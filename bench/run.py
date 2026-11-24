"""Run speed + quality benchmarks against models served by a local oMLX server.

  python run.py                 # every model in MODELS, skipping ones with results already saved
  python run.py Inkling-2-Mini-oQ4e-mtp
Results: ../results/<label>.json
"""
from __future__ import annotations

import json
import os
import random
import ssl
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tasks import all_tasks

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8005")
_KEYFILE = os.environ.get("BENCH_KEYFILE")
try:  # python.org builds ship without CA certs; verify against certifi's bundle
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
AUTH = {"authorization": "Bearer " + Path(_KEYFILE).expanduser().read_text().strip()} if _KEYFILE else {}
OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

# label -> (omlx model id, extra request fields, run speed?, run quality?)
MODELS: dict[str, tuple[str, dict, bool, bool]] = {
    "inkling-2-mini": ("Inkling-2-Mini-oQ4e-mtp", {}, True, True),
    "inkling-2-mini-low": ("Inkling-2-Mini-oQ4e-mtp", {"reasoning_effort": "low"}, False, True),
    "qwen3.8-27b": ("Qwen3.8-27B-oQ4e-mtp", {}, True, True),
    "qwen3.6-35b-a3b": ("mlx-community--Qwen3.6-35B-A3B-4bit", {}, True, True),
    "gemma-4-31b": ("gemma-4-31B-it-MLX-4bit", {}, True, False),
    "gemma-4-26b-a4b": ("gemma-4-26B-A4B-it-QAT-MLX-4bit", {}, True, True),
    "muse-glimmer-30b": ("Muse-Glimmer-30B-4bit", {}, True, False),
    "nemotron-3.5-lightning-30b-a3b": ("NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit", {}, True, True),
    "gpt-oss-20b": ("gpt-oss-20b-MXFP4-Q8", {}, True, True),
}
MAX_TOKENS = 16384
CONCURRENCY = 2  # 4 parallel 16k-token generations hit Metal OOM on a 36 GB Mac


def http(method: str, path: str, body: dict | None = None, timeout: float = 1800) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"content-type": "application/json", **AUTH})
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
        return json.loads(r.read() or b"{}")


def chat(model: str, prompt: str, max_tokens: int, extra: dict) -> dict:
    t0 = time.monotonic()
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, **extra}
    err = ""
    for attempt in range(4):  # oMLX's memory guard rejects requests (HTTP 400) under memory pressure; back off
        try:
            r = http("POST", "/v1/chat/completions", body)
            if r.get("choices"):
                break
            err = json.dumps(r)[:300]
        except urllib.error.HTTPError as e:
            err = f"{e.code} {e.read()[:300]!r}"
        except Exception as e:  # noqa: BLE001 - record and keep going
            err = repr(e)
        time.sleep(10 * (attempt + 1))
    else:
        return {"error": err, "wall": time.monotonic() - t0, "usage": {}}
    msg = r["choices"][0]["message"]
    return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
            "finish": r["choices"][0].get("finish_reason"), "usage": r.get("usage", {}),
            "wall": time.monotonic() - t0}


def unload_all() -> None:
    for m in http("GET", "/v1/models/status")["models"]:
        if m.get("loaded"):
            try:
                http("POST", f"/v1/models/{m['id']}/unload")
            except urllib.error.HTTPError:
                pass  # helper models (e.g. MarkItDown) can't be unloaded through this route
    time.sleep(2)


def filler(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    vocab = ("river stone engine maple orbit lantern copper meadow signal harbor quartz velvet tunnel ember "
             "falcon prism glacier canyon ledger beacon cobalt thistle anchor saffron").split()
    return " ".join(rng.choice(vocab) for _ in range(n_words))


def speed(model: str) -> dict:
    res: dict = {}
    unload_all()
    t0 = time.monotonic()
    http("POST", f"/v1/models/{model}/load")
    res["load_s"] = round(time.monotonic() - t0, 1)
    res["memory_gb"] = round(http("GET", "/api/status")["model_memory_used"] / 1e9, 2)
    chat(model, "Say hi.", 32, {})  # warm-up
    dec = []
    for i in range(3):
        r = chat(model, f"[{i}] Write a 600-word short story about a lighthouse keeper who finds a radio.", 700,
                 {"reasoning_effort": "low"} if "nkling" in model or "gpt-oss" in model else {})
        dec.append(r["usage"].get("generation_tokens_per_second", 0))
    res["decode_tps"] = round(statistics.median(dec), 1)
    res["decode_runs"] = dec
    for words in (6000, 24000):  # ~8k and ~32k tokens
        runs = []
        for i in range(2):
            p = f"Run {time.time_ns()}.\n{filler(words, i + words)}\n\nHow many times does 'ember' appear? Guess."
            r = chat(model, p, 4, {})
            u = r["usage"]
            runs.append({"prompt_tokens": u.get("prompt_tokens"), "prefill_tps": u.get("prompt_tokens_per_second"),
                         "ttft": u.get("time_to_first_token"), "cached": u.get("prompt_tokens_details", {})})
        res[f"prefill_{words}w"] = runs
    res["memory_gb_after_32k"] = round(http("GET", "/api/status")["model_memory_used"] / 1e9, 2)
    return res


def quality(model: str, extra: dict, partial: Path) -> list[dict]:
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            rec = json.loads(line)
            if not rec.get("error"):
                done[rec["id"]] = rec
    tasks = [t for t in all_tasks() if t.id not in done]
    lock = threading.Lock()

    def one(t):
        r = chat(model, t.prompt, MAX_TOKENS, extra)
        ok = bool(r.get("content")) and t.check(r["content"])
        u = r.get("usage", {})
        return {"id": t.id, "cat": t.cat, "pass": ok, "finish": r.get("finish"), "error": r.get("error"),
                "out_tokens": u.get("completion_tokens"),
                "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "wall": round(r["wall"], 1), "content": r.get("content", ""), "reasoning": r.get("reasoning", "")}

    def save(t):
        rec = one(t)
        with lock, partial.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(CONCURRENCY) as ex:
        new = list(ex.map(save, tasks))
    by_id = {**done, **{r["id"]: r for r in new}}
    return [by_id[t.id] for t in all_tasks()]


def main() -> None:
    only = set(sys.argv[1:])
    for label, (model, extra, do_speed, do_quality) in MODELS.items():
        if only and label not in only and model not in only:
            continue
        path = OUT / f"{label}.json"
        if path.exists():
            print("skip", label)
            continue
        print(f"== {label} ({model})", flush=True)
        rec: dict = {"label": label, "model": model, "extra": extra}
        sp = OUT / f"{label}.speed.json"
        if do_speed and sp.exists():
            rec["speed"] = json.loads(sp.read_text())
        elif do_speed:
            rec["speed"] = speed(model)
            sp.write_text(json.dumps(rec["speed"], indent=1))
            print("  speed", {k: v for k, v in rec["speed"].items() if not k.startswith("prefill")}, flush=True)
        if do_quality:
            t0 = time.monotonic()
            rec["quality"] = quality(model, extra, OUT / f"{label}.partial.jsonl")
            rec["quality_wall_s"] = round(time.monotonic() - t0, 1)
            q = rec["quality"]
            print(f"  quality {sum(x['pass'] for x in q)}/{len(q)} in {rec['quality_wall_s']}s, "
                  f"out_tokens={sum(x['out_tokens'] or 0 for x in q)}", flush=True)
        path.write_text(json.dumps(rec, indent=1))
    unload_all()


if __name__ == "__main__":
    main()
