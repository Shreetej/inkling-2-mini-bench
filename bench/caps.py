"""Run the model-card capability suite (caps_tasks.py) against local oMLX models or a remote OpenAI-compatible server.

  python3 caps.py inkling-2-mini qwen3.8-27b ...            # local labels from run.MODELS
  BENCH_BASE=... BENCH_KEYFILE=... python3 caps.py my-label=<served-name>   # any OpenAI-compatible server

Vision tasks go only to vision models; text-only models skip them. Results: ../results/caps-<label>.json
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

from caps_tasks import Cap, all_caps
from run import BASE, MODELS, OUT, http, unload_all

LOCAL = "127.0.0.1" in BASE
VISION = {"inkling-2-mini", "qwen3.8-27b", "qwen3.6-35b-a3b", "gemma-4-26b-a4b"}
TEXT = {"inkling-2-mini", "qwen3.8-27b", "qwen3.6-35b-a3b", "nemotron-3.5-lightning-30b-a3b"}
MAX_TOKENS = 16384
MAX_TOOL_TURNS = 15


def call(model: str, messages: list[dict], extra: dict, tools: list | None = None) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": MAX_TOKENS, **extra}
    if tools:
        body["tools"] = tools
    err = ""
    for attempt in range(4):
        try:
            r = http("POST", "/v1/chat/completions", body)
            if r.get("choices"):
                return r
            err = json.dumps(r)[:300]
        except urllib.error.HTTPError as e:
            err = f"{e.code} {e.read()[:300]!r}"
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        time.sleep(10 * (attempt + 1))
    return {"error": err}


def one(model: str, extra: dict, cap: Cap) -> dict:
    t0 = time.monotonic()
    msgs = [dict(m) for m in cap.messages]
    calls: list[tuple[str, dict]] = []
    out_tokens, finish, content, err, reasoning = 0, None, "", None, ""
    for _turn in range(MAX_TOOL_TURNS if cap.tools else 1):
        r = call(model, msgs, extra, cap.tools)
        if "error" in r:
            err = r["error"]
            break
        out_tokens += (r.get("usage") or {}).get("completion_tokens") or 0
        ch = r["choices"][0]
        m, finish = ch["message"], ch.get("finish_reason")
        content = m.get("content") or ""
        reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
        tcs = m.get("tool_calls") or []
        if not (cap.tools and tcs):
            break
        msgs.append({"role": "assistant", "content": content, "tool_calls": tcs})
        for tc in tcs:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append((fn["name"], args))
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": cap.tool_impl(fn["name"], args)})
    else:
        finish = "tool_turn_limit"
    ok = err is None and cap.check(content, calls)
    return {"id": cap.id, "area": cap.area, "pass": bool(ok), "finish": finish, "error": err, "out_tokens": out_tokens,
            "wall": round(time.monotonic() - t0, 1), "content": content, "calls": calls,
            "reasoning_len": len(reasoning), "reasoning_tail": reasoning[-6000:]}


def run_label(label: str, model: str, extra: dict) -> None:
    path = OUT / f"caps-{label}.json"
    part = OUT / f"caps-{label}.partial.jsonl"
    if path.exists():
        print("skip", label)
        return
    done = {}
    if part.exists():
        for ln in part.read_text().splitlines():
            rec = json.loads(ln)
            if not rec.get("error"):
                done[rec["id"]] = rec
    known = label in VISION | TEXT  # unknown labels (e.g. your own server) run every task
    caps = [c for c in all_caps() if not known or (label in VISION if c.vision else label in TEXT)]
    todo = [c for c in caps if c.id not in done]
    if LOCAL and todo:
        unload_all()
        http("POST", f"/v1/models/{model}/load")
    lock = threading.Lock()

    def save(c: Cap) -> dict:
        rec = one(model, extra, c)
        with lock, part.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  {label} {c.id}: {'PASS' if rec['pass'] else 'fail'} ({rec['out_tokens']} tok, {rec['wall']}s)"
              + (f" ERR {rec['error'][:80]}" if rec["error"] else ""), flush=True)
        return rec

    short = [c for c in todo if not c.long]
    with ThreadPoolExecutor(2) as ex:
        new = list(ex.map(save, short))
    new += [save(c) for c in todo if c.long]  # one at a time: 15-30K-token prompts
    by_id = {**done, **{r["id"]: r for r in new}}
    res = [by_id[c.id] for c in caps]
    path.write_text(json.dumps({"label": label, "model": model, "extra": extra, "results": res}, indent=1))
    print(f"== {label}: {sum(r['pass'] for r in res)}/{len(res)}", flush=True)


def main() -> None:
    for arg in sys.argv[1:]:
        label, _, remote_model = arg.partition("=")
        model, extra = (remote_model, {}) if remote_model else MODELS[label][:2]
        run_label(label, model, extra)
    if LOCAL:
        unload_all()


if __name__ == "__main__":
    main()
