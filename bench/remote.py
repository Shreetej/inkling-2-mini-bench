"""Same 52-task suite (and a streaming speed check) against a remote OpenAI-compatible server.

  BENCH_BASE=https://host BENCH_KEYFILE=~/.my_api_key python remote.py <label> <model>
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request

import run


def stream_speed(model: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(run.BASE + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **run.AUTH})
    t0 = time.monotonic()
    first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=600, context=run.SSL_CTX) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    first = time.monotonic()
    t1 = time.monotonic()
    n = usage.get("completion_tokens", 0)
    return {"ttft": round(first - t0, 2) if first else None, "tokens": n,
            "decode_tps": round((n - 1) / (t1 - first), 1) if first and n > 1 else None,
            "prompt_tokens": usage.get("prompt_tokens")}


def main() -> None:
    label, model = sys.argv[1], sys.argv[2]
    out = run.OUT / f"{label}.json"
    rec: dict = {"label": label, "model": model, "extra": {}, "remote": True}
    runs = [stream_speed(model, f"[{i}] Write a 600-word short story about a lighthouse keeper who finds a radio.",
                         700) for i in range(3)]
    long_runs = [stream_speed(model, f"Run {time.time_ns()}.\n{run.filler(24000, i)}\n\nHow many times does 'ember' "
                              "appear? Guess.", 4) for i in range(2)]
    rec["speed"] = {"decode_tps": statistics.median(r["decode_tps"] for r in runs), "decode_runs": runs,
                    "long_prompt_runs": long_runs}
    print("speed", rec["speed"], flush=True)
    t0 = time.monotonic()
    rec["quality"] = run.quality(model, {}, run.OUT / f"{label}.partial.jsonl")
    rec["quality_wall_s"] = round(time.monotonic() - t0, 1)
    q = rec["quality"]
    print(f"quality {sum(x['pass'] for x in q)}/{len(q)} in {rec['quality_wall_s']}s, "
          f"out_tokens={sum(x['out_tokens'] or 0 for x in q)}", flush=True)
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
