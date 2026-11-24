"""Interleaved speed check: 3 rounds, every model once per round in rotated order.

A sequential pass lets thermal or charging drift land on whichever model runs last. Interleaving spreads it
evenly. Per visit: fresh load, warm-up, 2 decode runs (700 tokens) and 1 prefill run (~8k tokens).
Writes ../results/interleaved-speed.json, which charts.py prefers over the per-model speed blocks.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time

import run

LABELS = ["inkling-2-mini", "qwen3.8-27b", "qwen3.6-35b-a3b", "gemma-4-31b", "gemma-4-26b-a4b",
          "nemotron-3.5-lightning-30b-a3b", "gpt-oss-20b"]
ROUNDS = 3


def on_ac() -> bool:
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    pm = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    return "AC Power" in batt and all(ln.split()[-1] == "0" for ln in pm.splitlines() if "powermode" in ln.lower())


def visit(label: str, rnd: int) -> dict:
    model = run.MODELS[label][0]
    run.unload_all()
    run.http("POST", f"/v1/models/{model}/load")
    mem = round(run.http("GET", "/api/status")["model_memory_used"] / 1e9, 2)
    run.chat(model, "Say hi.", 32, {})
    extra = {"reasoning_effort": "low"} if "nkling" in model or "gpt-oss" in model else {}
    dec = []
    for i in range(2):
        r = run.chat(model, f"[{rnd}.{i}] Write a 600-word short story about a lighthouse keeper who finds a radio.",
                     700, extra)
        dec.append(r["usage"].get("generation_tokens_per_second", 0))
    p = f"Run {time.time_ns()}.\n{run.filler(6000, rnd)}\n\nHow many times does 'ember' appear? Guess."
    u = run.chat(model, p, 4, {})["usage"]
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.split("\t")[-1].split(";")[0]
    return {"round": rnd, "decode": dec, "prefill_tps": u.get("prompt_tokens_per_second"),
            "prompt_tokens": u.get("prompt_tokens"), "memory_gb": mem, "battery": batt.strip()}


def main() -> None:
    if not on_ac():
        sys.exit("not on AC power / low power mode on; aborting")
    ver = run.http("GET", "/api/status")["version"]
    visits: dict[str, list[dict]] = {lbl: [] for lbl in LABELS}
    for rnd in range(ROUNDS):
        order = LABELS[rnd * 2 % len(LABELS):] + LABELS[:rnd * 2 % len(LABELS)]
        for lbl in order:
            v = visit(lbl, rnd)
            visits[lbl].append(v)
            print(lbl, v, flush=True)
    out = {"omlx": ver, "power": "ac", "rounds": ROUNDS, "models": {}}
    for lbl, vs in visits.items():
        dec = [d for v in vs for d in v["decode"]]
        out["models"][lbl] = {"decode_tps": round(statistics.median(dec), 1), "decode_min": min(dec),
                              "decode_max": max(dec),
                              "prefill8k_tps": round(statistics.median(v["prefill_tps"] for v in vs), 1),
                              "memory_gb": max(v["memory_gb"] for v in vs), "visits": vs}
    (run.OUT / (sys.argv[1] if len(sys.argv) > 1 else "interleaved-speed.json")).write_text(json.dumps(out, indent=1))
    run.unload_all()
    for lbl, m in out["models"].items():
        print(lbl, m["decode_tps"], f"({m['decode_min']}-{m['decode_max']})", m["prefill8k_tps"], m["memory_gb"])


if __name__ == "__main__":
    main()
