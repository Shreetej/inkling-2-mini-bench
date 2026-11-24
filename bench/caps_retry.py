"""Re-run only the capability tasks that hit the 16K output cap, with a 32K cap.

Separates "needs more room" from "stuck in a loop" (the card's repetitive-reasoning warning).
The headline scores stay at 16K; this writes ../results/caps-<label>.retry32k.json.
  python3 caps_retry.py inkling-2-mini ...      (remote: label=served-name with BENCH_BASE/BENCH_KEYFILE)
"""
from __future__ import annotations

import json
import sys

import caps
from caps_tasks import all_caps
from run import MODELS, OUT, http, unload_all

caps.MAX_TOKENS = 32768


def main() -> None:
    by_id = {c.id: c for c in all_caps()}
    for arg in sys.argv[1:]:
        label, _, remote_model = arg.partition("=")
        model, extra = (remote_model, {}) if remote_model else MODELS[label][:2]
        out = OUT / f"caps-{label}.retry32k.json"
        if out.exists():
            print("skip", label)
            continue
        res = json.loads((OUT / f"caps-{label}.json").read_text())["results"]
        # the two count-to-10,000 math problems need code, not more tokens (see caps-loop-sample-hosted.json)
        todo = [by_id[r["id"]] for r in res if r["finish"] == "length" and r["id"] not in
                ("math/divisors4", "math/euler_poly")]
        if not todo:
            print(label, "nothing truncated")
            continue
        if caps.LOCAL:
            unload_all()
            http("POST", f"/v1/models/{model}/load")
        done = []
        for c in todo:
            r = caps.one(model, extra, c)
            done.append(r)
            print(f"  {label} {c.id}: {'PASS' if r['pass'] else 'fail'} finish={r['finish']} "
                  f"tokens={r['out_tokens']} wall={r['wall']}s", flush=True)
        out.write_text(json.dumps({"label": label, "max_tokens": caps.MAX_TOKENS, "results": done}, indent=1))
    if caps.LOCAL:
        unload_all()


if __name__ == "__main__":
    main()
