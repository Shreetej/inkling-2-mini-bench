"""Regrade every saved answer with the current checkers and print the tables used in the post.

Regrading offline means a grader fix never needs a model re-run. Writes ../results/summary.json.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from tasks import all_tasks

R = Path(__file__).resolve().parent.parent / "results"
TASKS = {t.id: t for t in all_tasks()}
CATS = ["code", "debug", "instr", "reason", "fact", "abstain"]


HARMONY_FINAL = "<|channel|>final<|message|>"


def final_text(content: str) -> str:
    """oMLX returns gpt-oss's raw harmony channels in `content`; grade only the final channel."""
    if HARMONY_FINAL in content:
        return content.rsplit(HARMONY_FINAL, 1)[1].split("<|", 1)[0]
    return content


def regrade(q: list[dict]) -> list[dict]:
    for x in q:
        x["pass"] = bool(x.get("content")) and TASKS[x["id"]].check(final_text(x["content"]))
    return q


def main() -> None:
    summary = {}
    for p in sorted(R.glob("*.json")):
        if p.name.startswith(("agent-", "summary", "interleaved", "caps-", "destructive-")) or ".speed" in p.name:
            continue
        d = json.loads(p.read_text())
        row: dict = {"model": d["model"]}
        if d.get("remote"):
            row["remote_decode_tps"] = d["speed"]["decode_tps"]
            row["remote_ttft_38k_s"] = min(r["ttft"] for r in d["speed"]["long_prompt_runs"])
        elif "speed" in d:
            s = d["speed"]
            row |= {"decode_tps": s["decode_tps"], "memory_gb": s["memory_gb"],
                    "prefill8k_tps": max(r["prefill_tps"] or 0 for r in s["prefill_6000w"]),
                    "prefill32k_tps": max(r["prefill_tps"] or 0 for r in s["prefill_24000w"]),
                    "ttft32k_s": min(r["ttft"] or 1e9 for r in s["prefill_24000w"]),
                    "prompt_tokens_32k": s["prefill_24000w"][0]["prompt_tokens"]}
        if "quality" in d:
            q = regrade(d["quality"])
            row["total"] = sum(x["pass"] for x in q)
            row["by_cat"] = {c: f"{sum(x['pass'] for x in q if x['cat'] == c)}/{sum(1 for x in q if x['cat'] == c)}"
                             for c in CATS}
            row["out_tokens"] = sum(x["out_tokens"] or 0 for x in q)
            row["median_tokens_code"] = statistics.median(x["out_tokens"] or 0 for x in q if x["cat"] == "code")
            row["truncated"] = [x["id"] for x in q if x["finish"] == "length"]
            row["errors"] = [x["id"] for x in q if x.get("error")]
            row["failed"] = [x["id"] for x in q if not x["pass"]]
        summary[d["label"]] = row
    for p in sorted(R.glob("agent-*.json")):
        d = json.loads(p.read_text())
        runs = d["runs"]
        summary.setdefault(d["label"], {})["agent"] = {
            "passes": f"{sum(r['pass'] for r in runs)}/{len(runs)}",
            "median_wall_s": statistics.median(r["wall"] for r in runs),
            "median_turns": statistics.median(r["turns"] for r in runs),
            "median_tokens": statistics.median(r["out_tokens"] for r in runs),
            "stops": [r["stop"] for r in runs],
            "repeated_commands": [r["repeated_commands"] for r in runs],
            "tests_untouched": [r["tests_untouched"] for r in runs]}
    (R / "summary.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
