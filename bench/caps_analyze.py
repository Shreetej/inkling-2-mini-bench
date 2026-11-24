"""Regrade saved capability-suite answers with the current checkers and print per-area tables.

Writes ../results/caps-summary.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from caps_tasks import all_caps

R = Path(__file__).resolve().parent.parent / "results"
CAPS = {c.id: c for c in all_caps()}
AREAS = ["coding", "tools", "search", "work", "vision", "long", "math", "constraints", "multilingual",
         "safety_benign", "safety_harmful"]


def load(label: str) -> list[dict]:
    full = R / f"caps-{label}.json"
    if full.exists():
        res = json.loads(full.read_text())["results"]
    else:
        part = R / f"caps-{label}.partial.jsonl"
        by = {}
        for ln in part.read_text().splitlines() if part.exists() else []:
            rec = json.loads(ln)
            if not rec.get("error") or rec["id"] not in by:
                by[rec["id"]] = rec
        res = list(by.values())
    for r in res:
        c = CAPS[r["id"]]
        r["pass"] = r.get("error") is None and bool(c.check(r["content"] or "", [tuple(x) for x in r["calls"]]))
    return res


def main() -> None:
    labels = sorted({re.sub(r"(\.retry32k)?\.(partial\.jsonl|json)$", "", p.name[5:]) for p in R.glob("caps-*.json*")
                     if not any(x in p.name for x in ("summary", "retry32k", "loop-sample"))})
    summary = {}
    for lbl in labels:
        res = load(lbl)
        row = {}
        for a in AREAS:
            xs = [r for r in res if r["area"] == a]
            if xs:
                row[a] = [sum(r["pass"] for r in xs), len(xs)]
        row["total"] = [sum(r["pass"] for r in res), len(res)]
        row["truncated"] = [r["id"] for r in res if r["finish"] == "length"]
        row["failed"] = [r["id"] for r in res if not r["pass"]]
        row["out_tokens"] = sum(r["out_tokens"] or 0 for r in res)
        summary[lbl] = row
        print(lbl, {k: f"{v[0]}/{v[1]}" for k, v in row.items() if isinstance(v, list) and len(v) == 2
                    and isinstance(v[0], int)}, "trunc", row["truncated"])
        print("   failed:", row["failed"])
    (R / "caps-summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
