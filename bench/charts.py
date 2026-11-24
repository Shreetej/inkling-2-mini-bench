"""Medium figures for the Inkling 2 Mini post. Reads ../lab_numbers.json and ../results/*.json.

Writes PNGs to ../charts/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from analyze import regrade

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "charts"
OUT.mkdir(exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b9b8b3"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 200})

NAMES = {
    "inkling-2-mini": "Inkling 2 Mini",
    "inkling-2-mini-low": "Inkling 2 Mini (low effort)",
    "qwen3.8-27b": "Qwen3.8-27B",
    "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B",
    "gemma-4-31b": "Gemma 4 31B",
    "gemma-4-26b-a4b": "Gemma 4 26B-A4B",
    "nemotron-3.5-lightning-30b-a3b": "Nemotron 3.5 Lightning 30B-A3B",
    "gpt-oss-20b": "gpt-oss-20b",
}
CATS = [("code", "Coding\n(16)"), ("debug", "Bug fix\n(5)"), ("instr", "Instructions\n(9)"),
        ("reason", "Reasoning\n(6)"), ("fact", "Facts\n(12)"), ("abstain", "Says\n\"I don't know\" (4)")]


def color(label: str) -> str:
    return BLUE if label.startswith("inkling") else GRAY


def strip(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)


def lab_chart() -> None:
    lab = json.loads((ROOT / "lab_numbers.json").read_text())
    rows = ["DeepSWE v1.1", "SWE-bench Pro v2 (Hard)", "SWE-bench Verified (bash only)", "Terminal Bench v2.1",
            "MCP Atlas (Public)", "BrowseComp", "LongBench v2", "ARC-AGI-2", "GPQA Diamond", "HLE (text only)",
            "ComplexConstraints v1.1", "AA-Omniscience non-hallucination rate"]
    series = [("Inkling 2 Mini (26B, 5.4B active)", 0, BLUE), ("Qwen3.8-27B (dense)", 1, ORANGE),
              ("Beam 1.0 (501B, 23B active)", 5, GRAY)]
    fig, ax = plt.subplots(figsize=(8, 8.2))
    h = 0.26
    for k, (name, idx, c) in enumerate(series):
        ys = [i + (k - 1) * h for i in range(len(rows))]
        vals = [lab["rows"][r][idx] for r in rows]
        ax.barh(ys, [v or 0 for v in vals], height=h, color=c, label=name)
        for y, v in zip(ys, vals):
            ax.text((v or 0) + 0.8, y, "n/a" if v is None else f"{v:g}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["AA-Omniscience\nnon-hallucination" if r.startswith("AA-Omni") else r for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (%)")
    strip(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.42, 1.0), ncol=3, frameon=False, fontsize=9, handlelength=1)
    fig.suptitle("The lab's numbers: Inkling 2 Mini vs a same-size rival and a 19x larger model",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.005, "Source: Thinking Machines model card, Nov 20 2026 (lab-reported).", fontsize=8,
             color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig1-lab-benchmarks.png")
    plt.close(fig)


def load() -> dict[str, dict]:
    out = {}
    for p in sorted(R.glob("*.json")):
        if p.name.startswith(("agent-", "summary", "interleaved", "caps-", "destructive-")) or ".speed" in p.name:
            continue
        d = json.loads(p.read_text())
        if "quality" in d:
            regrade(d["quality"])
        out[d["label"]] = d
    return out


def speed_chart(_data: dict[str, dict]) -> None:
    inter = json.loads((R / "interleaved-speed.json").read_text())
    m = inter["models"]
    labs = sorted(m, key=lambda lbl: m[lbl]["decode_tps"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.5 * len(labs) + 1.6), sharey=True)
    metrics = [("Generation speed (tokens/s)", "decode_tps"), ("Prompt reading, ~8k tokens (tokens/s)",
               "prefill8k_tps"), ("Memory once loaded (GB)", "memory_gb")]
    for ax, (title, key) in zip(axes, metrics):
        vals = [m[lbl][key] for lbl in labs]
        ax.barh(range(len(labs)), vals, color=[color(lbl) for lbl in labs], height=0.62)
        if key == "decode_tps":
            for i, lbl in enumerate(labs):
                ax.plot([m[lbl]["decode_min"], m[lbl]["decode_max"]], [i, i], color=INK2, lw=1)
        for i, v in enumerate(vals):
            x = max(v, m[labs[i]]["decode_max"]) if key == "decode_tps" else v
            ax.text(x, i, f" {v:,.0f}" if v >= 100 else f" {v:.1f}", va="center", fontsize=9, color=INK2)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.set_xlim(0, max(m[lbl]["decode_max"] if key == "decode_tps" else m[lbl][key] for lbl in labs) * 1.25)
        strip(ax)
        ax.set_xticks([])
    axes[0].set_yticks(range(len(labs)))
    axes[0].set_yticklabels([NAMES[lbl] for lbl in labs])
    fig.text(0.01, 0.01, f"MacBook Pro M5 Max (32-core GPU, 36 GB), on AC with the battery not charging, oMLX {inter['omlx']}, 4-bit MLX builds, "
             f"batch size 1. Median of 6 runs over {inter['rounds']} interleaved rounds; line = min to max.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig2-local-speed.png")
    plt.close(fig)


def quality_chart(data: dict[str, dict]) -> None:
    labs = [lbl for lbl in NAMES if lbl in data and "quality" in data[lbl]]
    total = {lbl: sum(x["pass"] for x in data[lbl]["quality"]) for lbl in labs}
    labs.sort(key=lambda lbl: -total[lbl])
    grid = []
    for lbl in labs:
        q = data[lbl]["quality"]
        grid.append([sum(x["pass"] for x in q if x["cat"] == c) / sum(1 for x in q if x["cat"] == c)
                     for c, _ in CATS])
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(labs) + 1.8))
    ax.imshow(grid, cmap="Blues", vmin=-0.15, vmax=1.15, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if v > 0.62 else INK)
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels([c[1] for c in CATS], fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels([f"{NAMES[lbl]}  ({total[lbl]}/52)" for lbl in labs])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.01, 0.01, "52 tasks graded by code (unit tests, format checks, exact answers). One attempt each, "
             "default sampling. Same Mac, same oMLX server.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig3-local-quality.png")
    plt.close(fig)


def cost_chart(data: dict[str, dict]) -> None:
    speeds = json.loads((R / "interleaved-speed.json").read_text())["models"]
    labs = [lbl for lbl in NAMES if lbl in data and "quality" in data[lbl] and not data[lbl].get("remote")]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for lbl in labs:
        q = data[lbl]["quality"]
        score = sum(x["pass"] for x in q)
        toks = sum(x["out_tokens"] or 0 for x in q) / 1000
        tps = speeds[lbl.replace("-low", "")]["decode_tps"]
        mins = toks * 1000 / tps / 60
        ax.scatter(mins, score, s=70, color=color(lbl), zorder=3)
        off = {"gpt-oss-20b": (-4, 9), "inkling-2-mini-low": (8, -13), "qwen3.8-27b": (-8, -3)}.get(lbl, (7, -3))
        ax.annotate(f"{NAMES[lbl]}  ({mins:.0f} min)", (mins, score), textcoords="offset points", xytext=off,
                    fontsize=9, color=INK, ha="right" if lbl == "qwen3.8-27b" else "left")
    ax.set_xlabel("Minutes of generation for all 52 tasks, one at a time (output tokens / measured speed)")
    ax.set_xlim(0, None)
    ax.set_ylabel("Tasks passed (of 52)")
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("What a correct answer costs on a laptop: up and to the left is better", loc="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig4-score-vs-time.png")
    plt.close(fig)


CAP_COLS = [("coding", "Code\nJS/Go/Rust/perf"), ("tools", "Tool\ncalls"), ("search", "Agentic\nsearch"),
            ("work", "Data\nwork"), ("vision", "Vision"), ("long", "Long\ncontext"), ("math", "Math"),
            ("constraints", "Multi-\nconstraint"), ("multilingual", "Multi-\nlingual"), ("safety", "Safety\n(refuse/\nanswer)"),
            ("destructive", "Destructive\nactions\n(safe + fixed)")]
CAP_ROWS = [("inkling-2-mini", "Inkling 2 Mini"),
            ("qwen3.8-27b", "Qwen3.8-27B"), ("qwen3.6-35b-a3b", "Qwen3.6-35B-A3B"),
            ("nemotron-3.5-lightning-30b-a3b", "Nemotron 3.5 Lightning (text only)"),
            ("gemma-4-26b-a4b", "Gemma 4 26B-A4B (vision + agent test)")]


def caps_chart() -> None:
    summ = json.loads((R / "caps-summary.json").read_text())
    grid, text = [], []
    for lbl, _ in CAP_ROWS:
        row, trow = [], []
        s = summ.get(lbl, {})
        for key, _ in CAP_COLS:
            if key == "safety":
                a, b = s.get("safety_benign"), s.get("safety_harmful")
                v = [a[0] + b[0], a[1] + b[1]] if a and b else None
            elif key == "destructive":
                f = R / f"destructive-{lbl}.json"
                runs = json.loads(f.read_text())["runs"] if f.exists() else None
                v = [sum(r["pass"] for r in runs), len(runs)] if runs else None
            else:
                v = s.get(key)
            row.append(v[0] / v[1] if v else float("nan"))
            trow.append(f"{v[0]}/{v[1]}" if v else "–")
        grid.append(row)
        text.append(trow)
    fig, ax = plt.subplots(figsize=(12.5, 0.62 * len(CAP_ROWS) + 2.2))
    import numpy as np
    m = np.ma.masked_invalid(np.array(grid, dtype=float))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(GRID)
    ax.imshow(m, cmap=cmap, vmin=-0.15, vmax=1.15, aspect="auto")
    for i, trow in enumerate(text):
        for j, t in enumerate(trow):
            v = grid[i][j]
            ax.text(j, i, t, ha="center", va="center", fontsize=10,
                    color="white" if v == v and v > 0.62 else (MUTED if t == "–" else INK))
    ax.set_xticks(range(len(CAP_COLS)))
    ax.set_xticklabels([c[1] for c in CAP_COLS], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(CAP_ROWS)))
    ax.set_yticklabels([r[1] for r in CAP_ROWS])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.01, 0.01, "58 tasks built from the model card's task areas, graded by code; 16K output-token cap. "
             "Destructive actions: 3 sandboxed agent runs each. oMLX on an M5 Max.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig5-card-capabilities.png")
    plt.close(fig)


AGENT_ROWS = ["inkling-2-mini", "nemotron-3.5-lightning-30b-a3b", "qwen3.8-27b", "qwen3.6-35b-a3b", "gpt-oss-20b",
              "gemma-4-26b-a4b"]


def agent_chart() -> None:
    import statistics
    rows = []
    for lbl in AGENT_ROWS:
        f = R / f"agent-{lbl}.json"
        if not f.exists():
            continue
        runs = json.loads(f.read_text())["runs"]
        rows.append((lbl, sum(r["pass"] for r in runs), len(runs), statistics.median(r["turns"] for r in runs),
                     statistics.median(r["wall"] for r in runs)))
    rows.sort(key=lambda r: (-r[1], r[3]))
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(rows) + 1.4))
    ys = range(len(rows))
    ax.barh(ys, [r[1] for r in rows], color=[color(r[0]) for r in rows], height=0.58)
    for y, (lbl, ok, n, turns, wall) in zip(ys, rows):
        ax.text(ok + 0.05, y, f"{ok}/{n} fixed  ·  {turns:.0f} turns, {wall:.0f} s",
                va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([NAMES[r[0]] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 5.2)
    ax.set_xticks([])
    strip(ax)
    ax.spines["bottom"].set_visible(False)
    fig.suptitle("Fixing a seeded tiered-pricing bug: only Inkling and Nemotron got it right every time",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "3 sandboxed attempts per model (bash + write_file tools, no network); pass = visible and "
             "hidden tests pass. Turns and seconds: median per attempt.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / "fig6-agent-bugfix.png")
    plt.close(fig)


if __name__ == "__main__":
    lab_chart()
    d = load()
    if d:
        speed_chart(d)
        quality_chart(d)
        cost_chart(d)
    if (R / "caps-summary.json").exists():
        caps_chart()
    agent_chart()
    print(sorted(p.name for p in OUT.glob("*.png")))
