"""fig0: how Inkling 2 Mini's looped middle layers compare with a plain stack."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "charts"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#d9d8d3"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "figure.facecolor": SURFACE,
                     "savefig.dpi": 200})

fig, ax = plt.subplots(figsize=(10, 6.2))
ax.set_xlim(0, 10.2)
ax.set_ylim(0, 6.6)
ax.axis("off")


def block(x, y, w, h, color, text, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5, color=tc)


# Left: plain deep transformer
ax.text(2.0, 6.35, "A plain 48-layer transformer", ha="center", fontsize=13, color=INK, weight="bold")
ax.text(2.0, 6.0, "48 layers of depth = 48 layers of weights", ha="center", fontsize=10, color=INK2)
for i in range(8):
    block(0.9, 0.5 + i * 0.66, 2.2, 0.56, GRAY, f"layers {i * 6 + 1}-{i * 6 + 6}")
ax.text(2.0, 0.15, "every block stored separately", ha="center", fontsize=9.5, color=MUTED)

# Right: Inkling
ax.text(7.2, 6.35, "Inkling 2 Mini", ha="center", fontsize=13, color=INK, weight="bold")
ax.text(7.2, 6.0, "48 layer applications from 24 layers of weights", ha="center", fontsize=10, color=INK2)
block(6.1, 0.7, 2.2, 0.7, GRAY, "layers 1-6  (prelude, runs once)")
block(6.1, 1.75, 2.2, 1.0, BLUE, "layers 7-12", "white")
block(6.1, 2.95, 2.2, 1.0, BLUE, "layers 13-18", "white")
block(6.1, 4.3, 2.2, 0.7, GRAY, "layers 19-24  (runs once)")
for y0, y1 in [(1.4, 1.75), (3.95, 4.3)]:
    ax.add_patch(FancyArrowPatch((7.2, y0), (7.2, y1), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.2))
ax.add_patch(FancyArrowPatch((8.35, 3.85), (8.35, 1.85), connectionstyle="arc3,rad=-0.6", arrowstyle="-|>",
                             mutation_scale=14, color=ORANGE, lw=2))
ax.text(9.55, 2.85, "x3\nshared\nweights", ha="center", va="center", fontsize=11, color=ORANGE, weight="bold")
ax.text(4.9, 2.6, "a learned gate blends\neach new pass into\nthe previous one", ha="center", fontsize=9.5,
        color=INK2)
ax.add_patch(FancyArrowPatch((5.6, 2.6), (6.05, 2.6), arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1))
ax.text(7.2, 0.15, "~20 GB on disk  |  3.4B active per pass, ~5.4B effective", ha="center", fontsize=9.5,
        color=MUTED)
ax.text(7.2, 5.4, "output", ha="center", fontsize=9.5, color=MUTED)
ax.add_patch(FancyArrowPatch((7.2, 5.0), (7.2, 5.3), arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1))
fig.tight_layout()
fig.savefig(OUT / "fig0-loop-diagram.png")
