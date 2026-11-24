"""Deterministic images for the vision tasks. Every image is drawn from a fixed seed, so ground truth is exact.

Each builder returns (png_paths, truth). Images are written to ./caps_assets/.
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT = Path(__file__).resolve().parent / "caps_assets"
OUT.mkdir(exist_ok=True)
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
MONO = "/System/Library/Fonts/Menlo.ttc"


def f(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(MONO if mono else BOLD if bold else FONT, size)


# ---------------------------------------------------------------- documents / OCR
RECEIPT_ITEMS = [("Oat milk 1L", 2.49), ("Sourdough loaf", 4.75), ("Free-range eggs x12", 5.20),
                 ("Cherry tomatoes 250g", 2.15), ("Greek yoghurt 500g", 3.60), ("Dark chocolate 85%", 2.95),
                 ("Basmati rice 1kg", 3.30)]


def receipt(name: str, items: list[tuple[str, float]], invoice: str) -> tuple[Path, dict]:
    w, h = 620, 160 + 44 * len(items) + 200
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 30), "GREENLEAF MARKET", font=f(30, bold=True), fill="black")
    d.text((40, 75), f"Invoice no. {invoice}      Date: 2026-10-14", font=f(18), fill="#333")
    y = 130
    for it, p in items:
        d.text((40, y), it, font=f(20), fill="black")
        d.text((480, y), f"{p:6.2f}", font=f(20, mono=True), fill="black")
        y += 44
    sub = round(sum(p for _, p in items), 2)
    tax = round(sub * 0.08, 2)
    total = round(sub + tax, 2)
    d.line((40, y + 5, w - 40, y + 5), fill="black", width=2)
    for label, val in [("Subtotal", sub), ("Tax 8%", tax), ("TOTAL DUE", total)]:
        y += 40
        d.text((40, y), label, font=f(22, bold=label == "TOTAL DUE"), fill="black")
        d.text((480, y), f"{val:6.2f}", font=f(22, mono=True), fill="black")
    p = OUT / f"{name}.png"
    im.save(p)
    return p, {"invoice": invoice, "total": f"{total:.2f}", "sub": sub}


PARAGRAPH = ["Quarterly maintenance of the north filtration unit is",
             "scheduled for the second Tuesday of every third month,",
             "unless the turbidity reading exceeds 4.2 NTU beforehand."]


def paragraph() -> tuple[Path, dict]:
    im = Image.new("RGB", (900, 220), "#fbfaf6")
    d = ImageDraw.Draw(im)
    for i, line in enumerate(PARAGRAPH):
        d.text((30, 40 + i * 50), line, font=f(26), fill="#111")
    p = OUT / "paragraph.png"
    im.save(p)
    return p, {"line2": PARAGRAPH[1]}


CITIES = [("Tromsø", -3.4), ("Valletta", 18.9), ("Ulaanbaatar", -11.6), ("Hobart", 9.2), ("Quito", 13.8)]


def table() -> tuple[Path, dict]:
    im = Image.new("RGB", (560, 330), "white")
    d = ImageDraw.Draw(im)
    d.text((30, 20), "Mean January temperature (°C)", font=f(22, bold=True), fill="black")
    d.rectangle((30, 70, 530, 310), outline="black", width=2)
    d.text((45, 82), "City", font=f(20, bold=True), fill="black")
    d.text((360, 82), "Temp", font=f(20, bold=True), fill="black")
    for i, (c, t) in enumerate(CITIES):
        y = 122 + i * 37
        d.line((30, y - 6, 530, y - 6), fill="#999")
        d.text((45, y), c, font=f(20), fill="black")
        d.text((360, y), f"{t:+.1f}", font=f(20, mono=True), fill="black")
    p = OUT / "table.png"
    im.save(p)
    return p, {"city": "Ulaanbaatar", "temp": -11.6}


# ---------------------------------------------------------------- charts
def bar_chart() -> tuple[Path, dict]:
    cats = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    vals = [42, 67, 31, 58, 74, 49]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    ax.bar(cats, vals, color="#4a7bd0")
    ax.set_ylim(0, 80)
    ax.set_yticks(range(0, 81, 10))
    ax.grid(axis="y", color="#ddd")
    ax.set_axisbelow(True)
    ax.set_title("Units shipped (thousands)")
    p = OUT / "bar.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p, {"Delta": 58}


def line_chart() -> tuple[Path, dict]:
    years = list(range(2016, 2026))
    rev = [3.1, 3.4, 3.6, 3.5, 3.9, 4.6, 5.2, 5.6, 6.3, 6.9]
    cost = [4.0, 4.1, 4.3, 4.4, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    ax.plot(years, rev, marker="o", label="Revenue", color="#2a78d6")
    ax.plot(years, cost, marker="s", label="Costs", color="#eb6834")
    ax.set_xticks(years)
    ax.grid(color="#e5e5e5")
    ax.legend()
    ax.set_ylabel("$ millions")
    ax.set_title("Revenue vs costs")
    p = OUT / "line.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    first = next(y for y, r, c in zip(years, rev, cost) if r > c)
    return p, {"year": first}


def scatter() -> tuple[Path, dict]:
    rng = random.Random(11)
    pts = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(23)]
    pts = [(x, y) for x, y in pts if abs(y - (0.6 * x + 2)) > 0.6]  # keep points clearly off the line
    above = sum(1 for x, y in pts if y > 0.6 * x + 2)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    ax.scatter(*zip(*pts), color="#333", s=36)
    ax.plot([0, 10], [2, 8], "--", color="#c0392b")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_title("Samples vs threshold")
    p = OUT / "scatter.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p, {"above": above, "n": len(pts)}


# ---------------------------------------------------------------- GUI screenshots (ScreenSpot-style grounding)
def _button(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: str = "#f2f2f2") -> None:
    d.rounded_rectangle(box, radius=6, fill=fill, outline="#bbb")
    tw = d.textlength(text, font=f(15))
    d.text(((box[0] + box[2] - tw) / 2, box[1] + 8), text, font=f(15), fill="#111")


def app_window() -> tuple[Path, dict]:
    W, H = 1280, 800
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 40), fill="#e9e9ee")
    d.text((16, 11), "Ledgerly — Q3 report.xlsx", font=f(16, bold=True), fill="#222")
    boxes = {}
    x = 16
    for label in ["New", "Open", "Save", "Print", "Export CSV", "Export PDF", "Share", "Undo", "Redo"]:
        w = int(d.textlength(label, font=f(15))) + 28
        boxes[label] = (x, 52, x + w, 86)
        _button(d, boxes[label], label)
        x += w + 8
    d.rectangle((0, 98, 220, H), fill="#f6f6f8")
    for i, s in enumerate(["Dashboard", "Invoices", "Customers", "Reports", "Tax", "Settings", "Help"]):
        d.text((24, 120 + i * 40), s, font=f(16), fill="#333")
    for r in range(14):
        for c in range(6):
            d.rectangle((240 + c * 170, 110 + r * 44, 240 + (c + 1) * 170, 154 + r * 44), outline="#ddd")
            d.text((250 + c * 170, 122 + r * 44), f"{(r * 7 + c * 13) % 97 * 41:,}", font=f(14), fill="#444")
    p = OUT / "gui_app.png"
    im.save(p)
    return p, {"box": boxes["Export CSV"], "size": (W, H)}


def settings_form() -> tuple[Path, dict]:
    W, H = 1024, 768
    im = Image.new("RGB", (W, H), "#fafafa")
    d = ImageDraw.Draw(im)
    d.text((40, 30), "Notification settings", font=f(26, bold=True), fill="#111")
    boxes = {}
    opts = ["Email me about new comments", "Email me when I am mentioned", "Send weekly digest",
            "Send product announcements", "Push notifications on mobile", "Daily summary at 8 am"]
    for i, o in enumerate(opts):
        y = 100 + i * 56
        cb = (60, y, 84, y + 24)
        d.rectangle(cb, outline="#555", width=2, fill="white")
        if i % 2 == 0:
            d.line((64, y + 12, 70, y + 19, 80, y + 5), fill="#2a78d6", width=3)
        d.text((100, y + 1), o, font=f(18), fill="#222")
        boxes[o] = (60, y, 100 + int(d.textlength(o, font=f(18))), y + 24)  # checkbox + its label
    _button(d, (60, 470, 180, 510), "Save", fill="#2a78d6")
    _button(d, (196, 470, 316, 510), "Cancel")
    d.text((60, 560), "Delete account", font=f(15), fill="#c0392b")
    p = OUT / "gui_settings.png"
    im.save(p)
    dl = (60, 560, 60 + int(d.textlength("Delete account", font=f(15))), 578)
    return p, {"box": boxes["Send weekly digest"], "delete_box": dl, "size": (W, H)}


# ---------------------------------------------------------------- perception / counting / multi-image
def shapes(name: str, seed: int, n_red_tri: int, n_other: int) -> tuple[Path, dict]:
    rng = random.Random(seed)
    W, H = 800, 600
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    placed: list[tuple[int, int]] = []

    def spot() -> tuple[int, int]:
        while True:
            x, y = rng.randint(40, W - 40), rng.randint(40, H - 40)
            if all((x - a) ** 2 + (y - b) ** 2 > 60 ** 2 for a, b in placed):
                placed.append((x, y))
                return x, y

    def tri(x: int, y: int, col: str) -> None:
        d.polygon([(x, y - 18), (x - 17, y + 13), (x + 17, y + 13)], fill=col)

    for _ in range(n_red_tri):
        tri(*spot(), "#d62728")
    for i in range(n_other):
        x, y = spot()
        kind = i % 3
        if kind == 0:
            tri(x, y, "#1f77b4")
        elif kind == 1:
            d.ellipse((x - 16, y - 16, x + 16, y + 16), fill="#d62728")
        else:
            d.rectangle((x - 15, y - 15, x + 15, y + 15), fill="#2ca02c")
    p = OUT / f"{name}.png"
    im.save(p)
    return p, {"red_triangles": n_red_tri}


def spatial() -> tuple[Path, dict]:
    im = Image.new("RGB", (700, 500), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((430, 80, 510, 160), fill="#2ca02c")      # green circle, upper right
    d.rectangle((150, 320, 230, 400), fill="#1f77b4")   # blue square, lower left
    d.polygon([(330, 230), (300, 285), (360, 285)], fill="#ff7f0e")
    p = OUT / "spatial.png"
    im.save(p)
    return p, {"horiz": "left", "vert": "below"}


def dots(name: str, n: int, seed: int) -> tuple[Path, dict]:
    rng = random.Random(seed)
    im = Image.new("RGB", (500, 400), "white")
    d = ImageDraw.Draw(im)
    placed: list[tuple[int, int]] = []
    while len(placed) < n:
        x, y = rng.randint(30, 470), rng.randint(30, 370)
        if all((x - a) ** 2 + (y - b) ** 2 > 55 ** 2 for a, b in placed):
            placed.append((x, y))
            d.ellipse((x - 14, y - 14, x + 14, y + 14), fill="#7b3fa0")
    p = OUT / f"{name}.png"
    im.save(p)
    return p, {"n": n}


def build_all() -> dict:
    r1 = RECEIPT_ITEMS
    r2 = [(n, 4.15 if n == "Greek yoghurt 500g" else p) for n, p in RECEIPT_ITEMS]
    return {
        "receipt": receipt("receipt", r1, "GL-20481-K"),
        "receipt_b": receipt("receipt_b", r2, "GL-20481-K"),
        "paragraph": paragraph(),
        "table": table(),
        "bar": bar_chart(),
        "line": line_chart(),
        "scatter": scatter(),
        "gui_app": app_window(),
        "gui_settings": settings_form(),
        "shapes": shapes("shapes", 5, 7, 24),
        "spatial": spatial(),
        "dots_a": dots("dots_a", 6, 1),
        "dots_b": dots("dots_b", 8, 2),
    }


if __name__ == "__main__":
    for k, (p, t) in build_all().items():
        print(k, p.name, t)
