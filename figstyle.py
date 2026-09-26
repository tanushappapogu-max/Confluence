"""Shared figure style for the paper: one forest-green theme, one set of rcParams, one save helper.

Theme: dark forest / forest / sage / mint / lime, with coral red as the contrast color.
Color follows the ROLE, the same in every figure:
  OURS   (conserved flow)                         forest green
  BASE   (main baseline: top-1 gate, independent per-step softmax, unrolled autograd, exact Jacobian)  coral red
  BASE2  (secondary baseline: top-2 gate)         lime
  NEUTRAL (dense layer, reference lines)          gray
OURS / BASE / BASE2 pass the color-vision-deficiency check on all pairs (worst CVD dE 11.5, normal-vision
dE 22.9): the coral is kept much lighter than the forest green, so red/green-blind readers still separate
them by lightness. BASE and BASE2 are low-contrast on white, so every series is also direct-labeled and
uses a distinct marker (circle = ours, square = baseline, triangle = top-2).
The same hex values are defined as LaTeX colors in paper/main.tex.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "forest_dark": "#0f3d2e",   # headings, primary ink
    "forest":      "#166534",   # OURS
    "green":       "#2f855a",
    "sage":        "#5f7a6b",   # axes, secondary ink
    "sage_light":  "#a3b8ab",
    "mint":        "#d8efdf",   # light tint / card fill
    "wash":        "#f3f8f4",   # panel background
    "lime":        "#84a311",   # BASE2 / accent
    "coral":       "#f4978e",   # BASE (series fill)
    "red":         "#c0392b",   # illegal / violation strokes and text
    "gray":        "#8a8984",   # NEUTRAL
    "grid":        "#dde5df",
    "paper":       "#ffffff",
}

OURS = PALETTE["forest"]
BASE = PALETTE["coral"]
BASE2 = PALETTE["lime"]
NEUTRAL = PALETTE["gray"]
INK = PALETTE["forest_dark"]
INK2 = PALETTE["sage"]
GRID = PALETTE["grid"]
MARK = {"ours": "o", "base": "s", "base2": "^", "neutral": "D"}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "text.color": INK,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.5,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name, tight=True):
    """Save vector PDF (used by LaTeX) and PNG (preview) into paper/figs/."""
    if tight:
        fig.tight_layout()
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        paths.append(p)
    print("saved", *paths)
    return paths
