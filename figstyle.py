"""Shared figure style for the paper: one palette, one set of rcParams, one save helper.

Color follows the ROLE, the same in every figure:
  OURS   (conserved flow)                      blue
  BASE   (main baseline: top-1 gate, independent per-step softmax, unrolled autograd, exact Jacobian)  orange
  BASE2  (secondary baseline: top-2 gate)      aqua
  NEUTRAL (dense layer, reference lines)       gray
Palette validated for color-vision deficiency (all pairs of OURS/BASE/BASE2: CVD dE >= 9.2).
BASE2 is low-contrast on white, so every series is also direct-labeled.
The same hex values are defined as LaTeX colors in paper/main.tex (flowblue, baseorange, ...).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OURS = "#2a78d6"
BASE = "#eb6834"
BASE2 = "#1baf7a"
NEUTRAL = "#8a8984"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
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
})


def save(fig, name):
    """Save vector PDF (used by LaTeX) and PNG (preview) into paper/figs/."""
    fig.tight_layout()
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        paths.append(p)
    print("saved", *paths)
    return paths
