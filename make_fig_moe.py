"""MoE compute-vs-accuracy frontier: conserved-flow routing is Pareto-dominant.
Measured on the MoE-FFN-layer demo (moe_layer_demo.py, 3 seeds; raw log in
results/moe_layer_frontier_3seed.txt). Point = (active experts/layer, held-out MSE);
lower-left is better. The flow router sits at minimum compute AND minimum error, and its
routing respects the legal expert-transition graph while the top-k gate does not.
Saves paper/figs/moe_frontier.{pdf,png}. Run: python3 make_fig_moe.py
"""
import matplotlib.pyplot as plt
from figstyle import OURS, BASE, BASE2, NEUTRAL, INK2, save

# measured (results/moe_layer_frontier_3seed.txt): (active/layer, MSE, MSE_std, illegal, label, color, text offset)
pts = [
    (8.0, 1.278, 0.030, None,  "dense (all 8 experts)", NEUTRAL, (-0.3, 0.035), "right"),
    (1.0, 1.294, 0.033, 0.899, "top-1 gate",            BASE,    (0.25, 0.0),   "left"),
    (2.0, 1.478, 0.016, 0.364, "top-2 gate",            BASE2,   (0.25, 0.0),   "left"),
    (1.0, 1.170, 0.033, 0.150, "conserved flow (ours)", OURS,    (0.25, 0.0),   "left"),
]

fig, ax = plt.subplots(figsize=(3.5, 2.7))
for x, y, e, ill, label, c, (dx, dy), ha in pts:
    ax.errorbar(x, y, yerr=e, fmt="o", ms=7, color=c, capsize=2.5, zorder=3, mec="white", mew=1.2)
    tag = label + (f"\nillegal {ill:.2f}" if ill is not None else "")
    ax.text(x + dx, y + dy, tag, fontsize=7.5, color=INK2, va="center", ha=ha)
ax.set_xlabel("active experts per layer (compute)")
ax.set_ylabel("held-out MSE (lower is better)")
ax.set_xlim(0.4, 8.6); ax.set_ylim(1.10, 1.52)
ax.set_xticks([1, 2, 4, 6, 8])
save(fig, "moe_frontier")
