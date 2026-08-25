"""MoE compute-vs-accuracy frontier: conserved-flow routing is Pareto-dominant.
Measured on the MoE-FFN-layer demo (moe_layer_demo.py, 3 seeds; raw log in
results/moe_layer_frontier_3seed.txt). Point = (active experts/layer, held-out MSE);
lower-left is better. The flow router sits at minimum compute AND minimum error, and its
routing respects the legal expert-transition graph while the top-k gate does not.
Saves paper/figs/moe_frontier.png. Run: python3 make_fig_moe.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figs")
os.makedirs(OUT, exist_ok=True)

# measured (results/moe_layer_frontier_3seed.txt): (active/layer, MSE, MSE_std, illegal, label, color)
pts = [
    (8.0, 1.278, 0.030, None,  "dense (all experts)",   "#6b7280"),
    (1.0, 1.294, 0.033, 0.899, "top-1 gate (MoE)",       "#dc2626"),
    (2.0, 1.478, 0.016, 0.364, "top-2 gate (MoE)",       "#f59e0b"),
    (1.0, 1.170, 0.033, 0.150, "conserved flow (ours)",  "#2563eb"),
]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for x, y, e, ill, label, c in pts:
    ax.errorbar(x, y, yerr=e, fmt="o", ms=10, color=c, capsize=3, zorder=3)
    dx = 0.15 if label.startswith("conserved") else 0.15
    va = "top" if label.startswith("top-1") else "bottom"
    tag = label + (f"\nillegal {ill:.2f}" if ill is not None else "")
    ax.annotate(tag, (x, y), xytext=(x + dx, y + (0.02 if va == "bottom" else -0.02)),
                fontsize=8.5, color=c, va=va)
# Pareto arrow toward better
ax.annotate("better", xy=(0.6, 1.14), xytext=(2.3, 1.20), fontsize=9, color="#059669",
            arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5))
ax.set_xlabel("active experts per layer  (compute / FLOPs)")
ax.set_ylabel("held-out MSE  (lower better)")
ax.set_title("MoE routing: accuracy vs compute (coupled regime, 3 seeds)", fontsize=10.5)
ax.set_xlim(0.3, 8.7); ax.set_ylim(1.10, 1.52)
fig.tight_layout()
p = os.path.join(OUT, "moe_frontier.png")
fig.savefig(p, dpi=160, bbox_inches="tight")
print("saved", p)
