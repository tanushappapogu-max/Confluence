import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

alphas = [3.0, 2.0, 1.0]
x = [0, 1, 2]
# means/stds from mycelial_final.py (4 seeds)
mse = {
 "oracle":   ([0.001,0.001,0.001],[0,0,0]),
 "router":   ([0.083,0.506,1.419],[0.007,0.031,0.039]),
 "free (Routing-Free MoE)": ([0.158,0.583,1.319],[0.003,0.024,0.049]),
 "mycelial (ours)": ([0.054,0.392,1.246],[0.007,0.026,0.051]),
}
ill = {
 "router":   ([0.007,0.078,0.295],[0.003,0.010,0.021]),
 "free (Routing-Free MoE)": ([0.008,0.073,0.262],[0.004,0.011,0.019]),
 "mycelial (ours)": ([0.000,0.012,0.103],[0.000,0.002,0.007]),
}
col = {"oracle":"#999","router":"#e07b39","free (Routing-Free MoE)":"#c0392b","mycelial (ours)":"#2d7d46"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for k,(m,s) in mse.items():
    ax[0].errorbar(x, m, yerr=s, marker='o', capsize=3, lw=2, color=col[k], label=k)
ax[0].set_title("Held-out MSE on UNSEEN legal programs\n(systematic generalization; lower=better)")
ax[0].set_ylabel("MSE"); ax[0].set_xticks(x); ax[0].set_xticklabels([f"α={a}" for a in alphas])
ax[0].set_xlabel("routing-hint strength  (weaker →)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
for k,(m,s) in ill.items():
    ax[1].errorbar(x, m, yerr=s, marker='s', capsize=3, lw=2, color=col[k], label=k)
ax[1].set_title("Illegal-transition rate\n(fraction of paths violating the graph; lower=better)")
ax[1].set_ylabel("illegal rate"); ax[1].set_xticks(x); ax[1].set_xticklabels([f"α={a}" for a in alphas])
ax[1].set_xlabel("routing-hint strength  (weaker →)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.suptitle("Mycelial Routing vs Routing-Free MoE — constrained computation paths (4 seeds)", fontweight='bold')
fig.tight_layout()
fig.savefig("mycelial_result.png", dpi=130)
print("saved mycelial_result.png")
