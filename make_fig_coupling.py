"""Centerpiece figure: WHEN does conserved-flow routing help?
Measured on the constrained-routing density sweep (density_sweep.py, 3 seeds, K=3, M=8;
raw log in results/density_sweep_flow_vs_independent.txt). Flow = conserved-flow router
(mycelial); Independent = per-step softmax (router) -- the same class of baseline that
TIES the flow model on MetaQA 2-hop. As constraint density rises (reasoning steps become
coupled), the flow's advantage over independent scoring grows and crosses zero; at
density 1.0 (unconstrained / marginalizable, like MetaQA 2-hop) they tie.
Saves paper/figs/coupling.png. Run: python3 make_fig_coupling.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figs")
os.makedirs(OUT, exist_ok=True)

# --- measured numbers (results/density_sweep_flow_vs_independent.txt) ---
density   = [0.35, 0.50, 0.65, 0.80, 1.00]
mse_flow  = [0.586, 0.405, 0.460, 0.445, 0.509]   # MSE_myc
mse_indep = [0.745, 0.476, 0.520, 0.478, 0.490]   # MSE_rout
adv = [i - f for i, f in zip(mse_indep, mse_flow)] # flow advantage (independent - flow), >0 = flow wins

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
C_FLOW, C_IND = "#2563eb", "#dc2626"

ax[0].plot(density, mse_flow, "o-", color=C_FLOW, label="conserved flow (ours)")
ax[0].plot(density, mse_indep, "s--", color=C_IND, label="independent per-step softmax")
ax[0].set_xlabel("transition-graph density  (1.0 = unconstrained)")
ax[0].set_ylabel("held-out MSE  (lower better)")
ax[0].set_title("(a) accuracy vs coupling", fontsize=10)
ax[0].invert_xaxis()  # left = more coupled
ax[0].legend(frameon=False, fontsize=8)

ax[1].axhline(0, color="#9ca3af", lw=1)
ax[1].plot(density, adv, "o-", color=C_FLOW)
ax[1].fill_between(density, adv, 0, where=[a > 0 for a in adv], color=C_FLOW, alpha=0.12)
ax[1].set_xlabel("transition-graph density")
ax[1].set_ylabel("flow advantage  (indep. − flow MSE)")
ax[1].set_title("(b) flow wins iff steps are coupled", fontsize=10)
ax[1].invert_xaxis()
ax[1].annotate("MetaQA 2-hop\n(marginalizable → tie)", xy=(1.0, adv[-1]), xytext=(0.9, 0.09),
               fontsize=8, color="#6b7280", ha="center",
               arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=1))
ax[1].text(0.36, 0.13, "coupled → flow wins", fontsize=8, color=C_FLOW)

fig.tight_layout()
p = os.path.join(OUT, "coupling.png")
fig.savefig(p, dpi=160, bbox_inches="tight")
print("saved", p, "| advantage by density:", [round(a, 3) for a in adv])
