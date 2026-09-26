"""Centerpiece figure: WHEN does conserved-flow routing help?
Measured on the constrained-routing density sweep (density_sweep.py, 3 seeds, K=3, M=8;
raw log in results/density_sweep_flow_vs_independent.txt). Flow = conserved-flow router
(mycelial); Independent = per-step softmax (router) -- the same class of baseline that
TIES the flow model on MetaQA 2-hop. As constraint density rises (reasoning steps become
coupled), the flow's advantage over independent scoring grows and crosses zero; at
density 1.0 (unconstrained / marginalizable, like MetaQA 2-hop) they tie.
Saves paper/figs/coupling.png. Run: python3 make_fig_coupling.py
"""
import matplotlib.pyplot as plt
from figstyle import OURS, BASE, NEUTRAL, INK2, save

# --- measured numbers (results/density_sweep_flow_vs_independent.txt) ---
density   = [0.35, 0.50, 0.65, 0.80, 1.00]
mse_flow  = [0.586, 0.405, 0.460, 0.445, 0.509]   # MSE_myc
mse_indep = [0.745, 0.476, 0.520, 0.478, 0.490]   # MSE_rout
adv = [i - f for i, f in zip(mse_indep, mse_flow)] # flow advantage (independent - flow), >0 = flow wins

fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))

ax[0].plot(density, mse_indep, "s-", color=BASE)
ax[0].plot(density, mse_flow, "o-", color=OURS)
ax[0].text(0.73, 0.555, "independent\nper-step softmax", color=INK2, fontsize=7.5, va="bottom", ha="center")
ax[0].text(0.80, 0.425, "conserved flow (ours)", color=INK2, fontsize=7.5, va="top", ha="center")
ax[0].set_xlabel("transition-graph density (1.0 = unconstrained)")
ax[0].set_ylabel("held-out MSE (lower is better)")
ax[0].set_title("(a) error vs. coupling")
ax[0].invert_xaxis()  # left = more coupled

ax[1].axhline(0, color=NEUTRAL, lw=1)
ax[1].plot(density, adv, "o-", color=OURS)
ax[1].fill_between(density, adv, 0, where=[a > 0 for a in adv], color=OURS, alpha=0.12, lw=0)
ax[1].set_xlabel("transition-graph density")
ax[1].set_ylabel("flow advantage (indep. $-$ flow MSE)")
ax[1].set_title("(b) flow wins iff steps are coupled")
ax[1].invert_xaxis()
ax[1].annotate("MetaQA 2-hop regime\n(marginalizable: tie)", xy=(1.0, adv[-1]), xytext=(0.86, 0.10),
               fontsize=7.5, color=INK2, ha="center",
               arrowprops=dict(arrowstyle="->", color=NEUTRAL, lw=1))
ax[1].text(0.62, 0.125, "coupled:\nflow wins", fontsize=7.5, color=INK2, ha="center")

save(fig, "coupling")
print("advantage by density:", [round(a, 3) for a in adv])
