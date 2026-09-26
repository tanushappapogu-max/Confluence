import os
os.environ["MPLCONFIGDIR"] = os.path.dirname(os.path.abspath(__file__))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

dens = [0.35, 0.50, 0.65, 0.80, 1.00]
gap  = [0.397, 0.166, 0.124, 0.116, 0.060]      # free - mycelial MSE
ill_m= [0.024, 0.019, 0.019, 0.016, 0.000]
ill_f= [0.377, 0.077, 0.066, 0.058, 0.000]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3)); fig.patch.set_facecolor("white")
ax[0].plot(dens, gap, "o-", lw=2.4, color="#2d7d46")
ax[0].axhline(0, color="#999", lw=1, ls="--")
ax[0].fill_between(dens, 0, gap, color="#2d7d46", alpha=0.12)
ax[0].set_title("Mycelial's advantage scales with constraint strength", fontweight="bold", fontsize=11)
ax[0].set_xlabel("transition-graph density  (→ less constrained)"); ax[0].set_ylabel("MSE gap  (free − mycelial)")
ax[0].invert_xaxis(); ax[0].grid(alpha=.3)
ax[0].annotate("more constrained\n= bigger win", (0.35, 0.397), (0.55, 0.33), fontsize=8.5,
               color="#2d7d46", arrowprops=dict(arrowstyle="->", color="#2d7d46"))
ax[0].annotate("unconstrained\n≈ tie", (1.0, 0.060), (0.86, 0.16), fontsize=8.5,
               color="#666", arrowprops=dict(arrowstyle="->", color="#888"))

ax[1].plot(dens, ill_f, "s-", lw=2.2, color="#c0392b", label="free (Routing-Free MoE)")
ax[1].plot(dens, ill_m, "o-", lw=2.2, color="#2d7d46", label="mycelial (ours)")
ax[1].set_title("Illegal-transition rate (structural guarantee)", fontweight="bold", fontsize=11)
ax[1].set_xlabel("transition-graph density  (→ less constrained)"); ax[1].set_ylabel("illegal-transition rate")
ax[1].invert_xaxis(); ax[1].grid(alpha=.3); ax[1].legend(fontsize=9)
fig.suptitle("Density sweep — 3 seeds, MSE at hint α=2.0", fontweight="bold")
fig.tight_layout()
fig.savefig("mycelial_density.png", dpi=140, facecolor="white", bbox_inches="tight")
print("saved mycelial_density.png")
