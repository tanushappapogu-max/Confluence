import os
os.environ["MPLCONFIGDIR"] = os.path.dirname(os.path.abspath(__file__))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = [3.0, 2.0, 1.0]; x = [0, 1, 2]
mse = {
 "oracle":              ([0.002,0.002,0.002], "#999"),
 "router":              ([0.138,0.677,2.314], "#e07b39"),
 "pathmoe-lite (theirs)":([0.057,0.496,2.148], "#8e44ad"),
 "free (soft-penalty)": ([0.211,0.769,2.231], "#c0392b"),
 "confluence (ours)":   ([0.082,0.557,2.110], "#2d7d46"),
}
ill = {
 "router":              ([0.004,0.042,0.186], "#e07b39"),
 "pathmoe-lite (theirs)":([0.007,0.118,0.512], "#8e44ad"),
 "free (soft-penalty)": ([0.005,0.043,0.170], "#c0392b"),
 "confluence (ours)":   ([0.000,0.006,0.058], "#2d7d46"),
}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3)); fig.patch.set_facecolor("white")
for k,(y,c) in mse.items():
    ax[0].plot(x, y, "o-", lw=2.2, color=c, label=k)
ax[0].set_title("Held-out accuracy on UNSEEN retrieve+compute programs", fontweight="bold", fontsize=10.5)
ax[0].set_ylabel("MSE (lower=better)"); ax[0].set_xticks(x); ax[0].set_xticklabels([f"α={v}" for v in a])
ax[0].set_xlabel("routing-hint strength (→ noisier)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
for k,(y,c) in ill.items():
    ax[1].plot(x, y, "s-", lw=2.2, color=c, label=k)
ax[1].set_title("Illegal-hop rate (invalid retrieve/compute step)", fontweight="bold", fontsize=10.5)
ax[1].set_ylabel("illegal-hop rate (lower=better)"); ax[1].set_xticks(x); ax[1].set_xticklabels([f"α={v}" for v in a])
ax[1].set_xlabel("routing-hint strength (→ noisier)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[1].annotate("ours ≈ 0\nby construction", (2, 0.047), (0.7, 0.12), fontsize=8.5, color="#2d7d46",
               arrowprops=dict(arrowstyle="->", color="#2d7d46"))
fig.suptitle("Confluence go/no-go: one flow through DATA + COMPUTE nodes (3 seeds)  —  GO", fontweight="bold")
fig.tight_layout(); fig.savefig("confluence_result.png", dpi=140, facecolor="white", bbox_inches="tight")
print("saved confluence_result.png")
