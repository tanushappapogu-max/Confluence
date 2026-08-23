"""Figure for the Scalability section: implicit differentiation through the flow fixed point.
Three panels, all from live measurement (no synthetic/hand-drawn numbers):
  (a) correctness  -- relative error of the implicit gradient vs finite-difference gold, per N
  (b) memory       -- peak traced backward memory vs solver depth: unroll grows, implicit flat
  (c) scaling      -- wall-clock of exact-Jacobian vs matrix-free implicit gradient vs #edges
Saves paper/figs/implicit_diff.png. Run: python3 make_fig_implicit.py
"""
import os, time, tracemalloc, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import implicit_diff as m
torch.set_default_dtype(torch.float64)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figs")
os.makedirs(OUT, exist_ok=True)

# ---- (a) correctness: implicit vs finite-difference gold, across graph sizes ----
Ns, rels = [], []
for seed in range(6):
    N = 8 + seed * 4
    B, b, keep, E = m.random_graph(N, extra_edges=N, seed=seed)
    torch.manual_seed(100 + seed); theta = 0.5 * torch.randn(E); target = 0.5 * torch.rand(E)
    g_fd = m.grad_numeric(theta, B, b, keep, target)
    g_im, _ = m.grad_implicit(theta, B, b, keep, 4000, target)
    rel = ((g_fd - g_im).norm() / (g_fd.norm() + 1e-12)).item()
    Ns.append(N); rels.append(max(rel, 1e-12))
print("correctness rel-errs:", [f"{r:.1e}" for r in rels])

# ---- (b) memory vs solver depth ----
Bm, bm, keepm, Em = m.random_graph(30, extra_edges=60, seed=0)
torch.manual_seed(0); thv = 0.5 * torch.randn(Em); tgv = 0.5 * torch.rand(Em)
def peak(fn):
    tracemalloc.start(); fn(); _, pk = tracemalloc.get_traced_memory(); tracemalloc.stop(); return pk / 1e6
m.grad_unroll(thv, Bm, bm, keepm, 50, tgv); m.grad_implicit(thv, Bm, bm, keepm, 50, tgv)  # warm
steps = [50, 100, 200, 400]
mem_un = [peak(lambda s=s: m.grad_unroll(thv, Bm, bm, keepm, s, tgv)) for s in steps]
mem_im = [peak(lambda s=s: m.grad_implicit(thv, Bm, bm, keepm, s, tgv)) for s in steps]
print("mem unroll:", [f"{x:.2f}" for x in mem_un], "| implicit:", [f"{x:.2f}" for x in mem_im])

# ---- (c) wall-clock: exact-Jacobian vs matrix-free, across #edges ----
Es, t_ex, t_fr = [], [], []
for N in [40, 80, 160, 300]:
    B2, b2, keep2, E2 = m.random_graph(N, extra_edges=2 * N, seed=N)
    torch.manual_seed(N); th2 = 0.5 * torch.randn(E2); tg2 = 0.5 * torch.rand(E2)
    t = time.time(); m.grad_implicit(th2, B2, b2, keep2, 3000, tg2); te = (time.time() - t) * 1e3
    t = time.time(); m.grad_implicit_free(th2, B2, b2, keep2, 3000, tg2); tf = (time.time() - t) * 1e3
    Es.append(E2); t_ex.append(te); t_fr.append(tf)
print("edges:", Es, "| exact ms:", [f"{x:.0f}" for x in t_ex], "| free ms:", [f"{x:.0f}" for x in t_fr])

# ---- plot ----
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
C_IM, C_UN, C_FR = "#2563eb", "#dc2626", "#059669"

ax[0].semilogy(Ns, rels, "o-", color=C_IM)
ax[0].axhline(1e-6, ls="--", color="#9ca3af", lw=1)
ax[0].set_title("(a) gradient correctness", fontsize=10)
ax[0].set_xlabel("graph size $N$"); ax[0].set_ylabel("rel. error vs finite-diff")
ax[0].set_ylim(1e-10, 1e-2); ax[0].text(Ns[-1], 2e-6, "machine-precision band", ha="right", va="bottom", color="#6b7280", fontsize=8)

ax[1].plot(steps, mem_un, "s-", color=C_UN, label="unroll (autograd)")
ax[1].plot(steps, mem_im, "o-", color=C_IM, label="implicit (ours)")
ax[1].set_title("(b) backward memory", fontsize=10)
ax[1].set_xlabel("solver steps"); ax[1].set_ylabel("peak memory (MB)"); ax[1].legend(frameon=False, fontsize=8)

ax[2].plot(Es, t_ex, "s-", color="#7c3aed", label="exact Jacobian $O(E^2)$")
ax[2].plot(Es, t_fr, "o-", color=C_FR, label="matrix-free (ours)")
ax[2].set_title("(c) gradient wall-clock", fontsize=10)
ax[2].set_xlabel("edges $E$"); ax[2].set_ylabel("time (ms)"); ax[2].legend(frameon=False, fontsize=8)

fig.tight_layout()
p = os.path.join(OUT, "implicit_diff.png")
fig.savefig(p, dpi=160, bbox_inches="tight")
print("saved", p)
