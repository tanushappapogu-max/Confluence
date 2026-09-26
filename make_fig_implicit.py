"""Figure for the Scalability section: implicit differentiation through the flow fixed point.
Three panels, all from live measurement (no synthetic/hand-drawn numbers):
  (a) correctness  -- relative error of the implicit gradient vs finite-difference gold, per N
  (b) memory       -- peak traced backward memory vs solver depth: unroll grows, implicit flat
  (c) scaling      -- wall-clock of exact-Jacobian vs matrix-free implicit gradient vs #edges
Saves paper/figs/implicit_diff.{pdf,png}. Run: python3 make_fig_implicit.py
"""
import time, tracemalloc, torch
import matplotlib.pyplot as plt
from figstyle import OURS, BASE, NEUTRAL, INK2, save
import implicit_diff as m
torch.set_default_dtype(torch.float64)

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
fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.2))

ax[0].semilogy(Ns, rels, "o-", color=OURS)
ax[0].axhline(1e-6, ls="--", color=NEUTRAL, lw=1)
ax[0].set_title("(a) gradient correctness")
ax[0].set_xlabel("graph size $N$"); ax[0].set_ylabel("rel. error vs. finite diff.")
ax[0].set_ylim(1e-10, 1e-2)
ax[0].text(Ns[-1], 2e-6, "$10^{-6}$", ha="right", va="bottom", color=INK2, fontsize=7.5)

ax[1].plot(steps, mem_un, "s-", color=BASE)
ax[1].plot(steps, mem_im, "o-", color=OURS)
ax[1].text(steps[-1] * 1.04, mem_un[-1], "unrolled", color=INK2, fontsize=7.5, ha="left", va="center")
ax[1].text(steps[-1] * 1.04, mem_im[-1], "implicit\n(ours)", color=INK2, fontsize=7.5, ha="left", va="bottom")
ax[1].set_xlim(0, steps[-1] * 1.45); ax[1].set_ylim(0, max(mem_un) * 1.1)
ax[1].set_title("(b) backward memory")
ax[1].set_xlabel("solver steps"); ax[1].set_ylabel("peak memory (MB)")

ax[2].plot(Es, t_ex, "s-", color=BASE)
ax[2].plot(Es, t_fr, "o-", color=OURS)
ax[2].text(Es[-1] * 1.04, t_ex[-1], "exact\nJacobian", color=INK2, fontsize=7.5, ha="left", va="center")
ax[2].text(Es[-1] * 1.04, t_fr[-1], "matrix-\nfree (ours)", color=INK2, fontsize=7.5, ha="left", va="center")
ax[2].set_xlim(0, Es[-1] * 1.45)
ax[2].set_title("(c) gradient wall-clock")
ax[2].set_xlabel("edges $E$"); ax[2].set_ylabel("time (ms)")

save(fig, "implicit_diff")
