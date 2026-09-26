"""
Go/no-go spike for Mycelial Routing.

Tests the ONE thing the whole proposal lives or dies on:
can a differentiable Physarum flow-reinforcement dynamic converge to a
STABLE, SPARSE, near-single path -- and can we backprop a task loss
through that emergent path to CHOOSE which path forms?

Tero-Nakagaki model:
  flux   Q_ij = (D_ij / L_ij) * (p_i - p_j)
  Kirchhoff conservation at every node: sum_j Q_ij = b_i
     (b = +I0 at source, -I0 at sink, 0 elsewhere)
  -> solve weighted-Laplacian linear system for pressures p given conductances D
  reinforcement: D_ij <- D_ij + dt*( g(|Q_ij|) - D_ij ),  g(x)=x
     high-flux edges thicken, unused edges wither.

Everything is torch so gradients flow through the fixed point.
"""
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def solve_pressures(D, L, inc, b, sink):
    """Weighted-Laplacian solve. D: [E] conductances, L: [E] lengths,
    inc: [N,E] signed incidence, b: [N] source/sink injection, sink: int (grounded)."""
    w = D / L                                   # [E] edge weights
    Lap = inc @ torch.diag(w) @ inc.t()         # [N,N] graph Laplacian
    # ground the sink (pressures defined up to a constant) -> drop its row/col
    N = Lap.shape[0]
    keep = [i for i in range(N) if i != sink]
    idx = torch.tensor(keep)
    Lap_r = Lap[idx][:, idx]
    b_r = b[idx]
    p_r = torch.linalg.solve(Lap_r, b_r)
    p = torch.zeros(N, dtype=D.dtype)
    p[idx] = p_r
    return p


def flux(D, L, inc, p):
    dp = inc.t() @ p                            # [E] pressure drop across each edge
    return (D / L) * dp                         # [E] signed flux


def physarum(inc, L, b, sink, steps=400, dt=0.05, D0=1.0):
    """Run the reinforcement dynamic to a fixed point. Differentiable in L."""
    E = inc.shape[1]
    D = torch.full((E,), D0, dtype=L.dtype)
    for _ in range(steps):
        p = solve_pressures(D, L, inc, b, sink)
        Q = flux(D, L, inc, p)
        D = D + dt * (Q.abs() - D)              # Tero update, g(x)=x
        D = D.clamp(min=1e-9)
    return D, Q


def build(edges, N):
    """Signed incidence [N,E] from undirected edge list (orient low->high)."""
    E = len(edges)
    inc = torch.zeros(N, E)
    for e, (i, j) in enumerate(edges):
        inc[i, e] = 1.0
        inc[j, e] = -1.0
    return inc


def report(name, edges, D, thresh=0.05):
    surv = [(edges[e], round(D[e].item(), 3)) for e in range(len(edges)) if D[e] > thresh]
    print(f"  surviving edges (D>{thresh}): {surv}")


print("=" * 68)
print("TEST 1  — does a single path EMERGE, and is it the OPTIMAL one?")
print("=" * 68)
# Two routes from 0 to 4:  short 0-1-4 (2 edges)  vs  long 0-2-3-4 (3 edges)
edges = [(0, 1), (1, 4), (0, 2), (2, 3), (3, 4)]
N = 5
inc = build(edges, N)
L = torch.ones(len(edges))
b = torch.zeros(N); b[0] = 1.0; b[4] = -1.0
D, Q = physarum(inc, L, b, sink=4)
print("Graph: short route 0-1-4  vs  long route 0-2-3-4 (all unit length)")
report("t1", edges, D)
short = min(D[0].item(), D[1].item())
long_ = max(D[2].item(), D[3].item(), D[4].item())
print(f"  short-route conductance ~{short:.3f} | long-route conductance ~{long_:.3f}")
print(f"  VERDICT: {'short path won, long withered -> optimal single path emerged' if short > 0.5 and long_ < 0.1 else 'NO clean selection'}")

print()
print("=" * 68)
print("TEST 2  — many parallel routes: does it collapse to ONE sparse path?")
print("=" * 68)
# 3x3 grid-ish mesh, source corner 0 -> sink corner 8, lots of redundant routes
mesh = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),
        (0,3),(3,6),(1,4),(4,7),(2,5),(5,8),
        (0,4),(4,8)]  # + two diagonals (shortcuts)
N = 9
inc = build(mesh, N)
L = torch.ones(len(mesh))
# make the diagonal shortcuts geometrically longer (sqrt2) so they're a fair race
L[-2] = 2.0 ** 0.5; L[-1] = 2.0 ** 0.5
b = torch.zeros(N); b[0] = 1.0; b[8] = -1.0
D, Q = physarum(inc, L, b, sink=8)
n_surv = int((D > 0.05).sum().item())
print(f"  {len(mesh)} candidate edges -> {n_surv} survive above threshold")
report("t2", mesh, D)
print(f"  VERDICT: {'collapsed to a sparse path (not a diffuse web)' if n_surv <= 4 else 'stayed diffuse'}")

print()
print("=" * 68)
print("TEST 3  — is it DIFFERENTIABLE + TRAINABLE end-to-end?")
print("  Can a task loss backprop through the emergent path and FORCE")
print("  a DIFFERENT path to win than physics alone would choose?")
print("=" * 68)
# Reuse TEST-1 graph. Physics prefers the SHORT path (0-1-4).
# Task: we WANT flow to go through the LONG route's edge (2,3).
# Learn per-edge length multipliers so the emergent path routes through it.
edges = [(0, 1), (1, 4), (0, 2), (2, 3), (3, 4)]
N = 5
inc = build(edges, N)
b = torch.zeros(N); b[0] = 1.0; b[4] = -1.0
log_mult = torch.zeros(len(edges), requires_grad=True)   # learnable length multipliers
target_edge = 3  # edge (2,3) on the long route
opt = torch.optim.Adam([log_mult], lr=0.15)

print("  Physics-only preference is the SHORT path. Training to route through")
print("  the LONG path's edge (2,3) purely by gradient through the flow fixed point.\n")
for it in range(60):
    opt.zero_grad()
    L = torch.exp(log_mult)                       # positive lengths, differentiable
    D, Q = physarum(inc, L, b, sink=4, steps=120, dt=0.08)
    # loss: maximize conductance on the target edge (want it to survive)
    loss = -D[target_edge]
    loss.backward()
    opt.step()
    if it % 15 == 0 or it == 59:
        with torch.no_grad():
            print(f"  it{it:3d}  D(target 2-3)={D[target_edge].item():.3f}  "
                  f"D(short 0-1)={D[0].item():.3f}  grad_ok={log_mult.grad.abs().sum().item()>0}")

with torch.no_grad():
    L = torch.exp(log_mult)
    D, Q = physarum(inc, L, b, sink=4, steps=200, dt=0.08)
    won = D[target_edge].item() > D[0].item()
print(f"\n  final: D(long edge 2-3)={D[target_edge].item():.3f}  D(short edge 0-1)={D[0].item():.3f}")
print(f"  VERDICT: {'gradients through the fixed point RE-ROUTED the emergent path -> trainable' if won else 'training FAILED to re-route'}")
