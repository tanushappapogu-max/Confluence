"""
DECISIVE experiment: does conservation-coupled flow routing beat parallel
self-activation (Routing-Free MoE) when valid computation PATHS are constrained?

Task: legal programs are PATHS in a transition graph T over primitives:
  primitive m' may follow m only if T[m, m']=1.
The per-slot routing hint is NOISY (strength alpha). To route correctly under a
weak hint the model must respect transition legality.

  MYCELIAL: legality is baked into GRAPH TOPOLOGY -- illegal transitions have no
            edge, so flow can only form LEGAL paths. (ours)
  FREE    : Routing-Free MoE -- each slot self-activates INDEPENDENTLY from its
            hint (+ gets the adjacency as a feature); nothing stops it picking an
            illegal transition. (incumbent)
  ROUTER  : independent softmax gate (+ adjacency feature).
  ORACLE  : perfect routing (ceiling).

All share the SAME experts + SAME encoder (matched capacity). Baselines are even
GIVEN the adjacency as input; the only thing they lack is topological enforcement.
"""
import torch, torch.nn as nn, time, math
torch.manual_seed(0)
d, M, K = 6, 7, 3
NOISE = 0.7                                  # per-slot hint noise scale

# ground-truth non-commuting affine-ish primitives (learned by MLP experts)
true_A = []
for m in range(M):
    g = torch.randn(d, d); q, _ = torch.linalg.qr(g)
    r = torch.linalg.matrix_exp(0.6 * (g - g.t()) / 2)
    true_A.append(q @ r)
true_A = torch.stack(true_A); true_b = 0.3 * torch.randn(M, d)

# random transition graph (legality). density ~0.45
tg = torch.Generator().manual_seed(7)
T = (torch.rand(M, M, generator=tg) < 0.55).float()
T.fill_diagonal_(0.0)                       # no self-loops (forces real transitions)
legal = [(a, b, c) for a in range(M) for b in range(M) for c in range(M)
         if T[a, b] > 0 and T[b, c] > 0]
gp = torch.Generator().manual_seed(2); pm = torch.randperm(len(legal), generator=gp)
ntr = int(0.7 * len(legal))
trp = [legal[i] for i in pm[:ntr]]; tep = [legal[i] for i in pm[ntr:]]
print(f"transition graph: {int(T.sum())}/{M*M} legal transitions | "
      f"legal programs={len(legal)}  train={len(trp)} test(unseen legal)={len(tep)}")

def gen(progs, n):
    x = torch.randn(n, d); idx = torch.randint(0, len(progs), (n,))
    C = torch.tensor([progs[i] for i in idx]); h = x.clone()
    for k in range(K):
        h = torch.einsum('nij,nj->ni', true_A[C[:, k]], h) + true_b[C[:, k]]
    return x, C, h
Xtr, Ctr, Ytr = gen(trp, 6000); Xte, Cte, Yte = gen(tep, 3000)

# ---- layered graph with legality baked into topology (for mycelial) ----
N = 2 + K * M
def eid(k, m): return 2 + k * M + m
edges = []
for m in range(M): edges.append((0, eid(0, m)))                      # source->pos0 (any start)
for k in range(K - 1):
    for m in range(M):
        for mp in range(M):
            if T[m, mp] > 0:                                          # ONLY legal transitions
                edges.append((eid(k, m), eid(k + 1, mp)))
for m in range(M): edges.append((eid(K - 1, m), 1))                  # posK-1->sink
E = len(edges)
inc = torch.zeros(N, E); edge_dst = torch.full((E, 2), -1)
node_edges = torch.zeros(N, E)
for e, (i, j) in enumerate(edges):
    inc[i, e], inc[j, e] = 1.0, -1.0
    node_edges[i, e] = 1.0; node_edges[j, e] = 1.0
    if j >= 2: edge_dst[e] = torch.tensor([(j - 2) // M, (j - 2) % M])
b_inj = torch.zeros(N); b_inj[0] = 1.0; b_inj[1] = -1.0
keep = torch.tensor([i for i in range(N) if i != 1])

def physarum_select(cost, steps=26, dt=0.28, gamma=2.0):
    B = cost.shape[0]; L = torch.ones(B, E)
    kk, mm = edge_dst[:, 0], edge_dst[:, 1]; v = kk >= 0
    L[:, v] = cost[:, kk[v], mm[v]]; L = L.clamp(min=1e-3)
    D = torch.ones(B, E); binj = b_inj.unsqueeze(0).expand(B, -1)
    for _ in range(steps):
        w = D / L
        Lap = torch.einsum('ne,be,me->bnm', inc, w, inc)[:, keep][:, :, keep] + 1e-4 * torch.eye(N - 1)
        p_r = torch.linalg.solve(Lap, binj[:, keep])
        p = torch.zeros(B, N); p[:, keep] = p_r
        Q = w * torch.einsum('ne,bn->be', inc, p)
        D = (D + dt * (Q.abs() - D)).clamp(min=1e-9)
    thru = 0.5 * torch.einsum('ve,be->bv', node_edges, Q.abs())
    a = torch.zeros(B, K, M)
    for k in range(K):
        cols = torch.tensor([eid(k, m) for m in range(M)])
        tk = thru[:, cols].clamp(min=1e-9) ** gamma          # sharpen toward one-hot path
        a[:, k] = tk / (tk.sum(-1, keepdim=True) + 1e-9)
    return a

Tflat = T.reshape(-1)
class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 32), nn.GELU(),
                                                    nn.Linear(32, d)) for _ in range(M)])
        # encoder sees the NOISY per-slot hint AND the full adjacency (fair to baselines)
        self.enc = nn.Sequential(nn.Linear(K * M + M * M, 64), nn.Tanh(), nn.Linear(64, K * M))
        for p in self.enc[-1].parameters(): nn.init.zeros_(p)
        self.temp = nn.Parameter(torch.tensor(0.5)); self.thr = nn.Parameter(torch.tensor(0.0))
    def logits(self, C, alpha):
        B = C.shape[0]
        oh = torch.zeros(B, K, M); oh.scatter_(2, C.unsqueeze(-1), 1.0)
        hint = alpha * oh + NOISE * torch.randn(B, K, M)             # NOISY per-slot hint
        adj = Tflat.unsqueeze(0).expand(B, -1)
        z = hint + self.enc(torch.cat([hint.reshape(B, -1), adj], 1)).reshape(B, K, M)
        return z, oh
    def select(self, z, oh):
        if self.mode == "oracle":   return oh
        if self.mode == "router":   return torch.softmax(z / self.temp.clamp(min=0.1), -1)
        if self.mode == "free":
            act = torch.sigmoid(z - self.thr); return act / (act.sum(-1, keepdim=True) + 1e-9)
        if self.mode == "mycelial":
            return physarum_select(torch.nn.functional.softplus(-z) + 0.05)
    def forward(self, x, C, alpha):
        z, oh = self.logits(C, alpha); a = self.select(z, oh); h = x
        for k in range(K):
            outs = torch.stack([self.experts[m](h) for m in range(M)], 1)
            h = torch.einsum('bm,bmd->bd', a[:, k], outs)
        return h, a

def illegal_rate(a):
    """fraction of argmax paths that use an ILLEGAL transition."""
    pick = a.argmax(-1)                                              # [B,K]
    bad = torch.zeros(pick.shape[0])
    for k in range(K - 1):
        bad = bad + (T[pick[:, k], pick[:, k + 1]] == 0).float()
    return (bad > 0).float().mean().item()

def train(mode, alpha_train=2.5, epochs=60, bs=256, use_balance=False):
    torch.manual_seed(3); net = Model(mode); opt = torch.optim.Adam(net.parameters(), lr=4e-3)
    for ep in range(epochs):
        pi = torch.randperm(6000)
        for b in range(6000 // bs):
            idx = pi[b*bs:(b+1)*bs]
            pred, a = net(Xtr[idx], Ctr[idx], alpha_train)
            loss = ((pred - Ytr[idx])**2).mean()
            if use_balance: loss = loss + 0.1 * ((a.mean((0,1)) - 1/M)**2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return net

def ev(net, alpha):
    net.eval()
    with torch.no_grad():
        p, a = net(Xte, Cte, alpha)
        return ((p - Yte)**2).mean().item(), illegal_rate(a)

print(f"var(Yte)={Yte.var().item():.3f}  | trained at hint alpha=2.0. Lower alpha at test = noisier hint.\n")
alphas = [3.0, 2.0, 1.0]                          # strong -> weak hint
configs = [("oracle", {}), ("router", {}), ("free", dict(use_balance=True)), ("mycelial", {})]
hdr = f"{'variant':<12}{'params':>7}" + "".join(f"MSE@a={a}".rjust(11) for a in alphas) + "".join(f"illeg@{a}".rjust(10) for a in alphas) + f"{'sec':>6}"
print(hdr); print("-" * len(hdr))
for mode, kw in configs:
    t0 = time.time(); net = train(mode, **kw)
    res = [ev(net, a) for a in alphas]
    npar = sum(p.numel() for p in net.parameters())
    line = f"{mode:<12}{npar:>7}" + "".join(f"{m:.4f}".rjust(11) for m, _ in res) + "".join(f"{il:.2f}".rjust(10) for _, il in res) + f"{time.time()-t0:>6.1f}"
    print(line)
print("\nMSE@a  = held-out MSE on UNSEEN legal programs at hint strength a (lower=better)")
print("illeg@a= fraction of selected paths using an ILLEGAL transition (lower=better; oracle=0 by construction)")
