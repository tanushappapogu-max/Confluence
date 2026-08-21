"""
Mycelial Routing -- full architecture spike vs. Routing-Free MoE.

Task: ordered composition of NON-COMMUTING affine primitives.
  y = A_{c3} ( A_{c2} ( A_{c1} x ) ),  program c=(c1,c2,c3) drawn from M primitives.
  Order matters (affines don't commute). Experts are SHARED across positions
  (same primitive bank reused), so the test is systematic reuse.

Compositional-generalization split: train on a subset of programs,
TEST on HELD-OUT programs (unseen orderings of the same primitives).

Three selection mechanisms share the SAME experts + SAME context encoder
(matched capacity); they differ ONLY in how per-position selection a_k is formed:

  ROUTER  : a_k = softmax(z_k)            (standard learned gate)
  FREE    : a_k ~ sigmoid(z_k - thr), renorm; INDEPENDENT per expert;
            needs a load-balance loss   (Routing-Free MoE incumbent, B1)
  MYCELIAL: a_k = normalized node-throughput of a CONSERVATION-COUPLED
            Physarum flow over the layered (position x primitive) graph (ours, B2)

We also inject noise into z to test robustness: independent self-activation (FREE)
should degrade faster than a globally flow-conserved selection (MYCELIAL).
"""
import torch, torch.nn as nn, time, math
torch.manual_seed(0)
dev = "cpu"
F32 = torch.float32

# ----------------------------- data -----------------------------------------
d, M, K = 6, 4, 3            # dim, #primitives, path length
# ground-truth non-commuting affine primitives (near-orthogonal, distinct)
true_A = []
for m in range(M):
    g = torch.randn(d, d)
    q, _ = torch.linalg.qr(g)          # orthogonal -> stable composition
    r = torch.linalg.matrix_exp(0.6 * (g - g.t()) / 2)  # extra rotation, non-commuting
    true_A.append((q @ r))
true_A = torch.stack(true_A)           # [M,d,d]
true_b = 0.3 * torch.randn(M, d)

all_programs = [(a, b, c) for a in range(M) for b in range(M) for c in range(M)]  # M^K=64
g = torch.Generator().manual_seed(1); perm = torch.randperm(len(all_programs), generator=g)
n_train = 44
train_progs = [all_programs[i] for i in perm[:n_train]]
test_progs  = [all_programs[i] for i in perm[n_train:]]   # 20 UNSEEN orderings

def gen(progs, n):
    x = torch.randn(n, d)
    idx = torch.randint(0, len(progs), (n,))
    C = torch.tensor([progs[i] for i in idx])     # [n,K]
    h = x.clone()
    for k in range(K):
        Ak = true_A[C[:, k]]; bk = true_b[C[:, k]]
        h = torch.einsum('nij,nj->ni', Ak, h) + bk
    return x, C, h

Xtr, Ctr, Ytr = gen(train_progs, 6000)
Xte, Cte, Yte = gen(test_progs, 3000)          # held-out PROGRAMS

# --------------------- layered graph for Physarum ---------------------------
# nodes: 0=source, 1=sink, expert (k,m) -> 2 + k*M + m
N = 2 + K * M
def eid(k, m): return 2 + k * M + m
edges = []
for m in range(M): edges.append((0, eid(0, m)))                       # source -> pos0
for k in range(K - 1):
    for m in range(M):
        for mp in range(M): edges.append((eid(k, m), eid(k + 1, mp)))  # pos k -> k+1
for m in range(M): edges.append((eid(K - 1, m), 1))                   # posK-1 -> sink
E = len(edges)
inc = torch.zeros(N, E)
for e, (i, j) in enumerate(edges):
    inc[i, e], inc[j, e] = 1.0, -1.0
# map each edge to the (position,primitive) of its DESTINATION expert (for cost lookup)
edge_dst_kp = torch.full((E, 2), -1)
for e, (i, j) in enumerate(edges):
    if j >= 2:
        kk = (j - 2) // M; mm = (j - 2) % M
        edge_dst_kp[e] = torch.tensor([kk, mm])
# incident-edge mask per expert node, for throughput
node_edges = torch.zeros(N, E)
for e, (i, j) in enumerate(edges):
    node_edges[i, e] = 1.0; node_edges[j, e] = 1.0

b_inj = torch.zeros(N); b_inj[0] = 1.0; b_inj[1] = -1.0
keep = [i for i in range(N) if i != 1]           # ground the sink
keep_t = torch.tensor(keep)

def physarum_select(cost, steps=14, dt=0.18):
    """cost: [B,K,M] positive edge cost per (position,primitive).
    returns a: [B,K,M] per-position selection simplex (conservation-coupled)."""
    B = cost.shape[0]
    # gather per-edge cost from destination (k,m); source-edges use pos0 dest already covered
    L = torch.ones(B, E)
    kk = edge_dst_kp[:, 0]; mm = edge_dst_kp[:, 1]
    valid = kk >= 0
    L[:, valid] = cost[:, kk[valid], mm[valid]]
    L = L.clamp(min=1e-3)
    D = torch.ones(B, E)
    binj = b_inj.unsqueeze(0).expand(B, -1)
    for _ in range(steps):
        w = D / L                                              # [B,E]
        Lap = torch.einsum('ne,be,me->bnm', inc, w, inc)       # [B,N,N]
        Lap_r = Lap[:, keep_t][:, :, keep_t]
        Lap_r = Lap_r + 1e-4 * torch.eye(N - 1)
        p_r = torch.linalg.solve(Lap_r, binj[:, keep_t])
        p = torch.zeros(B, N); p[:, keep_t] = p_r
        Q = w * (torch.einsum('ne,bn->be', inc, p))            # [B,E] flux
        D = (D + dt * (Q.abs() - D)).clamp(min=1e-9)
    thru = 0.5 * torch.einsum('ve,be->bv', node_edges, Q.abs())  # [B,N] node throughput
    # per position, gather the M expert nodes and normalize
    a = torch.zeros(B, K, M)
    for k in range(K):
        cols = torch.tensor([eid(k, m) for m in range(M)])
        tk = thru[:, cols]
        a[:, k, :] = tk / (tk.sum(-1, keepdim=True) + 1e-9)
    return a

# ------------------------------- model --------------------------------------
class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 32), nn.GELU(),
                                                    nn.Linear(32, d)) for _ in range(M)])  # shared bank
        # routing signal is (near-)decodable from the program; all variants get it.
        # base 5*onehot makes CLEAN routing learnable; enc learns small corrections.
        self.enc = nn.Sequential(nn.Linear(K * M, 32), nn.Tanh(), nn.Linear(32, K * M))
        for p in self.enc[-1].parameters(): nn.init.zeros_(p)
        self.temp = nn.Parameter(torch.tensor(0.5))
        self.thr = nn.Parameter(torch.tensor(0.0))
    def logits(self, C, noise):
        B = C.shape[0]
        oh = torch.zeros(B, K, M); oh.scatter_(2, C.unsqueeze(-1), 1.0)
        z = 5.0 * oh + self.enc(oh.reshape(B, -1)).reshape(B, K, M)
        if noise > 0: z = z + noise * torch.randn_like(z)
        return z, oh
    def select(self, z, oh):
        if self.mode == "oracle":
            return oh                                         # ceiling: perfect routing
        if self.mode == "router":
            return torch.softmax(z / self.temp.clamp(min=0.1), dim=-1)
        if self.mode == "free":
            act = torch.sigmoid(z - self.thr)                 # independent self-activation
            return act / (act.sum(-1, keepdim=True) + 1e-9)
        if self.mode == "mycelial":
            cost = torch.nn.functional.softplus(-z) + 0.05    # low cost where logit high
            return physarum_select(cost)
    def forward(self, x, C, noise=0.0):
        z, oh = self.logits(C, noise)
        a = self.select(z, oh)                                # [B,K,M]
        h = x
        for k in range(K):
            outs = torch.stack([self.experts[m](h) for m in range(M)], 1)  # [B,M,d]
            h = torch.einsum('bm,bmd->bd', a[:, k, :], outs)  # sequential compose
        return h, a

def balance_loss(a):                      # encourage uniform primitive usage
    u = a.mean(dim=(0, 1)); return ((u - 1.0 / M) ** 2).sum()

def train(mode, epochs=40, bs=256, use_balance=False, noise_train=0.5):
    torch.manual_seed(3)
    net = Model(mode)
    opt = torch.optim.Adam(net.parameters(), lr=4e-3)
    nB = Xtr.shape[0] // bs
    for ep in range(epochs):
        pi = torch.randperm(Xtr.shape[0])
        for bnd in range(nB):
            idx = pi[bnd * bs:(bnd + 1) * bs]
            pred, a = net(Xtr[idx], Ctr[idx], noise=noise_train)
            loss = ((pred - Ytr[idx]) ** 2).mean()
            if use_balance: loss = loss + 0.1 * balance_loss(a)
            opt.zero_grad(); loss.backward(); opt.step()
    return net

def evaluate(net, noise=0.0):
    net.eval()
    with torch.no_grad():
        pte, ate = net(Xte, Cte, noise=noise)
        te = ((pte - Yte) ** 2).mean().item()
        usage = ate.mean(dim=(0, 1)); ent = -(usage * (usage + 1e-9).log()).sum().item()
        sharp = ate.max(-1).values.mean().item()      # 1.0 = perfectly one-hot
    return te, ent, sharp

def nparams(net): return sum(p.numel() for p in net.parameters())

print("Task: ordered non-commuting affine composition | held-out = UNSEEN program orderings")
print(f"primitives M={M}  path length K={K}  train progs={n_train}/64  test(unseen)={len(test_progs)}")
print(f"var(Yte)={Yte.var().item():.3f}  (MSE>=this ~= no learning). All trained with routing noise=0.5\n")

configs = [
    ("oracle",  dict()),                          # ceiling: perfect routing
    ("router",  dict()),                          # standard learned gate
    ("free",    dict(use_balance=True)),          # Routing-Free MoE incumbent (with balance loss)
    ("free",    dict(use_balance=False)),         # incumbent WITHOUT the balance hack
    ("mycelial",dict()),                          # OURS (no balance loss, no gate)
]
noises = [0.0, 1.0, 2.0]
rows = []
for mode, kw in configs:
    t0 = time.time()
    net = train(mode, **kw)
    tes = [evaluate(net, noise=n)[0] for n in noises]
    _, ent0, sh0 = evaluate(net, noise=0.0)
    tag = mode + ("+bal" if kw.get("use_balance") else ("-bal" if mode == "free" else ""))
    rows.append((tag, nparams(net), tes, ent0, sh0, time.time() - t0))

ncols = "".join(f"held@{n:.0f}".rjust(10) for n in noises)
hdr = f"{'variant':<14}{'params':>7}{ncols}{'balance':>9}{'sharp':>7}{'sec':>6}"
print(hdr); print("-" * len(hdr))
for tag, npar, tes, ent, sh, sec in rows:
    tescol = "".join(f"{t:.4f}".rjust(10) for t in tes)
    print(f"{tag:<14}{npar:>7}{tescol}{ent:>9.3f}{sh:>7.3f}{sec:>6.1f}")
print(f"\nheld@n  = MSE on UNSEEN program orderings with routing-signal noise n (lower=better)")
print(f"balance = usage entropy (max={math.log(M):.3f}=perfectly uniform). sharp = mean max prob (1=one-hot path)")
