"""
Multi-seed confirmation of the constrained-path result.
Each seed = a fresh transition graph + data + init. Reports mean +/- std across
seeds for held-out MSE (unseen legal programs) and illegal-transition rate.

Variants (matched capacity, shared experts+encoder):
  oracle   = perfect routing (ceiling)
  router   = independent softmax gate            (+ adjacency feature)
  free     = Routing-Free MoE self-activation    (+ adjacency feature)   [INCUMBENT]
  mycelial = conservation-coupled Physarum flow, legality baked into topology  [OURS]
"""
import torch, torch.nn as nn, time
d, M, K = 6, 7, 3
NOISE = 0.7
ALPHAS = [3.0, 2.0, 1.0]
SEEDS = [11, 22, 33, 44]

def build_graph(T):
    N = 2 + K * M
    eid = lambda k, m: 2 + k * M + m
    edges = [(0, eid(0, m)) for m in range(M)]
    for k in range(K - 1):
        for m in range(M):
            for mp in range(M):
                if T[m, mp] > 0: edges.append((eid(k, m), eid(k + 1, mp)))
    edges += [(eid(K - 1, m), 1) for m in range(M)]
    E = len(edges)
    inc = torch.zeros(N, E); edst = torch.full((E, 2), -1); nedg = torch.zeros(N, E)
    for e, (i, j) in enumerate(edges):
        inc[i, e], inc[j, e] = 1.0, -1.0; nedg[i, e] = nedg[j, e] = 1.0
        if j >= 2: edst[e] = torch.tensor([(j - 2) // M, (j - 2) % M])
    binj = torch.zeros(N); binj[0] = 1.0; binj[1] = -1.0
    keep = torch.tensor([i for i in range(N) if i != 1])
    return dict(N=N, E=E, eid=eid, inc=inc, edst=edst, nedg=nedg, binj=binj, keep=keep)

def run_seed(seed):
    torch.manual_seed(seed)
    true_A = []
    for m in range(M):
        g = torch.randn(d, d); q, _ = torch.linalg.qr(g)
        true_A.append(q @ torch.linalg.matrix_exp(0.6 * (g - g.t()) / 2))
    true_A = torch.stack(true_A); true_b = 0.3 * torch.randn(M, d)
    T = (torch.rand(M, M) < 0.55).float(); T.fill_diagonal_(0.0)
    legal = [(a, b, c) for a in range(M) for b in range(M) for c in range(M)
             if T[a, b] > 0 and T[b, c] > 0]
    if len(legal) < 30: return None
    pm = torch.randperm(len(legal)); ntr = int(0.7 * len(legal))
    trp = [legal[i] for i in pm[:ntr]]; tep = [legal[i] for i in pm[ntr:]]
    def gen(progs, n):
        x = torch.randn(n, d); idx = torch.randint(0, len(progs), (n,))
        C = torch.tensor([progs[i] for i in idx]); h = x.clone()
        for k in range(K): h = torch.einsum('nij,nj->ni', true_A[C[:, k]], h) + true_b[C[:, k]]
        return x, C, h
    Xtr, Ctr, Ytr = gen(trp, 6000); Xte, Cte, Yte = gen(tep, 3000)
    G = build_graph(T); Tflat = T.reshape(-1)

    def physarum(cost, steps=26, dt=0.28, gamma=2.0):
        B = cost.shape[0]; L = torch.ones(B, G['E'])
        kk, mm = G['edst'][:, 0], G['edst'][:, 1]; v = kk >= 0
        L[:, v] = cost[:, kk[v], mm[v]]; L = L.clamp(min=1e-3)
        D = torch.ones(B, G['E']); binj = G['binj'].unsqueeze(0).expand(B, -1)
        keep = G['keep']; inc = G['inc']
        for _ in range(steps):
            w = D / L
            Lap = torch.einsum('ne,be,me->bnm', inc, w, inc)[:, keep][:, :, keep] + 1e-4 * torch.eye(G['N'] - 1)
            p = torch.zeros(B, G['N']); p[:, keep] = torch.linalg.solve(Lap, binj[:, keep])
            Q = w * torch.einsum('ne,bn->be', inc, p)
            D = (D + dt * (Q.abs() - D)).clamp(min=1e-9)
        thru = 0.5 * torch.einsum('ve,be->bv', G['nedg'], Q.abs())
        a = torch.zeros(B, K, M)
        for k in range(K):
            cols = torch.tensor([G['eid'](k, m) for m in range(M)])
            tk = thru[:, cols].clamp(min=1e-9) ** gamma
            a[:, k] = tk / (tk.sum(-1, keepdim=True) + 1e-9)
        return a

    class Model(nn.Module):
        def __init__(self, mode):
            super().__init__(); self.mode = mode
            self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, d)) for _ in range(M)])
            self.enc = nn.Sequential(nn.Linear(K * M + M * M, 64), nn.Tanh(), nn.Linear(64, K * M))
            for p in self.enc[-1].parameters(): nn.init.zeros_(p)
            self.temp = nn.Parameter(torch.tensor(0.5)); self.thr = nn.Parameter(torch.tensor(0.0))
        def sel(self, C, alpha):
            B = C.shape[0]; oh = torch.zeros(B, K, M); oh.scatter_(2, C.unsqueeze(-1), 1.0)
            hint = alpha * oh + NOISE * torch.randn(B, K, M)
            z = hint + self.enc(torch.cat([hint.reshape(B, -1), Tflat.expand(B, -1)], 1)).reshape(B, K, M)
            if self.mode == "oracle": return oh
            if self.mode == "router": return torch.softmax(z / self.temp.clamp(min=0.1), -1)
            if self.mode == "free":
                act = torch.sigmoid(z - self.thr); return act / (act.sum(-1, keepdim=True) + 1e-9)
            return physarum(torch.nn.functional.softplus(-z) + 0.05)
        def forward(self, x, C, alpha):
            a = self.sel(C, alpha); h = x
            for k in range(K):
                outs = torch.stack([self.experts[m](h) for m in range(M)], 1)
                h = torch.einsum('bm,bmd->bd', a[:, k], outs)
            return h, a
    def illegal(a):
        pk = a.argmax(-1); bad = torch.zeros(pk.shape[0])
        for k in range(K - 1): bad += (T[pk[:, k], pk[:, k + 1]] == 0).float()
        return (bad > 0).float().mean().item()
    def train(mode, use_balance=False):
        torch.manual_seed(3); net = Model(mode); opt = torch.optim.Adam(net.parameters(), lr=4e-3)
        for ep in range(55):
            pi = torch.randperm(6000)
            for b in range(6000 // 256):
                idx = pi[b*256:(b+1)*256]; pred, a = net(Xtr[idx], Ctr[idx], 2.5)
                loss = ((pred - Ytr[idx])**2).mean()
                if use_balance: loss = loss + 0.1 * ((a.mean((0,1)) - 1/M)**2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        return net
    out = {}
    for mode, kw in [("oracle", {}), ("router", {}), ("free", dict(use_balance=True)), ("mycelial", {})]:
        net = train(mode, **kw); net.eval()
        with torch.no_grad():
            mse = {}; ill = {}
            for al in ALPHAS:
                p, a = net(Xte, Cte, al); mse[al] = ((p - Yte)**2).mean().item(); ill[al] = illegal(a)
        out[mode] = (mse, ill)
    return out, len(legal)

t0 = time.time(); agg = {}
print(f"Running {len(SEEDS)} seeds (fresh transition graph + data each)...\n")
for s in SEEDS:
    r = run_seed(s)
    if r is None: continue
    res, nlegal = r
    for mode, (mse, ill) in res.items():
        agg.setdefault(mode, []).append((mse, ill))
    print(f"  seed {s}: {nlegal} legal programs  | mycelial MSE@2.0={res['mycelial'][0][2.0]:.3f}  "
          f"free MSE@2.0={res['free'][0][2.0]:.3f}  mycelial illeg@1.0={res['mycelial'][1][1.0]:.2f} free={res['free'][1][1.0]:.2f}")

import statistics as st
def ms(vals): return f"{st.mean(vals):.3f}±{(st.pstdev(vals) if len(vals)>1 else 0):.3f}"
print(f"\n=== MEAN ± STD over {len(agg['mycelial'])} seeds ===")
hdr = f"{'variant':<10}" + "".join(f"MSE@{a}".rjust(13) for a in ALPHAS) + "".join(f"illeg@{a}".rjust(12) for a in ALPHAS)
print(hdr); print("-" * len(hdr))
for mode in ["oracle", "router", "free", "mycelial"]:
    runs = agg[mode]
    line = f"{mode:<10}"
    for a in ALPHAS: line += ms([mse[a] for mse, _ in runs]).rjust(13)
    for a in ALPHAS: line += ms([ill[a] for _, ill in runs]).rjust(12)
    print(line)
print(f"\ntotal {time.time()-t0:.0f}s | held-out MSE on UNSEEN legal programs; illeg=illegal-transition rate (lower better)")
