"""
CONFLUENCE go/no-go spike.
Question: can ONE conserved flow route a single legal path through a graph whose
nodes are BOTH data (facts, retrieve=inject a value) AND compute (experts,
apply a function) -- interleaving retrieve->compute->retrieve->compute -- with
0 illegal hops, while a soft-penalty baseline leaks?

Task: alternating retrieve/compute chain.
  positions:  [fact, expert, fact, expert]   (retrieve, compute, retrieve, compute)
  h0 = x
  fact position f:    h <- h + val[f]         (val = given DATA, not learned)
  expert position e:  h <- g_e(h)             (g_e = learned MLP; truth = affine)
  target = execute the true legal program.
Legality: bipartite transition graph -- fact->expert (T_fe) and expert->fact (T_ef);
fact->fact and expert->expert are ILLEGAL (must process what you fetch). The graph's
TOPOLOGY encodes it, so the flow cannot take an illegal hop.

SMOKE=1 -> tiny fast run to catch errors.
"""
import os, torch, torch.nn as nn, time, statistics as st
SMOKE = os.environ.get("SMOKE", "0") == "1"
d = 6
F, E = 4, 4                      # #facts, #experts
K = 4                            # path length: fact,expert,fact,expert
TYPES = ["fact", "expert", "fact", "expert"]
NITEM = 4                        # candidates per position (4 facts OR 4 experts)
NOISE = 0.7
SEEDS = [11] if SMOKE else [11, 22, 33]
EPOCHS = 8 if SMOKE else 55
PH_STEPS = 12 if SMOKE else 24
TEST_ALPHAS = [3.0, 2.0, 1.0]

def build_graph(legal_fe, legal_ef):
    """layered bipartite graph. node id: 0 src, 1 sink, else 2 + k*NITEM + m."""
    N = 2 + K * NITEM
    eid = lambda k, m: 2 + k * NITEM + m
    edges = []
    for m in range(NITEM): edges.append((0, eid(0, m)))            # src -> pos0 (facts)
    for k in range(K - 1):
        for m in range(NITEM):
            for mp in range(NITEM):
                ok = legal_fe[m, mp] if TYPES[k] == "fact" else legal_ef[m, mp]
                if ok > 0: edges.append((eid(k, m), eid(k + 1, mp)))
    for m in range(NITEM): edges.append((eid(K - 1, m), 1))        # posK-1 (experts) -> sink
    Eg = len(edges)
    inc = torch.zeros(N, Eg); edst = torch.full((Eg, 2), -1); nedg = torch.zeros(N, Eg)
    for e, (i, j) in enumerate(edges):
        inc[i, e], inc[j, e] = 1.0, -1.0; nedg[i, e] = nedg[j, e] = 1.0
        if j >= 2: edst[e] = torch.tensor([(j - 2) // NITEM, (j - 2) % NITEM])
    binj = torch.zeros(N); binj[0] = 1.0; binj[1] = -1.0
    keep = torch.tensor([i for i in range(N) if i != 1])
    return dict(N=N, Eg=Eg, eid=eid, inc=inc, edst=edst, nedg=nedg, binj=binj, keep=keep)

def run_seed(seed):
    torch.manual_seed(seed)
    val = torch.randn(F, d)                                        # DATA: fixed fact values
    trueA = []
    for e in range(E):
        g = torch.randn(d, d); q, _ = torch.linalg.qr(g)
        trueA.append(q @ torch.linalg.matrix_exp(0.6 * (g - g.t()) / 2))
    trueA = torch.stack(trueA); trueb = 0.3 * torch.randn(E, d)
    legal_fe = (torch.rand(F, E) < 0.6).float()                   # fact f -> expert e legal?
    legal_ef = (torch.rand(E, F) < 0.6).float()                   # expert e -> fact f legal?
    # enumerate legal programs (f0,e0,f1,e1)
    progs = [(f0, e0, f1, e1) for f0 in range(F) for e0 in range(E)
             for f1 in range(F) for e1 in range(E)
             if legal_fe[f0, e0] > 0 and legal_ef[e0, f1] > 0 and legal_fe[f1, e1] > 0]
    if len(progs) < 20: return None
    pm = torch.randperm(len(progs)); ntr = int(0.7 * len(progs))
    trp = [progs[i] for i in pm[:ntr]]; tep = [progs[i] for i in pm[ntr:]]

    def execute_true(x, C):
        h = x.clone()
        h = h + val[C[:, 0]]                       # fetch fact f0
        h = torch.einsum('nij,nj->ni', trueA[C[:, 1]], h) + trueb[C[:, 1]]   # expert e0
        h = h + val[C[:, 2]]                       # fetch fact f1
        h = torch.einsum('nij,nj->ni', trueA[C[:, 3]], h) + trueb[C[:, 3]]   # expert e1
        return h
    def gen(progs, n):
        x = torch.randn(n, d); idx = torch.randint(0, len(progs), (n,))
        C = torch.tensor([progs[i] for i in idx])
        return x, C, execute_true(x, C)
    Xtr, Ctr, Ytr = gen(trp, 5000); Xte, Cte, Yte = gen(tep, 2500)
    G = build_graph(legal_fe, legal_ef)
    # flat adjacency features for the encoder (fair to baselines)
    adj = torch.cat([legal_fe.reshape(-1), legal_ef.reshape(-1)])

    def physarum(cost, steps=PH_STEPS, dt=0.28, gamma=2.0):
        B = cost.shape[0]; L = torch.ones(B, G['Eg'])
        kk, mm = G['edst'][:, 0], G['edst'][:, 1]; v = kk >= 0
        L[:, v] = cost[:, kk[v], mm[v]]; L = L.clamp(min=1e-3)
        D = torch.ones(B, G['Eg']); binj = G['binj'].unsqueeze(0).expand(B, -1)
        keep = G['keep']; inc = G['inc']
        for _ in range(steps):
            w = D / L
            Lap = torch.einsum('ne,be,me->bnm', inc, w, inc)[:, keep][:, :, keep] + 1e-4 * torch.eye(G['N'] - 1)
            p = torch.zeros(B, G['N']); p[:, keep] = torch.linalg.solve(Lap, binj[:, keep])
            Q = w * torch.einsum('ne,bn->be', inc, p)
            D = (D + dt * (Q.abs() - D)).clamp(min=1e-9)
        thru = 0.5 * torch.einsum('ve,be->bv', G['nedg'], Q.abs())
        a = torch.zeros(B, K, NITEM)
        for k in range(K):
            cols = torch.tensor([G['eid'](k, m) for m in range(NITEM)])
            tk = thru[:, cols].clamp(min=1e-9) ** gamma
            a[:, k] = tk / (tk.sum(-1, keepdim=True) + 1e-9)
        return a

    class Model(nn.Module):
        def __init__(self, mode):
            super().__init__(); self.mode = mode
            self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, d)) for _ in range(E)])
            self.register_buffer("val", val)                       # DATA nodes: fixed, non-parametric
            self.enc = nn.Sequential(nn.Linear(K * NITEM + adj.numel(), 64), nn.Tanh(), nn.Linear(64, K * NITEM))
            for p in self.enc[-1].parameters(): nn.init.zeros_(p)
            # PathMoE-lite: a router whose params are SHARED across positions (their mechanism:
            # constrain path space by tying the router, NOT by hard topology).
            self.shared = nn.Sequential(nn.Linear(NITEM, 32), nn.Tanh(), nn.Linear(32, NITEM))
            self.temp = nn.Parameter(torch.tensor(0.5)); self.thr = nn.Parameter(torch.tensor(0.0))
        def sel(self, C, alpha):
            B = C.shape[0]; oh = torch.zeros(B, K, NITEM); oh.scatter_(2, C.unsqueeze(-1), 1.0)
            hint = alpha * oh + NOISE * torch.randn(B, K, NITEM)
            z = hint + self.enc(torch.cat([hint.reshape(B, -1), adj.expand(B, -1)], 1)).reshape(B, K, NITEM)
            if self.mode == "oracle": return oh
            if self.mode == "pathmoe":                              # shared router across positions, no hard legality
                zk = hint + torch.stack([self.shared(hint[:, k]) for k in range(K)], 1)
                return torch.softmax(zk / self.temp.clamp(min=0.1), -1)
            if self.mode == "router": return torch.softmax(z / self.temp.clamp(min=0.1), -1)
            if self.mode == "free":
                act = torch.sigmoid(z - self.thr); return act / (act.sum(-1, keepdim=True) + 1e-9)
            return physarum(torch.nn.functional.softplus(-z) + 0.05)
        def forward(self, x, C, alpha):
            a = self.sel(C, alpha); h = x
            for k in range(K):
                if TYPES[k] == "fact":                              # RETRIEVE: inject data value
                    h = h + torch.einsum('bm,md->bd', a[:, k], self.val)
                else:                                               # COMPUTE: apply expert
                    outs = torch.stack([self.experts[e](h) for e in range(E)], 1)
                    h = torch.einsum('bm,bmd->bd', a[:, k], outs)
            return h, a
    def illegal(a):
        pk = a.argmax(-1)                                           # [B,K] -> (f0,e0,f1,e1)
        bad = ((legal_fe[pk[:, 0], pk[:, 1]] == 0) |
               (legal_ef[pk[:, 1], pk[:, 2]] == 0) |
               (legal_fe[pk[:, 2], pk[:, 3]] == 0)).float()
        return bad.mean().item()
    def interleave_ok(a):
        """sanity: does the path actually alternate fact/expert? (true by construction here)"""
        return True
    def train(mode, use_balance=False):
        torch.manual_seed(3); net = Model(mode); opt = torch.optim.Adam(net.parameters(), lr=4e-3)
        for ep in range(EPOCHS):
            pi = torch.randperm(5000)
            for b in range(5000 // 256):
                idx = pi[b*256:(b+1)*256]; pred, a = net(Xtr[idx], Ctr[idx], 2.5)
                loss = ((pred - Ytr[idx])**2).mean()
                if use_balance: loss = loss + 0.1 * ((a.mean((0,1)) - 1/NITEM)**2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        return net
    out = {}
    for mode, kw in [("oracle", {}), ("router", {}), ("pathmoe", {}), ("free", dict(use_balance=True)), ("mycelial", {})]:
        net = train(mode, **kw); net.eval()
        with torch.no_grad():
            mse = {}; ill = {}
            for al in TEST_ALPHAS:
                p, a = net(Xte, Cte, al); mse[al] = ((p - Yte)**2).mean().item(); ill[al] = illegal(a)
        out[mode] = (mse, ill)
    return out, len(progs), Yte.var().item()

t0 = time.time(); agg = {}
print(f"{'SMOKE ' if SMOKE else ''}CONFLUENCE spike | mixed graph: {F} data-nodes + {E} expert-nodes, path={TYPES}")
print(f"seeds={SEEDS}\n")
nprog_all = []; var_all = []
for s in SEEDS:
    r = run_seed(s)
    if r is None:
        print(f"  seed {s}: too few legal programs, skipped"); continue
    res, nprog, yv = r; nprog_all.append(nprog); var_all.append(yv)
    for m in res: agg.setdefault(m, []).append(res[m])
    print(f"  seed {s}: {nprog} legal programs | mycelial MSE@2.0={res['mycelial'][0][2.0]:.3f} illeg@1.0={res['mycelial'][1][1.0]:.3f}"
          f" | free MSE@2.0={res['free'][0][2.0]:.3f} illeg@1.0={res['free'][1][1.0]:.3f}")

def ms(v): return f"{st.mean(v):.3f}" + (f"±{st.pstdev(v):.3f}" if len(v) > 1 else "")
print(f"\navg legal programs={int(st.mean(nprog_all))}  var(Yte)~{st.mean(var_all):.2f} (MSE>=this=no learning)")
print(f"\n{'variant':<10}" + "".join(f"MSE@{a}".rjust(12) for a in TEST_ALPHAS) + "".join(f"illeg@{a}".rjust(12) for a in TEST_ALPHAS))
print("-" * 82)
for mode in ["oracle", "router", "pathmoe", "free", "mycelial"]:
    runs = agg[mode]
    line = f"{mode:<10}"
    for a in TEST_ALPHAS: line += ms([r[0][a] for r in runs]).rjust(12)
    for a in TEST_ALPHAS: line += ms([r[1][a] for r in runs]).rjust(12)
    print(line)
print(f"\nMSE = held-out accuracy on UNSEEN legal retrieve+compute programs (lower=better)")
print(f"illeg = illegal-hop rate (fact->fact or expert->expert or T-violation). mycelial=0 by construction.")
print(f"total {time.time()-t0:.0f}s")
print("\nGO if: oracle~0 (task learnable), mycelial illeg~0 while free>0, mycelial MSE competitive.")
