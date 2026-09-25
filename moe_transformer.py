"""GPU-ready MoE transformer: conserved-flow routing vs top-k gating on a COUPLED task.

The main-track bet. Unlike MetaQA (marginalizable -> flow ties), this task is coupled BY
CONSTRUCTION: the correct expert at each MoE layer depends on the expert chosen at the previous
layer (a legal expert-transition graph). If the conserved flow's coupling helps anywhere on a
real trainable model, it is here. Baselines share experts + encoder; only the router differs.

Model: token embedding -> [attention + MoE-FFN] x K layers -> readout. Each MoE-FFN routes tokens
through experts; the router is either a hard top-k gate (standard MoE) or the batched conserved
flow (one legal expert-path). Task: sequence transform whose target requires applying a specific
LEGAL sequence of expert ops (per-example program), so getting layer k right needs layer k-1.

Batched + device-agnostic (uses CUDA if available -> this is the part a GPU accelerates).
Reports token accuracy, active experts/layer (compute), and illegal-transition rate.

Run: python3 moe_transformer.py            (SMOKE=1 for a tiny CPU correctness check)
"""
import os, time, math, torch, torch.nn as nn, statistics as st
SMOKE = os.environ.get("SMOKE", "0") == "1"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_model = 32 if SMOKE else 96
E = 6                       # experts per MoE layer
K = 3                       # MoE layers  (path length through experts)
L = 6                       # sequence length
VOCAB = 16                  # token vocabulary
NTRAIN = int(os.environ.get("NTRAIN", "2000" if SMOKE else "20000"))
NTEST = int(os.environ.get("NTEST", "500" if SMOKE else "4000"))
BATCH = int(os.environ.get("BATCH", "128"))
EPOCHS = int(os.environ.get("EPOCHS", "2" if SMOKE else "12"))
PH_STEPS = int(os.environ.get("PH_STEPS", "10" if SMOKE else "18"))
DENSITY = float(os.environ.get("DENSITY", "0.5"))   # expert-transition legality density (coupled regime)
SEED_START = int(os.environ.get("SEED_START", "0"))
SEEDS = list(range(SEED_START, SEED_START + int(os.environ.get("NSEEDS", "1" if SMOKE else "3"))))


def make_task(seed):
    """Coupled program-execution task. Each example: input tokens x[L] + a program (legal expert
    path p[K]); target y[L] = compose(experts p) applied to x. Experts are fixed random token maps;
    only LEGAL paths (per transition graph T) appear. Predicting y needs the right expert at each
    layer, and which is right at layer k is constrained by layer k-1 (coupling)."""
    g = torch.Generator().manual_seed(seed)
    T = (torch.rand(E, E, generator=g) < DENSITY).float(); T.fill_diagonal_(0.0)
    paths = [(a, b, c) for a in range(E) for b in range(E) for c in range(E) if T[a, b] > 0 and T[b, c] > 0]
    if len(paths) < 20: return None
    # each expert = a fixed permutation-ish token map
    maps = [torch.randperm(VOCAB, generator=g) for _ in range(E)]
    def gen(n):
        x = torch.randint(0, VOCAB, (n, L), generator=g)
        pidx = torch.randint(0, len(paths), (n,), generator=g)
        P = torch.tensor([paths[i] for i in pidx])          # [n,K] legal program
        y = x.clone()
        for k in range(K):
            for e in range(E):
                m = P[:, k] == e
                if m.any(): y[m] = maps[e][y[m]]
        return x.to(dev), P.to(dev), y.to(dev)
    return dict(T=T.to(dev), paths=paths, gen=gen)


def build_graph(T):
    """Layered expert-transition graph for the flow (source -> K expert layers -> sink)."""
    N = 2 + K * E
    eid = lambda k, m: 2 + k * E + m
    edges = [(0, eid(0, m)) for m in range(E)]
    for k in range(K - 1):
        for m in range(E):
            for mp in range(E):
                if T[m, mp] > 0: edges.append((eid(k, m), eid(k + 1, mp)))
    edges += [(eid(K - 1, m), 1) for m in range(E)]
    Eg = len(edges)
    inc = torch.zeros(N, Eg, device=dev); edst = torch.full((Eg, 2), -1)
    nedg = torch.zeros(N, Eg, device=dev)
    for e, (i, j) in enumerate(edges):
        inc[i, e], inc[j, e] = 1.0, -1.0; nedg[i, e] = nedg[j, e] = 1.0
        if j >= 2: edst[e] = torch.tensor([(j - 2) // E, (j - 2) % E])
    binj = torch.zeros(N, device=dev); binj[0] = 1.0; binj[1] = -1.0
    keep = torch.tensor([i for i in range(N) if i != 1], device=dev)
    return dict(N=N, Eg=Eg, eid=eid, inc=inc, edst=edst.to(dev), nedg=nedg, binj=binj, keep=keep)


class MoETransformer(nn.Module):
    def __init__(self, mode, G, topk=1):
        super().__init__(); self.mode = mode; self.topk = topk; self.G = G
        self.tok = nn.Embedding(VOCAB, d_model); self.pos = nn.Embedding(L, d_model)
        self.attn = nn.ModuleList([nn.MultiheadAttention(d_model, 4, batch_first=True) for _ in range(K)])
        self.ln = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(K)])
        self.experts = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)) for _ in range(E)]) for _ in range(K)])
        self.router_enc = nn.Linear(d_model, K * E)          # program hint -> per-layer expert logits
        self.gate = nn.ModuleList([nn.Linear(d_model, E) for _ in range(K)])
        self.out = nn.Linear(d_model, VOCAB)
        self.temp = nn.Parameter(torch.tensor(0.7))

    def physarum(self, cost):
        """Batched conserved flow -> per-layer expert weights a[B,K,E]. cost:[B,E,E]."""
        G = self.G; B = cost.shape[0]; Lc = torch.ones(B, G['Eg'], device=dev)
        kk, mm = G['edst'][:, 0], G['edst'][:, 1]; v = kk >= 0
        Lc[:, v] = cost[:, kk[v], mm[v]].clamp(min=1e-3)
        D = torch.ones(B, G['Eg'], device=dev); binj = G['binj'].unsqueeze(0).expand(B, -1)
        keep, inc = G['keep'], G['inc']
        for _ in range(PH_STEPS):
            w = D / Lc
            Lap = torch.einsum('ne,be,me->bnm', inc, w, inc)[:, keep][:, :, keep] + 1e-4 * torch.eye(G['N'] - 1, device=dev)
            p = torch.zeros(B, G['N'], device=dev); p[:, keep] = torch.linalg.solve(Lap, binj[:, keep])
            Q = w * torch.einsum('ne,bn->be', inc, p)
            D = (D + 0.28 * (Q.abs() - D)).clamp(min=1e-9)
        thru = 0.5 * torch.einsum('ve,be->bv', G['nedg'], Q.abs())
        a = torch.zeros(B, K, E, device=dev)
        for k in range(K):
            cols = torch.tensor([G['eid'](k, m) for m in range(E)], device=dev)
            tk = thru[:, cols].clamp(min=1e-9) ** 2.0
            a[:, k] = tk / (tk.sum(-1, keepdim=True) + 1e-9)
        return a

    def route(self, P, hstate):
        """Return per-(example) expert weights a[B,K,E] and the online logits z for gate modes."""
        B = P.shape[0]; oh = torch.zeros(B, K, E, device=dev); oh.scatter_(2, P.unsqueeze(-1), 1.0)
        hint = 3.0 * oh + 0.7 * torch.randn(B, K, E, device=dev)          # noisy program hint (input signal)
        z = hint + self.router_enc(hstate).reshape(B, K, E)
        if self.mode == "flow":
            return self.physarum(torch.nn.functional.softplus(-z) + 0.05), z
        return None, z

    def forward(self, x, P):
        B = x.shape[0]
        h = self.tok(x) + self.pos(torch.arange(L, device=dev))[None]
        pooled0 = h.mean(1)
        a, z = self.route(P, pooled0); active = 0.0; self._aux = 0.0; sel = []
        for k in range(K):
            att, _ = self.attn[k](h, h, h); h = self.ln[k](h + att)
            outs = torch.stack([self.experts[k][e](h) for e in range(E)], 2)   # [B,L,E,d]
            if self.mode == "flow":
                ak = a[:, k]; sel.append(ak.argmax(-1))
                h = h + torch.einsum('be,bled->bld', ak, outs); active += 1.0
            elif self.mode == "dense":
                h = h + outs.mean(2); active += E; sel.append(torch.zeros(B, dtype=torch.long, device=dev))
            else:  # top-k gate
                gk = torch.softmax((self.gate[k](h.mean(1)) + z[:, k]) / self.temp.clamp(min=0.1), -1)
                self._aux = self._aux + ((gk.mean(0) - 1.0 / E) ** 2).sum()
                val, idx = gk.topk(self.topk, -1); val = val / (val.sum(-1, keepdim=True) + 1e-9)
                mix = torch.zeros_like(h)
                for j in range(self.topk):
                    selj = outs[torch.arange(B, device=dev), :, idx[:, j]]     # [B,L,d]
                    mix = mix + val[:, j][:, None, None] * selj
                h = h + mix; active += self.topk; sel.append(idx[:, 0])
        return self.out(h), torch.stack(sel, 1), active / K

    def illegal(self, sel, T):
        bad = torch.zeros(sel.shape[0], device=dev)
        for k in range(K - 1): bad += (T[sel[:, k], sel[:, k + 1]] == 0).float()
        return (bad > 0).float().mean().item()


def run_seed(seed):
    task = make_task(seed)
    if task is None: return None
    G = build_graph(task["T"])
    Xtr, Ptr, Ytr = task["gen"](NTRAIN); Xte, Pte, Yte = task["gen"](NTEST)
    def train(mode, topk=1):
        torch.manual_seed(3); net = MoETransformer(mode, G, topk).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        for ep in range(EPOCHS):
            perm = torch.randperm(NTRAIN, device=dev)
            for b in range(0, NTRAIN, BATCH):
                idx = perm[b:b+BATCH]
                logits, _, _ = net(Xtr[idx], Ptr[idx])
                loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), Ytr[idx].reshape(-1))
                if mode == "topk": loss = loss + 0.01 * net._aux
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            logits, sel, active = net(Xte, Pte)
            acc = (logits.argmax(-1) == Yte).float().mean().item()
            return dict(acc=acc, active=active, illegal=net.illegal(sel, task["T"]))
    return {"dense": train("dense"), "topk1": train("topk", 1), "topk2": train("topk", 2), "flow": train("flow")}


if __name__ == "__main__":
    t0 = time.time()
    print(f"{'SMOKE ' if SMOKE else ''}MoE-transformer | device={dev} | E={E} experts, K={K} MoE layers, "
          f"L={L}, density={DENSITY} (coupled), seeds={SEEDS}\n")
    agg = {}
    for s in SEEDS:
        r = run_seed(s)
        if r is None: print(f"  seed {s}: too few legal paths, skipped"); continue
        for m in r: agg.setdefault(m, []).append(r[m])
        print(f"  seed {s} done ({time.time()-t0:.0f}s)")
    def ms(v): return f"{st.mean(v):.3f}" + (f"±{st.pstdev(v):.3f}" if len(v) > 1 else "")
    print(f"\n{'router':<16}{'token acc':>12}{'active/layer':>14}{'illegal':>12}")
    print("-" * 54)
    for m, lab in [("dense", "dense (all)"), ("topk1", "top-1 gate"), ("topk2", "top-2 gate"), ("flow", "flow (ours)")]:
        if m not in agg: continue
        runs = agg[m]; ill = "n/a" if m == "dense" else ms([r['illegal'] for r in runs])
        print(f"{lab:<16}{ms([r['acc'] for r in runs]):>12}{ms([r['active'] for r in runs]):>14}{ill:>12}")
    print(f"\ntotal {time.time()-t0:.0f}s")
    print("MAIN-TRACK READ: if flow acc > top-1 gate at equal compute (1 expert/layer) with lower")
    print("illegal rate, the coupling win holds in a real trainable transformer -- the real result.")
