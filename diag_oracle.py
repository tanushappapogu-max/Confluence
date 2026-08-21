"""Isolate the oracle (perfect one-hot routing). Can the shared expert bank
learn reusable primitives and generalize to unseen program orderings AT ALL?
If the oracle can't hit ~0 train AND ~0 held-out, the task/expert design is the
problem, not the router -- fix that before any routing comparison."""
import torch, torch.nn as nn
torch.manual_seed(0)
d, M, K = 6, 4, 3
true_A = []
for m in range(M):
    g = torch.randn(d, d); q, _ = torch.linalg.qr(g)
    r = torch.linalg.matrix_exp(0.6 * (g - g.t()) / 2)
    true_A.append(q @ r)
true_A = torch.stack(true_A); true_b = 0.3 * torch.randn(M, d)
allp = [(a, b, c) for a in range(M) for b in range(M) for c in range(M)]
gg = torch.Generator().manual_seed(1); perm = torch.randperm(len(allp), generator=gg)
trp = [allp[i] for i in perm[:44]]; tep = [allp[i] for i in perm[44:]]
def gen(progs, n):
    x = torch.randn(n, d); idx = torch.randint(0, len(progs), (n,))
    C = torch.tensor([progs[i] for i in idx]); h = x.clone()
    for k in range(K):
        h = torch.einsum('nij,nj->ni', true_A[C[:, k]], h) + true_b[C[:, k]]
    return x, C, h
Xtr, Ctr, Ytr = gen(trp, 6000); Xte, Cte, Yte = gen(tep, 3000)

def run(expert_kind, epochs=300, lr=5e-3):
    torch.manual_seed(3)
    if expert_kind == "linear":
        experts = nn.ModuleList([nn.Linear(d, d) for _ in range(M)])
    else:  # mlp
        experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, d)) for _ in range(M)])
    opt = torch.optim.Adam(experts.parameters(), lr=lr)
    for ep in range(epochs):
        pi = torch.randperm(6000)
        for b in range(6000 // 512):
            idx = pi[b*512:(b+1)*512]
            h = Xtr[idx]
            for k in range(K):
                outs = torch.stack([experts[m](h) for m in range(M)], 1)  # [B,M,d]
                oh = torch.zeros(len(idx), M); oh.scatter_(1, Ctr[idx, k:k+1], 1.0)
                h = torch.einsum('bm,bmd->bd', oh, outs)
            loss = ((h - Ytr[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    def ev(X, C, Y):
        with torch.no_grad():
            h = X
            for k in range(K):
                outs = torch.stack([experts[m](h) for m in range(M)], 1)
                oh = torch.zeros(len(X), M); oh.scatter_(1, C[:, k:k+1], 1.0)
                h = torch.einsum('bm,bmd->bd', oh, outs)
            return ((h - Y)**2).mean().item()
    return ev(Xtr, Ctr, Ytr), ev(Xte, Cte, Yte)

print(f"var(Yte)={Yte.var().item():.3f}")
for kind in ["linear", "mlp"]:
    tr, te = run(kind)
    print(f"experts={kind:7s}  oracle TRAIN mse={tr:.4f}  HELDOUT mse={te:.4f}")
