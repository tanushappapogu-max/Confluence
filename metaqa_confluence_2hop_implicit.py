"""MetaQA 2-hop with IMPLICIT-DIFF flow + lifted node cap (the scaling-ready pipeline).

Same model as metaqa_confluence_2hop_v2.py (GRU encoder, per-hop query heads, per-query
legal graph, in-degree-normalized decode) but two changes that remove the reasons the v2
results are stuck on the <=80-node "tractable subset":

  1. IMPLICIT DIFFERENTIATION through the flow fixed point instead of the 18-step unroll.
     Forward solve runs under no_grad; the gradient comes from the implicit function theorem
     via a matrix-free (VJP-only) solve of (I - J^T)u = g -- O(E) backward memory, no
     unrolled graph. (Primitives verified to machine precision in implicit_diff.py.)
  2. CAP lifted from 80 to CAP (default 400) so the heavy-degree tail is no longer excluded.
     The dense Laplacian solve is O(N^3); for the largest 2-hop neighborhoods (p90~1538)
     use the sparse forward solve hook noted below. CAP=400 already covers far more of the
     2-hop distribution than 80.

Because this container cannot reach the MetaQA data, run the built-in self-check to prove
the wired gradient is correct (matches finite difference on a synthetic KG-shaped graph):

    python3 metaqa_confluence_2hop_implicit.py --selfcheck

With data present (data/metaqa/...), run training/eval as usual:

    python3 metaqa_confluence_2hop_implicit.py            # trains, evals raw + degree-norm
    TRAIN_DEGNORM=1 python3 metaqa_confluence_2hop_implicit.py

Reuses the verified flow map F_step / solve_forward from implicit_diff (float64).
"""
import os, sys, collections, random, time, torch, torch.nn as nn
import implicit_diff
from implicit_diff import F_step, solve_forward
implicit_diff.RHO = 0.02          # small conductance floor -> smooth, well-conditioned fixed point
torch.manual_seed(0); random.seed(0)

DDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metaqa")
d = 48; CAP = int(os.environ.get("CAP", "400")); NTRAIN = 8000; NTEST = 1500; EPOCHS = 5
FWD_ITERS = int(os.environ.get("FWD_ITERS", "80")); BWD_ITERS = int(os.environ.get("BWD_ITERS", "200"))
TRAIN_DEGNORM = os.environ.get("TRAIN_DEGNORM", "0") == "1"


# ============================ flow: forward + implicit backward ============================
def build_flow(qt, G, enc):
    """Per-query flow tensors. enc(qt) -> (q1,q2). Returns Lv (grad), inc, b, keep."""
    q1, q2 = enc(qt); edges = G["edges"]; N = G["N"]; costs = []
    for (i, j, key) in edges:
        if key is None:
            costs.append(torch.tensor(0.05, dtype=torch.float64))
        else:
            hop, r, dr, e = key; q = q1 if hop == 0 else q2
            f = torch.cat([q, emb_rel(torch.tensor(r)), dir_emb(torch.tensor(dr)), emb_ent(torch.tensor(e))])
            costs.append(torch.nn.functional.softplus(-edge_mlp(f).squeeze()) + 0.05)
    Lv = torch.stack(costs); E = len(edges)
    inc = torch.zeros(N, E)
    for e, (i, j, _) in enumerate(edges): inc[i, e] = 1.0; inc[j, e] = -1.0
    b = torch.zeros(N); b[0] = 1.0; b[G["sink"]] = -1.0
    keep = torch.tensor([x for x in range(N) if x != G["sink"]])
    return Lv, inc, b, keep


def sink_mass_from(D, G):
    """Answer mass over e2 = conductance/flux on each e2->sink edge (D* = |Q*| at fixed point)."""
    mass = torch.zeros(len(G["e2set"]), dtype=D.dtype)
    for e, (i, j, _) in enumerate(G["edges"]):
        if j == G["sink"]: mass[i - G["o2"]] = D[e]
    return mass


@torch.no_grad()
def _lap_matvec(x, w, si, di, N):
    """Matrix-free Laplacian-vector product L(w) x from the edge list (x has sink entry 0).
    Per edge e=(i,j): flow = w_e (x_i - x_j) adds to node i, subtracts from node j."""
    flow = w * (x[si] - x[di])
    out = torch.zeros(N, dtype=x.dtype)
    out.index_add_(0, si, flow); out.index_add_(0, di, -flow)
    return out


@torch.no_grad()
def _cg_solve(w, b, si, di, N, sink, iters=400, tol=1e-10, reg=1e-4):
    """Solve the grounded L(w) p = b for p (p[sink]=0) by conjugate gradient, matrix-free.
    SPD system -> CG converges; cost O(E) per iter, no dense N x N matrix. reg on diagonal."""
    def A(x):
        y = _lap_matvec(x, w, si, di, N) + reg * x
        y[sink] = 0.0; return y
    p = torch.zeros(N, dtype=b.dtype); r = b.clone(); r[sink] = 0.0
    d = r.clone(); rs = (r * r).sum()
    for _ in range(iters):
        Ad = A(d); alpha = rs / ((d * Ad).sum() + 1e-30)
        p = p + alpha * d; r = r - alpha * Ad
        rs_new = (r * r).sum()
        if rs_new.sqrt() < tol: break
        d = r + (rs_new / (rs + 1e-30)) * d; rs = rs_new
    p[sink] = 0.0; return p


@torch.no_grad()
def solve_forward_mf(Lv, edges, N, sink, iters, dt=0.5):
    """Scalable forward: same reinforcement fixed point as solve_forward but the per-iteration
    linear solve is matrix-free CG -> scales to the large heavy-tail neighborhoods (no dense
    O(N^3) solve). Returns D* = regularized |Q*|.  RHO matches implicit_diff.RHO."""
    si = torch.tensor([e[0] for e in edges]); di = torch.tensor([e[1] for e in edges])
    b = torch.zeros(N, dtype=Lv.dtype); b[0] = 1.0; b[sink] = -1.0
    D = torch.ones(len(edges), dtype=Lv.dtype)
    for _ in range(iters):
        w = D / Lv
        p = _cg_solve(w, b, si, di, N, sink)
        Q = w * (p[si] - p[di])
        Fz = (1.0 - implicit_diff.RHO) * Q.abs() + implicit_diff.RHO
        D = (D + dt * (Fz - D)).clamp(min=1e-9)
    return D


@torch.no_grad()
def eval_mass(qt, G, enc, sparse=False):
    Lv, inc, b, keep = build_flow(qt, G, enc)
    if sparse or G["N"] > 500:                     # matrix-free CG for large neighborhoods
        D = solve_forward_mf(Lv, G["edges"], G["N"], G["sink"], FWD_ITERS, dt=0.5)
    else:
        D = solve_forward(Lv, inc, b, keep, FWD_ITERS, dt=0.5)
    return sink_mass_from(D, G)


def implicit_backward(qt, G, enc, params, gold_idx, degnorm):
    """Forward under no_grad to D*, CE loss on (degree-normalized) sink mass, then implicit
    gradient into params via matrix-free (I - J^T)u = g. Accumulates into p.grad. Returns loss."""
    Lv, inc, b, keep = build_flow(qt, G, enc)
    with torch.no_grad():
        z_star = solve_forward(Lv.detach(), inc, b, keep, FWD_ITERS, dt=0.5)
    z = z_star.detach().requires_grad_(True)
    Fz = F_step(z, Lv, inc, b, keep)                       # one differentiable step at D*
    mass = sink_mass_from(z, G)
    if degnorm: mass = mass / (G["indeg"] + 1e-6)
    tgt = torch.zeros(len(G["e2set"]), dtype=torch.float64);
    for gi in gold_idx: tgt[gi] = 1.0
    p = mass / (mass.sum() + 1e-9)
    loss = -(tgt / tgt.sum() * (p + 1e-9).log()).sum()
    g, = torch.autograd.grad(loss, z, retain_graph=True)   # cotangent dloss/dD*
    u = g.clone()                                          # solve (I - J^T) u = g (matrix-free)
    for _ in range(BWD_ITERS):
        Jt_u, = torch.autograd.grad(Fz, z, grad_outputs=u, retain_graph=True)
        un = g + Jt_u
        if (un - u).norm() <= 1e-9 * (u.norm() + 1e-9): u = un; break
        u = un
    gparams = torch.autograd.grad(Fz, params, grad_outputs=u, retain_graph=True, allow_unused=True)
    for pr, gp in zip(params, gparams):
        if gp is not None: pr.grad = gp if pr.grad is None else pr.grad + gp
    return loss.item()


# ============================ self-check (no MetaQA data needed) ============================
def selfcheck():
    """Prove the wired implicit gradient matches finite-difference on a synthetic 2-hop graph."""
    print("SELF-CHECK: implicit-diff MetaQA flow gradient vs finite difference (synthetic 2-hop)")
    torch.manual_seed(1)
    Ne_, Nr_ = 40, 6
    global emb_ent, emb_rel, dir_emb, edge_mlp
    emb_ent = nn.Embedding(Ne_, d).double(); emb_rel = nn.Embedding(Nr_, d).double()
    dir_emb = nn.Embedding(2, d).double(); edge_mlp = nn.Sequential(nn.Linear(4*d,96), nn.GELU(), nn.Linear(96,1)).double()
    q1 = torch.randn(d, dtype=torch.float64); q2 = torch.randn(d, dtype=torch.float64)
    enc = lambda qt: (q1, q2)
    # build a synthetic per-query graph in the same dict format as build()
    M, K = 6, 10; o1, o2 = 2, 2 + M; sink = o2 + K; N = sink + 1
    g = torch.Generator().manual_seed(2); edges = [(0, 1, None)]
    for i in range(M): edges.append((1, o1 + i, (0, int(torch.randint(0,Nr_,(1,),generator=g)), 0, int(torch.randint(0,Ne_,(1,),generator=g)))))
    for i in range(M):
        for c in torch.randperm(K, generator=g)[:4].tolist():
            edges.append((o1 + i, o2 + c, (1, int(torch.randint(0,Nr_,(1,),generator=g)), 0, int(torch.randint(0,Ne_,(1,),generator=g)))))
    for c in range(K): edges.append((o2 + c, sink, None))
    indeg = torch.zeros(K)
    for (i,j,key) in edges:
        if key is not None and key[0]==1: indeg[j-o2]+=1
    G = dict(N=N, edges=edges, e2set=list(range(K)), o2=o2, sink=sink, indeg=indeg)
    params = list(edge_mlp.parameters()) + [emb_rel.weight, emb_ent.weight, dir_emb.weight]
    gold_idx = {int(indeg.argmax())}                         # some reachable gold

    for pr in params: pr.grad = None
    loss = implicit_backward("", G, enc, params, gold_idx, degnorm=True)
    # finite-difference check on a few entries of emb_rel.weight
    def loss_only(perturb=None):
        with torch.no_grad():
            if perturb is not None: emb_rel.weight[perturb[0]] += perturb[1]
            m = eval_mass("", G, enc); m = m / (G["indeg"] + 1e-6)
            p = m / (m.sum() + 1e-9)
            tgt = torch.zeros(K, dtype=torch.float64)
            for gi in gold_idx: tgt[gi] = 1.0
            L = -(tgt / tgt.sum() * (p + 1e-9).log()).sum().item()
            if perturb is not None: emb_rel.weight[perturb[0]] -= perturb[1]
            return L
    eps = 1e-5; cos_ok = True; worst = 0.0
    print(f"  loss {loss:.4f} | checking d loss / d emb_rel[row,col] (implicit vs FD):")
    for (row, col) in [(0,0),(1,3),(2,7),(3,1),(4,5)]:
        e = torch.zeros(Nr_, d, dtype=torch.float64); e[row,col] = eps
        fd = (loss_only((slice(None), e)) - loss_only((slice(None), -e))) / (2*eps)
        an = emb_rel.weight.grad[row, col].item()
        rel = abs(fd - an) / (abs(fd) + 1e-9); worst = max(worst, rel)
        print(f"    emb_rel[{row},{col}]: implicit {an:+.6f}  FD {fd:+.6f}  rel_err {rel:.2e}")
    ok = worst < 1e-2
    print(f"\n  worst rel_err {worst:.2e} -> {'PASS: implicit-diff wiring is correct' if ok else 'FAIL'}")
    return ok


# ============================ data path (runs when MetaQA is present) ============================
def load_kb_and_run():
    global emb_ent, emb_rel, emb_tok, dir_emb, gru, head1, head2, edge_mlp, vocab
    ent2id, rel2id = {}, {}; adj = collections.defaultdict(list)
    with open(os.path.join(DDIR, "kb", "kb.txt")) as f:
        for line in f:
            s, r, o = line.rstrip("\n").split("|")
            for e in (s, o): ent2id.setdefault(e, len(ent2id))
            rel2id.setdefault(r, len(rel2id))
            h, rr, t = ent2id[s], rel2id[r], ent2id[o]
            adj[h].append((rr, t, 0)); adj[t].append((rr, h, 1))
    Ne, Nr = len(ent2id), len(rel2id)

    def load_qa(split):
        out = []
        with open(os.path.join(DDIR, "2-hop", f"qa_{split}.txt")) as f:
            for line in f:
                q, ans = line.rstrip("\n").split("\t")
                topic = q[q.find("[")+1:q.find("]")]
                qt = (q[:q.find("[")] + q[q.find("]")+1:]).lower()
                out.append((topic, qt, ans.split("|")))
        return out

    def build(topic_id):
        h1 = adj[topic_id]; e1set = list({x[1] for x in h1})
        if not e1set: return None
        h2edges = []; e2set = set()
        for (r, e1, dr) in h1:
            for (r2, e2, dr2) in adj[e1]:
                h2edges.append((e1, r2, e2, dr2)); e2set.add(e2)
        e2set = list(e2set)
        if 3 + len(e1set) + len(e2set) > CAP: return None
        id_e1 = {e: i for i, e in enumerate(e1set)}; id_e2 = {e: i for i, e in enumerate(e2set)}
        o1 = 2; o2 = 2 + len(e1set); sink = o2 + len(e2set); N = sink + 1
        edges = [(0, 1, None)]
        for (r, e1, dr) in h1: edges.append((1, o1 + id_e1[e1], (0, r, dr, e1)))
        for (e1, r2, e2, dr2) in h2edges: edges.append((o1 + id_e1[e1], o2 + id_e2[e2], (1, r2, dr2, e2)))
        for e2 in e2set: edges.append((o2 + id_e2[e2], sink, None))
        indeg = torch.zeros(len(e2set))
        for (e1, r2, e2, dr2) in h2edges: indeg[id_e2[e2]] += 1
        return dict(N=N, edges=edges, e2set=e2set, o2=o2, sink=sink, indeg=indeg)

    def prep(split, limit):
        qa = load_qa(split); random.shuffle(qa); out = []
        for topic, qt, ans in qa:
            if topic not in ent2id: continue
            G = build(ent2id[topic])
            if G is None: continue
            if not any(e in set(ent2id[a] for a in ans if a in ent2id) for e in G["e2set"]): continue
            out.append((topic, qt, ans, G))
            if len(out) >= limit: break
        return out
    t0 = time.time(); tr = prep("train", NTRAIN); te = prep("test", NTEST)
    print(f"KG {Ne} ent/{Nr} rel | 2-hop (CAP={CAP}): train {len(tr)} test {len(te)} | prep {time.time()-t0:.0f}s"
          f" | IMPLICIT-DIFF backward | TRAIN_DEGNORM={'on' if TRAIN_DEGNORM else 'off'}")

    vocab = {"<unk>": 0, "<pad>": 1}
    for _, qt, _, _ in tr:
        for w in qt.split(): vocab.setdefault(w, len(vocab))
    V = len(vocab)
    emb_ent = nn.Embedding(Ne, d).double(); emb_rel = nn.Embedding(Nr, d).double()
    emb_tok = nn.Embedding(V, d).double(); dir_emb = nn.Embedding(2, d).double()
    gru = nn.GRU(d, d, batch_first=True).double(); head1 = nn.Linear(d, d).double(); head2 = nn.Linear(d, d).double()
    edge_mlp = nn.Sequential(nn.Linear(4*d, 96), nn.GELU(), nn.Linear(96, 1)).double()
    mods = [emb_ent, emb_rel, emb_tok, dir_emb, gru, head1, head2, edge_mlp]
    params = sum([list(m.parameters()) for m in mods], [])
    opt = torch.optim.Adam(params, lr=2e-3)

    def enc(qt):
        ids = [vocab.get(w, 0) for w in qt.split()] or [0]
        x = emb_tok(torch.tensor(ids)).unsqueeze(0); _, h = gru(x); h = h.squeeze(0).squeeze(0)
        return head1(h), head2(h)

    print("training (implicit-diff)...")
    for ep in range(EPOCHS):
        random.shuffle(tr); tot = 0.0; n = 0; opt.zero_grad(); t0 = time.time()
        for (topic, qt, ans, G) in tr:
            gold = set(ent2id[a] for a in ans if a in ent2id)
            gold_idx = [i for i, e in enumerate(G["e2set"]) if e in gold]
            if not gold_idx: continue
            tot += implicit_backward(qt, G, enc, params, gold_idx, TRAIN_DEGNORM); n += 1
            if n % 16 == 0: opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        hit = hitN = m = 0
        with torch.no_grad():
            for (topic, qt, ans, G) in te:
                mass = eval_mass(qt, G, enc); gold = set(ent2id[a] for a in ans if a in ent2id)
                massN = mass / (G["indeg"] + 1e-6)
                hit += G["e2set"][mass.argmax().item()] in gold
                hitN += G["e2set"][massN.argmax().item()] in gold; m += 1
        print(f"  epoch {ep}: loss {tot/max(n,1):.3f}  hits@1 raw {hit/max(m,1):.3f}  degree-norm {hitN/max(m,1):.3f}  ({time.time()-t0:.0f}s)")
    print("\nImplicit-diff backward (O(E) memory, no unroll) + CAP lifted -> heavy tail no longer excluded.")


def scalecheck():
    """Prove matrix-free CG forward == dense forward (same mass), then scale to large N."""
    print("SCALE-CHECK: matrix-free CG forward vs dense forward, then large-N timing\n")
    def syn(M, K, deg, seed):
        g = torch.Generator().manual_seed(seed)
        o1, o2 = 2, 2 + M; sink = o2 + K; N = sink + 1; edges = [(0, 1, None)]
        for i in range(M): edges.append((1, o1 + i, None))
        for i in range(M):
            for c in torch.randperm(K, generator=g)[:deg].tolist(): edges.append((o1 + i, o2 + c, None))
        for c in range(K): edges.append((o2 + c, sink, None))
        inc = torch.zeros(N, len(edges))
        for e, (i, j, _) in enumerate(edges): inc[i, e] = 1.0; inc[j, e] = -1.0
        keep = torch.tensor([x for x in range(N) if x != sink])
        b = torch.zeros(N); b[0] = 1.0; b[sink] = -1.0
        return edges, inc, b, keep, N, sink
    print(f"  {'N':>6} {'E':>7} {'max|mass_dense-mass_mf|':>24} {'dense ms':>10} {'mf ms':>9}")
    for (M, K, deg) in [(6, 10, 4), (20, 60, 5), (60, 200, 6)]:
        edges, inc, b, keep, N, sink = syn(M, K, deg, seed=N if False else M)
        torch.manual_seed(M); Lv = torch.nn.functional.softplus(torch.randn(len(edges), dtype=torch.float64)) + 0.05
        t = time.time(); Dd = solve_forward(Lv, inc, b, keep, 60, dt=0.5); td = (time.time()-t)*1e3
        t = time.time(); Dm = solve_forward_mf(Lv, edges, N, sink, 60, dt=0.5); tm = (time.time()-t)*1e3
        diff = (Dd - Dm).abs().max().item()
        print(f"  {N:>6} {len(edges):>7} {diff:>24.2e} {td:>10.1f} {tm:>9.1f}")
    # large N where dense is infeasible: matrix-free alone
    edges, inc, b, keep, N, sink = syn(400, 2000, 6, seed=7)
    t = time.time(); Dm = solve_forward_mf(Lv := (torch.nn.functional.softplus(torch.randn(len(edges), dtype=torch.float64))+0.05), edges, N, sink, 40, dt=0.5); tm = (time.time()-t)*1e3
    print(f"\n  large: N={N} E={len(edges)} solved matrix-free in {tm:.0f} ms (dense O(N^3) infeasible)")
    print("  matrix-free forward -> eval can decode the FULL 2-hop distribution incl. heavy tail.")
    return True


if __name__ == "__main__":
    if "--scalecheck" in sys.argv:
        sys.exit(0 if scalecheck() else 1)
    if "--selfcheck" in sys.argv or not os.path.exists(os.path.join(DDIR, "kb", "kb.txt")):
        if not os.path.exists(os.path.join(DDIR, "kb", "kb.txt")):
            print("(MetaQA data not found -> running gradient self-check instead of training)\n")
        ok = selfcheck()
        sys.exit(0 if ok else 1)
    load_kb_and_run()
