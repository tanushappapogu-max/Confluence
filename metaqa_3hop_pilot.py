"""3-hop MetaQA PILOT: flow vs no-flow head-to-head on a small tractable subset.

Purpose: a cheap de-risking test before spending GPU. Trains BOTH the conserved-flow router and
the independent per-hop softmax on the SAME small 3-hop subset and prints hits@1 for each.
NOTE (conservative): the small-CAP subset is the SMALL-neighborhood = LEAST-coupled 3-hop
questions, i.e. the regime least favorable to the flow. So flow>no-flow here is a strong green
light; a tie is inconclusive (the coupled tail, unlocked by a GPU + large CAP, is where flow
should win).

Run: python3 metaqa_3hop_pilot.py     (env: CAP, NTRAIN, NTEST, EPOCHS)
"""
import os, collections, random, time, torch, torch.nn as nn
torch.manual_seed(0); random.seed(0)
DDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metaqa")
d = 48; CAP = int(os.environ.get("CAP", "150")); NTRAIN = int(os.environ.get("NTRAIN", "500"))
NTEST = int(os.environ.get("NTEST", "150")); EPOCHS = int(os.environ.get("EPOCHS", "5"))

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
    with open(os.path.join(DDIR, "3-hop", f"qa_{split}.txt")) as f:
        for line in f:
            q, ans = line.rstrip("\n").split("\t")
            topic = q[q.find("[")+1:q.find("]")]; qt = (q[:q.find("[")] + q[q.find("]")+1:]).lower()
            out.append((topic, qt, ans.split("|")))
    return out

def build(topic_id):
    h1 = adj[topic_id]; e1set = list({x[1] for x in h1})
    if not e1set: return None
    e2set = set(); hop2 = []
    for (r, e1, dr) in h1:
        for (r2, e2, dr2) in adj[e1]: e2set.add(e2); hop2.append((e1, r2, e2, dr2))
    e2set = list(e2set); e3set = set(); hop3 = []
    for e2 in e2set:
        for (r3, e3, dr3) in adj[e2]: e3set.add(e3); hop3.append((e2, r3, e3, dr3))
    e3set = list(e3set)
    n1, n2, n3 = len(e1set), len(e2set), len(e3set)
    if 4 + n1 + n2 + n3 > CAP: return None
    ide1 = {e: i for i, e in enumerate(e1set)}; ide2 = {e: i for i, e in enumerate(e2set)}; ide3 = {e: i for i, e in enumerate(e3set)}
    hop1 = [(ide1[e1], r, dr, e1) for (r, e1, dr) in h1]
    H2 = [(ide1[e1], ide2[e2], r2, dr2, e2) for (e1, r2, e2, dr2) in hop2]
    H3 = [(ide2[e2], ide3[e3], r3, dr3, e3) for (e2, r3, e3, dr3) in hop3]
    # incidence for the flow: nodes 0=src,1=hub, e1:[2..), e2, e3, sink
    o1 = 2; o2 = o1 + n1; o3 = o2 + n2; sink = o3 + n3; N = sink + 1
    edges = [(0, 1, None)]
    for (i1, r, dr, e1) in hop1: edges.append((1, o1 + i1, (0, r, dr, e1)))
    for (i1, i2, r2, dr2, e2) in H2: edges.append((o1 + i1, o2 + i2, (1, r2, dr2, e2)))
    for (i2, i3, r3, dr3, e3) in H3: edges.append((o2 + i2, o3 + i3, (2, r3, dr3, e3)))
    for i3 in range(n3): edges.append((o3 + i3, sink, None))
    indeg = torch.zeros(n3)
    for (_, i3, _, _, _) in H3: indeg[i3] += 1
    return dict(e1set=e1set, e2set=e2set, e3set=e3set, hop1=hop1, hop2=H2, hop3=H3,
                indeg=indeg, edges=edges, N=N, o3=o3, sink=sink)

def prep(split, limit):
    qa = load_qa(split); random.shuffle(qa); out = []
    for topic, qt, ans in qa:
        if topic not in ent2id: continue
        G = build(ent2id[topic])
        if G is None: continue
        if not any(e in set(ent2id[a] for a in ans if a in ent2id) for e in G["e3set"]): continue
        out.append((topic, qt, ans, G))
        if len(out) >= limit: break
    return out
t0 = time.time(); tr = prep("train", NTRAIN); te = prep("test", NTEST)
print(f"KG {Ne} ent/{Nr} rel | 3-hop pilot (CAP={CAP}): train {len(tr)} test {len(te)} | prep {time.time()-t0:.0f}s")
avg_nodes = sum(4+len(g["e1set"])+len(g["e2set"])+len(g["e3set"]) for _,_,_,g in tr)/max(len(tr),1)
print(f"avg graph size ~{avg_nodes:.0f} nodes (small = least-coupled subset -> conservative for flow)\n")

vocab = {"<unk>": 0}
for _, qt, _, _ in tr:
    for w in qt.split(): vocab.setdefault(w, len(vocab))
V = len(vocab)

def make_model():
    m = dict(ent=nn.Embedding(Ne, d), rel=nn.Embedding(Nr, d), tok=nn.Embedding(V, d), dir=nn.Embedding(2, d),
             gru=nn.GRU(d, d, batch_first=True), heads=nn.ModuleList([nn.Linear(d, d) for _ in range(3)]),
             emlp=nn.Sequential(nn.Linear(4*d, 96), nn.GELU(), nn.Linear(96, 1)))
    params = sum([list(v.parameters()) for v in m.values()], [])
    return m, params

def qheads(m, qt):
    ids = [vocab.get(w, 0) for w in qt.split()] or [0]
    x = m["tok"](torch.tensor(ids)).unsqueeze(0); _, h = m["gru"](x); h = h.squeeze(0).squeeze(0)
    return [hd(h) for hd in m["heads"]]

def escore(m, q, R, Dr, Ent):
    qx = q.unsqueeze(0).expand(R.shape[0], -1)
    return m["emlp"](torch.cat([qx, m["rel"](R), m["dir"](Dr), m["ent"](Ent)], 1)).squeeze(1)

def run_noflow(m, qt, G):
    q1, q2, q3 = qheads(m, qt); n1, n2, n3 = len(G["e1set"]), len(G["e2set"]), len(G["e3set"])
    h1, h2, h3 = G["hop1"], G["hop2"], G["hop3"]
    if not h1 or not h2 or not h3: return torch.zeros(n3)
    s1e = escore(m, q1, torch.tensor([e[1] for e in h1]), torch.tensor([e[2] for e in h1]), torch.tensor([e[3] for e in h1]))
    s1 = torch.full((n1,), -1e9).scatter_reduce(0, torch.tensor([e[0] for e in h1]), s1e, reduce="amax", include_self=True)
    s2e = escore(m, q2, torch.tensor([e[2] for e in h2]), torch.tensor([e[3] for e in h2]), torch.tensor([e[4] for e in h2]))
    s2 = torch.full((n2,), -1e9).scatter_reduce(0, torch.tensor([e[1] for e in h2]), s1[torch.tensor([e[0] for e in h2])] + s2e, reduce="amax", include_self=True)
    s3e = escore(m, q3, torch.tensor([e[2] for e in h3]), torch.tensor([e[3] for e in h3]), torch.tensor([e[4] for e in h3]))
    par3, chi3 = torch.tensor([e[0] for e in h3]), torch.tensor([e[1] for e in h3])
    return torch.zeros(n3).scatter_add(0, chi3, torch.softmax(s2[par3] + s3e, 0))

def run_flow(m, qt, G):
    q = qheads(m, qt); edges = G["edges"]; N = G["N"]; costs = []
    for (i, j, key) in edges:
        if key is None: costs.append(torch.tensor(0.05))
        else:
            hop, r, dr, e = key
            f = torch.cat([q[hop], m["rel"](torch.tensor(r)), m["dir"](torch.tensor(dr)), m["ent"](torch.tensor(e))])
            costs.append(torch.nn.functional.softplus(-m["emlp"](f).squeeze()) + 0.05)
    Lv = torch.stack(costs); E = len(edges); inc = torch.zeros(N, E)
    for e, (i, j, _) in enumerate(edges): inc[i, e] = 1; inc[j, e] = -1
    Dc = torch.ones(E); b = torch.zeros(N); b[0] = 1.0; b[G["sink"]] = -1.0
    keep = torch.tensor([x for x in range(N) if x != G["sink"]])
    for _ in range(16):
        w = Dc / Lv
        Lap = (inc @ torch.diag(w) @ inc.t())[keep][:, keep] + 1e-4 * torch.eye(N - 1)
        p = torch.zeros(N); p[keep] = torch.linalg.solve(Lap, b[keep])
        Q = w * (inc.t() @ p); Dc = (Dc + 0.25 * (Q.abs() - Dc)).clamp(min=1e-9)
    mass = torch.zeros(len(G["e3set"]))
    for e, (i, j, _) in enumerate(edges):
        if j == G["sink"]: mass[i - G["o3"]] = Q.abs()[e]
    return mass

def train_eval(name, runner):
    torch.manual_seed(0); m, params = make_model(); opt = torch.optim.Adam(params, lr=2e-3)
    t0 = time.time()
    for ep in range(EPOCHS):
        random.shuffle(tr); opt.zero_grad(); bl = 0.0; n = 0
        for (topic, qt, ans, G) in tr:
            mass = runner(m, qt, G); gold = set(ent2id[a] for a in ans if a in ent2id)
            tgt = torch.tensor([1.0 if e in gold else 0.0 for e in G["e3set"]])
            p = mass / (mass.sum() + 1e-9)
            bl = bl + -(tgt / tgt.sum() * (p + 1e-9).log()).sum(); n += 1
            if n % 16 == 0: (bl / 16).backward(); opt.step(); opt.zero_grad(); bl = 0.0
    hit = hitN = mt = 0
    with torch.no_grad():
        for (topic, qt, ans, G) in te:
            mass = runner(m, qt, G); gold = set(ent2id[a] for a in ans if a in ent2id)
            massN = mass / (G["indeg"] + 1e-6)
            hit += G["e3set"][mass.argmax().item()] in gold
            hitN += G["e3set"][massN.argmax().item()] in gold; mt += 1
    print(f"  {name:<10} hits@1 raw {hit/max(mt,1):.3f}  degree-norm {hitN/max(mt,1):.3f}  ({time.time()-t0:.0f}s)")
    return max(hit, hitN) / max(mt, 1)

print("training both on the SAME subset...")
nf = train_eval("no-flow", run_noflow)
fl = train_eval("flow", run_flow)
print(f"\nPILOT VERDICT: flow best {fl:.3f} vs no-flow best {nf:.3f} | "
      f"{'GREEN: flow ahead even on least-coupled subset -> worth GPU on the coupled tail' if fl > nf + 0.02 else ('TIE/negative on this easy subset -> flow win (if any) is only on the coupled tail; GPU is a gamble' if fl <= nf + 0.02 else '')}")
