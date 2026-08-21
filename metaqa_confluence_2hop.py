"""REAL 2-hop experiment: Confluence flow on MetaQA 2-hop.
Per query: build layered reasoning graph  source -> topic -> {hop1 entities} -> {hop2 entities} -> sink,
edges = real KG triples. Flow routes a legal 2-hop path; answer = hop2 entity with most flux.
Legality (answer reachable by a real 2-hop path) guaranteed by construction.
First run: filter to tractable neighborhoods (<= CAP nodes); heavy-degree tail deferred to the sparse solve.
"""
import os, collections, random, time, torch, torch.nn as nn
torch.manual_seed(0); random.seed(0)
DDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metaqa")
d = 32; CAP = 80

# ---------- KB ----------
ent2id, rel2id = {}, {}
adj = collections.defaultdict(list)     # ent -> [(rel, other, dir)]
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
            topic = q[q.find("[") + 1:q.find("]")]
            qt = (q[:q.find("[")] + q[q.find("]") + 1:]).lower()
            out.append((topic, qt, ans.split("|")))
    return out

# ---------- build layered 2-hop graph; return None if too big ----------
def build(topic_id):
    h1 = adj[topic_id]                                  # (rel, e1, dir)
    e1set = list({x[1] for x in h1})
    if not e1set: return None
    h2edges = []                                        # (e1, rel, e2, dir)
    e2set = set()
    for (r, e1, dr) in h1:
        for (r2, e2, dr2) in adj[e1]:
            h2edges.append((e1, r2, e2, dr2)); e2set.add(e2)
    e2set = list(e2set)
    n_nodes = 3 + len(e1set) + len(e2set)               # src, topic, sink + layers
    if n_nodes > CAP: return None
    id_e1 = {e: i for i, e in enumerate(e1set)}
    id_e2 = {e: i for i, e in enumerate(e2set)}
    # node indices: 0 src, 1 topic, 2..2+|e1|-1 hop1, then hop2, then sink last
    o1 = 2; o2 = 2 + len(e1set); sink = o2 + len(e2set); N = sink + 1
    edges = []                                          # (i,j, feat_key)  feat_key=(rel,dir,ent) or None
    edges.append((0, 1, None))                          # src->topic
    for (r, e1, dr) in h1: edges.append((1, o1 + id_e1[e1], (r, dr, e1)))
    for (e1, r2, e2, dr2) in h2edges: edges.append((o1 + id_e1[e1], o2 + id_e2[e2], (r2, dr2, e2)))
    for e2 in e2set: edges.append((o2 + id_e2[e2], sink, None))
    return dict(N=N, edges=edges, e2set=e2set, o2=o2, id_e2=id_e2, sink=sink)

# ---------- prepare data (filter to tractable) ----------
def prep(split, limit):
    qa = load_qa(split); random.shuffle(qa); out = []
    for topic, qt, ans in qa:
        if topic not in ent2id: continue
        G = build(ent2id[topic])
        if G is None: continue
        gold = set(ent2id[a] for a in ans if a in ent2id)
        golds_in = [e for e in G["e2set"] if e in gold]
        if not golds_in: continue                       # answer not reachable in (pruned) graph
        out.append((topic, qt, ans, G))
        if len(out) >= limit: break
    return out
t0 = time.time()
tr = prep("train", 3000); te = prep("test", 1000)
print(f"KG {Ne} ent/{Nr} rel | 2-hop tractable(<= {CAP} nodes): train {len(tr)} test {len(te)} | prep {time.time()-t0:.0f}s")

# ---------- vocab + model ----------
vocab = {"<unk>": 0}
for _, qt, _, _ in tr:
    for w in qt.split(): vocab.setdefault(w, len(vocab))
V = len(vocab)
emb_ent = nn.Embedding(Ne, d); emb_rel = nn.Embedding(Nr, d); emb_tok = nn.Embedding(V, d); dir_emb = nn.Embedding(2, d)
edge_mlp = nn.Sequential(nn.Linear(4 * d, 64), nn.GELU(), nn.Linear(64, 1))
params = sum([list(m.parameters()) for m in (emb_ent, emb_rel, emb_tok, dir_emb, edge_mlp)], [])
opt = torch.optim.Adam(params, lr=3e-3)
def qvec(qt):
    t = torch.tensor([vocab.get(w, 0) for w in qt.split()] or [0]); return emb_tok(t).mean(0)

def run_flow(qt, G):
    qv = qvec(qt); edges = G["edges"]; N = G["N"]
    costs = []
    for (i, j, key) in edges:
        if key is None: costs.append(torch.tensor(0.05))
        else:
            r, dr, e = key
            f = torch.cat([qv, emb_rel(torch.tensor(r)), dir_emb(torch.tensor(dr)), emb_ent(torch.tensor(e))])
            costs.append(torch.nn.functional.softplus(-edge_mlp(f).squeeze()) + 0.05)
    Lv = torch.stack(costs); E = len(edges)
    inc = torch.zeros(N, E)
    for e, (i, j, _) in enumerate(edges): inc[i, e] = 1; inc[j, e] = -1
    Dc = torch.ones(E); b = torch.zeros(N); b[0] = 1.0; b[G["sink"]] = -1.0
    keep = [x for x in range(N) if x != G["sink"]]; kt = torch.tensor(keep)
    for _ in range(18):
        w = Dc / Lv
        Lap = (inc @ torch.diag(w) @ inc.t())[kt][:, kt] + 1e-4 * torch.eye(N - 1)
        p = torch.zeros(N); p[kt] = torch.linalg.solve(Lap, b[kt])
        Q = w * (inc.t() @ p)
        Dc = (Dc + 0.25 * (Q.abs() - Dc)).clamp(min=1e-9)
    # answer mass per e2 = flux on its ->sink edge
    mass = torch.zeros(len(G["e2set"]))
    for e, (i, j, _) in enumerate(edges):
        if j == G["sink"]: mass[i - G["o2"]] = Q.abs()[e]
    return mass

print("training 2-hop flow...")
for ep in range(3):
    random.shuffle(tr); tot = 0.0; n = 0; opt.zero_grad(); bl = 0.0; t0 = time.time()
    for (topic, qt, ans, G) in tr:
        mass = run_flow(qt, G)
        gold = set(ent2id[a] for a in ans if a in ent2id)
        tgt = torch.tensor([1.0 if e in gold else 0.0 for e in G["e2set"]])
        p = mass / (mass.sum() + 1e-9)
        loss = -(tgt / tgt.sum() * (p + 1e-9).log()).sum(); bl = bl + loss; n += 1
        if n % 16 == 0:
            (bl / 16).backward(); opt.step(); opt.zero_grad(); tot += bl.item(); bl = 0.0
    print(f"  epoch {ep}: loss {tot/max(n,1):.3f}  ({time.time()-t0:.0f}s)")

# ---------- eval ----------
hit = 0; n = 0
with torch.no_grad():
    for (topic, qt, ans, G) in te:
        mass = run_flow(qt, G)
        gold = set(ent2id[a] for a in ans if a in ent2id)
        pred = G["e2set"][mass.argmax().item()]
        hit += pred in gold; n += 1
print(f"\nConfluence 2-hop hits@1 = {hit/max(n,1):.3f} on {n} tractable test Q")
print("legality: predicted answer reached by a REAL 2-hop KG path (0 hallucination by construction)")
print(f"[first 2-hop run: tractable subset only (<= {CAP} nodes); heavy tail -> sparse solve next]")
