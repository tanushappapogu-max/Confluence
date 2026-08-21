"""2-hop Confluence v2: GRU question encoder + PER-HOP query heads.
Motivation: a 2-hop question ("who wrote the films directed by [X]") needs relation r1
at hop1 (directed_by) and r2 at hop2 (written_by). v1 used one mean-pooled vector for
both hops -> can't disambiguate. v2: GRU -> two heads q1,q2; hop1 edges scored with q1,
hop2 edges with q2. More data + epochs. Everything else (flow, legality) unchanged.
"""
import os, collections, random, time, torch, torch.nn as nn
torch.manual_seed(0); random.seed(0)
DDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metaqa")
d = 48; CAP = 80; NTRAIN = 8000; NTEST = 1500; EPOCHS = 5

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
            topic = q[q.find("[") + 1:q.find("]")]
            qt = (q[:q.find("[")] + q[q.find("]") + 1:]).lower()
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
    for (r, e1, dr) in h1: edges.append((1, o1 + id_e1[e1], (0, r, dr, e1)))          # hop tag 0
    for (e1, r2, e2, dr2) in h2edges: edges.append((o1 + id_e1[e1], o2 + id_e2[e2], (1, r2, dr2, e2)))  # hop tag 1
    for e2 in e2set: edges.append((o2 + id_e2[e2], sink, None))
    return dict(N=N, edges=edges, e2set=e2set, o2=o2, sink=sink)

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
print(f"KG {Ne} ent/{Nr} rel | 2-hop tractable: train {len(tr)} test {len(te)} | prep {time.time()-t0:.0f}s")

vocab = {"<unk>": 0, "<pad>": 1}
for _, qt, _, _ in tr:
    for w in qt.split(): vocab.setdefault(w, len(vocab))
V = len(vocab)
emb_ent = nn.Embedding(Ne, d); emb_rel = nn.Embedding(Nr, d); emb_tok = nn.Embedding(V, d); dir_emb = nn.Embedding(2, d)
gru = nn.GRU(d, d, batch_first=True)
head1 = nn.Linear(d, d); head2 = nn.Linear(d, d)
edge_mlp = nn.Sequential(nn.Linear(4 * d, 96), nn.GELU(), nn.Linear(96, 1))
mods = [emb_ent, emb_rel, emb_tok, dir_emb, gru, head1, head2, edge_mlp]
params = sum([list(m.parameters()) for m in mods], [])
opt = torch.optim.Adam(params, lr=2e-3)

def qheads(qt):
    ids = [vocab.get(w, 0) for w in qt.split()] or [0]
    x = emb_tok(torch.tensor(ids)).unsqueeze(0)
    _, h = gru(x); h = h.squeeze(0).squeeze(0)
    return head1(h), head2(h)

def run_flow(qt, G):
    q1, q2 = qheads(qt); edges = G["edges"]; N = G["N"]; costs = []
    for (i, j, key) in edges:
        if key is None: costs.append(torch.tensor(0.05))
        else:
            hop, r, dr, e = key; q = q1 if hop == 0 else q2
            f = torch.cat([q, emb_rel(torch.tensor(r)), dir_emb(torch.tensor(dr)), emb_ent(torch.tensor(e))])
            costs.append(torch.nn.functional.softplus(-edge_mlp(f).squeeze()) + 0.05)
    Lv = torch.stack(costs); E = len(edges); inc = torch.zeros(N, E)
    for e, (i, j, _) in enumerate(edges): inc[i, e] = 1; inc[j, e] = -1
    Dc = torch.ones(E); b = torch.zeros(N); b[0] = 1.0; b[G["sink"]] = -1.0
    keep = torch.tensor([x for x in range(N) if x != G["sink"]])
    for _ in range(18):
        w = Dc / Lv
        Lap = (inc @ torch.diag(w) @ inc.t())[keep][:, keep] + 1e-4 * torch.eye(N - 1)
        p = torch.zeros(N); p[keep] = torch.linalg.solve(Lap, b[keep])
        Q = w * (inc.t() @ p); Dc = (Dc + 0.25 * (Q.abs() - Dc)).clamp(min=1e-9)
    mass = torch.zeros(len(G["e2set"]))
    for e, (i, j, _) in enumerate(edges):
        if j == G["sink"]: mass[i - G["o2"]] = Q.abs()[e]
    return mass

print("training v2 (GRU + per-hop heads)...")
for ep in range(EPOCHS):
    random.shuffle(tr); tot = 0.0; n = 0; opt.zero_grad(); bl = 0.0; t0 = time.time()
    for (topic, qt, ans, G) in tr:
        mass = run_flow(qt, G); gold = set(ent2id[a] for a in ans if a in ent2id)
        tgt = torch.tensor([1.0 if e in gold else 0.0 for e in G["e2set"]])
        p = mass / (mass.sum() + 1e-9)
        bl = bl + -(tgt / tgt.sum() * (p + 1e-9).log()).sum(); n += 1
        if n % 16 == 0: (bl / 16).backward(); opt.step(); opt.zero_grad(); tot += bl.item(); bl = 0.0
    # eval each epoch: raw-mass decode vs in-degree-normalized decode (degree-bias test)
    hit = 0; hitN = 0; m = 0
    with torch.no_grad():
        for (topic, qt, ans, G) in te:
            mass = run_flow(qt, G); gold = set(ent2id[a] for a in ans if a in ent2id)
            indeg = torch.zeros(len(G["e2set"]))
            for (i, j, key) in G["edges"]:
                if key is not None and key[0] == 1: indeg[j - G["o2"]] += 1   # hop2 edges into e2
            massN = mass / (indeg + 1e-6)
            hit += G["e2set"][mass.argmax().item()] in gold
            hitN += G["e2set"][massN.argmax().item()] in gold; m += 1
    print(f"  epoch {ep}: loss {tot/max(n,1):.3f}  hits@1 raw {hit/max(m,1):.3f}  degree-norm {hitN/max(m,1):.3f}  ({time.time()-t0:.0f}s)")
print("\nv1 was 0.833. If degree-norm >> raw, the plateau is connectivity bias -> retrain with it. 0 hallucination by construction.")
