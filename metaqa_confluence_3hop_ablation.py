"""3-hop MetaQA NO-FLOW baseline (the real-data coupled-regime test, independent-scoring side).

3-hop is where reasoning steps are more coupled than 2-hop, so the coupling thesis predicts the
conserved flow should beat this independent per-hop baseline here (unlike the 2-hop tie). This
script is the independent-scoring control: same encoder/graph/legality, path scored as
independent per-hop softmax then marginalized to answers. Run the flow model on the SAME prep to
complete the head-to-head (the flow side is compute-heavy -> best on a GPU locally).

Graph: source -> hub -> {e1} -> {e2} -> {e3=answers} -> sink, legal KG edges only.
Encoder: GRU + three per-hop query heads q1,q2,q3.
Metric: hits@1 (raw and in-degree-normalized), 0 hallucination by construction.

Needs data/metaqa/{kb/kb.txt, 3-hop/qa_{train,test}.txt}. CAP bounds neighborhood size.
"""
import os, collections, random, time, torch, torch.nn as nn
torch.manual_seed(0); random.seed(0)
DDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metaqa")
d = 48; CAP = int(os.environ.get("CAP", "120")); NTRAIN = int(os.environ.get("NTRAIN", "6000"))
NTEST = int(os.environ.get("NTEST", "1200")); EPOCHS = int(os.environ.get("EPOCHS", "5"))

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
    """3-hop expansion with de-duped node layers; returns edge lists per hop + answer set (e3)."""
    h1 = adj[topic_id]; e1set = list({x[1] for x in h1})
    if not e1set: return None
    e2set = set(); hop2 = []
    for (r, e1, dr) in h1:
        for (r2, e2, dr2) in adj[e1]: e2set.add(e2); hop2.append((e1, r2, e2, dr2))
    e2set = list(e2set)
    e3set = set(); hop3 = []
    for e2 in e2set:
        for (r3, e3, dr3) in adj[e2]: e3set.add(e3); hop3.append((e2, r3, e3, dr3))
    e3set = list(e3set)
    if 4 + len(e1set) + len(e2set) + len(e3set) > CAP: return None
    ide1 = {e: i for i, e in enumerate(e1set)}; ide2 = {e: i for i, e in enumerate(e2set)}; ide3 = {e: i for i, e in enumerate(e3set)}
    hop1 = [(ide1[e1], r, dr, e1) for (r, e1, dr) in h1]
    H2 = [(ide1[e1], ide2[e2], r2, dr2, e2) for (e1, r2, e2, dr2) in hop2]
    H3 = [(ide2[e2], ide3[e3], r3, dr3, e3) for (e2, r3, e3, dr3) in hop3]
    indeg = torch.zeros(len(e3set))
    for (_, i3, _, _, _) in H3: indeg[i3] += 1
    return dict(e1set=e1set, e2set=e2set, e3set=e3set, hop1=hop1, hop2=H2, hop3=H3, indeg=indeg)

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
print(f"KG {Ne} ent/{Nr} rel | 3-hop (CAP={CAP}): train {len(tr)} test {len(te)} | prep {time.time()-t0:.0f}s")

vocab = {"<unk>": 0}
for _, qt, _, _ in tr:
    for w in qt.split(): vocab.setdefault(w, len(vocab))
V = len(vocab)
emb_ent = nn.Embedding(Ne, d); emb_rel = nn.Embedding(Nr, d); emb_tok = nn.Embedding(V, d); dir_emb = nn.Embedding(2, d)
gru = nn.GRU(d, d, batch_first=True)
heads = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
edge_mlp = nn.Sequential(nn.Linear(4*d, 96), nn.GELU(), nn.Linear(96, 1))
mods = [emb_ent, emb_rel, emb_tok, dir_emb, gru, heads, edge_mlp]
params = sum([list(m.parameters()) for m in mods], [])
opt = torch.optim.Adam(params, lr=2e-3)

def qheads(qt):
    ids = [vocab.get(w, 0) for w in qt.split()] or [0]
    x = emb_tok(torch.tensor(ids)).unsqueeze(0); _, h = gru(x); h = h.squeeze(0).squeeze(0)
    return [hd(h) for hd in heads]

def escore(q, R, Dr, Ent):
    qx = q.unsqueeze(0).expand(R.shape[0], -1)
    f = torch.cat([qx, emb_rel(R), dir_emb(Dr), emb_ent(Ent)], 1)
    return edge_mlp(f).squeeze(1)

def run_noflow(qt, G):
    q1, q2, q3 = qheads(qt)
    n1, n2, n3 = len(G["e1set"]), len(G["e2set"]), len(G["e3set"])
    h1, h2, h3 = G["hop1"], G["hop2"], G["hop3"]
    if not h1 or not h2 or not h3: return torch.zeros(n3)
    s1e = escore(q1, torch.tensor([e[1] for e in h1]), torch.tensor([e[2] for e in h1]), torch.tensor([e[3] for e in h1]))
    s1 = torch.full((n1,), -1e9).scatter_reduce(0, torch.tensor([e[0] for e in h1]), s1e, reduce="amax", include_self=True)
    s2e = escore(q2, torch.tensor([e[2] for e in h2]), torch.tensor([e[3] for e in h2]), torch.tensor([e[4] for e in h2]))
    par2, chi2 = torch.tensor([e[0] for e in h2]), torch.tensor([e[1] for e in h2])
    s2 = torch.full((n2,), -1e9).scatter_reduce(0, chi2, s1[par2] + s2e, reduce="amax", include_self=True)  # best path score into each e2
    s3e = escore(q3, torch.tensor([e[2] for e in h3]), torch.tensor([e[3] for e in h3]), torch.tensor([e[4] for e in h3]))
    par3, chi3 = torch.tensor([e[0] for e in h3]), torch.tensor([e[1] for e in h3])
    p_path = torch.softmax(s2[par3] + s3e, 0)                # softmax over all legal 3-hop paths
    return torch.zeros(n3).scatter_add(0, chi3, p_path)      # marginalize to e3 answers

print(f"training 3-hop NO-FLOW baseline...")
for ep in range(EPOCHS):
    random.shuffle(tr); tot = 0.0; n = 0; opt.zero_grad(); bl = 0.0; t0 = time.time()
    for (topic, qt, ans, G) in tr:
        mass = run_noflow(qt, G); gold = set(ent2id[a] for a in ans if a in ent2id)
        tgt = torch.tensor([1.0 if e in gold else 0.0 for e in G["e3set"]])
        p = mass / (mass.sum() + 1e-9)
        bl = bl + -(tgt / tgt.sum() * (p + 1e-9).log()).sum(); n += 1
        if n % 16 == 0: (bl / 16).backward(); opt.step(); opt.zero_grad(); tot += bl.item(); bl = 0.0
    hit = hitN = m = 0
    with torch.no_grad():
        for (topic, qt, ans, G) in te:
            mass = run_noflow(qt, G); gold = set(ent2id[a] for a in ans if a in ent2id)
            massN = mass / (G["indeg"] + 1e-6)
            hit += G["e3set"][mass.argmax().item()] in gold
            hitN += G["e3set"][massN.argmax().item()] in gold; m += 1
    print(f"  epoch {ep}: loss {tot/max(n,1):.3f}  hits@1 raw {hit/max(m,1):.3f}  degree-norm {hitN/max(m,1):.3f}  ({time.time()-t0:.0f}s)")
print("\n3-hop independent-scoring baseline. Run the flow model on the same prep for the head-to-head:")
print("if flow >> this, the coupling thesis holds on REAL data (the result that lifts to main-track).")
