"""2-hop NO-FLOW ablation (the control that isolates Confluence's contribution).

Question this experiment answers: does the conserved min-cost flow buy ACCURACY, or
only the hard legality guarantee? To find out we hold EVERYTHING else fixed and remove
only the flow.

Shared with metaqa_confluence_2hop_v2.py (identical):
  - GRU question encoder + per-hop query heads q1, q2
  - entity / relation / direction embeddings, edge_mlp scorer
  - the per-query candidate graph (topic -> e1 -> e2), so legality-by-topology is
    IDENTICAL: only real KG edges are ever scored, illegal hops impossible in both.
  - in-degree-normalized decode option (degree-bias control)

Changed (the ONLY difference):
  - v2 selects the answer by a Tero-Nakagaki Physarum conserved-flow solve
    (weighted-Laplacian solve + reinforcement to a min-cost path, coupled across hops).
  - This ablation selects by INDEPENDENT PER-HOP SOFTMAX:
        p(e1)      = softmax over hop1 edge scores (q1)
        p(e2 | e1) = softmax over hop2 edge scores (q2), per e1
        mass(e2)   = sum_e1 p(e1) * p(e2 | e1)
    No Laplacian, no conservation, no reinforcement, no coupling budget.

Interpretation:
  - ablation << v2  -> the flow itself contributes accuracy (coupling / min-cost matter).
  - ablation ~= v2  -> Confluence's contribution is the GUARANTEE + unification, not raw
                       accuracy on this task. Both outcomes are honest paper results.

Run AFTER data is present (data/metaqa/{kb/kb.txt, 2-hop/qa_{train,test}.txt}).
Reuses v2's loader/graph builder verbatim so the comparison is apples-to-apples.
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
    """Per-query graph, IDENTICAL to v2. Returns hop1/hop2 edge lists so the ablation
    can score the same edges the flow would traverse."""
    h1 = adj[topic_id]; e1set = list({x[1] for x in h1})
    if not e1set: return None
    h2edges = []; e2set = set()
    for (r, e1, dr) in h1:
        for (r2, e2, dr2) in adj[e1]:
            h2edges.append((e1, r2, e2, dr2)); e2set.add(e2)
    e2set = list(e2set)
    if 3 + len(e1set) + len(e2set) > CAP: return None
    id_e1 = {e: i for i, e in enumerate(e1set)}; id_e2 = {e: i for i, e in enumerate(e2set)}
    # hop1 edges: (i_e1, r, dr, e1_ent) ;  hop2 edges: (i_e1, i_e2, r2, dr2, e2_ent)
    hop1 = [(id_e1[e1], r, dr, e1) for (r, e1, dr) in h1]
    hop2 = [(id_e1[e1], id_e2[e2], r2, dr2, e2) for (e1, r2, e2, dr2) in h2edges]
    return dict(e1set=e1set, e2set=e2set, id_e1=id_e1, id_e2=id_e2, hop1=hop1, hop2=hop2)

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
# SAME modules as v2 so capacity is matched exactly.
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

def edge_score(q, r, dr, e):
    f = torch.cat([q, emb_rel(torch.tensor(r)), dir_emb(torch.tensor(dr)), emb_ent(torch.tensor(e))])
    return edge_mlp(f).squeeze()          # raw logit (NO softplus/cost transform: this is a scorer, not a cost)

def run_noflow(qt, G):
    """Independent per-hop softmax over the SAME candidate edges -- no conservation solve."""
    q1, q2 = qheads(qt)
    n1, n2 = len(G["e1set"]), len(G["e2set"])
    # hop1: p(e1) = softmax over hop1 edge scores, aggregated to e1 nodes (logsumexp over parallel edges)
    s1 = torch.full((n1,), -1e9)
    for (i1, r, dr, e1) in G["hop1"]:
        s1[i1] = torch.logaddexp(s1[i1], edge_score(q1, r, dr, e1))
    p1 = torch.softmax(s1, 0)
    # hop2: for each e1, p(e2|e1) = softmax over its outgoing hop2 edges; mass(e2) = sum_e1 p(e1) p(e2|e1)
    s2 = torch.full((n1, n2), -1e9)
    for (i1, i2, r2, dr2, e2) in G["hop2"]:
        s2[i1, i2] = torch.logaddexp(s2[i1, i2], edge_score(q2, r2, dr2, e2))
    p2 = torch.softmax(s2, 1)             # [n1, n2]
    mass = (p1.unsqueeze(1) * p2).sum(0)  # [n2]
    return mass

def indeg_vec(G):
    indeg = torch.zeros(len(G["e2set"]))
    for (i1, i2, r2, dr2, e2) in G["hop2"]: indeg[i2] += 1
    return indeg

print("training NO-FLOW ablation (independent per-hop softmax, same encoder/graph)...")
for ep in range(EPOCHS):
    random.shuffle(tr); tot = 0.0; n = 0; opt.zero_grad(); bl = 0.0; t0 = time.time()
    for (topic, qt, ans, G) in tr:
        mass = run_noflow(qt, G); gold = set(ent2id[a] for a in ans if a in ent2id)
        tgt = torch.tensor([1.0 if e in gold else 0.0 for e in G["e2set"]])
        p = mass / (mass.sum() + 1e-9)
        bl = bl + -(tgt / tgt.sum() * (p + 1e-9).log()).sum(); n += 1
        if n % 16 == 0: (bl / 16).backward(); opt.step(); opt.zero_grad(); tot += bl.item(); bl = 0.0
    hit = 0; hitN = 0; m = 0
    with torch.no_grad():
        for (topic, qt, ans, G) in te:
            mass = run_noflow(qt, G); gold = set(ent2id[a] for a in ans if a in ent2id)
            massN = mass / (indeg_vec(G) + 1e-6)
            hit += G["e2set"][mass.argmax().item()] in gold
            hitN += G["e2set"][massN.argmax().item()] in gold; m += 1
    print(f"  epoch {ep}: loss {tot/max(n,1):.3f}  hits@1 raw {hit/max(m,1):.3f}  degree-norm {hitN/max(m,1):.3f}  ({time.time()-t0:.0f}s)")
print("\nCompare against v2 (flow). If flow >> this at matched encoder/graph, the conserved")
print("flow contributes accuracy. If ~equal, Confluence's contribution is the guarantee +")
print("unification, not raw accuracy on this task. Legality (0 hallucination) holds in BOTH.")
