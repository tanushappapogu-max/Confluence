"""First REAL-DATA run: Confluence flow on MetaQA 1-hop.
Per query: build the small reasoning graph from the topic entity's real KG edges,
route a conserved flow to the answer entity. Legality (answer is a real KG neighbor
via a real relation) is guaranteed by construction. Compare to an unconstrained
similarity classifier that can name ANY of 43k entities (and thus hallucinate).
"""
import os, collections, random, time, torch, torch.nn as nn
torch.manual_seed(0); random.seed(0)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metaqa")
d = 32

# ---------- load KB ----------
ent2id, rel2id = {}, {}
adj_full = collections.defaultdict(list)     # entity -> list of (rel, other, dir)  dir: 0 fwd(head), 1 rev(tail)
with open(os.path.join(D, "kb", "kb.txt")) as f:
    for line in f:
        s, r, o = line.rstrip("\n").split("|")
        for e in (s, o): ent2id.setdefault(e, len(ent2id))
        rel2id.setdefault(r, len(rel2id))
        h, rr, t = ent2id[s], rel2id[r], ent2id[o]
        adj_full[h].append((rr, t, 0)); adj_full[t].append((rr, h, 1))
Ne, Nr = len(ent2id), len(rel2id)

# ---------- load 1-hop QA ----------
def load_qa(split):
    out = []
    with open(os.path.join(D, "1-hop", f"qa_{split}.txt")) as f:
        for line in f:
            q, ans = line.rstrip("\n").split("\t")
            topic = q[q.find("[") + 1:q.find("]")]
            qtext = (q[:q.find("[")] + q[q.find("]") + 1:]).lower()
            out.append((topic, qtext, ans.split("|")))
    return out
tr = load_qa("train"); te = load_qa("test")
random.shuffle(tr); tr = tr[:6000]; te = te[:2000]

# ---------- vocab ----------
vocab = {"<unk>": 0}
for _, qt, _ in tr:
    for w in qt.split(): vocab.setdefault(w, len(vocab))
def toks(qt): return [vocab.get(w, 0) for w in qt.split()] or [0]
V = len(vocab)
print(f"KG {Ne} ent / {Nr} rel | train {len(tr)} test {len(te)} | vocab {V}")

# ---------- model ----------
emb_ent = nn.Embedding(Ne, d); emb_rel = nn.Embedding(Nr, d); emb_tok = nn.Embedding(V, d)
dir_emb = nn.Embedding(2, d)
edge_mlp = nn.Sequential(nn.Linear(4 * d, 64), nn.GELU(), nn.Linear(64, 1))
params = list(emb_ent.parameters()) + list(emb_rel.parameters()) + list(emb_tok.parameters()) \
    + list(dir_emb.parameters()) + list(edge_mlp.parameters())
opt = torch.optim.Adam(params, lr=3e-3)

def qvec(qt):
    t = torch.tensor(toks(qt)); return emb_tok(t).mean(0)

def flow_answer(topic, qt):
    """build 1-hop reasoning graph, return (candidate_ids, answer-mass over candidates)."""
    cands = adj_full[ent2id[topic]]
    if not cands: return [], None
    cand_ids = [c[1] for c in cands]
    qv = qvec(qt)
    rel = emb_rel(torch.tensor([c[0] for c in cands]))
    dr = dir_emb(torch.tensor([c[2] for c in cands]))
    ce = emb_ent(torch.tensor(cand_ids))
    feat = torch.cat([qv.expand(len(cands), d), rel, dr, ce], 1)
    score = edge_mlp(feat).squeeze(-1)                       # higher=cheaper
    cost = torch.nn.functional.softplus(-score) + 0.05
    # tiny star graph: source->topic (free) ; topic->cand_i (cost_i) ; cand_i->sink (free)
    C = len(cands); N = C + 3                                # src=0, topic=1, cands=2..2+C-1, sink=N-1
    inc = torch.zeros(N, 1 + 2 * C)
    inc[0, 0] = 1; inc[1, 0] = -1                            # src->topic
    L = [torch.tensor(0.05)]
    for i in range(C):
        e1 = 1 + i; inc[1, e1] = 1; inc[2 + i, e1] = -1; L.append(cost[i])       # topic->cand
        e2 = 1 + C + i; inc[2 + i, e2] = 1; inc[N - 1, e2] = -1; L.append(torch.tensor(0.05))  # cand->sink
    Lv = torch.stack(L)
    Dc = torch.ones(1 + 2 * C); b = torch.zeros(N); b[0] = 1.0; b[N - 1] = -1.0
    keep = list(range(N - 1)); kt = torch.tensor(keep)
    for _ in range(20):
        w = Dc / Lv
        Lap = (inc @ torch.diag(w) @ inc.t())[kt][:, kt] + 1e-4 * torch.eye(N - 1)
        p = torch.zeros(N); p[kt] = torch.linalg.solve(Lap, b[kt])
        Q = w * (inc.t() @ p)
        Dc = (Dc + 0.25 * (Q.abs() - Dc)).clamp(min=1e-9)
    ans_mass = Q.abs()[1 + C:1 + 2 * C]                     # flux on cand->sink edges
    return cand_ids, ans_mass

print("training Confluence flow on MetaQA 1-hop...")
for ep in range(3):
    random.shuffle(tr); tot = 0.0; n = 0; t0 = time.time()
    opt.zero_grad(); batch_loss = 0.0
    for qi, (topic, qt, ans) in enumerate(tr):
        cand_ids, mass = flow_answer(topic, qt)
        if mass is None: continue
        gold = set(ent2id[a] for a in ans if a in ent2id)
        tgt = torch.tensor([1.0 if c in gold else 0.0 for c in cand_ids])
        if tgt.sum() == 0: continue
        p = mass / (mass.sum() + 1e-9)
        loss = -(tgt / tgt.sum() * (p + 1e-9).log()).sum()
        batch_loss = batch_loss + loss; n += 1
        if n % 32 == 0:
            (batch_loss / 32).backward(); opt.step(); opt.zero_grad(); tot += batch_loss.item(); batch_loss = 0.0
    print(f"  epoch {ep}: loss {tot/max(n,1):.3f}  ({time.time()-t0:.0f}s)")

# ---------- eval: hits@1 + legality ----------
def evaluate():
    hit = 0; halluc = 0; n = 0
    with torch.no_grad():
        for topic, qt, ans in te:
            cand_ids, mass = flow_answer(topic, qt)
            if mass is None: continue
            gold = set(ent2id[a] for a in ans if a in ent2id)
            pred = cand_ids[mass.argmax().item()]
            hit += pred in gold; n += 1
            # legality: our pred is always a real KG neighbor by construction -> halluc impossible
    return hit / max(n, 1), n
acc, n = evaluate()
print(f"\nConfluence 1-hop hits@1 = {acc:.3f} on {n} test Q")
print("legality: predicted answer is ALWAYS a real KG neighbor via a real relation (0 hallucination by construction)")

# ---------- baseline: unconstrained similarity over ALL 43k entities ----------
print("\nbaseline: score query vs ALL entity embeddings (can name any of 43k -> can hallucinate)")
opt2 = torch.optim.Adam(params, lr=3e-3)   # reuse embeddings, add a query->entity scorer
for ep in range(2):
    random.shuffle(tr); tot = 0.0; n = 0
    for i in range(0, len(tr), 64):
        batch = tr[i:i+64]; loss = 0.0; c = 0
        for topic, qt, ans in batch:
            gold = [ent2id[a] for a in ans if a in ent2id]
            if not gold: continue
            logits = emb_ent.weight @ qvec(qt)              # [Ne] similarity to every entity
            loss = loss - torch.log_softmax(logits, 0)[gold].mean(); c += 1
        if c: (loss / c).backward(); opt2.step(); opt2.zero_grad(); tot += loss.item(); n += c
# eval baseline legality + acc
bhit = 0; bh = 0; bn = 0
with torch.no_grad():
    for topic, qt, ans in te[:500]:
        gold = set(ent2id[a] for a in ans if a in ent2id)
        pred = (emb_ent.weight @ qvec(qt)).argmax().item()
        neigh = set(c[1] for c in adj_full[ent2id[topic]])
        bhit += pred in gold; bh += pred not in neigh; bn += 1
print(f"baseline hits@1 = {bhit/bn:.3f} | HALLUCINATION rate (answer not a KG neighbor) = {bh/bn:.3f}")
print("\n^ the wedge on real data: our answer is guaranteed legal; the free classifier names non-neighbors.")
