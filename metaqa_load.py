"""MetaQA loader + stats. Parses kb.txt and qa files, reports the KG structure and
-- crucially -- the k-hop neighborhood sizes that determine how big the per-query
flow graphs will be (i.e. is this tractable, and do we need sparsification/implicit-diff)."""
import os, collections, statistics as st, random
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metaqa")

# ---- KB ----
ent2id, rel2id = {}, {}
triples = []
adj = collections.defaultdict(list)      # head_id -> list of (rel_id, tail_id)
adj_any = collections.defaultdict(set)   # head_id -> set of tail_ids (any relation), undirected-ish
with open(os.path.join(D, "kb", "kb.txt")) as f:
    for line in f:
        s, r, o = line.rstrip("\n").split("|")
        for e in (s, o): ent2id.setdefault(e, len(ent2id))
        rel2id.setdefault(r, len(rel2id))
        h, rr, t = ent2id[s], rel2id[r], ent2id[o]
        triples.append((h, rr, t)); adj[h].append((rr, t))
        adj_any[h].add(t); adj_any[t].add(h)      # treat KG as undirected for neighborhood expansion
Ne, Nr = len(ent2id), len(rel2id)
print(f"KG: {Ne} entities, {Nr} relations, {len(triples)} triples")
print("relations:", list(rel2id.keys()))
degs = [len(adj_any[e]) for e in range(Ne)]
print(f"degree (undirected): mean={st.mean(degs):.1f} median={sorted(degs)[len(degs)//2]} max={max(degs)}")

# ---- QA ----
def load_qa(hop, split):
    out = []
    with open(os.path.join(D, f"{hop}-hop", f"qa_{split}.txt")) as f:
        for line in f:
            q, ans = line.rstrip("\n").split("\t")
            topic = q[q.find("[") + 1: q.find("]")]
            answers = ans.split("|")
            out.append((topic, q, answers))
    return out
for hop in (1, 2, 3):
    qa = load_qa(hop, "train")
    miss = sum(1 for t, _, _ in qa if t not in ent2id)
    print(f"{hop}-hop train: {len(qa)} questions, topic-not-in-KG: {miss}")

# ---- neighborhood sizes (graph size per query) ----
def khop_size(topic_id, k):
    frontier = {topic_id}; seen = {topic_id}
    for _ in range(k):
        nxt = set()
        for e in frontier: nxt |= adj_any[e]
        nxt -= seen; seen |= nxt; frontier = nxt
    return len(seen)
random.seed(0)
for hop in (1, 2, 3):
    qa = load_qa(hop, "train")
    sample = random.sample(qa, 200)
    sizes = [khop_size(ent2id[t], hop) for t, _, _ in sample if t in ent2id]
    sizes.sort()
    print(f"{hop}-hop: per-query {hop}-hop neighborhood size  "
          f"median={sizes[len(sizes)//2]}  p90={sizes[int(0.9*len(sizes))]}  max={max(sizes)}")
print("\n(neighborhood size = #entity-nodes in the per-query flow graph. small=tractable now;")
print(" large=needs candidate pruning / sparsified solve before the full run.)")
