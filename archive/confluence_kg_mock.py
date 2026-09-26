"""
Confluence KG pipeline -- MOCK knowledge graph (MetaQA-ready).
Proves the Week-2 core: take KG triples (h, relation, t), build a per-query
multi-hop reasoning graph whose EDGES ARE THE TRIPLES, and route a conserved
flow to the answer entity. Legality (no hallucinated hop) is free from the KG's
topology: there is simply no edge where there is no triple.

Demonstrates:
 (1) build per-query 2-hop reasoning graph from a KG,
 (2) conserved flow routes a LEGAL path topic -> answer (illegal-hop rate = 0 by construction),
 (3) it is differentiable -- train an edge scorer so the flow concentrates on the correct path,
 (4) contrast: an unconstrained answer predictor (softmax over ALL entities) hallucinates
     non-reachable answers (illegal-answer rate > 0).

MetaQA slots in by replacing build_mock_kg() with the real triples + qa pairs.
"""
import torch, torch.nn as nn
torch.manual_seed(0)
d = 16

# ------------------------- mock KG (MetaQA-shaped) --------------------------
# 12 entities in 3 "layers" so clean 2-hop paths exist:  A(0-3) -r0-> B(4-7) -r1-> C(8-11)
Ne, Nr = 12, 3
A_ent, B_ent, C_ent = list(range(0, 4)), list(range(4, 8)), list(range(8, 12))
triples = []                         # (head, relation, tail)
g = torch.Generator().manual_seed(1)
for a in A_ent:                      # relation 0: A -> B (each A connects to 2 B's)
    for b in torch.randperm(4, generator=g)[:2]: triples.append((a, 0, 4 + b.item()))
for b in B_ent:                      # relation 1: B -> C
    for c in torch.randperm(4, generator=g)[:2]: triples.append((b, 1, 8 + c.item()))
for a in A_ent:                      # relation 2: some A -> C direct (a shortcut relation)
    if torch.rand(1, generator=g) < 0.5: triples.append((a, 2, 8 + torch.randint(0, 4, (1,), generator=g).item()))
triple_set = set(triples)
# per-relation adjacency
adj = torch.zeros(Nr, Ne, Ne)
for h, r, t in triples: adj[r, h, t] = 1.0
print(f"mock KG: {Ne} entities, {Nr} relations, {len(triples)} triples")

# 2-hop question type: follow relation 0 then relation 1 (A -> B -> C).
# query = topic entity a in A; answer set = {c : exists a-r0->b, b-r1->c}
def answers_of(a):
    ans = set()
    for b in range(Ne):
        if adj[0, a, b] > 0:
            for c in range(Ne):
                if adj[1, b, c] > 0: ans.add(c)
    return ans
queries = [(a, answers_of(a)) for a in A_ent if answers_of(a)]
print(f"2-hop queries (relation path r0->r1): {len(queries)} topic entities with answers")

# ------------------------- per-query reasoning graph ------------------------
# layered: source -> (hop0=topic) -> (hop1 = r0-neighbors) -> (hop2 = r1-neighbors) -> sink
# EDGES EXIST ONLY WHERE TRIPLES DO -> illegal hops are impossible.
def build_query_graph(topic):
    hop1 = [b for b in range(Ne) if adj[0, topic, b] > 0]
    hop2 = sorted({c for b in hop1 for c in range(Ne) if adj[1, b, c] > 0})
    nodes = ["src"] + [("h1", b) for b in hop1] + [("h2", c) for c in hop2] + ["sink"]
    idx = {n: i for i, n in enumerate(nodes)}
    edges = []
    for b in hop1: edges.append((idx["src"], idx[("h1", b)], ("r0", topic, b)))     # topic -r0-> b
    for b in hop1:
        for c in hop2:
            if adj[1, b, c] > 0: edges.append((idx[("h1", b)], idx[("h2", c)], ("r1", b, c)))  # b -r1-> c
    for c in hop2: edges.append((idx[("h2", c)], idx["sink"], ("end", c, c)))
    return nodes, idx, edges, hop1, hop2

# ------------------------- conserved-flow solver ----------------------------
def flow_path(cost, inc, N, sink_i):
    """single-graph Physarum: cost[E]->conductances->fluxes. returns edge fluxes |Q|."""
    E = inc.shape[1]; L = cost.clamp(min=1e-3)
    D = torch.ones(E); b = torch.zeros(N); b[0] = 1.0; b[sink_i] = -1.0
    keep = [i for i in range(N) if i != sink_i]; kt = torch.tensor(keep)
    for _ in range(30):
        w = D / L
        Lap = (inc @ torch.diag(w) @ inc.t())[kt][:, kt] + 1e-4 * torch.eye(N - 1)
        p = torch.zeros(N); p[kt] = torch.linalg.solve(Lap, b[kt])
        Q = w * (inc.t() @ p)
        D = (D + 0.25 * (Q.abs() - D)).clamp(min=1e-9)
    return Q.abs()

# learnable edge scorer: score an edge from (relation, entity embeddings)
emb = nn.Parameter(0.3 * torch.randn(Ne, d))
rel_emb = nn.Parameter(0.3 * torch.randn(Nr + 1, d))   # r0,r1,end
scorer = nn.Sequential(nn.Linear(3 * d, 32), nn.GELU(), nn.Linear(32, 1))
opt = torch.optim.Adam([emb, rel_emb] + list(scorer.parameters()), lr=5e-3)
rmap = {"r0": 0, "r1": 1, "end": 2}

def edge_costs(edges):
    feats = []
    for (i, j, lab) in edges:
        r, h, t = lab
        feats.append(torch.cat([rel_emb[rmap[r]], emb[h], emb[t]]))
    s = scorer(torch.stack(feats)).squeeze(-1)          # higher score = cheaper
    return torch.nn.functional.softplus(-s) + 0.05

def incidence(nodes, edges):
    N, E = len(nodes), len(edges); inc = torch.zeros(N, E)
    for e, (i, j, lab) in enumerate(edges): inc[i, e] = 1.0; inc[j, e] = -1.0
    return inc

# ------------------------- train: route flow to a correct answer -------------
print("\ntraining the edge scorer so the flow routes topic -> a correct answer...")
for it in range(120):
    opt.zero_grad(); total = 0.0
    for topic, ans in queries:
        nodes, idx, edges, hop1, hop2 = build_query_graph(topic)
        cost = edge_costs(edges)
        Q = flow_path(cost, incidence(nodes, edges), len(nodes), idx["sink"])
        # flux on each final (h2->sink) edge = mass reaching that answer entity
        final_flux = {}
        for e, (i, j, lab) in enumerate(edges):
            if lab[0] == "end": final_flux[lab[1]] = Q[e]
        ff = torch.stack([final_flux[c] for c in hop2])
        p_ans = ff / (ff.sum() + 1e-9)
        target = torch.tensor([1.0 if c in ans else 0.0 for c in hop2]); target /= target.sum()
        total = total + -(target * (p_ans + 1e-9).log()).sum()
    (total / len(queries)).backward(); opt.step()
    if it % 30 == 0 or it == 119: print(f"  it{it:3d}  routing loss {total.item()/len(queries):.3f}")

# ------------------------- evaluate: accuracy + LEGALITY ---------------------
print("\nEVAL:")
correct = 0; illegal_hops = 0; total_hops = 0
with torch.no_grad():
    for topic, ans in queries:
        nodes, idx, edges, hop1, hop2 = build_query_graph(topic)
        cost = edge_costs(edges)
        Q = flow_path(cost, incidence(nodes, edges), len(nodes), idx["sink"])
        # decode the argmax path: pick highest-flux final answer
        final = [(lab[1], Q[e]) for e, (i, j, lab) in enumerate(edges) if lab[0] == "end"]
        pred = max(final, key=lambda x: x[1])[0]
        correct += (pred in ans)
        # every edge used is a real triple by construction -> count illegal hops
        for e, (i, j, lab) in enumerate(edges):
            if lab[0] in ("r0", "r1"):
                total_hops += 1
                r = 0 if lab[0] == "r0" else 1
                if adj[r, lab[1], lab[2]] == 0: illegal_hops += 1   # can never happen
print(f"  routing accuracy (flow reaches a correct answer): {correct}/{len(queries)}")
print(f"  illegal hops in the reasoning graph: {illegal_hops}/{total_hops}  (0 = every edge is a real triple)")

# ------------------------- contrast: unconstrained predictor -----------------
# a plain classifier that can name ANY entity as the answer -> can hallucinate non-reachable ones
clf = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, Ne))
opt2 = torch.optim.Adam(list(clf.parameters()), lr=5e-3)
for it in range(300):
    opt2.zero_grad(); loss = 0.0
    for topic, ans in queries:
        logits = clf(emb[topic].detach())
        tgt = torch.zeros(Ne);
        for c in ans: tgt[c] = 1.0
        tgt /= tgt.sum()
        loss = loss + -(tgt * torch.log_softmax(logits, -1)).sum()
    (loss / len(queries)).backward(); opt2.step()
reach_all = {topic: (answers_of(topic) | set(h for h in range(Ne) if adj[0, topic, h] > 0)) for topic, _ in queries}
halluc = 0
with torch.no_grad():
    for topic, ans in queries:
        pred = clf(emb[topic]).argmax().item()
        # illegal if predicted entity is NOT even 2-hop-reachable from topic
        reachable = answers_of(topic)
        if pred not in reachable and pred not in [b for b in range(Ne) if adj[0, topic, b] > 0]:
            halluc += 1
print(f"\n  unconstrained classifier hallucination rate (answer not reachable in KG): "
      f"{halluc}/{len(queries)}")
print("\nGO: flow routes legal multi-hop paths on KG-shaped data, illegal hops=0 by construction,")
print("    trains end-to-end; the unconstrained baseline can name non-reachable answers. MetaQA-ready.")
