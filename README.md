# Confluence

**Unifying retrieval and computation as a single conserved-flow path.**

Confluence routes one differentiable, min-cost conserved flow through a single graph
whose nodes are **both** knowledge-base entries (data) **and** expert modules
(computation). A query is answered by one connected path that *retrieves and computes
at the same time*, and illegal reasoning steps are impossible **by construction** —
there is no edge where there is no triple.

## The idea

Standard pipelines split *retrieval* (search a memory) and *computation* (run a model)
into separate stages glued together by averaging the top-k results. Confluence collapses
them into one operation:

1. Build a per-query graph: retrieved entities become **data nodes**, relations/experts
   become **compute nodes**, and **legal transitions are the graph's edges**.
2. Set input-conditional edge costs; inject a conserved flow from the query.
3. A Physarum-style (slime-mold) reinforcement dynamic relaxes to the cheapest **legal**
   path — no learned gate.
4. Walk the path: at a data node, inject content; at a compute node, apply the expert.

**Two guarantees fall out of the mechanism:**
- **Legality by topology** — the selected path can only traverse real edges, so
  constraint violations are impossible (not merely penalized).
- **Conservation coupling** — flow-in = flow-out couples consecutive steps.

## Current results

| Setting | Metric | Result |
|---|---|---|
| Synthetic constrained routing (4 seeds) | held-out MSE / illegal-rate | best on both at every noise level |
| Density sweep (3 seeds) | advantage vs constraint strength | scales monotonically; ~tie when unconstrained (causal) |
| Mixed data+compute graph (3 seeds) | illegal-hop rate | ~0 by construction vs ~0.17 baselines |
| MetaQA 1-hop | hits@1 | 0.769, 0 hallucination |
| MetaQA 2-hop (tractable subset) | hits@1 | **0.948**, 0 hallucination |

All numbers are reproducible from the scripts below. See `PAPER_NOTES.md` for the full
record, exact numbers, honest caveats, and the novelty positioning.

## Repository

```
physarum_spike.py            flow mechanism (path emerges, optimal, differentiable)
diag_oracle.py               expert-learnability diagnostic (MLP required, not Linear)
mycelial_arch.py             backbone architecture + oracle ceiling
mycelial_final.py            constrained-routing win vs Routing-Free MoE (multi-seed)
density_sweep.py             causal proof: advantage scales with constraint strength
confluence_spike.py          go/no-go: one flow through data + compute nodes
confluence_kg_mock.py        KG pipeline on a mock knowledge graph
metaqa_load.py               MetaQA loader + KG / neighborhood stats
metaqa_confluence.py         MetaQA 1-hop
metaqa_confluence_2hop.py    MetaQA 2-hop
metaqa_confluence_2hop_v2.py MetaQA 2-hop (GRU + per-hop heads + degree-norm decode)
make_fig_*.py, eqns.py       figures
data/metaqa/                 MetaQA text (KB + 1/2/3-hop QA); not tracked in git
```

## Data

MetaQA (movie knowledge-graph QA): 43,234 entities, 9 relations, 134,741 triples, plus
1/2/3-hop question sets. Text mirror of the original dataset. `data/` is gitignored.

## Status

Research in progress. Working: mechanism, backbone, real-data pipeline, 2-hop at 0.948
with a hard legality guarantee. Next: fair baselines (EmbedKGQA / PathRAG / GFlowNet-KG),
implicit-differentiation and sparse solve for the heavy-degree tail and 3-hop, and
external validity on WebQSP/CWQ.

## Requirements

Python 3, PyTorch, NumPy, Matplotlib.
