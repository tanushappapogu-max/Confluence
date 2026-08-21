# Confluence — Paper Notes (private, gitignored)
*Master reference for writing the ICLR 2027 paper. Last updated 2026-08-18.*
*Deadlines: abstract Sept 18, paper Sept 25, 2026 (AOE).*

---

## 0. Identity
**Title candidates:** "Confluence: Unifying Retrieval and Computation as a Conserved-Flow Path" / "Legality by Topology".
**One-liner:** route ONE differentiable min-cost conserved flow through a single graph whose nodes are BOTH knowledge-base entries (data) AND expert modules (computation) — a query is answered by one legal path that retrieves and computes at once. Illegal reasoning steps are impossible by construction (no edge where there is no triple).

**Backbone → body:** Mycelial Routing (flow through expert nodes only) is the proven mechanism / special case; Confluence adds data nodes so retrieval + computation become one flow.

---

## 1. The novelty wedge (memorize for rebuttals)
Every neighbor is occupied; the CONJUNCTION is empty.

| axis | incumbents | Confluence (ours) |
|---|---|---|
| routes through | weights (PathMoE/CoE) OR data (PathRAG/GFlowNet/D-RAG) | **both, one graph** |
| constraint | soft/statistical (PathMoE, MoG) or none (GFlowNet, D-RAG) | **hard, by topology, input-conditional** |
| flow objective | reward-proportional SAMPLING (GFlowNet) | **min-cost optimal path** |
| learned? | PathRAG flow is non-parametric | **differentiable, jointly trained** |

**Scooped outcomes to concede honestly:** PathMoE (arXiv 2603.18297, Apple) already showed path-constrained MoE kills the load-balance loss + robustness + emergent paths, at 16B scale, via router-param-sharing. Chain-of-Experts (2506.18945) owns "sequential composition beats averaging." GFlowNet-KG owns flow-conservation-over-DAG for retrieval. → We do NOT claim those outcomes. We claim the mechanism (flow-as-router) + the HARD input-conditional legality guarantee + the retrieval/computation UNIFICATION.

**The killer demonstrated fact (toy):** PathMoE-style param-sharing gets the WORST legality under noise (illegal-hop 0.512) because tying router params can't condition on the transition. Topology gives 0. → "soft/shared-param constraints don't just fail to guarantee legality; they can make it worse."

---

## 2. Results (exact numbers — all reproducible)

### Backbone (Mycelial, synthetic)
- Flow mechanism spike: single path emerges + is optimal; 14 redundant edges → 2 survive; gradient through fixed point re-routes the path. (`physarum_spike.py`)
- Oracle (MLP experts) → **train 0.000, held-out 0.000** perfect systematic generalization. **Linear experts FAIL (train 0.80)** — must use MLP experts. (`diag_oracle.py`)
- Constrained-path win vs Routing-Free MoE, 4 seeds: mycelial best MSE + lowest illegal-rate at every noise. (`mycelial_final.py`)
  - MSE@α=3/2/1: myc 0.054/0.392/1.246 vs free 0.158/0.583/1.319 vs router 0.083/0.506/1.419
  - illegal@1.0: myc 0.103 vs free 0.262 vs router 0.295
- **Density sweep (causal proof), 3 seeds:** advantage scales monotonically with constraint strength; gap free−myc = 0.397 (dense-rules) → 0.060 (no rules). At density 1.0 the plain router slightly beats myc (0.492 vs 0.513) → HONEST boundary: wins iff structure exists. (`density_sweep.py`, fig `mycelial_density.png`)

### Confluence go/no-go (synthetic mixed data+compute), 3 seeds
- Mixed graph: 4 data-nodes (retrieve=inject value) + 4 expert-nodes (compute=apply MLP), path=[fact,expert,fact,expert].
- oracle→0.002 (learnable). Confluence best/tied MSE at every noise + illegal-hop 0.00/0.006/0.058 vs baselines ~0.17–0.19.
- PathMoE-lite (shared router across positions): illegal-hop 0.512 @noisy (WORST). (`confluence_spike.py`, fig `confluence_result.png`)

### Real data (MetaQA)
- KG: 43,234 entities, 9 relations, 134,741 triples. 1/2/3-hop QA (96k/119k/114k train).
- Mock-KG pipeline: 0/24 illegal hops (legality survives on KG data). (`confluence_kg_mock.py`)
- **1-hop hits@1 = 0.769** on 2k test, 0 hallucination by construction. (`metaqa_confluence.py`)
- **2-hop hits@1 = 0.833** on 1k tractable (≤80-node) test, loss 1.50→0.79, 0 hallucination. (`metaqa_confluence_2hop.py`)
- **2-hop v2 (GRU + per-hop query heads, 8k train, 5 ep): hits@1 = 0.864** — PLATEAUS ~0.85–0.86. (`metaqa_confluence_2hop_v2.py`)

### KEY FINDING (2026-08-18): flow CONNECTIVITY BIAS diagnosed AND fixed → 0.864 → 0.948
Encoder wasn't the bottleneck (GRU+per-hop heads only 0.833→0.864). Diagnosis: a conserved flow pools more flux at entities reachable by MORE legal paths → argmax biased to high-degree entities. FIX: normalize answer mass by in-degree. Result (eval-time norm, no retrain): 2-hop hits@1 **0.948** (was 0.864). This is a paper-worthy MECHANISM insight, not a hack — quantify degree bias + show the fix.
- Retrain with the degree-normalized objective → likely higher.
- 0.948 WITH 0-hallucination guarantee, on tractable subset, no entity-linking, no heavy tail. Approaching ~0.99 SOTA.
**Reframing still holds:** contribution = legality guarantee + unification; but now accuracy is competitive-enough that "0.95 + hard guarantee" is a strong standalone claim, not an excuse.
Still-open accuracy levers: retrain with degree-norm; hard relation-typed masking per hop; heavy-tail via sparse solve.

---

## 3. HONEST caveats (put in the paper before a reviewer does)
- MetaQA numbers are FIRST-PASS: 1-hop SOTA ~97%, 2-hop ~99%. Ours use a bare bag-of-words encoder, ~3 epochs, ≤6k of 96k+ Q. Not competitive yet.
- 2-hop 0.833 is on the TRACTABLE SUBSET (≤80-node graphs) → heavy-degree tail excluded = easier-question bias. Full coverage needs the sparse solve.
- The coded similarity baselines so far are BROKEN/undertrained (1-hop baseline 0.000/100% halluc). NOT fair comparisons — do not cite as wins.
- Flow solve is ~16× slower than a gate; per-query dense Laplacian O(N³). 3-hop median neighborhood 12k → intractable without sparsification/implicit-diff.
- γ (throughput sharpening) is a tuned hyperparameter; report its sensitivity.
- "Legality by construction" claim depends on the KG being the legality source — clean on KGQA, murkier where no native graph exists.

---

## 4. Baselines to build (the comparison that decides credibility)
- Fair KGQA: **EmbedKGQA** (KG-embedding QA), and a trained retrieval-classifier (not the broken one).
- Flow/graph rivals: **PathRAG** (non-parametric flow pruning), **GFlowNet-KG** (flow SAMPLER), **D-RAG** (differentiable subgraph SET), **MoG** (heuristic topology router).
- The wedge metric across all: **illegal-hop / hallucination rate** (ours 0 by construction) + accuracy at matched setup.

---

## 5. Reproduction (scripts, currently in session scratchpad → migrate to this folder)
`physarum_spike.py`, `diag_oracle.py`, `mycelial_arch.py`, `mycelial_final.py`, `density_sweep.py`,
`confluence_spike.py`, `confluence_kg_mock.py`, `metaqa_load.py`, `metaqa_confluence.py`, `metaqa_confluence_2hop.py`.
Figures: `mycelial_result.png`, `mycelial_density.png`, `confluence_result.png`, `mycelial_equations.png`.
Data: MetaQA text from HF mirror `camazlucas/MetaQA` (official Drive is resourcekey-gated → gdown 401). kb.txt + 1/2/3-hop vanilla qa.

## 6. Design decisions & gotchas (reproducibility + paper method)
- Experts MUST be MLP, not Linear (composed-linear optimization fails).
- Flow = Tero–Nakagaki Physarum: solve ℒ(w)p=b (weighted Laplacian), reinforce D←D+Δt(|Q|−D), fixed point D*=|Q*|.
- Selection = node throughput ^γ, normalized per position.
- Legality baked into TOPOLOGY: edge exists iff triple/transition legal → illegal flux impossible.
- Two ways to backprop the fixed point: unroll (current, expensive) vs implicit diff (DEQ-style, needed for scale).
- MetaQA per-query graph size = k-hop neighborhood: 1-hop med 7 / 2-hop med 18 (p90 1538) / 3-hop med 12k.

## 7. Open questions / limitations section fodder
- Scale: implicit diff + graph sparsification for 2-hop tail and 3-hop.
- Where does the constraint graph come from when there's no native KG? (answered for KGQA; open in general).
- Interpretability claim (readable reasoning paths) — measure, don't assert.
- Single domain (movies) — add WebQSP/CWQ for external validity (Week 3-4).

## 8. Timeline
Wk1 (done): mechanism proven, Confluence go/no-go GO, MetaQA in, 1-hop+2-hop first results.
Wk2: real encoder + full training (push accuracy); fair baselines; implicit diff.
Wk3: WebQSP/CWQ; full 2/3-hop coverage; ablations.
Wk4: analysis, figures, robustness, freeze ~Sept 14.
Wk5: write, abstract Sept 18, submit Sept 25.
