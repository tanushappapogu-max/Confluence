# PASSOVER — Confluence (complete handoff)

> Read this first in any new session. It is the full context dump for the project:
> the idea, the math, what worked, what didn't, the competitive landscape, the plan,
> what's done, and the exact next steps. Deliberately over-complete.

---

## 0. TL;DR / orientation
- **Project name:** Confluence.
- **Local repo:** `~/Documents/Confluence/` (git, branch `main`).
- **Remote:** private GitHub `github.com/tanushappapogu-max/Confluence` (origin, in sync).
- **Target:** ICLR 2027 — **abstract Sept 18, 2026; paper Sept 25, 2026 (AOE)**. ~5 weeks of runway from mid-Aug.
- **Author is solo** (no advisor/lab). ICLR needs no endorsement to submit.
- **Commit hygiene:** commits are authored solely by the human; do NOT add any AI/co-author trailer to commit messages. Keep the repo free of AI-attribution.
- **One-liner:** *Route one differentiable, min-cost conserved flow through a single graph whose nodes are BOTH knowledge-base entries (data) AND expert modules (computation), so a query is answered by one legal path that retrieves and computes at once — and illegal reasoning steps are impossible by construction.*
- **Status in one line:** mechanism proven; real-data pipeline works; MetaQA 2-hop at **hits@1 0.948 with a hard 0-hallucination legality guarantee**; next is fair baselines + scaling.

---

## 1. Lineage (how we got here — context)
This is the current head of a longer research thread. Earlier stages (shelved/superseded, but the DNA carries):
1. **DCN (Dynamic Coalition Network)** — the original "not a router" idea: experts form a *coalition* by resonance, validated by a lesion test. Shelved.
2. **Block-gated FFN frontier** — 3-seed result beat dense+MoE at k=8/k=4 but lost at k=2.
3. **Sparse-expert-control** — key diagnosis: *the penalty is averaging, not factorization*. i.e. MoE's convex-weighted-sum readout is bounded by its best expert → an "averaging ceiling."
4. **Mycelial Routing** — gateless conservation-flow *path*-MoE. Legality by topology. This is Confluence's proven backbone (flow through EXPERT nodes only).
5. **Confluence (current)** — add DATA nodes to the graph so retrieval + computation are one flow. RAG and MoE become the same operation.

The whole thread shares: gateless, self-organizing, "not a router", escape the averaging ceiling.

---

## 2. The idea (plain, then technical)

### Plain
Today, AI answers hard questions in two glued-together stages: **look things up** (search a memory / vector DB) and **think** (run computation). Standard RAG grabs the top-k most-similar facts and **averages** them (lossy; the "averaging ceiling"), then feeds a fixed reasoning stack — and nothing stops illegal logical jumps.

Confluence makes look-up and thinking the **same act**. Put every fact AND every compute-expert as **dots on a graph**, with **edges only between things allowed to connect**. A query **pours a conserved flow** in; it finds the cheapest **legal** path to the answer (like slime mold / water finding a route). The path passes through some fact-dots (retrieve) and some compute-dots (think) in one connected sweep. No boss/gate picks it — it self-organizes. Illegal jumps are **impossible** because there's no edge there.

### Technical (the mechanism, per step)
Graph with signed incidence `B` (N×E). Each edge has conductance `D_e`, cost `L_e`.
1. **Input → edge costs.** Encoder produces per-edge logits `z`; cost `L_e = softplus(−z) + ε` (relevant edges cheap).
2. **Conserved-flow dynamic (Tero–Nakagaki Physarum):**
   - weight `w_e = D_e / L_e`; flux `Q_e = w_e (Bᵀp)_e` (Ohm).
   - conservation (Kirchhoff): `ℒ(w) p = b`, where `ℒ(w) = B diag(w) Bᵀ` is the weighted Laplacian, `b` = +1 at source, −1 at sink. Solve for pressures `p` (ground the sink).
   - reinforce: `D_e ← D_e + Δt(|Q_e| − D_e)`. Iterate to fixed point `D*_e = |Q*_e(D*)|` ("a tube's thickness equals the flow it carries" → sparse emergent path).
3. **Flow → selection.** node throughput `t_v = ½ Σ_e |B_{v,e}| |Q_e|`; per position `a_{k,m} = t^γ / Σ t^γ` (γ sharpens toward one-hot).
4. **Sequential composition.** `h_0 = x`; at a compute node apply expert `g`, at a data node inject value; `ŷ = h_K` (function composition, NOT a weighted average → dodges the averaging ceiling).
5. **Learn.** loss backprops through the whole solve into the cost encoder + experts. Two ways to differentiate the fixed point: **unroll** (current, ~16× cost) or **implicit differentiation** (DEQ-style, needed for scale).

**Two guarantees fall out:**
- **Legality by topology:** `e ∉ E ⇒ Q_e = 0`. Illegal steps impossible, not penalized.
- **Conservation coupling:** `Σ_in Q = Σ_out Q` couples consecutive steps (a shared flow budget), unlike parallel self-activation.

It IS a bilevel / deep-equilibrium (DEQ) / differentiable-optimization-layer setup: inner loop = a physics/LP solve finding the min-cost legal path (no learned params); outer loop = SGD on task loss shaping the costs + experts. NOT "double gradient descent"; the inner objective ≠ the task loss.

---

## 3. Novelty / competitive landscape (CRUCIAL — we were partly scooped)
Individual ingredients are all occupied; the **conjunction** is the wedge.

| paper | what it owns | why it's NOT us |
|---|---|---|
| **PathMoE / Path-Constrained MoE** (arXiv 2603.18297, Apple) | path-constrained MoE, kills load-balance loss, robustness, emergent paths, at 16B | constraint is **soft/statistical** (shared router params) + **static** + routes **weights only** — no hard guarantee, no data |
| **Chain-of-Experts** (2506.18945) | sequential expert composition beats parallel averaging | learned per-iteration router; weights only; no legality |
| **Routing-Free MoE** (2604.00801) | gateless MoE (threshold self-activation) | parallel/independent; no path; no coupling |
| **GFlowNet-for-KG-retrieval** | flow-conservation over a DAG to sample retrieval paths | a **sampler** (diversity), not min-cost; no hard legality; data only |
| **PathRAG** | flow-based path pruning over graph-RAG | **non-parametric** (their own future work = a neural scorer); retrieval only |
| **D-RAG** (EMNLP 2025) | differentiable KG subgraph retrieval (Gumbel-softmax) | picks a **set**, not a connected legal path; data only |
| **MoG** (2605.31010) | MoE over knowledge graphs | **heuristic** topology router; soft; experts=graph regions |
| **Physarum diff-LP layer** (AAAI 2021 / 2004.14539) | Physarum as a differentiable LP solver | solves LPs, not expert routing |

**What survives as OUR novelty (the empty cell):**
1. a conserved-flow / Physarum dynamic used as the **router itself** (unoccupied — searches confirmed "Physarum + MoE routing" is unexplored);
2. **hard, input-conditional legality by TOPOLOGY** (violations impossible, not merely rare) vs everyone's soft/statistical constraint;
3. **min-cost optimal path** (vs GFlowNet sampling), **jointly trained** (vs PathRAG non-parametric);
4. **unification of retrieval + computation** into one flow through data+weight nodes — nobody does this; it sits between MoE and GraphRAG.

**Headline framing for the paper:** "Legality by topology" — routing that makes constraint violations *impossible*, not rare. Do NOT claim the scooped outcomes (path-constraint benefits = PathMoE; sequential-composition = CoE; flow-over-DAG = GFlowNet). Cite them; claim the mechanism + hard guarantee + unification.

**The single sharpest demonstrated fact:** a PathMoE-style shared-router baseline gets the WORST legality under noise (illegal-hop rate 0.512) because tying router params can't condition on the specific transition. → "soft/param-sharing constraints don't just fail to guarantee legality; they can make it worse."

---

## 4. WHAT HAS WORKED (all results, exact, reproducible)

### Synthetic (backbone = Mycelial)
- **Mechanism spike** (`physarum_spike.py`): a single path emerges + is the optimal one (short route D≈1.0, long withers to 0.001); 14 redundant edges → 2 survive (emergent sparsity); gradient through the fixed point RE-ROUTES the path (0.04→1.00). → flow is real, stable, differentiable, trainable.
- **Oracle** (`diag_oracle.py`): with **MLP experts**, oracle → **train 0.000, held-out 0.000** (perfect systematic generalization to unseen program orderings). Composition substrate is sound.
- **Constrained win vs Routing-Free MoE**, 4 seeds (`mycelial_final.py`): best MSE + lowest illegal-rate at every noise. MSE@α=3/2/1: myc 0.054/0.392/1.246 vs free 0.158/0.583/1.319 vs router 0.083/0.506/1.419. illegal@1.0: myc 0.103 vs free 0.262 vs router 0.295.
- **Density sweep** (causal proof), 3 seeds (`density_sweep.py`, fig `mycelial_density.png`): advantage scales monotonically with constraint strength; gap(free−myc) 0.397 (dense rules) → 0.060 (no rules). At density 1.0 the plain router slightly beats us (0.492 vs 0.513) → HONEST boundary: **wins iff structure exists**.

### Confluence proper (synthetic mixed data+compute), 3 seeds (`confluence_spike.py`, fig `confluence_result.png`)
- Mixed graph: 4 data-nodes (retrieve=inject value) + 4 expert-nodes (compute=apply MLP); path=[fact,expert,fact,expert].
- oracle→0.002 (learnable). Confluence best/tied MSE at every noise + illegal-hop 0.00/0.006/0.058 vs baselines ~0.17–0.19.
- **PathMoE-lite** (shared router across positions): illegal-hop **0.512** @noisy (worst) — the wedge.

### Real data — MetaQA
- KG: 43,234 entities, 9 relations, 134,741 triples. Relations: directed_by, written_by, starred_actors, release_year, in_language, has_tags, has_genre, has_imdb_votes, has_imdb_rating.
- **Mock KG pipeline** (`confluence_kg_mock.py`): 0/24 illegal hops — legality survives on KG-shaped data.
- **1-hop** (`metaqa_confluence.py`): hits@1 **0.769** on 2k test, 0 hallucination by construction.
- **2-hop** (`metaqa_confluence_2hop.py`): hits@1 **0.833** on 1k tractable (≤80-node) test, loss 1.50→0.79.
- **2-hop v2** (`metaqa_confluence_2hop_v2.py`): GRU + per-hop query heads → 0.864. Then **KEY FINDING** below → **0.948**.

### THE degree-bias finding (paper-worthy)
Better encoder only moved 0.833→0.864 → encoder was NOT the bottleneck. Diagnosis: a conserved flow pools MORE flux at entities reachable by MORE legal paths → argmax biased to high-degree entities. **Fix: normalize answer mass by in-degree.** Result (eval-time norm, no retrain): **2-hop hits@1 0.948**. This is a mechanism insight to quantify + present, not a hack.

---

## 5. WHAT HAS NOT WORKED / gotchas (be honest in the paper)
- **Linear experts FAIL** (oracle train 0.80) — composed-linear-map optimization is bad. MUST use MLP experts. (This wasted a first run — the initial arch got ~chance until this was found.)
- **First arch task was a NULL:** on an *unconstrained/factorized* task Confluence only TIES the incumbent → coupling needs structure to exploit. This motivated the constrained-task design and is itself an insight.
- **γ (throughput sharpening) is tuned** — without it Confluence only won in the noisy regime. Report γ-sensitivity.
- **Coded baselines so far are BROKEN/undertrained** (e.g. 1-hop similarity baseline: hits@1 0.000, 100% hallucination). These are NOT fair comparisons — do NOT cite as wins. Real baselines must be built.
- **2-hop 0.948 is on the TRACTABLE SUBSET** (≤80-node graphs) → heavy-degree tail excluded = easier-question bias. Full coverage needs the sparse solve.
- **Flow solve is ~16× slower than a gate**; per-query dense Laplacian is O(N³). 3-hop median neighborhood is ~12k nodes → intractable without sparsification/implicit-diff.
- MetaQA numbers are still first-pass (1-hop SOTA ~97%, 2-hop ~99%); no entity-linking pipeline yet.

---

## 6. Data tractability map (dictates the scaling strategy)
Per-query flow-graph size = the k-hop neighborhood of the topic entity:
- **1-hop:** median 7, p90 14, max 28 → trivially tractable now.
- **2-hop:** median 18, **p90 1538, max 7684** → bulk fine, heavy-degree tail (prolific actors) needs pruning.
- **3-hop:** median **12,114** → most queries touch a huge chunk → REQUIRES sparse/implicit-diff solve.

---

## 7. Phases / plan (5-week, ICLR Sept 25)
- **Wk1 (DONE):** mechanism proven; Confluence go/no-go GO; MetaQA downloaded; 1-hop + 2-hop first results; degree-bias fix → 0.948.
- **Wk2:** retrain with degree-norm objective; **fair baselines** (EmbedKGQA + a proper no-flow ablation); **implicit differentiation** (DEQ-style) to kill the 16× cost.
- **Wk3:** heavy-tail via sparse solve (unbias 2-hop, enable 3-hop); WebQSP/CWQ for external validity; ablations (mixed vs data-only vs compute-only; hard-topology vs soft-penalty; min-cost vs sampling; joint vs frozen).
- **Wk4:** multi-seed, variance bars, figures, robustness; freeze results ~Sept 14.
- **Wk5:** write; register abstract Sept 18; submit Sept 25.

**Contingency:** if scaling stalls, fall back to the "Legality by Topology" paper on the constrained-routing + Confluence-synthetic + MetaQA-1/2-hop results (workshop-to-borderline main).

---

## 8. NEXT STEPS (ranked, do these next)
1. **Retrain 2-hop with the in-degree-normalized objective** (currently it's only eval-time; aligning training should push >0.948). Small change to `metaqa_confluence_2hop_v2.py`.
2. **Build a FAIR baseline** — this is what makes any number a *comparison*:
   - a **no-flow ablation** (same GRU encoder + entity/relation embeddings, but direct softmax over candidates, no conservation solve) → isolates whether the flow helps accuracy or ONLY provides the guarantee;
   - **EmbedKGQA-style** trained baseline for a credible external number + its hallucination rate.
3. **Implicit differentiation** for the flow solve (DEQ / implicit function theorem through the Laplacian fixed point) → removes the 16× cost, unlocks larger graphs.
4. **Sparse solve + candidate pruning** → cover the 2-hop heavy tail (remove subset bias) and enable 3-hop.
5. **Reproduce rival baselines** (PathRAG, GFlowNet-KG, D-RAG, MoG) on the same MetaQA setup with the illegal-hop / hallucination metric.
6. Figures → paper draft around "Legality by Topology."

---

## 9. Reproduction (files)
```
physarum_spike.py             mechanism go/no-go
diag_oracle.py                MLP-vs-Linear expert diagnostic
mycelial_arch.py              backbone + oracle ceiling + noise sweep
mycelial_final.py             constrained win vs Routing-Free MoE (4 seeds)
density_sweep.py              causal density sweep
confluence_spike.py           mixed data+compute go/no-go + PathMoE-lite
confluence_kg_mock.py         KG pipeline on a mock graph
metaqa_load.py                MetaQA loader + KG/neighborhood stats
metaqa_confluence.py          MetaQA 1-hop
metaqa_confluence_2hop.py     MetaQA 2-hop
metaqa_confluence_2hop_v2.py  2-hop w/ GRU + per-hop heads + degree-norm decode (0.948)
make_fig_*.py, eqns.py        figures
PAPER_NOTES.md                condensed paper-writing reference (overlaps this file)
data/metaqa/                  MetaQA text (gitignored)
```
Run any script with `python3 <script>.py` from the repo root (they resolve `data/metaqa` relative to their own location). Requirements: Python 3, PyTorch, NumPy, Matplotlib.

**Data provenance:** official MetaQA is on a Google Drive folder that is resourcekey-gated (gdown returns 401). We used the clean text mirror HuggingFace `camazlucas/MetaQA` (kb.txt + 1/2/3-hop vanilla QA, no audio). If `data/metaqa/` is missing, re-download those files from that HF dataset into `data/metaqa/{kb/kb.txt, 1-hop, 2-hop, 3-hop}`.

---

## 10. Design decisions & gotchas (reproducibility)
- Experts MUST be MLP, not Linear.
- Flow = Tero–Nakagaki Physarum (see §2).
- Legality is baked into TOPOLOGY: an edge exists iff the triple/transition is legal → illegal flux is impossible.
- Selection = node-throughput^γ, normalized per position/hop.
- For multi-hop questions, use PER-HOP query heads (q1 for hop1 edges, q2 for hop2) — a single question vector can't disambiguate "relation A then relation B".
- Decode answers with **in-degree normalization** to kill connectivity bias.
- Two backprop options through the fixed point: unroll (simple, costly) vs implicit diff (needed for scale).

---

## 11. Open questions / limitations (paper limitations section)
- Where does the constraint graph come from when there is no native KG? (Answered for KGQA — the KG *is* the graph, per query; open in general.)
- Scale: implicit diff + sparsification are unproven at large graph sizes here.
- Single domain (movies) so far; add WebQSP/CWQ.
- Interpretability claim (readable reasoning paths) — must be measured, not asserted.
- γ and other hyperparameters — report sweeps.

---

## 12. Housekeeping / meta
- Repo is **private**; keep it private until after submission (scoop risk).
- **Never push without explicit human say-so; never add an AI/co-author trailer to commits.** Keep the repo and its history free of AI attribution.
- `data/`, `__pycache__/`, `*.pyc`, `.DS_Store` are gitignored.
- A condensed version of this lives in `PAPER_NOTES.md`; this `passover.md` is the fuller session/context dump. Keep both updated as work proceeds.
