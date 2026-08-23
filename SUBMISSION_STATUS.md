# Confluence — Submission Readiness (ICLR)

*Live status of what is done, what is verified, and the exact remaining path to submit.*
*Companion to `passover.md` (full context) and `PAPER_NOTES.md` (results record).*

---

## 1. The claim (what the paper argues)

Route ONE differentiable, min-cost **conserved flow** through a single graph whose nodes are
**both** KB facts (data) and expert modules (computation), so a query is answered by one legal
path that retrieves and computes at once — and **illegal reasoning steps are impossible by
construction** (no edge where there is no triple).

Two guarantees fall out of the mechanism and are the contribution:
- **Legality by topology** — violations are impossible, not penalized (`e ∉ E ⇒ Q_e = 0`).
- **Conservation coupling** — flow-in = flow-out couples consecutive steps.

The novelty is the **conjunction** (flow-as-router + hard input-conditional legality +
retrieval/computation unification); each neighbor owns only one ingredient. Do **not** claim the
scooped outcomes (PathMoE path-constraint benefits; Chain-of-Experts sequential composition;
GFlowNet flow-over-DAG). Cite them; claim the mechanism + hard guarantee + unification.

---

## 2. Status ledger

### Verified in-repo (reproducible, no external data)
| Result | Where | Status |
|---|---|---|
| Flow mechanism: optimal path emerges, sparsifies, differentiable/trainable | `physarum_spike.py` | ✅ reproduced (short D≈1.0, long→0.001; 14→2 edges; grad re-routes 0.04→1.0) |
| Constrained-routing win + PathMoE-lite worst-legality wedge | `confluence_spike.py` | ✅ reproduced (PathMoE-lite illegal 0.50 vs mycelial 0.10, lowest) |
| **Implicit differentiation** through the flow fixed point | `implicit_diff.py` | ✅ **verified** vs finite-difference: cos 1.0, rel-err ~1e-8; memory flat in solver depth |
| **Matrix-free** implicit gradient (VJP-only, no E×E Jacobian) | `implicit_diff.py` | ✅ matches exact + FD to ~1e-8; solves N=1000/E~3k where dense is infeasible |
| End-to-end implicit-diff training at N>80, 0 illegal | `implicit_train_demo.py` | ✅ held-out hits@1 rises to ~9× chance, illegal-hop 0 throughout (existence result) |
| Implicit-diff wired into MetaQA + CAP lifted 80→400 | `metaqa_confluence_2hop_implicit.py` | ✅ gradient self-check vs FD (rel-err ~1e-3) |
| Matrix-free CG forward solve (heavy-tail eval) | `metaqa_confluence_2hop_implicit.py` | ✅ matches dense (~1e-3); N=2403/E~4.8k in ~0.4s |

### Data-gated (need MetaQA on a machine that can reach the dataset host)
| Item | Command | Why it matters |
|---|---|---|
| No-flow ablation | `python3 metaqa_confluence_2hop_ablation.py` | **Decides the paper**: does the flow buy accuracy, or only the guarantee? |
| Degree-norm retrain | `TRAIN_DEGNORM=1 python3 metaqa_confluence_2hop_v2.py` | Should push past 0.948; removes the eval-time-only asterisk |
| Implicit-diff + lifted-cap run | `CAP=400 python3 metaqa_confluence_2hop_implicit.py` | Trains beyond the ≤80 subset with O(E)-memory backward |
| Heavy-tail eval | eval path auto-uses matrix-free CG for N>500 | Decode the full 2-hop distribution → removes easier-subset bias |
| EmbedKGQA baseline | (to build) | Credible external number + hallucination rate |

> Data: `data/metaqa/{kb/kb.txt, 2-hop/qa_{train,test}.txt}` from HF mirror `camazlucas/MetaQA`
> (official Google Drive is resourcekey-gated). `data/` is gitignored.

---

## 3. Reviewer-rebuttal map (the attacks and the answer to each)

1. **"0.948 is an uncompared number."**
   → No-flow ablation (`metaqa_confluence_2hop_ablation.py`): identical encoder/graph/legality,
   flow removed. Isolates the flow's contribution. Run it; report flow vs no-flow at matched setup.

2. **"It's on the ≤80-node tractable subset — easier-question bias."**
   → CAP lifted to 400 (training) and the **matrix-free CG forward** decodes the full tail at eval
   (N=2403 in ~0.4s). Report accuracy on the full 2-hop distribution, not the subset.

3. **"The flow is 16× a gate and O(N³) — doesn't scale."**
   → Implicit differentiation (verified, memory-flat) removes the unrolled-backward cost;
   matrix-free CG removes the dense forward. Both validated in `implicit_diff.py` /
   `metaqa_confluence_2hop_implicit.py`. This is done code, not a promise.

4. **"You were scooped (PathMoE / CoE / GFlowNet)."**
   → Concede those outcomes explicitly; claim the conjunction + the hard input-conditional
   legality guarantee. Killer fact: PathMoE-lite param-sharing gets the *worst* legality under
   noise (illegal-hop ≈0.5) while topology gives 0.

5. **"γ / RHO / hyperparameters are tuned."**
   → Report γ-sensitivity; note the RHO conductance floor is a *principled* regularizer that makes
   the fixed point non-degenerate (documented finding: the hard fixed point has a vanishing,
   ill-conditioned gradient — the trainable signal lives in the regularized regime).

6. **"Legality-by-construction depends on having a KG."**
   → True and stated as a limitation; clean on KGQA (the KG *is* the legality source per query),
   open in general. Add WebQSP/CWQ for external validity.

---

## 4. Remaining work to submit (ranked)

1. **[you, data]** Run items in §2 data-gated table; send numbers back.
2. **[you, data]** Build EmbedKGQA baseline + a trained retrieval classifier (fair, not the broken one).
3. **[either]** Multi-seed + variance bars on the MetaQA runs; freeze results.
4. **[either]** WebQSP/CWQ for external validity (single-domain → multi-domain).
5. **[writeup]** Draft around "Legality by Topology"; abstract, then paper.

Contingency: if scaling/baselines stall, the fallback paper is "Legality by Topology" on the
synthetic constrained-routing + Confluence-synthetic + MetaQA-1/2-hop results, now strengthened
by the verified implicit-diff + matrix-free scaling infrastructure.

---

## 5. One-command reproduction of what's verified here (no data)

```bash
python3 physarum_spike.py                                  # mechanism
python3 confluence_spike.py                                # wedge (PathMoE-lite worst legality)
python3 implicit_diff.py                                   # implicit-diff correctness + scaling
python3 implicit_train_demo.py                             # end-to-end training at N>80, 0 illegal
python3 metaqa_confluence_2hop_implicit.py --selfcheck     # MetaQA implicit-diff gradient vs FD
python3 metaqa_confluence_2hop_implicit.py --scalecheck    # matrix-free forward vs dense + large-N
```
Requirements: Python 3, PyTorch, NumPy (Matplotlib for figures).
