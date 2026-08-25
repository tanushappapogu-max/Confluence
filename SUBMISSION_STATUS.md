# Confluence — Submission Readiness (ICLR)

*Live status. Companion to `passover.md` (context) and `PAPER_NOTES.md` (record).*
*Framing (current): an MoE-routing paper — "When does conserved-flow routing help?"*

---

## 1. Thesis
Route ONE differentiable, min-cost **conserved flow** through a per-query graph of expert (and
optionally data) nodes whose edges are the legal transitions. It is a sparse **MoE router** that
selects one legal path of experts. Contribution = a **mechanism + a condition for its use**:
- **Coupling condition:** flow-routing beats an independent top-k gate **iff** reasoning steps are
  coupled (mutually constraining); on marginalizable tasks they tie.
- **Legality by topology:** flux is zero on illegal expert transitions by construction; the
  connected-path decode is legal by construction (safety for deployed MoE).
- **Cheap training at scale:** implicit differentiation + matrix-free solve (verified).

## 2. Results ledger (all real runs this session, on the branch)
| Result | File / figure | Status |
|---|---|---|
| Coupling crossover (flow vs independent, 3 seeds) | `results/density_sweep_flow_vs_independent.txt`, `paper/figs/coupling.png` | ✅ +0.159 (coupled) → −0.019 (unconstrained) |
| MoE-routing win vs top-k gate + Routing-Free MoE (4 seeds) | `results/moe_routing_comparison_4seed.txt` | ✅ MSE 0.392 vs 0.506 / 0.583; illegal 0.10 vs 0.30 / 0.26 |
| **MoE FFN-layer accuracy/compute frontier (3 seeds)** | `results/moe_layer_frontier_3seed.txt`, `paper/figs/moe_frontier.png` | ✅ flow Pareto-dominant: MSE 1.170 @ 1 expert/layer vs gate 1.294, dense 1.278 |
| Legality wedge (PathMoE-lite worst) | `confluence_spike.py` | ✅ shared-router ~0.5 vs 0 |
| MetaQA 2-hop no-flow ablation (real data, full) | `results/metaqa_2hop_noflow_ablation.txt` | ✅ no-flow 0.885/0.944 ≈ flow 0.864/0.948 — the predicted tie |
| Implicit-diff correctness + scaling | `implicit_diff.py`, `paper/figs/implicit_diff.png` | ✅ vs finite-diff ~1e-8, memory-flat, matrix-free to N=1000 |
| MetaQA implicit wiring + matrix-free forward | `metaqa_confluence_2hop_implicit.py` | ✅ gradient self-check ~1e-3; forward matches dense, N=2403 in 0.4s |
| EmbedKGQA baseline (ComplEx) | `metaqa_embedkgqa_baseline.py` | ✅ built + self-check; run on data for external anchor + hallucination |

Data: full MetaQA (43,234 ent / 9 rel / 134,741 triples; 1/2/3-hop QA) staged locally in
`data/metaqa/` (gitignored), sourced from a public GitHub mirror.

## 3. Paper
`paper/main.tex` — retitled "When Does Conserved-Flow Routing Help?", MoE-framed related work,
coupling figure, MoE-routing table, MoE frontier figure, honest MetaQA tie table, scalability
section + figure, limitations. Fill EmbedKGQA row from the data run.

## 4. Reviewer-rebuttal map
1. "Flow doesn't beat baselines" → it does in the coupled regime (Fig. coupling, MoE tables); the
   MetaQA tie is the marginalizable endpoint we predict, not a failure.
2. "Only synthetic MoE" → real MoE FFN layer frontier is honest but not an LM; next rung = MoE in a
   small transformer on a real task (see §5). Stated as limitation.
3. "Doesn't scale" → implicit-diff + matrix-free, verified; enables the coupled/deep regime.
4. "Scooped (PathMoE/CoE)" → concede outcomes; claim mechanism + hard legality + coupling condition.
5. "Legality really 0?" → flux/connected-path legal by construction; per-layer argmax decode 0.15
   (still 6× below the gate). Stated precisely.

## 5. Highest-impact remaining work (ranked)
1. **Real MoE in a small transformer** on a real task (the rung that turns "primitive" into
   "usable"; the commercial claim). Bigger infra build.
2. 3-hop MetaQA as a real-data COUPLED instance (data staged; needs the scalable solver; a win
   here is a real-data flow>no-flow result).
3. EmbedKGQA run for the external anchor + hallucination contrast.
4. Multi-seed everything + WebQSP/CWQ for external validity.

## 6. Reproduce (no external data)
```
python3 density_sweep.py        # coupling crossover
python3 mycelial_final.py        # MoE-routing win (4 seeds)
python3 moe_layer_demo.py        # MoE FFN-layer frontier (3 seeds)
python3 confluence_spike.py      # legality wedge
python3 implicit_diff.py         # implicit-diff correctness + scaling
python3 metaqa_confluence_2hop_implicit.py --selfcheck   # MetaQA implicit gradient vs FD
python3 metaqa_embedkgqa_baseline.py --selfcheck         # ComplEx baseline
```
