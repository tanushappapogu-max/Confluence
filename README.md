# Confluence

**When does conserved-flow routing help? A coupling condition for Mixture-of-Experts, with a legality guarantee.**

Sparse Mixture-of-Experts (MoE) gates pick experts independently at every layer. This repo studies a
*coupled* router instead: a differentiable Tero–Nakagaki (Physarum) conserved-flow dynamic that relaxes
to one min-cost path through a graph whose edges are exactly the legal expert transitions. Flux on an
illegal transition is zero by construction.

Main finding: coupled routing beats independent per-step gating **when the steps are mutually
constraining**, and ties when the output marginalizes over intermediate steps (MetaQA 2-hop).

## Results (paper tables)

| Experiment | Script | Raw log |
|---|---|---|
| Coupling crossover (density sweep, 3 seeds) | `density_sweep.py` | `results/density_sweep_flow_vs_independent.txt` |
| Constrained expert routing vs top-k / Routing-Free MoE (4 seeds) | `mycelial_final.py` | `results/moe_routing_comparison_4seed.txt` |
| MoE FFN accuracy/compute frontier (3 seeds) | `moe_layer_demo.py` | `results/moe_layer_frontier_3seed.txt` |
| MoE transformer, flow vs top-k gate (3 seeds + density sweep) | `moe_transformer.py` | `results/moe_transformer_*.txt` |
| MetaQA 2-hop: flow vs matched no-flow ablation | `metaqa_confluence_2hop_v2.py`, `metaqa_confluence_2hop_ablation.py` | `results/metaqa_2hop_noflow_ablation.txt` |
| Implicit differentiation check (vs finite differences) | `implicit_diff.py` | printed |

Other scripts: `metaqa_confluence_2hop_implicit.py` (MetaQA with implicit-diff training),
`metaqa_confluence_3hop_ablation.py` / `metaqa_3hop_pilot.py` (3-hop), `metaqa_embedkgqa_baseline.py`
(ComplEx/EmbedKGQA-style baseline), `implicit_train_demo.py`, `metaqa_load.py` (data stats).

## Run everything

- **Colab (GPU, checkpoints to Google Drive, resumable):** open
  [`Confluence_Colab.ipynb`](https://colab.research.google.com/github/tanushappapogu-max/Confluence/blob/main/Confluence_Colab.ipynb),
  set the runtime to GPU, and Run all.
- **Locally:** `pip install torch numpy matplotlib`, then e.g. `python3 moe_transformer.py`
  (env vars `NSEEDS`, `EPOCHS`, `NTRAIN`, `DENSITY`, `SEED_START` control scale).

## Figures and paper

- `figstyle.py` holds the shared palette (ours = blue, main baseline = orange, top-2 = aqua, dense = gray)
  and matplotlib settings; `make_fig_coupling.py`, `make_fig_moe.py`, `make_fig_implicit.py` write
  vector PDFs to `paper/figs/`.
- `paper/` is the ICLR 2027 LaTeX source (`main.tex`, `references.bib`, official style files).
  Compile with pdfLaTeX + BibTeX, or upload the folder to Overleaf.

## Data

MetaQA (KB + 2/3-hop QA) goes in `data/metaqa/` (not tracked). The Colab notebook fetches and stages it.

## Layout

```
*.py                 experiments (see table above) + figure scripts
figstyle.py          shared figure palette and style
results/             raw logs behind every number in the paper
paper/               ICLR 2027 LaTeX source + figures
archive/             early prototypes (mechanism spikes, 1-hop, mock KG); kept for history, not maintained
Confluence_Colab.ipynb   full-scale runs with Drive checkpointing
```
