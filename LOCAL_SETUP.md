# Running Confluence locally

Everything is on GitHub; going local is just clone + reinstall. `passover.md` and
`SUBMISSION_STATUS.md` are the full handoff — read those first in any new session.

## 1. Install Claude Code
```bash
# native installer (macOS/Linux):
curl -fsSL https://claude.ai/install.sh | bash
# or npm:
npm install -g @anthropic-ai/claude-code
```

## 2. Clone the repo + working branch
```bash
git clone https://github.com/tanushappapogu-max/Confluence.git
cd Confluence
git checkout claude/passover-iclr-review-r7292c
git config user.name "Tanush Appapogu"
git config user.email "tanush.appapogu@gmail.com"   # keep commits under your name
```

## 3. Re-stage the MetaQA data (gitignored — not in the repo)
```bash
git clone --depth 1 https://github.com/Tiny-Small/CS5284_Project /tmp/mqa
mkdir -p data/metaqa/kb data/metaqa/1-hop data/metaqa/2-hop data/metaqa/3-hop
SRC=/tmp/mqa/GNN-cluster/data/raw
cp "$SRC/kb.txt" data/metaqa/kb/kb.txt
for h in 1 2 3; do
  cp "$SRC/$h-hop/qa_train.txt" data/metaqa/$h-hop/
  cp "$SRC/$h-hop/qa_test.txt"  data/metaqa/$h-hop/
done
# sanity: kb.txt should be 134,741 lines; 2-hop qa_train 118,980
wc -l data/metaqa/kb/kb.txt data/metaqa/2-hop/qa_train.txt
```

## 4. Dependencies
```bash
pip install torch numpy matplotlib scipy
```

## 5. Start
```bash
claude
# then: "read passover.md and SUBMISSION_STATUS.md, then continue the project"
```

## Why local is better here
- The flow solves that took 20–75 min/epoch in the cloud sandbox run far faster on your
  machine; with a **GPU** the 3-hop and real-MoE-transformer experiments become practical.
- No network policy blocking dataset hosts; no proxy outages; you own the loop.

## Reproduce the current results (no external data needed)
```bash
python3 density_sweep.py                                  # coupling crossover
python3 mycelial_final.py                                 # MoE-routing win (4 seeds)
python3 moe_layer_demo.py                                 # MoE FFN-layer frontier (3 seeds)
python3 implicit_diff.py                                  # implicit-diff correctness + scaling
python3 metaqa_confluence_2hop_implicit.py --selfcheck    # MetaQA implicit gradient vs finite-diff
python3 metaqa_embedkgqa_baseline.py --selfcheck          # ComplEx baseline
```

## Experiments that need the staged data
```bash
python3 metaqa_confluence_2hop_ablation.py                # no-flow ablation (the decisive control)
TRAIN_DEGNORM=1 python3 metaqa_confluence_2hop_v2.py      # flow model, degree-norm training
CAP=400 python3 metaqa_confluence_2hop_implicit.py        # implicit-diff flow, lifted node cap
```
