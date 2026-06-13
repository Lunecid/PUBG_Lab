#!/bin/bash
# MR-STGNN full squad 본실험: MR(snapshot+phase) vs phase-only(141d) vs HGB, 동일 split.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?}"
OUT=model_stgcn/checkpoints/mr_full_squad
mkdir -p "$OUT"
SPLIT=$OUT/split.json

if [[ ! -f "$OUT/mr_test.json" ]]; then
    echo "== MR train+eval $(date) =="
    python3 -m model_stgcn.train_mr --mode squad --epochs 12 --batch 8 --device mps --cached \
        --split-json "$SPLIT" --checkpoint "$OUT/mr.pt"
    python3 -m model_stgcn.eval_mr --checkpoint "$OUT/mr.pt" --split test --cached --device mps --out "$OUT/mr_test.json"
fi
if [[ ! -f "$OUT/phase_test.json" ]]; then
    echo "== phase-only train+eval $(date) =="
    python3 -m model_stgcn.train_phase --mode squad --v2-node-features --epochs 12 --batch 8 --device mps --cached \
        --split-json "$SPLIT" --checkpoint "$OUT/phase.pt"
    python3 -m model_stgcn.eval_phase --checkpoint "$OUT/phase.pt" --split test --cached --device mps --out "$OUT/phase_test.json"
fi
if [[ ! -f "$OUT/hgb/feature_ablation_metrics.csv" ]]; then
    echo "== HGB $(date) =="
    python3 -m model_stgcn.feature_ablation --mode squad --split-json "$SPLIT" --align-to "" --out-dir "$OUT/hgb"
fi
echo "== full squad 3종 완료 $(date) =="
