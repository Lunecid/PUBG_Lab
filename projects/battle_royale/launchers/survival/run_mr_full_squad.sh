#!/bin/bash
# MR-STGNN full squad 본실험: MR(snapshot+phase) vs phase-only(141d) vs HGB, 동일 split.
set -euo pipefail

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
: "${PG_PASSWORD:?}"
OUT=pubg_lab.survival/checkpoints/mr_full_squad
mkdir -p "$OUT"
SPLIT=$OUT/split.json

if [[ ! -f "$OUT/mr_test.json" ]]; then
    echo "== MR train+eval $(date) =="
    python3 -m pubg_lab.survival.training.train_mr --mode squad --epochs 12 --batch 8 --device mps --cached \
        --split-json "$SPLIT" --checkpoint "$OUT/mr.pt"
    python3 -m pubg_lab.survival.evaluation.eval_mr --checkpoint "$OUT/mr.pt" --split test --cached --device mps --out "$OUT/mr_test.json"
fi
if [[ ! -f "$OUT/phase_test.json" ]]; then
    echo "== phase-only train+eval $(date) =="
    python3 -m pubg_lab.survival.training.train_phase --mode squad --v2-node-features --epochs 12 --batch 8 --device mps --cached \
        --split-json "$SPLIT" --checkpoint "$OUT/phase.pt"
    python3 -m pubg_lab.survival.evaluation.eval_phase --checkpoint "$OUT/phase.pt" --split test --cached --device mps --out "$OUT/phase_test.json"
fi
if [[ ! -f "$OUT/hgb/feature_ablation_metrics.csv" ]]; then
    echo "== HGB $(date) =="
    python3 -m pubg_lab.survival.experiments.feature_ablation --mode squad --split-json "$SPLIT" --align-to "" --out-dir "$OUT/hgb"
fi
echo "== full squad 3종 완료 $(date) =="
