#!/bin/bash
# solo+duo(TPP) 확장: MR vs phase-only vs HGB, 동일 split. RAM 한계로 squad와 분리.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?}"
OUT=model_stgcn/checkpoints/mr_solo_duo
mkdir -p "$OUT"
SPLIT=$OUT/split.json

if [[ ! -f "$SPLIT" ]]; then
    echo "== build solo+duo split $(date) =="
    python3 -c "
from model_stgcn.splits import load_or_create_split
from stdb.queries import list_phase_eligible_match_ids
ids = list_phase_eligible_match_ids(None,'solo') + list_phase_eligible_match_ids(None,'duo')
load_or_create_split(ids, split_json='$SPLIT', seed=42, val_frac=0.1, test_frac=0.1, metadata={'source':'solo_duo'})
print('solo+duo ids:', len(ids))
"
fi
if [[ ! -f "$OUT/mr_test.json" ]]; then
    echo "== MR solo+duo $(date) =="
    python3 -m model_stgcn.train_mr --split-json "$SPLIT" --epochs 12 --batch 8 --device mps --cached --checkpoint "$OUT/mr.pt"
    python3 -m model_stgcn.eval_mr --checkpoint "$OUT/mr.pt" --split test --cached --device mps --out "$OUT/mr_test.json"
fi
if [[ ! -f "$OUT/phase_test.json" ]]; then
    echo "== phase-only solo+duo $(date) =="
    python3 -m model_stgcn.train_phase --v2-node-features --split-json "$SPLIT" --epochs 12 --batch 8 --device mps --cached --checkpoint "$OUT/phase.pt"
    python3 -m model_stgcn.eval_phase --checkpoint "$OUT/phase.pt" --split test --cached --device mps --out "$OUT/phase_test.json"
fi
if [[ ! -f "$OUT/hgb/feature_ablation_metrics.csv" ]]; then
    echo "== HGB solo+duo $(date) =="
    python3 -m model_stgcn.feature_ablation --split-json "$SPLIT" --align-to "" --out-dir "$OUT/hgb"
fi
echo "== solo/duo 3종 완료 $(date) =="
