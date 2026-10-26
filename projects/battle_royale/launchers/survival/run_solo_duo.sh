#!/bin/bash
# solo+duo(TPP) 확장: MR vs phase-only vs HGB, 동일 split. RAM 한계로 squad와 분리.
set -euo pipefail

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
: "${PG_PASSWORD:?}"
OUT=pubg_lab.survival/checkpoints/mr_solo_duo
mkdir -p "$OUT"
SPLIT=$OUT/split.json

if [[ ! -f "$SPLIT" ]]; then
    echo "== build solo+duo split $(date) =="
    python3 -c "
from model_stgcn.splits import load_or_create_split
from pubg_lab.data.stdb.queries.queries import list_phase_eligible_match_ids
ids = list_phase_eligible_match_ids(None,'solo') + list_phase_eligible_match_ids(None,'duo')
load_or_create_split(ids, split_json='$SPLIT', seed=42, val_frac=0.1, test_frac=0.1, metadata={'source':'solo_duo'})
print('solo+duo ids:', len(ids))
"
fi
if [[ ! -f "$OUT/mr_test.json" ]]; then
    echo "== MR solo+duo $(date) =="
    python3 -m pubg_lab.survival.training.train_mr --split-json "$SPLIT" --epochs 12 --batch 8 --device mps --cached --checkpoint "$OUT/mr.pt"
    python3 -m pubg_lab.survival.evaluation.eval_mr --checkpoint "$OUT/mr.pt" --split test --cached --device mps --out "$OUT/mr_test.json"
fi
if [[ ! -f "$OUT/phase_test.json" ]]; then
    echo "== phase-only solo+duo $(date) =="
    python3 -m pubg_lab.survival.training.train_phase --v2-node-features --split-json "$SPLIT" --epochs 12 --batch 8 --device mps --cached --checkpoint "$OUT/phase.pt"
    python3 -m pubg_lab.survival.evaluation.eval_phase --checkpoint "$OUT/phase.pt" --split test --cached --device mps --out "$OUT/phase_test.json"
fi
if [[ ! -f "$OUT/hgb/feature_ablation_metrics.csv" ]]; then
    echo "== HGB solo+duo $(date) =="
    python3 -m pubg_lab.survival.experiments.feature_ablation --split-json "$SPLIT" --align-to "" --out-dir "$OUT/hgb"
fi
echo "== solo/duo 3종 완료 $(date) =="
