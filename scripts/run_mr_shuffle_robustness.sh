#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

run_eval() {
  local checkpoint="$1"
  local out="$2"
  local seed="$3"
  local device="${4:-mps}"
  if [[ -f "$out" ]]; then
    echo "== skip existing $out =="
    return
  fi
  echo "== shuffle eval checkpoint=$checkpoint seed=$seed out=$out =="
  python3 -m model_stgcn.eval_mr \
    --checkpoint "$checkpoint" \
    --split test \
    --cached \
    --device "$device" \
    --shuffle-time \
    --shuffle-seed "$seed" \
    --out "$out"
}

run_eval model_stgcn/checkpoints/mr_full_squad/mr.pt \
  model_stgcn/checkpoints/mr_full_squad/mr_shuffle_time_seed0_test.json 0
run_eval model_stgcn/checkpoints/mr_full_squad/mr.pt \
  model_stgcn/checkpoints/mr_full_squad/mr_shuffle_time_seed1_test.json 1
run_eval model_stgcn/checkpoints/mr_full_squad/mr.pt \
  model_stgcn/checkpoints/mr_full_squad/mr_shuffle_time_seed2_test.json 2

run_eval model_stgcn/checkpoints/mr_tiger_squad/mr.pt \
  model_stgcn/checkpoints/mr_tiger_squad/mr_seed42_shuffle_time_seed0_test.json 0
run_eval model_stgcn/checkpoints/mr_tiger_squad/mr_seed13.pt \
  model_stgcn/checkpoints/mr_tiger_squad/mr_seed13_shuffle_time_seed0_test.json 0
run_eval model_stgcn/checkpoints/mr_tiger_squad/mr_seed2026.pt \
  model_stgcn/checkpoints/mr_tiger_squad/mr_seed2026_shuffle_time_seed0_test.json 0

run_eval model_stgcn/checkpoints/mr_solo_duo/mr.pt \
  model_stgcn/checkpoints/mr_solo_duo/mr_shuffle_time_seed0_test.json 0

echo "== MR shuffle robustness done =="
