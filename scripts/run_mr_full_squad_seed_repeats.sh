#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

OUT=model_stgcn/checkpoints/mr_full_squad
SPLIT=$OUT/split.json
mkdir -p "$OUT"

echo "== MR full-squad seed repeats started $(date) =="
echo "split=$SPLIT"

# Normalize seed-42 names so downstream aggregation can treat all seeds equally.
if [[ ! -f "$OUT/mr_seed42_test.json" ]]; then
  python3 -m model_stgcn.eval_mr \
    --checkpoint "$OUT/mr.pt" --split test --cached --device mps \
    --out "$OUT/mr_seed42_test.json"
fi
if [[ ! -f "$OUT/phase_seed42_test.json" ]]; then
  python3 -m model_stgcn.eval_phase \
    --checkpoint "$OUT/phase.pt" --split test --cached --device mps \
    --out "$OUT/phase_seed42_test.json"
fi

for seed in 13 2026; do
  mr_ckpt="$OUT/mr_seed${seed}.pt"
  mr_eval="$OUT/mr_seed${seed}_test.json"
  mr_shuffle="$OUT/mr_seed${seed}_shuffle_time_seed0_test.json"
  phase_ckpt="$OUT/phase_seed${seed}.pt"
  phase_eval="$OUT/phase_seed${seed}_test.json"

  if [[ ! -f "$mr_eval" ]]; then
    echo
    echo "== train MR full-squad seed $seed $(date) =="
    python3 -m model_stgcn.train_mr \
      --mode squad --seed "$seed" \
      --epochs 12 --batch 8 --device mps --cached \
      --split-json "$SPLIT" \
      --checkpoint "$mr_ckpt"
    python3 -m model_stgcn.eval_mr \
      --checkpoint "$mr_ckpt" --split test --cached --device mps \
      --out "$mr_eval"
  else
    echo "== skip existing $mr_eval =="
  fi

  if [[ ! -f "$mr_shuffle" ]]; then
    echo
    echo "== eval MR full-squad seed $seed time-shuffle $(date) =="
    python3 -m model_stgcn.eval_mr \
      --checkpoint "$mr_ckpt" --split test --cached --device mps \
      --shuffle-time --shuffle-seed 0 \
      --out "$mr_shuffle"
  else
    echo "== skip existing $mr_shuffle =="
  fi

  if [[ ! -f "$phase_eval" ]]; then
    echo
    echo "== train phase-only full-squad seed $seed $(date) =="
    python3 -m model_stgcn.train_phase \
      --mode squad --v2-node-features --seed "$seed" \
      --epochs 12 --batch 8 --device mps --cached \
      --split-json "$SPLIT" \
      --checkpoint "$phase_ckpt"
    python3 -m model_stgcn.eval_phase \
      --checkpoint "$phase_ckpt" --split test --cached --device mps \
      --out "$phase_eval"
  else
    echo "== skip existing $phase_eval =="
  fi
done

echo "== MR full-squad seed repeats finished $(date) =="
