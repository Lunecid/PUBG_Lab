#!/bin/bash
# seed 반복 (1맵 Tiger squad, split 고정 / init seed만 변경): MR/phase ΔAUC 분산.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?}"
OUT=model_stgcn/checkpoints/mr_tiger_squad
SPLIT=$OUT/split.json

# seed 42 (기존 mr.pt/phase.pt) 재평가 → json 통일
[[ -f $OUT/mr_seed42_test.json ]]    || python3 -m model_stgcn.eval_mr    --checkpoint $OUT/mr.pt    --split test --cached --device mps --out $OUT/mr_seed42_test.json
[[ -f $OUT/phase_seed42_test.json ]] || python3 -m model_stgcn.eval_phase --checkpoint $OUT/phase.pt --split test --cached --device mps --out $OUT/phase_seed42_test.json

for seed in 13 2026; do
    if [[ ! -f $OUT/mr_seed${seed}_test.json ]]; then
        echo "== MR seed $seed $(date) =="
        python3 -m model_stgcn.train_mr --map Tiger_Main --mode squad --seed $seed \
            --epochs 12 --batch 8 --device mps --cached --split-json $SPLIT --checkpoint $OUT/mr_seed$seed.pt
        python3 -m model_stgcn.eval_mr --checkpoint $OUT/mr_seed$seed.pt --split test --cached --device mps --out $OUT/mr_seed${seed}_test.json
    fi
    if [[ ! -f $OUT/phase_seed${seed}_test.json ]]; then
        echo "== phase seed $seed $(date) =="
        python3 -m model_stgcn.train_phase --map Tiger_Main --mode squad --v2-node-features --seed $seed \
            --epochs 12 --batch 8 --device mps --cached --split-json $SPLIT --checkpoint $OUT/phase_seed$seed.pt
        python3 -m model_stgcn.eval_phase --checkpoint $OUT/phase_seed$seed.pt --split test --cached --device mps --out $OUT/phase_seed${seed}_test.json
    fi
done
echo "== seed repeat done $(date) =="
