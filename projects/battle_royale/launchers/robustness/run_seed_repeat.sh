#!/bin/bash
# seed 반복 (1맵 Tiger squad, split 고정 / init seed만 변경): MR/phase ΔAUC 분산.
set -euo pipefail

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
: "${PG_PASSWORD:?}"
OUT=pubg_lab.survival/checkpoints/mr_tiger_squad
SPLIT=$OUT/split.json

# seed 42 (기존 mr.pt/phase.pt) 재평가 → json 통일
[[ -f $OUT/mr_seed42_test.json ]]    || python3 -m pubg_lab.survival.evaluation.eval_mr    --checkpoint $OUT/mr.pt    --split test --cached --device mps --out $OUT/mr_seed42_test.json
[[ -f $OUT/phase_seed42_test.json ]] || python3 -m pubg_lab.survival.evaluation.eval_phase --checkpoint $OUT/phase.pt --split test --cached --device mps --out $OUT/phase_seed42_test.json

for seed in 13 2026; do
    if [[ ! -f $OUT/mr_seed${seed}_test.json ]]; then
        echo "== MR seed $seed $(date) =="
        python3 -m pubg_lab.survival.training.train_mr --map Tiger_Main --mode squad --seed $seed \
            --epochs 12 --batch 8 --device mps --cached --split-json $SPLIT --checkpoint $OUT/mr_seed$seed.pt
        python3 -m pubg_lab.survival.evaluation.eval_mr --checkpoint $OUT/mr_seed$seed.pt --split test --cached --device mps --out $OUT/mr_seed${seed}_test.json
    fi
    if [[ ! -f $OUT/phase_seed${seed}_test.json ]]; then
        echo "== phase seed $seed $(date) =="
        python3 -m pubg_lab.survival.training.train_phase --map Tiger_Main --mode squad --v2-node-features --seed $seed \
            --epochs 12 --batch 8 --device mps --cached --split-json $SPLIT --checkpoint $OUT/phase_seed$seed.pt
        python3 -m pubg_lab.survival.evaluation.eval_phase --checkpoint $OUT/phase_seed$seed.pt --split test --cached --device mps --out $OUT/phase_seed${seed}_test.json
    fi
done
echo "== seed repeat done $(date) =="
