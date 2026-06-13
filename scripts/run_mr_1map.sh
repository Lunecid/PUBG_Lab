#!/bin/bash
# MR-STGNN 1맵 검증 (Tiger squad): MR(snapshot+phase, deep-sup) vs phase-only(141d).
# suite가 mps를 비울 때까지 대기 후 동일 split으로 둘 다 학습 → val 비교.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?}"
OUT=model_stgcn/checkpoints/mr_tiger_squad
mkdir -p "$OUT"
SPLIT=$OUT/split.json

while pgrep -f "rerun_full_db_suite.sh" > /dev/null 2>&1; do
    echo "== waiting for suite (mps busy) $(date) =="; sleep 300
done
echo "== suite done — mps free $(date) =="

# MR: snapshot 10s + phase, deep supervision
if [[ ! -f "$OUT/mr.history.json" ]]; then
    echo "== train MR $(date) =="
    python3 -m model_stgcn.train_mr --map Tiger_Main --mode squad \
        --epochs 12 --batch 8 --device mps --cached \
        --split-json "$SPLIT" --checkpoint "$OUT/mr.pt"
fi
# phase-only baseline: same 141d nodes, no snapshot tier, same split
if [[ ! -f "$OUT/phase.history.json" ]]; then
    echo "== train phase-only baseline $(date) =="
    python3 -m model_stgcn.train_phase --map Tiger_Main --mode squad --v2-node-features \
        --epochs 12 --batch 8 --device mps --cached \
        --split-json "$SPLIT" --checkpoint "$OUT/phase.pt"
fi
echo "== MR 1-map validation done $(date) =="
