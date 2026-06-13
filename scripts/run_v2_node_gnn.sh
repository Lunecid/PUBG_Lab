#!/bin/bash
# B3/B4 (FEATURE_DESIGN_V2 §11.2): F6+F7 enriched 노드 피처(141d) 위에서
# graph on/off — "graph가 역할/압박 요약을 넘어 무엇을 더하는가"의 공정 검증.
#
#   B3 = --v2-node-features --edge-mode none   (graph off)
#   B4 = --v2-node-features --edge-mode full   (graph on, E=3)
#
# rerun_full_db_suite.sh(maskedpool 재산출)가 GPU를 쓰는 동안 대기했다가
# 시작한다. 동일 프로토콜: e12 / batch 8 / h64 / seed 42 / 동일 split.
#
# 사용:
#   PG_PASSWORD=... nohup bash scripts/run_v2_node_gnn.sh \
#       >> logs/run_v2_node_gnn.log 2>&1 &
#
# 멱등성: 완주 판정은 history.json (suite와 동일 규약).

set -euo pipefail
cd "$(dirname "$0")/.."

: "${PG_PASSWORD:?PG_PASSWORD must be set}"

ROOT=model_stgcn/checkpoints/full_db_reexperiment
OUT=$ROOT/v2_node_features
SPLIT=$ROOT/split_phase_all_current_seed42.json
REF_FULL_66D=$ROOT/maskedpool_retrain/phase_full_all_current_e12.pt
mkdir -p "$OUT"

echo "== v2-node-gnn started $(date) @ $(git rev-parse --short HEAD) =="

# ── 0) suite가 GPU를 비울 때까지 대기 ──
while pgrep -f "rerun_full_db_suite.sh" > /dev/null 2>&1; do
    echo "== waiting for rerun_full_db_suite to finish $(date) =="
    sleep 300
done
echo "== suite gone — GPU free $(date) =="

if [[ ! -f "$REF_FULL_66D" ]]; then
    echo "WARNING: 66d maskedpool full checkpoint missing — reference column will be skipped" >&2
fi

# ── 1) B3 / B4 학습 ── (bash 3.2 호환: associative array 대신 case)
for tag in v2_none v2_full; do
    case "$tag" in v2_none) mode=none ;; v2_full) mode=full ;; esac
    ckpt=$OUT/phase_${tag}_e12.pt
    history=$OUT/phase_${tag}_e12.history.json
    if [[ -f "$history" ]]; then
        echo "== skip train $tag (history exists — completed) =="
        continue
    fi
    rm -f "$ckpt"
    echo "== train $tag (edge-mode=$mode) started $(date) =="
    python3 -m model_stgcn.train_phase \
        --epochs 12 --batch 8 --edge-mode "$mode" --seed 42 \
        --v2-node-features --cached \
        --split-json "$SPLIT" \
        --checkpoint "$ckpt"
    echo "== train $tag finished $(date) =="
done

# ── 2) test 평가 + match-level bootstrap 95% CI ──
for tag in v2_none v2_full; do
    echo "== eval $tag $(date) =="
    python3 -m model_stgcn.eval_phase \
        --checkpoint "$OUT/phase_${tag}_e12.pt" \
        --split test --split-json "$SPLIT" \
        --bootstrap 1000 \
        --out "$OUT/eval_phase_${tag}_test.json"
done

# ── 3) aligned 비교 + Wilcoxon ──
# 슬롯 매핑(출력 모델명 주의):
#   dynamic_none_e12  = B3 (141d, graph off)
#   dynamic_full_e12  = B4 (141d, graph on)
#   dynamic_full_tuned = 참조용 66d maskedpool full (이름만 tuned 슬롯 재사용)
# HGB(72-col) 베이스라인은 분석 스크립트가 자체 적합한다.
echo "== phase_result_analysis $(date) =="
python3 -m model_stgcn.phase_result_analysis \
    --split-json "$SPLIT" \
    --none-checkpoint "$OUT/phase_v2_none_e12.pt" \
    --proximity-checkpoint "" \
    --proximity-combat-checkpoint "" \
    --full-checkpoint "$OUT/phase_v2_full_e12.pt" \
    --tuned-checkpoint "${REF_FULL_66D}" \
    --out-dir "$OUT/phase_result_analysis"

echo "== v2-node-gnn finished $(date) =="
