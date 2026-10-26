#!/bin/bash
# Full-DB 재실행 스위트 — masked-pool 코드(CODE_REVIEW C4) 기준으로
# 페이퍼 수치 체계 전체를 재산출한다.
#
# 전제: maskedpool_retrain/phase_full_all_current_e12.pt 가 이미 존재
#       (full 변형은 별도 선행 학습 — logs/retrain_maskedpool_full.log)
#
# 단계:
#   1) none / proximity / proximity_combat 재학습 (각 ~2.8h, 순차)
#   2) 4개 변형 eval_phase (test, match-level bootstrap 1000)
#   3) phase_result_analysis — baseline(HGB) 적합 + 5-model aligned 테이블
#      + match-paired Wilcoxon (report §2·§3의 원천)
#   4) calibration (model별 / model×phase_idx별, report §4의 원천)
#
# 사용:
#   PG_PASSWORD=... nohup bash projects/battle_royale/launchers/survival/rerun_full_db_suite.sh \
#       >> logs/rerun_full_db_suite.log 2>&1 &
#
# 멱등성: 이미 존재하는 학습 체크포인트는 건너뛴다 (재시작 안전).

set -euo pipefail

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

: "${PG_PASSWORD:?PG_PASSWORD must be set}"

ROOT=pubg_lab.survival/checkpoints/full_db_reexperiment
RETRAIN=$ROOT/maskedpool_retrain
SPLIT=$ROOT/split_phase_all_current_seed42.json

echo "== suite started $(date) @ $(git rev-parse --short HEAD) =="

if [[ ! -f "$RETRAIN/phase_full_all_current_e12.pt" ]]; then
    echo "ERROR: full checkpoint missing — 선행 full 학습이 끝나야 한다" >&2
    exit 1
fi

# ── 1) 잔여 변형 학습 (원 실험과 동일 설정: e12 / batch 8 / h64 / seed 42) ──
# 완주 판정은 history.json (train_phase가 전 epoch 종료 후에만 기록).
# .pt는 best-val 시점마다 갱신되므로 중단된 학습도 파일이 남는다.
for mode in none proximity proximity_combat; do
    ckpt=$RETRAIN/phase_${mode}_all_current_e12.pt
    history=$RETRAIN/phase_${mode}_all_current_e12.history.json
    if [[ -f "$history" ]]; then
        echo "== skip train $mode (history exists — completed) =="
        continue
    fi
    rm -f "$ckpt"  # 중단된 부분 학습 체크포인트 제거 후 처음부터
    echo "== train $mode started $(date) =="
    python3 -m pubg_lab.survival.training.train_phase \
        --epochs 12 --batch 8 --edge-mode "$mode" --seed 42 \
        --split-json "$SPLIT" \
        --checkpoint "$ckpt"
    echo "== train $mode finished $(date) =="
done

# ── 2) 변형별 test 평가 + 95% CI ──
for mode in full none proximity proximity_combat; do
    echo "== eval $mode $(date) =="
    python3 -m pubg_lab.survival.evaluation.eval_phase \
        --checkpoint "$RETRAIN/phase_${mode}_all_current_e12.pt" \
        --split test --split-json "$SPLIT" \
        --bootstrap 1000 \
        --out "$RETRAIN/eval_phase_${mode}_test.json"
done

# ── 3) aligned 비교 + Wilcoxon (tuned는 구코드 산출물이므로 제외) ──
echo "== phase_result_analysis $(date) =="
python3 -m pubg_lab.survival.experiments.phase_result_analysis \
    --split-json "$SPLIT" \
    --none-checkpoint "$RETRAIN/phase_none_all_current_e12.pt" \
    --proximity-checkpoint "$RETRAIN/phase_proximity_all_current_e12.pt" \
    --proximity-combat-checkpoint "$RETRAIN/phase_proximity_combat_all_current_e12.pt" \
    --full-checkpoint "$RETRAIN/phase_full_all_current_e12.pt" \
    --tuned-checkpoint "" \
    --out-dir "$RETRAIN/phase_result_analysis"

# ── 4) calibration (원 실험과 동일 그룹핑) ──
PRED=$RETRAIN/phase_result_analysis/hazard_predictions_test_aligned.csv
echo "== calibration $(date) =="
python3 -m pubg_lab.survival.evaluation.calibration \
    --predictions "$PRED" --group-cols model \
    --out-dir "$RETRAIN/calibration"
python3 -m pubg_lab.survival.evaluation.calibration \
    --predictions "$PRED" --group-cols model phase_idx \
    --out-dir "$RETRAIN/calibration_by_phase"

echo "== suite finished $(date) =="
