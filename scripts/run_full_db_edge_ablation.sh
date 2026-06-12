#!/usr/bin/env bash
set -euo pipefail

cd /Users/lunecid/project/PUBG

OUT_DIR="model_stgcn/checkpoints/full_db_reexperiment"
SPLIT_JSON="${OUT_DIR}/split_phase_all_current_seed42.json"
LOG_PATH="${OUT_DIR}/full_db_edge_ablation.log"
ANALYSIS_DIR="${OUT_DIR}/phase_result_analysis_edges"

mkdir -p "${OUT_DIR}"

run_train_eval() {
  local edge_mode="$1"
  local ckpt="${OUT_DIR}/phase_${edge_mode}_all_current_e12.pt"
  local eval_out="${OUT_DIR}/eval_phase_${edge_mode}_all_current_e12_test.json"

  echo
  echo "== train dynamic ${edge_mode} e12 =="
  python3 -u -m model_stgcn.train_phase \
    --split-json "${SPLIT_JSON}" \
    --checkpoint "${ckpt}" \
    --edge-mode "${edge_mode}" \
    --epochs 12 \
    --batch 8 \
    --seed 42 \
    --device auto

  echo
  echo "== evaluate dynamic ${edge_mode} e12 =="
  python3 -u -m model_stgcn.eval_phase \
    --checkpoint "${ckpt}" \
    --split-json "${SPLIT_JSON}" \
    --split test \
    --edge-mode "${edge_mode}" \
    --batch 16 \
    --device auto \
    --out "${eval_out}"
}

{
  echo "== full DB edge ablation started $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  echo "split=${SPLIT_JSON}"

  run_train_eval proximity
  run_train_eval proximity_combat

  echo
  echo "== aligned phase-result analysis with edge ablations =="
  python3 -u -m model_stgcn.phase_result_analysis \
    --split-json "${SPLIT_JSON}" \
    --none-checkpoint "${OUT_DIR}/phase_none_all_current_e12.pt" \
    --proximity-checkpoint "${OUT_DIR}/phase_proximity_all_current_e12.pt" \
    --proximity-combat-checkpoint "${OUT_DIR}/phase_proximity_combat_all_current_e12.pt" \
    --full-checkpoint "${OUT_DIR}/phase_full_all_current_e12.pt" \
    --tuned-checkpoint "" \
    --out-dir "${ANALYSIS_DIR}" \
    --batch 32 \
    --device auto

  echo
  echo "== calibration with edge ablations =="
  python3 -u -m model_stgcn.calibration \
    --predictions "${ANALYSIS_DIR}/hazard_predictions_test_aligned.csv" \
    --out-dir "${OUT_DIR}/calibration_edges" \
    --bins 10

  python3 -u -m model_stgcn.calibration \
    --predictions "${ANALYSIS_DIR}/hazard_predictions_test_aligned.csv" \
    --out-dir "${OUT_DIR}/calibration_edges_by_phase" \
    --bins 10 \
    --group-cols phase_idx \
    --include-overall

  echo "== full DB edge ablation finished $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
} 2>&1 | tee -a "${LOG_PATH}"
