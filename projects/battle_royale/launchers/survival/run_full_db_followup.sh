#!/usr/bin/env bash
set -euo pipefail

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"


OUT_DIR="pubg_lab.survival/checkpoints/full_db_reexperiment"
SPLIT_JSON="${OUT_DIR}/split_phase_all_current_seed42.json"
LOG_PATH="${OUT_DIR}/full_db_followup.log"

wait_for_file() {
  local path="$1"
  local label="$2"
  echo "waiting for ${label}: ${path}"
  until [[ -s "${path}" ]]; do
    sleep 60
  done
  echo "found ${label}: ${path}"
}

mkdir -p "${OUT_DIR}"

{
  echo "== full DB follow-up started $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  echo "split=${SPLIT_JSON}"

  wait_for_file "${OUT_DIR}/phase_full_all_current_e12.history.json" "full model history"

  echo
  echo "== evaluate dynamic full edge e12 =="
  python3 -u -m pubg_lab.survival.evaluation.eval_phase \
    --checkpoint "${OUT_DIR}/phase_full_all_current_e12.pt" \
    --split-json "${SPLIT_JSON}" \
    --split test \
    --edge-mode full \
    --batch 16 \
    --device auto \
    --out "${OUT_DIR}/eval_phase_full_all_current_e12_test.json"

  echo
  echo "== train dynamic no-graph e12 =="
  python3 -u -m pubg_lab.survival.training.train_phase \
    --split-json "${SPLIT_JSON}" \
    --checkpoint "${OUT_DIR}/phase_none_all_current_e12.pt" \
    --edge-mode none \
    --epochs 12 \
    --batch 8 \
    --seed 42 \
    --device auto

  echo
  echo "== evaluate dynamic no-graph e12 =="
  python3 -u -m pubg_lab.survival.evaluation.eval_phase \
    --checkpoint "${OUT_DIR}/phase_none_all_current_e12.pt" \
    --split-json "${SPLIT_JSON}" \
    --split test \
    --edge-mode none \
    --batch 16 \
    --device auto \
    --out "${OUT_DIR}/eval_phase_none_all_current_e12_test.json"

  echo "== full DB follow-up finished $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
} 2>&1 | tee -a "${LOG_PATH}"
