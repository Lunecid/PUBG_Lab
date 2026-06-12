#!/usr/bin/env bash
set -euo pipefail

cd /Users/lunecid/project/PUBG

OUT_DIR="model_stgcn/checkpoints/full_db_reexperiment"
SPLIT_JSON="${OUT_DIR}/split_phase_all_current_seed42.json"
LOG_PATH="${OUT_DIR}/full_db_reexperiment.log"

mkdir -p "${OUT_DIR}"

{
  echo "== full DB reexperiment started $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  echo "split=${SPLIT_JSON}"

  echo
  echo "== tabular baseline =="
  python3 -u -m model_stgcn.baselines \
    --split-json "${SPLIT_JSON}" \
    --out "${OUT_DIR}/baseline_phase_all_current_seed42.json"

  echo
  echo "== dynamic stgnn full edge e12 =="
  python3 -u -m model_stgcn.train_phase \
    --split-json "${SPLIT_JSON}" \
    --checkpoint "${OUT_DIR}/phase_full_all_current_e12.pt" \
    --edge-mode full \
    --epochs 12 \
    --batch 8 \
    --seed 42 \
    --device auto

  echo "== full DB reexperiment finished $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
} 2>&1 | tee -a "${LOG_PATH}"
