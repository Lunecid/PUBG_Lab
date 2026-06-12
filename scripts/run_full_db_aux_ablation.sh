#!/usr/bin/env bash
set -euo pipefail

cd /Users/lunecid/project/PUBG

OUT_DIR="model_stgcn/checkpoints/full_db_reexperiment"
SPLIT_JSON="${OUT_DIR}/split_phase_all_current_seed42.json"
ROBUST_LOG="${OUT_DIR}/full_db_robustness.log"
LOG_PATH="${OUT_DIR}/full_db_aux_ablation.log"
AUX_DIR="${OUT_DIR}/aux_ablation"

echo "waiting for robustness chain to finish..."
until grep -q "full DB robustness finished" "${ROBUST_LOG}" 2>/dev/null; do
  sleep 300
done

mkdir -p "${AUX_DIR}"
{
  echo "== aux-head ablation started $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  # Hazard-only variants: auxiliary placement/top-k loss weights set to zero.
  for MODE in full none; do
    CKPT="${AUX_DIR}/phase_${MODE}_hazonly_e12.pt"
    EVAL="${AUX_DIR}/eval_phase_${MODE}_hazonly_e12_test.json"
    echo
    echo "== train ${MODE} hazard-only $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
    python3 -u -m model_stgcn.train_phase \
      --split-json "${SPLIT_JSON}" \
      --checkpoint "${CKPT}" \
      --edge-mode "${MODE}" \
      --beta 0 --gamma 0 \
      --epochs 12 --batch 8 --seed 42 --device auto
    echo "== eval ${MODE} hazard-only =="
    python3 -u -m model_stgcn.eval_phase \
      --checkpoint "${CKPT}" \
      --split-json "${SPLIT_JSON}" \
      --split test --edge-mode "${MODE}" --batch 16 --device auto \
      --out "${EVAL}"
  done
  echo "== aux-head ablation finished $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
} 2>&1 | tee -a "${LOG_PATH}"
