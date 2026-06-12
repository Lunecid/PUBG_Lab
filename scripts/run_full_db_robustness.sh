#!/usr/bin/env bash
set -euo pipefail

cd /Users/lunecid/project/PUBG

OUT_DIR="model_stgcn/checkpoints/full_db_reexperiment"
SPLIT_JSON="${OUT_DIR}/split_phase_all_current_seed42.json"
LOG_PATH="${OUT_DIR}/full_db_robustness.log"

{
  echo "== full DB robustness started $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  echo "split=${SPLIT_JSON}"

  # 1) Edge sanity (inference-only): observed vs node-shuffled edges on the
  #    trained full-graph checkpoint. Shim directory satisfies the
  #    domain-layout contract of edge_sanity.py.
  SANITY_ROOT="${OUT_DIR}/sanity_root"
  DOM="${SANITY_ROOT}/All__current"
  mkdir -p "${DOM}"
  cp -f "${SPLIT_JSON}" "${DOM}/split_seed42.json"
  ln -sf "../../phase_full_all_current_e12.pt" "${DOM}/phase_full_e12.pt"

  echo
  echo "== edge sanity (node_shuffle x5) =="
  python3 -u -m model_stgcn.edge_sanity \
    --root "${SANITY_ROOT}" \
    --out-dir "${OUT_DIR}/edge_sanity" \
    --edge-mode full \
    --controls node_shuffle \
    --repeats 5 \
    --batch 16 \
    --device auto

  echo "== edge sanity done $(date -u +%Y-%m-%dT%H:%M:%SZ) =="

  # 2) Seed repeats: training-seed variance on the fixed canonical split.
  #    Seeds mirror the legacy snapshot experiment (13, 2026); seed 42 already
  #    exists from the main reexperiment.
  mkdir -p "${OUT_DIR}/seed_repeats"
  for SEED in 13 2026; do
    for MODE in full none; do
      CKPT="${OUT_DIR}/seed_repeats/phase_${MODE}_all_current_e12_seed${SEED}.pt"
      EVAL="${OUT_DIR}/seed_repeats/eval_phase_${MODE}_all_current_e12_seed${SEED}_test.json"

      echo
      echo "== train ${MODE} seed ${SEED} $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
      python3 -u -m model_stgcn.train_phase \
        --split-json "${SPLIT_JSON}" \
        --checkpoint "${CKPT}" \
        --edge-mode "${MODE}" \
        --epochs 12 \
        --batch 8 \
        --seed "${SEED}" \
        --device auto

      echo
      echo "== eval ${MODE} seed ${SEED} =="
      python3 -u -m model_stgcn.eval_phase \
        --checkpoint "${CKPT}" \
        --split-json "${SPLIT_JSON}" \
        --split test \
        --edge-mode "${MODE}" \
        --batch 16 \
        --device auto \
        --out "${EVAL}"
    done
  done

  echo "== full DB robustness finished $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
} 2>&1 | tee -a "${LOG_PATH}"
