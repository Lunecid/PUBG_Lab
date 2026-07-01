#!/usr/bin/env bash
# Overnight V3 runs. Chained, each isolated so one failure does not block the next.
# [1] V3-comparable tabular baselines (LR/HGB)  -- closes GNN-vs-tabular significance
# [2] V3 controls matrix (resolution/decoder/graph_control) on the SHARED split
#     -- resolution = MR-vs-phase headline ablation (front-loaded, fast phase models first)
set -uo pipefail
cd /Users/lunecid/project/PUBG
export PG_PASSWORD=104204
SPLIT=model_stgcn/checkpoints/v3_full_main_causal/split.json

echo "===== OVERNIGHT START $(date) ====="

echo "----- [1/2] $(date +%H:%M:%S) V3 tabular baseline (LR/HGB) -----"
python3 scripts/v3_tabular_baseline.py > logs/v3_baseline.log 2>&1 \
  && echo "[1] BASELINE OK $(date +%H:%M:%S)" \
  || echo "[1] BASELINE FAILED $(date +%H:%M:%S) (continuing)"
touch logs/.baseline_done

echo "----- [2/2] $(date +%H:%M:%S) V3 controls matrix (resolution,decoder,graph_control) -----"
python3 -m model_stgcn.v3.run_matrix \
  --groups resolution,decoder,graph_control \
  --out-dir model_stgcn/checkpoints/v3_full_controls \
  --split-json "$SPLIT" \
  --mode squad --epochs 12 --batch 8 --workers 4 --log-every 250 --device mps \
  > logs/v3_controls.log 2>&1 \
  && echo "[2] CONTROLS OK $(date +%H:%M:%S)" \
  || echo "[2] CONTROLS FAILED/INTERRUPTED $(date +%H:%M:%S)"
touch logs/.controls_done

echo "===== OVERNIGHT DONE $(date) ====="
