#!/usr/bin/env bash
# Overnight V3 runs. Chained, each isolated so one failure does not block the next.
# [1] V3-comparable tabular baselines (LR/HGB)  -- closes GNN-vs-tabular significance
# [2] V3 controls matrix (resolution/decoder/graph_control) on the SHARED split
#     -- resolution = MR-vs-phase headline ablation (front-loaded, fast phase models first)
set -uo pipefail

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
: "${PG_PASSWORD:?PG_PASSWORD must be set}"
SPLIT=pubg_lab.survival/checkpoints/v3_full_main_causal/split.json

echo "===== OVERNIGHT START $(date) ====="

echo "----- [1/2] $(date +%H:%M:%S) V3 tabular baseline (LR/HGB) -----"
python3 projects/battle_royale/analysis/robustness/v3_tabular_baseline.py > logs/v3_baseline.log 2>&1 \
  && echo "[1] BASELINE OK $(date +%H:%M:%S)" \
  || echo "[1] BASELINE FAILED $(date +%H:%M:%S) (continuing)"
touch logs/.baseline_done

echo "----- [2/2] $(date +%H:%M:%S) V3 controls matrix (resolution,decoder,graph_control) -----"
python3 -m pubg_lab.survival.v3.data.run_matrix \
  --groups resolution,decoder,graph_control \
  --out-dir pubg_lab.survival/checkpoints/v3_full_controls \
  --split-json "$SPLIT" \
  --require-existing-split \
  --mode squad --epochs 12 --batch 8 --workers 4 --log-every 250 --device mps \
  > logs/v3_controls.log 2>&1 \
  && echo "[2] CONTROLS OK $(date +%H:%M:%S)" \
  || echo "[2] CONTROLS FAILED/INTERRUPTED $(date +%H:%M:%S)"
touch logs/.controls_done

echo "===== OVERNIGHT DONE $(date) ====="
