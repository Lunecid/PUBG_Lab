#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

python3 -m model_stgcn.v3.run_matrix \
  --groups main \
  --out-dir model_stgcn/checkpoints/v3_pilot_main \
  --mode squad \
  --limit 120 \
  --epochs 2 \
  --batch 4 \
  --device mps \
  --cached

