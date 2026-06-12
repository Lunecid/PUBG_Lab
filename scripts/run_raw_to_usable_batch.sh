#!/usr/bin/env bash
set -euo pipefail

cd /Users/lunecid/project/PUBG
mkdir -p model_stgcn/etl_runs

python3 -u scripts/convert_raw_to_usable.py \
  --stages base spatial edge \
  --workers 12 \
  --progress-every 100 \
  --out model_stgcn/etl_runs/raw_to_usable_latest.json \
  2>&1 | tee -a model_stgcn/etl_runs/raw_to_usable_latest.log
