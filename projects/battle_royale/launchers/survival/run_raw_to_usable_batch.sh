#!/usr/bin/env bash
set -euo pipefail

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p pubg_lab.survival/etl_runs

python3 -u projects/data_preparation/convert_raw_to_usable.py \
  --stages base spatial edge \
  --workers 12 \
  --progress-every 100 \
  --out pubg_lab.survival/etl_runs/raw_to_usable_latest.json \
  2>&1 | tee -a pubg_lab.survival/etl_runs/raw_to_usable_latest.log
