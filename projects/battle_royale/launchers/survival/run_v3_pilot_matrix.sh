#!/usr/bin/env bash
set -euo pipefail

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

: "${PG_PASSWORD:?PG_PASSWORD must be set}"

python3 -m pubg_lab.survival.v3.data.run_matrix \
  --groups main \
  --out-dir pubg_lab.survival/checkpoints/v3_pilot_main \
  --mode squad \
  --limit 120 \
  --epochs 2 \
  --batch 4 \
  --device mps \
  --cached
