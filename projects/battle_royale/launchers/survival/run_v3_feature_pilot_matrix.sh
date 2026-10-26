#!/usr/bin/env bash
set -euo pipefail

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

: "${PG_PASSWORD:?PG_PASSWORD must be set}"

cmd=(
  python3 -m pubg_lab.survival.v3.data.run_matrix
  --groups feature
  --out-dir "${V3_OUT_DIR:-pubg_lab.survival/checkpoints/v3_pilot_feature}"
  --mode "${V3_MODE:-squad}"
  --limit "${V3_LIMIT:-120}"
  --epochs "${V3_EPOCHS:-2}"
  --batch "${V3_BATCH:-4}"
  --device "${V3_DEVICE:-mps}"
  --cached
)

if [[ -n "${V3_SPLIT_JSON:-}" ]]; then
  cmd+=(--split-json "$V3_SPLIT_JSON")
fi

"${cmd[@]}"
