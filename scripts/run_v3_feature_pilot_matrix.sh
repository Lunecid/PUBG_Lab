#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

cmd=(
  python3 -m model_stgcn.v3.run_matrix
  --groups feature
  --out-dir "${V3_OUT_DIR:-model_stgcn/checkpoints/v3_pilot_feature}"
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
