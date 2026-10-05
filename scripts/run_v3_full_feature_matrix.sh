#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

cmd=(
  python3 -m model_stgcn.v3.run_matrix
  --groups feature
  --out-dir "${V3_OUT_DIR:-model_stgcn/checkpoints/v3_full_feature}"
  --mode "${V3_MODE:-squad}"
  --epochs "${V3_EPOCHS:-12}"
  --batch "${V3_BATCH:-8}"
  --workers "${V3_WORKERS:-4}"
  --log-every "${V3_LOG_EVERY:-250}"
  --device "${V3_DEVICE:-mps}"
)

if [[ -n "${V3_LIMIT:-}" ]]; then
  cmd+=(--limit "$V3_LIMIT")
fi

if [[ -n "${V3_SPLIT_JSON:-}" ]]; then
  cmd+=(--split-json "$V3_SPLIT_JSON" --require-existing-split)
fi

if [[ "${V3_CACHED:-0}" == "1" ]]; then
  cmd+=(--cached)
fi

if [[ "${V3_FORCE:-0}" == "1" ]]; then
  cmd+=(--force)
fi

"${cmd[@]}"
