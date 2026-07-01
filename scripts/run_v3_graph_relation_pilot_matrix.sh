#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

cmd=(
  python3 -m model_stgcn.v3.run_matrix
  --groups graph_relation
  --out-dir "${V3_OUT_DIR:-model_stgcn/checkpoints/v3_pilot_graph_relation}"
  --mode "${V3_MODE:-squad}"
  --limit "${V3_LIMIT:-120}"
  --epochs "${V3_EPOCHS:-2}"
  --batch "${V3_BATCH:-4}"
  --workers "${V3_WORKERS:-4}"
  --log-every "${V3_LOG_EVERY:-250}"
  --device "${V3_DEVICE:-mps}"
)

if [[ -n "${V3_SPLIT_JSON:-}" ]]; then
  cmd+=(--split-json "$V3_SPLIT_JSON")
fi

if [[ "${V3_CACHED:-0}" == "1" ]]; then
  cmd+=(--cached)
fi

if [[ "${V3_FORCE:-0}" == "1" ]]; then
  cmd+=(--force)
fi

"${cmd[@]}"
