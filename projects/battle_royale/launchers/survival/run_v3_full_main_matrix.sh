#!/usr/bin/env bash
set -euo pipefail

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

: "${PG_PASSWORD:?PG_PASSWORD must be set}"

cmd=(
  python3 -m pubg_lab.survival.v3.data.run_matrix
  --groups main
  --out-dir "${V3_OUT_DIR:-pubg_lab.survival/checkpoints/v3_full_main}"
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

if [[ "${V3_CACHED:-0}" == "1" ]]; then
  cmd+=(--cached)
fi

if [[ "${V3_FORCE:-0}" == "1" ]]; then
  cmd+=(--force)
fi

"${cmd[@]}"
