#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${SNAP_OUT_DIR:-model_stgcn/checkpoints/snapshot_narrow_full}"
SEED="${SNAP_SEED:-42}"
EPOCHS="${SNAP_EPOCHS:-12}"
BATCH="${SNAP_BATCH:-2}"
HIDDEN="${SNAP_HIDDEN:-16}"
MID="${SNAP_MID:-32}"
NUM_BLOCKS="${SNAP_NUM_BLOCKS:-1}"
EDGE_HIDDEN="${SNAP_EDGE_HIDDEN:-8}"
DEVICE="${SNAP_DEVICE:-cpu}"
LIMIT="${SNAP_LIMIT:-}"
FORCE="${SNAP_FORCE:-0}"
CACHED="${SNAP_CACHED:-0}"
WORKERS="${SNAP_WORKERS:-0}"
EVAL_WORKERS="${SNAP_EVAL_WORKERS:-0}"
PREFETCH_FACTOR="${SNAP_PREFETCH_FACTOR:-2}"

mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --epochs "${EPOCHS}"
  --batch "${BATCH}"
  --hidden "${HIDDEN}"
  --mid "${MID}"
  --num-blocks "${NUM_BLOCKS}"
  --edge-hidden "${EDGE_HIDDEN}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --split-json "${OUT_DIR}/split_seed${SEED}.json"
  --workers "${WORKERS}"
  --eval-workers "${EVAL_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
)

if [[ -n "${LIMIT}" ]]; then
  COMMON_ARGS+=(--limit "${LIMIT}")
fi

if [[ "${CACHED}" == "1" ]]; then
  COMMON_ARGS+=(--cached)
fi

run_variant() {
  local name="$1"
  shift
  local ckpt="${OUT_DIR}/${name}_seed${SEED}.pt"
  local summary="${OUT_DIR}/${name}_seed${SEED}.summary.json"
  if [[ "${FORCE}" != "1" && -f "${summary}" ]]; then
    echo "SKIP ${name}: ${summary}"
    return
  fi
  echo "RUN ${name}"
  python3 -m model_stgcn.train_snapshot_phase \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --checkpoint "${ckpt}"
}

run_variant snapshot_last_only \
  --model-variant snapshot_last_only

run_variant snapshot_rgcn \
  --model-variant snapshot_rgcn

run_variant snapshot_rgcn_edge_shuffle \
  --model-variant snapshot_rgcn_edge_shuffle

run_variant snapshot_gate_only \
  --model-variant snapshot_gate_only

run_variant snapshot_relative_edge_sparse \
  --model-variant snapshot_relative_edge_sparse \
  --relative-edge-gate knn_combat \
  --relative-edge-k 5

run_variant snapshot_relative_edge_sparse_null \
  --model-variant snapshot_relative_edge_sparse_null \
  --relative-edge-gate knn_combat \
  --relative-edge-k 5

run_variant snapshot_relative_edge_sparse_topology_null \
  --model-variant snapshot_relative_edge_sparse_topology_null \
  --relative-edge-gate knn_combat \
  --relative-edge-k 5

python3 - <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

out_dir = Path(os.environ.get("SNAP_OUT_DIR", "model_stgcn/checkpoints/snapshot_narrow_full"))
seed = os.environ.get("SNAP_SEED", "42")
rows = []
for path in sorted(out_dir.glob(f"*_seed{seed}.summary.json")):
    data = json.loads(path.read_text())
    metrics = data.get("test_metrics", {})
    contract = data.get("channel_contract", {})
    rows.append(
        {
            "variant": data.get("model_variant"),
            "summary": str(path),
            "best_val": data.get("best_val"),
            "test_loss": metrics.get("loss"),
            "test_acc": metrics.get("hazard_acc"),
            "relation_handling": contract.get("relation_handling"),
            "tensor_owners": "+".join(contract.get("tensor_owners_consumed", [])),
            "relative_edge_null": contract.get("relative_edge_null"),
        }
    )

csv_path = out_dir / f"matrix_metrics_seed{seed}.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["variant"])
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {csv_path}")
PY
