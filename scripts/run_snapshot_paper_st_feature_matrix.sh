#!/usr/bin/env bash
set -euo pipefail

# Paper-facing ST representation matrix.
#
# Main question:
#   Does phase-aligned 10s spatio-temporal telemetry help beyond a basic
#   non-relational architecture, and do observed relation channels add signal
#   beyond temporal-only and graph-null controls?
#
# Core variants:
#   snapshot_last_only        : basic non-temporal, no-graph baseline
#   snapshot_tcn_no_graph     : temporal ST feature baseline without graph input
#   snapshot_rgcn             : observed relation-channel model
#   snapshot_rgcn_edge_shuffle: topology-null relation-channel control
#   snapshot_gate_only        : binary contact-gate/control-capacity model

OUT_DIR="${PAPER_ST_OUT_DIR:-model_stgcn/checkpoints/paper_st_feature_matrix}"
SEED="${PAPER_ST_SEED:-42}"
EPOCHS="${PAPER_ST_EPOCHS:-3}"
BATCH="${PAPER_ST_BATCH:-2}"
HIDDEN="${PAPER_ST_HIDDEN:-16}"
MID="${PAPER_ST_MID:-32}"
NUM_BLOCKS="${PAPER_ST_NUM_BLOCKS:-1}"
EDGE_HIDDEN="${PAPER_ST_EDGE_HIDDEN:-8}"
DEVICE="${PAPER_ST_DEVICE:-cpu}"
LIMIT="${PAPER_ST_LIMIT:-}"
FORCE="${PAPER_ST_FORCE:-0}"
CACHED="${PAPER_ST_CACHED:-0}"
WORKERS="${PAPER_ST_WORKERS:-2}"
EVAL_WORKERS="${PAPER_ST_EVAL_WORKERS:-1}"
PREFETCH_FACTOR="${PAPER_ST_PREFETCH_FACTOR:-2}"
PYTHON_BIN="${PAPER_ST_PYTHON:-python3}"
DRY_RUN="${PAPER_ST_DRY_RUN:-0}"
USE_CAFFEINATE="${PAPER_ST_CAFFEINATE:-1}"
INCLUDE_SPARSE="${PAPER_ST_INCLUDE_SPARSE:-0}"
ONLY="${PAPER_ST_ONLY:-}"

DEFAULT_SPLIT="model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/split_seed${SEED}.json"
if [[ -n "${PAPER_ST_SPLIT_JSON:-}" ]]; then
  SPLIT_JSON="${PAPER_ST_SPLIT_JSON}"
elif [[ -n "${LIMIT}" ]]; then
  SPLIT_JSON="${OUT_DIR}/split_seed${SEED}.json"
elif [[ -f "${DEFAULT_SPLIT}" ]]; then
  SPLIT_JSON="${DEFAULT_SPLIT}"
else
  SPLIT_JSON="${OUT_DIR}/split_seed${SEED}.json"
fi

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
  --split-json "${SPLIT_JSON}"
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

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
}

python_cmd() {
  if [[ "${USE_CAFFEINATE}" == "1" ]] && command -v caffeinate >/dev/null 2>&1; then
    run_cmd caffeinate -dimsu "${PYTHON_BIN}" "$@"
  else
    run_cmd "${PYTHON_BIN}" "$@"
  fi
}

is_selected() {
  local name="$1"
  if [[ -z "${ONLY}" ]]; then
    return 0
  fi
  IFS=',' read -ra selected <<< "${ONLY}"
  for item in "${selected[@]}"; do
    if [[ "${item}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

recover_summary_if_possible() {
  local name="$1"
  local ckpt="${OUT_DIR}/${name}_seed${SEED}.pt"
  local summary="${OUT_DIR}/${name}_seed${SEED}.summary.json"
  if [[ "${FORCE}" != "1" && -f "${ckpt}" && ! -f "${summary}" ]]; then
    echo "RECOVER ${name}: ${ckpt}"
    python_cmd tools/eval_snapshot_phase_checkpoint.py \
      --checkpoint "${ckpt}" \
      --device "${DEVICE}" \
      --eval-workers "${EVAL_WORKERS}" \
      --prefetch-factor "${PREFETCH_FACTOR}"
  fi
}

run_variant() {
  local name="$1"
  shift
  if ! is_selected "${name}"; then
    echo "SKIP ${name}: not selected"
    return
  fi

  local ckpt="${OUT_DIR}/${name}_seed${SEED}.pt"
  local summary="${OUT_DIR}/${name}_seed${SEED}.summary.json"

  recover_summary_if_possible "${name}"
  if [[ "${FORCE}" != "1" && -f "${summary}" ]]; then
    echo "SKIP ${name}: ${summary}"
    return
  fi

  echo "RUN ${name}"
  python_cmd -m model_stgcn.train_snapshot_phase \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --checkpoint "${ckpt}"
}

write_config() {
  "${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
seed = os.environ["SEED"]
config = {
    "purpose": "paper_st_feature_significance_matrix",
    "out_dir": str(out_dir),
    "seed": int(seed),
    "epochs": int(os.environ["EPOCHS"]),
    "batch": int(os.environ["BATCH"]),
    "hidden": int(os.environ["HIDDEN"]),
    "mid": int(os.environ["MID"]),
    "num_blocks": int(os.environ["NUM_BLOCKS"]),
    "edge_hidden": int(os.environ["EDGE_HIDDEN"]),
    "device": os.environ["DEVICE"],
    "limit": os.environ["LIMIT"] or None,
    "split_json": os.environ["SPLIT_JSON"],
    "workers": int(os.environ["WORKERS"]),
    "eval_workers": int(os.environ["EVAL_WORKERS"]),
    "prefetch_factor": int(os.environ["PREFETCH_FACTOR"]),
    "core_variants": [
        "snapshot_last_only",
        "snapshot_tcn_no_graph",
        "snapshot_rgcn",
        "snapshot_rgcn_edge_shuffle",
        "snapshot_gate_only",
    ],
    "include_sparse": os.environ["INCLUDE_SPARSE"] == "1",
}
(out_dir / f"paper_st_config_seed{seed}.json").write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)
PY
}

write_metrics() {
  "${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
seed = os.environ["SEED"]
variants = [
    "snapshot_last_only",
    "snapshot_tcn_no_graph",
    "snapshot_rgcn",
    "snapshot_rgcn_edge_shuffle",
    "snapshot_gate_only",
]
if os.environ["INCLUDE_SPARSE"] == "1":
    variants.extend(
        [
            "snapshot_relative_edge_sparse",
            "snapshot_relative_edge_sparse_null",
            "snapshot_relative_edge_sparse_topology_null",
        ]
    )

rows: list[dict[str, object]] = []
for variant in variants:
    summary_path = out_dir / f"{variant}_seed{seed}.summary.json"
    checkpoint_path = out_dir / f"{variant}_seed{seed}.pt"
    if not summary_path.exists():
        rows.append(
            {
                "variant": variant,
                "state": "missing_summary" if checkpoint_path.exists() else "not_started",
                "summary": "",
                "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else "",
                "best_val": "",
                "test_loss": "",
                "test_acc": "",
                "relation_handling": "",
                "tensor_owners": "",
                "input_channels": "",
                "graph_resolution": "",
                "relative_edge_null": "",
            }
        )
        continue

    data = json.loads(summary_path.read_text())
    metrics = data.get("test_metrics", {}) or {}
    contract = data.get("channel_contract", {}) or {}
    rows.append(
        {
            "variant": data.get("model_variant", variant),
            "state": "summary_complete",
            "summary": str(summary_path),
            "checkpoint": str(checkpoint_path),
            "best_val": data.get("best_val"),
            "test_loss": metrics.get("loss"),
            "test_acc": metrics.get("hazard_acc"),
            "relation_handling": contract.get("relation_handling"),
            "tensor_owners": "+".join(contract.get("tensor_owners_consumed", [])),
            "input_channels": "+".join(contract.get("input_channels_consumed", [])),
            "graph_resolution": contract.get("graph_resolution"),
            "relative_edge_null": contract.get("relative_edge_null"),
        }
    )

fieldnames = list(rows[0].keys()) if rows else ["variant"]
csv_path = out_dir / f"paper_st_metrics_seed{seed}.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
(out_dir / f"paper_st_status_seed{seed}.json").write_text(
    json.dumps(rows, indent=2),
    encoding="utf-8",
)
print(f"wrote {csv_path}")
PY
}

export OUT_DIR SEED EPOCHS BATCH HIDDEN MID NUM_BLOCKS EDGE_HIDDEN DEVICE LIMIT
export SPLIT_JSON WORKERS EVAL_WORKERS PREFETCH_FACTOR INCLUDE_SPARSE

write_config

run_variant snapshot_last_only \
  --model-variant snapshot_last_only

run_variant snapshot_tcn_no_graph \
  --model-variant snapshot_tcn_no_graph

run_variant snapshot_rgcn \
  --model-variant snapshot_rgcn

run_variant snapshot_rgcn_edge_shuffle \
  --model-variant snapshot_rgcn_edge_shuffle

run_variant snapshot_gate_only \
  --model-variant snapshot_gate_only

if [[ "${INCLUDE_SPARSE}" == "1" ]]; then
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
fi

write_metrics
