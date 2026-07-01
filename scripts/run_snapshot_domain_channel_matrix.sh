#!/usr/bin/env bash
set -euo pipefail

# Map/mode-sliced channel matrix for the prior-aligned Battleground ST feature
# design. This is intentionally separate from the pooled paper architecture
# runner because the research question here is:
#
#   Which ST feature channel matters in which Battleground domain?

OUT_DIR="${SNAP_DOMAIN_OUT_DIR:-model_stgcn/checkpoints/snapshot_domain_channel_matrix}"
PRESET="${SNAP_DOMAIN_PRESET:-primary_tpp}"
SEED="${SNAP_DOMAIN_SEED:-42}"
LIMIT="${SNAP_DOMAIN_LIMIT:-300}"
EPOCHS="${SNAP_DOMAIN_EPOCHS:-1}"
BATCH="${SNAP_DOMAIN_BATCH:-8}"
HIDDEN="${SNAP_DOMAIN_HIDDEN:-16}"
MID="${SNAP_DOMAIN_MID:-32}"
NUM_BLOCKS="${SNAP_DOMAIN_NUM_BLOCKS:-1}"
EDGE_HIDDEN="${SNAP_DOMAIN_EDGE_HIDDEN:-8}"
DEVICE="${SNAP_DOMAIN_DEVICE:-mps}"
WORKERS="${SNAP_DOMAIN_WORKERS:-4}"
EVAL_WORKERS="${SNAP_DOMAIN_EVAL_WORKERS:-2}"
PREFETCH_FACTOR="${SNAP_DOMAIN_PREFETCH_FACTOR:-2}"
PYTHON_BIN="${SNAP_DOMAIN_PYTHON:-python3}"
FORCE="${SNAP_DOMAIN_FORCE:-0}"
DRY_RUN="${SNAP_DOMAIN_DRY_RUN:-0}"
USE_CAFFEINATE="${SNAP_DOMAIN_CAFFEINATE:-1}"
ONLY="${SNAP_DOMAIN_ONLY:-}"
MIN_MATCHES="${SNAP_DOMAIN_MIN_MATCHES:-50}"

mkdir -p "${OUT_DIR}"

if [[ "${PRESET}" == "primary_tpp" ]]; then
  DOMAINS=(
    "Baltic_Main squad"
    "Tiger_Main squad"
    "Desert_Main squad"
    "Neon_Main squad"
    "Savage_Main squad"
  )
elif [[ "${PRESET}" == "fpp_aux" ]]; then
  DOMAINS=(
    "Baltic_Main squad-fpp"
    "Tiger_Main squad-fpp"
    "Desert_Main squad-fpp"
    "Neon_Main squad-fpp"
  )
elif [[ -n "${SNAP_DOMAIN_MAP:-}" && -n "${SNAP_DOMAIN_MODE:-}" ]]; then
  DOMAINS=("${SNAP_DOMAIN_MAP} ${SNAP_DOMAIN_MODE}")
else
  echo "Unknown SNAP_DOMAIN_PRESET=${PRESET}; use primary_tpp, fpp_aux, or SNAP_DOMAIN_MAP/SNAP_DOMAIN_MODE" >&2
  exit 2
fi

VARIANTS=(
  "last_full snapshot_last_only full full"
  "tcn_trajectory snapshot_tcn_no_graph trajectory none"
  "tcn_zone snapshot_tcn_no_graph zone none"
  "tcn_viability_resource snapshot_tcn_no_graph viability_resource none"
  "tcn_trajectory_zone snapshot_tcn_no_graph trajectory_zone none"
  "tcn_full snapshot_tcn_no_graph full none"
  "rgcn_proximity snapshot_rgcn full proximity"
  "rgcn_combat snapshot_rgcn full combat"
  "rgcn_full snapshot_rgcn full full"
  "rgcn_full_edge_shuffle snapshot_rgcn_edge_shuffle full full"
  "gate_only_full snapshot_gate_only full full"
)

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

slug() {
  echo "$1__$2" | tr '/' '_'
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

count_domain() {
  local map_name="$1"
  local mode="$2"
  PG_PASSWORD="${PG_PASSWORD:-}" "${PYTHON_BIN}" - "$map_name" "$mode" <<'PY'
from stdb.queries import list_phase_eligible_match_ids
import sys

print(len(list_phase_eligible_match_ids(sys.argv[1], sys.argv[2], None)))
PY
}

write_config() {
  OUT_DIR="${OUT_DIR}" PRESET="${PRESET}" SEED="${SEED}" LIMIT="${LIMIT}" EPOCHS="${EPOCHS}" \
  BATCH="${BATCH}" DEVICE="${DEVICE}" MIN_MATCHES="${MIN_MATCHES}" \
  "${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
config = {
    "purpose": "map_mode_channel_matrix",
    "preset": os.environ["PRESET"],
    "seed": int(os.environ["SEED"]),
    "limit_per_domain": int(os.environ["LIMIT"]) if os.environ["LIMIT"] else None,
    "epochs": int(os.environ["EPOCHS"]),
    "batch": int(os.environ["BATCH"]),
    "device": os.environ["DEVICE"],
    "min_matches": int(os.environ["MIN_MATCHES"]),
}
(out_dir / f"domain_channel_config_seed{os.environ['SEED']}.json").write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)
PY
}

write_metrics() {
  OUT_DIR="${OUT_DIR}" SEED="${SEED}" "${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
rows: list[dict[str, object]] = []
for path in sorted(out_dir.glob("*/*_seed*.summary.json")):
    data = json.loads(path.read_text())
    metrics = data.get("test_metrics", {}) or {}
    contract = data.get("channel_contract", {}) or {}
    rows.append(
        {
            "domain": path.parent.name,
            "map": data.get("map"),
            "mode": data.get("mode"),
            "variant_name": path.name.rsplit("_seed", 1)[0],
            "model_variant": data.get("model_variant"),
            "feature_preset": data.get("snapshot_feature_preset"),
            "edge_preset": data.get("snapshot_edge_preset"),
            "best_val": data.get("best_val"),
            "test_loss": metrics.get("loss"),
            "test_acc": metrics.get("hazard_acc"),
            "tensor_owners": "+".join(contract.get("tensor_owners_consumed", [])),
            "input_channels": "+".join(contract.get("input_channels_consumed", [])),
            "domain_mechanisms": "+".join(contract.get("domain_mechanisms_consumed", [])),
            "summary": str(path),
        }
    )
csv_path = out_dir / f"domain_channel_metrics_seed{os.environ['SEED']}.csv"
if rows:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
else:
    csv_path.write_text("domain,map,mode,variant_name\n", encoding="utf-8")
(out_dir / f"domain_channel_status_seed{os.environ['SEED']}.json").write_text(
    json.dumps(rows, indent=2),
    encoding="utf-8",
)
print(f"wrote {csv_path}")
PY
}

write_config

for domain in "${DOMAINS[@]}"; do
  read -r MAP_NAME MODE <<<"${domain}"
  N_MATCHES="$(count_domain "${MAP_NAME}" "${MODE}")"
  if (( N_MATCHES < MIN_MATCHES )); then
    echo "SKIP ${MAP_NAME}/${MODE}: n=${N_MATCHES} < min_matches=${MIN_MATCHES}"
    continue
  fi

  DOMAIN_SLUG="$(slug "${MAP_NAME}" "${MODE}")"
  DOMAIN_DIR="${OUT_DIR}/${DOMAIN_SLUG}"
  mkdir -p "${DOMAIN_DIR}"
  SPLIT_JSON="${DOMAIN_DIR}/split_seed${SEED}.json"
  echo "== ${MAP_NAME}/${MODE}: n=${N_MATCHES}, limit=${LIMIT}, split=${SPLIT_JSON} =="

  for row in "${VARIANTS[@]}"; do
    read -r VARIANT_NAME MODEL_VARIANT FEATURE_PRESET EDGE_PRESET <<<"${row}"
    if ! is_selected "${VARIANT_NAME}"; then
      echo "SKIP ${DOMAIN_SLUG}/${VARIANT_NAME}: not selected"
      continue
    fi

    CKPT="${DOMAIN_DIR}/${VARIANT_NAME}_seed${SEED}.pt"
    SUMMARY="${DOMAIN_DIR}/${VARIANT_NAME}_seed${SEED}.summary.json"
    if [[ "${FORCE}" != "1" && -f "${SUMMARY}" ]]; then
      echo "SKIP ${DOMAIN_SLUG}/${VARIANT_NAME}: ${SUMMARY}"
      continue
    fi

    echo "RUN ${DOMAIN_SLUG}/${VARIANT_NAME}"
    ARGS=(
      -m model_stgcn.train_snapshot_phase
      --map "${MAP_NAME}"
      --mode "${MODE}"
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
      --model-variant "${MODEL_VARIANT}"
      --snapshot-feature-preset "${FEATURE_PRESET}"
      --snapshot-edge-preset "${EDGE_PRESET}"
      --checkpoint "${CKPT}"
    )
    if [[ -n "${LIMIT}" ]]; then
      ARGS+=(--limit "${LIMIT}")
    fi
    python_cmd "${ARGS[@]}"
  done
done

write_metrics
