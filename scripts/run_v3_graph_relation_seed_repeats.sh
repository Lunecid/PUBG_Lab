#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

ONLY="${V3_ONLY:-graph_core_habitat_no_graph,graph_core_habitat_proximity,graph_core_habitat_proximity_combat,graph_core_habitat_full_observed,graph_core_habitat_full_edge_shuffle}"
OUT_ROOT="${V3_OUT_ROOT:-model_stgcn/checkpoints/v3_graph_relation_seed_repeats}"
SPLIT_JSON="${V3_SPLIT_JSON:-model_stgcn/checkpoints/v3_full_main_causal/split.json}"
SEEDS="${V3_SEEDS:-13 2026}"

for seed in ${SEEDS}; do
  echo "== graph relation seed repeat seed=${seed} started $(date) =="
  cmd=(
    python3 -m model_stgcn.v3.run_matrix
    --groups graph_relation
    --only "${ONLY}"
    --out-dir "${OUT_ROOT}/seed${seed}"
    --split-json "${SPLIT_JSON}"
    --mode "${V3_MODE:-squad}"
    --epochs "${V3_EPOCHS:-12}"
    --batch "${V3_BATCH:-8}"
    --workers "${V3_WORKERS:-4}"
    --log-every "${V3_LOG_EVERY:-250}"
    --device "${V3_DEVICE:-mps}"
    --seed "${seed}"
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

  if [[ "${V3_FORCE_EVAL:-0}" == "1" ]]; then
    cmd+=(--force-eval)
  fi

  "${cmd[@]}"
  echo "== graph relation seed repeat seed=${seed} finished $(date) =="
done

echo "== graph relation seed repeats finished $(date) =="
