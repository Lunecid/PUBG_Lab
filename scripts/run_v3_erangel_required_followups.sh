#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

COHORT="${V3_COHORT:-Erangel TPP squad}"
SPLIT_JSON="${V3_SPLIT_JSON:-model_stgcn/checkpoints/br_om_ladder_required_v1/erangel_tpp_squad/split.json}"
ROOT="${V3_OUT_ROOT:-model_stgcn/checkpoints/followup_erangel_required_v1}"
EPOCHS="${V3_EPOCHS:-12}"
BATCH="${V3_BATCH:-8}"
WORKERS="${V3_WORKERS:-4}"
DEVICE="${V3_DEVICE:-mps}"
LOG_EVERY="${V3_LOG_EVERY:-250}"
CACHED="${V3_CACHED:-1}"
FORCE="${V3_FORCE:-0}"

run_matrix_common=(
  python3 -m model_stgcn.v3.run_matrix
  --cohort "${COHORT}"
  --split-json "${SPLIT_JSON}"
  --require-existing-split
  --epochs "${EPOCHS}"
  --batch "${BATCH}"
  --workers "${WORKERS}"
  --log-every "${LOG_EVERY}"
  --device "${DEVICE}"
)

if [[ "${CACHED}" == "1" ]]; then
  run_matrix_common+=(--cached)
fi
if [[ "${FORCE}" == "1" ]]; then
  run_matrix_common+=(--force)
fi

echo "== follow-up 1/2: V3 feature-preset comparison (${COHORT}) =="
"${run_matrix_common[@]}" \
  --groups feature \
  --out-dir "${ROOT}/feature_preset"

echo "== follow-up 2/2: V3 seed repeats for core representative models (${COHORT}) =="
SEEDS="${V3_SEEDS:-13 2026}"
ONLY="${V3_ONLY:-mr_tcn,mr_farnet,mr_st_gnn_mamba}"
for seed in ${SEEDS}; do
  echo "== seed repeat seed=${seed} =="
  "${run_matrix_common[@]}" \
    --groups main \
    --only "${ONLY}" \
    --seed "${seed}" \
    --out-dir "${ROOT}/seed_repeats/seed${seed}"
done

echo "== follow-ups finished =="
