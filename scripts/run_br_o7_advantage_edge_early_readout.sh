#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "${O7_TRAIN_ONLY:-0}" != "1" || "${O7_SKIP_AUDIT:-0}" != "1" ]]; then
  : "${PG_PASSWORD:?PG_PASSWORD must be set}"
fi

RUN_ROOT="${O7_ROOT:-model_stgcn/checkpoints/br_o7_advantage_edge_required_v1}"
OM_ROOT="${O7_OM_ROOT:-model_stgcn/checkpoints/br_om_ladder_required_v1}"
PYTHON_BIN="${O7_PYTHON:-python3}"
DEVICE="${O7_DEVICE:-mps}"
EPOCHS="${O7_EPOCHS:-12}"
BATCH="${O7_BATCH:-64}"
SEED="${O7_SEED:-42}"
MAX_TRAIN="${O7_MAX_TRAIN:-8000}"
MAX_VAL="${O7_MAX_VAL:-1521}"
MAX_TEST="${O7_MAX_TEST:-1521}"
FEATURE_PRESET="${O7_FEATURE_PRESET:-om5_relation_structure}"
GATES="${O7_GATES:-knn}"

COHORTS=(
  "Erangel TPP squad"
  "Taego TPP squad"
  "Erangel TPP solo"
  "Miramar / Desert TPP squad"
)

if [[ -n "${O7_COHORT:-}" ]]; then
  COHORTS=("${O7_COHORT}")
elif [[ -n "${O7_COHORTS:-}" ]]; then
  IFS="|" read -r -a COHORTS <<< "${O7_COHORTS}"
fi

mkdir -p "${RUN_ROOT}"
RUN_INDEX="${RUN_ROOT}/early_readout_run_index.tsv"
if [[ ! -f "${RUN_INDEX}" ]]; then
  printf "started_at\tfinished_at\tstatus\tcohort_slug\tcohort\tout_dir\n" > "${RUN_INDEX}"
fi

cohort_slug() {
  "${PYTHON_BIN}" -c 'from model_stgcn.v3.cohorts import resolve_cohort; import sys; print(resolve_cohort(sys.argv[1]).slug)' "$1"
}

for cohort in "${COHORTS[@]}"; do
  slug="$(cohort_slug "${cohort}")"
  out_dir="${RUN_ROOT}/${slug}"
  cache_dir="${out_dir}/adv_cache"
  split_json="${OM_ROOT}/${slug}/split.json"
  mkdir -p "${out_dir}"
  log_file="${out_dir}/o7_early_readout.log"
  started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  status="failed"
  echo "== O7 advantage-edge early readout: ${cohort} -> ${out_dir} =="
  {
    echo "started_at=${started_at}"
    echo "cohort=${cohort}"
    echo "split_json=${split_json}"
    echo "feature_preset=${FEATURE_PRESET}"
    echo "max_train=${MAX_TRAIN} max_val=${MAX_VAL} max_test=${MAX_TEST}"
  } > "${log_file}"

  if [[ "${O7_SKIP_AUDIT:-0}" != "1" ]]; then
    "${PYTHON_BIN}" tools/audit_o7_advantage_edge_shuffle.py \
      --cohort "${cohort}" \
      --split-json "${split_json}" \
      --out "${out_dir}/o7_edge_shuffle_audit.json" \
      --max-matches "${O7_AUDIT_MATCHES:-8}" \
      >> "${log_file}" 2>&1
  fi

  if [[ "${O7_TRAIN_ONLY:-0}" != "1" ]]; then
    precompute_cmd=(
      "${PYTHON_BIN}" scripts/adv_precompute.py
      --cohort "${cohort}"
      --split-json "${split_json}"
      --out-dir "${cache_dir}"
      --feature-preset "${FEATURE_PRESET}"
      --max-train "${MAX_TRAIN}"
      --max-val "${MAX_VAL}"
      --max-test "${MAX_TEST}"
      --gates "${GATES}"
    )
    if [[ "${O7_NO_NODE_ABS:-0}" == "1" ]]; then
      precompute_cmd+=(--no-node-abs)
    fi
    "${precompute_cmd[@]}" >> "${log_file}" 2>&1
  fi

  if [[ "${O7_CACHE_ONLY:-0}" != "1" ]]; then
    "${PYTHON_BIN}" -m model_stgcn.v3.adv_model \
      --cache-dir "${cache_dir}" \
      --gate none \
      --control observed \
      --epochs "${EPOCHS}" \
      --batch "${BATCH}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --out "${out_dir}/A_none_observed" \
      >> "${log_file}" 2>&1
    "${PYTHON_BIN}" -m model_stgcn.v3.adv_model \
      --cache-dir "${cache_dir}" \
      --gate knn \
      --control observed \
      --epochs "${EPOCHS}" \
      --batch "${BATCH}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --out "${out_dir}/B_knn_observed" \
      >> "${log_file}" 2>&1
    "${PYTHON_BIN}" -m model_stgcn.v3.adv_model \
      --cache-dir "${cache_dir}" \
      --gate knn \
      --control edge_shuffle \
      --epochs "${EPOCHS}" \
      --batch "${BATCH}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --out "${out_dir}/C_knn_edge_shuffle" \
      >> "${log_file}" 2>&1
  fi

  status="passed"
  finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${started_at}" "${finished_at}" "${status}" "${slug}" "${cohort}" "${out_dir}" >> "${RUN_INDEX}"
  echo "passed: ${cohort} log=${log_file}"
done
