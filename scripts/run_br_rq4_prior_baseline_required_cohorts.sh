#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

RUN_ROOT="${BR_RQ4_BASELINE_ROOT:-model_stgcn/checkpoints/br_rq4_prior_baseline_required_v1}"
OM_ROOT="${BR_RQ4_OM_ROOT:-model_stgcn/checkpoints/br_om_ladder_required_v1}"
FEATURE_VIEW="${BR_RQ4_FEATURE_VIEW:-stdb.v_team_phase_feature_v2}"
CONFIG_STYLE="${BR_RQ4_CONFIG_STYLE:-both}"
PYTHON_BIN="${BR_RQ4_PYTHON:-python3}"
LEARNERS="${BR_RQ4_LEARNERS:-hgb}"
MAX_ITER="${BR_RQ4_MAX_ITER:-200}"
N_ESTIMATORS="${BR_RQ4_N_ESTIMATORS:-300}"
N_JOBS="${BR_RQ4_N_JOBS:-0}"

COHORTS=(
  "Erangel TPP squad"
  "Taego TPP squad"
  "Miramar / Desert TPP squad"
  "Rondo TPP squad"
  "Sanhok TPP squad"
  "Erangel TPP solo"
  "Erangel TPP duo"
  "Taego TPP duo"
  "Vikendi TPP squad"
  "Taego FPP squad"
  "Erangel FPP squad"
  "Paramo TPP squad"
  "Karakin TPP squad"
)

if [[ -n "${BR_RQ4_COHORT:-}" ]]; then
  COHORTS=("${BR_RQ4_COHORT}")
elif [[ -n "${BR_RQ4_COHORTS:-}" ]]; then
  IFS="|" read -r -a COHORTS <<< "${BR_RQ4_COHORTS}"
fi

mkdir -p "${RUN_ROOT}"
RUN_INDEX="${RUN_ROOT}/run_index.tsv"
if [[ ! -f "${RUN_INDEX}" ]]; then
  printf "started_at\tfinished_at\tstatus\tcohort_slug\tcohort\tout_dir\n" > "${RUN_INDEX}"
fi

cohort_slug() {
  "${PYTHON_BIN}" -c 'from model_stgcn.v3.cohorts import resolve_cohort; import sys; print(resolve_cohort(sys.argv[1]).slug)' "$1"
}

for cohort in "${COHORTS[@]}"; do
  slug="$(cohort_slug "${cohort}")"
  out_dir="${RUN_ROOT}/${slug}"
  split_json="${OM_ROOT}/${slug}/split.json"
  log_file="${out_dir}/run.log"
  mkdir -p "${out_dir}"
  if [[ -n "${BR_RQ4_LIMIT:-}" ]]; then
    split_json="${out_dir}/split_limit_${BR_RQ4_LIMIT}.json"
  fi
  if [[ ! -f "${split_json}" ]]; then
    if [[ -n "${BR_RQ4_LIMIT:-}" ]]; then
      echo "creating smoke split: ${split_json}"
    else
      echo "missing split_json: ${split_json}" >&2
      exit 1
    fi
  fi

  started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  status="failed"
  cmd=(
    "${PYTHON_BIN}" -m model_stgcn.feature_ablation
    --cohort "${cohort}"
    --split-json "${split_json}"
    --align-to ""
    --out-dir "${out_dir}"
    --feature-view "${FEATURE_VIEW}"
    --config-style "${CONFIG_STYLE}"
    --learners "${LEARNERS}"
    --max-iter "${MAX_ITER}"
    --n-estimators "${N_ESTIMATORS}"
    --n-jobs "${N_JOBS}"
  )
  if [[ -n "${BR_RQ4_LIMIT:-}" ]]; then
    cmd+=(--limit "${BR_RQ4_LIMIT}")
  fi
  if [[ -n "${BR_RQ4_CONFIGS:-}" ]]; then
    cmd+=(--configs "${BR_RQ4_CONFIGS}")
  fi
  if [[ "${BR_RQ4_SKIP_NETWORK:-0}" == "1" ]]; then
    cmd+=(--skip-network)
  fi
  if [[ "${BR_RQ4_SKIP_HABITAT:-0}" == "1" ]]; then
    cmd+=(--skip-habitat)
  fi

  echo "== RQ4 prior baseline: ${cohort} -> ${out_dir} =="
  echo "+ ${cmd[*]}" > "${log_file}"
  if "${cmd[@]}" >> "${log_file}" 2>&1; then
    status="passed"
    echo "passed: ${cohort} log=${log_file}"
  else
    echo "failed: ${cohort} log=${log_file}" >&2
    tail -80 "${log_file}" >&2
  fi
  finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${started_at}" "${finished_at}" "${status}" "${slug}" "${cohort}" "${out_dir}" >> "${RUN_INDEX}"
  if [[ "${status}" != "passed" ]]; then
    exit 1
  fi
done

"${PYTHON_BIN}" tools/collect_rq4_baseline_metrics.py --root "${RUN_ROOT}"
