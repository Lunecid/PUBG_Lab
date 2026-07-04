#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

RUN_ROOT="${BR_OM6Z_ROOT:-model_stgcn/checkpoints/br_om6z_required_v1}"
SPLIT_ROOT="${BR_OM6Z_SPLIT_ROOT:-model_stgcn/checkpoints/br_om_ladder_required_v1}"
EPOCHS="${BR_OM6Z_EPOCHS:-12}"
BATCH="${BR_OM6Z_BATCH:-8}"
WORKERS="${BR_OM6Z_WORKERS:-4}"
DEVICE="${BR_OM6Z_DEVICE:-mps}"
SEED="${BR_OM6Z_SEED:-42}"
LOG_EVERY="${BR_OM6Z_LOG_EVERY:-250}"
EVAL_SPLIT="${BR_OM6Z_EVAL_SPLIT:-test}"
PYTHON_BIN="${BR_OM6Z_PYTHON:-python3}"

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

if [[ -n "${BR_OM6Z_COHORT:-}" ]]; then
  COHORTS=("${BR_OM6Z_COHORT}")
elif [[ -n "${BR_OM6Z_COHORTS:-}" ]]; then
  IFS="|" read -r -a COHORTS <<< "${BR_OM6Z_COHORTS}"
fi

mkdir -p "${RUN_ROOT}"
RUN_INDEX="${RUN_ROOT}/run_index.tsv"
if [[ ! -f "${RUN_INDEX}" ]]; then
  printf "started_at\tfinished_at\tstatus\tcohort_slug\tcohort\tout_dir\tsplit_json\n" > "${RUN_INDEX}"
fi

cohort_slug() {
  "${PYTHON_BIN}" -c 'from model_stgcn.v3.cohorts import resolve_cohort; import sys; print(resolve_cohort(sys.argv[1]).slug)' "$1"
}

for cohort in "${COHORTS[@]}"; do
  slug="$(cohort_slug "${cohort}")"
  out_dir="${RUN_ROOT}/${slug}"
  split_json="${SPLIT_ROOT}/${slug}/split.json"
  if [[ ! -f "${split_json}" ]]; then
    echo "missing split json for ${cohort}: ${split_json}" >&2
    exit 1
  fi

  mkdir -p "${out_dir}"
  log_file="${out_dir}/run.log"
  started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  status="failed"
  cmd=(
    "${PYTHON_BIN}" -m model_stgcn.v3.run_matrix
    --groups om6z_sensitivity
    --cohort "${cohort}"
    --out-dir "${out_dir}"
    --split-json "${split_json}"
    --epochs "${EPOCHS}"
    --batch "${BATCH}"
    --workers "${WORKERS}"
    --log-every "${LOG_EVERY}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --eval-split "${EVAL_SPLIT}"
  )
  if [[ -n "${BR_OM6Z_LIMIT:-}" ]]; then
    cmd+=(--limit "${BR_OM6Z_LIMIT}")
  fi
  if [[ "${BR_OM6Z_CACHED:-0}" == "1" ]]; then
    cmd+=(--cached)
  fi
  if [[ "${BR_OM6Z_FORCE:-0}" == "1" ]]; then
    cmd+=(--force)
  fi
  if [[ "${BR_OM6Z_FORCE_EVAL:-0}" == "1" ]]; then
    cmd+=(--force-eval)
  fi
  if [[ "${BR_OM6Z_EVAL_ONLY:-0}" == "1" ]]; then
    cmd+=(--eval-only)
  fi
  echo "== OM6Z cohort: ${cohort} -> ${out_dir} =="
  echo "+ ${cmd[*]}" > "${log_file}"
  if "${cmd[@]}" >> "${log_file}" 2>&1; then
    status="passed"
    echo "passed: ${cohort} log=${log_file}"
  else
    echo "failed: ${cohort} log=${log_file}" >&2
    tail -80 "${log_file}" >&2
  fi
  finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${started_at}" "${finished_at}" "${status}" "${slug}" "${cohort}" "${out_dir}" "${split_json}" >> "${RUN_INDEX}"
  if [[ "${status}" != "passed" ]]; then
    exit 1
  fi
done

"${PYTHON_BIN}" tools/collect_om_ladder_metrics.py \
  --root "${RUN_ROOT}" \
  --out "${RUN_ROOT}/om6z_metrics_all_cohorts.csv" \
  --summary-out "${RUN_ROOT}/om6z_metrics_summary.json"
