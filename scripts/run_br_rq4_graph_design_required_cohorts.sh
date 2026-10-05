#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

RUN_ROOT="${BR_RQ4_GRAPH_ROOT:-model_stgcn/checkpoints/br_rq4_graph_design_required_v1}"
OM_ROOT="${BR_RQ4_OM_ROOT:-model_stgcn/checkpoints/br_om_ladder_required_v1}"
EPOCHS="${BR_RQ4_GRAPH_EPOCHS:-12}"
BATCH="${BR_RQ4_GRAPH_BATCH:-8}"
WORKERS="${BR_RQ4_GRAPH_WORKERS:-4}"
DEVICE="${BR_RQ4_GRAPH_DEVICE:-mps}"
SEED="${BR_RQ4_GRAPH_SEED:-42}"
LOG_EVERY="${BR_RQ4_GRAPH_LOG_EVERY:-250}"
PYTHON_BIN="${BR_RQ4_GRAPH_PYTHON:-python3}"

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

if [[ -n "${BR_RQ4_GRAPH_COHORT:-}" ]]; then
  COHORTS=("${BR_RQ4_GRAPH_COHORT}")
elif [[ -n "${BR_RQ4_GRAPH_COHORTS:-}" ]]; then
  IFS="|" read -r -a COHORTS <<< "${BR_RQ4_GRAPH_COHORTS}"
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
  if [[ -n "${BR_RQ4_GRAPH_LIMIT:-}" ]]; then
    split_json="${out_dir}/split_limit_${BR_RQ4_GRAPH_LIMIT}.json"
  fi
  if [[ ! -f "${split_json}" ]]; then
    if [[ -n "${BR_RQ4_GRAPH_LIMIT:-}" ]]; then
      echo "creating smoke split: ${split_json}"
    else
      echo "missing split_json: ${split_json}" >&2
      exit 1
    fi
  fi

  started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  status="failed"
  cmd=(
    "${PYTHON_BIN}" -m model_stgcn.v3.run_matrix
    --groups graph_relation
    --cohort "${cohort}"
    --split-json "${split_json}"
    --out-dir "${out_dir}"
    --epochs "${EPOCHS}"
    --batch "${BATCH}"
    --workers "${WORKERS}"
    --log-every "${LOG_EVERY}"
    --device "${DEVICE}"
    --seed "${SEED}"
  )
  if [[ -n "${BR_RQ4_GRAPH_LIMIT:-}" ]]; then
    cmd+=(--limit "${BR_RQ4_GRAPH_LIMIT}")
  fi
  if [[ -n "${BR_RQ4_GRAPH_ONLY:-}" ]]; then
    cmd+=(--only "${BR_RQ4_GRAPH_ONLY}")
  fi
  if [[ "${BR_RQ4_GRAPH_CACHED:-0}" == "1" ]]; then
    cmd+=(--cached)
  fi
  if [[ "${BR_RQ4_GRAPH_FORCE:-0}" == "1" ]]; then
    cmd+=(--force)
  fi
  if [[ "${BR_RQ4_GRAPH_FORCE_EVAL:-0}" == "1" ]]; then
    cmd+=(--force-eval)
  fi
  if [[ "${BR_RQ4_GRAPH_EVAL_ONLY:-0}" == "1" ]]; then
    cmd+=(--eval-only)
  fi

  echo "== RQ4 graph design: ${cohort} -> ${out_dir} =="
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

"${PYTHON_BIN}" tools/collect_rq4_graph_design_metrics.py --root "${RUN_ROOT}"
