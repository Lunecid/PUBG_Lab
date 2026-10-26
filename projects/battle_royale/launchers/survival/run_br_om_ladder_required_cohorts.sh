#!/usr/bin/env bash
set -euo pipefail

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

: "${PG_PASSWORD:?PG_PASSWORD must be set}"

RUN_ROOT="${BR_OM_ROOT:-pubg_lab.survival/checkpoints/br_om_ladder_required_v1}"
EPOCHS="${BR_OM_EPOCHS:-12}"
BATCH="${BR_OM_BATCH:-8}"
WORKERS="${BR_OM_WORKERS:-4}"
DEVICE="${BR_OM_DEVICE:-mps}"
SEED="${BR_OM_SEED:-42}"
LOG_EVERY="${BR_OM_LOG_EVERY:-250}"
EVAL_SPLIT="${BR_OM_EVAL_SPLIT:-test}"
PYTHON_BIN="${BR_OM_PYTHON:-python3}"

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

if [[ -n "${BR_OM_COHORT:-}" ]]; then
  COHORTS=("${BR_OM_COHORT}")
elif [[ -n "${BR_OM_COHORTS:-}" ]]; then
  IFS="|" read -r -a COHORTS <<< "${BR_OM_COHORTS}"
fi

mkdir -p "${RUN_ROOT}"
RUN_INDEX="${RUN_ROOT}/run_index.tsv"
if [[ ! -f "${RUN_INDEX}" ]]; then
  printf "started_at\tfinished_at\tstatus\tcohort_slug\tcohort\tout_dir\n" > "${RUN_INDEX}"
fi

cohort_slug() {
  "${PYTHON_BIN}" -c 'from pubg_lab.survival.v3.data.cohorts import resolve_cohort; import sys; print(resolve_cohort(sys.argv[1]).slug)' "$1"
}

for cohort in "${COHORTS[@]}"; do
  slug="$(cohort_slug "${cohort}")"
  out_dir="${RUN_ROOT}/${slug}"
  mkdir -p "${out_dir}"
  log_file="${out_dir}/run.log"
  started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  status="failed"
  cmd=(
    "${PYTHON_BIN}" -m pubg_lab.survival.v3.data.run_matrix
    --groups om_ladder
    --cohort "${cohort}"
    --out-dir "${out_dir}"
    --epochs "${EPOCHS}"
    --batch "${BATCH}"
    --workers "${WORKERS}"
    --log-every "${LOG_EVERY}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --eval-split "${EVAL_SPLIT}"
  )
  if [[ -n "${BR_OM_LIMIT:-}" ]]; then
    cmd+=(--limit "${BR_OM_LIMIT}")
  fi
  if [[ "${BR_OM_CACHED:-0}" == "1" ]]; then
    cmd+=(--cached)
  fi
  if [[ "${BR_OM_FORCE:-0}" == "1" ]]; then
    cmd+=(--force)
  fi
  if [[ "${BR_OM_FORCE_EVAL:-0}" == "1" ]]; then
    cmd+=(--force-eval)
  fi
  if [[ "${BR_OM_EVAL_ONLY:-0}" == "1" ]]; then
    cmd+=(--eval-only)
  fi
  echo "== OM ladder cohort: ${cohort} -> ${out_dir} =="
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

"${PYTHON_BIN}" projects/battle_royale/analysis/ontology/collect_om_ladder_metrics.py --root "${RUN_ROOT}"
