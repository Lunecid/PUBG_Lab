#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

RUN_ROOT="${BR_RQ4_HIER_ROOT:-model_stgcn/checkpoints/br_rq4_hier_survival_required_v1}"
OM_ROOT="${BR_RQ4_HIER_OM_ROOT:-model_stgcn/checkpoints/br_om_ladder_required_v1}"
PYTHON_BIN="${BR_RQ4_HIER_PYTHON:-python3}"
SEED="${BR_RQ4_HIER_SEED:-42}"
EPOCHS="${BR_RQ4_HIER_EPOCHS:-12}"
BATCH="${BR_RQ4_HIER_BATCH:-8}"
HIDDEN="${BR_RQ4_HIER_HIDDEN:-48}"
MID="${BR_RQ4_HIER_MID:-96}"
NUM_BLOCKS="${BR_RQ4_HIER_NUM_BLOCKS:-1}"
NUM_HEADS="${BR_RQ4_HIER_NUM_HEADS:-4}"
KERNEL_SIZE="${BR_RQ4_HIER_KERNEL_SIZE:-3}"
EDGE_HIDDEN="${BR_RQ4_HIER_EDGE_HIDDEN:-16}"
DEVICE="${BR_RQ4_HIER_DEVICE:-cpu}"
WORKERS="${BR_RQ4_HIER_WORKERS:-0}"
EVAL_WORKERS="${BR_RQ4_HIER_EVAL_WORKERS:-0}"
PREFETCH_FACTOR="${BR_RQ4_HIER_PREFETCH_FACTOR:-2}"
CACHED="${BR_RQ4_HIER_CACHED:-1}"
FORCE="${BR_RQ4_HIER_FORCE:-0}"
DRY_RUN="${BR_RQ4_HIER_DRY_RUN:-0}"
RUN_TABULAR="${BR_RQ4_HIER_RUN_TABULAR:-1}"
TABULAR_CONFIGS="${BR_RQ4_HIER_TABULAR_CONFIGS:-snapshot_last_hgb,snapshot_summary_hgb}"
TABULAR_LEARNERS="${BR_RQ4_HIER_TABULAR_LEARNERS:-hgb}"
TABULAR_MAX_ITER="${BR_RQ4_HIER_TABULAR_MAX_ITER:-200}"
TABULAR_N_ESTIMATORS="${BR_RQ4_HIER_TABULAR_N_ESTIMATORS:-300}"
TABULAR_N_JOBS="${BR_RQ4_HIER_TABULAR_N_JOBS:-0}"
SNAPSHOT_CACHE_ROOT="${BR_RQ4_HIER_SNAPSHOT_CACHE_DIR:-}"

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

VARIANTS=(
  "snapshot_last_only"
  "snapshot_tcn_no_graph"
  "snapshot_gru_no_graph"
  "snapshot_lstm_no_graph"
  "snapshot_transformer_no_graph"
)

if [[ -n "${BR_RQ4_HIER_COHORT:-}" ]]; then
  COHORTS=("${BR_RQ4_HIER_COHORT}")
elif [[ -n "${BR_RQ4_HIER_COHORTS:-}" ]]; then
  IFS="|" read -r -a COHORTS <<< "${BR_RQ4_HIER_COHORTS}"
fi

if [[ -n "${BR_RQ4_HIER_VARIANTS:-}" ]]; then
  IFS="|" read -r -a VARIANTS <<< "${BR_RQ4_HIER_VARIANTS}"
fi

mkdir -p "${RUN_ROOT}"
RUN_INDEX="${RUN_ROOT}/run_index.tsv"
if [[ ! -f "${RUN_INDEX}" ]]; then
  printf "started_at\tfinished_at\tstatus\tcohort_slug\tcohort\tvariant\tout_dir\n" > "${RUN_INDEX}"
fi

resolve_cohort() {
  "${PYTHON_BIN}" -c 'from model_stgcn.v3.cohorts import resolve_cohort; import sys
spec = resolve_cohort(sys.argv[1])
print(f"{spec.slug}\t{spec.map_name}\t{spec.game_mode}")' "$1"
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

for cohort in "${COHORTS[@]}"; do
  IFS=$'\t' read -r slug map_name game_mode <<< "$(resolve_cohort "${cohort}")"
  split_json="${OM_ROOT}/${slug}/split.json"
  if [[ -n "${BR_RQ4_HIER_LIMIT:-}" ]]; then
    split_json="${RUN_ROOT}/${slug}/split_limit_${BR_RQ4_HIER_LIMIT}.json"
  elif [[ ! -f "${split_json}" ]]; then
    echo "missing split_json: ${split_json}" >&2
    exit 1
  fi

  if [[ "${RUN_TABULAR}" == "1" ]]; then
    tabular_dir_name="snapshot_tabular_hgb"
    if [[ "${TABULAR_LEARNERS}" != "hgb" ]]; then
      tabular_dir_name="snapshot_tabular_zoo"
    fi
    out_dir="${RUN_ROOT}/${slug}/${tabular_dir_name}"
    summary_json="${out_dir}/snapshot_tabular_summary.json"
    log_file="${out_dir}/run.log"
    mkdir -p "${out_dir}"
    tabular_args=(
      -m model_stgcn.snapshot_tabular_baseline
      --map "${map_name}"
      --mode "${game_mode}"
      --seed "${SEED}"
      --split-json "${split_json}"
      --out-dir "${out_dir}"
      --configs "${TABULAR_CONFIGS}"
      --learners "${TABULAR_LEARNERS}"
      --max-iter "${TABULAR_MAX_ITER}"
      --n-estimators "${TABULAR_N_ESTIMATORS}"
      --n-jobs "${TABULAR_N_JOBS}"
    )
    if [[ -n "${BR_RQ4_HIER_LIMIT:-}" ]]; then
      tabular_args+=(--limit "${BR_RQ4_HIER_LIMIT}")
    fi
    started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    status="failed"
    echo "== RQ4 same-condition tabular: ${cohort} -> ${out_dir} =="
    {
      printf '+ %q' "${PYTHON_BIN}"
      printf ' %q' "${tabular_args[@]}"
      printf '\n'
    } > "${log_file}"
    if [[ "${FORCE}" == "1" || ! -f "${summary_json}" ]]; then
      if run_cmd "${PYTHON_BIN}" "${tabular_args[@]}" >> "${log_file}" 2>&1; then
        status="passed"
      else
        echo "failed tabular baseline: ${cohort} log=${log_file}" >&2
        tail -80 "${log_file}" >&2
        exit 1
      fi
    else
      echo "SKIP tabular: ${summary_json}" | tee -a "${log_file}"
      status="passed"
    fi
    finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${started_at}" "${finished_at}" "${status}" "${slug}" "${cohort}" "${tabular_dir_name}" "${out_dir}" >> "${RUN_INDEX}"
  fi

  for variant in "${VARIANTS[@]}"; do
    out_dir="${RUN_ROOT}/${slug}/${variant}"
    snapshot_cache_dir=""
    if [[ -n "${SNAPSHOT_CACHE_ROOT}" && "${SNAPSHOT_CACHE_ROOT}" != "none" ]]; then
      snapshot_cache_dir="${SNAPSHOT_CACHE_ROOT}/${slug}"
    fi
    ckpt="${out_dir}/model.pt"
    summary_json="${out_dir}/model.summary.json"
    bridge_json="${out_dir}/survival_bridge_test.json"
    bridge_csv="${out_dir}/survival_bridge_test.pred.csv"
    log_file="${out_dir}/run.log"
    mkdir -p "${out_dir}"

    train_args=(
      -m model_stgcn.train_snapshot_phase
      --map "${map_name}"
      --mode "${game_mode}"
      --epochs "${EPOCHS}"
      --batch "${BATCH}"
      --hidden "${HIDDEN}"
      --mid "${MID}"
      --num-blocks "${NUM_BLOCKS}"
      --num-heads "${NUM_HEADS}"
      --kernel-size "${KERNEL_SIZE}"
      --edge-hidden "${EDGE_HIDDEN}"
      --device "${DEVICE}"
      --seed "${SEED}"
      --split-json "${split_json}"
      --workers "${WORKERS}"
      --eval-workers "${EVAL_WORKERS}"
      --prefetch-factor "${PREFETCH_FACTOR}"
      --model-variant "${variant}"
      --checkpoint "${ckpt}"
    )
    if [[ -n "${snapshot_cache_dir}" ]]; then
      train_args+=(--snapshot-cache-dir "${snapshot_cache_dir}")
    fi
    if [[ -n "${BR_RQ4_HIER_LIMIT:-}" ]]; then
      train_args+=(--limit "${BR_RQ4_HIER_LIMIT}")
    fi
    if [[ "${CACHED}" == "1" ]]; then
      train_args+=(--cached)
    fi

    eval_args=(
      -m tools.eval_snapshot_survival_bridge
      --checkpoint "${ckpt}"
      --split test
      --device "${DEVICE}"
      --batch "${BATCH}"
      --out "${bridge_json}"
      --pred-out "${bridge_csv}"
    )
    if [[ -n "${snapshot_cache_dir}" ]]; then
      eval_args+=(--snapshot-cache-dir "${snapshot_cache_dir}")
    fi
    if [[ "${CACHED}" == "1" ]]; then
      eval_args+=(--cached)
    fi

    started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    status="failed"
    echo "== RQ4 hierarchical survival: ${cohort} / ${variant} -> ${out_dir} =="
    {
      printf '+ %q' "${PYTHON_BIN}"
      printf ' %q' "${train_args[@]}"
      printf '\n'
      printf '+ %q' "${PYTHON_BIN}"
      printf ' %q' "${eval_args[@]}"
      printf '\n'
    } > "${log_file}"

    if [[ "${FORCE}" == "1" || ! -f "${summary_json}" ]]; then
      if run_cmd "${PYTHON_BIN}" "${train_args[@]}" >> "${log_file}" 2>&1; then
        status="trained"
      else
        echo "failed training: ${cohort} / ${variant} log=${log_file}" >&2
        tail -80 "${log_file}" >&2
        exit 1
      fi
    else
      echo "SKIP train: ${summary_json}" | tee -a "${log_file}"
      status="train_skipped"
    fi

    if [[ "${FORCE}" == "1" || ! -f "${bridge_json}" ]]; then
      if run_cmd "${PYTHON_BIN}" "${eval_args[@]}" >> "${log_file}" 2>&1; then
        status="passed"
      else
        echo "failed bridge eval: ${cohort} / ${variant} log=${log_file}" >&2
        tail -80 "${log_file}" >&2
        exit 1
      fi
    else
      echo "SKIP eval: ${bridge_json}" | tee -a "${log_file}"
      status="passed"
    fi

    finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${started_at}" "${finished_at}" "${status}" "${slug}" "${cohort}" "${variant}" "${out_dir}" >> "${RUN_INDEX}"
  done
done

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" tools/collect_snapshot_survival_bridge_metrics.py --root "${RUN_ROOT}"
fi
