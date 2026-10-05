#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

ROOT="${V3_OUT_ROOT:-model_stgcn/checkpoints/full_required_core13_v1}"
EPOCHS="${V3_EPOCHS:-12}"
BATCH="${V3_BATCH:-8}"
WORKERS="${V3_WORKERS:-4}"
DEVICE="${V3_DEVICE:-mps}"
LOG_EVERY="${V3_LOG_EVERY:-250}"
FORCE="${V3_FORCE:-0}"
CACHED="${V3_CACHED:-1}"
ONLY="${V3_ONLY:-mr_tcn,mr_sage,mr_rgcn,mr_farnet,mr_st_gnn_mamba}"

cohorts=(
  "erangel_tpp_squad|Erangel TPP squad"
  "taego_tpp_squad|Taego TPP squad"
  "miramar_desert_tpp_squad|Miramar / Desert TPP squad"
  "rondo_tpp_squad|Rondo TPP squad"
  "sanhok_tpp_squad|Sanhok TPP squad"
  "erangel_tpp_solo|Erangel TPP solo"
  "erangel_tpp_duo|Erangel TPP duo"
  "taego_tpp_duo|Taego TPP duo"
  "vikendi_tpp_squad|Vikendi TPP squad"
  "taego_fpp_squad|Taego FPP squad"
  "erangel_fpp_squad|Erangel FPP squad"
  "paramo_tpp_squad|Paramo TPP squad"
  "karakin_tpp_squad|Karakin TPP squad"
)

for item in "${cohorts[@]}"; do
  slug="${item%%|*}"
  cohort="${item#*|}"
  split_json="model_stgcn/checkpoints/br_om_ladder_required_v1/${slug}/split.json"
  out_dir="${ROOT}/${slug}"

  echo "== V3 core13: ${cohort} -> ${out_dir} =="
  cmd=(
    python3 -m model_stgcn.v3.run_matrix
    --groups main
    --only "${ONLY}"
    --cohort "${cohort}"
    --split-json "${split_json}"
    --out-dir "${out_dir}"
    --epochs "${EPOCHS}"
    --batch "${BATCH}"
    --workers "${WORKERS}"
    --log-every "${LOG_EVERY}"
    --device "${DEVICE}"
  )
  if [[ "${CACHED}" == "1" ]]; then
    cmd+=(--cached)
  fi
  if [[ "${FORCE}" == "1" ]]; then
    cmd+=(--force)
  fi

  "${cmd[@]}"
done
