#!/usr/bin/env bash
set -euo pipefail

ROOT="${O7_ROOT:-model_stgcn/checkpoints/br_o7_advantage_edge_required_v1}"
BOOTSTRAP="${O7_BOOTSTRAP:-200}"
FEATURE_PRESET="${O7_FEATURE_PRESET:-om5_relation_structure}"
COHORTS="${O7_COHORTS:-erangel_tpp_squad|taego_tpp_squad|erangel_tpp_solo|miramar_desert_tpp_squad|rondo_tpp_squad|sanhok_tpp_squad|vikendi_tpp_squad|paramo_tpp_squad|karakin_tpp_squad|erangel_tpp_duo|taego_tpp_duo|taego_fpp_squad|erangel_fpp_squad}"

IFS='|' read -r -a cohort_array <<< "$COHORTS"
for slug in "${cohort_array[@]}"; do
  echo "== O7 hazard-implied rank bridge: ${slug} =="
  python3 tools/analyze_o7_advantage_rank_bridge.py \
    --root "${ROOT}/${slug}" \
    --bootstrap "${BOOTSTRAP}" \
    --feature-preset "${FEATURE_PRESET}"
done

python3 tools/aggregate_o7_advantage_readout.py \
  --root "${ROOT}" \
  --cohorts "${COHORTS}" \
  --input-name o7_rank_bridge_phase_group_deltas.csv \
  --out-prefix o7_required_13cohort_rank_bridge
