#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PG_PASSWORD:?PG_PASSWORD must be set}"

RUN_ROOT="${BR_OM_ROOT:-model_stgcn/checkpoints/br_om_ladder_required_v1}"
QUEUE_SLEEP="${BR_OM_QUEUE_SLEEP:-300}"
PYTHON_BIN="${BR_OM_PYTHON:-/Users/lunecid/.pyenv/versions/3.11.5/bin/python3}"

mkdir -p "${RUN_ROOT}"
QUEUE_LOG="${RUN_ROOT}/required_remaining_queue.log"
exec >> "${QUEUE_LOG}" 2>&1

echo "== required cohort queue started at $(date -u +"%Y-%m-%dT%H:%M:%SZ") =="
echo "waiting for Erangel TPP squad completion markers"
echo "run root: ${RUN_ROOT}"
echo "python: ${PYTHON_BIN}"

while ! "${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
from pathlib import Path
import sys

run_root = Path(sys.argv[1])
root = run_root / "erangel_tpp_squad" / "om_ladder"
presets = [
    "om0_context",
    "om1_survival_capital",
    "om2_movement",
    "om3_pressure_habitat",
    "om4_threat_exposure",
    "om5_relation_structure",
    "om6_spatial_proxy",
]
required = ("model.pt", "train_complete.json", "eval_test.json")
missing = [str(root / preset / name) for preset in presets for name in required if not (root / preset / name).exists()]
if missing:
    print(f"Erangel TPP squad not complete yet: {len(missing)} missing marker(s).")
    print(f"next missing: {missing[0]}")
    raise SystemExit(1)
print("Erangel TPP squad completion markers verified.")
PY
do
  echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") sleeping ${QUEUE_SLEEP}s before next completion check"
  sleep "${QUEUE_SLEEP}"
done

unset BR_OM_COHORT
export BR_OM_ROOT="${RUN_ROOT}"
export BR_OM_PYTHON="${PYTHON_BIN}"
export BR_OM_COHORTS="Taego TPP squad|Miramar / Desert TPP squad|Rondo TPP squad|Sanhok TPP squad|Erangel TPP solo|Erangel TPP duo|Taego TPP duo|Vikendi TPP squad|Taego FPP squad|Erangel FPP squad|Paramo TPP squad|Karakin TPP squad"

echo "starting remaining cohorts at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
scripts/run_br_om_ladder_required_cohorts.sh
echo "required cohort queue finished at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
