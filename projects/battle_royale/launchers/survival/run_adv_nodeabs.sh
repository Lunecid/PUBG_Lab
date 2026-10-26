#!/bin/zsh
# CONTROL: give phase NODES the absolute z/alive/health/zone. Then test whether
# the advantage EDGE still beats edge_shuffle. If yes -> genuinely relational
# (pairwise Delta beyond node absolutes). If ns -> it was just z-injection.
set -e

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
: "${PG_PASSWORD:?PG_PASSWORD must be set}"
EP=${ADV_EPOCHS:-12}
CACHE=pubg_lab.survival/checkpoints/adv_cache_nodeabs
OUT=pubg_lab.survival/checkpoints/adv_runs_nodeabs

echo "=== $(date +%H:%M:%S) precompute (node-abs, knn) ==="
python3 projects/battle_royale/analysis/advantage/adv_precompute.py --k 5 --gates knn --node-abs --out-dir "$CACHE" > logs/adv_pre_nodeabs.log 2>&1
echo "precompute done $(date +%H:%M:%S)"

for C in none_observed knn_observed knn_edge_shuffle; do
  GATE=${C%%_*}; CTRL=${C#*_}
  echo "=== $(date +%H:%M:%S) gate=$GATE control=$CTRL (node-abs nodes) ==="
  python3 -m pubg_lab.survival.v3.models.adv_model --cache-dir "$CACHE" --gate "$GATE" --control "$CTRL" \
    --epochs "$EP" --device mps --out "$OUT/${C}"
done
echo "NODE-ABS CONTROL DONE $(date +%H:%M:%S)"
