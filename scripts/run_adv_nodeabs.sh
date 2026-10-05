#!/bin/zsh
# CONTROL: give phase NODES the absolute z/alive/health/zone. Then test whether
# the advantage EDGE still beats edge_shuffle. If yes -> genuinely relational
# (pairwise Delta beyond node absolutes). If ns -> it was just z-injection.
set -e
cd /Users/lunecid/project/PUBG
: "${PG_PASSWORD:?PG_PASSWORD must be set}"
EP=${ADV_EPOCHS:-12}
CACHE=model_stgcn/checkpoints/adv_cache_nodeabs
OUT=model_stgcn/checkpoints/adv_runs_nodeabs

echo "=== $(date +%H:%M:%S) precompute (node-abs, knn) ==="
python3 scripts/adv_precompute.py --k 5 --gates knn --node-abs --out-dir "$CACHE" > logs/adv_pre_nodeabs.log 2>&1
echo "precompute done $(date +%H:%M:%S)"

for C in none_observed knn_observed knn_edge_shuffle; do
  GATE=${C%%_*}; CTRL=${C#*_}
  echo "=== $(date +%H:%M:%S) gate=$GATE control=$CTRL (node-abs nodes) ==="
  python3 -m model_stgcn.v3.adv_model --cache-dir "$CACHE" --gate "$GATE" --control "$CTRL" \
    --epochs "$EP" --device mps --out "$OUT/${C}"
done
echo "NODE-ABS CONTROL DONE $(date +%H:%M:%S)"
