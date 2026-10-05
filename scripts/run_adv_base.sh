#!/bin/zsh
# Re-base the WHOLE advantage experiment on the correct node spec (node-abs ON):
# nodes carry z/alive/health/zone; ablations test what the advantage EDGE adds
# ON TOP of complete nodes. Main 8-condition matrix + k-ablation, all node-abs base.
set -e
cd /Users/lunecid/project/PUBG
: "${PG_PASSWORD:?PG_PASSWORD must be set}"
EP=${ADV_EPOCHS:-12}

echo "=== $(date +%H:%M:%S) precompute base (all gates, node-abs) + k3/k8 (knn) parallel ==="
python3 scripts/adv_precompute.py --k 5 --out-dir model_stgcn/checkpoints/adv_cache_base > logs/adv_pre_base.log 2>&1 &
python3 scripts/adv_precompute.py --k 3 --gates knn --out-dir model_stgcn/checkpoints/adv_cache_base_k3 > logs/adv_pre_base_k3.log 2>&1 &
python3 scripts/adv_precompute.py --k 8 --gates knn --out-dir model_stgcn/checkpoints/adv_cache_base_k8 > logs/adv_pre_base_k8.log 2>&1 &
wait
echo "precompute done $(date +%H:%M:%S)"

BASE=model_stgcn/checkpoints/adv_cache_base
ROOT=model_stgcn/checkpoints/adv_runs_base
mrun() { echo "=== $(date +%H:%M:%S) gate=$1 control=$2 ==="; python3 -m model_stgcn.v3.adv_model --cache-dir "$BASE" --gate "$1" --control "$2" --epochs "$EP" --device mps --out "$ROOT/${1}_${2}"; }
mrun none   observed
mrun knn    observed
mrun knn    node_shuffle
mrun knn    edge_shuffle
mrun strict observed
mrun strict edge_shuffle
mrun full   observed
mrun full   edge_shuffle

for K in 3 8; do
  for C in observed edge_shuffle; do
    echo "=== $(date +%H:%M:%S) k=$K knn $C ==="
    python3 -m model_stgcn.v3.adv_model --cache-dir "model_stgcn/checkpoints/adv_cache_base_k${K}" --gate knn --control "$C" --epochs "$EP" --device mps --out "model_stgcn/checkpoints/adv_runs_base_k${K}/knn_${C}"
  done
done
echo "ADV BASE RE-RUN DONE $(date +%H:%M:%S)"
