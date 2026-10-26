#!/bin/zsh
# k-ablation for the fused kNN advantage gate: confirm observed>shuffle holds
# across k in {3,5,8} (k=5 already in adv_runs). knn gate only -> fast/small.
set -e

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
: "${PG_PASSWORD:?PG_PASSWORD must be set}"
EP=${ADV_EPOCHS:-12}

echo "=== $(date +%H:%M:%S) precompute k=3 and k=8 (knn only, parallel) ==="
python3 projects/battle_royale/analysis/advantage/adv_precompute.py --k 3 --gates knn --out-dir pubg_lab.survival/checkpoints/adv_cache_k3 > logs/adv_pre_k3.log 2>&1 &
python3 projects/battle_royale/analysis/advantage/adv_precompute.py --k 8 --gates knn --out-dir pubg_lab.survival/checkpoints/adv_cache_k8 > logs/adv_pre_k8.log 2>&1 &
wait
echo "precompute done $(date +%H:%M:%S)"

for K in 3 8; do
  CACHE=pubg_lab.survival/checkpoints/adv_cache_k${K}
  OUT=pubg_lab.survival/checkpoints/adv_runs_k${K}
  for CTRL in observed edge_shuffle; do
    echo "=== $(date +%H:%M:%S) k=$K gate=knn control=$CTRL ==="
    python3 -m pubg_lab.survival.v3.models.adv_model --cache-dir "$CACHE" --gate knn --control "$CTRL" \
      --epochs "$EP" --device mps --out "$OUT/knn_${CTRL}"
  done
done
echo "K-ABLATION DONE $(date +%H:%M:%S)"
