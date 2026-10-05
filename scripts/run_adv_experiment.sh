#!/bin/zsh
# Learned-advantage edge experiment: 8 conditions on the cached phase tier (MPS).
# Tests whether contact-gated, relative-advantage edges beat null controls -
# i.e. does relation TYPE (advantage) carry signal where proximity did not.
set -e
cd /Users/lunecid/project/PUBG
: "${PG_PASSWORD:?PG_PASSWORD must be set}"
ROOT=model_stgcn/checkpoints/adv_runs
DEV=${ADV_DEV:-mps}
EP=${ADV_EPOCHS:-12}

run() {  # gate control
  echo "=== $(date +%H:%M:%S) gate=$1 control=$2 ==="
  python3 -m model_stgcn.v3.adv_model --gate "$1" --control "$2" --epochs "$EP" \
    --device "$DEV" --out "$ROOT/${1}_${2}"
}

run none      observed       # no-graph baseline
run knn       observed       # fused-gate advantage graph
run knn       node_shuffle   # break team<->advantage alignment
run knn       edge_shuffle   # destroy topology
run strict    observed       # combat-gated advantage
run strict    edge_shuffle
run full      observed       # original null graph + advantage features
run full      edge_shuffle
echo "ADV EXPERIMENT DONE $(date +%H:%M:%S)"
