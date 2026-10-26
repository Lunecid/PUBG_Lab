#!/bin/zsh
# Regenerate V3 per-cell predictions (--pred-out) for significance bootstrap.
# Forward-only eval on saved checkpoints, shared test split, no training.
set -e

# Resolve source imports and artifact paths independently of the invoking directory.
PUBG_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PUBG_WORKSPACE"
export PYTHONPATH="$PUBG_WORKSPACE/src:$PUBG_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
: "${PG_PASSWORD:?PG_PASSWORD must be set}"
SPLIT=pubg_lab.survival/checkpoints/v3_full_main_causal/split.json
OUT=pubg_lab.survival/checkpoints/v3_pred
mkdir -p "$OUT"

CKPTS="v3_full_main_causal/main/mr_gat
v3_full_graph_relation/graph_relation/graph_core_habitat_no_graph
v3_full_graph_relation/graph_relation/graph_core_habitat_full_observed
v3_full_graph_relation/graph_relation/graph_core_habitat_full_edge_shuffle"

echo "$CKPTS" | while IFS= read -r c; do
  [ -z "$c" ] && continue
  name=$(echo "$c" | tr '/' '_')
  echo "=== $(date +%H:%M:%S) eval $c ==="
  python3 -m pubg_lab.survival.v3.evaluation.eval \
    --checkpoint "pubg_lab.survival/checkpoints/$c/model.pt" \
    --split test --split-json "$SPLIT" --require-existing-split --seed 42 --batch 8 \
    --device cpu --cached --mode squad \
    --out "$OUT/$name.eval.json" \
    --pred-out "$OUT/$name.pred.csv"
  echo "  -> wrote $OUT/$name.pred.csv ($(wc -l < "$OUT/$name.pred.csv") rows)"
done
echo "ALL DONE $(date +%H:%M:%S)"
