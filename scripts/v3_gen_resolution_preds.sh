#!/bin/zsh
# Regenerate predictions for the resolution group (MR vs phase) for match-level
# significance. Forward-only on saved checkpoints, shared test split, device cpu
# (so it does not fight the MPS training still running).
set -e
cd /Users/lunecid/project/PUBG
: "${PG_PASSWORD:?PG_PASSWORD must be set}"
SPLIT=model_stgcn/checkpoints/v3_full_main_causal/split.json
OUT=model_stgcn/checkpoints/v3_pred
mkdir -p "$OUT"

CKPTS="resolution/phase_tcn
resolution/phase_rgcn
resolution/mr_tcn
resolution/mr_rgcn"

echo "$CKPTS" | while IFS= read -r c; do
  [ -z "$c" ] && continue
  name="resolution_$(basename "$c")"
  echo "=== $(date +%H:%M:%S) eval $c ==="
  python3 -m model_stgcn.v3.eval \
    --checkpoint "model_stgcn/checkpoints/v3_full_controls/$c/model.pt" \
    --split test --split-json "$SPLIT" --require-existing-split --seed 42 --batch 8 \
    --device cpu --cached --mode squad \
    --out "$OUT/$name.eval.json" \
    --pred-out "$OUT/$name.pred.csv"
  echo "  -> wrote $OUT/$name.pred.csv"
done
echo "RESOLUTION PREDS DONE $(date +%H:%M:%S)"
