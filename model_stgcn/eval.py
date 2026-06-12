"""
model_stgcn/eval.py
===================
Evaluate a trained STGCN checkpoint on its held-out test split.

Metrics:
  • Regression on final rank
        - MAE
        - RMSE
        - Spearman ρ            ← most important for ranking
  • Classification on top-5 / top-10
        - ROC-AUC
        - Accuracy @ 0.5 threshold

Results are written as JSON to
`presentation_build/assets/eval_table.json` so the deck builder can render
them on slide 11 without re-running anything.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from model_stgcn.dataset import CachedMatchDataset, collate
from model_stgcn.stgcn import STGCN

logger = logging.getLogger("model_stgcn.eval")


def evaluate(ckpt_path: str, *, device: str = "cpu") -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    test_ids: list[str] = ckpt["test_ids"]
    args_d: dict = ckpt["args"]

    model = STGCN(
        in_features=8,
        hidden=args_d.get("hidden", 64),
        num_blocks=args_d.get("num_blocks", 2),
        dropout=args_d.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    ds = CachedMatchDataset(test_ids)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)

    rank_pred_list: list[float] = []
    rank_true_list: list[float] = []
    top5_pred_list: list[float] = []
    top5_true_list: list[float] = []
    top10_pred_list: list[float] = []
    top10_true_list: list[float] = []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            A = batch["A_t"].to(device)
            mask = batch["mask"]
            rp, cp = model(X, A)
            rp_cpu = rp.cpu()
            cp_cpu = torch.sigmoid(cp.cpu())
            for b in range(rp_cpu.size(0)):
                m = mask[b].numpy().astype(bool)
                rank_pred_list.extend((rp_cpu[b].numpy()[m] * 25.0).tolist())
                rank_true_list.extend(batch["y_reg"][b].numpy()[m].tolist())
                top5_pred_list.extend(cp_cpu[b].numpy()[m, 0].tolist())
                top5_true_list.extend(batch["y_cls"][b].numpy()[m, 0].tolist())
                top10_pred_list.extend(cp_cpu[b].numpy()[m, 1].tolist())
                top10_true_list.extend(batch["y_cls"][b].numpy()[m, 1].tolist())

    rp_arr = np.asarray(rank_pred_list)
    rt_arr = np.asarray(rank_true_list)
    top5_pred = np.asarray(top5_pred_list)
    top5_true = np.asarray(top5_true_list)
    top10_pred = np.asarray(top10_pred_list)
    top10_true = np.asarray(top10_true_list)

    mae = float(np.mean(np.abs(rp_arr - rt_arr))) if rp_arr.size else float("nan")
    rmse = float(np.sqrt(np.mean((rp_arr - rt_arr) ** 2))) if rp_arr.size else float("nan")
    spearman = float(spearmanr(rp_arr, rt_arr)[0]) if rp_arr.size > 1 else float("nan")

    def safe_auc(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))

    top5_auc = safe_auc(top5_true, top5_pred)
    top10_auc = safe_auc(top10_true, top10_pred)
    top5_acc = float(accuracy_score(top5_true, (top5_pred > 0.5).astype(int))) if top5_true.size else float("nan")
    top10_acc = float(accuracy_score(top10_true, (top10_pred > 0.5).astype(int))) if top10_true.size else float("nan")

    return {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "best_val_loss": ckpt.get("best_val"),
        "n_test_matches": len(test_ids),
        "n_test_teams": int(rp_arr.size),
        "mae": mae,
        "rmse": rmse,
        "spearman_rho": spearman,
        "top5_auc": top5_auc,
        "top5_accuracy": top5_acc,
        "top10_auc": top10_auc,
        "top10_accuracy": top10_acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="model_stgcn/checkpoints/best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default="presentation_build/assets/eval_table.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = evaluate(args.checkpoint, device=args.device)

    print(json.dumps(metrics, indent=2))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info("metrics written to %s", out_path)


if __name__ == "__main__":
    main()
