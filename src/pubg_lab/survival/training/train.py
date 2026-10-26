"""
model_stgcn/train.py
====================
Train the STGCN on STDB-extracted match tensors.

Multi-task loss:
    L = α · MAE(rank) + β · BCE(top5) + γ · BCE(top10)

Defaults assume Baltic_Main squad-fpp; override with --map / --mode.
Checkpoints land under model_stgcn/checkpoints/<map>_<mode>_best.pt by default.

Usage examples:
    python -m model_stgcn.train --limit 50 --epochs 30
    python -m model_stgcn.train --map Baltic_Main --mode squad-fpp --epochs 80
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from pubg_lab.survival.data.dataset import CachedMatchDataset, collate, train_val_test_split
from pubg_lab.survival.models.stgcn import STGCN, stgcn_loss
from pubg_lab.data.stdb.queries.queries import FEATURE_DIM, list_eligible_match_ids

logger = logging.getLogger("pubg_lab.survival.training.train")


def evaluate_loss(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    parts_sum = {"l_reg": 0.0, "l_top5": 0.0, "l_top10": 0.0}
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            A = batch["A_t"].to(device)
            yr = batch["y_reg"].to(device)
            yc = batch["y_cls"].to(device)
            m = batch["mask"].to(device)
            rp, cp = model(X, A)
            loss, parts = stgcn_loss(rp, cp, yr, yc, m)
            bs = X.size(0)
            total += float(loss.item()) * bs
            n += bs
            for k in parts_sum:
                parts_sum[k] += parts[k] * bs
    if n == 0:
        return float("nan"), {k: float("nan") for k in parts_sum}
    return total / n, {k: v / n for k, v in parts_sum.items()}


def train_loop(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ids = list_eligible_match_ids(args.map, args.mode, args.limit)
    if len(ids) < 10:
        logger.error(
            "Only %d matches available — run more `stdb.etl` ingest first.",
            len(ids),
        )
        return

    train_ids, val_ids, test_ids = train_val_test_split(
        ids, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    logger.info(
        "split: train=%d val=%d test=%d",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )

    logger.info("loading train …")
    train_ds = CachedMatchDataset(train_ids)
    logger.info("loading val …")
    val_ds = CachedMatchDataset(val_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    device = torch.device(args.device)
    model = STGCN(
        in_features=8,
        hidden=args.hidden,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)
    logger.info(
        "model params: %.2f M",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history: list[dict] = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for batch in train_loader:
            X = batch["X"].to(device)
            A = batch["A_t"].to(device)
            yr = batch["y_reg"].to(device)
            yc = batch["y_cls"].to(device)
            m = batch["mask"].to(device)

            rp, cp = model(X, A)
            loss, parts = stgcn_loss(
                rp,
                cp,
                yr,
                yc,
                m,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = X.size(0)
            loss_sum += float(loss.item()) * bs
            n += bs
        train_loss = loss_sum / max(n, 1)
        sched.step()

        val_loss, val_parts = evaluate_loss(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_l_reg": val_parts["l_reg"],
                "val_l_top5": val_parts["l_top5"],
                "val_l_top10": val_parts["l_top10"],
                "lr": sched.get_last_lr()[0],
            }
        )

        logger.info(
            "epoch %3d  train=%.4f  val=%.4f  (reg=%.4f top5=%.4f top10=%.4f)  lr=%.5f",
            epoch,
            train_loss,
            val_loss,
            val_parts["l_reg"],
            val_parts["l_top5"],
            val_parts["l_top10"],
            sched.get_last_lr()[0],
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_val": val_loss,
                    "feature_dim": FEATURE_DIM,
                    "args": vars(args),
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "test_ids": test_ids,
                },
                ckpt_path,
            )
            logger.info("  → saved best ckpt to %s", ckpt_path)

    elapsed = time.time() - t_start
    logger.info("done in %.1fs  best_val=%.4f", elapsed, best_val)

    hist_path = ckpt_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    logger.info("wrote history to %s", hist_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="Baltic_Main")
    ap.add_argument("--mode", default="squad-fpp")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0, help="MAE weight")
    ap.add_argument("--beta", type=float, default=0.3, help="BCE top5 weight")
    ap.add_argument("--gamma", type=float, default=0.3, help="BCE top10 weight")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument(
        "--checkpoint",
        default="model_stgcn/checkpoints/best.pt",
        help="Output checkpoint path",
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    train_loop(parse_args())
