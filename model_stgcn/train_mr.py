"""
Train the Multi-Resolution ST-GNN (snapshot 10s + phase).

Phase-only baseline reproduction: --delta 0 disables snapshot survival
supervision; with snapshot tier still feeding the phase tier you get the
"snapshot representation only" ablation rung. Set --no-snapshot to fall back
to the pure phase model (use train_phase.py for that).

Protocol mirrors train_phase.py (AdamW, cosine, e12/seed42, match-level split)
so MR vs phase-only comparisons stay controlled.
"""

from __future__ import annotations

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

from model_stgcn.mr_dataset import CachedMRDataset, LazyMRDataset, mr_collate
from model_stgcn.mr_stgnn import MRSTGNN, mr_loss
from model_stgcn.splits import load_or_create_split
from stdb.queries import PHASE_FEATURE_DIM_V2, list_phase_eligible_match_ids
from stdb.snapshot_queries import SNAP_FEATURE_DIM

logger = logging.getLogger("model_stgcn.train_mr")


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _move(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _make_dataset(ids: list[str], *, cached: bool):
    cls = CachedMRDataset if cached else LazyMRDataset
    return cls(ids, include_v2=True)


def _run_epoch(model, loader, device, args, *, train: bool, opt=None):
    model.train(train)
    sums: dict[str, float] = {}
    total_n = 0
    for batch in loader:
        batch = _move(batch, device)
        hazard, survival, rank, topk = model(
            batch["X_s"].float(), batch["A_s"].float(),
            batch["phase_of_snap"], batch["node_mask_s"].bool(),
            batch["X"].float(), batch["A"].float(),
            phase_mask=batch["phase_mask"],
        )
        loss, parts = mr_loss(
            hazard, survival, rank, topk, batch,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
            horizon=args.horizon, hazard_pos_weight=args.hazard_pos_weight,
        )
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        bs = batch["X"].shape[0]
        total_n += bs
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v * bs
    return {k: v / max(total_n, 1) for k, v in sums.items()}


def train_loop(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ids = list_phase_eligible_match_ids(args.map, args.mode, args.limit)
    if len(ids) < 2:
        raise SystemExit(f"not enough matches: {len(ids)}")
    train_ids, val_ids, test_ids = load_or_create_split(
        ids, split_json=args.split_json, val_frac=args.val_frac,
        test_frac=args.test_frac, seed=args.seed,
        metadata={"source": "model_stgcn.train_mr", "map": args.map, "limit": args.limit},
    )
    logger.info("ids total=%d train=%d val=%d test=%d", len(ids), len(train_ids), len(val_ids), len(test_ids))

    train_ds = _make_dataset(train_ids, cached=args.cached)
    val_ds = _make_dataset(val_ids, cached=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=mr_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=mr_collate, num_workers=0)

    device = _device(args.device)
    model = MRSTGNN(
        snap_in_features=SNAP_FEATURE_DIM,
        phase_in_features=PHASE_FEATURE_DIM_V2,
        snap_hidden=args.snap_hidden,
        phase_hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    logger.info("device=%s params=%.3fM delta=%.2f", device, sum(p.numel() for p in model.parameters()) / 1e6, args.delta)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: list[dict] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = _run_epoch(model, train_loader, device, args, train=True, opt=opt)
        with torch.no_grad():
            va = _run_epoch(model, val_loader, device, args, train=False)
        sched.step()
        history.append({"epoch": epoch, "train": tr, "val": va})
        logger.info(
            "epoch %03d train=%.4f haz=%.4f surv=%.4f val=%.4f val_haz=%.4f val_surv=%.4f",
            epoch, tr.get("loss", float("nan")), tr.get("l_hazard", float("nan")),
            tr.get("l_survival", float("nan")), va.get("loss", float("nan")),
            va.get("l_hazard", float("nan")), va.get("l_survival", float("nan")),
        )
        if va.get("loss", float("inf")) < best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch, "best_val": best_val, "args": vars(args),
                    "snap_feature_dim": SNAP_FEATURE_DIM,
                    "phase_feature_dim": PHASE_FEATURE_DIM_V2,
                    "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
                },
                ckpt_path,
            )
            logger.info("saved best checkpoint to %s", ckpt_path)

    ckpt_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    logger.info("done in %.1fs best_val=%.4f", time.time() - started, best_val)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--snap-hidden", type=int, default=48)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--delta", type=float, default=0.3, help="snapshot survival weight (0 = repr-only)")
    ap.add_argument("--horizon", type=int, default=3, help="snapshot survival window (snapshots)")
    ap.add_argument("--hazard-pos-weight", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--checkpoint", default="model_stgcn/checkpoints/mr_stgnn.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--cached", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    train_loop(parse_args())
