"""
Train DynamicSTGNN on phase-level PUBG tensors.

Primary objective:
    X(phase <= t) -> eliminated during phase t+1

Smoke test:
    python -m model_stgcn.train_phase --limit 8 --epochs 30 --batch 4 --overfit
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

from model_stgcn.dataset import (
    CachedPhaseMatchDataset,
    LazyPhaseMatchDataset,
    phase_collate,
)
from model_stgcn.dyn_stgnn import DynamicSTGNN, dynamic_stgnn_loss
from model_stgcn.edge_modes import EDGE_MODE_CHOICES, apply_edge_mode, edge_active_mask
from model_stgcn.splits import load_or_create_split
from stdb.queries import (
    PHASE_EDGE_TYPES,
    PHASE_EDGE_TYPES_DIRECTED,
    PHASE_FEATURE_DIM,
    list_phase_eligible_match_ids,
)

logger = logging.getLogger("model_stgcn.train_phase")


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _metrics_from_logits(
    hazard_logits: torch.Tensor,
    y_hazard: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        mask = target_mask.bool()
        if mask.sum() == 0:
            return {"hazard_acc": float("nan"), "hazard_pos_rate": float("nan")}
        probs = torch.sigmoid(hazard_logits)
        pred = probs >= 0.5
        truth = y_hazard >= 0.5
        acc = (pred[mask] == truth[mask]).float().mean()
        pos_rate = truth[mask].float().mean()
        return {
            "hazard_acc": float(acc.detach().cpu().item()),
            "hazard_pos_rate": float(pos_rate.detach().cpu().item()),
        }


def evaluate(model: DynamicSTGNN, loader: DataLoader, device: torch.device, args) -> tuple[float, dict[str, float]]:
    model.eval()
    total = 0.0
    n = 0
    sums = {"l_hazard": 0.0, "l_rank": 0.0, "l_topk": 0.0, "hazard_acc": 0.0, "hazard_pos_rate": 0.0}
    with torch.no_grad():
        active_mask = edge_active_mask(args.edge_mode, args.edge_types, device=device)
        for batch in loader:
            batch = _move_batch(batch, device)
            A = apply_edge_mode(batch["A"].float(), args.edge_mode, args.edge_types)
            yh, yr, yk = model(batch["X"].float(), A, edge_active_mask=active_mask, phase_mask=batch["phase_mask"])
            loss, parts = dynamic_stgnn_loss(
                yh,
                yr,
                yk,
                batch["y_hazard"].float(),
                batch["target_mask"],
                batch["y_rank"].float(),
                batch["y_topk"].float(),
                batch["team_mask"],
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                hazard_pos_weight=args.hazard_pos_weight,
            )
            metrics = _metrics_from_logits(yh, batch["y_hazard"].float(), batch["target_mask"])
            bs = batch["X"].size(0)
            total += float(loss.item()) * bs
            n += bs
            for key in ("l_hazard", "l_rank", "l_topk"):
                sums[key] += parts[key] * bs
            for key in ("hazard_acc", "hazard_pos_rate"):
                sums[key] += metrics[key] * bs
    if n == 0:
        return float("nan"), {k: float("nan") for k in sums}
    return total / n, {k: v / n for k, v in sums.items()}


def _make_dataset(ids: list[str], *, cached: bool, include_combat_in: bool = False):
    cls = CachedPhaseMatchDataset if cached else LazyPhaseMatchDataset
    return cls(ids, include_combat_in=include_combat_in)


def train_loop(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 엣지 채널 레이아웃: --combat-in이면 E=4 (combat_in 추가)
    args.edge_types = list(PHASE_EDGE_TYPES_DIRECTED if args.combat_in else PHASE_EDGE_TYPES)

    ids = list_phase_eligible_match_ids(args.map, args.mode, args.limit)
    if len(ids) < 2:
        raise SystemExit(f"not enough phase-eligible matches: {len(ids)}")

    if args.overfit:
        train_ids = ids
        val_ids = ids
        test_ids: list[str] = []
    else:
        train_ids, val_ids, test_ids = load_or_create_split(
            ids,
            split_json=args.split_json,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            metadata={
                "map": args.map,
                "mode": args.mode,
                "limit": args.limit,
                "source": "model_stgcn.train_phase",
            },
        )

    logger.info(
        "ids: total=%d train=%d val=%d test=%d overfit=%s",
        len(ids),
        len(train_ids),
        len(val_ids),
        len(test_ids),
        args.overfit,
    )

    train_ds = _make_dataset(train_ids, cached=args.cached, include_combat_in=args.combat_in)
    val_ds = _make_dataset(val_ids, cached=True, include_combat_in=args.combat_in)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=not args.overfit,
        collate_fn=phase_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=phase_collate,
        num_workers=0,
    )

    device = _device(args.device)
    model = DynamicSTGNN(
        in_features=PHASE_FEATURE_DIM,
        hidden=args.hidden,
        mid=args.mid,
        num_blocks=args.num_blocks,
        num_edge_types=len(args.edge_types),
        dropout=args.dropout,
    ).to(device)
    logger.info("device=%s params=%.3fM", device, sum(p.numel() for p in model.parameters()) / 1e6)
    logger.info("edge_mode=%s edge_types=%s", args.edge_mode, args.edge_types)
    active_mask = edge_active_mask(args.edge_mode, args.edge_types, device=device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history: list[dict] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        sums = {"l_hazard": 0.0, "l_rank": 0.0, "l_topk": 0.0}
        for batch in train_loader:
            batch = _move_batch(batch, device)
            A = apply_edge_mode(batch["A"].float(), args.edge_mode, args.edge_types)
            yh, yr, yk = model(batch["X"].float(), A, edge_active_mask=active_mask, phase_mask=batch["phase_mask"])
            loss, parts = dynamic_stgnn_loss(
                yh,
                yr,
                yk,
                batch["y_hazard"].float(),
                batch["target_mask"],
                batch["y_rank"].float(),
                batch["y_topk"].float(),
                batch["team_mask"],
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                hazard_pos_weight=args.hazard_pos_weight,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            bs = batch["X"].size(0)
            total += float(loss.item()) * bs
            n += bs
            for key in sums:
                sums[key] += parts[key] * bs

        sched.step()
        train_loss = total / max(n, 1)
        train_parts = {k: v / max(n, 1) for k, v in sums.items()}
        val_loss, val_parts = evaluate(model, val_loader, device, args)
        edge_weights = [w.detach().cpu().tolist() for w in model.edge_weights(edge_active_mask=active_mask)]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_l_hazard": train_parts["l_hazard"],
            "train_l_rank": train_parts["l_rank"],
            "train_l_topk": train_parts["l_topk"],
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_parts.items()},
            "lr": sched.get_last_lr()[0],
            "edge_weights": edge_weights,
        }
        history.append(row)

        logger.info(
            "epoch %03d train=%.4f haz=%.4f rank=%.4f topk=%.4f "
            "val=%.4f val_haz=%.4f acc=%.3f pos=%.3f lr=%.5g ew=%s",
            epoch,
            train_loss,
            train_parts["l_hazard"],
            train_parts["l_rank"],
            train_parts["l_topk"],
            val_loss,
            val_parts["l_hazard"],
            val_parts["hazard_acc"],
            val_parts["hazard_pos_rate"],
            sched.get_last_lr()[0],
            edge_weights[-1] if edge_weights else None,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "args": vars(args),
                    "feature_dim": PHASE_FEATURE_DIM,
                    "edge_types": list(args.edge_types),
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "test_ids": test_ids,
                },
                ckpt_path,
            )
            logger.info("saved best checkpoint to %s", ckpt_path)

    hist_path = ckpt_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    logger.info("done in %.1fs best_val=%.4f history=%s", time.time() - started, best_val, hist_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--mid", type=int, default=128)
    ap.add_argument("--num-blocks", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--hazard-pos-weight", type=float, default=1.0)
    ap.add_argument("--edge-mode", default="full", choices=EDGE_MODE_CHOICES)
    ap.add_argument(
        "--combat-in",
        action="store_true",
        help="add transposed combat channel (E=4) so victims aggregate attacker state (CODE_REVIEW C5)",
    )
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--checkpoint", default="model_stgcn/checkpoints/dynamic_stgnn_phase.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split-json",
        default=None,
        help="Load this match split manifest, or create it if it does not exist",
    )
    ap.add_argument("--cached", action="store_true", help="Cache train tensors in memory")
    ap.add_argument("--overfit", action="store_true", help="Use the same small set for train/val")
    return ap.parse_args()


if __name__ == "__main__":
    train_loop(parse_args())
