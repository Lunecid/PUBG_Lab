"""Train V3 factorized models.

Examples:

    python3 -m model_stgcn.v3.train --model phase --graph-encoder rgcn --global-fusion concat
    python3 -m model_stgcn.v3.train --model mr --graph-encoder gat --global-fusion film
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

from model_stgcn.edge_modes import EDGE_MODE_CHOICES, edge_active_mask
from model_stgcn.splits import load_or_create_split
from model_stgcn.v3.edge_controls import EDGE_CONTROL_CHOICES, apply_phase_edge_control
from model_stgcn.v3.factorized_data import (
    CachedFactorizedMRDataset,
    CachedFactorizedPhaseDataset,
    LazyFactorizedMRDataset,
    LazyFactorizedPhaseDataset,
    factorized_collate,
)
from model_stgcn.v3.models import (
    GLOBAL_FUSIONS,
    GRAPH_ENCODERS,
    TEAM_DECODERS,
    FactorizedMRGNN,
    FactorizedPhaseGNN,
    factorized_mr_loss,
    factorized_phase_loss,
)
from model_stgcn.v3.spec import FEATURE_PRESETS
from stdb.queries import list_phase_eligible_match_ids
from stdb.snapshot_queries import SNAP_FEATURE_DIM

logger = logging.getLogger("model_stgcn.v3.train")


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


def _dataset(ids: list[str], args: argparse.Namespace, *, cached: bool):
    if args.model == "mr":
        cls = CachedFactorizedMRDataset if cached else LazyFactorizedMRDataset
    else:
        cls = CachedFactorizedPhaseDataset if cached else LazyFactorizedPhaseDataset
    return cls(
        ids,
        include_v2_features=not args.no_v2_features,
        include_combat_in=args.combat_in,
        feature_preset=args.feature_preset,
    )


def _build_model(sample: dict, args: argparse.Namespace) -> torch.nn.Module:
    node_dim = int(sample["X_node"].shape[-1])
    global_dim = int(sample["G"].shape[-1])
    edge_types = list(sample["edge_types"])
    if args.model == "mr":
        return FactorizedMRGNN(
            snap_features=SNAP_FEATURE_DIM,
            node_features=node_dim,
            global_features=global_dim,
            snap_hidden=args.snap_hidden,
            phase_hidden=args.hidden,
            phase_mid=args.mid,
            phase_blocks=args.num_blocks,
            phase_edge_types=len(edge_types),
            graph_encoder=args.graph_encoder,
            global_fusion=args.global_fusion,
            team_decoder=args.team_decoder,
            decoder_heads=args.decoder_heads,
            decoder_layers=args.decoder_layers,
            dropout=args.dropout,
        )
    return FactorizedPhaseGNN(
        node_features=node_dim,
        global_features=global_dim,
        hidden=args.hidden,
        mid=args.mid,
        num_blocks=args.num_blocks,
        num_edge_types=len(edge_types),
        graph_encoder=args.graph_encoder,
        global_fusion=args.global_fusion,
        team_decoder=args.team_decoder,
        decoder_heads=args.decoder_heads,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        )


def _edge_control_generator(args: argparse.Namespace, *, train: bool, epoch: int | None) -> torch.Generator | None:
    if args.edge_control == "observed":
        return None
    gen = torch.Generator()
    offset = (epoch or 0) * 1009 + (0 if train else 100_000)
    gen.manual_seed(int(args.edge_control_seed) + offset)
    return gen


def _run_epoch(model, loader, device, args, *, train: bool, opt=None, epoch: int | None = None) -> dict[str, float]:
    model.train(train)
    sums: dict[str, float] = {}
    total_n = 0
    total_batches = len(loader)
    edge_control_generator = _edge_control_generator(args, train=train, epoch=epoch)
    for batch_idx, batch in enumerate(loader, start=1):
        batch = _move(batch, device)
        edge_types = batch["edge_types"][0] if batch["edge_types"] and isinstance(batch["edge_types"][0], list) else batch["edge_types"]
        phase_edge_mask = edge_active_mask(
            args.edge_mode,
            edge_types,
            device=device,
        )
        A_phase = apply_phase_edge_control(
            batch["A"].float(),
            batch["node_mask"].bool(),
            control=args.edge_control,
            generator=edge_control_generator,
        )
        if args.model == "mr":
            hazard, survival, rank, topk = model(
                batch["X_s"].float(),
                batch["A_s"].float(),
                batch["phase_of_snap"],
                batch["node_mask_s"].bool(),
                batch["X_node"].float(),
                batch["G"].float(),
                A_phase,
                phase_edge_mask=phase_edge_mask,
                phase_mask=batch["phase_mask"],
                team_mask=batch["team_mask"],
            )
            loss, parts = factorized_mr_loss(
                hazard,
                survival,
                rank,
                topk,
                batch,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                delta=args.delta,
                pairwise_rank_weight=args.pairwise_rank_weight,
                horizon=args.horizon,
                hazard_pos_weight=args.hazard_pos_weight,
            )
        else:
            hazard, rank, topk = model(
                batch["X_node"].float(),
                batch["G"].float(),
                A_phase,
                edge_active_mask=phase_edge_mask,
                phase_mask=batch["phase_mask"],
                team_mask=batch["team_mask"],
            )
            loss, parts = factorized_phase_loss(
                hazard,
                rank,
                topk,
                batch,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                pairwise_rank_weight=args.pairwise_rank_weight,
                hazard_pos_weight=args.hazard_pos_weight,
            )
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        bs = int(batch["X_node"].shape[0])
        total_n += bs
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v) * bs
        if (
            train
            and args.log_every > 0
            and (batch_idx % args.log_every == 0 or batch_idx == total_batches)
        ):
            avg_loss = sums.get("loss", 0.0) / max(total_n, 1)
            avg_haz = sums.get("l_hazard", 0.0) / max(total_n, 1)
            logger.info(
                "epoch %03d progress %d/%d batches loss=%.4f haz=%.4f",
                epoch or 0,
                batch_idx,
                total_batches,
                avg_loss,
                avg_haz,
            )
    return {k: v / max(total_n, 1) for k, v in sums.items()}


def run(args: argparse.Namespace) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ids = list_phase_eligible_match_ids(args.map, args.mode, args.limit)
    train_ids, val_ids, test_ids = load_or_create_split(
        ids,
        split_json=args.split_json,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        metadata={
            "source": "model_stgcn.v3.train",
            "model": args.model,
            "graph_encoder": args.graph_encoder,
            "global_fusion": args.global_fusion,
            "edge_mode": args.edge_mode,
            "edge_control": args.edge_control,
            "edge_control_seed": args.edge_control_seed,
            "feature_preset": args.feature_preset,
        },
    )
    logger.info("ids total=%d train=%d val=%d test=%d", len(ids), len(train_ids), len(val_ids), len(test_ids))

    train_ds = _dataset(train_ids, args, cached=args.cached)
    val_ds = _dataset(val_ids, args, cached=True)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("empty train/val dataset")

    sample = train_ds[0]
    model = _build_model(sample, args)
    device = _device(args.device)
    model.to(device)
    logger.info(
        "model=%s graph=%s global=%s node_dim=%d global_dim=%d params=%.3fM",
        args.model,
        args.graph_encoder,
        args.global_fusion,
        sample["X_node"].shape[-1],
        sample["G"].shape[-1],
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    logger.info(
        "feature_preset=%s node_features=%d global_features=%d",
        args.feature_preset,
        len(sample["node_feature_names"]),
        len(sample["global_feature_names"]),
    )

    loader_kwargs = {
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=factorized_collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=factorized_collate)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: list[dict] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = _run_epoch(model, train_loader, device, args, train=True, opt=opt, epoch=epoch)
        with torch.no_grad():
            va = _run_epoch(model, val_loader, device, args, train=False, epoch=epoch)
        sched.step()
        history.append({"epoch": epoch, "train": tr, "val": va})
        logger.info(
            "epoch %03d train=%.4f haz=%.4f val=%.4f val_haz=%.4f",
            epoch,
            tr.get("loss", float("nan")),
            tr.get("l_hazard", float("nan")),
            va.get("loss", float("nan")),
            va.get("l_hazard", float("nan")),
        )
        if va.get("loss", float("inf")) < best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "args": vars(args),
                    "node_feature_names": sample["node_feature_names"],
                    "global_feature_names": sample["global_feature_names"],
                    "edge_types": sample["edge_types"],
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "test_ids": test_ids,
                },
                ckpt_path,
            )
            logger.info("saved best checkpoint to %s", ckpt_path)

    hist_path = ckpt_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    summary = {
        "checkpoint": str(ckpt_path),
        "history": str(hist_path),
        "best_val": best_val,
        "elapsed_s": round(time.time() - started, 3),
        "split": {"total": len(ids), "train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
    }
    logger.info("summary=%s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["phase", "mr"], default="phase")
    ap.add_argument("--graph-encoder", choices=GRAPH_ENCODERS, default="rgcn")
    ap.add_argument("--global-fusion", choices=GLOBAL_FUSIONS, default="concat")
    ap.add_argument("--team-decoder", choices=TEAM_DECODERS, default="mlp")
    ap.add_argument("--decoder-heads", type=int, default=4)
    ap.add_argument("--decoder-layers", type=int, default=1)
    ap.add_argument("--edge-mode", choices=EDGE_MODE_CHOICES, default="full")
    ap.add_argument("--edge-control", choices=EDGE_CONTROL_CHOICES, default="observed")
    ap.add_argument("--edge-control-seed", type=int, default=42)
    ap.add_argument("--combat-in", action="store_true")
    ap.add_argument("--no-v2-features", action="store_true")
    ap.add_argument("--feature-preset", choices=FEATURE_PRESETS, default="full")
    ap.add_argument("--map", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--mid", type=int, default=128)
    ap.add_argument("--num-blocks", type=int, default=2)
    ap.add_argument("--snap-hidden", type=int, default=48)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--pairwise-rank-weight", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--hazard-pos-weight", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--checkpoint", default="model_stgcn/checkpoints/v3/factorized_phase.pt")
    ap.add_argument("--cached", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
