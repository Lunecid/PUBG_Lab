"""Train snapshot-resolution STGNN with phase-cut survival outputs.

This is the corrected design for the intended task:

    10s snapshot tensors -> causal ST encoder -> phase-cut hazard head

Phase-level tables are used for supervision only.  They are not the model's
primary input representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model_stgcn.snapshot_baselines import (
    SnapshotGRUNoGraphHazard,
    SnapshotHierGRUNoGraphHazard,
    SnapshotLastOnlyHazard,
    SnapshotLSTMNoGraphHazard,
    SnapshotTCNNoGraphHazard,
    SnapshotTransformerNoGraphHazard,
)
from model_stgcn.snapshot_edge_controls import apply_snapshot_edge_control
from model_stgcn.snapshot_phase_dataset import (
    CachedSnapshotPhaseDataset,
    LazySnapshotPhaseDataset,
    snapshot_phase_collate,
)
from model_stgcn.snapshot_phase_survival import (
    SnapshotPhaseSurvivalGNN,
    snapshot_phase_hazard_loss,
)
from model_stgcn.snapshot_relative_edge import SnapshotRelativeEdgeHazard
from model_stgcn.splits import load_or_create_split
from stdb.queries import PHASE_T_MAX, list_phase_eligible_match_ids
from stdb.snapshot_queries import SNAP_EDGE_TYPES, SNAP_FEATURE_DIM, SNAP_FEATURE_NAMES

logger = logging.getLogger("model_stgcn.train_snapshot_phase")

NO_GRAPH_VARIANTS = {
    "snapshot_last_only",
    "snapshot_tcn_no_graph",
    "snapshot_gru_no_graph",
    "snapshot_lstm_no_graph",
    "snapshot_transformer_no_graph",
    "snapshot_hier_gru_no_graph",
}
RGCN_VARIANTS = {"snapshot_rgcn", "snapshot_rgcn_edge_shuffle", "snapshot_gate_only"}
RELATIVE_EDGE_VARIANTS = {
    "snapshot_relative_edge",
    "snapshot_relative_edge_sparse",
    "snapshot_relative_edge_sparse_null",
    "snapshot_relative_edge_sparse_topology_null",
}

SNAPSHOT_FEATURE_PRESETS: dict[str, list[str]] = {
    "full": list(SNAP_FEATURE_NAMES),
    "trajectory": ["pos_x", "pos_y", "pos_z", "speed_mps"],
    "zone": [
        "rel_zone_x",
        "rel_zone_y",
        "dist_blue_boundary_m",
        "inside_blue",
        "dist_target_center_m",
    ],
    "viability": ["mean_health", "min_health", "alive_n"],
    "resource": ["vehicle_ratio"],
    "viability_resource": ["mean_health", "min_health", "alive_n", "vehicle_ratio"],
    "trajectory_zone": [
        "pos_x",
        "pos_y",
        "pos_z",
        "speed_mps",
        "rel_zone_x",
        "rel_zone_y",
        "dist_blue_boundary_m",
        "inside_blue",
        "dist_target_center_m",
    ],
}

SNAPSHOT_FEATURE_MECHANISMS: dict[str, str] = {
    "pos_x": "space",
    "pos_y": "space",
    "pos_z": "space",
    "rel_zone_x": "zone",
    "rel_zone_y": "zone",
    "speed_mps": "rotation",
    "dist_blue_boundary_m": "zone",
    "inside_blue": "zone",
    "dist_target_center_m": "zone",
    "mean_health": "roster",
    "min_health": "roster",
    "alive_n": "roster",
    "vehicle_ratio": "resource",
}

SNAPSHOT_EDGE_PRESETS: dict[str, list[str]] = {
    "full": list(SNAP_EDGE_TYPES),
    "proximity": ["proximity"],
    "combat": ["combat"],
    "none": [],
}

SNAPSHOT_EDGE_MECHANISMS: dict[str, str] = {
    "proximity": "relation",
    "combat": "combat",
}


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


def _make_dataset(
    ids: list[str],
    *,
    cached: bool,
    include_edges: bool,
    cache_dir: str | None = None,
    cache_label: str | None = None,
):
    cls = CachedSnapshotPhaseDataset if cached else LazySnapshotPhaseDataset
    if cached:
        return cls(
            ids,
            include_edges=include_edges,
            cache_dir=cache_dir,
            cache_label=cache_label,
        )
    return cls(ids, include_edges=include_edges)


def _make_model(args: argparse.Namespace) -> nn.Module:
    if args.model_variant in RGCN_VARIANTS:
        return SnapshotPhaseSurvivalGNN(
            snap_in_features=SNAP_FEATURE_DIM,
            snap_hidden=args.hidden,
            snap_mid=args.mid,
            snap_blocks=args.num_blocks,
            snap_edge_types=len(SNAP_EDGE_TYPES),
            t_phase=PHASE_T_MAX,
            dropout=args.dropout,
        )
    if args.model_variant in RELATIVE_EDGE_VARIANTS:
        relative_edge_gate = (
            "observed" if args.model_variant == "snapshot_relative_edge" else args.relative_edge_gate
        )
        relative_edge_null = (
            "feature_roll"
            if args.model_variant == "snapshot_relative_edge_sparse_null"
            else (
                "topology_roll"
                if args.model_variant == "snapshot_relative_edge_sparse_topology_null"
                else args.relative_edge_null
            )
        )
        return SnapshotRelativeEdgeHazard(
            SNAP_FEATURE_DIM,
            feature_names=list(SNAP_FEATURE_NAMES),
            hidden=args.hidden,
            num_layers=args.num_blocks,
            edge_hidden=args.edge_hidden,
            kernel_size=args.kernel_size,
            t_phase=PHASE_T_MAX,
            dropout=args.dropout,
            relative_edge_gate=relative_edge_gate,
            relative_edge_k=args.relative_edge_k,
            relative_edge_null=relative_edge_null,
        )
    if args.model_variant == "snapshot_last_only":
        return SnapshotLastOnlyHazard(
            SNAP_FEATURE_DIM,
            hidden=args.hidden,
            t_phase=PHASE_T_MAX,
            dropout=args.dropout,
        )
    if args.model_variant == "snapshot_tcn_no_graph":
        return SnapshotTCNNoGraphHazard(
            SNAP_FEATURE_DIM,
            hidden=args.hidden,
            num_layers=args.num_blocks,
            kernel_size=args.kernel_size,
            t_phase=PHASE_T_MAX,
            dropout=args.dropout,
        )
    if args.model_variant == "snapshot_gru_no_graph":
        return SnapshotGRUNoGraphHazard(
            SNAP_FEATURE_DIM,
            hidden=args.hidden,
            num_layers=args.num_blocks,
            t_phase=PHASE_T_MAX,
            dropout=args.dropout,
        )
    if args.model_variant == "snapshot_lstm_no_graph":
        return SnapshotLSTMNoGraphHazard(
            SNAP_FEATURE_DIM,
            hidden=args.hidden,
            num_layers=args.num_blocks,
            t_phase=PHASE_T_MAX,
            dropout=args.dropout,
        )
    if args.model_variant == "snapshot_transformer_no_graph":
        return SnapshotTransformerNoGraphHazard(
            SNAP_FEATURE_DIM,
            hidden=args.hidden,
            num_layers=args.num_blocks,
            num_heads=args.num_heads,
            t_phase=PHASE_T_MAX,
            dropout=args.dropout,
        )
    if args.model_variant == "snapshot_hier_gru_no_graph":
        return SnapshotHierGRUNoGraphHazard(
            SNAP_FEATURE_DIM,
            hidden=args.hidden,
            num_layers=args.num_blocks,
            t_phase=PHASE_T_MAX,
            dropout=args.dropout,
        )
    raise ValueError(f"unknown model_variant={args.model_variant}")


def _edge_control_for_variant(model_variant: str) -> str:
    if model_variant == "snapshot_rgcn_edge_shuffle":
        return "edge_shuffle"
    if model_variant == "snapshot_gate_only":
        return "gate_only"
    if model_variant == "snapshot_relative_edge":
        return "gate_only"
    if model_variant in {
        "snapshot_relative_edge_sparse",
        "snapshot_relative_edge_sparse_null",
        "snapshot_relative_edge_sparse_topology_null",
    }:
        return "observed"
    return "observed"


def _selected_snapshot_features(preset: str) -> list[str]:
    try:
        return list(SNAPSHOT_FEATURE_PRESETS[preset])
    except KeyError as exc:
        raise ValueError(f"unknown snapshot_feature_preset={preset}") from exc


def _selected_snapshot_edges(preset: str) -> list[str]:
    try:
        return list(SNAPSHOT_EDGE_PRESETS[preset])
    except KeyError as exc:
        raise ValueError(f"unknown snapshot_edge_preset={preset}") from exc


def _mask_snapshot_features(X_s: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.snapshot_feature_preset == "full":
        return X_s.float()
    keep = set(_selected_snapshot_features(args.snapshot_feature_preset))
    idx = [i for i, name in enumerate(SNAP_FEATURE_NAMES) if name in keep]
    mask = torch.zeros((X_s.shape[-1],), dtype=X_s.dtype, device=X_s.device)
    if idx:
        mask[idx] = 1.0
    return X_s.float() * mask.view(1, 1, 1, -1)


def _mask_snapshot_edges(A_s: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.snapshot_edge_preset == "full":
        return A_s.float()
    keep = set(_selected_snapshot_edges(args.snapshot_edge_preset))
    idx = [i for i, name in enumerate(SNAP_EDGE_TYPES) if name in keep]
    mask = torch.zeros((A_s.shape[2],), dtype=A_s.dtype, device=A_s.device)
    if idx:
        mask[idx] = 1.0
    return A_s.float() * mask.view(1, 1, -1, 1, 1)


def _batch_edge_control_generator(batch: dict, args: argparse.Namespace) -> torch.Generator:
    match_key = "\n".join(str(mid) for mid in batch.get("match_id", []))
    digest = hashlib.sha1(match_key.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return torch.Generator().manual_seed(args.edge_control_seed + offset)


def _forward_model(
    model: nn.Module,
    batch: dict,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    X_s = _mask_snapshot_features(batch["X_s"], args)
    node_mask_s = batch["node_mask_s"].bool()
    if args.model_variant in NO_GRAPH_VARIANTS:
        active_nodes = batch.get("team_mask")
        if active_nodes is not None:
            active_nodes = active_nodes.bool().any(dim=0)
        else:
            active_nodes = node_mask_s.any(dim=(0, 1))
        if bool(active_nodes.any()) and not bool(active_nodes.all()):
            hazard_active, cut_mask = model(
                X_s[:, :, active_nodes],
                batch["phase_of_snap"],
                node_mask_s[:, :, active_nodes],
            )
            hazard = X_s.new_zeros(
                (X_s.shape[0], hazard_active.shape[1], X_s.shape[2])
            )
            hazard[:, :, active_nodes] = hazard_active
            return hazard, cut_mask
        return model(
            X_s,
            batch["phase_of_snap"],
            node_mask_s,
        )
    A_s = _mask_snapshot_edges(batch["A_s"], args)
    edge_control = _edge_control_for_variant(args.model_variant)
    if edge_control != "observed":
        A_s = apply_snapshot_edge_control(
            A_s,
            batch["node_mask_s"].bool(),
            control=edge_control,
            generator=_batch_edge_control_generator(batch, args),
        )
    if args.model_variant in RELATIVE_EDGE_VARIANTS:
        return model(
            X_s,
            A_s,
            batch["phase_of_snap"],
            batch["node_mask_s"].bool(),
        )
    return model(
        X_s,
        A_s,
        batch["phase_of_snap"],
        batch["node_mask_s"].bool(),
    )


def _channel_contract(args: argparse.Namespace) -> dict:
    no_graph = args.model_variant in NO_GRAPH_VARIANTS
    relative_edge = args.model_variant in RELATIVE_EDGE_VARIANTS
    sparse_relative_edge = args.model_variant in {
        "snapshot_relative_edge_sparse",
        "snapshot_relative_edge_sparse_null",
        "snapshot_relative_edge_sparse_topology_null",
    }
    sparse_relative_edge_null = args.model_variant == "snapshot_relative_edge_sparse_null"
    sparse_relative_edge_topology_null = (
        args.model_variant == "snapshot_relative_edge_sparse_topology_null"
    )
    selected_features = _selected_snapshot_features(args.snapshot_feature_preset)
    selected_edges = [] if no_graph else _selected_snapshot_edges(args.snapshot_edge_preset)
    if no_graph:
        snapshot_edge_types: list[str] = []
    elif sparse_relative_edge_topology_null:
        snapshot_edge_types = selected_edges + ["knn_combat_sparse_topology_null"]
    elif sparse_relative_edge_null:
        snapshot_edge_types = selected_edges + ["knn_combat_sparse_relative_advantage_null"]
    elif sparse_relative_edge:
        snapshot_edge_types = selected_edges + ["knn_combat_sparse_relative_advantage"]
    elif relative_edge:
        snapshot_edge_types = selected_edges + ["contact_gated_relative_advantage"]
    else:
        snapshot_edge_types = selected_edges
    domain_mechanisms = sorted(
        set(
            SNAPSHOT_FEATURE_MECHANISMS[name]
            for name in selected_features
            if name in SNAPSHOT_FEATURE_MECHANISMS
        )
        | (
            set()
            if no_graph
            else {
                SNAPSHOT_EDGE_MECHANISMS[name]
                for name in selected_edges
                if name in SNAPSHOT_EDGE_MECHANISMS
            }
        )
    )
    if relative_edge:
        domain_mechanisms.append(
            "relative_topology_null_control"
            if sparse_relative_edge_topology_null
            else (
                "relative_advantage_null_control"
                if sparse_relative_edge_null
                else "relative_advantage"
            )
        )
    return {
        "model_variant": args.model_variant,
        "model_family": (
            (
                "M3_hierarchical_recurrent_survival_model"
                if args.model_variant == "snapshot_hier_gru_no_graph"
                else (
                    "M3_transformer_snapshot_survival_model"
                    if args.model_variant == "snapshot_transformer_no_graph"
                    else (
                        "M3_recurrent_snapshot_survival_model"
                        if args.model_variant in {"snapshot_gru_no_graph", "snapshot_lstm_no_graph"}
                        else "M2_snapshot_temporal_model"
                    )
                )
            )
            if no_graph
            else (
                "M6_edge_conditioned_relative_advantage_model"
                if relative_edge
                else "M5_snapshot_first_st_survival_model"
            )
        ),
        "temporal_resolution": "10s_snapshot",
        "output_resolution": "phase_cut",
        "graph_resolution": "none" if no_graph else "10s_snapshot",
        "relation_handling": (
            "none"
            if no_graph
            else (
                "edge_shuffle_null"
                if args.model_variant == "snapshot_rgcn_edge_shuffle"
                else (
                    (
                        (
                            (
                                "knn_combat_sparse_topology_roll_null"
                                if sparse_relative_edge_topology_null
                                else (
                                    "knn_combat_sparse_relative_edge_feature_roll_null"
                                    if sparse_relative_edge_null
                                    else "knn_combat_sparse_relative_edge"
                                )
                            )
                        )
                        if sparse_relative_edge
                        else "contact_gated_relative_edge"
                    )
                    if relative_edge
                    else (
                        "contact_gate_only"
                        if args.model_variant == "snapshot_gate_only"
                        else "relation_channels_preserved"
                    )
                )
            )
        ),
        "edge_control": "none" if no_graph else _edge_control_for_variant(args.model_variant),
        "edge_control_seed": None if no_graph else args.edge_control_seed,
        "snapshot_feature_preset": args.snapshot_feature_preset,
        "snapshot_edge_preset": "none" if no_graph else args.snapshot_edge_preset,
        "relative_edge_gate": (
            None
            if not relative_edge
            else ("observed_contact_gate" if not sparse_relative_edge else args.relative_edge_gate)
        ),
        "relative_edge_k": args.relative_edge_k if sparse_relative_edge else None,
        "relative_edge_null": (
            (
                "topology_roll"
                if sparse_relative_edge_topology_null
                else "feature_roll"
            )
            if (sparse_relative_edge_topology_null or sparse_relative_edge_null)
            else (args.relative_edge_null if relative_edge else None)
        ),
        "feature_clocks_consumed": ["snapshot_10s"],
        "input_channels_consumed": (
            ["C1_snapshot_team_state"]
            if (no_graph or not selected_edges)
            else ["C1_snapshot_team_state", "C2_snapshot_team_relation"]
        ),
        "supervision_channels": ["C8_label_mask"],
        "tensor_owners_consumed": (
            ["X_s"]
            if no_graph
            else (
                ["X_s", "A_s", "E_s_derived_topology_null"]
                if sparse_relative_edge_topology_null
                else ["X_s", "A_s", "E_s_derived_null"]
                if sparse_relative_edge_null
                else (["X_s", "A_s", "E_s_derived"] if relative_edge else ["X_s", "A_s"])
            )
        ),
        "snapshot_feature_names": selected_features,
        "snapshot_feature_names_available": list(SNAP_FEATURE_NAMES),
        "snapshot_edge_types": snapshot_edge_types,
        "snapshot_edge_feature_names": (
            list(SnapshotRelativeEdgeHazard.edge_feature_names) if relative_edge else []
        ),
        "domain_mechanisms_consumed": domain_mechanisms,
        "heads_enabled": ["next_phase_hazard"],
        "label_and_mask_inputs_forbidden": True,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    train: bool,
    opt: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train(train)
    sums: dict[str, float] = {}
    total_n = 0
    for batch in loader:
        batch = _move(batch, device)
        hazard, cut_mask = _forward_model(model, batch, args)
        loss, parts = snapshot_phase_hazard_loss(
            hazard,
            batch["y_hazard"].float(),
            batch["target_mask"],
            cut_mask=cut_mask,
            hazard_pos_weight=args.hazard_pos_weight,
        )
        if train:
            assert opt is not None
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        bs = int(batch["X_s"].shape[0])
        total_n += bs
        for key, value in parts.items():
            sums[key] = sums.get(key, 0.0) + float(value) * bs
    return {key: value / max(total_n, 1) for key, value in sums.items()}


def _loader_worker_kwargs(workers: int, prefetch_factor: int) -> dict:
    kwargs = {"num_workers": workers}
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def train_loop(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ids = list_phase_eligible_match_ids(args.map, args.mode, args.limit)
    if len(ids) < 2:
        raise SystemExit(f"not enough labeled matches: {len(ids)}")
    train_ids, val_ids, test_ids = load_or_create_split(
        ids,
        split_json=args.split_json,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        metadata={
            "source": "model_stgcn.train_snapshot_phase",
            "map": args.map,
            "mode": args.mode,
            "limit": args.limit,
            "input_resolution": "10s_snapshot",
            "output_resolution": "phase_cut",
        },
    )
    logger.info("ids total=%d train=%d val=%d test=%d", len(ids), len(train_ids), len(val_ids), len(test_ids))

    include_edges = args.model_variant not in NO_GRAPH_VARIANTS
    logger.info(
        "build train dataset include_edges=%s cached=%s ids=%d cache_dir=%s",
        include_edges,
        args.cached,
        len(train_ids),
        args.snapshot_cache_dir,
    )
    train_ds = _make_dataset(
        train_ids,
        cached=args.cached,
        include_edges=include_edges,
        cache_dir=args.snapshot_cache_dir,
        cache_label="train",
    )
    logger.info(
        "build val dataset include_edges=%s cached=%s ids=%d cache_dir=%s",
        include_edges,
        args.cached,
        len(val_ids),
        args.snapshot_cache_dir,
    )
    val_ds = _make_dataset(
        val_ids,
        cached=args.cached,
        include_edges=include_edges,
        cache_dir=args.snapshot_cache_dir,
        cache_label="val",
    )
    logger.info(
        "build test dataset include_edges=%s cached=%s ids=%d cache_dir=%s",
        include_edges,
        args.cached,
        len(test_ids),
        args.snapshot_cache_dir,
    )
    test_ds = _make_dataset(
        test_ids,
        cached=args.cached,
        include_edges=include_edges,
        cache_dir=args.snapshot_cache_dir,
        cache_label="test",
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("empty train/val dataset")

    logger.info(
        "dataset include_edges=%s cached=%s workers=%d eval_workers=%d",
        include_edges,
        args.cached,
        args.workers,
        args.eval_workers,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=snapshot_phase_collate,
        **_loader_worker_kwargs(args.workers, args.prefetch_factor),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=snapshot_phase_collate,
        **_loader_worker_kwargs(args.eval_workers, args.prefetch_factor),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=snapshot_phase_collate,
        **_loader_worker_kwargs(args.eval_workers, args.prefetch_factor),
    ) if len(test_ds) else None

    device = _device(args.device)
    model = _make_model(args).to(device)
    contract = _channel_contract(args)
    logger.info(
        "variant=%s device=%s params=%.3fM input=10s_snapshots output=phase_cut_hazard",
        args.model_variant,
        device,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

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
        history.append({"epoch": epoch, "train": tr, "val": va, "lr": sched.get_last_lr()[0]})
        logger.info(
            "epoch %03d train=%.4f acc=%.3f val=%.4f val_acc=%.3f lr=%.5g",
            epoch,
            tr.get("loss", float("nan")),
            tr.get("hazard_acc", float("nan")),
            va.get("loss", float("nan")),
            va.get("hazard_acc", float("nan")),
            sched.get_last_lr()[0],
        )
        if va.get("loss", float("inf")) < best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "args": vars(args),
                    "model": model.__class__.__name__,
                    "model_variant": args.model_variant,
                    "snapshot_feature_preset": args.snapshot_feature_preset,
                    "snapshot_edge_preset": args.snapshot_edge_preset,
                    "input_resolution": "10s_snapshot",
                    "output_resolution": "phase_cut",
                    "snap_feature_dim": SNAP_FEATURE_DIM,
                    "snap_feature_names": list(SNAP_FEATURE_NAMES),
                    "snap_edge_types": list(contract["snapshot_edge_types"]),
                    "snap_edge_feature_names": list(contract["snapshot_edge_feature_names"]),
                    "channel_contract": contract,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "test_ids": test_ids,
                },
                ckpt_path,
            )
            logger.info("saved best checkpoint to %s", ckpt_path)

    hist_path = ckpt_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    summary: dict[str, object] = {
        "model_variant": args.model_variant,
        "checkpoint": str(ckpt_path),
        "history": str(hist_path),
        "map": args.map,
        "mode": args.mode,
        "limit": args.limit,
        "seed": args.seed,
        "snapshot_feature_preset": args.snapshot_feature_preset,
        "snapshot_edge_preset": args.snapshot_edge_preset,
        "snapshot_cache_dir": args.snapshot_cache_dir,
        "best_val": best_val,
        "best_epoch": min(history, key=lambda r: r["val"]["loss"])["epoch"] if history else None,
        "split_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "train_loaded": len(train_ds),
            "val_loaded": len(val_ds),
            "test_loaded": len(test_ds),
        },
        "channel_contract": contract,
    }
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state"])
    with torch.no_grad():
        summary["best_val_metrics"] = _run_epoch(model, val_loader, device, args, train=False)
        if test_loader is not None:
            summary["test_metrics"] = _run_epoch(model, test_loader, device, args, train=False)
    summary_path = ckpt_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "done in %.1fs best_val=%.4f history=%s summary=%s",
        time.time() - started,
        best_val,
        hist_path,
        summary_path,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--mid", type=int, default=96)
    ap.add_argument("--num-blocks", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--edge-hidden", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--hazard-pos-weight", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--checkpoint", default="model_stgcn/checkpoints/snapshot_phase_survival.pt")
    ap.add_argument(
        "--model-variant",
        default="snapshot_rgcn",
        choices=[
            "snapshot_rgcn",
            "snapshot_rgcn_edge_shuffle",
            "snapshot_gate_only",
            "snapshot_relative_edge",
            "snapshot_relative_edge_sparse",
            "snapshot_relative_edge_sparse_null",
            "snapshot_relative_edge_sparse_topology_null",
            "snapshot_last_only",
            "snapshot_tcn_no_graph",
            "snapshot_gru_no_graph",
            "snapshot_lstm_no_graph",
            "snapshot_transformer_no_graph",
            "snapshot_hier_gru_no_graph",
        ],
    )
    ap.add_argument("--edge-control-seed", type=int, default=20260622)
    ap.add_argument(
        "--snapshot-feature-preset",
        default="full",
        choices=sorted(SNAPSHOT_FEATURE_PRESETS),
        help="Mask snapshot node features to a prior-aligned feature channel.",
    )
    ap.add_argument(
        "--snapshot-edge-preset",
        default="full",
        choices=sorted(SNAPSHOT_EDGE_PRESETS),
        help="Mask snapshot relation channels for graph variants.",
    )
    ap.add_argument(
        "--relative-edge-gate",
        default="knn_combat",
        choices=["observed", "knn_combat"],
    )
    ap.add_argument("--relative-edge-k", type=int, default=5)
    ap.add_argument(
        "--relative-edge-null",
        default="none",
        choices=["none", "feature_roll", "topology_roll"],
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--snapshot-cache-dir", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--eval-workers", type=int, default=0)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    return ap.parse_args()


if __name__ == "__main__":
    train_loop(parse_args())
