"""Evaluate DynamicSTGNN phase checkpoints with baseline-compatible metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from model_stgcn.bootstrap_ci import cluster_bootstrap_ci
from model_stgcn.dataset import CachedPhaseMatchDataset, LazyPhaseMatchDataset, phase_collate
from model_stgcn.dyn_stgnn import DynamicSTGNN, dynamic_stgnn_loss
from model_stgcn.edge_modes import EDGE_MODE_CHOICES, apply_edge_mode, edge_active_mask
from model_stgcn.splits import load_or_create_split
from stdb.queries import PHASE_EDGE_TYPES, PHASE_FEATURE_DIM, list_phase_eligible_match_ids

logger = logging.getLogger("model_stgcn.eval_phase")


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
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan"),
        "pos_rate": float(np.mean(y_true)) if y_true.size else float("nan"),
        "n": int(y_true.size),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc"] = float("nan")
    try:
        out["ap"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        out["ap"] = float("nan")
    return out


def _dataset(ids: list[str], cached: bool, *, include_combat_in: bool = False):
    cls = CachedPhaseMatchDataset if cached else LazyPhaseMatchDataset
    return cls(ids, include_combat_in=include_combat_in)


def _checkpoint_splits(ckpt: dict[str, Any], available_ids: list[str]) -> tuple[list[str], list[str], list[str]] | None:
    required = ("train_ids", "val_ids", "test_ids")
    if not all(k in ckpt for k in required):
        return None
    available = set(available_ids)
    splits = tuple(list(ckpt[k]) for k in required)
    missing = [mid for part in splits for mid in part if mid not in available]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"checkpoint split has {len(missing)} ids outside current candidate set: {preview}")
    return splits  # type: ignore[return-value]


def _resolve_eval_ids(args: argparse.Namespace, ckpt: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    ids = list_phase_eligible_match_ids(args.map, args.mode, args.limit)
    if len(ids) < 2:
        raise SystemExit(f"not enough phase-eligible matches: {len(ids)}")

    if args.split_json:
        train_ids, val_ids, test_ids = load_or_create_split(
            ids,
            split_json=args.split_json,
            seed=args.seed,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            metadata={
                "map": args.map,
                "mode": args.mode,
                "limit": args.limit,
                "source": "model_stgcn.eval_phase",
            },
        )
    else:
        ckpt_splits = _checkpoint_splits(ckpt, ids)
        if ckpt_splits is not None:
            train_ids, val_ids, test_ids = ckpt_splits
        else:
            train_ids, val_ids, test_ids = load_or_create_split(
                ids,
                split_json=None,
                seed=args.seed,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
            )

    split_map = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "all": train_ids + val_ids + test_ids,
    }
    return split_map[args.split], {
        "total": len(ids),
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
        "eval": len(split_map[args.split]),
    }


def run(args: argparse.Namespace) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    eval_ids, split_counts = _resolve_eval_ids(args, ckpt)
    logger.info("split=%s counts=%s", args.split, split_counts)

    feature_dim = int(ckpt.get("feature_dim", PHASE_FEATURE_DIM))
    edge_types = list(ckpt.get("edge_types", PHASE_EDGE_TYPES))
    edge_mode = args.edge_mode or ckpt_args.get("edge_mode", "full")
    model = DynamicSTGNN(
        in_features=feature_dim,
        hidden=int(ckpt_args.get("hidden", args.hidden)),
        mid=int(ckpt_args.get("mid", args.mid)),
        num_blocks=int(ckpt_args.get("num_blocks", args.num_blocks)),
        num_edge_types=len(edge_types),
        dropout=float(ckpt_args.get("dropout", args.dropout)),
    )
    model.load_state_dict(ckpt["model_state"])
    device = _device(args.device)
    model.to(device)
    model.eval()
    active_mask = edge_active_mask(edge_mode, edge_types=edge_types, device=device)

    ds = _dataset(eval_ids, cached=args.cached, include_combat_in="combat_in" in edge_types)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=phase_collate, num_workers=0)

    hazard_prob: list[np.ndarray] = []
    hazard_true: list[np.ndarray] = []
    hazard_match: list[np.ndarray] = []
    top5_prob: list[np.ndarray] = []
    top5_true: list[np.ndarray] = []
    top10_prob: list[np.ndarray] = []
    top10_true: list[np.ndarray] = []
    rank_pred: list[np.ndarray] = []
    rank_true: list[np.ndarray] = []
    total_loss = 0.0
    total_matches = 0
    loss_parts = {"l_hazard": 0.0, "l_rank": 0.0, "l_topk": 0.0}

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            A = apply_edge_mode(batch["A"].float(), edge_mode, edge_types=edge_types)
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

            bs = int(batch["X"].size(0))
            total_loss += float(loss.item()) * bs
            total_matches += bs
            for key in loss_parts:
                loss_parts[key] += float(parts[key]) * bs

            hmask = batch["target_mask"].bool()
            tmask = batch["team_mask"].bool()
            hazard_prob.append(torch.sigmoid(yh)[hmask].detach().cpu().numpy())
            hazard_true.append(batch["y_hazard"][hmask].detach().cpu().numpy())
            # match-level bootstrap용: hmask 선택 순서(row-major)에 맞춘 match_id
            cell_counts = hmask.reshape(hmask.shape[0], -1).sum(dim=1).cpu().numpy()
            hazard_match.append(np.repeat(np.asarray(batch["match_id"]), cell_counts))

            topk_prob = torch.sigmoid(yk)
            top5_prob.append(topk_prob[:, :, 0][tmask].detach().cpu().numpy())
            top5_true.append(batch["y_topk"][:, :, 0][tmask].detach().cpu().numpy())
            top10_prob.append(topk_prob[:, :, 1][tmask].detach().cpu().numpy())
            top10_true.append(batch["y_topk"][:, :, 1][tmask].detach().cpu().numpy())
            rank_pred.append(yr[tmask].detach().cpu().numpy())
            rank_true.append(batch["y_rank"][tmask].detach().cpu().numpy())

    yh_true = np.concatenate(hazard_true) if hazard_true else np.array([])
    yh_prob = np.concatenate(hazard_prob) if hazard_prob else np.array([])
    yh_match = np.concatenate(hazard_match) if hazard_match else np.array([])
    y5_true = np.concatenate(top5_true) if top5_true else np.array([])
    y5_prob = np.concatenate(top5_prob) if top5_prob else np.array([])
    y10_true = np.concatenate(top10_true) if top10_true else np.array([])
    y10_prob = np.concatenate(top10_prob) if top10_prob else np.array([])
    yr_true = np.concatenate(rank_true) if rank_true else np.array([])
    yr_pred = np.concatenate(rank_pred) if rank_pred else np.array([])

    denom = max(total_matches, 1)
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "split_counts": split_counts,
        "feature_dim": feature_dim,
        "edge_types": edge_types,
        "edge_mode": edge_mode,
        "loss": total_loss / denom,
        "loss_parts": {k: v / denom for k, v in loss_parts.items()},
        "hazard": _binary_metrics(yh_true, yh_prob),
        "top5": _binary_metrics(y5_true, y5_prob),
        "top10": _binary_metrics(y10_true, y10_prob),
        "rank_mae_norm": float(mean_absolute_error(yr_true, yr_pred)) if yr_true.size else float("nan"),
        "n_rank": int(yr_true.size),
        "edge_weights": [w.detach().cpu().tolist() for w in model.edge_weights(edge_active_mask=active_mask)],
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_val": ckpt.get("best_val"),
    }

    if args.bootstrap > 0 and yh_true.size:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

        result["hazard_ci"] = {
            "n_boot": args.bootstrap,
            "auc": cluster_bootstrap_ci(yh_true, yh_prob, yh_match, roc_auc_score, n_boot=args.bootstrap, seed=args.seed),
            "ap": cluster_bootstrap_ci(yh_true, yh_prob, yh_match, average_precision_score, n_boot=args.bootstrap, seed=args.seed),
            "brier": cluster_bootstrap_ci(yh_true, yh_prob, yh_match, brier_score_loss, n_boot=args.bootstrap, seed=args.seed),
        }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        logger.info("wrote %s", out_path)
    logger.info("result=%s", result)
    return result


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="model_stgcn/checkpoints/dynamic_stgnn_phase.pt")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    ap.add_argument("--map", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--mid", type=int, default=128)
    ap.add_argument("--num-blocks", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--hazard-pos-weight", type=float, default=1.0)
    ap.add_argument("--edge-mode", default=None, choices=EDGE_MODE_CHOICES)
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="match-level bootstrap 반복 수 (0=off); hazard AUC/AP/Brier 95%% CI 산출",
    )
    ap.add_argument("--out", default="model_stgcn/checkpoints/eval_phase.json")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
