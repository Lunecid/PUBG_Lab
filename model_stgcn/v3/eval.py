"""Evaluate V3 factorized checkpoints.

The primary metric is next-phase elimination hazard over at-risk cells.  Rank
and top-k are evaluated as causal cut-phase forecasts over the same at-risk
cells, not as full-match retrospective reconstructions.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from model_stgcn.edge_modes import edge_active_mask
from model_stgcn.splits import load_or_create_split
from model_stgcn.v3.cohorts import resolve_map_mode
from model_stgcn.v3.edge_controls import EDGE_CONTROL_CHOICES, apply_phase_edge_control
from model_stgcn.v3.factorized_data import (
    CachedFactorizedMRDataset,
    CachedFactorizedPhaseDataset,
    LazyFactorizedMRDataset,
    LazyFactorizedPhaseDataset,
    factorized_collate,
)
from model_stgcn.v3.models import FactorizedMRGNN, FactorizedPhaseGNN
from model_stgcn.v3.spec import FEATURE_PRESETS
from stdb.queries import list_phase_eligible_match_ids
from stdb.snapshot_queries import SNAP_FEATURE_DIM

logger = logging.getLogger("model_stgcn.v3.eval")


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _ece(y: np.ndarray, p: np.ndarray, *, n_bins: int = 10) -> float:
    if y.size == 0:
        return float("nan")
    p = np.clip(p.astype(np.float64), 0.0, 1.0)
    y = y.astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        weight = float(mask.mean())
        ece += weight * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def _binary_metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(np.int64)
    p = np.clip(np.asarray(p).astype(np.float64), 1e-7, 1.0 - 1e-7)
    pred = (p >= 0.5).astype(np.int64)
    out = {
        "n": int(y.size),
        "pos_rate": float(y.mean()) if y.size else float("nan"),
        "acc": float(accuracy_score(y, pred)) if y.size else float("nan"),
        "bce": float(log_loss(y, p, labels=[0, 1])) if y.size else float("nan"),
        "brier": float(brier_score_loss(y, p)) if y.size else float("nan"),
        "ece": _ece(y, p),
    }
    out["auc"] = float(roc_auc_score(y, p)) if y.size and y.min() != y.max() else float("nan")
    out["ap"] = float(average_precision_score(y, p)) if y.size and y.min() != y.max() else float("nan")
    return out


def _prefixed(prefix: str, data: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in data.items()}


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks without requiring scipy."""
    a = np.asarray(a)
    if a.size == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks_sorted = np.empty(a.size, dtype=np.float64)
    start = 0
    while start < a.size:
        end = start + 1
        while end < a.size and sorted_a[end] == sorted_a[start]:
            end += 1
        # 1-based average rank for ties.
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    rt = _rankdata(y_true)
    rp = _rankdata(y_pred)
    if np.std(rt) == 0 or np.std(rp) == 0:
        return float("nan")
    return float(np.corrcoef(rt, rp)[0, 1])


def _pairwise_rank_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    if y_true.size < 2:
        return 0, 0
    true_diff = y_true[:, None] - y_true[None, :]
    pred_diff = y_pred[:, None] - y_pred[None, :]
    upper = np.triu(np.ones_like(true_diff, dtype=bool), k=1)
    valid = upper & (np.abs(true_diff) > 1e-8) & (np.abs(pred_diff) > 1e-8)
    if not np.any(valid):
        return 0, 0
    correct = np.sign(true_diff[valid]) == np.sign(pred_diff[valid])
    return int(correct.sum()), int(valid.sum())


def _rank_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    pair_correct: int,
    pair_total: int,
) -> dict[str, float | int]:
    return {
        "n": int(y_true.size),
        "mae_norm": float(mean_absolute_error(y_true, y_pred)) if y_true.size else float("nan"),
        "spearman": _spearman(y_true, y_pred),
        "pairwise_acc": float(pair_correct / pair_total) if pair_total else float("nan"),
        "pairwise_n": int(pair_total),
    }


def _new_phase_bin() -> dict[str, Any]:
    return {
        "matches": set(),
        "hazard_y": [],
        "hazard_p": [],
        "rank_y": [],
        "rank_p": [],
        "top5_y": [],
        "top5_p": [],
        "top10_y": [],
        "top10_p": [],
        "pair_correct": 0,
        "pair_total": 0,
    }


def _concat(parts: list[np.ndarray], *, dtype: np.dtype) -> np.ndarray:
    if not parts:
        return np.asarray([], dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def _phase_metrics(phase_bins: dict[int, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cut_phase, data in sorted(phase_bins.items()):
        hy = _concat(data["hazard_y"], dtype=np.int64)
        hp = _concat(data["hazard_p"], dtype=np.float64)
        ry = _concat(data["rank_y"], dtype=np.float64)
        rp = _concat(data["rank_p"], dtype=np.float64)
        t5y = _concat(data["top5_y"], dtype=np.int64)
        t5p = _concat(data["top5_p"], dtype=np.float64)
        t10y = _concat(data["top10_y"], dtype=np.int64)
        t10p = _concat(data["top10_p"], dtype=np.float64)
        out[str(cut_phase)] = {
            "cut_phase": int(cut_phase),
            "n_matches": int(len(data["matches"])),
            "hazard": _binary_metric(hy, hp),
            "top5": _binary_metric(t5y, t5p),
            "top10": _binary_metric(t10y, t10p),
            "rank": _rank_metrics(
                ry,
                rp,
                pair_correct=int(data["pair_correct"]),
                pair_total=int(data["pair_total"]),
            ),
        }
    return out


def _write_prediction_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "edge_mode",
        "edge_control",
        "match_id",
        "team_id",
        "phase_slot",
        "cut_phase",
        "hazard_true",
        "hazard_prob",
        "final_rank_norm",
        "rank_pred_norm",
        "final_top5",
        "top5_prob",
        "final_top10",
        "top10_prob",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _dataset(ids: list[str], ckpt_args: dict[str, Any], *, cached: bool):
    model_name = ckpt_args.get("model", "phase")
    cls = CachedFactorizedMRDataset if model_name == "mr" and cached else None
    if model_name == "mr" and not cached:
        cls = LazyFactorizedMRDataset
    if model_name != "mr" and cached:
        cls = CachedFactorizedPhaseDataset
    if model_name != "mr" and not cached:
        cls = LazyFactorizedPhaseDataset
    return cls(
        ids,
        include_v2_features=not ckpt_args.get("no_v2_features", False),
        include_combat_in=ckpt_args.get("combat_in", False),
        feature_preset=ckpt_args.get("feature_preset", "full"),
    )


def _build_model(ckpt: dict[str, Any], sample: dict) -> torch.nn.Module:
    args = ckpt.get("args", {})
    node_dim = int(sample["X_node"].shape[-1])
    global_dim = int(sample["G"].shape[-1])
    edge_types = list(sample["edge_types"])
    if args.get("model", "phase") == "mr":
        return FactorizedMRGNN(
            snap_features=SNAP_FEATURE_DIM,
            node_features=node_dim,
            global_features=global_dim,
            snap_hidden=args.get("snap_hidden", 48),
            phase_hidden=args.get("hidden", 64),
            phase_mid=args.get("mid", 128),
            phase_blocks=args.get("num_blocks", 2),
            phase_edge_types=len(edge_types),
            graph_encoder=args.get("graph_encoder", "rgcn"),
            global_fusion=args.get("global_fusion", "concat"),
            team_decoder=args.get("team_decoder", "mlp"),
            decoder_heads=args.get("decoder_heads", 4),
            decoder_layers=args.get("decoder_layers", 1),
            dropout=args.get("dropout", 0.1),
        )
    return FactorizedPhaseGNN(
        node_features=node_dim,
        global_features=global_dim,
        hidden=args.get("hidden", 64),
        mid=args.get("mid", 128),
        num_blocks=args.get("num_blocks", 2),
        num_edge_types=len(edge_types),
        graph_encoder=args.get("graph_encoder", "rgcn"),
        global_fusion=args.get("global_fusion", "concat"),
        team_decoder=args.get("team_decoder", "mlp"),
        decoder_heads=args.get("decoder_heads", 4),
        decoder_layers=args.get("decoder_layers", 1),
        dropout=args.get("dropout", 0.1),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    cohort_arg = args.cohort
    map_arg = args.map
    mode_arg = args.mode
    if not (cohort_arg or map_arg or mode_arg):
        ckpt_cohort = ckpt_args.get("cohort")
        if ckpt_cohort:
            cohort_arg = ckpt_cohort
        else:
            map_arg = ckpt_args.get("map")
            mode_arg = ckpt_args.get("mode")
    try:
        map_filter, mode_filter, cohort_spec = resolve_map_mode(
            cohort=cohort_arg,
            map_name=map_arg,
            mode=mode_arg,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.map = map_filter
    args.mode = mode_filter
    args.cohort = cohort_spec.label if cohort_spec else cohort_arg
    args.cohort_slug = cohort_spec.slug if cohort_spec else None
    ids = list_phase_eligible_match_ids(args.map, args.mode, args.limit)
    if args.split_json:
        train_ids, val_ids, test_ids = load_or_create_split(
            ids,
            split_json=args.split_json,
            seed=args.seed,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            metadata={"source": "model_stgcn.v3.eval"},
        )
    else:
        train_ids = list(ckpt.get("train_ids", []))
        val_ids = list(ckpt.get("val_ids", []))
        test_ids = list(ckpt.get("test_ids", []))
        if not test_ids:
            train_ids, val_ids, test_ids = load_or_create_split(
                ids,
                seed=args.seed,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
            )
    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    eval_ids = split_map[args.split]
    ds = _dataset(eval_ids, ckpt_args, cached=args.cached)
    if len(ds) == 0:
        raise SystemExit("empty eval dataset")
    sample = ds[0]
    model = _build_model(ckpt, sample)
    model.load_state_dict(ckpt["model_state"])
    device = _device(args.device)
    model.to(device).eval()
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=factorized_collate)
    edge_types = sample["edge_types"]
    active_mask = edge_active_mask(
        args.edge_mode or ckpt_args.get("edge_mode", "full"),
        edge_types,
        device=device,
    )
    edge_control = args.edge_control or ckpt_args.get("edge_control", "observed")
    edge_control_generator = None
    if edge_control != "observed":
        edge_control_generator = torch.Generator()
        edge_control_generator.manual_seed(int(args.edge_control_seed))

    probs: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    rank_pred: list[np.ndarray] = []
    rank_true: list[np.ndarray] = []
    top5_prob: list[np.ndarray] = []
    top5_true: list[np.ndarray] = []
    top10_prob: list[np.ndarray] = []
    top10_true: list[np.ndarray] = []
    pair_correct = 0
    pair_total = 0
    phase_bins: dict[int, dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            A_phase = apply_phase_edge_control(
                moved["A"].float(),
                moved["node_mask"].bool(),
                control=edge_control,
                generator=edge_control_generator,
            )
            if ckpt_args.get("model", "phase") == "mr":
                hazard, _surv, rank, topk = model(
                    moved["X_s"].float(),
                    moved["A_s"].float(),
                    moved["phase_of_snap"],
                    moved["node_mask_s"].bool(),
                    moved["X_node"].float(),
                    moved["G"].float(),
                    A_phase,
                    phase_edge_mask=active_mask,
                    phase_mask=moved["phase_mask"],
                    team_mask=moved["team_mask"],
                )
            else:
                hazard, rank, topk = model(
                    moved["X_node"].float(),
                    moved["G"].float(),
                    A_phase,
                    edge_active_mask=active_mask,
                    phase_mask=moved["phase_mask"],
                    team_mask=moved["team_mask"],
                )
            p = torch.sigmoid(hazard).detach().cpu().numpy()
            y = batch["y_hazard"].numpy()
            m = batch["target_mask"].numpy().astype(bool)
            probs.append(p[m])
            trues.append(y[m])
            forecast_mask = batch["target_mask"].numpy().astype(bool)
            rank_np = rank.detach().cpu().numpy()
            y_rank_np = batch["y_rank"].numpy()
            topk_np = torch.sigmoid(topk).detach().cpu().numpy()
            y_topk_np = batch["y_topk"].numpy()
            y_rank_cut = np.broadcast_to(y_rank_np[:, None, :], rank_np.shape)
            y_topk_cut = np.broadcast_to(y_topk_np[:, None, :, :], topk_np.shape)
            rank_pred.append(rank_np[forecast_mask])
            rank_true.append(y_rank_cut[forecast_mask])
            top5_prob.append(topk_np[:, :, :, 0][forecast_mask])
            top5_true.append(y_topk_cut[:, :, :, 0][forecast_mask])
            top10_prob.append(topk_np[:, :, :, 1][forecast_mask])
            top10_true.append(y_topk_cut[:, :, :, 1][forecast_mask])
            for b in range(forecast_mask.shape[0]):
                phase_indices = batch["phase_indices"][b]
                match_id = str(batch["match_id"][b])
                team_ids = batch["team_ids"][b]
                for t in range(forecast_mask.shape[1]):
                    cut_phase = int(phase_indices[t]) if t < len(phase_indices) and phase_indices[t] else t + 1
                    if cut_phase <= 0:
                        continue
                    valid = forecast_mask[b, t]
                    if not np.any(valid):
                        continue
                    phase_bin = phase_bins.setdefault(cut_phase, _new_phase_bin())
                    phase_bin["matches"].add(match_id)
                    phase_bin["hazard_y"].append(y[b, t][valid].astype(np.int64))
                    phase_bin["hazard_p"].append(p[b, t][valid].astype(np.float64))
                    phase_bin["rank_y"].append(y_rank_np[b][valid].astype(np.float64))
                    phase_bin["rank_p"].append(rank_np[b, t][valid].astype(np.float64))
                    phase_bin["top5_y"].append(y_topk_np[b, valid, 0].astype(np.int64))
                    phase_bin["top5_p"].append(topk_np[b, t, valid, 0].astype(np.float64))
                    phase_bin["top10_y"].append(y_topk_np[b, valid, 1].astype(np.int64))
                    phase_bin["top10_p"].append(topk_np[b, t, valid, 1].astype(np.float64))
                    correct, total = _pairwise_rank_counts(y_rank_np[b][valid], rank_np[b, t][valid])
                    pair_correct += correct
                    pair_total += total
                    phase_bin["pair_correct"] += correct
                    phase_bin["pair_total"] += total
                    if args.pred_out:
                        for i, team_id in enumerate(team_ids):
                            if not valid[i] or not team_id:
                                continue
                            prediction_rows.append(
                                {
                                    "split": args.split,
                                    "edge_mode": args.edge_mode or ckpt_args.get("edge_mode", "full"),
                                    "edge_control": edge_control,
                                    "match_id": match_id,
                                    "team_id": int(team_id),
                                    "phase_slot": int(t + 1),
                                    "cut_phase": int(cut_phase),
                                    "hazard_true": int(y[b, t, i]),
                                    "hazard_prob": float(p[b, t, i]),
                                    "final_rank_norm": float(y_rank_np[b, i]),
                                    "rank_pred_norm": float(rank_np[b, t, i]),
                                    "final_top5": int(y_topk_np[b, i, 0]),
                                    "top5_prob": float(topk_np[b, t, i, 0]),
                                    "final_top10": int(y_topk_np[b, i, 1]),
                                    "top10_prob": float(topk_np[b, t, i, 1]),
                                }
                            )
    P = np.concatenate(probs)
    Y = np.concatenate(trues).astype(np.int64)
    R_pred = np.concatenate(rank_pred) if rank_pred else np.asarray([], dtype=np.float32)
    R_true = np.concatenate(rank_true) if rank_true else np.asarray([], dtype=np.float32)
    T5_p = np.concatenate(top5_prob) if top5_prob else np.asarray([], dtype=np.float32)
    T5_y = np.concatenate(top5_true).astype(np.int64) if top5_true else np.asarray([], dtype=np.int64)
    T10_p = np.concatenate(top10_prob) if top10_prob else np.asarray([], dtype=np.float32)
    T10_y = np.concatenate(top10_true).astype(np.int64) if top10_true else np.asarray([], dtype=np.int64)
    hazard_metrics = _binary_metric(Y, P)
    out = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "model": ckpt_args.get("model", "phase"),
        "graph_encoder": ckpt_args.get("graph_encoder", "rgcn"),
        "global_fusion": ckpt_args.get("global_fusion", "concat"),
        "team_decoder": ckpt_args.get("team_decoder", "mlp"),
        "edge_mode": args.edge_mode or ckpt_args.get("edge_mode", "full"),
        "edge_control": edge_control,
        "feature_preset": ckpt_args.get("feature_preset", "full"),
        "cohort": args.cohort,
        "cohort_slug": args.cohort_slug,
        "map": args.map,
        "mode": args.mode,
        "metric_unit": "at_risk_team_phase",
        "phase_metric_unit": "at_risk_team_phase_by_cut_phase",
        **hazard_metrics,
        **_prefixed("hazard", hazard_metrics),
        **_prefixed("top5", _binary_metric(T5_y, T5_p)),
        **_prefixed("top10", _binary_metric(T10_y, T10_p)),
        "rank_n": int(R_true.size),
        "rank_eval_unit": "at_risk_team_phase",
        "topk_eval_unit": "at_risk_team_phase",
        "rank_mae_norm": float(mean_absolute_error(R_true, R_pred)) if R_true.size else float("nan"),
        "rank_spearman": _spearman(R_true, R_pred),
        "rank_pairwise_acc": float(pair_correct / pair_total) if pair_total else float("nan"),
        "rank_pairwise_n": int(pair_total),
        "phase_metrics": _phase_metrics(phase_bins),
    }
    logger.info("result=%s", out)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    if args.pred_out:
        _write_prediction_csv(args.pred_out, prediction_rows)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--edge-mode", default=None)
    ap.add_argument("--edge-control", choices=EDGE_CONTROL_CHOICES, default=None)
    ap.add_argument("--edge-control-seed", type=int, default=42)
    ap.add_argument("--cohort", default=None, help="named cohort, e.g. 'Erangel TPP squad'")
    ap.add_argument("--map", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--feature-preset", choices=FEATURE_PRESETS, default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--pred-out",
        default=None,
        help="optional CSV export of at-risk team-phase forecasts for live/phase trajectory analysis",
    )
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
