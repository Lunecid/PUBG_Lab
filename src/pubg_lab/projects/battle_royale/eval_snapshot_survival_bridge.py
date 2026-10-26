"""Evaluate snapshot-phase hazard checkpoints as cumulative survival forecasts.

The snapshot models predict next-phase elimination hazard.  This tool keeps
that training contract intact and derives the thesis-facing bridge:

    S_i(k) = product_{j <= k} (1 - hazard_i(j))

where S_i(k) is the cumulative survival score after phase cut k.  The derived
score is then compared with final top-k and rank labels for interpretation;
it is not a separately trained rank or top-k head.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
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

from pubg_lab.survival.provenance import git_provenance
from pubg_lab.survival.snapshot_phase_dataset import snapshot_phase_collate
from pubg_lab.survival.train_snapshot_phase import (
    NO_GRAPH_VARIANTS,
    _device,
    _forward_model,
    _loader_worker_kwargs,
    _make_dataset,
    _make_model,
    _move,
)

logger = logging.getLogger("tools.eval_snapshot_survival_bridge")


def _namespace_from_checkpoint(
    ckpt: dict[str, Any], cli_args: argparse.Namespace
) -> SimpleNamespace:
    saved_args = dict(ckpt.get("args") or {})
    saved_args["checkpoint"] = str(cli_args.checkpoint)
    saved_args.setdefault("model_variant", ckpt.get("model_variant", "snapshot_rgcn"))
    saved_args.setdefault("device", "cpu")
    saved_args.setdefault("cached", False)
    saved_args.setdefault("batch", 2)
    saved_args.setdefault("eval_workers", 0)
    saved_args.setdefault("prefetch_factor", 2)
    saved_args.setdefault("hidden", 48)
    saved_args.setdefault("mid", 96)
    saved_args.setdefault("num_blocks", 2)
    saved_args.setdefault("kernel_size", 3)
    saved_args.setdefault("edge_hidden", 16)
    saved_args.setdefault("dropout", 0.1)
    saved_args.setdefault("hazard_pos_weight", 1.0)
    saved_args.setdefault("grad_clip", 5.0)
    saved_args.setdefault("edge_control_seed", 20260622)
    saved_args.setdefault("snapshot_feature_preset", "full")
    saved_args.setdefault("snapshot_edge_preset", "full")
    saved_args.setdefault("relative_edge_gate", "knn_combat")
    saved_args.setdefault("relative_edge_k", 5)
    saved_args.setdefault("relative_edge_null", "none")
    saved_args.setdefault("snapshot_cache_dir", None)
    if cli_args.device is not None:
        saved_args["device"] = cli_args.device
    if cli_args.batch is not None:
        saved_args["batch"] = cli_args.batch
    if cli_args.eval_workers is not None:
        saved_args["eval_workers"] = cli_args.eval_workers
    if cli_args.prefetch_factor is not None:
        saved_args["prefetch_factor"] = cli_args.prefetch_factor
    if cli_args.cached:
        saved_args["cached"] = True
    if cli_args.snapshot_cache_dir is not None:
        saved_args["snapshot_cache_dir"] = cli_args.snapshot_cache_dir
    return SimpleNamespace(**saved_args)


def _split_ids(ckpt: dict[str, Any], split: str) -> list[str]:
    if split == "all":
        ids: list[str] = []
        for key in ("train_ids", "val_ids", "test_ids"):
            ids.extend(str(mid) for mid in (ckpt.get(key) or []))
        return ids
    return [str(mid) for mid in (ckpt.get(f"{split}_ids") or [])]


def _loader(ids: list[str], args: SimpleNamespace, include_edges: bool, split: str) -> DataLoader:
    ds = _make_dataset(
        ids,
        cached=bool(args.cached),
        include_edges=include_edges,
        cache_dir=getattr(args, "snapshot_cache_dir", None),
        cache_label=split,
    )
    return DataLoader(
        ds,
        batch_size=int(args.batch),
        shuffle=False,
        collate_fn=snapshot_phase_collate,
        **_loader_worker_kwargs(int(args.eval_workers), int(args.prefetch_factor)),
    )


def _ece(y: np.ndarray, p: np.ndarray, *, n_bins: int = 10) -> float:
    if y.size == 0:
        return float("nan")
    p = np.clip(p.astype(np.float64), 0.0, 1.0)
    y = y.astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    out = 0.0
    for bin_idx in range(n_bins):
        mask = bins == bin_idx
        if not np.any(mask):
            continue
        out += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(out)


def _binary_metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y).astype(np.int64)
    p = np.clip(np.asarray(p).astype(np.float64), 1e-7, 1.0 - 1e-7)
    pred = (p >= 0.5).astype(np.int64)
    out: dict[str, float | int] = {
        "n": int(y.size),
        "pos_rate": float(y.mean()) if y.size else float("nan"),
        "acc_at_0_5": float(accuracy_score(y, pred)) if y.size else float("nan"),
        "bce": float(log_loss(y, p, labels=[0, 1])) if y.size else float("nan"),
        "brier": float(brier_score_loss(y, p)) if y.size else float("nan"),
        "ece": _ece(y, p),
    }
    out["auc"] = float(roc_auc_score(y, p)) if y.size and y.min() != y.max() else float("nan")
    out["ap"] = (
        float(average_precision_score(y, p)) if y.size and y.min() != y.max() else float("nan")
    )
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
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


def _rank_metric(
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


def _empty_phase_bin() -> dict[str, Any]:
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
            "top5_from_survival": _binary_metric(t5y, t5p),
            "top10_from_survival": _binary_metric(t10y, t10p),
            "rank_from_survival": _rank_metric(
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
        "model_variant",
        "match_id",
        "team_idx",
        "team_id",
        "phase_slot",
        "cut_phase",
        "hazard_true",
        "hazard_prob",
        "cumulative_survival_score",
        "rank_pred_norm",
        "final_rank_norm",
        "final_top5",
        "top5_prob",
        "final_top10",
        "top10_prob",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _latest_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["match_id"]), int(row["team_idx"]))
        if key not in latest or int(row["cut_phase"]) > int(latest[key]["cut_phase"]):
            latest[key] = row
    return list(latest.values())


def _final_bridge_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        empty = np.asarray([], dtype=np.float64)
        return {
            "n": 0,
            "n_matches": 0,
            "top5_from_survival": _binary_metric(empty.astype(np.int64), empty),
            "top10_from_survival": _binary_metric(empty.astype(np.int64), empty),
            "rank_from_survival": _rank_metric(
                empty,
                empty,
                pair_correct=0,
                pair_total=0,
            ),
        }
    t5y = np.asarray([int(row["final_top5"]) for row in rows], dtype=np.int64)
    t10y = np.asarray([int(row["final_top10"]) for row in rows], dtype=np.int64)
    score = np.asarray([float(row["cumulative_survival_score"]) for row in rows], dtype=np.float64)
    rank_true = np.asarray([float(row["final_rank_norm"]) for row in rows], dtype=np.float64)
    rank_pred = np.asarray([float(row["rank_pred_norm"]) for row in rows], dtype=np.float64)

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y": [], "p": []})
    for row in rows:
        group = grouped[str(row["match_id"])]
        group["y"].append(float(row["final_rank_norm"]))
        group["p"].append(float(row["rank_pred_norm"]))
    pair_correct = 0
    pair_total = 0
    for values in grouped.values():
        correct, total = _pairwise_rank_counts(
            np.asarray(values["y"], dtype=np.float64),
            np.asarray(values["p"], dtype=np.float64),
        )
        pair_correct += correct
        pair_total += total

    return {
        "n": int(len(rows)),
        "n_matches": int(len(grouped)),
        "top5_from_survival": _binary_metric(t5y, score),
        "top10_from_survival": _binary_metric(t10y, score),
        "rank_from_survival": _rank_metric(
            rank_true,
            rank_pred,
            pair_correct=pair_correct,
            pair_total=pair_total,
        ),
    }


def evaluate(cli_args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checkpoint = Path(cli_args.checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu")
    args = _namespace_from_checkpoint(ckpt, cli_args)
    device = _device(args.device)
    model = _make_model(args).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    include_edges = args.model_variant not in NO_GRAPH_VARIANTS
    ids = _split_ids(ckpt, cli_args.split)
    if not ids:
        raise SystemExit(f"{checkpoint} has no ids for split={cli_args.split}")
    loader = _loader(ids, args, include_edges, cli_args.split)
    if len(loader.dataset) == 0:
        raise SystemExit(f"empty dataset for split={cli_args.split}")

    logger.info(
        "evaluate survival bridge checkpoint=%s variant=%s split=%s ids=%d loaded=%d",
        checkpoint,
        args.model_variant,
        cli_args.split,
        len(ids),
        len(loader.dataset),
    )

    hazard_y: list[np.ndarray] = []
    hazard_p: list[np.ndarray] = []
    rank_y: list[np.ndarray] = []
    rank_p: list[np.ndarray] = []
    top5_y: list[np.ndarray] = []
    top5_p: list[np.ndarray] = []
    top10_y: list[np.ndarray] = []
    top10_p: list[np.ndarray] = []
    phase_bins: dict[int, dict[str, Any]] = defaultdict(_empty_phase_bin)
    rank_groups: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"y": [], "p": []}
    )
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            logits, cut_mask = _forward_model(model, batch, args)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            y = batch["y_hazard"].detach().cpu().numpy()
            valid_mask = (
                (batch["target_mask"].bool() & cut_mask.bool().unsqueeze(-1)).detach().cpu().numpy()
            )
            y_rank_np = batch["y_rank"].detach().cpu().numpy()
            y_topk_np = batch["y_topk"].detach().cpu().numpy()
            match_ids = list(batch["match_id"])
            team_ids_batch = list(batch["team_ids"])

            B, T_out, N = prob.shape
            for b in range(B):
                match_id = str(match_ids[b])
                team_ids = team_ids_batch[b]
                for i in range(N):
                    team_id = int(team_ids[i]) if i < len(team_ids) else 0
                    if team_id == 0:
                        continue
                    survival_score = 1.0
                    for t in range(T_out):
                        if not bool(valid_mask[b, t, i]):
                            continue
                        cut_phase = int(t + 1)
                        hp = float(prob[b, t, i])
                        survival_score *= 1.0 - hp
                        survival_score = float(np.clip(survival_score, 0.0, 1.0))
                        rank_pred_norm = 1.0 - survival_score
                        hy = int(y[b, t, i])
                        ry = float(y_rank_np[b, i])
                        t5y = int(y_topk_np[b, i, 0])
                        t10y = int(y_topk_np[b, i, 1])

                        hazard_y.append(np.asarray([hy], dtype=np.int64))
                        hazard_p.append(np.asarray([hp], dtype=np.float64))
                        rank_y.append(np.asarray([ry], dtype=np.float64))
                        rank_p.append(np.asarray([rank_pred_norm], dtype=np.float64))
                        top5_y.append(np.asarray([t5y], dtype=np.int64))
                        top5_p.append(np.asarray([survival_score], dtype=np.float64))
                        top10_y.append(np.asarray([t10y], dtype=np.int64))
                        top10_p.append(np.asarray([survival_score], dtype=np.float64))

                        phase_bin = phase_bins[cut_phase]
                        phase_bin["matches"].add(match_id)
                        phase_bin["hazard_y"].append(np.asarray([hy], dtype=np.int64))
                        phase_bin["hazard_p"].append(np.asarray([hp], dtype=np.float64))
                        phase_bin["rank_y"].append(np.asarray([ry], dtype=np.float64))
                        phase_bin["rank_p"].append(np.asarray([rank_pred_norm], dtype=np.float64))
                        phase_bin["top5_y"].append(np.asarray([t5y], dtype=np.int64))
                        phase_bin["top5_p"].append(np.asarray([survival_score], dtype=np.float64))
                        phase_bin["top10_y"].append(np.asarray([t10y], dtype=np.int64))
                        phase_bin["top10_p"].append(np.asarray([survival_score], dtype=np.float64))
                        rank_groups[(match_id, cut_phase)]["y"].append(ry)
                        rank_groups[(match_id, cut_phase)]["p"].append(rank_pred_norm)

                        prediction_rows.append(
                            {
                                "split": cli_args.split,
                                "model_variant": str(args.model_variant),
                                "match_id": match_id,
                                "team_idx": int(i),
                                "team_id": team_id,
                                "phase_slot": cut_phase,
                                "cut_phase": cut_phase,
                                "hazard_true": hy,
                                "hazard_prob": hp,
                                "cumulative_survival_score": survival_score,
                                "rank_pred_norm": rank_pred_norm,
                                "final_rank_norm": ry,
                                "final_top5": t5y,
                                "top5_prob": survival_score,
                                "final_top10": t10y,
                                "top10_prob": survival_score,
                            }
                        )

    pair_correct = 0
    pair_total = 0
    for (_match_id, cut_phase), values in rank_groups.items():
        correct, total = _pairwise_rank_counts(
            np.asarray(values["y"], dtype=np.float64),
            np.asarray(values["p"], dtype=np.float64),
        )
        pair_correct += correct
        pair_total += total
        phase_bins[cut_phase]["pair_correct"] += correct
        phase_bins[cut_phase]["pair_total"] += total

    Y = _concat(hazard_y, dtype=np.int64)
    P = _concat(hazard_p, dtype=np.float64)
    R_true = _concat(rank_y, dtype=np.float64)
    R_pred = _concat(rank_p, dtype=np.float64)
    T5_y = _concat(top5_y, dtype=np.int64)
    T5_p = _concat(top5_p, dtype=np.float64)
    T10_y = _concat(top10_y, dtype=np.int64)
    T10_p = _concat(top10_p, dtype=np.float64)
    latest_rows = _latest_prediction_rows(prediction_rows)

    out: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "split": cli_args.split,
        "model_variant": str(args.model_variant),
        "metric_unit": "at_risk_team_phase",
        "final_bridge_eval_unit": "latest_at_risk_team",
        "bridge_definition": "cumulative_survival_score=prod_{j<=cut_phase}(1-hazard_prob_j)",
        "topk_bridge_note": "top5_prob/top10_prob are cumulative survival scores, not separately calibrated top-k heads.",
        "rank_bridge_note": "rank_pred_norm=1-cumulative_survival_score, so lower predicted rank means higher survival.",
        "bridge_metric_is_headline_safe": False,
        "split_counts": {
            "ids_requested": len(ids),
            "matches_loaded": len(loader.dataset),
            "prediction_rows": len(prediction_rows),
        },
        "channel_contract": ckpt.get("channel_contract", {}),
        "hazard": _binary_metric(Y, P),
        "top5_from_survival": _binary_metric(T5_y, T5_p),
        "top10_from_survival": _binary_metric(T10_y, T10_p),
        "rank_from_survival": _rank_metric(
            R_true,
            R_pred,
            pair_correct=pair_correct,
            pair_total=pair_total,
        ),
        "latest_cut_bridge": _final_bridge_metrics(latest_rows),
        "phase_metrics": _phase_metrics(phase_bins),
        "provenance": git_provenance(),
    }

    if cli_args.out:
        out_path = Path(cli_args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        logger.info("wrote %s", out_path)
    pred_out = cli_args.pred_out
    if pred_out is None and cli_args.out:
        pred_out = str(Path(cli_args.out).with_suffix(".pred.csv"))
    if pred_out:
        _write_prediction_csv(pred_out, prediction_rows)
        out["prediction_csv"] = str(pred_out)
        if cli_args.out:
            Path(cli_args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        logger.info("wrote %s", pred_out)
    logger.info("result=%s", out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--out", default=None)
    parser.add_argument("--pred-out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--snapshot-cache-dir", default=None)
    parser.add_argument("--eval-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
