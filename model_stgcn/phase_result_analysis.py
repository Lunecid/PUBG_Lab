"""Phase-wise hazard analysis and paired statistical tests.

This script turns the trained DynamicSTGNN checkpoints and the tabular baseline
into analysis-ready tables:

* overall hazard metrics
* phase-wise hazard metrics
* match-level paired Wilcoxon tests

The unit of statistical comparison is one match, not one team/phase row.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from model_stgcn.baselines import EXCLUDE_COLUMNS, _classifier, _matrix, load_phase_rows
from model_stgcn.dataset import CachedPhaseMatchDataset, LazyPhaseMatchDataset, phase_collate
from model_stgcn.dyn_stgnn import DynamicSTGNN
from model_stgcn.edge_modes import apply_edge_mode, edge_active_mask
from model_stgcn.splits import load_or_create_split
from stdb.queries import PHASE_EDGE_TYPES, PHASE_FEATURE_DIM, list_phase_eligible_match_ids

logger = logging.getLogger("model_stgcn.phase_result_analysis")


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


def _bce(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    p = np.clip(y_prob.astype(float), 1e-7, 1.0 - 1e-7)
    y = y_true.astype(float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    out: dict[str, float | int] = {
        "n": int(y_true.size),
        "pos_rate": float(np.mean(y_true)) if y_true.size else float("nan"),
        "acc": float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan"),
        "bce": _bce(y_true, y_prob),
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


def _metric_rows(df: pd.DataFrame, *, group_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(_binary_metrics(g["y"].to_numpy(), g["prob"].to_numpy()))
        rows.append(row)
    return rows


def _paired_metric_by_match(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, match_id), g in df.groupby(["model", "match_id"]):
        metrics = _binary_metrics(g["y"].to_numpy(), g["prob"].to_numpy())
        rows.append({"model": model, "match_id": match_id, **metrics})
    return pd.DataFrame(rows)


def _paired_test(
    per_match: pd.DataFrame,
    *,
    left_model: str,
    right_model: str,
    metric: str,
    larger_is_better: bool,
) -> dict[str, Any]:
    left = per_match[per_match["model"] == left_model][["match_id", metric]].rename(columns={metric: "left"})
    right = per_match[per_match["model"] == right_model][["match_id", metric]].rename(columns={metric: "right"})
    joined = left.merge(right, on="match_id", how="inner").dropna()
    if joined.empty:
        return {
            "left_model": left_model,
            "right_model": right_model,
            "metric": metric,
            "n": 0,
            "p_value": float("nan"),
            "median_improvement": float("nan"),
            "mean_improvement": float("nan"),
            "rank_biserial": float("nan"),
        }

    # Positive improvement means left_model is better.
    improvement = (
        joined["left"].to_numpy() - joined["right"].to_numpy()
        if larger_is_better
        else joined["right"].to_numpy() - joined["left"].to_numpy()
    )
    improvement = improvement[np.isfinite(improvement)]
    nonzero = improvement[improvement != 0]
    if nonzero.size == 0:
        p_value = 1.0
        rank_biserial = 0.0
    else:
        stat = wilcoxon(nonzero, alternative="greater", zero_method="wilcox", mode="auto")
        p_value = float(stat.pvalue)
        ranks = rankdata(np.abs(nonzero))
        pos_sum = float(ranks[nonzero > 0].sum())
        neg_sum = float(ranks[nonzero < 0].sum())
        total = pos_sum + neg_sum
        rank_biserial = (pos_sum - neg_sum) / total if total else 0.0

    return {
        "left_model": left_model,
        "right_model": right_model,
        "metric": metric,
        "larger_is_better": larger_is_better,
        "n": int(improvement.size),
        "left_mean": float(joined["left"].mean()),
        "right_mean": float(joined["right"].mean()),
        "median_improvement": float(np.median(improvement)) if improvement.size else float("nan"),
        "mean_improvement": float(np.mean(improvement)) if improvement.size else float("nan"),
        "p_value_greater": p_value,
        "rank_biserial": float(rank_biserial),
    }


def _load_dynamic_model(checkpoint: str, device: torch.device) -> tuple[DynamicSTGNN, dict[str, Any], str, list[str]]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    edge_types = list(ckpt.get("edge_types", PHASE_EDGE_TYPES))
    edge_mode = ckpt_args.get("edge_mode", "full")
    model = DynamicSTGNN(
        in_features=int(ckpt.get("feature_dim", PHASE_FEATURE_DIM)),
        hidden=int(ckpt_args.get("hidden", 64)),
        mid=int(ckpt_args.get("mid", 128)),
        num_blocks=int(ckpt_args.get("num_blocks", 2)),
        num_edge_types=len(edge_types),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt, edge_mode, edge_types


def dynamic_hazard_predictions(
    *,
    model_name: str,
    checkpoint: str,
    match_ids: list[str],
    device: torch.device,
    batch_size: int,
    cached: bool,
) -> pd.DataFrame:
    model, ckpt, edge_mode, edge_types = _load_dynamic_model(checkpoint, device)
    active_mask = edge_active_mask(edge_mode, edge_types=edge_types, device=device)
    # V2 node layout (F6+F7) is recorded in ckpt args; fall back to feature_dim.
    include_v2 = bool(ckpt.get("args", {}).get("v2_node_features")) or int(
        ckpt.get("feature_dim", PHASE_FEATURE_DIM)
    ) > PHASE_FEATURE_DIM
    ds_cls = CachedPhaseMatchDataset if cached else LazyPhaseMatchDataset
    ds = ds_cls(
        match_ids,
        include_combat_in="combat_in" in edge_types,
        include_v2_features=include_v2,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=phase_collate, num_workers=0)
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            A = apply_edge_mode(batch["A"].float(), edge_mode, edge_types=edge_types)
            yh, _yr, _yk = model(batch["X"].float(), A, edge_active_mask=active_mask, phase_mask=batch["phase_mask"])
            probs = torch.sigmoid(yh).detach().cpu().numpy()
            y = batch["y_hazard"].detach().cpu().numpy()
            mask = batch["target_mask"].detach().cpu().numpy().astype(bool)

            for b, match_id in enumerate(batch["match_id"]):
                team_ids = batch["team_ids"][b]
                phase_indices = batch["phase_indices"][b]
                for out_t in range(probs.shape[1]):
                    cut_phase = int(phase_indices[out_t]) if out_t < len(phase_indices) and phase_indices[out_t] else out_t + 1
                    for i, team_id in enumerate(team_ids):
                        if not team_id or not mask[b, out_t, i]:
                            continue
                        rows.append(
                            {
                                "model": model_name,
                                "match_id": match_id,
                                "team_id": int(team_id),
                                "phase_idx": cut_phase,
                                "y": int(y[b, out_t, i]),
                                "prob": float(probs[b, out_t, i]),
                            }
                        )
    return pd.DataFrame(rows)


def baseline_hazard_predictions(
    *,
    model_name: str,
    train_ids: list[str],
    test_ids: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    train_rows, cols = load_phase_rows(train_ids)
    test_rows, _ = load_phase_rows(test_ids)
    feature_cols = [c for c in cols if c not in EXCLUDE_COLUMNS]
    X_train, yh_train, *_ = _matrix(train_rows, feature_cols)
    X_test, yh_test, *_ = _matrix(test_rows, feature_cols)
    model = _classifier(feature_cols)
    logger.info("fit tabular baseline hazard rows=%d features=%d", len(train_rows), len(feature_cols))
    model.fit(X_train, yh_train)
    prob = model.predict_proba(X_test)[:, 1]
    out_rows = [
        {
            "model": model_name,
            "match_id": row["match_id"],
            "team_id": int(row["team_id"]),
            "phase_idx": int(row["phase_idx"]),
            "y": int(row["hazard_next"]),
            "prob": float(p),
        }
        for row, p in zip(test_rows, prob)
    ]
    return pd.DataFrame(out_rows), feature_cols


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ids = list_phase_eligible_match_ids(args.map, args.mode, args.limit)
    train_ids, val_ids, test_ids = load_or_create_split(
        ids,
        split_json=args.split_json,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        metadata={"source": "model_stgcn.phase_result_analysis", "limit": args.limit},
    )
    logger.info("split total=%d train=%d val=%d test=%d", len(ids), len(train_ids), len(val_ids), len(test_ids))

    device = _device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_frames: list[pd.DataFrame] = []
    baseline_df, feature_cols = baseline_hazard_predictions(
        model_name="tabular_baseline_hgb",
        train_ids=train_ids,
        test_ids=test_ids,
    )
    pred_frames.append(baseline_df)

    dynamic_specs: list[tuple[str, str]] = []
    for model_name, checkpoint in [
        ("dynamic_none_e12", args.none_checkpoint),
        ("dynamic_proximity_e12", args.proximity_checkpoint),
        ("dynamic_proximity_combat_e12", args.proximity_combat_checkpoint),
        ("dynamic_full_e12", args.full_checkpoint),
        ("dynamic_full_tuned", args.tuned_checkpoint),
    ]:
        if not checkpoint:
            continue
        if not Path(checkpoint).exists():
            logger.warning("skip missing checkpoint for %s: %s", model_name, checkpoint)
            continue
        dynamic_specs.append((model_name, checkpoint))

    for model_name, checkpoint in dynamic_specs:
        logger.info("predict %s from %s", model_name, checkpoint)
        pred_frames.append(
            dynamic_hazard_predictions(
                model_name=model_name,
                checkpoint=checkpoint,
                match_ids=test_ids,
                device=device,
                batch_size=args.batch,
                cached=args.cached,
            )
        )

    preds = pd.concat(pred_frames, ignore_index=True)
    raw_preds_path = out_dir / "hazard_predictions_test_raw.csv"
    raw_preds_path.write_text(preds.to_csv(index=False), encoding="utf-8")

    key_cols = ["match_id", "team_id", "phase_idx"]
    n_models = preds["model"].nunique()
    common_keys = (
        preds.groupby(key_cols)["model"]
        .nunique()
        .reset_index(name="n_models")
        .query("n_models == @n_models")
        .drop(columns=["n_models"])
    )
    preds = preds.merge(common_keys, on=key_cols, how="inner")
    preds_path = out_dir / "hazard_predictions_test_aligned.csv"
    preds.to_csv(preds_path, index=False)

    overall = pd.DataFrame(_metric_rows(preds, group_cols=["model"]))
    phase = pd.DataFrame(_metric_rows(preds, group_cols=["model", "phase_idx"]))
    per_match = _paired_metric_by_match(preds)

    overall_path = out_dir / "hazard_overall_metrics.csv"
    phase_path = out_dir / "hazard_phase_metrics.csv"
    per_match_path = out_dir / "hazard_per_match_metrics.csv"
    overall.to_csv(overall_path, index=False)
    phase.to_csv(phase_path, index=False)
    per_match.to_csv(per_match_path, index=False)

    tests = []
    comparisons = [
        ("dynamic_full_e12", "dynamic_none_e12"),
        ("dynamic_full_e12", "dynamic_proximity_e12"),
        ("dynamic_full_e12", "dynamic_proximity_combat_e12"),
        ("dynamic_full_e12", "tabular_baseline_hgb"),
        ("dynamic_full_tuned", "tabular_baseline_hgb"),
        ("dynamic_full_tuned", "dynamic_none_e12"),
    ]
    available_models = set(per_match["model"])
    for left, right in comparisons:
        if left not in available_models or right not in available_models:
            continue
        tests.append(_paired_test(per_match, left_model=left, right_model=right, metric="bce", larger_is_better=False))
        tests.append(_paired_test(per_match, left_model=left, right_model=right, metric="auc", larger_is_better=True))
        tests.append(_paired_test(per_match, left_model=left, right_model=right, metric="ap", larger_is_better=True))

    tests_df = pd.DataFrame(tests)
    tests_path = out_dir / "hazard_paired_tests.csv"
    tests_df.to_csv(tests_path, index=False)

    summary = {
        "split": {
            "total": len(ids),
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "feature_count": len(feature_cols),
        "outputs": {
            "raw_predictions": str(raw_preds_path),
            "aligned_predictions": str(preds_path),
            "overall_metrics": str(overall_path),
            "phase_metrics": str(phase_path),
            "per_match_metrics": str(per_match_path),
            "paired_tests": str(tests_path),
        },
        "alignment": {
            "key_columns": key_cols,
            "n_models": int(n_models),
            "aligned_rows": int(len(preds)),
        },
        "overall_metrics": overall.to_dict(orient="records"),
        "paired_tests": tests,
    }
    summary_path = out_dir / "phase_result_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", summary_path)
    logger.info("overall=%s", overall.to_dict(orient="records"))
    logger.info("paired=%s", tests)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-json", default="model_stgcn/checkpoints/split_phase_all_seed42.json")
    ap.add_argument("--none-checkpoint", default="model_stgcn/checkpoints/phase_ablation_none_e12.pt")
    ap.add_argument("--proximity-checkpoint", default="model_stgcn/checkpoints/phase_ablation_proximity_masked_e12.pt")
    ap.add_argument("--proximity-combat-checkpoint", default="model_stgcn/checkpoints/phase_ablation_proximity_combat_masked_e12.pt")
    ap.add_argument("--full-checkpoint", default="model_stgcn/checkpoints/phase_full_5951_e12.pt")
    ap.add_argument("--tuned-checkpoint", default="model_stgcn/checkpoints/phase_full_tuned_h96_e30.pt")
    ap.add_argument("--out-dir", default="model_stgcn/checkpoints/phase_result_analysis")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--cached", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
