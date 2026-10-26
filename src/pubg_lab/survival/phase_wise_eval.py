"""Phase-wise evaluation for domain edge ablation checkpoints.

This module evaluates trained DynamicSTGNN checkpoints at the cut-phase level:

    cut_phase k: X up to phase k -> hazard of elimination before phase k + 1

It writes prediction rows plus per-domain/per-phase metrics so the edge ablation
can be discussed as a spatio-temporal effect rather than only one pooled score.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[2]))

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader

from pubg_lab.survival.config import DEFAULT_PHASE_EPOCHS, phase_checkpoint_name
from pubg_lab.survival.dataset import LazyPhaseMatchDataset, phase_collate
from pubg_lab.survival.dyn_stgnn import DynamicSTGNN
from pubg_lab.survival.edge_modes import apply_edge_mode, edge_active_mask
from pubg_lab.data.stdb.queries import PHASE_EDGE_TYPES, PHASE_FEATURE_DIM

logger = logging.getLogger("model_stgcn.phase_wise_eval")


EDGE_ORDER = ("none", "proximity", "proximity_combat", "full")


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _read_test_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        return list(data["splits"]["test"])
    except KeyError as exc:
        raise ValueError(f"split file lacks splits.test: {path}") from exc


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    out: dict[str, float | int] = {
        "n": int(y_true.size),
        "pos_rate": float(np.mean(y_true)) if y_true.size else float("nan"),
        "acc": float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan"),
        "bce": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1]))
        if y_true.size
        else float("nan"),
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


def _load_model(
    checkpoint: Path, edge_mode: str, device: torch.device
) -> tuple[DynamicSTGNN, list[str], torch.Tensor]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    feature_dim = int(ckpt.get("feature_dim", PHASE_FEATURE_DIM))
    edge_types = list(ckpt.get("edge_types", PHASE_EDGE_TYPES))
    model = DynamicSTGNN(
        in_features=feature_dim,
        hidden=int(ckpt_args.get("hidden", 64)),
        mid=int(ckpt_args.get("mid", 128)),
        num_blocks=int(ckpt_args.get("num_blocks", 2)),
        num_edge_types=len(edge_types),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, edge_types, edge_active_mask(edge_mode, edge_types=edge_types, device=device)


def _domain_parts(domain_dir: Path) -> tuple[str, str]:
    if "__" not in domain_dir.name:
        raise ValueError(f"domain dir must be Map__mode: {domain_dir}")
    map_name, mode = domain_dir.name.split("__", 1)
    return map_name, mode


def _prediction_rows_for_checkpoint(
    *,
    domain_dir: Path,
    edge_mode: str,
    test_ids: list[str],
    batch_size: int,
    device: torch.device,
    epochs: int = DEFAULT_PHASE_EPOCHS,
) -> list[dict[str, Any]]:
    checkpoint = domain_dir / phase_checkpoint_name(edge_mode, epochs)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    map_name, mode = _domain_parts(domain_dir)
    model, edge_types, active_mask = _load_model(checkpoint, edge_mode, device)
    loader = DataLoader(
        LazyPhaseMatchDataset(test_ids, include_combat_in="combat_in" in edge_types),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=phase_collate,
        num_workers=0,
    )

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            A = apply_edge_mode(batch["A"].float(), edge_mode, edge_types=edge_types)
            hazard_logits, _rank_pred, _topk_logits = model(
                batch["X"].float(),
                A,
                edge_active_mask=active_mask,
                phase_mask=batch["phase_mask"],
            )
            probs = torch.sigmoid(hazard_logits).detach().cpu().numpy()
            y_true = batch["y_hazard"].detach().cpu().numpy()
            target_mask = batch["target_mask"].detach().cpu().numpy().astype(bool)

            match_ids = raw_batch["match_id"]
            team_ids_batch = raw_batch["team_ids"]
            phase_indices_batch = raw_batch["phase_indices"]
            bsz, t_max, n_max = probs.shape
            for b in range(bsz):
                for t in range(t_max):
                    cut_phase = (
                        int(phase_indices_batch[b][t]) if t < len(phase_indices_batch[b]) else t + 1
                    )
                    if cut_phase <= 0:
                        continue
                    for n in range(n_max):
                        if not target_mask[b, t, n]:
                            continue
                        team_id = int(team_ids_batch[b][n])
                        if team_id <= 0:
                            continue
                        rows.append(
                            {
                                "map_name": map_name,
                                "mode": mode,
                                "edge_mode": edge_mode,
                                "match_id": match_ids[b],
                                "team_id": team_id,
                                "cut_phase": cut_phase,
                                "y": int(y_true[b, t, n]),
                                "prob": float(probs[b, t, n]),
                            }
                        )
    return rows


def _group_metrics(rows: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        y = np.array([r["y"] for r in group], dtype=int)
        p = np.array([r["prob"] for r in group], dtype=float)
        rec = dict(zip(group_cols, key))
        rec.update(_binary_metrics(y, p))
        rec["n_matches"] = len({r["match_id"] for r in group})
        out.append(rec)
    return out


def _phase_deltas(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["map_name"], r["mode"], int(r["cut_phase"]), r["edge_mode"]): r for r in metrics}
    out: list[dict[str, Any]] = []
    for (map_name, mode, cut_phase, edge_mode), row in sorted(by_key.items()):
        if edge_mode == "none":
            continue
        base = by_key.get((map_name, mode, cut_phase, "none"))
        if not base:
            continue
        rec = {
            "map_name": map_name,
            "mode": mode,
            "cut_phase": cut_phase,
            "edge_mode": edge_mode,
            "n": row["n"],
            "pos_rate": row["pos_rate"],
        }
        for metric in ("auc", "ap", "bce", "acc"):
            rec[f"{metric}_{edge_mode}"] = row[metric]
            rec[f"{metric}_none"] = base[metric]
            rec[f"delta_{metric}"] = float(row[metric]) - float(base[metric])
        out.append(rec)
    return out


def _paired_bce_tests(
    rows: list[dict[str, Any]], comparisons: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    by_cell = {
        (
            r["map_name"],
            r["mode"],
            r["match_id"],
            int(r["team_id"]),
            int(r["cut_phase"]),
            r["edge_mode"],
        ): r
        for r in rows
    }
    keys = sorted({(r["map_name"], r["mode"], int(r["cut_phase"])) for r in rows})
    out: list[dict[str, Any]] = []
    for map_name, mode, cut_phase in keys:
        domain_rows = [
            r
            for r in rows
            if r["map_name"] == map_name and r["mode"] == mode and int(r["cut_phase"]) == cut_phase
        ]
        cell_ids = sorted({(r["match_id"], int(r["team_id"])) for r in domain_rows})
        for left, right in comparisons:
            diffs: list[float] = []
            for match_id, team_id in cell_ids:
                l = by_cell.get((map_name, mode, match_id, team_id, cut_phase, left))
                r = by_cell.get((map_name, mode, match_id, team_id, cut_phase, right))
                if not l or not r:
                    continue
                y = int(l["y"])
                lp = float(np.clip(float(l["prob"]), 1e-7, 1 - 1e-7))
                rp = float(np.clip(float(r["prob"]), 1e-7, 1 - 1e-7))
                l_bce = -(y * np.log(lp) + (1 - y) * np.log(1 - lp))
                r_bce = -(y * np.log(rp) + (1 - y) * np.log(1 - rp))
                diffs.append(float(l_bce - r_bce))
            rec: dict[str, Any] = {
                "map_name": map_name,
                "mode": mode,
                "cut_phase": cut_phase,
                "left_model": left,
                "right_model": right,
                "n_cells": len(diffs),
                "mean_bce_delta_left_minus_right": float(np.mean(diffs)) if diffs else float("nan"),
            }
            if diffs and any(d != 0 for d in diffs):
                stat, p = wilcoxon(diffs, alternative="less", zero_method="wilcox", method="auto")
                rec["wilcoxon_stat"] = float(stat)
                rec["wilcoxon_p_less"] = float(p)
            else:
                rec["wilcoxon_stat"] = float("nan")
                rec["wilcoxon_p_less"] = float("nan")
            out.append(rec)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "phase_wise_eval"
    device = _device(args.device)
    domain_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and "__" in p.name]
    if args.domains:
        requested = set(args.domains)
        domain_dirs = [p for p in domain_dirs if p.name in requested]
    if not domain_dirs:
        raise SystemExit("no domain directories found")

    all_predictions: list[dict[str, Any]] = []
    for domain_dir in domain_dirs:
        split_path = domain_dir / "split_seed42.json"
        if not split_path.exists():
            logger.warning("skip %s: missing split_seed42.json", domain_dir)
            continue
        test_ids = _read_test_ids(split_path)
        logger.info("%s test_matches=%d", domain_dir.name, len(test_ids))
        for edge_mode in args.edge_modes:
            logger.info("evaluate %s edge=%s", domain_dir.name, edge_mode)
            all_predictions.extend(
                _prediction_rows_for_checkpoint(
                    domain_dir=domain_dir,
                    edge_mode=edge_mode,
                    test_ids=test_ids,
                    batch_size=args.batch,
                    device=device,
                    epochs=args.epochs,
                )
            )

    phase_metrics = _group_metrics(all_predictions, ["map_name", "mode", "edge_mode", "cut_phase"])
    phase_deltas = _phase_deltas(phase_metrics)
    phase_tests = _paired_bce_tests(
        all_predictions,
        comparisons=[("full", "none"), ("proximity_combat", "none"), ("proximity", "none")],
    )
    aggregate_metrics = _group_metrics(all_predictions, ["edge_mode", "cut_phase"])
    aggregate_deltas = _phase_deltas(
        [{"map_name": "ALL", "mode": "ALL", **row} for row in aggregate_metrics]
    )

    prediction_path = out_dir / "phase_predictions.csv"
    metrics_path = out_dir / "phase_metrics.csv"
    deltas_path = out_dir / "phase_deltas.csv"
    tests_path = out_dir / "phase_bce_wilcoxon.csv"
    aggregate_metrics_path = out_dir / "phase_metrics_aggregate.csv"
    aggregate_deltas_path = out_dir / "phase_deltas_aggregate.csv"
    summary_path = out_dir / "phase_wise_summary.json"

    _write_csv(prediction_path, all_predictions)
    _write_csv(metrics_path, phase_metrics)
    _write_csv(deltas_path, phase_deltas)
    _write_csv(tests_path, phase_tests)
    _write_csv(aggregate_metrics_path, aggregate_metrics)
    _write_csv(aggregate_deltas_path, aggregate_deltas)

    summary = {
        "root": str(root),
        "domains": [p.name for p in domain_dirs],
        "edge_modes": list(args.edge_modes),
        "n_predictions": len(all_predictions),
        "outputs": {
            "predictions": str(prediction_path),
            "phase_metrics": str(metrics_path),
            "phase_deltas": str(deltas_path),
            "phase_bce_wilcoxon": str(tests_path),
            "phase_metrics_aggregate": str(aggregate_metrics_path),
            "phase_deltas_aggregate": str(aggregate_deltas_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s", summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="model_stgcn/checkpoints/domain_runs_expanded_e12")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_PHASE_EPOCHS,
        help="checkpoint epochs suffix (must match run_domain_training)",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--edge-modes", nargs="+", default=list(EDGE_ORDER), choices=list(EDGE_ORDER))
    ap.add_argument("--domains", nargs="*", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
