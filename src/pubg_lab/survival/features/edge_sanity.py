"""Random-edge sanity checks for phase-level DynamicSTGNN checkpoints.

This is an inference-time control: it keeps the trained model fixed and compares
observed graph edges against randomized edges on the same test cells.  The
default `node_shuffle` control preserves each active subgraph's density and
degree sequence up to a node relabeling, but breaks the alignment between team
features and team-team relations.

Example:
    python -m model_stgcn.edge_sanity --repeats 5
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
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader

from pubg_lab.survival.config import PHASE_CHECKPOINT_TEMPLATE
from pubg_lab.survival.data.dataset import LazyPhaseMatchDataset, phase_collate
from pubg_lab.survival.models.dyn_stgnn import DynamicSTGNN
from pubg_lab.survival.features.edge_modes import apply_edge_mode, edge_active_mask
from pubg_lab.data.stdb.queries.queries import PHASE_EDGE_TYPES, PHASE_FEATURE_DIM

logger = logging.getLogger("pubg_lab.survival.features.edge_sanity")


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
    return list(data["splits"]["test"])


def _domain_parts(domain_dir: Path) -> tuple[str, str]:
    if "__" not in domain_dir.name:
        raise ValueError(f"domain dir must be Map__mode: {domain_dir}")
    return tuple(domain_dir.name.split("__", 1))  # type: ignore[return-value]


def _load_model(
    checkpoint: Path, edge_mode: str, device: torch.device
) -> tuple[DynamicSTGNN, list[str], torch.Tensor, bool]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    edge_types = list(ckpt.get("edge_types", PHASE_EDGE_TYPES))
    feature_dim = int(ckpt.get("feature_dim", PHASE_FEATURE_DIM))
    include_v2 = bool(ckpt_args.get("v2_node_features")) or feature_dim > PHASE_FEATURE_DIM
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
    return (
        model,
        edge_types,
        edge_active_mask(edge_mode, edge_types=edge_types, device=device),
        include_v2,
    )


def _active_indices(mask: torch.Tensor) -> torch.Tensor:
    idx = torch.nonzero(mask.bool(), as_tuple=False).flatten()
    return idx


def _node_shuffle_edges(
    A: torch.Tensor, node_mask: torch.Tensor, *, generator: torch.Generator
) -> torch.Tensor:
    """Relabel alive nodes per (batch, phase), preserving edge structure.

    Phase-local (FEATURE_DESIGN_V2 §9): the permutation is drawn from the
    nodes alive at that phase (`node_mask[b, t]`), so dead/padding rows never
    mix into the alive block. One permutation is shared across edge channels
    within a phase — a node relabeling assigns each team a single fake
    identity, so all of its relations must move together. Entries outside the
    alive block (none expected; edges are built from alive snapshots) are
    zeroed.
    """
    out = torch.zeros_like(A)
    bsz, t_max, e_max, _n, _n2 = A.shape
    for b in range(bsz):
        for t in range(t_max):
            active_nodes = _active_indices(node_mask[b, t])
            if active_nodes.numel() <= 1:
                continue
            perm = torch.randperm(active_nodes.numel(), generator=generator).to(A.device)
            for e in range(e_max):
                sub = A[b, t, e][active_nodes][:, active_nodes]
                shuffled = sub[perm][:, perm]
                out[b, t, e][active_nodes[:, None], active_nodes[None, :]] = shuffled
    return out


def _edge_shuffle_edges(
    A: torch.Tensor, node_mask: torch.Tensor, *, generator: torch.Generator
) -> torch.Tensor:
    """Shuffle alive-block adjacency entries per (batch, phase, channel).

    Phase-local; preserves each phase subgraph's density but not its degree
    sequence. Per-channel shuffling is intentional here — this control has no
    node-identity semantics to keep consistent.
    """
    out = torch.zeros_like(A)
    bsz, t_max, e_max, _n, _n2 = A.shape
    for b in range(bsz):
        for t in range(t_max):
            active_nodes = _active_indices(node_mask[b, t])
            if active_nodes.numel() <= 1:
                continue
            for e in range(e_max):
                sub = A[b, t, e][active_nodes][:, active_nodes]
                flat = sub.flatten()
                perm = torch.randperm(flat.numel(), generator=generator).to(A.device)
                shuffled = flat[perm].reshape_as(sub)
                out[b, t, e][active_nodes[:, None], active_nodes[None, :]] = shuffled
    return out


def _row_normalize_edges(A: torch.Tensor) -> torch.Tensor:
    row_sum = A.sum(dim=-1, keepdim=True)
    return torch.where(row_sum > 0, A / row_sum.clamp_min(1e-12), A)


def _apply_control(
    A: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    control: str,
    generator: torch.Generator,
) -> torch.Tensor:
    if control == "observed":
        return A
    if control == "node_shuffle":
        return _row_normalize_edges(_node_shuffle_edges(A, node_mask, generator=generator))
    if control == "edge_shuffle":
        return _row_normalize_edges(_edge_shuffle_edges(A, node_mask, generator=generator))
    raise ValueError(f"unknown edge control: {control}")


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob.astype(float), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= 0.5).astype(int)
    out: dict[str, float | int] = {
        "n": int(y_true.size),
        "pos_rate": float(np.mean(y_true)) if y_true.size else float("nan"),
        "acc": float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan"),
        "bce": float(log_loss(y_true, y_prob, labels=[0, 1])) if y_true.size else float("nan"),
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


def _prediction_rows(
    *,
    domain_dir: Path,
    checkpoint: Path,
    edge_mode: str,
    test_ids: list[str],
    batch_size: int,
    device: torch.device,
    control: str,
    repeat: int,
    seed: int,
) -> list[dict[str, Any]]:
    map_name, mode = _domain_parts(domain_dir)
    model, edge_types, active_mask, include_v2 = _load_model(checkpoint, edge_mode, device)
    loader = DataLoader(
        LazyPhaseMatchDataset(
            test_ids,
            include_combat_in="combat_in" in edge_types,
            include_v2_features=include_v2,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=phase_collate,
        num_workers=0,
    )
    generator = torch.Generator()
    generator.manual_seed(seed + repeat * 1009)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            A = apply_edge_mode(batch["A"].float(), edge_mode, edge_types=edge_types)
            A = _apply_control(A, batch["node_mask"].bool(), control=control, generator=generator)
            hazard_logits, _rank_pred, _topk_logits = model(
                batch["X"].float(),
                A,
                edge_active_mask=active_mask,
                phase_mask=batch["phase_mask"],
            )
            probs = torch.sigmoid(hazard_logits).detach().cpu().numpy()
            y_true = batch["y_hazard"].detach().cpu().numpy()
            target_mask = batch["target_mask"].detach().cpu().numpy().astype(bool)

            for b, match_id in enumerate(raw_batch["match_id"]):
                team_ids = raw_batch["team_ids"][b]
                phase_indices = raw_batch["phase_indices"][b]
                for t in range(probs.shape[1]):
                    cut_phase = (
                        int(phase_indices[t])
                        if t < len(phase_indices) and phase_indices[t]
                        else t + 1
                    )
                    for i, team_id in enumerate(team_ids):
                        if not team_id or not target_mask[b, t, i]:
                            continue
                        rows.append(
                            {
                                "map_name": map_name,
                                "mode": mode,
                                "edge_mode": edge_mode,
                                "control": control,
                                "repeat": repeat,
                                "match_id": match_id,
                                "team_id": int(team_id),
                                "cut_phase": cut_phase,
                                "y": int(y_true[b, t, i]),
                                "prob": float(probs[b, t, i]),
                            }
                        )
    return rows


def _group_metrics(rows: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[c] for c in group_cols), []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        rec = dict(zip(group_cols, key))
        rec.update(
            _binary_metrics(np.array([r["y"] for r in group]), np.array([r["prob"] for r in group]))
        )
        rec["n_matches"] = len({r["match_id"] for r in group})
        out.append(rec)
    return out


def _control_deltas(
    metrics: list[dict[str, Any]], *, baseline_control: str = "observed"
) -> list[dict[str, Any]]:
    by_key = {
        (r.get("map_name"), r.get("mode"), r.get("edge_mode"), r.get("repeat"), r.get("control")): r
        for r in metrics
    }
    out: list[dict[str, Any]] = []
    for row in metrics:
        if row["control"] == baseline_control:
            continue
        base = by_key.get(
            (row.get("map_name"), row.get("mode"), row.get("edge_mode"), 0, baseline_control)
        )
        if not base:
            continue
        rec = {
            "map_name": row.get("map_name"),
            "mode": row.get("mode"),
            "edge_mode": row.get("edge_mode"),
            "control": row.get("control"),
            "repeat": row.get("repeat"),
            "n": row.get("n"),
        }
        for metric in ("auc", "ap", "bce", "acc"):
            rec[f"{metric}_{row['control']}"] = row.get(metric)
            rec[f"{metric}_{baseline_control}"] = base.get(metric)
            rec[f"delta_{metric}"] = float(row[metric]) - float(base[metric])
        out.append(rec)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "edge_sanity"
    device = _device(args.device)
    domain_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and "__" in p.name]
    if args.domains:
        requested = set(args.domains)
        domain_dirs = [p for p in domain_dirs if p.name in requested]
    if not domain_dirs:
        raise SystemExit(f"no domain directories found in {root}")

    all_rows: list[dict[str, Any]] = []
    for domain_dir in domain_dirs:
        split_path = domain_dir / f"split_seed{args.seed}.json"
        if not split_path.exists():
            split_path = domain_dir / "split_seed42.json"
        if not split_path.exists():
            logger.warning("skip %s: missing split json", domain_dir)
            continue
        checkpoint = domain_dir / args.checkpoint_template.format(
            edge_mode=args.edge_mode, epochs=args.epochs
        )
        if not checkpoint.exists():
            logger.warning("skip %s: missing %s", domain_dir, checkpoint.name)
            continue
        test_ids = _read_test_ids(split_path)
        if args.max_matches:
            test_ids = test_ids[: args.max_matches]
        logger.info(
            "%s test_matches=%d checkpoint=%s", domain_dir.name, len(test_ids), checkpoint.name
        )

        all_rows.extend(
            _prediction_rows(
                domain_dir=domain_dir,
                checkpoint=checkpoint,
                edge_mode=args.edge_mode,
                test_ids=test_ids,
                batch_size=args.batch,
                device=device,
                control="observed",
                repeat=0,
                seed=args.seed,
            )
        )
        for control in args.controls:
            for repeat in range(args.repeats):
                logger.info("%s control=%s repeat=%d", domain_dir.name, control, repeat)
                all_rows.extend(
                    _prediction_rows(
                        domain_dir=domain_dir,
                        checkpoint=checkpoint,
                        edge_mode=args.edge_mode,
                        test_ids=test_ids,
                        batch_size=args.batch,
                        device=device,
                        control=control,
                        repeat=repeat,
                        seed=args.seed,
                    )
                )

    prediction_path = out_dir / "edge_sanity_predictions.csv"
    metrics_path = out_dir / "edge_sanity_metrics.csv"
    aggregate_path = out_dir / "edge_sanity_metrics_aggregate.csv"
    deltas_path = out_dir / "edge_sanity_deltas.csv"
    summary_path = out_dir / "edge_sanity_summary.json"

    domain_metrics = _group_metrics(
        all_rows, ["map_name", "mode", "edge_mode", "control", "repeat"]
    )
    aggregate_metrics = _group_metrics(all_rows, ["edge_mode", "control", "repeat"])
    deltas = _control_deltas(domain_metrics)

    _write_csv(prediction_path, all_rows)
    _write_csv(metrics_path, domain_metrics)
    _write_csv(aggregate_path, aggregate_metrics)
    _write_csv(deltas_path, deltas)
    summary = {
        "root": str(root),
        "edge_mode": args.edge_mode,
        "controls": list(args.controls),
        "repeats": args.repeats,
        "n_predictions": len(all_rows),
        "outputs": {
            "predictions": str(prediction_path),
            "domain_metrics": str(metrics_path),
            "aggregate_metrics": str(aggregate_path),
            "deltas": str(deltas_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s", summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="model_stgcn/checkpoints/domain_runs_expanded_e12")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--edge-mode", default="full", choices=["full", "proximity", "proximity_combat"]
    )
    ap.add_argument("--checkpoint-template", default=PHASE_CHECKPOINT_TEMPLATE)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--controls", nargs="+", default=["node_shuffle"], choices=["node_shuffle", "edge_shuffle"]
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--max-matches", type=int, default=None, help="Limit test matches for smoke checks"
    )
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--domains", nargs="*", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
