"""Diagnose the minimal snapshot relative-edge pilot.

This is a read-only pilot diagnostic.  It checks contact-gate coverage and
derived relative-edge feature scales on a split manifest before any larger
training run.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pubg_lab.survival.snapshot_edge_controls import apply_snapshot_edge_control
from pubg_lab.survival.snapshot_phase_dataset import fetch_snapshot_phase
from pubg_lab.survival.snapshot_phase_survival import phase_cut_indices
from pubg_lab.survival.snapshot_relative_edge import SnapshotRelativeEdgeHazard
from pubg_lab.data.stdb.snapshot_queries import SNAP_FEATURE_DIM, SNAP_FEATURE_NAMES


class FeatureStats:
    def __init__(self, names: list[str], *, sample_cap: int = 200_000) -> None:
        self.names = names
        self.sample_cap = sample_cap
        self.count = 0
        self.sum = torch.zeros(len(names), dtype=torch.float64)
        self.sumsq = torch.zeros(len(names), dtype=torch.float64)
        self.min = torch.full((len(names),), float("inf"), dtype=torch.float64)
        self.max = torch.full((len(names),), float("-inf"), dtype=torch.float64)
        self.zero = torch.zeros(len(names), dtype=torch.float64)
        self.samples: list[torch.Tensor] = []

    def add(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        v = values.detach().cpu().to(torch.float64)
        self.count += int(v.shape[0])
        self.sum += v.sum(dim=0)
        self.sumsq += (v * v).sum(dim=0)
        self.min = torch.minimum(self.min, v.min(dim=0).values)
        self.max = torch.maximum(self.max, v.max(dim=0).values)
        self.zero += (v.abs() < 1e-8).sum(dim=0).to(torch.float64)
        have = sum(s.shape[0] for s in self.samples)
        room = max(self.sample_cap - have, 0)
        if room <= 0:
            return
        if v.shape[0] > room:
            idx = torch.linspace(0, v.shape[0] - 1, room).long()
            v = v[idx]
        self.samples.append(v.to(torch.float32))

    def finish(self) -> dict[str, dict[str, float | int]]:
        if self.count == 0:
            return {name: {"count": 0} for name in self.names}
        mean = self.sum / self.count
        var = (self.sumsq / self.count) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=0.0))
        sample = torch.cat(self.samples, dim=0).to(torch.float64) if self.samples else None
        out: dict[str, dict[str, float | int]] = {}
        for i, name in enumerate(self.names):
            row: dict[str, float | int] = {
                "count": self.count,
                "mean": float(mean[i]),
                "std": float(std[i]),
                "min": float(self.min[i]),
                "max": float(self.max[i]),
                "zero_frac": float(self.zero[i] / self.count),
            }
            if sample is not None and sample.shape[0] > 0:
                qs = torch.quantile(
                    sample[:, i], torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64)
                )
                row.update({"q05": float(qs[0]), "q50": float(qs[1]), "q95": float(qs[2])})
            out[name] = row
        return out


def _load_split(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text())
    return {k: list(v) for k, v in data["splits"].items()}


def _pairwise_gate_edges(A: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    gate = (A > 0).to(dtype=A.dtype)
    alive = node_mask.bool()
    alive_block = alive[:, :, None, :, None] & alive[:, :, None, None, :]
    n = A.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=A.device).view(1, 1, 1, n, n)
    gate = torch.where(alive_block & ~eye, gate, torch.zeros_like(gate))
    row_sum = gate.sum(dim=-1, keepdim=True)
    return torch.where(row_sum > 0, gate / row_sum.clamp_min(1e-12), gate)


def _mode_A(A: torch.Tensor, node_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return A
    if mode == "train_gate":
        return apply_snapshot_edge_control(A, node_mask, control="gate_only")
    if mode == "pairwise_gate":
        return _pairwise_gate_edges(A, node_mask)
    raise ValueError(mode)


def _empty_counter() -> dict[str, float | int]:
    return {
        "matches": 0,
        "valid_snapshots": 0,
        "snapshots_with_edge": 0,
        "alive_nodes": 0,
        "alive_ordered_pairs": 0,
        "edges": 0,
        "alive_rows_with_out_edge": 0,
        "alive_rows_with_in_edge": 0,
        "target_cells": 0,
        "target_cells_out_edge_at_cut": 0,
        "target_cells_in_edge_at_cut": 0,
        "target_cells_any_edge_at_cut": 0,
    }


def _accumulate_counter(dst: dict[str, float | int], src: dict[str, float | int]) -> None:
    for key, value in src.items():
        dst[key] = int(dst.get(key, 0)) + int(value)


def _analyze_item(
    item: dict,
    model: SnapshotRelativeEdgeHazard,
    *,
    mode: str,
    stats: FeatureStats,
) -> dict[str, float | int]:
    X = item["X_s"].float().unsqueeze(0)
    A = item["A_s"].to_dense().float().unsqueeze(0)
    node_mask = item["node_mask_s"].bool().unsqueeze(0)
    phase_of_snap = item["phase_of_snap"].long().unsqueeze(0)
    target_mask = item["target_mask"].bool().unsqueeze(0)

    A_used = _mode_A(A, node_mask, mode)
    with torch.no_grad():
        edge_feat, edge_mask = model.build_edge_features(X, A_used, node_mask)

    valid_snap = node_mask.any(dim=-1)
    alive_n = node_mask.sum(dim=-1)
    alive_pairs = (alive_n * (alive_n - 1)).clamp_min(0)
    out_rows = edge_mask.any(dim=-1) & node_mask
    in_rows = edge_mask.any(dim=-2) & node_mask

    counter = _empty_counter()
    counter["matches"] = 1
    counter["valid_snapshots"] = int(valid_snap.sum())
    counter["snapshots_with_edge"] = int(edge_mask.any(dim=(-1, -2)).sum())
    counter["alive_nodes"] = int(node_mask.sum())
    counter["alive_ordered_pairs"] = int(alive_pairs.sum())
    counter["edges"] = int(edge_mask.sum())
    counter["alive_rows_with_out_edge"] = int(out_rows.sum())
    counter["alive_rows_with_in_edge"] = int(in_rows.sum())

    if edge_mask.any():
        stats.add(edge_feat[edge_mask])

    cut_idx, cut_valid = phase_cut_indices(phase_of_snap, node_mask, t_phase=8)
    for p in range(min(7, target_mask.shape[1])):
        if not bool(cut_valid[0, p]):
            continue
        t = int(cut_idx[0, p])
        tm = target_mask[0, p]
        if not bool(tm.any()):
            continue
        out_at = edge_mask[0, t].any(dim=-1)
        in_at = edge_mask[0, t].any(dim=-2)
        counter["target_cells"] += int(tm.sum())
        counter["target_cells_out_edge_at_cut"] += int((tm & out_at).sum())
        counter["target_cells_in_edge_at_cut"] += int((tm & in_at).sum())
        counter["target_cells_any_edge_at_cut"] += int((tm & (out_at | in_at)).sum())

    return counter


def _rates(counter: dict[str, float | int]) -> dict[str, float]:
    def div(a: str, b: str) -> float:
        denom = float(counter.get(b, 0))
        return float(counter.get(a, 0)) / denom if denom else 0.0

    return {
        "edge_density_alive_pairs": div("edges", "alive_ordered_pairs"),
        "snapshot_edge_coverage": div("snapshots_with_edge", "valid_snapshots"),
        "alive_row_out_coverage": div("alive_rows_with_out_edge", "alive_nodes"),
        "alive_row_in_coverage": div("alive_rows_with_in_edge", "alive_nodes"),
        "target_cut_out_coverage": div("target_cells_out_edge_at_cut", "target_cells"),
        "target_cut_in_coverage": div("target_cells_in_edge_at_cut", "target_cells"),
        "target_cut_any_coverage": div("target_cells_any_edge_at_cut", "target_cells"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split-json",
        type=Path,
        default=Path("model_stgcn/checkpoints/snapshot_pilots/split_limit20_seed42.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_diagnostics.json"
        ),
    )
    ap.add_argument("--modes", default="train_gate,raw,pairwise_gate")
    ap.add_argument("--relative-edge-gate", default="observed", choices=["observed", "knn_combat"])
    ap.add_argument("--relative-edge-k", type=int, default=5)
    ap.add_argument(
        "--relative-edge-null", default="none", choices=["none", "feature_roll", "topology_roll"]
    )
    args = ap.parse_args()

    splits = _load_split(args.split_json)
    modes = [m for m in args.modes.split(",") if m]
    model = SnapshotRelativeEdgeHazard(
        SNAP_FEATURE_DIM,
        feature_names=list(SNAP_FEATURE_NAMES),
        hidden=8,
        relative_edge_gate=args.relative_edge_gate,
        relative_edge_k=args.relative_edge_k,
        relative_edge_null=args.relative_edge_null,
    )
    report: dict[str, Any] = {
        "split_json": str(args.split_json),
        "relative_edge_gate": args.relative_edge_gate,
        "relative_edge_k": args.relative_edge_k,
        "relative_edge_null": args.relative_edge_null,
        "modes": {},
        "edge_feature_names": list(model.edge_feature_names),
    }
    for mode in modes:
        by_split: dict[str, Any] = {}
        total = _empty_counter()
        stats = FeatureStats(list(model.edge_feature_names))
        for split_name, ids in splits.items():
            split_counter = _empty_counter()
            for mid in ids:
                item = fetch_snapshot_phase(mid)
                if item is None:
                    continue
                c = _analyze_item(item, model, mode=mode, stats=stats)
                _accumulate_counter(split_counter, c)
                _accumulate_counter(total, c)
            by_split[split_name] = {
                "counts": split_counter,
                "rates": _rates(split_counter),
            }
        report["modes"][mode] = {
            "by_split": by_split,
            "total_counts": total,
            "total_rates": _rates(total),
            "feature_stats": stats.finish(),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    for mode, data in report["modes"].items():
        rates = data["total_rates"]
        counts = data["total_counts"]
        print(
            f"{mode}: edges={counts['edges']} density={rates['edge_density_alive_pairs']:.4f} "
            f"target_any={rates['target_cut_any_coverage']:.3f} "
            f"out={rates['target_cut_out_coverage']:.3f} in={rates['target_cut_in_coverage']:.3f}"
        )


if __name__ == "__main__":
    main()
