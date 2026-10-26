"""Evaluate a snapshot-phase checkpoint and write a train-style summary.

This fills the gap when a run was interrupted after saving the best checkpoint
but before writing ``*.summary.json``.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader

from pubg_lab.survival.snapshot_phase_dataset import snapshot_phase_collate
from pubg_lab.survival.train_snapshot_phase import (
    NO_GRAPH_VARIANTS,
    _device,
    _loader_worker_kwargs,
    _make_dataset,
    _make_model,
    _run_epoch,
)

logger = logging.getLogger("tools.eval_snapshot_phase_checkpoint")


def _namespace_from_checkpoint(
    ckpt: dict[str, Any], cli_args: argparse.Namespace
) -> SimpleNamespace:
    saved_args = dict(ckpt.get("args") or {})
    saved_args["checkpoint"] = str(cli_args.checkpoint)
    if cli_args.device is not None:
        saved_args["device"] = cli_args.device
    if cli_args.eval_workers is not None:
        saved_args["eval_workers"] = cli_args.eval_workers
    if cli_args.prefetch_factor is not None:
        saved_args["prefetch_factor"] = cli_args.prefetch_factor
    saved_args.setdefault("cached", False)
    saved_args.setdefault("batch", 2)
    saved_args.setdefault("eval_workers", 0)
    saved_args.setdefault("prefetch_factor", 2)
    saved_args.setdefault("hazard_pos_weight", 1.0)
    saved_args.setdefault("grad_clip", 5.0)
    saved_args.setdefault("edge_control_seed", 20260622)
    saved_args.setdefault("relative_edge_gate", "knn_combat")
    saved_args.setdefault("relative_edge_k", 5)
    saved_args.setdefault("relative_edge_null", "none")
    return SimpleNamespace(**saved_args)


def _loader(ids: list[str], args: SimpleNamespace, include_edges: bool) -> DataLoader:
    ds = _make_dataset(ids, cached=bool(args.cached), include_edges=include_edges)
    return DataLoader(
        ds,
        batch_size=int(args.batch),
        shuffle=False,
        collate_fn=snapshot_phase_collate,
        **_loader_worker_kwargs(int(args.eval_workers), int(args.prefetch_factor)),
    )


def evaluate(cli_args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checkpoint = Path(cli_args.checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu")
    args = _namespace_from_checkpoint(ckpt, cli_args)
    device = _device(args.device)
    model = _make_model(args).to(device)
    model.load_state_dict(ckpt["model_state"])

    include_edges = args.model_variant not in NO_GRAPH_VARIANTS
    val_ids = list(ckpt.get("val_ids") or [])
    test_ids = list(ckpt.get("test_ids") or [])
    if not val_ids and not test_ids:
        raise SystemExit(f"{checkpoint} has no val_ids/test_ids")

    logger.info(
        "evaluate checkpoint=%s variant=%s val=%d test=%d include_edges=%s",
        checkpoint,
        args.model_variant,
        len(val_ids),
        len(test_ids),
        include_edges,
    )

    summary: dict[str, Any] = {
        "model_variant": ckpt.get("model_variant", args.model_variant),
        "checkpoint": str(checkpoint),
        "history": str(checkpoint.with_suffix(".history.json")),
        "best_val": ckpt.get("best_val"),
        "best_epoch": ckpt.get("epoch"),
        "split_counts": {
            "train": len(ckpt.get("train_ids") or []),
            "val": len(val_ids),
            "test": len(test_ids),
            "train_loaded": len(ckpt.get("train_ids") or []),
            "val_loaded": len(val_ids),
            "test_loaded": len(test_ids),
        },
        "channel_contract": ckpt.get("channel_contract", {}),
    }

    with torch.no_grad():
        if val_ids:
            summary["best_val_metrics"] = _run_epoch(
                model,
                _loader(val_ids, args, include_edges),
                device,
                args,
                train=False,
            )
        if test_ids:
            summary["test_metrics"] = _run_epoch(
                model,
                _loader(test_ids, args, include_edges),
                device,
                args,
                train=False,
            )

    out_path = Path(cli_args.out) if cli_args.out else checkpoint.with_suffix(".summary.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_path)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-workers", type=int, default=None)
    ap.add_argument("--prefetch-factor", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
