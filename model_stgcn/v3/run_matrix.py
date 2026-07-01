"""Run V3 experiment groups with a shared split.

The runner shells out to `model_stgcn.v3.train` and `model_stgcn.v3.eval` so
each experiment releases cached tensors before the next model starts.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from model_stgcn.v3.experiment_matrix import EXPERIMENT_GROUPS, all_experiments


def _cmd_train(row: dict[str, str], args: argparse.Namespace, checkpoint: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "model_stgcn.v3.train",
        "--model",
        row["model"],
        "--graph-encoder",
        row["graph_encoder"],
        "--global-fusion",
        row["global_fusion"],
        "--team-decoder",
        row["team_decoder"],
        "--pairwise-rank-weight",
        row["pairwise_rank_weight"],
        "--edge-mode",
        row["edge_mode"],
        "--edge-control",
        row.get("edge_control", "observed"),
        "--feature-preset",
        row.get("feature_preset", "full"),
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--log-every",
        str(args.log_every),
        "--device",
        args.device,
        "--checkpoint",
        str(checkpoint),
        "--split-json",
        str(args.split_json),
        "--seed",
        str(args.seed),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.map is not None:
        cmd += ["--map", args.map]
    if args.mode is not None:
        cmd += ["--mode", args.mode]
    if args.cached:
        cmd.append("--cached")
    if args.combat_in:
        cmd.append("--combat-in")
    if args.no_v2_features:
        cmd.append("--no-v2-features")
    return cmd


def _cmd_eval(row: dict[str, str], args: argparse.Namespace, checkpoint: Path, out: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "model_stgcn.v3.eval",
        "--checkpoint",
        str(checkpoint),
        "--split",
        args.eval_split,
        "--edge-mode",
        row["edge_mode"],
        "--edge-control",
        row.get("edge_control", "observed"),
        "--split-json",
        str(args.split_json),
        "--seed",
        str(args.seed),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--out",
        str(out),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.map is not None:
        cmd += ["--map", args.map]
    if args.mode is not None:
        cmd += ["--mode", args.mode]
    if args.cached:
        cmd.append("--cached")
    return cmd


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _read_metric(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "group",
        "name",
        "feature_preset",
        "model",
        "graph_encoder",
        "global_fusion",
        "edge_mode",
        "edge_control",
        "team_decoder",
        "pairwise_rank_weight",
        "n",
        "pos_rate",
        "acc",
        "bce",
        "brier",
        "ece",
        "auc",
        "ap",
        "rank_mae_norm",
        "rank_spearman",
        "rank_pairwise_acc",
        "top5_auc",
        "top5_ap",
        "top5_brier",
        "top10_auc",
        "top10_ap",
        "top10_brier",
        "checkpoint",
        "eval_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.split_json is None:
        args.split_json = root / "split.json"
    else:
        args.split_json = Path(args.split_json)

    groups = args.groups.split(",") if args.groups else ["main"]
    unknown = [g for g in groups if g not in EXPERIMENT_GROUPS]
    if unknown:
        raise SystemExit(f"unknown groups: {unknown}; choices={sorted(EXPERIMENT_GROUPS)}")
    experiments = all_experiments(groups=groups)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        experiments = [row for row in experiments if row["name"] in wanted]

    started = time.time()
    metric_rows: list[dict[str, Any]] = []
    manifest = {
        "out_dir": str(root),
        "groups": groups,
        "epochs": args.epochs,
        "limit": args.limit,
        "map": args.map,
        "mode": args.mode,
        "seed": args.seed,
        "split_json": str(args.split_json),
        "eval_only": args.eval_only,
        "force": args.force,
        "force_eval": args.force_eval,
        "experiments": experiments,
    }
    (root / "matrix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for row in experiments:
        exp_dir = root / row["group"] / row["name"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = exp_dir / "model.pt"
        eval_json = exp_dir / f"eval_{args.eval_split}.json"

        if args.eval_only and not checkpoint.exists():
            raise SystemExit(f"missing checkpoint for --eval-only: {checkpoint}")

        if not args.eval_only and (not checkpoint.exists() or args.force):
            _run(_cmd_train(row, args, checkpoint), cwd=Path.cwd())
        else:
            print(f"skip existing checkpoint {checkpoint}", flush=True)

        if not eval_json.exists() or args.force or args.force_eval:
            _run(_cmd_eval(row, args, checkpoint, eval_json), cwd=Path.cwd())
        else:
            print(f"skip existing eval {eval_json}", flush=True)

        metric = _read_metric(eval_json)
        metric_rows.append(
            {
                **row,
                **{
                    k: metric.get(k)
                    for k in (
                        "n",
                        "pos_rate",
                        "acc",
                        "bce",
                        "brier",
                        "ece",
                        "auc",
                        "ap",
                        "rank_mae_norm",
                        "rank_spearman",
                        "rank_pairwise_acc",
                        "top5_auc",
                        "top5_ap",
                        "top5_brier",
                        "top10_auc",
                        "top10_ap",
                        "top10_brier",
                    )
                },
                "checkpoint": str(checkpoint),
                "eval_json": str(eval_json),
            }
        )
        _write_summary(root / "matrix_metrics.csv", metric_rows)

    summary = {
        "elapsed_s": round(time.time() - started, 3),
        "metrics": metric_rows,
        "metrics_csv": str(root / "matrix_metrics.csv"),
    }
    (root / "matrix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="main", help="comma-separated: main,resolution,decoder,graph_control,graph_relation,feature")
    ap.add_argument("--only", default=None, help="comma-separated experiment names")
    ap.add_argument("--out-dir", default="model_stgcn/checkpoints/v3_matrix")
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--map", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--combat-in", action="store_true")
    ap.add_argument("--no-v2-features", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-eval", action="store_true", help="recompute eval JSON/CSV without forcing retrain")
    ap.add_argument("--eval-only", action="store_true", help="require existing checkpoints and only run evaluation")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
