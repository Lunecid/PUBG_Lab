"""Run and aggregate required seed-repeat DynamicSTGNN experiments.

The paper-facing goal is to check whether the graph gain is stable across
match-level splits and initialization seeds.  This wrapper intentionally keeps
each seed in its own directory so split manifests and checkpoints cannot
overwrite the existing seed-42 ablation outputs.

Example:
    python -m model_stgcn.seed_repeat_experiments --eval --seeds 13 42 2026
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from projects.battle_royale.pipelines.run_domain_training import PRIMARY_TPP_DOMAINS
from pubg_lab.data.stdb.queries.queries import list_phase_eligible_match_ids


DEFAULT_EDGE_MODES = ("none", "proximity_combat", "full")
DEFAULT_SEEDS = (13, 42, 2026)


def _slug(map_name: str, mode: str) -> str:
    return f"{map_name}__{mode}".replace("/", "_")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def _domains(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.map and args.mode:
        return [(args.map, args.mode)]
    return PRIMARY_TPP_DOMAINS


def _metric_from_eval(
    eval_path: Path, *, seed: int, map_name: str, mode: str, edge_mode: str
) -> dict[str, Any]:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    hazard = data.get("hazard", {})
    split_counts = data.get("split_counts", {})
    return {
        "seed": seed,
        "map_name": map_name,
        "mode": mode,
        "edge_mode": edge_mode,
        "total_matches": split_counts.get("total"),
        "test_matches": split_counts.get("test"),
        "loss": data.get("loss"),
        "hazard_auc": hazard.get("auc"),
        "hazard_ap": hazard.get("ap"),
        "hazard_acc": hazard.get("acc"),
        "hazard_n": hazard.get("n"),
        "hazard_pos_rate": hazard.get("pos_rate"),
        "rank_mae_norm": data.get("rank_mae_norm"),
        "top5_auc": data.get("top5", {}).get("auc"),
        "top10_auc": data.get("top10", {}).get("auc"),
        "checkpoint_epoch": data.get("checkpoint_epoch"),
        "checkpoint_best_val": data.get("checkpoint_best_val"),
        "eval_path": str(eval_path),
    }


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


def _aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_base = {(r["seed"], r["map_name"], r["mode"]): r for r in rows if r["edge_mode"] == "none"}
    deltas: list[dict[str, Any]] = []
    for row in rows:
        if row["edge_mode"] == "none":
            continue
        base = by_base.get((row["seed"], row["map_name"], row["mode"]))
        if not base:
            continue
        rec = {
            "seed": row["seed"],
            "map_name": row["map_name"],
            "mode": row["mode"],
            "edge_mode": row["edge_mode"],
        }
        for metric in ("loss", "hazard_auc", "hazard_ap", "hazard_acc", "rank_mae_norm"):
            left = row.get(metric)
            right = base.get(metric)
            if left is None or right is None:
                rec[f"delta_{metric}"] = None
            else:
                rec[f"delta_{metric}"] = float(left) - float(right)
        deltas.append(rec)

    summary: list[dict[str, Any]] = []
    edge_modes = sorted({r["edge_mode"] for r in rows})
    for edge_mode in edge_modes:
        subset = [r for r in rows if r["edge_mode"] == edge_mode]
        rec: dict[str, Any] = {
            "edge_mode": edge_mode,
            "n_runs": len(subset),
            "n_seeds": len({r["seed"] for r in subset}),
            "n_domains": len({(r["map_name"], r["mode"]) for r in subset}),
        }
        for metric in ("loss", "hazard_auc", "hazard_ap", "hazard_acc"):
            vals = np.array(
                [float(r[metric]) for r in subset if r.get(metric) is not None], dtype=float
            )
            rec[f"{metric}_mean"] = float(np.nanmean(vals)) if vals.size else float("nan")
            rec[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else float("nan")
        summary.append(rec)

    for edge_mode in sorted({r["edge_mode"] for r in deltas}):
        subset = [r for r in deltas if r["edge_mode"] == edge_mode]
        rec = {
            "edge_mode": f"{edge_mode}_vs_none",
            "n_runs": len(subset),
            "n_seeds": len({r["seed"] for r in subset}),
            "n_domains": len({(r["map_name"], r["mode"]) for r in subset}),
        }
        for metric in ("delta_loss", "delta_hazard_auc", "delta_hazard_ap", "delta_hazard_acc"):
            vals = np.array(
                [float(r[metric]) for r in subset if r.get(metric) is not None], dtype=float
            )
            rec[f"{metric}_mean"] = float(np.nanmean(vals)) if vals.size else float("nan")
            rec[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else float("nan")
            if metric != "delta_loss":
                rec[f"{metric}_positive"] = int(np.sum(vals > 0)) if vals.size else 0
        summary.append(rec)

    return deltas, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_dir)
    rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        for map_name, mode in _domains(args):
            ids = list_phase_eligible_match_ids(map_name, mode, args.limit)
            if len(ids) < args.min_matches:
                print(
                    f"skip {map_name}/{mode} seed={seed}: n={len(ids)} < {args.min_matches}",
                    flush=True,
                )
                continue

            domain_dir = out_root / f"seed_{seed}" / _slug(map_name, mode)
            domain_dir.mkdir(parents=True, exist_ok=True)
            split_json = domain_dir / f"split_seed{seed}.json"
            print(f"\n== seed={seed} {map_name}/{mode}: {len(ids)} matches ==", flush=True)

            for edge_mode in args.edge_modes:
                ckpt = domain_dir / f"phase_{edge_mode}_e{args.epochs}.pt"
                eval_out = domain_dir / f"eval_{edge_mode}_e{args.epochs}_test.json"

                if args.skip_train:
                    print(f"skip train requested {ckpt}", flush=True)
                elif ckpt.exists() and not args.force:
                    print(f"skip train existing {ckpt}", flush=True)
                else:
                    train_cmd = [
                        sys.executable,
                        "-m",
                        "pubg_lab.survival.training.train_phase",
                        "--map",
                        map_name,
                        "--mode",
                        mode,
                        "--split-json",
                        str(split_json),
                        "--epochs",
                        str(args.epochs),
                        "--batch",
                        str(args.batch),
                        "--edge-mode",
                        edge_mode,
                        "--checkpoint",
                        str(ckpt),
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                    ]
                    if args.cached:
                        train_cmd.append("--cached")
                    _run(train_cmd, dry_run=args.dry_run)

                if args.eval and (ckpt.exists() or args.dry_run):
                    eval_cmd = [
                        sys.executable,
                        "-m",
                        "pubg_lab.survival.evaluation.eval_phase",
                        "--checkpoint",
                        str(ckpt),
                        "--map",
                        map_name,
                        "--mode",
                        mode,
                        "--split-json",
                        str(split_json),
                        "--split",
                        "test",
                        "--batch",
                        str(args.batch),
                        "--out",
                        str(eval_out),
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                    ]
                    if args.cached:
                        eval_cmd.append("--cached")
                    _run(eval_cmd, dry_run=args.dry_run)

                if eval_out.exists():
                    rows.append(
                        _metric_from_eval(
                            eval_out, seed=seed, map_name=map_name, mode=mode, edge_mode=edge_mode
                        )
                    )

    metrics_path = out_root / "seed_repeat_metrics.csv"
    deltas_path = out_root / "seed_repeat_deltas.csv"
    summary_path = out_root / "seed_repeat_summary.json"
    _write_csv(metrics_path, rows)
    deltas, summary = _aggregate(rows)
    _write_csv(deltas_path, deltas)
    summary_payload = {
        "seeds": list(args.seeds),
        "edge_modes": list(args.edge_modes),
        "epochs": args.epochs,
        "metrics_path": str(metrics_path),
        "deltas_path": str(deltas_path),
        "summary": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    return summary_payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument(
        "--edge-modes",
        nargs="+",
        default=list(DEFAULT_EDGE_MODES),
        choices=["none", "proximity", "proximity_combat", "full"],
    )
    ap.add_argument("--map", default=None, help="Run one explicit map; requires --mode")
    ap.add_argument("--mode", default=None, help="Run one explicit mode; requires --map")
    ap.add_argument(
        "--out-dir", default="model_stgcn/checkpoints/required_experiments/seed_repeats"
    )
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-matches", type=int, default=300)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if bool(args.map) != bool(args.mode):
        ap.error("--map and --mode must be provided together")
    return args


if __name__ == "__main__":
    run(parse_args())
