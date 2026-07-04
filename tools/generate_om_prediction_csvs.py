"""Generate per-team-phase prediction CSVs for a completed OM ladder run.

The completed OM result JSONs contain aggregate metrics. Paired bootstrap needs
row-level predictions with match identifiers, so this utility re-runs eval only
on saved checkpoints and writes `--pred-out` CSVs. It does not train models.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOT = Path("model_stgcn/checkpoints/br_om_ladder_required_v1")
DEFAULT_METRICS = DEFAULT_ROOT / "om_ladder_metrics_all_cohorts.csv"
DEFAULT_PRED_DIR = DEFAULT_ROOT / "predictions"


def _split_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def _load_rows(path: Path, cohorts: set[str] | None, presets: set[str] | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, str]] = []
    for row in rows:
        if cohorts and row["cohort_slug"] not in cohorts and row["cohort"] not in cohorts:
            continue
        if presets and row["feature_preset"] not in presets and row["name"] not in presets:
            continue
        out.append(row)
    out.sort(key=lambda r: (r["cohort_slug"], r["feature_preset"]))
    return out


def _run_eval(row: dict[str, str], args: argparse.Namespace) -> None:
    pred_dir = Path(args.pred_dir) / row["cohort_slug"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    preset = row["feature_preset"]
    pred_out = pred_dir / f"{preset}.pred.csv"
    eval_out = pred_dir / f"{preset}.eval.json"
    if pred_out.exists() and eval_out.exists() and not args.force:
        print(f"skip existing {row['cohort']} / {preset}", flush=True)
        return
    split_json = Path(row["run_dir"]) / "split.json"
    cmd = [
        sys.executable,
        "-m",
        "model_stgcn.v3.eval",
        "--checkpoint",
        row["checkpoint"],
        "--split",
        args.split,
        "--split-json",
        str(split_json),
        "--seed",
        args.seed,
        "--batch",
        args.batch,
        "--device",
        args.device,
        "--cohort",
        row["cohort"],
        "--edge-mode",
        row["edge_mode"],
        "--edge-control",
        row["edge_control"],
        "--out",
        str(eval_out),
        "--pred-out",
        str(pred_out),
    ]
    if args.cached:
        cmd.append("--cached")
    print(f"eval {row['cohort']} / {preset}", flush=True)
    if args.verbose:
        print("+ " + " ".join(cmd), flush=True)
    env = dict(os.environ)
    if args.pg_password:
        env["PG_PASSWORD"] = args.pg_password
    if args.verbose:
        subprocess.run(cmd, cwd=args.cwd, env=env, check=True)
    else:
        result = subprocess.run(
            cmd,
            cwd=args.cwd,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout, flush=True)
            raise subprocess.CalledProcessError(result.returncode, cmd)
    n_lines = 0
    if pred_out.exists():
        with pred_out.open("r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
    print(f"  wrote {pred_out} ({max(n_lines - 1, 0)} rows)", flush=True)


def run(args: argparse.Namespace) -> None:
    rows = _load_rows(Path(args.metrics_csv), _split_csv(args.cohorts), _split_csv(args.presets))
    if not rows:
        raise SystemExit("no OM rows matched the requested filters")
    print(f"matched {len(rows)} OM eval rows", flush=True)
    for row in rows:
        _run_eval(row, args)
    print("prediction CSV generation complete", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", default=str(DEFAULT_METRICS))
    parser.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR))
    parser.add_argument("--cohorts", default=None, help="comma-separated cohort labels or slugs")
    parser.add_argument("--presets", default=None, help="comma-separated OM feature presets")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--seed", default="42")
    parser.add_argument("--batch", default="8")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cached", action="store_true", default=True)
    parser.add_argument("--no-cached", dest="cached", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pg-password", default=os.environ.get("PG_PASSWORD"))
    parser.add_argument("--cwd", default=str(Path.cwd()))
    parser.add_argument("--verbose", action="store_true", help="stream full eval logs")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
