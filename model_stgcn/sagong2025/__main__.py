"""``python -m model_stgcn.sagong2025 {plan,extract,fit} ...``"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

DEFAULT_DUMP_DIR = "model_stgcn/dumps/bigcomp2027_phase_v1"
DEFAULT_PLAN_DIR = "model_stgcn/dumps/bigcomp2027_sagong_v1"
DEFAULT_RUNS_DIR = "model_stgcn/checkpoints/bigcomp2027_ptr_v1"


def _cohorts(dump_dir, only):
    from model_stgcn.ptr.cohorts import discover_cohorts

    have = discover_cohorts(dump_dir)
    if only:
        missing = [c for c in only if c not in have]
        if missing:
            raise SystemExit(f"unknown cohorts {missing}; available: {have}")
        return list(only)
    return have


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m model_stgcn.sagong2025")
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("plan", "extract", "fit"):
        p = sub.add_parser(name)
        p.add_argument("--dump-dir", default=DEFAULT_DUMP_DIR)
        p.add_argument("--plan-dir", default=DEFAULT_PLAN_DIR)
        p.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
        p.add_argument("--cohort", action="append", help="repeatable; default = every complete cohort of the dump dir")
        if name == "extract":
            p.add_argument("--workers", type=int, default=2)
            p.add_argument("--batch", type=int, default=6)
            p.add_argument("--limit", type=int, default=None, help="first N matches only (smoke test, writes nothing)")
            p.add_argument("--force", action="store_true")
        if name == "fit":
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--n-jobs", type=int, default=2)
            p.add_argument("--models", default="xgb,rf,lr,hgb", help="subset of xgb,rf,lr,hgb (hgb = HGB-cut classifier control)")
            p.add_argument("--scope", default="pooled", choices=("pooled", "per_phase"))
            p.add_argument("--sr-mode", default="paper", choices=("paper", "label"))
            p.add_argument("--out-dir", default=None, help="default: --runs-dir (pooled scope)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import warnings

    warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)  # numpy 2 + Accelerate noise
    cohorts = _cohorts(args.dump_dir, args.cohort)
    if args.stage == "plan":
        from model_stgcn.sagong2025.plan import build_plan

        for c in cohorts:
            a = build_plan(args.dump_dir, c, args.plan_dir, ref_runs=args.runs_dir)
            print(c, json.dumps({s: v["n_target"] for s, v in a["splits"].items()}), flush=True)
    elif args.stage == "extract":
        from model_stgcn.sagong2025.extract import run_extract

        for c in cohorts:
            r = run_extract(args.plan_dir, c, workers=args.workers, batch=args.batch, limit=args.limit, force=args.force)
            print(c, json.dumps({k: v for k, v in r.items() if k != "event_types"}), flush=True)
    else:
        from model_stgcn.sagong2025.fit import run_cohort

        for c in cohorts:
            run_cohort(args.plan_dir, c, args.out_dir or args.runs_dir, seed=args.seed, n_jobs=args.n_jobs,
                       models=tuple(args.models.split(",")), scope=args.scope, runs_ref=args.runs_dir, sr_mode=args.sr_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
