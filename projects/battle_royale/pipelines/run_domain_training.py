"""Run map/mode-specific DynamicSTGNN training experiments.

This is an orchestration wrapper around ``model_stgcn.train_phase`` and
``model_stgcn.eval_phase``.  It keeps domain-specific splits/checkpoints in a
stable directory structure so map-wise and perspective-wise ablations do not
accidentally collapse into a single pooled experiment.
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
import subprocess
import sys
from pathlib import Path

from pubg_lab.survival.config import phase_checkpoint_name
from pubg_lab.data.stdb.queries.queries import list_phase_eligible_match_ids


PRIMARY_TPP_DOMAINS = [
    ("Baltic_Main", "squad"),
    ("Tiger_Main", "squad"),
    ("Desert_Main", "squad"),
    ("Neon_Main", "squad"),
    ("Savage_Main", "squad"),
]

FPP_AUX_DOMAINS = [
    ("Baltic_Main", "squad-fpp"),
    ("Tiger_Main", "squad-fpp"),
    ("Desert_Main", "squad-fpp"),
    ("Neon_Main", "squad-fpp"),
]

EDGE_MODES = ["none", "proximity", "proximity_combat", "full"]


def _slug(map_name: str, mode: str) -> str:
    return f"{map_name}__{mode}".replace("/", "_")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _domains(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.map and args.mode:
        return [(args.map, args.mode)]
    if args.preset == "primary_tpp":
        return PRIMARY_TPP_DOMAINS
    if args.preset == "fpp_aux":
        return FPP_AUX_DOMAINS
    if args.preset == "primary_plus_fpp":
        return PRIMARY_TPP_DOMAINS + FPP_AUX_DOMAINS
    raise ValueError(f"unknown preset: {args.preset}")


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_dir)
    edge_modes = args.edge_modes or EDGE_MODES

    for map_name, mode in _domains(args):
        ids = list_phase_eligible_match_ids(map_name, mode, args.limit)
        n = len(ids)
        if n < args.min_matches:
            print(f"skip {map_name}/{mode}: n={n} < min_matches={args.min_matches}", flush=True)
            continue

        domain_dir = out_root / _slug(map_name, mode)
        domain_dir.mkdir(parents=True, exist_ok=True)
        split_json = domain_dir / f"split_seed{args.seed}.json"

        print(f"\n== {map_name}/{mode}: {n} graph-eligible matches ==", flush=True)
        for edge_mode in edge_modes:
            ckpt = domain_dir / phase_checkpoint_name(edge_mode, args.epochs)
            eval_out = domain_dir / f"eval_{edge_mode}_e{args.epochs}_test.json"

            if ckpt.exists() and not args.force:
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
                    str(args.seed),
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
                    str(args.seed),
                ]
                if args.cached:
                    eval_cmd.append("--cached")
                _run(eval_cmd, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preset", default="primary_tpp", choices=["primary_tpp", "fpp_aux", "primary_plus_fpp"]
    )
    ap.add_argument("--map", default=None, help="Run one explicit map; requires --mode")
    ap.add_argument("--mode", default=None, help="Run one explicit mode; requires --map")
    ap.add_argument("--edge-modes", nargs="*", default=None, choices=EDGE_MODES)
    ap.add_argument("--out-dir", default="model_stgcn/checkpoints/domain_runs")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-matches", type=int, default=300)
    ap.add_argument(
        "--cached", action="store_true", help="Cache train tensors in memory per domain"
    )
    ap.add_argument(
        "--eval", action="store_true", help="Evaluate each checkpoint on its domain test split"
    )
    ap.add_argument("--force", action="store_true", help="Retrain even if checkpoint exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if bool(args.map) != bool(args.mode):
        ap.error("--map and --mode must be provided together")
    return args


if __name__ == "__main__":
    run(parse_args())
