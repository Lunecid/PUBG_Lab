#!/usr/bin/env python3
"""Dependency-aware local queues for pre-specified controls (no external scheduling)."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = REPO_ROOT
BASE = REPO / "reports/sac2027_patch_v14"


def wait_for(paths):
    deadline = time.monotonic() + 8 * 3600
    while not all(p.exists() for p in paths):
        if time.monotonic() > deadline:
            raise TimeoutError("upstream selections/results have not finished within 8 hours")
        time.sleep(20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lane", choices=["population_ptr", "population_krafton", "history", "mapping"])
    args = ap.parse_args()
    for cohort in ("erangel_tpp_squad", "taego_tpp_squad"):
        if args.lane.startswith("population"):
            architecture = args.lane.split("_")[1]
            wait_for([BASE / "tuning" / cohort / architecture / "selection.json"])
            cmd = [
                sys.executable,
                "projects/sac2027/evaluation/controls/sac2027_population_controls.py",
                "--cohort",
                cohort,
                "--architecture",
                architecture,
                "--device",
                "cpu",
            ]
        elif args.lane == "history":
            wait_for([BASE / "tuning" / cohort / a / "selection.json" for a in ("ptr", "krafton")])
            cmd = [
                sys.executable,
                "projects/sac2027/evaluation/history/sac2027_history_controls.py",
                "--cohort",
                cohort,
                "--device",
                "cpu",
            ]
        else:
            wait_for(
                [
                    BASE / "objective_controls" / cohort / o / f"seed{s}" / "metrics.json"
                    for o in ("weighted_full", "bce")
                    for s in (42, 13, 2026)
                ]
            )
            continue
        print("START", args.lane, cohort, flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
    if args.lane == "mapping":
        subprocess.run(
            [sys.executable, "projects/sac2027/evaluation/controls/sac2027_mapping_factorial.py"],
            cwd=REPO,
            check=True,
        )
    (BASE / f"{args.lane}_complete.json").write_text('{"completed":true}\n')
    print("COMPLETE", args.lane, flush=True)


if __name__ == "__main__":
    main()
