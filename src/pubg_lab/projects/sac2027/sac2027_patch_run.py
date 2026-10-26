#!/usr/bin/env python3
"""Launch isolated patch experiments after their data-validation gate completes."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = REPO_ROOT
BASE = REPO / "reports/sac2027_patch_v14"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lane", choices=["main", "neural_search", "gbdt_search"])
    args = ap.parse_args()
    deadline = time.monotonic() + 3600
    while not (BASE / "validation_complete.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("validation gate not completed within one hour")
        time.sleep(10)
    gate = json.loads((BASE / "validation_complete.json").read_text())
    assert not gate["errors"], gate
    py = sys.executable
    if args.lane == "main":
        cmd = [
            py,
            str(REPO / "src/pubg_lab/projects/battle_royale/rerun_v2.py"),
            "--dump-dir",
            str(BASE / "dumps/phase_v2"),
            "--runs-dir",
            str(BASE / "checkpoints/ptr_v2"),
            "--kf-dir",
            str(BASE / "dumps/kraftonfeat_v2"),
            "--krafton-data-dir",
            str(BASE / "dumps/krafton_winprob_v2"),
            "--sagong-dir",
            str(BASE / "dumps/sagong_v2"),
            "--results-dir",
            str(BASE / "results_v2"),
            "--krafton-reports-dir",
            str(BASE / "krafton_reports_v2"),
            "--paper-dir",
            str(BASE / "paper"),
            "--cpu-workers",
            "4",
            "--cpu-threads",
            "2",
            "--mps-streams",
            "1",
            "--mps-threads",
            "2",
            "--with-exploratory",
            "--with-m11r-union",
            "--leak-tests",
            "tests/test_phase_v2_unit.py tests/test_ptr_data.py tests/test_ptr_model.py",
            "--skip",
            "tables.,numbers.",
        ]
        print("MAIN RUN START", flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
    else:
        architectures = ["ptr", "krafton"] if args.lane == "neural_search" else ["gbdt"]
        for cohort in ["erangel_tpp_squad", "taego_tpp_squad"]:
            for architecture in architectures:
                cmd = [
                    py,
                    str(REPO / "src/pubg_lab/projects/sac2027/sac2027_fair_tune.py"),
                    "--cohort",
                    cohort,
                    "--architecture",
                    architecture,
                    "--device",
                    "cpu" if architecture == "gbdt" else "mps",
                ]
                print("SEARCH START", cohort, architecture, flush=True)
                subprocess.run(cmd, cwd=REPO, check=True)
    (BASE / f"{args.lane}_complete.json").write_text(json.dumps({"completed": True}) + "\n")


if __name__ == "__main__":
    main()
