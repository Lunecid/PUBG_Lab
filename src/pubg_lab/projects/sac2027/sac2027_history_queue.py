#!/usr/bin/env python3
"""Independent history conditions run concurrently after their frozen selections."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))
from pubg_lab.projects.sac2027.sac2027_followup_run import wait_for, BASE

KINDS = (
    "krafton_current_padded",
    "krafton_history_summary",
    "ptr_ordered_prefix",
    "ptr_shuffled_prefix",
)


def worker(cohort, kind):
    architecture = "ptr" if kind.startswith("ptr") else "krafton"
    wait_for([BASE / "tuning" / cohort / architecture / "selection.json"])
    with (BASE / f"history_{cohort}_{kind}.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "src/pubg_lab/projects/sac2027/sac2027_history_controls.py",
                "--cohort",
                cohort,
                "--kind",
                kind,
                "--device",
                "cpu",
            ],
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("COMPLETE", cohort, kind, flush=True)


def main():
    with ThreadPoolExecutor(max_workers=8) as executor:
        jobs = [
            executor.submit(worker, c, k)
            for c in ("erangel_tpp_squad", "taego_tpp_squad")
            for k in KINDS
        ]
        for job in jobs:
            job.result()
    (BASE / "history_complete.json").write_text('{"completed":true}\n')


if __name__ == "__main__":
    main()
