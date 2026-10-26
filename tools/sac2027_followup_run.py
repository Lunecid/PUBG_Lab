#!/usr/bin/env python3
"""Dependency-aware local queues for pre-specified controls (no external scheduling)."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
BASE=REPO/"reports/sac2027_patch_v14"

def wait_for(paths):
    deadline=time.monotonic()+8*3600
    while not all(p.exists() for p in paths):
        if time.monotonic()>deadline:raise TimeoutError("upstream selections/results have not finished within 8 hours")
        time.sleep(20)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("lane",choices=["population_ptr","population_krafton","history","mapping"]);args=ap.parse_args()
    for cohort in ("erangel_tpp_squad","taego_tpp_squad"):
        if args.lane.startswith("population"):
            architecture=args.lane.split("_")[1]
            wait_for([BASE/"tuning"/cohort/architecture/"selection.json"])
            cmd=[sys.executable,"tools/sac2027_population_controls.py","--cohort",cohort,"--architecture",architecture,"--device","cpu"]
        elif args.lane=="history":
            wait_for([BASE/"tuning"/cohort/a/"selection.json" for a in ("ptr","krafton")])
            cmd=[sys.executable,"tools/sac2027_history_controls.py","--cohort",cohort,"--device","cpu"]
        else:
            wait_for([BASE/"objective_controls"/cohort/o/f"seed{s}"/"metrics.json" for o in ("weighted_full","bce") for s in (42,13,2026)])
            continue
        print("START",args.lane,cohort,flush=True);subprocess.run(cmd,cwd=REPO,check=True)
    if args.lane=="mapping":subprocess.run([sys.executable,"tools/sac2027_mapping_factorial.py"],cwd=REPO,check=True)
    (BASE/f"{args.lane}_complete.json").write_text('{"completed":true}\n')
    print("COMPLETE",args.lane,flush=True)

if __name__=="__main__":main()
