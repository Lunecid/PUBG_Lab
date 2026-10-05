"""Sequential runner of the KRAFTON baseline matrix (ONE training process at a time, MPS or CPU).

    python -m adaptation.run_all --stage main            # 12 cohorts x {transformer, mlp}, seed 42
    python -m adaptation.run_all --stage seeds           # seeds 13 and 2026 for erangel_tpp_squad and taego_tpp_squad
    python -m adaptation.run_all --stage ourfeat         # K2 = KRAFTON-tf-ourfeat, 12 cohorts, seed 42 (needs the KRAFTON-tf runs)
    python -m adaptation.run_all --stage hgbkf           # HGB-kraftonfeat, 12 cohorts, seed 42 (needs the feature table)
    python -m adaptation.run_all --stage k3              # K3a (bce) and K3 (hazard) per cohort, 12 cohorts, seed 42 (needs K1 + K2 runs)

Every run is a subprocess (``python -m adaptation.train``) whose stdout / stderr go to
``<out>/logs/<cohort>__<arm>__seed<seed>.log``; the status of every run is appended to
``<out>/krafton_run_status.csv`` (a run whose metrics.json already exists is skipped, a failed run is recorded and the
matrix continues).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DEFAULT_OUT = REPO / "model_stgcn/checkpoints/bigcomp2027_ptr_v1"
DEFAULT_DATA = REPO / "model_stgcn/dumps/krafton_winprob_v1"
ARM = {"transformer": "KRAFTON-tf", "mlp": "KRAFTON-mlp", "ourfeat": "KRAFTON-tf-ourfeat", "hgbkf": "HGB-kraftonfeat",
       "bce": "KRAFTON-tf-ourfeat-bce", "hazard": "KRAFTON-tf-ourfeat-hazard"}
# small cohorts first: failures show up early; the two large cohorts last
ORDER = ["paramo_tpp_squad", "karakin_tpp_squad", "erangel_fpp_squad", "taego_fpp_squad", "erangel_tpp_solo", "taego_tpp_duo",
         "erangel_tpp_duo", "sanhok_tpp_squad", "rondo_tpp_squad", "miramar_desert_tpp_squad", "erangel_tpp_squad", "taego_tpp_squad"]  # fmt: skip
FIELDS = ["run_id", "cohort", "arm", "seed", "status", "exit_code", "start_utc", "end_utc", "elapsed_sec", "log", "metrics", "note"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["main", "seeds", "ourfeat", "hgbkf", "k3"], default="main")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default="", help="comma separated cohorts")
    ap.add_argument("--encoders", default="transformer,mlp")
    a = ap.parse_args(argv)
    cohorts = [c for c in a.only.split(",") if c] or ORDER
    if a.stage == "seeds":
        runs = [(c, e, s) for c in ("erangel_tpp_squad", "taego_tpp_squad") for s in (13, 2026) for e in a.encoders.split(",") if c in cohorts]
    elif a.stage == "ourfeat":
        runs = [(c, "ourfeat", 42) for c in cohorts]
    elif a.stage == "hgbkf":
        runs = [(c, "hgbkf", 42) for c in cohorts]
    elif a.stage == "k3":
        runs = [(c, v, 42) for c in cohorts for v in ("bce", "hazard")]
    else:
        runs = [(c, e, 42) for c in cohorts for e in a.encoders.split(",")]
    status = a.out_dir / "krafton_run_status.csv"
    (a.out_dir / "logs").mkdir(parents=True, exist_ok=True)
    new = not status.exists()
    for cohort, enc, seed in runs:
        arm = ARM[enc]
        out = a.out_dir / cohort / arm / f"seed{seed}"
        run_id = f"{cohort}/{arm}/seed{seed}"
        if (out / "metrics.json").is_file():
            print(f"skip {run_id} (done)", flush=True)
            continue
        log = a.out_dir / "logs" / f"{cohort}__{arm}__seed{seed}.log"
        if enc == "hgbkf":
            cmd = [sys.executable, "-m", "adaptation.hgb_kraftonfeat", "--cohort", cohort, "--seed", str(seed), "--out-dir", str(a.out_dir),
                   "--data-dir", str(a.data_dir), "--threads", "1"]  # fmt: skip
        elif enc in ("bce", "hazard"):
            cmd = [sys.executable, "-m", "adaptation.ourfeat_hazard", "--cohort", cohort, "--variant", enc, "--seed", str(seed), "--out-dir", str(a.out_dir),
                   "--data-dir", str(a.data_dir), "--device", a.device, "--threads", str(min(a.threads, 3))]  # fmt: skip
        elif enc == "ourfeat":
            cmd = [sys.executable, "-m", "adaptation.ourfeat", "--cohort", cohort, "--seed", str(seed), "--out-dir", str(a.out_dir),
                   "--data-dir", str(a.data_dir), "--device", a.device, "--threads", str(a.threads)]  # fmt: skip
        else:
            cmd = [sys.executable, "-m", "adaptation.train", "--cohort", cohort, "--encoder", enc, "--seed", str(seed), "--out-dir", str(a.out_dir),
                   "--data-dir", str(a.data_dir), "--device", a.device, "--threads", str(a.threads)]  # fmt: skip
        t0 = time.time()
        start = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{start}] start {run_id}", flush=True)
        with open(log, "w") as lf:
            env = dict(os.environ)
            if enc == "hgbkf":
                env["OMP_NUM_THREADS"] = "1"  # resource cap: the boosting runs next to a training process
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
        ok = rc == 0 and (out / "metrics.json").is_file()
        row = {"run_id": run_id, "cohort": cohort, "arm": arm, "seed": seed, "status": "done" if ok else "failed", "exit_code": rc,
               "start_utc": start, "end_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "elapsed_sec": round(time.time() - t0, 1),
               "log": str(log.relative_to(REPO)), "metrics": str((out / "metrics.json").relative_to(REPO)) if ok else "", "note": ""}  # fmt: skip
        with open(status, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
                new = False
            w.writerow(row)
        print(f"  -> {row['status']} in {row['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
