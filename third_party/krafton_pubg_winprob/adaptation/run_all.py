"""Sequential runner of the KRAFTON baseline matrix (ONE training process at a time, MPS or CPU) and the stages that hang off it.

    python -m adaptation.run_all --stage main            # 12 cohorts x {transformer, mlp}, seed 42
    python -m adaptation.run_all --stage seeds           # seeds 13 and 2026 for erangel_tpp_squad and taego_tpp_squad
    python -m adaptation.run_all --stage ourfeat         # K2 = KRAFTON-tf-ourfeat, 12 cohorts, seed 42 (needs the KRAFTON-tf runs)
    python -m adaptation.run_all --stage hgbkf           # HGB-kraftonfeat, 12 cohorts, seed 42 (needs the feature table)
    python -m adaptation.run_all --stage k3              # K3a (bce) and K3 (hazard) per cohort, 12 cohorts, seed 42 (needs K1 + K2 runs)
    python -m adaptation.run_all --stage union           # HGB-union (post hoc, design doc 4.6): run_matrix --union on the kf table (CPU workers)
    python -m adaptation.run_all --stage h0              # the H0 arms (adaptation.h0_baseline): needs the K1 and K2 checkpoints

Every path is explicit and forwarded to every child (K1, K2, K3a, K3, HGB-kraftonfeat, union, h0); the defaults are the v2 roots of
``model_stgcn.ptr.paths``:

    --out-dir    run directory <out>/<cohort>/<arm>/seed<s>/ (PTR arms are read from it for the row assertion; the KRAFTON arms are written to it)
    --data-dir   KRAFTON label tables, shards and extraction logs (``adaptation.labels`` / ``extract``)
    --dump-dir   phase dump (K2 / K3 / HGB-kraftonfeat / union / h0 read the tokens and labels from it)
    --table-dir  KRAFTON feature table (``adaptation.kftable``; HGB-kraftonfeat and the union runs)

Before anything starts the dump version (``meta['version']``) is compared with the version of the other three roots: a v1 root next to a v2 dump
(or the reverse) is refused, and the run directory is stamped (``.rerun_stamp.json``).  Nothing is read from or written to a v1 root.

Every run is a subprocess (``python -m adaptation.train`` ...) whose stdout / stderr go to ``<out>/logs/<cohort>__<arm>__seed<seed>.log``; the status of
every run is appended to ``<out>/<status-name>`` (default ``krafton_run_status.csv``; give two concurrent streams two names).  A run whose
``metrics.json`` already exists is skipped (resumable), a failed run is recorded and the matrix continues; the exit code is 1 if any run failed.
``--dry-run`` prints the commands and writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
for _p in (str(ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_stgcn.ptr import paths as P  # noqa: E402  (the v2 roots and the version guard)

DEFAULT_OUT = P.RUNS_DIR
DEFAULT_DATA = P.KRAFTON_DATA_DIR
DEFAULT_DUMP = P.DUMP_DIR
DEFAULT_TABLE = P.KF_DIR
STAGES = ("main", "seeds", "ourfeat", "hgbkf", "k3", "union", "h0")
ARM = {"transformer": "KRAFTON-tf", "mlp": "KRAFTON-mlp", "ourfeat": "KRAFTON-tf-ourfeat", "hgbkf": "HGB-kraftonfeat",
       "bce": "KRAFTON-tf-ourfeat-bce", "hazard": "KRAFTON-tf-ourfeat-hazard"}
# small cohorts first: failures show up early; the two large cohorts last
ORDER = ["paramo_tpp_squad", "karakin_tpp_squad", "erangel_fpp_squad", "taego_fpp_squad", "erangel_tpp_solo", "taego_tpp_duo",
         "erangel_tpp_duo", "sanhok_tpp_squad", "rondo_tpp_squad", "miramar_desert_tpp_squad", "erangel_tpp_squad", "taego_tpp_squad"]  # fmt: skip
FIELDS = ["run_id", "cohort", "arm", "seed", "status", "exit_code", "start_utc", "end_utc", "elapsed_sec", "log", "metrics", "note"]
UNION_ARMS = ("HGB-union", "M11R-union")


@dataclass(frozen=True)
class Roots:
    """The four roots every child is given (absolute paths: the children run in ``third_party/krafton_pubg_winprob``)."""

    out: Path = DEFAULT_OUT
    data: Path = DEFAULT_DATA
    dump: Path = DEFAULT_DUMP
    table: Path = DEFAULT_TABLE

    def __post_init__(self) -> None:
        for name in ("out", "data", "dump", "table"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())

    def check_versions(self) -> str:
        """Refuse a v1 root next to a v2 dump (or the reverse); returns the dump's version."""
        try:
            return P.check_same_version(self.dump, {"runs (--out-dir)": self.out, "KRAFTON data (--data-dir)": self.data, "kf table (--table-dir)": self.table})
        except P.VersionMixError as exc:
            raise SystemExit(f"refusing to run: {exc}") from exc


@dataclass
class Job:
    """One child process of the matrix."""

    run_id: str
    cohort: str
    arm: str
    seed: int
    cmd: list[str]
    cwd: Path
    out: Path  # the run directory <out>/<cohort>/<arm>/seed<seed>: done when metrics.json is there
    log: Path
    env: dict[str, str] = field(default_factory=dict)  # added to os.environ

    @property
    def done(self) -> bool:
        return (self.out / "metrics.json").is_file()


def resolve_cohorts(only: str, dump: Path) -> list[str]:
    """``--only`` if given; else the cohorts of the dump directory in :data:`ORDER` (extra ones last); else :data:`ORDER`."""
    given = [c for c in only.split(",") if c]
    if given:
        return given
    if Path(dump).is_dir():
        from model_stgcn.ptr.cohorts import discover_cohorts

        have = set(discover_cohorts(dump))
        if have:
            return [c for c in ORDER if c in have] + sorted(have - set(ORDER))
    return list(ORDER)


def build_job(stage_run: tuple[str, str, int], roots: Roots, *, device: str, threads: int, status_python: str | None = None) -> Job:
    """The child process of one ``(cohort, encoder-or-variant, seed)`` run of the stages main / seeds / ourfeat / hgbkf / k3."""
    cohort, enc, seed = stage_run
    arm = ARM[enc]
    py = status_python or sys.executable
    run_dir = roots.out / cohort / arm / f"seed{seed}"
    log = roots.out / "logs" / f"{cohort}__{arm}__seed{seed}.log"
    common = ["--cohort", cohort, "--seed", str(seed), "--out-dir", str(roots.out), "--data-dir", str(roots.data)]
    env: dict[str, str] = {}
    if enc == "hgbkf":
        cmd = [py, "-m", "adaptation.hgb_kraftonfeat", *common, "--table-dir", str(roots.table), "--dump-dir", str(roots.dump), "--threads", "1"]
        env["OMP_NUM_THREADS"] = "1"  # resource cap: the boosting runs next to a training process
    elif enc in ("bce", "hazard"):
        cmd = [py, "-m", "adaptation.ourfeat_hazard", *common, "--variant", enc, "--dump-dir", str(roots.dump), "--device", device, "--threads", str(min(threads, 3))]
    elif enc == "ourfeat":
        cmd = [py, "-m", "adaptation.ourfeat", *common, "--dump-dir", str(roots.dump), "--device", device, "--threads", str(threads)]
    else:  # K1: the transformer / mlp on the extracted shards (no dump: its tokens are the shards)
        cmd = [py, "-m", "adaptation.train", *common, "--encoder", enc, "--device", device, "--threads", str(threads)]
    return Job(f"{cohort}/{arm}/seed{seed}", cohort, arm, seed, cmd, ROOT, run_dir, log, env)


def stage_runs(stage: str, cohorts: list[str], encoders: list[str]) -> list[tuple[str, str, int]]:
    if stage == "seeds":
        return [(c, e, s) for c in ("erangel_tpp_squad", "taego_tpp_squad") for s in (13, 2026) for e in encoders if c in cohorts]
    if stage == "ourfeat":
        return [(c, "ourfeat", 42) for c in cohorts]
    if stage == "hgbkf":
        return [(c, "hgbkf", 42) for c in cohorts]
    if stage == "k3":
        return [(c, v, 42) for c in cohorts for v in ("bce", "hazard")]
    return [(c, e, 42) for c in cohorts for e in encoders]


def union_command(roots: Roots, *, arms: tuple[str, ...], workers: int, threads: int, cohorts: str = "", python: str | None = None) -> list[str]:
    """``run_matrix --union`` (HGB-union, optionally M11R-union) on the kf table: CPU workers, resumable by the config hash of every run."""
    cmd = [python or sys.executable, "-m", "model_stgcn.ptr.run_matrix", "--union", "--dump-dir", str(roots.dump), "--out-dir", str(roots.out),
           "--extra-features", str(roots.table), "--status-name", "run_matrix_status_union.csv", "--workers", str(workers), "--threads", str(threads)]  # fmt: skip
    cmd += ["--only", "arm=" + ",".join(arms)]
    if cohorts:
        cmd += ["--only", "cohort=" + cohorts]
    return cmd


def h0_command(roots: Roots, *, device: str, threads: int, cohorts: str = "", redo: bool = False, python: str | None = None) -> list[str]:
    """``adaptation.h0_baseline`` with the three roots it needs (run from ``ROOT``)."""
    cmd = [python or sys.executable, "-m", "adaptation.h0_baseline", "--ckpt-dir", str(roots.out), "--dump-dir", str(roots.dump), "--data-dir", str(roots.data),
           "--device", device, "--threads", str(min(threads, 3))]  # fmt: skip
    if cohorts:
        cmd += ["--cohorts", cohorts]
    if redo:
        cmd.append("--redo")
    return cmd


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=list(STAGES), default="main")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="run directory (default: the v2 root)")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="KRAFTON labels / shards / logs (default: the v2 root)")
    ap.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP, help="phase dump (default: the v2 dump)")
    ap.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE, help="KRAFTON feature table (default: the v2 root)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default="", help="comma separated cohorts")
    ap.add_argument("--encoders", default="transformer,mlp")
    ap.add_argument("--status-name", default="krafton_run_status.csv", help="status CSV inside --out-dir (one name per concurrent stream)")
    ap.add_argument("--cpu-workers", type=int, default=6, help="stage union: parallel runs of run_matrix")
    ap.add_argument("--cpu-threads", type=int, default=2, help="stage union: threads per run")
    ap.add_argument("--union-arms", default="HGB-union", help="stage union: arms to run (HGB-union is the one the decomposition needs; M11R-union is optional)")
    ap.add_argument("--redo", action="store_true", help="stage h0: refit arms that exist")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, write nothing, start nothing")
    return ap.parse_args(argv)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command_stage(cmd: list[str], cwd: Path, log: Path, env: dict[str, str] | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as lf:
        return subprocess.run(cmd, cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}).returncode


def main(argv=None) -> int:
    a = parse_args(argv)
    roots = Roots(a.out_dir, a.data_dir, a.dump_dir, a.table_dir)
    roots.check_versions()
    if a.stage in ("union", "h0"):
        arms = tuple(x for x in a.union_arms.split(",") if x)
        bad = [x for x in arms if x not in UNION_ARMS]
        if a.stage == "union" and bad:
            raise SystemExit(f"--union-arms: unknown {bad}; known {UNION_ARMS}")
        cohorts = ",".join(c for c in a.only.split(",") if c)
        if a.stage == "union":
            cmd, cwd, name = union_command(roots, arms=arms, workers=a.cpu_workers, threads=a.cpu_threads, cohorts=cohorts), REPO, "run_all__union"
        else:
            cmd, cwd, name = h0_command(roots, device="cpu" if a.device == "mps" else a.device, threads=a.threads, cohorts=cohorts, redo=a.redo), ROOT, "run_all__h0"
        log = roots.out / "logs" / f"{name}.log"
        print(f"[{_utc()}] {a.stage}: {' '.join(cmd)}", flush=True)
        if a.dry_run:
            return 0
        P.claim_root(roots.out, roots.dump, role="runs")
        rc = run_command_stage(cmd, cwd, log, {"PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH", "")]))})
        print(f"  -> {'done' if rc == 0 else 'failed'} (exit {rc}); log {log}", flush=True)
        return 1 if rc else 0

    cohorts = resolve_cohorts(a.only, roots.dump)
    jobs = [build_job(r, roots, device=a.device, threads=a.threads) for r in stage_runs(a.stage, cohorts, [e for e in a.encoders.split(",") if e])]
    if a.dry_run:
        for j in jobs:
            print(f"{'skip (done)' if j.done else 'run'} {j.run_id}: {' '.join(j.cmd)}", flush=True)
        return 0
    P.claim_root(roots.out, roots.dump, role="runs")
    status = roots.out / a.status_name
    (roots.out / "logs").mkdir(parents=True, exist_ok=True)
    new = not status.exists()
    failed = 0
    for job in jobs:
        if job.done:
            print(f"skip {job.run_id} (done)", flush=True)
            continue
        t0 = time.time()
        start = _utc()
        print(f"[{start}] start {job.run_id}", flush=True)
        env = dict(os.environ)
        env.update(job.env)
        with open(job.log, "w") as lf:
            rc = subprocess.run(job.cmd, cwd=str(job.cwd), stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
        ok = rc == 0 and job.done
        failed += 0 if ok else 1
        row = {"run_id": job.run_id, "cohort": job.cohort, "arm": job.arm, "seed": job.seed, "status": "done" if ok else "failed", "exit_code": rc,
               "start_utc": start, "end_utc": _utc(), "elapsed_sec": round(time.time() - t0, 1),
               "log": _rel(job.log), "metrics": _rel(job.out / "metrics.json") if ok else "", "note": ""}  # fmt: skip
        with open(status, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
                new = False
            w.writerow(row)
        print(f"  -> {row['status']} in {row['elapsed_sec']}s", flush=True)
    return 1 if failed else 0


def _rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(REPO))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
