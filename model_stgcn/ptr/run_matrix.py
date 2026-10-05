"""Run the whole PTR experiment matrix on parallel CPU workers; resumable, one log per run, a status CSV.

    python -m model_stgcn.ptr.run_matrix --dump-dir model_stgcn/dumps/bigcomp2027_phase_v1 \\
        --out-dir model_stgcn/checkpoints/bigcomp2027_ptr_v1 [--workers 7] [--dry-run]

Matrix (design doc sections 4 / 4.1), cohorts = the complete cohort directories of the dump directory
(:func:`model_stgcn.ptr.cohorts.discover_cohorts`), ``C`` of them, ``L`` of which are the two large cohorts:

========== ============================================================== ============================
group      runs                                                           count
========== ============================================================== ============================
main       the 9 ``MAIN_ARMS`` x every cohort, seed 42                    9 C
exploratory M11R-noProx / -noCombat / -noCopress on the large cohorts, 42 3 L
seeds      M00, M11, M11R, M11R-shuf, M11RZ on the large cohorts, 13, 2026 10 L
hgb        HGB-cut / HGB-lag / HGB-noHab x every cohort, seed 42          3 C
========== ============================================================== ============================

i.e. ``12 C + 13 L`` runs (170 for the 12 expected cohorts).  The large cohorts are ``erangel_tpp_squad`` and
``taego_tpp_squad`` (``--large-cohorts``); one that is not in the dump directory is skipped with a note
(``--strict`` makes that an error).

* Every run is one subprocess (``python -m model_stgcn.ptr.train|hgb ...``) with its stdout / stderr in
  ``<out-dir>/logs/<cohort>__<arm>__seed<seed>.log``; ``--workers`` of them run at once (default: cores / 2) with
  ``--threads`` threads each (default 2, 1 if cores are short; set both as arguments and as OMP / MKL / OpenBLAS
  environment variables).  The heaviest runs (estimated) start first.
* Resumable: a run is skipped when its ``metrics.json`` exists and carries the config hash that the same command
  would stamp now (``train.config_hash`` / ``hgb.config_hash``: arguments that change a result, the dump files'
  size and mtime, the source of the result-affecting modules).  Unreadable / stale / missing ``metrics.json`` means
  "run again"; ``--force`` re-runs everything.  A finished run never needs the runner again.
* ``<out-dir>/run_matrix_status.csv`` lists every run of the plan with status ``planned | skipped_done | running |
  done | failed | interrupted`` and is rewritten atomically after every change.  A run fails if its process exits
  non-zero or exits cleanly without a matching ``metrics.json``; the others continue and the exit code is 1.
* ``--dry-run`` prints the plan (counts per group, how many are done, the estimated wall-clock) and touches nothing
  (``--list`` adds every run); ``--only key=value[,value]`` (repeatable; keys cohort, arm, seed, kind, group; shell
  patterns allowed, e.g. ``arm=HGB-*``) restricts the plan, different keys are ANDed, repeated keys are ORed.
  ``--projected`` prints the plan of the cohorts kept in ``<dump-dir>/population.json`` (fallback: the built-in 12-cohort
  registry); no tensors are read.

The wall-clock model (:func:`estimate_seconds`, constants in ``CALIBRATION``) is an upper bound: it assumes that every
run uses all ``--max-epochs`` epochs although early stopping (patience 3) usually ends sooner.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from model_stgcn.ptr import hgb as H
from model_stgcn.ptr import train as T
from model_stgcn.ptr.arms import EXPLORATORY_ARMS, MAIN_ARMS
from model_stgcn.ptr.cohorts import discover_cohorts
from model_stgcn.ptr.data import DEFAULT_LABELS, DEFAULT_MATCH_TYPES, LABEL_CHOICES
from model_stgcn.ptr.hgb import HGB_VARIANTS
from model_stgcn.ptr.util import REPO_ROOT

logger = logging.getLogger("model_stgcn.ptr.run_matrix")

MAIN_SEED = 42
EXTRA_SEEDS: tuple[int, ...] = (13, 2026)
SEED_ARMS: tuple[str, ...] = ("M00", "M11", "M11R", "M11R-shuf", "M11RZ")
LARGE_COHORTS: tuple[str, ...] = ("erangel_tpp_squad", "taego_tpp_squad")
GROUPS: tuple[str, ...] = ("main", "exploratory", "seeds", "hgb")
ONLY_KEYS: tuple[str, ...] = ("cohort", "arm", "seed", "kind", "group")
STATUS_FIELDS = (
    "run_id", "group", "kind", "cohort", "arm", "seed", "labels", "config_hash", "status", "exit_code",
    "start_utc", "end_utc", "elapsed_sec", "log", "metrics", "note",
)  # fmt: skip

# --------------------------------------------------------------------------- sizes used for the wall-clock model
# mean teams per match and per-split matches of the cohorts, estimated for the registry that remains after the
# design v2.3 population rules (reports/bigcomp2027_data_profile/csv/cohort_summary.csv and cohort_match_type.csv:
# airoyale matches and Vikendi removed, the stored 80 / 10 / 10 split kept).  Only used for ``--projected``.
_REGISTRY_SIZES = {  # cohort: (matches after the exclusions, mean teams per match)
    "erangel_tpp_squad": (3283, 26.4),
    "taego_tpp_squad": (4159, 26.8),
    "miramar_desert_tpp_squad": (1543, 24.0),
    "rondo_tpp_squad": (1039, 22.8),
    "sanhok_tpp_squad": (940, 28.0),
    "erangel_tpp_solo": (224, 99.4),
    "erangel_tpp_duo": (656, 49.8),
    "taego_tpp_duo": (573, 50.0),
    "taego_fpp_squad": (435, 29.7),
    "erangel_fpp_squad": (430, 26.8),
    "karakin_tpp_squad": (333, 17.7),
    "paramo_tpp_squad": (333, 17.7),
}  # fmt: skip


@dataclass(frozen=True)
class CohortStats:
    n_train: int
    n_val: int
    n_test: int
    mean_teams: float  # teams per match
    rows_per_match: float  # risk-set rows per match (hazard rows)


def _projected_stats() -> dict[str, CohortStats]:
    out = {}
    for name, (n, teams) in _REGISTRY_SIZES.items():
        n_val = n_test = round(n * 0.1)
        out[name] = CohortStats(n - n_val - n_test, n_val, n_test, teams, 2.2 * teams)
    return out


PROJECTED_REGISTRY: dict[str, CohortStats] = _projected_stats()  # fallback when the dump dir has no population.json
DEFAULT_TEAMS = 26.0  # mean teams per match for a kept cohort that is not in ``_REGISTRY_SIZES``


def population_stats(dump_dir: str | Path) -> dict[str, CohortStats] | None:
    """Kept cohorts and split sizes from ``<dump_dir>/population.json`` (None if the file is absent).

    Matches per split are the ``after`` counts of the population report; the mean team count (not in that file)
    comes from ``_REGISTRY_SIZES`` when the cohort is listed there, else ``DEFAULT_TEAMS``."""
    path = Path(dump_dir) / "population.json"
    if not path.is_file():
        return None
    out = {}
    for name, info in json.loads(path.read_text(encoding="utf-8")).get("cohorts", {}).items():
        if info.get("status") != "kept":
            continue
        after = info["after"]
        teams = _REGISTRY_SIZES.get(name, (0, DEFAULT_TEAMS))[1]
        out[name] = CohortStats(int(after["train"]), int(after["val"]), int(after["test"]), teams, 2.2 * teams)
    return out



# --------------------------------------------------------------------------- wall-clock model
# Measured on this machine (Apple M5 Pro, 15 cores) with the runner itself: 7 concurrent workers x 2 threads,
# real dumps; see the report of the work package.  Epoch seconds include the validation pass.
CALIBRATION = {
    "startup_s": 5.0,  # interpreter + imports + dump loading + standardiser, per run
    # ms per training match and epoch = a + b * mean_teams, per arm class
    "epoch_ms": {"base": (1.9, 0.095), "rel": (2.4, 0.135), "relzone": (2.7, 0.165)},
    "predict_epochs": 0.5,  # final val / test predictions, in epochs of the same arm
    # HGB fit seconds = us_per_cell * rows * columns / 1e6
    "hgb_us_per_cell": 1.4,
}
ARM_CLASS = {
    **{a: "base" for a in ("M00", "M10", "M01", "M11")},
    **{a: "rel" for a in ("M11R", "M11R-shuf", "M11R-noHab", "M11R-noProx", "M11R-noCombat", "M11R-noCopress")},
    **{a: "relzone" for a in ("M11RZ", "M11RZ-noHab")},
}


@dataclass(frozen=True)
class RunSpec:
    kind: str  # 'neural' | 'hgb'
    cohort: str
    arm: str  # arm name, or the HGB variant
    seed: int
    group: str  # main | exploratory | seeds | hgb

    @property
    def run_id(self) -> str:
        return f"{self.cohort}/{self.arm}/seed{self.seed}"

    def run_dir(self, out_dir: str | Path) -> Path:
        return Path(out_dir) / self.cohort / self.arm / f"seed{self.seed}"

    def log_path(self, out_dir: str | Path) -> Path:
        return Path(out_dir) / "logs" / f"{self.cohort}__{self.arm}__seed{self.seed}.log"


@dataclass
class Settings:
    dump_dir: str
    out_dir: str
    labels: str = DEFAULT_LABELS
    match_types: tuple[str, ...] | None = DEFAULT_MATCH_TYPES
    max_epochs: int = 12
    device: str = "cpu"
    threads: int = 2
    python: str = field(default_factory=lambda: sys.executable)


# --------------------------------------------------------------------------- enumeration and filters
def enumerate_runs(
    cohorts: Iterable[str], large: Sequence[str] = LARGE_COHORTS, *, strict: bool = False
) -> tuple[list[RunSpec], list[str]]:
    """The run matrix for the given cohorts and notes about what had to be skipped."""
    cohorts = sorted(set(cohorts))
    present = [c for c in large if c in cohorts]
    missing = [c for c in large if c not in cohorts]
    notes: list[str] = []
    if missing:
        msg = f"large cohort(s) {missing} not in the dump directory: no exploratory / extra-seed runs for them"
        if strict:
            raise ValueError(msg)
        notes.append(msg)
    runs = [RunSpec("neural", c, a, MAIN_SEED, "main") for c in cohorts for a in MAIN_ARMS]
    runs += [RunSpec("neural", c, a, MAIN_SEED, "exploratory") for c in present for a in EXPLORATORY_ARMS]
    runs += [RunSpec("neural", c, a, s, "seeds") for c in present for s in EXTRA_SEEDS for a in SEED_ARMS]
    runs += [RunSpec("hgb", c, v, MAIN_SEED, "hgb") for c in cohorts for v in HGB_VARIANTS]
    return runs, notes


def parse_only(entries: Iterable[str]) -> dict[str, list[str]]:
    """``['arm=M00,M11', 'seed=42']`` -> ``{'arm': ['M00', 'M11'], 'seed': ['42']}`` (repeated keys are merged)."""
    out: dict[str, list[str]] = {}
    for entry in entries:
        key, sep, value = entry.partition("=")
        if not sep or key not in ONLY_KEYS or not value:
            raise ValueError(f"--only expects key=value[,value] with key in {ONLY_KEYS}, got {entry!r}")
        values = [v for v in value.split(",") if v]
        if key == "seed":
            for v in values:
                int(v)  # ValueError for anything that is not an integer
        out.setdefault(key, []).extend(values)
    return out


def apply_only(runs: Sequence[RunSpec], only: dict[str, list[str]]) -> list[RunSpec]:
    def keep(run: RunSpec) -> bool:
        return all(
            any(fnmatch.fnmatchcase(str(getattr(run, key)), pattern) for pattern in patterns)
            for key, patterns in only.items()
        )

    return [r for r in runs if keep(r)]


# --------------------------------------------------------------------------- command lines and hashes
def build_argv(spec: RunSpec, st: Settings) -> list[str]:
    module = "model_stgcn.ptr.train" if spec.kind == "neural" else "model_stgcn.ptr.hgb"
    argv = [
        st.python, "-m", module,
        "--dump-dir", str(st.dump_dir), "--cohort", spec.cohort,
        "--arm" if spec.kind == "neural" else "--variant", spec.arm,
        "--seed", str(spec.seed), "--out-dir", str(st.out_dir), "--labels", st.labels,
        "--match-types", "any" if st.match_types is None else ",".join(st.match_types),
        "--threads", str(st.threads),
    ]  # fmt: skip
    if spec.kind == "neural":
        argv += ["--device", st.device, "--max-epochs", str(st.max_epochs)]
    return argv


def expected_hash(spec: RunSpec, st: Settings) -> str:
    """The config hash the run's own CLI will stamp (computed with the CLI's own parser and hash function)."""
    argv = build_argv(spec, st)[3:]
    if spec.kind == "neural":
        return T.config_hash(T.parse_args(argv))
    return H.config_hash(H.parse_args(argv))


def is_done(spec: RunSpec, st: Settings, expected: str | None = None) -> bool:
    """A finished run: ``metrics.json`` parses and carries the hash the same command would stamp now."""
    path = spec.run_dir(st.out_dir) / "metrics.json"
    try:
        stamped = json.loads(path.read_text(encoding="utf-8")).get("config_hash")
    except (OSError, ValueError):
        return False
    return stamped is not None and stamped == (expected or expected_hash(spec, st))


# --------------------------------------------------------------------------- cost model
def cohort_stats(dump_dir: str | Path, cohort: str, labels: str = DEFAULT_LABELS) -> CohortStats:
    """Split sizes from the dump manifest and the team / row counts of the validation split (small to read)."""
    base = Path(dump_dir) / cohort
    sizes = {}
    try:
        manifest = json.loads((base / "dump_manifest.json").read_text(encoding="utf-8"))
        sizes = {s: int(manifest["splits"][s]["n_dumped"]) for s in ("train", "val", "test")}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    mean_teams, rows_per_match = _guess_teams(cohort), 2.2 * _guess_teams(cohort)
    try:
        import torch

        val = torch.load(base / "val.pt", map_location="cpu", weights_only=False)
        matches = val["matches"]
        mean_teams = sum(len(m["team_ids"]) for m in matches) / len(matches)
        key = "target_mask_v2" if labels == "v2" and "target_mask_v2" in matches[0] else "target_mask"
        rows_per_match = sum(int(m[key].sum()) for m in matches) / len(matches)
        sizes.setdefault("val", len(matches))
        sizes.setdefault("train", 8 * len(matches))
        sizes.setdefault("test", len(matches))
    except Exception:  # unreadable validation file: the name-based guess is good enough for an estimate
        sizes = sizes or {"train": 300, "val": 40, "test": 40}
    return CohortStats(sizes["train"], sizes["val"], sizes["test"], mean_teams, rows_per_match)


def _guess_teams(cohort: str) -> float:
    return 99.0 if "solo" in cohort else 50.0 if "duo" in cohort else 27.0


def estimate_seconds(spec: RunSpec, stats: CohortStats, st: Settings) -> float:
    """Upper-bound seconds of one run: every epoch used (neural), measured per-cell cost (HGB)."""
    c = CALIBRATION
    if spec.kind == "neural":
        a, b = c["epoch_ms"][ARM_CLASS.get(spec.arm, "relzone")]
        epoch = stats.n_train * (a + b * stats.mean_teams) / 1000.0
        return c["startup_s"] + epoch * (st.max_epochs + c["predict_epochs"])
    cols = 3 * 150 + 1 if spec.arm == "HGB-lag" else 150 + 1 - (44 if spec.arm == "HGB-noHab" else 0)
    return c["startup_s"] + c["hgb_us_per_cell"] * stats.n_train * stats.rows_per_match * cols / 1e6


def makespan(costs: Sequence[float], workers: int) -> float:
    """Wall-clock of list scheduling (longest first onto the earliest free worker) of the given run times."""
    free = [0.0] * max(1, workers)
    for cost in sorted(costs, reverse=True):
        i = min(range(len(free)), key=free.__getitem__)
        free[i] += cost
    return max(free) if costs else 0.0


# --------------------------------------------------------------------------- plan
@dataclass
class PlanItem:
    spec: RunSpec
    expected_hash: str
    done: bool
    est_seconds: float


@dataclass
class Plan:
    items: list[PlanItem]
    notes: list[str]
    cohorts: list[str]
    projected: bool = False

    @property
    def todo(self) -> list[PlanItem]:
        return [i for i in self.items if not i.done]


def make_plan(
    runs: Sequence[RunSpec],
    st: Settings,
    *,
    stats: dict[str, CohortStats],
    force: bool = False,
    check_done: bool = True,
    cohorts: Sequence[str] = (),
    notes: Sequence[str] = (),
    projected: bool = False,
) -> Plan:
    items = []
    for spec in runs:
        h = expected_hash(spec, st) if check_done or not projected else ""
        done = bool(check_done and not force and is_done(spec, st, h))
        est = estimate_seconds(spec, stats[spec.cohort], st) if spec.cohort in stats else 0.0
        items.append(PlanItem(spec, h, done, est))
    return Plan(items, list(notes), list(cohorts), projected)


def format_plan(plan: Plan, st: Settings, workers: int, *, list_runs: bool = False) -> str:
    counts = {g: sum(1 for i in plan.items if i.spec.group == g) for g in GROUPS}
    todo = plan.todo
    n_done = len(plan.items) - len(todo)
    lines = [
        f"{'projected registry (no dump read)' if plan.projected else 'dump: ' + str(st.dump_dir)}"
        f" | labels {st.labels} | match types {'any' if st.match_types is None else ','.join(st.match_types)}"
        f" | max epochs {st.max_epochs} | device {st.device}",
        f"cohorts ({len(plan.cohorts)}): {', '.join(plan.cohorts) if plan.cohorts else '-'}",
    ]
    lines += [f"note: {n}" for n in plan.notes]
    lines.append(
        f"runs: {len(plan.items)} = main {counts['main']} + exploratory {counts['exploratory']}"
        f" + seeds {counts['seeds']} + hgb {counts['hgb']}"
    )
    lines.append(f"to run: {len(todo)} (already done with the same config hash: {n_done})")
    est = [i.est_seconds for i in todo]
    hours = makespan(est, workers) / 3600.0
    when = f"~{hours:.1f} h" if hours >= 0.1 else f"~{hours * 60:.0f} min"
    lines.append(f"workers: {workers} x {st.threads} threads")
    lines.append(
        f"estimated wall-clock with {workers} workers: {when} (upper bound: all {st.max_epochs} epochs used; "
        f"summed run time {sum(est) / 3600:.1f} h)"
    )
    if list_runs:
        lines.append("run id | group | est. min | state")
        for i in sorted(plan.items, key=lambda i: -i.est_seconds):
            lines.append(f"{i.spec.run_id} | {i.spec.group} | {i.est_seconds / 60:.1f} | {'done' if i.done else 'todo'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- execution
def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def default_threads(workers: int) -> int:
    return max(1, min(2, (os.cpu_count() or 2) // max(1, workers)))


def child_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")]))
    return env


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StatusTable:
    """The status CSV: one row per planned run, rewritten atomically after every change."""

    def __init__(self, path: Path, items: Sequence[PlanItem], st: Settings) -> None:
        self.path = path
        self.rows: dict[str, dict[str, str]] = {}
        for item in items:
            s = item.spec
            self.rows[s.run_id] = {
                "run_id": s.run_id, "group": s.group, "kind": s.kind, "cohort": s.cohort, "arm": s.arm,
                "seed": str(s.seed), "labels": st.labels, "config_hash": item.expected_hash,
                "status": "skipped_done" if item.done else "planned", "exit_code": "", "start_utc": "",
                "end_utc": "", "elapsed_sec": "", "log": str(s.log_path(st.out_dir)),
                "metrics": str(s.run_dir(st.out_dir) / "metrics.json"), "note": "",
            }  # fmt: skip

    def update(self, run_id: str, **fields) -> None:
        self.rows[run_id].update({k: str(v) for k, v in fields.items()})
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows.values())
        os.replace(tmp, self.path)


def execute(
    plan: Plan,
    st: Settings,
    workers: int,
    *,
    popen: Callable = subprocess.Popen,
    poll_seconds: float = 1.0,
    status_path: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Run the runs of the plan that are not done, at most ``workers`` at once; returns the status rows."""
    out = Path(st.out_dir)
    table = StatusTable(status_path or out / "run_matrix_status.csv", plan.items, st)
    table.write()
    queue = [i for i in sorted(plan.todo, key=lambda i: -i.est_seconds)]
    running: dict[str, tuple[object, PlanItem, float, object]] = {}
    env = child_env(st.threads)
    logger.info("%d runs to start, %d workers x %d threads", len(queue), workers, st.threads)

    def finish(run_id: str, proc, item: PlanItem, started: float, logfile, interrupted: bool = False) -> None:
        logfile.close()
        code = proc.returncode
        elapsed = time.time() - started
        if interrupted:
            table.update(run_id, status="interrupted", exit_code=code, end_utc=_utc(), elapsed_sec=f"{elapsed:.1f}")
        elif code != 0:
            table.update(run_id, status="failed", exit_code=code, end_utc=_utc(), elapsed_sec=f"{elapsed:.1f}",
                         note=f"process exited with code {code}; see the log")  # fmt: skip
        elif not is_done(item.spec, st, item.expected_hash):
            table.update(run_id, status="failed", exit_code=code, end_utc=_utc(), elapsed_sec=f"{elapsed:.1f}",
                         note="exit code 0 but no metrics.json with the expected config hash")  # fmt: skip
        else:
            table.update(run_id, status="done", exit_code=code, end_utc=_utc(), elapsed_sec=f"{elapsed:.1f}")
        logger.info("%-55s %s (%.1fs)", run_id, table.rows[run_id]["status"], elapsed)

    try:
        while queue or running:
            for run_id in [r for r, (p, *_rest) in running.items() if p.poll() is not None]:
                proc, item, started, logfile = running.pop(run_id)
                finish(run_id, proc, item, started, logfile)
            while queue and len(running) < workers:
                item = queue.pop(0)
                spec = item.spec
                log_path = spec.log_path(st.out_dir)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                logfile = open(log_path, "w", encoding="utf-8")
                proc = popen(build_argv(spec, st), stdout=logfile, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))
                running[spec.run_id] = (proc, item, time.time(), logfile)
                table.update(spec.run_id, status="running", start_utc=_utc())
            if queue or running:
                time.sleep(poll_seconds)
    except KeyboardInterrupt:
        for run_id, (proc, item, started, logfile) in list(running.items()):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:  # pragma: no cover - a child that ignores SIGTERM
                pass
            finish(run_id, proc, item, started, logfile, interrupted=True)
        raise
    return table.rows


# --------------------------------------------------------------------------- command line
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m model_stgcn.ptr.run_matrix", description=__doc__.split("\n")[0])
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--labels", default=DEFAULT_LABELS, choices=LABEL_CHOICES)
    ap.add_argument("--match-types", default=",".join(DEFAULT_MATCH_TYPES), type=T._parse_match_types,
                    help="allowed match_type values, comma separated ('any' = no check)")  # fmt: skip
    ap.add_argument("--max-epochs", type=int, default=12)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--workers", type=T._positive_int, default=None, help="parallel runs (default: cores / 2)")
    ap.add_argument("--threads", type=T._positive_int, default=None,
                    help="threads per worker (default: 2, or fewer if cores are short)")  # fmt: skip
    ap.add_argument("--large-cohorts", default=",".join(LARGE_COHORTS),
                    help="cohorts that also get the exploratory arms and the extra seeds")  # fmt: skip
    ap.add_argument("--only", action="append", default=[], metavar="KEY=VALUE[,VALUE]",
                    help=f"restrict the plan; keys {', '.join(ONLY_KEYS)}; patterns allowed; repeatable")  # fmt: skip
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; nothing is written")
    ap.add_argument("--list", action="store_true", help="with --dry-run: list every run")
    ap.add_argument("--projected", action="store_true",
                    help="with --dry-run: plan the expected 12-cohort registry from the data profile sizes")  # fmt: skip
    ap.add_argument("--force", action="store_true", help="re-run runs that are already done")
    ap.add_argument("--strict", action="store_true", help="fail if a large cohort is missing from the dump dir")
    ap.add_argument("--poll-seconds", type=float, default=1.0)
    ap.add_argument("--python", default=sys.executable)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None, popen: Callable = subprocess.Popen) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.projected and not args.dry_run:
        raise SystemExit("--projected only makes sense with --dry-run (there are no dumps to run)")
    workers = args.workers or default_workers()
    threads = args.threads or default_threads(workers)
    st = Settings(
        dump_dir=args.dump_dir, out_dir=args.out_dir, labels=args.labels, match_types=args.match_types,
        max_epochs=args.max_epochs, device=args.device, threads=threads, python=args.python,
    )  # fmt: skip
    large = [c for c in args.large_cohorts.split(",") if c]
    if args.projected:
        stats = population_stats(args.dump_dir) or dict(PROJECTED_REGISTRY)
        cohorts = sorted(stats)
    else:
        cohorts = discover_cohorts(args.dump_dir)
        stats = {c: cohort_stats(args.dump_dir, c, args.labels) for c in cohorts} if args.dry_run or cohorts else {}
    runs, notes = enumerate_runs(cohorts, large, strict=args.strict)
    if not args.projected:
        kept = population_stats(args.dump_dir)
        if kept is not None and set(kept) - set(cohorts):
            notes.append(f"population.json keeps {len(kept)} cohorts; not in the dump dir yet: {sorted(set(kept) - set(cohorts))}")
    runs = apply_only(runs, parse_only(args.only))
    plan = make_plan(runs, st, stats=stats, force=args.force, cohorts=cohorts, notes=notes,
                     projected=args.projected, check_done=not args.projected)  # fmt: skip
    print(format_plan(plan, st, workers, list_runs=args.list))
    if args.dry_run:
        return 0
    rows = execute(plan, st, workers, popen=popen, poll_seconds=args.poll_seconds)
    failed = [r for r in rows.values() if r["status"] in ("failed", "interrupted")]
    n_done = sum(1 for r in rows.values() if r["status"] == "done")
    print(f"finished: {n_done} done, {len(rows) - n_done - len(failed)} skipped, {len(failed)} failed "
          f"(status: {Path(st.out_dir) / 'run_matrix_status.csv'})")  # fmt: skip
    for r in failed:
        print(f"  failed: {r['run_id']} -> {r['log']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
