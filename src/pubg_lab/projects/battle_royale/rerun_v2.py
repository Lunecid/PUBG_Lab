#!/usr/bin/env python3
"""Orchestrate the whole BigComp 2027 re-run on the leak-free v2 dump: gate, CPU lane, MPS lane(s), aggregation, tables, paper numbers.

    python tools/rerun_v2.py --dry-run                      # print the plan (dump, roots, budget, lanes, steps, commands, estimates); writes nothing
    python tools/rerun_v2.py [--mps-streams 2] [--with-lc]  # run it; resumable: run it again after an interruption

Order (manifest ``docs/experiment_design/BIGCOMP2027_RERUN_MANIFEST_v2_2026-10-06.md`` D.1 / D.5)::

    1  gate        the v2 dump meta is valid (``paths.validate_v2_dump``), no v1 root is mixed in, the leak tests pass (``--leak-tests``)
    2  CPU lane    HGB family FIRST (the Sagong and KRAFTON stages assert their rows against HGB-cut's) -> Sagong (plan, extract, habcols, fit)
                   -> EHM / RQ1 (fit, rq1, rq1 physical) -> PTR main + seeds -> norank -> union
    3  MPS lane    the KRAFTON chain, after the HGB family: labels, extract, kf table, HGB-kraftonfeat, K2, K3a / K3, official arms (K1), H0 fit
                   (one stream: that order; two streams: K1 on the second one)
    4  aggregate   paired bootstrap, pooling, P1-P5 + Holm, the decomposition steps, RQ2
    5  supplement  within match-phase AUC, REML + HKSJ sensitivity (reads the aggregator's outputs)
    6  tables      tab_decomp (re-render), the KRAFTON tables (h0_report, k3_contrasts, k3_facts, phase_auc), make_paper_tables, EHM report
    7  numbers     tools/paper_numbers_v2.py -> <results>/paper_numbers.json

Budget: ``--cpu-workers`` (default 6) x ``--cpu-threads`` (2) for every ``run_matrix`` step, plus ``--mps-streams`` (default 1, at most 2) training
processes with ``--mps-threads`` (3) threads each; the lanes run concurrently (a step waits for the steps it needs on another lane).

Resumable: ``<results>/run_state.json`` records every step (status, times, exit code, the hash of its command); a step that is ``done`` with the same command
and its outputs in place is skipped, a failed / interrupted one runs again (every underlying tool skips what it has finished: runs with the same config hash,
shards, tables).  ``--restart`` forgets the state (not the outputs).  One log per step: ``<results>/logs/<step>.log``.

Provenance: ``<results>/run_manifest.json`` records git HEAD, the SHA-1 of ``git status --porcelain``, the SHA-1 of the result-affecting modules and the dump meta
hash.  A resume with another HEAD / result-module hash / dump meta hash is refused (finished runs would go stale or mix two builds) unless ``--allow-code-change``.

Not in the plan: stage 2 (the player layer, ``M11RZ-P`` / ``M11RZ-P-tm``): it needs the player dump, which is not rebuilt, and the paper does not use it.
``validate_plan`` refuses any command that names those arms, ``--stage2`` or a player dump.

Code snapshot: commits wait for the user's approval, so every launch saves the code that made the results into ``<results>/code_snapshot/``: ``git_diff_HEAD.patch``
(``git diff HEAD --binary``: every tracked change, binary-safe), ``untracked_files.tar`` (the untracked ``.py`` / ``.md`` / ``.json`` files under ``model_stgcn/``,
``stdb/``, ``third_party/krafton_pubg_winprob/adaptation/``, ``tools/``, ``tests/`` and ``paper/bigcomp2027/tables/``, where ``make_paper_tables.py`` lives),
``snapshot.json`` (HEAD, the sha256 of the patch, the tar and every file),
``SHA256SUMS`` and ``untracked_files.sha256``.  A later launch with other code adds ``code_snapshot/launch_<UTC>/`` and keeps the first one.  Restore:
``git checkout <HEAD> && git apply --binary code_snapshot/git_diff_HEAD.patch && tar -xf code_snapshot/untracked_files.tar``.  The dry run only previews it.

Roots: every path is an option, the defaults are the v2 roots of ``model_stgcn.ptr.paths``; a v1 root next to the v2 dump is refused.  The database stages need the
credentials in the environment (``set -a; source <pubg_collector>/.env; set +a``); they are never printed.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT

import argparse
import hashlib
import io
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

REPO = REPO_ROOT
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pubg_lab.survival.ptr import paths as P  # noqa: E402
from pubg_lab.survival.ptr import run_matrix as RM  # noqa: E402
from pubg_lab.survival.ptr.arms import STAGE2_ARMS  # noqa: E402
from pubg_lab.survival.ptr.util import code_fingerprint  # noqa: E402

KR = REPO / "third_party" / "krafton_pubg_winprob"
MAX_MPS_STREAMS = 2
SAGONG_PYTHON = "/usr/local/bin/python3"  # xgboost + pyarrow (the default interpreter has neither)
DEFAULT_LEAK_TESTS = (
    "tests/test_phase_v2_unit.py",
    "tests/test_phase_v2_dump_files.py",
    "tests/test_phase_v2_db.py",
    "tests/test_ptr_data.py",
    "tests/test_ptr_model.py",
)
VALIDATE_CAUSAL_COHORTS = (
    "karakin_tpp_squad",
    "erangel_tpp_squad",
    "taego_tpp_duo",
    "erangel_tpp_solo",
    "sanhok_tpp_squad",
)
LANES = ("main", "cpu", "mps0", "mps1")
PLAYER_LAYER_TOKENS = (
    "--stage2",
    "--player-dump-dir",
    *STAGE2_ARMS,
)  # stage 2 (the player layer) is not part of the v2 re-run: the player dump is not rebuilt
SNAPSHOT_DIR_NAME = "code_snapshot"
SNAPSHOT_ROOTS = (
    "model_stgcn",
    "stdb",
    "third_party/krafton_pubg_winprob/adaptation",
    "tools",
    "tests",
    "paper/bigcomp2027/tables",
)  # the last one is not in the coordinator's list: make_paper_tables.py (untracked) is run by the plan; its .tex outputs are not code
SNAPSHOT_EXTENSIONS = (".py", ".md", ".json")
SNAPSHOT_MAX_FILE_BYTES = (
    50 * 1024 * 1024
)  # a larger untracked file is listed as skipped, not captured (it is not code)
SNAPSHOT_MAX_FILES = 2000  # more than this means that something that is not code slipped past the ignore rules: refuse, do not archive it
SNAPSHOT_MAX_TOTAL_BYTES = 300 * 1024 * 1024
SNAPSHOT_DIFF_ARGS = (
    "diff",
    "HEAD",
    "--binary",
    "--full-index",
    "--no-ext-diff",
    "--no-color",
    "--no-renames",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)


# =========================================================================== configuration
@dataclass(frozen=True)
class Config:
    paths: P.RerunPaths = field(default_factory=P.RerunPaths)
    cpu_workers: int = 6
    cpu_threads: int = 2
    mps_streams: int = 1
    mps_threads: int = 3
    python: str = sys.executable
    python_sagong: str = SAGONG_PYTHON if Path(SAGONG_PYTHON).exists() else sys.executable
    n_boot: int = 2000
    with_lc: bool = False
    with_mlp: bool = False
    with_m11r_union: bool = False
    with_exploratory: bool = False
    validate_causal: bool = True
    leak_tests: tuple[str, ...] = DEFAULT_LEAK_TESTS
    skip_leak_tests: bool = False
    cohorts: tuple[str, ...] = ()  # filled from the dump (or the projected registry)

    def __post_init__(self) -> None:
        if not 1 <= self.mps_streams <= MAX_MPS_STREAMS:
            raise ValueError(f"--mps-streams must be 1 or {MAX_MPS_STREAMS}")
        if self.cpu_workers < 1 or self.cpu_threads < 1 or self.mps_threads < 1:
            raise ValueError("workers and threads must be positive")


def _order() -> list[str]:
    """The KRAFTON cohort order (small cohorts first, the two large ones last): ``adaptation.run_all.ORDER`` (imported without leaving the KRAFTON package on the path)."""
    sys.path.insert(0, str(KR))
    try:
        from adaptation.run_all import ORDER

        return list(ORDER)
    finally:
        sys.path.remove(str(KR))


def cohorts_of(paths: P.RerunPaths) -> tuple[str, ...]:
    order = _order()
    if paths.dump.is_dir():
        from pubg_lab.survival.ptr.cohorts import discover_cohorts

        have = set(discover_cohorts(paths.dump))
        if have:
            return tuple([c for c in order if c in have] + sorted(have - set(order)))
    return tuple(order)


# =========================================================================== the plan
@dataclass
class Step:
    id: str
    lane: str
    desc: str
    cmd: list[str] | None = None  # None: an in-process step (``fn``)
    cwd: Path = REPO
    env: dict[str, str] = field(default_factory=dict)
    deps: tuple[str, ...] = ()
    est_min: float = 1.0
    needs_db: bool = False
    outputs: tuple[Path, ...] = ()  # a step counts as done only if these exist
    fn: Callable[["Step", Path], int] | None = None
    optional: bool = False
    always: bool = False  # never skipped as 'done before': the gate runs at every launch

    def signature(self) -> str:
        body = json.dumps(
            {
                "cmd": self.cmd,
                "cwd": str(self.cwd),
                "env": self.env,
                "fn": self.fn.__name__ if self.fn else None,
            },
            sort_keys=True,
        )
        return hashlib.sha1(body.encode()).hexdigest()[:16]


def _rm(cfg: Config, *extra: str) -> list[str]:
    p = cfg.paths
    return [cfg.python, "-m", "model_stgcn.ptr.run_matrix", "--dump-dir", str(p.dump), "--out-dir", str(p.runs), "--workers", str(cfg.cpu_workers),
            "--threads", str(cfg.cpu_threads), *extra]  # fmt: skip


def _kr(cfg: Config, module: str, *args: str) -> list[str]:
    return [cfg.python, "-m", f"adaptation.{module}", *args]


def _run_all(cfg: Config, stage: str, stream: str, *extra: str) -> list[str]:
    p = cfg.paths
    return _kr(cfg, "run_all", "--stage", stage, "--out-dir", str(p.runs), "--data-dir", str(p.krafton_data), "--dump-dir", str(p.dump), "--table-dir", str(p.kf),
               "--status-name", f"krafton_run_status_{stream}.csv", *extra)  # fmt: skip


def build_plan(cfg: Config) -> list[Step]:
    """The whole DAG as a list of steps (a pure function of the configuration)."""
    p = cfg.paths
    cohorts = ",".join(cfg.cohorts or cohorts_of(p))
    py, pys = cfg.python, cfg.python_sagong
    sg = [pys, "-m", "model_stgcn.sagong2025"]
    sg_args = ["--dump-dir", str(p.dump), "--plan-dir", str(p.sagong), "--runs-dir", str(p.runs)]
    eco = [
        py,
        "-m",
        "model_stgcn.ptr.ecomodel",
        "--dump-dir",
        str(p.dump),
        "--runs-dir",
        str(p.runs),
        "--out-dir",
        str(p.ecological),
    ]
    eco_env = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    mps = {"--device": "mps", "--threads": str(cfg.mps_threads)}
    chain_args = ["--device", mps["--device"], "--threads", mps["--threads"]]
    k1_encoders = "transformer,mlp" if cfg.with_mlp else "transformer"
    s0, s1 = "mps0", ("mps1" if cfg.mps_streams > 1 else "mps0")
    main_groups = "group=main,seeds" + (",exploratory" if cfg.with_exploratory else "")
    union_arms = "HGB-union" + (",M11R-union" if cfg.with_m11r_union else "")
    steps: list[Step] = []

    def add(**kw) -> Step:
        s = Step(**kw)
        steps.append(s)
        return s

    # ---- 1 gate ----------------------------------------------------------------------------------------------------------
    add(
        id="gate.dump",
        lane="main",
        desc="the v2 dump meta is valid, no v1 root is mixed in",
        fn=gate_dump_fn(cfg),
        est_min=0.5,
        outputs=(),
        always=True,
    )
    leak = [py, "-m", "pytest", "-q", "-x", *cfg.leak_tests] if not cfg.skip_leak_tests else None
    if leak:
        add(
            id="gate.leak_tests",
            lane="main",
            desc="the leak tests pass (the dump builder's tests, the loader and model tests)",
            cmd=leak,
            deps=("gate.dump",),
            est_min=3.0,
            always=True,
        )
    gate = ("gate.leak_tests",) if leak else ("gate.dump",)

    # ---- 2 CPU lane --------------------------------------------------------------------------------------------------------
    add(
        id="cpu.hgb",
        lane="cpu",
        desc="HGB family (HGB-cut, HGB-lag, HGB-noHab) x every cohort, FIRST: the Sagong and KRAFTON stages assert their rows against HGB-cut's",
        cmd=_rm(cfg, "--only", "group=hgb", "--status-name", "run_matrix_status_hgb.csv"),
        deps=gate,
        est_min=3,
    )
    add(
        id="cpu.sagong_plan",
        lane="cpu",
        desc="Sagong re-implementation: plan (units, windows)",
        cmd=[*sg, "plan", *sg_args],
        deps=("cpu.hgb",),
        est_min=0.5,
    )
    add(
        id="cpu.sagong_extract",
        lane="cpu",
        desc="Sagong: extract the player features (database, 2 workers)",
        cmd=[*sg, "extract", *sg_args, "--workers", "2"],
        deps=("cpu.sagong_plan",),
        est_min=4.5,
        needs_db=True,
    )
    add(
        id="cpu.sagong_habcols",
        lane="cpu",
        desc="Sagong: the 44 zone / habitat columns (needs HGB-cut and HGB-noHab)",
        cmd=[*sg, "habcols", *sg_args],
        deps=("cpu.sagong_extract",),
        est_min=0.3,
    )
    add(
        id="cpu.sagong_fit",
        lane="cpu",
        desc="Sagong: SAGONG-xgb / rf / lr / hgb / hgb-hab",
        cmd=[*sg, "fit", *sg_args, "--models", "xgb,rf,lr,hgb,hgb-hab", "--n-jobs", "2"],
        deps=("cpu.sagong_habcols",),
        est_min=1.5,
    )
    add(
        id="cpu.ehm_fit",
        lane="cpu",
        desc="ecological hazard model: PHASE-only, EHM, EHM-int, EHM-lite arms",
        cmd=[*eco, "--stage", "fit"],
        env=eco_env,
        deps=("cpu.sagong_fit",),
        est_min=2,
    )
    add(
        id="cpu.ehm_rq1",
        lane="cpu",
        desc="RQ1 coefficients (SD units)",
        cmd=[*eco, "--stage", "rq1"],
        env=eco_env,
        deps=("cpu.ehm_fit",),
        est_min=3,
    )
    add(
        id="cpu.ehm_rq1_physical",
        lane="cpu",
        desc="RQ1 coefficients (physical units + AME)",
        cmd=[*eco, "--stage", "rq1", "--rq1-units", "physical"],
        env=eco_env,
        deps=("cpu.ehm_rq1",),
        est_min=3,
    )
    add(
        id="cpu.ptr_main",
        lane="cpu",
        desc="PTR main matrix (9 arms) + extra seeds (13 / 2026 on the two large cohorts); no stage 2 (player layer)",
        cmd=_rm(cfg, "--only", main_groups),
        deps=("cpu.ehm_rq1_physical",),
        est_min=50,
    )
    add(
        id="cpu.ptr_norank",
        lane="cpu",
        desc="M01-norank and M11R-norank (rank-loss control: BCE only)",
        cmd=_rm(cfg, "--norank", "--status-name", "run_matrix_status_norank.csv"),
        deps=("cpu.ptr_main",),
        est_min=6,
    )
    add(
        id="cpu.union",
        lane="cpu",
        desc="HGB-union (extension) via run_all --stage union on the kf table"
        + (" + M11R-union" if cfg.with_m11r_union else ""),
        cmd=_run_all(
            cfg,
            "union",
            "cpu",
            "--cpu-workers",
            str(cfg.cpu_workers),
            "--cpu-threads",
            str(cfg.cpu_threads),
            "--union-arms",
            union_arms,
        ),
        cwd=KR,
        deps=("cpu.ptr_norank", "mps.kftable"),
        est_min=3 + (9 if cfg.with_m11r_union else 0),
    )
    cpu_tail = "cpu.union"
    if cfg.with_lc:
        lc = [
            py,
            "-m",
            "model_stgcn.ptr.run_matrix",
            "--dump-dir",
            str(p.dump),
            "--workers",
            str(cfg.cpu_workers),
            "--threads",
            str(cfg.cpu_threads),
            "--lc",
        ]
        add(
            id="cpu.lc",
            lane="cpu",
            desc="learning curve, fixed epochs (optional)",
            cmd=[*lc, "--out-dir", str(p.runs_lc)],
            deps=(cpu_tail,),
            est_min=21,
        )
        add(
            id="cpu.lc_steps",
            lane="cpu",
            desc="learning curve, steps matched (optional)",
            cmd=[*lc, "--out-dir", str(p.runs_lc_steps), "--lc-steps-matched"],
            deps=("cpu.lc",),
            est_min=17,
        )
        cpu_tail = "cpu.lc_steps"

    # ---- 3 MPS lane(s): the KRAFTON chain ----------------------------------------------------------------------------------
    add(
        id="mps.labels",
        lane=s0,
        desc="KRAFTON label tables from the dump (after the HGB family: the KRAFTON stages assert their rows against HGB-cut's)",
        cmd=_kr(cfg, "labels", "--dump-dir", str(p.dump), "--out-dir", str(p.krafton_data)),
        cwd=KR,
        deps=(*gate, "cpu.hgb"),
        est_min=1,
    )
    add(
        id="mps.extract",
        lane=s0,
        desc="KRAFTON features at the cuts (database, 3 workers)",
        cmd=_kr(
            cfg,
            "extract",
            "--cohorts",
            cohorts,
            "--splits",
            "val,test,train",
            "--workers",
            "3",
            "--out-dir",
            str(p.krafton_data),
        ),
        cwd=KR,
        deps=("mps.labels",),
        est_min=7,
        needs_db=True,
    )
    last = "mps.extract"
    if cfg.validate_causal:
        vc = [c for c in VALIDATE_CAUSAL_COHORTS if c in (cfg.cohorts or cohorts_of(p))] or list(
            (cfg.cohorts or cohorts_of(p))[:5]
        )
        add(
            id="mps.validate_causal",
            lane=s0,
            desc="KRAFTON features: prefix replay at the cuts (database)",
            cwd=KR,
            deps=("mps.extract",),
            est_min=5,
            needs_db=True,
            cmd=_kr(
                cfg,
                "validate_causal",
                "--cohorts",
                ",".join(vc),
                "--data-dir",
                str(p.krafton_data),
                "--out",
                str(p.krafton_data / "logs" / "validate_causal.json"),
            ),
        )
        last = "mps.validate_causal"
    add(
        id="mps.kftable",
        lane=s0,
        desc="KRAFTON feature table at the cuts (database, 2 workers, --verify)",
        cwd=KR,
        deps=(last,),
        est_min=6,
        needs_db=True,
        cmd=_kr(
            cfg,
            "kftable",
            "--cohorts",
            cohorts,
            "--workers",
            "2",
            "--verify",
            "--label-dir",
            str(p.krafton_data),
            "--data-dir",
            str(p.krafton_data),
            "--out-dir",
            str(p.kf),
        ),
    )
    add(
        id="mps.hgbkf",
        lane=s0,
        desc="HGB-kraftonfeat (a GBDT on KRAFTON's inputs, 1 thread)",
        cmd=_run_all(cfg, "hgbkf", s0),
        cwd=KR,
        deps=("mps.kftable", "cpu.hgb"),
        est_min=1.5,
    )
    add(
        id="mps.k2",
        lane=s0,
        desc="K2 = KRAFTON-tf-ourfeat (their transformer + Cox loss on our features)",
        cmd=_run_all(cfg, "ourfeat", s0, *chain_args),
        cwd=KR,
        deps=("mps.hgbkf", "cpu.hgb"),
        est_min=40,
    )
    add(
        id="mps.k3",
        lane=s0,
        desc="K3a = -ourfeat-bce and K3 = -ourfeat-hazard",
        cmd=_run_all(cfg, "k3", s0, *chain_args),
        cwd=KR,
        deps=("mps.k2",),
        est_min=76,
    )
    k1_deps = ("mps.extract", "cpu.hgb") if s1 != s0 else ("mps.k3", "cpu.hgb")
    add(
        id="mps.k1",
        lane=s1,
        desc="official arms: KRAFTON-tf" + (" + KRAFTON-mlp" if cfg.with_mlp else ""),
        cmd=_run_all(cfg, "main", s1, *chain_args, "--encoders", k1_encoders),
        cwd=KR,
        deps=k1_deps,
        est_min=68 + (39 if cfg.with_mlp else 0),
    )
    add(
        id="mps.h0",
        lane=s0,
        desc="H0 arms: phase baseline (grouped-time Cox) on the K1 / K2 scores",
        cmd=_run_all(cfg, "h0", s0, "--device", "cpu", "--threads", "3"),
        cwd=KR,
        deps=("mps.k3", "mps.k1"),
        est_min=1.5,
    )

    # ---- 4-7 analysis: the main lane after every producer ------------------------------------------------------------------
    producers = tuple(s.id for s in steps if s.lane != "main")
    add(
        id="analysis.aggregate",
        lane="main",
        desc="aggregate: paired bootstrap, DL pooling, P1-P5 + Holm, decomposition steps, RQ2",
        est_min=7,
        deps=producers,
        needs_db=False,
        cmd=[
            py,
            "-m",
            "model_stgcn.ptr.aggregate",
            "--runs-dir",
            str(p.runs),
            "--out-dir",
            str(p.results),
            "--dump-dir",
            str(p.dump),
            "--workers",
            str(cfg.cpu_workers),
            "--n-boot",
            str(cfg.n_boot),
        ],
        outputs=(p.results / "contrasts_pooled.csv", p.results / "summary.json"),
    )
    add(
        id="analysis.supplement",
        lane="main",
        desc="supplement: within match-phase AUC, REML + HKSJ sensitivity",
        est_min=8,
        deps=("analysis.aggregate",),
        cmd=[
            py,
            "-m",
            "model_stgcn.ptr.supplement",
            "--runs-dir",
            str(p.runs),
            "--results-dir",
            str(p.results),
            "--out-dir",
            str(p.supplement),
            "--dump-dir",
            str(p.dump),
            "--workers",
            "3",
        ],
        outputs=(p.supplement / "supplement_summary.json",),
        env={"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
    )
    add(
        id="reports.ehm_report",
        lane="main",
        desc="ecological model report (share of discrimination, coefficients)",
        cmd=[*eco, "--stage", "report"],
        env=eco_env,
        est_min=5,
        deps=("analysis.supplement",),
    )
    prev = "reports.ehm_report"
    if cfg.with_lc:
        add(
            id="reports.learning_curve",
            lane="main",
            desc="learning-curve report",
            est_min=1,
            deps=(prev,),
            outputs=(p.results / "learning_curve.csv",),
            cmd=[
                py,
                "-m",
                "model_stgcn.ptr.learning_curve",
                "--lc-dir",
                str(p.runs_lc),
                "--main-dir",
                str(p.runs),
                "--steps-dir",
                str(p.runs_lc_steps),
                "--out-dir",
                str(p.results),
            ],
        )
        prev = "reports.learning_curve"
    add(
        id="tables.tab_decomp",
        lane="main",
        desc="tab_decomp (steps (a)-(g) + rank-loss group) and every aggregator table, re-rendered from the CSVs",
        est_min=0.5,
        deps=(prev,),
        cmd=[
            py,
            "-m",
            "model_stgcn.ptr.aggregate",
            "--rerender-tables",
            "--out-dir",
            str(p.results),
        ],
        outputs=(p.results / "tables" / "tab_decomp.tex",),
    )
    kr_out = p.krafton_reports
    add(
        id="tables.kftable_h0_report",
        lane="main",
        desc="KRAFTON tables: H0 arms next to the Cox, hazard and PTR arms (h0_report)",
        est_min=3,
        deps=("tables.tab_decomp",),
        cwd=KR,
        cmd=_kr(
            cfg,
            "h0_report",
            "--ckpt-dir",
            str(p.runs),
            "--dump-dir",
            str(p.dump),
            "--out-dir",
            str(kr_out),
        ),
        outputs=(kr_out / "h0_contrasts.csv",),
    )
    add(
        id="tables.kftable_k3_contrasts",
        lane="main",
        desc="KRAFTON tables: K3 paired contrasts (k3_contrasts)",
        est_min=1,
        deps=("tables.kftable_h0_report",),
        cwd=KR,
        cmd=_kr(
            cfg,
            "k3_contrasts",
            "--ckpt-dir",
            str(p.runs),
            "--out",
            str(kr_out / "k3_contrasts.csv"),
        ),
        outputs=(kr_out / "k3_contrasts.csv",),
    )
    add(
        id="tables.kftable_k3_facts",
        lane="main",
        desc="KRAFTON tables: run facts of the K arms (k3_facts)",
        est_min=0.5,
        deps=("tables.kftable_k3_contrasts",),
        cwd=KR,
        cmd=_kr(
            cfg, "k3_facts", "--ckpt-dir", str(p.runs), "--out", str(kr_out / "k3_run_facts.csv")
        ),
        outputs=(kr_out / "k3_run_facts.csv",),
    )
    add(
        id="tables.kftable_phase_auc",
        lane="main",
        desc="KRAFTON tables: within-phase AUC of the K arms (phase_auc)",
        est_min=1,
        deps=("tables.kftable_k3_facts",),
        cwd=KR,
        cmd=_kr(cfg, "phase_auc", "--ckpt-dir", str(p.runs), "--out-dir", str(kr_out)),
    )
    add(
        id="tables.make_paper_tables",
        lane="main",
        desc="paper tables and figures from the aggregator outputs",
        est_min=1,
        deps=("tables.kftable_phase_auc",),
        cmd=[
            py,
            str(P.PAPER_DIR / "tables" / "make_paper_tables.py"),
            "--results-dir",
            str(p.results),
            "--paper-dir",
            str(p.paper),
            "--dump-dir",
            str(p.dump),
        ],
    )  # the script is the repository's
    add(
        id="numbers.paper_numbers",
        lane="main",
        desc="every dump-dependent number of the paper drafts, with provenance",
        est_min=25,
        deps=("tables.make_paper_tables",),
        needs_db=True,
        cmd=[
            py,
            str(REPO / "tools" / "paper_numbers_v2.py"),
            "--dump-dir",
            str(p.dump),
            "--runs-dir",
            str(p.runs),
            "--results-dir",
            str(p.results),
            "--out",
            str(p.paper_numbers),
        ],
        outputs=(p.paper_numbers,),
    )
    return steps


# =========================================================================== the gate (in-process)
def gate_dump_fn(cfg: Config) -> Callable[[Step, Path], int]:
    def gate_dump(step: Step, log: Path) -> int:
        p = cfg.paths
        lines: list[str] = []
        rc = 0
        try:
            dv = P.check_same_version(
                p.dump, {name: root for name, root in p.artifact_roots().items()}
            )
            lines.append(f"dump {p.dump}: version {dv}")
            if dv != P.VERSION:
                raise P.VersionMixError(f"the dump {p.dump} is {dv}, not {P.VERSION}")
        except P.VersionMixError as exc:
            lines.append(f"VERSION: {exc}")
            rc = 1
        res = P.validate_v2_dump(
            p.dump, require_prefix_replay=True
        )  # meta.version == 'v2' of every cohort, the record invariants, prefix_replay.json without mismatches
        lines += [f"error: {e}" for e in res["errors"]] + [f"warning: {w}" for w in res["warnings"]]
        if res.get("prefix_replay"):
            lines.append(f"prefix replay: {res['prefix_replay']}")
        lines.append(f"cohorts: {len(res['cohorts'])}; dump meta sha1 {res['dump_meta_sha1']}")
        rc = rc or (1 if res["errors"] else 0)
        if cfg.cohorts and set(cfg.cohorts) != set(res["cohorts"]):
            lines.append(
                f"error: cohorts of the dump {res['cohorts']} differ from the planned {list(cfg.cohorts)}"
            )
            rc = 1
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return rc

    return gate_dump


# =========================================================================== validation of the plan (scope, lane order + dependencies)
def check_scope(steps: Sequence[Step]) -> None:
    """The v2 re-run leaves stage 2 out: the player layer (``M11RZ-P`` / ``M11RZ-P-tm``) needs the player dump, which is not rebuilt, and the paper does not use it.
    No command of the plan may name those arms, ask for ``--stage2`` or pass a player dump."""
    for s in steps:
        for tok in s.cmd or []:
            hit = [t for t in PLAYER_LAYER_TOKENS if t in str(tok)]
            if hit:
                raise ValueError(
                    f"step {s.id} asks for the player layer ({hit[0]!r} in {str(tok)!r}): stage 2 is not part of the v2 re-run (the player dump is not rebuilt)"
                )


def validate_plan(steps: Sequence[Step], *, universe: Sequence[Step] | None = None) -> None:
    """Every dependency exists, ids are unique, and running every lane strictly in order cannot deadlock (checked by simulation).  For a selection of the plan
    (``--only`` / ``--skip``) pass the full plan as ``universe``: a dependency outside the selection counts as satisfied."""
    ids = [s.id for s in steps]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate step ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    check_scope(steps)
    known = set(ids)
    everything = {s.id for s in (universe if universe is not None else steps)}
    for s in steps:
        for d in s.deps:
            if d not in everything:
                raise ValueError(f"step {s.id} depends on unknown step {d}")
    lanes: dict[str, list[Step]] = OrderedDict()
    for s in steps:
        lanes.setdefault(s.lane, []).append(s)
    done: set[str] = set()
    heads = {lane: 0 for lane in lanes}
    while any(heads[lane] < len(v) for lane, v in lanes.items()):
        progressed = False
        for lane, v in lanes.items():
            while heads[lane] < len(v) and all(
                d in done or d not in known for d in v[heads[lane]].deps
            ):
                done.add(v[heads[lane]].id)
                heads[lane] += 1
                progressed = True
        if not progressed:
            stuck = {lane: v[heads[lane]].id for lane, v in lanes.items() if heads[lane] < len(v)}
            raise ValueError(f"the plan would deadlock: every lane waits ({stuck})")


def simulate(steps: Sequence[Step]) -> dict[str, tuple[float, float]]:
    """Start / end minutes of every step when each lane runs in order and a step waits for its dependencies (list scheduling with the estimates)."""
    lanes: dict[str, list[Step]] = OrderedDict()
    for s in steps:
        lanes.setdefault(s.lane, []).append(s)
    end: dict[str, float] = {}
    out: dict[str, tuple[float, float]] = {}
    free = {lane: 0.0 for lane in lanes}
    heads = {lane: 0 for lane in lanes}
    known = {s.id for s in steps}
    while any(heads[lane] < len(v) for lane, v in lanes.items()):
        for lane, v in lanes.items():
            while heads[lane] < len(v) and all(
                d in end or d not in known for d in v[heads[lane]].deps
            ):
                s = v[heads[lane]]
                start = max([free[lane], *[end[d] for d in s.deps if d in end]])
                end[s.id] = start + s.est_min
                out[s.id] = (start, end[s.id])
                free[lane] = end[s.id]
                heads[lane] += 1
    return out


# =========================================================================== provenance: git state and the run manifest
def git_state(repo: Path = REPO) -> dict:
    def run(*args: str) -> bytes | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=str(repo),
                stderr=subprocess.DEVNULL,
                timeout=60,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except Exception:
            return None

    head = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    return {
        "head": head.decode().strip() if head else None, "branch": branch.decode().strip() if branch else None,
        "status_porcelain_sha1": hashlib.sha1(status).hexdigest() if status is not None else None,
        "n_status_lines": len(status.splitlines()) if status is not None else None,
    }  # fmt: skip


def current_provenance(cfg: Config, *, repo: Path = REPO) -> dict:
    p = cfg.paths
    version, meta_sha1 = "unknown", None
    if p.dump.is_dir():
        try:
            version = P.dump_version(p.dump)
            meta_sha1 = P.dump_meta_sha1(p.dump, deep=True) if version != "unknown" else None
        except Exception:  # a dump that is still being written: the gate reports it
            meta_sha1 = None
    return {
        "git": git_state(repo),
        "result_modules_sha1": code_fingerprint(),
        "dump": {"dir": str(p.dump), "version": version, "meta_sha1": meta_sha1},
    }


def check_resume(manifest: dict | None, now: dict) -> tuple[list[str], list[str]]:
    """``(errors, warnings)`` of resuming a run whose manifest is ``manifest`` with the provenance ``now``."""
    if not manifest:
        return [], []
    old = manifest.get("provenance", manifest)
    errors, warnings = [], []
    if old["git"]["head"] != now["git"]["head"]:
        errors.append(f"git HEAD changed: {old['git']['head']} -> {now['git']['head']}")
    if old["result_modules_sha1"] != now["result_modules_sha1"]:
        errors.append(
            "the result-affecting modules (arms, data, extra, hgb, loss, metrics, model, player, relations, train, util) changed: finished runs would be stale"
        )
    if (
        old["dump"]["meta_sha1"]
        and now["dump"]["meta_sha1"]
        and old["dump"]["meta_sha1"] != now["dump"]["meta_sha1"]
    ):
        errors.append("the dump meta hash changed: the dump was rebuilt after the first launch")
    if old["git"]["status_porcelain_sha1"] != now["git"]["status_porcelain_sha1"]:
        warnings.append(
            "git status --porcelain changed since the first launch (the result modules and HEAD are what decide whether finished steps stay valid)"
        )
    return errors, warnings


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =========================================================================== the code snapshot (commits wait for the user's approval)
def _git(repo: Path, *args: str, timeout: int = 300) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        timeout=timeout,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )  # read-only: no index refresh under another agent
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args[:2])} failed in {repo}: {proc.stderr.decode(errors='replace').strip()[:300]}"
        )
    return proc.stdout


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def in_snapshot_scope(rel: str) -> bool:
    return rel.endswith(SNAPSHOT_EXTENSIONS) and any(
        rel == r or rel.startswith(r + "/") for r in SNAPSHOT_ROOTS
    )


def list_untracked(repo: Path) -> tuple[list[str], list[str]]:
    """``(in scope, outside scope)`` of the untracked files, sorted POSIX paths relative to the repository.  ``git ls-files --others --exclude-standard``: what the
    ignore rules keep out (``reports/``, ``model_stgcn/dumps/``, the checkpoints) stays out."""
    raw = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    paths = sorted(os.fsdecode(x) for x in raw.split(b"\0") if x)
    return [x for x in paths if in_snapshot_scope(x)], [
        x for x in paths if not in_snapshot_scope(x)
    ]


def count_diff_files(diff: bytes) -> int:
    return sum(1 for line in diff.split(b"\n") if line.startswith(b"diff --git "))


def take_code_snapshot(repo: Path, out_dir: Path, *, max_file_bytes: int | None = None) -> dict:
    """Write the snapshot of ``repo`` into ``out_dir`` and return its summary: ``git_diff_HEAD.patch`` (``git diff HEAD --binary``), ``untracked_files.tar`` (the untracked
    files in scope, read once: the hash is of the bytes in the tar), ``snapshot.json``, ``SHA256SUMS`` (the four files) and ``untracked_files.sha256`` (every untracked
    file, ``sha256sum`` format, repository-relative paths).  An implausibly large untracked set is refused (something that is not code slipped past the ignore rules)."""
    limit = SNAPSHOT_MAX_FILE_BYTES if max_file_bytes is None else max_file_bytes
    repo, out = Path(repo), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    porcelain = _git(repo, "status", "--porcelain")
    diff = _git(repo, *SNAPSHOT_DIFF_ARGS, timeout=900)
    patch, tar_path = out / "git_diff_HEAD.patch", out / "untracked_files.tar"
    patch.write_bytes(diff)
    in_scope, outside = list_untracked(repo)
    if len(in_scope) > SNAPSHOT_MAX_FILES:
        raise RuntimeError(
            f"{len(in_scope)} untracked files in scope (more than {SNAPSHOT_MAX_FILES}): something that is not code is not ignored; look at `git status`"
        )
    files: list[dict] = []
    skipped: list[dict] = []
    total = 0
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tf:
        for rel in in_scope:
            path = repo / rel
            try:
                st = path.lstat()
            except OSError as exc:
                skipped.append({"path": rel, "reason": f"unreadable ({exc.strerror})"})
                continue
            if not stat.S_ISREG(st.st_mode):
                skipped.append({"path": rel, "reason": "not a regular file (a symbolic link)"})
                continue
            if st.st_size > limit:
                skipped.append(
                    {"path": rel, "reason": f"larger than {limit} bytes", "size": st.st_size}
                )
                continue
            data = path.read_bytes()
            total += len(data)
            if total > SNAPSHOT_MAX_TOTAL_BYTES:
                raise RuntimeError(
                    f"the untracked files in scope add up to more than {SNAPSHOT_MAX_TOTAL_BYTES} bytes: something that is not code is not ignored"
                )
            info = tarfile.TarInfo(rel)
            info.size, info.mode, info.mtime = len(data), st.st_mode & 0o777, int(st.st_mtime)
            tf.addfile(info, io.BytesIO(data))
            files.append(
                {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            )
    diff_sha, tar_sha = hashlib.sha256(diff).hexdigest(), sha256_file(tar_path)
    digest = hashlib.sha256(
        "\n".join([diff_sha, *[f"{f['sha256']}  {f['path']}" for f in files]]).encode()
    ).hexdigest()  # a function of the code alone
    n_changed = count_diff_files(diff)
    (out / "untracked_files.sha256").write_text(
        "".join(f"{f['sha256']}  {f['path']}\n" for f in files), encoding="utf-8"
    )
    doc = {
        "created_utc": _utc(), "repo": str(repo), "head": head, "branch": branch, "status_porcelain_sha1": hashlib.sha1(porcelain).hexdigest(),
        "n_status_lines": len(porcelain.splitlines()), "digest": digest,
        "git_diff_HEAD": {"file": patch.name, "sha256": diff_sha, "bytes": len(diff), "n_files_changed": n_changed, "command": "git " + " ".join(SNAPSHOT_DIFF_ARGS)},
        "untracked": {"file": tar_path.name, "sha256": tar_sha, "bytes": tar_path.stat().st_size, "n_files": len(files), "files": files, "skipped": skipped,
                      "scope": {"roots": list(SNAPSHOT_ROOTS), "extensions": list(SNAPSHOT_EXTENSIONS)},
                      "outside_scope": {"n": len(outside), "first": outside[:200], "note": "untracked files that the snapshot does not capture (other directories or extensions)"}},
        "restore": f"git checkout {head} && git apply --binary {patch.name} && tar -xf {tar_path.name}   (from the repository root, with the two files of this directory)",
    }  # fmt: skip
    (out / "snapshot.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    (out / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(out / n)}  {n}\n"
            for n in (patch.name, tar_path.name, "snapshot.json", "untracked_files.sha256")
        ),
        encoding="utf-8",
    )
    return {"dir": str(out), "digest": digest, "head": head, "git_diff_HEAD": {"sha256": diff_sha, "bytes": len(diff), "n_files_changed": n_changed},
            "untracked_tar": {"sha256": tar_sha, "bytes": tar_path.stat().st_size, "n_files": len(files), "n_skipped": len(skipped)}, "n_untracked_outside_scope": len(outside)}  # fmt: skip


def verify_code_snapshot(directory: Path) -> list[str]:
    """The problems of a snapshot directory (an empty list: every hash matches): the patch and the tar against ``snapshot.json``, every member of the tar against the
    recorded sha256 of its file, ``SHA256SUMS`` against the files."""
    d = Path(directory)
    try:
        doc = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"snapshot.json is unreadable: {exc}"]
    problems: list[str] = []
    for key, name in (
        ("git_diff_HEAD", "git_diff_HEAD.patch"),
        ("untracked", "untracked_files.tar"),
    ):
        f = d / name
        if not f.is_file():
            problems.append(f"{name} is missing")
        elif sha256_file(f) != doc[key]["sha256"]:
            problems.append(f"{name}: the sha256 differs from snapshot.json")
    recorded = {f["path"]: f["sha256"] for f in doc["untracked"]["files"]}
    try:
        with tarfile.open(d / "untracked_files.tar") as tf:
            members = {m.name: m for m in tf.getmembers() if m.isfile()}
            if set(members) != set(recorded):
                problems.append(
                    f"the tar holds {len(members)} files, snapshot.json lists {len(recorded)}"
                )
            for name, sha in recorded.items():
                m = members.get(name)
                if m is not None and hashlib.sha256(tf.extractfile(m).read()).hexdigest() != sha:
                    problems.append(f"{name}: the sha256 differs inside the tar")
    except (OSError, tarfile.TarError) as exc:
        problems.append(f"untracked_files.tar is unreadable: {exc}")
    sums = d / "SHA256SUMS"
    if not sums.is_file():
        problems.append("SHA256SUMS is missing")
    else:
        for line in sums.read_text(encoding="utf-8").splitlines():
            sha, _, name = line.partition("  ")
            if not (d / name).is_file() or sha256_file(d / name) != sha:
                problems.append(f"SHA256SUMS: {name} does not match")
    return problems


def snapshot_launch(repo: Path, results: Path) -> dict:
    """Save the code of this launch under ``<results>/code_snapshot/``: the first snapshot of a results root goes into that directory itself, a later one with other
    code into ``launch_<UTC>/`` (the first one is kept: it is the code that made the first results); the same code as a saved snapshot adds nothing.  Returns the
    summary of the snapshot that holds this launch's code, with ``dir`` and ``unchanged``."""
    root = Path(results) / SNAPSHOT_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        info = take_code_snapshot(repo, tmp)
        problems = verify_code_snapshot(tmp)
        if problems:
            raise RuntimeError("the snapshot does not verify: " + "; ".join(problems[:3]))
        known: dict[str, Path] = {}
        for d in [root, *sorted(root.glob("launch_*"))]:
            try:
                known[json.loads((d / "snapshot.json").read_text(encoding="utf-8"))["digest"]] = d
            except (OSError, ValueError, KeyError):
                continue
        if info["digest"] in known:
            target, unchanged = known[info["digest"]], True
        elif not (root / "snapshot.json").is_file():
            for f in tmp.iterdir():
                os.replace(f, root / f.name)
            target, unchanged = root, False
        else:
            target = root / f"launch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            k = 1
            while target.exists():
                k += 1
                target = (
                    root / f"launch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{k}"
                )
            os.replace(tmp, target)
            unchanged = False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {**info, "dir": str(target), "unchanged": unchanged}


def preview_code_snapshot(repo: Path = REPO) -> dict:
    """What a launch would save, without writing anything (the dry run): counts and sizes."""
    try:
        diff = _git(repo, *SNAPSHOT_DIFF_ARGS, timeout=900)
        in_scope, outside = list_untracked(repo)
        size = 0
        for rel in in_scope:
            try:
                st = (Path(repo) / rel).lstat()
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode) and st.st_size <= SNAPSHOT_MAX_FILE_BYTES:
                size += st.st_size
        return {
            "ok": True,
            "n_files_changed": count_diff_files(diff),
            "diff_bytes": len(diff),
            "n_untracked": len(in_scope),
            "untracked_bytes": size,
            "n_outside_scope": len(outside),
        }
    except Exception as exc:  # git missing, not a repository
        return {"ok": False, "error": str(exc)}


class State:
    """``run_state.json``: the status of every step; thread-safe, rewritten atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        try:
            self.steps: dict[str, dict] = json.loads(path.read_text(encoding="utf-8")).get(
                "steps", {}
            )
        except (OSError, ValueError):
            self.steps = {}

    def set(self, step_id: str, **fields) -> None:
        with self.lock:
            self.steps.setdefault(step_id, {}).update(fields)
            write_json_atomic(self.path, {"updated_utc": _utc(), "steps": self.steps})

    def clear(self) -> None:
        with self.lock:
            self.steps = {}
            write_json_atomic(self.path, {"updated_utc": _utc(), "steps": {}})

    def is_done(self, step: Step) -> bool:
        if step.always:
            return False
        rec = self.steps.get(step.id)
        return bool(
            rec
            and rec.get("status") == "done"
            and rec.get("signature") == step.signature()
            and all(Path(o).exists() for o in step.outputs)
        )


# =========================================================================== the runner
class Runner:
    """Runs the lanes concurrently: a lane executes its steps strictly in order and waits for the steps it depends on."""

    def __init__(self, steps: Sequence[Step], state: State, logs: Path, *, base_env: dict[str, str] | None = None, stop_on_failure: bool = False,
                 log: Callable[[str], None] = print, universe: Sequence[Step] | None = None) -> None:  # fmt: skip
        validate_plan(steps, universe=universe)
        self.steps, self.state, self.logs, self.stop_on_failure, self.log = (
            list(steps),
            state,
            logs,
            stop_on_failure,
            log,
        )
        self.base_env = dict(os.environ if base_env is None else base_env)
        self.cv = threading.Condition()
        self.status: dict[str, str] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.aborted = False
        self.known = {s.id for s in steps}

    # ---- one step
    def _execute(self, step: Step) -> int:
        self.logs.mkdir(parents=True, exist_ok=True)
        log = self.logs / f"{step.id}.log"
        if step.fn is not None:
            return step.fn(step, log)
        env = dict(self.base_env)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH", "")]))
        env.update(step.env)
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"# {_utc()} start {step.id} (cwd {step.cwd})\n# {' '.join(step.cmd or [])}\n")
            lf.flush()
            proc = subprocess.Popen(
                step.cmd, cwd=str(step.cwd), stdout=lf, stderr=subprocess.STDOUT, env=env
            )
            self.procs[step.id] = proc
            try:
                rc = proc.wait()
            finally:
                self.procs.pop(step.id, None)
            lf.write(f"# {_utc()} end {step.id} exit {rc}\n")
        return rc

    def _lane(self, lane: str, steps: list[Step]) -> None:
        for step in steps:
            with self.cv:
                while True:
                    if self.aborted:
                        return
                    deps = [
                        self.status.get(d, "pending") for d in step.deps if d in self.known
                    ]  # a dependency outside the plan (--only) counts as satisfied
                    if any(s in ("failed", "skipped", "interrupted") for s in deps):
                        self.status[step.id] = "skipped"
                        self.cv.notify_all()
                        break
                    if all(s in ("done", "cached") for s in deps):
                        break
                    self.cv.wait(timeout=0.5)
            if self.status.get(step.id) == "skipped":
                self.state.set(
                    step.id,
                    status="skipped",
                    signature=step.signature(),
                    note="a dependency did not finish",
                )
                self.log(f"[{lane}] {step.id}: skipped (a dependency did not finish)")
                continue
            if self.state.is_done(step):
                with self.cv:
                    self.status[step.id] = "cached"
                    self.cv.notify_all()
                self.log(
                    f"[{lane}] {step.id}: done before (same command, outputs in place), skipped"
                )
                continue
            self.state.set(
                step.id,
                status="running",
                start_utc=_utc(),
                signature=step.signature(),
                lane=lane,
                log=str(self.logs / f"{step.id}.log"),
            )
            self.log(f"[{lane}] {step.id}: start")
            t0 = time.time()
            try:
                rc = self._execute(step)
            except Exception as exc:  # the command could not even start
                rc = 127
                (self.logs / f"{step.id}.log").write_text(
                    f"could not run: {exc!r}\n", encoding="utf-8"
                )
            ok = rc == 0 and all(Path(o).exists() for o in step.outputs)
            status = "done" if ok else ("interrupted" if self.aborted else "failed")
            self.state.set(step.id, status=status, exit_code=rc, end_utc=_utc(), elapsed_sec=round(time.time() - t0, 1),
                           note="" if ok or rc != 0 else f"exit 0 but outputs missing: {[str(o) for o in step.outputs if not Path(o).exists()]}")  # fmt: skip
            with self.cv:
                self.status[step.id] = status
                if not ok and self.stop_on_failure:
                    self.aborted = True
                    for proc in list(self.procs.values()):
                        proc.terminate()
                self.cv.notify_all()
            self.log(f"[{lane}] {step.id}: {status} in {time.time() - t0:.0f}s (exit {rc})")

    def run(self) -> int:
        lanes: dict[str, list[Step]] = OrderedDict()
        for s in self.steps:
            lanes.setdefault(s.lane, []).append(s)
        threads = [
            threading.Thread(target=self._lane, args=(lane, v), name=f"lane-{lane}", daemon=True)
            for lane, v in lanes.items()
        ]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.5)
        except KeyboardInterrupt:
            self.log("interrupted: terminating the running steps")
            with self.cv:
                self.aborted = True
                for proc in list(self.procs.values()):
                    proc.terminate()
                self.cv.notify_all()
            for t in threads:
                t.join(timeout=30)
            for step in self.steps:
                if (
                    self.status.get(step.id) is None
                    and self.state.steps.get(step.id, {}).get("status") == "running"
                ):
                    self.state.set(step.id, status="interrupted")
            return 130
        bad = [
            s.id
            for s in self.steps
            if self.status.get(s.id) in ("failed", "skipped", "interrupted")
        ]
        return 1 if bad else 0


# =========================================================================== command line
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    d = P.RerunPaths()
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit; nothing is written, started or read beyond the dump's manifests",
    )
    ap.add_argument("--dump-dir", default=str(d.dump), help="phase dump (default: the v2 dump)")
    ap.add_argument(
        "--runs-dir", default=str(d.runs), help="run directory of every arm (default: the v2 root)"
    )
    ap.add_argument(
        "--kf-dir", default=str(d.kf), help="KRAFTON feature table (default: the v2 root)"
    )
    ap.add_argument(
        "--krafton-data-dir",
        default=str(d.krafton_data),
        help="KRAFTON labels / shards / logs (default: the v2 root)",
    )
    ap.add_argument(
        "--sagong-dir", default=str(d.sagong), help="Sagong cache (default: the v2 root)"
    )
    ap.add_argument(
        "--results-dir",
        default=str(d.results),
        help="aggregator / supplement / logs / manifest / paper numbers (default: the v2 root)",
    )
    ap.add_argument(
        "--krafton-reports-dir",
        default=str(d.krafton_reports),
        help="KRAFTON report tables (default: the v2 root)",
    )
    ap.add_argument(
        "--paper-dir", default=str(d.paper), help="paper directory with tables/ and figures/"
    )
    ap.add_argument("--lc-dir", default=None, help="learning curve runs (default: <runs-dir>_lc)")
    ap.add_argument(
        "--lc-steps-dir",
        default=None,
        help="steps-matched learning curve runs (default: <runs-dir>_lc_steps)",
    )
    ap.add_argument("--cpu-workers", type=int, default=6)
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument(
        "--mps-streams",
        type=int,
        default=1,
        choices=[1, 2],
        help="concurrent KRAFTON training processes (2: K1 on the second stream)",
    )
    ap.add_argument("--mps-threads", type=int, default=3)
    ap.add_argument(
        "--python", default=sys.executable, help="interpreter of the PTR / KRAFTON steps"
    )
    ap.add_argument(
        "--python-sagong",
        default=SAGONG_PYTHON if Path(SAGONG_PYTHON).exists() else sys.executable,
        help="interpreter of the Sagong steps (xgboost + pyarrow)",
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument(
        "--with-lc",
        action="store_true",
        help="also the learning curve (fixed epochs, steps matched) and its report [optional in the plan]",
    )
    ap.add_argument(
        "--with-mlp", action="store_true", help="also KRAFTON-mlp (39 MPS minutes) [optional]"
    )
    ap.add_argument(
        "--with-m11r-union", action="store_true", help="also M11R-union (45 CPU-minutes) [optional]"
    )
    ap.add_argument(
        "--with-exploratory",
        action="store_true",
        help="also the three M11R-no* relation-removal arms [not in the plan]",
    )
    ap.add_argument(
        "--no-validate-causal", action="store_true", help="skip adaptation.validate_causal"
    )
    ap.add_argument(
        "--leak-tests",
        default=" ".join(DEFAULT_LEAK_TESTS),
        help="pytest targets of the gate (space separated)",
    )
    ap.add_argument(
        "--skip-leak-tests",
        action="store_true",
        help="do not run the leak tests (recorded as skipped; not recommended)",
    )
    ap.add_argument(
        "--only", default="", help="comma separated step ids or prefixes ('cpu.', 'mps.k')"
    )
    ap.add_argument("--skip", default="", help="comma separated step ids or prefixes to leave out")
    ap.add_argument(
        "--restart",
        action="store_true",
        help="forget the recorded state (the outputs stay; the tools skip what they have finished)",
    )
    ap.add_argument(
        "--allow-code-change",
        action="store_true",
        help="resume although HEAD / the result modules / the dump meta hash changed since the first launch",
    )
    ap.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="terminate everything at the first failed step (default: only its dependents are skipped)",
    )
    ap.add_argument("--no-code-snapshot", action="store_true", help="do not save the code (git diff HEAD + the untracked sources) into <results>/code_snapshot/ [not recommended: the manifest "
                    "would not hold the code that made the results]")  # fmt: skip
    return ap.parse_args(argv)


def config_from_args(a: argparse.Namespace) -> Config:
    runs = Path(a.runs_dir)
    paths = P.RerunPaths(dump=Path(a.dump_dir), kf=Path(a.kf_dir), krafton_data=Path(a.krafton_data_dir), sagong=Path(a.sagong_dir), runs=runs,
                         runs_lc=Path(a.lc_dir) if a.lc_dir else runs.with_name(runs.name + "_lc"),
                         runs_lc_steps=Path(a.lc_steps_dir) if a.lc_steps_dir else runs.with_name(runs.name + "_lc_steps"),
                         results=Path(a.results_dir), krafton_reports=Path(a.krafton_reports_dir), paper=Path(a.paper_dir))  # fmt: skip
    return Config(paths=paths, cpu_workers=a.cpu_workers, cpu_threads=a.cpu_threads, mps_streams=a.mps_streams, mps_threads=a.mps_threads, python=a.python,
                  python_sagong=a.python_sagong, n_boot=a.n_boot, with_lc=a.with_lc, with_mlp=a.with_mlp, with_m11r_union=a.with_m11r_union,
                  with_exploratory=a.with_exploratory, validate_causal=not a.no_validate_causal, leak_tests=tuple(a.leak_tests.split()), skip_leak_tests=a.skip_leak_tests,
                  cohorts=cohorts_of(paths))  # fmt: skip


def select(steps: list[Step], only: str, skip: str) -> list[Step]:
    def match(step_id: str, spec: str) -> bool:
        return any(
            step_id == tok or step_id.startswith(tok)
            for tok in (t.strip() for t in spec.split(","))
            if tok
        )

    out = [s for s in steps if (not only or match(s.id, only)) and not (skip and match(s.id, skip))]
    if only and not out:
        raise SystemExit(f"--only {only!r} matches no step; ids: {[s.id for s in steps]}")
    return out


def check_roots(cfg: Config) -> None:
    """Refuse a v1 root next to the v2 dump before anything is planned."""
    try:
        P.check_same_version(cfg.paths.dump, cfg.paths.artifact_roots())
    except P.VersionMixError as exc:
        raise SystemExit(f"refusing to run: {exc}") from exc


def plan_counts(cfg: Config) -> dict[str, int]:
    """Run counts of the run_matrix steps (from the dump's cohorts, or the projected 12-cohort registry)."""
    cohorts = list(cfg.cohorts) or list(RM.PROJECTED_REGISTRY)
    runs, _ = RM.enumerate_runs(cohorts)
    n = {g: sum(1 for r in runs if r.group == g) for g in ("main", "exploratory", "seeds", "hgb")}
    n["norank"] = len(RM.enumerate_norank_runs(cohorts))
    n["union_hgb"] = len(cohorts)
    n["union_m11r"] = len(cohorts) + sum(2 for c in cohorts if c in RM.LARGE_COHORTS)
    return n


def _db_note(steps: Sequence[Step]) -> str:
    need = [s.id for s in steps if s.needs_db]
    have = bool(os.environ.get("PG_PASSWORD") or os.environ.get("PGPASSWORD"))
    if not need:
        return "not needed by the selected steps"
    return (
        "yes"
        if have
        else f"NO (steps {', '.join(need)} need them: set -a; source <pubg_collector>/.env; set +a)"
    )


def _mb(n: int) -> str:
    return f"{n / 1048576:.1f} MB"


def format_plan(cfg: Config, steps: list[Step], state: State | None, prov: dict, *, manifest: dict | None = None, errors: Sequence[str] = (), warnings: Sequence[str] = (),
                snapshot: dict | None = None) -> str:  # fmt: skip
    p = cfg.paths
    sim = simulate(steps)
    wall = max((e for _, e in sim.values()), default=0.0)
    ncpu = os.cpu_count() or 1
    cores = cfg.cpu_workers * cfg.cpu_threads + cfg.mps_streams * cfg.mps_threads
    counts = plan_counts(cfg)
    lines = [
        f"BigComp 2027 re-run on the v2 dump: {len(steps)} steps, lanes {', '.join(OrderedDict.fromkeys(s.lane for s in steps))}",
        f"dump          {p.dump}  version {prov['dump']['version']}  meta sha1 {prov['dump']['meta_sha1'] or '-'}  cohorts {len(cfg.cohorts)}: {', '.join(cfg.cohorts)}",
        f"roots         runs {p.runs}",
        f"              kf table {p.kf} | KRAFTON data {p.krafton_data} | Sagong {p.sagong}",
        f"              results {p.results} | KRAFTON reports {p.krafton_reports} | paper {p.paper}",
        f"git           HEAD {prov['git']['head']} ({prov['git']['branch']}), status --porcelain sha1 {prov['git']['status_porcelain_sha1']} ({prov['git']['n_status_lines']} lines), "
        f"result modules sha1 {prov['result_modules_sha1'][:12]}",
        f"budget        {cfg.cpu_workers} CPU workers x {cfg.cpu_threads} threads + {cfg.mps_streams} MPS stream(s) x {cfg.mps_threads} threads = {cores} cores of {ncpu}"
        + ("  (over-subscribed)" if cores > ncpu else ""),
        f"runs          main {counts['main']} + seeds {counts['seeds']}"
        + (f" + exploratory {counts['exploratory']}" if cfg.with_exploratory else "")
        + f" (hgb {counts['hgb']} first), norank {counts['norank']}, union HGB {counts['union_hgb']}"
        + (f" + M11R {counts['union_m11r']}" if cfg.with_m11r_union else ""),
        f"excluded      stage 2 (player layer: {', '.join(STAGE2_ARMS)}): it needs the player dump, which is not rebuilt, and the paper does not use it; no step runs it, the aggregator leaves its rows out",
        f"database      credentials in the environment: {_db_note(steps)}",
        f"estimate      about {wall / 60:.1f} h wall-clock with {cfg.mps_streams} MPS stream(s) (the estimates of the manifest, D.2; 20 % contingency not included)",
    ]
    gate = P.validate_v2_dump(p.dump, deep=False, require_prefix_replay=True)
    lines.append(
        "gate          "
        + (
            "the dump passes the manifest-level checks (the gate also reads the validation splits and runs the leak tests)"
            if not gate["errors"]
            else f"WOULD FAIL now: {len(gate['errors'])} error(s): " + "; ".join(gate["errors"][:3])
        )
    )
    if snapshot is not None:
        where = p.results / SNAPSHOT_DIR_NAME
        if snapshot.get("ok"):
            lines.append(f"code snapshot {where} (written at launch, not by this dry run): git diff HEAD --binary = {snapshot['n_files_changed']} tracked files changed, {_mb(snapshot['diff_bytes'])}; "
                         f"tar of {snapshot['n_untracked']} untracked {'/'.join(SNAPSHOT_EXTENSIONS)} files under {', '.join(SNAPSHOT_ROOTS)} ({_mb(snapshot['untracked_bytes'])}); sha256 of each; "
                         f"{snapshot['n_outside_scope']} other untracked files are not captured")  # fmt: skip
        else:
            lines.append(
                f"code snapshot {where}: UNAVAILABLE now ({snapshot.get('error')}); a launch would refuse to start without --no-code-snapshot"
            )
    if manifest:
        lines.append(f"manifest      resuming run first launched {manifest.get('created_utc')}")
    if prov["git"]["n_status_lines"]:
        lines.append(f"warning       the working tree has {prov['git']['n_status_lines']} uncommitted / untracked entries: the code snapshot saves them with the manifest; a commit (design doc 6.6) "
                     "after the user's approval makes HEAD name the code as well")  # fmt: skip
    lines += [f"ERROR         {e}" for e in errors] + [f"warning       {w}" for w in warnings]
    lines.append("")
    lines.append(
        f"{'step':<28} {'lane':<5} {'state':<8} {'start':>6} {'end':>6}  depends on / command"
    )
    for s in steps:
        st = "done" if state is not None and state.is_done(s) else "todo"
        a, b = sim[s.id]
        cmd = "(in-process) " + s.desc if s.cmd is None else " ".join(s.cmd)
        lines.append(
            f"{s.id:<28} {s.lane:<5} {st:<8} {a:6.0f} {b:6.0f}  after {', '.join(s.deps) or '-'}"
        )
        lines.append(f"{'':<28} {'':<5} {'':<8} {'':>6} {'':>6}  {s.desc}")
        lines.append(
            f"{'':<28} {'':<5} {'':<8} {'':>6} {'':>6}  $ {cmd}"
            + (f"   [cwd {s.cwd}]" if s.cwd != REPO else "")
            + ("   [needs the database]" if s.needs_db else "")
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    cfg = config_from_args(a)
    check_roots(cfg)
    full = build_plan(cfg)
    validate_plan(full)
    steps = select(full, a.only, a.skip)
    validate_plan(steps, universe=full)
    p = cfg.paths
    prov = current_provenance(cfg)
    manifest = None
    try:
        manifest = (
            json.loads(p.run_manifest.read_text(encoding="utf-8"))
            if p.run_manifest.is_file()
            else None
        )
    except (OSError, ValueError):
        manifest = None
    errors, warnings = check_resume(manifest, prov) if manifest and not a.restart else ([], [])
    state = State(p.run_state) if p.run_state.is_file() else None
    if a.dry_run:
        print(format_plan(cfg, steps, None if (a.restart or state is None) else state, prov, manifest=manifest, errors=errors, warnings=warnings,
                          snapshot=None if a.no_code_snapshot else preview_code_snapshot(REPO)))  # fmt: skip
        return 0
    if errors and not a.allow_code_change:
        raise SystemExit(
            "refusing to resume: "
            + "; ".join(errors)
            + "  (--restart to start over, --allow-code-change to accept)"
        )
    if any(s.needs_db for s in steps) and not (
        os.environ.get("PG_PASSWORD") or os.environ.get("PGPASSWORD")
    ):
        raise SystemExit(
            "the database stages need the credentials in the environment: set -a; source <pubg_collector>/.env; set +a"
        )
    p.results.mkdir(parents=True, exist_ok=True)
    if (
        a.no_code_snapshot
    ):  # the code first: a launch that cannot save it changes nothing (state, manifest, roots)
        snap: dict = {"skipped": True, "reason": "--no-code-snapshot"}
        print(
            "warning: no code snapshot (--no-code-snapshot): the manifest does not hold the code that made the results",
            flush=True,
        )
    else:
        try:
            snap = snapshot_launch(REPO, p.results)
        except Exception as exc:
            raise SystemExit(
                f"the code snapshot failed: {exc} (fix that, or pass --no-code-snapshot to run without it)"
            ) from exc
        print(
            f"code snapshot: {snap['dir']} ({'the same code as a saved snapshot' if snap['unchanged'] else 'saved'}; HEAD {snap['head'][:10]}, {snap['git_diff_HEAD']['n_files_changed']} tracked files "
            f"changed, {snap['untracked_tar']['n_files']} untracked files, digest {snap['digest'][:12]})",
            flush=True,
        )
    state = State(p.run_state)
    if a.restart:
        state.clear()
    now = _utc()
    launch = {
        "start_utc": now,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "steps": [s.id for s in steps],
        "provenance_changes": errors + warnings,
        "code_snapshot": snap,
    }
    doc = {"created_utc": (manifest or {}).get("created_utc", now), "updated_utc": now, "provenance": (manifest or {}).get("provenance", prov) if not a.restart else prov,
           "current": prov, "code_snapshot_dir": str(p.results / SNAPSHOT_DIR_NAME), "roots": p.as_dict(), "budget": {"cpu_workers": cfg.cpu_workers, "cpu_threads": cfg.cpu_threads, "mps_streams": cfg.mps_streams, "mps_threads": cfg.mps_threads},
           "launches": [*((manifest or {}).get("launches", []) if not a.restart else []), launch]}  # fmt: skip
    write_json_atomic(p.run_manifest, doc)
    for w in warnings:
        print(f"warning: {w}", flush=True)
    for root in (
        p.runs,
        p.krafton_data,
        p.kf,
        p.sagong,
        p.results,
    ):  # stamp the roots (v2 dump): every tool checks them again
        P.claim_root(root, p.dump, role=root.name)
    (p.krafton_data / "logs").mkdir(
        parents=True, exist_ok=True
    )  # adaptation.extract opens <out>/logs/extract_<t>.jsonl without creating it
    runner = Runner(
        steps,
        state,
        p.logs,
        stop_on_failure=a.stop_on_failure,
        universe=full,
        log=lambda s: print(f"{datetime.now().strftime('%H:%M:%S')} {s}", flush=True),
    )

    def _terminate(
        signum, frame
    ):  # SIGTERM = Ctrl-C: the runner stops its children and records the steps as interrupted
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)
    rc = runner.run()
    doc["launches"][-1]["end_utc"] = _utc()
    doc["launches"][-1]["exit_code"] = rc
    doc["launches"][-1]["status"] = {s.id: runner.status.get(s.id, "not started") for s in steps}
    write_json_atomic(p.run_manifest, doc)
    print(
        f"finished with exit code {rc}; state {p.run_state}, manifest {p.run_manifest}, logs {p.logs}"
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
