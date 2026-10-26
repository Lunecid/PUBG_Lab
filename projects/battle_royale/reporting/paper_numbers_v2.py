#!/usr/bin/env python3
"""Recompute every dump-dependent number of the BigComp 2027 paper drafts from the v2 dump and the database (manifest section E).

    python tools/paper_numbers_v2.py [--dump-dir model_stgcn/dumps/bigcomp2027_phase_v2] [--runs-dir ...] [--results-dir reports/bigcomp2027_results_v2]
        [--out reports/bigcomp2027_results_v2/paper_numbers.json] [--stages population,risk_set,...] [--no-db]

The database is read-only (``set -a; source <pubg_collector>/.env; set +a`` first; the credentials are only read from the environment and never
printed or written).  Nothing is copied from v1: a number that cannot be recomputed is listed under ``not_computed`` with the reason.

Output: one JSON file ``{"meta": ..., "numbers": {key: {"value", "unit", "provenance"}}, "not_computed": [...], "errors": [...]}``.  Every number carries its
provenance: the stage, the source (``dump`` | ``db`` | ``runs`` | ``results``), the definition, the inputs (files, tables, the SQL), the manifest reference
(``E.1 L14`` = section E.1, line 14 of ``BIGCOMP2027_SEC3_DATA_KO_v1``) and the sample (seed, size) where there is one.

Stages (manifest section E; default: all, the database stages only when the environment has credentials and ``--no-db`` is not given)::

    population    matches, cohorts, split and match-type sizes, cohort size range, the two largest cohorts       population.json          E.1 L14, L71, L77, E.2 L158
    risk_set      risk-set rows, positives, rate, hazard by cut, comeback rows, excluded cuts, feature counts    the dump records         E.1 L78, E.2 L47
    correlations  Spearman of required speed / migration deficit with the outside depth (pooled train rows)       hgb.build_rows           E.2 L42
    db_counts     records per telemetry table of the benchmark matches (Table I) and the match date range        database                 E.1 L15, L26-37
    intervals     position sampling interval statistics (sample)                                                  database                 E.1 L9
    clock         offset of the elapsed_time FIELD (game clock) from LogMatchStart: a description only (v2 is on W)  database                 E.1 L49
    coverage      share of alive players with a position sample within 15 s before the cut (sample)              database + dump cuts     E.1 L52
    lookahead     the prefix-replay no-look-ahead check on 400 random matches and a_k = b_k - 10 s                phase_v2.replay          E.1 L50, L51
    label_audit   agreement of the dump labels with an independent elimination time from kills / participants   database + dump labels   E.4 (4)
    headline      the headline numbers of the RQ blueprint, read from the aggregator outputs                      results                  E.3

Test on v1: ``--dump-dir model_stgcn/dumps/bigcomp2027_phase_v1 --allow-v1`` (the version guard refuses a v1 dump otherwise); the population, risk set,
correlation and interval cores reproduce the documented v1 values (``tests/test_paper_numbers_v2.py``).
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

REPO = REPO_ROOT
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pubg_lab.survival.ptr.data.paths as P  # noqa: E402
from pubg_lab.survival.ptr.common.util import git_info, to_jsonable  # noqa: E402

SPLITS = P.SPLITS
STAGES: tuple[str, ...] = (
    "population",
    "risk_set",
    "correlations",
    "db_counts",
    "intervals",
    "clock",
    "coverage",
    "lookahead",
    "label_audit",
    "headline",
)
DB_STAGES = frozenset({"db_counts", "intervals", "clock", "coverage", "lookahead", "label_audit"})
CUTS = 7  # landmark cuts k = 1..7 on the 8-slot axis
TELEMETRY_TABLES: tuple[tuple[str, str, str], ...] = (  # (key, table, Table I row of SEC3)
    ("positions", "telem_positions", "positions"), ("damage", "telem_damage", "damage"), ("item_pickup", "telem_item_pickup", "item pick-up and drop"),
    ("item_equip", "telem_item_equip", "equip"), ("item_use", "telem_item_use", "use"), ("game_states", "telem_game_states", "zone state"),
    ("parachute_landing", "telem_parachute_landing", "landing"), ("kills", "telem_kills", "kills"), ("groggy", "telem_groggy", "knocks"),
    ("phase_changes", "telem_phase_changes", "phase changes"),
)  # fmt: skip
CORR_COLUMNS = (
    "hab_required_speed_to_safe_mps",
    "hab_signed_boundary_distance_end_m",
    "hab_migration_deficit_mps",
)


# =========================================================================== the collector
class Numbers:
    """The numbers with their provenance; ``skip`` records what could not be computed and why."""

    def __init__(self) -> None:
        self.numbers: dict[str, dict] = {}
        self.not_computed: list[dict] = []
        self.errors: list[dict] = []

    def add(self, key: str, value: Any, *, stage: str, source: str, definition: str, inputs: Iterable = (), ref: str | None = None, unit: str | None = None,
            **extra: Any) -> None:  # fmt: skip
        if key in self.numbers:
            raise KeyError(f"number {key!r} recorded twice")
        prov = {
            "stage": stage,
            "source": source,
            "definition": definition,
            "inputs": [str(i) for i in inputs],
            "manifest": ref,
        }
        prov.update({k: to_jsonable(v) for k, v in extra.items()})
        self.numbers[key] = {"value": to_jsonable(value), "unit": unit, "provenance": prov}

    def skip(self, key: str, reason: str, *, stage: str, ref: str | None = None) -> None:
        self.not_computed.append({"key": key, "stage": stage, "reason": reason, "manifest": ref})

    def error(self, stage: str, exc: BaseException) -> None:
        self.errors.append({"stage": stage, "error": f"{type(exc).__name__}: {exc}"})

    def get(self, key: str) -> Any:
        return self.numbers[key]["value"]


# =========================================================================== context: the dump, the sample, the connection
class Ctx:
    """What the stages share: the dump, the roots, the sample size and seed, a lazily loaded match-id table, a connection factory."""

    def __init__(self, dump_dir: str | Path, *, runs_dir: str | Path | None = None, results_dir: str | Path | None = None, seed: int = 42,
                 per_cohort: int = 250, audit_per_cohort: int = 0, lookahead_n: int = 400, limit_matches: int | None = None, workers: int = 2,
                 connect: Callable[[], Any] | None = None, cohorts: Sequence[str] | None = None) -> None:  # fmt: skip
        self.dump_dir = Path(dump_dir)
        self.runs_dir = None if runs_dir is None else Path(runs_dir)
        self.results_dir = None if results_dir is None else Path(results_dir)
        self.seed, self.per_cohort, self.audit_per_cohort, self.lookahead_n = (
            int(seed),
            int(per_cohort),
            int(audit_per_cohort),
            int(lookahead_n),
        )
        self.limit_matches, self.workers = limit_matches, max(1, int(workers))
        self._connect = connect or connect_readonly
        self._cohorts = list(cohorts) if cohorts is not None else None
        self._ids: dict[str, dict[str, list[str]]] | None = None
        self._local = threading.local()

    def cohorts(self) -> list[str]:
        if self._cohorts is None:
            from pubg_lab.survival.ptr.data.cohorts import discover_cohorts

            self._cohorts = discover_cohorts(self.dump_dir)
        return list(self._cohorts)

    def load(self, cohort: str, split: str) -> dict:
        import torch

        return torch.load(
            self.dump_dir / cohort / f"{split}.pt", map_location="cpu", weights_only=False
        )

    def match_ids(self) -> dict[str, dict[str, list[str]]]:
        """``{cohort: {split: [match_id]}}`` of the dumped matches (loaded once)."""
        if self._ids is None:
            self._ids = {
                c: {s: [str(m["match_id"]) for m in self.load(c, s)["matches"]] for s in SPLITS}
                for c in self.cohorts()
            }
        return self._ids

    def all_ids(self) -> list[str]:
        out = sorted({i for per in self.match_ids().values() for ids in per.values() for i in ids})
        return out[: self.limit_matches] if self.limit_matches else out

    def sample(self, per_cohort: int) -> list[tuple[str, str, str]]:
        """``[(cohort, split, match_id)]``: ``per_cohort`` random matches of every cohort (``RandomState(seed)``, cohorts in sorted order); ``per_cohort <= 0`` = every
        match of the cohort (no sampling: the numbers are those of the whole corpus)."""
        rng = np.random.RandomState(self.seed)
        out: list[tuple[str, str, str]] = []
        for c in sorted(self.match_ids()):
            pairs = sorted((i, s) for s, ids in self.match_ids()[c].items() for i in ids)
            n = len(pairs) if per_cohort <= 0 else min(per_cohort, len(pairs))
            if self.limit_matches:
                n = min(n, self.limit_matches)
            pick = (
                rng.choice(len(pairs), size=n, replace=False)
                if n < len(pairs)
                else np.arange(len(pairs))
            )
            out += [(c, pairs[j][1], pairs[j][0]) for j in sorted(pick)]
        return out

    def conn(self):
        """One read-only connection per thread."""
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._local.conn = self._connect()
        return c


def connect_readonly(timeout_s: int = 900):
    """A read-only PostgreSQL session; the credentials come from the environment (``PG_HOST`` / ``PG_PORT`` / ``PG_DATABASE`` / ``PG_USER`` / ``PG_PASSWORD``)."""
    import psycopg2

    g = lambda *names, default=None: next(
        (os.environ[n] for n in names if os.environ.get(n)), default
    )  # noqa: E731
    conn = psycopg2.connect(host=g("PG_HOST", "PGHOST", default="localhost"), port=int(g("PG_PORT", "PGPORT", default="5432")),
                            dbname=g("PG_DBNAME", "PG_DATABASE", "PGDATABASE", default="pubg_survival"), user=g("PG_USER", "PGUSER", default="postgres"),
                            password=g("PG_PASSWORD", "PGPASSWORD"))  # fmt: skip
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = '{int(timeout_s)}s'")
    return conn


def db_available() -> bool:
    if not os.environ.get("PG_PASSWORD") and not os.environ.get("PGPASSWORD"):
        return False
    try:
        c = connect_readonly(30)
        with c.cursor() as cur:
            cur.execute("SELECT 1")
        c.close()
        return True
    except Exception:
        return False


# =========================================================================== population (population.json)
def population_numbers(pop: dict) -> dict:
    """Pure core: sizes from ``population.json`` (``after`` counts of the cohorts with status ``kept``)."""
    cohorts = pop.get("cohorts", {})
    kept = {c: i for c, i in cohorts.items() if i.get("status") == "kept"}
    by_split = {s: int(sum(i["after"][s] for i in kept.values())) for s in SPLITS}
    types: Counter = Counter()
    for i in kept.values():
        types.update({k: int(v) for k, v in (i.get("match_type_after") or {}).items()})
    size = {c: int(i["after"]["total"]) for c, i in kept.items()}
    ranked = sorted(size, key=lambda c: (-size[c], c))
    return {
        "n_matches": int(sum(size.values())), "n_cohorts": len(kept), "by_split": by_split, "match_types": dict(types), "cohort_sizes": size,
        "size_min": min(size.values()) if size else None, "size_max": max(size.values()) if size else None, "smallest": ranked[-1] if ranked else None,
        "two_largest": ranked[:2], "excluded": {c: i.get("reason", "") for c, i in cohorts.items() if i.get("status") != "kept"},
        "before_total": int(sum(i["before"]["total"] for i in cohorts.values() if "before" in i)), "rules": pop.get("rules", {}),
    }  # fmt: skip


def stage_population(ctx: Ctx, nums: Numbers) -> None:
    path = ctx.dump_dir / "population.json"
    n = population_numbers(json.loads(path.read_text(encoding="utf-8")))
    kw = dict(stage="population", source="dump", inputs=[path])
    nums.add(
        "population.n_matches",
        n["n_matches"],
        definition="sum of after.total over the cohorts with status 'kept'",
        ref="E.1 L14",
        unit="matches",
        **kw,
    )
    nums.add(
        "population.n_cohorts",
        n["n_cohorts"],
        definition="cohorts with status 'kept'",
        ref="E.1 L14",
        **kw,
    )
    for s in SPLITS:
        nums.add(
            f"population.matches_{s}",
            n["by_split"][s],
            definition=f"sum of after.{s} over the kept cohorts (the stored 80/10/10 split)",
            ref="E.1 L77",
            unit="matches",
            **kw,
        )
    for t, v in sorted(n["match_types"].items()):
        nums.add(
            f"population.matches_{t}",
            v,
            definition=f"sum of match_type_after['{t}'] over the kept cohorts",
            ref="E.1 L71",
            unit="matches",
            **kw,
        )
    nums.add(
        "population.cohort_sizes",
        n["cohort_sizes"],
        definition="after.total per kept cohort",
        ref="E.1 L77",
        unit="matches",
        **kw,
    )
    nums.add(
        "population.cohort_size_min",
        n["size_min"],
        definition=f"smallest kept cohort ({n['smallest']})",
        ref="E.1 L77",
        unit="matches",
        **kw,
    )
    nums.add(
        "population.cohort_size_max",
        n["size_max"],
        definition="largest kept cohort",
        ref="E.1 L77",
        unit="matches",
        **kw,
    )
    nums.add(
        "population.two_largest_cohorts",
        n["two_largest"],
        definition="the two kept cohorts with the most matches: the seeds 13 / 2026 and the learning curve run on them",
        ref="E.2 L158-167",
        **kw,
    )
    nums.add(
        "population.matches_before_exclusion",
        n["before_total"],
        definition="sum of before.total over every cohort of the report (before the airoyale / small-cohort rules)",
        ref="E.1 L70",
        unit="matches",
        **kw,
    )
    nums.add(
        "population.excluded_cohorts",
        n["excluded"],
        definition="cohorts that population.json does not keep, with the reason",
        ref="E.1 L70",
        **kw,
    )
    nums.add(
        "population.rules",
        n["rules"],
        definition="the population rules of the dump (exclude_match_types, min_matches_per_cohort, split assignment)",
        ref="E.1 L70",
        **kw,
    )


# =========================================================================== risk set (the dump records)
def risk_set_counts(matches: Sequence[dict], select: Callable | None = None) -> dict:
    """Pure core over loaded matches: loader risk-set rows (``select_labels``) and positives per cut slot, the raw stored rows, comeback rows
    (risk-set rows without a key token, v2 records only), excluded (match, cut) pairs and the winner count."""
    import torch

    if select is None:
        from pubg_lab.survival.ptr.data.data import (
            select_labels as select,
        )  # the definition every arm trains and is scored on
    rows = np.zeros(CUTS, np.int64)
    pos = np.zeros(CUTS, np.int64)
    raw_rows = np.zeros(CUTS, np.int64)
    raw_pos = np.zeros(CUTS, np.int64)
    comeback = np.zeros(CUTS, np.int64)
    excluded = Counter()
    n_win = n_excl_matches = 0
    has_query = False
    max_delta = 0.0
    n_delta = 0
    for m in matches:
        y, target = select(m, "v2")
        t = target.numpy()[:CUTS]
        yy = (y.numpy()[:CUTS] > 0.5) & t
        rows += t.sum(axis=1)
        pos += yy.sum(axis=1)
        rt = torch.as_tensor(m["target_mask_v2"]).bool().numpy()[:CUTS]
        ry = (torch.as_tensor(m["y_hazard_v2"]).numpy()[:CUTS] > 0.5) & rt
        raw_rows += rt.sum(axis=1)
        raw_pos += ry.sum(axis=1)
        if "query_mask" in m and "node_mask" in m:
            has_query = True
            comeback += (rt & ~torch.as_tensor(m["node_mask"]).bool().numpy()[:CUTS]).sum(axis=1)
        for k, why in (m.get("cut_excluded") or {}).items():
            excluded[str(why)] += 1
        n_excl_matches += int(bool(m.get("cut_excluded")))
        for key in ("clock_delta_s", "clock_delta_median_s"):
            if key in m:
                n_delta += 1
                max_delta = max(
                    max_delta,
                    abs(float(torch.as_tensor(m[key]).double().abs().max()))
                    if torch.as_tensor(m[key]).numel()
                    else 0.0,
                )
        if "team_won" in m:
            n_win += int(torch.as_tensor(m["team_won"]).bool().sum())
    return {
        "n_matches": len(matches), "rows_by_cut": rows.tolist(), "positives_by_cut": pos.tolist(), "raw_rows_by_cut": raw_rows.tolist(),
        "raw_positives_by_cut": raw_pos.tolist(), "comeback_rows_by_cut": comeback.tolist() if has_query else None, "excluded_match_cuts": dict(excluded),
        "matches_with_excluded_cut": n_excl_matches, "winners": n_win, "max_abs_clock_delta_s": max_delta, "n_clock_delta_fields": n_delta,
    }  # fmt: skip


def _sum_counts(a: dict | None, b: dict) -> dict:
    if a is None:
        return {
            k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v))
            for k, v in b.items()
        }
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, list):
            out[k] = None if (v is None or a[k] is None) else [x + y for x, y in zip(a[k], v)]
        elif isinstance(v, dict):
            c = Counter(a[k])
            c.update(v)
            out[k] = dict(c)
        elif v is None:
            out[k] = None if a[k] is None else a[k]
        elif k == "max_abs_clock_delta_s":
            out[k] = max(a[k], v)
        else:
            out[k] = a[k] + v
    return out


def stage_risk_set(ctx: Ctx, nums: Numbers) -> None:
    per_cohort: dict[str, dict] = {}
    per_split: dict[str, dict | None] = {s: None for s in SPLITS}
    total: dict | None = None
    meta0: dict | None = None
    stats_check: list[str] = []
    for c in ctx.cohorts():
        cohort_total = None
        for s in SPLITS:
            d = ctx.load(c, s)
            matches = d["matches"][: ctx.limit_matches] if ctx.limit_matches else d["matches"]
            cnt = risk_set_counts(matches)
            cohort_total = _sum_counts(cohort_total, cnt)
            per_split[s] = _sum_counts(per_split[s], cnt)
            if meta0 is None and s == "val":
                meta0 = dict(d["meta"])
            st = d["meta"].get("stats") or {}
            if (
                not ctx.limit_matches
                and "rows_by_cut" in st
                and list(st["rows_by_cut"]) != cnt["raw_rows_by_cut"]
            ):
                stats_check.append(
                    f"{c}/{s}: meta stats rows_by_cut {st['rows_by_cut']} != recount {cnt['raw_rows_by_cut']}"
                )
            del d
        per_cohort[c] = cohort_total
        total = _sum_counts(total, cohort_total)
    assert total is not None, "no cohort"
    kw = dict(stage="risk_set", source="dump", inputs=[ctx.dump_dir / "<cohort>" / "<split>.pt"])
    rows, pos = sum(total["rows_by_cut"]), sum(total["positives_by_cut"])
    d_rows = "risk-set rows = sum over the cut slots 1..7 of model_stgcn.ptr.data.select_labels(match, 'v2')[1] (the rows every arm is trained and scored on)"
    nums.add("risk_set.n_rows", rows, definition=d_rows, ref="E.1 L78", unit="rows", **kw)
    nums.add(
        "risk_set.n_positives",
        pos,
        definition="rows with y = 1 (eliminated before the next cut) among the risk-set rows",
        ref="E.1 L78",
        unit="rows",
        **kw,
    )
    nums.add(
        "risk_set.positive_rate",
        pos / rows if rows else None,
        definition="n_positives / n_rows",
        ref="E.1 L78",
        unit="share",
        **kw,
    )
    nums.add(
        "risk_set.rows_by_cut",
        total["rows_by_cut"],
        definition=d_rows + ", per cut k = 1..7",
        ref="E.1 L56-60",
        **kw,
    )
    nums.add(
        "risk_set.positives_by_cut",
        total["positives_by_cut"],
        definition="positives per cut k = 1..7",
        ref="E.1 L56-60",
        **kw,
    )
    nums.add(
        "risk_set.hazard_by_cut",
        [p / r if r else None for p, r in zip(total["positives_by_cut"], total["rows_by_cut"])],
        definition="positives / rows per cut k = 1..7, pooled over the cohorts (the v1 dump gives 0.195, 0.193, 0.248, 0.350, 0.375, 0.502, 0.547; the 0.208 ... 0.557 of design doc 4.2 is a "
        "corrected-label figure of the audit, not a ratio of this dump)",
        ref="Appendix gate 3",
        unit="share",
        **kw,
    )
    for s in SPLITS:
        nums.add(
            f"risk_set.n_rows_{s}",
            sum(per_split[s]["rows_by_cut"]),
            definition=d_rows + f", split {s}",
            ref="E.1 L78",
            unit="rows",
            **kw,
        )
        nums.add(
            f"risk_set.n_positives_{s}",
            sum(per_split[s]["positives_by_cut"]),
            definition=f"positives, split {s}",
            ref="E.1 L78",
            unit="rows",
            **kw,
        )
    nums.add(
        "risk_set.n_rows_by_cohort",
        {c: sum(v["rows_by_cut"]) for c, v in per_cohort.items()},
        definition=d_rows + ", per cohort",
        ref="E.1 L78",
        unit="rows",
        **kw,
    )
    nums.add(
        "risk_set.n_positives_by_cohort",
        {c: sum(v["positives_by_cut"]) for c, v in per_cohort.items()},
        definition="positives per cohort",
        ref="E.1 L78",
        unit="rows",
        **kw,
    )
    nums.add(
        "risk_set.n_matches_dumped",
        total["n_matches"],
        definition="matches in the dump files (train + val + test)",
        ref="E.1 L14",
        unit="matches",
        **kw,
    )
    nums.add(
        "risk_set.raw_rows",
        sum(total["raw_rows_by_cut"]),
        definition="sum of the stored target_mask_v2 (before the loader's winner override); equal to n_rows when the dump pre-applies winner censoring",
        ref="E.1 L78",
        unit="rows",
        **kw,
    )
    nums.add(
        "risk_set.raw_equals_loader",
        total["raw_rows_by_cut"] == total["rows_by_cut"]
        and total["raw_positives_by_cut"] == total["positives_by_cut"],
        definition="the stored risk set and positives equal what select_labels returns, per cut (False: the loader adds / removes rows)",
        ref="E.1 L78",
        **kw,
    )
    if total["comeback_rows_by_cut"] is not None:
        nums.add(
            "risk_set.comeback_rows",
            sum(total["comeback_rows_by_cut"]),
            definition="risk-set rows whose team has no key token (target_mask_v2 and not node_mask): causally dead at the cut, back later",
            ref="B.0 8",
            unit="rows",
            by_cut=total["comeback_rows_by_cut"],
            **kw,
        )
    else:
        nums.skip(
            "risk_set.comeback_rows",
            "records carry no query_mask (a v1 dump)",
            stage="risk_set",
            ref="B.0 8",
        )
    nums.add(
        "risk_set.excluded_match_cuts",
        total["excluded_match_cuts"],
        definition="(match, cut) pairs excluded for a truncated log, by reason (cut_excluded of the records)",
        ref="E.1 L56-60",
        **kw,
    )
    nums.add(
        "risk_set.winners",
        total["winners"],
        definition="teams with team_won over every dumped match (one per match expected)",
        ref="E.1 L78",
        **kw,
    )
    nums.add(
        "risk_set.meta_stats_agree",
        not stats_check,
        definition="rows_by_cut of every split's meta['stats'] equals the recount of target_mask_v2"
        if not ctx.limit_matches
        else "not checked with --limit-matches",
        ref="E.1 L78",
        mismatches=stats_check[:10],
        **kw,
    )
    if total["n_clock_delta_fields"]:
        nums.add(
            "clock.max_abs_clock_delta_s",
            total["max_abs_clock_delta_s"],
            definition="largest |clock_delta_s| / |clock_delta_median_s| over every dumped match: 0 = no game-clock offset alignment is applied (v2: every table is on W)",
            ref="E.1 L49",
            unit="s",
            **kw,
        )
    if meta0 is not None:
        clock_meta = meta0.get("clock") or {}
        nums.add(
            "clock.v2_definition",
            clock_meta.get("W") or "not recorded in the meta (a v1 dump has two clocks)",
            definition="meta['clock']['W'] of the dump: the clock every table of the v2 inputs is on",
            ref="E.1 L49",
            stage="risk_set",
            source="dump",
            inputs=[ctx.dump_dir / ctx.cohorts()[0] / "val.pt"],
            single_clock=meta0.get("single_clock"),
        )
    _check_run_summary(ctx, nums, total, per_cohort)
    if meta0 is not None:
        groups = {g: len(cols) for g, cols in (meta0.get("feature_groups") or {}).items()}
        nums.add(
            "features.n_features",
            int(meta0.get("n_features", len(meta0.get("feature_names", [])))),
            definition="meta['n_features'] of the validation split",
            ref="E.2 L47-48",
            stage="risk_set",
            source="dump",
            inputs=[ctx.dump_dir / ctx.cohorts()[0] / "val.pt"],
            n_feature_names=len(meta0.get("feature_names", [])),
        )
        nums.add(
            "features.group_sizes",
            groups,
            definition="columns per meta['feature_groups'] entry (the 'nine families' of the text are a hand-made mapping of these groups: not stored in the dump)",
            ref="E.2 L47-48",
            stage="risk_set",
            source="dump",
            inputs=[ctx.dump_dir / ctx.cohorts()[0] / "val.pt"],
        )
        nums.add(
            "features.version",
            meta0.get("version", "v1"),
            definition="meta['version'] of the dump (absent = the legacy v1 format)",
            ref="B.0",
            stage="risk_set",
            source="dump",
            inputs=[ctx.dump_dir / ctx.cohorts()[0] / "val.pt"],
        )
    if ctx.runs_dir is not None:
        _check_against_runs(ctx, nums, per_cohort)


def _check_run_summary(ctx: Ctx, nums: Numbers, total: dict, per_cohort: dict[str, dict]) -> None:
    """Cross-check with ``<dump>/run_summary.json`` (the dump builder's own totals): rows, positives and their per-cut vectors, comeback rows, per cohort rows."""
    path = ctx.dump_dir / "run_summary.json"
    if ctx.limit_matches or not path.is_file():
        return
    try:
        rs = json.loads(path.read_text(encoding="utf-8"))
        bad = []
        if int(rs["rows"]) != sum(total["raw_rows_by_cut"]):
            bad.append(f"rows {rs['rows']} vs {sum(total['raw_rows_by_cut'])}")
        if int(rs["positives"]) != sum(total["raw_positives_by_cut"]):
            bad.append(f"positives {rs['positives']} vs {sum(total['raw_positives_by_cut'])}")
        if (
            list(rs["rows_by_cut"]) != total["raw_rows_by_cut"]
            or list(rs["positives_by_cut"]) != total["raw_positives_by_cut"]
        ):
            bad.append("per-cut rows / positives differ")
        for c, v in (rs.get("per_cohort") or {}).items():
            if c in per_cohort and int(v["rows"]) != sum(per_cohort[c]["raw_rows_by_cut"]):
                bad.append(f"{c}: rows {v['rows']} vs {sum(per_cohort[c]['raw_rows_by_cut'])}")
        if (
            total["comeback_rows_by_cut"] is not None
            and list(rs["comeback_rows_by_cut"]) != total["comeback_rows_by_cut"]
        ):
            bad.append("comeback rows by cut differ")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        nums.skip(
            "risk_set.matches_run_summary",
            f"run_summary.json unreadable: {exc}",
            stage="risk_set",
            ref="E.1 L78",
        )
        return
    nums.add(
        "risk_set.matches_run_summary",
        not bad,
        definition="the recount of the stored risk set equals run_summary.json of the dump builder (rows, positives, per cut, per cohort, comeback rows)",
        ref="E.1 L78",
        stage="risk_set",
        source="dump",
        inputs=[path],
        mismatches=bad,
    )


def _check_against_runs(ctx: Ctx, nums: Numbers, per_cohort: dict[str, dict]) -> None:
    """Cross-check: the risk-set rows of every cohort equal ``data.n_target_rows`` of the M00 runs (train + val + test)."""
    bad, seen = [], 0
    for c, cnt in per_cohort.items():
        p = ctx.runs_dir / c / "M00" / "seed42" / "metrics.json"
        if not p.is_file():
            continue
        try:
            n = json.loads(p.read_text(encoding="utf-8"))["data"]["n_target_rows"]
        except (OSError, ValueError, KeyError):
            continue
        seen += 1
        if int(sum(n.values())) != int(sum(cnt["rows_by_cut"])):
            bad.append(f"{c}: M00 metrics.json {sum(n.values())} vs dump {sum(cnt['rows_by_cut'])}")
    nums.add(
        "risk_set.matches_m00_metrics",
        None if not seen or ctx.limit_matches else not bad,
        definition="the dump's risk-set rows per cohort equal data.n_target_rows of the M00 run (train + val + test)",
        ref="E.1 L78",
        stage="risk_set",
        source="runs",
        inputs=[ctx.runs_dir / "<cohort>" / "M00" / "seed42" / "metrics.json"],
        cohorts_compared=seen,
        mismatches=bad[:10],
    )


# =========================================================================== correlations
def spearman_pairs(cols: dict[str, np.ndarray]) -> dict:
    """Pure core: Spearman (and Pearson) of the required speed and of the migration deficit with the outside depth ``max(0, -d)`` over pooled rows."""
    from scipy import stats

    req, d, deficit = (np.asarray(cols[c], dtype=np.float64) for c in CORR_COLUMNS)
    depth = np.maximum(0.0, -d)
    ok = np.isfinite(req) & np.isfinite(depth) & np.isfinite(deficit)
    req, depth, deficit = req[ok], depth[ok], deficit[ok]
    return {
        "n_rows": int(ok.sum()),
        "spearman_required_speed_outside_depth": float(stats.spearmanr(req, depth)[0]),
        "spearman_migration_deficit_outside_depth": float(stats.spearmanr(deficit, depth)[0]),
        "spearman_required_speed_migration_deficit": float(stats.spearmanr(req, deficit)[0]),
        "pearson_required_speed_outside_depth": float(np.corrcoef(req, depth)[0, 1]),
        "pearson_migration_deficit_outside_depth": float(np.corrcoef(deficit, depth)[0, 1]),
    }  # fmt: skip


def stage_correlations(ctx: Ctx, nums: Numbers) -> None:
    import pubg_lab.survival.ptr.models.hgb as H
    from pubg_lab.survival.ptr.data.data import load_dump

    parts = []
    for c in ctx.cohorts():
        dump = load_dump(ctx.dump_dir / c / "train.pt", "v2")
        if ctx.limit_matches:
            dump = {**dump, "matches": dump["matches"][: ctx.limit_matches]}
        rows = H.build_rows(dump, "HGB-cut", "v2")
        names = list(rows.feature_names)
        parts.append(np.asarray(rows.X)[:, [names.index(n) for n in CORR_COLUMNS]])
    X = np.concatenate(parts)
    res = spearman_pairs({n: X[:, i] for i, n in enumerate(CORR_COLUMNS)})
    kw = dict(stage="correlations", source="dump", ref="E.2 L42", inputs=[ctx.dump_dir / "<cohort>" / "train.pt"],
              definition_rows="the pooled TRAIN rows of model_stgcn.ptr.hgb.build_rows(train, 'HGB-cut', 'v2') over every cohort")  # fmt: skip
    nums.add(
        "correlations.n_rows",
        res["n_rows"],
        definition="train rows entering the correlations (finite values)",
        unit="rows",
        **kw,
    )
    nums.add(
        "correlations.spearman_required_speed_outside_depth",
        res["spearman_required_speed_outside_depth"],
        definition="Spearman of hab_required_speed_to_safe_mps with max(0, -hab_signed_boundary_distance_end_m) (the text says 0.79; v1 gives 0.795)",
        unit="rho",
        **kw,
    )
    nums.add(
        "correlations.spearman_migration_deficit_outside_depth",
        res["spearman_migration_deficit_outside_depth"],
        definition="Spearman of hab_migration_deficit_mps with max(0, -hab_signed_boundary_distance_end_m) (the text says 0.41; v1 gives 0.386: no producer of 0.41 exists)",
        unit="rho",
        **kw,
    )
    nums.add(
        "correlations.spearman_required_speed_migration_deficit",
        res["spearman_required_speed_migration_deficit"],
        definition="Spearman of the required speed with the migration deficit",
        unit="rho",
        **kw,
    )
    nums.add(
        "correlations.pearson_required_speed_outside_depth",
        res["pearson_required_speed_outside_depth"],
        definition="Pearson of the same pair (ecomodel's collinearity table reports 0.90 per cohort)",
        unit="r",
        **kw,
    )
    nums.add(
        "correlations.pearson_migration_deficit_outside_depth",
        res["pearson_migration_deficit_outside_depth"],
        definition="Pearson of the same pair",
        unit="r",
        **kw,
    )


# =========================================================================== database stages
def _batches(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def stage_db_counts(ctx: Ctx, nums: Numbers) -> None:
    ids = ctx.all_ids()
    counts: dict[str, int] = {}
    cur = ctx.conn().cursor()
    for key, table, _row in TELEMETRY_TABLES:
        counts[key] = sum(
            _scalar(cur, f"SELECT count(*) FROM pubg.{table} WHERE match_id = ANY(%s)", (b,))
            for b in _batches(ids, 1000)
        )
    cur.execute(
        "SELECT min(created_at), max(created_at), count(*) FROM pubg.matches WHERE match_id = ANY(%s)",
        (ids,),
    )
    first, last, n_found = cur.fetchone()
    kw = dict(
        stage="db_counts",
        source="db",
        ref="E.1 L15, L26-37",
        n_match_ids=len(ids),
        inputs=["pubg.<table>", ctx.dump_dir],
    )
    for key, table, row in TELEMETRY_TABLES:
        nums.add(
            f"records.{key}",
            counts[key],
            definition=f"SELECT count(*) FROM pubg.{table} WHERE match_id = ANY(<the dumped matches>) (Table I: {row})",
            unit="records",
            sql=f"pubg.{table}",
            **kw,
        )
    total = sum(counts.values())
    nums.add(
        "records.total", total, definition="sum of the ten tables of Table I", unit="records", **kw
    )
    nums.add(
        "records.events",
        total - counts["positions"],
        definition="total minus positions ('225.97 M events')",
        unit="records",
        **kw,
    )
    nums.add(
        "records.positions_share",
        counts["positions"] / total if total else None,
        definition="positions / total",
        unit="share",
        **kw,
    )
    nums.add(
        "matches.created_first",
        first.date().isoformat() if first else None,
        definition="min(pubg.matches.created_at) over the dumped matches",
        ref="E.1 L15",
        stage="db_counts",
        source="db",
        inputs=["pubg.matches"],
    )
    nums.add(
        "matches.created_last",
        last.date().isoformat() if last else None,
        definition="max(pubg.matches.created_at) over the dumped matches",
        ref="E.1 L15",
        stage="db_counts",
        source="db",
        inputs=["pubg.matches"],
    )
    nums.add(
        "matches.found_in_db",
        int(n_found),
        definition="dumped match ids found in pubg.matches (equals population.n_matches unless the id list was limited)",
        ref="E.1 L14",
        stage="db_counts",
        source="db",
        inputs=["pubg.matches"],
    )


def _scalar(cur, sql: str, params) -> int:
    cur.execute(sql, params)
    return int(cur.fetchone()[0])


INTERVAL_SQL = """
WITH p AS (
  SELECT match_id, account_id, elapsed_time AS t FROM pubg.telem_positions
   WHERE match_id = ANY(%s) AND is_game >= 0.1 AND account_id IS NOT NULL AND team_id IS NOT NULL
     AND pos_x IS NOT NULL AND pos_y IS NOT NULL AND elapsed_time IS NOT NULL),
d AS (SELECT t - lag(t) OVER (PARTITION BY match_id, account_id ORDER BY t) AS dt FROM p)
SELECT least(greatest(dt, 0), 120)::int AS b, count(*)::bigint FROM d WHERE dt IS NOT NULL GROUP BY 1
"""


def interval_stats(hist: np.ndarray) -> dict:
    """Pure core (``03_summarize_sample.py``): statistics of the histogram of consecutive position intervals, bin ``i`` = floor(interval) = i seconds (clipped at 120)."""
    h = np.asarray(hist, dtype=np.float64)
    tot = float(h.sum())
    if tot == 0:
        return {"n_intervals": 0}
    cdf = h.cumsum() / tot
    q = lambda p: int(np.searchsorted(cdf, p))  # noqa: E731
    return {
        "n_intervals": int(tot), "mean_s": float((np.arange(len(h)) * h).sum() / tot), "median_s": q(0.5), "p10_s": q(0.1), "p90_s": q(0.9), "p99_s": q(0.99),
        "share_le11": float(h[:12].sum() / tot), "share_eq10": float(h[10] / tot), "share_gt20": float(h[21:].sum() / tot), "share_gt60": float(h[61:].sum() / tot),
    }  # fmt: skip


def stage_intervals(ctx: Ctx, nums: Numbers) -> None:
    sample = ctx.sample(ctx.per_cohort)
    by_cohort: dict[str, list[str]] = {}
    for c, _s, mid in sample:
        by_cohort.setdefault(c, []).append(mid)
    jobs = [(c, b) for c, ids in by_cohort.items() for b in _batches(ids, 25)]

    def work(job):
        c, batch = job
        cur = ctx.conn().cursor()
        cur.execute(INTERVAL_SQL, (batch,))
        h = np.zeros(121, np.int64)
        for b, n in cur.fetchall():
            h[int(b)] += int(n)
        return c, h

    hist = {c: np.zeros(121, np.int64) for c in by_cohort}
    with ThreadPoolExecutor(ctx.workers) as pool:
        for c, h in pool.map(work, jobs):
            hist[c] += h
    pooled = sum(hist.values())
    s = interval_stats(pooled)
    kw = dict(stage="intervals", source="db", ref="E.1 L9", inputs=["pubg.telem_positions"], n_matches_sampled=len(sample), per_cohort=ctx.per_cohort, seed=ctx.seed,
              sql=" ".join(INTERVAL_SQL.split()), sample="random matches of the dump, per cohort, RandomState(seed)")  # fmt: skip
    nums.add(
        "intervals.n_intervals",
        s["n_intervals"],
        definition="consecutive position samples of one player (is_game >= 0.1, finite position), seconds between them (elapsed_time is an integer)",
        unit="intervals",
        **kw,
    )
    nums.add(
        "intervals.median_s",
        s["median_s"],
        definition="median interval (bin = floor seconds)",
        unit="s",
        **kw,
    )
    nums.add(
        "intervals.share_within_11s",
        s["share_le11"],
        definition="share of intervals with floor(interval) <= 11 s ('99.94 % within 11 s')",
        unit="share",
        **kw,
    )
    nums.add(
        "intervals.share_exactly_10s",
        s["share_eq10"],
        definition="share of intervals of exactly 10 s",
        unit="share",
        **kw,
    )
    nums.add(
        "intervals.share_over_20s",
        s["share_gt20"],
        definition="share of intervals above 20 s",
        unit="share",
        **kw,
    )
    nums.add(
        "intervals.share_over_60s",
        s["share_gt60"],
        definition="share of intervals above 60 s",
        unit="share",
        **kw,
    )
    nums.add(
        "intervals.p99_s", s["p99_s"], definition="99th percentile of the interval", unit="s", **kw
    )
    nums.add(
        "intervals.by_cohort",
        {c: interval_stats(h) for c, h in hist.items()},
        definition="the same statistics per cohort",
        **kw,
    )


CLOCK_SQL = """
SELECT p.match_id, percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch FROM p.event_time) - p.elapsed_time)::float8 AS o_pos, extract(epoch FROM s.t0)::float8 AS s_ep
  FROM pubg.telem_positions p
  JOIN (SELECT match_id, min(event_time) AS t0 FROM pubg.telem_match_start WHERE match_id = ANY(%s) GROUP BY 1) s ON s.match_id = p.match_id
 WHERE p.match_id = ANY(%s) AND p.is_game >= 0.1 AND p.elapsed_time IS NOT NULL
 GROUP BY p.match_id, s.t0
"""


def clock_stats(delta: np.ndarray) -> dict:
    """Pure core: median, p95 and max of the per-match clock offsets ``delta_pos = median(event_time - elapsed_time) - LogMatchStart`` (seconds)."""
    d = np.asarray(delta, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n_matches": 0}
    return {
        "n_matches": int(d.size),
        "median_s": float(np.median(d)),
        "p95_s": float(np.quantile(d, 0.95)),
        "max_s": float(d.max()),
        "min_s": float(d.min()),
    }


def stage_clock(ctx: Ctx, nums: Numbers) -> None:
    sample = ctx.sample(ctx.per_cohort)

    def work(batch):
        cur = ctx.conn().cursor()
        cur.execute(CLOCK_SQL, (batch, batch))
        return [(r[1] - r[2]) for r in cur.fetchall() if r[1] is not None and r[2] is not None]

    deltas: list[float] = []
    with ThreadPoolExecutor(ctx.workers) as pool:
        for part in pool.map(work, list(_batches([m for _, _, m in sample], 25))):
            deltas += part
    s = clock_stats(np.asarray(deltas))
    kw = dict(stage="clock", source="db", ref="E.1 L49", inputs=["pubg.telem_positions", "pubg.telem_match_start"], n_matches_sampled=len(sample), per_cohort=ctx.per_cohort, seed=ctx.seed,
              sql=" ".join(CLOCK_SQL.split()),
              note="a description of the elapsed_time FIELD only (the game clock G, which starts later than LogMatchStart); the v2 clock is W = event_time - LogMatchStart.event_time for every table "
                   "and no offset is estimated or applied (clock_delta_s = 0 in the dump records, see clock.*)")  # fmt: skip
    for key, label in (("median_s", "median"), ("p95_s", "95th percentile"), ("max_s", "maximum")):
        nums.add(
            f"elapsed_time_offset.{key}",
            s.get(key),
            definition=f"{label} over the sampled matches of delta_pos = median(epoch(event_time) - elapsed_time over the positions) - epoch(LogMatchStart): "
            "how far the elapsed_time field lags LogMatchStart",
            unit="s",
            **kw,
        )
    nums.add(
        "elapsed_time_offset.n_matches",
        s["n_matches"],
        definition="matches with an offset",
        unit="matches",
        **kw,
    )


def cut_arrays(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(cut time on W [P], valid [P])`` of a dump record: ``cut_wall_s`` (v1: the W-clock cut; v2: identical to ``cut_t``), and ``cut_valid`` (v2) or
    'phases k and k + 1 exist' (v1)."""
    cut = np.asarray(m["cut_wall_s"] if "cut_wall_s" in m else m["cut_t"], dtype=np.float64)
    if "cut_valid" in m:
        valid = np.asarray(m["cut_valid"]).astype(bool)
    else:
        pm = np.asarray(m["phase_mask"]).astype(bool)
        valid = np.zeros(len(pm), dtype=bool)
        valid[:-1] = pm[:-1] & pm[1:]
    return cut, valid & np.isfinite(cut)


def player_cut_coverage(
    pos_w: np.ndarray,
    pos_acct: np.ndarray,
    alive: Sequence[tuple[int, float]],
    windows: Sequence[float] = (15.0, 30.0),
) -> dict:
    """Pure core: for every ``(account index, cut time)`` of an alive player, the age of its last position sample before the cut (``pos_w < cut``); counts the players
    without a sample inside each window (and without any sample)."""
    order = np.lexsort((pos_w, pos_acct))
    w, a = pos_w[order], pos_acct[order]
    vals, first = np.unique(a, return_index=True)
    bounds = np.r_[first, len(a)]
    span = {
        int(v): (int(bounds[i]), int(bounds[i + 1])) for i, v in enumerate(vals)
    }  # account index -> its slice of the sorted samples
    miss = {float(x): 0 for x in windows}
    none = 0
    for acct, cut in alive:
        sl = span.get(int(acct))
        if sl is None:
            none += 1
            for x in miss:
                miss[x] += 1
            continue
        i0, i1 = sl
        j = int(np.searchsorted(w[i0:i1], cut, side="left")) - 1  # last sample with w < cut
        age = float("inf") if j < 0 else cut - float(w[i0 + j])
        none += int(not math.isfinite(age))
        for x in miss:
            miss[x] += int(age > x)
    return {
        "n_alive": len(alive),
        "missing": {f"{x:g}": v for x, v in miss.items()},
        "without_any_sample": none,
    }


def stage_coverage(ctx: Ctx, nums: Numbers) -> None:
    import pubg_lab.survival.data.phase_v2.alive as AL
    from pubg_lab.survival.data.phase_v2.source import fetch_match_source

    sample = ctx.sample(ctx.audit_per_cohort)

    records: dict[tuple[str, str], dict[str, dict]] = {}
    for (
        c,
        s,
        mid,
    ) in sample:  # cut times and cut validity come from the dump records of the sampled matches
        records.setdefault((c, s), {})[mid] = {}
    for (c, s), want in records.items():
        for m in ctx.load(c, s)["matches"]:
            if str(m["match_id"]) in want:
                cut, valid = cut_arrays(m)
                want[str(m["match_id"])] = {"cut_t": cut, "cut_valid": valid}

    def run(job):
        cohort, split, mid = job
        rec = records[(cohort, split)][mid]
        cur = ctx.conn().cursor()
        src = fetch_match_source(cur, mid)
        if src is None or not rec:
            return cohort, None
        tl = AL.timelines_from_source(src)
        acct = src.account_index()
        members = [(pid, acct[pid]) for (_team, pid, _dt, _ts) in src.roster if pid in acct]
        out = {"by_cut": {}}
        for k in range(1, CUTS + 1):
            if not rec["cut_valid"][k - 1]:
                continue
            a = float(rec["cut_t"][k - 1])
            alive = [(idx, a) for _pid, idx in members if AL.player_alive_at(tl, idx, a)]
            out["by_cut"][k] = player_cut_coverage(src.pos["w"], src.pos["acct"], alive)
        return cohort, out

    tot = {"n_alive": 0, "15": 0, "30": 0, "none": 0}
    by_cut: dict[int, dict] = {}
    by_cohort: dict[str, dict] = {}
    n_ok = 0
    with ThreadPoolExecutor(ctx.workers) as pool:
        for cohort, res in pool.map(run, sample):
            if res is None:
                continue
            n_ok += 1
            for k, c in res["by_cut"].items():
                for d in (
                    tot,
                    by_cut.setdefault(k, {"n_alive": 0, "15": 0, "30": 0, "none": 0}),
                    by_cohort.setdefault(cohort, {"n_alive": 0, "15": 0, "30": 0, "none": 0}),
                ):
                    d["n_alive"] += c["n_alive"]
                    d["15"] += c["missing"]["15"]
                    d["30"] += c["missing"]["30"]
                    d["none"] += c["without_any_sample"]
    kw = dict(stage="coverage", source="db", ref="E.1 L52", inputs=["pubg.telem_positions", "pubg.telem_kills", "pubg.participants", ctx.dump_dir], n_matches_sampled=n_ok, per_cohort=ctx.audit_per_cohort,
              seed=ctx.seed, sample="every match of the dump" if ctx.audit_per_cohort <= 0 else f"{ctx.audit_per_cohort} random matches per cohort, RandomState(seed)")  # fmt: skip
    d = "alive players at the cut a_k = cut_t of the dump record (phase_v2.alive.player_alive_at: no kill event <= a_k, or revive evidence), cuts 1..7 that are valid"
    nums.add("coverage.n_alive_players", tot["n_alive"], definition=d, unit="player-cuts", **kw)
    nums.add(
        "coverage.share_with_sample_15s",
        1 - tot["15"] / tot["n_alive"] if tot["n_alive"] else None,
        definition="share of those players with a position sample in [a_k - 15 s, a_k) (v1: 99.45 % with the grid-based cut)",
        unit="share",
        **kw,
    )
    nums.add(
        "coverage.share_with_sample_30s",
        1 - tot["30"] / tot["n_alive"] if tot["n_alive"] else None,
        definition="the same with a 30 s window",
        unit="share",
        **kw,
    )
    nums.add(
        "coverage.share_without_any_sample",
        tot["none"] / tot["n_alive"] if tot["n_alive"] else None,
        definition="players with no position sample before the cut at all",
        unit="share",
        **kw,
    )
    nums.add(
        "coverage.by_cut",
        {
            str(k): {
                "n_alive": v["n_alive"],
                "share_15s": 1 - v["15"] / v["n_alive"] if v["n_alive"] else None,
            }
            for k, v in sorted(by_cut.items())
        },
        definition="per cut k",
        **kw,
    )


def stage_lookahead(ctx: Ctx, nums: Numbers) -> None:
    """The prefix-replay no-look-ahead check of the dump builder (``model_stgcn.phase_v2.replay.run_prefix_replay``) on ``lookahead_n`` random matches, plus the
    dump-level identity ``cut_t = phase_end_s - 10`` over every dumped match."""
    import pubg_lab.survival.data.phase_v2.replay as RP

    cur = ctx.conn().cursor()
    per = math.ceil(ctx.lookahead_n / 12)
    pool = RP.stratified_matches(cur, per_cohort=per, tag=f"paper-numbers-{ctx.seed}")
    rng = np.random.RandomState(ctx.seed)
    pick = [pool[i] for i in sorted(rng.permutation(len(pool))[: ctx.lookahead_n])]
    res = RP.run_prefix_replay(cur, pick, mutation_matches=3)
    kw = dict(
        stage="lookahead",
        source="db",
        ref="E.1 L50",
        inputs=["model_stgcn.phase_v2.replay.run_prefix_replay", "pubg.telem_*"],
        seed=ctx.seed,
    )
    nums.add(
        "lookahead.n_matches",
        res["n_matches"],
        definition="matches replayed (random, stratified by cohort)",
        unit="matches",
        **kw,
    )
    nums.add(
        "lookahead.n_cuts",
        res["n_cuts"],
        definition="cuts replayed: inputs of slot k rebuilt from the full source and from a source truncated by the DATABASE at a_k",
        unit="cuts",
        **kw,
    )
    nums.add(
        "lookahead.n_mismatches",
        len(res["mismatches"]),
        definition="cuts whose inputs (X, edges, node_mask, alive counts, team_pos, zone) differ between the two sources, or whose last kept record is not before a_k",
        unit="cuts",
        **kw,
    )
    nums.add(
        "lookahead.min_gap_s",
        res.get("min_gap_between_last_kept_record_and_cut_s"),
        definition="smallest a_k - (last record time kept by the truncated source) over all cuts (> 0: no record at or after the cut is read)",
        unit="s",
        **kw,
    )
    nums.add(
        "lookahead.mutation_detected",
        res["mutation_check"],
        definition="the comparison catches a builder that forgets to cut the event tables (cuts detected as leaky / compared)",
        **kw,
    )
    rp_path = ctx.dump_dir / P.PREFIX_REPLAY_FILE
    if rp_path.is_file():
        rp = json.loads(rp_path.read_text(encoding="utf-8"))
        kw_d = dict(stage="lookahead", source="dump", ref="E.1 L50", inputs=[rp_path])
        nums.add(
            "lookahead.dump_replay_n_matches",
            rp.get("n_matches"),
            definition="matches of the dump builder's own prefix replay (prefix_replay.json)",
            unit="matches",
            **kw_d,
        )
        nums.add(
            "lookahead.dump_replay_n_cuts",
            rp.get("n_cuts"),
            definition="cuts of that replay",
            unit="cuts",
            **kw_d,
        )
        nums.add(
            "lookahead.dump_replay_n_mismatches",
            len(rp.get("mismatches") or []),
            definition="cuts of that replay whose inputs differ between the full and the database-truncated source",
            unit="cuts",
            **kw_d,
        )
        nums.add(
            "lookahead.dump_replay_min_gap_s",
            rp.get("min_gap_between_last_kept_record_and_cut_s"),
            definition="smallest gap between the last kept record and the cut in that replay",
            unit="s",
            **kw_d,
        )
    ident = dump_cut_identity(ctx)
    nums.add(
        "lookahead.cut_equals_boundary_minus_10s",
        ident["all_equal"],
        definition="cut_t == phase_end_s - 10 s for every valid cut of every dumped match (a_k = b_k - 10 s)",
        ref="E.1 L51",
        unit="bool",
        stage="lookahead",
        source="dump",
        inputs=[ctx.dump_dir / "<cohort>" / "<split>.pt"],
        n_cuts=ident["n_cuts"],
        max_abs_error_s=ident["max_abs_error"],
    )


def dump_cut_identity(ctx: Ctx) -> dict:
    n, worst = 0, 0.0
    for c in ctx.cohorts():
        for s in SPLITS:
            for m in ctx.load(c, s)["matches"][: ctx.limit_matches or None]:
                end, cut = (
                    np.asarray(m["phase_end_s"], dtype=np.float64),
                    np.asarray(m["cut_t"], dtype=np.float64),
                )
                ok = np.isfinite(end) & np.isfinite(cut)
                n += int(ok.sum())
                if ok.any():
                    worst = max(worst, float(np.abs(cut[ok] - (end[ok] - P.V2_CUT_LEAD_S)).max()))
    return {"n_cuts": n, "max_abs_error": worst, "all_equal": worst <= 1e-6}


# ---------------------------------------------------------------- label audit
AUDIT_SQL_PARTICIPANTS = """
SELECT p.player_id, r.team_id, p.death_type, p.time_survived::float8 FROM pubg.participants p
  JOIN pubg.rosters r ON r.roster_id = p.roster_id AND r.match_id = p.match_id WHERE p.match_id = %s
"""


def independent_team_death(
    members: Sequence[tuple[Any, str | None, float | None]], first_kill_w: dict, offset: float
) -> float:
    """Pure core: the elimination time of a team by the label audit's rule, independent of the builder: a member dies at its FIRST kill event; a member without
    one dies at ``time_survived + offset`` unless it was alive at the end; the team dies with its last member (``inf`` while one lives)."""
    deaths = []
    for pid, death_type, ts in members:
        if pid in first_kill_w:
            deaths.append(float(first_kill_w[pid]))
        elif death_type is None or death_type == "alive" or ts is None or not math.isfinite(ts):
            deaths.append(math.inf)
        else:
            deaths.append(float(ts) + offset)
    return max(deaths) if deaths else math.inf


def label_agreement(
    truth_t: np.ndarray,
    boundary: np.ndarray,
    window_end: np.ndarray,
    dump_target: np.ndarray,
    dump_y: np.ndarray,
    cut_valid: np.ndarray,
) -> dict:
    """Pure core: per cut, the dump's risk set / positives against the ones implied by independent team death times ``truth_t`` [N]:
    ``risk = T > b_k``, ``positive = risk and T <= window_end_k`` (b_k = ``boundary`` [P], window end = ``window_end`` [P]); counts only valid cuts."""
    out = {
        k: 0
        for k in ("rows", "risk_true", "risk_dump", "risk_both", "pos_true", "pos_dump", "pos_both")
    }
    for p in range(len(boundary)):
        if not cut_valid[p] or not np.isfinite(boundary[p]):
            continue
        risk_t = truth_t > boundary[p]
        pos_t = risk_t & (truth_t <= window_end[p])
        risk_d = dump_target[p].astype(bool)
        pos_d = risk_d & (dump_y[p] > 0.5)
        out["rows"] += int(len(truth_t))
        out["risk_true"] += int(risk_t.sum())
        out["risk_dump"] += int(risk_d.sum())
        out["risk_both"] += int((risk_t & risk_d).sum())
        out["pos_true"] += int(pos_t.sum())
        out["pos_dump"] += int(pos_d.sum())
        out["pos_both"] += int((pos_t & pos_d).sum())
    return out


def stage_label_audit(ctx: Ctx, nums: Numbers) -> None:
    from pubg_lab.survival.data.phase_v2.source import fetch_match_source

    sample = ctx.sample(ctx.audit_per_cohort)
    records: dict[tuple[str, str], dict[str, dict]] = {}
    for c, s, mid in sample:
        records.setdefault((c, s), {})[mid] = {}
    for (c, s), want in records.items():
        for m in ctx.load(c, s)["matches"]:
            if str(m["match_id"]) in want:
                want[str(m["match_id"])] = m

    def run(job):
        cohort, split, mid = job
        m = records[(cohort, split)][mid]
        cur = ctx.conn().cursor()
        src = fetch_match_source(cur, mid)
        if src is None or not m:
            return None
        by_acct_first: dict[str, float] = {}
        accounts = src.accounts
        for w, v in zip(src.kills["w"], src.kills["victim"]):
            pid = accounts[int(v)] if 0 <= int(v) < len(accounts) else None
            if pid is not None and (pid not in by_acct_first or w < by_acct_first[pid]):
                by_acct_first[pid] = float(w)
        offs = [
            by_acct_first[pid] - ts
            for (_t, pid, dt, ts) in src.roster
            if pid in by_acct_first and ts is not None and math.isfinite(ts)
        ]
        offset = float(np.median(offs)) if offs else 0.0
        teams: dict[int, list] = {}
        for team, pid, dt, ts in src.roster:
            teams.setdefault(int(team), []).append((pid, dt, ts))
        truth = np.asarray(
            [
                independent_team_death(teams.get(int(t), []), by_acct_first, offset)
                if int(t) in teams
                else math.nan
                for t in src.team_ids
            ]
        )
        truth = np.where(np.asarray(src.team_won, dtype=bool), math.inf, truth)
        ok = np.isfinite(truth) | np.isposinf(truth)
        t_for = np.where(ok, truth, math.nan)
        dump_t = np.asarray(m["target_mask_v2"]).astype(bool)
        dump_y = np.asarray(m["y_hazard_v2"], dtype=np.float64)
        end = np.asarray(m["phase_end_s"], dtype=np.float64)
        wend = (
            np.asarray(m["label_window_end_s"], dtype=np.float64)
            if "label_window_end_s" in m
            else np.r_[end[1:], np.nan]
        )
        _cut, valid = cut_arrays(m)
        # teams without a roster (unknown T) have no truth: the comparison ignores them on both sides
        known = ok[None, :]
        return label_agreement(
            np.where(ok, truth, -math.inf),
            end,
            np.where(np.isfinite(wend), wend, np.inf),
            dump_t & known,
            dump_y,
            valid,
        )

    tot = {
        k: 0
        for k in ("rows", "risk_true", "risk_dump", "risk_both", "pos_true", "pos_dump", "pos_both")
    }
    n_ok = 0
    with ThreadPoolExecutor(ctx.workers) as pool:
        for res in pool.map(run, sample):
            if res is None:
                continue
            n_ok += 1
            for k, v in res.items():
                tot[k] += v
    kw = dict(stage="label_audit", source="db", ref="E.4 (4)", inputs=["pubg.telem_kills", "pubg.participants", "pubg.rosters", ctx.dump_dir], n_matches_sampled=n_ok, per_cohort=ctx.audit_per_cohort, seed=ctx.seed,
              rule="independent truth: a member dies at its first kill event (else time_survived + median offset unless alive at the end), the team with its last member, winners censored; risk = T > b_k, positive = risk and T <= b_{k+1} (or C_m)")  # fmt: skip
    nums.add(
        "label_audit.risk_set_agreement",
        tot["risk_both"] / tot["risk_dump"] if tot["risk_dump"] else None,
        definition="share of the dump's risk-set rows that are also in the independent risk set",
        unit="share",
        **kw,
    )
    nums.add(
        "label_audit.positive_precision",
        tot["pos_both"] / tot["pos_dump"] if tot["pos_dump"] else None,
        definition="share of the dump's positives that are independent positives (v1 pipeline labels: 70.4 %)",
        unit="share",
        **kw,
    )
    nums.add(
        "label_audit.positive_recall",
        tot["pos_both"] / tot["pos_true"] if tot["pos_true"] else None,
        definition="share of the independent positives that the dump labels (v1 pipeline labels: 75.2 %)",
        unit="share",
        **kw,
    )
    nums.add(
        "label_audit.counts",
        tot,
        definition="the counts behind the three shares (valid cuts 1..7, sampled matches)",
        **kw,
    )
    nums.skip(
        "label_audit.naive_rule_mislabel_11_4pct",
        "refers to the v1 pipeline labels (stdb.team_phase_labels), which v2 does not have; keep the sentence as a statement about the v1 rule or drop it",
        stage="label_audit",
        ref="E.4 (4)",
    )
    nums.skip(
        "label_audit.leak_27pct_of_matches",
        "the share of matches that leak without clock alignment is a property of the v1 two-clock pipeline; v2 has one clock and a replay check instead (lookahead.*)",
        stage="label_audit",
        ref="E.4 (4)",
    )


# =========================================================================== headline numbers (aggregator outputs)
HEADLINE_CONTRASTS: tuple[str, ...] = (
    "phase_main", "team_main", "rel_vs_shuf", "zone_nohab", "m11r_vs_krafton_tf", "m11r_vs_krafton_tf_h0", "m11r_vs_sagong_xgb", "hgbcut_vs_sagong_hgb", "hgbcut_vs_hgb_kraftonfeat",
    "k2h0_vs_k2", "k3a_vs_k2h0", "k3a_vs_k2", "k3_vs_k3a", "m01norank_vs_k3", "m11rnorank_vs_m01norank", "hgb_union_vs_hgbcut", "m01_vs_m01norank", "m11r_vs_m11rnorank", "m11r_vs_m01", "m01_vs_k3",
)  # fmt: skip


def stage_headline(ctx: Ctx, nums: Numbers) -> None:
    res = ctx.results_dir
    if res is None or not (res / "contrasts_pooled.csv").is_file():
        nums.skip(
            "headline",
            f"no aggregator output (contrasts_pooled.csv) in {res}",
            stage="headline",
            ref="E.3",
        )
        return
    cp = pd.read_csv(res / "contrasts_pooled.csv")
    cp = cp[cp["stratum"] == "all"]
    for name in HEADLINE_CONTRASTS:
        for metric in ("auc", "auc_within_phase", "spearman"):
            r = cp[(cp["contrast"] == name) & (cp["metric"] == metric)]
            if r.empty or (metric == "spearman" and name not in ("team_main",)):
                continue
            r = r.iloc[0]
            nums.add(f"headline.{name}.{metric}", {"theta": r.theta, "ci95": [r.ci95_low, r.ci95_high], "k": int(r.k), "k_total": int(r.k_total), "same_sign": int(r.same_sign), "verdict": r.verdict,
                                                   "p_holm": r.p_holm}, definition="pooled (DerSimonian-Laird) paired difference, stratum all", stage="headline", source="results", ref="E.3", unit="delta " + metric,
                     inputs=[res / "contrasts_pooled.csv"])  # fmt: skip
    am_path = res / "arm_metrics_mean.csv"
    if am_path.is_file():
        am = pd.read_csv(am_path)
        am = am[(am["stratum"] == "all")]
        means = {(r.arm, r.metric): float(r["mean"]) for _, r in am.iterrows()}
        for arm in (
            "M00",
            "M01",
            "M11R",
            "HGB-cut",
            "KRAFTON-tf-ourfeat-bce",
            "KRAFTON-tf-ourfeat",
            "KRAFTON-tf-ourfeat-hazard",
            "KRAFTON-tf",
            "SAGONG-xgb",
            "M01-norank",
            "M11R-norank",
        ):
            if (arm, "auc") in means:
                nums.add(f"headline.mean_auc.{arm}", means[(arm, "auc")], definition="unweighted mean over the cohorts of the test AUC, seed 42, stratum all", stage="headline", source="results", ref="E.3",
                         unit="AUC", inputs=[am_path])  # fmt: skip
        hab = {
            a: means.get((a, "auc"))
            for a in ("SAGONG-hgb", "SAGONG-hgb-hab", "HGB-cut", "HGB-noHab")
        }
        if all(v is not None for v in hab.values()):
            kw = dict(
                stage="headline",
                source="results",
                ref="E.4 (1)",
                unit="AUC",
                inputs=[am_path],
                recipe="summary_hab_increment: step_hab = hab - hgb, step_rest = cut - hab, step_total = cut - hgb, cut_minus_nohab = cut - nohab",
            )
            nums.add(
                "headline.hab_increment.step_hab",
                hab["SAGONG-hgb-hab"] - hab["SAGONG-hgb"],
                definition="habitat share of the representation effect: SAGONG-hgb-hab - SAGONG-hgb (mean test AUC)",
                **kw,
            )
            nums.add(
                "headline.hab_increment.step_rest",
                hab["HGB-cut"] - hab["SAGONG-hgb-hab"],
                definition="the rest: HGB-cut - SAGONG-hgb-hab",
                **kw,
            )
            nums.add(
                "headline.hab_increment.step_total",
                hab["HGB-cut"] - hab["SAGONG-hgb"],
                definition="HGB-cut - SAGONG-hgb",
                **kw,
            )
            nums.add(
                "headline.hab_increment.cut_minus_nohab",
                hab["HGB-cut"] - hab["HGB-noHab"],
                definition="HGB-cut - HGB-noHab",
                **kw,
            )
        else:
            nums.skip(
                "headline.hab_increment",
                "needs SAGONG-hgb, SAGONG-hgb-hab, HGB-cut and HGB-noHab in arm_metrics_mean.csv",
                stage="headline",
                ref="E.4 (1)",
            )
    sh = res / "ecological_model" / "share_of_discrimination.csv"
    if sh.is_file():
        df = pd.read_csv(sh)
        nums.add(
            "headline.ehm_share_of_discrimination",
            df.to_dict("records"),
            definition="rows of ecological_model/share_of_discrimination.csv (EHM, EHM-lite and the others)",
            stage="headline",
            source="results",
            ref="E.3",
            inputs=[sh],
        )
    mr = res / "metareg.csv"
    if mr.is_file():
        df = pd.read_csv(mr)
        nums.add(
            "headline.metareg_crossover",
            {
                r.panel: {
                    "crossover_n": r.crossover_n,
                    "ci": [r.crossover_ci_low, r.crossover_ci_high],
                    "slope": r.slope,
                }
                for r in df.itertuples()
            },
            definition="the crossover training size per panel of the meta-regression",
            stage="headline",
            source="results",
            ref="E.3",
            inputs=[mr],
        )


# =========================================================================== not recomputed: why
def stage_not_computed(nums: Numbers) -> None:
    nums.skip(
        "ai_accounts.airoyale_share",
        "a property of the account ids of the match set (the population is the same in v2); not dump-dependent",
        stage="population",
        ref="E.1 L67-69",
    )
    nums.skip(
        "edge_by_phase_elimination_rates",
        "no producer and no definition of 'outside' / 'deep inside' exist (manifest E.4 (5)); write the definition first",
        stage="risk_set",
        ref="E.4 (5)",
    )
    nums.skip(
        "constants.model_settings",
        "the training settings and the SESOI are constants of the code, not dump-dependent (ptr/hgb.py, ptr/train.py, ptr/aggregate.py)",
        stage="population",
        ref="E.2 L75-77",
    )


# =========================================================================== driver
STAGE_FUNCS: dict[str, Callable[[Ctx, Numbers], None]] = {
    "population": stage_population, "risk_set": stage_risk_set, "correlations": stage_correlations, "db_counts": stage_db_counts, "intervals": stage_intervals,
    "clock": stage_clock, "coverage": stage_coverage, "lookahead": stage_lookahead, "label_audit": stage_label_audit, "headline": stage_headline,
}  # fmt: skip


def run_stages(
    ctx: Ctx, stages: Sequence[str], *, log: Callable[[str], None] = lambda s: None
) -> Numbers:
    nums = Numbers()
    for st in stages:
        t0 = time.time()
        log(f"stage {st} ...")
        try:
            STAGE_FUNCS[st](ctx, nums)
        except Exception as exc:  # a failed stage is recorded; the other stages still run
            nums.error(st, exc)
            log(f"stage {st} FAILED: {exc}")
        else:
            log(f"stage {st} done in {time.time() - t0:.0f}s")
    stage_not_computed(nums)
    return nums


def build_output(ctx: Ctx, nums: Numbers, args: dict, stages: Sequence[str]) -> dict:
    ver = P.dump_version(ctx.dump_dir) if ctx.dump_dir.is_dir() else "unknown"
    return {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_info(), "dump_dir": str(ctx.dump_dir), "dump_version": ver,
            "dump_meta_sha1": P.dump_meta_sha1(ctx.dump_dir) if ctx.dump_dir.is_dir() else None, "stages": list(stages), "args": to_jsonable(args), "seed": ctx.seed,
            "script": 'projects/battle_royale/reporting/paper_numbers_v2.py', "n_numbers": len(nums.numbers),
        },
        "numbers": nums.numbers, "not_computed": nums.not_computed, "errors": nums.errors,
    }  # fmt: skip


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump-dir", default=str(P.DUMP_DIR), help="phase dump (default: the v2 dump)")
    ap.add_argument(
        "--runs-dir",
        default=None,
        help="run directory: cross-check of the risk-set rows against data.n_target_rows of M00 (default: the v2 root if it exists)",
    )
    ap.add_argument(
        "--results-dir",
        default=None,
        help="aggregator output for the headline stage (default: the v2 results root)",
    )
    ap.add_argument(
        "--out", default=None, help="JSON output (default: <results-dir>/paper_numbers.json)"
    )
    ap.add_argument(
        "--stages",
        default="all",
        help=f"comma separated of {', '.join(STAGES)}; 'all' (default) = every stage, the database stages only if credentials are in the environment",
    )
    ap.add_argument("--no-db", action="store_true", help="skip the database stages")
    ap.add_argument(
        "--allow-v1",
        action="store_true",
        help="run on a v1 dump (a test of the script): the version guard otherwise requires a v2 dump",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sample-per-cohort",
        type=int,
        default=250,
        help="matches per cohort of the interval and clock samples",
    )
    ap.add_argument(
        "--audit-per-cohort",
        type=int,
        default=0,
        help="matches per cohort of the coverage and label-audit stages (they fetch every record of the match; 0 = every match, "
        "about 12 minutes per stage with 2 workers)",
    )
    ap.add_argument("--lookahead-matches", type=int, default=400)
    ap.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="debug: at most N matches per split / per cohort sample; the numbers are NOT the paper's",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="database connections (the manifest recommends at most 2)",
    )
    return ap.parse_args(argv)


def resolve_stages(spec: str, *, no_db: bool, have_db: bool) -> list[str]:
    stages = list(STAGES) if spec == "all" else [s for s in spec.split(",") if s]
    unknown = [s for s in stages if s not in STAGE_FUNCS]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; known {STAGES}")
    if no_db or (spec == "all" and not have_db):
        stages = [s for s in stages if s not in DB_STAGES]
    return stages


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    dump = Path(a.dump_dir)
    results = Path(a.results_dir) if a.results_dir else P.RESULTS_DIR
    runs = Path(a.runs_dir) if a.runs_dir else (P.RUNS_DIR if P.RUNS_DIR.is_dir() else None)
    out = Path(a.out) if a.out else results / "paper_numbers.json"
    try:
        dv = P.check_same_version(
            dump, {"runs (--runs-dir)": runs, "results (--results-dir)": results}
        )
        out_v = P.name_version(
            out.parent
        )  # an explicit --out below a v1-named directory is a v1 location (any other place is fine: it is not a root)
        if out_v is not None and dv != "unknown" and out_v != dv:
            raise P.VersionMixError(
                f"the dump {dump} is {dv} but the output directory {out.parent} is {out_v}"
            )
    except P.VersionMixError as exc:
        raise SystemExit(f"refusing to run: {exc}") from exc
    if dv != P.VERSION and not a.allow_v1:
        raise SystemExit(
            f"refusing to run: the dump {dump} is {dv}, not {P.VERSION} (a test of this script on v1 needs --allow-v1)"
        )
    have_db = db_available() if (a.stages == "all" and not a.no_db) else True
    stages = resolve_stages(a.stages, no_db=a.no_db, have_db=have_db)
    ctx = Ctx(dump, runs_dir=runs, results_dir=results, seed=a.seed, per_cohort=a.sample_per_cohort, audit_per_cohort=a.audit_per_cohort, lookahead_n=a.lookahead_matches,
              limit_matches=a.limit_matches, workers=a.workers)  # fmt: skip
    nums = run_stages(ctx, stages, log=lambda s: print(s, flush=True))
    doc = build_output(ctx, nums, vars(a), stages)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(
        f"wrote {out}: {len(nums.numbers)} numbers, {len(nums.not_computed)} not computed, {len(nums.errors)} stage error(s)"
    )
    for e in nums.errors:
        print(f"  error in {e['stage']}: {e['error']}", file=sys.stderr)
    return 1 if nums.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
