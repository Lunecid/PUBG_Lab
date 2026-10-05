"""Fix B rerun: rebuild the phase-tier derived tables with capped grid joins.

For every selected match, in ONE transaction (a failure leaves the old derived rows):

    1. stage   the base grid rows (team centroids on the 10 s grid, spatial events, team
               proximity) from the raw pubg.* telemetry with the original stdb.etl code
               (stdb.base_stage).  stdb.moving_objects / spatial_events / team_proximity are
               EMPTY in the database, so they are staged in SESSION-LOCAL TEMP TABLES of the
               same names (``TempBaseCursor`` redirects the schema-qualified SQL): no WAL, no
               dead tuples in the shared tables, a private ANALYZE (without fresh statistics the
               planner mis-orders the exposure joins: 17 s instead of 0.07 s per match).
    2. rebuild in dependency order (every window on ONE clock, then capped: stdb/causal_cut.py)
         match_phase_global  (rq1 global block: game-state window shifted, zone grid reads capped)
         team_phase_state    (rq1 raw position window shifted and capped)
         team_phase_trajectory + team_phase_exposure   (phase_spatial_feature_etl)
         team_phase_edge_cache                          (phase_edge_etl)
         team_phase_habitat_features                    (habitat_feature_etl; reads 1,2)
         team_phase_network_features                    (network_feature_etl; reads edges)
         team_phase_z_affordance_features               (z_affordance_feature_etl; reads habitat)
         team_phase_f0_extra                            (f0_feature_etl)
    3. guard   row counts of the rebuilt tables must not collapse against the old rows

The shared base tables are never written.  Run ``tools/fixb_backup_tables.py`` FIRST: this
script overwrites derived rows (the backup pre-flight below refuses to run otherwise).

Usage (credentials come from the environment, see stdb/conn.py):
    python -m tools.run_fixb_phase_etl --scope study --workers 12
    python -m tools.run_fixb_phase_etl --match-id <id> --rollback     # dry run, nothing kept

Resumable: ``<out-dir>/done.jsonl`` lists finished matches; resume is the default.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_stgcn.v3.cohorts import REQUIRED_COHORTS
from stdb.conn import connect

logger = logging.getLogger("tools.run_fixb_phase_etl")

DEFAULT_OUT_DIR = Path("model_stgcn/etl_runs/fixb_2026-10-05")
SPLIT_ROOT = Path("model_stgcn/checkpoints/br_om_ladder_required_v1")

# table -> minimum (rows after / rows before) accepted by the guard
GUARDED_TABLES = {
    "match_phase_global": 1.0,
    "team_phase_state": 1.0,
    "team_phase_trajectory": 0.7,
    "team_phase_exposure": 0.7,
    "team_phase_edge_cache": 0.7,
    "team_phase_habitat_features": 1.0,
    "team_phase_network_features": 1.0,
    "team_phase_z_affordance_features": 1.0,
    "team_phase_f0_extra": 1.0,
}


class SkipMatch(Exception):
    """Raised inside the transaction to leave the match's old rows untouched."""


# --------------------------------------------------------------------------------------
# match selection
# --------------------------------------------------------------------------------------
def study_match_ids_by_cohort() -> list[tuple[str, list[str]]]:
    """(cohort slug, match ids) from the stored 13-cohort split manifests, smallest first."""
    out = []
    for spec in REQUIRED_COHORTS:
        path = SPLIT_ROOT / spec.slug / "split.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ids: list[str] = []
        for part in ("train", "val", "test"):
            ids.extend(manifest["splits"][part])
        out.append((spec.slug, ids))
    out.sort(key=lambda kv: len(kv[1]))
    return out


EXCLUDED_COHORTS = ("vikendi_tpp_squad",)  # design v2.3 population rule
EXCLUDED_MATCH_TYPES = ("airoyale",)


_RAISED_CONN = None


def _is_raised(match_id: str) -> bool:
    """Worker: is the clock delta of this match raised by the drift-aware rule (stdb.causal_cut)?"""
    global _RAISED_CONN
    from stdb import causal_cut

    if _RAISED_CONN is None:
        _RAISED_CONN = connect()
        _RAISED_CONN.set_session(readonly=True, autocommit=True)
    with _RAISED_CONN.cursor() as cur:
        return bool(causal_cut.fetch_clock_delta_detail(cur, match_id).raised)


def select_matches(scope: str) -> list[tuple[str, str]]:
    """List of (match_id, group) in processing order.

    ``kept``  study matches that survive the v2.3 population rules (not airoyale, not Vikendi),
              smallest cohort first -- the matches that are dumped;
    ``raised`` the ``kept`` matches whose clock delta is raised by the drift-aware rule
              (stdb.causal_cut.safe_clock_delta): they were first rebuilt with the plain median delta and must
              be rebuilt again (writes ``fixb_bump_since.txt``, which model_stgcn/dump_phase_tier.py requires);
    ``study`` kept + the excluded study matches (airoyale / Vikendi);
    ``rest``  every other labelled match;
    ``all``   study + rest.
    """
    by_cohort = study_match_ids_by_cohort()
    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for slug, ids in by_cohort:
        for mid in ids:
            if mid not in seen:
                seen.add(mid)
                ordered.append((mid, slug))

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_id, match_type FROM pubg.matches WHERE match_id = ANY(%s)",
                ([m for m, _ in ordered],),
            )
            mtype = dict(cur.fetchall())
            labeled: list[str] = []
            if scope in ("rest", "all"):
                cur.execute(
                    """
                    SELECT m.match_id
                      FROM stdb.matches m
                     WHERE EXISTS (SELECT 1 FROM stdb.team_phase_labels l WHERE l.match_id = m.match_id)
                     ORDER BY m.match_id
                    """
                )
                labeled = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()

    kept = [
        (m, c) for m, c in ordered
        if c not in EXCLUDED_COHORTS and mtype.get(m) not in EXCLUDED_MATCH_TYPES
    ]
    kept_ids = {m for m, _ in kept}
    dropped = [(m, c) for m, c in ordered if m not in kept_ids]
    if scope == "kept":
        return kept
    if scope == "raised":
        ctx = mp.get_context("spawn")  # fork + libpq is unsafe on macOS
        with ctx.Pool(processes=8) as pool:
            flags = pool.map(_is_raised, [m for m, _ in kept], chunksize=64)
        return [(m, c) for (m, c), raised in zip(kept, flags) if raised]
    if scope == "study":
        return kept + dropped
    rest = [(mid, "rest") for mid in labeled if mid not in seen]
    if scope == "rest":
        return rest
    if scope == "all":
        return kept + dropped + rest
    raise ValueError(f"unknown scope {scope!r}")


# --------------------------------------------------------------------------------------
# one match
# --------------------------------------------------------------------------------------
def _counts(cur, match_id: str) -> dict[str, int]:
    out = {}
    for table in GUARDED_TABLES:
        cur.execute(f"SELECT count(*) FROM stdb.{table} WHERE match_id = %s", (match_id,))
        out[table] = int(cur.fetchone()[0])
    return out


def rebuild_match(cur, match_id: str) -> dict[str, Any]:
    """Stage -> rebuild -> guard for ONE match inside the caller's transaction.

    ``cur`` is a ``stdb.base_stage.TempBaseCursor`` (base tables -> session temp tables).  The
    caller commits (or rolls back); nothing is written to the shared base tables.
    """
    from stdb import (
        f0_feature_etl,
        habitat_feature_etl,
        network_feature_etl,
        phase_edge_etl,
        phase_spatial_feature_etl,
        rq1_feature_etl,
        z_affordance_feature_etl,
    )
    from stdb.base_stage import analyze_temp_base_tables, reset_temp_base_tables, stage_base_rows

    timings: dict[str, float] = {}

    def timed(name, fn, *args):
        t0 = time.time()
        res = fn(*args)
        timings[name] = round(time.time() - t0, 3)
        return res

    from stdb import causal_cut

    clock = causal_cut.fetch_clock_delta_detail(cur, match_id, use_cache=False)  # cached for the builders below
    before = _counts(cur, match_id)
    timed("reset", reset_temp_base_tables, cur)
    staged = timed(
        "stage", lambda c, m: stage_base_rows(c, m, before_proximity=analyze_temp_base_tables), cur, match_id
    )
    if staged["team_rows"] == 0 or staged["proximity_rows"] == 0:
        raise SkipMatch(f"no raw grid rows could be staged: {staged}")
    timed("analyze", analyze_temp_base_tables, cur)

    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM stdb.team_snapshot_features WHERE match_id = %s)",
        (match_id,),
    )
    has_snapshot_features = bool(cur.fetchone()[0])

    timed("match_phase_global", rq1_feature_etl.build_match_phase_global, cur, match_id)
    timed("team_phase_state", rq1_feature_etl.build_team_phase_state, cur, match_id)
    spatial = timed("spatial", phase_spatial_feature_etl.build_match, cur, match_id)
    edge = timed("edge", phase_edge_etl.build_match, cur, match_id)
    timed("habitat", habitat_feature_etl.build_match, cur, match_id)
    timed("network", network_feature_etl.build_match, cur, match_id)
    if has_snapshot_features or before["team_phase_z_affordance_features"] > 0:
        timed("z_affordance", z_affordance_feature_etl.build_match, cur, match_id)
    timed("f0", f0_feature_etl.build_match, cur, match_id)

    after = _counts(cur, match_id)
    ratios = {}
    for table, min_ratio in GUARDED_TABLES.items():
        old, new = before[table], after[table]
        ratios[table] = round(new / old, 4) if old else None
        if old and new < min_ratio * old:
            raise SkipMatch(
                f"row-count guard failed for {table}: before={old} after={new} "
                f"(min ratio {min_ratio})"
            )
    return {
        "staged": staged,
        "spatial": spatial,
        "edge": edge,
        "before": before,
        "after": after,
        "ratios": ratios,
        "timings": timings,
        "clock": {"median_s": round(clock.median_s, 3), "delta_s": round(clock.safe_s, 3), "raised": clock.raised},
    }


_CONN = None


def _worker_connection():
    """One long-lived connection per worker process; its temp tables are reused per match."""
    global _CONN
    from stdb.base_stage import create_temp_base_tables

    if _CONN is None or _CONN.closed:
        _CONN = connect()
        create_temp_base_tables(_CONN)
    return _CONN


def _drop_connection() -> None:
    global _CONN
    try:
        if _CONN is not None:
            _CONN.close()
    except Exception:
        pass
    _CONN = None


def process_match(args: tuple[str, str, bool]) -> dict[str, Any]:
    from stdb.base_stage import TempBaseCursor

    match_id, cohort, rollback = args
    t0 = time.time()
    try:
        conn = _worker_connection()
        try:
            with conn.cursor() as raw_cur:
                detail = rebuild_match(TempBaseCursor(raw_cur), match_id)
            if rollback:
                conn.rollback()
            else:
                conn.commit()
            return {
                "match_id": match_id,
                "cohort": cohort,
                "ok": True,
                "status": "rolled_back" if rollback else "ok",
                "elapsed_s": round(time.time() - t0, 3),
                **detail,
            }
        except SkipMatch as exc:
            conn.rollback()
            return {
                "match_id": match_id,
                "cohort": cohort,
                "ok": False,
                "status": "skipped",
                "reason": str(exc),
                "elapsed_s": round(time.time() - t0, 3),
            }
    except Exception as exc:  # keep large batches moving; the match keeps its old rows
        try:
            if _CONN is not None and not _CONN.closed:
                _CONN.rollback()
        except Exception:
            _drop_connection()
        return {
            "match_id": match_id,
            "cohort": cohort,
            "ok": False,
            "status": "failed",
            "reason": repr(exc),
            "traceback": traceback.format_exc(limit=6),
            "elapsed_s": round(time.time() - t0, 3),
        }


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def _load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok":
                done.add(rec["match_id"])
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scope", choices=["kept", "raised", "study", "rest", "all"], default="kept")
    ap.add_argument("--match-id", action="append", default=None, help="repeatable; overrides --scope")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--rollback", action="store_true", help="dry run: rebuild, report, roll back")
    ap.add_argument("--no-resume", action="store_true", help="ignore done.jsonl")
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--skip-backup-check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done_name = "done_raised.jsonl" if args.scope == "raised" else "done.jsonl"
    done_path = args.out_dir / ("dryrun_done.jsonl" if args.rollback else done_name)
    if not args.rollback:
        marker = args.out_dir / "fixb_since.txt"
        if not args.skip_backup_check:
            # never overwrite derived rows before the backup schema is complete and verified
            from tools.fixb_backup_tables import verify_backups

            checks = verify_backups(exact=False)  # compares with the counts recorded at backup time
            (args.out_dir / "backup_preflight.json").write_text(
                json.dumps(checks, indent=2) + "\n", encoding="utf-8"
            )
            bad = [c for c in checks if not c["ok"]]
            if bad:
                logger.error("backup pre-flight failed, refusing to run: %s", bad)
                return 3
            logger.info("backup pre-flight ok: %s", checks)
        # model_stgcn/dump_phase_tier.py refuses to dump matches whose derived rows are older
        # than this marker (i.e. that were not rebuilt with the Fix B cap)
        if not marker.exists():
            marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        bump_marker = args.out_dir / "fixb_bump_since.txt"
        if args.scope == "raised" and not args.match_id and not bump_marker.exists():
            # from here on every rebuild uses the drift-aware delta; dumps of raised matches must be newer
            bump_marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")

    if args.match_id:
        selected = [(mid, "adhoc") for mid in args.match_id]
    else:
        selected = select_matches(args.scope)
    if not args.no_resume and not args.rollback:
        done = _load_done(done_path)
        selected = [(m, c) for m, c in selected if m not in done]
    if args.limit:
        selected = selected[: args.limit]
    logger.info(
        "selected=%d scope=%s workers=%d rollback=%s out=%s",
        len(selected), args.scope, args.workers, args.rollback, args.out_dir,
    )
    if not selected:
        return 0

    started = time.time()
    n_ok = n_fail = n_skip = 0
    stage_sum: dict[str, float] = {}
    jobs = [(m, c, args.rollback) for m, c in selected]
    ctx = mp.get_context("spawn")  # fork + libpq is unsafe on macOS
    with ctx.Pool(processes=max(1, args.workers)) as pool, done_path.open("a", encoding="utf-8") as done_f:
        for i, res in enumerate(pool.imap_unordered(process_match, jobs, chunksize=1), start=1):
            done_f.write(json.dumps(res, default=str) + "\n")
            done_f.flush()
            if res["status"] in ("ok", "rolled_back"):
                n_ok += 1
                for k, v in res.get("timings", {}).items():
                    stage_sum[k] = stage_sum.get(k, 0.0) + v
            elif res["status"] == "skipped":
                n_skip += 1
                logger.warning("SKIPPED %s (%s): %s", res["match_id"], res["cohort"], res["reason"])
            else:
                n_fail += 1
                logger.error("FAILED %s (%s): %s", res["match_id"], res["cohort"], res["reason"])
            if i == 1 or i % args.progress_every == 0 or i == len(jobs):
                wall = time.time() - started
                rate = i / wall
                eta = (len(jobs) - i) / rate if rate > 0 else float("nan")
                logger.info(
                    "progress %d/%d ok=%d skipped=%d failed=%d rate=%.2f/s eta=%.0fs",
                    i, len(jobs), n_ok, n_skip, n_fail, rate, eta,
                )

    wall = time.time() - started
    summary = {
        "scope": args.scope,
        "selected": len(selected),
        "ok": n_ok,
        "skipped": n_skip,
        "failed": n_fail,
        "rollback": args.rollback,
        "workers": args.workers,
        "wall_s": round(wall, 1),
        "mean_stage_s": {k: round(v / max(n_ok, 1), 3) for k, v in sorted(stage_sum.items())},
    }
    (args.out_dir / ("dryrun_summary.json" if args.rollback else f"summary_{args.scope}.json")).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("summary: %s", summary)
    return 0 if n_fail == 0 and n_skip == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
