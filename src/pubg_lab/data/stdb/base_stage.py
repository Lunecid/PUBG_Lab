"""Transient re-materialisation of the base STDB rows for ONE match.

Situation (2026-10-05): ``stdb.moving_objects``, ``stdb.spatial_events`` and
``stdb.team_proximity`` are EMPTY in the working database (they were purged after the
original ETL), while ``stdb.dynamic_regions`` and every derived phase table are intact.
The Fix B reruns (stdb/phase_spatial_feature_etl.py, phase_edge_etl.py, habitat /
network / z-affordance / f0 builders) and the DB-gated leakage tests need those three
tables, so this module rebuilds them for a single match from the raw ``pubg.*``
telemetry using EXACTLY the code ``stdb.etl`` uses (same 10 s grid, same nearest-sample
rule within +-SNAPSHOT_HALF_WIN, same event ingestion, same proximity construction).

What is deliberately NOT rebuilt:

* ``stdb.dynamic_regions`` (35 GB, intact; rewriting it would bloat the table),
* player-level ``moving_objects`` rows (no Fix B consumer reads them; ``team_only``),
* ``stdb.team_phase_features`` and every RQ1 table.

Both functions run inside the CALLER's transaction and never commit, so a failed ETL
leaves the database untouched.  ``tools/run_fixb_phase_etl.py`` stages -> rebuilds ->
purges inside one transaction per match; test fixtures stage and then roll back.
"""

from __future__ import annotations

import time

from pubg_lab.data.stdb.etl import (
    _build_team_proximity,
    _ingest_moving_objects,
    _ingest_spatial_events,
    _select_match_meta,
)

BASE_TABLES: tuple[str, ...] = (
    "stdb.moving_objects",
    "stdb.spatial_events",
    "stdb.team_proximity",
)

# Session-local twins of the base tables (see TempBaseCursor): no WAL, no autovacuum, no
# cross-session lock on ANALYZE, nothing committed to the shared tables.
TEMP_SCHEMA = "pg_temp"


def stage_base_rows(cur, match_id: str, *, team_only: bool = True, before_proximity=None) -> dict:
    """Insert grid / event / proximity rows for ``match_id`` (caller commits or rolls back).

    ``before_proximity(cur)`` runs after the grid and the events are staged and before the
    proximity self-join / encounter update, which are mis-planned on tables without statistics
    (the Fix B driver passes an ANALYZE of the session-local temp tables here)."""
    t0 = time.time()
    meta = _select_match_meta(cur, match_id)
    if meta is None:
        raise LookupError(f"match {match_id} not found in pubg.matches")
    if meta.get("duration") is None:
        raise ValueError(f"match {match_id} has no duration; cannot build the 10 s grid")

    n_player, n_team = _ingest_moving_objects(
        cur, match_id, int(meta["duration"]), team_only=team_only
    )
    t1 = time.time()
    n_event = _ingest_spatial_events(cur, match_id, meta["started_at"])
    if before_proximity is not None:
        before_proximity(cur)
    t2 = time.time()
    n_prox = _build_team_proximity(cur, match_id)
    t3 = time.time()
    return {
        "match_id": match_id,
        "player_rows": n_player,
        "team_rows": n_team,
        "event_rows": n_event,
        "proximity_rows": n_prox,
        "grid_s": round(t1 - t0, 3),
        "events_s": round(t2 - t1, 3),
        "proximity_s": round(t3 - t2, 3),
    }


def purge_base_rows(cur, match_id: str) -> dict:
    """Delete the match's base rows again (restores the empty-base-table state)."""
    out = {}
    for table in BASE_TABLES:
        cur.execute(f"DELETE FROM {table} WHERE match_id = %s", (match_id,))
        out[table] = int(cur.rowcount)
    return out


# --------------------------------------------------------------------------------------
# session-local staging (used by tools/run_fixb_phase_etl.py)
# --------------------------------------------------------------------------------------
def rewrite_to_temp(sql):
    """Point the three base tables of a SQL string (str or bytes) at their pg_temp twins."""
    if isinstance(sql, bytes):
        for table in BASE_TABLES:
            sql = sql.replace(table.encode(), f"{TEMP_SCHEMA}.{table.split('.', 1)[1]}".encode())
        return sql
    for table in BASE_TABLES:
        sql = sql.replace(table, f"{TEMP_SCHEMA}.{table.split('.', 1)[1]}")
    return sql


class TempBaseCursor:
    """Cursor proxy that redirects ``stdb.moving_objects`` / ``stdb.spatial_events`` /
    ``stdb.team_proximity`` to the session's temporary tables of the same name.

    The ETL builders and ``stage_base_rows`` write SQL against the schema-qualified shared
    tables; routing them through this proxy lets the Fix B rerun stage the base grid in
    session-local temp tables (no WAL, no dead tuples in the shared tables, a private ANALYZE)
    without touching the ETL code under test.  Only ``execute`` / ``mogrify`` are rewritten
    (``psycopg2.extras.execute_batch`` needs exactly those); every other attribute is forwarded.
    """

    def __init__(self, cur) -> None:
        self._cur = cur

    def execute(self, sql, params=None):
        return self._cur.execute(rewrite_to_temp(sql), params)

    def mogrify(self, sql, params=None):
        return self._cur.mogrify(rewrite_to_temp(sql), params)

    def __getattr__(self, name):
        return getattr(self._cur, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def create_temp_base_tables(conn, *, temp_buffers: str = "128MB") -> None:
    """Create the session-local twins (idempotent).  ``temp_buffers`` must be set before the
    session first touches a temp table."""
    with conn.cursor() as cur:
        cur.execute(f"SET temp_buffers = '{temp_buffers}'")
        for table in BASE_TABLES:
            name = table.split(".", 1)[1]
            cur.execute(f"CREATE TEMP TABLE IF NOT EXISTS {name} (LIKE {table} INCLUDING ALL)")
    conn.commit()


def reset_temp_base_tables(cur) -> None:
    """Empty the twins (call at the start of every match; transactional like any TRUNCATE)."""
    names = ", ".join(f"{TEMP_SCHEMA}.{t.split('.', 1)[1]}" for t in BASE_TABLES)
    cur.execute(f"TRUNCATE {names}")


def analyze_temp_base_tables(cur) -> None:
    """Fresh planner statistics for the staged rows (local to this session: no lock contention)."""
    for table in BASE_TABLES:
        cur.execute(f"ANALYZE {TEMP_SCHEMA}.{table.split('.', 1)[1]}")
