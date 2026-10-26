"""Back up the derived stdb tables that the Fix B ETL rerun overwrites.

Fix B (2026-10-05) re-materialises the phase-tier feature / edge tables with the grid
joins capped at ``phase_end_s - SNAPSHOT_INTERVAL``.  Before any of them is touched,
each is copied (data only) into the schema ``stdb_bak_20261005`` and the row counts
of source and copy are compared.  Raw ``pubg.*`` tables are never touched.

Usage (credentials come from the environment, see stdb/conn.py):
    python -m tools.fixb_backup_tables [--tables a b ...] [--out model_stgcn/etl_runs/fixb_2026-10-05/backup.json]

The script is idempotent: a backup table that already exists with the same row count
as its source is left alone; a mismatching one aborts (never overwritten).
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from pubg_lab.data.stdb.connection.conn import connect

logger = logging.getLogger("projects.battle_royale.pipelines.fixb_backup_tables")

BACKUP_SCHEMA = "stdb_bak_20261005"

# Every derived table the Fix B reruns can DELETE/INSERT into (see tools/run_fixb_phase_etl.py).
DERIVED_TABLES: tuple[str, ...] = (
    "match_phase_global",
    "team_phase_state",  # added 2026-10-05 with the one-clock fix (position window shifted + capped)
    "team_phase_trajectory",
    "team_phase_exposure",
    "team_phase_edge_cache",
    "team_phase_habitat_features",
    "team_phase_network_features",
    "team_phase_z_affordance_features",
    "team_phase_f0_extra",
)

DEFAULT_OUT = Path("model_stgcn/etl_runs/fixb_2026-10-05/backup.json")


def _count(cur, schema: str, table: str) -> int:
    cur.execute(f'SELECT count(*) FROM "{schema}"."{table}"')
    return int(cur.fetchone()[0])


def _exists(cur, schema: str, table: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
        (schema, table),
    )
    return cur.fetchone() is not None


def backup_table(table: str) -> dict:
    started = time.time()
    conn = connect(autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{BACKUP_SCHEMA}"')
            src_n = _count(cur, "stdb", table)
            if _exists(cur, BACKUP_SCHEMA, table):
                dst_n = _count(cur, BACKUP_SCHEMA, table)
                if dst_n != src_n:
                    raise RuntimeError(
                        f"backup {BACKUP_SCHEMA}.{table} exists with {dst_n} rows but source has "
                        f"{src_n}; refusing to overwrite or proceed"
                    )
                logger.info("backup %s already present (%d rows)", table, dst_n)
                action = "already_present"
            else:
                logger.info(
                    "copying stdb.%s (%d rows) -> %s.%s", table, src_n, BACKUP_SCHEMA, table
                )
                cur.execute(f'CREATE TABLE "{BACKUP_SCHEMA}"."{table}" AS TABLE stdb."{table}"')
                dst_n = _count(cur, BACKUP_SCHEMA, table)
                action = "created"
            if dst_n != src_n:
                raise RuntimeError(f"row-count mismatch for {table}: src={src_n} backup={dst_n}")
            cur.execute(
                "SELECT pg_total_relation_size(%s::regclass)",
                (f'"{BACKUP_SCHEMA}"."{table}"',),
            )
            size_bytes = int(cur.fetchone()[0])
    finally:
        conn.close()
    return {
        "table": table,
        "backup": f"{BACKUP_SCHEMA}.{table}",
        "action": action,
        "source_rows": src_n,
        "backup_rows": dst_n,
        "rows_match": src_n == dst_n,
        "backup_bytes": size_bytes,
        "elapsed_s": round(time.time() - started, 1),
    }


def _recorded_counts(paths: tuple[Path, ...]) -> dict[str, int]:
    """Row counts of the sources at backup time, from the backup JSON logs."""
    out: dict[str, int] = {}
    for path in paths:
        if path.exists():
            for rec in json.loads(path.read_text(encoding="utf-8")):
                out[rec["table"]] = int(rec["source_rows"])
    return out


def verify_backups(
    tables: tuple[str, ...] = DERIVED_TABLES,
    *,
    exact: bool = True,
    recorded_logs: tuple[Path, ...] = (
        Path("model_stgcn/etl_runs/fixb_2026-10-05/backup.json"),
        Path("model_stgcn/etl_runs/fixb_2026-10-05/backup_team_phase_state.json"),
    ),
) -> list[dict]:
    """Pre-flight for the rerun: every backup table exists and still has the row count that its
    source had when the copy was made (taken from the backup logs, so it stays valid after the
    sources have been rewritten).  ``exact`` additionally compares with the live source, which is
    only meaningful before the first overwrite."""
    recorded = _recorded_counts(recorded_logs)
    out = []
    conn = connect(autocommit=True)
    try:
        with conn.cursor() as cur:
            for table in tables:
                if not _exists(cur, BACKUP_SCHEMA, table):
                    out.append({"table": table, "ok": False, "reason": "backup table missing"})
                    continue
                backup_n = _count(cur, BACKUP_SCHEMA, table)
                rec = {"table": table, "backup_rows": backup_n, "ok": True}
                if table in recorded:
                    rec["recorded_source_rows"] = recorded[table]
                    rec["ok"] = backup_n == recorded[table]
                    if not rec["ok"]:
                        rec["reason"] = (
                            "backup row count differs from the count recorded at backup time"
                        )
                elif exact:
                    rec["ok"] = False
                    rec["reason"] = "no recorded count for this backup"
                if exact and rec["ok"]:
                    src_n = _count(cur, "stdb", table)
                    rec["source_rows"] = src_n
                    rec["ok"] = src_n == backup_n
                    if not rec["ok"]:
                        rec["reason"] = "source/backup row counts differ"
                out.append(rec)
    finally:
        conn.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", nargs="+", default=list(DERIVED_TABLES))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    unknown = sorted(set(args.tables) - set(DERIVED_TABLES))
    if unknown:
        ap.error(f"not a Fix B derived table: {unknown}")

    results = []
    for table in args.tables:
        res = backup_table(table)
        results.append(res)
        logger.info("backup result: %s", res)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    ok = all(r["rows_match"] for r in results)
    logger.info("ALL BACKUPS %s", "VERIFIED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
