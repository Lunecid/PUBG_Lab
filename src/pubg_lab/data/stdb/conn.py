"""
stdb/conn.py
============
psycopg2 connection helpers for the STDB layer.

Defaults match main.py's DB_CONFIG so any existing env where main.py works will
also work for stdb.  Override via PG_HOST / PG_PORT / PG_DBNAME / PG_USER /
PG_PASSWORD env vars.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import contextlib
import os
from pathlib import Path

import psycopg2
import psycopg2.extensions
import psycopg2.extras

DB_CONFIG = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": int(os.environ.get("PG_PORT", "5432")),
    "dbname": os.environ.get("PG_DBNAME", "pubg_survival"),
    "user": os.environ.get("PG_USER", "postgres"),
}


def require_pg_password() -> str:
    """Resolve the DB password from the environment at connect time.

    Resolved lazily (not at import time) so modules that only need constants
    from stdb.queries can be imported without a database environment.
    """
    pw = os.environ.get("PG_PASSWORD")
    if not pw:
        raise RuntimeError(
            "PG_PASSWORD environment variable is not set. "
            "Export it before running anything that touches the database:\n"
            "  export PG_PASSWORD='<your password>'\n"
            "(see .env.example at the repo root)"
        )
    return pw


def _register_decimal_to_float() -> None:
    dec2float = psycopg2.extensions.new_type(
        psycopg2.extensions.DECIMAL.values,
        "DEC2FLOAT",
        lambda value, _curs: float(value) if value is not None else None,
    )
    psycopg2.extensions.register_type(dec2float)


_register_decimal_to_float()


def connect(*, autocommit: bool = False) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(password=require_pg_password(), **DB_CONFIG)
    if autocommit:
        conn.autocommit = True
    return conn


@contextlib.contextmanager
def get_conn(*, autocommit: bool = False):
    conn = connect(autocommit=autocommit)
    try:
        yield conn
    finally:
        conn.close()


@contextlib.contextmanager
def get_cur(*, autocommit: bool = False, dict_cursor: bool = False):
    with get_conn(autocommit=autocommit) as conn:
        if dict_cursor:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cur = conn.cursor()
        try:
            yield conn, cur
        finally:
            cur.close()


def apply_sql_file(path: str | Path) -> None:
    """Execute every statement in a .sql file (autocommit)."""
    sql = Path(path).read_text()
    with get_conn(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def healthcheck() -> dict:
    """Return basic server / postgis / stdb facts.  Read-only."""
    out: dict = {"config": {k: v for k, v in DB_CONFIG.items() if k != "password"}}
    with get_cur() as (_conn, cur):
        cur.execute("SELECT version()")
        out["server_version"] = cur.fetchone()[0]
        try:
            cur.execute("SELECT postgis_version()")
            out["postgis_version"] = cur.fetchone()[0]
        except Exception:
            cur.execute("ROLLBACK")
            out["postgis_version"] = None
        cur.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='stdb'")
        out["stdb_tables"] = cur.fetchone()[0]
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(healthcheck(), indent=2, default=str))
