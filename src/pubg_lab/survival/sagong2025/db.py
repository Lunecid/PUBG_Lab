"""Read-only PostgreSQL access for the Sagong et al. (2025) feature extraction (SELECT only).

Credentials come from the environment (``PG_HOST`` / ``PG_PORT`` / ``PG_DBNAME`` or ``PG_DATABASE`` / ``PG_USER`` /
``PG_PASSWORD``); nothing is printed or stored.  The session is opened ``readonly=True`` (``default_transaction_read_only``).
"""

from __future__ import annotations

import os

import pandas as pd

from pubg_lab.survival.sagong2025.features import MatchRaw

SQL_START = "SELECT match_id, extract(epoch FROM started_at)::float8 FROM stdb.matches WHERE match_id = ANY(%s)"
SQL_PART = """
SELECT p.match_id, p.player_id, r.team_id, coalesce(p.kills, 0), coalesce(p.damage_dealt, 0)::float8
  FROM pubg.participants p
  JOIN pubg.rosters r ON r.roster_id = p.roster_id AND r.match_id = p.match_id
 WHERE p.match_id = ANY(%s) AND r.team_id IS NOT NULL AND p.player_id IS NOT NULL
"""

CLOCKS = ("epoch", "w")


def _ts(clock: str, alias: str) -> str:
    """Event time column: ``epoch`` (v1: seconds since 1970, ``start_epoch`` is subtracted later) or ``w`` (v2: seconds since
    LogMatchStart computed by the database exactly like the dump producer, ``extract(epoch FROM (event_time - started_at))``)."""
    if clock == "w":
        return f"extract(epoch FROM ({alias}.event_time - m.started_at))::float8"
    if clock == "epoch":
        return f"extract(epoch FROM {alias}.event_time)::float8"
    raise ValueError(f"clock must be one of {CLOCKS}, got {clock!r}")


def _join(clock: str, alias: str) -> str:
    return f"JOIN stdb.matches m ON m.match_id = {alias}.match_id" if clock == "w" else ""


def sql_pos(clock: str = "epoch") -> str:
    return f"""
SELECT x.match_id, x.account_id, {_ts(clock, "x")}, x.pos_x::float8 / 100.0, x.pos_y::float8 / 100.0
  FROM pubg.telem_positions x {_join(clock, "x")}
 WHERE x.match_id = ANY(%s) AND x.is_game >= 0.1 AND x.account_id IS NOT NULL AND x.pos_x IS NOT NULL AND x.pos_y IS NOT NULL
"""


def sql_dmg(clock: str = "epoch") -> str:
    return f"""
SELECT x.match_id, x.attacker_id, x.victim_id, x.damage::float8, {_ts(clock, "x")}
  FROM pubg.telem_damage x {_join(clock, "x")}
 WHERE x.match_id = ANY(%s) AND x.attacker_id IS NOT NULL AND x.victim_id IS NOT NULL AND x.damage > 0
"""


def sql_kill(clock: str = "epoch") -> str:
    return f"""
SELECT x.match_id, x.killer_id, x.victim_id, x.is_suicide, {_ts(clock, "x")}
  FROM pubg.telem_kills x {_join(clock, "x")} WHERE x.match_id = ANY(%s) AND x.victim_id IS NOT NULL
"""


def sql_pick(clock: str = "epoch") -> str:
    return f"""
SELECT x.match_id, x.account_id, x.event_type, {_ts(clock, "x")}
  FROM pubg.telem_item_pickup x {_join(clock, "x")}
 WHERE x.match_id = ANY(%s) AND x.event_type LIKE 'pickup%%' AND x.account_id IS NOT NULL
"""


# the v1 (epoch) statements under their old names
SQL_POS, SQL_DMG, SQL_KILL, SQL_PICK = sql_pos(), sql_dmg(), sql_kill(), sql_pick()


def connect_readonly(statement_timeout_s: int = 180):
    import psycopg2

    def env(*names, default=None):
        return next((os.environ[n] for n in names if os.environ.get(n)), default)

    conn = psycopg2.connect(
        host=env("PG_HOST", "PGHOST", default="localhost"),
        port=int(env("PG_PORT", "PGPORT", default="5432")),
        dbname=env("PG_DBNAME", "PG_DATABASE", "PGDATABASE", default="pubg_survival"),
        user=env("PG_USER", "PGUSER", default="postgres"),
        password=env("PG_PASSWORD", "PGPASSWORD"),
    )
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = '{int(statement_timeout_s)}s'")
    return conn


def cur_rows(cur, sql: str, ids: list[str]) -> list:
    cur.execute(sql, (ids,))
    return cur.fetchall()


def _frame(cur, sql: str, ids: list[str], columns: list[str]) -> pd.DataFrame:
    cur.execute(sql, (ids,))
    return pd.DataFrame(cur.fetchall(), columns=columns)


def fetch_raw_batch(cur, match_ids: list[str], clock: str = "epoch") -> dict[str, MatchRaw]:
    """Raw telemetry of several matches (one set of SELECTs); matches without ``started_at`` are left out.

    ``clock='epoch'`` (v1): ``ep`` columns are epoch seconds and ``MatchRaw.start_epoch`` is ``stdb.matches.started_at``.
    ``clock='w'`` (v2): ``ep`` columns are already seconds since LogMatchStart, computed by the database
    (``extract(epoch FROM (event_time - started_at))``, bit for bit the clock of the dump) and ``start_epoch`` is 0."""
    if clock not in CLOCKS:
        raise ValueError(f"clock must be one of {CLOCKS}, got {clock!r}")
    start = dict(cur_rows(cur, SQL_START, match_ids))
    if clock == "w":
        start = {m: 0.0 for m, v in start.items() if v is not None}
    part = _frame(
        cur, SQL_PART, match_ids, ["match_id", "player_id", "team_id", "kills", "damage_dealt"]
    )
    pos = _frame(cur, sql_pos(clock), match_ids, ["match_id", "acc", "ep", "x", "y"])
    dmg = _frame(cur, sql_dmg(clock), match_ids, ["match_id", "attacker", "victim", "damage", "ep"])
    kill = _frame(
        cur, sql_kill(clock), match_ids, ["match_id", "killer", "victim", "is_suicide", "ep"]
    )
    pick = _frame(cur, sql_pick(clock), match_ids, ["match_id", "acc", "event_type", "ep"])
    by = {
        name: {m: g for m, g in df.groupby("match_id", sort=False)}
        for name, df in (("part", part), ("pos", pos), ("dmg", dmg), ("kill", kill), ("pick", pick))
    }
    out: dict[str, MatchRaw] = {}
    for mid in match_ids:
        if mid not in start or start[mid] is None:
            continue
        p = by["part"].get(mid, part.iloc[0:0])
        team_of = {str(a): int(t) for a, t in zip(p["player_id"], p["team_id"])}
        out[mid] = MatchRaw(
            match_id=mid,
            start_epoch=float(start[mid]),
            team_of=team_of,
            pos=by["pos"].get(mid, pos.iloc[0:0]).drop(columns="match_id"),
            dmg=by["dmg"].get(mid, dmg.iloc[0:0]).drop(columns="match_id"),
            kill=by["kill"].get(mid, kill.iloc[0:0]).drop(columns="match_id"),
            pick=by["pick"].get(mid, pick.iloc[0:0]).drop(columns="match_id"),
            official={
                "kills": float(p["kills"].sum()),
                "damage_dealt": float(p["damage_dealt"].sum()),
            },
        )
    return out
