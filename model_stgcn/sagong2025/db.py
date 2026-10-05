"""Read-only PostgreSQL access for the Sagong et al. (2025) feature extraction (SELECT only).

Credentials come from the environment (``PG_HOST`` / ``PG_PORT`` / ``PG_DBNAME`` or ``PG_DATABASE`` / ``PG_USER`` /
``PG_PASSWORD``); nothing is printed or stored.  The session is opened ``readonly=True`` (``default_transaction_read_only``).
"""

from __future__ import annotations

import os

import pandas as pd

from model_stgcn.sagong2025.features import MatchRaw

SQL_START = "SELECT match_id, extract(epoch FROM started_at)::float8 FROM stdb.matches WHERE match_id = ANY(%s)"
SQL_PART = """
SELECT p.match_id, p.player_id, r.team_id, coalesce(p.kills, 0), coalesce(p.damage_dealt, 0)::float8
  FROM pubg.participants p
  JOIN pubg.rosters r ON r.roster_id = p.roster_id AND r.match_id = p.match_id
 WHERE p.match_id = ANY(%s) AND r.team_id IS NOT NULL AND p.player_id IS NOT NULL
"""
SQL_POS = """
SELECT match_id, account_id, extract(epoch FROM event_time)::float8, pos_x::float8 / 100.0, pos_y::float8 / 100.0
  FROM pubg.telem_positions
 WHERE match_id = ANY(%s) AND is_game >= 0.1 AND account_id IS NOT NULL AND pos_x IS NOT NULL AND pos_y IS NOT NULL
"""
SQL_DMG = """
SELECT match_id, attacker_id, victim_id, damage::float8, extract(epoch FROM event_time)::float8
  FROM pubg.telem_damage
 WHERE match_id = ANY(%s) AND attacker_id IS NOT NULL AND victim_id IS NOT NULL AND damage > 0
"""
SQL_KILL = """
SELECT match_id, killer_id, victim_id, is_suicide, extract(epoch FROM event_time)::float8
  FROM pubg.telem_kills WHERE match_id = ANY(%s) AND victim_id IS NOT NULL
"""
SQL_PICK = """
SELECT match_id, account_id, event_type, extract(epoch FROM event_time)::float8
  FROM pubg.telem_item_pickup
 WHERE match_id = ANY(%s) AND event_type LIKE 'pickup%%' AND account_id IS NOT NULL
"""


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


def fetch_raw_batch(cur, match_ids: list[str]) -> dict[str, MatchRaw]:
    """Raw telemetry of several matches (one set of SELECTs); matches without ``started_at`` are left out."""
    start = dict(cur_rows(cur, SQL_START, match_ids))
    part = _frame(cur, SQL_PART, match_ids, ["match_id", "player_id", "team_id", "kills", "damage_dealt"])
    pos = _frame(cur, SQL_POS, match_ids, ["match_id", "acc", "ep", "x", "y"])
    dmg = _frame(cur, SQL_DMG, match_ids, ["match_id", "attacker", "victim", "damage", "ep"])
    kill = _frame(cur, SQL_KILL, match_ids, ["match_id", "killer", "victim", "is_suicide", "ep"])
    pick = _frame(cur, SQL_PICK, match_ids, ["match_id", "acc", "event_type", "ep"])
    by = {name: {m: g for m, g in df.groupby("match_id", sort=False)} for name, df in
          (("part", part), ("pos", pos), ("dmg", dmg), ("kill", kill), ("pick", pick))}
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
            official={"kills": float(p["kills"].sum()), "damage_dealt": float(p["damage_dealt"].sum())},
        )
    return out
