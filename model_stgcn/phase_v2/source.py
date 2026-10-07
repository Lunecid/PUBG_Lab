"""Raw-telemetry source of ONE match on the LogMatchStart clock W (DB-free container + read-only fetch).

Every temporal record carries ``w`` = ``event_time - telem_match_start.event_time`` in seconds (float64, one
clock for positions, game states, kills, damage, groggy, item events and phase changes).  The container is the
only thing the pure builder (``model_stgcn.phase_v2.builder``) sees: it never touches the database, so the
prefix-replay test can hand it a source that was truncated either in memory (:func:`truncate`) or by the
database itself (``fetch_match_source(..., cut_w=a_k)`` adds ``W < a_k`` to every temporal query).

Static information that is known before the first shot (roster, team ids, the zone schedule) is not temporal and
is never truncated.  Post-match summaries (``final_rank``, ``won``, ``death_type``, ``time_survived``) are static
fields too, but only the label side of the builder reads them (never any input feature).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

CM_TO_M = 100.0
# vehicle types that do not count as "in a vehicle" (stdb/rq1_feature_etl.py + stdb/etl.py)
AIRCRAFT = ("TransportAircraft", "DummyTransportAircraft_C")


def _empty(dtypes: dict[str, Any]) -> dict[str, np.ndarray]:
    return {k: np.zeros(0, dtype=v) for k, v in dtypes.items()}


PHASE_EVENT_DTYPES = {"w": np.float64, "is_game": np.float64}
GAME_STATE_DTYPES = {
    "w": np.float64, "n_teams": np.float64, "n_players": np.float64,
    "safe_x": np.float64, "safe_y": np.float64, "safe_r": np.float64,
    "poison_x": np.float64, "poison_y": np.float64, "poison_r": np.float64,
}  # fmt: skip
POS_DTYPES = {
    "w": np.float64, "g": np.int64, "acct": np.int64, "team": np.int64, "health": np.float64,
    "x": np.float64, "y": np.float64, "z": np.float64, "veh": np.bool_, "vspeed": np.float64,
}  # fmt: skip
KILL_DTYPES = {
    "w": np.float64, "victim": np.int64, "killer": np.int64, "victim_team": np.int64,
    "vx": np.float64, "vy": np.float64, "kx": np.float64, "ky": np.float64, "rank": np.int64,
}  # fmt: skip
DAMAGE_DTYPES = {
    "w": np.float64, "attacker": np.int64, "victim": np.int64, "damage": np.float64,
    "ax": np.float64, "ay": np.float64, "vx": np.float64, "vy": np.float64,
}  # fmt: skip
GROGGY_DTYPES = {
    "w": np.float64, "attacker": np.int64, "victim": np.int64,
    "ax": np.float64, "ay": np.float64, "vx": np.float64, "vy": np.float64,
}  # fmt: skip
ITEM_DTYPES = {
    "w": np.float64, "team": np.int64, "item": object, "cat": object, "sub": object,
    "stack": np.int64, "carepackage": np.bool_, "lootbox": np.bool_,
}  # fmt: skip

ITEM_KINDS = ("pickup", "equip", "use")


@dataclass
class MatchSource:
    """Plain numpy / python container (see the module docstring).  ``cut_w`` is not None on truncated copies."""

    match_id: str
    map_name: str
    game_mode: str
    # ---- static -------------------------------------------------------------------------------------
    team_ids: np.ndarray  # (N,) int64 ascending (stdb.teams)
    team_num_players: np.ndarray  # (N,) float64 (stdb.teams.num_players)
    team_final_rank: np.ndarray  # (N,) float64 (NaN = unknown); label side only
    team_won: np.ndarray  # (N,) bool; label side only
    roster: list[tuple[int, str, str | None, float | None]]  # (team_id, player_id, death_type, time_survived)
    match_type: str | None = None
    schedule: dict[int, float] = field(default_factory=dict)  # analysis phase k -> scheduled seconds
    schedule_source: str = "missing"
    # ---- temporal (all on W) ------------------------------------------------------------------------
    end_w: float | None = None  # aligned LogMatchEnd (C_m); None on truncated copies
    phase_events: dict[str, np.ndarray] = field(default_factory=lambda: _empty(PHASE_EVENT_DTYPES))
    gs: dict[str, np.ndarray] = field(default_factory=lambda: _empty(GAME_STATE_DTYPES))
    pos: dict[str, np.ndarray] = field(default_factory=lambda: _empty(POS_DTYPES))
    kills: dict[str, np.ndarray] = field(default_factory=lambda: _empty(KILL_DTYPES))
    damage: dict[str, np.ndarray] = field(default_factory=lambda: _empty(DAMAGE_DTYPES))
    groggy: dict[str, np.ndarray] = field(default_factory=lambda: _empty(GROGGY_DTYPES))
    items: dict[str, dict[str, np.ndarray]] = field(
        default_factory=lambda: {k: _empty(ITEM_DTYPES) for k in ITEM_KINDS}
    )
    accounts: list[str] = field(default_factory=list)  # account index -> account id
    cut_w: float | None = None

    @property
    def n_teams(self) -> int:
        return int(len(self.team_ids))

    def account_index(self) -> dict[str, int]:
        return {a: i for i, a in enumerate(self.accounts)}


def _take(table: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in table.items()}


def truncate(src: MatchSource, cut_w: float) -> MatchSource:
    """The source restricted to records with ``w < cut_w`` (strict).  Static fields are shared, ``end_w`` is
    dropped (a truncated log has no known end), ``cut_w`` is recorded."""
    c = float(cut_w)
    return replace(
        src,
        end_w=None,
        phase_events=_take(src.phase_events, src.phase_events["w"] < c),
        gs=_take(src.gs, src.gs["w"] < c),
        pos=_take(src.pos, src.pos["w"] < c),
        kills=_take(src.kills, src.kills["w"] < c),
        damage=_take(src.damage, src.damage["w"] < c),
        groggy=_take(src.groggy, src.groggy["w"] < c),
        items={k: _take(v, v["w"] < c) for k, v in src.items.items()},
        cut_w=c,
    )


def temporal_row_counts(src: MatchSource) -> dict[str, int]:
    """Number of records per temporal table (used by the prefix-replay test and the dump statistics)."""
    out = {
        "phase_events": len(src.phase_events["w"]),
        "game_states": len(src.gs["w"]),
        "positions": len(src.pos["w"]),
        "kills": len(src.kills["w"]),
        "damage": len(src.damage["w"]),
        "groggy": len(src.groggy["w"]),
    }
    for k, v in src.items.items():
        out[f"item_{k}"] = len(v["w"])
    return out


def max_record_time(src: MatchSource) -> float:
    """Largest ``w`` over every temporal table (-inf if empty)."""
    tabs = [src.phase_events, src.gs, src.pos, src.kills, src.damage, src.groggy, *src.items.values()]
    return max((float(t["w"].max()) for t in tabs if len(t["w"])), default=float("-inf"))


# --------------------------------------------------------------------------------------
# blue-zone schedule
# --------------------------------------------------------------------------------------
def parse_blue_zone_options(value: Any) -> list[dict]:
    """``telem_match_start.blue_zone_options`` is a jsonb *string* holding a JSON list (``"[]"`` in 100 % of the
    corpus).  Returns the list of option dicts, ``[]`` if empty / unparsable."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if isinstance(value, str):  # double encoded
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    return [o for o in value if isinstance(o, dict)]


def schedule_from_options(options: list[dict]) -> dict[int, float]:
    """Scheduled seconds of analysis phase k from the match's own zone options: wait + shrink of the zone the
    phase is mapped to (``warningDuration + releaseDuration`` of ``phaseNum == k``).

    Analysis phase k = isGame k.0 (zone k appears: waiting) + isGame k.5 (zone k shrinks) (PUBG telemetry docs,
    https://documentation.pubg.com/en/telemetry-objects.html#common).  The key names follow the LogMatchStart
    ``blueZoneCustomOptions`` JSON; no match in the corpus carries a non-empty list, so this path is exercised
    by a unit test only and an empty / incomplete list returns ``{}`` (the caller then falls back to the cohort
    median schedule and counts the match)."""
    out: dict[int, float] = {}
    for o in options:
        try:
            k = int(o["phaseNum"])
            out[k] = float(o["warningDuration"]) + float(o["releaseDuration"])
        except (KeyError, TypeError, ValueError):
            return {}
    return out


# --------------------------------------------------------------------------------------
# DB fetch (read-only SELECTs)
# --------------------------------------------------------------------------------------
def _arr(rows: list[tuple], idx: int, dtype) -> np.ndarray:
    return np.asarray([r[idx] for r in rows], dtype=dtype)


def _f(rows: list[tuple], idx: int) -> np.ndarray:
    """float64 column with None -> NaN."""
    return np.asarray([np.nan if r[idx] is None else r[idx] for r in rows], dtype=np.float64)


def fetch_match_source(
    cur,
    match_id: str,
    *,
    cut_w: float | None = None,
    schedule: dict[int, float] | None = None,
    schedule_source: str | None = None,
) -> MatchSource | None:
    """Read one match into a :class:`MatchSource`.

    ``cut_w`` (seconds on W) is applied INSIDE every temporal query (``extract(epoch from event_time - S) <
    cut_w``): the database never returns a record at or after the cut.  ``schedule`` is the fallback zone
    schedule (cohort median) used when the match's own ``blue_zone_options`` is empty.  Returns None if the
    match has no ``telem_match_start`` row or no teams.
    """
    cur.execute(
        """
        SELECT s.event_time, m.map_name, m.game_mode, s.blue_zone_options, pm.match_type,
               extract(epoch FROM (e.event_time - s.event_time))::float8
          FROM stdb.matches m
          JOIN pubg.telem_match_start s ON s.match_id = m.match_id
          JOIN pubg.matches pm ON pm.match_id = m.match_id
          LEFT JOIN pubg.telem_match_end e ON e.match_id = m.match_id
         WHERE m.match_id = %s
        """,
        (match_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    started_at, map_name, game_mode, bz_opts, match_type, end_w = row
    cur.execute(
        "SELECT team_id, num_players, final_rank, won FROM stdb.teams WHERE match_id = %s ORDER BY team_id",
        (match_id,),
    )
    trows = cur.fetchall()
    if not trows:
        return None
    team_ids = np.asarray([r[0] for r in trows], dtype=np.int64)
    team_num_players = np.asarray([np.nan if r[1] is None else r[1] for r in trows], dtype=np.float64)
    team_final_rank = np.asarray([np.nan if r[2] is None else r[2] for r in trows], dtype=np.float64)
    team_won = np.asarray([bool(r[3]) for r in trows], dtype=bool)

    cur.execute(
        """
        SELECT r.team_id, p.player_id, p.death_type, p.time_survived::float8
          FROM pubg.participants p
          JOIN pubg.rosters r ON r.match_id = p.match_id AND r.roster_id = p.roster_id
         WHERE p.match_id = %s AND r.team_id IS NOT NULL
         ORDER BY r.team_id, p.player_id
        """,
        (match_id,),
    )
    roster = [(int(t), str(pid), dt, None if ts is None else float(ts)) for t, pid, dt, ts in cur.fetchall()]

    options = parse_blue_zone_options(bz_opts)
    own = schedule_from_options(options)
    if own:
        sched, sched_src = own, "match_options"
    elif schedule:
        sched, sched_src = dict(schedule), schedule_source or "cohort_median"
    else:
        sched, sched_src = {}, "missing"

    wexpr = "extract(epoch FROM (event_time - %(S)s))::float8"
    cutsql = f" AND {wexpr} < %(cut)s" if cut_w is not None else ""
    prm = {"mid": match_id, "S": started_at, "cut": None if cut_w is None else float(cut_w)}

    acct_index: dict[str, int] = {}
    accounts: list[str] = []

    def aidx(a: str | None) -> int:
        if a is None:
            return -1
        i = acct_index.get(a)
        if i is None:
            i = len(accounts)
            acct_index[a] = i
            accounts.append(a)
        return i

    for _t, pid, _dt, _ts in roster:
        aidx(pid)

    # --- phase changes ------------------------------------------------------------------------------
    cur.execute(
        f"SELECT {wexpr}, is_game::float8 FROM pubg.telem_phase_changes WHERE match_id = %(mid)s{cutsql} "
        "ORDER BY event_time, id",
        prm,
    )
    rows = cur.fetchall()
    phase_events = {"w": _arr(rows, 0, np.float64), "is_game": _arr(rows, 1, np.float64)}

    # --- game states --------------------------------------------------------------------------------
    cur.execute(
        f"""
        SELECT {wexpr}, num_alive_teams, num_alive_players,
               safe_zone_x, safe_zone_y, safe_zone_radius, poison_zone_x, poison_zone_y, poison_zone_radius
          FROM pubg.telem_game_states WHERE match_id = %(mid)s{cutsql} ORDER BY event_time, id
        """,
        prm,
    )
    rows = cur.fetchall()
    gs = {
        "w": _arr(rows, 0, np.float64),
        "n_teams": _f(rows, 1),
        "n_players": _f(rows, 2),
        "safe_x": _f(rows, 3) / CM_TO_M,
        "safe_y": _f(rows, 4) / CM_TO_M,
        "safe_r": _f(rows, 5) / CM_TO_M,
        "poison_x": _f(rows, 6) / CM_TO_M,
        "poison_y": _f(rows, 7) / CM_TO_M,
        "poison_r": _f(rows, 8) / CM_TO_M,
    }
    for pre in ("safe", "poison"):  # a zone with a missing centre or radius <= 0 does not exist (stdb.etl)
        bad = ~(np.isfinite(gs[f"{pre}_x"]) & np.isfinite(gs[f"{pre}_y"]) & (gs[f"{pre}_r"] > 0))
        for c in ("x", "y", "r"):
            gs[f"{pre}_{c}"] = np.where(bad, np.nan, gs[f"{pre}_{c}"])

    # --- positions (the stdb ETL filters) -----------------------------------------------------------
    cur.execute(
        f"""
        SELECT {wexpr}, elapsed_time, account_id, team_id, health::float8,
               pos_x::float8, pos_y::float8, pos_z::float8, vehicle_type, vehicle_speed::float8
          FROM pubg.telem_positions
         WHERE match_id = %(mid)s AND account_id IS NOT NULL AND team_id IS NOT NULL
           AND elapsed_time IS NOT NULL AND is_game >= 0.1 AND pos_x IS NOT NULL AND pos_y IS NOT NULL
           {cutsql}
         ORDER BY event_time, id
        """,
        prm,
    )
    rows = cur.fetchall()
    pos = {
        "w": _arr(rows, 0, np.float64),
        "g": _arr(rows, 1, np.int64),
        "acct": np.asarray([aidx(r[2]) for r in rows], dtype=np.int64),
        "team": _arr(rows, 3, np.int64),
        "health": _f(rows, 4),
        "x": _f(rows, 5) / CM_TO_M,
        "y": _f(rows, 6) / CM_TO_M,
        "z": np.nan_to_num(_f(rows, 7), nan=0.0) / CM_TO_M,
        "veh": np.asarray([(r[8] is not None and r[8] not in AIRCRAFT) for r in rows], dtype=bool),
        "vspeed": _f(rows, 9),
    }

    # --- kills --------------------------------------------------------------------------------------
    cur.execute(
        f"""
        SELECT {wexpr}, victim_id, killer_id, victim_team_id,
               victim_x::float8, victim_y::float8, killer_x::float8, killer_y::float8, victim_rank
          FROM pubg.telem_kills WHERE match_id = %(mid)s AND victim_id IS NOT NULL{cutsql}
         ORDER BY event_time, id
        """,
        prm,
    )
    rows = cur.fetchall()
    kills = {
        "w": _arr(rows, 0, np.float64),
        "victim": np.asarray([aidx(r[1]) for r in rows], dtype=np.int64),
        "killer": np.asarray([aidx(r[2]) for r in rows], dtype=np.int64),
        "victim_team": np.asarray([-1 if r[3] is None else r[3] for r in rows], dtype=np.int64),
        "vx": _f(rows, 4) / CM_TO_M,
        "vy": _f(rows, 5) / CM_TO_M,
        "kx": _f(rows, 6) / CM_TO_M,
        "ky": _f(rows, 7) / CM_TO_M,
        "rank": np.asarray([0 if r[8] is None else r[8] for r in rows], dtype=np.int64),
    }

    # --- damage -------------------------------------------------------------------------------------
    cur.execute(
        f"""
        SELECT {wexpr}, attacker_id, victim_id, damage::float8,
               attacker_x::float8, attacker_y::float8, victim_x::float8, victim_y::float8
          FROM pubg.telem_damage WHERE match_id = %(mid)s{cutsql} ORDER BY event_time, id
        """,
        prm,
    )
    rows = cur.fetchall()
    damage = {
        "w": _arr(rows, 0, np.float64),
        "attacker": np.asarray([aidx(r[1]) for r in rows], dtype=np.int64),
        "victim": np.asarray([aidx(r[2]) for r in rows], dtype=np.int64),
        "damage": _f(rows, 3),
        "ax": _f(rows, 4) / CM_TO_M,
        "ay": _f(rows, 5) / CM_TO_M,
        "vx": _f(rows, 6) / CM_TO_M,
        "vy": _f(rows, 7) / CM_TO_M,
    }

    # --- groggy -------------------------------------------------------------------------------------
    cur.execute(
        f"""
        SELECT {wexpr}, attacker_id, victim_id,
               attacker_x::float8, attacker_y::float8, victim_x::float8, victim_y::float8
          FROM pubg.telem_groggy WHERE match_id = %(mid)s{cutsql} ORDER BY event_time, id
        """,
        prm,
    )
    rows = cur.fetchall()
    groggy = {
        "w": _arr(rows, 0, np.float64),
        "attacker": np.asarray([aidx(r[1]) for r in rows], dtype=np.int64),
        "victim": np.asarray([aidx(r[2]) for r in rows], dtype=np.int64),
        "ax": _f(rows, 3) / CM_TO_M,
        "ay": _f(rows, 4) / CM_TO_M,
        "vx": _f(rows, 5) / CM_TO_M,
        "vy": _f(rows, 6) / CM_TO_M,
    }

    # --- item events --------------------------------------------------------------------------------
    items: dict[str, dict[str, np.ndarray]] = {}
    item_sql = {
        "pickup": "SELECT {w}, team_id, item_id, category, sub_category, coalesce(stack_count, 1), "
                  "(carepackage_id IS NOT NULL), (lootbox_owner_team_id IS NOT NULL) "
                  "FROM pubg.telem_item_pickup WHERE match_id = %(mid)s AND team_id IS NOT NULL{cut} "
                  "ORDER BY event_time, id",
        "equip": "SELECT {w}, team_id, item_id, category, sub_category, 1, false, false "
                 "FROM pubg.telem_item_equip WHERE match_id = %(mid)s AND team_id IS NOT NULL{cut} "
                 "ORDER BY event_time, id",
        "use": "SELECT {w}, team_id, item_id, category, sub_category, coalesce(stack_count, 1), false, false "
               "FROM pubg.telem_item_use WHERE match_id = %(mid)s AND team_id IS NOT NULL{cut} "
               "ORDER BY event_time, id",
    }  # fmt: skip
    for kind in ITEM_KINDS:
        cur.execute(item_sql[kind].format(w=wexpr, cut=cutsql), prm)
        rows = cur.fetchall()
        items[kind] = {
            "w": _arr(rows, 0, np.float64),
            "team": _arr(rows, 1, np.int64),
            "item": np.asarray([r[2] for r in rows], dtype=object),
            "cat": np.asarray([r[3] for r in rows], dtype=object),
            "sub": np.asarray([r[4] for r in rows], dtype=object),
            "stack": _arr(rows, 5, np.int64),
            "carepackage": _arr(rows, 6, np.bool_),
            "lootbox": _arr(rows, 7, np.bool_),
        }

    # canonical account order (sorted ids): the relative order of two accounts is then the same in a full and in a
    # truncated source, so every order-dependent floating point accumulation is reproduced bit for bit
    order = sorted(range(len(accounts)), key=lambda i: accounts[i])
    remap = np.empty(len(accounts), dtype=np.int64)
    remap[np.asarray(order, dtype=np.int64)] = np.arange(len(accounts), dtype=np.int64)
    accounts = [accounts[i] for i in order]

    def _re(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        ok = a >= 0
        out[ok] = remap[a[ok]]
        return out

    pos["acct"] = _re(pos["acct"])
    kills["victim"], kills["killer"] = _re(kills["victim"]), _re(kills["killer"])
    damage["attacker"], damage["victim"] = _re(damage["attacker"]), _re(damage["victim"])
    groggy["attacker"], groggy["victim"] = _re(groggy["attacker"]), _re(groggy["victim"])

    return MatchSource(
        match_id=match_id,
        map_name=map_name,
        game_mode=game_mode,
        team_ids=team_ids,
        team_num_players=team_num_players,
        team_final_rank=team_final_rank,
        team_won=team_won,
        roster=roster,
        match_type=match_type,
        schedule=sched,
        schedule_source=sched_src,
        end_w=None if (end_w is None or cut_w is not None) else float(end_w),
        phase_events=phase_events,
        gs=gs,
        pos=pos,
        kills=kills,
        damage=damage,
        groggy=groggy,
        items=items,
        accounts=accounts,
        cut_w=None if cut_w is None else float(cut_w),
    )
