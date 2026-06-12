"""
DB에서 매치별 교전 이벤트 로드.

main.py의 connection 패턴을 그대로 사용. 각 매치에 대해:
  - telem_match_start로 기준 시각 획득
  - telem_damage 전체 이벤트 (attacker→victim, 시각, 데미지)
  - telem_kills도 함께 (탈락 시점 정확도 향상)
"""

import psycopg2
import psycopg2.extensions
from collections import defaultdict

from network_analysis.config import DB_CONFIG, SCHEMA, EXCLUDE_CAUSERS, require_pg_password


def get_conn():
    # Decimal → float 자동 변환
    DEC2FLOAT = psycopg2.extensions.new_type(
        psycopg2.extensions.DECIMAL.values,
        "DEC2FLOAT",
        lambda v, c: float(v) if v is not None else None,
    )
    psycopg2.extensions.register_type(DEC2FLOAT)
    return psycopg2.connect(password=require_pg_password(), **DB_CONFIG)


def load_match_start_time(cur, match_id):
    cur.execute(
        f"SELECT event_time FROM {SCHEMA}.telem_match_start "
        f"WHERE match_id = %s LIMIT 1",
        (match_id,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def load_damage_events(cur, match_id, match_start_time):
    """
    Returns: list of dict sorted by elapsed:
        {attacker_id, victim_id, elapsed, damage, causer}
    Self-damage, bluezone, None ids 제외.
    """
    cur.execute(
        f"""
        SELECT attacker_id, victim_id, event_time, damage, damage_causer
        FROM {SCHEMA}.telem_damage
        WHERE match_id = %s
          AND attacker_id IS NOT NULL
          AND victim_id IS NOT NULL
        ORDER BY event_time
        """,
        (match_id,),
    )
    out = []
    for a, v, t, d, c in cur.fetchall():
        if a == v:
            continue
        if c in EXCLUDE_CAUSERS:
            continue
        if t is None or match_start_time is None:
            continue
        elapsed = (t - match_start_time).total_seconds()
        if elapsed < 0:
            continue
        out.append({
            "attacker": a, "victim": v, "elapsed": elapsed,
            "damage": float(d or 0.0), "causer": c,
        })
    return out


def aggregate_combat_edges(damage_events, t_now, window_sec):
    """
    [t_now - window, t_now] 구간의 (attacker, victim) 쌍별 총 데미지 합산.

    Returns:
        dict: {(attacker_id, victim_id) -> total_damage}
    """
    t_lo = t_now - window_sec
    agg = defaultdict(float)
    for e in damage_events:
        if e["elapsed"] < t_lo:
            continue
        if e["elapsed"] > t_now:
            break  # events are sorted
        agg[(e["attacker"], e["victim"])] += e["damage"]
    return dict(agg)


def load_match_damage(match_id):
    """편의 함수: 단일 커넥션으로 매치의 모든 이벤트 로드."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        ms = load_match_start_time(cur, match_id)
        if ms is None:
            return {"start": None, "events": []}
        events = load_damage_events(cur, match_id, ms)
        return {"start": ms, "events": events}
    finally:
        conn.close()


def load_phase_intervals(match_id, match_duration_s=None, landing_offset_s=90.0):
    """Return match phase intervals as elapsed seconds.

    `telem_phase_changes.elapsed_time` is often null in the current DB, so the
    interval start is derived from `event_time - LogMatchStart.event_time`.
    Multiple rows can exist for one integer phase; the earliest row for each
    phase is treated as that phase's start.  The final phase ends at the match
    duration when available.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        match_start = load_match_start_time(cur, match_id)
        if match_start is None:
            return []
        cur.execute(
            f"""
            SELECT event_time, phase
            FROM {SCHEMA}.telem_phase_changes
            WHERE match_id = %s
              AND event_time IS NOT NULL
              AND phase IS NOT NULL
            ORDER BY event_time
            """,
            (match_id,),
        )
        starts = {}
        for event_time, phase in cur.fetchall():
            try:
                phase_i = int(float(phase))
            except (TypeError, ValueError):
                continue
            if phase_i <= 0:
                continue
            elapsed = (event_time - match_start).total_seconds()
            if elapsed < 0:
                continue
            starts[phase_i] = min(float(elapsed), starts.get(phase_i, float("inf")))

        if not starts:
            return []

        duration = float(match_duration_s or 0.0)
        phases = sorted(starts)
        intervals = []
        for i, phase_i in enumerate(phases):
            start_s = max(float(landing_offset_s), float(starts[phase_i]))
            if i + 1 < len(phases):
                end_s = float(starts[phases[i + 1]])
            else:
                end_s = duration if duration > start_s else start_s
            if end_s <= start_s:
                continue
            intervals.append({
                "phase": phase_i,
                "phase_start_s": start_s,
                "phase_end_s": end_s,
                "phase_duration_s": end_s - start_s,
            })
        return intervals
    finally:
        conn.close()


def load_team_ranked_stats(match_id):
    """Team-level ranked-performance ingredients from participants.

    PUBG ranked RP exact weights are not public, but PUBG Support states that
    kills, assists, and team placement are reflected in RP.  This helper returns
    those observable ingredients for a match.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                r.team_id,
                r.rank,
                COUNT(p.player_id) AS n_players,
                COALESCE(SUM(p.kills), 0) AS team_kills,
                COALESCE(SUM(p.assists), 0) AS team_assists,
                COALESCE(SUM(p.dbnos), 0) AS team_dbnos,
                COALESCE(SUM(p.damage_dealt), 0) AS team_damage_dealt
            FROM {SCHEMA}.rosters r
            JOIN {SCHEMA}.participants p
              ON r.roster_id = p.roster_id
             AND r.match_id = p.match_id
            WHERE r.match_id = %s
            GROUP BY r.team_id, r.rank
            """,
            (match_id,),
        )
        out = {}
        for row in cur.fetchall():
            team_id, rank, n_players, kills, assists, dbnos, damage = row
            out[int(team_id)] = {
                "ranked_team_rank": int(rank) if rank is not None else None,
                "ranked_n_players": int(n_players or 0),
                "team_kills": int(kills or 0),
                "team_assists": int(assists or 0),
                "team_dbnos": int(dbnos or 0),
                "team_damage_dealt": float(damage or 0.0),
            }
        return out
    finally:
        conn.close()
