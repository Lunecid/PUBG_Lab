"""Ground-truth phase timing/zone profile for squad matches from the ST-DB.

Answers: what real-game minute does each cut phase correspond to, and what is
the zone/survivor state there (for e-sports situational interpretation).
Read-only aggregation over stdb.v_team_phase_feature_v2.
"""
from __future__ import annotations
import json
from stdb.conn import connect

TABLE = ("stdb", "v_team_phase_feature_v2")


def discover_columns(cur):
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
        TABLE,
    )
    return [r[0] for r in cur.fetchall()]


def pick(cols, *cands):
    for c in cands:
        if c in cols:
            return c
    # substring fallback
    for c in cols:
        for cand in cands:
            if cand in c:
                return c
    return None


def main():
    with connect() as conn:
        cur = conn.cursor()
        cols = discover_columns(cur)
        # candidate semantic columns
        col_start = pick(cols, "phase_start_s")
        col_end = pick(cols, "phase_end_s")
        col_dur = pick(cols, "phase_duration_s")
        col_radius = pick(cols, "safe_radius_m", "cur_safe_radius_m", "next_safe_radius_m", "safe_radius")
        col_alive_t = pick(cols, "alive_teams", "teams_alive", "n_alive_teams", "alive_team_count")
        col_alive_p = pick(cols, "alive_players_end", "alive_players", "players_alive", "n_alive_players")
        print("# resolved columns:")
        for nm, c in [("start", col_start), ("end", col_end), ("dur", col_dur),
                      ("safe_radius", col_radius), ("alive_teams", col_alive_t),
                      ("alive_players", col_alive_p)]:
            print(f"   {nm:14} -> {c}")
        # show any safe/zone/alive-ish columns for transparency
        zone_like = [c for c in cols if any(k in c for k in
                     ("safe", "radius", "zone", "circle", "alive", "blue", "ring", "area"))]
        print("# zone/alive-like columns available:", zone_like)

        sel = ["phase_idx", "count(DISTINCT match_id) AS n_match"]
        def agg(col, alias):
            if col:
                sel.append(f"avg({col}) AS {alias}_avg")
                sel.append(f"percentile_cont(0.5) WITHIN GROUP (ORDER BY {col}) AS {alias}_med")
        agg(col_start, "start")
        agg(col_end, "end")
        agg(col_dur, "dur")
        agg(col_radius, "radius")
        agg(col_alive_t, "aliveT")
        agg(col_alive_p, "aliveP")
        q = (
            "SELECT " + ", ".join(sel) +
            f" FROM {TABLE[0]}.{TABLE[1]} "
            "WHERE game_mode = 'squad' "
            "GROUP BY phase_idx ORDER BY phase_idx"
        )
        cur.execute(q)
        rows = cur.fetchall()
        names = [d[0] for d in cur.description]
        print("\n# squad phase profile (avg / median):")
        print("\t".join(names))
        for r in rows:
            out = []
            for v in r:
                if v is None:
                    out.append("-")
                elif isinstance(v, float):
                    out.append(f"{v:.1f}")
                else:
                    out.append(str(v))
            print("\t".join(out))
        cur.close()


if __name__ == "__main__":
    main()
