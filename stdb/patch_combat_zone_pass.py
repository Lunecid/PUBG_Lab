"""
Patch stdb.team_phase_exposure.combat_zone_pass_count to remove the
cross-cut leak (FEATURE_OVERLAP / snoopy plan #2).

The old ETL counted combat events by floor(event_t/10)*10 == snapshot_t with no
phase-window check, so a team's last snapshot of phase k could absorb up to ~10s
of phase-(k+1) (post-cut) combat. This recomputes the count with
event_t BETWEEN phase_start_s AND phase_end_s and UPDATEs only that one column
(no trajectory rebuild). The ETL itself is already fixed for future builds.

Idempotent UPDATE; match-scoped. Mirrors the network/habitat batch runner.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time

from stdb.conn import connect

logger = logging.getLogger("stdb.patch_combat_zone_pass")

PATCH_SQL = """
WITH phase_bounds AS (
    SELECT phase_idx, phase_start_s, phase_end_s
      FROM stdb.v_match_phase_bounds
     WHERE match_id = %(m)s AND phase_idx > 0
),
snaps AS (
    SELECT b.phase_idx, mo.snapshot_t, ST_Force2D(mo.pos) AS pos2d
      FROM phase_bounds b
      JOIN stdb.moving_objects mo
        ON mo.match_id = %(m)s AND mo.agent_kind = 'team' AND mo.is_alive
       AND mo.snapshot_t BETWEEN b.phase_start_s AND b.phase_end_s
),
team_snaps AS (
    SELECT b.phase_idx, mo.team_id, mo.snapshot_t, ST_Force2D(mo.pos) AS pos2d
      FROM phase_bounds b
      JOIN stdb.moving_objects mo
        ON mo.match_id = %(m)s AND mo.agent_kind = 'team' AND mo.is_alive
       AND mo.snapshot_t BETWEEN b.phase_start_s AND b.phase_end_s
),
combat_points AS (
    SELECT event_id, event_t,
           floor(event_t / 10.0)::int * 10 AS snapshot_t,
           COALESCE(ST_Force2D(victim_pos), ST_Force2D(attacker_pos)) AS event_pos
      FROM stdb.spatial_events
     WHERE match_id = %(m)s
       AND event_kind IN ('kill', 'damage', 'groggy')
       AND COALESCE(victim_pos, attacker_pos) IS NOT NULL
),
combat_pass AS (
    SELECT s.team_id, s.phase_idx, count(DISTINCT cp.event_id) AS cnt
      FROM team_snaps s
      JOIN phase_bounds pb ON pb.phase_idx = s.phase_idx
      JOIN combat_points cp
        ON cp.snapshot_t = s.snapshot_t
       AND cp.event_t BETWEEN pb.phase_start_s AND pb.phase_end_s
       AND ST_DWithin(s.pos2d, cp.event_pos, 200.0)
     GROUP BY s.team_id, s.phase_idx
),
universe AS (
    SELECT team_id, phase_idx FROM stdb.team_phase_exposure WHERE match_id = %(m)s
)
UPDATE stdb.team_phase_exposure e
   SET combat_zone_pass_count = coalesce(cp.cnt, 0)
  FROM universe u
  LEFT JOIN combat_pass cp ON cp.team_id = u.team_id AND cp.phase_idx = u.phase_idx
 WHERE e.match_id = %(m)s AND e.team_id = u.team_id AND e.phase_idx = u.phase_idx
"""


def patch_match(cur, match_id: str) -> int:
    cur.execute(PATCH_SQL, {"m": match_id})
    return int(cur.rowcount)


def select_match_ids(*, limit, match_id, shards=1, shard_index=0):
    if match_id:
        return [match_id]
    with connect() as conn:
        with conn.cursor() as cur:
            sql = "SELECT DISTINCT match_id FROM stdb.team_phase_exposure ORDER BY match_id"
            if limit:
                sql += " LIMIT %s"
                cur.execute(sql, (limit,))
            else:
                cur.execute(sql)
            ids = [r[0] for r in cur.fetchall()]
    if shards > 1:
        ids = [m for m in ids if int(hashlib.md5(m.encode()).hexdigest()[:8], 16) % shards == shard_index]
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match-id")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ids = select_match_ids(limit=args.limit, match_id=args.match_id, shards=args.shards, shard_index=args.shard_index)
    logger.info("patching %s matches (shard %s/%s)", len(ids), args.shard_index, args.shards)
    started = time.time(); ok = failed = 0
    for idx, mid in enumerate(ids, 1):
        conn = connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    patch_match(cur, mid)
            ok += 1
            if idx % 500 == 0 or idx == len(ids):
                logger.info("[%s/%s] patched", idx, len(ids))
        except Exception:
            conn.rollback(); failed += 1
            logger.exception("FAILED %s", mid)
        finally:
            conn.close()
    logger.info("done ok=%s failed=%s elapsed=%.1fs", ok, failed, time.time() - started)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
