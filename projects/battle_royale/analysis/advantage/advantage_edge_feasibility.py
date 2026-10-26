"""Feasibility for the contact-gated, learned-advantage edge design.
Confirms: (1) edge cache coverage (proximity=loose gate, combat=strict gate),
(2) per-team-per-phase state for relative features (z/alive/health/zone),
(3) gate coverage: what fraction of at-risk team-phases actually HAVE a contact
edge to carry advantage. Read-only, sampled.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))
import sys
from pubg_lab.data.stdb.connection.conn import connect

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def main():
    with connect() as conn:
        cur = conn.cursor()

        # (1) edge cache global status
        cur.execute(
            "SELECT edge_type, count(*), count(DISTINCT match_id) FROM stdb.team_phase_edge_cache GROUP BY 1 ORDER BY 1"
        )
        print("=== (1) team_phase_edge_cache (전체) ===")
        for et, rows, m in cur.fetchall():
            print(f"  {et:11} rows={rows:>10,} matches={m:>7,}")

        # sample matches present in BOTH edge cache and snapshot state
        cur.execute(
            """
            SELECT DISTINCT e.match_id
            FROM stdb.team_phase_edge_cache e
            WHERE EXISTS (SELECT 1 FROM stdb.team_snapshot_features s WHERE s.match_id=e.match_id)
            LIMIT %s
        """,
            (N,),
        )
        mids = [r[0] for r in cur.fetchall()]
        print(f"\nsampled {len(mids)} matches with both edges + state")

        # (2) per-team-per-phase state availability (the relative-feature inputs)
        cur.execute(
            """
            SELECT count(*) AS rows,
                   count(pos_z) AS z, count(alive_n) AS alive,
                   count(mean_health) AS health, count(dist_blue_boundary_m) AS zone
            FROM (
              SELECT match_id, team_id, phase_idx,
                     avg(pos_z) pos_z, avg(alive_n) alive_n,
                     avg(mean_health) mean_health, avg(dist_blue_boundary_m) dist_blue_boundary_m
              FROM stdb.team_snapshot_features
              WHERE match_id = ANY(%s)
              GROUP BY 1,2,3
            ) q
        """,
            (mids,),
        )
        r = cur.fetchone()
        print("\n=== (2) per-(match,team,phase) 상대-상태 피처 가용성 ===")
        print(
            f"  team-phase rows={r[0]:,} | pos_z={r[1]:,} alive_n={r[2]:,} mean_health={r[3]:,} dist_blue_boundary={r[4]:,}"
        )

        # (3) gate coverage: of at-risk team-phases, how many have >=1 combat / >=1 proximity edge
        cur.execute(
            """
            WITH atrisk AS (   -- team-phases present in state (proxy for at-risk/alive)
                SELECT DISTINCT match_id, team_id, phase_idx
                FROM stdb.team_snapshot_features WHERE match_id = ANY(%s)
            ),
            comb AS (
                SELECT DISTINCT match_id, phase_idx, dst_team_id AS team_id
                FROM stdb.team_phase_edge_cache
                WHERE match_id = ANY(%s) AND edge_type='combat'
            ),
            prox AS (
                SELECT DISTINCT match_id, phase_idx, dst_team_id AS team_id
                FROM stdb.team_phase_edge_cache
                WHERE match_id = ANY(%s) AND edge_type='proximity'
            )
            SELECT a.phase_idx,
                   count(*) AS atrisk_n,
                   count(c.team_id) AS has_combat,
                   count(p.team_id) AS has_prox
            FROM atrisk a
            LEFT JOIN comb c ON c.match_id=a.match_id AND c.phase_idx=a.phase_idx AND c.team_id=a.team_id
            LEFT JOIN prox p ON p.match_id=a.match_id AND p.phase_idx=a.phase_idx AND p.team_id=a.team_id
            GROUP BY a.phase_idx ORDER BY a.phase_idx
        """,
            (mids, mids, mids),
        )
        print("\n=== (3) 접촉 게이트 커버리지 (at-risk team-phase 중 incoming 엣지 보유 비율) ===")
        print(f"  {'phase':>5}{'at-risk':>9}{'combat(strict)':>16}{'proximity(loose)':>18}")
        for ph, n, hc, hp in cur.fetchall():
            print(f"  {ph:>5}{n:>9,}{hc:>9,}({100 * hc / n:4.0f}%){hp:>11,}({100 * hp / n:4.0f}%)")
        cur.close()


if __name__ == "__main__":
    main()
