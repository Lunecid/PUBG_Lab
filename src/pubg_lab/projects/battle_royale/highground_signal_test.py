"""NOW-WORK step 1: does high-ground actually predict next-phase elimination?

For proximate (engagement-range) team pairs in a phase, test whether the
LOWER-elevation team is eliminated more often before the next phase. If yes,
building a directed high-ground edge channel is justified; if not, it won't help.
Read-only; samples matches to avoid scanning the 191GB proximity table fully.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))
import sys
from pubg_lab.data.stdb.conn import connect

N_MATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 600
DIST_M = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0


def main():
    with connect() as conn:
        cur = conn.cursor()
        # sample squad matches that have snapshot features
        cur.execute(
            """
            SELECT DISTINCT s.match_id
            FROM stdb.team_snapshot_features s
            JOIN stdb.match_phase_global g ON g.match_id = s.match_id
            LIMIT %s
        """,
            (N_MATCHES,),
        )
        mids = [r[0] for r in cur.fetchall()]
        print(f"sampled {len(mids)} matches; engagement distance <= {DIST_M} m")

        q = """
        WITH tz AS (
            SELECT match_id, team_id, phase_idx, AVG(pos_z) AS z
            FROM stdb.team_snapshot_features
            WHERE match_id = ANY(%(mids)s)
            GROUP BY 1,2,3
        ),
        last_alive AS (
            SELECT match_id, team_id, MAX(phase_idx) AS last_phase
            FROM stdb.team_snapshot_features
            WHERE match_id = ANY(%(mids)s)
            GROUP BY 1,2
        ),
        snap_phase AS (
            SELECT DISTINCT match_id, snapshot_t, phase_idx
            FROM stdb.team_snapshot_features
            WHERE match_id = ANY(%(mids)s)
        ),
        pairs AS (
            SELECT sp.match_id, sp.phase_idx, tp.team_a, tp.team_b,
                   MIN(tp.distance_m) AS min_d,
                   MAX(tp.encounter_recently::int) AS enc
            FROM stdb.team_proximity tp
            JOIN snap_phase sp
              ON sp.match_id = tp.match_id AND sp.snapshot_t = tp.snapshot_t
            WHERE tp.match_id = ANY(%(mids)s)
            GROUP BY 1,2,3,4
        )
        SELECT p.phase_idx, p.min_d, p.enc,
               za.z AS z_a, zb.z AS z_b,
               (la.last_phase = p.phase_idx)::int AS elim_a,
               (lb.last_phase = p.phase_idx)::int AS elim_b
        FROM pairs p
        JOIN tz za ON za.match_id=p.match_id AND za.team_id=p.team_a AND za.phase_idx=p.phase_idx
        JOIN tz zb ON zb.match_id=p.match_id AND zb.team_id=p.team_b AND zb.phase_idx=p.phase_idx
        JOIN last_alive la ON la.match_id=p.match_id AND la.team_id=p.team_a
        JOIN last_alive lb ON lb.match_id=p.match_id AND lb.team_id=p.team_b
        WHERE p.min_d <= %(dist)s
        """
        cur.execute(q, {"mids": mids, "dist": DIST_M})
        rows = cur.fetchall()
        cur.close()

    print(f"proximate team-pair-phases: {len(rows)}")

    # discordant pairs: exactly one team eliminated next phase
    import math

    def analyze(rows, label, enc_only=False):
        low_dies = 0  # lower-elevation team is the eliminated one (supports high-ground)
        high_dies = 0  # higher-elevation team eliminated (against)
        tie = 0
        for ph, d, enc, za, zb, ea, eb in rows:
            if enc_only and not enc:
                continue
            if ea + eb != 1:  # need exactly one eliminated
                continue
            if za == zb:
                tie += 1
                continue
            elim_is_a = ea == 1
            elim_z = za if elim_is_a else zb
            other_z = zb if elim_is_a else za
            if elim_z < other_z:
                low_dies += 1
            else:
                high_dies += 1
        n = low_dies + high_dies
        if n == 0:
            print(f"  [{label}] no discordant pairs")
            return
        frac = low_dies / n
        # binomial test vs 0.5 (normal approx)
        se = math.sqrt(0.25 / n)
        z = (frac - 0.5) / se
        print(
            f"  [{label}] discordant n={n} | lower-elev dies {low_dies} ({frac:.3f}) vs higher {high_dies} "
            f"| z={z:+.2f} {'(SIG高지우위)' if abs(z) > 1.96 else '(ns)'}  ties={tie}"
        )

    print("\n=== 고지 우위 검정 (낮은 팀이 더 죽으면 frac>0.5) ===")
    analyze(rows, "ALL proximate <=%dm" % DIST_M)
    analyze(rows, "engagement (encounter_recently)", enc_only=True)
    print("\n=== phase별 ===")
    for ph in sorted(set(r[0] for r in rows)):
        analyze([r for r in rows if r[0] == ph], f"phase {ph}")


if __name__ == "__main__":
    main()
