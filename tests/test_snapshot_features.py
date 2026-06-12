"""Snapshot-resolution 팀 노드 피처 불변식 — stdb/snapshot_feature_etl.py (MR-STGNN Tier-S).

전부 실 DB. 트랜잭션 안에서 매치 1개 재빌드 후 rollback.

핵심: snapshot 노드 피처는 순간/과거(lag)만 → 그 자체로 미래 무관(누수 0).
fetch 단계의 cumulative cut은 별도(test_mr_leakage). 여기선 노드 피처의 인과성을 본다.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_db


def _pick_match_id(cur) -> str:
    cur.execute(
        """
        SELECT match_id FROM stdb.team_phase_labels
         GROUP BY match_id HAVING max(cut_phase) >= 3
         ORDER BY match_id LIMIT 1
        """
    )
    row = cur.fetchone()
    if row is None:
        pytest.skip("no labeled match")
    return row[0]


@pytest.fixture()
def rebuilt_match():
    from stdb.conn import connect
    from stdb.snapshot_feature_etl import apply_schema, build_match

    apply_schema()
    conn = connect()
    try:
        with conn.cursor() as cur:
            match_id = _pick_match_id(cur)
            rows = build_match(cur, match_id)
            assert rows > 0
            yield conn, cur, match_id
    finally:
        conn.rollback()
        conn.close()


@requires_db
def test_speed_is_past_only_lag(rebuilt_match):
    """speed = 직전 snapshot 변위/Δt (lag=과거). 미래 안 봄."""
    _conn, cur, mid = rebuilt_match
    cur.execute(
        """
        WITH ref AS (
            SELECT team_id, snapshot_t,
                   ST_Distance(ST_Force2D(pos), lag(ST_Force2D(pos)) OVER w)
                     / NULLIF(snapshot_t - lag(snapshot_t) OVER w, 0) AS spd
              FROM stdb.moving_objects
             WHERE match_id = %(m)s AND agent_kind = 'team' AND is_alive
            WINDOW w AS (PARTITION BY team_id ORDER BY snapshot_t)
        )
        SELECT count(*) FILTER (WHERE r.spd IS NOT NULL AND abs(f.speed_mps - r.spd) > 1e-2),
               count(*) FILTER (WHERE r.spd IS NULL AND f.speed_mps IS NOT NULL),
               count(*)
          FROM stdb.team_snapshot_features f
          JOIN ref r ON r.team_id = f.team_id AND r.snapshot_t = f.snapshot_t
         WHERE f.match_id = %(m)s
        """,
        {"m": mid},
    )
    bad, spurious, n = cur.fetchone()
    assert n > 0
    assert bad == 0 and spurious == 0, f"speed not past-only: bad={bad} spurious_nonnull={spurious}"


@requires_db
def test_signed_boundary_consistent_with_inside(rebuilt_match):
    _conn, cur, mid = rebuilt_match
    cur.execute(
        """
        SELECT count(*) FILTER (WHERE inside_blue AND dist_blue_boundary_m < 0),
               count(*) FILTER (WHERE NOT inside_blue AND dist_blue_boundary_m > 0),
               count(*) FILTER (WHERE inside_blue IS NOT NULL)
          FROM stdb.team_snapshot_features WHERE match_id = %(m)s
        """,
        {"m": mid},
    )
    in_neg, out_pos, n = cur.fetchone()
    assert n > 0
    assert in_neg == 0 and out_pos == 0, "signed boundary sign must match inside_blue"


@requires_db
def test_phase_idx_matches_dynamic_regions(rebuilt_match):
    _conn, cur, mid = rebuilt_match
    cur.execute(
        """
        SELECT count(*) FILTER (WHERE f.phase_idx IS DISTINCT FROM s.phase_idx), count(*)
          FROM stdb.team_snapshot_features f
          JOIN stdb.dynamic_regions s
            ON s.match_id = f.match_id AND s.region_kind = 'safe' AND s.snapshot_t = f.snapshot_t
         WHERE f.match_id = %(m)s
        """,
        {"m": mid},
    )
    bad, n = cur.fetchone()
    assert n > 0 and bad == 0, "phase_idx must match dynamic_regions safe snapshot"


@requires_db
def test_future_snapshot_blindness(rebuilt_match):
    """미래 snapshot 삭제 후 과거 snapshot 행이 불변 — speed가 lag(과거)만 보는 증명."""
    from stdb.snapshot_feature_etl import build_match

    _conn, cur, mid = rebuilt_match
    cur.execute(
        "SELECT max(snapshot_t) FROM stdb.team_snapshot_features WHERE match_id = %s", (mid,)
    )
    t_max = cur.fetchone()[0]
    cutoff = int(t_max * 0.5)

    def snap():
        cur.execute(
            """
            SELECT * FROM stdb.team_snapshot_features
             WHERE match_id = %s AND snapshot_t <= %s
             ORDER BY team_id, snapshot_t
            """,
            (mid, cutoff),
        )
        cols = [d[0] for d in cur.description]
        drop = {cols.index("created_at")}
        return [tuple(v for i, v in enumerate(r) if i not in drop) for r in cur.fetchall()]

    before = snap()
    assert before, "fixture must have early snapshots"
    cur.execute(
        "DELETE FROM stdb.moving_objects WHERE match_id = %s AND snapshot_t > %s", (mid, cutoff)
    )
    assert cur.rowcount > 0
    build_match(cur, mid)
    assert before == snap(), "early snapshot features changed after deleting future snapshots"


def test_build_sql_references_no_label_sources():
    from stdb.snapshot_feature_etl import BUILD_SQL

    sql = BUILD_SQL.lower()
    forbidden = [
        "team_phase_labels", "final_rank", "hazard_next", "rosters", "won",
        "telem_match_end", "game_results", "victim_rank",
        "floor(",  # grid-aligned snapshot_t join only — no floor() bucket leak
    ]
    hits = [t for t in forbidden if t in sql]
    assert not hits, f"BUILD_SQL references forbidden/leaky sources: {hits}"
