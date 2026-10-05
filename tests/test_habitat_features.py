"""F7 Tier-1 habitat feature 불변식 — stdb/habitat_feature_etl.py.

전부 실 DB 기반(requires_db). 트랜잭션 안에서 매치 1개를 재빌드하고 rollback.

검증 축:
1. 전역 블록이 match_phase_global 재계산과 일치 (상대오차, float4 감안)
2. 범위/부호 불변식 + refuge scarcity 공식 재검
3. 타깃 거리 0 ⟺ cut 시점 흰 원 내부 (의미론 핵심)
4. 미래 차단: phase>2 입력(regions/positions/proximity) 삭제 후 재빌드해도
   cut<=2 피처 불변
5. 테이블 컬럼 ⊇ manifest F7 계약(32개)
6. BUILD_SQL이 라벨/최종순위 소스를 참조하지 않음
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import requires_db

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "model_stgcn" / "feature_manifests" / "pubg_representation_v2.json"

REL_TOL = 1e-6  # 면적 ~1e8 m^2의 float4 저장 오차(~1.5e-8) 흡수


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
        pytest.skip("no labeled match with >=3 phases")
    return row[0]


@pytest.fixture()
def rebuilt_match():
    from stdb.base_stage import (
        TempBaseCursor,
        analyze_temp_base_tables,
        create_temp_base_tables,
        stage_base_rows,
    )
    from stdb.conn import connect
    from stdb.habitat_feature_etl import apply_schema, build_match

    apply_schema()
    conn = connect()
    try:
        create_temp_base_tables(conn)
        with conn.cursor() as raw_cur:
            # stdb.moving_objects / team_proximity are empty in the working DB (purged after the
            # original ETL): without staging, every team-relative feature is NULL and most of
            # the invariants below pass vacuously.  The staged rows live in session-local temp
            # tables that the proxy cursor substitutes for the shared ones (nothing is written
            # to the shared base tables, and everything is rolled back at the end).
            cur = TempBaseCursor(raw_cur)
            match_id = _pick_match_id(cur)
            stage_base_rows(cur, match_id)
            analyze_temp_base_tables(cur)
            rows = build_match(cur, match_id)
            assert rows > 0
            yield conn, cur, match_id
    finally:
        conn.rollback()
        conn.close()


@requires_db
def test_global_block_matches_mpg(rebuilt_match):
    _conn, cur, mid = rebuilt_match
    cur.execute(
        """
        SELECT
            count(*) FILTER (WHERE abs(h.hab_safe_area_start_m2 - pi()*g.safe_radius_start_m^2)
                             > %(tol)s * pi()*g.safe_radius_start_m^2),
            count(*) FILTER (WHERE abs(h.hab_safe_area_end_m2 - pi()*g.safe_radius_end_m^2)
                             > %(tol)s * pi()*g.safe_radius_end_m^2),
            count(*) FILTER (WHERE h.hab_alive_teams <> g.end_alive_teams),
            count(*) FILTER (WHERE abs(h.hab_crowding_pressure
                                        - g.end_alive_teams/(pi()*g.safe_radius_end_m^2)) > 1e-9),
            count(*) FILTER (WHERE abs(h.hab_loss_ratio
                                        - (1.0 - g.safe_radius_end_m^2/g.safe_radius_start_m^2)) > 1e-4),
            count(*) FILTER (WHERE h.hab_zone_gap_start_m IS DISTINCT FROM g.zone_gap_start_m),
            count(*)
        FROM stdb.team_phase_habitat_features h
        JOIN stdb.match_phase_global g
          ON g.match_id = h.match_id AND g.phase_idx = h.phase_idx
        WHERE h.match_id = %(m)s
        """,
        {"m": mid, "tol": REL_TOL},
    )
    *mismatches, n = cur.fetchone()
    assert n > 0
    assert all(c == 0 for c in mismatches), f"global block mismatches: {mismatches}"


@requires_db
def test_range_sign_and_formula_invariants(rebuilt_match):
    _conn, cur, mid = rebuilt_match
    cur.execute(
        """
        SELECT
            count(*) FILTER (WHERE hab_safe_alignment_to_target NOT BETWEEN -1.0001 AND 1.0001),
            count(*) FILTER (WHERE hab_edge_pressure_ratio NOT BETWEEN 0 AND 1),
            count(*) FILTER (WHERE hab_rotation_delay_ratio NOT BETWEEN 0 AND 1),
            count(*) FILTER (WHERE hab_outside_safe_ratio NOT BETWEEN 0 AND 1),
            count(*) FILTER (WHERE hab_distance_to_target_safe_m < 0),
            count(*) FILTER (WHERE abs(abs(hab_signed_boundary_distance_start_m)
                                       - hab_dist_to_safe_boundary_start_m) > 0.01),
            count(*) FILTER (WHERE abs(abs(hab_signed_boundary_distance_end_m)
                                       - hab_dist_to_safe_boundary_end_m) > 0.01),
            count(*) FILTER (WHERE hab_local_enemy_count_100m > hab_local_enemy_count_200m),
            count(*) FILTER (WHERE hab_local_enemy_count_200m > hab_local_enemy_count_500m),
            count(*) FILTER (WHERE hab_safe_entry_enemy_count_200m IS NOT NULL
                             AND hab_safe_area_end_m2 IS NOT NULL
                             AND abs(hab_refuge_scarcity_simple
                                     - hab_safe_entry_enemy_count_200m
                                       / (pi() * (sqrt(hab_safe_area_end_m2/pi()))^2 + 1.0))
                                 > greatest(1e-12, 1e-3 * hab_refuge_scarcity_simple)),
            count(*) FILTER (WHERE hab_migration_deficit_mps IS NOT NULL
                             AND abs(hab_migration_deficit_mps
                                     - (hab_required_speed_to_safe_mps - hab_observed_speed_mps)) > 1e-3),
            count(*)
        FROM stdb.team_phase_habitat_features
        WHERE match_id = %(m)s
        """,
        {"m": mid},
    )
    *violations, n = cur.fetchone()
    assert n > 0
    assert all(c == 0 for c in violations), f"invariant violations: {violations}"


@requires_db
def test_target_distance_containment_consistency(rebuilt_match):
    from stdb.causal_cut import GUARD_S, fetch_clock_delta

    _conn, cur, mid = rebuilt_match
    delta = fetch_clock_delta(cur, mid)  # one clock: grid windows are [start - delta, end - delta - GUARD_S]
    cur.execute(
        f"""
        WITH b AS (
            SELECT phase_idx, phase_start_s, phase_end_s
              FROM stdb.v_match_phase_bounds WHERE match_id = %(m)s AND phase_idx > 0
        ),
        ct AS (
            SELECT b.phase_idx, dr.region FROM b
            JOIN LATERAL (
                SELECT region FROM stdb.dynamic_regions
                 WHERE match_id = %(m)s AND region_kind = 'poison'
                   AND snapshot_t BETWEEN b.phase_start_s - %(d)s AND b.phase_end_s - %(d)s - {GUARD_S}
                 ORDER BY snapshot_t DESC LIMIT 1
            ) dr ON TRUE
        ),
        lastpos AS (
            SELECT b.phase_idx, mo.team_id,
                   (array_agg(ST_Force2D(mo.pos) ORDER BY mo.snapshot_t DESC))[1] AS pos
              FROM b JOIN stdb.moving_objects mo
                ON mo.match_id = %(m)s AND mo.agent_kind = 'team' AND mo.is_alive
               AND mo.snapshot_t BETWEEN b.phase_start_s - %(d)s AND b.phase_end_s - %(d)s - {GUARD_S}
             GROUP BY 1, 2
        )
        SELECT count(*) FILTER (WHERE (h.hab_distance_to_target_safe_m = 0)
                                <> ST_Contains(ct.region, lp.pos)),
               count(*)
          FROM stdb.team_phase_habitat_features h
          JOIN lastpos lp ON lp.phase_idx = h.phase_idx AND lp.team_id = h.team_id
          JOIN ct ON ct.phase_idx = h.phase_idx
         WHERE h.match_id = %(m)s
        """,
        {"m": mid, "d": delta},
    )
    mismatched, n = cur.fetchone()
    assert n > 0
    assert mismatched == 0, "distance==0 must coincide with cut-time target containment"


@requires_db
def test_future_phase_blindness(rebuilt_match):
    """phase>2 입력을 지우고 재빌드해도 cut<=2 피처가 불변이어야 한다."""
    from stdb.habitat_feature_etl import build_match

    _conn, cur, mid = rebuilt_match

    def snapshot():
        cur.execute(
            """
            SELECT * FROM stdb.team_phase_habitat_features
             WHERE match_id = %s AND phase_idx <= 2
             ORDER BY team_id, phase_idx
            """,
            (mid,),
        )
        cols = [d[0] for d in cur.description]
        drop = {cols.index("created_at")}
        return [tuple(v for i, v in enumerate(r) if i not in drop) for r in cur.fetchall()]

    before = snapshot()
    assert before, "fixture match must have phase<=2 rows"

    cur.execute(
        """
        SELECT max(phase_end_s) FROM stdb.v_match_phase_bounds
         WHERE match_id = %s AND phase_idx = 2
        """,
        (mid,),
    )
    p2_end = cur.fetchone()[0]
    assert p2_end is not None

    cur.execute(
        "DELETE FROM stdb.dynamic_regions WHERE match_id = %s AND phase_idx > 2",
        (mid,),
    )
    deleted_regions = cur.rowcount
    cur.execute(
        "DELETE FROM stdb.moving_objects WHERE match_id = %s AND snapshot_t > %s",
        (mid, p2_end),
    )
    cur.execute(
        "DELETE FROM stdb.team_proximity WHERE match_id = %s AND snapshot_t > %s",
        (mid, p2_end),
    )
    assert deleted_regions > 0, "fixture match must have phase>2 regions to delete"

    build_match(cur, mid)
    after = snapshot()
    assert before == after, "features at cut<=2 changed after deleting future inputs"


@requires_db
def test_table_satisfies_manifest_contract(rebuilt_match):
    _conn, cur, _mid = rebuilt_match
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted = manifest["families"]["f7_habitat_pressure_tier1"]["columns"]
    assert len(wanted) == 32

    cur.execute(
        """
        SELECT column_name FROM information_schema.columns
         WHERE table_schema = 'stdb' AND table_name = 'team_phase_habitat_features'
        """
    )
    have = {r[0] for r in cur.fetchall()}
    missing = [c for c in wanted if c not in have]
    assert not missing, f"manifest columns missing from table: {missing}"


def test_build_sql_references_no_label_sources():
    from stdb.habitat_feature_etl import BUILD_SQL

    sql = BUILD_SQL.lower()
    forbidden = [
        "team_phase_labels",
        "final_rank",
        "hazard_next",
        "rosters",
        "won",
        "telem_match_end",
        "game_results",
        "victim_rank",
    ]
    hits = [t for t in forbidden if t in sql]
    assert not hits, f"BUILD_SQL references forbidden label sources: {hits}"
