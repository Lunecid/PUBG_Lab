"""F6 Tier-1 network feature 불변식 — stdb/network_feature_etl.py.

전부 실 DB 기반(requires_db). 각 테스트는 자체 트랜잭션 안에서 매치 1개를
재빌드하고 검증 후 rollback하므로 DB 상태를 바꾸지 않는다.

검증 축:
1. phase-local 값이 edge cache 독립 재계산과 일치
2. cumulative 값이 phases<=k 삼각 참조와 일치 + 단조 증가
3. 미래 phase 차단: phase>2 엣지를 지우고 재빌드해도 cut<=2 피처 불변 (누수 방지)
4. 테이블 컬럼이 V2 manifest의 F6 계약(43개)을 전부 포함
5. BUILD_SQL이 라벨/최종순위 계열 입력을 참조하지 않음 (정적 가드)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import requires_db

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / 'src/pubg_lab/survival/features/manifests/pubg_representation_v2.json'

FLOAT_TOL = 1e-3


def _pick_match_id(cur) -> str:
    """combat 엣지가 있고 phase가 3개 이상인 매치 1개."""
    cur.execute(
        """
        SELECT match_id
          FROM stdb.team_phase_edge_cache
         WHERE edge_type = 'combat'
         GROUP BY match_id
        HAVING max(phase_idx) >= 3
         ORDER BY match_id
         LIMIT 1
        """
    )
    row = cur.fetchone()
    if row is None:
        pytest.skip("no combat-bearing match in edge cache")
    return row[0]


@pytest.fixture()
def rebuilt_match():
    """트랜잭션 안에서 매치 1개를 빌드하고 (conn, cur, match_id)를 내준 뒤 rollback."""
    from pubg_lab.data.stdb.connection.conn import connect
    from pubg_lab.data.stdb.pipelines.network_feature_etl import apply_schema, build_match

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
def test_phase_local_matches_reference(rebuilt_match):
    _conn, cur, mid = rebuilt_match
    cur.execute(
        """
        WITH ours AS (
            SELECT team_id, phase_idx,
                   net_combat_out_degree, net_combat_in_degree,
                   net_combat_out_strength, net_combat_in_strength,
                   net_combat_total_degree, net_combat_unique_targets,
                   net_prox_degree_200m, net_prox_min_distance_m,
                   net_prox_strength_decay, net_prox_isolation_m,
                   net_copress_degree, net_copress_strength
              FROM stdb.team_phase_network_features WHERE match_id = %(m)s
        ),
        ref_out AS (
            SELECT src_team_id AS team_id, phase_idx,
                   count(*) AS od, sum(weight) AS os
              FROM stdb.team_phase_edge_cache
             WHERE match_id = %(m)s AND edge_type = 'combat' GROUP BY 1, 2
        ),
        ref_in AS (
            SELECT dst_team_id AS team_id, phase_idx,
                   count(*) AS idg, sum(weight) AS ist
              FROM stdb.team_phase_edge_cache
             WHERE match_id = %(m)s AND edge_type = 'combat' GROUP BY 1, 2
        ),
        ref_prox AS (
            SELECT src_team_id AS team_id, phase_idx,
                   count(*) FILTER (WHERE avg_distance_m < 200) AS d200,
                   min(avg_distance_m) AS mind, sum(weight) AS sd
              FROM stdb.team_phase_edge_cache
             WHERE match_id = %(m)s AND edge_type = 'proximity' GROUP BY 1, 2
        ),
        ref_cop AS (
            SELECT src_team_id AS team_id, phase_idx,
                   count(*) AS deg, sum(weight) AS st
              FROM stdb.team_phase_edge_cache
             WHERE match_id = %(m)s AND edge_type = 'copressure' GROUP BY 1, 2
        )
        SELECT
            count(*) FILTER (WHERE o.net_combat_out_degree <> coalesce(ro.od, 0)),
            count(*) FILTER (WHERE abs(o.net_combat_out_strength - coalesce(ro.os, 0)) > %(tol)s),
            count(*) FILTER (WHERE o.net_combat_in_degree <> coalesce(ri.idg, 0)),
            count(*) FILTER (WHERE abs(o.net_combat_in_strength - coalesce(ri.ist, 0)) > %(tol)s),
            count(*) FILTER (WHERE o.net_combat_total_degree
                             <> coalesce(ro.od, 0) + coalesce(ri.idg, 0)),
            count(*) FILTER (WHERE o.net_combat_unique_targets <> coalesce(ro.od, 0)),
            count(*) FILTER (WHERE o.net_prox_degree_200m <> coalesce(rp.d200, 0)),
            count(*) FILTER (WHERE rp.mind IS NOT NULL
                             AND abs(o.net_prox_min_distance_m - rp.mind) > 0.01),
            count(*) FILTER (WHERE rp.team_id IS NULL AND o.net_prox_min_distance_m IS NOT NULL),
            count(*) FILTER (WHERE o.net_prox_isolation_m IS DISTINCT FROM o.net_prox_min_distance_m),
            count(*) FILTER (WHERE abs(o.net_prox_strength_decay - coalesce(rp.sd, 0)) > %(tol)s),
            count(*) FILTER (WHERE o.net_copress_degree <> coalesce(rc.deg, 0)),
            count(*) FILTER (WHERE abs(o.net_copress_strength - coalesce(rc.st, 0)) > %(tol)s),
            count(*)
        FROM ours o
        LEFT JOIN ref_out  ro ON ro.team_id = o.team_id AND ro.phase_idx = o.phase_idx
        LEFT JOIN ref_in   ri ON ri.team_id = o.team_id AND ri.phase_idx = o.phase_idx
        LEFT JOIN ref_prox rp ON rp.team_id = o.team_id AND rp.phase_idx = o.phase_idx
        LEFT JOIN ref_cop  rc ON rc.team_id = o.team_id AND rc.phase_idx = o.phase_idx
        """,
        {"m": mid, "tol": FLOAT_TOL},
    )
    *mismatches, n = cur.fetchone()
    assert n > 0
    assert all(c == 0 for c in mismatches), f"phase-local mismatches: {mismatches}"


@requires_db
def test_cumulative_matches_reference_and_monotone(rebuilt_match):
    _conn, cur, mid = rebuilt_match
    cur.execute(
        """
        WITH ours AS (
            SELECT team_id, phase_idx,
                   cum_combat_out_strength, cum_combat_in_strength,
                   cum_unique_targets, cum_unique_attackers,
                   cum_prox_close_exposure_200m
              FROM stdb.team_phase_network_features WHERE match_id = %(m)s
        ),
        ref AS (
            SELECT o.team_id, o.phase_idx,
                   coalesce(sum(e.weight) FILTER (
                       WHERE e.edge_type = 'combat' AND e.src_team_id = o.team_id), 0) AS cos,
                   coalesce(sum(e.weight) FILTER (
                       WHERE e.edge_type = 'combat' AND e.dst_team_id = o.team_id), 0) AS cis,
                   count(DISTINCT e.dst_team_id) FILTER (
                       WHERE e.edge_type = 'combat' AND e.src_team_id = o.team_id) AS cut,
                   count(DISTINCT e.src_team_id) FILTER (
                       WHERE e.edge_type = 'combat' AND e.dst_team_id = o.team_id) AS cua,
                   count(*) FILTER (
                       WHERE e.edge_type = 'proximity' AND e.src_team_id = o.team_id
                         AND e.avg_distance_m < 200) AS cpx
              FROM ours o
         LEFT JOIN stdb.team_phase_edge_cache e
                ON e.match_id = %(m)s AND e.phase_idx <= o.phase_idx
             GROUP BY 1, 2
        )
        SELECT
            count(*) FILTER (WHERE abs(o.cum_combat_out_strength - r.cos) > %(tol)s),
            count(*) FILTER (WHERE abs(o.cum_combat_in_strength - r.cis) > %(tol)s),
            count(*) FILTER (WHERE o.cum_unique_targets <> r.cut),
            count(*) FILTER (WHERE o.cum_unique_attackers <> r.cua),
            count(*) FILTER (WHERE o.cum_prox_close_exposure_200m <> r.cpx),
            count(*)
        FROM ours o JOIN ref r ON r.team_id = o.team_id AND r.phase_idx = o.phase_idx
        """,
        {"m": mid, "tol": FLOAT_TOL},
    )
    *mismatches, n = cur.fetchone()
    assert n > 0
    assert all(c == 0 for c in mismatches), f"cumulative mismatches: {mismatches}"

    cur.execute(
        """
        SELECT count(*) FROM (
            SELECT cum_combat_total_strength
                       - lag(cum_combat_total_strength) OVER w AS d_strength,
                   cum_unique_targets - lag(cum_unique_targets) OVER w AS d_targets,
                   cum_prox_close_exposure_200m
                       - lag(cum_prox_close_exposure_200m) OVER w AS d_prox
              FROM stdb.team_phase_network_features
             WHERE match_id = %(m)s
            WINDOW w AS (PARTITION BY team_id ORDER BY phase_idx)
        ) x
        WHERE d_strength < -1e-5 OR d_targets < 0 OR d_prox < 0
        """,
        {"m": mid},
    )
    assert cur.fetchone()[0] == 0, "cumulative columns must be non-decreasing in phase"


@requires_db
def test_future_phase_blindness(rebuilt_match):
    """cut phase k의 피처는 phase>k 정보와 무관해야 한다 (누수 방지 핵심).

    같은 트랜잭션 안에서 phase>2 엣지를 삭제하고 재빌드 → phase<=2 행이
    바이트 단위로 동일해야 한다.
    """
    from pubg_lab.data.stdb.pipelines.network_feature_etl import build_match

    _conn, cur, mid = rebuilt_match

    def snapshot():
        cur.execute(
            """
            SELECT * FROM stdb.team_phase_network_features
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
        "DELETE FROM stdb.team_phase_edge_cache WHERE match_id = %s AND phase_idx > 2",
        (mid,),
    )
    assert cur.rowcount > 0, "fixture match must have phase>2 edges to delete"
    build_match(cur, mid)
    after = snapshot()

    assert before == after, "features at cut<=2 changed after deleting future-phase edges"


@requires_db
def test_table_satisfies_manifest_contract(rebuilt_match):
    _conn, cur, _mid = rebuilt_match
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted = manifest["families"]["f6_network_phase_role_tier1"]["columns"]
    assert len(wanted) == 43

    cur.execute(
        """
        SELECT column_name FROM information_schema.columns
         WHERE table_schema = 'stdb' AND table_name = 'team_phase_network_features'
        """
    )
    have = {r[0] for r in cur.fetchall()}
    missing = [c for c in wanted if c not in have]
    assert not missing, f"manifest columns missing from table: {missing}"


def test_build_sql_references_no_label_sources():
    """F6 입력은 edge cache + 우주(teams/bounds/global)뿐이어야 한다."""
    from pubg_lab.data.stdb.pipelines.network_feature_etl import BUILD_SQL

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
