"""hazard 라벨 정렬·인과성 불변식 — CODE_REVIEW A1을 코드로 고정.

검증하는 정렬:
    hazard_logits[:, k] ← hidden(1-indexed phase k+1까지 관측)
    y_hazard[k, i]      ← cut_phase k+1: "phase k+1 종료 시 생존 & phase k+2에서 탈락"
"""

from __future__ import annotations

import numpy as np

from tests.conftest import requires_db


@requires_db
def test_phase_indices_one_indexed(phase_tensors):
    real = [p for p in phase_tensors.phase_indices if p > 0]
    assert real == sorted(real)
    assert real[0] == 1, "phase는 1-indexed여야 함 (v_match_phase_bounds)"
    assert all(1 <= p <= 8 for p in real)


@requires_db
def test_target_alignment_against_label_table(phase_tensors):
    """텐서의 (out_t, i)가 team_phase_labels의 (cut_phase=out_t+1, team)과 일치."""
    from stdb.conn import get_cur

    mt = phase_tensors
    team_index = {tid: i for i, tid in enumerate(mt.team_ids) if tid > 0}

    with get_cur() as (_conn, cur):
        cur.execute(
            """
            SELECT team_id, cut_phase, target_defined, hazard_next
              FROM stdb.team_phase_labels
             WHERE match_id = %s
            """,
            (mt.match_id,),
        )
        rows = cur.fetchall()

    checked = 0
    for team_id, cut_phase, target_defined, hazard_next in rows:
        team_id, cut_phase = int(team_id), int(cut_phase)
        if team_id not in team_index or not (1 <= cut_phase < mt.y_hazard.shape[0] + 1):
            continue
        out_t = cut_phase - 1
        i = team_index[team_id]
        assert bool(mt.target_mask[out_t, i]) == bool(target_defined), (
            f"target_mask[{out_t},{i}] != target_defined(cut_phase={cut_phase})"
        )
        if target_defined and hazard_next is not None:
            assert mt.y_hazard[out_t, i] == float(hazard_next)
            checked += 1
    assert checked > 0, "검증된 라벨이 0개 — 매치/쿼리 확인 필요"


@requires_db
def test_alive_start_end_gap_bounded():
    """데이터 품질 회귀 가드 (CODE_REVIEW E20).

    이상적으로는 target_defined(phase 종료 시 생존) ⊆ alive_at_phase_start지만,
    실측 DB에서 타겟의 ~2.5%가 start=0 & end>0 (전원 DBNO 상태이거나 스냅샷
    공백).  라벨 누수는 아니나 해당 팀은 그 phase에서 proximity self-loop을 받지
    못한다.  ETL 보강 전까지는 이 비율이 조용히 증가하지 않는지만 감시한다.
    """
    from stdb.conn import get_cur

    with get_cur() as (_conn, cur):
        cur.execute(
            """
            SELECT count(*) FILTER (WHERE target_defined) AS targets,
                   count(*) FILTER (WHERE target_defined AND NOT alive_at_phase_start) AS gap
              FROM stdb.team_phase_labels
            """
        )
        targets, gap = cur.fetchone()
    assert targets > 0
    rate = gap / targets
    assert rate < 0.03, (
        f"alive start/end gap {rate:.2%} (>{gap:,}/{targets:,}) — "
        "ETL 변경으로 데이터 품질이 악화됨"
    )


@requires_db
def test_team_order_not_rank_sorted(phase_tensors):
    """팀 순서는 team_id 정렬 (final_rank 정렬이면 라벨 누수 — A4 회귀 방지)."""
    real = [t for t in phase_tensors.team_ids if t > 0]
    assert real == sorted(real), "team_ids must be sorted by team_id, not by rank"
