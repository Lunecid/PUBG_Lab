"""마스크 상호 일관성 — 패딩 의미론이 조용히 깨지는 것을 방지 (CODE_REVIEW D10)."""

from __future__ import annotations

import numpy as np

from tests.conftest import requires_db


@requires_db
def test_phase_mask_matches_phase_indices(phase_tensors):
    mt = phase_tensors
    n_real = len([p for p in mt.phase_indices if p > 0])
    assert mt.phase_mask[:n_real].all()
    assert not mt.phase_mask[n_real:].any()


@requires_db
def test_team_mask_matches_team_ids(phase_tensors):
    mt = phase_tensors
    n_real = len([t for t in mt.team_ids if t > 0])
    # team_mask는 피처 row가 있는 팀만 True일 수 있으므로 상한만 확인
    assert not mt.team_mask[n_real:].any(), "패딩 위치에 team_mask True"
    assert mt.team_mask.sum() > 0


@requires_db
def test_node_mask_zero_on_padded_phases(phase_tensors):
    mt = phase_tensors
    pad_phases = ~mt.phase_mask
    assert not mt.node_mask[pad_phases].any(), "존재하지 않는 phase에 생존 팀"


@requires_db
def test_targets_only_on_real_teams(phase_tensors):
    mt = phase_tensors
    n_real = len([t for t in mt.team_ids if t > 0])
    assert not mt.target_mask[:, n_real:].any()
    assert not mt.y_hazard[:, n_real:].any()


@requires_db
def test_features_zero_on_padding(phase_tensors):
    mt = phase_tensors
    n_real = len([t for t in mt.team_ids if t > 0])
    assert np.abs(mt.X[:, n_real:, :]).sum() == 0.0, "패딩 노드 피처는 0이어야 함"
