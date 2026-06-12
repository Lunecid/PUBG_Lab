"""인접행렬 불변식 — stdb.queries._fetch_phase_edges가 보장해야 하는 것들.

논문이 주장하는 그래프 구성 안전성(row-stochastic, 패딩 비오염, self-loop
정책)을 코드로 고정한다.  docs/CODE_REVIEW_2026-06-12.md A3, D10.
"""

from __future__ import annotations

import numpy as np

from tests.conftest import requires_db


@requires_db
def test_rows_stochastic_or_zero(phase_tensors):
    A = phase_tensors.A  # (T, E, N, N)
    rs = A.sum(axis=-1)
    ok = np.isclose(rs, 1.0, atol=1e-3) | np.isclose(rs, 0.0, atol=1e-6)
    assert ok.all(), f"non-stochastic rows: {np.argwhere(~ok)[:5]}"


@requires_db
def test_padding_rows_and_cols_zero(phase_tensors):
    mt = phase_tensors
    n_real = int(mt.team_mask.sum())
    A = mt.A
    assert A[:, :, n_real:, :].sum() == 0.0, "padded rows must be zero"
    assert A[:, :, :, n_real:].sum() == 0.0, "padded cols must be zero"


@requires_db
def test_selfloop_policy(phase_tensors):
    """proximity만 alive 노드에 self-loop; combat/copressure는 대각 0."""
    mt = phase_tensors
    e = {name: i for i, name in enumerate(mt.edge_types)}
    for t in range(mt.A.shape[0]):
        alive = np.where(mt.node_mask[t])[0]
        diag_prox = np.diagonal(mt.A[t, e["proximity"]])
        assert (diag_prox[alive] > 0).all(), f"phase t={t}: alive 노드 self-loop 누락"
        for name in ("combat", "copressure"):
            assert np.diagonal(mt.A[t, e[name]]).sum() == 0.0, f"{name}는 대각 0이어야 함"


@requires_db
def test_combat_in_is_support_transpose(phase_tensors, phase_tensors_directed):
    """combat_in의 support(>0 위치)는 combat의 transpose와 정확히 일치."""
    e3 = {name: i for i, name in enumerate(phase_tensors.edge_types)}
    e4 = {name: i for i, name in enumerate(phase_tensors_directed.edge_types)}
    assert "combat_in" in e4 and "combat_in" not in e3

    combat = phase_tensors_directed.A[:, e4["combat"]]
    combat_in = phase_tensors_directed.A[:, e4["combat_in"]]
    sup_out = combat > 0
    sup_in = combat_in > 0
    assert (sup_in == np.swapaxes(sup_out, -1, -2)).all()

    # E=3 vs E=4에서 기존 3채널은 동일해야 함
    for name in ("proximity", "combat", "copressure"):
        np.testing.assert_allclose(
            phase_tensors.A[:, e3[name]],
            phase_tensors_directed.A[:, e4[name]],
            atol=1e-6,
            err_msg=f"channel {name} changed by include_combat_in",
        )
