"""MR-STGNN 누수 회귀 가드 — hierarchical 시간 인과성.

대부분 합성(빠름). 핵심 4축:
1. snapshot encoder causal — 미래 snapshot 바꿔도 과거 임베딩 불변
2. phase pooling phase-local — phase>k snapshot 바꿔도 phase<=k 풀 불변
3. survival 라벨 = 미래 death만(입력은 <=t, 라벨은 look-ahead = 표준 survival)
4. (실DB) combat 엣지 과거 귀속 — 미래 event 삭제해도 snapshot t 그래프 불변
"""

from __future__ import annotations

import numpy as np
import torch

from pubg_lab.survival.models.mr_stgnn import mr_survival_targets
from pubg_lab.survival.models.snapshot_encoder import SnapshotEncoder, pool_snapshots_to_phase
from tests.conftest import requires_db


def _rand_rowstoch(B, T, E, N, n_alive, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.zeros(B, T, E, N, N)
    d = torch.rand(B, T, E, n_alive, n_alive, generator=g)
    d = d * (torch.rand(B, T, E, n_alive, n_alive, generator=g) < 0.3)
    d[:, :, 0] += torch.eye(n_alive)
    rs = d.sum(-1, keepdim=True)
    d = torch.where(rs > 0, d / rs.clamp(min=1e-9), d)
    A[:, :, :, :n_alive, :n_alive] = d
    return A


def test_snapshot_encoder_is_causal():
    """snapshot t 임베딩은 <=t만 관측 — 미래 snapshot을 바꿔도 t<=cut 임베딩 불변."""
    B, T, N, F, E, na = 1, 12, 16, 13, 2, 10
    torch.manual_seed(0)
    X = torch.zeros(B, T, N, F); X[:, :, :na] = torch.randn(B, T, na, F)
    A = _rand_rowstoch(B, T, E, N, na)
    enc = SnapshotEncoder(in_features=F, num_edge_types=E).eval()
    with torch.no_grad():
        emb0 = enc(X, A)
        cut = 6
        X2 = X.clone(); X2[:, cut + 1:] = torch.randn_like(X2[:, cut + 1:]) * 5
        A2 = A.clone(); A2[:, cut + 1:] = _rand_rowstoch(B, T, E, N, na, seed=9)[:, cut + 1:]
        emb1 = enc(X2, A2)
    assert torch.allclose(emb0[:, :cut + 1], emb1[:, :cut + 1], atol=1e-5), \
        "snapshot encoder leaked future into past embeddings"


def test_phase_pooling_is_phase_local():
    """phase k 풀은 phase_of_snap==k snapshot만 — phase>k 바꿔도 phase<=k 풀 불변."""
    B, T, N, C = 1, 12, 16, 8
    torch.manual_seed(1)
    emb = torch.randn(B, T, N, C)
    nmask = torch.zeros(B, T, N, dtype=torch.bool); nmask[:, :, :10] = True
    # phase_of_snap: 0,0,1,1,1,2,2,3,3,4,4,5
    pos = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5]])
    pooled0 = pool_snapshots_to_phase(emb, pos, nmask, t_phase=8)
    # phase>2 (slots) snapshot 임베딩 교란 — phase 1,2 (slots 0,1) 풀은 불변이어야
    emb2 = emb.clone()
    future = pos.squeeze(0) >= 3
    emb2[:, future] = torch.randn_like(emb2[:, future]) * 9
    pooled1 = pool_snapshots_to_phase(emb2, pos, nmask, t_phase=8)
    assert torch.allclose(pooled0[:, :2], pooled1[:, :2], atol=1e-6), \
        "phase pooling mixed future-phase snapshots into earlier phases"


def test_survival_targets_label_only_future_deaths():
    B, T, N = 2, 20, 8
    nmask = torch.zeros(B, T, N, dtype=torch.bool); nmask[:, :, :5] = True
    death = torch.full((B, N), -1, dtype=torch.long)
    death[0, :3] = torch.tensor([5, 10, 15])
    tgt, at_risk = mr_survival_targets(death, nmask, horizon=3)
    t_idx = torch.arange(T).view(1, T, 1)
    future_death = (death.view(B, 1, N) > t_idx)
    # 모든 positive는 미래 death(>t) 위치여야 한다
    assert bool((tgt.bool() <= future_death).all()), "survival label includes non-future death"
    # horizon 밖(death > t+horizon)은 0
    beyond = death.view(B, 1, N) > t_idx + 3
    assert float((tgt * beyond.float()).sum()) == 0.0, "survival label leaked beyond horizon"
    # at-risk는 alive만
    assert bool((at_risk.bool() <= nmask).all())


@requires_db
def test_combat_edges_attributed_to_past_window():
    """combat 엣지는 event_t in (t-10, t] (과거)에 귀속 — 미래 event 삭제해도 snapshot 그래프 불변."""
    from pubg_lab.data.stdb.connection.conn import connect
    from pubg_lab.data.stdb.pipelines.snapshot_feature_etl import apply_schema, build_match as build_feat
    from pubg_lab.data.stdb.queries.snapshot_queries import fetch_snapshot_match_tensors

    apply_schema()
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT match_id FROM stdb.spatial_events
                 WHERE event_kind IN ('damage','kill','groggy')
                   AND attacker_team IS NOT NULL AND victim_team IS NOT NULL
                 GROUP BY match_id HAVING count(*) > 50 ORDER BY match_id LIMIT 1
                """
            )
            row = cur.fetchone()
            if row is None:
                import pytest; pytest.skip("no combat-bearing match")
            mid = row[0]
            build_feat(cur, mid)
            conn.commit()
    finally:
        conn.close()

    st0 = fetch_snapshot_match_tensors(mid)
    assert st0 is not None
    T_real = int((st0.snapshot_ts >= 0).sum())
    cut_t = int(st0.snapshot_ts[T_real // 2])

    conn = connect()
    try:
        with conn.cursor() as cur:
            # 미래(> cut_t) combat event 삭제
            cur.execute("DELETE FROM stdb.spatial_events WHERE match_id=%s AND event_t > %s", (mid, cut_t))
            deleted = cur.rowcount
            st1 = fetch_snapshot_match_tensors(mid)
        # combat 채널(인덱스 1)에서 snapshot_t <= cut_t 행이 불변이어야 한다
        idx = {int(s): i for i, s in enumerate(st0.snapshot_ts) if s >= 0}
        idx1 = {int(s): i for i, s in enumerate(st1.snapshot_ts) if s >= 0}
        common = [s for s in idx if s <= cut_t and s in idx1]
        assert common and deleted > 0
        for s in common:
            a0 = st0.A_s[idx[s], 1]
            a1 = st1.A_s[idx1[s], 1]
            assert np.allclose(a0, a1, atol=1e-5), f"combat graph at snap {s} changed after deleting future events"
    finally:
        conn.rollback()
        conn.close()
