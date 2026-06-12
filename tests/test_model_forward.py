"""모델 forward 불변식 — DB 불필요 (합성 텐서).

1. 패딩 노드 비오염: 패딩 노드 피처를 바꿔도 실 노드 hazard 불변
2. masked pooling: 패딩 phase 노이즈가 rank/topk에 영향 없음 (C4)
3. 인과성: 미래 phase 노이즈가 과거 hazard에 영향 없음 (A1)
4. 인접행렬 검증: 비정규화 A는 명확한 에러 (C6)
5. (legacy) GroupGNN: 비연속 팀 ID 정렬 + 계약 위반 에러 (B3)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _make_inputs(n_alive=12, n_phase=5, B=2, T=8, E=3, N=32, F=66, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.zeros(B, T, N, F)
    X[:, :n_phase, :n_alive] = torch.randn(B, n_phase, n_alive, F, generator=g)

    A = torch.zeros(B, T, E, N, N)
    dense = torch.rand(B, n_phase, E, n_alive, n_alive, generator=g)
    dense = dense * (torch.rand(B, n_phase, E, n_alive, n_alive, generator=g) < 0.3)
    dense[:, :, 0] = dense[:, :, 0] + torch.eye(n_alive)
    rs = dense.sum(dim=-1, keepdim=True)
    dense = torch.where(rs > 0, dense / rs.clamp(min=1e-9), dense)
    A[:, :n_phase, :, :n_alive, :n_alive] = dense

    phase_mask = torch.zeros(B, T, dtype=torch.bool)
    phase_mask[:, :n_phase] = True
    return X, A, phase_mask


@pytest.fixture(scope="module")
def model():
    from model_stgcn.dyn_stgnn import DynamicSTGNN

    torch.manual_seed(0)
    return DynamicSTGNN(in_features=66).eval()


def test_padding_nodes_do_not_pollute(model):
    X, A, pm = _make_inputs()
    with torch.no_grad():
        yh, _, _ = model(X, A, phase_mask=pm)
        X2 = X.clone()
        X2[:, :, 12:] += 50.0  # 패딩 노드(A의 row/col이 0)에 노이즈
        yh2, _, _ = model(X2, A, phase_mask=pm)
    assert torch.allclose(yh[:, :, :12], yh2[:, :, :12], atol=1e-5)


def test_masked_pooling_ignores_padded_phases(model):
    X, A, pm = _make_inputs()
    with torch.no_grad():
        _, yr, yk = model(X, A, phase_mask=pm)
        X2 = X.clone()
        X2[:, 5:] += 100.0  # 존재하지 않는 phase에 노이즈
        _, yr2, yk2 = model(X2, A, phase_mask=pm)
        _, yr3, _ = model(X2, A)  # 마스크 없으면 새어 들어가야 함 (대조)
    assert torch.allclose(yr, yr2, atol=1e-5)
    assert torch.allclose(yk, yk2, atol=1e-5)
    assert not torch.allclose(yr, yr3, atol=1e-3)


def test_causality_no_future_leak(model):
    X, A, pm = _make_inputs()
    with torch.no_grad():
        yh, _, _ = model(X, A, phase_mask=pm)
        X2 = X.clone()
        X2[:, 4] += 100.0  # phase 5(텐서 idx 4)에만 노이즈
        yh2, _, _ = model(X2, A, phase_mask=pm)
    # hazard_logits[:, k]는 텐서 idx ≤ k만 관측 → idx 0..3 불변이어야 함
    assert torch.allclose(yh[:, :4], yh2[:, :4], atol=1e-5)
    assert not torch.allclose(yh[:, 4], yh2[:, 4], atol=1e-3)


def test_unnormalized_adjacency_rejected(model):
    X, A, pm = _make_inputs()
    bad = A.clone()
    bad[:, 0, 0, 0, :12] = 0.5  # row sum = 6
    model._adjacency_checked = False
    with pytest.raises(ValueError, match="row"):
        model(X, bad, phase_mask=pm)
    model._adjacency_checked = False


def test_legacy_groupgnn_alignment():
    """B3 회귀 방지: 비연속 글로벌 팀 ID + 플레이어 없는 생존 팀."""
    torch.manual_seed(0)
    from torch_geometric.data import Data

    from model.arena_survival_net import ArenaSurvivalNet  # legacy/ (conftest sys.path)

    net = ArenaSurvivalNet(
        agent_feat_dim=39, hidden_dim=32, n_encoder_layers=1,
        n_group_gnn_layers=2, n_gru_layers=1, n_heads=4, dropout=0.0,
    ).eval()

    agent_h = torch.randn(4, 32)
    p_team_idx = torch.tensor([3, 3, 7, 12])

    with torch.no_grad():
        pooled, _ = net._pool_alive_teams(agent_h, p_team_idx, [3, 7, 12])
        assert pooled.shape == (3, 32)
        ref, _ = net.group_pooling(agent_h, p_team_idx, 13)
        assert torch.allclose(pooled[2], ref[12])

        # 플레이어 row가 없는 생존 팀(과거 IndexError) — 크래시 없이 행 생성
        pooled2, _ = net._pool_alive_teams(agent_h, p_team_idx, [3, 7, 12, 20])
        assert pooled2.shape == (4, 32)

        tg = Data()
        tg.x = torch.randn(3, 8)
        tg.num_nodes = 3
        tg.edge_index = torch.tensor([[0, 1], [1, 0]])
        tg.edge_attr = torch.randn(2, 4)
        tg.alive_teams = [3, 7, 12]
        out, _ = net.group_gnn(pooled, tg)
        assert out.shape == (3, 32)

        with pytest.raises(ValueError, match="num_nodes"):
            net.group_gnn(torch.randn(5, 32), tg)
