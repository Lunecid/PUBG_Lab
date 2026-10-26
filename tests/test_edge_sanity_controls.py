"""edge_sanity 셔플 컨트롤 불변식 — phase-local 교정(FEATURE_DESIGN_V2 §9) 회귀 가드.

합성 텐서만 사용 (DB 불필요). 핵심 보장:
- 셔플이 phase별 생존 노드 블록 안에서만 일어난다 (사망/패딩 행 미오염)
- 생존 블록의 값 multiset·행합 multiset(=강도 시퀀스)이 보존된다
- node_shuffle은 채널 간 동일 순열 (노드 재라벨 의미론)
- 같은 seed → 같은 결과 (재현성)
"""

from __future__ import annotations

import torch

from pubg_lab.survival.features.edge_sanity import _edge_shuffle_edges, _node_shuffle_edges

B, T, E, N = 1, 3, 2, 6
ALIVE = {0: [0, 1, 2, 3], 1: [0, 1, 2], 2: [0, 1]}  # 매 phase 줄어드는 생존 집합


def _synthetic(channel_identical: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    A = torch.zeros(B, T, E, N, N)
    node_mask = torch.zeros(B, T, N, dtype=torch.bool)
    for t, alive in ALIVE.items():
        node_mask[0, t, alive] = True
        for e in range(E):
            for i in alive:
                for j in alive:
                    if i == j:
                        continue
                    base = (t + 1) * 100.0 + i * 10.0 + j
                    A[0, t, e, i, j] = base if channel_identical else base + e * 0.5
    return A, node_mask


def _dead_entries_zero(out: torch.Tensor) -> bool:
    for t, alive in ALIVE.items():
        dead = [n for n in range(N) if n not in alive]
        if out[0, t][:, dead, :].abs().sum() > 0 or out[0, t][:, :, dead].abs().sum() > 0:
            return False
    return True


def test_node_shuffle_is_phase_local():
    A, node_mask = _synthetic()
    gen = torch.Generator().manual_seed(7)
    out = _node_shuffle_edges(A, node_mask, generator=gen)
    assert _dead_entries_zero(out), "사망/패딩 노드 행·열은 0이어야 한다"


def test_node_shuffle_preserves_values_and_strength_sequence():
    A, node_mask = _synthetic()
    gen = torch.Generator().manual_seed(7)
    out = _node_shuffle_edges(A, node_mask, generator=gen)
    for t, alive in ALIVE.items():
        idx = torch.tensor(alive)
        for e in range(E):
            sub_in = A[0, t, e][idx][:, idx]
            sub_out = out[0, t, e][idx][:, idx]
            assert torch.equal(
                sub_in.flatten().sort().values, sub_out.flatten().sort().values
            ), f"값 multiset 변형 t={t} e={e}"
            assert torch.allclose(
                sub_in.sum(dim=1).sort().values, sub_out.sum(dim=1).sort().values
            ), f"행합(수신 강도) 시퀀스 변형 t={t} e={e}"
            assert torch.allclose(
                sub_in.sum(dim=0).sort().values, sub_out.sum(dim=0).sort().values
            ), f"열합 시퀀스 변형 t={t} e={e}"


def test_node_shuffle_shares_permutation_across_channels():
    A, node_mask = _synthetic(channel_identical=True)
    gen = torch.Generator().manual_seed(11)
    out = _node_shuffle_edges(A, node_mask, generator=gen)
    assert torch.equal(out[0, :, 0], out[0, :, 1]), (
        "채널 간 인접행렬이 같으면 셔플 결과도 같아야 한다 (단일 노드 재라벨)"
    )


def test_node_shuffle_changes_alignment_for_some_seed():
    A, node_mask = _synthetic()
    changed = False
    for seed in range(5):
        gen = torch.Generator().manual_seed(seed)
        out = _node_shuffle_edges(A, node_mask, generator=gen)
        if not torch.equal(out, A):
            changed = True
            break
    assert changed, "다섯 seed 모두 항등 순열일 수는 없다"


def test_controls_deterministic_given_seed():
    A, node_mask = _synthetic()
    for fn in (_node_shuffle_edges, _edge_shuffle_edges):
        a = fn(A, node_mask, generator=torch.Generator().manual_seed(3))
        b = fn(A, node_mask, generator=torch.Generator().manual_seed(3))
        assert torch.equal(a, b), f"{fn.__name__} 비결정적"


def test_edge_shuffle_is_phase_local_and_density_preserving():
    A, node_mask = _synthetic()
    gen = torch.Generator().manual_seed(5)
    out = _edge_shuffle_edges(A, node_mask, generator=gen)
    assert _dead_entries_zero(out)
    for t, alive in ALIVE.items():
        idx = torch.tensor(alive)
        for e in range(E):
            sub_in = A[0, t, e][idx][:, idx]
            sub_out = out[0, t, e][idx][:, idx]
            assert (sub_in != 0).sum() == (sub_out != 0).sum(), f"밀도 변형 t={t} e={e}"
            assert torch.equal(
                sub_in.flatten().sort().values, sub_out.flatten().sort().values
            ), f"값 multiset 변형 t={t} e={e}"
