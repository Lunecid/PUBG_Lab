from __future__ import annotations

import torch

from model_stgcn.v3.adv_model import shuffle_adj
from model_stgcn.v3.edge_controls import apply_phase_edge_control


def _rows_as_tuples(x: torch.Tensor) -> list[tuple[float, ...]]:
    return sorted(tuple(float(v) for v in row) for row in x.reshape(-1, x.shape[-1]))


def test_v3_edge_control_observed_returns_input():
    A = torch.rand(2, 3, 2, 4, 4)
    node_mask = torch.ones(2, 3, 4, dtype=torch.bool)
    out = apply_phase_edge_control(A, node_mask, control="observed")
    assert out is A


def test_v3_node_shuffle_preserves_shape_and_row_stochastic_alive_block():
    A = torch.zeros(1, 1, 2, 4, 4)
    A[0, 0, :, :3, :3] = torch.tensor(
        [
            [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.25, 0.25, 0.5]],
            [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [1.0, 0.0, 0.0]],
        ]
    )
    node_mask = torch.tensor([[[True, True, True, False]]])
    gen = torch.Generator().manual_seed(7)
    out = apply_phase_edge_control(A, node_mask, control="node_shuffle", generator=gen)
    assert out.shape == A.shape
    row_sums = out[0, 0, :, :3, :3].sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums))
    assert torch.count_nonzero(out[0, 0, :, 3]).item() == 0


def test_v3_edge_shuffle_preserves_density():
    A = torch.zeros(1, 1, 1, 3, 3)
    A[0, 0, 0] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 1.0, 0.0]])
    node_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    gen = torch.Generator().manual_seed(11)
    out = apply_phase_edge_control(A, node_mask, control="edge_shuffle", generator=gen)
    assert out.shape == A.shape
    assert int(torch.count_nonzero(out)) == int(torch.count_nonzero(A))


def test_advantage_edge_shuffle_is_phase_local_and_preserves_feature_multiset():
    A_feat = torch.zeros(1, 2, 4, 4, 3)
    A_mask = torch.zeros(1, 2, 4, 4)
    node_mask = torch.tensor([[[True, True, True, False], [True, True, False, False]]])
    for t, alive in enumerate(([0, 1, 2], [0, 1])):
        for i in alive:
            for j in alive:
                if i == j:
                    continue
                A_mask[0, t, i, j] = 1.0
                A_feat[0, t, i, j] = torch.tensor([float(t + 1), float(i), float(j)])

    out_feat, out_mask = shuffle_adj(
        A_feat,
        A_mask,
        node_mask,
        "edge_shuffle",
        torch.Generator().manual_seed(17),
    )

    assert out_feat.shape == A_feat.shape
    assert out_mask.shape == A_mask.shape
    assert out_mask[0, 0, 3].sum() == 0
    assert out_mask[0, 0, :, 3].sum() == 0
    assert out_feat[0, 1, 2:].abs().sum() == 0
    assert out_feat[0, 1, :, 2:].abs().sum() == 0

    for t, alive in enumerate(([0, 1, 2], [0, 1])):
        idx = torch.tensor(alive)
        in_block = torch.cat(
            [A_mask[0, t][idx][:, idx].reshape(-1, 1), A_feat[0, t][idx][:, idx].reshape(-1, 3)],
            dim=1,
        )
        out_block = torch.cat(
            [out_mask[0, t][idx][:, idx].reshape(-1, 1), out_feat[0, t][idx][:, idx].reshape(-1, 3)],
            dim=1,
        )
        assert _rows_as_tuples(in_block) == _rows_as_tuples(out_block)
        assert int(out_mask[0, t][idx][:, idx].sum()) == int(A_mask[0, t][idx][:, idx].sum())
