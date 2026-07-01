from __future__ import annotations

import torch

from model_stgcn.v3.edge_controls import apply_phase_edge_control


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
