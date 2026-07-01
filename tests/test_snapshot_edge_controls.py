from __future__ import annotations

import torch

from model_stgcn.snapshot_edge_controls import (
    _edge_shuffle_edges,
    apply_snapshot_edge_control,
)


def _synthetic() -> tuple[torch.Tensor, torch.Tensor]:
    A = torch.zeros(1, 2, 2, 5, 5)
    node_mask = torch.zeros(1, 2, 5, dtype=torch.bool)
    alive_by_t = {0: [0, 1, 2, 3], 1: [0, 1, 2]}
    for t, alive in alive_by_t.items():
        node_mask[0, t, alive] = True
        for e in range(2):
            for i in alive:
                for j in alive:
                    if i != j:
                        A[0, t, e, i, j] = (t + 1) * 100.0 + e * 10.0 + i * 3.0 + j
    return A, node_mask


def test_snapshot_edge_shuffle_stays_inside_alive_block():
    A, node_mask = _synthetic()
    out = _edge_shuffle_edges(
        A,
        node_mask,
        generator=torch.Generator().manual_seed(5),
    )

    assert out[0, 0, :, 4].abs().sum() == 0
    assert out[0, 0, :, :, 4].abs().sum() == 0
    assert out[0, 1, :, 3:].abs().sum() == 0
    assert out[0, 1, :, :, 3:].abs().sum() == 0


def test_snapshot_edge_shuffle_preserves_alive_block_value_multiset():
    A, node_mask = _synthetic()
    out = _edge_shuffle_edges(
        A,
        node_mask,
        generator=torch.Generator().manual_seed(7),
    )

    for t, alive in {0: [0, 1, 2, 3], 1: [0, 1, 2]}.items():
        idx = torch.tensor(alive)
        for e in range(2):
            sub_in = A[0, t, e][idx][:, idx]
            sub_out = out[0, t, e][idx][:, idx]
            assert torch.equal(sub_in.flatten().sort().values, sub_out.flatten().sort().values)


def test_apply_snapshot_edge_shuffle_row_normalizes_alive_rows():
    A = torch.zeros(1, 1, 1, 3, 3)
    A[0, 0, 0] = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 2.0],
            [3.0, 0.0, 0.0],
        ]
    )
    node_mask = torch.ones(1, 1, 3, dtype=torch.bool)

    out = apply_snapshot_edge_control(
        A,
        node_mask,
        control="edge_shuffle",
        generator=torch.Generator().manual_seed(11),
    )

    row_sums = out.sum(dim=-1)
    assert torch.all((row_sums == 0) | torch.isclose(row_sums, torch.ones_like(row_sums)))
    assert int(torch.count_nonzero(out)) == int(torch.count_nonzero(A))


def test_apply_snapshot_gate_only_binarizes_and_row_normalizes():
    A = torch.zeros(1, 1, 1, 4, 4)
    A[0, 0, 0] = torch.tensor(
        [
            [0.0, 0.2, 3.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 4.0],
            [9.0, 9.0, 9.0, 9.0],
        ]
    )
    node_mask = torch.tensor([[[True, True, True, False]]])

    out = apply_snapshot_edge_control(A, node_mask, control="gate_only")

    expected = torch.zeros_like(A)
    expected[0, 0, 0, :3, :3] = torch.tensor(
        [
            [0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    torch.testing.assert_close(out, expected)
