from __future__ import annotations

import torch

from pubg_lab.survival.models.snapshot_phase_survival import SnapshotPhaseSurvivalGNN, phase_cut_indices, snapshot_phase_hazard_loss


def _identity_adjacency(B: int, T: int, E: int, N: int) -> torch.Tensor:
    A = torch.zeros(B, T, E, N, N)
    eye = torch.eye(N).view(1, 1, N, N)
    A[:, :, 0] = eye
    return A


def test_phase_cut_indices_select_last_observed_snapshot_before_cut():
    phase_of_snap = torch.tensor([[0, 0, 1, 2, 2, 4, 4, 0, 0]])
    node_mask = torch.zeros(1, 9, 3, dtype=torch.bool)
    node_mask[:, :7] = True

    idx, mask = phase_cut_indices(phase_of_snap, node_mask, t_phase=5)

    assert idx.tolist() == [[2, 4, 4, 6, 6]]
    assert mask.tolist() == [[True, True, True, True, True]]


def test_snapshot_phase_model_forward_shape():
    B, Ts, N, F, E = 2, 18, 6, 13, 2
    X = torch.randn(B, Ts, N, F)
    A = _identity_adjacency(B, Ts, E, N)
    phase_of_snap = (torch.arange(Ts).view(1, Ts).repeat(B, 1) // 2).clamp(max=8)
    node_mask = torch.ones(B, Ts, N, dtype=torch.bool)

    model = SnapshotPhaseSurvivalGNN(
        F,
        snap_hidden=8,
        snap_mid=12,
        snap_blocks=1,
        snap_edge_types=E,
        dropout=0.0,
    )
    hazard, cut_mask = model(X, A, phase_of_snap, node_mask)

    assert hazard.shape == (B, 7, N)
    assert cut_mask.shape == (B, 7)


def test_snapshot_phase_model_has_no_future_snapshot_leak_to_past_cuts():
    B, Ts, N, F, E = 1, 18, 6, 13, 2
    torch.manual_seed(7)
    X = torch.randn(B, Ts, N, F)
    A = _identity_adjacency(B, Ts, E, N)
    phase_of_snap = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8]])
    node_mask = torch.ones(B, Ts, N, dtype=torch.bool)
    model = SnapshotPhaseSurvivalGNN(
        F,
        snap_hidden=8,
        snap_mid=12,
        snap_blocks=1,
        snap_edge_types=E,
        dropout=0.0,
    ).eval()

    with torch.no_grad():
        hazard0, _ = model(X, A, phase_of_snap, node_mask)
        X2 = X.clone()
        X2[:, phase_of_snap.squeeze(0) > 3] = torch.randn_like(X2[:, phase_of_snap.squeeze(0) > 3]) * 9
        hazard1, _ = model(X2, A, phase_of_snap, node_mask)

    assert torch.allclose(hazard0[:, :3], hazard1[:, :3], atol=1e-5)
    assert not torch.allclose(hazard0[:, 3], hazard1[:, 3], atol=1e-3)


def test_snapshot_phase_loss_masks_invalid_cuts():
    hazard = torch.zeros(1, 7, 3)
    y = torch.ones(1, 7, 3)
    target_mask = torch.ones(1, 7, 3, dtype=torch.bool)
    cut_mask = torch.tensor([[True, False, False, False, False, False, False]])

    loss, parts = snapshot_phase_hazard_loss(hazard, y, target_mask, cut_mask=cut_mask)

    assert torch.isfinite(loss)
    assert parts["hazard_n"] == 3.0
