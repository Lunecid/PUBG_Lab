from __future__ import annotations

import torch

from model_stgcn.snapshot_baselines import SnapshotLastOnlyHazard, SnapshotTCNNoGraphHazard
from model_stgcn.snapshot_phase_dataset import snapshot_phase_collate


def test_snapshot_last_only_baseline_forward_shape():
    B, Ts, N, F = 2, 18, 6, 13
    X = torch.randn(B, Ts, N, F)
    phase_of_snap = (torch.arange(Ts).view(1, Ts).repeat(B, 1) // 2).clamp(max=8)
    node_mask = torch.ones(B, Ts, N, dtype=torch.bool)
    model = SnapshotLastOnlyHazard(F, hidden=8, dropout=0.0)

    hazard, cut_mask = model(X, phase_of_snap, node_mask)

    assert hazard.shape == (B, 7, N)
    assert cut_mask.shape == (B, 7)


def test_snapshot_tcn_no_graph_baseline_forward_shape():
    B, Ts, N, F = 2, 18, 6, 13
    X = torch.randn(B, Ts, N, F)
    phase_of_snap = (torch.arange(Ts).view(1, Ts).repeat(B, 1) // 2).clamp(max=8)
    node_mask = torch.ones(B, Ts, N, dtype=torch.bool)
    model = SnapshotTCNNoGraphHazard(F, hidden=8, num_layers=1, dropout=0.0)

    hazard, cut_mask = model(X, phase_of_snap, node_mask)

    assert hazard.shape == (B, 7, N)
    assert cut_mask.shape == (B, 7)


def test_snapshot_tcn_no_graph_has_no_future_snapshot_leak_to_past_cuts():
    B, Ts, N, F = 1, 18, 6, 13
    torch.manual_seed(11)
    X = torch.randn(B, Ts, N, F)
    phase_of_snap = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8]])
    node_mask = torch.ones(B, Ts, N, dtype=torch.bool)
    model = SnapshotTCNNoGraphHazard(F, hidden=8, num_layers=1, dropout=0.0).eval()

    with torch.no_grad():
        hazard0, _ = model(X, phase_of_snap, node_mask)
        X2 = X.clone()
        X2[:, phase_of_snap.squeeze(0) > 3] = torch.randn_like(X2[:, phase_of_snap.squeeze(0) > 3]) * 9
        hazard1, _ = model(X2, phase_of_snap, node_mask)

    assert torch.allclose(hazard0[:, :3], hazard1[:, :3], atol=1e-5)
    assert not torch.allclose(hazard0[:, 3], hazard1[:, 3], atol=1e-3)


def test_no_graph_collate_can_omit_edge_tensor():
    batch = []
    for i in range(2):
        batch.append(
            {
                "X_s": torch.randn(5, 4, 3),
                "phase_of_snap": torch.arange(5),
                "node_mask_s": torch.ones(5, 4, dtype=torch.bool),
                "snapshot_ts": torch.arange(5),
                "y_hazard": torch.zeros(7, 4),
                "target_mask": torch.ones(7, 4, dtype=torch.bool),
                "phase_mask": torch.ones(8, dtype=torch.bool),
                "team_mask": torch.ones(4, dtype=torch.bool),
                "match_id": f"m{i}",
                "team_ids": [1, 2, 3, 4],
                "phase_indices": list(range(1, 9)),
            }
        )

    out = snapshot_phase_collate(batch)

    assert "A_s" not in out
    assert out["X_s"].shape == (2, 5, 4, 3)
    assert out["match_id"] == ["m0", "m1"]
