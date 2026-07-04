# Snapshot-Resolution Phase Survival Design

## Corrected Model Contract

The model input must stay at the minimum available temporal resolution:

```text
X_s[t, team, feature]          10s team snapshot features
A_s[t, edge_type, team, team]  10s dynamic graph
phase_of_snap[t]               phase id for output alignment only
```

The model predicts phase-level survival/hazard at causal cut points:

```text
snapshots <= phase k  ->  P(eliminated during phase k+1)
```

Phase is therefore an output/evaluation resolution, not the input resolution.

## Why The Old Design Was Wrong For This Goal

`train_phase.py` and V3 `model=phase` consume `fetch_phase_match_tensors()`, whose
time axis is `PHASE_T_MAX=8`. Its features come from phase-level tables such as
`team_phase_state`, `team_phase_combat`, `team_phase_loot`,
`team_phase_network_features`, and `team_phase_habitat_features`. Those tables
summarize telemetry inside each phase with aggregates such as start/end, avg,
min, max, count, and sum.

That is a valid phase-level graph model, but it is not a 10s snapshot ST model.

MR models (`train_mr.py`, V3 `model=mr`) do ingest `X_s/A_s`, but they pool
snapshot embeddings back into phase vectors before the main phase head. They are
snapshot-informed phase models, not pure snapshot-resolution phase survival
models.

## New Code Path

Use this path for the corrected design:

```text
stdb.team_snapshot_features
  -> stdb.snapshot_queries.fetch_snapshot_match_tensors()
  -> model_stgcn.snapshot_phase_dataset
  -> model_stgcn.snapshot_phase_survival.SnapshotPhaseSurvivalGNN
  -> model_stgcn.train_snapshot_phase
```

The phase label tables are still used for supervision:

```text
stdb.team_phase_labels
  -> y_hazard / target_mask
```

They are not used as the primary model input.

## Architecture

```text
10s snapshot sequence
    X_s, A_s
        |
        v
causal SnapshotEncoder
    h[t, team]
        |
        v
phase cut selector
    h_cut[k, team] = h[last snapshot with phase <= k]
        |
        v
hazard head
    P(eliminated in next phase | snapshots <= phase k)
```

The selector is implemented in `phase_cut_indices()` and
`gather_phase_cut_embeddings()`.

## Reproduction Command

```bash
python3 -m model_stgcn.train_snapshot_phase \
  --mode squad \
  --epochs 12 \
  --batch 8 \
  --device auto \
  --cached \
  --checkpoint model_stgcn/checkpoints/snapshot_phase_survival.pt
```

## Causality Invariant

For any cut phase `k`, predictions for phase cuts `<= k` must be unchanged if
snapshots from phases `> k` are perturbed or removed.

The synthetic regression guard is:

```bash
python3 -m pytest tests/test_snapshot_phase_survival.py -q
```

