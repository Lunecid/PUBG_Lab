# Paper ST Domain-Channel Result (2026-06-23)

## Purpose

This result corrects the pooled fast run by slicing experiments by Battleground
domain:

```text
map_name x game_mode/user-version x feature/relation channel
```

The run tests whether the prior-aligned ST feature families behave differently
by map and user-version instead of mixing all matches into one pooled cohort.

## Code Changes

Trainer:

```text
model_stgcn/train_snapshot_phase.py
```

Added:

```text
--snapshot-feature-preset full|trajectory|zone|viability|resource|viability_resource|trajectory_zone
--snapshot-edge-preset full|proximity|combat|none
```

Runner:

```text
scripts/run_snapshot_domain_channel_matrix.sh
```

This runner preserves a separate split per `map/mode` domain and runs channel
variants on that same split.

## Run Envelopes

TPP squad primary maps:

```text
SNAP_DOMAIN_OUT_DIR=model_stgcn/checkpoints/snapshot_domain_channel_primary_tpp_mps300_e1_20260623
SNAP_DOMAIN_PRESET=primary_tpp
SNAP_DOMAIN_LIMIT=300
SNAP_DOMAIN_EPOCHS=1
SNAP_DOMAIN_BATCH=8
SNAP_DOMAIN_DEVICE=mps
```

FPP squad auxiliary maps:

```text
SNAP_DOMAIN_OUT_DIR=model_stgcn/checkpoints/snapshot_domain_channel_fpp_aux_mps250_e1_20260623
SNAP_DOMAIN_PRESET=fpp_aux
SNAP_DOMAIN_LIMIT=250
SNAP_DOMAIN_EPOCHS=1
SNAP_DOMAIN_BATCH=8
SNAP_DOMAIN_DEVICE=mps
```

Artifacts:

```text
model_stgcn/checkpoints/snapshot_domain_channel_primary_tpp_mps300_e1_20260623/domain_channel_metrics_seed42.csv
model_stgcn/checkpoints/snapshot_domain_channel_fpp_aux_mps250_e1_20260623/domain_channel_metrics_seed42.csv
```

## Domain Winners

| Map | User version / mode | Best channel variant | Test loss | Best val |
| --- | --- | --- | ---: | ---: |
| `Baltic_Main` | `squad` | `tcn_viability_resource` | 0.6038 | 0.6095 |
| `Tiger_Main` | `squad` | `tcn_viability_resource` | 0.5608 | 0.5662 |
| `Desert_Main` | `squad` | `tcn_viability_resource` | 0.5770 | 0.5771 |
| `Neon_Main` | `squad` | `tcn_viability_resource` | 0.5788 | 0.5606 |
| `Savage_Main` | `squad` | `tcn_viability_resource` | 0.5826 | 0.5872 |
| `Baltic_Main` | `squad-fpp` | `tcn_viability_resource` | 0.5874 | 0.6083 |
| `Tiger_Main` | `squad-fpp` | `tcn_viability_resource` | 0.6236 | 0.6026 |
| `Desert_Main` | `squad-fpp` | `tcn_viability_resource` | 0.6148 | 0.5966 |
| `Neon_Main` | `squad-fpp` | `tcn_viability_resource` | 0.6187 | 0.6049 |

## Mean Channel Ranking Across 9 Domains

| Variant | Mean test loss | SD | Interpretation |
| --- | ---: | ---: | --- |
| `tcn_viability_resource` | 0.5942 | 0.0206 | strongest channel; roster survival capital dominates fast pilot |
| `tcn_trajectory` | 0.6178 | 0.0181 | second-best node channel; movement history matters |
| `rgcn_full` | 0.6203 | 0.0176 | relation channel helps less than viability/resource |
| `rgcn_proximity` | 0.6203 | 0.0176 | local opponent field roughly equals full relation |
| `gate_only_full` | 0.6204 | 0.0179 | graph capacity/contact control tied with observed relation |
| `rgcn_full_edge_shuffle` | 0.6206 | 0.0181 | topology-null tied with observed relation |
| `rgcn_combat` | 0.6214 | 0.0174 | combat-only relation slightly behind proximity |
| `last_full` | 0.6311 | 0.0125 | latest snapshot baseline |
| `tcn_trajectory_zone` | 0.6320 | 0.0157 | trajectory+zone did not improve in 1-epoch fast pilot |
| `tcn_full` | 0.6362 | 0.0214 | full node channel likely needs longer training/tuning |
| `tcn_zone` | 0.6411 | 0.0155 | weakest standalone channel |

## Interpretation

Supported by this corrected fast run:

```text
The strongest Battleground ST channel is team viability/resource state:
health, alive roster count, and vehicle/mobility resource. It wins in all
tested map/user-version cohorts.
```

Secondary signal:

```text
Trajectory state is consistently useful and ranks second on average. Movement
history is meaningful, but less directly predictive than current roster
survival capital in the 1-epoch fast pilot.
```

Not supported yet:

```text
Observed relation topology. R-GCN full, proximity-only, edge-shuffle null, and
gate-only controls are tightly grouped. This suggests graph capacity or contact
scaffolding, not semantic observed topology, at this pilot scale.
```

Important caveat:

```text
This is a fast MPS pilot with one epoch per domain. It is appropriate for
feature/channel triage, not final paper metrics. The full node feature channel
may need longer training because it has more active inputs than the masked
single-channel variants.
```

## Paper-Facing Takeaway

The paper should not claim that every ST channel contributes equally. The
corrected domain-channel matrix supports a more precise claim:

```text
Across map and squad/FPP user-version slices, causal team survival-capital
features are the dominant ST signal; trajectory features provide a consistent
secondary signal; zone-only and observed graph-topology signals require more
careful modeling or longer confirmation.
```
