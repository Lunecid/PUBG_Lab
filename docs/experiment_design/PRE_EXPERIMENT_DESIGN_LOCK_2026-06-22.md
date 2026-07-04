# Pre-Experiment Design Lock (2026-06-22)

## Stage Definition

Current stage:

```text
Pre-experiment design lock + pilot validation
```

This is not the full-training stage.

The research direction is fixed:

```text
Battle Royale is formalized as a Dynamic Shrinking-Arena Competitive Graph
(DSACG), and PUBG telemetry is used as a team-level projection to predict
phase-wise survival hazard and final placement.
```

The next work is to lock the experimental contract in code before large model
training.

## Fixed Decisions

| Item | Decision |
| --- | --- |
| Genre formalism | Dynamic Shrinking-Arena Competitive Graph |
| Empirical case | PUBG telemetry |
| Projection | team-level graph projection |
| Node | team |
| Minimum input clock | 10-second snapshot |
| Output clock | phase cut |
| Main target | next-phase elimination hazard |
| Secondary target | final rank / top-k placement |
| Required controls | no-graph, edge-shuffle, gate-only, relative-edge |

## Validation Code Added

| File | Purpose |
| --- | --- |
| `tools/validate_genre_feature_manifest.py` | validates feature/channel manifest schema, allowed values, leakage rules, and coverage against legacy feature-family manifest |
| `tools/smoke_snapshot_phase_contract.py` | checks snapshot tensor shape, phase label/mask shape, at-risk target consistency, and one-pass model forward |
| `model_stgcn/snapshot_baselines.py` | adds `snapshot_last_only` and `snapshot_tcn_no_graph` no-graph baseline modules |
| `tests/test_snapshot_baselines.py` | synthetic forward and causal-blindness tests for no-graph baselines |

## Validation Results

Manifest validation:

```text
python3 tools/validate_genre_feature_manifest.py
```

Result:

```text
rows=229
errors=0
warnings=0
```

The v0 wildcard rows such as `net_combat_*`, `hab_safe_area_*`, and
`created_*` have been expanded to physical DB/model feature columns. The
manifest also separates the snapshot `vehicle_ratio` channel from the phase
aggregate `vehicle_ratio` column.

Synthetic forward checks:

```text
direct test functions passed
```

`pytest` is not installed in the current system Python, so the test functions
were executed directly.

DB tensor/label smoke:

```text
PG_PASSWORD=... python3 tools/smoke_snapshot_phase_contract.py --limit 5 --run-forward
```

Result:

```text
requested=5 checked=5 failures=0
X_s = (200, 128, 13)
A_s = (200, 2, 128, 128)
y_hazard = (7, 128)
forward = True
```

Tiny no-graph pilots:

```text
PG_PASSWORD=... python3 -m model_stgcn.train_snapshot_phase \
  --model-variant snapshot_last_only --limit 20 --epochs 2 --cached ...

PG_PASSWORD=... python3 -m model_stgcn.train_snapshot_phase \
  --model-variant snapshot_tcn_no_graph --limit 20 --epochs 2 --cached ...
```

Result:

Important correction after diagnostics:

```text
The first pilot summaries without the phasefix suffix are stale for performance
interpretation. They verified code paths, but `phase_of_snap` could select
match-tail snapshots as the phase cut in some held-out matches.

`stdb/snapshot_queries.py` now derives phase_of_snap from
stdb.v_match_phase_bounds, so the current valid tiny-pilot artifacts are the
*_phasefix_limit20 summaries below.
```

| Variant | Split | Best val loss | Test loss | Test acc | Artifact |
| --- | --- | ---: | ---: | ---: | --- |
| `snapshot_last_only` | 16/2/2 matches | 0.6518 | 0.6452 | 0.6667 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_last_only_phasefix_limit20.summary.json` |
| `snapshot_tcn_no_graph` | 16/2/2 matches | 0.6810 | 0.6774 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_tcn_no_graph_phasefix_limit20.summary.json` |

Tiny graph pilot:

```text
PG_PASSWORD=... python3 -m model_stgcn.train_snapshot_phase \
  --model-variant snapshot_rgcn --limit 20 --epochs 2 --cached ...
```

Result:

| Variant | Split | Best val loss | Test loss | Test acc | Artifact |
| --- | --- | ---: | ---: | ---: | --- |
| `snapshot_rgcn` | 16/2/2 matches | 0.6526 | 0.6474 | 0.6667 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_rgcn_phasefix_limit20.summary.json` |
| `snapshot_rgcn_edge_shuffle` | 16/2/2 matches | 0.6520 | 0.6474 | 0.6667 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_rgcn_edge_shuffle_phasefix_limit20.summary.json` |
| `snapshot_gate_only` | 16/2/2 matches | 0.6522 | 0.6466 | 0.6667 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_gate_only_phasefix_limit20.summary.json` |
| `snapshot_relative_edge` | 16/2/2 matches | 0.6416 | 0.6151 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_phasefix_limit20.summary.json` |

Interpretation:

- `snapshot_rgcn` confirms the C1 + C2 snapshot graph path trains end-to-end.
- `snapshot_rgcn_edge_shuffle` confirms the topology-null path trains end-to-end.
- `snapshot_gate_only` confirms the binary contact-gate path trains end-to-end.
- `snapshot_relative_edge` confirms the derived `E_s` contact-gated relative
  edge path trains end-to-end.
- After the phase-cut fix, observed R-GCN still does not beat
  `snapshot_last_only` on loss and still does not beat the edge-shuffle null.
- Contact-gate-only is effectively tied with observed R-GCN and edge-shuffle,
  so this tiny run does not support contact topology alone as the missing
  signal.
- The corrected minimal `snapshot_relative_edge` pilot is the best tiny run on
  val/test loss, but this is not yet evidence of observed graph-topology
  benefit. It is evidence that the derived relative-edge feature path can carry
  signal under the locked channel contract.
- The diagnostic artifact
  `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_diagnostics_after_phase_fix.json`
  reports target-cut coverage recovered to 0.9579 overall, but the current
  contact gate is still dense: edge density over alive ordered pairs is 1.0 and
  `contact_gate` mean is 1.0. Therefore the current gate is effectively an
  alive-pair complete graph, not a sparse contact graph.
- The current conclusion is: raw proximity/combat topology is not supported by
  this tiny pilot; relative feature construction is promising but needs a real
  sparse/contact or advantage gate before scale-up.

Sparse relative-edge follow-up:

```text
PG_PASSWORD=... python3 tools/diagnose_snapshot_relative_edge.py \
  --relative-edge-gate knn_combat --relative-edge-k 5 --modes raw ...

PG_PASSWORD=... python3 -m model_stgcn.train_snapshot_phase \
  --model-variant snapshot_relative_edge_sparse \
  --relative-edge-gate knn_combat --relative-edge-k 5 \
  --limit 20 --epochs 2 --cached ...
```

Result:

| Variant | Split | Gate density | Best val loss | Test loss | Test acc | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `snapshot_relative_edge_sparse` | 16/2/2 matches | 0.1569 | 0.6440 | 0.6144 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k5_phasefix_limit20.summary.json` |

Diagnostic artifact:

```text
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k5_diagnostics.json
```

Interpretation:

- The new `knn_combat` gate fixes the previous complete-graph problem:
  edge density over alive ordered pairs falls from 1.0 to 0.1569.
- Target-cut coverage remains usable: target any-edge coverage is 0.9579,
  out-edge coverage is 0.9579, and in-edge coverage is 0.9421.
- The sparse pilot remains better than R-GCN, edge-shuffle, gate-only, and
  no-graph last-only on this tiny split's test loss. It is close to dense
  relative-edge on validation loss and slightly better on test loss.
- This is still not a full claim. It supports the next design step:
  run sparse relative-edge null controls and k-sensitivity before full training.

Sparse relative-edge feature null:

```text
PG_PASSWORD=... python3 tools/diagnose_snapshot_relative_edge.py \
  --relative-edge-gate knn_combat --relative-edge-k 5 \
  --relative-edge-null feature_roll --modes raw ...

PG_PASSWORD=... python3 -m model_stgcn.train_snapshot_phase \
  --model-variant snapshot_relative_edge_sparse_null \
  --relative-edge-gate knn_combat --relative-edge-k 5 \
  --limit 20 --epochs 2 --cached ...
```

Result:

| Variant | Split | Gate density | Best val loss | Test loss | Test acc | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `snapshot_relative_edge_sparse_null` | 16/2/2 matches | 0.1569 | 0.6437 | 0.6143 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_null_k5_phasefix_limit20.summary.json` |

Diagnostic artifact:

```text
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_null_k5_diagnostics.json
```

Interpretation:

- The null preserves the same sparse gate density and target coverage as the
  real sparse variant: density 0.1569, target any-edge coverage 0.9579.
- The null is effectively tied with the real sparse relative-edge pilot and is
  microscopically lower on this tiny split's val/test loss.
- Therefore this tiny pilot does not support a claim that aligned relative-edge
  feature semantics are the source of the gain. The safer interpretation is
  that sparse kNN/combat pooling, model capacity/regularization, or neighbor
  scaffold effects are carrying most of the observed tiny-pilot improvement.
- Next step before scale-up: k-sensitivity and a topology-level sparse null
  that breaks kNN destinations while preserving row degree.

Sparse topology null:

```text
PG_PASSWORD=... python3 tools/diagnose_snapshot_relative_edge.py \
  --relative-edge-gate knn_combat --relative-edge-k 5 \
  --relative-edge-null topology_roll --modes raw ...

PG_PASSWORD=... python3 -m model_stgcn.train_snapshot_phase \
  --model-variant snapshot_relative_edge_sparse_topology_null \
  --relative-edge-gate knn_combat --relative-edge-k 5 \
  --limit 20 --epochs 2 --cached ...
```

Result:

| Variant | Split | Gate density | Best val loss | Test loss | Test acc | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `snapshot_relative_edge_sparse_topology_null` | 16/2/2 matches | 0.1569 | 0.6412 | 0.6155 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_topology_null_k5_phasefix_limit20.summary.json` |

Diagnostic artifact:

```text
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_topology_null_k5_diagnostics.json
```

Interpretation:

- The topology null preserves the same alive-pair edge density as real sparse
  k=5: 0.1569. It preserves target out/any coverage at 0.9579; target in
  coverage changes slightly to 0.9523 because destinations are rolled.
- The topology null is close to real sparse: better on tiny validation loss,
  slightly worse on tiny test loss.
- This further weakens any claim that kNN destination semantics are established
  by the tiny pilot. The robust observation is sparse relation scaffolding, not
  yet semantic topology.

k-sensitivity:

| Variant | k | Gate density | Best val loss | Test loss | Test acc | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `snapshot_relative_edge_sparse` | 3 | 0.0951 | 0.6438 | 0.6144 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k3_phasefix_limit20.summary.json` |
| `snapshot_relative_edge_sparse` | 5 | 0.1569 | 0.6440 | 0.6144 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k5_phasefix_limit20.summary.json` |
| `snapshot_relative_edge_sparse` | 8 | 0.2464 | 0.6430 | 0.6148 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k8_phasefix_limit20.summary.json` |

Interpretation:

- The k=3/5/8 pilots are nearly tied on this tiny split.
- The gain is not sensitive to this small k range, again suggesting the current
  signal is a sparse-neighbor scaffold effect more than a precise kNN topology
  or relative-feature semantic effect.

Pilot-scale seed repeats:

```text
seeds = 41, 42, 43
limit = 20
epochs = 2
k = 5
families = real sparse, feature-roll null, topology-roll null
```

Result:

| Seed | Variant | Best val loss | Test loss | Test acc | Artifact |
| ---: | --- | ---: | ---: | ---: | --- |
| 41 | `real_sparse` | 0.6702 | 0.6653 | 0.6415 | `model_stgcn/checkpoints/snapshot_pilot_repeats/seed41_sparse_k5.summary.json` |
| 41 | `feature_null` | 0.6703 | 0.6654 | 0.6415 | `model_stgcn/checkpoints/snapshot_pilot_repeats/seed41_sparse_feature_null_k5.summary.json` |
| 41 | `topology_null` | 0.6707 | 0.6650 | 0.6509 | `model_stgcn/checkpoints/snapshot_pilot_repeats/seed41_sparse_topology_null_k5.summary.json` |
| 42 | `real_sparse` | 0.6440 | 0.6144 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilot_repeats/seed42_sparse_k5.summary.json` |
| 42 | `feature_null` | 0.6437 | 0.6143 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilot_repeats/seed42_sparse_feature_null_k5.summary.json` |
| 42 | `topology_null` | 0.6412 | 0.6155 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilot_repeats/seed42_sparse_topology_null_k5.summary.json` |
| 43 | `real_sparse` | 0.6745 | 0.6523 | 0.6667 | `model_stgcn/checkpoints/snapshot_pilot_repeats/seed43_sparse_k5.summary.json` |
| 43 | `feature_null` | 0.6745 | 0.6523 | 0.6667 | `model_stgcn/checkpoints/snapshot_pilot_repeats/seed43_sparse_feature_null_k5.summary.json` |
| 43 | `topology_null` | 0.6754 | 0.6543 | 0.6594 | `model_stgcn/checkpoints/snapshot_pilot_repeats/seed43_sparse_topology_null_k5.summary.json` |

Aggregate:

| Variant | Mean test loss | Test loss sd | Mean test acc |
| --- | ---: | ---: | ---: |
| `real_sparse` | 0.6440 | 0.0216 | 0.6783 |
| `feature_null` | 0.6440 | 0.0217 | 0.6783 |
| `topology_null` | 0.6450 | 0.0213 | 0.6790 |

Interpretation:

- Real sparse and feature-null are indistinguishable across the three pilot
  seeds.
- Topology-null is only slightly worse on mean test loss, and the gap is tiny
  compared with split/seed variation.
- The current repeat evidence supports only a sparse-scaffold effect. It still
  does not support aligned relative-edge feature semantics or precise kNN
  destination semantics.

Main-comparator pilot repeats:

```text
seeds = 41, 42, 43
limit = 20
epochs = 2
families = last_only, tcn_no_graph, rgcn, edge_shuffle, gate_only
```

Aggregate:

| Variant | Mean test loss | Test loss sd | Mean test acc |
| --- | ---: | ---: | ---: |
| `snapshot_last_only` | 0.6639 | 0.0173 | 0.6520 |
| `snapshot_tcn_no_graph` | 0.6508 | 0.0202 | 0.6734 |
| `snapshot_rgcn` | 0.6476 | 0.0009 | 0.6583 |
| `snapshot_rgcn_edge_shuffle` | 0.6478 | 0.0008 | 0.6583 |
| `snapshot_gate_only` | 0.6476 | 0.0012 | 0.6583 |
| `snapshot_relative_edge_sparse` | 0.6440 | 0.0216 | 0.6783 |
| `snapshot_relative_edge_sparse_null` | 0.6440 | 0.0217 | 0.6783 |
| `snapshot_relative_edge_sparse_topology_null` | 0.6450 | 0.0213 | 0.6790 |

Interpretation:

- On mean test loss, the sparse family is slightly lower than R-GCN/gate-only
  and clearly lower than last-only, but the sparse-family standard deviation is
  large because split seed 42 is much more favorable than seeds 41 and 43.
- Per seed, the direction is not stable: R-GCN/gate-only beat sparse on seeds
  41 and 43, while sparse beats them strongly on seed 42.
- R-GCN, edge-shuffle, and gate-only remain almost identical, so observed
  proximity/combat topology still has no stable advantage over graph nulls.
- Current pilot evidence does not justify a semantic graph win or a stable
  sparse-scaffold win over all baselines. It just identifies the candidate
  family worth testing at a larger, properly repeated scale.

Limit-50 scale-up gate pilot:

```text
seeds = 41, 42, 43
limit = 50
epochs = 2
families = last_only, rgcn, edge_shuffle, gate_only,
           sparse_real, sparse_feature_null, sparse_topology_null
```

Sparse k=5 diagnostic:

```text
model_stgcn/checkpoints/snapshot_pilot_repeats_limit50/sparse_k5_diagnostics.json
edge_density_alive_pairs = 0.1918
target_cut_any_coverage = 0.970
```

Aggregate:

| Variant | Mean test loss | Test loss sd | Mean test acc |
| --- | ---: | ---: | ---: |
| `snapshot_last_only` | 0.6476 | 0.0116 | 0.6729 |
| `snapshot_rgcn` | 0.6289 | 0.0039 | 0.6729 |
| `snapshot_rgcn_edge_shuffle` | 0.6299 | 0.0035 | 0.6729 |
| `snapshot_gate_only` | 0.6299 | 0.0036 | 0.6729 |
| `snapshot_relative_edge_sparse` | 0.6255 | 0.0110 | 0.6769 |
| `snapshot_relative_edge_sparse_null` | 0.6255 | 0.0110 | 0.6769 |
| `snapshot_relative_edge_sparse_topology_null` | 0.6248 | 0.0108 | 0.6769 |

Interpretation:

- At limit 50, sparse-family mean loss is modestly lower than R-GCN/gate-only
  and clearly lower than last-only.
- The improvement is still not semantic: feature-null is tied with real sparse,
  and topology-null is slightly lower than real sparse on mean loss.
- R-GCN, edge-shuffle, and gate-only remain nearly identical, so observed
  proximity/combat topology still does not beat graph nulls.
- The scale-up gate outcome is: sparse scaffold is a viable candidate family
  for a narrowly scoped full matrix, but the research claim must be framed as
  sparse-scaffold/regularization unless a later full-scale semantic null gap
  appears.

Narrow full matrix runner:

```text
scripts/run_snapshot_narrow_full_matrix.sh
```

Runner scope:

```text
snapshot_last_only
snapshot_rgcn
snapshot_rgcn_edge_shuffle
snapshot_gate_only
snapshot_relative_edge_sparse
snapshot_relative_edge_sparse_null
snapshot_relative_edge_sparse_topology_null
```

Smoke validation:

```text
PG_PASSWORD=... SNAP_OUT_DIR=model_stgcn/checkpoints/snapshot_narrow_full_smoke20_v2 \
SNAP_LIMIT=20 SNAP_EPOCHS=1 SNAP_FORCE=1 \
./scripts/run_snapshot_narrow_full_matrix.sh
```

Result:

```text
model_stgcn/checkpoints/snapshot_narrow_full_smoke20_v2/matrix_metrics_seed42.csv
```

Important runtime note:

- A full lazy debug command on all 21,824 eligible matches started correctly
  with split 17,460/2,182/2,182, but after about 2.5 minutes it was still in
  the first epoch with low CPU utilization, indicating DB-bound lazy fetch.
- That debug run was interrupted intentionally before producing metrics.
- Do not treat the narrow full matrix as completed. The runner is ready, but a
  practical full run needs an I/O strategy: worker tuning, disk tensor cache,
  or an overnight/background execution plan with logs.
- The runner now exposes `SNAP_WORKERS`, `SNAP_EVAL_WORKERS`, and
  `SNAP_PREFETCH_FACTOR` for lazy-loader tuning.
- No-graph snapshot training now requests `include_edges=False`, so `A_s` is
  not fetched for last-only/TCN batches. Graph and relative-edge variants still
  request `A_s`.

Six-hour all-match sleep run, recovered:

```text
SNAP_OUT_DIR=model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022
SNAP_EPOCHS=1 SNAP_SEED=42 SNAP_BATCH=2 SNAP_HIDDEN=16 SNAP_MID=32
SNAP_NUM_BLOCKS=1 SNAP_EDGE_HIDDEN=8 SNAP_WORKERS=2 SNAP_EVAL_WORKERS=1
SNAP_PREFETCH_FACTOR=2 SNAP_DEVICE=cpu SNAP_FORCE=1
```

Scope:

```text
all 21,824 eligible matches
split = 17,460 train / 2,182 val / 2,182 test
timeout = 6 hours
```

Result artifacts:

```text
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/matrix_metrics_seed42_recovered.csv
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/matrix_metrics_seed42_partial.csv
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/matrix_status_seed42_recovered.json
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/matrix_status_seed42_partial.json
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/run.log
```

Completed metrics:

| Variant | State | Best val loss | Test loss | Test acc | Tensor owners |
| --- | --- | ---: | ---: | ---: | --- |
| `snapshot_last_only` | summary complete | 0.5308 | 0.5337 | 0.7398 | `X_s` |
| `snapshot_rgcn` | summary complete | 0.5212 | 0.5235 | 0.7455 | `X_s+A_s` |
| `snapshot_rgcn_edge_shuffle` | summary complete | 0.5263 | 0.5291 | 0.7433 | `X_s+A_s` |
| `snapshot_gate_only` | summary complete | 0.5252 | 0.5283 | 0.7435 | `X_s+A_s` |
| `snapshot_relative_edge_sparse` | summary complete | 0.5354 | 0.5378 | 0.7439 | `X_s+A_s+E_s_derived` |
| `snapshot_relative_edge_sparse_null` | summary complete after recovery | 0.5369 | 0.5397 | 0.7428 | `X_s+A_s+E_s_derived_null` |
| `snapshot_relative_edge_sparse_topology_null` | summary complete after recovery | 0.5384 | 0.5411 | 0.7426 | `X_s+A_s+E_s_derived_topology_null` |

Interpretation:

- This is not full training: it is one seed, one epoch, and timeboxed.
- Among completed variants, `snapshot_rgcn` is best on validation loss, test
  loss, and test accuracy.
- `snapshot_rgcn` improves test loss over `snapshot_last_only` by about 0.0102,
  over `snapshot_rgcn_edge_shuffle` by about 0.0056, and over
  `snapshot_gate_only` by about 0.0048.
- This gives tentative all-match evidence that relation channels may help
  beyond no-graph, edge-shuffle, and gate-only controls under the current
  model/loader settings.
- The sparse relative-edge family is worse than R-GCN, edge-shuffle, gate-only,
  and last-only in this run. Within the sparse family, real sparse is better
  than feature-null and topology-null, but all three are behind the R-GCN
  comparator set.
- Do not claim a semantic graph win yet. This is still one seed and one epoch,
  with no repeat or longer-training confirmation.

No-graph summaries record consumed channel families:

```text
C1_snapshot_team_state
C8_label_mask for supervision only
graph_resolution = none
relation_handling = none
```

The `snapshot_rgcn` summary records:

```text
C1_snapshot_team_state + C2_snapshot_team_relation
C8_label_mask for supervision only
graph_resolution = 10s_snapshot
relation_handling = relation_channels_preserved
snapshot_edge_types = proximity, combat
```

The `snapshot_rgcn_edge_shuffle` summary records the same C1 + C2 channel
consumption but with:

```text
relation_handling = edge_shuffle_null
edge_control = edge_shuffle
```

The `snapshot_gate_only` summary records the same C1 + C2 channel consumption
but with:

```text
relation_handling = contact_gate_only
edge_control = gate_only
```

The `snapshot_relative_edge` summary records:

```text
C1_snapshot_team_state + C2_snapshot_team_relation + E_s_derived
relation_handling = contact_gated_relative_edge
edge_control = gate_only
edge_features = prox_weight, combat_weight, contact_gate, delta_z,
                delta_alive, delta_health, delta_blue_boundary,
                delta_target_center, delta_rel_zone_x, delta_rel_zone_y
```

## Phase-Cut At-Risk Semantics

Important correction:

```text
For phase-cut prediction after observing phase k, the at-risk condition is
alive_at_phase_end for phase k, not alive_at_phase_start.
```

Reason:

```text
The model observes snapshots through phase k and predicts elimination before
phase k+1. Therefore a valid hazard target exists only for teams that survive
to the phase-k cut.
```

The smoke script reports `phase_start_disagreements` separately. These are not
failures; they indicate cases where `target_defined=True` under the correct
phase-end cut semantics while `alive_at_phase_start=False` in the phase label
table.

## Do Not Do Yet

Do not start the full ST-GNN matrix yet.

Before full experiments:

1. record consumed channel families in every subsequent experiment output;
2. use `scripts/run_snapshot_paper_st_feature_matrix.sh` for paper-facing
   ST feature/significance runs instead of the broader sparse exploratory
   matrix;
3. before any full matrix claim, choose whether the main DSACG story should
   present sparse scaffolding, observed relation channels, or both as
   hypothesis-generating rather than as a semantic graph-topology win.

## Next Pilot Ladder

```text
snapshot_last_only
snapshot_tcn_no_graph
snapshot_rgcn
snapshot_rgcn_edge_shuffle
snapshot_gate_only
snapshot_relative_edge
snapshot_relative_edge_sparse
```

Paper-facing runner:

```text
scripts/run_snapshot_paper_st_feature_matrix.sh
```

Default paper core:

```text
snapshot_last_only
snapshot_tcn_no_graph
snapshot_rgcn
snapshot_rgcn_edge_shuffle
snapshot_gate_only
```

Smoke artifact:

```text
model_stgcn/checkpoints/paper_st_feature_smoke20/paper_st_metrics_seed42.csv
```

The first question is not "how high is the AUC?"

The first question is:

```text
Can relation semantics recover signal over temporal-only and edge-shuffled
controls under a locked feature/channel contract?
```
