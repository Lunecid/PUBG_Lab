# Next Chat Handoff (2026-06-22)

## One-Line State

Current stage:

```text
Pre-experiment design lock + recovered all-match narrow matrix
```

Do not launch the broad full matrix. The feature/channel manifest, graph-null
pilots, sparse relative-edge pilots, seed repeats, and limit-50 scale-up gate
are done. A six-hour, all-match, one-epoch narrow matrix was run and then
recovered from saved sparse-null checkpoints after interruption. The next
practical step is a focused continuation, not broadening the matrix.

## Current Research Position

The project is no longer framed as:

```text
Apply GNN to PUBG.
```

It is framed as:

```text
Formalize Battle Royale as a Dynamic Shrinking-Arena Competitive Graph
(DSACG), then empirically instantiate its PUBG team-level projection for
phase-wise survival hazard and final-placement prediction.
```

Safe claim:

```text
We propose a genre-level graph formalism based on Battle Royale mechanics and
evaluate its observable team-level projection on PUBG telemetry.
```

Do not claim:

```text
The model generalizes to every Battle Royale title.
```

Cross-title validation remains future work. Current validation is within-title:
map, mode, phase band, pressure band, graph null controls.

## Fixed Design Decisions

| Item | Decision |
| --- | --- |
| Formalism | Dynamic Shrinking-Arena Competitive Graph |
| Empirical case | PUBG telemetry |
| Projection | team-level graph projection |
| Node | team |
| Minimum input clock | 10-second snapshot |
| Output clock | phase cut |
| Main target | next-phase elimination hazard |
| Secondary target | final rank / top-k |
| Main warning | full DSACG is heterogeneous; PUBG experiment is only a team-level projection |

## Key Documents

Read these first in the next chat:

| File | Purpose |
| --- | --- |
| `docs/FUTURE_PLAN.md` | living project plan and completion tracker |
| `docs/PRE_EXPERIMENT_DESIGN_LOCK_2026-06-22.md` | current stage, validation results, next pilot ladder |
| `docs/DSACG_METHOD_SKELETON.md` | RQ1-RQ5 and paper method skeleton |
| `docs/feature_engineering/ST_FEATURE_DEFINITION_2026-06-23.md` | canonical definition of ST features for the paper |
| `docs/ST_FEATURE_PRIOR_ALIGNED_DESIGN_2026-06-23.md` | prior-work-aligned ST feature blocks and project novelty |
| `docs/ST_FEATURE_BATTLEGROUND_MAPPING_2026-06-23.md` | table mapping prior-work ST concepts to Battleground/PUBG feature families |
| `docs/PAPER_ST_FEATURE_MATRIX_RUNBOOK_2026-06-23.md` | paper-facing ST feature/significance runner and interpretation plan |
| `docs/PAPER_ST_FEATURE_FAST_MPS_RESULT_2026-06-23.md` | 500-match/2-epoch MPS fast result for the paper ST matrix |
| `docs/PAPER_ST_DOMAIN_CHANNEL_RESULT_2026-06-23.md` | corrected map/mode/user-version channel-sliced MPS results |
| `docs/PAPER_ST_FEATURE_SET_NOTE_2026-06-23.md` | reference-level interpretation of the domain-channel results by feature set |
| `docs/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv` | feature/channel manifest v0 |
| `docs/BATTLE_ROYALE_GRAPH_FORMALISM_2026-06-22.md` | genre-level DSACG formalism |
| `docs/GRAPH_DATA_MODELING_FOUNDATION_2026-06-22.md` | graph object, tensor contracts, ablation logic |
| `docs/RELATED_WORK_FRAME_2026-06-22.md` | canonical related-work structure |
| `docs/PRIOR_WORK_SURVEY_2026-06-22.md` | candidate bibliography and baseline inventory |
| `docs/INPUT_MODEL_CHANNEL_CONTRACT_2026-06-22.md` | channel definitions and model-stream contract |

## New Code / Validation Artifacts

| File | Purpose |
| --- | --- |
| `tools/validate_genre_feature_manifest.py` | validates manifest schema, allowed owners/clocks/streams, leakage rules, and legacy feature coverage |
| `tools/smoke_snapshot_phase_contract.py` | DB smoke for snapshot tensors, phase labels/masks, at-risk consistency, and one-pass model forward |
| `model_stgcn/snapshot_baselines.py` | minimal no-graph baselines: `SnapshotLastOnlyHazard`, `SnapshotTCNNoGraphHazard` |
| `tests/test_snapshot_baselines.py` | synthetic forward and causal-blindness tests for no-graph baselines |
| `scripts/run_snapshot_paper_st_feature_matrix.sh` | paper-facing core ST feature matrix runner |
| `tools/eval_snapshot_phase_checkpoint.py` | recovers missing snapshot summary metrics from saved checkpoints |

## Validation Results Already Run

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

Interpretation:

- No schema/leakage/legacy coverage errors.
- Wildcard rows such as `net_combat_*`, `hab_safe_area_*`, and `created_*`
  have been expanded into physical DB/model feature columns.
- The manifest explicitly separates snapshot `vehicle_ratio` from the phase
  aggregate `vehicle_ratio` column.

Synthetic forward checks:

```text
direct test functions passed
```

Note:

```text
pytest was not installed in the active system Python, so tests were invoked
directly as Python functions.
```

DB smoke:

```text
PG_PASSWORD=<local_password> python3 tools/smoke_snapshot_phase_contract.py --limit 5 --run-forward
```

Result:

```text
requested=5 checked=5 failures=0
X_s=(200,128,13)
A_s=(200,2,128,128)
y_hazard=(7,128)
forward=True
```

Important label correction:

```text
For phase-cut hazard after observing phase k, the at-risk condition is
alive_at_phase_end, not alive_at_phase_start.
```

Reason:

```text
The model observes snapshots through phase k and predicts elimination before
phase k+1. Therefore the team must survive to the phase-k cut.
```

`phase_start_disagreements` are reported separately by the smoke script. They
are not failures under the corrected phase-cut semantics.

Tiny no-graph pilots:

```text
PG_PASSWORD=<local_password> python3 -m model_stgcn.train_snapshot_phase \
  --model-variant snapshot_last_only --limit 20 --epochs 2 --cached ...

PG_PASSWORD=<local_password> python3 -m model_stgcn.train_snapshot_phase \
  --model-variant snapshot_tcn_no_graph --limit 20 --epochs 2 --cached ...
```

Result:

Important correction:

```text
The first pilot summaries without the phasefix suffix are stale for performance
interpretation. They verified code paths, but `phase_of_snap` could select
match-tail snapshots as phase cuts for some held-out matches.

`stdb/snapshot_queries.py` now derives phase_of_snap from
stdb.v_match_phase_bounds. Treat the *_phasefix_limit20 summaries as the
current valid tiny-pilot artifacts.
```

| Variant | Split | Best val loss | Test loss | Test acc | Artifact |
| --- | --- | ---: | ---: | ---: | --- |
| `snapshot_last_only` | 16/2/2 matches | 0.6518 | 0.6452 | 0.6667 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_last_only_phasefix_limit20.summary.json` |
| `snapshot_tcn_no_graph` | 16/2/2 matches | 0.6810 | 0.6774 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_tcn_no_graph_phasefix_limit20.summary.json` |

Both summaries record channel consumption:

```text
C1_snapshot_team_state
C8_label_mask for supervision only
graph_resolution = none
relation_handling = none
```

Tiny graph pilot:

```text
PG_PASSWORD=<local_password> python3 -m model_stgcn.train_snapshot_phase \
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

```text
The C1 + C2 snapshot graph path, edge-shuffle null path, binary contact-gate
path, and derived E_s relative-edge path all train end-to-end under the locked
channel contract. After the phase-cut fix, snapshot_rgcn still does not beat
snapshot_last_only on loss and does not beat snapshot_rgcn_edge_shuffle; it is
also effectively tied with snapshot_gate_only. The corrected
snapshot_relative_edge pilot is best on this tiny split, but this should be
read as evidence for derived relative-edge features, not as evidence that the
current observed proximity/combat topology is useful.

The relative-edge diagnostic recovered target-cut coverage to 0.9579 overall,
but edge density over alive ordered pairs is 1.0 and contact_gate mean is 1.0.
The current gate is therefore an alive-pair complete graph, not a sparse
contact graph. Before scaling, replace it with an explicit sparse/contact or
advantage gate and rerun the same pilot ladder.
```

Sparse relative-edge follow-up now implemented:

```text
snapshot_relative_edge_sparse
relative_edge_gate = knn_combat
relative_edge_k = 5
```

Diagnostic:

```text
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k5_diagnostics.json
```

Result:

| Variant | Split | Gate density | Best val loss | Test loss | Test acc | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `snapshot_relative_edge_sparse` | 16/2/2 matches | 0.1569 | 0.6440 | 0.6144 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k5_phasefix_limit20.summary.json` |

Interpretation:

```text
The sparse gate fixes the complete-graph issue: edge density over alive ordered
pairs drops from 1.0 to 0.1569 while target any-edge coverage remains 0.9579.
On this tiny split, sparse relative-edge stays better than R-GCN, edge-shuffle,
gate-only, and no-graph last-only on test loss. It is close to the dense
relative-edge pilot on validation loss and slightly better on test loss.

This is still not a main-result claim. Treat it as a passed design smoke:
relative-edge semantics can survive a real sparse/contact gate.
```

Sparse relative-edge feature null now implemented:

```text
snapshot_relative_edge_sparse_null
relative_edge_gate = knn_combat
relative_edge_k = 5
relative_edge_null = feature_roll
```

Diagnostic:

```text
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_null_k5_diagnostics.json
```

Result:

| Variant | Split | Gate density | Best val loss | Test loss | Test acc | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `snapshot_relative_edge_sparse_null` | 16/2/2 matches | 0.1569 | 0.6437 | 0.6143 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_null_k5_phasefix_limit20.summary.json` |

Interpretation:

```text
The feature-roll null preserves the same sparse density and target coverage as
the real sparse variant. It is effectively tied with the real sparse model and
is microscopically lower on this tiny split's val/test loss. Therefore the
current tiny result does not support a claim that aligned relative-edge feature
semantics are driving the gain. The safer reading is that sparse kNN/combat
pooling, model capacity/regularization, or neighbor-scaffold effects are doing
most of the work.
```

Sparse topology null now implemented:

```text
snapshot_relative_edge_sparse_topology_null
relative_edge_gate = knn_combat
relative_edge_k = 5
relative_edge_null = topology_roll
```

Diagnostic:

```text
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_topology_null_k5_diagnostics.json
```

Result:

| Variant | Split | Gate density | Best val loss | Test loss | Test acc | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `snapshot_relative_edge_sparse_topology_null` | 16/2/2 matches | 0.1569 | 0.6412 | 0.6155 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_topology_null_k5_phasefix_limit20.summary.json` |

Interpretation:

```text
The topology null preserves the same density as real sparse k=5 and breaks kNN
destination semantics by rolling destinations within each source row's alive
candidate set. It is close to real sparse: better on tiny validation loss,
slightly worse on tiny test loss. This also does not support a semantic
topology claim yet.
```

k-sensitivity now run:

| Variant | k | Gate density | Best val loss | Test loss | Test acc | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `snapshot_relative_edge_sparse` | 3 | 0.0951 | 0.6438 | 0.6144 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k3_phasefix_limit20.summary.json` |
| `snapshot_relative_edge_sparse` | 5 | 0.1569 | 0.6440 | 0.6144 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k5_phasefix_limit20.summary.json` |
| `snapshot_relative_edge_sparse` | 8 | 0.2464 | 0.6430 | 0.6148 | 0.7267 | `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k8_phasefix_limit20.summary.json` |

Interpretation:

```text
k=3/5/8 are nearly tied on the tiny split. The current robust observation is
sparse relation scaffolding, not precise kNN topology or aligned relative-edge
feature semantics.
```

Pilot-scale seed repeats now run:

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

```text
Real sparse and feature-null are indistinguishable over these pilot repeats.
Topology-null is only slightly worse on mean test loss, and the gap is tiny
relative to split/seed variation. This supports sparse-scaffold behavior, not a
semantic edge-feature or kNN-destination claim.
```

Main-comparator pilot repeats now run:

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

```text
Sparse-family means are slightly lower than R-GCN/gate-only, but the direction
is split-sensitive: R-GCN/gate-only beat sparse on seeds 41 and 43, while sparse
wins strongly on seed 42. R-GCN, edge-shuffle, and gate-only remain almost
identical. This still does not justify a semantic graph win or a stable sparse
win; it identifies the candidate family for larger repeated pilots.
```

Limit-50 scale-up gate pilot now run:

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

```text
At limit 50, sparse-family mean loss is modestly lower than R-GCN/gate-only and
clearly lower than last-only. However, feature-null is tied with real sparse and
topology-null is slightly lower than real sparse, so the signal is still sparse
scaffold/regularization, not semantic edge topology. R-GCN, edge-shuffle, and
gate-only remain nearly identical.
```

## Architecture Direction

Canonical target architecture:

```text
10-second dynamic team graph
  -> snapshot graph/no-graph encoder
  -> causal temporal encoder
  -> last snapshot at phase cut
  -> team hazard / rank / top-k heads
```

Do not describe the legacy V3/MR result as a fully snapshot-resolution survival
model. It encoded 10-second snapshots and then mean-pooled them to phase before
phase-resolution heads.

## Edge Interpretation

Old interaction topology:

```text
proximity + combat + co-pressure
```

Observed issue:

```text
observed full graph AUC ~= 0.845
edge-shuffle AUC ~= 0.847
```

Interpretation:

```text
The old topology behaved more like smoothing than true observed-topology signal.
```

Promising direction:

```text
contact-gated directed relative advantage edge
```

Preliminary relative-edge result:

```text
real advantage edges - edge-shuffled null = +0.0033 AUC
node-absolute control still positive     = +0.0023 AUC
```

Keep this as preliminary / hypothesis-generating until rebased through the new
locked pipeline.

## Six-Hour All-Match Partial Run

Run envelope:

```text
SNAP_OUT_DIR=model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022
SNAP_EPOCHS=1 SNAP_SEED=42 SNAP_BATCH=2 SNAP_HIDDEN=16 SNAP_MID=32
SNAP_NUM_BLOCKS=1 SNAP_EDGE_HIDDEN=8 SNAP_WORKERS=2 SNAP_EVAL_WORKERS=1
SNAP_PREFETCH_FACTOR=2 SNAP_DEVICE=cpu SNAP_FORCE=1
```

Scope:

```text
all eligible matches = 21,824
split = 17,460 train / 2,182 val / 2,182 test
timebox = 6 hours
```

Artifacts:

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

```text
This is not full training. It is one seed, one epoch, and timeboxed.
```

On the completed all-match variants, `snapshot_rgcn` is the current leader:
test loss 0.5235 versus 0.5337 for last-only, 0.5291 for edge-shuffle, 0.5283
for gate-only, 0.5378 for sparse real, 0.5397 for sparse feature-null, and
0.5411 for sparse topology-null. That is tentative evidence that
relation-channel consumption is doing something beyond no-graph and simple
graph null controls under this run envelope.

Do not overclaim. Sparse real is slightly better than its sparse null controls,
but the whole sparse family underperforms R-GCN, edge-shuffle, gate-only, and
last-only here. There are still no seed or epoch repeats. The safe next claim
is: all-match one-epoch evidence now favors R-GCN among completed variants,
but semantic graph benefit remains open.

## Next Work Order

Do this next:

1. Do not launch the broad full matrix.
2. Treat the 6-hour matrix as recovered/completed for the one-epoch seed-42
   narrow scope.
3. Use `scripts/run_snapshot_paper_st_feature_matrix.sh` for paper-facing
   ST feature runs. It separates the core paper matrix from sparse exploratory
   variants and includes `snapshot_tcn_no_graph`.
4. If using another sleep-window run, prefer a focused continuation:
   `snapshot_rgcn`, `snapshot_rgcn_edge_shuffle`, `snapshot_gate_only`, and
   `snapshot_last_only`, plus `snapshot_tcn_no_graph`, for 2-3 epochs or seed
   repeats on the same split.
5. Keep sparse relative-edge in the matrix, but do not treat it as the default
   winner after the all-match 1-epoch result.
6. Frame the current evidence as hypothesis-generating: observed R-GCN relation
   channels have the best all-match 1-epoch metrics, but semantic graph benefit
   still requires repeats and longer training.
7. Record feature/channel families consumed by every subsequent experiment
   result.

## Immediate Questions For Next Chat

Answer these before full training:

| Question | Required answer |
| --- | --- |
| Are all physical features mapped to genre primitives? | yes for current DB/model manifest columns; validator reports 229 rows, 0 warnings, 0 errors |
| Are labels/masks forbidden as model input? | validator says yes |
| Does snapshot tensor contract work on DB? | yes, 5-match smoke passed |
| Is no-graph temporal baseline implemented? | yes; tiny `snapshot_last_only` and `snapshot_tcn_no_graph` pilots ran on a 20-match split |
| Is snapshot R-GCN pilot implemented/trained? | yes; tiny 20-match pilot ran and records C1 + C2 consumption |
| Is edge-shuffle graph pilot implemented in new locked path? | yes; tiny 20-match pilot ran and records `edge_control=edge_shuffle` |
| Is gate-only graph pilot implemented in new locked path? | yes; tiny 20-match pilot ran and records `edge_control=gate_only` |
| Is relative-edge pilot implemented in new locked path? | yes; corrected tiny 20-match phasefix pilot ran and records `E_s_derived`; it is best on tiny val/test loss but still uses a dense alive-pair gate |
| Is sparse relative-edge implemented? | yes; `snapshot_relative_edge_sparse` with `relative_edge_gate=knn_combat`, `relative_edge_k=5`, density 0.1569, test loss 0.6144 |
| Is sparse relative-edge feature null implemented? | yes; `snapshot_relative_edge_sparse_null` with `relative_edge_null=feature_roll`, same density 0.1569, test loss 0.6143 |
| Is sparse topology null implemented? | yes; `snapshot_relative_edge_sparse_topology_null` with `relative_edge_null=topology_roll`, same density 0.1569, test loss 0.6155 |
| Is k-sensitivity checked? | yes for k=3/5/8; all are nearly tied on tiny test loss |
| Are sparse scaffold seed repeats done? | yes for seeds 41/42/43; real sparse and nulls are effectively tied |
| Are main comparator seed repeats done? | yes for seeds 41/42/43; sparse family is slightly lower on mean loss but split-sensitive |
| Is limit-50 scale-up gate done? | yes; sparse family is modestly better than R-GCN/gate-only on mean loss, but nulls are tied or better |
| Is the narrow full-matrix runner ready? | yes; `scripts/run_snapshot_narrow_full_matrix.sh` covers last-only, R-GCN, edge-shuffle, gate-only, sparse real, sparse feature-null, and sparse topology-null |
| Did the narrow runner smoke pass? | yes; after the loader update, `SNAP_LIMIT=20`, `SNAP_EPOCHS=1`, seed 42 completed all 7 variants and wrote `model_stgcn/checkpoints/snapshot_narrow_full_smoke20_v2/matrix_metrics_seed42.csv` |
| Is the paper-facing ST feature runner ready? | yes; `scripts/run_snapshot_paper_st_feature_matrix.sh` covers last-only, TCN no-graph, R-GCN, edge-shuffle, and gate-only, with optional sparse variants |
| Did the paper ST runner smoke pass? | yes; `PAPER_ST_LIMIT=20`, `PAPER_ST_EPOCHS=1`, seed 42 completed all 5 core variants and wrote `model_stgcn/checkpoints/paper_st_feature_smoke20/paper_st_metrics_seed42.csv` |
| Did the fast MPS paper ST matrix run? | yes; `PAPER_ST_LIMIT=500`, `PAPER_ST_EPOCHS=2`, `PAPER_ST_BATCH=8`, `PAPER_ST_DEVICE=mps`, seed 42 completed all 5 core variants and wrote `model_stgcn/checkpoints/paper_st_feature_mps_fast500_e2_20260623/paper_st_metrics_seed42.csv` |
| Is the actual narrow full matrix complete? | yes for the recovered all-match seed-42, 1-epoch narrow scope; all 7 summaries exist in `snapshot_narrow_full_sleep6h_20260623_0022` |
| Which all-match 1-epoch variant leads so far? | `snapshot_rgcn`: test loss 0.5235, better than last-only 0.5337, edge-shuffle 0.5291, gate-only 0.5283, sparse real 0.5378, sparse feature-null 0.5397, and sparse topology-null 0.5411 |
| Does that prove graph semantics? | no; it is one seed/one epoch, but it is the first all-match signal favoring observed R-GCN relation channels |
| Which fast MPS paper variant leads? | `snapshot_tcn_no_graph`: test loss 0.5746, better than last-only 0.6246, R-GCN 0.5823, edge-shuffle 0.5829, and gate-only 0.5819 |
| What does the fast MPS paper run prove? | it supports temporal ST node dynamics on the 500-match split; it does not support observed graph topology because graph variants are close to controls and behind TCN |
| Did the corrected domain/channel matrix run? | yes; primary TPP squad 5-map x 11 variants and FPP squad 4-map x 11 variants completed on MPS |
| Which channel wins after map/user-version slicing? | `tcn_viability_resource` wins in all 9 tested map/mode cohorts; mean test loss 0.5942 vs trajectory 0.6178 and graph variants about 0.620 |
| Does no-graph training still fetch graph tensors? | no; no-graph snapshot training now constructs datasets with `include_edges=False`, so `A_s` is omitted from those batches |
| Are non-phasefix pilot metrics still valid? | no; treat them as code-path smoke only because `phase_of_snap` was corrected afterward |

## Files Are Not Committed

At the end of this handoff, the new docs/code are still untracked in git. Do
not assume they were committed.

Notable new or updated files include:

```text
docs/handoffs/NEXT_CHAT_HANDOFF_2026-06-22.md
docs/PRE_EXPERIMENT_DESIGN_LOCK_2026-06-22.md
docs/DSACG_METHOD_SKELETON.md
docs/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv
docs/FUTURE_PLAN.md
docs/feature_engineering/ST_FEATURE_DEFINITION_2026-06-23.md
docs/ST_FEATURE_PRIOR_ALIGNED_DESIGN_2026-06-23.md
docs/ST_FEATURE_BATTLEGROUND_MAPPING_2026-06-23.md
docs/PAPER_ST_FEATURE_MATRIX_RUNBOOK_2026-06-23.md
docs/PAPER_ST_FEATURE_FAST_MPS_RESULT_2026-06-23.md
docs/PAPER_ST_DOMAIN_CHANNEL_RESULT_2026-06-23.md
docs/PAPER_ST_FEATURE_SET_NOTE_2026-06-23.md
tools/validate_genre_feature_manifest.py
tools/smoke_snapshot_phase_contract.py
tools/eval_snapshot_phase_checkpoint.py
model_stgcn/snapshot_baselines.py
model_stgcn/snapshot_edge_controls.py
model_stgcn/snapshot_relative_edge.py
model_stgcn/train_snapshot_phase.py
scripts/run_snapshot_narrow_full_matrix.sh
scripts/run_snapshot_paper_st_feature_matrix.sh
model_stgcn/checkpoints/paper_st_feature_smoke20/paper_st_metrics_seed42.csv
model_stgcn/checkpoints/paper_st_feature_smoke20/paper_st_config_seed42.json
model_stgcn/checkpoints/snapshot_narrow_full_smoke20_v2/matrix_metrics_seed42.csv
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/matrix_metrics_seed42_recovered.csv
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/matrix_metrics_seed42_partial.csv
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/matrix_status_seed42_recovered.json
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/matrix_status_seed42_partial.json
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/run.log
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/snapshot_last_only_seed42.summary.json
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/snapshot_rgcn_seed42.summary.json
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/snapshot_rgcn_edge_shuffle_seed42.summary.json
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/snapshot_gate_only_seed42.summary.json
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/snapshot_relative_edge_sparse_seed42.summary.json
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/snapshot_relative_edge_sparse_null_seed42.summary.json
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/snapshot_relative_edge_sparse_topology_null_seed42.summary.json
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k5_phasefix_limit20.summary.json
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k5_diagnostics.json
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k3_phasefix_limit20.summary.json
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k3_diagnostics.json
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k8_phasefix_limit20.summary.json
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k8_diagnostics.json
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_null_k5_phasefix_limit20.summary.json
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_null_k5_diagnostics.json
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_topology_null_k5_phasefix_limit20.summary.json
model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_topology_null_k5_diagnostics.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed41_sparse_k5.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed41_sparse_feature_null_k5.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed41_sparse_topology_null_k5.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed42_sparse_k5.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed42_sparse_feature_null_k5.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed42_sparse_topology_null_k5.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed43_sparse_k5.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed43_sparse_feature_null_k5.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed43_sparse_topology_null_k5.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed41_last_only.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed41_tcn_no_graph.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed41_rgcn.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed41_rgcn_edge_shuffle.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed41_gate_only.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed42_last_only.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed42_tcn_no_graph.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed42_rgcn.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed42_rgcn_edge_shuffle.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed42_gate_only.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed43_last_only.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed43_tcn_no_graph.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed43_rgcn.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed43_rgcn_edge_shuffle.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats/seed43_gate_only.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats_limit50/seed41_sparse_k5.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats_limit50/seed42_sparse_k5.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats_limit50/seed43_sparse_k5.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats_limit50/seed41_rgcn.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats_limit50/seed42_rgcn.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats_limit50/seed43_rgcn.summary.json
model_stgcn/checkpoints/snapshot_pilot_repeats_limit50/sparse_k5_diagnostics.json
tools/diagnose_snapshot_relative_edge.py
tests/test_snapshot_baselines.py
tests/test_snapshot_edge_controls.py
tests/test_snapshot_relative_edge.py
```

## Next Chat Opening Prompt

Use this to continue:

```text
Read docs/handoffs/NEXT_CHAT_HANDOFF_2026-06-22.md and continue from the pre-experiment
design lock stage. Do not run full training yet. The manifest physical-column
expansion, no-graph pilots, tiny snapshot_rgcn pilot, and tiny
snapshot_rgcn_edge_shuffle null pilot, and tiny snapshot_gate_only pilot are
done. Phase-cut alignment was fixed in stdb/snapshot_queries.py, so use the
*_phasefix_limit20 summaries as the current pilot metrics. The corrected
snapshot_relative_edge pilot is best on tiny val/test loss, but diagnostics
show the original contact gate is an alive-pair complete graph. The sparse
construction is now implemented as snapshot_relative_edge_sparse with
relative_edge_gate=knn_combat and relative_edge_k=5. It reduces alive-pair edge
density to 0.1569 and gets test_loss=0.6144 on the same tiny split.
Feature-roll sparse null is also implemented as
snapshot_relative_edge_sparse_null. It preserves density 0.1569 and gets
test_loss=0.6143, effectively tied with real sparse. Topology null and
k-sensitivity are now done. Topology null preserves density 0.1569 and gets
test_loss=0.6155. k=3/5/8 real sparse are nearly tied on test loss. Sparse
scaffold seed repeats are also done for seeds 41/42/43: real sparse and nulls
are effectively tied. Main comparator seed repeats are also done for seeds
41/42/43. Sparse-family mean loss is slightly lower, but the direction is
split-sensitive and not a stable graph win. Limit-50 scale-up gate is now done:
sparse-family mean loss is modestly lower than R-GCN/gate-only and clearly
lower than last-only, but feature/topology nulls remain tied or slightly better
than real sparse. The narrow full-matrix runner now exists at
scripts/run_snapshot_narrow_full_matrix.sh, and the limit-20/1-epoch smoke
completed all 7 variants with metrics in
model_stgcn/checkpoints/snapshot_narrow_full_smoke20_v2/matrix_metrics_seed42.csv.
A six-hour all-match 1-epoch sleep run was then launched on 21,824 eligible
matches with split 17,460/2,182/2,182, then recovered after interruption by
evaluating saved sparse-null checkpoints. Recovered metrics are in
model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/matrix_metrics_seed42_recovered.csv.
Completed test losses are: snapshot_last_only 0.5337, snapshot_rgcn 0.5235,
snapshot_rgcn_edge_shuffle 0.5291, snapshot_gate_only 0.5283,
snapshot_relative_edge_sparse 0.5378, snapshot_relative_edge_sparse_null
0.5397, and snapshot_relative_edge_sparse_topology_null 0.5411. This is not
full training, but it is the first all-match signal favoring observed R-GCN
relation channels over no-graph, edge-shuffle, gate-only, and sparse
relative-edge variants. Do not claim semantic graph benefit yet. No-graph
snapshot training now uses include_edges=False so A_s is not fetched for
last-only/TCN baselines. A paper-facing ST feature runner is now separated at
scripts/run_snapshot_paper_st_feature_matrix.sh, with runbook
docs/PAPER_ST_FEATURE_MATRIX_RUNBOOK_2026-06-23.md. Its limit-20/1-epoch smoke
completed the five core variants: last-only, TCN no-graph, R-GCN, edge-shuffle,
and gate-only. A fast MPS paper matrix then completed on 500 matches, 2 epochs,
batch 8, seed 42, with metrics in
model_stgcn/checkpoints/paper_st_feature_mps_fast500_e2_20260623/paper_st_metrics_seed42.csv.
Test losses: last-only 0.6246, TCN no-graph 0.5746, R-GCN 0.5823,
edge-shuffle 0.5829, gate-only 0.5819. This fast result strongly supports
temporal ST node dynamics, but not observed graph topology. Next step: either
repeat the 500-match fast matrix across seeds or run a focused all-match
continuation for TCN vs R-GCN controls.
```
