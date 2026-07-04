# Future Plan

Living plan for the Battle Royale / PUBG survival-prediction project.

Created: 2026-06-22  
Status convention:

- `[ ]` Planned / not started
- `[~]` In progress
- `[x]` Completed
- `-- Result:` concrete output exists. Use this only when an artifact, metric, figure, or document has actually been produced.

Update rule:

> A task becomes `[x]` only when it has a concrete `-- Result:` line or an explicitly reviewed deliverable path. If a result is not available yet, keep it `[ ]` or `[~]`.

---

## 1. Current Understanding From Discussion

### Problem Definition

We must make the problem definition explicit before presenting the model.

Core problem:

> Formalize a Battle Royale match as a Dynamic Shrinking-Arena Competitive Graph,
> then instantiate it with PUBG telemetry as a team-level multi-relational graph
> to predict phase-wise survival hazard and final-placement outcome.

Clarifications:

- Genre-level object: Battle Royale dynamic shrinking-arena competitive graph.
- Empirical case study: PUBG team-level graph projection.
- Prediction unit: at-risk `team x phase` cell.
- Node: team, not player.
- Target: team next-phase elimination, final rank, top-5/top-10.
- Output time: phase cut points, not every 10-second tick.
- Team size differences are encoded as node features such as `alive_n`, `team_size`, health statistics, and team spread.

### Spatial Definition

The "spatial" component must be explained more clearly.

Spatial means:

1. map position: `x`, `y`, `z`;
2. zone-relative state: distance to blue-zone boundary, inside/outside blue, target safe-zone center, safe-zone geometry;
3. team-to-team spatial/strategic relations: proximity, combat, co-pressure, and future relative-advantage edges.

Short wording:

> Spatial = where a team is, where it is relative to the zone, and which other teams it is spatially or strategically connected to.

### Legacy Model Interpretation

The legacy V3/MR model is not a fully snapshot-resolution survival model.

Correct interpretation:

> 10-second snapshots are encoded first, then mean-pooled into phase-level embeddings. Main graph reasoning and final prediction heads run at phase resolution.

Legacy model contract:

```text
10-second snapshot input
  -> snapshot ST-GNN
  -> mean-pool to phase
  -> fuse with phase features and global context
  -> phase ST-Encoder
  -> phase-cut hazard/rank/top-k heads
```

### Graph Null-Control Interpretation

Existing phase graph relation result:

```text
observed full graph AUC ~= 0.845
edge_shuffle AUC ~= 0.847
```

Interpretation:

> The original proximity/combat/co-pressure topology mainly gives smoothing, not reliable observed-topology signal.

Relative advantage edge follow-up:

```text
real advantage edges - edge-shuffled null = +0.0033 AUC
node-absolute control still positive     = +0.0023 AUC
```

Interpretation:

> The negative topology result motivates changing the relation type. Contact-gated relative advantage gives preliminary evidence that relation type, not only GNN architecture, determines whether topology is informative.

Important wording change:

- Prefer `real advantage edges` or `unshuffled advantage graph`.
- Avoid ambiguous slide wording such as `observed - node-absolute control`.

### Sagong / Prior Work Data

Sagong et al. use phase-specific PUBG telemetry/log data, not the public Kaggle post-game aggregate dataset.

Important distinction:

- Sagong exact data: confidential, player-level solo logs, Jan 1-7 2021, 0.5 phase intervals.
- Public Kaggle prior benchmark: `PUBG Finish Placement Prediction (Kernels Only)`, post-game aggregate placement prediction.
- Our data: telemetry-derived team-level dynamic graphs, mainly squad, 10-second snapshots plus phase cuts.

Correct framing:

> We cannot exactly reproduce Sagong et al.'s confidential dataset, but we can implement a Sagong-style phase strategy baseline on our telemetry-derived data and optionally include the Kaggle PUBG Finish Placement dataset as an external public post-game benchmark.

---

## 2. Completed / Frozen Items

- [x] Create canonical input/model channel contract.
  -- Result: `docs/INPUT_MODEL_CHANNEL_CONTRACT_2026-06-22.md`

- [x] Create Battle Royale genre-level graph formalism.
  -- Result: `docs/BATTLE_ROYALE_GRAPH_FORMALISM_2026-06-22.md`

- [x] Create PUBG graph-modeling research frame.
  -- Result: `docs/PUBG_GRAPH_MODELING_RESEARCH_FRAME_2026-06-22.md`

- [x] Create graph data modeling foundation.
  -- Result: `docs/GRAPH_DATA_MODELING_FOUNDATION_2026-06-22.md`

- [x] Add PUBG domain-aware design layer.
  -- Result: `docs/PUBG_DOMAIN_DESIGN_LAYER_2026-06-22.md`

- [x] Create prioritized prior-work survey.
  -- Result: `docs/PRIOR_WORK_SURVEY_2026-06-22.md`

- [x] Create expanded related-work frame around Battle Royale formalism.
  -- Result: `docs/RELATED_WORK_FRAME_2026-06-22.md`

- [x] Create DSACG method skeleton.
  -- Result: `docs/DSACG_METHOD_SKELETON.md`

- [x] Define paper-facing ST feature contract.
  -- Result: `docs/feature_engineering/ST_FEATURE_DEFINITION_2026-06-23.md`

- [x] Redesign ST feature framing against prior work and novelty.
  -- Result: `docs/ST_FEATURE_PRIOR_ALIGNED_DESIGN_2026-06-23.md`

- [x] Map prior-work ST concepts to Battleground/PUBG feature families.
  -- Result: `docs/ST_FEATURE_BATTLEGROUND_MAPPING_2026-06-23.md`

- [x] Run fast MPS paper ST feature matrix.
  -- Result: `docs/PAPER_ST_FEATURE_FAST_MPS_RESULT_2026-06-23.md`,
  `model_stgcn/checkpoints/paper_st_feature_mps_fast500_e2_20260623/paper_st_metrics_seed42.csv`

- [x] Run corrected map/user-version/channel-sliced MPS matrix.
  -- Result: `docs/PAPER_ST_DOMAIN_CHANNEL_RESULT_2026-06-23.md`,
  `model_stgcn/checkpoints/snapshot_domain_channel_primary_tpp_mps300_e1_20260623/domain_channel_metrics_seed42.csv`,
  `model_stgcn/checkpoints/snapshot_domain_channel_fpp_aux_mps250_e1_20260623/domain_channel_metrics_seed42.csv`

- [x] Write reference-level feature-set interpretation for the domain-channel result.
  -- Result: `docs/PAPER_ST_FEATURE_SET_NOTE_2026-06-23.md`

- [x] Create genre primitive feature manifest v0.
  -- Result: `docs/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv`

- [x] Add pre-experiment design lock validation artifacts.
  -- Result: `docs/PRE_EXPERIMENT_DESIGN_LOCK_2026-06-22.md`, `tools/validate_genre_feature_manifest.py`, `tools/smoke_snapshot_phase_contract.py`

- [x] Add minimal no-graph snapshot baseline modules.
  -- Result: `model_stgcn/snapshot_baselines.py`, `tests/test_snapshot_baselines.py`

- [x] Create next-chat handoff document.
  -- Result: `docs/handoffs/NEXT_CHAT_HANDOFF_2026-06-22.md`

- [x] Freeze legacy result interpretation.
  -- Result: `docs/LEGACY_CODE_RESULT_MAP_2026-06-22.md`

- [x] Map legacy result artifacts to code paths and checkpoints.
  -- Result: `docs/LEGACY_RESULT_CODE_REGISTRY_2026-06-22.md`, `docs/legacy_result_code_registry_2026-06-22.json`

- [x] Create snapshot-first redesign audit.
  -- Result: `docs/SNAPSHOT_FIRST_REDESIGN_AUDIT_2026-06-22.md`

- [x] Create file-level code inventory for redesign migration.
  -- Result: `docs/SNAPSHOT_FIRST_CODE_INVENTORY_2026-06-22.md`

- [x] Add corrected snapshot-first model seed code.
  -- Result: `model_stgcn/snapshot_phase_survival.py`, `model_stgcn/snapshot_phase_dataset.py`, `model_stgcn/train_snapshot_phase.py`

- [x] Add unit/smoke tests for snapshot-first model seed.
  -- Result: `tests/test_snapshot_phase_survival.py`; direct function smoke and `py_compile` passed earlier.

- [x] Rewrite architecture design spec for legacy V3/MR drawing.
  -- Result: `docs/archive/generated_presentation_2026-06/deliverables/MODEL_ARCHITECTURE_DESIGN_SPEC_2026-06-22.md`

---

## 3. Immediate Priority: Channel Contract Implementation

- [~] Stabilize the project around explicit input/model channels.
  -- Result: `docs/INPUT_MODEL_CHANNEL_CONTRACT_2026-06-22.md`, `docs/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv`

  Remaining work:

  - validate that every current feature belongs to exactly one semantic channel and at least one PUBG domain mechanism;
  - record consumed channels in every subsequent experiment result;
  - record consumed domain mechanisms in every experiment result;

- [x] Add feature/channel manifest validation tooling.
  -- Result: `tools/validate_genre_feature_manifest.py`

  Latest smoke result:

  ```text
  rows=229
  errors=0
  warnings=0
  ```

  Interpretation:

  - no schema/leakage/coverage errors against the legacy feature-family manifest;
  - wildcard family rows have been expanded to physical DB/model feature columns.

- [x] Add tensor shape + label/mask smoke validation.
  -- Result: `tools/smoke_snapshot_phase_contract.py`

  Latest DB smoke:

  ```text
  requested=5 checked=5 failures=0
  X_s=(200,128,13)
  A_s=(200,2,128,128)
  y_hazard=(7,128)
  forward=True
  ```

  At-risk semantics:

  - phase-cut hazard target uses `alive_at_phase_end`, not `alive_at_phase_start`;
  - `phase_start_disagreements` are reported separately and are not failures.
  - align presentation diagrams and ablation tables with the same channel taxonomy.

- [ ] Create machine-readable feature channel manifest.
  -- Result:

  Required channel families:

  - domain/cohort;
  - snapshot team-state;
  - snapshot team-relation;
  - phase team-state;
  - phase global context;
  - phase team-relation;
  - individual-derived team aggregate;
  - prior-work strategy;
  - labels/masks.

  Required feature-clock fields:

  - `clock`: match_static, snapshot_10s, phase_cut, phase_window, label;
  - `entity`: team, pair, global, label;
  - `semantic_domain`: roster, movement, zone, combat, resource, relation,
    strategy, global;
  - `causal_window`: instant, trailing_10s, trailing_30s, phase_to_date,
    phase_window, match_static;
  - `model_stream`: snapshot_node, snapshot_edge, phase_node, phase_edge,
    global_context, label.

  Required domain mechanisms:

  - population rule-set;
  - roster survival capital;
  - rotation tempo;
  - zone/habitat pressure;
  - local opponent field;
  - combat exchange;
  - resource readiness;
  - strategic behavior;
  - relation topology;
  - phase horizon;
  - skill/lobby strength.

- [ ] Add channel validation tooling.
  -- Result:

  Requirements:

  - every input feature has one feature clock;
  - every input feature has one semantic channel;
  - label/mask fields cannot enter model input;
  - global features are not silently treated as team node features;
  - snapshot-first models cannot accidentally consume phase aggregate features unless explicitly configured.
  - every input feature has at least one declared PUBG domain mechanism.

- [ ] Add model-channel declaration to experiment outputs.
  -- Result: `model_stgcn/checkpoints/snapshot_pilots/snapshot_last_only_phasefix_limit20.summary.json`, `model_stgcn/checkpoints/snapshot_pilots/snapshot_tcn_no_graph_phasefix_limit20.summary.json` record the no-graph pilot channel contracts. `model_stgcn/checkpoints/snapshot_pilots/snapshot_rgcn_phasefix_limit20.summary.json` records C1 + C2 graph-channel consumption for the tiny R-GCN pilot. `model_stgcn/checkpoints/snapshot_pilots/snapshot_rgcn_edge_shuffle_phasefix_limit20.summary.json` records the same C1 + C2 consumption with `edge_control=edge_shuffle`. `model_stgcn/checkpoints/snapshot_pilots/snapshot_gate_only_phasefix_limit20.summary.json` records the same C1 + C2 consumption with `edge_control=gate_only`. `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_phasefix_limit20.summary.json` records C1 + C2 + derived E_s relative-edge consumption.

  Note: non-phasefix pilot summaries are stale for performance interpretation
  because `phase_of_snap` was corrected afterward. Current diagnostics show
  target-cut coverage recovered, but the contact gate remains dense
  (`edge_density_alive_pairs=1.0`, `contact_gate` mean 1.0), so full training
  should wait for a sparse/contact or advantage-gated relative-edge rerun.

  Sparse follow-up result:
  `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k5_phasefix_limit20.summary.json`
  records C1 + C2 + `E_s_derived` with `relative_edge_gate=knn_combat` and
  `relative_edge_k=5`. Diagnostic artifact
  `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_k5_diagnostics.json`
  reports `edge_density_alive_pairs=0.1569` with target any-edge coverage
  0.9579. Tiny pilot result: best val loss 0.6440, test loss 0.6144, test acc
  0.7267.

  Sparse feature-null result:
  `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_null_k5_phasefix_limit20.summary.json`
  records C1 + C2 + `E_s_derived_null` with
  `relative_edge_null=feature_roll`. Diagnostic artifact
  `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_null_k5_diagnostics.json`
  preserves the same density 0.1569 and target any-edge coverage 0.9579.
  Tiny pilot result: best val loss 0.6437, test loss 0.6143, test acc 0.7267.
  This is effectively tied with real sparse and does not support a feature
  semantics claim yet.

  Sparse topology-null result:
  `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_topology_null_k5_phasefix_limit20.summary.json`
  records C1 + C2 + `E_s_derived_topology_null` with
  `relative_edge_null=topology_roll`. Diagnostic artifact
  `model_stgcn/checkpoints/snapshot_pilots/snapshot_relative_edge_sparse_topology_null_k5_diagnostics.json`
  preserves density 0.1569. Tiny pilot result: best val loss 0.6412, test loss
  0.6155, test acc 0.7267. This is close to real sparse and does not support a
  topology-semantics claim yet.

  k-sensitivity result:
  k=3 density 0.0951, test loss 0.6144; k=5 density 0.1569, test loss 0.6144;
  k=8 density 0.2464, test loss 0.6148. The tiny pilot is nearly flat across
  this k range.

  Sparse scaffold repeat result:
  seeds 41, 42, and 43 were run with limit 20, epochs 2, and k=5 for real
  sparse, feature-roll null, and topology-roll null. Mean test losses were
  0.6440, 0.6440, and 0.6450 respectively, with standard deviations around
  0.021. Real sparse and nulls are effectively tied at this pilot scale.

  Main comparator repeat result:
  the same seed-41/42/43 protocol was run for last-only, TCN no-graph, R-GCN,
  edge-shuffle, and gate-only. Mean test losses were 0.6639, 0.6508, 0.6476,
  0.6478, and 0.6476 respectively. Sparse-family mean loss is slightly lower,
  but seed-level direction is not stable; R-GCN/gate-only beat sparse on seeds
  41 and 43, while sparse wins strongly on seed 42.

  Limit-50 scale-up gate:
  seeds 41, 42, and 43 were rerun with limit 50 and epochs 2 for last-only,
  R-GCN, edge-shuffle, gate-only, sparse real, sparse feature-null, and sparse
  topology-null. Mean test losses were 0.6476, 0.6289, 0.6299, 0.6299,
  0.6255, 0.6255, and 0.6248 respectively. Sparse-family mean loss is
  modestly lower than R-GCN/gate-only, but nulls remain tied or slightly better
  than real sparse.

  Narrow full-matrix readiness:
  `scripts/run_snapshot_narrow_full_matrix.sh` now runs the approved narrow
  matrix: last-only, R-GCN, edge-shuffle, gate-only, sparse real, sparse
  feature-null, and sparse topology-null. The smoke command
  `SNAP_OUT_DIR=model_stgcn/checkpoints/snapshot_narrow_full_smoke20_v2
  SNAP_LIMIT=20 SNAP_EPOCHS=1 SNAP_FORCE=1 ./scripts/run_snapshot_narrow_full_matrix.sh`
  completed all seven variants and wrote
  `model_stgcn/checkpoints/snapshot_narrow_full_smoke20_v2/matrix_metrics_seed42.csv`.
  A six-hour all-match one-epoch sleep run then wrote partial metrics, and the
  interrupted sparse-null summaries were recovered from checkpoints. The
  current recovered CSV is
  `model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/matrix_metrics_seed42_recovered.csv`
  on all 21,824 eligible matches with split 17,460/2,182/2,182. Completed test
  losses were: last-only 0.5337, R-GCN 0.5235, edge-shuffle 0.5291, gate-only
  0.5283, sparse relative-edge 0.5378, sparse feature-null 0.5397, and sparse
  topology-null 0.5411.
  Interpretation: all-match 1-epoch evidence currently favors observed R-GCN
  relation channels among completed variants, while semantic graph benefit
  remains unclaimed until repeats/longer training finish.

  Paper ST feature runner:
  `scripts/run_snapshot_paper_st_feature_matrix.sh` now separates the
  paper-facing core matrix from sparse exploratory variants. Default core
  variants are last-only, TCN no-graph, R-GCN, edge-shuffle, and gate-only.
  Runbook:
  `docs/PAPER_ST_FEATURE_MATRIX_RUNBOOK_2026-06-23.md`. Smoke command
  `PAPER_ST_OUT_DIR=model_stgcn/checkpoints/paper_st_feature_smoke20
  PAPER_ST_LIMIT=20 PAPER_ST_EPOCHS=1 PAPER_ST_FORCE=1
  ./scripts/run_snapshot_paper_st_feature_matrix.sh` completed all five core
  variants and wrote
  `model_stgcn/checkpoints/paper_st_feature_smoke20/paper_st_metrics_seed42.csv`.
  The next paper-facing run should be a focused 2-3 epoch/seed continuation on
  the recovered all-match split, centered on ST representation value and R-GCN
  vs controls.

  Channel-contract implementation note:
  no-graph snapshot training now constructs datasets with `include_edges=False`,
  so `A_s` is omitted from last-only/TCN batches instead of being fetched and
  ignored. Graph and relative-edge variants still request `A_s`.

  Every result should declare:

  - feature clocks consumed;
  - input channels consumed;
  - domain mechanisms consumed;
  - temporal resolution;
  - graph resolution;
  - relation handling;
  - heads enabled;
  - metrics emitted.

- [ ] Define model families in experiment matrix.
  -- Result:

  Required families:

  - tabular phase baseline;
  - temporal tabular model;
  - snapshot temporal model;
  - phase graph model;
  - legacy pooled-MR ST-GNN;
  - snapshot-first ST survival model;
  - edge-conditioned relative-advantage model.

---

## 4. Domain-Aware Design Plan

- [ ] Add domain mechanism labels to the feature manifest.
  -- Result:

  Required labels:

  - D0 population rule-set;
  - D1 roster survival capital;
  - D2 rotation tempo;
  - D3 zone/habitat pressure;
  - D4 local opponent field;
  - D5 combat exchange;
  - D6 resource readiness;
  - D7 strategic behavior;
  - D8 relation topology;
  - D9 phase horizon.

- [ ] Redesign feature groups around PUBG survival mechanisms.
  -- Result:

  Required feature redesign candidates:

  - map-specific coordinate scaling;
  - roster-normalized damage/resource features;
  - snapshot-level rotation urgency;
  - contact-gated relative edge features;
  - third-party/crossfire exposure;
  - phase-band labels for early/mid/late reporting;
  - resource-readiness ratios or inventory-quality proxies;
  - causal temporal deltas over 10/30/60 seconds;
  - skill/lobby tier source and leakage-safe tier buckets.

- [ ] Redesign feature groups around feature clocks.
  -- Result: `docs/FEATURE_CLOCK_CHANNEL_REDESIGN_2026-06-22.md`

  Required decisions:

  - split snapshot-changing features from phase-changing features before
    semantic grouping;
  - treat snapshot proximity/combat edges and phase proximity/combat/co-pressure
    edges as different relation streams;
  - mark phase aggregates as auxiliary/context for snapshot-first models;
  - compare early mean-pooling against late fusion as an explicit ablation.

- [ ] Add tier-aware training design.
  -- Result: `docs/TIER_PROXY_DESIGN_2026-06-22.md`, `docs/SKILL_RATING_PROXY_DESIGN_2026-06-22.md`, `stdb/skill_proxy_etl.py`, `stdb/skill_rating_proxy_etl.py`, `stdb/skill_coverage_audit.py`

  Required decisions:

  - audit whether raw data contains official ranked tier/MMR;
  - if official tier is absent, do not use tier as a main input feature;
  - audit repeated-player coverage before any proxy-tier analysis;
  - forbid current-match final rank, survival time, win/loss, or top-k outcome
    from tier construction;
  - forbid the existing `ranked_rp_proxy` as a tier input because it is a
    current-match outcome proxy;
  - leave official tier/MMR as a future data-collection extension if repeated
    player coverage is too low;
  - run tier-conditioned or tier-specialist experiments only if an official
    tier exists or repeated-player coverage is empirically sufficient.

  Current proxy status:

  - simple historical averages are not valid tier proxies;
  - opponent-adjusted ratings are still conditional because they require enough
    repeated `player_id` coverage and match-graph connectivity;
  - unless coverage audit passes, keep tier as a limitation/future-work axis,
    not a current model channel.

- [ ] Add domain-aware model streams to the architecture spec.
  -- Result:

  Required streams:

  - snapshot team-state stream;
  - snapshot relation/message-passing stream;
  - phase team-state stream;
  - phase relation/message-passing stream;
  - global zone/phase context stream;
  - optional tier/domain conditioning stream;
  - strategy/prior-work stream;
  - hazard/rank/top-k head stream.

- [ ] Add domain-aware evaluation slices.
  -- Result:

  Required slices:

  - phase band;
  - map;
  - mode/team size;
  - perspective if available;
  - skill/lobby tier if available or derived;
  - alive roster band;
  - zone-pressure band;
  - local opponent-density band;
  - graph/null-control condition.

---

## 5. Immediate Presentation Fixes

- [ ] Add or revise a Problem Definition slide.
  -- Result:

  Required content:

  ```text
  Given team states and team-to-team relations observed up to phase k,
  predict each alive team's elimination risk before phase k+1.
  ```

  Must distinguish:

  - team-level, not player-level;
  - phase-cut prediction, not every-10-second output;
  - dynamic graph forecasting, not static post-game rank prediction.

- [ ] Add or revise a Spatial Definition slide.
  -- Result:

  Required content:

  ```text
  Spatial signals = map position + zone-relative pressure + team-to-team relations.
  ```

  Must show:

  - map coordinates and elevation;
  - shrinking safe-zone context;
  - proximity/combat/co-pressure graph relations.

- [ ] Redesign the relative-advantage slide.
  -- Result:

  Required structure:

  ```text
  Old interaction topology failed null control
    -> smoothing, not topology
    -> redesign relation as contact-gated relative advantage
    -> real advantage edges beat edge-shuffled null
  ```

  Required wording changes:

  - `real advantage edges - edge-shuffled null`
  - `with node-absolute features: real advantage edges - edge-shuffled null`
  - keep `preliminary / not main result` caveat visible.

- [ ] Update architecture slide based on the revised design spec.
  -- Result:

  Must include:

  - two input lanes;
  - explicit `Mean-pool to phase`;
  - phase-resolution main encoder;
  - output heads and supervised loss;
  - visual grammar where each shape/arrow/text has meaning.

---

## 6. Baseline / Benchmark Plan

### 6.0 Prior-Work Survey

- [~] Expand and verify prior literature.
  -- Result: `docs/PRIOR_WORK_SURVEY_2026-06-22.md`, `docs/RELATED_WORK_FRAME_2026-06-22.md`

  Priority order:

  - journal / venue reputation;
  - direct PUBG;
  - battle royale;
  - feature design, ablation design, and game telemetry modeling;
  - graph and spatio-temporal graph modeling;
  - useful adjacent domains such as team sports and dynamic survival analysis.

  Remaining verification tasks:

  - keep `docs/RELATED_WORK_FRAME_2026-06-22.md` as the canonical Related Work
    section structure;
  - verify Battle Royale genre papers and formal game ontology citations;
  - verify candidate Electronics 2025 PUBG rank-prediction paper metadata;
  - manually search ACM DL / IEEE Xplore for hidden PUBG or battle-royale papers;
  - add BibTeX entries for accepted references;
  - rewrite the paper's Related Work section around:
    Battle Royale genre -> formal game mechanics -> telemetry/spatial analytics
    -> PUBG prediction -> multi-agent spatio-temporal graph -> relational
    survival modeling.

### 6.1 Sagong-Style Baseline On Our Data

- [ ] Implement Sagong-style feature extraction on our telemetry-derived data.
  -- Result:

  Team-level adaptation of Sagong features:

  Aggressiveness:

  - phase attack count;
  - phase kill count;
  - phase total damage;
  - phase assist count.

  Activeness:

  - phase distance;
  - phase max density within local radius;
  - phase item count;
  - carepackage use.

- [ ] Compute FightScore and ActiveScore.
  -- Result:

  Required formula:

  ```text
  FightScore = sum(aggressiveness features)
  ActiveScore = sum(activeness features)
  ```

- [ ] Derive phase strategy type.
  -- Result:

  Strategy categories:

  - Hunter: high FightScore, high ActiveScore;
  - Sniper: high FightScore, low ActiveScore;
  - Traveler: low FightScore, high ActiveScore;
  - Camper: low FightScore, low ActiveScore.

- [ ] Build prior-work feature benchmark.
  -- Result:

  Same input, different models:

  - Logistic Regression;
  - Random Forest;
  - XGBoost / HistGradientBoosting;
  - MLP;
  - Phase TCN if appropriate.

  Purpose:

  > Test model-only gains under the same Sagong-style feature representation.

### 6.2 Public Kaggle External Benchmark

- [ ] Download and stage Kaggle PUBG Finish Placement Prediction data.
  -- Result:

  Dataset:

  ```text
  PUBG Finish Placement Prediction (Kernels Only)
  https://www.kaggle.com/competitions/pubg-finish-placement-prediction
  ```

  Notes:

  - current workspace does not contain `train_V2.csv` or `test_V2.csv`;
  - Kaggle API package/credentials are not currently available;
  - this benchmark is post-game aggregate placement prediction, not phase survival prediction.

- [ ] Implement external Kaggle baseline.
  -- Result:

  Models:

  - LightGBM / XGBoost / HGB;
  - Random Forest;
  - MLP if useful.

  Metrics:

  - MAE for `winPlacePerc`;
  - optional rank/placement correlation.

---

## 7. Representation Ablation Plan

- [ ] B0: Sagong-style strategy features only.
  -- Result:

- [ ] B1: B0 + our phase spatial/zone features.
  -- Result:

- [ ] B2: B1 + phase graph relations.
  -- Result:

- [ ] B3: B2 + legacy pooled-MR 10-second snapshot encoder.
  -- Result:

- [ ] B4: snapshot-first phase survival model.
  -- Result:

Purpose:

> Separate the gain from better model architecture from the gain from richer spatio-temporal representation.

- [ ] Add domain-channel ablation matrix.
  -- Result:

  Required ladder:

  ```text
  D1 roster only
    -> + D2 rotation
    -> + D3 zone/habitat pressure
    -> + D4 local opponent field
    -> + D5 combat exchange
    -> + D6 resource readiness
    -> + D7 strategy scores
    -> + D8 relation topology
  ```

- [ ] Add spatial semantics ablation.
  -- Result:

  Required variants:

  - no explicit spatial features;
  - absolute coordinates only;
  - zone-relative only;
  - enemy-relative/local density only;
  - zone-relative + enemy-relative;
  - full spatial + map-specific scaling.

- [ ] Add temporal-resolution ablation.
  -- Result:

  Required variants:

  - phase aggregate only;
  - last snapshot at phase cut;
  - 30-second snapshot sampling;
  - 10-second snapshot sequence;
  - 10-second sequence plus causal deltas.

---

## 8. Snapshot-First Redesign Plan

- [ ] Add label-only phase API.
  -- Result:

  Motivation:

  > Snapshot-first dataset should not depend on phase aggregate features except for labels/masks.

- [ ] Add snapshot-first evaluator.
  -- Result:

  Required outputs:

  - AUC, AP, BCE, Brier, ECE;
  - phase-band metrics;
  - prediction CSV with `match_id`, `phase_idx`, `team_id`, `y`, `p`.

- [ ] Add snapshot-level graph controls.
  -- Result:

  Controls:

  - observed;
  - no_graph;
  - node_shuffle;
  - edge_shuffle;
  - relation drop;
  - temporal shuffle if needed.

- [ ] Add snapshot experiment matrix and run scripts.
  -- Result:

  Proposed output roots:

  - `model_stgcn/checkpoints/snapshot_phase_survival_v1_smoke`
  - `model_stgcn/checkpoints/snapshot_phase_survival_v1_pilot`
  - `model_stgcn/checkpoints/snapshot_phase_survival_v1_full`

---

## 9. Writing / Framing Plan

- [ ] Rewrite problem statement in paper and deck.
  -- Result:

- [ ] Rewrite spatial definition in paper and deck.
  -- Result:

- [ ] Add prior-work benchmark section.
  -- Result:

  Required distinction:

  ```text
  Sagong-style phase strategy baseline on our data
  vs.
  public Kaggle post-game placement benchmark
  ```

- [ ] Add legacy-result caveat to architecture/results sections.
  -- Result:

  Required wording:

  > The legacy V3/MR model uses 10-second observations, but final prediction heads operate at phase resolution after snapshot-to-phase pooling.

- [ ] Keep relative-advantage result as future/preliminary.
  -- Result: corrected tiny `snapshot_relative_edge` phasefix pilot is best on
  tiny val loss; the sparse `snapshot_relative_edge_sparse` k=5 follow-up
  fixes the dense-gate issue and has slightly better tiny test loss. However,
  the feature-roll sparse null and topology-roll sparse null are effectively
  tied with real sparse, so treat this as support for sparse relation
  scaffolding, not yet for aligned relative-edge feature or topology semantics.
  Seed repeats keep the same interpretation. Comparator repeats show the sparse
  family is a candidate for scale-up, but not a stable pilot-scale win.
  Limit-50 repeats make sparse scaffold a viable narrow full-matrix candidate,
  still not a semantic graph claim. The narrow full-matrix runner is ready and
  smoke-tested. A recovered six-hour all-match 1-epoch run changes the
  immediate priority: completed metrics favor `snapshot_rgcn` over last-only,
  edge-shuffle, gate-only, sparse relative-edge, and sparse null controls.
  Treat relative-edge as future/preliminary and treat the R-GCN signal as
  hypothesis-generating until repeats/longer training finish.

  Required wording:

  > Preliminary phase-tier follow-up; not part of the present main result.
