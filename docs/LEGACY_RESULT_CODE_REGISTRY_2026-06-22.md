# Legacy Result-Code Registry (2026-06-22)

## Purpose

This registry answers one practical question:

> "Which code produced which result artifact?"

Use this document when presenting or citing current results before the redesigned
snapshot-first architecture has new outputs.

## Registry Identity

- Registry id: `legacy-result-code-registry-2026-06-22`
- Legacy result label: `legacy-v3-pooled-mr-phase-head-2026-06-22`
- Created: 2026-06-22
- Git HEAD at bookmark time: `93cfda996f30d938800340999b43c460c0e02e25`
- Worktree state: dirty. Do not treat Git HEAD alone as the reproducibility key.
- Companion JSON: `docs/legacy_result_code_registry_2026-06-22.json`
- Parent interpretation doc: `docs/LEGACY_CODE_RESULT_MAP_2026-06-22.md`

## One-Line Interpretation

The current result set is produced by a legacy V3 pooled multi-resolution
phase-head model: 10-second snapshots are encoded first, pooled to phase
resolution, fused with phase features/global context, and evaluated at phase
cut points.

It is not a strict 10-second snapshot-resolution survival model.

## How To Trace A Result

For V3 neural matrices, the canonical key is a row in:

```text
model_stgcn/checkpoints/<run_root>/matrix_metrics.csv
```

Each row contains:

- experiment name: `name`
- architectural knobs: `model`, `graph_encoder`, `edge_mode`,
  `global_fusion`, `team_decoder`
- result metrics: `auc`, `ap`, rank/top-k metrics
- checkpoint pointer: `checkpoint`
- evaluation pointer: `eval_json`

Therefore the trace is:

```text
matrix_metrics.csv row
  -> checkpoint = .../model.pt
  -> eval_json  = .../eval_test.json
  -> experiment definition in model_stgcn/v3/experiment_matrix.py
  -> runner in model_stgcn/v3/run_matrix.py
  -> train/eval modules in model_stgcn/v3/train.py and model_stgcn/v3/eval.py
```

Prediction exports in `model_stgcn/checkpoints/v3_pred` are forward-only
exports from selected saved checkpoints. They should be cited as exported
predictions, not separate training runs.

## Legacy Code Flow

```text
stdb/snapshot_queries.py::fetch_snapshot_match_tensors
  -> 10-second tensors: X_s, A_s, snap_phase_idx

stdb/queries.py::fetch_phase_match_tensors
  -> phase tensors: X_node, G, A, labels, masks, rank targets

model_stgcn/v3/factorized_data.py::fetch_factorized_mr_match
  -> joins snapshot and phase tensors for one match

model_stgcn/snapshot_encoder.py::SnapshotEncoder
  -> encodes 10-second snapshot sequence

model_stgcn/snapshot_encoder.py::pool_snapshots_to_phase
  -> mean-pools snapshot embeddings into phase bins

model_stgcn/v3/models.py::FactorizedMRGNN.forward
  -> concatenates phase node features with pooled snapshot embedding
  -> applies phase-level graph/global encoder and heads

model_stgcn/v3/eval.py
  -> computes hazard AUC/AP, rank metrics, top-5/top-10 metrics
```

This flow is the reason the current V3/MR results should be described as
"snapshot-informed phase-level survival prediction."

## Run Family Map

| Family id | Result root | Runner | Experiment group | Code contract | Main artifact |
| --- | --- | --- | --- | --- | --- |
| `v3_full_main_causal` | `model_stgcn/checkpoints/v3_full_main_causal` | `scripts/run_v3_full_main_matrix.sh` with `V3_OUT_DIR=model_stgcn/checkpoints/v3_full_main_causal` | `main` | V3 pooled-MR architecture comparison | `matrix_metrics.csv` |
| `v3_full_controls` | `model_stgcn/checkpoints/v3_full_controls` | `scripts/run_v3_full_controls_matrix.sh` | manifest: `resolution`, `decoder`, `graph_control`; completed metrics rows: `resolution`, partial `decoder` | phase vs pooled-MR and decoder controls | `matrix_metrics.csv` |
| `v3_full_graph_relation` | `model_stgcn/checkpoints/v3_full_graph_relation` | `scripts/run_v3_full_graph_relation_matrix.sh` | `graph_relation` | phase graph relation/null-control matrix | `matrix_metrics.csv` |
| `v3_full_feature_causal` | `model_stgcn/checkpoints/v3_full_feature_causal` | `scripts/run_v3_full_feature_matrix.sh` | `feature` | V3 pooled-MR feature preset ablation | `matrix_metrics.csv` |
| `feature_ablation_full_db_manifest_v2` | `model_stgcn/checkpoints/feature_ablation_full_db_manifest_v2` | `model_stgcn/feature_ablation.py` | HGB tabular ablation | phase-row tabular feature-family baseline | `feature_ablation_metrics.csv` |
| `v3_pred` | `model_stgcn/checkpoints/v3_pred` | `scripts/v3_gen_predictions.sh`, `scripts/v3_gen_resolution_preds.sh` | selected saved checkpoints | prediction export for significance/plotting | `*.eval.json`, `*.pred.csv` |

## V3 Main Architecture Results

Source:
`model_stgcn/checkpoints/v3_full_main_causal/matrix_metrics.csv`

Code path:

```text
scripts/run_v3_full_main_matrix.sh
  -> python3 -m model_stgcn.v3.run_matrix --groups main
  -> model_stgcn.v3.train
  -> model_stgcn.v3.eval
```

| Row name | Model | Graph encoder | Global fusion | Decoder | AUC | AP | Checkpoint / eval root |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| `mr_tcn` | `mr` | `none` | `concat` | `mlp` | 0.8358 | 0.7189 | `v3_full_main_causal/main/mr_tcn` |
| `mr_gcn` | `mr` | `gcn` | `concat` | `mlp` | 0.8461 | 0.7380 | `v3_full_main_causal/main/mr_gcn` |
| `mr_sage` | `mr` | `sage` | `concat` | `mlp` | 0.8438 | 0.7346 | `v3_full_main_causal/main/mr_sage` |
| `mr_gat` | `mr` | `gat` | `concat` | `mlp` | 0.8496 | 0.7453 | `v3_full_main_causal/main/mr_gat` |
| `mr_rgcn` | `mr` | `rgcn` | `concat` | `mlp` | 0.8460 | 0.7388 | `v3_full_main_causal/main/mr_rgcn` |
| `mr_rgcn_film` | `mr` | `rgcn` | `film` | `mlp` | 0.8465 | 0.7392 | `v3_full_main_causal/main/mr_rgcn_film` |
| `mr_rgcn_attn_rank` | `mr` | `rgcn` | `concat` | `attention` | 0.8449 | 0.7374 | `v3_full_main_causal/main/mr_rgcn_attn_rank` |
| `mr_farnet` | `mr` | `rgcn` | `film` | `attention` | 0.8454 | 0.7381 | `v3_full_main_causal/main/mr_farnet` |

Safe claim:

> In the implemented V3 pooled-MR phase-head setting, graph-aware neural
> variants reach hazard AUC around 0.84-0.85, with `mr_gat` the strongest row
> in this matrix.

## Resolution And Decoder Control Results

Source:
`model_stgcn/checkpoints/v3_full_controls/matrix_metrics.csv`

Code path:

```text
scripts/run_v3_full_controls_matrix.sh
  -> python3 -m model_stgcn.v3.run_matrix --groups resolution,decoder,graph_control
```

Important: `matrix_manifest.json` lists 12 planned experiments, but the current
`matrix_metrics.csv` contains 6 completed rows. Cite only the rows listed below
unless the missing graph-control rows are regenerated and added to the metrics
CSV.

| Row name | Contract | AUC | AP | Checkpoint / eval root |
| --- | --- | ---: | ---: | --- |
| `phase_tcn` | phase features, no graph | 0.7872 | 0.6679 | `v3_full_controls/resolution/phase_tcn` |
| `phase_rgcn` | phase features plus phase R-GCN | 0.7926 | 0.6770 | `v3_full_controls/resolution/phase_rgcn` |
| `mr_tcn` | snapshot-informed MR, no graph | 0.8349 | 0.7182 | `v3_full_controls/resolution/mr_tcn` |
| `mr_rgcn` | snapshot-informed MR plus phase R-GCN | 0.8467 | 0.7391 | `v3_full_controls/resolution/mr_rgcn` |
| `mr_rgcn_hazard_rank` | MR R-GCN, hazard/rank MLP decoder | 0.8460 | 0.7382 | `v3_full_controls/decoder/mr_rgcn_hazard_rank` |
| `mr_rgcn_pairwise_rank` | MR R-GCN, pairwise rank loss enabled | 0.8462 | 0.7387 | `v3_full_controls/decoder/mr_rgcn_pairwise_rank` |

Safe claim:

> Adding the 10-second snapshot encoder before phase-level prediction is a much
> larger change than swapping the phase decoder in this result set.

## Graph Relation / Null-Control Results

Source:
`model_stgcn/checkpoints/v3_full_graph_relation/matrix_metrics.csv`

Code path:

```text
scripts/run_v3_full_graph_relation_matrix.sh
  -> python3 -m model_stgcn.v3.run_matrix --groups graph_relation
  -> edge controls from model_stgcn/v3/edge_controls.py
```

| Row name | Relation/control | AUC | AP | Checkpoint / eval root |
| --- | --- | ---: | ---: | --- |
| `graph_core_habitat_no_graph` | no graph | 0.8343 | 0.7178 | `v3_full_graph_relation/graph_relation/graph_core_habitat_no_graph` |
| `graph_core_habitat_proximity` | proximity only | 0.8455 | 0.7370 | `v3_full_graph_relation/graph_relation/graph_core_habitat_proximity` |
| `graph_core_habitat_combat` | combat only | 0.8373 | 0.7235 | `v3_full_graph_relation/graph_relation/graph_core_habitat_combat` |
| `graph_core_habitat_copressure` | co-pressure only | 0.8367 | 0.7233 | `v3_full_graph_relation/graph_relation/graph_core_habitat_copressure` |
| `graph_core_habitat_proximity_combat` | proximity + combat | 0.8462 | 0.7386 | `v3_full_graph_relation/graph_relation/graph_core_habitat_proximity_combat` |
| `graph_core_habitat_proximity_copressure` | proximity + co-pressure | 0.8455 | 0.7372 | `v3_full_graph_relation/graph_relation/graph_core_habitat_proximity_copressure` |
| `graph_core_habitat_full_observed` | full unshuffled phase graph | 0.8453 | 0.7371 | `v3_full_graph_relation/graph_relation/graph_core_habitat_full_observed` |
| `graph_core_habitat_full_node_shuffle` | node-shuffled phase graph | 0.8448 | 0.7361 | `v3_full_graph_relation/graph_relation/graph_core_habitat_full_node_shuffle` |
| `graph_core_habitat_full_edge_shuffle` | edge-shuffled phase graph | 0.8469 | 0.7381 | `v3_full_graph_relation/graph_relation/graph_core_habitat_full_edge_shuffle` |

Safe claim:

> In the implemented phase-tier graph relation matrix, the full unshuffled
> proximity/combat/co-pressure graph does not robustly beat the edge-shuffle
> null. This motivates redesigning relation semantics; it does not prove that
> graph topology is useless in general.

## V3 Feature Preset Results

Source:
`model_stgcn/checkpoints/v3_full_feature_causal/matrix_metrics.csv`

Code path:

```text
scripts/run_v3_full_feature_matrix.sh
  -> python3 -m model_stgcn.v3.run_matrix --groups feature
```

| Row name | Feature preset | AUC | AP | Checkpoint / eval root |
| --- | --- | ---: | ---: | --- |
| `feat_core` | `core` | 0.8306 | 0.7102 | `v3_full_feature_causal/feature/feat_core` |
| `feat_stdb` | `stdb` | 0.8280 | 0.7052 | `v3_full_feature_causal/feature/feat_stdb` |
| `feat_network` | `network` | 0.8284 | 0.7044 | `v3_full_feature_causal/feature/feat_network` |
| `feat_habitat` | `habitat` | 0.8339 | 0.7139 | `v3_full_feature_causal/feature/feat_habitat` |
| `feat_core_stdb` | `core_stdb` | 0.8303 | 0.7119 | `v3_full_feature_causal/feature/feat_core_stdb` |
| `feat_core_network` | `core_network` | 0.8312 | 0.7117 | `v3_full_feature_causal/feature/feat_core_network` |
| `feat_core_habitat` | `core_habitat` | 0.8343 | 0.7178 | `v3_full_feature_causal/feature/feat_core_habitat` |
| `feat_base66` | `base66` | 0.8297 | 0.7107 | `v3_full_feature_causal/feature/feat_base66` |
| `feat_full` | `full` | 0.8359 | 0.7201 | `v3_full_feature_causal/feature/feat_full` |

Safe claim:

> Under the pooled-MR phase-head architecture, feature-preset differences are
> smaller than the phase-vs-MR resolution difference, and `feat_full` is the
> strongest feature preset in this matrix.

## Tabular Phase Feature Ablation Results

Source:
`model_stgcn/checkpoints/feature_ablation_full_db_manifest_v2/feature_ablation_metrics.csv`

Code path:

```text
model_stgcn/feature_ablation.py
  -> sklearn HistGradientBoostingClassifier
  -> phase-row tabular features
```

| Row name | Features | AUC | AP |
| --- | ---: | ---: | ---: |
| `a0_official_raw` | 78 | 0.7589 | 0.6564 |
| `a1_prior_raw` | 12 | 0.7185 | 0.5704 |
| `a2_prior_scores` | 5 | 0.6488 | 0.4547 |
| `a3_current_conventional` | 57 | 0.7523 | 0.6477 |
| `a4_stdb_spatial` | 18 | 0.7212 | 0.6007 |
| `a5_raw_plus_stdb` | 95 | 0.7676 | 0.6667 |
| `a6_raw_plus_network` | 121 | 0.7644 | 0.6621 |
| `a7_raw_plus_habitat` | 110 | 0.7828 | 0.6833 |
| `a8_raw_stdb_network` | 138 | 0.7692 | 0.6685 |
| `a9_raw_stdb_habitat` | 127 | 0.7834 | 0.6843 |
| `a10_raw_network_habitat` | 153 | 0.7832 | 0.6841 |
| `a11_full_representation` | 170 | 0.7838 | 0.6846 |

Safe claim:

> This is a phase-row tabular baseline, not a neural ST-Graph result. It is
> useful for feature-family ablation and for showing that habitat/zone pressure
> features add signal to conventional features.

## Prediction Export Map

Source root:
`model_stgcn/checkpoints/v3_pred`

| Export artifact pattern | Produced from | Script | Use |
| --- | --- | --- | --- |
| `v3_full_main_causal_main_mr_gat.{eval.json,pred.csv}` | `v3_full_main_causal/main/mr_gat/model.pt` | `scripts/v3_gen_predictions.sh` | selected main-model prediction export |
| `v3_full_graph_relation_*full_observed.{eval.json,pred.csv}` | full observed graph checkpoint | `scripts/v3_gen_predictions.sh` | graph-control significance/plotting |
| `v3_full_graph_relation_*full_edge_shuffle.{eval.json,pred.csv}` | edge-shuffle graph checkpoint | `scripts/v3_gen_predictions.sh` | null-control significance/plotting |
| `v3_full_graph_relation_*no_graph.{eval.json,pred.csv}` | no-graph checkpoint | `scripts/v3_gen_predictions.sh` | no-graph comparison |
| `resolution_phase_tcn.{eval.json,pred.csv}` | `v3_full_controls/resolution/phase_tcn/model.pt` | `scripts/v3_gen_resolution_preds.sh` | phase-only resolution export |
| `resolution_phase_rgcn.{eval.json,pred.csv}` | `v3_full_controls/resolution/phase_rgcn/model.pt` | `scripts/v3_gen_resolution_preds.sh` | phase R-GCN export |
| `resolution_mr_tcn.{eval.json,pred.csv}` | `v3_full_controls/resolution/mr_tcn/model.pt` | `scripts/v3_gen_resolution_preds.sh` | MR no-graph export |
| `resolution_mr_rgcn.{eval.json,pred.csv}` | `v3_full_controls/resolution/mr_rgcn/model.pt` | `scripts/v3_gen_resolution_preds.sh` | MR R-GCN export |
| `baseline_hgb.pred.csv`, `baseline_lr.pred.csv` | tabular baselines | separate baseline export | non-neural comparator predictions |

## Code Checksum Registry

| Path | SHA-256 | Role |
| --- | --- | --- |
| `model_stgcn/v3/models.py` | `79c0d6729af3f43419b34fc3e65b70818d1e90b55ea94c96419e43ca36554cd9` | V3 phase and pooled-MR neural architectures |
| `model_stgcn/v3/factorized_data.py` | `e8e681e4b5dd759c8fd5a8af1a323d09db98f8a9dbcdabd5c9915614e757816c` | V3 phase/MR dataset assembly |
| `model_stgcn/v3/spec.py` | `f3c8df2a5d2b3d9b3c036a256427af14a0bfc7c6e24f0d4327ddf079464a5902` | V3 feature preset/spec definitions |
| `model_stgcn/v3/train.py` | `1c2f341591ebd70991fff6ef13759d700de5f2c3de3182e4642b609c27e97286` | V3 trainer |
| `model_stgcn/v3/eval.py` | `0f91ebc22eabf4af7829539e53ed5d73159b5a63f2ad1bd5333b036c9313abdc` | V3 evaluator |
| `model_stgcn/v3/run_matrix.py` | `20cae4d56b726334c3df689e94a1630a562231312aebdcd1ed5f9e18742fb52a` | V3 matrix runner |
| `model_stgcn/v3/experiment_matrix.py` | `be830298019bdaf65d468e2b14fa8a52b0870d3f1da046e7fdbf5212ad2f1c30` | V3 experiment definitions |
| `model_stgcn/v3/edge_controls.py` | `6fd7c61a621f172137970004b02df468901e72a1f051871204b07ab50f9ea44e` | V3 graph null controls |
| `model_stgcn/snapshot_encoder.py` | `acb643199a196007d70d655d254fbbe7ecb346951ea8064fea975f14ee3ce4e9` | 10-second snapshot encoder and phase pooling |
| `model_stgcn/dyn_stgnn.py` | `d826a20f44fe049f357a3dd6dff3f413981c739837d6e620847975f44882bc85` | dynamic spatial-temporal block |
| `stdb/queries.py` | `f6b5ee3262cd01eb73bdad9fa714b7039083a280fdc3fc6fdf21e829f8899715` | phase tensors, labels, phase graph |
| `stdb/snapshot_queries.py` | `2e85babec0057988b3b98b9e5faa08968b2db50cc11ccaef32c61f9acefb55cb` | 10-second snapshot tensors |
| `scripts/run_v3_full_main_matrix.sh` | `1ab82dc832a0c0b5bfe1eda2cecff49d68c13800b9513a35d81aff12e1eea279` | main matrix shell entrypoint |
| `scripts/run_v3_full_controls_matrix.sh` | `2a517aabc76770cb5f9d3a52cb2a80e74ef659569d82ef97a17018135d9d8bc0` | control matrix shell entrypoint |
| `scripts/run_v3_full_graph_relation_matrix.sh` | `1921bc670ebe669bff66ac7d603d70d1cde5dbb34c85db5858a3173a84934ba6` | graph relation matrix shell entrypoint |
| `scripts/run_v3_full_feature_matrix.sh` | `e562496c6b3f05228d8f1274afe3e8997ea091eee4184a5bfbad04505e233710` | feature matrix shell entrypoint |
| `scripts/v3_gen_predictions.sh` | `d47077dd54656a8a4e59df2b53d48cb3d5d6d1257ed9715618990ee2b4e85cea` | selected V3 prediction export |
| `scripts/v3_gen_resolution_preds.sh` | `fabb8e9350453dbae9872c68489bed6891b543f3faab763f01441d38eaa9bad9` | resolution prediction export |

## Result Checksum Registry

| Path | SHA-256 | Interpretation |
| --- | --- | --- |
| `model_stgcn/checkpoints/v3_full_main_causal/matrix_metrics.csv` | `d7542a1bb329f02b7e90b3fdece847ee9a87f2a50a49d6dc4edbe3110102d4ab` | V3 main architecture matrix |
| `model_stgcn/checkpoints/v3_full_controls/matrix_metrics.csv` | `01b0329bbda431c3467b7bd8c1334f38369d55adf276454c171e947e0421a5b2` | V3 resolution/decoder controls |
| `model_stgcn/checkpoints/v3_full_graph_relation/matrix_metrics.csv` | `c03d640d92a8cb8734a29cf61d74d8cb0456a06d2db37a36e6d4525da3f60cc0` | V3 phase graph relation/null controls |
| `model_stgcn/checkpoints/v3_full_feature_causal/matrix_metrics.csv` | `41d6e9af7a961a5aafe86ac19451aff0f4ff4e385709eb1e664f9e72b3247b8a` | V3 feature preset matrix |
| `model_stgcn/checkpoints/feature_ablation_full_db_manifest_v2/feature_ablation_metrics.csv` | `3f3a8d70835a160debb10d82e59d71ddff332a93a674f2145bcff1f00d920b24` | phase-row tabular feature ablation |

## Freeze Rules

1. Do not overwrite any result root listed in this document.
2. If a listed code file changes, keep this registry as historical and create a
   new registry id for new results.
3. New snapshot-first runs should use new roots, for example:
   `model_stgcn/checkpoints/snapshot_phase_survival_v1_*`.
4. Presentation slides should cite the family id plus the artifact path, not
   only the metric number.
