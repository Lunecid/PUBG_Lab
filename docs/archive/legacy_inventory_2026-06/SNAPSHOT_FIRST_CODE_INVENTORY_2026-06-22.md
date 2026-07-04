# Snapshot-First Code Inventory (2026-06-22)

## Legend

- `target`: 새 snapshot-first canonical path에 직접 포함.
- `modify`: 새 설계에 맞게 수정하거나 새 wrapper가 필요.
- `legacy`: 기존 결과 재현/설명용으로 보존.
- `support`: 데이터 구축, 분석, 문서/그림 생성 등 모델 contract와 직접 충돌하지 않음.

이 파일은 삭제 목록이 아니다. 기존 결과를 보존하면서 새 설계로 넘어가기 위한 분류표이다.

## `model_stgcn`

| File | Class | Action |
| --- | --- | --- |
| `model_stgcn/snapshot_phase_survival.py` | target | 새 canonical model core로 유지하고 ablation hooks 확장 |
| `model_stgcn/snapshot_phase_dataset.py` | modify | label-only phase API로 바꿔 phase aggregate feature dependency 제거 |
| `model_stgcn/train_snapshot_phase.py` | modify | evaluator, checkpoint metadata, matrix runner와 연결 |
| `model_stgcn/snapshot_encoder.py` | target | snapshot encoder component로 재사용. pooling 전제가 없는지 call site에서 통제 |
| `model_stgcn/config.py` | modify | snapshot-first constants, output roots, model contract names 추가 |
| `model_stgcn/splits.py` | target | legacy/new 결과 비교를 위해 같은 split 재사용 |
| `model_stgcn/calibration.py` | modify | snapshot evaluator output schema에 연결 |
| `model_stgcn/bootstrap_ci.py` | modify | prediction CSV schema를 새 evaluator와 맞춤 |
| `model_stgcn/train_phase.py` | legacy | phase-only baseline으로 보존 |
| `model_stgcn/eval_phase.py` | legacy | phase-only baseline evaluator |
| `model_stgcn/dyn_stgnn.py` | legacy | phase-resolution ST-GNN baseline |
| `model_stgcn/dataset.py` | legacy | phase dataset path |
| `model_stgcn/train_mr.py` | legacy | pooled-MR baseline으로 보존 |
| `model_stgcn/eval_mr.py` | legacy | pooled-MR baseline evaluator |
| `model_stgcn/mr_stgnn.py` | legacy | snapshot encoder 후 phase pooling하는 legacy MR architecture |
| `model_stgcn/mr_dataset.py` | legacy | legacy MR dataset |
| `model_stgcn/feature_ablation.py` | legacy | phase feature-only A0-A11 baseline |
| `model_stgcn/manifest_importance.py` | legacy | phase manifest importance 분석 |
| `model_stgcn/domain_ablation.py` | legacy | phase/legacy result robustness |
| `model_stgcn/seed_repeat_experiments.py` | modify | snapshot-first runner 지원 추가 필요 |
| `model_stgcn/edge_sanity.py` | modify | phase edge sanity와 snapshot edge sanity를 분리 |
| `model_stgcn/edge_modes.py` | modify | snapshot relation mode와 phase relation mode 명시 분리 |
| `model_stgcn/baselines.py` | modify | legacy baselines와 snapshot baselines를 함께 export |
| `model_stgcn/phase_wise_eval.py` | modify | snapshot prediction schema 지원 필요 |
| `model_stgcn/phase_result_analysis.py` | legacy | legacy phase/MR result 분석 |
| `model_stgcn/novelty_analysis.py` | support | 분석용. 새 result schema 필요 시 adapter 추가 |
| `model_stgcn/train.py` | legacy | older STGCN training path |
| `model_stgcn/eval.py` | legacy | older STGCN evaluation path |
| `model_stgcn/infer.py` | legacy | older inference path |
| `model_stgcn/stgcn.py` | legacy | older STGCN architecture |
| `model_stgcn/__init__.py` | support | package marker |

## `model_stgcn/v3`

| File | Class | Action |
| --- | --- | --- |
| `model_stgcn/v3/spec.py` | legacy | V3 pooled-MR spec로 보존 |
| `model_stgcn/v3/factorized_data.py` | legacy | V3 phase-head data path |
| `model_stgcn/v3/models.py` | legacy | factorized pooled-MR V3 architecture |
| `model_stgcn/v3/train.py` | legacy | V3 pooled-MR training |
| `model_stgcn/v3/eval.py` | legacy | V3 pooled-MR evaluation |
| `model_stgcn/v3/experiment_matrix.py` | legacy | V3 matrix. 새 matrix와 혼합하지 말 것 |
| `model_stgcn/v3/run_matrix.py` | legacy | V3 runner. 새 snapshot runner는 별도 생성 권장 |
| `model_stgcn/v3/edge_controls.py` | legacy | phase graph controls |
| `model_stgcn/v3/advantage_edges.py` | legacy | future/pilot relation design |
| `model_stgcn/v3/adv_model.py` | legacy | advantage-edge pilot model |
| `model_stgcn/v3/__init__.py` | support | package marker |

## `stdb`

| File | Class | Action |
| --- | --- | --- |
| `stdb/snapshot_queries.py` | target | canonical snapshot tensor source |
| `stdb/queries.py` | modify | label-only phase fetch 추가. phase feature fetch는 legacy로 유지 |
| `stdb/snapshot_feature_etl.py` | target | snapshot feature source. 10초 grid assumptions 문서화 |
| `stdb/phase_edge_etl.py` | legacy | phase graph source. snapshot graph와 이름/문서 분리 |
| `stdb/rq1_feature_etl.py` | legacy | phase engineered features |
| `stdb/f0_feature_etl.py` | legacy | phase official/raw features |
| `stdb/network_feature_etl.py` | legacy | phase network features |
| `stdb/habitat_feature_etl.py` | legacy | phase habitat features |
| `stdb/phase_spatial_feature_etl.py` | legacy | phase spatial features |
| `stdb/v2_view.py` | legacy | phase feature view |
| `stdb/etl.py` | support | base ETL utilities |
| `stdb/build_available.py` | support | availability build utility |
| `stdb/audit_raw.py` | support | raw data audit |
| `stdb/patch_combat_zone_pass.py` | support | data patch utility |
| `stdb/conn.py` | support | DB connection utilities |
| `stdb/__init__.py` | support | package marker |

## `scripts`

| File | Class | Action |
| --- | --- | --- |
| `scripts/run_v3_full_main_matrix.sh` | legacy | legacy V3 main result reproduction |
| `scripts/run_v3_full_controls_matrix.sh` | legacy | legacy V3 controls |
| `scripts/run_v3_full_feature_matrix.sh` | legacy | legacy V3 feature matrix |
| `scripts/run_v3_full_graph_relation_matrix.sh` | legacy | legacy V3 graph relation |
| `scripts/run_v3_pilot_matrix.sh` | legacy | legacy V3 pilot |
| `scripts/run_v3_feature_pilot_matrix.sh` | legacy | legacy V3 feature pilot |
| `scripts/run_v3_graph_relation_pilot_matrix.sh` | legacy | legacy V3 graph relation pilot |
| `scripts/run_v3_graph_relation_seed_repeats.sh` | legacy | legacy V3 seed repeat |
| `scripts/v3_gen_predictions.sh` | legacy | V3 prediction export |
| `scripts/v3_gen_resolution_preds.sh` | legacy | V3 resolution prediction export |
| `scripts/v3_phase_significance.py` | legacy | V3 significance analysis |
| `scripts/v3_resolution_significance.py` | legacy | V3 resolution significance |
| `scripts/v3_tabular_baseline.py` | legacy | V3 tabular baseline |
| `scripts/run_mr_full_squad.sh` | legacy | pooled-MR full run |
| `scripts/run_mr_full_squad_seed_repeats.sh` | legacy | pooled-MR seed repeats |
| `scripts/run_mr_shuffle_robustness.sh` | legacy | pooled-MR temporal shuffle result |
| `scripts/mr_paired_significance.py` | legacy | pooled-MR significance |
| `scripts/run_seed_repeat.sh` | legacy | old phase/MR seed repeat |
| `scripts/run_full_db_reexperiment.sh` | legacy | old full DB suite |
| `scripts/run_full_db_edge_ablation.sh` | legacy | phase edge ablation |
| `scripts/run_full_db_aux_ablation.sh` | legacy | phase auxiliary ablation |
| `scripts/run_full_db_robustness.sh` | legacy | legacy robustness |
| `scripts/run_full_db_followup.sh` | legacy | legacy follow-up |
| `scripts/rerun_full_db_suite.sh` | legacy | legacy suite orchestration |
| `scripts/run_v2_node_gnn.sh` | legacy | node/phase GNN experiment |
| `scripts/run_mr_1map.sh` | legacy | pooled-MR one-map experiment |
| `scripts/run_solo_duo.sh` | legacy | mode/domain run |
| `scripts/run_adv_base.sh` | legacy | advantage-edge pilot |
| `scripts/run_adv_experiment.sh` | legacy | advantage-edge pilot |
| `scripts/run_adv_kablation.sh` | legacy | advantage-edge pilot |
| `scripts/run_adv_nodeabs.sh` | legacy | advantage-edge pilot |
| `scripts/adv_precompute.py` | legacy | advantage-edge precompute |
| `scripts/adv_analysis.py` | legacy | advantage-edge analysis |
| `scripts/advantage_edge_feasibility.py` | legacy | future design feasibility |
| `scripts/analyze_graph_signal.py` | legacy | existing graph signal analysis |
| `scripts/highground_signal_test.py` | legacy | existing signal test |
| `scripts/phase_significance.py` | legacy | phase/MR significance |
| `scripts/phase_time_profile.py` | legacy | phase timing profile |
| `scripts/audit_spatial_features.py` | support | data audit |
| `scripts/build_data_availability_domains.py` | support | data availability |
| `scripts/build_experiment_domain_datasets.py` | support | dataset/domain manifest |
| `scripts/update_data_availability_summary.py` | support | docs/data summary |
| `scripts/convert_raw_to_usable.py` | support | raw data conversion |
| `scripts/run_raw_to_usable_batch.sh` | support | raw data conversion orchestration |
| `scripts/generate_dynamic_graph_example.py` | support | docs/example figure |
| `scripts/generate_team_phase_feature_example.py` | support | docs/example figure |
| `scripts/make_labeling_figure.py` | support | presentation figure |
| `scripts/make_phase_summary_figure.py` | support | presentation figure |
| `scripts/make_paper_figures_svg.py` | support | paper figure |
| `scripts/make_presentation_figures.py` | support | presentation figure |
| `scripts/make_realmap_figure.py` | support | presentation figure |
| `scripts/create_pubg_safe_zone_proposal_docx.py` | support | document artifact |
| `scripts/overnight_runs.sh` | modify | 새 runner가 생기면 legacy/new 분리 필요 |

## `tests`

| File | Class | Action |
| --- | --- | --- |
| `tests/test_snapshot_phase_survival.py` | target | 새 canonical unit tests |
| `tests/test_snapshot_features.py` | target | snapshot feature assumptions 유지 |
| `tests/test_mr_leakage.py` | legacy | pooled-MR leakage guard로 보존 |
| `tests/test_v3_factorized.py` | legacy | V3 behavior guard |
| `tests/test_v3_edge_controls.py` | legacy | phase graph controls guard |
| `tests/test_edge_sanity_controls.py` | legacy | existing edge controls guard |
| `tests/test_label_alignment.py` | target | labels must remain valid for new dataset |
| `tests/test_masks.py` | target | masks remain critical for snapshot cuts |
| `tests/test_adjacency.py` | modify | snapshot adjacency cases 추가 |
| `tests/test_model_forward.py` | legacy | old model forward guard |
| `tests/test_network_features.py` | legacy | phase feature family guard |
| `tests/test_v2_node_features.py` | legacy | phase/node feature guard |
| `tests/test_habitat_features.py` | legacy | phase habitat feature guard |
| `tests/conftest.py` | support | shared test setup |

## Missing Target Files

These files do not exist yet and are required for the redesigned system:

- `model_stgcn/eval_snapshot_phase.py`
- `model_stgcn/snapshot_edge_controls.py`
- `model_stgcn/snapshot_experiment_matrix.py`
- `model_stgcn/run_snapshot_matrix.py`
- `scripts/run_snapshot_phase_smoke.sh`
- `scripts/run_snapshot_phase_pilot_matrix.sh`
- `scripts/run_snapshot_phase_full_matrix.sh`
- `tests/test_snapshot_edge_controls.py`
- `tests/test_snapshot_phase_eval.py`

