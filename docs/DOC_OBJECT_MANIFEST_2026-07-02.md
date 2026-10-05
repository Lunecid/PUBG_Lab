# Documentation Object Manifest

Date: 2026-07-02

This manifest treats each Markdown file as a documentation object. It is generated from the current `docs/` tree after the OOP-style cleanup pass.

## Lifecycle Legend

| State | Meaning |
| --- | --- |
| `active` | current public interface, policy, or package object |
| `supporting` | useful background but not the canonical contract |
| `handoff` | operational continuation state |
| `generated` | reproducible presentation/report artifact |
| `legacy` | historical object kept for provenance |

## Object Registry

| Path | Class | Lifecycle | Responsibility |
| --- | --- | --- | --- |
| `docs/DESIGN_DOC_STRUCTURE_2026-07-02.md` | `DocStructurePolicy` | `active` | folder routing policy |
| `docs/DOC_OBJECT_MANIFEST_2026-07-02.md` | `DocObjectManifest` | `active` | documentation object registry |
| `docs/DOC_OBJECT_MODEL_2026-07-02.md` | `DocObjectModel` | `active` | documentation object model |
| `docs/README.md` | `RootPackage` | `active` | public entry point |
| `docs/archive/README.md` | `DeprecatedObjectStoreInterface` | `active` | archive public interface |
| `docs/archive/generated_presentation_2026-06/deliverables/01_graph_presentation.md` | `GeneratedArtifact` | `generated` | archived historical/generated object |
| `docs/archive/generated_presentation_2026-06/deliverables/01_graph_research_report.md` | `GeneratedArtifact` | `generated` | archived historical/generated object |
| `docs/archive/generated_presentation_2026-06/deliverables/02_stdb_paper_sections.md` | `GeneratedArtifact` | `generated` | archived historical/generated object |
| `docs/archive/generated_presentation_2026-06/deliverables/03_fusion_interim_report.md` | `GeneratedArtifact` | `generated` | archived historical/generated object |
| `docs/archive/generated_presentation_2026-06/deliverables/GRAPH_DECK_MASTER_SPEC_AND_DESIGN_2026-06-21.md` | `GeneratedArtifact` | `generated` | archived historical/generated object |
| `docs/archive/generated_presentation_2026-06/deliverables/MODEL_ARCHITECTURE_DESIGN_SPEC_2026-06-22.md` | `GeneratedArtifact` | `generated` | archived historical/generated object |
| `docs/archive/generated_presentation_2026-06/deliverables/design_spec_graph_deck.md` | `GeneratedArtifact` | `generated` | archived historical/generated object |
| `docs/archive/generated_presentation_2026-06/deliverables/design_spec_graph_deck_academic_revised.md` | `GeneratedArtifact` | `generated` | archived historical/generated object |
| `docs/archive/generated_presentation_2026-06/deliverables/design_spec_labeling_metrics.md` | `GeneratedArtifact` | `generated` | archived historical/generated object |
| `docs/archive/legacy_inventory_2026-06/CODE_REVIEW_2026-06-12.md` | `LegacyInventory` | `legacy` | archived historical/generated object |
| `docs/archive/legacy_inventory_2026-06/FEATURE_OVERLAP_AUDIT_2026-06-12.md` | `LegacyInventory` | `legacy` | archived historical/generated object |
| `docs/archive/legacy_inventory_2026-06/LEGACY_CODE_RESULT_MAP_2026-06-22.md` | `LegacyInventory` | `legacy` | archived historical/generated object |
| `docs/archive/legacy_inventory_2026-06/LEGACY_RESULT_CODE_REGISTRY_2026-06-22.md` | `LegacyInventory` | `legacy` | archived historical/generated object |
| `docs/archive/legacy_inventory_2026-06/SNAPSHOT_FIRST_CODE_INVENTORY_2026-06-22.md` | `LegacyInventory` | `legacy` | archived historical/generated object |
| `docs/archive/legacy_inventory_2026-06/SNAPSHOT_FIRST_REDESIGN_AUDIT_2026-06-22.md` | `LegacyInventory` | `legacy` | archived historical/generated object |
| `docs/archive/legacy_status_reports_2026-06/CHAT_RECORD_2026-06-23_ST_SHORT_PAPER.md` | `LegacyStatus` | `legacy` | archived historical/generated object |
| `docs/archive/legacy_status_reports_2026-06/PRESENTATION_DATA_2026-06-21.md` | `LegacyStatus` | `legacy` | archived historical/generated object |
| `docs/data_layer/BATTLE_ROYALE_COHORT_REGISTRY_2026-07-03.md` | `DataContract` | `active` | locked map / perspective / team-size cohort registry |
| `docs/data_layer/BATTLE_ROYALE_PRIMARY_COHORT_DECISION_2026-07-03.md` | `DataContract` | `supporting` | supporting group-level TPP squad cohort note |
| `docs/data_layer/BR_DATA_LAYER_CONTRACT.md` | `DataContract` | `active` | package documentation object |
| `docs/data_layer/DATA_AVAILABILITY_BY_DOMAIN.md` | `DataContract` | `active` | package documentation object |
| `docs/data_layer/EXPERIMENT_DATASETS_BY_DOMAIN.md` | `DataContract` | `active` | package documentation object |
| `docs/data_layer/PUBG_OFFICIAL_TELEMETRY_OBSERVABILITY_TABLE_2026-07-05.md` | `DataContract` | `active` | RQ2 table of what PUBG official telemetry provides, derives, labels, proxies, and does not directly provide |
| `docs/data_layer/README.md` | `DataContractInterface` | `active` | package public interface |
| `docs/data_layer/RESEARCH_COHORT_FLAT.md` | `DataContract` | `active` | package documentation object |
| `docs/data_layer/RQ1_STDB_REDESIGN.md` | `DataContract` | `active` | package documentation object |
| `docs/experiment_design/FULL_DB_REEXPERIMENT_REPORT.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/experiment_design/MODEL_RESULTS.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/experiment_design/MR_STGNN_RESULTS_2026-06-13.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/experiment_design/PAPER_ST_DOMAIN_CHANNEL_RESULT_2026-06-23.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/experiment_design/PAPER_ST_FEATURE_FAST_MPS_RESULT_2026-06-23.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/experiment_design/PAPER_ST_FEATURE_MATRIX_RUNBOOK_2026-06-23.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/experiment_design/PAPER_ST_FEATURE_SET_NOTE_2026-06-23.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/experiment_design/PRE_EXPERIMENT_DESIGN_LOCK_2026-06-22.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/experiment_design/README.md` | `ExperimentContractInterface` | `active` | package public interface |
| `docs/experiment_design/BATTLE_ROYALE_FRAMEWORK_VALIDATION_PLAN_2026-07-02.md` | `ExperimentContract` | `active` | Battle Royale framework validation plan |
| `docs/experiment_design/BATTLE_ROYALE_RESEARCH_EXECUTION_CHECKLIST_2026-07-03.md` | `ExperimentContract` | `active` | current Battle Royale research execution checklist |
| `docs/experiment_design/BATTLE_ROYALE_EXPERIMENT_EVIDENCE_LEDGER_2026-07-05.md` | `ExperimentResult` | `active` | detailed evidence ledger mapping experiments to supported and bounded claims |
| `docs/experiment_design/BATTLE_ROYALE_CLAIM_EXPERIMENT_MAPPING_2026-07-06.md` | `ExperimentSynthesis` | `active` | manuscript-facing RQ1-RQ4 claim-to-experiment/result mapping and remaining gap summary |
| `docs/experiment_design/BATTLE_ROYALE_EXPERIMENT_STATUS_AND_NEXT_DECISION_2026-07-06.md` | `ExperimentDecision` | `active` | current revised RQ1-RQ4 experiment status, model comparison conclusion, and board-baseline decision |
| `docs/experiment_design/BATTLE_ROYALE_RQ4_PRIOR_BASELINE_RESULT_2026-07-05.md` | `ExperimentResult` | `active` | revised-RQ3 Sagong-style/basic tabular baseline result; graph design removed from core |
| `docs/experiment_design/BATTLE_ROYALE_RQ4_DYNAMIC_PROBABILITY_FLUCTUATION_RESULT_2026-07-06.md` | `ExperimentResult` | `active` | revised-RQ3 dynamic probability trajectory result for realtime survival readouts |
| `docs/experiment_design/BATTLE_ROYALE_RQ4_PHASE_TRANSITION_RISK_ADVANTAGE_BOARD_RESULT_2026-07-06.md` | `ExperimentResult` | `active` | RQ4 phase-transition risk/advantage board result |
| `docs/experiment_design/BATTLE_ROYALE_RQ4_BOARD_BASELINE_COMPARISON_PROTOCOL_2026-07-06.md` | `ExperimentContract` | `active` | frozen RQ4 board-baseline comparison protocol |
| `docs/experiment_design/BATTLE_ROYALE_RQ4_BOARD_BASELINE_COMPARISON_RESULT_2026-07-06.md` | `ExperimentResult` | `active` | RQ4 board-baseline comparison result and claim boundary |
| `docs/experiment_design/BATTLE_ROYALE_RQ4_HIERARCHICAL_SURVIVAL_PROTOCOL_2026-07-05.md` | `ExperimentContract` | `active` | revised-RQ3 hierarchical / 10-second phase-survival model protocol and cumulative-survival bridge smoke status |
| `docs/experiment_design/BATTLE_ROYALE_REQUIRED_COHORT_SET_2026-07-03.md` | `ExperimentContract` | `active` | required cohort set for main training and cohort ablations |
| `docs/experiment_design/BATTLE_ROYALE_ONTOLOGY_VALIDATION_PROTOCOL_2026-07-03.md` | `ExperimentContract` | `active` | staged ontology validation protocol before predictive claims |
| `docs/experiment_design/BATTLE_ROYALE_FEATURE_AVAILABILITY_AUDIT_2026-07-03.md` | `ExperimentContract` | `active` | OM0-OM6 feature availability audit for required PUBG cohorts |
| `docs/experiment_design/BATTLE_ROYALE_OM_LADDER_RUNBOOK_2026-07-03.md` | `ExperimentRunbook` | `active` | runbook and artifact contract for cohort-separated OM0-OM6 runs |
| `docs/experiment_design/BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md` | `ExperimentResult` | `active` | completed required-cohort OM0-OM6 result, cohort-size, and feature-block audit |
| `docs/experiment_design/BATTLE_ROYALE_PHASE_STRATIFIED_OM_RESULT_2026-07-04.md` | `ExperimentResult` | `active` | phase-stratified required-cohort OM result and within-phase mechanism check |
| `docs/experiment_design/BATTLE_ROYALE_O1_SURVIVAL_CAPITAL_INTEGRITY_AUDIT_2026-07-04.md` | `ExperimentResult` | `active` | O1 survival-capital tensor integrity and univariate hazard-signal audit |
| `docs/experiment_design/BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md` | `ExperimentResult` | `active` | paired match-cluster bootstrap result for phase-group OM deltas |
| `docs/experiment_design/BATTLE_ROYALE_O6_Z_LADDER_SENSITIVITY_PROTOCOL_2026-07-04.md` | `ExperimentContract` | `active` | corrected O6 Z-ladder sensitivity protocol and pre-run gates |
| `docs/experiment_design/BATTLE_ROYALE_O6_Z_LADDER_FEATURE_AVAILABILITY_AUDIT_2026-07-04.md` | `ExperimentResult` | `active` | Z1-Z3 source/tensor availability audit before OM6Z training |
| `docs/experiment_design/BATTLE_ROYALE_O6_Z_LADDER_RESULT_2026-07-05.md` | `ExperimentResult` | `active` | corrected O6Z required-cohort sensitivity result and claim boundary |
| `docs/experiment_design/BATTLE_ROYALE_O7_EDGE_STREAM_NULL_GUARD_PROTOCOL_2026-07-05.md` | `ExperimentContract` | `active` | O7 advantage-edge stream 3-arm protocol and null-guard contract |
| `docs/experiment_design/BATTLE_ROYALE_GEN1_GEN2_RESULT_BOUNDARY_2026-07-05.md` | `ExperimentBoundary` | `active` | decision note separating legacy gen-1 pilots from current gen-2 manuscript evidence |
| `docs/experiment_design/BATTLE_ROYALE_O7_EDGE_SHUFFLE_AUDIT_2026-07-05.md` | `ExperimentResult` | `active` | O7 advantage-edge edge-shuffle audit and cohort-aware runner smoke result |
| `docs/experiment_design/BATTLE_ROYALE_O7_EARLY_READOUT_ERANGEL_2026-07-05.md` | `ExperimentResult` | `active` | first gen-2 O7 advantage-edge A/B/C early readout on Erangel TPP squad |
| `docs/experiment_design/BATTLE_ROYALE_O7_EARLY_READOUT_4COHORT_RESULT_2026-07-05.md` | `ExperimentResult` | `active` | four-cohort O7 advantage-edge early readout and expansion decision |
| `docs/experiment_design/BATTLE_ROYALE_O7_REQUIRED_COHORT_HAZARD_RESULT_2026-07-05.md` | `ExperimentResult` | `active` | 13-cohort O7 advantage-edge hazard-AUC result and claim boundary |
| `docs/experiment_design/BATTLE_ROYALE_O7_REQUIRED_COHORT_RANK_BRIDGE_RESULT_2026-07-05.md` | `ExperimentResult` | `active` | 13-cohort O7 hazard-implied rank bridge result and claim boundary |
| `docs/experiment_design/REPRESENTATION_ABLATION_STATUS_2026-06-14.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/experiment_design/REQUIRED_EXPERIMENTS.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/experiment_design/V3_FINAL_DESIGN_STATUS_2026-06-15.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/experiment_design/V3_LEAKAGE_AUDIT_2026-06-15.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/experiment_design/V3_PILOT_RESULTS_2026-06-15.md` | `ExperimentContract` | `active` | package documentation object |
| `docs/feature_engineering/FEATURE_CLOCK_CHANNEL_REDESIGN_2026-06-22.md` | `FeatureContract` | `active` | package documentation object |
| `docs/feature_engineering/BATTLE_ROYALE_FEATURE_ONTOLOGY_2026-07-02.md` | `FeatureContract` | `active` | genre-level Battle Royale feature ontology; O2 is contraction / elimination pressure |
| `docs/feature_engineering/PUBG_FEATURE_OPERATIONAL_MANIFEST_2026-07-02.md` | `FeatureContract` | `active` | PUBG-operational feature contract and ablation ladder |
| `docs/feature_engineering/PUBG_NETWORK_STRUCTURE_FEATURE_SPEC_2026-07-02.md` | `FeatureContract` | `active` | PUBG network-structure feature design for O7 relation features |
| `docs/feature_engineering/FEATURE_DESIGN_V2.md` | `FeatureContract` | `active` | package documentation object |
| `docs/feature_engineering/FEATURE_INVENTORY_AND_ABLATION_PLAN.md` | `FeatureContract` | `active` | package documentation object |
| `docs/feature_engineering/SPATIAL_TACTICAL_AFFORDANCE_FEATURE_SPEC_2026-07-02.md` | `FeatureContract` | `active` | spatial-tactical affordance feature spec |
| `docs/feature_engineering/NETWORK_ANALYSIS_FOUNDATION_CONCEPTS_2026-07-01.md` | `FeatureContract` | `active` | package documentation object |
| `docs/feature_engineering/README.md` | `FeatureContractInterface` | `active` | package public interface |
| `docs/feature_engineering/SKILL_RATING_PROXY_DESIGN_2026-06-22.md` | `FeatureContract` | `active` | package documentation object |
| `docs/feature_engineering/SPATIAL_EFFECT_FEATURE_DESIGN.md` | `FeatureContract` | `active` | package documentation object |
| `docs/feature_engineering/ST_FEATURE_BATTLEGROUND_MAPPING_2026-06-23.md` | `FeatureContract` | `active` | package documentation object |
| `docs/feature_engineering/ST_FEATURE_DEFINITION_2026-06-23.md` | `FeatureContract` | `active` | package documentation object |
| `docs/feature_engineering/ST_FEATURE_PRIOR_ALIGNED_DESIGN_2026-06-23.md` | `FeatureContract` | `active` | package documentation object |
| `docs/feature_engineering/TELEMETRY_DB_FEATURE_INVENTORY_2026-06-23.md` | `FeatureContract` | `active` | package documentation object |
| `docs/feature_engineering/TERRAIN_EXPERIENCE_FEATURE_DESIGN_2026-07-02.md` | `FeatureContract` | `active` | package documentation object |
| `docs/feature_engineering/TERRAIN_EXPERIENCE_RESEARCH_LINEAGE_2026-07-02.md` | `FeatureContract` | `active` | package documentation object |
| `docs/feature_engineering/TIER_PROXY_DESIGN_2026-06-22.md` | `FeatureContract` | `active` | package documentation object |
| `docs/handoffs/NEXT_CHAT_HANDOFF_2026-06-22.md` | `ContinuationState` | `handoff` | package documentation object |
| `docs/handoffs/NEXT_CHAT_HANDOFF_2026-07-01_CALDERA_ST_3D.md` | `ContinuationState` | `handoff` | package documentation object |
| `docs/handoffs/NEXT_CHAT_HANDOFF_2026-07-05_BATTLE_ROYALE_FRAMEWORK.md` | `ContinuationState` | `handoff` | current Battle Royale framework continuation state after OM6Z |
| `docs/handoffs/README.md` | `ContinuationStateInterface` | `active` | package public interface |
| `docs/model_design/BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md` | `ModelContract` | `active` | model-facing survival formula for the locked ontology |
| `docs/model_design/BATTLE_ROYALE_REALTIME_RISK_ADVANTAGE_FRAMEWORK_2026-07-06.md` | `ModelContract` | `active` | realtime elimination-risk prediction and phase-transition risk/advantage board framework |
| `docs/model_design/BATTLE_ROYALE_GRAPH_FORMALISM_2026-06-22.md` | `ModelContract` | `active` | package documentation object |
| `docs/model_design/BR_PARALLEL_MODEL_VISUALIZATION_ROADMAP.md` | `ModelContract` | `active` | package documentation object |
| `docs/model_design/DSACG_METHOD_SKELETON.md` | `ModelContract` | `active` | package documentation object |
| `docs/model_design/FINAL_EXPERIMENT_CODE_DESIGN.md` | `ModelContract` | `active` | package documentation object |
| `docs/model_design/FUTURE_DESIGN_3D_RELATIONAL_2026-06-21.md` | `ModelContract` | `active` | package documentation object |
| `docs/model_design/GRAPH_DATA_MODELING_FOUNDATION_2026-06-22.md` | `ModelContract` | `active` | package documentation object |
| `docs/model_design/INPUT_MODEL_CHANNEL_CONTRACT_2026-06-22.md` | `ModelContract` | `active` | package documentation object |
| `docs/model_design/MODEL_SPEC_CODE_GROUNDED_2026-06-21.md` | `ModelContract` | `active` | package documentation object |
| `docs/model_design/PUBG_GRAPH_MODELING_RESEARCH_FRAME_2026-06-22.md` | `ModelContract` | `active` | package documentation object |
| `docs/model_design/README.md` | `ModelContractInterface` | `active` | package public interface |
| `docs/model_design/SNAPSHOT_PHASE_SURVIVAL_DESIGN.md` | `ModelContract` | `active` | package documentation object |
| `docs/paper_writing/BATTLE_ROYALE_MANUSCRIPT_ASSEMBLY_PLAN_2026-07-05.md` | `ManuscriptSurface` | `active` | next-action plan for manuscript section drafts, figure/table storyboard, captions, and claim-boundary prose |
| `docs/paper_writing/BATTLE_ROYALE_RQ1_MANUSCRIPT_CLAIM_2026-07-06.md` | `ManuscriptSurface` | `active` | manuscript-facing RQ1 build-up from general genre definition to PUBG empirical instantiation |
| `docs/paper_writing/README.md` | `ManuscriptSurfaceInterface` | `active` | package public interface |
| `docs/research_framing/CALDERA_DATASET_STATUS_REDECISION_2026-07-02.md` | `ClaimBoundary` | `active` | Caldera label/zone/redeploy status decision |
| `docs/research_framing/BATTLE_ROYALE_PROBLEM_DEFINITION_2026-07-03.md` | `ClaimBoundary` | `active` | problem definition, motivation, and contribution boundary |
| `docs/research_framing/BATTLE_ROYALE_PAPER_ARGUMENT_CHAIN_2026-07-03.md` | `ClaimBoundary` | `active` | paper-level premise and argument order for the Battle Royale framework |
| `docs/research_framing/BATTLE_ROYALE_PROBLEM_METHOD_EXPERIMENT_STRUCTURE_2026-07-05.md` | `ClaimBoundary` | `active` | paper-writing rule from problem formalization to required methodology, experiments/results, and claim boundary |
| `docs/research_framing/BATTLE_ROYALE_MANUSCRIPT_CANONICAL_FLOW_2026-07-05.md` | `ClaimBoundary` | `active` | final manuscript flow from genre formalization to PUBG telemetry mapping, prior/tabular plus 10-second realtime survival model, and phase-transition risk/advantage board |
| `docs/research_framing/BATTLE_ROYALE_RESEARCH_QUESTION_CHAIN_2026-07-05.md` | `ClaimBoundary` | `active` | RQ1-RQ4 manuscript spine and auxiliary-experiment placement |
| `docs/research_framing/BATTLE_ROYALE_RQ_OBJECT_CHAIN_2026-07-06.md` | `ClaimBoundary` | `active` | object-oriented RQ1-RQ4 claim, experiment, evidence, result, and boundary chain |
| `docs/research_framing/BATTLE_ROYALE_RQ1_GENRE_FORMALIZATION_OBJECT_2026-07-06.md` | `ClaimBoundary` | `active` | RQ1 genre formalization claim object |
| `docs/research_framing/BATTLE_ROYALE_RQ2_PUBG_REPRESENTATION_OBJECT_2026-07-06.md` | `ClaimBoundary` | `active` | RQ2 PUBG official-telemetry representation claim object |
| `docs/research_framing/BATTLE_ROYALE_RQ3_REALTIME_SURVIVAL_READOUT_OBJECT_2026-07-06.md` | `ClaimBoundary` | `active` | RQ3 realtime survival readout claim object |
| `docs/research_framing/BATTLE_ROYALE_RQ4_RISK_ADVANTAGE_BOARD_OBJECT_2026-07-06.md` | `ClaimBoundary` | `active` | RQ4 phase-transition risk/advantage board claim object |
| `docs/research_framing/BATTLE_ROYALE_ANALYTIC_FRAMEWORK_NARRATIVE_FOR_MANUSCRIPT_2026-07-04.md` | `ClaimBoundary` | `active` | manuscript-facing analytic-framework narrative and reviewer-defense map |
| `docs/research_framing/BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md` | `ClaimBoundary` | `active` | locked O0-O8 ontology decision for the current prediction cycle |
| `docs/research_framing/BATTLE_ROYALE_ANALYTIC_FRAMEWORK_STATUS_2026-07-04.md` | `ClaimBoundary` | `active` | claim boundary for presenting an ontology-guided Battle Royale analytic framework |
| `docs/research_framing/BATTLE_ROYALE_ONTOLOGY_CORPUS_DESIGN_2026-07-02.md` | `ClaimBoundary` | `active` | confirmatory corpus design for Battle Royale ontology validation |
| `docs/research_framing/BATTLE_ROYALE_MINI_CORPUS_COVERAGE_TABLE_2026-07-04.md` | `ClaimBoundary` | `active` | source-grounded mini corpus coverage table for non-PUBG ontology support |
| `docs/research_framing/DESIGN.md` | `ClaimBoundary` | `active` | package documentation object |
| `docs/research_framing/DOCS_CLEANUP_AUDIT_2026-07-02.md` | `ClaimBoundary` | `active` | package documentation object |
| `docs/research_framing/FUTURE_PLAN.md` | `ClaimBoundary` | `active` | package documentation object |
| `docs/research_framing/PAPER_PREP_RESEARCH_SUMMARY.md` | `ClaimBoundary` | `active` | package documentation object |
| `docs/research_framing/PRIOR_WORK_SURVEY_2026-06-22.md` | `ClaimBoundary` | `active` | package documentation object |
| `docs/research_framing/PROJECT_FLOW_SUMMARY.md` | `ClaimBoundary` | `active` | package documentation object |
| `docs/research_framing/PUBG_DOMAIN_DESIGN_LAYER_2026-06-22.md` | `ClaimBoundary` | `active` | package documentation object |
| `docs/research_framing/README.md` | `ClaimBoundaryInterface` | `active` | package public interface |
| `docs/research_framing/RELATED_WORK_FRAME_2026-06-22.md` | `ClaimBoundary` | `active` | package documentation object |
| `docs/research_framing/RESEARCH_STATUS_2026-06-10.md` | `ClaimBoundary` | `active` | package documentation object |
| `docs/research_framing/TERRAIN_AFFORDANCE_CAPABILITY_BOUNDARY_2026-07-02.md` | `ClaimBoundary` | `active` | package documentation object |
| `docs/spatial_world_modeling/CALDERA_3D_VISUALIZATION_PLAN.md` | `WorldModel` | `active` | package documentation object |
| `docs/spatial_world_modeling/CALDERA_ST_3D_DESIGN_CONCEPTS_2026-07-01.md` | `WorldModel` | `active` | package documentation object |
| `docs/spatial_world_modeling/WARZONE_CALDERA_RQ3_3D_RENDER_EVIDENCE_PACKAGE_2026-07-06.md` | `WorldModel` | `active` | supplementary Warzone/Caldera 3D render evidence package, figure selection, captions, and claim boundary |
| `docs/spatial_world_modeling/CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_SCOPE_2026-07-05.md` | `WorldModel` | `active` | bounded Caldera O6 construct-validation artifact scope |
| `docs/spatial_world_modeling/CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_RESULT_2026-07-05.md` | `WorldModel` | `active` | completed Caldera O6 construct-validation artifact package and claim boundary |
| `docs/spatial_world_modeling/PUBG_WARZONE_OBSERVABILITY_GAP_BRIDGE_2026-07-05.md` | `WorldModel` | `active` | supplementary bridge from PUBG official telemetry gaps to Warzone / Caldera geometry representations |
| `docs/spatial_world_modeling/README.md` | `WorldModelInterface` | `active` | package public interface |

## Refactoring Rule

When a document changes class, move it to the package that owns that class and update package README references. Archive generated and legacy objects instead of mixing them with active design contracts.
