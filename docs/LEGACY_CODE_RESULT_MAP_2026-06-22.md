# Legacy Code-Result Map (2026-06-22)

## Purpose

이 문서는 새 snapshot-first 산출물이 나오기 전까지, 현재 코드와 현재 결과를 하나의 legacy 묶음으로 고정하기 위한 북마크이다.

핵심은 다음 한 문장이다.

> 이 결과들은 strict 10초 snapshot-resolution survival model의 결과가 아니라, phase-level head를 가진 phase / pooled-MR / V3 계열 결과이다.

## Bookmark Metadata

- Legacy label: `legacy-v3-pooled-mr-phase-head-2026-06-22`
- 작성일: 2026-06-22
- Git HEAD at bookmark time: `93cfda996f30d938800340999b43c460c0e02e25`
- Worktree status at bookmark time: dirty. 따라서 HEAD만으로는 완전한 frozen state가 아니며, 아래 checksum과 result path를 함께 보아야 한다.
- 새 canonical 설계명: `snapshot_phase_survival_v1`
- 새 canonical 코드 초안: `model_stgcn/snapshot_phase_survival.py`, `model_stgcn/snapshot_phase_dataset.py`, `model_stgcn/train_snapshot_phase.py`
- Detailed code-result registry: `docs/LEGACY_RESULT_CODE_REGISTRY_2026-06-22.md`
- Machine-readable registry: `docs/legacy_result_code_registry_2026-06-22.json`

## Legacy Model Contracts

| Legacy family | Input contract | Prediction contract | Correct interpretation |
| --- | --- | --- | --- |
| `phase` | `fetch_phase_match_tensors()`에서 만든 `(phase, team, phase_feature)` 입력 | phase별 next-phase elimination hazard | phase aggregate feature model |
| `mr` | 10초 snapshot encoder를 거친 뒤 phase로 pooling | phase별 next-phase elimination hazard | snapshot-informed phase model |
| `v3` MR | V3 factorized model 안에서 snapshot tier를 phase로 요약하고 phase tier에서 graph/global/head 수행 | phase별 survival/rank/top-k metrics | pooled-MR phase-head model |
| feature A0-A11 | `(match_id, team_id, phase_idx)` 단위의 tabular phase features | phase-row binary classifier | feature-only phase baseline |
| graph relation/control | phase graph tensor `A` 또는 V3 phase graph relation | phase-tier graph ablation | snapshot graph ablation이 아님 |

## Result Mapping

For row-level traceability, use `matrix_metrics.csv` as the primary key:

```text
matrix_metrics.csv row
  -> checkpoint column
  -> eval_json column
  -> model_stgcn/v3/experiment_matrix.py row definition
  -> model_stgcn/v3/run_matrix.py
  -> model_stgcn/v3/train.py and model_stgcn/v3/eval.py
```

The detailed family-by-family registry is in
`docs/LEGACY_RESULT_CODE_REGISTRY_2026-06-22.md`.

| Result artifact | Code path | Contract | Representative numbers | Allowed claim | Do not claim |
| --- | --- | --- | --- | --- | --- |
| `model_stgcn/checkpoints/v3_full_main_causal/matrix_metrics.csv` | `model_stgcn.v3.run_matrix` -> `model_stgcn.v3.train/eval` | V3 pooled-MR phase head | `mr_gat` AUC 0.8496, `mr_rgcn_film` AUC 0.8465, `mr_farnet` AUC 0.8454 | Factorized pooled-MR V3 improves over simple phase baselines | Strict 10초 snapshot-resolution ST survival model |
| `model_stgcn/checkpoints/v3_full_controls/matrix_metrics.csv` | V3 control matrix | phase vs pooled-MR resolution ablation | `phase_tcn` AUC 0.7872, `phase_rgcn` AUC 0.7926, `mr_tcn` AUC 0.8349, `mr_rgcn` AUC 0.8467 | 10초 snapshot information helps before phase-level prediction | phase output head가 없거나 모든 10초 step에서 직접 survival을 예측했다 |
| `model_stgcn/checkpoints/v3_full_graph_relation/matrix_metrics.csv` | V3 graph relation matrix | phase-tier graph relation/control | `full_observed` AUC 0.8453, `full_edge_shuffle` AUC 0.8469 | current phase graph relation does not clearly beat shuffled edge control in this run | snapshot graph relation has no value |
| `model_stgcn/checkpoints/v3_full_feature_causal/matrix_metrics.csv` | V3 feature preset matrix | pooled-MR with phase feature presets | feature preset ablation under V3 phase-head contract | V3 feature family contribution under pooled-MR | snapshot-first feature ablation |
| `model_stgcn/checkpoints/feature_ablation_full_db_manifest_v2/feature_ablation_metrics.csv` | `model_stgcn/feature_ablation.py` | HGB over phase feature rows | A0 raw AUC 0.7589, A7 raw+habitat AUC 0.7828, A11 full AUC 0.7838 | tabular phase feature family baseline | neural snapshot temporal dynamics |
| `model_stgcn/checkpoints/mr_full_squad*` and MR reports | `model_stgcn/train_mr.py`, `eval_mr.py` | legacy MR, snapshot encoder then phase pooling | documented as MR mean AUC around 0.84 in existing reports | pooled-MR is stronger than phase-only | strict snapshot cut survival |
| `model_stgcn/checkpoints/adv_*` and `docs/FUTURE_DESIGN_3D_RELATIONAL_2026-06-21.md` | advantage edge scripts and preliminary model paths | future/pilot graph design | preliminary feasibility artifacts | future relation-design direction | final canonical snapshot-first result |

## Core Code Registry

These checksums identify the legacy code/result files used by this bookmark. If any file changes, treat the bookmark as a historical mapping, not as a fresh reproducibility guarantee.

| Path | SHA-256 | Role |
| --- | --- | --- |
| `stdb/queries.py` | `f6b5ee3262cd01eb73bdad9fa714b7039083a280fdc3fc6fdf21e829f8899715` | phase tensor and labels |
| `stdb/snapshot_queries.py` | `2e85babec0057988b3b98b9e5faa08968b2db50cc11ccaef32c61f9acefb55cb` | 10초 snapshot tensor source |
| `model_stgcn/train_phase.py` | `000c6dffc0e3034841429b4502ad7c0d6aa048c3cdd0a782a6b8a233cd23d1b7` | legacy phase training |
| `model_stgcn/train_mr.py` | `700a0574b316584e673458aea67734389bd56caf16671ae344e90c84828db517` | legacy MR training |
| `model_stgcn/mr_stgnn.py` | `1635f121d112a9415b4e6964ef33028440be452a1568ec6ae72a7fc6292b722e` | legacy MR architecture |
| `model_stgcn/v3/train.py` | `1c2f341591ebd70991fff6ef13759d700de5f2c3de3182e4642b609c27e97286` | V3 trainer |
| `model_stgcn/v3/eval.py` | `0f91ebc22eabf4af7829539e53ed5d73159b5a63f2ad1bd5333b036c9313abdc` | V3 evaluator |
| `model_stgcn/v3/experiment_matrix.py` | `be830298019bdaf65d468e2b14fa8a52b0870d3f1da046e7fdbf5212ad2f1c30` | V3 matrix definitions |
| `model_stgcn/v3/run_matrix.py` | `20cae4d56b726334c3df689e94a1630a562231312aebdcd1ed5f9e18742fb52a` | V3 matrix runner |
| `model_stgcn/v3/edge_controls.py` | `6fd7c61a621f172137970004b02df468901e72a1f051871204b07ab50f9ea44e` | phase graph controls |
| `model_stgcn/feature_ablation.py` | `9cb2a5e5ade17007d2b67276704c66dc1b03e04d65b3cff537367738481d09a4` | tabular phase feature ablation |

## Result File Registry

| Path | SHA-256 | Interpretation |
| --- | --- | --- |
| `model_stgcn/checkpoints/v3_full_main_causal/matrix_metrics.csv` | `d7542a1bb329f02b7e90b3fdece847ee9a87f2a50a49d6dc4edbe3110102d4ab` | legacy V3 main matrix |
| `model_stgcn/checkpoints/v3_full_controls/matrix_metrics.csv` | `01b0329bbda431c3467b7bd8c1334f38369d55adf276454c171e947e0421a5b2` | legacy V3 resolution/decoder/control matrix |
| `model_stgcn/checkpoints/v3_full_graph_relation/matrix_metrics.csv` | `c03d640d92a8cb8734a29cf61d74d8cb0456a06d2db37a36e6d4525da3f60cc0` | legacy phase graph relation matrix |
| `model_stgcn/checkpoints/v3_full_feature_causal/matrix_metrics.csv` | `41d6e9af7a961a5aafe86ac19451aff0f4ff4e385709eb1e664f9e72b3247b8a` | legacy V3 feature preset matrix |
| `model_stgcn/checkpoints/feature_ablation_full_db_manifest_v2/feature_ablation_metrics.csv` | `3f3a8d70835a160debb10d82e59d71ddff332a93a674f2145bcff1f00d920b24` | legacy phase tabular feature ablation |

## Presentation Guardrails

Use this wording:

- "The current implemented V3 model is a factorized pooled-MR phase-head model: 10-second snapshots are encoded first, then summarized to phase-level decision points."
- "The reported improvement shows that high-resolution temporal observations improve phase-level survival prediction."
- "The graph relation ablation is about the current phase-tier relation tensor, not about all possible snapshot graph dynamics."

Avoid this wording:

- "The current result is a strict 10-second snapshot-resolution survival model."
- "The model predicts survival at every 10-second tick."
- "Graph dynamics are useless." The current evidence only says the implemented phase-tier graph relation did not robustly beat controls.

## Legacy Freeze Rule

Do not overwrite these legacy result directories when running the redesigned model. New runs should use separate output roots, for example:

- `model_stgcn/checkpoints/snapshot_phase_survival_v1_smoke`
- `model_stgcn/checkpoints/snapshot_phase_survival_v1_pilot`
- `model_stgcn/checkpoints/snapshot_phase_survival_v1_full`
