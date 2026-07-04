# Snapshot-First Redesign Audit (2026-06-22)

## Decision

기존 V3/MR 결과는 `legacy-v3-pooled-mr-phase-head-2026-06-22`로 보존한다. 새 canonical 설계는 `snapshot_phase_survival_v1`로 분리한다.

새 설계의 원칙은 명확하다.

> 모델 입력의 최소 시간 해상도는 10초 snapshot이다. phase는 입력을 뭉개는 단위가 아니라, 예측/평가 cut point이다.

## Target Contract

For each match:

- Input snapshots: `X_s[t, team, feature]`, 10초 간격의 팀 상태.
- Input snapshot graph: `A_s[t, relation, src_team, dst_team]`, 10초 간격의 relation graph.
- Phase index per snapshot: `phase_of_snap[t]`.
- Valid team mask: `node_mask_s[t, team]`.
- Phase cut `k`: phase `k`까지 관측된 snapshot만 사용한다.
- Output: `hazard[k, team]`, phase `k`에서 다음 phase까지 탈락할 위험 또는 다음 phase 생존확률로 변환 가능한 logit.
- Survival probability: `S[k, team] = prod_{i <= k} (1 - hazard[i, team])`.

이 계약에서는 `phase_feature_long` 같은 phase aggregate feature가 primary input이면 안 된다. 필요하다면 auxiliary/context ablation으로만 별도 표시한다.

## Current New Code Status

| File | Status | Notes |
| --- | --- | --- |
| `model_stgcn/snapshot_phase_survival.py` | new target seed | 10초 snapshot encoder 후 phase cut별 representation을 gather해서 hazard 예측 |
| `model_stgcn/snapshot_phase_dataset.py` | new target seed | snapshot tensor를 primary input으로 사용하되 label/mask는 현재 phase fetch를 재사용 |
| `model_stgcn/train_snapshot_phase.py` | new target seed | corrected contract의 첫 training entrypoint |
| `tests/test_snapshot_phase_survival.py` | new target seed | phase cut causality, forward shape, masked loss smoke |
| `docs/SNAPSHOT_PHASE_SURVIVAL_DESIGN.md` | new target seed | corrected architecture contract 초안 |

현재 초안은 방향은 맞지만, 전체 실험 체계에 아직 연결되지 않았다. 따라서 아직 새 결과 산출용 canonical pipeline으로 완성된 상태는 아니다.

File-by-file classification is tracked in `docs/SNAPSHOT_FIRST_CODE_INVENTORY_2026-06-22.md`.

## System-Wide Audit

| Area | Files | Current role | Redesign action |
| --- | --- | --- | --- |
| Phase data fetch | `stdb/queries.py` | phase features, phase graph, labels를 함께 반환 | label/mask only fetch를 분리한다. 새 snapshot model이 phase features를 의도치 않게 main input으로 사용하지 않게 한다. |
| Snapshot data fetch | `stdb/snapshot_queries.py` | `X_s`, `A_s`, `phase_of_snap`, `node_mask_s` 제공 | canonical input source로 승격한다. relation mode와 missing snapshot 정책을 문서화한다. |
| Feature ETL | `stdb/rq1_feature_etl.py`, `stdb/f0_feature_etl.py`, `stdb/network_feature_etl.py`, `stdb/habitat_feature_etl.py`, `stdb/v2_view.py` | phase-row engineered features 생성 | legacy phase baselines로 유지한다. snapshot-first에서는 causal cut context 여부를 별도 ablation으로만 연결한다. |
| Phase graph ETL | `stdb/phase_edge_etl.py` | phase graph relation 생성 | legacy V3용으로 유지한다. snapshot graph relation과 혼동하지 않게 이름/문서 분리한다. |
| Legacy phase model | `model_stgcn/train_phase.py`, `model_stgcn/dyn_stgnn.py` | phase-resolution baseline | baseline으로 유지하되 새 main으로 홍보하지 않는다. |
| Legacy MR model | `model_stgcn/train_mr.py`, `model_stgcn/mr_stgnn.py`, `model_stgcn/eval_mr.py` | 10초 snapshot encoder 후 phase pooling | `legacy_mr_pooled` baseline으로 rename 또는 matrix alias를 둔다. |
| V3 package | `model_stgcn/v3/*` | factorized pooled-MR phase-head matrix | legacy V3로 보존한다. 새 snapshot-first runner와 matrix를 별도 파일로 만들거나 명시적으로 `snapshot_cut` route를 추가한다. |
| Feature ablation | `model_stgcn/feature_ablation.py`, `model_stgcn/feature_manifests/*` | tabular phase feature A0-A11 | phase feature-only baseline으로 유지한다. snapshot-first ablation 결과와 같은 표에 넣을 때는 family를 분리한다. |
| Edge controls | `model_stgcn/v3/edge_controls.py`, `model_stgcn/edge_sanity.py` | phase graph shuffle/control | snapshot-level control module이 새로 필요하다. |
| Evaluation | `model_stgcn/v3/eval.py`, `model_stgcn/eval_mr.py` | legacy output schema | `eval_snapshot_phase.py`가 필요하다. prediction CSV, calibration, phase-band metrics를 새 schema에 맞춘다. |
| Scripts | `scripts/run_v3_*.sh`, `scripts/run_mr_*.sh` | legacy result production | overwrite 금지. 새 scripts는 `run_snapshot_phase_*` 이름으로 만든다. |
| Presentation docs | `docs/deliverables/MODEL_ARCHITECTURE_DESIGN_SPEC_2026-06-22.md` | current V3 pooled-MR 설명 | 오늘 발표용 current architecture로는 정확하다. 새 canonical 설계 설명으로 재사용하면 안 된다. |

## Required Code Changes

1. Label-only phase API

Create a function such as `fetch_phase_labels(match_id)` that returns only:

- phase labels or hazard targets
- `phase_mask`
- `node_mask`
- optional rank/top-k labels

This removes accidental dependency on phase aggregate features in the snapshot-first dataset.

2. Snapshot-first evaluator

Add `model_stgcn/eval_snapshot_phase.py` with the same metric family as V3:

- AUC, AP, BCE, Brier, ECE
- phase-band metrics
- top-k metrics if enabled
- prediction export with `match_id`, `phase_idx`, `team_id`, `y`, `p`

3. Snapshot graph controls

Add snapshot-level controls over `A_s`, not phase `A`:

- `observed`
- `no_graph`
- `node_shuffle`
- `edge_shuffle`
- `relation_drop: proximity/combat/copressure`
- optional `time_shuffle` for temporal dynamics

4. Snapshot-first experiment runner

Either add a separate runner:

- `model_stgcn/snapshot_experiment_matrix.py`
- `model_stgcn/run_snapshot_matrix.py`

or extend V3 runner with an explicit `model=snapshot_cut` route. Separate files are safer because the legacy V3 package already encodes pooled-MR assumptions.

5. Model extension points

The initial `SnapshotPhaseSurvivalGNN` should be expanded only where it answers a concrete ablation:

- temporal-only encoder
- snapshot graph encoder variants
- last-snapshot-only baseline
- optional global cut context
- optional rank/top-k decoder

6. Result directories

New outputs must not reuse legacy directories. Use:

- `model_stgcn/checkpoints/snapshot_phase_survival_v1_smoke`
- `model_stgcn/checkpoints/snapshot_phase_survival_v1_pilot`
- `model_stgcn/checkpoints/snapshot_phase_survival_v1_full`
- `model_stgcn/checkpoints/snapshot_phase_survival_v1_controls`

## Redesigned Ablation Matrix

| Ablation block | Rows | Purpose |
| --- | --- | --- |
| E0 legacy bridge | `phase_tcn`, `legacy_mr_pooled`, `snapshot_cut` | old results와 새 결과를 같은 평가 split에서 연결 |
| E1 temporal resolution | phase-only, pooled-MR, snapshot-cut, last-snapshot-only, time-shuffle, within-phase-shuffle | 10초 동역학이 실제로 쓰이는지 검증 |
| E2 snapshot graph | no graph, proximity only, combat only, copressure only, full, node shuffle, edge shuffle | graph relation의 효과를 snapshot graph 기준으로 재검증 |
| E3 feature/context | snapshot raw 13, +global cut context, +causal engineered context, +phase aggregate auxiliary | phase aggregate가 main input으로 새는지 통제 |
| E4 model family | temporal-only, snapshot RGCN+TCN, snapshot GAT/SAGE variants, optional GRU/Transformer | architecture gain과 data-contract gain을 분리 |
| E5 decoder | hazard-only, hazard+rank, hazard+top-k | prediction target 확장의 효과 검증 |
| E6 robustness | seed repeat, map/mode split, calibration, phase-band | 발표/논문용 안정성 검증 |

## Interpretation Reset

Old interpretation:

- "MR improves phase-level survival prediction after encoding 10초 snapshots and pooling to phase."

New interpretation target:

- "A model that continuously ingests 10초 snapshots can estimate phase cut survival without collapsing the input stream into phase aggregate features."

The old result is not useless. It becomes the bridge baseline showing that high-resolution observations help. The new result must answer the stronger question: whether preserving snapshot-level dynamics up to each phase cut improves survival prediction and graph reasoning.

## Migration Plan

1. Freeze legacy

- Keep `docs/LEGACY_CODE_RESULT_MAP_2026-06-22.md`.
- Do not overwrite legacy checkpoint/result directories.
- In future tables, label old rows as `legacy_phase` or `legacy_mr_pooled`.

2. Complete canonical data/eval path

- Add label-only phase API.
- Remove unnecessary phase feature loading from snapshot dataset.
- Add snapshot-first evaluator and prediction export.

3. Add controls and matrix

- Implement snapshot graph controls over `A_s`.
- Add temporal controls.
- Add `snapshot_phase_survival_v1` matrix runner.

4. Re-run in increasing cost

- smoke: import, one batch, tiny train
- pilot: small split, all matrix rows
- full: frozen split, seed repeat, calibration

5. Update docs/results

- New architecture design spec should describe `snapshot_phase_survival_v1`, not V3 pooled-MR.
- Legacy docs remain valid only for old result interpretation.
- Final tables should include a contract column: `phase`, `legacy_mr_pooled`, `snapshot_cut`.

## Immediate Priority

The next code work should be:

1. `fetch_phase_labels()` or equivalent label-only API.
2. `eval_snapshot_phase.py`.
3. `snapshot_edge_controls.py`.
4. `snapshot_phase_experiment_matrix.py` plus run scripts.

Only after these are in place should full reruns begin.
