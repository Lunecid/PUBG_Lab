# V3 Final Experiment Design Status

작성일: 2026-06-15

이 문서는 현재까지 합의한 최종 실험 설계 상태를 정리한다. 실험 결과 수치는 의도적으로 제외한다. 결과 해석은 별도 결과 문서와 최종 논문 표에서 다룬다.

## 1. 연구 방향

이 프로젝트의 중심은 단일 모델 성능 경쟁이 아니다. 동일한 PUBG telemetry를 어떤 표현 단위로 재구성할 때 생존 과정을 가장 잘 보존하는지 검증하는 representation-first 연구다.

핵심 명제는 다음과 같다.

```text
PUBG 팀 생존 위험과 최종 순위 구조는 flat/raw aggregate보다,
team/node state, global phase context, dynamic team graph, snapshot temporal signal을
분리한 factorized multi-resolution representation에서 더 명확하게 보존된다.
```

따라서 최종 모델은 목적 그 자체가 아니라, 이 표현 설계를 검증하기 위한 구현체다.

## 2. 현재 코드 구조

새 최종 실험 코드는 기존 legacy 결과를 덮어쓰지 않고 `model_stgcn/v3/`에 병렬로 추가했다.

| 파일 | 역할 |
| --- | --- |
| `spec.py` | phase feature를 node/global 역할로 분리 |
| `factorized_data.py` | 기존 STDB tensor를 `X_node`, `G`, `A`로 변환 |
| `models.py` | factorized phase/MR GNN 모델 정의 |
| `train.py` | phase 또는 MR 모델 학습 |
| `eval.py` | hazard, causal rank/top-k forecast, calibration 평가 |
| `experiment_matrix.py` | 최종 실험 matrix 선언 |
| `run_matrix.py` | 동일 split으로 matrix train/eval 실행 |

관련 문서와 스크립트:

| 파일 | 역할 |
| --- | --- |
| `docs/FINAL_EXPERIMENT_CODE_DESIGN.md` | 코드 설계 원칙 |
| `docs/V3_FINAL_DESIGN_STATUS_2026-06-15.md` | 현재 설계 상태 요약 |
| `docs/V3_LEAKAGE_AUDIT_2026-06-15.md` | leakage audit와 causal rank/top-k 수정 기록 |
| `scripts/run_v3_pilot_matrix.sh` | pilot main matrix 실행 |
| `scripts/run_v3_full_main_matrix.sh` | full main architecture matrix 실행 |
| `scripts/run_v3_full_controls_matrix.sh` | full control matrix 실행 |
| `tests/test_v3_factorized.py` | V3 feature split/model forward smoke tests |

## 3. 입력 표현 설계

기존 legacy phase tensor는 다음 형태였다.

```text
X_legacy[p, i] = [global phase context broadcast to team i ; team/node features]
```

V3에서는 이를 명시적으로 분리한다.

```text
X_node[p, i] = team i의 phase p 상태
G[p]         = phase p의 match/global context
A[p]         = phase p의 team-team relation graph
X_s[t, i]    = 10초 snapshot에서 team i 상태
A_s[t]       = snapshot-level team graph
```

이 분리는 논문 방어에 중요하다. safe-zone radius, alive teams, phase duration 같은 값은 특정 팀의 속성이 아니라 match/phase context이므로 node feature에 broadcast하는 것보다 별도 global input으로 주입하는 것이 더 엄밀하다.

## 4. Feature Role

### 4.1 Node Features

Node feature는 팀 또는 팀-페이즈에 귀속되는 값이다.

주요 범주:

| 범주 | 예시 |
| --- | --- |
| official/team raw | team size, alive players, combat/loot aggregate |
| STDB team state | health, spread, vehicle ratio, movement state |
| trajectory | path length, displacement, speed, sinuosity, safe alignment |
| exposure | distance to boundary, outside-safe ratio, enemy density |
| combat | damage dealt/taken, kills, groggy, distinct opponents |
| loot/readiness | pickup/equip/use, weapon/ammo/armor/heal/boost counts |
| network role | combat degree/strength, proximity degree, reciprocity |
| local habitat pressure | team-level zone deficit, local enemy pressure, accessible pressure |

### 4.2 Global Features

Global feature는 특정 팀이 아니라 phase 또는 match 환경에 귀속되는 값이다.

주요 범주:

| 범주 | 예시 |
| --- | --- |
| phase context | phase index, phase duration |
| survival context | alive teams, alive players |
| zone context | safe radius, poison radius, zone gap, safe center shift |
| global habitat pressure | habitat loss, shrink rate, safe area per alive team, crowding pressure |

### 4.3 Graph Features

Graph는 feature로 요약하지 않고 adjacency로 직접 입력되는 관계 표현이다.

현재 구현된 relation channel:

| Channel | 의미 |
| --- | --- |
| `proximity` | phase 내 가까운 팀 관계 |
| `combat` | phase 내 교전/damage 관계 |
| `copressure` | 같은 zone/habitat pressure를 공유하는 관계 |

향후 추가해야 할 control:

| Control | 목적 |
| --- | --- |
| edge-shuffled graph | 실제 연결 구조의 의미 검증 |
| degree-preserving shuffle | degree 효과 통제 |
| density-matched random graph | edge 수 효과 통제 |
| relation-label shuffle | relation type 의미 검증 |
| time-shuffled snapshots | MR temporal order 검증 |

### 4.4 Snapshot Features

Snapshot tier는 phase 내부의 10초 단위 변화를 요약한다.

역할:

| 요소 | 의미 |
| --- | --- |
| `X_s` | 10초 단위 팀 위치/위험/상태 |
| `A_s` | snapshot-level 관계 그래프 |
| snapshot encoder | 짧은 시간 해상도 움직임을 phase embedding으로 압축 |
| snapshot survival head | near-future death 보조 supervision |

## 5. Label and Mask Protocol

주요 endpoint는 next-phase elimination hazard다.

```text
y_hazard[i, p] = 1 if team i is alive at phase p and eliminated before phase p+1
               = 0 if team i is alive at phase p and survives to phase p+1
```

중요한 mask:

| Mask | 역할 |
| --- | --- |
| `target_mask` | hazard loss/eval 대상 risk cell |
| `team_mask` | match에 실제 존재하는 team |
| `node_mask` | phase/team 단위 유효 node |
| `phase_mask` | 유효 phase |
| `node_mask_s` | snapshot/team 단위 유효 node |

평가 원칙:

| Task | Evaluation Unit |
| --- | --- |
| hazard | at-risk team-phase cell |
| rank | at-risk team-phase cell; cut phase별 final-rank forecast |
| top-k | at-risk team-phase cell; cut phase별 final top-k forecast |
| calibration | hazard probability over same risk cells |

이미 탈락한 팀이나 target이 정의되지 않은 phase를 hazard 평가에 섞으면 안 된다.

### 5.1 Phase-Wise / Live Forecast Output

모델은 최종 match 하나당 하나의 값만 내는 구조가 아니다. 각 match, cut phase, team에 대해 다음 출력을 만든다.

```text
hazard_prob[m, p, i] = cut phase p까지 관측했을 때 team i가 다음 phase에 탈락할 확률
top5_prob[m, p, i]   = cut phase p까지 관측했을 때 team i가 최종 top5가 될 확률
top10_prob[m, p, i]  = cut phase p까지 관측했을 때 team i가 최종 top10이 될 확률
rank_pred[m, p, i]   = cut phase p까지 관측했을 때 team i의 최종 rank-normalized forecast
```

따라서 live service 또는 replay 분석에서는 같은 팀의 확률이 phase 진행에 따라 바뀌는 trajectory로 표시되어야 한다.

현재 V3 evaluator는 두 가지 산출물을 지원한다.

| Output | 역할 |
| --- | --- |
| `phase_metrics` in eval JSON | cut phase별 hazard/top5/top10/rank metric |
| `--pred-out` CSV | `match_id`, `team_id`, `cut_phase`, 예측 확률을 포함한 live trajectory row export |

전체 AUC/Spearman은 모든 at-risk team-phase cell을 합친 요약값일 뿐이다. top10처럼 phase가 진행될수록 양성 비율이 급격히 높아지는 task는 반드시 phase별 pos_rate와 함께 해석해야 한다.

## 6. Model Architecture

### 6.1 Phase Model

Phase-only factorized model:

```text
X_node -> node encoder
G      -> global encoder
A      -> graph encoder
fused hidden -> causal temporal encoder -> task heads
```

지원 graph encoder:

| Encoder | 의미 |
| --- | --- |
| `none` | graph 없는 neural baseline |
| `gcn` | basic graph convolution |
| `sage` | GraphSAGE-style aggregation |
| `gat` | graph attention |
| `rgcn` | relation-aware graph convolution |

지원 global fusion:

| Fusion | 의미 |
| --- | --- |
| `none` | global context 제거 |
| `concat` | node hidden과 global hidden concat |
| `add` | projected global context additive fusion |
| `film` | global context로 node hidden을 scale/shift |

지원 decoder:

| Decoder | 의미 |
| --- | --- |
| `mlp` | per-team output heads |
| `attention` | team set self-attention 후 rank/top-k head |

### 6.2 MR Model

MR model은 phase model 앞에 snapshot encoder를 붙인다.

```text
X_s, A_s -> snapshot encoder -> phase-level pooled embedding
phase pooled snapshot embedding + X_node -> phase model
```

MR은 hazard만을 위한 장치가 아니다. phase 내부 시간 변화가 final rank/top-k 구조를 더 잘 보존하는지 같이 검증해야 한다.

## 7. Multi-Task Loss

현재 V3 loss:

```text
L = alpha * hazard_BCE over target_mask
  + beta  * causal rank_MAE over target_mask
  + gamma * causal topk_BCE over target_mask
  + delta * snapshot_survival_BCE
  + eta   * causal pairwise_rank_loss over target_mask
```

역할:

| Loss | 역할 |
| --- | --- |
| hazard BCE | main next-phase survival endpoint |
| rank MAE | cut phase별 final rank forecast auxiliary |
| top-k BCE | cut phase별 final bracket forecast auxiliary |
| snapshot survival BCE | MR deep supervision |
| pairwise rank loss | team ordering auxiliary |

논문에서는 hazard가 main task이고, rank/top-k는 downstream 또는 auxiliary task로 정리하는 것이 안전하다.

## 8. Experiment Matrix

### 8.1 Main Architecture Matrix

모든 main architecture는 MR input을 기준으로 비교한다.

| ID | 목적 |
| --- | --- |
| `mr_tcn` | MR input, graph 없음 |
| `mr_gcn` | basic graph convolution |
| `mr_sage` | GraphSAGE aggregation |
| `mr_gat` | graph attention |
| `mr_rgcn` | relation-aware graph convolution |
| `mr_rgcn_film` | global FiLM conditioning |
| `mr_rgcn_attn_rank` | attention rank decoder |
| `mr_farnet` | MR + R-GCN + FiLM + AttnRank final candidate |

### 8.2 Resolution Ablation

MR이 실제로 필요한지 확인한다.

| ID | 목적 |
| --- | --- |
| `phase_tcn` | phase-only, graph 없음 |
| `phase_rgcn` | phase-only, R-GCN |
| `mr_tcn` | MR, graph 없음 |
| `mr_rgcn` | MR, R-GCN |

### 8.3 Decoder Ablation

rank/top-k decoder와 pairwise rank loss의 효과를 분리한다.

| ID | 목적 |
| --- | --- |
| `mr_rgcn_hazard_rank` | 기본 hazard/rank/top-k heads |
| `mr_rgcn_pairwise_rank` | pairwise rank loss 추가 |
| `mr_rgcn_attn_rank` | team-set attention decoder 추가 |

### 8.4 Graph-Control Ablation

graph relation 자체의 의미를 검증한다.

| ID | 목적 |
| --- | --- |
| `mr_rgcn_no_graph` | graph 없음 |
| `mr_rgcn_proximity` | proximity channel only |
| `mr_rgcn_combat_only` | combat channel only |
| `mr_rgcn_copressure_only` | copressure channel only |
| `mr_rgcn_full_observed` | observed relation graph 전체 |

아직 추가해야 할 stronger graph controls:

| Control | 필요성 |
| --- | --- |
| shuffled edge | observed structure 검증 |
| density-matched random edge | edge density 효과 통제 |
| degree-preserving shuffle | node degree 효과 통제 |

## 9. Baseline and Representation Ablation

논문 방어를 위해 neural/GNN만 비교하면 부족하다. 같은 split과 같은 risk cells에서 강한 tabular baseline이 필요하다.

필수 baseline:

| Baseline | 목적 |
| --- | --- |
| logistic regression | simple linear reference |
| HGB/LightGBM/XGBoost | strong tabular reference |
| phase TCN | non-graph neural phase baseline |
| MR TCN | non-graph MR baseline |

Representation ablation은 다음 층으로 정리한다.

| Family | 의미 |
| --- | --- |
| official raw | 공식/raw aggregate baseline |
| node only | team state만 사용 |
| node + global | global context 분리 주입 |
| node + global + STDB | 시공간 노출/이동 표현 |
| node + global + network | network role 표현 |
| node + global + habitat | shrinking-space pressure 표현 |
| full feature | 모든 feature family |
| full + graph | observed graph 추가 |
| full + graph + MR | 최종 factorized MR |

## 10. 현재 구현 상태

완료:

| 항목 | 상태 |
| --- | --- |
| node/global feature split | 구현 |
| factorized phase dataset | 구현 |
| factorized MR dataset | 구현 |
| graph encoder variants | 구현 |
| global fusion variants | 구현 |
| attention rank decoder | 구현 |
| pairwise rank loss | 구현 |
| matrix runner | 구현 |
| hazard/rank/top-k/calibration evaluator | 구현 |
| pilot/full run scripts | 구현 |
| manual V3 smoke tests | 구현 |
| train progress logging | 구현 |
| lazy DB loader workers | 구현 |
| direct forbidden feature guard | 구현 |
| causal rank/top-k head | 구현 |
| phase-wise metrics | 구현 |
| live prediction CSV export | 구현 |

미완료 또는 추가 필요:

| 항목 | 필요 작업 |
| --- | --- |
| shuffled/random graph controls | graph structure 검증 강화 |
| time-shuffled snapshot control | MR temporal order 검증 |
| alive-only rank evaluation | phase별 생존 팀 rank 평가 |
| strong tabular baseline shared split | HGB/LightGBM/XGBoost 재정렬 |
| feature group importance | representation family 해석 |
| seed repeats | 통계적 안정성 |
| full-domain/domain-wise runs | map/mode/domain robustness |
| habitat graph extension | latent competition graph 추가 |

## 11. Full Run Plan

현재 full run은 다음 순서로 진행한다.

1. `v3_full_main`
   - main architecture matrix
   - MR input 기준 graph encoder/global fusion/decoder 비교

2. `v3_full_controls`
   - resolution ablation
   - decoder ablation
   - graph-control ablation

3. shared split 기반 tabular baseline
   - official raw
   - manifest feature families
   - full feature set

4. robustness/control
   - seed repeats
   - graph shuffle/random controls
   - phase-wise evaluation
   - calibration and feature importance

## 12. 논문 서사

최종 논문 서사는 다음 형태가 가장 안전하다.

```text
Battle royale survival is a phase-indexed survival process shaped by
shrinking playable space, spatial exposure, combat interaction, and team-to-team
competition. We reconstruct PUBG telemetry into factorized multi-resolution
representations and evaluate which representation components preserve
predictive and interpretable signals for next-phase elimination hazard and
final survival ranking.
```

한국어 요약:

```text
배틀로얄 생존은 단순 전투 통계의 결과가 아니라, 축소되는 플레이 가능 공간,
시공간 노출, 교전 상호작용, 팀 간 경쟁 관계가 phase마다 누적되는 생존 과정이다.
본 연구는 PUBG telemetry를 node/global/graph/snapshot으로 분리한
factorized multi-resolution representation으로 재구성하고, 어떤 표현 요소가
다음 phase 탈락 위험과 최종 생존 순위 신호를 가장 잘 보존하는지 검증한다.
```

## 13. 주의점

이 문서에는 결과 수치를 넣지 않는다. 설계 문서와 결과 문서를 분리해야, 논문 작성 시 방법론과 실험 결과를 명확하게 나눌 수 있다.

또한 최종 주장은 "MR-FARNet이 무조건 최고다"가 아니라 다음이어야 한다.

```text
PUBG 생존 예측에서 node/global/graph/snapshot을 분리한 representation 설계가
flat 또는 broadcast-only 표현보다 hazard와 ranking 구조를 더 잘 보존하는지
체계적으로 검증한다.
```
