# 논문 작성 전 연구 정리: 실제 수행 내용, 동기, 결과물

이 문서는 ACM/SIGPLAN 템플릿으로 논문을 작성하기 전에, 현재 PUBG 프로젝트에서 실제로 수행한 작업과 그 동기, 실험 결과, 논문으로 주장할 수 있는 범위를 정리하기 위한 준비 문서이다.

참고한 기존 보고서:

- `/Users/lunecid/Downloads/202589713_백성은.pdf`

논문 작성용 LaTeX 템플릿:

- `/Users/lunecid/Downloads/Association_for_Computing_Machinery__ACM____SIGPLAN_Proceedings_Template`

## 1. 연구의 출발점

초기 연구 동기는 PUBG Telemetry를 단순한 경기 로그가 아니라, 시간에 따라 움직이는 객체와 변화하는 공간 영역을 포함한 시공간 데이터베이스(STDB) 문제로 다루는 것이었다.

기존 보고서의 중심 질문은 다음과 같았다.

1. PUBG Telemetry는 어떤 시공간 데이터베이스 구조로 표현될 수 있는가?
2. 안전구역 축소에 따라 팀의 이동 궤적과 생존 환경은 어떻게 변화하는가?
3. 시공간 데이터를 통해 팀의 등수 또는 생존 결과를 예측할 수 있는가?

프로젝트를 진행하면서 연구 초점은 더 구체화되었다. 최종 등수 예측은 경기 전체의 사후 결과와 강하게 연결되므로, 시공간 데이터베이스와 동적 상호작용의 기여를 선명하게 보기 어렵다. 따라서 실제 논문에서 중심으로 삼을 과제는 다음과 같이 정리된다.

> PUBG Telemetry를 STDB로 재구성하고, 팀을 동적 그래프의 노드로 표현하여, 각 phase에서 다음 phase까지의 탈락 위험을 예측한다.

즉, 최종 순위 예측 자체보다 `phase-level next-phase elimination hazard`를 중심 과제로 삼는 것이 현재 결과와 가장 잘 맞는다.

## 2. 우리가 실제로 한 일

### 2.1 Telemetry 수집 및 정제

PUBG Open API 기반 Telemetry를 수집하고, match 단위 로그를 분석 가능한 형태로 정제했다.

현재 확보된 데이터 흐름은 다음과 같다.

| 단계 | match 수 | 의미 |
| --- | ---: | --- |
| Raw fetched matches | 27,491 | 원천 Telemetry 수집량 |
| Base STDB converted matches | 17,984 | STDB 변환 완료 |
| Phase feature matches | 17,983 | phase 단위 feature 생성 |
| Target label matches | 17,977 | phase-level target 생성 |
| Edge cache matches | 17,983 | 동적 edge 생성 |
| Graph-eligible matches | 17,977 | `N_max=128` 조건에서 그래프 학습 가능 |

최종 분석은 TPP squad 중심의 주요 map을 대상으로 진행했다.

| Map | match 수 |
| --- | ---: |
| Baltic_Main | 6,069 |
| Tiger_Main | 4,051 |
| Desert_Main | 1,495 |
| Neon_Main | 1,024 |
| Savage_Main | 908 |

### 2.2 STDB 구조화

Telemetry를 단순 이벤트 목록이 아니라 다음 요소를 가진 시공간 데이터베이스로 재해석했다.

- 이동 객체: player, team
- 동적 공간 영역: safe zone, blue zone, play phase
- 공간 이벤트: combat, damage, death, movement, zone transition
- 시간 단위: match time, phase time, phase boundary
- 분석 단위: team-phase

이 구조화 덕분에 단순 누적 통계가 아니라, 특정 phase에서 팀이 어떤 공간적 압력과 상호작용을 겪었는지 표현할 수 있게 되었다.

### 2.3 Phase 재구성

PUBG match는 안전구역 축소와 blue zone 진행에 의해 phase가 나뉜다. 프로젝트에서는 Telemetry의 phase 관련 이벤트를 이용해 match별 phase boundary를 복원하고, 각 team을 phase 단위로 정렬했다.

그 결과 예측 단위는 다음처럼 정의된다.

> 어떤 team이 현재 phase의 상태와 관계 정보를 바탕으로 다음 phase까지 탈락할 것인가?

이 과제는 최종 등수보다 더 시공간적이다. 왜냐하면 매 phase마다 위치, zone 노출, 전투 압력, 주변 팀 관계가 달라지고, 그 변화가 바로 다음 생존 위험에 반영되기 때문이다.

### 2.4 Feature 설계

팀의 상태를 설명하기 위해 여러 feature group을 구성했다.

| Feature group | 의미 |
| --- | --- |
| Prior scores | 기존식 요약 점수 또는 간단한 prior 정보 |
| Conventional telemetry | kill, damage, alive count, DBNO 등 일반 전투/상태 통계 |
| Team state | 팀 생존 인원, 누적 상태, phase별 상태 |
| Loot/readiness | 장비, 회복, 준비도 관련 proxy |
| Combat | 교전 강도와 전투 이벤트 |
| STDB exposure | zone, playable area, 위험 공간 노출 |
| Trajectory morphology | 이동 궤적의 형태와 이동 패턴 |
| Global phase context | 현재 phase와 match-level context |

중요한 점은 STDB 관련 정보가 단일 feature가 아니라, 노출, 궤적, phase context 등으로 분해되어 모델에 들어갔다는 것이다.

### 2.5 동적 팀 그래프 구성

각 phase에서 team을 node로 보고, team 간 관계를 edge로 구성했다.

사용한 edge type은 다음과 같다.

| Edge type | 의미 |
| --- | --- |
| Proximity | 공간적으로 가까운 팀 관계 |
| Combat | 직접 또는 근접 전투 관계 |
| Copressure | 같은 zone 압력 또는 공간 위험을 공유하는 관계 |

이 그래프는 match 전체에서 고정된 그래프가 아니라, phase마다 바뀌는 동적 그래프이다. 따라서 모델은 팀의 개별 상태뿐 아니라, 같은 시점의 다른 팀들과의 구조적 관계도 사용할 수 있다.

### 2.6 모델 구현

주요 모델은 DynamicSTGNN 계열이다.

비교를 위해 다음 모델/조건을 사용했다.

| 모델 또는 조건 | 목적 |
| --- | --- |
| Tabular HGB baseline | 강한 tabular baseline |
| DynamicSTGNN without graph | 그래프 없이 sequence/node feature만 사용 |
| DynamicSTGNN proximity only | proximity edge만 사용 |
| DynamicSTGNN proximity + combat | proximity와 combat edge 사용 |
| DynamicSTGNN full 3-edge | proximity, combat, copressure 모두 사용 |
| Tuned full | ranking 성능을 더 높이기 위한 tuned variant |

## 3. 핵심 결과

### 3.1 단일 split 기준 주요 성능

기본 실험은 phase-ready 5,950 matches를 사용했다.

| Split | match 수 |
| --- | ---: |
| Train | 4,760 |
| Validation | 595 |
| Test | 595 |

Test set에는 DynamicSTGNN 공통 비교 기준 30,166개의 aligned target cell이 포함되었다.

| Model | AUC | AP | Accuracy | BCE |
| --- | ---: | ---: | ---: | ---: |
| Tabular HGB baseline | 0.7678 | 0.6471 | 0.7504 | 0.5141 |
| DynamicSTGNN no graph | 0.7593 | 0.6293 | 0.7460 | 0.5227 |
| DynamicSTGNN proximity only | 0.7665 | 0.6393 | 0.7496 | 0.5167 |
| DynamicSTGNN proximity + combat | 0.7665 | 0.6378 | 0.7481 | 0.5171 |
| DynamicSTGNN full 3-edge | 0.7693 | 0.6423 | 0.7515 | 0.5147 |
| Tuned full | 0.7708 | 0.6469 | 0.7166 | 0.5544 |

Tabular HGB baseline의 AUC/AP/Accuracy는 `model_stgcn/checkpoints/baseline_phase_5951_shared.json`의 test hazard metrics에서 재추출한 값이다. BCE는 해당 JSON에 저장되어 있지 않아 기존 aligned analysis의 `phase_result_analysis/hazard_overall_metrics.csv` 값을 유지했다. 최종 논문 표는 한 산출 스크립트에서 모든 baseline 지표를 다시 생성해 provenance를 통일해야 한다.

해석:

- Untuned full 3-edge 모델은 가장 균형 잡힌 graph model이다.
- Tuned full은 AUC/AP ranking 성능은 가장 높지만, BCE와 calibration이 나빠졌다.
- Tabular HGB baseline은 매우 강한 baseline이며, DynamicSTGNN full은 이와 비슷하거나 일부 지표에서 근접한 성능을 보인다.
- 논문에서는 단순히 "GNN이 baseline을 압도했다"가 아니라, "동적 관계 edge가 STGNN 내부에서 일관된 개선 신호를 만든다"는 방향이 더 정확하다.

### 3.2 Feature ablation 결과

Feature group의 기여를 보기 위해 prior-style baseline부터 STDB feature를 포함한 full feature까지 비교했다.

| Feature set | AUC | AP | BCE |
| --- | ---: | ---: | ---: |
| Prior scores | 0.6680 | 0.4657 | 0.5862 |
| Prior raw | 0.7287 | 0.5649 | 0.5500 |
| Conventional telemetry | 0.7581 | 0.6338 | 0.5221 |
| STDB spatial only | 0.7101 | 0.5672 | 0.5555 |
| Conventional + STDB spatial | 0.7674 | 0.6459 | 0.5144 |
| Full node features | 0.7682 | 0.6465 | 0.5140 |

주요 통계 검정 결과:

| 비교 | 결과 |
| --- | --- |
| Full features vs prior scores | AUC +0.0989, p = 2.31e-87 |
| Full features vs prior raw | AUC +0.0369, p = 1.44e-50 |
| Conventional + STDB spatial vs conventional | AUC +0.0085, p = 4.81e-13 |
| Conventional + STDB spatial vs conventional | AP +0.0091, p = 4.05e-08 |
| Conventional + STDB spatial vs conventional | BCE -0.0081, p = 7.03e-22 |
| STDB spatial only vs prior scores | AUC +0.0388, p = 2.58e-26 |

해석:

- STDB spatial feature는 단독으로도 prior score보다 더 강한 신호를 가진다.
- Conventional telemetry에 STDB spatial feature를 추가하면 작지만 통계적으로 매우 안정적인 성능 향상이 있다.
- 따라서 논문에서 주장할 수 있는 핵심은 "STDB 표현이 단순 사후 통계를 대체한다"가 아니라, "기존 telemetry feature에 공간적 노출과 phase context를 추가해 예측력을 보강한다"이다.

### 3.3 Feature group importance

Full feature 모델에서 feature group 제거 실험을 수행했다.

| Removed group | AUC drop |
| --- | ---: |
| Loot/readiness | 0.0572 |
| Combat | 0.0364 |
| Team state | 0.0358 |
| STDB exposure | 0.0309 |
| Global phase context | 0.0234 |
| Prior scores | 0.0081 |
| Trajectory morphology | 0.0070 |

해석:

- 가장 큰 단일 신호는 loot/readiness이다.
- 하지만 STDB exposure는 상당히 큰 기여를 한다.
- trajectory morphology만으로는 효과가 작으므로, 논문에서 "이동 궤적 자체가 가장 강한 요인"이라고 주장하면 안 된다.
- 더 정확한 표현은 "공간 노출과 phase context가 생존 위험 예측에 의미 있는 STDB 신호를 제공한다"이다.

### 3.4 Edge ablation 결과

동적 그래프 edge의 기여를 확인하기 위해 edge type별 ablation을 수행했다.

단일 split의 paired test 결과:

| 비교 | 결과 |
| --- | --- |
| Full vs no graph | BCE -0.0080, p = 1.84e-17 |
| Full vs no graph | AUC +0.0075, p = 5.86e-10 |
| Full vs no graph | AP +0.0087, p = 2.16e-07 |
| Full vs proximity only | AUC +0.0026, p = 0.0044 |
| Full vs proximity + combat | AUC +0.0016, p = 0.0114 |

해석:

- 그래프 edge는 no-graph 모델 대비 작지만 일관된 성능 향상을 만든다.
- proximity만으로도 많은 신호가 들어오지만, combat과 copressure를 포함한 full edge가 전체적으로 더 낫다.
- phase별로는 초중반 phase에서 graph uplift가 더 강하고, 후반 phase에서는 표본 수와 경기 구조 변화 때문에 효과가 약하거나 불안정하다.

### 3.5 Seed repeat 필수 실험

필수 실험으로 3개 seed 반복 실험을 추가 수행했다.

출력 위치:

- `model_stgcn/checkpoints/required_experiments/seed_repeats/seed_repeat_summary.json`
- `model_stgcn/checkpoints/required_experiments/seed_repeats/seed_repeat_metrics.csv`
- `model_stgcn/checkpoints/required_experiments/seed_repeats/seed_repeat_deltas.csv`

설정:

- Seeds: 13, 42, 2026
- Edge modes: none, proximity_combat, full
- Maps: Baltic_Main, Tiger_Main, Desert_Main, Neon_Main, Savage_Main
- 총 45개 학습/평가 run

전체 평균 결과:

| Edge mode | BCE mean | AUC mean | AP mean | Accuracy mean |
| --- | ---: | ---: | ---: | ---: |
| None | 0.5728 | 0.7415 | 0.5883 | 0.7475 |
| Proximity + combat | 0.5676 | 0.7474 | 0.5966 | 0.7495 |
| Full | 0.5655 | 0.7499 | 0.6003 | 0.7499 |

Full vs none:

| Metric | Mean delta | Positive count |
| --- | ---: | ---: |
| BCE | -0.0072 | lower is better |
| AUC | +0.0084 | 15 / 15 |
| AP | +0.0120 | 14 / 15 |
| Accuracy | +0.0024 | 13 / 15 |

Proximity + combat vs none:

| Metric | Mean delta | Positive count |
| --- | ---: | ---: |
| BCE | -0.0051 | lower is better |
| AUC | +0.0058 | 15 / 15 |
| AP | +0.0083 | 14 / 15 |

Map별 full vs none의 평균 AUC 개선은 모든 map에서 양수였다.

| Map | AUC delta | AP delta |
| --- | ---: | ---: |
| Baltic_Main | +0.0092 | +0.0108 |
| Tiger_Main | +0.0083 | +0.0081 |
| Desert_Main | +0.0057 | +0.0114 |
| Neon_Main | +0.0070 | +0.0174 |
| Savage_Main | +0.0119 | +0.0124 |

주의할 점:

- Full edge가 no-graph 대비 안정적으로 낫다는 주장은 가능하다.
- 그러나 full edge가 모든 map에서 proximity + combat보다 항상 낫다고 말하면 안 된다.
- 특히 Neon_Main에서는 full과 proximity + combat의 차이가 약하거나 역전되는 구간이 있어, copressure의 효과는 map-dependent하다고 쓰는 것이 안전하다.

### 3.6 Random edge sanity check

동적 edge의 효과가 단순히 edge density나 regularization 때문인지 확인하기 위해 random edge sanity check를 수행했다.

출력 위치:

- `model_stgcn/checkpoints/domain_runs_expanded_e12/edge_sanity/edge_sanity_summary.json`

핵심 결과:

| Condition | AUC | AP | BCE |
| --- | ---: | ---: | ---: |
| Observed full edges | 0.7631 | 0.6213 | 0.5083 |
| Node-shuffled edges | 약 0.6966-0.6985 | 약 0.5242-0.5270 | 약 0.5653-0.5679 |

해석:

- 실제 phase에 정렬된 team relation이 성능에 중요하다.
- 그래프가 단순히 edge 수를 늘려서 regularization처럼 작동한 것이 아니라, 관측된 동적 관계 자체가 정보를 제공한다.

### 3.7 Calibration 결과

Calibration 분석도 수행했다.

출력 위치:

- `model_stgcn/checkpoints/required_experiments/calibration/calibration_metrics.csv`
- `model_stgcn/checkpoints/required_experiments/calibration_by_phase/calibration_metrics.csv`

주요 결과:

| Model | Brier | BCE | ECE |
| --- | ---: | ---: | ---: |
| Dynamic full e12 | 0.1705 | 0.5147 | 0.0076 |
| Dynamic full tuned | 0.1871 | 0.5544 | 0.1236 |
| Dynamic none e12 | 0.1737 | 0.5227 | 0.0067 |
| Dynamic proximity + combat e12 | 0.1715 | 0.5171 | 0.0041 |
| Dynamic proximity e12 | 0.1713 | 0.5167 | 0.0037 |
| Tabular HGB baseline | 0.1704 | 0.5141 | 0.0066 |

해석:

- Untuned full model은 ranking과 calibration의 균형이 좋다.
- Tuned full은 AUC/AP는 높지만 calibration이 크게 나빠진다.
- 논문에서는 tuned full을 최종 대표 모델로 삼기보다, ranking upper bound 또는 보조 결과로 다루는 편이 좋다.

## 4. 우리가 만든 결과물

### 4.1 코드 산출물

대표적인 신규 또는 주요 코드 산출물:

- `model_stgcn/seed_repeat_experiments.py`
- `model_stgcn/edge_sanity.py`
- `model_stgcn/calibration.py`
- `model_stgcn/phase_strategy_type_analysis.py`
- `model_stgcn/st_hypothesis_analysis.py`
- `network_analysis/build_analysis_dataset.py`

### 4.2 실험 산출물

대표 결과 위치:

- `model_stgcn/checkpoints/required_experiments/seed_repeats/`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/edge_sanity/`
- `model_stgcn/checkpoints/required_experiments/calibration/`
- `model_stgcn/checkpoints/required_experiments/calibration_by_phase/`
- `model_stgcn/checkpoints/required_experiments/`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/`

### 4.3 문서 산출물

현재 논문 준비에 직접 참고할 수 있는 문서:

- `docs/PROJECT_FLOW_SUMMARY.md`
- `docs/MODEL_RESULTS.md`
- `docs/REQUIRED_EXPERIMENTS.md`
- `docs/PAPER_PREP_RESEARCH_SUMMARY.md`

## 5. 논문에서 주장할 수 있는 것

현재 결과에 근거해 비교적 안전하게 주장할 수 있는 내용은 다음과 같다.

1. PUBG Telemetry는 STDB 관점에서 moving teams, dynamic safe zones, spatial events, phase transitions로 구조화할 수 있다.
2. Phase-level next-phase elimination hazard는 PUBG의 시공간 구조를 반영하는 적절한 예측 과제이다.
3. STDB spatial feature는 prior-style feature보다 강한 예측 신호를 제공한다.
4. Conventional telemetry에 STDB spatial feature를 추가하면 통계적으로 유의한 성능 향상이 있다.
5. Dynamic team graph는 no-graph STGNN 대비 작지만 일관된 개선을 만든다.
6. 3-seed 반복 실험에서 full graph는 no-graph 대비 모든 map-seed 조합에서 AUC 개선을 보였다.
7. Random edge sanity check는 관측된 edge alignment가 중요하다는 점을 뒷받침한다.
8. Calibration 분석상 tuned model보다 untuned full model이 최종 대표 모델로 더 적합하다.

## 6. 논문에서 조심해야 할 것

현재 결과만으로 강하게 주장하면 안 되는 내용은 다음과 같다.

1. "GNN이 모든 baseline을 압도했다"라고 쓰면 안 된다. Tabular HGB baseline은 매우 강하다.
2. "Copressure edge는 모든 map에서 항상 도움이 된다"라고 쓰면 안 된다. map-dependent한 결과가 있다.
3. "이동 궤적 morphology가 가장 중요한 feature다"라고 쓰면 안 된다. 현재는 STDB exposure가 더 강한 STDB signal이다.
4. "최종 등수 예측 문제를 해결했다"라고 쓰면 안 된다. 현재 핵심 과제는 phase-level next-phase elimination hazard이다.
5. "전략적 인과 관계를 증명했다"라고 쓰면 안 된다. 결과는 예측적 연관성과 구조적 신호에 관한 것이다.
6. Held-out map generalization을 강하게 주장하면 안 된다. 현재는 map-wise 반복과 ablation이 중심이다.

## 7. 논문 서사의 중심

논문의 중심 서사는 다음 흐름이 가장 자연스럽다.

1. PUBG Telemetry는 단순 로그가 아니라 대규모 시공간 이벤트 데이터이다.
2. 그러나 기존 game analytics는 주로 사후 통계나 최종 결과 예측에 치우쳐 있다.
3. 우리는 Telemetry를 STDB로 재구성하고, match를 phase-level team trajectories로 변환했다.
4. 각 phase에서 team 간 proximity, combat, copressure 관계를 동적 그래프로 구성했다.
5. 이 표현을 이용해 다음 phase까지의 team elimination hazard를 예측했다.
6. 실험 결과, STDB spatial feature와 dynamic graph relation은 작지만 안정적인 성능 향상을 제공했다.
7. 특히 seed repeat와 random edge sanity check는 이 효과가 우연한 split이나 단순 edge density가 아니라는 점을 보강한다.

## 8. 논문 제목 후보

후보 1:

> Modeling PUBG Telemetry as a Spatio-Temporal Database for Phase-Level Team Elimination Prediction

후보 2:

> Dynamic Team Graphs over Spatio-Temporal PUBG Telemetry for Phase-Level Survival Modeling

후보 3:

> From Telemetry Logs to Dynamic Spatial Graphs: Phase-Level Elimination Risk Prediction in PUBG

현재 결과와 가장 잘 맞는 제목은 후보 1 또는 후보 2이다.

## 9. 논문 구조 초안

ACM/SIGPLAN 템플릿에 맞춘 1차 구조는 다음과 같이 잡을 수 있다.

### Abstract

- PUBG Telemetry를 STDB로 재구성한 동기
- Phase-level elimination hazard 과제 정의
- Dynamic team graph 모델 개요
- STDB feature와 graph edge의 실험적 기여 요약

### 1. Introduction

- PUBG Telemetry의 특성: moving objects, dynamic zones, sparse combat events
- 기존 최종 등수/사후 통계 중심 접근의 한계
- 본 연구의 질문과 기여

### 2. Background and Related Work

- Spatio-temporal databases
- Sports/esports analytics
- Graph neural networks for dynamic interactions
- PUBG/game telemetry analysis

### 3. Data and STDB Modeling

- Telemetry 수집 및 정제
- Match, player, team, phase schema
- Safe zone과 phase reconstruction
- Team-phase record 생성

### 4. Prediction Task

- Next-phase elimination hazard 정의
- Label 생성
- Train/validation/test split
- Evaluation metrics: AUC, AP, BCE, accuracy, Brier, ECE

### 5. Dynamic Team Graph Model

- Node feature groups
- Edge type: proximity, combat, copressure
- DynamicSTGNN architecture
- Baselines

### 6. Experiments

- Main comparison
- Feature ablation
- Edge ablation
- Phase-wise evaluation
- Seed repeat
- Random edge sanity check
- Calibration

### 7. Results and Discussion

- STDB feature의 기여
- Graph relation의 안정적이지만 작은 개선
- Map-dependent edge behavior
- Calibration trade-off
- 해석 가능한 한계

### 8. Threats to Validity / Limitations

- Telemetry 누락 및 match filtering
- Map별 데이터 불균형
- Patch/version 차이 가능성
- Causal claim이 아닌 predictive modeling
- Generalization 범위

### 9. Conclusion

- PUBG Telemetry를 STDB와 dynamic graph로 모델링할 수 있음을 요약
- Phase-level survival modeling에서 STDB spatial context와 dynamic relation이 유의미한 신호임을 정리

## 10. 다음 작업

논문 작성 전 다음 작업을 정리해야 한다.

1. LaTeX 템플릿을 workspace 안으로 복사해 논문 작업용 파일을 만든다.
2. 표 4-6개를 최종 선정한다.
3. 그림 3-4개를 만든다.
   - 데이터 처리 pipeline
   - STDB schema 또는 conceptual diagram
   - Dynamic team graph construction
   - 주요 ablation 결과 plot
4. `Abstract`와 `Introduction`의 주장 수위를 현재 결과에 맞게 고정한다.
5. 관련 연구 bibliography를 구성한다.
6. 실험 결과 파일에서 논문용 숫자를 다시 한 번 자동 추출해 표와 본문 숫자를 일치시킨다.
7. Limitations를 먼저 써서 과장된 claim을 방지한다.

## 11. 현재 확인된 리스크와 처리 방침

논문 작성 전에 다음 문제들을 명시적으로 통제해야 한다.

| 항목 | 확인 내용 | 처리 방침 |
| --- | --- | --- |
| 모델 트리 이름 충돌 | 루트의 `dataset.py`, `train.py`, `model/arena_survival_net.py`는 이전 snapshot 기반 `ArenaSurvivalNet` 라인이고, `model_stgcn/dataset.py`, `train_phase.py`, `eval_phase.py`가 현재 논문용 phase hazard 라인이다. | 논문과 재현 명령은 `model_stgcn` 파이프라인만 사용한다. 루트 모델은 legacy/exploratory로 분리해서 설명한다. |
| GNN 절대 이득 미미 | 단일 shared split에서 full graph AUC 0.7693, tabular baseline AUC 0.7678로 거의 비등하다. | "GNN이 tabular baseline을 압도했다"는 서사를 금지한다. 핵심 주장은 "STDB feature가 강하고, graph edge는 no-graph neural model 대비 안정적인 보완 신호를 준다"로 둔다. |
| AP 역전 | 단일 shared split에서 full graph AP 0.6423, tabular baseline AP 0.6471이다. | AUC만 강조하지 않는다. AP, BCE, calibration을 함께 보고하고, tabular baseline이 강하다는 점을 Results/Limitations에 명시한다. |
| 데이터 수치 불일치 | DB 및 summary 기준 `phase_eligible_all=17,977`, `phase_feature_all=17,983`, primary TPP 5-map 합계 `13,547`, shared split `5,950`이다. | 세 숫자를 같은 모집단처럼 쓰지 않는다. Data section에서 "전체 가용량", "primary TPP domain runs", "shared split main comparison"을 분리한다. |
| `N_max=128` padding | 현재 DynamicSTGNN tensor는 `N=128` padding과 mask를 사용한다. squad 실제 팀 수는 보통 약 25-32팀이므로 계산 비효율이 있다. | 이미 수행된 실험과 재현성을 위해 논문 결과는 `N_max=128` 기준으로 보고한다. 효율화는 future work 또는 implementation limitation으로 둔다. |
| FPP와 late phase 검정력 | 현재 논문 핵심 실험은 TPP squad 중심이며, FPP는 충분한 규모의 본 실험으로 수행하지 않았다. 또한 phase 6-7은 표본이 작고 생존자 편향이 크다. | 본 논문의 중심 claim은 TPP squad와 phase 1-6 중심으로 둔다. FPP와 late phase 결과는 수행 범위 밖이거나 보조적/방향성 결과로만 언급한다. |
| tuned checkpoint 오선택 | tuned full은 AUC/AP가 높지만 ECE와 BCE가 크게 나쁘다. | 최종 대표 모델은 untuned `phase_full_5951_e12.pt`로 둔다. tuned 모델은 ranking-oriented supplementary result로만 둔다. |
| 산출 라인 분리 | webapp/PPTX는 `network_analysis` 중심의 가설검정/시각화 서사이고, 논문 핵심은 `model_stgcn` hazard 서사이다. | 논문 본문은 `model_stgcn`의 phase-level hazard를 중심으로 쓴다. webapp/PPTX는 supporting artifact 또는 exploratory analysis로 분리한다. |
| `adapters/` 미사용 | `adapters/pubg_adapter.py`의 `CanonicalMatch` 변환은 현재 DynamicSTGNN 파이프라인에서 직접 사용되지 않는다. | 논문 방법론에서 제외한다. 필요하면 코드 정리 단계에서 exploratory adapter로 표시한다. |

이 리스크들을 반영하면 Introduction의 claim은 다음 문장 수준이 안전하다.

> The main contribution is not that a graph neural network universally outperforms all tabular baselines, but that a spatio-temporal database representation of PUBG telemetry exposes spatial and relational signals that complement conventional telemetry features and consistently improve no-graph neural variants.

## 12. 현재 결론

현재 프로젝트는 단순 모델 실험이 아니라, PUBG Telemetry를 시공간 데이터베이스로 재구성하고 phase-level dynamic graph prediction까지 연결한 연구로 정리할 수 있다.

가장 강한 논문 메시지는 다음이다.

> PUBG의 raw telemetry를 STDB와 dynamic team graph로 재구성하면, phase-level team elimination hazard를 예측하는 데 있어 기존 telemetry feature를 보완하는 안정적인 공간 및 관계 신호를 얻을 수 있다.

이 방향은 기존 보고서의 문제의식과도 연결되며, 현재 완료된 필수 실험 결과와도 가장 잘 맞는다.
