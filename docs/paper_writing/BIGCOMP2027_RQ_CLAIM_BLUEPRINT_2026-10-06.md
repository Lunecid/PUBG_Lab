# BigComp 2027 RQ–주장–근거 설계서 (2026-10-06)

이 문서는 원고의 기준이다. 문장을 쓸 때는 이 문서의 주장 문구와 근거, 표기 규칙을 따른다.

- 수치 출처: `reports/bigcomp2027_results/`(contrasts_pooled.csv, tab_decomp, ecological_model/), `model_stgcn/dumps/bigcomp2027_sagong_v1/summary_hab_increment.csv`, 데이터 감사 문서.
- 작성 원칙: Widom, research-paper-writing 스킬, BigComp 서론 양식(빅데이터 관점).

## 0. 논문의 한 문장

> On a rare, large-scale corpus of 10-second spatio-temporal telemetry, treating the shrinking safe zone as stepwise habitat loss yields a phase-conditioned local-extinction hazard whose habitat covariates explain most of the predictable elimination risk. Against prior designs, the decisive elements are this formulation and this habitat representation, not graph or sequence structure.

## 1. RQ 사슬 (확정 문구)

| RQ | EN (원고 문구) | 앞 RQ와의 관계 |
|---|---|---|
| RQ1 | Can next-phase elimination be formulated as local extinction under stepwise habitat loss, with habitat variables tracking the risk? | 출발점: 이론에서 정식화를 세우고 데이터로 검증한다 |
| RQ2 | What model design does this formulation imply relative to prior designs, and does it improve prediction on identical samples? | RQ1이 성립해야 설계를 도출할 수 있다 |
| RQ3 | Which design element accounts for the difference? | RQ2에서 차이가 확인되어야 분해할 수 있다 |

결과를 해석할 때는 RQ3의 최대 요인(정식화)을 다시 RQ1의 메커니즘(서식지가 줄수록 위험이 커짐)으로 설명한다. 그래서 사슬이 고리로 닫힌다.

## 2. RQ별 주장 · 근거 · 위치 · 논증 · 한계

### RQ1: 정식화와 서식지 변수

- **주장 1a (정식화).** Elimination can be cast as a phase-conditioned discrete-time local-extinction hazard, with phases as habitat-loss steps.
  - 근거: 이론적 대응(Fahrig 2003, Hanski 1994, MacKenzie et al. 2003)과 III장의 정의식.
  - 위치: IV-A (식), 서론 P3.
- **주장 1b (서식지 변수가 위험을 추적한다).** Habitat variables track elimination risk consistently across cohorts.
  - 근거 1, EHM: 단계와 서식지 공변량 6개로 만든 로지스틱 모델이 HGB-cut 판별력의 **80%** [0.79, 0.82]를 설명한다. 단계 안 기준으로는 70%다.
  - 근거 2, 계수 재현:
    - 분산 요구 +0.47, 12/12 코호트
    - 국지 밀집 +0.32, 12/12
    - 생존 팀원 −0.68, 11/11
    - 경쟁 팀 수 −0.07, 11/12
    - EHM-lite 결과로 최종 확정한다.
  - 근거 3, 가장자리 기울기: 바깥 깊이 × 단계 상호작용 +0.060 (10/12). 같은 단계 안에서 경계 밖 팀의 탈락률은 단계 4 기준 0.589, 깊은 안쪽은 0.243이다. 단계 1–2에서는 차이가 거의 없다.
  - 위치: VI-A. 그림 2 = EHM 계수의 코호트별 forest plot 또는 가장자리 분위 × 단계 그림.
  - 분석 유형: 탐색적, 기술적.
- **논증 순서.** 이론 → 정식화 → 사전에 정한 방향 예측 → 데이터 확인.
  - 방향 예측 1: 위험은 분산 요구와 함께 증가한다.
  - 방향 예측 2: 위험은 국지 밀집과 함께 증가한다.
  - 방향 예측 3: 가장자리 기울기는 서식지가 줄수록 가팔라진다.
  - 세 예측 모두 확인되었다. 생태 틀은 실험 이전에 있었다(2026년 4월 발표자료와 F7 특징). 이것을 원고에 직접 쓰지는 않고, "the habitat representation was designed before the experiments" 정도로만 쓴다.
- **한계.**
  - 연관이지 인과가 아니다.
  - 밀도 변수(남은 팀 수)는 생존자 편향과 교란될 수 있다.
  - 바깥 깊이와 필요 속도는 다중공선성이 있어 EHM-lite로 해석한다.

### RQ2: 설계 차이와 성능

- **주장 2a (설계).** The formulation implies two design changes relative to prior work.
  - 추정 대상: 다음 단계 hazard로 바꾼다. 단계가 추정 대상 안에 들어간다.
  - 표현: 서식지 맥락으로 바꾼다. 가장자리 노출, 분산 요구, 밀집, 개체군 크기다.
  - 팀 간 결합(attention, 관계 bias)은 정식화가 요구하는 것이 아니라 검증할 선택지다.
  - 근거: 네 설계 축 비교표 (표 II).
  - 위치: IV-B, IV-C.
- **주장 2b (성능).** The resulting design outperforms adapted versions of both prior predictors on identical samples.
  - M11R − KRAFTON-tf: **+0.062** [0.052, 0.073], 12/12 (단계 안 +0.044)
  - M11R − SAGONG-xgb: **+0.103** [0.096, 0.109], 12/12 (단계 안 +0.143)
  - 위치: VI-B, 표 III의 첫 행들 또는 본문.
  - 분석 유형: 탐색적 외부 비교. 같은 행, 같은 라벨, 각자의 설정을 쓴다.
- **논증.** 선행 설계를 재현한 것이 아니라 우리 과제에 맞춰 적용한 것임을 먼저 밝힌다. 이어서 공정성 장치(같은 행, 원 설정, 감사한 결함 수정, Platt 보정)를 쓰고, 그다음에 차이를 제시한다.
- **한계.** KRAFTON 입력 22개는 근사치이고 1개는 0이다. Sagong은 플레이어 단위를 팀 단위로, 반 단계를 전체 단계로 바꾸었다.

### RQ3: 분해

- **주장 3a (정식화가 KRAFTON 격차를 설명한다).** Within the same transformer, inputs and samples, replacing the Cox time-of-death objective with the next-phase hazard raises AUC by **0.051** [0.044, 0.058] (12/12). With this objective, KRAFTON's backbone matches our reference model (0.791 vs 0.793).
  - 보조 근거: 명시적 단계 기저의 효과는 0이다(단계가 이미 입력에 있다). 단계별 보정 오차는 0.074에서 0.016으로 준다.
- **주장 3b (표현이 Sagong 격차를 설명하고, 그 대부분이 서식지다).** With the learner fixed, the representation adds **0.091** [0.085, 0.097]. Habitat context alone accounts for **0.053** of it (12/12), and the remaining context accounts for 0.039.
- **주장 3c (구조의 몫은 작고 데이터에 의존한다).** Architecture adds 0.002 [0.0003, 0.003] and structure 0.002 [−0.001, 0.006]. The pre-specified tests find phase history and relation graphs equivalent to null. Structure helps only in cohorts with roughly 700 or more training matches.
- **보조 3d.** Input differences versus KRAFTON's features are small: 0.014 [0.012, 0.016].
- **고리 논증 (RQ1로 귀환).** 정식화 효과의 절반가량(전체 0.051 vs 단계 안 0.027)은 단계 사이의 위험 상승분이다. 이는 서식지가 줄수록 멸종 위험이 커진다는 메커니즘에 해당한다. Cox는 시점별 상수 이동에 불변이어서 바로 이 상승을 표현하지 못한다.
- **위치.** VI-C. 표 III = 분해 7단계 (전체 / 단계 안). 그림 3 = 학습 곡선 (선택).
- **표기.** P1–P5만 사전 지정 분석이다. 분해는 사후 탐색이다.

### 데이터 기여 (III장)

- **주장 D.** A rare public corpus of 10-second in-match telemetry, about 470M records from 27,491 matches, turned into a leakage-safe benchmark: 13,948 matches, 12 cohorts, 829,640 samples. A naive snapshot rule mislabels 11.4% of samples.
- **위치.** III, 표 I.

## 3. 장별 설계도 (분량 8쪽)

| 장 | 단락 역할 | 담당 주장 | 분량 |
|---|---|---|---|
| Abstract | 문제(빅데이터 위험 예측), 데이터·파이프라인, 정식화, 분해 결과 3개 | D, 1b, 3a, 3b, 3c | 0.25 |
| I | P1 시공간 빅데이터 과제 / P2 PUBG 코퍼스 (III 요약) / P3 추정 대상과 생태 렌즈 / P4 공백 / P5 RQ와 접근 / 기여 3개 | 전체 미리보기 | 0.8 |
| II | 시공간 위험 예측, BR·e스포츠 예측, 생존분석(이산 hazard, Cox, DeepHit), 생태 모델 | 위치 짓기 | 0.5 |
| III | A 코퍼스(희소성, 규모) / B 로그 → 표본(시계, 인과 컷, 라벨, 감사) / C 벤치마크 | D | 0.9 |
| IV | A 서식지 손실 hazard 정식화(식, Cox 대비) / B 서식지 표현 / C 참조 모델(그래프·시계열 스위치) / D 선행 설계 적용과 단일 요인 사다리 | 1a, 2a | 1.3 |
| V | 지표(전체 AUC 주 지표, 단계 안 보완), bootstrap, DL, SESOI, 사전 지정 vs 탐색 | — | 0.4 |
| VI | A RQ1 (EHM, 가장자리) / B RQ2 (전체 대비) / C RQ3 (분해 표, 학습 곡선) | 1b, 2b, 3a–c | 1.6 |
| VII | 논의(추정 대상 정합, 표현 > 구조, 데이터 규모, 생태 해석), 한계, 결론 | — | 0.6 |
| Refs | | | 0.8 |

## 4. 그림과 표 (최대 5개)

| 번호 | 내용 | 담당 |
|---|---|---|
| Fig. 1 | 개요: 로그 → 파이프라인 → 서식지 손실 hazard → 분해 사다리 | I, III, IV |
| Table I | 텔레메트리 코퍼스 규모 | III |
| Table II | 네 설계 축: Sagong / KRAFTON / 우리 | IV |
| Fig. 2 | RQ1: EHM 계수 forest (코호트별) + 가장자리 × 단계 | VI-A |
| Table III | 분해 7단계 + 전체 대비 | VI-B, VI-C |
| (선택) Fig. 3 | 학습 곡선 | VI-C |

## 5. 심사위원 질문 예상과 답

| 예상 질문 | 답 (근거) |
|---|---|
| 생태학은 은유일 뿐 아닌가? | 공변량 6개의 생태 모델이 판별력의 80%를 설명하고, 계수 방향이 12/12에서 재현된다. 방향 예측 3개가 모두 확인되었다. 서식지 표현은 실험 전에 설계되었다 |
| 평가 타깃으로 학습했으니 당연하지 않나? | 증거가 양방향이다. KRAFTON은 최종 top-5에서 앞선다(그 목적에 맞기 때문이다). 추정 대상의 정합성이 핵심이라는 주장과 일치한다 |
| 선행 기준선이 불공정하지 않나? | 같은 행·라벨, 원 설정, 감사 결함 수정, 단일 요인 사다리를 썼다. 같은 백본에서 목적만 바꾼 실험(K3a)이 직접적인 증거다 |
| AUC 차이가 작지 않나? | SESOI 0.005 대비 정식화 10배, 표현 18배다. 잘못 순서 매기는 쌍의 비율이 약 25–33% 줄어든다 |
| 누수는 없나? | 시계 정렬, 경계 10초 전 인과 컷, 라벨 감사, 누수 테스트를 했다(III-B) |
| 사후 분석 아닌가? | P1–P5는 사전 지정이다. 분해, 외부 비교, EHM은 탐색적이라고 명시했다 |
| 일반화는? | 12개 코호트(맵 × 시점 × 팀 규모)에서 일관된다. 단일 게임이라는 점은 한계로 쓴다 |

## 6. 표기 규칙 요약

- 사전 지정: P1–P5.
- 사후 탐색: 분해, 외부 비교, EHM, [A]·[B], 단계 안 AUC, 합집합, 데이터 규모 분석.
- 금지 표현: "first", "state of the art", "outperforms KRAFTON"(이 표현 대신 "adapted versions on identical samples"를 쓴다), 생태 인과 표현, "pre-registered".


## 갱신 (EHM-lite 확정, 2026-10-06)

RQ1 주장 1b의 주 근거를 **EHM-lite**로 바꾼다.

- 모델: 단계 + 공변량 4개(분산 요구, 국지 밀집, 경쟁 팀 수의 로그, 생존 팀원 수). 다중공선성이 없다(최대 VIF 1.32, EHM은 11.9).
- AUC: 전체 0.734, 단계 안 0.680.
- 판별력 비율: **0.794** [0.780, 0.808], 단계 안 0.678 [0.654, 0.705].
- 표준화 계수와 부호 일치:

| 공변량 | 계수 [95% CI] | 부호 일치 |
|---|---|---|
| 분산 요구 | +0.10 [0.07, 0.13] | 12/12 |
| 국지 밀집 | +0.31 [0.25, 0.36] | 12/12 |
| 경쟁 팀 수(로그) | −0.06 [−0.10, −0.03] | 11/12 |
| 생존 팀원 | −0.68 [−0.76, −0.61] | 11/11 |

- EHM과의 AUC 차이는 +0.003에 그친다. 가장자리 변수는 상호작용 분석(EHM-int, +0.060, 10/12)에서만 보고한다.
- 원고 문구: "a four-covariate ecological hazard model captures 79% of the full model's discrimination, with consistent signs across cohorts."
