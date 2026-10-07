# BigComp 2027 원고 설계도 (단락 단위, 2026-10-06)

- 원칙: 사용자와 함께 쓴다. 이 문서는 문장이 아니라 각 단락의 역할, 주장, 근거 수치를 정한다.
- 수치 출처: `BIGCOMP2027_PAPER_FOCUS_2026-10-06.md`(v5)와 `reports/bigcomp2027_results/`
- 분량: 8쪽, 참고문헌 포함

## 제목 후보

1. *What Do Battle Royale Survival Predictors Miss? Decomposing Representation, Formulation, and Structure*
2. *Representation and Formulation, Not Architecture: Decomposing Elimination-Risk Prediction in Battle Royale Games*
3. *Phase-Conditioned Hazards for Battle Royale Elimination Prediction: A Controlled Decomposition of Prior Designs*

## 초록 (약 180단어, 5문장)

1. 문제: 줄어드는 안전구역 아래에서 팀의 다음 단계 탈락 위험을 실시간으로 추정하는 일.
2. 공백: 두 선행 설계(단계별 활동 지표 분류, 사망 시각 Cox 기반 우승확률 Transformer)는 위험 구간, 추정 대상, 맥락, 팀 간 결합이 동시에 다르다. 그래서 무엇이 성능을 좌우하는지 알 수 없다.
3. 방법: 공개 PUBG 13,948경기, 12개 코호트에서 같은 행과 사건 기반 라벨로 한 번에 한 요인씩 바꾸는 분해를 한다.
4. 결과:
   - 맥락 표현이 첫 설계와의 차이를 설명한다 (+0.091 AUC).
   - 단계 조건부 hazard 정식화가 둘째 설계와의 차이를 설명한다 (같은 백본에서 목적만 바꿔 +0.051).
   - 아키텍처와 관계 구조는 +0.002 수준이다.
5. 의미: BR 위험 예측의 설계 우선순위는 추정 대상을 맞추고 맥락을 표현하는 것이다. 구조는 데이터가 충분할 때만 이득이다.

## 절별 설계 (분량)

### I. 서론 (0.9쪽)

| 단락 | 역할 | 핵심 내용 |
|---|---|---|
| 1 | 문제와 필요 | 안전구역 단계가 경기 시계를 정한다. 단계별 위험은 중계, 분석, 코칭에 쓰인다 |
| 2 | 선행 두 갈래 | Sagong(플레이어·반 단계·활동 지표), KRAFTON(squad·임의 시점·Cox·우승확률). 네 축이 동시에 달라 비교가 불가능하다 |
| 3 | 질문과 접근 | 무엇이 차이를 만드는가. 같은 데이터·행·라벨에서 단일 요인 분해 |
| 4 | 기여 | C1–C3 + 사건 기반 라벨과 공개 다조건 데이터 (한 줄) |

### II. 관련연구 (0.5쪽)

BR 예측, e스포츠 실시간 승률, 생존분석(이산 hazard, Cox, DeepHit), 관계·attention 모델 순으로 위치를 잡는다.

### III. 문제와 데이터 (1.0쪽)

- A. 단계 조건부 이산 hazard
  - 정의: 위험 집합, 라벨, hazard
  - Cox와의 관계: 시점마다 상수만큼 이동해도 결과가 같다. 그래서 단계 간 기저율을 표현하지 못한다. C2의 이론적 근거다.
- B. 데이터
  - 공개 API, 12개 코호트, 13,948경기
  - airoyale과 Vikendi 제외, 경기 단위 분할
- C. 라벨: 사건 기반 정의를 쓴다. snapshot 규칙은 11.4%를 오분류한다는 2~3문장을 넣는다.

### IV. 비교하는 설계 (1.2쪽)

`BIGCOMP2027_BASELINES_AND_CONTRAST_2026-10-06.md`를 압축한다.

- A. 네 설계 축 (표: Sagong / KRAFTON / 우리)
- B. Sagong 이식의 핵심 판단 3~4개
- C. KRAFTON 이식: 지표 재구성(정확 32, 근사 22), 감사한 결함 수정, 학습과 평가 시점
- D. 우리 표현과 참조 모델(PTR-Former) 개요
- E. 단일 요인 사다리 (표의 행 = 분해 단계)

### V. 평가 (0.5쪽)

- 지표: 전체 AUC(사전 지정 주 지표), 단계 안 AUC(보완 지표)
- 통계: paired bootstrap, DL 통합, SESOI
- 사전 지정 분석과 사후 탐색의 구분

### VI. 결과 (1.5쪽)

- A. 분해 표 (7단계): C1, C2
  - 같은 백본에서 목적만 바꾸면 KRAFTON이 우리 모델에 도달한다 (0.791 vs 0.793)
  - 보정 오차 0.074 → 0.016
- B. 구조: P1–P5 (작은 표 또는 본문)와 학습 곡선 그림. C3를 뒷받침한다.
- C. 보조: 확장 표현 +0.008, top-5에서의 추정 대상 차이

### VII. 논의와 결론 (0.7쪽)

- 함의: 정식화 > 표현 > 구조
- 한계: 단일 게임, 공개 경기, 이식, 사후 분석
- 결론

## 그림과 표 (최대 5개)

| 번호 | 내용 | 필수 여부 |
|---|---|---|
| 표 I | 네 설계 축 비교 | 필수 |
| 표 II | 분해 7단계 (전체 / 단계 안 ΔAUC) | 필수 |
| 그림 1 | 학습 곡선 (2개 코호트) | 필수 |
| 표 III | P1–P5 판정 | 선택 (본문 대체 가능) |
| 표 IV | 입력 × 학습기 격자 | 선택 |


---

## 확정 사항 (2026-10-06, 사용자 결정)

- 제목: *What Do Battle Royale Survival Predictors Miss? Decomposing Representation, Formulation, and Structure*
- 결과 순서: C2(정식화) → C1(표현) → C3(구조)
- PTR-Former는 분해용 참조 모델로만 소개한다.

## 초록 초안 v1 (공동 검토용)

> Battle royale matches unfold as a sequence of shrinking safe-zone phases, and the risk that a team is eliminated before the next phase is a natural quantity for real-time analytics. Two recent predictors approach this risk from different directions: phase-wise activity scores classified per player, and a cross-squad transformer trained on time of death for win probability. They differ at once in risk interval, estimand, observed context and cross-team coupling, so it is unclear what drives their performance. We port both designs to 13,948 public PUBG matches in 12 map, perspective and team-size cohorts, evaluate them on identical rows with audited event-based labels, and decompose the performance gap one design factor at a time. Within the same transformer backbone and inputs, replacing the time-of-death Cox objective with a phase-conditioned discrete hazard raises next-phase elimination AUC by 0.051 and closes the gap to our reference model. With the learner held fixed, a context representation of zone geometry, local competition and team state adds 0.091 AUC over phase-wise activity indicators. Architecture and explicit relation structure add about 0.002 and help only in data-rich cohorts. Aligning the estimand and representing the arena context matter more than model structure.

## 기여 초안 v1 (서론 마지막 단락)

- **Formulation.** On identical inputs and backbone, a phase-conditioned discrete-time hazard objective improves next-phase elimination AUC by 0.051 over a time-of-death Cox objective. Cox scores are invariant to per-time-point shifts and so cannot encode phase-level base rates.
- **Representation.** With the learner fixed, a context representation of zone geometry, local competition and team state improves AUC by 0.091 over phase-wise activity indicators and by 0.014 over cumulative squad statistics.
- **Structure.** Cross-team attention, phase history and typed relation biases add at most 0.006 AUC. Pre-specified equivalence tests find no meaningful gain from history or relations, and structure helps only above roughly 700 training matches.
- **Data.** We construct event-based elimination labels for 12 public cohorts. Snapshot-based alive inference mislabels 11.4% of team-phase rows. Release scope: TBD.


---

## 생태학 틀 (2026-10-06, 사용자 결정: Fahrig 2003 중심)

- **근거 문헌** (서지 검증 완료):
  - Fahrig, L. (2003). Effects of habitat fragmentation on biodiversity. *Annu. Rev. Ecol. Evol. Syst.* 34:487–515. 서식지 **손실**은 크고 일관된 부정적 효과를 내고, 단편화 자체의 효과는 약하다.
  - Nathan, R. et al. (2008). A movement ecology paradigm for unifying organismal movement research. *PNAS* 105(49):19052–19059. 분산 비용의 근거로 쓴다.
- **대응** (개념적 유비이며 인과 주장은 아니다):
  - 안전구역 축소는 단편화 없는 순수한 서식지 손실이다.
  - 팀 탈락은 국지적 멸종이다.
  - 단계마다 손실이 한 번씩 일어나므로 단계 조건부 이산 hazard로 정식화한다.
  - habitat 표현은 남은 안전 면적, 가장자리 거리, 분산 요구(안전구역 도달 필요 속도), 국지 밀집으로 구성한다.
- **원고 배치**:
  - 서론 1단락: 생태 유비로 문제를 정의한다.
  - III-A: 국지적 멸종 hazard를 정식화한다.
  - IV-D: habitat 표현의 설계 근거를 쓴다.
  - VI: habitat 블록 제거 시 −0.011 (신경망), −0.0106 (HGB), 12/12. 사후 탐색으로 표기한다.
- **쓰지 않는 것**:
  - "자기장 피해 = 서식지 손실 사망" (해당 변수가 없음)
  - 원문 초록으로 뒷받침되지 않는 Fahrig·Nathan 인용 문장
  - 출처 없는 z값

## 초록 초안 v2 (생태학 틀 반영)

> Battle royale matches unfold as a sequence of shrinking safe-zone phases, an artificial analogue of habitat loss, and the risk that a team is eliminated before the next phase (its local extinction risk) is a natural target for real-time analytics. Two recent predictors approach this risk from different directions: phase-wise activity scores classified per player, and a cross-squad transformer trained on time of death for win probability. They differ at once in risk interval, estimand, observed context and cross-team coupling, so it is unclear what drives their performance. Drawing on habitat-loss ecology, we formulate elimination as a phase-conditioned discrete-time hazard and represent each team's habitat context: remaining safe area, edge distance, dispersal demand and local crowding. We adapt both designs to 13,948 public PUBG matches in 12 map, perspective and team-size cohorts, and decompose the performance gap one factor at a time on identical rows with audited event-based labels. Within the same transformer backbone and inputs, replacing the time-of-death Cox objective with the phase-conditioned hazard raises next-phase elimination AUC by 0.051. With the learner fixed, a habitat-aware context representation adds 0.091 over phase-wise activity indicators. Architecture and explicit relation structure add about 0.002 and help only in data-rich cohorts.


---

## 선행연구 대비와 기여: 강한 대비 (2026-10-06)

### 공백: 선행 연구가 답하지 못한 것

| | Sagong et al. (2025, EC) | KRAFTON pubg-winprob (2026, 공개 코드) | 남는 공백 |
|---|---|---|---|
| 위험을 무엇으로 보나 | 다음 반 단계의 생존 여부 (플레이어 분류) | 미래 사망 시각의 순위를 우승확률로 변환 | 경기의 축소 단계마다 발생하는 **국지적 멸종 위험**으로 정식화한 연구가 없다 |
| 단계의 역할 | 지표를 모으는 시간 창 | 표본 추출과 평가에만 사용. 추정 대상 밖이다 | 단계별 기저 위험이 추정 대상에 없다 |
| 다른 팀과 경기장 | 100 m 안 인원수 하나 | 150 m 안 인원수, 누적 행동, 장비, 같은 시점 attention | 서식지 맥락(남은 면적, 가장자리, 분산 요구, 밀집)을 표현한 연구가 없다 |
| 단위와 데이터 | 솔로 플레이어, 비공개 1주 | esports squad, 데이터 비공개 | 공개 다조건 팀 데이터가 없다 |
| 평가 | 정확도 (양성 약 85%) | 우승 정확도, C-index | 같은 데이터와 같은 행에서 비교한 적이 없다. 그래서 **무엇이 성능을 좌우하는지 모른다** |

### 기여 4개 (선행 → 우리 → 근거)

1. **정식화: 사망 시각 순위에서 단계 조건부 국지 멸종 hazard로** (KRAFTON 대비)
   - 선행: Cox 순위 목적은 시점마다 상수만큼 이동해도 결과가 같다. 그래서 단계 정보가 입력에 있어도 단계별 기저율을 표현하지 못한다.
   - 우리: 축소 단계를 서식지 손실의 한 단계로 보고, 다음 단계 안의 탈락 확률을 직접 추정한다.
   - 근거: 같은 백본, 입력, 표본에서 목적만 바꾸면 **+0.051 AUC (12/12)**. 단계별 보정 오차는 0.074에서 0.016으로 준다. 이 정식화를 주면 KRAFTON 백본이 우리 참조 모델과 같아진다 (0.791 vs 0.793). 차이는 아키텍처가 아니라 정식화에서 온다.
2. **표현: 개인 활동 점수에서 서식지 맥락으로** (Sagong 대비)
   - 선행: 플레이어 자신의 활동 8개 지표. 다른 플레이어에 대해서는 100 m 안 인원수 하나만 본다.
   - 우리: 서식지 손실 생태학(Fahrig 2003)에 근거한 팀 단위 맥락. 남은 안전 면적, 가장자리 거리, 분산 요구(Nathan 2008), 국지 밀집, 팀 상태, 팀 쌍 상호작용 이력으로 구성한다.
   - 근거: 같은 학습기에서 **+0.091 AUC (12/12, 단계 안 +0.119)**. habitat 블록 하나를 빼면 −0.011 (12/12).
3. **방법: 비교 불가능한 시스템에서 단일 요인 분해로**
   - 선행: 각자 다른 데이터, 지표, 추정 대상으로 보고했다.
   - 우리: 두 설계를 같은 공개 데이터, 같은 행, 사건 기반 라벨에 적용한다. 입력, 목적, 기저, 아키텍처, 구조를 한 번에 하나씩 바꾼다. 구조 효과는 사전 지정 동등성 검정으로 판정한다.
   - 근거: 아키텍처와 구조는 각 +0.002다. 단계 이력과 관계 그래프는 효과가 없다고 판정된다. 구조는 학습 경기 약 700개 이상에서만 이득이다.
4. **데이터: snapshot 추론에서 감사된 사건 기반 라벨과 공개 다조건 데이터로**
   - 선행: 비공개 단일 조건 데이터. 라벨 감사가 없다.
   - 우리: 공개 13,948경기, 12개 코호트(맵 × 시점 × 팀 규모), 사건 기반 라벨, 시계 정렬, 봇 위주 경기 제외.
   - 근거: snapshot 생존 추론은 팀-단계 행의 11.4%를 오분류한다.

### 의의 (논의·결론 절의 메시지)

- **설계 우선순위**: 추정 대상을 의사결정 시간 단위(다음 단계)에 맞추는 선택의 효과는 +0.05로, 아키텍처 효과의 약 25배다. 반대로 경기 최종 순위가 목표라면 사망 시각 목적이 낫다 (KRAFTON의 top-5 우위).
- **표현 우선순위**: 위험 정보는 서식지 손실 변수에 있다. 깊은 관계 구조가 이를 대신하지 못한다 (κ = 0.29).
- **연구 방법**: BR 예측기들 사이의 성능 차이는 아키텍처가 아니라 정식화와 표현에서 나온다. 통제된 분해 없이는 이 차이가 보이지 않는다.

### 새롭다고 주장하지 않는 것 (심사 대비)

생존분석의 BR 적용(KRAFTON), 팀 간 attention(KRAFTON), 단계 기반 예측(Sagong), 관계 bias attention(Graphormer 계열)

### 서론 원고 초안 (영문, 공백 단락 + 기여)

> **Gap.** Prior battle royale predictors are hard to compare because they differ along several design axes at once. Sagong et al. classify individual players at half-phase boundaries from activity scores, observing other players only through a head-count within 100 m. KRAFTON's predictor ranks squads by future time of death with a Cox objective and converts the scores to win probabilities, using game phases only for sampling and evaluation. Neither treats the shrinking safe zone as what it mechanically is, a habitat that is lost step by step. The two also report different metrics on different, unreleased data. Which design choice actually determines how well elimination can be predicted therefore remains unknown.
>
> **Contributions.** (i) We formulate elimination as a phase-conditioned discrete-time hazard, the local-extinction risk at each step of habitat loss. Within the same backbone and inputs it improves next-phase AUC by 0.051 over a time-of-death Cox objective, enough for KRAFTON's backbone to match our reference model. (ii) We represent each team's habitat context. With the learner fixed, it adds 0.091 AUC over phase-wise activity indicators. (iii) We decompose the gap one factor at a time on identical public data. Architecture and relational structure add about 0.002, and only data-rich cohorts benefit from them. (iv) We release audited event-based labels for 12 public cohorts, after showing that snapshot-based alive inference mislabels 11.4% of rows.

공개 범위(iv)는 미정이다.


---

## 동기 ("왜 해야 하는가") 근거 수치 (2026-10-06, 탐색적·기술적)

- **계산 방법**: KRAFTON-tf 점수를 생존 팀끼리 softmax해서 우승확률을 만들었다. 온도는 val에서 우승 팀 NLL이 최소가 되도록 맞췄다. 평가는 12개 코호트의 test 단계 경계에서 했다.
- **비교 대상**: 같은 행에서 M11R이 예측한 다음 단계 hazard의 분포.

| 단계 | 생존 팀 수 (중앙값) | 우승확률이 2/N 미만인 팀 비율 | 1위 팀 우승확률 (평균) | hazard 사분위 범위 (M11R) | hazard > 0.5인 팀 비율 |
|---|---|---|---|---|---|
| 1 | 16 | 94% | 0.17 | 0.08–0.29 | 12% |
| 2 | 13 | 93% | 0.22 | 0.08–0.28 | 11% |
| 3 | 11 | 91% | 0.28 | 0.11–0.38 | 16% |
| 4 | 8 | 89% | 0.36 | 0.16–0.55 | 30% |
| 5 | 6 | 88% | 0.48 | 0.18–0.60 | 34% |
| 6 | 4 | 89% | 0.58 | 0.26–0.74 | 49% |
| 7 | 3 | 93% | 0.65 | 0.33–0.78 | 54% |

**해석**
- 모든 단계에서 약 10팀 중 9팀은 우승확률이 균등 기준의 2배 미만이다. 이런 팀들에게 우승확률은 사실상 정보가 없다.
- 같은 팀들의 다음 단계 hazard는 넓게 퍼져 있다. 사분위 범위가 단계 1에서 0.08–0.29, 단계 4에서 0.16–0.55다.
- 따라서 단계마다 "지금 누가 위험한가"를 가르는 양은 hazard이고, 우승확률은 소수 선두 팀에만 집중된다.

**서론 동기 단락 v2 (영문)**

> In battle royale games, decisions are made one safe zone at a time. Players and coaches decide whether to rotate, hold or fight before the next shrink, broadcasters decide which of dozens of teams is in danger now, and designers tune each phase's pressure. All of them need the probability that a team is eliminated before the next phase. The tool built so far is win probability, and it answers a different question: in our data, about nine in ten surviving teams hold less than twice the uniform win probability at every phase, while their next-phase elimination risk spans an interquartile range of 0.08–0.29 in the first phase and 0.16–0.55 by the fourth. A model trained for win probability also misstates next-phase risk by 7.4 percentage points on average.
