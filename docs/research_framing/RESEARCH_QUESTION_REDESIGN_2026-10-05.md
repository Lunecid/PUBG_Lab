# 연구질문 재설계 (2026-10-05)

상태: **저널 확장용 참고**. BigComp 2027 논문은 `docs/experiment_design/BIGCOMP2027_PTR_STUDY_DESIGN_2026-10-05.md`를 따른다. 학위논문에는 쓰지 않기로 했다(2026-10-05 사용자 결정).

이 문서의 목적은 새 실험을 돌리기 전에 질문, 예측, 판정 기준을 고정하는 것이다. 지금까지 관찰한 결과는 모두 **탐색적 근거**로만 쓴다. 이 결과를 보고 만든 가설을 같은 데이터로 "확증"했다고 쓰지 않는다.

최신 학위논문 원고는 저장소 밖 `/Users/lunecid/학위논문`에 있다(git 관리 안 됨). 저장소의 `paper/`는 6월 시점의 이전 원고다.

---

## 0. 요약

1. KRAFTON `pubg-winprob`가 "PUBG + 생존분석 + 전체 squad 동시 고려"를 이미 구현해 공개했다. 공개된 것은 코드뿐이고 논문 서지는 아직 확인되지 않았다. 따라서 문제 설정 자체를 새롭다고 주장할 수 없다.
2. 지금까지 결과 중 견고한 것은 **안전구역·경계거리·국지 밀집(habitat) 정보**의 효과뿐이다. 시간 순서, 명시적 그래프, 모델 구조(Mamba/GNN)의 효과는 약하거나 없다. 149차원 HGB가 13개 코호트 중앙값에서 모든 신경망 MR 모델보다 높다.
3. 그래서 중심 질문을 바꾼다. 이전 질문은 "어떤 모델이 좋은가"였다. 새 질문은 "KRAFTON식 현재 상태 snapshot만으로 충분한가, 아니면 어떤 정보가 추가 정보를 주는가"이다. 동등성 검정과 음성 대조군을 미리 정해 두어, 결과가 양성이든 음성이든 해석할 수 있게 한다.
4. 학위논문의 4-RQ 골격은 유지한다. RQ1과 RQ2에는 검증 가능한 분석을 더한다. RQ3는 정보 충분성 질문으로 다시 쓴다. RQ4에는 KRAFTON식 우승확률과의 비교를 넣는다.

---

## 1. 고정된 관찰 (탐색적)

약어: SEQ = `docs/experiment_design/BATTLE_ROYALE_RQ4_SAME_CONDITION_SEQUENCE_RESULT_2026-07-05.md`, LED = `.../BATTLE_ROYALE_EXPERIMENT_EVIDENCE_LEDGER_2026-07-05.md`, SYN = `.../BATTLE_ROYALE_RESULT_SYNTHESIS_2026-07-10.md`, REP = `.../REPRESENTATION_ABLATION_STATUS_2026-06-14.md`, CK = `model_stgcn/checkpoints`. 지표는 다음 단계 탈락 hazard AUC이고, "n/13"은 차이가 양수인 코호트 수다. **(집계)** 표시는 체크포인트 CSV에서 에이전트가 계산한 값이다. 이 값은 문서에 없으므로 인용 전에 다시 계산해야 한다.

| ID | 관찰 | 수치 | 해석 상태 | 출처 |
|---|---|---|---|---|
| O1 | 자기 팀 상태 궤적 요약이 마지막 snapshot보다 낫다 (HGB) | 중앙값 0.7411 → 0.7550, 코호트별 차이 중앙값 +0.0074 (12/13) | 지지. 단, 두 arm 모두 누적 사건 통계와 다른 팀 정보가 없다 | SEQ:219-229 |
| O2 | 순서를 쓰는 신경망은 순서 없는 요약 HGB보다 낮다 | SUMMARY_HGB − Transformer +0.0090 (11/13), − LSTM +0.0159 (11/13) | "순서 > 요약"의 근거 없음. 모델 계열 교란 | SEQ:97-106 (집계) |
| O3 | 신경망 이력 모델 > 마지막 snapshot MLP | Transformer 0.7471 vs 0.7213 | 약함. HGB-마지막 snapshot과 비교하면 −0.0007 (6/13) | SEQ:67-106 |
| O4 | 시간 섞기 대조 | 0.8422 → 0.5124 (추론 단계에서만 섞음, gen-1 모델) | 순서 정보의 증거가 아니다. 분포 이동에 따른 붕괴일 가능성이 크다 | REP:61-102 |
| O5 | 안전구역·경계·habitat 블록 | OM2→OM3 +0.0103 (13/13). HGB +habitat (공식 원자료 대비) +0.0192 (13/13, 12개 p<.05) | 지지. 구성요소 분리 실험은 하지 않았다 | LED:421-430, SYN:167-177 (집계) |
| O6 | 명시적 관계 vs edge shuffle | +0.0023 (방향 13/13, CI 하한>0 6/13) | 약함 | `BATTLE_ROYALE_O7_REQUIRED_COHORT_HAZARD_RESULT_2026-07-05.md`:116-125 |
| O7 | 구조 비교 (MR, 13개 코호트 중앙값) | GraphSAGE 0.7639, R-GCN 0.7625, FARNet 0.7599, Mamba 0.7595, MR-TCN 0.7560, **HGB-149d 0.7753** | 구조 간 차이 없음. HGB가 가장 높다 | CK/full_required_core13_v1 (집계) |
| O8 | 같은 단계 팀 간 attention (Erangel, `mr_rgcn_attn_rank` vs `mr_rgcn`) | hazard AUC 0.7963 = 0.7963, rank Spearman 0.6897 vs 0.6600 | 상대 순위에는 기여하고 절대 위험에는 기여하지 않는다. rank loss와 교란된다 | `model_stgcn/v3/models.py`:272-327 (집계) |
| O9 | 보정·순위 지표 | Brier/ECE 차이는 seed 편차 안. C-index, IBS는 계산하지 않았다 | 미실시 | — |

---

## 2. 선행연구 경계 (검색일 2026-10-05)

검색 범위: WebSearch, GitHub API, arXiv 검색 페이지, Crossref, OpenAlex, Semantic Scholar, DBLP. 출판사 페이지 일부는 403으로 열리지 않았다.

| 문헌 | 관련성 | 확인 수준 |
|---|---|---|
| KRAFTON AI, *Real-Time Win Probability Prediction in Battle Royale Games via Survival Analysis*, github.com/krafton-ai/pubg-winprob (commit 99ed9d1, 마지막 push 2026-08-26) | squad 단위, 한 시점 입력, 최대 16 squad self-attention, 누적 행동 28개 + 상태 특징, Cox/weighted-Cox, 우승확률. 이산시간 hazard는 LogisticHazard 기준선으로만 등장한다 | 코드 확인. **논문 서지(arXiv, 학회, 저널)는 찾지 못함** |
| Sagong, Woo, Kim (2025), Entertainment Computing 55:101040 | 단계별 FightScore/ActiveScore, 실시간 생존·승리 분류 | 메타데이터 확인. 이미 인용 중 |
| Sagong & Kim (2025), *Survival Sequences: Win Prediction from a Strategy Sequence Approach*, IEEE BigComp, pp.105-106, doi:10.1109/bigcomp64353.2025.00030 | 단계별 전략 시퀀스를 LSTM/GRU로 학습, 우승 정확도 88%. snapshot 기준선은 보고하지 않았다 | 초록 확인. **원고 인용 여부 확인 필요.** 시간 이력 주장과 겹치므로 인용해야 한다 |
| Ringer et al. (2023), Machine Learning with Applications 100466 (Dota 2) | 단일 프레임 F1 .54, GRU 15-step .62, 3-step .32, 30-step .45. 이력의 효과가 설계에 따라 달라진다 | PDF 확인 |
| Liu & Schulte (2018), IJCAI (아이스하키) | LSTM trace 길이 10 vs 1 비교. 대부분 지표에서 이력이 도움이 된다 | PDF 확인 |
| Lee et al. (2018) DeepHit; Lee et al. (2020) Dynamic-DeepHit; van Houwelingen (2007) landmarking; Gensheimer & Narasimhan (2019) Nnet-survival; Ren et al. (2019) DRSA | 이산시간·동적 생존분석 방법론 | 초록 또는 메타데이터 확인 |
| Xenopoulos & Silva (2021); Wang et al. (2024) TacticAI | 스포츠·e스포츠 관계 그래프 모델 | 확인 |

이번 검색 범위 안에서 **찾지 못한** 것은 두 가지다.

- (b) 배틀로얄에서 "현재 상태 + 누적 통계" 대비 순서 있는 이력의 추가 정보를 검정한 연구
- (c) 배틀로얄 생존 모델이 맵, 모드, 시점(TPP/FPP), 팀 규모 사이에서 전이되는지 본 연구

원고에는 "선행연구가 없다"가 아니라 "검색 범위 안에서 확인되지 않았다"라고 쓴다.

### 피해야 할 주장

- PUBG에 생존분석을 처음 적용했다.
- 실시간 또는 경기 중 생존 예측이 새롭다.
- 시간 이력이 핵심이다. 순서 효과는 아직 검정하지 않았다.
- 그래프 모델이나 Mamba가 우수하다. O7과 모순된다.
- 안전구역 압박이 탈락을 **일으킨다**. 모든 결과는 예측 정보에 관한 것이고 인과 주장이 아니다.
- KRAFTON 결과를 재현했다. 원 데이터, 분할, 체크포인트가 공개되지 않았다. 우리가 할 수 있는 것은 "KRAFTON식 구조를 우리 과제에 맞춘 적용(task adaptation)"이다.

---

## 3. 중심 질문

> **KO.** 배틀로얄에서 팀의 다음 단계 탈락 위험을 예측할 때, 전체 팀의 현재 상태와 누적 통계(KRAFTON식 snapshot)만으로 충분한가? 충분하지 않다면 어떤 정보가 추가 예측 정보를 주는가?
>
> **EN.** Is a cross-team current-state snapshot with cumulative statistics sufficient for predicting next-phase elimination in battle royale games? If not, which information sources add predictive information?

이 질문을 고른 이유는 네 가지다.

1. 가장 가까운 선행연구(KRAFTON)의 설계 가정을 직접 검정한다.
2. 어느 결과가 나와도 해석할 수 있다. snapshot이 충분하다면 snapshot 기반 시스템을 정당화하는 근거가 된다. 충분하지 않다면 무엇이 빠져 있는지 알려준다.
3. 가장 강한 결과(공간 제약)와 약한 결과(순서, 그래프)를 같은 프로토콜 안에서 그대로 보고할 수 있다.
4. 기존 학위논문 RQ3 문장이 이미 "단계별 상태, 이전 시점의 기록, 팀 관계"라는 세 정보원을 나열한다. 모델 선택 질문을 정보원 질문으로 바꾸는 것이 자연스럽다.

### 이력의 두 종류

KRAFTON의 누적 행동 통계는 **사건 이력**(kill, damage, heal의 누적 횟수)이다. 우리가 추가로 시험할 것은 **상태 궤적**이다. 상태 궤적은 위치, 체력, 구역 거리가 시간에 따라 어떻게 변했는지를 말한다. 이 구분을 원고 전체에서 일관되게 쓴다. "시간 이력이 도움이 된다"는 문장은 둘 중 어느 쪽인지 밝히지 않으면 쓰지 않는다.

---

## 4. 연구질문

### RQ1. 위험의 시간 단위 (정식화)

> **KO.** 배틀로얄 탈락 과정을 안전구역 단계를 위험 구간으로 하는 이산시간 생존 과정으로 정식화할 수 있는가? 이 단계 기준 시간축은 경기 시계 기준보다 맵과 모드에 걸쳐 위험 구조를 더 일관되게 정렬하는가?
>
> **EN.** Can battle-royale elimination be formalized as a discrete-time survival process whose risk intervals are safe-zone phases, and does this phase clock align risk across maps and modes more consistently than elapsed game time?

- 주장 유형: 기술적·연관적.
- 후보 가설 H1: 탈락 위험은 경기 시계보다 안전구역 단계에 맞춰 구조화되어 있다.
- 경쟁 설명:
  - 맵마다 단계 시각이 고정되어 있다. 그렇다면 맵 안에서는 단계 정렬과 시계 정렬을 구별할 수 없다. 구별은 맵 사이 비교에서만 가능하다.
  - 위험 변화는 생존 팀 수(risk set 크기)의 감소 때문일 수 있다.
- 예측:
  - **P1.1** 같은 자유도에서 단계 구간 기저위험 모형의 로그우도가 균등 시계 구간 모형보다 높다.
  - **P1.2** 단계로 정렬한 위험 곡선은 시계로 정렬한 곡선보다 코호트 간 거리(예: Jensen–Shannon)가 작다.
  - **P1.3** 생존 팀 수를 공변량으로 넣어도 단계 효과가 남는다.
- 반증 조건: P1.2에서 차이가 없거나, 시계 정렬이 더 일관적이다.
- 데이터와 비용: 기존 사망 시각, 단계 경계, 라벨로 계산할 수 있다. 1~2일.

### RQ2. 표현과 조건 간 일반화

> **KO.** PUBG 공식 기록으로 구성한 단계 조건부 표현은 13개 조건(맵 × 시점 × 팀 규모)에서 다음 단계 탈락을 구분하는가? 그리고 조건 사이에서 전이되는가?
>
> **EN.** Does the phase-conditioned representation built from official PUBG records discriminate next-phase elimination across 13 conditions (map × perspective × team size), and does it transfer between conditions?

- 주장 유형: 예측적.
- **P2.1** (기존 결과, 탐색적) 코호트별 판별력. 이미 보고했다.
- **P2.2** (신규) cross-cohort 전이. A로 학습하고 B로 평가하는 13×13 행렬을 HGB로 만든다.
  - 예측: 팀 규모가 같으면 전이 손실이 작고, 팀 규모가 다르면 커진다.
  - 예측: 단계 정규화 특징은 절대 시각·절대 좌표 특징보다 전이 손실이 작다.
  - 반증: 전이 손실이 팀 규모 일치 여부와 무관하다. 또는 정규화 특징에 이점이 없다.
- 경쟁 설명: 작은 코호트의 표본 크기, 맵 고유 지형.
- 비용: 2~3일. 문헌 검색에서 이 질문을 다룬 BR 연구는 찾지 못했다.

### RQ3. 정보 충분성 (핵심)

> **KO.** 전체 팀의 현재 상태와 누적 통계가 주어졌을 때, (a) 상태 궤적 이력, (b) 단계 조건부 공간 제약, (c) 명시적 팀 관계는 다음 단계 탈락 위험에 추가 예측 정보를 주는가? 상태 궤적의 정보는 순서에 의존하는가?
>
> **EN.** Given the current state and cumulative statistics of all teams, do (a) state-trajectory history, (b) phase-conditioned spatial constraints, and (c) explicit inter-team relations add predictive information about next-phase elimination? Does the trajectory information depend on temporal order?

- 주장 유형: 예측 정보량. 인과 주장이 아니다.

#### 정보 집합

| 집합 | 정의 | KRAFTON 대응 | 데이터 상태 |
|---|---|---|---|
| S0 | 자기 팀 현재 상태: 단계 경계 직전 10초 격자의 13개 변수(위치 x/y/z, 속도, 구역 상대위치, 경계거리, 구역 안 여부, 목표 중심 거리, 평균·최소 체력, 생존 인원, 차량 비율) + 단계 번호 | 부분 | 있음 (`stdb/snapshot_queries.py`:38-52) |
| S1 | S0 + 누적 사건 통계(kill, finish, 기절시킴, 준·받은 피해 횟수와 합, 획득, 회복 아이템 사용, 장비 수준) + 팀 상태(총 체력, 구역 밖 인원) + 150 m 안 아군·적 수 + 다음 구역 거리 | KRAFTON 55차원에 대응 | 도출 가능. drop, 회복량, 투척물, 부활은 저장되지 않아 근사하거나 뺀다 |
| S1× | S1을 같은 시점의 생존 팀 전체에 대해 masked self-attention으로 인코딩 | KRAFTON 구조 (mask와 Cox 부호 문제를 고친 버전) | 신규 구현 |
| S2 | S1× + 순서 없는 상태 궤적 요약(접두 구간과 현재 단계의 mean, min, max, std, last−first) | 없음 | 요약 특징은 있음 (SUMMARY arm) |
| S3 | S1× + 10초 상태 시퀀스의 인과 시퀀스 인코더 | 없음 | 신경망 코드는 있음 (`model_stgcn/snapshot_baselines.py`) |
| S3-perm | S3와 같은 입력을 창 안에서 무작위 순서로 섞어 **학습과 평가 모두** 수행 | — | 신규. 순서에 대한 음성 대조군 |
| +Z | 단계 조건부 공간 제약. 구성요소는 셋이다. Z-zone(축소 속도, 다음 축소까지 시간, 구역 간격, 다음 구역 부호 거리), Z-comp(100/200/500 m 적 수, 밀집도), Z-loc(절대 위치, 지형 고도) | 부분 (구역 거리, 150 m 적 수) | 있음 (habitat, OM3) |
| +G | 유형이 있는 근접·교전 그래프의 message passing | 없음 | 있음. 대조군은 차수 보존 edge shuffle과 유형 shuffle을 **재학습**해서 쓴다 |

#### 주 대조

| 대조 | 비교 | 후보 가설 | 기존 근거 (탐색적) | 반증 조건 |
|---|---|---|---|---|
| C3.0 | S1× vs S1 | 다른 팀의 현재 상태가 위험 예측에 정보를 더한다 (KRAFTON의 핵심 가정) | O8: hazard에는 0, 순위에는 양수 | 동등성 성립 |
| C3.1 | S2 vs S1× | 상태 궤적 요약이 사건 이력 위에 정보를 더한다 | O1: +0.0074. 누적 통계가 들어가면 줄어들 것으로 본다 | 동등성 성립 |
| C3.2 | S3 vs S2, S3 vs S3-perm | 순서가 요약 이상의 정보를 준다 | O2: 근거 없음 | 두 대조 모두 동등성 성립 |
| C3.3 | S1× + Z vs S1× (구성요소별 포함) | 단계 조건부 공간 제약이 KRAFTON식 구역 특징 위에 정보를 더한다 | O5: +0.0103 ~ +0.0192. S1에 이미 구역 거리와 150 m 적 수가 있어 줄어들 수 있다 | 동등성 성립 |
| C3.4 | S1× + G vs S1× (재학습한 null 그래프 포함) | 명시적 관계가 학습된 attention 이상의 정보를 준다 | O6: +0.0023 | 관찰 그래프가 null 그래프와 동등 |

#### 설계

- 모든 arm은 같은 risk set, 라벨, 분할, 평가 행 집합을 쓴다.
- 신경망 골격은 하나로 고정한다. 구조는 팀 token 인코더 → 팀 간 masked transformer → 단계 hazard head다. arm마다 token 입력만 바꾸고 파라미터 수를 맞춘다.
- 표 형태 arm(S0, S1, S2, +Z)은 HGB로도 돌린다. 모델 계열 교란을 분리하기 위해서다.
- 순서 대조는 재학습으로 한다. 추론 단계에서만 섞으면 O4처럼 분포 이동을 측정하게 된다.
- 결과는 단계 구간(초반, 중반, 후반)별로도 보고한다.

#### 경쟁 설명과 대응

| 경쟁 설명 | 대응 |
|---|---|
| 모델 계열·용량 차이 | 같은 골격, 같은 파라미터 수. HGB 병행 |
| 미래 정보 누수 (위치 정렬 ±5초) | 선결 조건 1 |
| 분포 이동 | permutation 대조를 재학습으로 수행 |
| 코호트 이질성, 작은 테스트 집합 | random-effects 메타분석, τ² 보고 |
| 단계별 risk set 구성 차이 | 단계 구간별 보고 |
| 긴 경기 절단 (2000초) | 선결 조건 5 |
| 패치·시기 변화 | 새로 수집한 시간 holdout으로 확증 |

### RQ4. 상대 생존 판독

> **KO.** 단계별 위험을 누적한 상대 생존 지표는 직접 우승확률 추정(KRAFTON식)과 비교해 순위·우승 판별과 보정에서 어떤 성능을 보이는가? 팀 간 정보는 절대 위험과 상대 순위 중 어디에 기여하는가?
>
> **EN.** How does a relative survival index composed from phase hazards compare with direct win-probability estimation (KRAFTON-style) in ranking, winner discrimination, and calibration? Does cross-team information contribute to absolute risk or to relative standing?

- 주장 유형: 예측적.
- **P4.1** 팀 간 attention의 이득은 hazard AUC/NLL보다 상대 지표(순위 Spearman, 우승 정확도, 경기 내 C-index)에서 크다. 기존 근거는 O8이다.
- **P4.2** hazard로 만든 상대 생존 지표 π는 같은 입력의 직접 우승 head(팀 softmax + CE, 또는 부호를 고친 weighted Cox)와 비교해 우승 지표에서 SESOI 안에 든다. 동시에 단계별 hazard 보정을 제공한다.
  - 반증: π가 우승 지표에서 SESOI보다 크게 뒤진다. 이 경우 단계 분해가 순위 능력을 희생한다는 뜻이다.
- 지표:
  - 단계 hazard: NLL, AUC, Brier, ECE (단계별).
  - 상대 판독: 경기 내 C-index(탈락 순서), 우승 정확도, 우승 Brier.
  - 우승 Brier는 KRAFTON 코드의 "IBS"와 같은 정의다(시점마다 Σ_alive (p − 1[winner])²를 평균). 시간 적분 IBS와 다르다는 점을 원고에 명시한다.

---

## 5. 통계와 판정 규칙

- **주 지표**: ΔNLL. 정보 집합을 늘렸을 때 줄어드는 음의 로그우도이며, "예측 정보 이득"이라 부른다. 모델이 잘 맞으면 조건부 상호정보량에 근사한다. 보조 지표는 ΔAUC, ΔBrier, ECE다. AUC는 기존 결과와 비교하기 위해 계속 보고한다.
- **코호트 통합**: 코호트별 Δ와 경기 단위 bootstrap 표준오차로 random-effects 메타분석(REML)을 한다. τ²를 함께 보고한다. 방향이 같은 코호트 수는 보조로만 쓴다.
- **SESOI(의미 있는 최소 효과)**: ΔAUC 0.005에 대응하는 ΔNLL을 기존 habitat ablation에서 환산해 실행 전에 고정한다. 0.005는 관계 효과(+0.0023)의 약 두 배이고 habitat 효과(+0.0192)의 약 4분의 1이다. 최종값은 지도교수와 확인한다.
- **판정**:
  - 지지: 통합 Δ의 95% CI 하한이 0보다 크고, 점추정이 SESOI 이상이다.
  - 동등(추가 정보 없음): 90% CI가 ±SESOI 안에 있다(TOST).
  - 그 외는 불확정으로 보고한다.
- **다중성**: RQ3 주 대조 다섯 개에 Holm 보정을 한다.
- **반복**: 신경망은 seed 3개 이상. HGB는 seed를 고정하고 같은 설정의 반복 실행 편차를 측정한다.
- **확증 데이터**: 사전 등록 이후 새로 수집한 경기로 시간 holdout을 만든다. 주 대조를 이 데이터에서 한 번만 평가한다. 기존 데이터 결과는 탐색적이라고 표시한다.

---

## 6. 선결 조건 (새 실험 전)

1. **위치 정렬의 미래 참조.** 10초 격자에 가장 가까운 위치 표본을 앞뒤 5초 안에서 고른다(`stdb/etl.py`:176-183). 최대 5초의 미래 위치가 들어갈 수 있다. 직전 표본만 쓰도록 고치고 테스트를 추가한다. 현재 `tests/test_snapshot_features.py`:111은 이 경우를 다루지 않는다.
2. **HGB 반복 실행 편차.** 같은 설정으로 다시 돌렸을 때 코호트당 최대 0.0048 차이가 난다. 이 크기는 SESOI와 같으므로 원인을 먼저 없앤다.
3. **평가 행 집합 통일.** 표 형태 모델은 104,120행, 신경망·MR은 103,860행이다.
4. **arm 정의 검증.** 기존 OM0에는 snapshot tier가 포함되어 있었다. "no-graph" arm에서는 snapshot tier 그래프가 켜져 있었다. 새 사다리는 각 arm이 받는 입력을 코드에서 확인하는 검증 스크립트를 둔다.
5. **긴 경기 절단.** `SNAP_T_MAX=200`(2000초) 때문에 적격 경기의 7.0%, solo-FPP의 48%가 잘린다. 한도를 늘리거나 민감도 분석을 한다.
6. **gen-1과 gen-2 차이.** 같은 Taego squad에서 HGB 대비 MR 이득이 +0.0425에서 +0.0061로 줄었고 원인은 확인되지 않았다. gen-1 결과는 근거에서 계속 제외한다.
7. **재현성.** 현재 dirty 파일이 118개이고 BR 문서 대부분이 untracked다. 커밋하고 manifest를 고정한다.
8. **(집계) 값 재계산.** 1절의 (집계) 값을 스크립트로 다시 계산하고 문서에 기록한다.
9. **사전 등록.** 이 문서의 예측과 판정 규칙, SESOI를 `docs/experiment_design/`에 커밋한 뒤 새 arm을 실행한다.

---

## 7. 학위논문과 후속 논문 매핑

| 기존 학위논문 RQ (`01_introduction.tex`:108-111) | 새 RQ | 변경 |
|---|---|---|
| RQ1 장르 정식화 | RQ1 | 소. P1 분석을 근거로 추가 |
| RQ2 PUBG 표현 | RQ2 | 소~중. 전이 분석 추가 |
| RQ3 판별력과 적합한 모형 | RQ3 정보 충분성 | 대. MR-FARNet은 "제안 모형"에서 비교 구조 중 하나로 위치를 바꾼다 |
| RQ4 누적 생존 점수와 유리·불리 팀 | RQ4 상대 생존 판독 | 중. KRAFTON식 직접 우승 head와 비교, 지표 정의 명시 |

함께 고칠 것:

- 관련연구에 KRAFTON `pubg-winprob`와 Sagong & Kim (BigComp 2025)을 추가한다.
- 기여 문장에서 문제 설정의 새로움 대신 아래 세 가지를 말한다.
  - 단계 조건부 정식화와 그 근거(RQ1)
  - 13개 조건의 전이(RQ2)
  - 통제된 정보 충분성 검정(RQ3)

후속 저널 논문은 RQ3을 중심에 두고 RQ2 전이와 RQ4 요약을 붙인다. 가제는 *Is the Current State Enough? Information Sources for Phase-Conditioned Elimination Risk in Battle Royale Games*.

---

## 8. 실행 순서 (제안)

| 단계 | 내용 | 기간 | 선행 |
|---|---|---|---|
| M0 | 선결 조건 1~5, 7 | 2~3일 | — |
| M1 | 사전 등록 문서와 SESOI 고정 | 0.5~1일 | M0 |
| M2 | 확증용 새 경기 수집 시작 (백그라운드). PUBG API 경기 보존 기간(약 14일, 확인 필요) 때문에 달력 시간이 든다 | 2~3주 병행 | M1 |
| M3 | RQ1(단계 vs 시계), RQ2(13×13 전이), HGB | 2~4일 | M0 |
| M4 | S1 누적 사건 통계 ETL + 접두 구간 재생 누수 테스트 | 3~5일 | M0 |
| M5 | 팀 집합 데이터셋 + KRAFTON식 masked encoder + hazard head. 탐색용 quick probe(기존 `TeamSetDecoder`를 hazard head에 연결, 생존 팀 mask)는 1일이면 먼저 할 수 있다 | 3~5일 | M4 |
| M6 | RQ3 사다리: 13개 코호트 × arm × seed 3 | 계산 3~5일 | M1, M5 |
| M7 | RQ4 판독 비교 | 2~3일 | M6 |
| M8 | 확증 holdout 1회 평가, 원고 반영 | 2~3일 | M2, M6, M7 |

전체 4~6주. 학위논문 일정에 따라 M2와 M8을 저널 단계로 미룰 수 있다.

---

## 9. 결정이 필요한 항목

1. 학위논문 심사·제출 일정. 이 일정이 RQ3 재작성 범위와 M2, M8의 시기를 정한다.
2. KRAFTON 대응 전체 구현(1.5~2.5주)과 quick probe(1일) 중 무엇을 먼저 할지.
3. SESOI 값.
4. 확증용 새 데이터 수집을 시작할지.
