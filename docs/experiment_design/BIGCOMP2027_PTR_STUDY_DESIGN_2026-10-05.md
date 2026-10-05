# BigComp 2027 연구 설계: 단계·팀·관계 축 분해 (v2, 2026-10-05)

상태: **확정안 v2**. 실험을 실행하기 전에 고정한다. 바꿀 때는 날짜와 이유를 문서 끝 변경 기록에 남긴다.

- 대상: IEEE BigComp 2027 full paper. 8쪽 이하(참고문헌 포함), IEEE 2단, double-blind.
- 마감: 2026-10-16 AoE (한국시간 10/17 20:59). EasyChair(ieeebc27)에서 다시 확인한다.
- 원고: `paper/bigcomp2027/`
- 상위 문서: `docs/research_framing/RESEARCH_QUESTION_REDESIGN_2026-10-05.md` (저널 확장용)

---

## 1. 논문의 주장

새 모델이 기준선을 이긴다는 주장이 아니다. 같은 데이터, 라벨, 분할에서 정보 축을 하나씩 켜고 끄며 다음 단계 탈락 정보가 어디에 있는지 분리한다.

| 축 | 직접 선행 | 모델 구성요소 |
|---|---|---|
| 단계 축 | Sagong et al. 2025 (EC), Sagong & Kim 2025 (BigComp) | 팀별로 단계 1..k에 걸친 causal attention |
| 팀 축 | KRAFTON `pubg-winprob` (팀 간 attention, `--encoder mlp` ablation 스위치 보유) | 같은 단계 생존 팀 전체에 대한 masked attention |
| 관계 | 이종 상호작용 그래프 (HIGFormer, Graphormer) | 관계 유형별 attention bias |
| 안전구역 객체 | 찾지 못함 | 단계마다 안전구역을 별도 객체 노드로 둠 |

이름 붙인 모델은 모든 축과 안전구역 객체를 켠 구성이다. 작업 이름은 **PTR-Former**이고 바꿀 수 있다.

### 노벨티 위치 (2026-10-05 검증)

- 구성요소별로는 모두 선행이 있다. 아래 표현은 쓰지 않는다.
  - "새 구조"
  - "팀 간 attention의 첫 검정" (KRAFTON 코드에 이미 attention vs MLP 스위치가 있다)
- 주장할 것은 네 가지다.
  1. 다음 단계 hazard에서 축을 통제해 분해했다.
  2. BR 고유 관계를 그래프로 표현했다: 교전 격화, 구역 공동압박, 팀원.
  3. 줄어드는 안전구역을 명시적 객체로 표현했다. 이 부분은 선행을 찾지 못했다.
  4. 공개 경기 13개 조건과 고정 분할을 썼다.
- 문헌 공백은 "검색 범위 안에서 찾지 못했다"로만 쓴다.
- Sagong et al. 2025 원문 확인 결과 (2026-10-05):
  - 분석 단위는 플레이어이고 솔로 경기만 썼다. 팀 전략 영향을 이유로 듀오·스쿼드를 제외했다.
  - 라벨은 다음 반 단계 생존 여부다(이진 분류). 생존분석 방법은 쓰지 않았다.
  - 다른 플레이어에 관한 정보는 100 m 안 인원수라는 스칼라 하나뿐이다.
  - 데이터는 비공개 1주, 7,719경기다.
  - 팀 협동 변수와 시퀀스 모델을 향후 과제로 남겼다.
  - 따라서 우리의 확장 방향은 네 가지다. 팀 단위와 팀원 관계, 이산시간 hazard, 스칼라 밀도(Sagong 100 m, KRAFTON 150 m)에서 유형 있는 관계로의 확장, 공개 13개 조건이다.
  - 그들의 정확도 87.90%와는 비교하지 않는다. 라벨의 약 85%가 양성이고 데이터도 다르다.

---

## 2. 연구질문

| RQ | EN | KO |
|---|---|---|
| RQ1 단계 축 | Does encoding each team's phase-by-phase history improve next-phase elimination prediction beyond its current-phase state? | 팀의 단계별 이력이 현재 단계 상태 이상의 정보를 주는가? |
| RQ2 팀 축 | When KRAFTON-style cross-team attention is applied to next-phase hazard, does it improve absolute elimination risk, within-match relative standing, or both? | KRAFTON식 팀 간 attention을 다음 단계 hazard에 쓰면 절대 위험과 상대 순위 중 무엇이 개선되는가? |
| RQ3 관계 | Do explicit battle-royale relations (proximity, combat escalation, zone co-pressure, and teammates) add information beyond learned cross-team attention? | 명시적 BR 관계(근접, 교전 격화, 구역 공동압박, 팀원)는 학습된 팀 간 attention 이상의 정보를 주는가? (유형별 기여는 탐색적으로 분석한다) |
| RQ4 안전구역 객체 | Does representing the shrinking safe zone as an explicit object node recover the information carried by hand-crafted zone and habitat features? | 안전구역을 명시적 객체 노드로 넣으면 수작업 구역·habitat 특징이 담던 정보를 회복하는가? |

모든 주장은 예측 정보에 관한 것이며 인과 표현을 쓰지 않는다. RQ3의 팀원 관계는 2단계(플레이어 계층)에서만 시험한다.

---

## 3. 그래프 설계

모든 노드 특징과 엣지는 **단계 k 경계의 10초 전까지**의 표본만 쓴다(Fix B, 6절).

### 1단계: 팀 수준 (확보용)

- **팀 노드**: 단계 시작 시 생존한 팀. 특징은 phase tier 150차원과 경계 시점 팀 중심 좌표(정규화)다.
- **안전구역 객체 노드**: 단계마다 1개. 특징은 원시 구역 기하뿐이다.
  - 현재 구역 중심과 반경
  - 다음 구역 중심과 반경
  - 단계 경과 비율, 단계 번호
  - 구역 정보를 미리 가공한 특징(경계거리, 필요 속도 등)은 넣지 않는다. RQ4가 이 정보를 모델이 스스로 회복하는지 묻기 때문이다.
- **팀–팀 엣지** (기존 `A_p`):
  - 근접: 대칭, `exp(−평균 거리/200 m)`
  - 교전: 방향(공격→피격), `ln(1 + damage + 100·kills + 50·groggy)`. 같은 쌍에 교전이 있으면 근접 대신 교전 유형으로 바꾼다(교전 격화). 교전 bias는 근접 bias와 별도로 학습한다.
  - 구역 공동압박: 대칭. 두 팀이 모두 구역 밖이거나 경계 안쪽 150 m 이내에 있고 거리가 200 m 이하인 비율로 가중한다.
- **팀–객체 엣지**: 팀 중심의 구역 상대 위치에서 계산하는 연속 bias다. 입력은 (중심 − 구역 중심)/반경과 거리/반경으로, 현재·다음 구역 각각에 대해 구한다.

### 2단계: 플레이어 계층 (10/9 18:00까지 검증되면 포함)

- **플레이어 노드**: 단계 시작 시 생존한 플레이어. 특징은 다음과 같다.
  - 경계 시점 위치(팀 중심 기준, 맵 기준)와 체력
  - 단계 안에서 준·받은 피해
  - 기절시킴, kill
  - 누적 kill, 회복 아이템 사용 수
  - 팀 특징 150차원을 함께 붙인다.
- **엣지**:
  - 팀원: 같은 팀 생존자 사이, 대칭, 거리 특징 포함
  - 적 근접: 다른 팀 플레이어 사이, 반경 300 m 안, 거리 구간별
  - 교전: 공격자→피격자, 같은 쌍의 근접을 대체하는 격화 유형
  - 객체: 플레이어–구역
- **흐름**: 플레이어 토큰 → 관계 bias attention → 팀별 attention pooling → 단계 축 → hazard.

### 결합 방식

관계 bias attention (Graphormer식)을 쓴다.

`logit(i,j) = q_i·k_j/√d + Σ_r b_{r,h} · w_r(i,j)`

b는 head와 block마다 학습한다. 팀 축을 끄면 attention이 자기 자신으로 제한된다. 플레이어 계층에서는 같은 팀원으로 제한된다.

---

## 4. 실험 arm

모든 신경망 arm은 같은 골격, 같은 손실(hazard BCE + 0.1 × pairwise 순위 손실), 같은 학습 설정을 쓴다. hazard head는 attention을 거친 표현을 읽는다.

| arm | 단계 축 | 팀 축 | 관계 bias | 구역 객체 | 특징 |
|---|---|---|---|---|---|
| M00 | 끔 | 끔 | 끔 | 끔 | 전체 |
| M10 | 켬 | 끔 | 끔 | 끔 | 전체 |
| M01 | 끔 | 켬 | 끔 | 끔 | 전체 |
| M11 | 켬 | 켬 | 끔 | 끔 | 전체 |
| M11R | 켬 | 켬 | 켬 | 끔 | 전체 |
| M11R-shuf | 켬 | 켬 | 경기·단계 안에서 팀 순열로 섞음, 재학습 | 끔 | 전체 |
| M11RZ (PTR-Former) | 켬 | 켬 | 켬 | 켬 | 전체 |
| M11R-noHab | 켬 | 켬 | 켬 | 끔 | 구역·habitat 계열 제거 |
| M11RZ-noHab | 켬 | 켬 | 켬 | 켬 | 구역·habitat 계열 제거 |
| M11RZ-P (2단계) | 켬 | 켬 | 켬 + 팀원 | 켬 | 전체 + 플레이어 |

참조 기준선:

- HGB-cut: 단계 k의 150차원 행
- HGB-lag: HGB-cut + 직전 단계 값과 차분
- HGB-noHab: HGB-cut에서 구역·habitat 계열을 제거

주 대조:

| RQ | 대조 |
|---|---|
| RQ1 | M10 − M00, M11 − M01 |
| RQ2 | M01 − M00, M11 − M10. hazard 지표와 순위 지표를 따로 본다 |
| RQ3 | M11R − M11, M11R − M11R-shuf. 대형 코호트 2개에서는 유형별 제거도 한다 |
| RQ4 | 회복률 = (M11RZ-noHab − M11R-noHab) / (M11R − M11R-noHab). M11RZ − M11R도 함께 본다 |
| 2단계 | M11RZ-P − M11RZ |

- seed는 모든 arm과 코호트에 42를 쓴다.
- Erangel TPP squad와 Taego TPP squad에서는 M00, M11, M11R, M11R-shuf, M11RZ에 seed 13과 2026을 추가한다.

### 4.1 분석 규칙 보충 (v2.1)

**주 대조 5개.** Holm 보정은 이 다섯 개에만 적용한다.

| # | RQ | 주 대조 | 지표 |
|---|---|---|---|
| P1 | RQ1 | 단계 축 주효과 = ½[(M10 − M00) + (M11 − M01)] | hazard AUC |
| P2 | RQ2 | 팀 축 주효과 = ½[(M01 − M00) + (M11 − M10)] | hazard AUC |
| P3 | RQ2 | 팀 축 주효과 (P2와 같은 식) | 순위 Spearman |
| P4 | RQ3 | M11R − M11R-shuf | hazard AUC |
| P5 | RQ4 | M11RZ-noHab − M11R-noHab | hazard AUC |

보조 대조는 Holm 보정을 하지 않고 탐색적으로 보고한다.
- 단순 효과
- 2×2 상호작용 = (M11 − M10) − (M01 − M00)
- M11R − M11
- M11RZ − M11R
- 나머지 지표

**회복률 κ.** κ = (M11RZ-noHab − M11R-noHab) / (M11R − M11R-noHab)는 서술용이다. 분모의 통합 95% CI 하한이 SESOI 이상일 때만 κ와 bootstrap CI를 보고하고, 그렇지 않으면 "추정 불가"라고 쓴다. RQ4 판정은 P5로만 한다.

**관계 유형별 제거 (탐색적).** Erangel TPP squad와 Taego TPP squad에서 seed 42로 M11R−prox, M11R−combat, M11R−copress를 돌린다.

**seed.** 모든 대조는 seed 42로 계산한다. 추가 seed(13, 2026)는 안정성 확인용이며 seed 간 표준편차만 보고하고 통합 추정에는 넣지 않는다.

**관계 가중치와 구역 객체 세부.**
- 관계 가중치는 행 정규화 없이 원시값을 쓴다.
  - prox = exp(−d/200)
  - combat_out과 combat_in = log1p(원시 가중치)
  - copress = 기존 ETL 값 (비율 × exp(−d/200))
  - 같은 쌍에 교전이 있으면 그 쌍의 prox는 0으로 둔다.
- 구역 bias는 2층 MLP다. 구조는 6 → 16 → H.
  - 입력은 현재·다음 구역 각각에 대해 (중심 − 구역 중심)/반경 2차원과 거리/반경 1차원이다.
- 구역 토큰은 팀 축 attention에서 query이자 key이고, block마다 갱신된다. 단계 축 attention에는 넣지 않는다.
- 팀 중심 좌표는 2차원이다.

**2단계 세부.**
- 플레이어 계층도 생존 플레이어 전체에 attend한다. 엣지는 bias로만 작용한다. 적 근접 bias는 300 m 안에서만 준다.
- 팀원 관계를 따로 떼어 보기 위해 M11RZ-P−tm(팀원 bias만 제거)을 추가한다. RQ3의 팀원 대조는 M11RZ-P − M11RZ-P−tm이다.
- 2단계를 하지 않으면 RQ3 문장에서 "teammates"를 지운다.

**기타.**
- top-10 라벨은 쓰지 않는다.
- HGB-lag에서 k = 1의 직전 단계 값은 NaN으로 둔다. HGB는 결측값을 그대로 처리한다. 신경망은 표준화 후 0으로 채운다.
- 인용 추가: HIGFormer (arXiv:2507.10626), DerSimonian & Laird (1986), Holm (1979).

**판정과 Holm (v2.2).**
- 개선: 주 대조의 random-effects 통합 z 검정(단측, H0: Δ ≤ 0)에서 Holm 보정 p < 0.05이고, 통합 점추정이 SESOI 이상이다.
- 동등: 통합 90% CI가 ±SESOI 안에 있다 (TOST, 대조마다 따로 보며 보정하지 않음).
- 그 외는 불확정이다.
- 표에 보이는 95% CI는 보정하지 않은 값이다.

**세부 확인 (v2.2).**
- RQ3의 확증 검정은 P4 하나다(관계 전체 vs 섞기). 유형별 기여는 탐색적으로만 본다. combat을 제거하면 combat_out과 combat_in을 함께 제거한다.
- 구역 토큰이 query일 때의 bias는 같은 구역 MLP 출력을 대칭으로 쓴다. 구역 MLP는 block마다 따로 둔다.
- 관계 bias의 대각 성분은 0이다. 자기 자신에 대한 attention은 허용한다.
- noHab은 spec의 zone·habitat 그룹을 뺀다. 단, 생존 인원 계열 변수는 공간 특징이 아니므로 남긴다.
- 순위 손실은 pairwise logistic이다. 최종 순위가 더 좋은 i와 j의 쌍마다 log(1 + exp(−(s_i − s_j)))를 평균한다. s가 클수록 생존에 유리하다.
- 신경망 입력은 train 토큰으로 맞춘 특징별 z-score로 표준화하고, 표준화 뒤 NaN을 0으로 채운다.

### 4.2 데이터·라벨 수정 (v2.3, 새 모델 결과를 보기 전에 결정)

근거 문서:
- `docs/data_layer/BIGCOMP2027_DATA_PROFILE_2026-10-05.md`
- `docs/data_layer/BIGCOMP2027_LABEL_AUDIT_2026-10-05.md`

**라벨 v2.**
- 기존 라벨(`team_phase_labels.hazard_next`)은 팀의 마지막 (팀, 초) snapshot에 건강한 플레이어가 있는지로 정해진다. 이 규칙은 학습 행의 11.4%를 틀리게 만든다. 양성 정밀도는 70.4%, 재현율은 75.2%다.
- 새 정의는 다음과 같다.
  - T = 팀원별 마지막 kill 사건 시각의 최댓값 − started_at (LogMatchStart 시계). 경기 종료 시 생존자가 있으면 T = ∞ (우승 팀 censoring).
  - target_mask(k) = T > phase_end_s(k)이고 단계 k+1이 존재한다.
  - y(k) = T ≤ phase_end_s(k+1). 단계 k+1이 경기의 마지막 단계이면 그 구간은 경기 종료까지로 본다.
- 공식 `time_survived`는 AI와 부활한 플레이어에서 쓸 수 없으므로 정답으로 쓰지 않는다.
- 보정 후 단계별 탈락률(k = 1..7)은 0.208, 0.211, 0.280, 0.393, 0.392, 0.506, 0.557이다.

**단일 시계.**
- 단계 경계는 LogMatchStart 시계를 쓰고, 위치 표본은 elapsed 시계를 쓴다. 두 시계는 δ만큼 어긋난다 (중앙값 5.7초, 최대 41.9초).
- 경기마다 δ를 추정해 두 시계를 하나로 맞춘 뒤 Fix B의 −10초 상한을 적용한다. 원래 설계대로 하면 약 27%의 경기에 누수가 남는다.

**생존 인원.** `team_phase_state.alive_players_*`에서 나온 특징은 사건 기반 생존 인원(단계 시작·종료 시점)으로 바꾼다.

**코호트별 학습.** 코호트는 맵 × 시점(TPP/FPP) × 팀 규모(솔로·듀오·스쿼드)로 나눈다. 모든 모델(신경망 arm과 HGB)은 코호트마다 따로 학습하고 평가한다. 코호트를 합쳐 학습하지 않으며, 코호트를 묶는 것은 random-effects 통합 단계뿐이다.

**모집단.**
- match_type이 airoyale인 경기를 제외한다.
- Vikendi 코호트를 제외한다.
- 남은 경기는 기존 분할 배정을 그대로 쓴다.
- 제외 후 200경기 미만인 코호트는 뺀다.
- 팀 특징에 AI 계정 비율(`team_ai_frac`)을 더한다.
- 결과는 match_type(official, competitive)별로도 보고한다.

**이전 결과.** 기존 라벨로 계산한 과거 결과(학위논문 포함)는 이 논문에서 비교 근거로 쓰지 않는다.

---

## 5. 지표와 판정

- hazard 지표: AUC, AP, NLL, Brier.
- 순위 지표: (경기, 단계)별 최종 순위 Spearman의 평균, top-5 AUC, 경기 내 C-index.
- 불확실성: 코호트별로 경기 단위 bootstrap(2,000회) 95% CI를 구한다. 13개 코호트는 DerSimonian–Laird random-effects로 통합하고 τ²를 보고한다.
- SESOI: hazard AUC 0.005, Spearman 0.01, top-5 AUC와 C-index 0.005.
  - 의미 있는 개선: 통합 95% CI 하한이 0보다 크고 점추정이 SESOI 이상이다.
  - 차이 없음: 통합 90% CI가 ±SESOI 안이다.
  - 그 외는 불확정이다.
- 다중성: 주 대조 5개에 Holm 보정을 한다. 5개는 RQ1, RQ2 hazard, RQ2 순위, RQ3, RQ4다.

---

## 6. 무결성 (실행 전)

1. **Fix B**: 격자 조인을 `phase_end_s − SNAPSHOT_INTERVAL`로 제한한다.
   - 대상: `stdb/phase_spatial_feature_etl.py`, `stdb/habitat_feature_etl.py`, `stdb/phase_edge_etl.py`, `stdb/network_feature_etl.py`, `stdb/z_affordance_feature_etl.py`
   - 덮어쓸 파생 테이블은 먼저 백업 스키마에 복사한다. 원시 `pubg.*` 테이블은 건드리지 않는다.
   - 경계 이후 표본이 바뀌어도 특징이 그대로인지 확인하는 테스트를 추가한다.
2. **HGB**: `early_stopping=False`로 두고 `ORDER BY match_id, team_id, cut_phase`로 행 순서를 고정한다. 대상은 `snapshot_tabular_baseline.py`와 `feature_ablation.py`다. 모든 HGB를 다시 돌린다.
3. **평가 행**: 표 형태 쪽도 단계 1~7로 제한한다 (`feature_ablation.py`:233).
4. **mask**: 팀 축 key mask는 단계 시작 시 생존 팀(`node_mask`)으로 둔다.
5. **덤프**: Fix B 이후 코호트별 텐서를 디스크에 한 번 저장한다. 학습은 DB 없이 돌린다.
6. **커밋**: 실행 전에 이 문서와 코드를 커밋한다. 사전 등록 기준 커밋은 `e536ecb`다.

---

## 7. 일정

| 날짜 | 작업 |
|---|---|
| 10/5 | 설계 v2. Fix B와 HGB 수정 시작. PTR-Former 구현 시작 |
| 10/6 | ETL 재실행, 덤프, HGB 재실행. 모델 테스트 통과 |
| 10/7 | 1단계 smoke run. 플레이어 ETL 시작 |
| 10/8 | 1단계 전체 실행 (야간) |
| 10/9 | 18:00에 플레이어 계층 포함 여부를 결정. 포함하면 야간에 2단계 실행 |
| 10/10~11 | 분석, 표와 그림 |
| 10/9~14 | 집필. 10/14에 전체 초고 |
| 10/15 | 모의심사와 수정 |
| 10/16 | 제출 |

**대체 경로**: 10/10까지 1단계 실행이 끝나지 않으면 기존 결과만으로 쓴다. 기존 결과는 snapshot zoo, MR 변형, O7 관계 대조, `TeamSetDecoder` 순위 효과다. 이 경우 모든 결과를 탐색적이라고 쓴다.

---

## 변경 기록

- 2026-10-05 v1: 축 분해, RQ1~RQ4, arm 7개.
- 2026-10-05 v2: 사용자와 그래프를 함께 설계해 반영했다. 바뀐 점은 아래와 같다.
  - 플레이어 + 안전구역 객체 노드
  - 팀원·적 근접·교전 격화 엣지
  - 관계 bias attention
  - 노벨티 검증 결과에 따라 RQ2 표현 수정, RQ3 관계 유형 확장, RQ4를 안전구역 객체 회복 질문으로 변경
  - Fix B와 HGB 수정 확정
  - 플레이어 계층은 10/9 18:00 컷오프를 둔 2단계로 함
- 2026-10-05 v2.1: 원고 담당 에이전트가 지적한 공백 12건을 §4.1에서 해소했다.
  - 주 대조 5개 지정 (요인 주효과, 셔플 대조, noHab 대조)
  - κ 보고 규칙
  - 관계 유형별 제거 arm
  - seed 사용 규칙
  - 관계 가중치와 구역 토큰 세부
  - 2단계 연결 방식과 팀원 분리 arm
  - HGB-lag 결측 처리
- 2026-10-05 v2.3: 데이터 감사 결과를 반영했다. 라벨 v2(사건 기반 + 마지막 단계 규칙), 단일 시계 보정, 생존 인원 재계산, airoyale·Vikendi 제외, AI 비율 특징.
- 2026-10-05 v2.2: Holm 판정 규칙을 정했다. RQ3 문구를 관계 전체의 정보 여부로 바꾸고 유형별 기여는 탐색적이라고 명시했다. 구역 토큰, noHab, 순위 손실, 표준화 세부를 확정했다.
