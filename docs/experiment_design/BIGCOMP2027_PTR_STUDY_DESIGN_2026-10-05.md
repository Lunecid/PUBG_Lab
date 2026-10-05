# BigComp 2027 연구 설계: 단계·팀·관계 축 분해 (2026-10-05)

상태: **확정안 v1**. 실험을 실행하기 전에 고정한다. 바꾸면 날짜와 이유를 이 문서 끝에 기록한다.

- 대상: IEEE BigComp 2027 full paper (8쪽 이하, 참고문헌 포함, IEEE 2단, double-blind)
- 마감: 2026-10-16 AoE (한국시간 10/17 20:59). 출처는 bigcomp2027-website.pages.dev이고 EasyChair(ieeebc27)에서 재확인해야 한다.
- 상위 문서: `docs/research_framing/RESEARCH_QUESTION_REDESIGN_2026-10-05.md`. 그 문서는 저널 확장용으로 남긴다.

---

## 1. 논문의 주장

새 모델이 기준선을 이긴다는 주장이 아니다. 같은 데이터, 라벨, 분할에서 세 정보 축을 하나씩 켜고 끄며 어느 축에 다음 단계 탈락 정보가 있는지 분리한다. 세 축은 각각 선행연구 하나에 대응한다.

| 축 | 대응 선행연구 | 모델 안의 구성요소 |
|---|---|---|
| 단계 축: 팀의 단계별 이력 | Sagong et al. 2025 (EC), Sagong & Kim 2025 (BigComp) | 팀마다 단계 1..k에 걸친 causal attention |
| 팀 축: 같은 단계의 다른 팀 | KRAFTON `pubg-winprob` | 같은 단계의 생존 팀 전체에 대한 masked self-attention |
| 관계: 명시적 근접·교전 관계 | 그래프 기반 스포츠 예측 | 팀 축 attention logit에 관계 유형별 bias를 더함 (Graphormer식) |

이름 붙인 모델은 세 축을 모두 켠 구성이다. 작업 이름은 **PTR-Former** (Phase–Team–Relation Transformer)이며 바꿀 수 있다.

## 2. 연구질문

| RQ | EN | KO |
|---|---|---|
| RQ1 단계 축 | Does encoding each team's phase-by-phase history improve next-phase elimination prediction beyond its current-phase state? | 팀의 단계별 이력을 넣으면 현재 단계 상태만 쓸 때보다 다음 단계 탈락 예측이 나아지는가? |
| RQ2 팀 축 | Does attention over all teams alive in the same phase improve absolute elimination risk, within-match relative standing, or both? | 같은 단계 생존 팀 전체에 대한 attention은 절대 탈락 위험과 경기 내 상대 순위 중 무엇을 개선하는가? |
| RQ3 관계 | Do explicit proximity and combat relations, injected as attention biases, add information beyond learned cross-team attention? | 근접·교전 관계를 attention bias로 넣으면 학습된 팀 간 attention 이상의 정보가 더해지는가? |
| RQ4 공간 맥락 (시간이 남으면) | When safe-zone and local-competition features are removed, does the cross-team model recover the lost information? | 안전구역·국지 경쟁 특징을 빼면 팀 간 모델이 그 정보를 대신 학습하는가? |

모든 주장은 예측 정보에 관한 것이다. 인과 표현을 쓰지 않는다.

## 3. 데이터와 과제

- 13개 코호트 (맵 × 시점 × 팀 규모), 17,943경기. 코호트별 경기 단위 80/10/10 분할(seed 42)을 저장된 파일 그대로 쓴다.
- 예측 단위는 (경기, 팀, 단계 경계 k)이고 k = 1..7이다. 라벨은 단계 k 종료 시 생존한 팀이 단계 k+1 안에 탈락하는지 여부다.
- 입력은 phase tier만 쓴다.
  - 특징 텐서 `X_p` (8 × 128 × 150)
  - 관계 텐서 `A_p` (8 × 3 × 128 × 128)
  - 10초 snapshot tier는 쓰지 않는다. 이유는 둘이다. ±5초 정렬 문제를 피하고, arm 정의를 단순하게 유지하기 위해서다.
- 보조 라벨: 최종 순위(정규화), top-5, top-10.

## 4. 실험 arm

모든 신경망 arm은 같은 골격, 같은 손실, 같은 학습 설정을 쓴다. 손실은 hazard BCE + η × 순위 손실이며 η는 모든 arm에서 같다. 순위 손실을 고정하는 이유는 기존 O8의 교란(attention과 rank loss가 함께 바뀜)을 없애기 위해서다.

| arm | 단계 축 | 팀 축 | 관계 bias | 답하는 RQ |
|---|---|---|---|---|
| M00 | 끔 (단계 k token만) | 끔 | 끔 | 기준 |
| M10 | 켬 | 끔 | 끔 | RQ1 |
| M01 | 끔 | 켬 | 끔 | RQ2 |
| M11 | 켬 | 켬 | 끔 | RQ1, RQ2 |
| M11R (PTR-Former) | 켬 | 켬 | 켬 | RQ3 |
| M11R-shuf | 켬 | 켬 | 경기 안에서 섞은 관계, 재학습 | RQ3 대조 |
| M11R-noZ | 켬 | 켬 | 켬, 안전구역·habitat 특징 제거 | RQ4 |

참조 기준선:

- HGB-cut: 단계 k의 150차원 행
- HGB-lag: HGB-cut + 직전 단계 값과 차분. 단계 축의 표 형태 대응이다.

주 대조:

| 대조 | 비교 |
|---|---|
| RQ1 | M10 − M00, M11 − M01 |
| RQ2 | M01 − M00, M11 − M10. hazard 지표와 순위 지표를 따로 본다 |
| RQ3 | M11R − M11, M11R − M11R-shuf |
| RQ4 | M11R − M11R-noZ. HGB의 같은 특징 제거 효과와 비교한다 |

## 5. 지표와 판정

- hazard 지표: AUC, AP, NLL, Brier.
- 상대 순위 지표: 최종 순위 Spearman, top-5 AUC, 경기 내 C-index.
- 코호트별로 경기 단위 bootstrap(2,000회) 95% CI를 구한다. 13개 코호트는 random-effects로 통합하고, 방향이 같은 코호트 수를 함께 보고한다.
- SESOI는 ΔAUC 0.005다.
  - 의미 있는 개선: 통합 95% CI 하한이 0보다 크고 점추정이 0.005 이상이다.
  - 차이 없음: 통합 90% CI가 ±0.005 안이다.
  - 그 외는 불확정으로 보고한다.
- seed는 모든 코호트에 42를 쓴다. Erangel TPP squad와 Taego TPP squad는 13, 2026을 추가한다.

### 기존 근거에서 나온 예상 (탐색적, 판정에 쓰지 않음)

| RQ | 예상 | 근거 |
|---|---|---|
| RQ1 | 소폭 양(+) | 상태 요약 +0.0074 (12/13) |
| RQ2 | hazard 0, 순위 양(+) | Erangel Spearman +0.0297, AUC 동일 |
| RQ3 | 0 근처 | +0.0023, CI 하한>0 6/13 |
| RQ4 | 특징 제거 손실이 일부만 회복 | habitat +0.0192 (13/13, HGB) |

## 6. 선결 조건 (실행 전)

1. phase tier 특징에 단계 경계 이후 정보(±5초 정렬)가 들어가는지 확인한다. 들어가면 고치거나 한계로 명시한다.
2. HGB 같은 설정 재실행 편차(코호트당 최대 0.0048)의 원인을 제거한다.
3. 모든 arm과 기준선의 평가 행 집합을 같게 맞춘다.
4. 팀 축 attention의 key mask는 단계 시작 시 생존 팀으로 둔다. 현재 `TeamSetDecoder`는 존재하는 팀 전체를 mask로 써서 탈락 팀도 attend한다.
5. 실행 전에 이 문서와 코드를 커밋한다 (사전 등록 타임스탬프).

## 7. 일정

| 날짜 | 작업 |
|---|---|
| 10/5 | 설계 확정, 무결성 점검(조사), 원고 골격 |
| 10/6 | 무결성 수정, 모델 구현 시작 |
| 10/7~8 | 모델과 테스트, 코호트 1개 smoke run, HGB 기준선 재실행 |
| 10/8~10 | 전체 실행 (야간 2회) |
| 10/10~11 | 분석, 표와 그림 |
| 10/9~14 | 집필 병행. 10/14 전체 초고 |
| 10/15 | 모의심사 1회와 수정 |
| 10/16 | 제출 |

## 8. 대체 경로

10/10까지 새 모델 실행이 끝나지 않으면 기존 결과만으로 같은 질문을 다룬다. 기존 결과는 snapshot zoo, MR 변형, O7 관계 대조, `TeamSetDecoder` 순위 효과다. 이 경우 모든 결과는 탐색적이라고 쓰고, 축이 깨끗하게 분리되지 않는 점(snapshot tier가 항상 켜져 있음 등)을 한계로 명시한다.

## 변경 기록

- 2026-10-05 v1 작성.
