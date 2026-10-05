# BigComp 2027 결과 서사 브리프 (2026-10-06)

이 문서는 원고 작성 에이전트가 따라야 할 기준이다. 수치는 반드시 `reports/bigcomp2027_results/`의 CSV에서 다시 읽어 넣는다. 아래 수치는 확인용이며, CSV와 다르면 CSV를 따른다.

스토리라인은 **표현 + 분해 + 데이터 규모**이다 (사용자 결정, 2026-10-06).

---

## 1. 한 문장 주장

> On public PUBG matches with audited event-based labels, *what* a model sees matters far more than *how* it relates teams over time: a phase-conditioned representation of zone and local-competition context outperforms prior battle-royale approaches by 0.06–0.10 AUC, whereas phase history, explicit relation graphs and intra-team relations add little on average—and neural structure overtakes gradient boosting only once a cohort offers roughly 700+ training matches.

## 2. 기여 (서론 마지막 문단, 이 순서)

1. **Audited labels and data.** Snapshot-based alive inference mislabels 11.4% of team-phase rows (positive precision 70.4%). We give event-based labels with clock alignment. Bot-dominated (`airoyale`) matches and a map with non-player combatants are excluded, which leaves 13,948 matches in 12 map × perspective × team-size cohorts.
2. **Representation matters most.** On identical rows, our phase-conditioned representation beats a re-implementation of Sagong et al. by +0.103 AUC (12/12 cohorts) and KRAFTON's official code adapted to our task by +0.057 (K2 result → specify whether the gap is due to the input).
3. **Pre-specified decomposition.** Phase history (P1), explicit relations (P4) and teammate relations do not add meaningful information (equivalence under SESOI 0.005).
   - Cross-team attention adds a small but highly consistent gain: +0.0033 AUC and +0.0094 Spearman.
   - An explicit safe-zone object recovers about 29% of the information in hand-crafted zone/habitat features.
4. **Data-scale regime (exploratory).** The learning curves and a cohort-level meta-regression locate the point where the axis model overtakes GBDT at about 700 training matches.

## 3. 결과 절 구성 (분량 배분)

| 절 | 내용 | 표/그림 | 분량 |
|---|---|---|---|
| A. 선행 방식 대비 | SAGONG-xgb/rf/lr, KRAFTON-tf/mlp, K2(우리 특징 + KRAFTON 백본), HGB-cut, M11R의 12개 코호트 평균 AUC와 통합 Δ | `tab_external` 1개 (축약) | 0.35쪽 |
| B. 사전 지정 분해 | P1–P5 판정 표와 forest plot | `tab_primary` + `forest_primary` | 0.45쪽 |
| C. 데이터 규모 | 학습 곡선(step 수를 맞춘 버전이 있으면 그것)과 메타회귀 핵심 3행(M11R − HGB, P1, P2) | `learning_curve` 그림 1개 | 0.35쪽 |
| D. 플레이어 계층 | M11RZ-P − M11RZ: AUC +0.0014 [−0.0009, 0.0037], top-5 AUC −0.020. 팀원 관계는 −0.0001 | 본문만 | 0.1쪽 |

## 4. 해석 문장 (논의 절에 그대로 쓰거나 다듬어서)

- **표현이 핵심이다.** 같은 학습기(GBDT)끼리 비교해도 Sagong 특징과 우리 특징의 차이가 0.10 AUC다. 그러므로 차이는 모델 용량이 아니라 입력 표현에서 나온다. K2 결과로 KRAFTON 쪽도 같은 결론인지 확인한다.
- **구조의 효과는 데이터에 달려 있다.**
  - 단계 이력의 효과는 코호트 규모에 따라 부호가 바뀐다. 기울기는 ln N당 +0.0025이고 교차점은 약 590경기다.
  - 팀 간 attention의 효과는 규모와 무관하다.
  - 작은 코호트에서는 GBDT, 큰 코호트에서는 축 모델이 낫다.
- **관계 그래프는 attention을 넘지 못한다.** 근접·교전·공동압박 bias는 섞은 관계와 동등했다(P4). 팀원 관계도 동등했다. 학습된 팀 간 attention이 이 정보를 이미 담고 있다고 해석한다. 인과 표현은 쓰지 않는다.
- **팀 간 정보는 상대 순위 쪽에서 더 크다.** P3(Spearman +0.0094)가 P2(AUC +0.0033)보다 크다. KRAFTON식 구조의 이득은 절대 위험보다 상대 순위에 있다는 해석과 맞는다.
- **수작업 공간 특징을 학습형 객체가 대체하지 못한다.** κ = 0.29다. 도메인 지식에 기반한 특징 설계가 여전히 필요하다.

## 5. 한계 (반드시 포함)

- 단일 게임(PUBG), 공개 경기, 2026년 3~6월 패치 구간이다. 시간 holdout은 없다.
- 외부 기준선은 재현이 아니라 우리 과제에 맞춘 적용이다.
  - KRAFTON: 입력 일부를 재구성하지 못했고, 원래 타깃은 우승확률이다.
  - Sagong: 플레이어·반 단계 방식을 팀·전체 단계로 집계했다.
- 데이터 규모 분석은 사후 탐색이다. 메타회귀의 규모 효과는 맵·모드 차이와 교란된다. 학습 곡선은 두 코호트에서만 했다.
- 주 대조 외의 결과는 Holm 보정 없이 탐색적으로 보고했다.
- AI 계정이 섞여 있다. 팀별 AI 비율을 입력에 넣고 match_type별로도 보고했다.

## 6. 표현 규칙

- "first", "state-of-the-art", "outperform KRAFTON/Sagong"는 쓰지 않는다. 대신 "on our task and data, adapted implementations of …"라고 쓴다.
- "pre-registered"는 쓰지 않는다. "pre-specified (analysis plan fixed in a timestamped commit before the experiments)"라고 쓴다.
- 통계적으로 유의하지만 SESOI 미만인 효과는 "reliable but small"로 쓴다.
- 인과 표현을 쓰지 않는다.

## 7. 분량 관리 (8쪽, 참고문헌 포함)

- 분해 대상은 4개 RQ로 유지한다. 외부 기준선과 데이터 규모는 RQ 밖의 "additional analyses"로 둔다.
- 삭제·축약 후보:
  - 코호트 표 (본문에서 범위로 서술)
  - arm 표와 대조 표 통합
  - 관련연구를 0.5쪽으로
  - 탐색 표는 `\resizebox` 또는 부록 없이 삭제
