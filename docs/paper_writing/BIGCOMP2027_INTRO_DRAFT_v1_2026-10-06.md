# Introduction draft v1 (2026-10-06)

이 초안은 사용자와 함께 쓰기 위한 것이다. 작성 기준은 다음과 같다.

- Widom과 research-paper-writing 스킬을 따른다. 단락마다 메시지는 하나이고, 첫 문장이 그 메시지를 말한다.
- 배경과 동기에는 우리 결과를 쓰지 않는다.
- 선행연구는 공백을 세우는 데 필요한 만큼만 서론에 두고, 상세한 내용은 Section II로 보낸다.
- `[TBD]`는 실행 중인 분석(EHM, 서식지 증분)의 결과가 나오면 채운다.

## Outline

1. P1 (opening): PUBG는 무엇이고, 왜 연구 대상인가. 외생적으로 줄어드는 공간 제약과 공개된 상세 기록이 이유다.
2. P2 (task): 결정은 안전구역 단계마다 내려지므로, 다음 단계 탈락 위험이 결정에 필요한 양이다. 경기 단위 우승확률은 다른 질문에 답한다.
3. P3 (lens): 안전구역 축소는 서식지 손실과 대응하므로, 국지 멸종 hazard로 모델링한다.
4. P4 (challenge): 선행 두 설계의 기술적 한계와 비교 불가능성.
5. P5 (approach): RQ1–RQ3, 그리고 데이터와 단일 요인 분해 방법.
6. P6 (contributions): 기여 목록과 결과 미리보기.
7. P7 (organization): 논문 구성.

## Draft

**[P1 · opening: PUBG as a research object]**
PUBG: BATTLEGROUNDS (PUBG) is a battle royale game in which up to 100 players, alone or in teams of two or four, land on a large map, gather equipment, and fight until a single team remains. The feature that defines the genre is an exogenous spatial constraint. The playable area, the safe zone, shrinks in scheduled phases, and players caught outside it take continuous damage. Each match is therefore a multi-agent survival process in which dozens of teams compete within a habitat that is lost step by step. PUBG also records this process in detail: its public telemetry logs player positions at roughly ten-second intervals, together with combat, item and zone events [cite: PUBG API]. A known, shrinking spatial constraint combined with large-scale, fine-grained public records makes PUBG an unusually clean testbed for studying survival under spatial pressure.

**[P2 · task: why next-phase elimination risk]**
Decisions in such a match are made one safe zone at a time. At the start of each phase the next safe zone is revealed, and the boundary then contracts toward it. Before the contraction ends, a team must decide whether to rotate, hold, or engage. The quantity that informs this decision is the probability that the team is eliminated before the next phase. Broadcasters need the same quantity to choose which of dozens of teams to show, and designers need it to tune the pressure of each phase. A match-level win probability, the measure commonly reported in esports analytics [cite], answers a different question. With N teams its prior is 1/N, and it blends a team's immediate danger with its long-run strength.

**[P3 · lens: shrinking zone as habitat loss]**
The mechanics of the shrinking zone mirror habitat loss in ecology. The habitable area contracts in discrete steps without fragmenting, so teams are pushed into closer contact at its shrinking edge. In ecology, habitat loss has large and consistently negative effects on biodiversity [Fahrig03]. Metapopulation models relate local persistence to habitat patch area and isolation [Hanski94], and occupancy models estimate local extinction as a probability per time period [MacKenzie03]. Through this lens, the elimination of a team is a local extinction. The natural model is then a discrete-time hazard indexed by habitat-loss steps (phases) and conditioned on each team's habitat context: its position relative to the edge, the movement it needs to reach the next zone, and the density of competitors around it.

**[P4 · challenge: prior designs and why they leave the question open]**
Existing predictors for PUBG approach elimination differently. Sagong et al. classify whether an individual player survives to the next half-phase, using phase-wise activity scores; they observe other players only through a head-count within 100 m [Sagong25a]. A follow-up models sequences of phase-wise strategies [Sagong25b]. KRAFTON's recently released predictor ranks squads by their future time of death with a Cox objective and converts the scores into win probabilities; it uses phases only to sample and stratify time points [KRAFTON]. Neither design represents the arena as shrinking habitat. In addition, a Cox partial likelihood is unchanged when every score at a time point is shifted by the same constant, so its scores cannot express how risk rises from one phase to the next. The two designs also differ in unit of analysis, data and evaluation metric. It therefore remains unclear which design choice determines how well elimination can be predicted.

**[P5 · approach: research questions and method]**
This paper asks whether the habitat-loss lens leads to better models of elimination risk, and why. We pose three research questions:

- **RQ1.** Can next-phase elimination be formulated as local extinction under stepwise habitat loss, and do habitat variables track that risk in the data?
- **RQ2.** What model design does this formulation imply relative to prior designs, and does it change predictive performance on identical data?
- **RQ3.** When the performance gap is decomposed one factor at a time, which design element contributes most?

We answer these questions on 13,948 public PUBG matches spanning 12 cohorts defined by map, perspective and team size. Elimination labels are constructed from kill events rather than position snapshots. We adapt both prior designs to the same matches, rows and labels, and change one factor at a time: input representation, training objective, phase baseline, architecture and relational structure.

**[P6 · contributions]**
Our contributions are as follows.

- **An ecological formulation of elimination risk** as a phase-conditioned, discrete-time local-extinction hazard, with evidence that habitat variables track this risk [TBD: EHM coefficient replication across cohorts; the edge-exposure gradient steepens as the zone shrinks] (Sections III and VI-A).
- **A habitat-aware model design** that differs from prior designs in estimand and representation and outperforms adapted versions of both on identical data (Sections IV and VI-B).
- **A single-factor decomposition.**
  - The gap to the time-of-death design comes from the training objective: +0.051 AUC within the same backbone and inputs.
  - The gap to the activity-score design comes from the representation: +0.091 AUC with the learner fixed, of which habitat context alone accounts for +0.053 (12/12 cohorts).
  - Architecture and relational structure contribute about 0.002 (Section VI-C).
- **Event-based elimination labels for 12 public cohorts.** We show that snapshot-based alive inference mislabels 11.4% of team-phase rows (Section III-C).

**[P7 · organization]**
The rest of the paper is organized as follows. Section II reviews related work. Section III formulates the problem and describes the data. Section IV presents the compared designs. Section V describes the evaluation protocol, and Section VI reports the results. Section VII discusses implications and limitations and concludes the paper.

## Self-review checklist

| 항목 | 상태 |
|---|---|
| 단락마다 메시지 하나, 첫 문장이 메시지 | ✓ |
| 배경·동기에 우리 결과 없음 (P1–P4) | ✓ 우리 수치는 P6에만 있음 |
| 용어 일관성 | ✓ safe zone, phase, elimination, local extinction, habitat context, next-phase elimination risk |
| 인용이 필요한 곳 | P1의 PUBG API 문서, P2의 e스포츠 승률 문헌, P3의 Hanski (서지 검증 중) |
| 분량 | 약 770단어, IEEE 2단 기준 약 0.8쪽 |

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| 다음 단계 탈락 위험이 결정에 필요한 양이다 | 게임 메커니즘 서술. 활용은 동기 수준의 주장이다 | 논증으로 지지됨. 사용자 가치는 평가하지 않았다 |
| 우승확률의 사전값은 1/N이고, 당장의 위험과 장기적 강함을 섞는다 | 논리적 논증 | 지지됨 |
| Cox 부분우도는 시점별 상수 이동에 불변이다 | 수학적 성질 | 지지됨 (Section III에서 식으로 보인다) |
| 서식지 손실은 생물다양성에 크고 일관된 부정적 효과를 낸다 | Fahrig 2003 초록 | 지지됨 |
| 메타개체군 모델은 국지 지속을 패치 면적과 고립도에 연결한다. occupancy 모델은 기간별 국지 멸종 확률을 추정한다 | Hanski 1994 초록, MacKenzie et al. 2003 | 지지됨 (원문으로 확인된 범위까지만 씀) |
| 정식화 효과 +0.051, 표현 효과 +0.091, 구조 +0.002 | tab_decomp (12/12) | 지지됨 |
| 서식지 변수가 위험을 추적한다 | EHM과 [B] 가장자리 기울기 | EHM 결과 대기 |
| 표현 이득의 약 58%가 서식지 맥락에서 온다 | SAGONG-hgb-hab: +0.053 / 합계 +0.092 (12/12) | 지지됨 |
| snapshot 라벨의 11.4% 오분류 | 라벨 감사 | 지지됨 |
