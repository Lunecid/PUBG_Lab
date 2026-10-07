# Introduction draft v2 (2026-10-06): BigComp template

v1 대비 바뀐 점은 다음과 같다. 근거는 BigComp 2023–2026 논문 4편의 서론을 조사한 결과다.

- 5단락과 기여 4개 항목, 약 590단어로 줄였다. 조사한 논문은 400–700단어였다.
- 구성 단락을 삭제했다. 조사한 논문 4편 모두 이 단락이 없었다.
- RQ는 유지하되 P5 안에 짧은 질문 형태로 넣었다.
- 수치는 기여 항목에만 두었다.
- 상세 선행연구는 Section II로 보낸다.

## Draft

**[P1 · context]** PUBG: BATTLEGROUNDS (PUBG) is a battle royale game in which up to 100 players, alone or in teams of two or four, land on a large map and fight until one team remains. Its defining mechanic is an exogenous spatial constraint. The safe zone shrinks in scheduled phases, and players outside it take continuous damage. Each match is thus a multi-agent survival process within a habitat that is lost step by step. Public telemetry records this process in detail, including player positions at roughly ten-second intervals and combat, item and zone events [API].

**[P2 · task]** In such a process, decisions are made one safe zone at a time. Once a phase reveals the next zone, each team must rotate, hold or engage before the boundary closes. These decisions turn on the probability that a team is eliminated before the next phase, and broadcasters and designers need the same quantity. A match-level win probability, the usual output of esports predictors [cites], answers a different question. With N teams its prior is 1/N, and it blends a team's immediate danger with its long-run strength.

**[P3 · lens]** The shrinking zone mirrors habitat loss in ecology. Habitat loss has large and consistently negative effects on biodiversity [Fahrig03]. Metapopulation models relate local persistence to patch area and isolation [Hanski94], and occupancy models estimate local extinction per time period [MacKenzie03]. Read this way, the elimination of a team is a local extinction. It is best modeled as a discrete-time hazard over habitat-loss steps, conditioned on the team's habitat context: its exposure at the zone edge, the movement it needs to reach the next zone, and the density of competitors around it.

**[P4 · gap]** However, existing PUBG predictors take other routes. Sagong et al. classify whether a player survives to the next half-phase from activity scores, and they observe other players only as a head-count within 100 m [Sagong25a, Sagong25b]. KRAFTON's predictor ranks squads by future time of death with a Cox objective and converts the scores into win probabilities. It uses phases only to sample time points [KRAFTON]. Neither design represents the arena as shrinking habitat. A Cox partial likelihood is also invariant to shifting all scores at a time point, so its scores cannot express how risk rises across phases. Because the two designs further differ in unit, data and metric, it remains unknown which choice drives predictive performance.

**[P5 · approach]** We therefore ask three questions.

- RQ1: Can next-phase elimination be formulated as local extinction under stepwise habitat loss, with habitat variables tracking the risk?
- RQ2: What model design does this formulation imply relative to prior designs, and does it improve prediction on identical data?
- RQ3: Which design element accounts for the difference?

We answer them on 13,948 public matches from 12 cohorts defined by map, perspective and team size. Labels are event-based. We adapt both prior designs to the same rows and change one factor at a time.

**[Contributions]** Our contributions are summarized as follows.

- An ecological formulation of elimination as a phase-conditioned, discrete-time local-extinction hazard, with evidence that habitat variables track the risk across cohorts [TBD: EHM].
- A habitat-aware design that outperforms adapted versions of both prior predictors on identical rows.
- A single-factor decomposition:
  - The training objective explains the gap to the time-of-death design (+0.051 AUC).
  - Habitat context explains most of the gap to the activity-score design (+0.053 of +0.092 AUC).
  - Architecture adds about 0.002.
- Event-based elimination labels for 12 public cohorts, replacing snapshot labels that mislabel 11.4% of team-phase rows.

## Notes

- Citations needed:
  - [API]: PUBG developer documentation. Check whether a URL reference is acceptable.
  - [cites] in P2: e스포츠 승률 문헌 2–3편. 학위논문 bib에 있는 Hodge 2017, Yang 2020, Junior 2023 중에서 고른다.
  - 서론의 인용은 약 8개다. 조사한 논문은 12–16개였으므로, Section II에서 보강한다.
- Fig. 1 (선택): 조사한 논문 3편 중 2편이 서론 쪽에 개요 그림을 두었다. 제안하는 그림은 "안전구역 축소 = 서식지 손실, 그리고 단일 요인 분해 사다리"다.
- 남은 `[TBD]`: 기여 1번의 EHM 결과.
