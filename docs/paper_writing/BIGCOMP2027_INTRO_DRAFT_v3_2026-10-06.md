# Abstract + Introduction draft v3 (2026-10-06): big-data framing

사용자 지시에 따라 게임 특화 용어를 줄이고 세 관점을 앞에 세웠다.

1. 대규모 시공간 텔레메트리 로그
2. 그래프·시계열 기반 hazard 예측
3. 누수 없는 데이터 엔지니어링과 딥러닝 파이프라인

규모 수치는 DB 통계(pg_class reltuples 추정치)에서 왔다.

- 수집한 원시 데이터: 27,491경기
  - 위치 표본: 155.4M (telem_positions 2026-03~06 파티션 합계)
  - 교전·아이템·상태 이벤트: 313.7M (damage 129.6M, item_pickup 87.0M, item_equip 68.5M, item_use 19.8M, game_states 4.0M, landing 2.8M, kills 2.0M)
  - 합계 약 470M 레코드, DB 전체 261 GB
- 분석에 쓴 데이터: 13,948경기, 12개 코호트, 위험 집합 829,640행

## Abstract v3

> Spatio-temporal telemetry logs capture how many interacting agents move and survive in environments whose feasible region changes over time. Predicting which agents will fail next is difficult because risk depends jointly on the shrinking region, on agent interactions, and on irregular, noisy event streams. We study this problem on a large public log corpus from the battle royale game PUBG: about 470 million position and event records from 27,491 matches, of which 13,948 matches in 12 cohorts form a leakage-safe benchmark. Our pipeline aligns asynchronous clocks, cuts features causally at each phase boundary, and derives event-based elimination labels. An audit shows that snapshot-based labeling mislabels 11.4% of samples. Drawing on habitat-loss ecology, we formulate elimination as a phase-conditioned discrete-time hazard, represent each team's habitat context, and model the hazard with a graph- and sequence-based reference model. We then adapt two prior predictors to identical samples and decompose the performance gap one factor at a time. Replacing a time-of-death Cox objective with the phase-conditioned hazard in the same transformer raises AUC by 0.051. Habitat context accounts for 0.053 of a 0.092 representation gain. Graph and sequence structure add about 0.002, mainly in data-rich cohorts.

## Introduction v3

**[P1 · spatio-temporal big data and the task]** Telemetry from multi-agent systems such as online games, vehicle fleets and animal tracking produces large volumes of spatio-temporal logs. These logs hold agent positions sampled over time, interaction events, and changes to the environment [ST-survey-1, ST-survey-2]. A recurring analytic task on such data is risk prediction: estimating which agents will fail within a horizon from their recent trajectories and interactions. The task is hardest when the environment itself changes. An agent's risk then depends on how the feasible region evolves as well as on its competitors, and the logs that record both are irregular and noisy.

**[P2 · PUBG as a large-scale testbed, and the data challenges]** Battle royale games make this setting observable at scale. In PUBG: BATTLEGROUNDS, up to 100 players in teams of one, two or four compete on a large map. The safe zone shrinks in scheduled phases, players outside it take continuous damage, and the last team standing wins. The public telemetry of 27,491 matches comprises about 470 million records: 155 million position samples and 314 million combat, item and state events (261 GB). Turning these logs into reliable learning samples is itself non-trivial. Event clocks are not aligned with phase boundaries, and knocked players must be distinguished from eliminated ones. Positions arrive only every ten seconds, and any feature computed across a phase boundary leaks the outcome.

**[P3 · estimand and the ecological lens]** In this setting the decision-relevant risk is next-phase elimination: whether a team is eliminated before the next shrink. This is the horizon at which teams choose to rotate, hold or engage. Match-level win probability, the usual output of game predictors [cites], answers a different question. Its prior is 1/N for N teams, and it blends immediate danger with long-run strength. The shrinking zone instead resembles habitat loss. Habitat loss has large and consistently negative effects on biodiversity [Fahrig03]. Metapopulation models relate local persistence to patch area and isolation [Hanski94], and occupancy models estimate local extinction per period [MacKenzie03]. This suggests modeling elimination as a discrete-time local-extinction hazard over phases. The hazard is conditioned on each team's habitat context: its edge exposure, the movement it needs to reach the next zone, and the density of competitors.

**[P4 · gap]** However, existing predictors for this data follow other designs. Sagong et al. classify whether an individual player survives to the next half-phase from activity scores, and they observe others only as a head-count within 100 m [Sagong25a, Sagong25b]. KRAFTON's predictor feeds squad statistics to a cross-squad transformer trained with a Cox objective on time of death, and it outputs win probabilities. It uses phases only to sample time points [KRAFTON]. Sagong et al.'s representation omits the region's geometry altogether. KRAFTON's representation includes zone distances and nearby-enemy counts, but its Cox objective is invariant to shifting all scores at a time point, so it cannot express how risk rises across phases. The designs further differ in unit, data and metric, so it is unknown which choice drives predictive performance: the formulation, the representation or the model structure.

**[P5 · approach]** We address three questions.

- RQ1: Can next-phase elimination be formulated as local extinction under stepwise habitat loss, with habitat variables tracking the risk?
- RQ2: What model design does this formulation imply relative to prior designs, and does it improve prediction on identical samples?
- RQ3: Which design element accounts for the difference?

We build a leakage-safe pipeline that turns raw logs into phase-aligned team tensors with typed interaction graphs. We evaluate on 13,948 matches in 12 cohorts, adapt both prior designs to identical samples, and change one factor at a time.

**[Contributions]** Our contributions are summarized as follows.

- **A leakage-safe spatio-temporal pipeline and benchmark.** It produces clock-aligned, causally cut and event-labeled samples from 13,948 public matches in 12 cohorts. An accompanying audit shows that snapshot-based labels mislabel 11.4% of samples.
- **A habitat-loss hazard formulation** and a graph- and sequence-based reference model for team elimination risk, with evidence that habitat variables track the risk: a four-covariate ecological hazard model captures 79% of the full model's discrimination, and its dispersal-demand, crowding and team-size effects keep the same sign in every cohort.
- **A single-factor decomposition against two prior designs.**
  - The training objective explains the gap to the time-of-death transformer (+0.051 AUC).
  - Habitat context explains most of the gap to the activity-score design (+0.053 of +0.092).
  - Graph and sequence structure add about 0.002, mainly in data-rich cohorts.

## Notes

- **기여 순서**: 학회 성격에 맞춰 데이터 파이프라인을 먼저 두었다. Section VI 결과는 C2 → C1 → C3 순서를 유지한다.
- **게임 용어**: PUBG는 1회만 쓴다. 안전구역은 "feasible region"과 함께 정의한다. 기절은 데이터 난점으로만 언급한다.
- **인용 검증 필요 ([VERIFY])**:
  - ST-survey-1: Atluri, Karpatne, Kumar 2018, ACM Computing Surveys 51(4), "Spatio-temporal data mining: a survey of problems and methods"
  - ST-survey-2: Wang, Cao, Yu, IEEE TKDE, "Deep learning for spatio-temporal data mining: a survey"
  - (선택) Zheng 2015, ACM TIST, "Trajectory data mining: an overview"
- **분량**: 서론 약 620단어로, BigComp 범위인 400–700단어 안에 든다. 초록은 약 200단어다.


- **사실 정정 (2026-10-06)**: KRAFTON은 구역 기하(중심·반경 임베딩, 구역까지의 거리)와 150 m 적 수를 입력으로 쓴다. 그래서 "구역을 무시한다"는 표현을 쓰지 않는다. 격차의 원천은 정식화다.
