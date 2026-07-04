# STDB Paper Sections — EN + KO Mirror

> Source of all current-result numbers: `docs/PRESENTATION_DATA_2026-06-21.md`. No headline statistic appears here that is not on that sheet. The single Future Work paragraph cites preliminary numbers from `docs/FUTURE_DESIGN_3D_RELATIONAL_2026-06-21.md` and is labelled as such. Figures are referenced by filename only (`docs/presentation_figures/`).

---

## 1. Methods — Phase-Aligned STDB, Multi-Resolution Snapshots, and the Next-Phase Hazard Task

### English

We reconstruct each match from its raw telemetry stream into a phase-aligned spatio-temporal database (STDB), the shared time and space axis on which every downstream quantity is defined. A match is partitioned into numbered phases, where each phase announces the next safe zone and then contracts toward it; we align every team trajectory, zone exposure, and combat event to this phase grid so that the unit of observation is the team-phase rather than the post-match aggregate.

On top of the phase grid we add a finer snapshot tier sampled every ten seconds, which supplies the model with multi-resolution temporal input. The coarse phase boundaries carry the game's strategic structure, while the ten-second snapshots preserve the within-phase trajectory that phase aggregation discards. The snapshot tier records each team's position in three dimensions, including elevation, so the representation that produces every headline result below already observes vertical team position rather than a flattened two-dimensional footprint.

We then cast prediction as a next-phase elimination hazard. At each cut phase we identify the teams still at risk and ask whether an at-risk team is eliminated before the next phase begins; the observation window always ends at the cut phase boundary, so every input is drawn from telemetry available up to that point and the future appears only in the label. This reformulation gives a single discrete-time frame in which the multi-resolution snapshot representation and the phase-level structure answer the same question — when, and under what conditions, a surviving team's risk crosses into elimination.

### 한국어 (Korean mirror)

우리는 각 매치의 원시 텔레메트리 스트림을 phase-정렬 시공간 데이터베이스(STDB)로 재구성한다. 이 STDB는 이후 모든 산출량이 정의되는 공통의 시간·공간 축이다. 매치는 번호가 매겨진 phase로 분할되며, 각 phase는 다음 safe zone을 제시한 뒤 그 zone을 향해 수축한다. 우리는 모든 팀 trajectory, 구역 노출, 전투 사건을 이 phase 격자에 정렬하여, 관측 단위가 사후 집계가 아니라 team-phase가 되도록 한다.

phase 격자 위에 우리는 10초마다 표집한 더 미세한 스냅샷 계층을 추가하며, 이는 모델에 다해상도 시간 입력을 공급한다. 거친 phase 경계는 게임의 전략적 구조를 담고, 10초 스냅샷은 phase 집계가 버리는 phase 내부 trajectory를 보존한다. 스냅샷 계층은 각 팀의 위치를 고도를 포함한 3차원으로 기록하므로, 아래의 모든 헤드라인 결과를 산출하는 표현은 이미 평탄화된 2차원 자취가 아니라 팀의 수직 위치까지 관측하고 있다.

그런 다음 우리는 예측을 다음-phase 탈락 hazard로 정식화한다. 각 cut phase에서 여전히 위험에 처한 팀을 식별하고, 위험 상태의 팀이 다음 phase가 시작되기 전에 탈락하는지를 묻는다. 관측 구간은 항상 cut phase 경계에서 끝나므로, 모든 입력은 그 시점까지 관측 가능한 텔레메트리에서만 구성되고 미래는 오직 label에만 나타난다. 이 재정식화는 다해상도 스냅샷 표현과 phase 수준 구조가 같은 질문 — 생존 팀의 위험이 언제, 어떤 조건에서 탈락으로 넘어가는가 — 에 답하는 단일 이산시간 프레임을 제공한다.

---

## 2. Results — The Resolution Ladder Breaks the Phase Ceiling

### English

Temporal resolution, not model family, sets the accuracy ceiling. When the input is restricted to phase-level aggregation, performance saturates near 0.79: a tabular gradient-boosting model on phase features reaches a hazard AUC of 0.786, and a phase-only graph model — given the same phase aggregation — reaches 0.793, essentially the same value. Adding the ten-second snapshot tier breaks this ceiling. The multi-resolution snapshot model reaches a hazard AUC of 0.847, an improvement of roughly +0.06 over the phase-aggregation plateau. The gain is therefore a property of representation rather than of architecture: two unrelated model families converge to the same value once they see only aggregated phases, and both are surpassed only when the within-phase trajectory is restored at ten-second resolution.

The advantage holds at every cut phase, not only on average. On the shared split, the multi-resolution model beats the tabular gradient-boosting baseline at all seven cut phases, with an overall ΔAUC of +0.063 (95% CI [+0.060, +0.067]) by match-level cluster bootstrap. The per-phase improvements range from +0.059 to +0.087, and every per-phase confidence interval excludes zero, so the gain is not confined to a single regime. A match-paired Wilcoxon test corroborates the result with p = 2.5e-144. The multi-resolution representation is thus statistically distinguishable from the phase-aggregation baseline across the full phase ladder rather than only where risk-sets are largest. 〔Figures: fig1_resolution_ladder.png, fig4_hazard_vs_baseline_ci.png〕

### 한국어 (Korean mirror)

정확도 천장을 정하는 것은 모델 계열이 아니라 시간 해상도다. 입력이 phase 수준 집계로 제한되면 성능은 0.79 부근에서 포화한다. phase 피처를 쓰는 tabular gradient-boosting 모델은 hazard AUC 0.786에 도달하고, 동일한 phase 집계를 받는 phase-only graph 모델은 0.793에 도달하여 사실상 같은 값을 보인다. 10초 스냅샷 계층을 더하면 이 천장이 깨진다. 다해상도 스냅샷 모델은 hazard AUC 0.847에 도달하며, 이는 phase 집계 plateau 대비 약 +0.06의 향상이다. 따라서 이 이득은 아키텍처가 아니라 표현의 속성이다. 서로 무관한 두 모델 계열이 집계된 phase만 볼 때 같은 값으로 수렴하고, 둘 모두 phase 내부 trajectory가 10초 해상도로 복원될 때에만 추월당한다.

이 우위는 평균에서만이 아니라 모든 cut phase에서 성립한다. 공유 split에서 다해상도 모델은 일곱 개 cut phase 전부에서 tabular gradient-boosting baseline을 능가하며, match-level cluster bootstrap 기준 전체 ΔAUC는 +0.063 (95% CI [+0.060, +0.067])이다. phase별 향상폭은 +0.059에서 +0.087까지 분포하고, 모든 phase별 신뢰구간이 0을 배제하므로, 이 이득은 단일 국면에 국한되지 않는다. match-paired Wilcoxon 검정은 p = 2.5e-144로 이 결과를 뒷받침한다. 따라서 다해상도 표현은 risk-set이 가장 큰 구간에만 집중된 것이 아니라 phase 사다리 전체에 걸쳐 phase 집계 baseline과 통계적으로 구별된다. 〔그림: fig1_resolution_ladder.png, fig4_hazard_vs_baseline_ci.png〕

---

## 3. Results — The Decision Window and Rank Lock-In

### English

The final survival ordering crystallizes during the mid-to-late zone collapse, and the data locate this decision window precisely. We measure how closely each cut phase's predicted ordering aligns with the final rank, using Spearman alignment between predicted and observed placement. The alignment rises significantly from cut phase 3 to cut phase 6, by +0.085 (95% CI [+0.070, +0.101]). The increase then plateaus from phase 5 to phase 6, where the change is +0.013 (95% CI [−0.002, +0.025]) and not significant.

Mapped onto measured game time, this window spans roughly minutes 16 to 24, over which the safe-zone radius contracts from about 660 m to about 190 m and the number of surviving teams falls from about nine to about four. The survival ordering therefore locks in as the playable area collapses in the mid-to-late game, and it has largely settled by the time the endgame setup begins. This places the moment at which a match's outcome becomes predictable squarely inside the zone-contraction process rather than at landing or at the final fight. 〔Figure: fig3_rank_lockin.png〕

### 한국어 (Korean mirror)

최종 생존 서열은 중후반 존 붕괴 구간에서 굳으며, 데이터는 이 decision window를 정밀하게 위치 짓는다. 우리는 각 cut phase의 예측 서열이 최종 순위와 얼마나 일치하는지를, 예측 순위와 관측 순위 사이의 Spearman alignment로 측정한다. 이 정렬은 cut phase 3에서 cut phase 6으로 가며 +0.085 (95% CI [+0.070, +0.101])만큼 유의하게 상승한다. 이후 phase 5에서 phase 6 사이에서는 변화가 +0.013 (95% CI [−0.002, +0.025])으로 유의하지 않게 plateau에 든다.

실측 게임 시각에 대응시키면 이 구간은 대략 16분에서 24분에 걸치며, 그 동안 safe-zone 반경은 약 660 m에서 약 190 m로 수축하고 생존 팀 수는 약 9에서 약 4로 줄어든다. 따라서 생존 서열은 중후반 게임에서 playable area가 붕괴할 때 잠기며, 엔드게임 setup이 시작될 무렵에는 대부분 정해져 있다. 이는 매치의 결과가 예측 가능해지는 순간을 낙하 시점이나 최종 교전이 아니라 존 수축 과정의 한가운데에 위치시킨다. 〔그림: fig3_rank_lockin.png〕

---

## 4. Graph Contribution — Smoothing, Not Topology

### English

The dynamic team graph improves accuracy, but its contribution is mean-field smoothing rather than the specific inter-team topology. We test this against three row-normalized null controls that preserve aggregation scale while destroying or altering structure. The result is unambiguous: an edge-shuffled control reaches a hazard AUC of 0.847, consistently matching or exceeding the observed graph across all three seeds, while a no-graph control sits lower at 0.834 and the observed graph reaches 0.845. The smoothing therefore helps — the graph models beat the no-graph control — but the observed topology does not: where the two differ significantly in the early phases, the observed graph is in fact the worse of the two, and no phase shows the observed topology significantly ahead of its shuffle. A node-shuffle control that randomizes team identity matches the observed graph, confirming that team identity carries no signal either. Read together, these controls show that the graph operates as a low-pass filter over team-phase features, not as a carrier of relational structure. We report this as an honest negative. 〔Figure: fig2_graph_null.png〕

### 한국어 (Korean mirror)

동적 팀 graph는 정확도를 높이지만, 그 기여는 특정 팀 간 위상이 아니라 mean-field 평활화다. 우리는 이를, 집계 규모를 보존하면서 구조를 파괴하거나 변형하는 세 개의 row-normalized null 대조군으로 검정한다. 결과는 분명하다. edge-shuffle 대조군은 hazard AUC 0.847에 도달하여 세 시드 전부에서 관측 graph와 일관되게 같거나 그 이상을 보이는 반면, no-graph 대조군은 0.834로 더 낮고 관측 graph는 0.845에 도달한다. 따라서 평활화는 도움이 된다 — graph 모델은 no-graph 대조군을 능가한다 — 그러나 관측 위상은 그렇지 않다. 두 값이 초반 phase에서 유의하게 갈리는 곳에서 관측 graph는 오히려 더 나쁜 쪽이며, 어떤 phase에서도 관측 위상이 그 shuffle보다 유의하게 앞서지 않는다. 팀 신원을 무작위화하는 node-shuffle 대조군은 관측 graph와 일치하여, 팀 신원 역시 신호를 담지 않음을 확인한다. 이들을 종합하면, graph는 관계 구조의 운반체가 아니라 team-phase 피처에 대한 low-pass filter로 작동한다. 우리는 이를 정직한 음성 결과로 보고한다. 〔그림: fig2_graph_null.png〕

---

## 5. Future Work — Relation Type as the Carrier of Relational Signal

### English

The negative graph result concerns interaction-based relations — proximity, combat, and co-pressure, which encode whether and how intensely teams interact rather than which team holds the advantage — and a complementary line of preliminary work suggests that the *type* of relation, not the presence of a graph, governs whether topology is informative. When edges are recast from these interaction relations to a contact-gated relative-advantage relation, the observed graph separates significantly from every null. In phase-tier experiments the contact-gated advantage edge beats an edge-shuffled control by +0.0033 (95% CI [+0.0023, +0.0043]) and remains significant against node-shuffle (+0.0023) and no-graph (+0.0051) controls. Critically, the effect survives a node-absolute control: when the absolute per-team quantities that a relative edge could be a proxy for — elevation, alive count, health, zone state — are injected directly into the node features, the advantage edge still contributes +0.0023 (95% CI [+0.0013, +0.0033]), about seventy percent of the original effect. The signal is therefore not reducible to node attributes; a pairwise "I am lower than the adjacent enemy" cannot be reconstructed by mean aggregation over node states. A descriptive physical check is consistent: among nearby team pairs the lower team is more likely to be eliminated in the next phase, at a magnitude of z = +10.5. These figures are preliminary and require a full re-base; we report them to indicate that the honest negative on the interaction-relation topology is specific to relation type rather than general to graphs, and that contact-gated relative advantage is a promising direction for recovering irreducibly relational signal.

### 한국어 (Korean mirror)

graph의 음성 결과는 상호작용 기반 관계 — 근접·교전·존압박, 즉 팀이 *상호작용하는가·얼마나 강하게*를 담을 뿐 *누가 유리한가*는 담지 않는 관계 — 에 관한 것이며, 이를 보완하는 예비 연구는 graph의 존재가 아니라 관계의 *종류*가 위상의 정보성을 좌우함을 시사한다. 엣지를 이 상호작용 관계에서 접촉-게이트 상대-우위 관계로 다시 정식화하면, 관측 graph는 모든 null과 유의하게 분리된다. phase-tier 실험에서 접촉-게이트 우위 엣지는 edge-shuffle 대조군을 +0.0033 (95% CI [+0.0023, +0.0043])만큼 능가하고, node-shuffle (+0.0023) 및 no-graph (+0.0051) 대조군에 대해서도 유의하게 유지된다. 결정적으로 이 효과는 node-absolute 통제를 통과한다. 상대 엣지가 대리할 수 있는 팀별 절대량 — 고도, 생존 인원, 체력, zone 상태 — 을 노드 피처에 직접 주입해도, 우위 엣지는 여전히 +0.0023 (95% CI [+0.0013, +0.0033])을 기여하여 원래 효과의 약 70%가 잔존한다. 따라서 이 신호는 노드 속성으로 환원되지 않는다. "내가 옆 적보다 낮다"는 쌍별 관계는 노드 상태에 대한 mean aggregation으로 복원할 수 없다. 기술적 물리 검정도 이와 일치한다. 근접한 팀 쌍에서 더 낮은 팀이 다음 phase에 탈락할 가능성이 더 높으며, 그 크기는 z = +10.5다. 이 수치들은 예비적이며 전체 재-베이스가 필요하다. 우리는 상호작용 관계 위상에 대한 정직한 음성 결과가 graph 일반이 아니라 관계의 종류에 특정된 것임을, 그리고 접촉-게이트 상대 우위가 환원 불가능한 관계 신호를 회복하는 유망한 방향임을 나타내기 위해 이를 보고한다.
