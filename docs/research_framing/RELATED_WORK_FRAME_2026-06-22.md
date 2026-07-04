# Related Work Frame For Battle Royale Survival Modeling (2026-06-22)

## Purpose

This document is the canonical related-work frame for the paper.

The related work should not be organized as a narrow list of "PUBG prediction
papers." The paper is stronger if it is framed as:

```text
Battle Royale is a genre-level dynamic survival system.
PUBG is the empirical case study.
The proposed contribution is a mechanic-aligned, relational, spatio-temporal
survival representation of that system.
```

Detailed candidate bibliography and priority ranking remain in:

```text
docs/PRIOR_WORK_SURVEY_2026-06-22.md
```

Method skeleton and feature-manifest follow-through:

```text
docs/DSACG_METHOD_SKELETON.md
docs/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv
```

This document defines the argument structure that the final Related Work
section should follow.

## Claim Guardrail

Do not claim:

```text
No one has studied PUBG.
No one has studied Battle Royale.
No one has used graphs for games.
We are the first to predict survival in PUBG.
```

Safer core claim:

```text
Existing studies have examined Battle Royale mechanics, game telemetry,
PUBG-specific prediction, multi-agent spatio-temporal modeling, graph neural
networks, and survival analysis largely as separate threads. What remains
underdeveloped is a genre-level computational formalism that represents Battle
Royale matches as dynamic survival systems shaped jointly by shrinking arena
pressure, spatial exposure, combat interaction, resource competition, and
elimination-based ranking.
```

Korean shorthand:

```text
선행연구가 없다는 주장이 아니다. 선행연구들은 존재하지만, 배틀로얄 장르의
핵심 메커닉을 하나의 계산 가능한 시공간-관계형 생존 시스템으로 정식화한
연구 축이 약하다는 것이 우리의 gap이다.
```

## Eight Related-Work Axes

| Axis | Question it answers | Role in our paper |
| --- | --- | --- |
| Battle Royale genre definition | What is the domain object? | Establish that the target is a genre mechanic, not only PUBG |
| Formal game studies / ontology | How can game rules and mechanics become analytic primitives? | Justify translating loot, zone, combat, movement, and elimination into variables |
| Game telemetry and analytics | Why can gameplay be modeled from logs? | Justify official telemetry and event/snapshot reconstruction |
| Spatial and spatio-temporal game analytics | What does "spatial" mean beyond coordinates? | Define spatial exposure, density, movement, and arena pressure |
| PUBG / Battle Royale prediction | What has already been predicted in this domain? | Position Sagong-style phase survival and Kaggle placement baselines |
| Multi-agent spatio-temporal representation | How are moving agents and team outcomes modeled elsewhere? | Borrow representation logic from sports and esports analytics |
| Survival analysis / time-to-event modeling | Why is hazard a better target than plain classification? | Formalize at-risk team-phase cells and next-phase elimination |
| Graph and dynamic graph learning | How should relations be learned? | Justify message passing, typed relations, temporal graphs, and null controls |

## Final Paper Section Structure

Use this six-part structure for Section 2.

```text
2.1 Battle Royale as a Game Genre
2.2 Formal Game Modeling and Game Mechanics
2.3 Game Telemetry and Spatial Game Analytics
2.4 Battle Royale Prediction and PUBG Analytics
2.5 Multi-Agent Spatiotemporal Graph Representation
2.6 Relational Survival Modeling for Battle Royale
```

The order matters. It moves from domain definition to formalization, then from
data availability to predictive modeling, and finally to the proposed graph
survival formulation.

## 2.1 Battle Royale As A Game Genre

Purpose:

```text
Define the studied system before introducing the model.
```

What to establish:

- Battle Royale is not merely a shooter label.
- The genre combines many agents or teams, shared arena entry, scavenging or
  resource acquisition, movement, shrinking safe zone pressure, combat, one-life
  or near-one-life elimination, and final survivor/rank outcome.
- PUBG is a representative empirical instance, not the entire conceptual claim.

Sources already usable:

| Source | Status | Use |
| --- | --- | --- |
| PUBG official game-info page | verified | PUBG has up to 100 players, landing, looting, vehicles, Blue Zone, solo/duo/squad modes |
| General Battle Royale genre description | usable for background, prefer stronger scholarly source later | Genre definition: last-man-standing, survival/scavenging, shrinking safe area |
| Choi and Kim, "Battle Royale Game: In Search of a New Game Genre" | verify | Candidate scholarly genre framing |
| Google Play or industry Battle Royale genre report | verify | Optional market/genre motivation, not core scientific claim |

How this section leads to our work:

```text
If Battle Royale is defined by shared survival mechanics, then the model should
represent those mechanics directly: arena compression, spatial exposure, combat
interaction, resource state, and elimination order.
```

## 2.2 Formal Game Modeling And Game Mechanics

Purpose:

```text
Show that turning game mechanics into computational primitives is a legitimate
research move, not arbitrary feature engineering.
```

What to establish:

- Game studies provides vocabulary for rules, mechanics, state, agents, and
  player actions.
- For this paper, mechanics are not just presentation or design terms. They are
  the causal operators that change match state.
- In Battle Royale, the relevant mechanics are:
  - movement / rotation;
  - looting / resource acquisition;
  - combat / damage / elimination;
  - zone avoidance and forced movement;
  - team coordination and revival where available.

Sources already usable:

| Source | Status | Use |
| --- | --- | --- |
| Hunicke, LeBlanc, Zubek, MDA framework | verified PDF | Mechanics as the rule/data/algorithm layer that generates dynamics |
| Sicart, "Defining Game Mechanics" | verified | Mechanics as methods invoked by agents to interact with game state |
| Game Ontology Project | verify metadata | Candidate ontology vocabulary for formal game elements |
| Formal Abstract Design Tools / Church / Grünvogel | verify | Candidate support for formal design decomposition |

How this section leads to our work:

```text
Our DSACG formalism maps Battle Royale mechanics into graph primitives:
agents become nodes, interactions become typed edges, the zone becomes global
and spatial context, and elimination becomes a survival event.
```

## 2.3 Game Telemetry And Spatial Game Analytics

Purpose:

```text
Justify the data substrate: game logs can be transformed into analyzable
temporal, spatial, and behavioral representations.
```

What to establish:

- Game telemetry provides quantitative event traces of in-game behavior.
- PUBG official telemetry contains match event objects whose structure depends
  on event type, making it suitable for event reconstruction.
- Spatial analysis in games should include movement trajectories, local density,
  exposure to boundaries, resource field, and interaction range, not just raw
  x/y coordinates.

Sources already usable:

| Source | Status | Use |
| --- | --- | --- |
| PUBG Developer API telemetry docs | verified | Official source for match telemetry event data |
| PUBG API assets / data dictionaries | verified | Event/object schema support |
| Drachen et al., behavioral telemetry clustering | verified arXiv | Telemetry can produce actionable behavioral profiles |
| Bunian et al., HMM game behavior modeling | verified arXiv | Sequential game behavior matters beyond aggregate statistics |
| Spatial game analytics / trajectory analytics papers | verify | Strengthen the spatial analytics subsection |
| Activision Warzone Caldera dataset | optional / verify primary source | Public example of Battle Royale spatial/movement data release |

How this section leads to our work:

```text
Telemetry alone is not the contribution. The contribution is the reconstruction
of telemetry into causal 10-second snapshots, phase cuts, team states, spatial
context, and typed survival relations.
```

## 2.4 Battle Royale Prediction And PUBG Analytics

Purpose:

```text
Place the work against the closest domain-specific prediction studies.
```

What to establish:

- The closest prior work is Sagong et al. (2025), which models PUBG solo
  phase survival with FightScore, ActiveScore, and strategy types.
- Kaggle-style PUBG studies typically use public post-game aggregate features
  and predict final placement, not phase-causal survival.
- PUBG telemetry has also been used for related targets such as viewer
  engagement, which supports telemetry usefulness but does not solve our task.

Sources already usable:

| Source | Status | Use |
| --- | --- | --- |
| Sagong et al. (2025), "Are You a Hunter or Camper?" | local PDF + BibTeX | Main direct prior work; phase survival, FightScore/ActiveScore, solo setting |
| Rokad et al., "Survival of the Fittest in PlayerUnknown BattleGround" | verified arXiv | Public Kaggle post-game placement baseline family |
| Sen et al., PUBG final-rank prediction | verified arXiv | Feature-selection and aggregate placement baseline |
| Lee and Lee, PUBG rank factors | BibTeX has DOI; source page needs manual verification | Candidate journal paper for rank-prediction factors |
| Melhart et al., PUBG Twitch engagement | verified arXiv | Fine-grained PUBG telemetry can predict moment-level engagement |
| Kaggle PUBG Finish Placement Prediction | known dataset; page access limited in browser | Public external benchmark, but not phase survival |

How this section leads to our work:

```text
We do not replace Sagong et al.; we extend the problem setting. Their work is
phase-aware and Battle-Royale-specific, but solo and score-based. Our study
asks whether team survival can be represented as a dynamic relational system
with graph-null controls and relation ablations.
```

Presentation wording:

```text
Prior PUBG survival research models individual strategy and phase survival.
This study models team-level survival dependencies under shrinking-arena
pressure.
```

## 2.5 Multi-Agent Spatiotemporal Graph Representation

Purpose:

```text
Explain why a moving multi-agent match should not be modeled only as independent
rows of tabular features.
```

What to establish:

- In sports and esports, outcome-relevant behavior depends on movement and
  interactions among agents, not only individual state.
- Spatio-temporal graph models can encode agent positions, pairwise relations,
  temporal dynamics, and outcome prediction.
- Battle Royale differs from sports: the graph is many-team, adversarial,
  partially observed through telemetry, compressed by a shrinking arena, and
  terminated by elimination/ranking.

Sources already usable:

| Source | Status | Use |
| --- | --- | --- |
| Gudmundsson and Horton, team sports spatio-temporal survey | verified arXiv / DOI available | Moving agents and team interactions require spatio-temporal analysis |
| Hu and Sukthankar, ST-GCN team performance | verified arXiv | ST-GCN over behavioral traces, spatial relations, temporal dynamics, ablations |
| Antonini et al., time-window spatial activity graphs | verified arXiv | Constructing graph features from GPS movement windows |
| Dota 2 spatio-temporal team behavior | verified arXiv | Esports analogy for spatial behavior and skill/style differences |
| Tactical Graph Networks / sports graph papers | verify | Candidate strengthening references |

How this section leads to our work:

```text
The sports/esports literature supports the representation principle, but Battle
Royale needs a different graph semantics: edges are not teammate coordination
links only; they are survival dependencies under combat, proximity, and zone
pressure.
```

## 2.6 Relational Survival Modeling For Battle Royale

Purpose:

```text
Connect the target definition and the model family.
```

What to establish:

- Phase-wise elimination is naturally a discrete-time survival / event-history
  problem: each alive team-phase cell is at risk until elimination or match end.
- Time-varying covariates matter, so static classification is not enough.
- Graph neural networks provide message passing over entities and relations.
- R-GCN and other typed-edge operators are relevant because Battle Royale edges
  have different semantics: proximity, combat, zone co-pressure, resource
  competition, or contact-gated relative advantage.
- Dynamic graph models motivate future event-level extensions.

Sources already usable:

| Source | Status | Use |
| --- | --- | --- |
| Allison, discrete-time event-history methods | BibTeX verified | Statistical foundation for phase hazard |
| Singer and Willett, discrete-time survival | BibTeX verified | Event-history explanation for at-risk intervals |
| MATCH-Net | verified arXiv | Dynamic survival with temporal covariates |
| DeepHazard | verified arXiv | Neural time-varying risk |
| Gilmer et al., MPNN | verified arXiv | Message-passing abstraction |
| Battaglia et al., Graph Networks | verified arXiv | Relational inductive bias with entities and relations |
| Schlichtkrull et al., R-GCN | verified arXiv | Relation-specific graph convolution |
| Temporal Graph Networks | verified arXiv | Dynamic graph/event memory direction |

How this section leads to our work:

```text
The proposed DSACG model combines these threads: it is a discrete-time
survival model whose risk is conditioned on node state, global shrinking-arena
state, and typed relational messages among teams.
```

## Integrated Gap Statement

Use this as the bridge from Related Work to Method.

```text
Taken together, prior work establishes that Battle Royale games have distinctive
survival mechanics, that game telemetry can support behavioral modeling, that
PUBG outcomes can be predicted from handcrafted or aggregate features, and that
spatio-temporal graph and survival models are effective in other dynamic
multi-agent domains. However, these threads have not yet been integrated into a
genre-level computational formalism for Battle Royale survival. This study
addresses that gap by defining a Dynamic Shrinking-Arena Competitive Graph
(DSACG) and instantiating it on PUBG telemetry to predict phase-wise team
hazard and final ranking.
```

Korean explanation:

```text
선행연구의 각 조각은 있다. 장르 설명도 있고, 텔레메트리 분석도 있고, PUBG
예측도 있고, ST-GNN도 있고, survival analysis도 있다. 하지만 이 조각들을
배틀로얄의 장르 메커닉인 shrinking arena, 공간 노출, 교전, 자원 경쟁,
팀 생존/탈락 순위로 묶어 하나의 계산 가능한 그래프-생존 모델로 정식화한
축이 약하다. 우리가 세우는 기둥은 바로 이 통합 formalism이다.
```

## Method Implications

The related work should force the method design to answer five concrete
questions.

| Related-work pressure | Method requirement |
| --- | --- |
| Battle Royale genre mechanics | Inputs must be grouped by survival mechanism, not arbitrary feature lists |
| Formal mechanics | Every graph primitive must map to an in-game rule or mechanic |
| Telemetry analytics | Raw logs must be reconstructed into causal snapshots and phase cuts |
| Spatial analytics | Space must include exposure, density, movement, zone pressure, and relative position |
| PUBG prior work | Include Sagong-style strategy and Kaggle aggregate baselines where feasible |
| Multi-agent ST modeling | Include temporal-only, graph-only, and spatio-temporal graph ablations |
| Survival analysis | Use at-risk masks, phase hazard, calibration, and rank-aware outputs |
| Graph learning | Include no-graph, edge-shuffle, node-shuffle, relation-drop, and typed-edge controls |

## Minimum Baseline Set Implied By Related Work

| Baseline | Why it is necessary |
| --- | --- |
| Sagong-style FightScore / ActiveScore | Closest phase-aware Battle Royale survival prior |
| Kaggle-style aggregate placement model | Public PUBG benchmark family, even if task differs |
| Tabular snapshot/phase model | Tests whether graph is needed beyond features |
| Temporal-only model | Tests whether time explains most signal |
| Static graph model | Tests graph without temporal dynamics |
| ST graph model | Tests combined spatio-temporal relational hypothesis |
| Edge-shuffle / node-shuffle nulls | Tests whether topology/relations carry real signal |
| Relation-drop | Tests which relation family matters |

## Source Map

| Work / source | Category | Status | Link or local source |
| --- | --- | --- | --- |
| PUBG: BATTLEGROUNDS game info | Genre / PUBG mechanics | verified web | https://pubg.com/en/game-info/overview |
| PUBG Developer API Telemetry | Telemetry | verified web | https://documentation.pubg.com/en/telemetry.html |
| PUBG API assets | Telemetry schema | verified web | https://github.com/pubg/api-assets |
| Sagong et al. (2025) | Direct PUBG survival | local + BibTeX | `paper/battleground_paper.pdf`, `paper/references.bib` |
| Kaggle PUBG Finish Placement Prediction | Public PUBG benchmark | known dataset; browser page limited | https://www.kaggle.com/c/pubg-finish-placement-prediction |
| Rokad et al. (2019) | PUBG aggregate placement | verified arXiv | https://arxiv.org/abs/1905.06052 |
| Sen et al. (2021) | PUBG rank prediction | verified arXiv | https://arxiv.org/abs/2107.09016 |
| Lee and Lee (2025) | PUBG rank factors | BibTeX verified; web page manual verify | https://www.mdpi.com/2079-9292/14/3/626 |
| Melhart et al. (2020) | PUBG telemetry engagement | verified arXiv | https://arxiv.org/abs/2008.07207 |
| Hunicke, LeBlanc, Zubek (2004) | Formal game mechanics / MDA | verified PDF | https://users.cs.northwestern.edu/~hunicke/MDA.pdf |
| Sicart (2008) | Game mechanics | verified web | https://gamestudies.org/0802/articles/sicart |
| Game Ontology Project | Game ontology | verify metadata | https://www.gameontology.com/ |
| Drachen et al. | Behavioral telemetry | verified arXiv | https://arxiv.org/abs/1407.3950 |
| Bunian et al. | Sequential player modeling | verified arXiv | https://arxiv.org/abs/1804.00245 |
| Gudmundsson and Horton | Team sports ST survey | verified arXiv | https://arxiv.org/abs/1602.06994 |
| Hu and Sukthankar | ST-GCN team performance | verified arXiv | https://arxiv.org/abs/2206.10720 |
| Antonini et al. | Time-window spatial activity graphs | verified arXiv | https://arxiv.org/abs/2407.13109 |
| Dota 2 spatio-temporal team behavior | Esports spatial analogy | verified arXiv | https://arxiv.org/abs/1603.07738 |
| Allison (1982) | Discrete-time event history | BibTeX verified | `paper/references.bib` |
| Singer and Willett (1993) | Discrete-time survival | BibTeX verified | `paper/references.bib` |
| MATCH-Net | Dynamic survival | verified arXiv | https://arxiv.org/abs/1811.10746 |
| DeepHazard | Time-varying risk | verified arXiv | https://arxiv.org/abs/2007.13218 |
| Gilmer et al. | Message passing | verified arXiv | https://arxiv.org/abs/1704.01212 |
| Battaglia et al. | Graph networks / relational bias | verified arXiv | https://arxiv.org/abs/1806.01261 |
| Schlichtkrull et al. | R-GCN | verified arXiv | https://arxiv.org/abs/1703.06103 |
| Rossi et al. | Temporal Graph Networks | verified arXiv | https://arxiv.org/abs/2006.10637 |

## Remaining Literature Tasks

- [ ] Verify Choi and Kim's Battle Royale genre paper metadata and venue.
- [ ] Verify whether a Google Play / industry Battle Royale genre report is
  appropriate for motivation; do not use it as a scientific foundation.
- [ ] Manually search ACM DL / IEEE Xplore for Battle Royale genre, PUBG,
  game telemetry, and spatial game analytics papers hidden from generic search.
- [ ] Verify Game Ontology Project citation metadata.
- [ ] Verify Grünvogel / Formal Abstract Design Tools citation chain.
- [ ] Verify KRAFTON AI win-probability or Battle Royale prediction material, if
  it is to be cited.
- [ ] Add accepted BibTeX entries to `paper/references.bib`.

## One-Paragraph Related-Work Pitch

```text
This paper sits at the intersection of Battle Royale game mechanics, game
telemetry analytics, multi-agent spatio-temporal representation, graph neural
networks, and discrete-time survival modeling. Prior PUBG studies have shown
that phase-level behavior scores or aggregate match statistics can predict
survival or final placement, while broader game analytics and sports analytics
work demonstrates the value of telemetry, movement traces, and interaction
modeling. However, Battle Royale has not been sufficiently formalized as a
genre-level relational survival system. We address this gap by defining a
Dynamic Shrinking-Arena Competitive Graph and using PUBG telemetry as a
team-level empirical instantiation for phase-wise hazard and final-rank
prediction.
```
