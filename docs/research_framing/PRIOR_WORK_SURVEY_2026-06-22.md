# Prior Work Survey And Positioning Plan (2026-06-22)

This document ranks prior work for the PUBG survival-prediction project.

Canonical related-work structure:

```text
docs/RELATED_WORK_FRAME_2026-06-22.md
```

Use that file when writing the paper's Related Work section. This file remains
the candidate bibliography, priority list, and baseline-planning inventory.
The final paper should not be organized as a narrow list of "PUBG prediction
papers"; it should follow the Battle Royale genre -> formal game mechanics ->
telemetry/spatial analytics -> PUBG prediction -> multi-agent ST graph ->
relational survival modeling structure defined in the related-work frame.

The ranking follows the requested priority order:

1. venue / journal reputation;
2. direct PUBG relevance;
3. battle-royale relevance;
4. feature design, ablation design, and game-data modeling relevance;
5. graph and spatio-temporal graph modeling relevance;
6. other useful adjacent domains.

This is a living survey. Items marked `verify` should not be used as final
paper claims until their publication metadata has been checked.

---

## 1. Current Positioning

The nearest direct prior work is:

```text
Sagong et al. (2025), Entertainment Computing
Are you a Hunter or Camper? A phase-based framework for real-time survival
prediction in battle royale using FightScore and ActiveScore
```

Why it matters:

- direct PUBG / battle-royale survival prediction;
- phase-specific prediction, not only post-game aggregate placement;
- uses behavior scores, strategy types, and next-phase survival prediction;
- local PDF says the data are confidential, solo-only, collected January 1-7,
  2021, with position data recorded every 10 seconds.

Our differentiator should not be stated as simply "we use a GNN."

Stronger positioning:

```text
Prior PUBG survival work models phase-specific solo-player strategy through
Aggressiveness/Activeness scores. We extend the problem to team-level dynamic
survival prediction by representing each match as a time-varying team graph,
where nodes are teams, edges encode contact/combat/zone pressure, and outputs
are at-risk team-phase hazard forecasts.
```

---

## 2. Priority Tier A: Direct PUBG / Battle-Royale Prior Work

| Rank | Work | Venue / status | Relevance | Data / task | How to use |
| --- | --- | --- | --- | --- | --- |
| A1 | Sagong et al., "Are you a Hunter or Camper?" | Entertainment Computing 55, 101040, 2025; local PDF | Highest direct prior | PUBG solo logs; phase survival/win prediction; FightScore/ActiveScore | Main prior-work baseline and motivation for phase-aware survival |
| A2 | Melhart et al., "Moment-to-moment Engagement Prediction through the Eyes of the Observer: PUBG Streaming on Twitch" | arXiv/HCI, 2020 | PUBG telemetry but different target | PUBG streaming telemetry + Twitch chat; engagement prediction | Supports use of fine-grained PUBG telemetry for moment-level prediction |
| A3 | Rokad et al., "Survival of the Fittest in PlayerUnknown BattleGround" | arXiv, 2019 | PUBG but post-game placement | Kaggle 29-feature post-game aggregate; `winPlacePerc` regression | Public baseline family; low relevance to dynamic survival |
| A4 | Sen et al., "Prediction of the final rank of Players in PUBG with the optimal number of features" | arXiv, 2021 | PUBG but post-game placement | Kaggle 29-feature final-rank prediction | Feature-selection/post-game benchmark only |
| A5 | Kaggle PUBG Finish Placement Prediction | Kaggle competition, 2018 | Public PUBG data source | post-game aggregate placement prediction | External public benchmark, not a replacement for Sagong data |
| A6 | Lee & Lee, "A Study on the Factors Influencing Rank Prediction in PUBG" | `verify`; local `references.bib` says Electronics 14(3):626, 2025 | Potential PUBG journal paper | rank prediction | Verify DOI/page before citing |

Directness breakdown:

| Work | Directness | Temporal causality | Unit | Target | Use |
| --- | --- | --- | --- | --- | --- |
| Sagong et al. | high | phase-causal | solo player | next-phase survival / win | closest baseline |
| Lee & Lee | medium | mostly aggregate | player | rank factor / prediction | feature-prior reference |
| Rokad et al. | medium | post-game | player | `winPlacePerc` | Kaggle benchmark family |
| Sen et al. | medium | post-game | player | final rank | feature-selection benchmark |
| Melhart et al. | medium | moment-level but different target | observer / engagement | viewer engagement | telemetry usefulness, not survival target |

Why this matters:

```text
Sagong et al. is the closest baseline because it is phase-aware and Battle
Royale survival-oriented. Kaggle-style studies are useful public benchmarks,
but they mostly evaluate post-game aggregate placement prediction rather than
phase-causal survival hazard.
```

Sources:

- Sagong et al.: local file `paper/battleground_paper.pdf`
- Rokad et al.: https://arxiv.org/abs/1905.06052
- Sen et al.: https://arxiv.org/abs/2107.09016
- Melhart et al.: https://arxiv.org/abs/2008.07207
- Kaggle PUBG competition: https://www.kaggle.com/competitions/pubg-finish-placement-prediction

Key distinction:

```text
Sagong-style data = private phase-level PUBG solo telemetry.
Kaggle data = public post-game aggregate placement data.
Our data = telemetry-derived team-level dynamic graph data with snapshots and
phase cuts.
```

---

## 3. Tier B: Game Telemetry, Strategy, And Player Modeling

These works are not PUBG survival papers, but they justify feature design and
behavioral abstraction.

| Work | Venue / status | Main idea | Use for our project |
| --- | --- | --- | --- |
| Drachen et al., "A Comparison of Methods for Player Clustering via Behavioral Telemetry" | FDG 2013 / arXiv | large-scale telemetry clustering and actionable behavioral profiles | Supports strategy-type baselines and behavioral feature grouping |
| Bunian et al., "Modeling Individual Differences in Game Behavior using HMM" | arXiv, 2018 | sequential game actions can reveal player differences better than aggregates | Supports temporal behavior modeling over phase/snapshot history |
| Melhart et al., "Towards General Models of Player Experience: A Study Within Genres" | IEEE CoG 2021 / arXiv | abstract gameplay metrics can generalize within genre; time features are important | Supports domain-level feature families and temporal dynamics |

Sources:

- Drachen et al.: https://arxiv.org/abs/1407.3950
- Bunian et al.: https://arxiv.org/abs/1804.00245
- Melhart et al. genre models: https://arxiv.org/abs/2110.00978

Design implication:

```text
Do not only report raw feature lists. Group features into interpretable
behavioral/domain mechanisms: roster, rotation, zone pressure, combat,
resource readiness, strategy, and topology.
```

---

## 4. Tier C: Graph And Spatio-Temporal Graph Modeling Foundations

These works are method foundations rather than game-domain prior work.

| Work | Venue / status | Main idea | Use for our project |
| --- | --- | --- | --- |
| Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" | ICLR 2017 / arXiv | first-order graph convolution over node features and adjacency | Baseline GCN explanation |
| Hamilton et al., "Inductive Representation Learning on Large Graphs" | NeurIPS 2017 / arXiv | GraphSAGE inductive neighborhood aggregation | Useful because matches contain unseen team-node sets |
| Velickovic et al., "Graph Attention Networks" | ICLR 2018 / arXiv | learn neighbor importance with attention | Justifies GAT variant |
| Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks" | ESWC 2018 / arXiv | R-GCN with relation-specific weights | Justifies preserving proximity/combat/co-pressure channels separately |
| Yu et al., "Spatio-Temporal Graph Convolutional Networks" | IJCAI 2018 / arXiv | TCN + graph convolution for time-varying graph signals | Main architectural ancestor for ST-GNN framing |
| Li et al., "Diffusion Convolutional Recurrent Neural Network" | ICLR 2018 / arXiv | directed graph diffusion + recurrent temporal modeling | Supports directed combat/contact relations |
| Rossi et al., "Temporal Graph Networks" | arXiv, 2020 | dynamic graph event sequence with memory and ablations | Future target for event-level PUBG telemetry |
| Xu et al., "Inductive Representation Learning on Temporal Graphs" | arXiv, 2020 | temporal graph attention and time encoding | Supports continuous-time/contact-event graph design |

Sources:

- GCN: https://arxiv.org/abs/1609.02907
- GraphSAGE: https://arxiv.org/abs/1706.02216
- GAT: https://arxiv.org/abs/1710.10903
- R-GCN: https://arxiv.org/abs/1703.06103
- STGCN: https://arxiv.org/abs/1709.04875
- DCRNN: https://arxiv.org/abs/1707.01926
- TGN: https://arxiv.org/abs/2006.10637
- TGAT: https://arxiv.org/abs/2002.07962

Design implication:

```text
GCN = homogeneous relation smoothing.
R-GCN = relation-aware message passing.
Edge-conditioned / temporal graph models = better fit for contact-gated
relative advantage edges.
```

---

## 5. Tier D: Evaluation, Ablation, And Null-Control Warnings

These are important because our graph controls already showed that topology can
be misleading.

| Work | Venue / status | Main idea | Use for our project |
| --- | --- | --- | --- |
| Shchur et al., "Pitfalls of Graph Neural Network Evaluation" | arXiv, 2018 | model rankings change under splits/tuning; simple baselines can win | Justifies seed repeats, fixed splits, fair tuning |
| Liu et al., "Do We Really Need Graph Neural Networks for Traffic Forecasting?" | arXiv, 2023 | simpler spatial-temporal baselines can rival STGNNs | Justifies no-graph, temporal-only, and spatial-only controls |
| Gong & Cheng, "Exploiting Edge Features in Graph Neural Networks" | arXiv, 2018 | edge features can carry important information beyond adjacency | Supports contact-gated relative edge features |

Sources:

- Pitfalls: https://arxiv.org/abs/1811.05868
- Do We Really Need GNNs: https://arxiv.org/abs/2301.12603
- Edge features: https://arxiv.org/abs/1809.02709

Design implication:

```text
Graph contribution is not accepted unless real/unshuffled edges beat:
no_graph, edge_shuffle, node_shuffle, relation_drop, and temporal-only baselines
under the same split/seed/tuning budget.
```

---

## 6. Tier E: Spatio-Temporal Team Sports As An Analogy

Team sports are not battle royale, but they provide strong analogies for
tracking multiple agents, spatial relations, and team-level outcomes.

| Work | Venue / status | Main idea | Use for our project |
| --- | --- | --- | --- |
| Gudmundsson & Horton, "Spatio-Temporal Analysis of Team Sports: A Survey" | ACM CSUR related DOI / arXiv, 2016 | team performance requires movement and interactions of all players | Supports spatial definition and team-interaction framing |
| Hu & Sukthankar, "Predicting Team Performance with Spatial Temporal Graph Convolutional Networks" | ICPR 2022 / arXiv | ST-GCN over behavioral traces; ablation of architecture components | Strong analogy for team movement + graph + outcome prediction |
| Antonini et al., "A Framework for Spatio-Temporal Graph Analytics In Field Sports" | arXiv, 2024 | time-window spatial activity graphs from GPS data | Supports windowed graph construction from trajectories |

Sources:

- Team sports survey: https://arxiv.org/abs/1602.06994
- Team performance ST-GCN: https://arxiv.org/abs/2206.10720
- Field sports graph analytics: https://arxiv.org/abs/2407.13109

Design implication:

```text
Spatial is not only x/y coordinates. It is position, movement, local density,
relative pressure, and interaction structure.
```

---

## 7. Tier F: Survival / Dynamic Risk Modeling

These works justify treating the task as phase-cut hazard prediction rather
than generic classification.

| Work | Venue / status | Main idea | Use for our project |
| --- | --- | --- | --- |
| Allison, "Discrete-Time Methods for the Analysis of Event Histories" | Sociological Methodology, 1982 | discrete-time hazard modeling | Statistical foundation for phase-cut survival |
| Singer & Willett, "It's About Time" | Journal of Educational Statistics, 1993 | discrete-time survival/event history framing | Supports at-risk interval formulation |
| Jarrett et al., "MATCH-Net" | arXiv, 2018 | temporal convolution for dynamic survival prediction with longitudinal covariates | Supports snapshot history -> hazard |
| Rava & Bradic, "DeepHazard" | arXiv, 2020 | neural model for time-varying risks | Supports time-varying covariates and non-proportional hazards |

Sources:

- MATCH-Net: https://arxiv.org/abs/1811.10746
- DeepHazard: https://arxiv.org/abs/2007.13218

Design implication:

```text
The clean target is an at-risk team-phase cell:
given information up to phase k, predict elimination before phase k+1.
```

---

## 8. What This Means For Our Research Design

### 8.1 Prior-Work Baseline

Implement a Sagong-style baseline on our data:

```text
FightScore = phase_attack_count + phase_kill_count
           + phase_total_damage + phase_assist_count

ActiveScore = phase_distance + phase_max_density
            + phase_item_count + carepackage_use

Strategy type = Hunter / Sniper / Traveler / Camper
```

Important adaptation:

```text
Sagong is solo-player.
Our main setting is team-level.
Therefore, all Sagong-style features must be team-aggregated and
roster-normalized where appropriate.
```

### 8.2 Main Novelty Claim

Do not claim:

```text
We are the first to predict survival in PUBG.
```

Safer claim:

```text
We extend phase-specific battle-royale survival prediction from individual
strategy scores to team-level dynamic spatio-temporal graphs, with explicit
graph-null controls and relation-type ablations.
```

### 8.3 Required Ablations

Minimum ablation ladder:

```text
Sagong-style strategy baseline
  -> + zone/rotation features
  -> + team resource/roster features
  -> + temporal model
  -> + phase graph
  -> + snapshot graph
  -> + contact-gated relative edges
```

Minimum graph controls:

```text
no_graph
edge_shuffle
node_shuffle
relation_drop
node-only temporal baseline
same-split same-seed repeats
```

Minimum slices:

```text
phase band
map
mode/team size
zone pressure band
local opponent density band
alive roster band
graph condition
```

---

## 9. Search Gaps / Next Literature Tasks

- [ ] Verify Lee & Lee (2025), Electronics 14(3):626, DOI `10.3390/electronics14030626`.
- [ ] Search IEEE Xplore / ACM DL manually for PUBG, battle royale, and game telemetry papers hidden from generic web search.
- [ ] Search Korean DBs if the work targets a Korean venue or thesis ecosystem.
- [ ] Add BibTeX entries for all accepted references.
- [ ] Separate final related-work section into:
  - PUBG and battle-royale prediction;
  - game telemetry and strategy modeling;
  - dynamic graph and ST-GNN modeling;
  - discrete-time survival / hazard prediction.

---

## 10. Presentation-Ready Short Summary

```text
The closest prior work is Sagong et al. (Entertainment Computing, 2025),
which predicts PUBG solo-player next-phase survival using phase-specific
FightScore/ActiveScore strategy types. Most other PUBG studies use the public
Kaggle dataset and predict post-game placement from aggregate statistics.
Our work moves beyond post-game aggregate prediction and solo strategy scoring
by modeling team-level PUBG survival as a dynamic spatio-temporal graph
problem: teams are nodes, contact/zone/combat relations are edges, and the
target is next-phase elimination hazard for each at-risk team.
```
