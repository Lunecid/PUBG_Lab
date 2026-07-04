# PUBG Graph Modeling Research Frame (2026-06-22)

## Purpose

This document defines the empirical PUBG graph-modeling frame under the larger
Battle Royale DSACG formalism.

The center is not:

```text
Is PUBG a homophily graph?
```

That is only a diagnostic.

The center is:

```text
What does correct graph modeling mean for PUBG survival prediction?
```

## Core Thesis

PUBG match is graph-modelable when it is represented as a dynamic,
multi-relational, edge-attributed survival dependency graph rather than a
homophilic similarity graph.

Korean:

> PUBG 매치는 연결된 팀들이 서로 비슷하다는 의미의 그래프가 아니라, 팀
> 간 공간 노출, 교전, 안전공간 경쟁이 미래 생존 조건을 변화시키는 동적
> 다중관계 생존 의존성 그래프로 모델링될 때 타당하다.

## Strong Research Frame

Weak frame:

```text
We model PUBG matches as graphs and apply a GNN.
```

Strong frame:

```text
A PUBG match is not a set of independent teams. It is a dynamic,
attributed, multi-relational survival system in which team interactions and
shared arena constraints evolve over time. This study defines PUBG matches as
game-mechanic-aligned survival graphs and evaluates which node, edge, and global
representations best explain phase-wise hazard and final rank.
```

Key terms:

- dynamic;
- attributed;
- multi-relational;
- game-mechanic-aligned;
- survival dependency graph.

## Formal Graph Definition

For each match:

```text
G_1:T = {G_t}_{t=1}^T
```

At each 10-second snapshot or phase-aligned cut:

```text
G_t = (
  V_t,
  E_t^spatial,
  E_t^combat,
  E_t^habitat,
  X_t,
  A_t,
  U_t
)
```

| Component | Meaning |
| --- | --- |
| `V_t` | alive team nodes at time `t` |
| `X_t` | team state features |
| `E_t^spatial` | spatial exposure relation |
| `E_t^combat` | directed combat/damage relation |
| `E_t^habitat` | safe-space, route, chokepoint, destination competition relation |
| `A_t` | edge attributes |
| `U_t` | global arena context: phase, safe zone, alive teams, map context |

Prediction targets:

```text
yhat_i^{t+Delta}
  = P(team i eliminated in (t, t+Delta] | G_1:t)
```

Final rank:

```text
rhat_i = f(G_1:t, i)
```

## When Graph Modeling Is Valid

Do not claim:

```text
PUBG is a graph.
```

Claim:

```text
PUBG graph modeling is valid if a team's future survival state is better
explained by its own state plus relational edge features and global arena
context than by its own state alone.
```

Conditional-dependency form:

```text
Y_i^{t+Delta} not independent of (X_j^t, e_ij^{r,t})
given X_i^t and U_t
```

In words:

> Team i's future survival is not fully explained by team i's own state and the
> global arena context; another team's state and relation to i add information.

This is the precise meaning of an informative edge.

## Edge Meaning

PUBG edges do not represent team similarity.

They represent survival-relevant dependency:

> One team's state or action changes another team's future survival condition.

| Edge type | Weak interpretation | Correct interpretation |
| --- | --- | --- |
| proximity | nearby teams are similar | spatial exposure increases encounter/fight possibility |
| combat | teams are connected by a fight | directed damage/pressure changes survival risk |
| zone/path | teams share the same pressure | teams compete for the same survivable space or route |
| global context | repeated node feature | shared arena rule/phase condition that changes all relations |

Important sentence:

> Nearby teams are not similar teams. They are teams that can change each
> other's survival probability.

## Node Feature Contract

Team nodes are survival-state actors, not independent rows.

| Node feature group | Examples |
| --- | --- |
| team state | alive players, health, knocked count, revive status |
| combat state | recent damage dealt/received, kills, knocks, fight duration |
| mobility state | centroid position, spread, speed, heading, vehicle use |
| zone state | inside/outside safe zone, boundary distance, required rotation distance |
| resource/position advantage | loot proxy, terrain height, compound occupancy, cover proxy |

Rule:

> Node features describe the team's own state. Edge features describe the team's
> relation to another team or arena object.

Example:

- "team is outside safe zone" -> node feature;
- "two teams must enter the same safe route/chokepoint" -> edge feature.

## Edge Taxonomy

### E1. Spatial Exposure Edge

Do not define spatial relation only as a binary threshold.

Weak:

```text
d_ij < 100m -> edge
```

Better:

```text
e_ij^spatial =
[
  distance,
  relative angle,
  relative speed,
  closing speed,
  same sector,
  terrain advantage,
  cover/visibility proxy,
  zone-relative position
]
```

Meaning:

> Two teams are in a spatial relation where encounter, observation, or pressure
> becomes possible.

### E2. Combat Interaction Edge

Combat edges must be directed.

```text
e_{j -> i}^combat != e_{i -> j}^combat
```

Features:

```text
e_{j -> i}^combat =
[
  damage_{j -> i},
  knock_{j -> i},
  kill_{j -> i},
  time since last damage,
  combat duration,
  damage imbalance_{j,i}
]
```

Meaning:

> Team j directly changed team i's survival state through damage or combat
> pressure.

### E3. Habitat / Zone-Competition Edge

Do not connect teams merely because both are under zone pressure.

Better:

```text
e_ij^habitat =
[
  path overlap,
  same target sector,
  arrival time gap,
  shared chokepoint,
  safe-area competition,
  destination local density
]
```

Meaning:

> Two teams compete for the same survivable space, route, chokepoint, compound,
> or destination.

Safe zone decomposition:

| Location | Meaning |
| --- | --- |
| global context `U_t` | how compressed the arena is |
| node feature `z_i^t` | how pressured team i is |
| edge feature `e_ij^habitat` | whether teams i and j compete for the same survival space/path |

Safe zone should not be forced into team-team edges only.

## Message Passing Principle

Learning should not be neighbor averaging.

It should be relation interpretation:

```text
m_{j -> i}^{r,t}
  = phi_r(h_i^t, h_j^t, e_{j -> i}^{r,t}, U_t)
```

This matters because the same neighbor can mean different things depending on
team i's condition and global arena context.

Example:

- If team i is inside safe zone on high ground and team j is outside blue zone,
  team j may be under pressure.
- If team i is rotating outside safe zone and team j is holding a compound,
  team j may be a severe threat.

Therefore:

```text
neighbor state + ego state + edge relation + phase context
  -> survival message to ego team
```

## Relation-Specific Learning

Do not collapse all relations into one adjacency unless it is a baseline.

Weak:

```text
A = A_spatial + A_combat + A_habitat
```

Better:

```text
m_i^t =
  m_i^{spatial,t}
  concat m_i^{combat,t}
  concat m_i^{habitat,t}
```

or:

```text
m_i^t = sum_r beta_r^t m_i^{r,t}
```

Relation-specific message:

```text
m_i^{r,t}
  = Pool_{j in N_r(i)}
      alpha_ij^{r,t} phi_r(h_i^t, h_j^t, e_ij^{r,t}, U_t)
```

R-GCN is a typed-relation baseline. For PUBG, relation-specific weights should
eventually be extended with edge attributes and temporal context.

## Attention As Relation Selection

Attention is useful only if it answers:

> Which opponent or relation matters most for the current team's survival risk?

Relation-aware attention:

```text
alpha_ij^{r,t}
  = softmax_j(psi_r(h_i^t, h_j^t, e_ij^{r,t}, U_t))
```

Interpretation examples:

- spatial attention: who is the most dangerous positional neighbor?
- combat attention: who recently increased survival risk?
- habitat attention: who is likely to collide on the same safe-space route?

## Temporal Structure

PUBG is not a static graph.

```text
G_1 -> G_2 -> ... -> G_T
```

Recommended first implementation:

```text
snapshot-based dynamic multi-relational graph
```

Flow:

```text
G_t -> Graph Encoder -> H_t -> Temporal Encoder -> yhat_{t+Delta}, rhat
```

Event-based temporal graph modeling is a stronger future direction, but
snapshot-based dynamic graphs are more realistic for the current implementation.

## Research Questions

### RQ1. Graph Formalization

What dynamic graph representation can formalize a PUBG match as teams,
relations, and arena context?

Korean:

> 배틀그라운드 한 매치는 팀 노드, 관계 엣지, 경기장 context로 구성된
> 어떤 동적 그래프 표현으로 정식화될 수 있는가?

### RQ2. Edge Semantics

Which edge semantics best capture survival-relevant relations: spatial
exposure, combat interaction, or habitat/zone competition?

Korean:

> 팀 간 근접, 교전, 안전구역/경로 경쟁 관계 중 어떤 엣지 정의가 생존
> 예측에 가장 유효한가?

### RQ3. Relation Learning

How should graph relations be learned: smoothing, general attention,
relation-specific aggregation, edge-conditioned message passing, or temporal
relational MPNN?

Korean:

> PUBG의 팀 간 관계를 단순 평균 집계가 아니라 관계별 메시지로 학습할 때
> 생존 예측과 해석 가능성이 개선되는가?

### RQ4. Conditional Usefulness

When does graph modeling matter: by phase, team density, combat occurrence, or
zone pressure?

Korean:

> 그래프 모델링은 경기 전체에서 동일하게 중요한가, 아니면 특정 phase와
> 고압박 상황에서 특히 중요한가?

## Experiment Design

### Step 1. Representation Family

| Row | Meaning |
| --- | --- |
| `R0_node_only` | team state only |
| `R1_node_global` | team state + safe zone/phase |
| `R2_spatial_graph` | add spatial exposure edge |
| `R3_combat_graph` | add directed combat edge |
| `R4_habitat_graph` | add safe-space/route competition edge |
| `R5_full_multirelational` | spatial + combat + habitat |

This answers:

> Which graph representation is valid for PUBG survival prediction?

### Step 2. Learning Operator

Use the same representation while changing the learning operator.

| Operator | Meaning |
| --- | --- |
| standard GCN | smoothing baseline |
| GraphSAGE / GAT | general neighbor aggregation / attention |
| R-GCN | relation type separation |
| edge-conditioned MPNN | edge attributes shape messages |
| ego-preserving relational MPNN | keeps ego and neighbor states distinct |
| temporal relational MPNN | learns graph sequence dynamics |

This answers:

> How should edges be learned so they represent PUBG survival dependency rather
> than neighbor averaging?

### Step 3. Conditional Analysis

Report not only overall metrics but also:

- early/mid/late phase metrics;
- alive-team-count bins;
- safe-zone pressure bins;
- local density bins;
- before/after combat-event windows;
- map/mode slices if available.

This answers:

> When is graph modeling useful?

## Validity Criteria

### Construct Validity

Does the graph definition reflect PUBG mechanics?

Checks:

- team node choice is justified;
- spatial/combat/habitat edges represent different mechanics;
- zone is decomposed into global, node, and edge roles;
- no future information leaks into features or edges.

### Predictive Validity

Does relational information improve hazard and rank prediction beyond
node/global features?

Full graph does not need to dominate every phase. It should help in plausible
conditions such as mid/late phase, high density, active combat, or strong zone
pressure.

### Relational Validity

Do edge types play different roles?

Evidence:

- relation ablation;
- relation-specific attention/gate analysis;
- phase-wise importance;
- edge shuffle tests.

Contribution score:

```text
Delta_r = Performance_full - Performance_without_r
```

or:

```text
Delta_r = Loss_without_r - Loss_full
```

### Mechanistic Validity

Does the model react in PUBG-like ways?

| Situation | Expected relation |
| --- | --- |
| early phase | individual loot/mobility, zone distance |
| mid phase | proximity, rotation conflict |
| late phase | habitat competition, local density, combat pressure |
| after direct fight | combat edge |
| before safe-zone shrink | zone/habitat edge |

## Homophily And Smoothing Position

Homophily is not the main research question.

Use it as:

- negative motivation;
- baseline diagnosis;
- explanation for why standard GCN may fail;
- motivation for relation-aware, edge-conditioned, ego-preserving message
  passing.

Recommended wording:

```text
Standard GCN-style aggregation may be limited because PUBG relations are
competitive survival dependencies rather than homophilic similarity links.
However, this study does not aim to classify PUBG as homophilic or
heterophilic. The focus is to define game-mechanic-aligned edge semantics and
learn relation-specific survival messages.
```

Do not make homophily the thesis.

## Contribution Framing

The contribution does not need to be a brand-new GNN architecture.

Valid contributions:

1. Define PUBG match graph representation.
2. Propose Battle-Royale/PUBG survival edge taxonomy.
3. Test predictive validity of edge semantics.
4. Analyze smoothing-based vs relation-aware message passing.
5. Show when graph modeling matters through phase/density/pressure analyses.

## Abstract Core

Korean:

> 본 연구는 PUBG: Battlegrounds의 한 매치를 팀 단위의 독립 표본 집합이
> 아니라, 시간에 따라 변화하는 팀 간 상호작용과 경기장 제약이 결합된
> 동적 다중관계 그래프로 정식화한다. 본 연구에서 엣지는 팀 간 유사성을
> 의미하지 않고, 근접 노출, 실제 교전, 안전구역 및 이동 경로 경쟁처럼
> 한 팀의 미래 생존 조건을 변화시키는 관계적 의존성을 의미한다. 이를
> 바탕으로 본 연구는 노드 feature, global arena context,
> relation-specific edge feature를 분리하여 구성하고, 어떤 edge semantics와
> message-passing 방식이 phase-wise survival hazard와 final rank prediction에
> 유효한지 검증한다.

English:

> This study formalizes a PUBG: Battlegrounds match not as a collection of
> independent teams, but as a dynamic multi-relational graph in which team
> survival is shaped by evolving spatial exposure, combat interactions, and
> safe-zone-mediated competition. In this formulation, edges do not represent
> similarity between teams; rather, they encode survival-relevant relational
> dependencies that alter a team's future hazard. We investigate how node
> attributes, global arena context, and relation-specific edge attributes should
> be defined and learned to support phase-wise survival and final rank
> prediction.

## Title Candidates

Stable:

```text
Game-Mechanic-Aligned Graph Modeling of PUBG Matches for Phase-wise Team Survival Prediction
```

Stronger:

```text
Defining Battle Royale Matches as Dynamic Multi-Relational Graphs for Team Survival Prediction
```

Question style:

```text
What Makes a Battle Royale Match a Graph? Dynamic Multi-Relational Representation Learning for PUBG Survival Modeling
```

