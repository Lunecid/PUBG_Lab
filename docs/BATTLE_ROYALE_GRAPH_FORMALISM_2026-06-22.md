# Battle Royale Graph Formalism (2026-06-22)

## Purpose

This document reframes the project from:

```text
Can PUBG be modeled as a graph?
```

to:

```text
Can Battle Royale matches be formalized as a genre-level relational survival
system, and can that representation explain survival hazard and final ranking?
```

PUBG is the empirical case study. The larger object is the Battle Royale genre.

Empirical PUBG research frame:

```text
docs/PUBG_GRAPH_MODELING_RESEARCH_FRAME_2026-06-22.md
```

Related-work frame supporting this formalism:

```text
docs/RELATED_WORK_FRAME_2026-06-22.md
```

## Careful Scope

Do not claim that no Battle Royale analytics research exists. Existing work
includes title-specific prediction, movement analysis, safe-zone analysis,
randomness analysis, and match-statistics modeling.

The safer claim is:

> Existing studies are often title-specific or feature-based. A genre-level,
> mechanic-aligned, relational graph formalism for Battle Royale survival
> modeling is still underdeveloped.

Citation guardrail:

> Before this wording enters the final paper, cite representative prior work on
> PUBG placement prediction, Battle Royale movement/zone analysis, safe-zone or
> randomness studies, and general game telemetry mining. The claim is
> "underdeveloped genre-level formalism," not "no prior work exists."

## Problem Definition

### English

Battle Royale games share a distinctive competitive structure in which many
players or teams enter a common arena, acquire limited resources, move under a
progressively shrinking safe area, and compete through combat and elimination
until the final survivor or squad remains. Existing game analytics studies have
largely focused on title-specific match statistics or handcrafted predictive
features, leaving a lack of genre-level computational formalism that captures
the shared mechanics of Battle Royale games. This study formalizes a Battle
Royale match as a dynamic shrinking-arena competitive graph, where agents,
spaces, resources, zones, and combat events form time-evolving relational
dependencies. We investigate whether this graph representation can explain and
predict phase-wise survival hazard and final ranking.

### Korean

배틀로얄 장르는 다수의 플레이어 또는 팀이 동일한 arena 안에서 제한된
자원을 획득하고, 시간에 따라 축소되는 안전 공간 안으로 이동하며, 교전과
제거를 통해 최종 생존자를 결정하는 독특한 경쟁 구조를 가진다. 그러나
기존 게임 분석 연구는 주로 특정 게임의 match statistics나 개별 feature를
사용한 예측에 집중해 왔고, 배틀로얄 장르의 공통 메커닉을 장르 수준에서
일반화할 수 있는 수학적, 컴퓨터공학적 표현은 충분히 정식화되어 있지
않다. 본 연구는 배틀로얄 매치를 agent, space, resource, zone, combat
event가 시간에 따라 관계를 형성하는 dynamic shrinking-arena competitive
graph로 정의하고, 이 그래프 표현이 phase-wise survival hazard와 final
ranking을 설명하고 예측하는 데 유효한지 검증한다.

## Research Stack

| Layer | Content |
| --- | --- |
| Domain problem | Battle Royale is expanding, but genre-level computational modeling is weak |
| Representation problem | Survival, space, shrinking zone, combat, resources, and rank must be represented together |
| Proposed solution | Define a Battle Royale match as a Dynamic Shrinking-Arena Competitive Graph |
| Empirical task | Predict phase-wise survival hazard and final rank |
| Validation | Use PUBG as a case study; test map/mode/phase transfer and relation ablations |

## Battle Royale As A Survival System

Define a Battle Royale match as:

```text
M = <A, Omega, T, Z_1:T, R_1:T, C_1:T, S_1:T, Y>
```

| Symbol | Meaning |
| --- | --- |
| `A` | players or teams |
| `Omega` | spatial arena / map |
| `T` | match time or phase sequence |
| `Z_1:T` | time-varying safe zone / playable area |
| `R_1:T` | resources: loot, vehicle, cover, terrain, supply |
| `C_1:T` | combat events: damage, knock, kill, revive |
| `S_1:T` | agent/team state trajectories |
| `Y` | elimination time, survival rank, winner |

Core properties:

```text
|A| >> 2
```

The system is large-scale multi-agent competition, not a two-team duel.

```text
Omega_t subseteq Omega_{t-1}
```

The effective safe/playable space shrinks over time.

```text
a_i in A -> tau_i
```

Each agent/team has an elimination time.

```text
Y_i = rank(tau_i)
```

Final ranking is tied to survival time and elimination order.

Short definition:

> Battle Royale is a dynamic survival system in which many agents compete over
> space, resources, combat, and position pressure inside a shrinking arena.

## Genre-Level Primitive Vocabulary

The representation should not depend on PUBG-specific variable names. It should
start from genre-level primitives.

| Primitive | Meaning | PUBG-specific example |
| --- | --- | --- |
| Agent | player or team survival subject | player/team |
| Arena | playable spatial domain | Erangel/Miramar coordinates |
| Hazard field | unsafe area or damaging boundary | blue zone |
| Compression phase | shrinking-area stage | circle phase |
| Exposure | chance of contact or being seen/fought | proximity, local density |
| Encounter | possible or actual combat | damage/knock/kill exchange |
| Resource field | loot, vehicle, weapon, healing, objective | vehicle, item, drop |
| Habitat | survivable/defensible space | compound, cover, high ground |
| Rotation | movement toward viable space | route to safe zone |
| Elimination | absorbing survival event | team death |
| Rank | survival-time-based ordering | final placement |

PUBG-specific features should be mapped to these primitives before model design.

## Dynamic Shrinking-Arena Competitive Graph

Recommended model name:

```text
Dynamic Shrinking-Arena Competitive Graph (DSACG)
```

Rationale:

- Dynamic: state and relations evolve over time.
- Shrinking-Arena: safe/hazard zone compresses the playable space.
- Competitive: teams compete over survival conditions.
- Graph: entities and dependencies are relational.

Define the general Battle Royale graph sequence:

```text
G_1:T^BR = {G_t^BR}_{t=1}^T
```

Full heterogeneous form:

```text
G_t^BR =
(
  V_t^A,
  V_t^S,
  V_t^R,
  V_t^Z,
  E_t^AA,
  E_t^AS,
  E_t^AR,
  E_t^AZ,
  E_t^SS,
  X_t,
  A_t,
  U_t
)
```

| Component | Meaning |
| --- | --- |
| `V_t^A` | agent/team nodes |
| `V_t^S` | spatial region nodes |
| `V_t^R` | resource nodes |
| `V_t^Z` | zone/hazard nodes |
| `E_t^AA` | agent-agent relations: exposure, combat, competition |
| `E_t^AS` | agent-space relations: occupancy, movement, cover |
| `E_t^AR` | agent-resource relations: loot, vehicle, supply access |
| `E_t^AZ` | agent-zone relations: pressure, distance, hazard exposure |
| `E_t^SS` | space-space relations: adjacency, route, chokepoint |
| `X_t` | node attributes |
| `A_t` | edge attributes |
| `U_t` | global match/arena context |

## Team-Level Projection For PUBG

The full DSACG is an agent-environment heterogeneous graph. In this project,
given data availability and prediction target, we instantiate it as a
team-level projection:

```text
Full BR graph -> team-level competitive survival graph
```

Projection:

```text
Gbar_t^BR =
(
  Vbar_t^team,
  Ebar_t^exposure,
  Ebar_t^combat,
  Ebar_t^habitat,
  Ebar_t^resource,
  Ebar_t^zone,
  Xbar_t,
  Abar_t,
  U_t
)
```

Interpretation:

> PUBG team nodes are not the whole genre-level graph. They are a practical
> projection of agent, space, resource, zone, and event relations into
> team-team and team-context tensors.

This prevents the research from being trapped as "a PUBG team graph." The
paper can claim a genre-level formalism and a PUBG team-level instantiation.

Projection scope:

> The proposed DSACG is a general heterogeneous formalism over agents, spaces,
> resources, zones, and combat events. In the empirical PUBG setting, we use a
> team-level projection because the target variable is team survival and the
> available telemetry can be causally aggregated into team states and
> team-to-team relations. This projection does not claim to exhaust the full
> DSACG; rather, it instantiates the subset of DSACG relations observable and
> relevant for team-level hazard prediction.

Korean shorthand:

> DSACG의 일반형은 agent, space, resource, zone, combat event를 포함하는
> 이종 그래프다. 그러나 본 연구의 실증 단계에서는 예측 대상이 팀 생존이고,
> telemetry가 팀 단위 상태와 팀-팀 관계로 안정적으로 집계될 수 있기 때문에
> team-level projection을 사용한다. 이 projection은 DSACG 전체를 완전히
> 구현한다는 의미가 아니라, 팀 생존 hazard 예측에 관측 가능하고 관련 있는
> 관계들을 구현한 empirical instantiation이다.

Projection owner map:

| Primitive | General DSACG owner | PUBG team-level projection |
| --- | --- | --- |
| Agent/team state | agent/team node | team node feature |
| Zone pressure | zone node / global context / agent-zone edge | team node feature + global context |
| Spatial exposure | agent-agent / agent-space edge | team-team edge |
| Combat interaction | combat event / directed agent-agent edge | directed team-team edge |
| Habitat competition | space node / agent-space / agent-agent projection | team-team habitat edge |
| Resource readiness | resource node / agent-resource edge | team node feature or projected resource-advantage edge |
| Temporal continuity | temporal edge | team hidden-state sequence |

## Edge Meaning: Survival Dependency

In Battle Royale, an edge should mean:

> One entity's state changes another entity's future survival condition.

Generic edge function:

```text
e_ij^{r,t} = psi_r(s_i^t, s_j^t, u_t, omega_t)
```

| Symbol | Meaning |
| --- | --- |
| `s_i^t` | state of entity/team i |
| `s_j^t` | state of entity/team j |
| `u_t` | global context: phase, safe zone, alive teams |
| `omega_t` | environmental context: map, terrain, resource field |
| `r` | relation type |

Good edge question:

```text
Under the current phase and spatial condition, how does team j change team i's
future survival/rank condition?
```

Weak edge question:

```text
Are team i and team j connected?
```

## Edge Taxonomy

| Edge type | Meaning | Examples | Direction |
| --- | --- | --- | --- |
| spatial exposure | teams can encounter or pressure each other spatially | distance, bearing, closing speed, line-of-sight proxy, cover, sector | often symmetric but can be directed via advantage |
| combat interaction | actual damage/knock/kill pressure | damage `i -> j`, knock, kill, time since hit, damage imbalance | directed |
| zone pressure | shrinking arena pressure on team survival | boundary distance, required speed, time-to-zone, hazard exposure | team-zone or team-context |
| habitat/route competition | teams compete for viable space/routes | same compound, choke point, high ground, route overlap, destination density | usually team-team |
| resource competition | teams compete for limited resources | loot zone, vehicle, drop, resource advantage | team-resource or projected team-team |
| temporal continuity | same team evolves through time | `team_i^t -> team_i^{t+1}` | directed through time |

Important:

> Proximity/combat/co-pressure are baseline projections of richer DSACG
> relations, not the final edge ontology.

## Relational Survival Learning Objective

The learning problem is not just "apply a GNN." It is relational survival
learning.

Hidden-state update:

```text
h_i^t =
Update(
  h_i^{t-1},
  x_i^t,
  u_t,
  OPLUS_{r in R} AGG_j phi_r(h_i^t, h_j^t, e_ij^{r,t}, u_t)
)
```

Survival hazard:

```text
lambda_i^t = rho(h_i^t, u_t)
```

Discrete phase-cut hazard:

```text
p_i^{t+Delta} = sigmoid(f_theta(G_1:t^BR, i))
```

Final ranking:

```text
rhat_i = g_theta(G_1:t^BR, i)
```

or survival time:

```text
tauhat_i = h_theta(G_1:t^BR, i)
```

Main empirical outputs:

- phase-wise survival hazard;
- final rank / final placement score;
- optional top-k placement probability;
- calibration of hazard probabilities.

## Validity Framework

A genre-level graph formalism needs more than high AUC.

### 1. Construct Validity

Question:

> Does the graph definition represent Battle Royale mechanics?

Checks:

- large-scale multi-agent survival is represented by agent/team nodes;
- shrinking arena is represented by global, zone, or edge relations;
- combat is represented as directed interaction;
- resource/habitat competition is represented as relation;
- elimination is represented as an absorbing survival event.

### 2. Predictive Validity

Question:

> Does adding relational information improve hazard and rank prediction?

Suggested representation ladder:

| Row | Representation |
| --- | --- |
| `R0` | agent/team only |
| `R1` | agent/team + global arena |
| `R2` | + spatial exposure |
| `R3` | + combat |
| `R4` | + zone/habitat |
| `R5` | full DSACG projection |

### 3. Mechanistic Validity

Question:

> Does the model react in Battle-Royale-like ways?

Expected patterns:

| Situation | Relation expected to matter |
| --- | --- |
| early phase | resource, mobility, initial location |
| mid phase | rotation, exposure, zone pressure |
| late phase | habitat competition, local density, combat |
| after direct fight | directed combat edge |
| before safe-zone shrink | route conflict, zone pressure |
| many teams in small area | spatial exposure, habitat competition |

### 4. Transfer Validity

Question:

> Does the representation generalize beyond one handcrafted PUBG feature set?

Practical levels:

| Level | Validation |
| --- | --- |
| 1 | PUBG map transfer |
| 2 | mode transfer: solo/duo/squad |
| 3 | phase-band transfer: early/mid/late |
| 4 | public/synthetic Battle Royale simulator rule transfer |
| 5 | cross-title transfer if another telemetry source becomes available |

For the current paper, Level 1-3 are realistic. Cross-title transfer should be
positioned as future work unless data is obtained.

## Data Availability Motivation

Detailed game telemetry is often controlled by game companies and is not always
available to external researchers across titles. This can be used as motivation,
but carefully:

```text
Because cross-title official telemetry is difficult to obtain, a useful Battle
Royale representation should depend on mechanic-level primitives rather than
one game's variable names.
```

This supports the DSACG direction:

```text
PUBG blue zone distance       -> hazard boundary distance
PUBG circle phase             -> arena compression phase
PUBG vehicle count            -> mobility resource availability
PUBG damage dealt             -> directed combat pressure
PUBG team spread              -> squad cohesion
PUBG final placement          -> survival rank
```

The current project should therefore report both:

- PUBG-specific feature names for reproducibility;
- genre-level primitive names for generalization.

## Contributions

Recommended contribution framing:

1. Genre-level formalization: define Battle Royale as a dynamic
   shrinking-arena competitive survival system.
2. Graph schema: propose a DSACG schema over agents, spaces, zones, resources,
   combat events, and temporal relations.
3. Empirical instantiation: implement a PUBG team-level multi-relational
   survival graph.
4. Generalization protocol: evaluate map/mode/phase transfer and relation/null
   controls.

## Current Project Implications

This formalism changes the research design as follows.

| Area | Decision |
| --- | --- |
| problem statement | Battle Royale genre-level formalism; PUBG is case study |
| model name | Dynamic Shrinking-Arena Competitive Graph, DSACG |
| empirical graph | team-level projection of agent-space-resource-zone-event graph |
| outputs | phase-wise hazard + final rank; top-k optional |
| edge design | survival dependency, not mere proximity |
| key edge candidate | contact-gated directed relative edge |
| validation | construct, predictive, mechanistic, transfer validity |

Immediate next step:

> Convert the current feature/channel manifest into a two-name schema:
> `pubg_feature_name` plus `genre_primitive`.

## Relationship To Other Modeling Views

Do not claim graph modeling is the only valid view. Position it as a relational
structure for survival learning.

| View | Strength | Limitation | How DSACG uses it |
| --- | --- | --- | --- |
| multi-agent survival process | natural hazard/rank framing | weak relational structure if all relations are features | DSACG adds relational survival dependencies |
| stochastic/Markov game | models decisions and policy | actions/rewards are hard to observe in telemetry | DSACG models observable relational state |
| habitat compression | explains shrinking-zone pressure | does not cover combat/resource/skill alone | DSACG represents habitat as zone/space/team relations |
| event graph / point process | captures fine combat event timing | must add spatial/zone structure separately | DSACG can add event nodes or projected event edges |
| game-mechanic ontology | gives transferable vocabulary | not itself a predictive model | DSACG converts ontology into graph schema |

## Title Candidates

Preferred:

```text
Dynamic Shrinking-Arena Competitive Graphs for Battle Royale Survival Prediction
```

Alternatives:

- Game-Mechanic-Aligned Graph Modeling of Battle Royale Matches for Survival Prediction
- Toward a Genre-Level Graph Formalism for Battle Royale Survival Modeling
- What Makes a Battle Royale Match a Graph? A Dynamic Relational Formalism for Survival Modeling

## One-Sentence Thesis

```text
Battle Royale is not merely a combat game; it is a dynamic survival system in
which agents, spaces, resources, zones, combat, and elimination are relationally
coupled inside a shrinking arena, and this system can be modeled as a dynamic
multi-relational graph.
```
