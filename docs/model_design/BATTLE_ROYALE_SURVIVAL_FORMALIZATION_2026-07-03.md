# Battle Royale Survival Formalization

Date: 2026-07-03

## Purpose

This document turns the locked Battle Royale ontology into a testable survival
formula.

The ontology defines the concept families.  This document defines the
model-facing formula that the PUBG OM0-OM6 experiments are meant to test.

Related documents:

```text
docs/research_framing/BATTLE_ROYALE_PAPER_ARGUMENT_CHAIN_2026-07-03.md
docs/research_framing/BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md
docs/feature_engineering/BATTLE_ROYALE_FEATURE_ONTOLOGY_2026-07-02.md
docs/feature_engineering/PUBG_FEATURE_OPERATIONAL_MANIFEST_2026-07-02.md
docs/experiment_design/BATTLE_ROYALE_FRAMEWORK_VALIDATION_PLAN_2026-07-02.md
docs/experiment_design/BATTLE_ROYALE_RQ4_DYNAMIC_PROBABILITY_FLUCTUATION_RESULT_2026-07-06.md
docs/model_design/BATTLE_ROYALE_REALTIME_RISK_ADVANTAGE_FRAMEWORK_2026-07-06.md
```

## Unit Of Prediction

The supervised prediction unit is:

```text
(match m, team i, phase cut k)
```

Input window:

```text
history through phase cut k
```

Primary target:

```text
Y_i(k+1) = 1 if team i is eliminated before or during the next phase,
           0 otherwise
```

Primary prediction:

```text
lambda_i(k) = Pr(Y_i(k+1) = 1 | history through k)
```

`lambda_i(k)` is the next-phase survival hazard.  It is the main quantity the
model estimates.

Secondary prediction path:

```text
phase-wise hazard
-> cumulative survival probability
-> match-local dynamic survival share
-> final-rank score / final-rank evaluation
```

The paper should not skip directly from raw result statistics to final rank.
Final rank is connected to the framework through phase-wise survival dynamics.

## State Variables

For each match `m`, phase cut `k`, and team `i`:

| Symbol | Ontology | Meaning | PUBG projection examples |
| --- | --- | --- | --- |
| `U_m(k)` | O0 | rule and match context | map, mode, team size, phase index, elapsed time |
| `C_i(k)` | O1 | survival capital | alive roster, health, team spread, vehicle occupancy |
| `Z_m(k)` | O2 | contraction / elimination pressure schedule | safe radius, poison radius, phase duration, alive population pressure |
| `G_i(k)` | O6 | spatial / environment affordance observed for team i | position, relative height, zone-relative location, geometry-rich affordance when available |
| `H_i(k)` | O2 x O6 | survivable habitat under pressure | safe area pressure, zone gap, required speed, habitat crowding |
| `R_i(k)` | O5 | resources and support | heals, boosts, loot, vehicles, mobility resources |
| `B_i(k)` | O3 | behavioral response | movement, rotation timing, path shape, speed, entry timing |
| `Q_ij(k)` | O7 | relation between team i and team j | proximity edge, combat edge, co-pressure edge, advantage edge |
| `Q_i(k)` | O7 | ego/global relation summary for team i | degree, relation role, bridge score, centrality, graph state |
| `E_i(k)` | O4 | interaction / threat exposure for team i | damage, kills, contacts, nearby combat, scalar ego exposure |
| `Y_i(k+1)` | O8 | next outcome / win-condition label | next-phase hazard, survival, rank, top-k, winner |

## Core Formula

The Battle Royale survival formula is:

```text
lambda_i(k)
  = Pr(Y_i(k+1) = 1 | history through k)
  = F(
      U_m(k),
      C_i(k),
      H_i(k),
      B_i(k),
      E_i(k),
      R_i(k),
      Q_i(k)
    )
```

In words:

```text
next-phase elimination risk
= function of match context, survival capital, pressure-conditioned habitat,
  behavior, interaction exposure, resources, and relation structure.
```

This is not a claim that the true game system is linear or additive.  `F` can be
a logistic model, tree model, neural model, graph model, or other supervised
estimator.  The formula defines the information set and mechanism families,
not the final model architecture.

## Habitat Formula

Habitat is the key Battle Royale environment term:

```text
H_i(k) = h(Z_m(k), G_i(k), A_m(k))
```

Where:

| Symbol | Meaning |
| --- | --- |
| `Z_m(k)` | contraction / elimination pressure schedule |
| `G_i(k)` | spatial or environmental affordance available to team i |
| `A_m(k)` | alive / at-risk population state at phase cut k |

Interpretation:

```text
habitat = pressure schedule applied to the available affordance field
```

Examples:

- the same safe-zone radius has different meaning in open terrain, dense urban
  space, water, mountain, road network, or a 2D sector map;
- the same position has different survival meaning under early, mid, and late
  pressure;
- the same alive count has different meaning when the safe area is large versus
  compressed.

Ablation rule:

```text
H_i(k) belongs to OM3 as an O2 x O6 habitat interaction.
OM6 adds standalone spatial affordance signals and must not duplicate H_i(k).
```

## Exposure Formula

Interaction / threat exposure is:

```text
E_i(k) = e(B_i(k), Q_i(k), event_i(k))
```

Where:

| Symbol | Meaning |
| --- | --- |
| `B_i(k)` | movement / rotation behavior |
| `Q_i(k)` | relation field around team i |
| `event_i(k)` | realized contact, combat, contest, or threat events |

Interpretation:

```text
exposure = behavior through a relation field plus realized threat events
```

Boundary rule:

```text
O4 owns realized hostile events and scalar ego exposure summaries.
O7 owns edges, graph roles, centrality, bridge structure, and global graph
state.
```

Every O4 exposure summary must declare the O7 relation channel, radius/window,
or event family it marginalizes.

## From Phase Hazard To Final Rank

The model's primary object is the next-phase hazard `lambda_i(k)`.  A final-rank
prediction can be derived from the sequence of phase hazards by accumulating
survival probability.

Survival curve:

```text
S_i(0) = 1
S_i(k+1) = S_i(k) * (1 - lambda_i(k))
```

Cumulative elimination risk:

```text
RISK_i(K) = 1 - S_i(K)
```

Match-local dynamic survival share:

```text
pi_i(k) = S_i(k) / sum_j S_j(k)
```

`pi_i(k)` compares teams within the same match and phase cut.  It is a dynamic
survival-share or win-probability proxy, not a fully calibrated betting-style
win probability.

Rank score:

```text
rank_score_i(K) = G(S_i(1:K), lambda_i(0:K-1), state_i(K))
```

`G` may be a simple ordering score, a calibrated rank model, or an auxiliary
rank head.  The important rule is that `state_i(K)` and all hazards used in the
score must be computed from information available through the chosen prediction
cut `K`.

Technical assumptions:

| Assumption | Design rule |
| --- | --- |
| rank is match-local | compare `rank_score_i(K)` among teams within the same match and cut; avoid pooled cross-match rank regression unless match size and phase structure are explicitly controlled |
| phase hazards have limited resolution | phase-level hazards can support phase-band and top-k ranking, but cannot fully recover the order of teams eliminated inside the same phase without finer event-time labels |
| calibration matters | check per-phase calibration before interpreting cumulative survival scores; multiplicative accumulation can amplify per-phase miscalibration |
| hazards are scores, not a simulator | independent Bernoulli hazards do not enforce the exact number of eliminations per phase, so `G` is an evaluation/scoring bridge rather than a full match-generating model |

This gives the manuscript a clear order:

```text
phase-wise survival probability first,
cumulative survival probability second,
dynamic match-local probability trajectory third,
final-rank prediction/evaluation fourth.
```

## State Feedback

The formula is evaluated at phase cut `k`, but the match evolves with feedback.

```text
C_i(k+1) = update_C(C_i(k), E_i(k), R_i(k), Y_i(k+1))
A_m(k+1) = update_alive_set(A_m(k), Y_*(k+1))
Q_ij(k+1) = update_Q(Q_ij(k), B_i(k), B_j(k), A_m(k+1), Z_m(k+1))
H_i(k+1) = h(Z_m(k+1), G_i(k+1), A_m(k+1))
```

Important feedback loops:

| Feedback | Meaning |
| --- | --- |
| `O4 -> O1` | threat and combat deplete survival capital |
| `O8 -> O2/O7` | eliminations change crowding and relation structure |
| `O3_j/O7_ij -> H_i` | other teams can block, camp, third-party, or reshape local habitat |

To avoid leakage, a prediction at cut `k` can use only states observed through
`k`; feedback affects later cuts, not the current input window.

## OM Ablation Mapping

The OM ladder tests this formula by adding terms in interpretable blocks.

| Block | Formula terms first emphasized | Interpretation |
| --- | --- | --- |
| `OM0` | baseline subset of `U_m(k)` and legacy core fields | sanity baseline |
| `OM1` | `C_i(k)` | survival capital matters |
| `OM2` | `B_i(k)` | behavioral response matters |
| `OM3` | `Z_m(k)` and `H_i(k)` | contraction pressure and habitat matter |
| `OM4` | `E_i(k)` | interaction / threat exposure matters |
| `OM5` | `Q_i(k)` / `Q_ij(k)` | relation structure matters |
| `OM6` | standalone `G_i(k)` / spatial-affordance proxies | spatial affordance adds signal beyond habitat |
| `OM7` | `R_i(k)` where available | resource / support sensitivity |

The main paper path is OM0-OM6.  OM7 is useful but can remain sensitivity if
resource mechanisms are incomplete or confounded with combat and roster state.

## What Makes This A Formula

This formalization is valid for experiments only if each condition holds.

| Requirement | Meaning | Status |
| --- | --- | --- |
| defined variables | every term has an ontology owner | satisfied by O0-O8 lock |
| time index | every input is cut at `k`, target is `k+1` | required by manifest clock rules |
| observability mask | missing geometry, labels, or telemetry are explicit | required by manifest and Caldera boundary |
| operational mapping | each term maps to concrete features where available | satisfied for PUBG headline blocks |
| falsifiability | adding/removing terms changes testable prediction claims | tested by OM0-OM6 |

The formula can fail empirically.  For example:

- if OM3 adds no signal, the habitat-pressure claim is weak;
- if OM5 adds no signal after controls, the relation-structure claim is weak;
- if OM6 adds no signal, standalone spatial-affordance proxies are not useful
  in the current PUBG observability regime;
- if Caldera cannot instantiate a term, the claim is masked rather than false.

## Claim Boundary

Allowed claim:

```text
We formalize Battle Royale survival as next-phase hazard under contraction
pressure, survival capital, pressure-conditioned habitat, behavioral response,
interaction exposure, resources, and relation structure.
```

Stronger claim only after experiments:

```text
The terms in this formula improve PUBG next-phase survival prediction in the
OM0-OM6 ablation ladder.
```

Avoid:

```text
The formula proves causal effects.
The formula is the final mathematical theory of all Battle Royale games.
Caldera validates supervised survival prediction without labels.
Independent phase hazards simulate the exact final placement order.
```

## Paper-Level Short Form

For the manuscript, the compact form is:

```text
lambda_i(k) = Pr(Y_i(k+1)=1 | H_k)
            = F(C_i(k), H_i(k), B_i(k), E_i(k), R_i(k), Q_i(k), U_m(k))

H_i(k) = h(Z_m(k), G_i(k), A_m(k))
```

Where `H_k` is history through phase cut `k`.
