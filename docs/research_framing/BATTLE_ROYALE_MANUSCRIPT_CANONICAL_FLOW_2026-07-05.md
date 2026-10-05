# Battle Royale Manuscript Canonical Flow

Date: 2026-07-05
Class: `ClaimBoundary`
Lifecycle: `active`

## Purpose

This document fixes the manuscript's final argument form.

The paper should not be organized as a chronology of completed experiments.
It should be organized as problem formalization, methodology required by that
formalization, and experiments/results that validate or bound the methodology.

The paper-writing rule is fixed in:

```text
docs/research_framing/BATTLE_ROYALE_PROBLEM_METHOD_EXPERIMENT_STRUCTURE_2026-07-05.md
```

The four-step argument is:

```text
Battle Royale genre formalization
-> exact PUBG official-telemetry representation and mapping
-> prior tabular/phase-aware prediction positioning and 10-second realtime
   elimination-risk model
-> phase-transition risk/advantage board for most unfavorable and most
   favorable teams
```

## Canonical Claim Chain

| Step | Manuscript question | Required answer | Primary evidence |
| --- | --- | --- | --- |
| 1 | How can Battle Royale be formalized as a genre? | Battle Royale is a phase-conditioned multi-agent survival system with reusable mechanism families O0-O8. | ontology lock, mini corpus coverage, survival formalization |
| 2 | How is that formalization represented in the representative Battle Royale game PUBG? | PUBG official telemetry is transformed into leakage-guarded team x phase-cut variables and 10-second snapshot histories mapped to the formal families. | PUBG official telemetry observability table, operational manifest, feature availability audit, required cohort registry, OM0-OM6 feature blocks, snapshot tensor contract |
| 3 | How can prior tabular and phase-aware prediction be extended into realtime elimination-risk prediction? | Sagong-style and basic tabular baselines are used as prior-work positioning, then 10-second PUBG telemetry is modeled with causal temporal survival readouts that update hazard, cumulative survival, and match-local survival share. | prior baseline result with legacy `RQ4` filename, same-condition hierarchical survival protocol/result, dynamic probability fluctuation result, 10-second prefix readout demo |
| 4 | Can the realtime model identify the most unfavorable and most favorable teams at phase transitions? | The model output becomes a risk board and advantage board: highest hazard teams are most at risk, and highest match-local survival-share teams are most advantaged. | realtime risk-advantage framework, phase-transition risk/advantage board result, hazard-to-rank/top-k bridge |

## Problem-Method-Experiment Rule

Every main paper unit must be written in this order:

```text
problem
-> method / representation
-> experiment or validation
-> result
-> claim boundary
```

Experiments do not define the paper order.  They are evidence for a prior
problem and method.  Any experiment that does not clearly validate or bound one
of the four RQs belongs in supplementary or future work.

## Section Contract

### 1. Genre Formalization

This section must start from the genre, not from PUBG features.

Core statement:

```text
Battle Royale can be formalized as phase-conditioned multi-agent survival:
teams repeatedly face changing contraction pressure, interaction exposure,
resource state, spatial affordance, and relation structure until final
survival / win condition.
```

Minimum formal objects:

| Object | Meaning |
| --- | --- |
| `(m, i, k)` | match, team, phase cut |
| `lambda_i(k)` | next-phase elimination hazard |
| `S_i(k)` | cumulative survival probability through phase `k` |
| `O0-O8` | mechanism families used to name the information set |

Do not claim:

```text
We prove a universal exhaustive ontology of all Battle Royale games.
```

### 2. PUBG Official-Telemetry Representation

This section must show that the formalization is not just prose. It is mapped
onto PUBG official telemetry.

Source of truth for the observability surface:

```text
docs/data_layer/PUBG_OFFICIAL_TELEMETRY_OBSERVABILITY_TABLE_2026-07-05.md
```

The required mapping surface is:

| Formal term | Ontology | PUBG official-telemetry representation | OM location |
| --- | --- | --- | --- |
| `U_m(k)` | O0 | map, mode, team size, phase index, elapsed time | `OM0` |
| `C_i(k)` | O1 | alive roster, health / damage-derived survival capital, team state | `OM1` |
| `Z_m(k)` | O2 | safe-zone / blue-zone pressure schedule and alive-population pressure | `OM3` via habitat |
| `B_i(k)` | O3 | movement / rotation response, speed, zone-relative movement | `OM2` |
| `H_i(k)` | O2 x O6 | pressure-conditioned survival habitat, zone gap, crowding / pressure interactions | `OM3` |
| `E_i(k)` | O4 | threat / combat / exposure summaries | `OM4` |
| `R_i(k)` | O5 | resources, vehicles, mobility/support proxies where available | within operational feature blocks |
| `Q_i(k)` | O7 | relation summaries or edge-stream evidence, with O7 edge stream treated as supplementary | `OM5`, supplementary O7 |
| `Y_i(k+1)` | O8 | next-phase hazard, final rank, top-k / win labels | prediction targets |

This section should emphasize:

```text
PUBG is the label-rich supervised case because it has official telemetry,
phase/outcome labels, and enough cohort coverage for controlled prediction.
```

Do not claim:

```text
Every formal construct is fully observed in PUBG.
```

### 3. Prior Baselines And Realtime Survival Model

This section must start from prior prediction work, not from model novelty for
its own sake.

Core statement:

```text
Prior PUBG work motivates phase-aware and tabular prediction, but a Battle
Royale survival formulation requires a model that updates team-level
elimination risk as the match unfolds.  PUBG's 10-second official telemetry
snapshots allow this realtime survival readout.
```

Required bridge:

```text
Sagong-style phase / score baseline
-> basic official telemetry tabular baseline
-> same-snapshot-surface tabular and temporal model comparison
-> 10-second causal telemetry survival readout
-> phase-cut and prefix dynamic probability trajectories
```

Current supported rows:

| Layer | Evidence | Current status |
| --- | --- | --- |
| prior-work score baseline | Sagong-style team-phase adapted scores | completed; weaker than representation-rich features |
| basic official telemetry baseline | official raw and conventional tabular features | completed |
| strong representation baseline | pressure-habitat HGB/XGB family | completed; strongest hazard-AUC family |
| temporal model form | LSTM/GRU/TCN same-condition comparison | completed; sequence helps over neural last-only, not over strongest HGB |
| dynamic readout | phase-cut `lambda -> S -> pi` trajectory | completed across 13 cohorts |
| realtime demo | 10-second prefix probability readout | descriptive demo complete; not separately validated streaming metric |

Source of truth:

```text
docs/experiment_design/BATTLE_ROYALE_RQ4_PRIOR_BASELINE_RESULT_2026-07-05.md
docs/experiment_design/BATTLE_ROYALE_RQ4_DYNAMIC_PROBABILITY_FLUCTUATION_RESULT_2026-07-06.md
docs/model_design/BATTLE_ROYALE_REALTIME_RISK_ADVANTAGE_FRAMEWORK_2026-07-06.md
```

Do not claim:

```text
Deep sequence models generally dominate tabular baselines.
```

Do not claim:

```text
The 10-second prefix demo is a separately validated deployment metric.
```

### 4. Phase-Transition Risk And Advantage Board

This section turns the realtime survival model into the final interpretive
framework.

Core statement:

```text
The realtime survival model should not only output row-level hazard scores.  At
each phase transition, it should identify the teams most likely to be
eliminated next and the teams currently holding the strongest survival
advantage.
```

The required prediction chain is:

```text
next-phase hazard lambda_i(k)
-> cumulative survival S_i(k+1) = S_i(k) * (1 - lambda_i(k))
-> match-local survival share pi_i(k)
-> phase-transition risk/advantage board
-> final-rank / top-k bridge
```

The required board definitions are:

```text
RiskBoard_m(k) = sort_i(lambda_i(k), descending=True)
AdvantageBoard_m(k) = sort_i(pi_i(k), descending=True)
```

Primary empirical claim:

```text
The dynamic survival readout can be converted into phase-transition boards that
identify high next-phase elimination risk and strong match-local survival
advantage.
```

Do not collapse the paper into:

```text
We predict final rank from a bag of match-level statistics.
```

Do not claim:

```text
The board is final before comparison against simple and prior-work board
baselines.
```

## Correct Artifact Placement

| Artifact family | Paper role |
| --- | --- |
| ontology lock + mini corpus | Step 1: genre formalization |
| survival formalization | Step 1, Step 3, and Step 4: phase survival equations |
| PUBG official telemetry observability table | Step 2: direct / derived / proxy / label / missing telemetry boundary |
| PUBG operational manifest / feature audit | Step 2: official-telemetry mapping |
| OM ladder and feature-block audit | Step 2 and Step 3: executable mapping and hazard mechanism |
| prior baseline result, legacy `RQ4` filename | Step 3: comparison against Sagong-style/basic tabular alternatives |
| hierarchical survival protocol, legacy `RQ4` filename | Step 3: same-snapshot-surface tabular/DL comparison and temporal model-form boundary |
| dynamic probability fluctuation result, legacy `RQ4` filename | Step 3: phase hazard to cumulative survival and match-local probability trajectory readout |
| realtime risk-advantage framework | Step 4: model contract for most-at-risk and most-advantaged team identification |
| RQ4 phase-transition risk/advantage board result | Step 4: board-level evaluation at phase cuts |
| phase-stratified / bootstrap / rank bridge | Step 3 and Step 4: survival mechanism and final-outcome bridge |
| O6Z / Warzone / Caldera artifacts | supplementary observability boundary and future O6 terrain extension |
| O7 advantage-edge | supplementary reviewer defense, not core spine |

## Manuscript Skeleton

Use this ordering:

1. Introduction / problem: Battle Royale outcomes are dynamic phase survival, not merely post-game rank labels.
2. Problem formalization: define the genre as phase-conditioned multi-agent survival with O0-O8.
3. Methodology 1: map the formalization to PUBG official telemetry through observed, derived, proxy, label, and 10-second snapshot surfaces.
4. Methodology 2: position prior tabular/phase-aware prediction and define the causal 10-second realtime elimination-risk model.
5. Methodology 3: define the phase-transition risk/advantage board.
6. Experiments/results: mini corpus, feature availability, prior/tabular baselines, same-condition model comparison, dynamic probability trajectories, risk/advantage board, OM2->OM3 mechanism and robustness, rank/top-k bridge.
7. Supplement / robustness: O1 integrity, O7 null-control, gen-1/gen-2 boundary, O6Z and Warzone/Caldera observability boundary.

The active assembly plan is:

```text
docs/paper_writing/BATTLE_ROYALE_MANUSCRIPT_ASSEMBLY_PLAN_2026-07-05.md
```

## One-Sentence Version

```text
We formalize Battle Royale as phase-conditioned multi-agent survival, map that
formalization onto PUBG official telemetry, extend prior tabular and
phase-aware PUBG prediction into a causal 10-second realtime elimination-risk
model, and use its survival readouts to identify the most-at-risk and
most-advantaged teams at phase transitions.
```
