# Battle Royale RQ3 Realtime Survival Readout Object

Date: 2026-07-06
Package: research_framing
Lifecycle: active
Responsibility: Define the RQ3 claim, evidence objects, experiments, results, and boundaries.
Depends on: docs/experiment_design/BATTLE_ROYALE_RQ4_DYNAMIC_PROBABILITY_FLUCTUATION_RESULT_2026-07-06.md
Used by: docs/research_framing/BATTLE_ROYALE_RQ_OBJECT_CHAIN_2026-07-06.md

## RQObject

Object id:

```text
RQ3RealtimeSurvivalReadout
```

Research question:

```text
How can prior tabular and phase-aware PUBG prediction be extended into a
realtime elimination-risk model using 10-second telemetry?
```

Role in the paper:

```text
Bridge prior phase-aware prediction to a causal team-level survival readout
that updates hazard, cumulative survival, and match-local survival share.
```

## Main Claim

```text
Prior phase-aware PUBG prediction can be extended into a causal telemetry
survival readout that updates team-level elimination risk and match-local
survival share over time.
```

Manuscript-safe wording:

```text
Compared with prior-style and basic tabular baselines, the proposed
representation supports a phase-wise survival readout that can be accumulated
into changing match-local survival probabilities from 10-second telemetry.
```

## Claim Objects

| Claim object | Claim | Evidence object | Result | Status |
| --- | --- | --- | --- | --- |
| `RQ3.C1.PriorPositioning` | The thesis must compare against prior-style and basic tabular prediction. | `PriorBaselineResult` | representation-rich pressure/habitat features beat Sagong-style and basic official baselines. | supported |
| `RQ3.C2.GraphNotCore` | Graph architecture is not the core contribution. | `PriorBaselineResult`, `GraphDesignEarlyReadout` | observed graph did not beat no-graph in the bounded early readout. | not retained as core |
| `RQ3.C3.SameConditionModelForm` | 10-second state histories can be modeled causally and compared on the same snapshot surface. | `HierSurvivalProtocol`, `SameConditionSequenceResult` | sequence accumulation beats neural last-only in most cohorts, but snapshot summary HGB remains strongest same-condition baseline. | supported with boundary |
| `RQ3.C4.DynamicProbabilityReadout` | The model can output a changing probability trajectory rather than only a static score. | `DynamicProbabilityResult` | LSTM phase hazards produce dynamic survival-share trajectories across 13 cohorts. | supported with phase-cut boundary |
| `RQ3.C5.PrefixDemo` | A 10-second prefix readout is technically demonstrable. | `RealtimePrefixDemo` | one Erangel TPP squad test-match demo exists. | descriptive |

## Experiment Objects

| Experiment object | Question | Method | Evidence produced |
| --- | --- | --- | --- |
| `RQ3.E1.PriorBaselineComparison` | How strong are prior-style and basic tabular baselines? | Train/evaluate Sagong-style, official raw, conventional, and representation-rich tabular arms across required cohorts. | `BATTLE_ROYALE_RQ4_PRIOR_BASELINE_RESULT_2026-07-05.md` |
| `RQ3.E2.SameConditionSequenceComparison` | Do temporal sequence models add value on the same official snapshot surface? | Compare snapshot HGB reductions, neural last-only, TCN, GRU, and LSTM/no-graph sequence arms. | `BATTLE_ROYALE_RQ4_SAME_CONDITION_SEQUENCE_RESULT_2026-07-05.md` |
| `RQ3.E3.DynamicProbabilityTrajectory` | Can phase hazards become a dynamic survival-share trajectory? | Convert `lambda -> S -> pi` using saved phase-hazard predictions. | `BATTLE_ROYALE_RQ4_DYNAMIC_PROBABILITY_FLUCTUATION_RESULT_2026-07-06.md` |
| `RQ3.E4.PrefixReadoutDemo` | Can the same readout be shown at 10-second prefix resolution? | Build one-match descriptive prefix probability trajectory. | `RQ4_REALTIME_PREFIX_PROBABILITY_DEMO_2026-07-06.md` |

## Evidence Objects

| Evidence object | Path | What it supports |
| --- | --- | --- |
| `PriorBaselineResult` | `docs/experiment_design/BATTLE_ROYALE_RQ4_PRIOR_BASELINE_RESULT_2026-07-05.md` | prior-work and basic tabular positioning |
| `HierSurvivalProtocol` | `docs/experiment_design/BATTLE_ROYALE_RQ4_HIERARCHICAL_SURVIVAL_PROTOCOL_2026-07-05.md` | same-condition survival model protocol |
| `SameConditionSequenceResult` | `docs/experiment_design/BATTLE_ROYALE_RQ4_SAME_CONDITION_SEQUENCE_RESULT_2026-07-05.md` | temporal model-form result and boundary |
| `DynamicProbabilityResult` | `docs/experiment_design/BATTLE_ROYALE_RQ4_DYNAMIC_PROBABILITY_FLUCTUATION_RESULT_2026-07-06.md` | phase-cut dynamic probability trajectories |
| `RealtimePrefixDemo` | `reports/rq4_dynamic_probability_fluctuation/RQ4_REALTIME_PREFIX_PROBABILITY_DEMO_2026-07-06.md` | one-match 10-second prefix demonstration |
| `RiskAdvantageFramework` | `docs/model_design/BATTLE_ROYALE_REALTIME_RISK_ADVANTAGE_FRAMEWORK_2026-07-06.md` | `lambda`, `S`, and `pi` readout contract |

## Result Objects

### RQ3.R1 Prior Baseline Positioning

Representative result:

```text
Representation-rich habitat/full features beat prior-style and basic official
telemetry baselines across the required cohorts.
```

Manuscript interpretation:

```text
The thesis is not merely reproducing prior phase-score prediction; it uses the
Battle Royale representation to improve the phase-risk surface before realtime
readout.
```

### RQ3.R2 Same-Condition Model-Form Boundary

Supported:

```text
Sequence accumulation helps relative to neural last-only baselines.
```

Bounded:

```text
Deep sequence models do not generally dominate strong tabular snapshot-summary
baselines.
```

### RQ3.R3 Dynamic Probability Trajectory

Core readout:

```text
lambda_i(k) = predicted next-phase hazard
S_i(k)      = product_{j <= k} (1 - lambda_i(j))
pi_i(k)     = S_i(k) / sum_j S_j(k)
```

Coverage:

| Field | Value |
| --- | ---: |
| cohorts | 13 |
| matches | 1,789 |
| teams | 33,584 |
| trajectory rows | 105,491 |
| median latest top-5 AUC from `pi` | 0.9290 |
| median latest rank Spearman from `pi` | 0.7914 |

## Boundary Object

Allowed:

```text
The current model produces phase-cut dynamic survival trajectories and a
descriptive 10-second prefix readout.
```

Forbidden:

```text
Deep sequence models are generally superior to all tabular baselines.
```

Forbidden:

```text
The one-match 10-second prefix demo is a validated realtime deployment metric.
```

Forbidden:

```text
The cumulative survival readout is a fully calibrated betting-style win
probability.
```

## Interface To RQ4

RQ3 exports the following readout objects to RQ4:

```text
lambda_i(k): next-phase risk
S_i(k): cumulative survival score
pi_i(k): match-local survival share
dynamic probability trajectory rows
phase-cut and prefix readout boundary
```

RQ4 must convert these readouts into board-level decision objects.
