# Battle Royale Ontology Validation Protocol

Date: 2026-07-03
Class: `ExperimentContract`
Lifecycle: `active`

## Purpose

This document defines how the project validates the locked Battle Royale
ontology before using it as the basis for predictive experiments.

The goal is not to prove a universal law for every possible Battle Royale game.
The goal is to establish that the ontology is stable enough, operational enough,
and useful enough for the current paper path:

```text
genre formulation
-> ontology families
-> phase-wise PUBG survival prediction
-> cross-cohort robustness
-> Warzone/Caldera construct projection for geometry-rich affordances
```

## Core Decision

The framework should be tested with a staged evidence chain.

```text
ontology validity
= conceptual coverage
+ operational feature availability
+ phase-mechanism sanity
+ supervised predictive utility
+ cross-cohort robustness
+ geometry-rich construct projection
```

Official game descriptions can support conceptual coverage.  They do not, by
themselves, prove predictive usefulness.

PUBG provides the label-rich supervised test bed.  Warzone/Caldera provides
geometry-rich construct evidence, not supervised survival prediction under the
current data boundary.

## Anchor Versus Proof

`Erangel TPP squad` may be used as the first execution anchor because it is the
largest and most comparable cohort to prior PUBG work.

However, `Erangel TPP squad` alone is not the framework proof.  It is only the
first smoke or anchor run.

Framework evidence requires separately trained and separately evaluated runs for
the required map / perspective / team-size cohorts.  Do not pool cohorts for the
main OM0-OM6 claim.

## Validation Ladder

| Stage | Question | Evidence | Success Criterion | Claim Enabled |
| --- | --- | --- | --- | --- |
| OV0. Ontology lock | Are the top-level families stable enough to stop redesigning? | O0-O8 lock, survival formalization, paper argument chain | No family renaming during the OM cycle | The paper has a fixed analytic vocabulary |
| OV1. Corpus coverage | Are O0-O8 not just PUBG-specific vocabulary? | 15 source-rich official/publisher-controlled rows in `research_framing/BATTLE_ROYALE_MINI_CORPUS_COVERAGE_TABLE_2026-07-04.md`; optional negative controls remain appendix/future work | Representative Battle Royale descriptions map to O0-O8; edge cases add projections, not new top-level families | The ontology is a defensible genre framework |
| OV2. PUBG operational availability | Can the ontology be instantiated as legal inputs in PUBG? | Feature availability audit for each required cohort and OM block | OM0-OM6 has ready/partial/missing status by cohort; labels exist; forbidden inputs are absent | The ontology can become executable feature blocks |
| OV3. Phase mechanism sanity | Does contraction/elimination pressure actually change the environment and behavior? | Phase-wise shifts in zone pressure, density, movement, contact, and relation features | Expected phase-conditioned changes appear in the required cohorts or are explicitly bounded | The formulation has empirical mechanism support |
| OV4. Predictive utility | Do ontology-derived blocks help predict phase survival? | OM0-OM6 ablations trained separately for each required cohort | Feature-family deltas are measured by phase and cohort with calibration checks | The ontology has supervised utility in PUBG |
| OV5. Cross-cohort robustness | Does the same formulation survive map, perspective, and team-size changes? | Required cohort comparisons after separate training | Gains or failures are interpretable under map/perspective/team-size differences | The framework is not a single-cohort artifact |
| OV6. Warzone/Caldera construct projection | Can spatial affordance constructs be measured in a geometry-rich Battle Royale setting? | Trajectory-grounded terrain, local mesh, and LOS/visibility artifacts | O6/O7 constructs are computable and visually auditable without survival-label claims | The spatial part of the ontology has a richer construct instantiation |

## Required PUBG Cohorts For OV2-OV5

The supervised validation path uses the required cohort set defined in:

```text
docs/experiment_design/BATTLE_ROYALE_REQUIRED_COHORT_SET_2026-07-03.md
```

Each cohort below is a separate OM0-OM6 training and evaluation job:

| Cohort Description | Role |
| --- | --- |
| Erangel TPP squad | first execution anchor and required map/team/perspective cohort |
| Taego TPP squad | large required map cohort |
| Miramar / Desert TPP squad | required map cohort |
| Rondo TPP squad | required map cohort |
| Sanhok TPP squad | required map cohort |
| Vikendi TPP squad | required map cohort |
| Paramo TPP squad | required map cohort |
| Karakin TPP squad | required map cohort |
| Erangel TPP solo | required team-size contrast |
| Erangel TPP duo | required team-size contrast |
| Taego TPP duo | required team-size contrast |
| Taego FPP squad | required perspective contrast |
| Erangel FPP squad | required perspective contrast |

The cohort descriptions above are the user-facing names.  Internal registry IDs
may be used in scripts, but paper prose and experiment explanations should use
the explicit map / perspective / team-size names.

## How Each Evidence Source Is Used

| Evidence Source | What It Can Establish | What It Cannot Establish |
| --- | --- | --- |
| Official game descriptions and store/manual texts | O0-O8 are recognizable genre mechanisms beyond PUBG | Predictive performance |
| PUBG required cohorts | Label-rich phase survival and rank-related supervised utility | Native 3D terrain affordance at Caldera resolution |
| Other PUBG cohorts and auxiliary cohorts | Robustness and sensitivity outside the required set | Independent cross-game generality by themselves |
| Warzone/Caldera | Geometry-rich measurability of spatial/terrain/relation affordances | Supervised rank or phase survival prediction with the current labels |

## Decision Gate Before Predictive Runs

Before claiming OM0-OM6 results as ontology tests, complete these minimum gates:

1. OV0 is complete: O0-O8 are locked for the current cycle.
2. OV1 has at least a mini coverage table or a documented source-coding plan.
3. OV2 is complete for the required PUBG cohorts.
4. The OM ladder does not use forbidden outcome, future, or post-cut inputs.
5. Each required cohort is trained and evaluated separately.

The full corpus saturation protocol is confirmatory.  It is not allowed to
block PUBG predictive experiments unless it reveals a genuine top-level family
failure.

## Allowed Claims By Stage

| Completed Stage | Allowed Claim |
| --- | --- |
| OV0 | We have a locked analytic vocabulary for the current paper path |
| OV0-OV1 | The vocabulary is not merely a PUBG feature list |
| OV0-OV2 | The vocabulary can be operationalized on the required PUBG cohorts |
| OV0-OV4 | Ontology-derived features have measured supervised utility in PUBG |
| OV0-OV5 | The utility and failure modes have been tested across map, perspective, and team-size cohorts |
| OV0-OV6 | The spatial affordance construct has a geometry-rich Warzone/Caldera instantiation |

## Forbidden Collapses

Do not write or imply:

- "Official game descriptions prove the ontology."
- "One PUBG cohort proves the whole Battle Royale framework."
- "A pooled map/perspective/team-size run is the main cohort result."
- "Warzone/Caldera validates supervised survival prediction."
- "A feature family that fails one cohort should be deleted from the ontology
  before checking observability and proxy quality."

## Immediate Next Step

The next operational step is:

```text
Feature availability audit for OM0-OM6 on required cohorts
```

That audit must answer, for every required cohort and every OM block:

- which ontology family is represented;
- which formula term it supports;
- which feature columns or presets exist now;
- whether the feature is direct, proxy, masked, or unavailable;
- whether the feature obeys the causal clock and leakage rules.
