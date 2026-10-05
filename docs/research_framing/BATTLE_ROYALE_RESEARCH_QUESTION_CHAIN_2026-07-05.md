# Battle Royale Research Question Chain

Date: 2026-07-05
Class: `ClaimBoundary`
Lifecycle: `active`

## Purpose

This document fixes the manuscript spine after the current experiment cycle.

The paper should not be organized around every completed auxiliary experiment.
It should be organized around four connected research questions.

The writing structure is:

```text
problem formalization
-> methodology required by that formalization
-> experiments / validation
-> claim boundary
```

The detailed paper-writing rule is fixed in:

```text
docs/research_framing/BATTLE_ROYALE_PROBLEM_METHOD_EXPERIMENT_STRUCTURE_2026-07-05.md
```

The final manuscript flow is fixed in:

```text
docs/research_framing/BATTLE_ROYALE_MANUSCRIPT_CANONICAL_FLOW_2026-07-05.md
```

The compressed order is:

```text
Battle Royale genre formalization
-> PUBG official-telemetry representation and mapping to the formalization
-> prior phase/tabular prediction positioning and 10-second telemetry realtime
   elimination-risk model development
-> phase-transition most-at-risk / most-advantaged team prediction
```

## Research Question Spine

| RQ | Question | Role in paper | Primary evidence |
| --- | --- | --- | --- |
| `RQ1` | How can the Battle Royale genre be formalized? | Define Battle Royale as phase-conditioned multi-agent survival with reusable mechanism families | O0-O8 lock, mini corpus coverage, phase-conditioned survival formalization |
| `RQ2` | How can that formalization be represented in the representative Battle Royale game PUBG? | Map formal objects to leakage-guarded PUBG official-telemetry team x phase-cut and 10-second snapshot representations | PUBG official telemetry observability table, operational manifest, feature availability audit, required cohort set, OM0-OM6 ladder |
| `RQ3` | How can prior tabular and phase-wise PUBG prediction be extended into a realtime elimination-risk model using 10-second telemetry? | Position against Sagong-style / basic tabular baselines, then develop a causal snapshot-sequence model that updates elimination risk over time | prior baseline result with legacy `RQ4` filename, same-condition snapshot/tabular vs temporal model comparison, dynamic probability fluctuation result, 10-second prefix readout demo |
| `RQ4` | Can the realtime survival model identify the most unfavorable and most favorable teams at phase transitions? | Convert elimination hazard and cumulative survival into risk and advantage boards | realtime risk-advantage framework, phase-transition risk/advantage board result, final top-k/rank bridge |

## Problem-Method-Experiment Spine

| RQ | Problem | Methodology | Experiment / validation |
| --- | --- | --- | --- |
| `RQ1` | Battle Royale needs a genre-level survival formulation rather than a PUBG-only feature list. | O0-O8 mechanism families and phase survival formalization. | ontology lock, mini corpus coverage |
| `RQ2` | The formalization must be represented in a real representative game system. | PUBG official telemetry observability table, leakage-guarded team x phase-cut features, 10-second snapshot tensors, cohort contract. | feature availability audit, OM0-OM6 readiness, snapshot data contract |
| `RQ3` | Prior work motivates phase-aware prediction but remains largely score/tabular or phase-cut oriented. | Same-risk-set prior/tabular comparison, causal 10-second snapshot sequence model, phase-cut and prefix dynamic probability readouts. | Sagong-style/basic tabular baseline, same-condition model comparison, LSTM dynamic probability trajectory, 10-second prefix demo |
| `RQ4` | A realtime survival model should produce interpretable match-state decisions, not only row-level AUC. | Risk board from elimination hazard and advantage board from match-local survival share at phase transitions. | phase-transition risk/advantage board metrics, final top-k/rank bridge |

## Narrative Order

The manuscript should move in this order:

1. Formalize the Battle Royale genre as phase-conditioned multi-agent survival.
2. Represent that formalization in PUBG official telemetry and show the mapping
   from formal terms to executable team-phase and 10-second snapshot features.
3. Position the prediction task against prior tabular and phase-aware PUBG
   work, then develop a causal 10-second telemetry model that updates
   elimination risk as the match unfolds.
4. Use the realtime survival readout to identify, at phase transitions, the
   teams most likely to be eliminated next and the teams currently holding the
   strongest survival advantage.

This order is the paper's main argument.  Auxiliary experiments should be
placed only where they defend or qualify one of these steps.  No experiment
should be presented before the problem and methodology that make it necessary.

## Core Claims

### RQ1 Claim

```text
Battle Royale can be formalized as a phase-conditioned multi-agent survival
system with reusable mechanism families O0-O8.
```

Allowed evidence:

- O0-O8 ontology lock;
- 15-game mini corpus coverage;
- survival formalization.

Not allowed:

```text
We prove a universal exhaustive Battle Royale ontology.
```

### RQ2 Claim

```text
PUBG official telemetry can instantiate the Battle Royale formalization as
leakage-guarded team x phase-cut feature blocks across exact map /
perspective / team-size cohorts.
```

Allowed evidence:

- PUBG official telemetry observability table;
- feature availability audit;
- required cohort registry;
- formal object to PUBG feature-block mapping;
- OM0-OM6 ladder readiness and execution.

### RQ3 Claim

```text
Prior PUBG phase-aware and tabular prediction work can be extended into a
causal 10-second telemetry survival model that updates team-level elimination
risk and match-local survival share as the match progresses.
```

Allowed evidence:

- Sagong-style / basic tabular baseline comparison;
- same-condition snapshot tabular versus causal temporal model comparison;
- phase-cut dynamic probability trajectory;
- 10-second prefix readout demo.

Not allowed:

```text
Deep sequence models generally beat all tabular baselines.
```

```text
The current 10-second prefix demo is already a separately validated realtime
deployment metric.
```

### RQ4 Claim

```text
The realtime survival model can be converted into phase-transition risk and
advantage boards that identify the most unfavorable team states and the most
favorable team states in a PUBG match.
```

Primary evidence:

- phase-transition risk/advantage board metrics over 10,616 phase boards;
- risk top-1 / top-3 hit rates for next-phase elimination;
- advantage-board relation to final top-5 / top-10 outcomes;
- final rank/top-k bridge as secondary outcome evidence.

The central board outputs are:

```text
RiskBoard_m(k) = sort_i(lambda_i(k), descending=True)
AdvantageBoard_m(k) = sort_i(pi_i(k), descending=True)
```

Do not claim:

```text
The board result is final until it is compared against Sagong-style, official
raw, health/alive-count, zone-distance, and pressure-habitat board baselines.
```

## Supplementary Boundary: Warzone / Caldera

Warzone / Caldera is no longer a core RQ in the revised manuscript spine.

Correct placement:

```text
Supplementary observability and future-work boundary for geometry-rich terrain
constructs not directly provided by PUBG official telemetry.
```

Its role:

- explains why PUBG official telemetry is label-rich but geometry-limited;
- supports future O6 terrain/topology extensions;
- remains outside the supervised realtime risk/advantage claim.

## Auxiliary Experiments

O7 advantage-edge is auxiliary.

Correct placement:

```text
Supplementary PUBG relation-topology null-control.
```

Its role:

- defend that a specific relation-edge stream can carry signal when tested
  against an edge-shuffled null;
- prevent the false conclusion that all relation topology is useless in PUBG;
- provide reviewer-defense material if O7 / graph topology is challenged.

Its non-role:

- it is not the main empirical answer to RQ4;
- it is not the realtime prediction evidence for RQ3;
- it does not validate all O7 graph structures;
- it does not replace the central pressure-habitat result.

Safe wording:

```text
As a supplementary null-control, one PUBG relation stream, contact-gated
relative-advantage edges, shows topology-specific signal over an edge-shuffled
null.
```

Avoid:

```text
O7 is a core mechanism result of the paper.
```

## Paper Section Mapping

| Section | Main job | Evidence |
| --- | --- | --- |
| Introduction / Problem | Motivate Battle Royale as dynamic phase survival rather than post-game rank prediction | RQ1 problem definition and survival formalization |
| Genre Formalization | Define O0-O8, phase survival, and cumulative final-outcome logic | RQ1 ontology lock, mini corpus, survival formula |
| PUBG Official-Telemetry Mapping | Show exact formal-term to PUBG team-phase and 10-second snapshot feature mapping | RQ2 official telemetry observability table, operational manifest, feature audit, cohort contract |
| Realtime Survival Model | Position against Sagong-style/tabular prior work and develop the 10-second telemetry survival readout | RQ3 prior baseline, same-condition model comparison, dynamic trajectory, prefix demo |
| Risk / Advantage Board | Identify most-at-risk and most-advantaged teams at phase transitions | RQ4 realtime risk-advantage framework, phase-transition board metrics |
| Supplement / Robustness | Defend boundaries and side mechanisms | O1 integrity, O7 edge-shuffle, gen-1/gen-2 boundary, Warzone/Caldera observability note |

## Current Next Step

The next core-path task is not another O7 or Warzone / Caldera experiment.

```text
Assemble manuscript figures and prose around the revised RQ1-RQ4 chain, then
compare the risk/advantage board against simple and prior-work board baselines.
```

The execution plan for that step is:

```text
docs/paper_writing/BATTLE_ROYALE_MANUSCRIPT_ASSEMBLY_PLAN_2026-07-05.md
```
