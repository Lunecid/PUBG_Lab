# Battle Royale Problem-Method-Experiment Structure

Date: 2026-07-05
Class: `ClaimBoundary`
Lifecycle: `active`

## Purpose

This document fixes the paper-writing rule for the Battle Royale manuscript.

The paper must not be organized as an inventory of experiments.  It must be
organized as:

```text
problem formalization
-> methodology required by that formalization
-> experiments that validate or bound the methodology
-> claim boundary
```

Every experiment is included only after the problem and methodological need
that make the experiment necessary have been stated.

## Global Paper Logic

| Layer | Manuscript job | What belongs here | What does not belong here |
| --- | --- | --- | --- |
| Problem formalization | Explain what problem exists and why the existing view is insufficient | Battle Royale as phase-conditioned multi-agent survival; final outcome as accumulated phase survival; prior tabular/phase-aware prediction lacks a realtime team-level risk/advantage board | raw experiment chronology |
| Methodology | Define the representation needed to solve the problem | O0-O8 ontology, phase-cut survival formalism, PUBG official telemetry observability map, 10-second snapshot survival model, risk/advantage board design | result numbers without a formal reason |
| Experiments / validation | Test whether the methodology is usable and what it supports | mini corpus coverage, feature availability audit, prior/tabular baseline, same-condition temporal comparison, dynamic probability trajectory, phase-transition board, OM mechanism robustness | new claims not tied to the method |
| Claim boundary | State what can and cannot be concluded | supported / bounded / pending / not-claimed language | universal proof language |

## RQ-Level Structure

| RQ | Problem formalization | Methodology required | Experiments / validation | Allowed result claim |
| --- | --- | --- | --- | --- |
| `RQ1` Genre establishment | Battle Royale should be treated as a genre-level survival system, not just a set of PUBG features or final-rank labels. | Formalize Battle Royale as phase-conditioned multi-agent survival with reusable mechanism families `O0-O8` and phase survival variables. | ontology lock, survival formalization, mini corpus coverage | O0-O8 is supported as a framework vocabulary across representative Battle Royale cases, not as an exhaustive universal ontology. |
| `RQ2` PUBG representation | A genre formalization is not useful unless it can be represented in an observed representative game system. | Map formal objects to official PUBG telemetry-derived, leakage-guarded team x phase-cut and 10-second snapshot representations. Separate direct telemetry, derived pre-cut features, proxies, labels, and missing constructs. | PUBG official telemetry observability table, operational manifest, feature availability audit, required cohort registry, OM0-OM6 readiness, snapshot data contract | PUBG is a label-rich supervised instantiation of the formalization. |
| `RQ3` Realtime survival prediction | Prior PUBG work motivates phase-aware and tabular prediction, but does not provide a team-level realtime risk process from 10-second telemetry. | Compare against Sagong-style/basic tabular baselines, evaluate same-snapshot-surface tabular versus causal temporal models, and derive dynamic probability trajectories from phase hazards. | Sagong-style/basic tabular baseline result, same-condition hierarchical survival model comparison, dynamic probability fluctuation result, 10-second prefix demo | Phase-aware prediction can be extended into a causal realtime survival readout; sequence models support dynamic readouts but do not generally dominate strong tabular baselines. |
| `RQ4` Risk / advantage board | A realtime survival model should produce interpretable phase-transition decisions, not only row-level AUC or final-rank scores. | Convert elimination hazard to a risk board and match-local survival share to an advantage board. | realtime risk-advantage framework, phase-transition risk/advantage board result, final top-k/rank bridge | The model can identify most-at-risk and most-advantaged teams at phase transitions, pending explicit board-baseline comparison. |

## Experiment Placement Rule

An experiment can enter the main manuscript only if it answers one of these
questions:

1. Which problem does this experiment address?
2. Which methodological component does it validate or bound?
3. What claim becomes safer after the result?
4. What claim remains unsafe?

If an experiment cannot answer all four questions, it belongs in supplementary
or a future-work note.

## Current Experiment Placement

| Experiment / artifact | Problem it answers | Methodological role | Manuscript placement |
| --- | --- | --- | --- |
| ontology lock | Battle Royale needs a genre-level vocabulary | freezes `O0-O8` mechanism families | RQ1 method |
| mini corpus coverage | `O0-O8` might be PUBG-only | checks representative non-PUBG coverage | RQ1 validation |
| survival formalization | final rank alone misses phase dynamics | defines hazard and cumulative survival variables | RQ1/RQ4 method |
| PUBG official telemetry observability table | formal terms need measurable PUBG evidence | separates observed, derived, proxy, label, and missing fields | RQ2 method table |
| feature availability audit | the mapping may not be executable | confirms current feature and label surfaces | RQ2 validation |
| OM0-OM6 ladder | feature blocks need predictive validation | tests staged representation families | RQ2/RQ4 validation |
| Sagong-style/basic tabular baseline | RQ3 needs prior-work positioning | compares prior-work strategy scores and official raw telemetry baselines on the same team-phase risk set | RQ3 validation before realtime model result |
| recurrent survival model | phase survival should be modeled as accumulated observed state and phase-cut hazard recurrence | compares same-snapshot-surface HGB reductions, last-only neural, causal temporal, and causal recurrent no-graph model forms | RQ3 model-form validation complete; sequence beats last-only in 12/13 cohorts but does not generally beat summary HGB |
| dynamic probability trajectory | realtime prediction needs a changing team-level probability readout | converts phase hazards into cumulative survival and match-local survival share; includes one-match 10-second prefix demo | RQ3 realtime readout validation |
| phase-transition risk/advantage board | realtime prediction should identify actionable match states | ranks teams by next-phase hazard and survival share at phase cuts | RQ4 board-level validation |
| graph-design early readout | graph architecture might distract from the representation claim | bounded audit showed observed graph did not beat no-graph hazard AUC and shuffle controls were comparable | excluded from core / optional supplement |
| `OM2 -> OM3` hazard result | pressure-habitat may not matter | tests contraction and habitat after movement | RQ4 headline result |
| phase-stratified analysis | OM3 could be only pooled phase base rate | checks within-cut-phase deltas | RQ4 robustness |
| paired bootstrap | cohort metrics need uncertainty control | tests paired match-cluster deltas | RQ4 robustness |
| hazard-to-rank/top-k bridge | hazard and advantage boards should relate to final outcome | evaluates phase survival scores against final outcome labels | RQ4 final-outcome bridge |
| corrected O6Z | PUBG `pos_z` may validate terrain affordance | bounds vertical-proxy-only terrain claims | supplementary observability boundary |
| PUBG-Warzone observability bridge | PUBG lacks geometry but the genre may need it in future | maps PUBG gaps to Warzone / Caldera geometry representation | supplement / future O6 extension |
| Caldera construct artifact | geometry bridge needs visible construct evidence | shows terrain/building/route construct representation | supplement / future O6 extension |
| O7 advantage-edge | relation topology may carry signal | supplementary null-control for one edge stream | supplement / reviewer defense |

## Manuscript Section Pattern

Each main section should follow the same internal order:

```text
Problem: what is missing or underspecified?
Method: what representation or formal object solves it?
Experiment: how did we test whether that representation is usable?
Result: what is supported?
Boundary: what must not be claimed?
```

This means the Results section should not start from a run list.  It should
start from the realtime prediction problem: prior/tabular baselines, causal
10-second survival readout, phase-transition risk/advantage board, and then
mechanism/robustness checks such as OM2->OM3, phase stratification, bootstrap,
and rank/top-k bridges.

## One-Sentence Rule

```text
We do not run experiments to find a story; we state the Battle Royale problem,
define the methodology required by that problem, and then use experiments to
test which parts of the methodology are supported, bounded, or still pending.
```
