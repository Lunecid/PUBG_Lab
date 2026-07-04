# Battle Royale Paper Argument Chain

Date: 2026-07-03
Class: `ClaimBoundary`
Lifecycle: `active`

## Purpose

This document fixes the paper-level premise and argument order for the current
Battle Royale framework study.

It does not replace the ontology, feature manifest, model formalization, or
experiment plan.  It states how those objects should be connected in the
manuscript narrative.

The manuscript-facing compression of the first paper question is now owned by:

```text
docs/research_framing/BATTLE_ROYALE_ANALYTIC_FRAMEWORK_NARRATIVE_FOR_MANUSCRIPT_2026-07-04.md
```

Use that document when drafting Introduction, Theory, and Validation prose for
the analytic-framework/formulation claim.

## Core Premise

The paper should be framed as an extension of two prior lines:

1. prior PUBG work showed that outcome-level or result-statistical information
   can be used to predict victory, rank, or performance;
2. the Sagong line of work motivates decomposing PUBG analysis by phase rather
   than treating the match as one undifferentiated block;
3. the direct Sagong-style prior is solo-player and score-based, while our
   target setting is team-level and relation-aware.

Our contribution begins after those premises:

```text
result-level PUBG prediction exists
-> phase-aware PUBG prediction is necessary
-> but phase pressure is not only a PUBG Blue Zone trick
-> contraction / elimination pressure is a genre-level mechanism
-> Battle Royale needs a general formulation beyond PUBG
-> the formulation becomes an ontology of genre mechanisms
-> the ontology is instantiated as team-level, relation-aware phase survival prediction
-> phase hazards accumulate into survival curves
-> cumulative survival scores support final-rank prediction/evaluation
```

Sagong citation metadata and the solo/phase/FightScore/ActiveScore reading were
checked against the local PDF and BibTeX on 2026-07-03.  Other rank-prediction
and aggregate-statistics prior citations still require final bibliography
verification before manuscript submission.

## Paper Argument Order

| Step | Claim | Evidence / Method | Owning Document |
| --- | --- | --- | --- |
| 1 | PUBG victory/rank prediction has prior precedent using result-level or aggregate statistics | Related-work synthesis; citation verification pending | `RELATED_WORK_FRAME_2026-06-22.md` |
| 2 | Phase decomposition is needed because survival meaning changes across the match | Sagong-style phase-aware framing; local PDF confirms solo, phase-specific FightScore/ActiveScore, next-phase survival/win prediction | `DESIGN.md`, `PRIOR_WORK_SURVEY_2026-06-22.md` |
| 3 | The prior-work gap is team-level, relation-aware survival under phase pressure | Sagong excludes duo/squad and uses player-level score summaries; our unit is team in a dynamic relation field | `PUBG_NETWORK_STRUCTURE_FEATURE_SPEC_2026-07-02.md`, `BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md` |
| 4 | The bridge to a general formulation is that contraction / elimination pressure is genre-defining, while PUBG Blue Zone is one projection | O2 rename, official-source mini corpus plan, non-PUBG boundary cases | `BATTLE_ROYALE_ONTOLOGY_CORPUS_DESIGN_2026-07-02.md`, `BATTLE_ROYALE_FEATURE_ONTOLOGY_2026-07-02.md` |
| 5 | We define Battle Royale as a general survival system, not a PUBG-only task | Lightweight ontology vocabulary, claim-boundary lock, and post-OM analytic-framework status | `BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md`, `BATTLE_ROYALE_ANALYTIC_FRAMEWORK_STATUS_2026-07-04.md` |
| 6 | We define the genre by its reusable mechanism families | O0-O8 ontology, source/corpus confirmatory plan, and OM ladder mechanism evidence | `BATTLE_ROYALE_ONTOLOGY_CORPUS_DESIGN_2026-07-02.md`, `BATTLE_ROYALE_FEATURE_ONTOLOGY_2026-07-02.md`, `BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md` |
| 7 | We instantiate that formulation in PUBG because PUBG is label-rich | PUBG operational manifest, feature availability, phase labels | `PUBG_FEATURE_OPERATIONAL_MANIFEST_2026-07-02.md` |
| 8 | We predict team-level, relation-aware phase survival hazard | Phase-cut hazard formula, leakage-safe cut rule, O7 relation features | `BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md`, `PUBG_NETWORK_STRUCTURE_FEATURE_SPEC_2026-07-02.md` |
| 9 | We derive cumulative survival probability from phase hazards | Survival curve accumulation | `BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md` |
| 10 | We use cumulative survival scores and/or an auxiliary head to evaluate final rank | Match-local rank evaluation under PUBG labels | `BATTLE_ROYALE_FRAMEWORK_VALIDATION_PLAN_2026-07-02.md` |
| 11 | We use Warzone/Caldera as ontology and construct evidence, not supervised prediction | Geometry-rich but label-poor observability boundary | `CALDERA_DATASET_STATUS_REDECISION_2026-07-02.md`, `TERRAIN_AFFORDANCE_CAPABILITY_BOUNDARY_2026-07-02.md` |

## Direct Prior-Work Gap

Sagong et al. (2025) is the closest direct prior because it already shows that
PUBG survival prediction should be phase-aware.  The verified local reading
supports three constraints on our positioning:

| Prior-work property | Implication for our paper |
| --- | --- |
| solo-player setting | our unit shift is player to team |
| FightScore / ActiveScore strategy summaries | our representation shift is derived strategy scores to ontology-grounded telemetry blocks |
| next-phase survival and final win prediction | our task shift is not merely "hazard instead of survival", but team-level survival under relation structure |

Therefore, the core differentiator should be stated as:

```text
We extend phase-aware PUBG survival prediction from solo score-based strategy
analysis to team-level, relation-aware survival modeling under a general
Battle Royale ontology.
```

## Formal Narrative

Let `lambda_i(k)` be the predicted probability that team `i` is eliminated in
the next phase after observing history through phase cut `k`.

The phase-wise survival curve is:

```text
S_i(0) = 1
S_i(k+1) = S_i(k) * (1 - lambda_i(k))
```

The cumulative elimination risk by phase `K` is:

```text
RISK_i(K) = 1 - S_i(K)
```

A final-rank predictor can then use:

```text
rank_score_i(K) = G(S_i(1:K), lambda_i(0:K-1), state_i(K))
```

where `state_i(K)` is restricted to information available through the chosen cut
`K`.  This allows the paper to connect next-phase survival prediction to final
rank prediction without collapsing the model back into a single result-level
classifier.

Technical boundary:

| Requirement | Meaning |
| --- | --- |
| match-local ranking | `rank_score_i(K)` is evaluated within the same match and cut, not by pooling teams across unrelated matches |
| phase-resolution limit | phase hazards distinguish phase bands; teams eliminated in the same phase may not be fully orderable by `S_i` alone |
| calibration prerequisite | per-phase calibration must be checked because multiplicative survival accumulation amplifies hazard miscalibration |
| scoring not simulation | independent hazards produce rank scores, not a generative simulator that enforces the exact number of eliminations per phase |

## Warzone / Caldera Role

Warzone/Caldera should not be presented as the supervised survival-prediction
dataset in the current paper path.

Current role:

- supports ontology construction by showing that the O6 spatial / terrain /
  environment family is not a PUBG-only idea;
- demonstrates how geometry-rich Battle Royale telemetry can instantiate richer
  spatial affordance measurements;
- helps define observability masks: a construct can exist even when a dataset
  lacks labels, zone phase records, or final rank outcomes.

Current boundary:

- no rank/survival supervised claim from Caldera;
- no phase-hazard experiment from Caldera without valid outcome labels and
  phase schedule;
- no claim that geometry alone validates the predictive framework.

## Claim Levels

| Claim Level | Allowed Wording |
| --- | --- |
| framework claim | We propose an ontology-guided analytic framework that formulates Battle Royale as a phase-conditioned survival system with O0-O8 mechanism families |
| PUBG prediction claim | We test whether ontology-derived feature blocks improve team-level, relation-aware phase survival prediction in PUBG |
| final-rank claim | We evaluate whether calibrated phase-wise hazards and cumulative survival scores support match-local final-rank prediction |
| Caldera claim | We use Warzone/Caldera as geometry-rich construct evidence for spatial affordance measurement |

Current status after the required-cohort OM ladder:

```text
The current paper presents an ontology-guided Battle Royale survival analysis
framework.  The allowed claim is not a formal ontology or final exhaustive genre
taxonomy; it is a stable analytic framework whose central pressure / habitat
mechanism improves PUBG phase-survival prediction across all required cohorts.
```

## Forbidden Collapses

Do not collapse the paper into any of these weaker or unsupported framings:

- "We predict PUBG final rank from a bag of match statistics."
- "Warzone validates supervised Battle Royale survival prediction."
- "The ontology is proven only because official game descriptions mention
  similar words."
- "The phase model is only a preprocessing trick rather than the core survival
  formulation."
- "Independent phase hazards are a full match simulator that recover exact
  final placement order."
- "The general ontology is needed merely because PUBG has phases."

## Immediate Methodological Consequence

The next execution step must check whether the current PUBG code and feature
tables can support this argument order:

```text
O0-O8 ontology
-> OM0-OM6 feature blocks
-> team-level, relation-aware next-phase hazard
-> cumulative survival curve
-> match-local final-rank evaluation
```

That is the purpose of the feature availability audit in the execution
checklist.
