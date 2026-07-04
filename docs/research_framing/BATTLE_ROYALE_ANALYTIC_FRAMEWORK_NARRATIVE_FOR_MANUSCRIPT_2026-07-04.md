# Battle Royale Analytic Framework Narrative For Manuscript

Date: 2026-07-04
Class: `ClaimBoundary`
Lifecycle: `active`

## Purpose

This document fixes the manuscript-facing narrative for the first paper
question:

```text
Can we formulate Battle Royale as an ontology-guided survival analysis
framework rather than as a PUBG-specific feature list?
```

It turns the internal vocabulary, corpus, operationalization, and OM ladder
results into a compact paper argument.  It is not a replacement for the feature
manifest, model formalization, or result audit.  It states how those artifacts
should be used in Introduction, Theory, and Validation.

## Narrative Decision

The paper should not claim that it proves the final taxonomy of all Battle
Royale games.

The paper should claim:

```text
We propose an ontology-guided Battle Royale survival analysis framework and
validate its central mechanism through cohort-separated PUBG prediction
experiments.
```

The strongest defense is not that official game descriptions contain similar
words.  The strongest defense is that the framework produced a pre-specified
testable implication before the result audit:

```text
If the pressure-habitat block OM3 did not add signal after context, survival
capital, and movement were already available, the central habitat claim would
be weak.
```

The completed OM ladder supported that implication:

```text
OM2 -> OM3 is positive in 13/13 required cohorts and 8/8 map codes.
```

This is the spine of the framework narrative.

## One-Sentence Thesis

Use this as the manuscript anchor sentence:

```text
We propose an ontology-guided analytic framework that formulates Battle Royale
as a phase-conditioned multi-agent survival process in which contraction
pressure acting on spatial affordance produces a shrinking survival habitat;
operationalizing this framework as a leakage-guarded feature ladder in PUBG, we
find that the pressure-habitat mechanism improves next-phase survival
prediction in all 13 map / perspective / team-size cohorts and all 8 map
groups.
```

The sentence intentionally does not claim:

- causal identification;
- a universal final taxonomy;
- a formal ontology in the semantic-web or knowledge-representation sense;
- a full O7 graph-encoder result;
- a full O6 geometry-rich terrain result.

## Why This Is Not Just Feature-Group Naming

The expected reviewer attack is:

```text
Is this framework only a set of arbitrary feature groups with plausible names?
```

The answer has four layers.

| Defense layer | Evidence | Manuscript use |
| --- | --- | --- |
| Pre-specified central implication | The framework made `O2 x O6 = H` the central mechanism; failure of OM3 would weaken the habitat claim | Put this before the result |
| Cohort-separated result | `OM2 -> OM3` AUC gain is positive in 13/13 required cohorts | Main validation headline |
| Map-level robustness | Collapsing repeated Erangel/Taego cohorts still gives 8/8 positive map groups | Prevents the "same map repeated" objection |
| Feature-count rebuttal | OM3 adds 47 features and gains median `+0.0103`; OM5 adds 43 features and gains median `+0.0003` | Shows that gain is about feature family content, not simply adding many columns |

Recommended validation sentence:

```text
The ladder also argues against a simple feature-count explanation: OM3 adds 47
features and yields the only universally positive block-level AUC gain, whereas
OM5 adds 43 relation-summary features and contributes approximately zero at the
median.
```

## Core Mechanism

The framework's central construct is:

```text
H_i(k) = h(Z_k, G_i(k), A(k))
```

where:

| Symbol | Meaning | Ontology owner |
| --- | --- | --- |
| `Z_k` | phase-specific contraction / elimination pressure | `O2` |
| `G_i(k)` | spatial, terrain, route, obstacle, or environmental affordance available to team or agent `i` | `O6` |
| `A(k)` | alive population, crowding, and survival-capital context | `O1` / `O2` |
| `H_i(k)` | phase-conditioned survival habitat available to team or agent `i` | `O2 x O6` composite |

The predictive target remains:

```text
lambda_i(k) = P(team i is eliminated before the next phase | history through k)
```

The theory chain is a primary explanatory direction, not a complete one-way
causal graph:

```text
O2 pressure x O6 affordance -> H_i(k)
O1 survival capital, O5 resources, and H_i(k) -> O3 behavior
O3 behavior and O7 relation field -> O4 interaction / threat exposure
O1, H_i(k), O4, and O7 -> O8 survival / rank / qualification outcome
```

The manuscript should also acknowledge feedback:

```text
Combat can deplete survival capital; eliminations reshape alive-team density
and relation structure; other teams' movement can change a team's local
habitat.
```

## What Is New About Habitat

The paper should not pretend that zone-distance features are new.  They are not.

The new claim is:

```text
Zone, storm, ring, or staged-elimination pressure is treated as one projection
of a broader contraction / elimination mechanism.  Habitat is not a single zone
distance feature; it is the survival environment produced when that pressure
acts on spatial affordance and alive-population context.
```

Three defenses should be stated explicitly.

| Defense | Meaning |
| --- | --- |
| Composite formulation | `H = h(Z, G, A)`, not merely "distance to zone" |
| Conservative ladder position | OM3 is added after context, survival capital, and movement, so it must explain residual signal |
| Genre-level abstraction | Fall Guys, Tetris 99, F-ZERO 99, and 2D/top-down variants show that `O2` is broader than PUBG Blue Zone |

Recommended theory sentence:

```text
Zone-distance features have appeared in prior PUBG modeling; our contribution
is to formalize contraction pressure as a genre-level mechanism that transforms
available affordance into a survival habitat, then to test that mechanism as a
separate block after conventional context, capital, and movement information
has already been provided.
```

## Ablation Order Explanation

The theoretical chain begins with pressure and habitat, but the OM ladder tests
movement before pressure-habitat:

```text
OM0 context
-> OM1 survival capital
-> OM2 movement
-> OM3 pressure / habitat
-> OM4 threat exposure
-> OM5 relation structure
-> OM6 spatial proxy
```

This is not a contradiction.  It is a conservative test.

Recommended methods sentence:

```text
Although the theory treats pressure-habitat as an upstream mechanism, the
ablation ladder introduces movement before OM3 so that the habitat block is
credited only for signal remaining after conventional motion and rotation
behavior is already available.
```

## Manuscript Section Plan

### Introduction

Use four compact paragraphs.

| Paragraph | Job | Key content |
| --- | --- | --- |
| 1 | Prior work and gap | Prior PUBG work predicts victory/rank from result-level or aggregate statistics; Sagong-style work motivates phase-aware PUBG analysis but is solo/player-score oriented |
| 2 | Formulation | Battle Royale is a phase-conditioned survival system, not only a final-rank prediction task |
| 3 | Central implication and evidence preview | The framework predicts that pressure-habitat should add signal; OM3 improves next-phase survival prediction in 13/13 cohorts and 8/8 map groups |
| 4 | Contributions | Ontology/formulation, leakage-guarded OM ladder, pressure-habitat validation, observability-mask frame with Caldera as geometry-rich construct validation |

Draft paragraph skeleton:

```text
Battle Royale games are often modeled as outcome-prediction tasks, where match
statistics are used to forecast final placement or victory.  However, phase-
aware PUBG work suggests that survival meaning changes over the course of a
match.  Existing phase-aware work remains primarily player-level and score-
based, leaving open how to model team survival under changing pressure,
movement, interaction, and relation structure.
```

```text
We formulate Battle Royale as a phase-conditioned multi-agent survival process.
In this formulation, contraction or elimination pressure transforms the
available spatial or environmental affordance into a shrinking survival habitat.
Teams respond through movement, resource use, interaction, and relation
structure, producing next-phase elimination hazards and ultimately rank
outcomes.
```

```text
This formulation has an explicit validation criterion.  If the pressure-habitat block
failed to improve prediction after context, survival capital, and movement were
already available, the central habitat claim would be weak.  Instead, in 13
separately trained PUBG map / perspective / team-size cohorts, the
pressure-habitat block improves next-phase survival AUC in every cohort; the
same sign consistency holds across all 8 map groups.
```

### Theory

The Theory section should contain only the compact pieces needed for the paper:

- an O0-O8 table;
- the `H_i(k) = h(Z_k, G_i(k), A(k))` definition;
- the phase hazard `lambda_i(k)`;
- the primary explanatory chain plus one feedback sentence;
- the claim boundary that the theory is predictive/associational here, not a
  causal identification design.

Avoid long corpus details in Theory.  Use the corpus only to show that `O2`
and `O6` are not PUBG-only concepts.

### Validation

Use three validation layers.

| Layer | Question | Artifact | Main statement |
| --- | --- | --- | --- |
| Coverage | Is O0-O8 PUBG-only? | mini corpus table | 15 representative games/modes map to O0-O8 without new top-level family |
| Operationalization | Can O0-O8 become legal PUBG inputs? | OM feature audit and `spec.py` | OM0-OM6 implements a leakage-guarded feature ladder |
| Predictive implication | Does the central mechanism add signal? | OM ladder result | OM3 is positive in 13/13 cohorts and 8/8 map groups |

Suggested subsection names:

```text
Framework coverage beyond PUBG
Operationalizing the framework as an ablation ladder
The pressure-habitat validation test
Environment-conditioned family importance
```

## Secondary Finding: Environment-Conditioned Family Importance

OM4 and OM5 should not be presented only as limitations.

The first-run result pattern is:

- OM4 threat exposure has a small median gain but larger gains in compact or
  high-contact settings;
- OM5 relation-summary features are near zero on average, but meaningful gains
  appear mainly in small-map cohorts such as Karakin and Paramo;
- this is directionally consistent with the ontology claim that feature-family
  importance changes with survival environment density.

This should be framed as exploratory secondary validation, not as a confirmed
headline result.

Recommended sentence:

```text
As a secondary, exploratory pattern, the gains from interaction and relation
features are largest in compact or high-density cohorts, suggesting that
feature-family importance is itself conditioned by the survival environment.
This pattern is consistent with the framework but should be tested with explicit
density and uncertainty analyses before being treated as confirmatory.
```

Needed figure:

```text
Delta AUC for OM3->OM4 and OM4->OM5 versus map area or team density.
```

## Claim Boundaries

Use these verbs:

- `predicts`;
- `is associated with`;
- `supports`;
- `is consistent with`;
- `survival is shaped by`;
- `transforms the playable habitat available to teams`.

Avoid these verbs unless a separate causal design is added:

- `causes`;
- `proves`;
- `identifies`;
- `determines`.

Allowed wording:

```text
The results support the framework's central mechanism: survival is shaped by how
contraction pressure transforms the playable habitat available to teams.
```

Forbidden wording:

```text
The Battle Royale ontology is established because official game descriptions
mention similar mechanics.
```

Forbidden wording:

```text
The results prove that relation structure universally improves survival
prediction in Battle Royale games.
```

## Reviewer Attack Map

| Attack | Response |
| --- | --- |
| "This is just named feature groups." | The framework produced a pre-result testable implication: OM3 should add signal; that implication is supported in 13/13 cohorts and 8/8 maps. |
| "You just added more features." | OM3 adds 47 features and gains; OM5 adds 43 features and is near zero.  Content, not count alone, explains the headline. |
| "Habitat is just zone distance." | Habitat is `h(Z, G, A)`: contraction pressure applied to spatial affordance and alive-population context. |
| "Why does movement come before habitat in the ladder?" | The order is conservative: habitat must add residual signal after motion is already available. |
| "Multi-agent is oversold because O7 is weak." | The formulation is multi-agent; current O7 node-summary gains are conditional and full edge-stream tests remain future/sensitivity work. |
| "13 cohorts are not independent because maps repeat." | The sign consistency also holds after collapsing to map code: 8/8 map groups positive. |
| "Official pages do not prove an ontology." | Correct.  They support cross-game coverage of the framework vocabulary only; predictive utility is tested separately in PUBG telemetry. |

## Evidence Numbers To Use

| Quantity | Value |
| --- | ---: |
| Required cohorts | 13 |
| Map groups | 8 |
| `OM2 -> OM3` positive cohorts | 13/13 |
| `OM2 -> OM3` positive map groups | 8/8 |
| `OM2 -> OM3` median AUC gain | `+0.0103` |
| `OM2 -> OM3` gain range | `+0.0057` to `+0.0211` |
| Two-sided sign test, 13/13 | approximately `p = 0.00024` |
| Two-sided sign test, 8/8 | approximately `p = 0.0078` |
| OM3 added features | 47 |
| OM5 added features | 43 |
| OM5 median AUC gain | `+0.0003` |

Do not overstate the sign tests as definitive independent-sample inference.
They are compact descriptive robustness checks.  Pair them with match-level
bootstrap and phase-stratified analyses before final submission.

## Open Dependencies Before Final Manuscript

| Dependency | Why it matters |
| --- | --- |
| Phase-stratified OM aggregation | Separates pooled phase-base-rate effects from within-phase feature value |
| Paired bootstrap intervals | Turns small OM4/OM5/OM6 deltas into defensible or bounded claims |
| Gen-1/gen-2 boundary cleanup | Needed if older core-habitat null-control results are cited |
| Final citation verification | Sagong and non-Sagong PUBG/rank-prediction lines must be bibliography-clean |
| O6 Z-ladder run | Needed before claiming full spatial-affordance proxy validation |
| O7 edge-stream test | Needed before claiming full relation-structure modeling benefit |

## Current Conclusion

For the first paper question, the answer is:

```text
Yes, for the current paper path.  We can present an ontology-guided Battle
Royale survival analysis framework, and the framework is supported by coverage
evidence, legal operationalization, and a pre-specified pressure-habitat
validation result.  The claim remains an analytic and predictive framework, not
an exhaustive taxonomy, formal ontology, or causal proof.
```
