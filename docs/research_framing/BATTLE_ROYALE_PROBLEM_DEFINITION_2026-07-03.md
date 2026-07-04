# Battle Royale Problem Definition

Date: 2026-07-03
Class: `ClaimBoundary`
Lifecycle: `active`

## Purpose

This document fixes the problem definition for the Battle Royale framework
paper.

It answers:

```text
Why are we doing this?
What problem does the framework solve?
What scientific or methodological progress does it make?
```

## One-Sentence Problem

Battle Royale telemetry is often treated as game-specific result prediction,
but the genre is better understood as a phase-conditioned, multi-agent survival
process in which contraction or elimination pressure changes the survivable
environment, agent behavior, relation structure, and cumulative risk.

Therefore, the project asks whether a general Battle Royale ontology can turn
raw telemetry into interpretable, phase-wise survival features that predict and
explain team survival across map, perspective, and team-size cohorts.

## Core Research Problem

Given observations through phase cut `k` for team `i` in match `m`, estimate:

```text
lambda_i(k) = Pr(team i is eliminated before or during the next phase
                 | history through phase cut k)
```

Then use the sequence of phase hazards to derive:

```text
cumulative survival probability
-> final-rank score or rank-related evaluation
```

The main research question is:

```text
Can ontology-derived Battle Royale feature families explain and predict
phase-wise team survival better than conventional or game-specific feature
sets, while remaining interpretable across map, perspective, team-size, and
observability differences?
```

## Why Existing Framing Is Not Enough

| Existing Framing | Limitation | Our Reframing |
| --- | --- | --- |
| Predict victory or final rank from aggregate result statistics | Often post-hoc and weakly tied to when and why survival changes | Predict next-phase hazard, then connect hazards to cumulative survival and rank |
| Treat PUBG phase as a PUBG-only Blue Zone detail | Misses that shrinking or eliminating pressure is genre-defining across Battle Royale variants | Define contraction / elimination pressure as a genre mechanism, with PUBG Blue Zone as one projection |
| Use player-level or score-based phase summaries | Does not capture team-level survival capital or relations among teams | Model team-level, relation-aware survival |
| Engineer many telemetry features ad hoc | Makes ablations hard to interpret and hard to transfer | Organize features by O0-O8 ontology families and explicit observability masks |
| Assume one dataset sees everything | PUBG is label-rich but geometry-poor; Warzone/Caldera is geometry-rich but label-poor | Use each dataset for the claim it can actually support |

## Why This Helps

### Scientific Contribution

The project turns Battle Royale from a loose genre label into a formal survival
system:

```text
rules and match context
-> survival capital
-> contraction / elimination pressure
-> pressure-conditioned habitat
-> behavior
-> interaction exposure
-> relation structure
-> next-phase outcome
-> cumulative rank-related result
```

This lets the paper ask mechanistic questions rather than only leaderboard
questions:

- How does phase pressure change the survival environment?
- How do teams alter movement, contact, and positioning under pressure?
- When do relation features matter more than individual movement features?
- Which mechanisms matter early, mid, and late in a match?

### Methodological Contribution

The framework changes the experiment from a single feature bundle into a
controlled ablation system.

```text
OM0 baseline
-> OM1 survival capital
-> OM2 movement / rotation
-> OM3 contraction / habitat pressure
-> OM4 interaction / threat exposure
-> OM5 relation structure
-> OM6 spatial / terrain proxies
```

Each block has a conceptual owner, a causal-clock rule, and an observability
status.  This prevents the paper from claiming that "more features help" without
knowing which mechanism helped.

### Empirical Contribution

The required PUBG cohorts make the framework testable under meaningful game
variation:

- map variation through separate TPP squad cohorts such as Erangel TPP squad,
  Taego TPP squad, Miramar / Desert TPP squad, Sanhok TPP squad, and others;
- perspective variation through same-map TPP/FPP squad comparisons such as
  Erangel TPP squad versus Erangel FPP squad and Taego TPP squad versus Taego
  FPP squad;
- team-size variation through same-map squad/duo/solo comparisons such as
  Erangel TPP squad, Erangel TPP duo, and Erangel TPP solo.

The goal is not to train one pooled model and call it general.  The goal is to
train and evaluate the ontology ladder separately by cohort, then ask which
mechanisms remain useful, drift, or fail.

### Construct Contribution

Warzone/Caldera helps with a different problem:

```text
Can spatial / terrain / visibility affordance constructs be measured when a
Battle Royale dataset has richer geometry than PUBG?
```

Because the current Caldera data lacks usable rank/survival labels and phase
zone states, it cannot validate supervised survival prediction.  It can validate
that the spatial part of the ontology is not merely a PUBG coordinate proxy.

## What Progress The Project Claims

The project advances the research line in four steps:

| From | To |
| --- | --- |
| final-result prediction | phase-wise survival process |
| PUBG-specific feature engineering | Battle Royale ontology projections |
| individual or score-based summaries | team-level, relation-aware hazard modeling |
| one dataset, one observability regime | label-rich PUBG plus geometry-rich Caldera construct evidence |

## What The Paper Should Not Claim

Do not claim:

- that the ontology is proven only because official game descriptions mention
  similar mechanics;
- that one PUBG cohort proves the whole Battle Royale framework;
- that Warzone/Caldera supports supervised survival prediction in the current
  data state;
- that phase hazards are a full match simulator;
- that the goal is merely to improve final-rank accuracy with a larger feature
  list.

## Paper-Level Problem Statement

Use this as the manuscript-facing version:

```text
Existing game-telemetry studies can predict Battle Royale outcomes from
aggregate or game-specific statistics, but they often under-specify the genre's
central dynamic: survival unfolds through phases in which contraction or
elimination pressure reshapes the available habitat, the behavior of teams,
and the relation field among opponents.  We formulate Battle Royale as a
phase-conditioned multi-agent survival process, operationalize that formulation
as an ontology of feature families, and test whether those families improve
team-level next-phase survival prediction and rank-related evaluation across
map, perspective, and team-size cohorts in PUBG.  We use Warzone/Caldera not as
a supervised label source, but as geometry-rich construct evidence that the
same spatial affordance vocabulary can be instantiated under a different
observability regime.
```

## Immediate Consequence

The next work item remains:

```text
Feature availability audit for OM0-OM6 on required cohorts
```

That audit is no longer just an engineering check.  It is the first test of
whether the problem definition can be operationalized without leakage, without
pooled-cohort shortcuts, and without collapsing back into final-result
statistics.

