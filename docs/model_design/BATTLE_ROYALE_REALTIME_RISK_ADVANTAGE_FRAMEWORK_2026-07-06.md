# Battle Royale Realtime Risk-Advantage Framework

Date: 2026-07-06
Class: `ModelContract`
Lifecycle: `active`

## Purpose

This document defines the manuscript-facing model design for RQ4:

```text
Realtime elimination-risk prediction and phase-transition risk/advantage board
for Battle Royale matches.
```

The model is not presented merely as a higher-AUC classifier.  It is presented
as a survival-monitoring framework that answers:

1. which team is most likely to be eliminated next;
2. which team is currently in the most favorable survival position;
3. how those rankings change when the phase changes.

## Core Thesis Role

This framework gives the RQ4 contribution a concrete form:

```text
Battle Royale outcome prediction is a dynamic survival-monitoring problem.
At each time or phase cut, the model updates team-level elimination risk,
cumulative survival, and match-local advantage rankings.
```

This separates the thesis from prior phase-score work.  A Sagong-style score
baseline can say that a phase state is more or less favorable.  The proposed
framework produces an evolving probability board over teams in the same match.

## Terminology

Use these terms consistently.

| Term | Symbol | Meaning |
| --- | --- | --- |
| elimination risk | `lambda_i(tau)` | probability that team `i` is eliminated over the next horizon, given information through time `tau` |
| cumulative survival | `S_i(tau)` | survival score accumulated from earlier phase risks and the current prefix risk |
| match-local survival share | `pi_i(tau)` | normalized survival share among alive or at-risk teams in the same match |
| most at-risk team | `argmax_i lambda_i(tau)` | team currently most likely to be eliminated next |
| most advantaged team | `argmax_i pi_i(tau)` | team currently holding the strongest match-local survival position |
| risk spike | `Delta lambda_i(tau)` | sharp increase in elimination risk after a phase, zone, combat, or movement change |
| advantage swing | `Delta pi_i(tau)` | sharp change in match-local survival share |

Important wording rule:

```text
"Most dangerous team" should mean "most at-risk team" unless a separate
threat-to-others score is explicitly defined.
```

If the manuscript wants to discuss the team that is most dangerous to other
teams, use a separate term such as:

```text
threat source score
```

That is an O4/O7 exposure or relation output, not the same as elimination risk.

## Time And Phase Contract

Let `tau` denote a 10-second snapshot time and `k(tau)` the current phase.

The model uses only information observed up to `tau`:

```text
X_m(<= tau)
```

The main real-time hazard is:

```text
lambda_i(tau) = Pr(team i is eliminated before the next phase boundary
                   or over the chosen horizon | X_m(<= tau))
```

For the current phase-cut implementation, `tau` is restricted to phase cuts:

```text
tau = t_k
lambda_i(k) = Pr(Y_i(k+1)=1 | X_m(<= t_k))
```

For the 10-second prefix readout, `tau` can be any snapshot time inside phase
`k`.

## Model Architecture

The framework is architecture-agnostic but the current executable path is:

```text
10s official telemetry snapshots
-> causal temporal encoder
-> phase / horizon-conditioned hazard head
-> survival accumulation
-> phase-transition risk/advantage board
```

Current executable encoder:

```text
snapshot_lstm_no_graph
```

Candidate future encoders:

| Encoder | Role |
| --- | --- |
| LSTM / GRU | compact causal sequence model for snapshot histories |
| Transformer | longer-range temporal attention over snapshot histories |
| tabular HGB/XGB | strong phase-cut baseline and calibration reference |
| relation-aware model | optional supplementary edge/threat modeling, not the core claim unless validated |

## Prediction Heads

### Primary Head

```text
lambda_i(tau)
```

The primary head predicts next-horizon elimination risk.

### Derived Survival Readouts

Phase-cut version:

```text
S_i(k)  = product_{j <= k} (1 - lambda_i(j))
pi_i(k) = S_i(k) / sum_j S_j(k)
```

10-second prefix version inside phase `k`:

```text
S_i(tau)  = S_i(k-1) * (1 - lambda_i(k | X_m(<= tau)))
pi_i(tau) = S_i(tau) / sum_j S_j(tau)
```

`pi_i` is a match-local survival-share or advantage proxy.  It should not be
called a fully calibrated betting-style win probability.

### Optional Heads

Optional later heads can be added only after the primary risk board is stable:

| Head | Target | Status |
| --- | --- | --- |
| top-k head | final top-5 / top-10 | currently evaluated as a bridge from survival score |
| final rank head | normalized final rank | currently a bounded bridge, not the main trained head |
| threat source head | risk imposed on nearby opponents | future O4/O7 extension |

## Phase-Transition Board

At every phase transition, compute:

```text
RiskBoard_m(k) =
  sort_i(lambda_i(t_k), descending=True)

AdvantageBoard_m(k) =
  sort_i(pi_i(t_k), descending=True)
```

The report should include:

| Output | Definition | Interpretation |
| --- | --- | --- |
| most at-risk team | team with highest `lambda_i` | likely next eliminated team |
| top-r at-risk teams | highest `lambda_i` teams | phase-transition danger list |
| most advantaged team | team with highest `pi_i` | strongest survival position |
| top-r advantaged teams | highest `pi_i` teams | likely top-k / late-game candidates |
| biggest risk spike | largest positive `Delta lambda_i` | team whose situation worsened most |
| biggest advantage gain | largest positive `Delta pi_i` | team whose position improved most |
| leader change | change in `argmax_i pi_i` | phase transition altered match advantage |

## Evaluation Protocol

The framework should be evaluated as a board, not only as row-level AUC.

### Existing Metrics

Already available:

| Metric | Meaning |
| --- | --- |
| hazard AUC / AP / BCE | row-level next-phase elimination discrimination |
| top-5/top-10 AUC from `pi` | whether latest survival share ranks final top-k teams |
| rank Spearman | match-local ordering relation with final rank |
| leader changes | how often the advantaged team changes over phases |
| risk/probability shocks | whether the board captures major shifts |

### Needed Board Metrics

Next experiment should add:

| Metric | Definition |
| --- | --- |
| top-1 risk hit | whether the highest-risk team is eliminated in the next phase |
| top-r risk recall | whether an eliminated team appears in the top-r risk list |
| risk-board MRR | reciprocal rank of the first actually eliminated team |
| advantage top-k hit | whether the top-advantage team finishes top-5/top-10/wins |
| advantage-board NDCG | rank quality of the advantage board against final placement |
| phase-transition delta accuracy | whether risk/advantage changes after phase transition predict outcome changes |

These metrics should be compared against:

```text
Sagong-style score baseline
official raw / tabular baseline
health/alive-count baseline
zone-distance baseline
best mechanism-aligned pressure-habitat baseline
```

## Current Evidence

Completed evidence:

```text
docs/experiment_design/BATTLE_ROYALE_RQ4_DYNAMIC_PROBABILITY_FLUCTUATION_RESULT_2026-07-06.md
```

Current status:

| Evidence | Status |
| --- | --- |
| phase-cut dynamic probability trajectory across 13 cohorts | complete |
| 10-second prefix readout for one Erangel TPP squad test match | descriptive demo complete |
| phase-transition board metrics | first phase-cut board result complete |
| strict 10-second supervised streaming validation | pending |
| threat-to-others model | future extension |

First board result:

```text
docs/experiment_design/BATTLE_ROYALE_RQ4_PHASE_TRANSITION_RISK_ADVANTAGE_BOARD_RESULT_2026-07-06.md
```

Headline bounded readout:

| Metric | Value |
| --- | ---: |
| phase-transition boards | 10,616 |
| median risk top-1 hit | 0.6393 |
| median risk top-3 hit | 0.9172 |
| median advantage top-team final top-10 | 0.9490 |
| median advantage top-5 contains final top-10 | 1.0000 |

## Supported Claim

```text
The Battle Royale survival formulation can be operationalized as a realtime
risk-advantage framework: phase-wise hazards produce dynamic team-level
elimination risk, cumulative survival, and match-local advantage rankings.
```

## Boundary

Do not claim:

```text
The current LSTM is the best overall predictor.
```

The strongest hazard-AUC models remain the mechanism-aligned tabular
pressure-habitat models.

Do not claim:

```text
The 10-second prefix readout is already a fully validated realtime deployment
metric.
```

The current 10-second prefix artifact is a descriptive demo from a
phase-cut-trained checkpoint.  The phase-transition board evaluation is the
next required experiment.

## Manuscript Wording

Recommended wording:

```text
We propose a realtime risk-advantage framework for Battle Royale matches.  The
model continuously estimates each team's elimination risk and converts the
phase-wise hazards into cumulative survival and match-local advantage rankings.
At phase transitions, the framework identifies the teams most at risk, the
teams currently most advantaged, and the largest risk/advantage shifts.
```

Avoid:

```text
We simply predict final placement from accumulated match statistics.
```

Avoid:

```text
The model predicts exact live win probability in the bookmaker sense.
```
