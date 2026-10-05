# Battle Royale Experiment Status And Next Decision

Date: 2026-07-06
Class: `ExperimentDecision`
Lifecycle: `active`

## Purpose

This document consolidates the current Battle Royale thesis experiment state
after the revised-RQ3 prior-baseline, same-condition sequence, dynamic
probability trajectory, RQ4 board, O7 supplementary, and Warzone / Caldera
observability-gap work.

It answers three practical questions:

1. what evidence is already available for the manuscript research questions;
2. what the current model comparison actually supports;
3. what experiment should be run next without turning the thesis into an
   unfocused model sweep.

This is a decision document, not a replacement for the detailed evidence
ledger.

Source of truth for detailed evidence:

```text
docs/experiment_design/BATTLE_ROYALE_EXPERIMENT_EVIDENCE_LEDGER_2026-07-05.md
```

## Current Manuscript Spine

The thesis should remain organized as:

```text
RQ1. Battle Royale genre formalization
RQ2. PUBG official-telemetry representation as the representative game case
RQ3. prior tabular / phase-aware prediction and 10-second realtime
     elimination-risk model
RQ4. phase-transition most-at-risk / most-advantaged team prediction
```

Do not reorganize the manuscript around the chronological order of
experiments.  Each experiment should appear only after the problem and method
that make it necessary.

## Current Bottom Line

The main thesis statement is:

```text
Battle Royale can be formalized as phase-conditioned multi-agent survival,
represented in PUBG official telemetry, extended from prior tabular/phase-aware
prediction into a 10-second realtime elimination-risk model, and rendered as a
phase-transition risk/advantage board.
```

The strongest mechanism-level empirical statement remains:

```text
In PUBG, Battle Royale mechanism-aligned representation design, especially the
contraction-pressure plus habitat / survival-environment block, improves
phase-wise survival hazard prediction across required cohorts.
```

The current evidence does not support:

```text
Deep sequence models are generally better than tabular reductions.
```

The current evidence also does not support:

```text
Graph architecture is a core contribution.
```

The safe thesis framing is therefore:

```text
The contribution is a Battle Royale survival formalization and a
leakage-guarded representation pipeline, not a claim that one neural
architecture universally dominates.
```

The RQ3 realtime readout addendum supports:

```text
Phase-wise hazard predictions can be rendered as dynamic match-local survival
probability trajectories, which static Sagong-style score baselines do not
directly provide.
```

The RQ4 board addendum supports:

```text
Those trajectories can be converted into phase-transition risk/advantage boards
that identify the most-at-risk and most-advantaged team states.
```

## RQ-Level Status

| RQ | Question | Current status | Main evidence | Manuscript-safe claim |
| --- | --- | --- | --- | --- |
| `RQ1` | How can the Battle Royale genre be formalized? | supported as framework vocabulary | O0-O8 lock and 15-game mini corpus | Battle Royale can be modeled as phase-conditioned multi-agent survival with reusable mechanism families. |
| `RQ2` | How can the formalization be represented in the representative Battle Royale game PUBG? | supported as data contract | PUBG observability table, feature manifest, feature availability audit, required cohort set, snapshot tensor contract | PUBG official telemetry can instantiate the formalization as leakage-guarded team x phase-cut and 10-second snapshot features. |
| `RQ3` | How can prior tabular and phase-aware prediction be extended into realtime elimination-risk modeling? | supported for phase-cut dynamic trajectories; descriptive for 10-second prefix readout | Sagong-style/basic tabular comparison, same-condition sequence comparison, dynamic probability fluctuation result, 10-second prefix demo | Prior phase-aware prediction can be extended into a causal telemetry survival readout that updates hazard, cumulative survival, and match-local survival share. |
| `RQ4` | Can the realtime model identify the most unfavorable and most favorable teams at phase transitions? | supported with completed baseline comparison and HGB-scoring boundary | realtime risk-advantage framework, phase-transition board result, board-baseline comparison result | Phase-transition risk and advantage boards identify most-at-risk and most-advantaged teams; LSTM is the realtime readout form, while pressure-habitat HGB is slightly stronger as a phase-cut scorer. |

## Completed Evidence Map

| Evidence block | Completed artifact | Main result | Claim status |
| --- | --- | --- | --- |
| Genre coverage | mini corpus coverage | O0-O8 covered 15/15 source-rich Battle Royale and Battle-Royale-like games/modes without requiring O9 | supports framework vocabulary, not universal ontology proof |
| PUBG telemetry mapping | official telemetry observability table | direct telemetry, pre-cut derived features, proxy-only O6 signals, label-only outcomes, and not-provided geometry are separated | supports RQ2 data contract |
| Feature availability | OM0-OM6 audit | required cohorts have executable leakage-guarded feature blocks | supports operational readiness |
| OM ladder | required-cohort OM0-OM6 run | `OM2 -> OM3` pressure-habitat improves hazard AUC in 13/13 cohorts | central mechanism support |
| Phase stratification | within-phase OM analysis | pressure-habitat effect survives phase stratification | weakens pooled phase-base-rate objection |
| Paired bootstrap | match-cluster bootstrap | early/mid pressure-habitat cells receive uncertainty support | supports mechanism robustness |
| O6Z sensitivity | corrected vertical-proxy ladder | `pos_z`-based O6Z does not robustly improve over OM5 | bounded negative PUBG terrain-affordance result |
| Warzone / Caldera bridge | observability-gap bridge and construct artifact | geometry-rich data can express terrain, route, building, rough-mobility, and compound constructs missing from official PUBG telemetry | supplementary observability / future O6 boundary, not core RQ |
| Prior/basic baseline | Sagong-style and official raw baseline comparison | representation-rich pressure/habitat features beat prior-style scores and official raw tabular baselines | supports RQ3 prior-work positioning |
| Same-condition sequence | snapshot tabular vs TCN/GRU/LSTM | sequence accumulation beats neural last-only in most cohorts, but summary HGB remains strongest same-condition baseline | bounded RQ3 model-form result |
| Dynamic probability trajectory | phase-wise hazard to cumulative survival and match-local survival share | LSTM phase-cut predictions produce dynamic team probability curves across 13 cohorts; median latest top-5 AUC from `pi` is 0.9290. A one-match 10-second prefix demo exists, but it is not a separately validated streaming metric | supports RQ3 realtime readout-form difference from static phase-score baselines |
| Phase-transition risk/advantage board | risk and advantage rankings at each phase cut | 10,616 LSTM phase boards across 13 cohorts; median risk top-1 hit is 0.7004, median risk top-3 hit is 0.9172, median advantage top-team final top-10 is 0.9490 | supports RQ4 risk/advantage framework |
| Board-baseline comparison | random, health/alive-count, zone-distance, Sagong-style, official-raw, pressure-habitat HGB, and LSTM boards | LSTM beats simple/prior boards in most cohort comparisons; pressure-habitat HGB is stronger on median phase-cut risk top-1 (0.7188 vs 0.7004) | supports RQ4 with HGB-scoring boundary |
| O7 advantage-edge | required-cohort A/B/C null control | observed contact-gated advantage edges beat edge-shuffled null for hazard and hazard-implied rank bridge | supplementary relation-topology evidence only |

## Current Model Comparison Conclusion

The complete comparable model set now contains:

```text
35 representation / prior tabular models
11 same-condition official snapshot model-zoo models
3 earlier same-condition sequence arms retained for comparison
49 comparable completed required-cohort rows total
```

Primary source files:

```text
model_stgcn/checkpoints/br_rq4_prior_model_zoo_required_v1/rq4_prior_baseline_metrics_summary.csv
model_stgcn/checkpoints/br_rq4_sequence_model_zoo_required_v1/snapshot_survival_bridge_metrics_summary.csv
model_stgcn/checkpoints/br_rq4_same_condition_snapshot_required_v2/snapshot_survival_bridge_metrics_summary.csv
```

### Top And Manuscript-Relevant Models

Sorted by required-cohort median hazard AUC:

| Role | Model | Cohorts | Median AUC | Mean AUC | Median AP | Median BCE | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| primary candidate | `conventional_stdb_network_habitat__hgb` | 13 | 0.7753 | 0.7693 | 0.6577 | 0.5063 | best current required-cohort model |
| close representation candidate | `a7_raw_plus_habitat__hgb` | 13 | 0.7742 | 0.7696 | 0.6566 | 0.5063 | shows habitat pressure is sufficient for near-best result |
| close full representation | `a11_full_representation__hgb` | 13 | 0.7740 | 0.7694 | 0.6586 | 0.5065 | full feature-only representation; no major gain over habitat |
| primary sensitivity learner | `conventional_stdb_network_habitat__xgb` | 13 | 0.7717 | 0.7668 | 0.6517 | 0.5099 | same representation with different tree learner |
| strong official snapshot baseline | `snapshot_summary_hgb` | 13 | 0.7550 | 0.7490 | 0.6457 | 0.5154 | strongest same-condition official snapshot model |
| official raw baseline | `a0_official_raw__hgb` | 13 | 0.7521 | 0.7479 | 0.6307 | 0.5245 | direct official telemetry aggregate baseline |
| conventional baseline | `a3_current_conventional__hgb` | 13 | 0.7458 | 0.7408 | 0.6174 | 0.5310 | current conventional feature baseline |
| best sequence-family addendum | `snapshot_lstm_no_graph` | 13 | 0.7441 | 0.7379 | 0.6209 | 0.5355 | improves sequence family, supports dynamic trajectory readout, but does not beat summary HGB |
| earlier sequence baseline | `snapshot_gru_no_graph` | 13 | 0.7355 | 0.7345 | 0.6148 | 0.5482 | sequence accumulation baseline |
| earlier temporal conv baseline | `snapshot_tcn_no_graph` | 13 | 0.7325 | 0.7316 | 0.6238 | 0.5253 | sequence accumulation baseline |
| neural last-only lower bound | `snapshot_last_only` | 13 | 0.7213 | 0.7233 | 0.6091 | 0.5352 | lower bound for sequence accumulation |
| Sagong-style adapted score baseline | `a2_prior_scores__hgb` | 13 | 0.6525 | 0.6462 | 0.4480 | 0.6006 | much weaker under team-phase hazard setting |

### Interpretation

The model comparison supports:

```text
Representation design dominates the current RQ4 evidence.
```

It does not support:

```text
The thesis should be written as a deep-learning superiority claim.
```

The best current model is a tree learner over a mechanism-aligned
pressure-habitat representation.  Sequence models are still useful because they
show that phase survival can be modeled over cumulative 10s snapshots and
converted into changing probability trajectories, but they are not the aggregate
performance winner under the same official snapshot feature surface.

Transformer-style sequence models are not included in the completed
required-cohort comparison until they finish a full required-cohort run.

## Erangel Versus Generalization Boundary

There are two different claims that must not be mixed.

| Claim type | Current status | Meaning |
| --- | --- | --- |
| Cohort-separated robustness | partially supported by existing required-cohort runs | the same representation/model family performs well across independently trained map / perspective / team-size cohorts |
| Strict external map transfer | not yet completed | a design selected on one source setting is applied to held-out maps or styles under a frozen protocol |

The current 13-cohort tables already show repeated support across maps and
settings.  However, if the thesis wants a cleaner statement that the result is
not an Erangel-specific artifact, the next experiment should be a frozen-design
map/style stability test rather than another broad model zoo.

## Decision: Do Not Re-Run Every Model Everywhere

Running all 49 completed model arms across every new map/style condition should
not be the main next step.

Reasons:

1. It turns the thesis into a model-search paper rather than a Battle Royale
   representation paper.
2. It increases selection bias unless a new holdout protocol is locked.
3. It delays the board-baseline comparison now needed for the revised RQ4.
4. The current evidence already says the main effect is representation, not
   architecture variety.

The correct next step is:

```text
Freeze the board metrics and compare the risk/advantage board against simple
and prior-work board baselines without reopening model selection.
```

## Frozen Panel For The Next Experiment

Recommended panel:

| Role | Arm |
| --- | --- |
| primary model | `conventional_stdb_network_habitat__hgb` |
| primary learner sensitivity | `conventional_stdb_network_habitat__xgb` |
| habitat sufficiency check | `a7_raw_plus_habitat__hgb` |
| official raw baseline | `a0_official_raw__hgb` |
| conventional feature baseline | `a3_current_conventional__hgb` |
| Sagong-style adapted baseline | `a2_prior_scores__hgb` |
| official snapshot baseline | `snapshot_summary_hgb` |
| sequence boundary baseline | `snapshot_lstm_no_graph` |

Optional, if runtime is acceptable:

```text
snapshot_gru_no_graph
snapshot_tcn_no_graph
a11_full_representation__hgb
```

This panel is enough to answer:

```text
Does the primary Battle Royale representation remain stronger than prior-style,
official raw, conventional, snapshot-summary, and compact sequence alternatives
across different map/style settings?
```

## Completed Next Experiment

Name:

```text
RQ4 risk/advantage board baseline comparison
```

Primary question:

```text
The realtime risk and advantage boards were compared against simple and
prior-work board baselines at identifying next-phase eliminated teams and
final top-k teams.
```

Baseline boards:

| Baseline | Risk-board score | Advantage-board score |
| --- | --- | --- |
| Sagong-style adapted score | inverse strategy-survival score or highest risk score | highest strategy-survival score |
| official raw / tabular | official raw hazard probability | official raw survival share |
| health / alive-count | low survival capital | high survival capital |
| zone-distance | outside / far from safe zone | safe-zone centered / lower pressure |
| pressure-habitat HGB | HGB hazard probability | HGB survival share |
| LSTM realtime board | LSTM hazard probability | LSTM survival share |

Primary metrics:

```text
risk top-1 hit
risk top-3 / top-5 hit
risk-board MRR
advantage top-team final top-5 / top-10
advantage top-5 contains final top-10
advantage-board NDCG@5
```

Primary comparison:

```text
LSTM realtime board - Sagong-style board
LSTM realtime board - health/alive-count board
LSTM realtime board - zone-distance board
LSTM realtime board - official raw/tabular board
LSTM realtime board - pressure-habitat HGB board
```

Result:

```text
The LSTM realtime board beats random, health/alive-count, zone-distance, and
Sagong-style boards across the core risk metrics.  Pressure-habitat HGB is
slightly stronger on median phase-cut risk top-1, so the manuscript should
frame the LSTM as a realtime readout model, not as the strongest board scorer.
```

Preferred reporting:

```text
aggregate table + cohort table + sign-count summary + one phase-transition
example figure
```

Do not tune a new winner inside each cohort.

## Warzone / Caldera Placement

Warzone / Caldera is no longer a core RQ in the revised manuscript spine.
It should not be placed as the main RQ3 section.

```text
Core spine:
RQ1 genre formalization
-> RQ2 PUBG official-telemetry mapping
-> RQ3 prior/tabular plus realtime 10-second survival model
-> RQ4 risk/advantage board
```

Its manuscript role is:

```text
Supplementary observability boundary: PUBG official telemetry is label-rich
for supervised survival prediction, but does not directly provide some
geometry-rich terrain/topology constructs.  Warzone / Caldera can illustrate
future O6 terrain extensions, not the core supervised model.
```

Allowed Warzone / Caldera claim:

```text
Warzone / Caldera supports supplementary geometry-rich construct validation for
future O6 terrain/topology extensions.
```

Forbidden Warzone / Caldera claim:

```text
Warzone / Caldera validates the PUBG supervised survival model.
```

Forbidden until implemented:

```text
The current Warzone artifact proves LOS, occlusion, or cover effects.
```

## Final-Rank And Win Prediction Boundary

The thesis needs final rank / top-k / win prediction because Battle Royale is
not only next-phase survival.  But the current evidence is stronger for
phase-wise hazard than for a general final-rank bridge.

Current safe wording:

```text
Phase-wise hazard prediction is the supported core.  Final rank and top-k are
evaluated as phase-cut forecasts or hazard-implied bridges, with calibration
and final-outcome heads treated as the next refinement.
```

Do not overstate:

```text
The current cumulative survival score already solves final rank prediction.
```

The next fixed-panel experiment should report final top-k / win where the
metric is stable, but the headline should remain phase survival unless the
final-outcome bridge clearly improves.

## Immediate Work Queue

| Priority | Work item | Output |
| ---: | --- | --- |
| 1 | Register this status/decision document | README and manifest entries |
| 2 | Build board-baseline comparison protocol | complete: `experiment_design/BATTLE_ROYALE_RQ4_BOARD_BASELINE_COMPARISON_PROTOCOL_2026-07-06.md` |
| 3 | Run board-baseline comparison | complete: aggregate + cohort + sign-count table in `reports/rq4_board_baseline_comparison/` |
| 4 | Decide final-rank/top-k reporting discipline | top-k / win bridge table or bounded limitation |
| 5 | Move Warzone / Caldera to supplement/future-work boundary | short observability-boundary note |

## One-Sentence Current Decision

```text
The RQ4 board-baseline comparison is complete: keep the board result, frame
LSTM as the realtime readout form, report pressure-habitat HGB as the stronger
phase-cut scorer where applicable, and move Warzone / Caldera to supplementary
observability boundary.
```
