# Battle Royale RQ4 Risk Advantage Board Object

Date: 2026-07-06
Package: research_framing
Lifecycle: active
Responsibility: Define the RQ4 claim, evidence objects, experiments, results, and boundaries.
Depends on: docs/experiment_design/BATTLE_ROYALE_RQ4_BOARD_BASELINE_COMPARISON_RESULT_2026-07-06.md
Used by: docs/research_framing/BATTLE_ROYALE_RQ_OBJECT_CHAIN_2026-07-06.md

## RQObject

Object id:

```text
RQ4RiskAdvantageBoard
```

Research question:

```text
Can the realtime survival readout identify the most unfavorable and most
favorable teams at phase transitions?
```

Role in the paper:

```text
Turn the RQ3 survival readout into an interpretable phase-transition board for
most-at-risk and most-advantaged team states.
```

## Main Claim

```text
The realtime survival readout can be converted into phase-transition boards
that identify immediate elimination risk and match-local survival advantage.
```

Manuscript-safe wording:

```text
Phase-transition boards can identify most-at-risk and most-advantaged teams;
LSTM supplies the realtime readout form, while pressure-habitat HGB is the
stronger phase-cut board scorer where applicable.
```

## Claim Objects

| Claim object | Claim | Evidence object | Result | Status |
| --- | --- | --- | --- | --- |
| `RQ4.C1.RiskBoardDefinition` | Hazard can be interpreted as next-phase elimination risk. | `RiskAdvantageFramework` | `RiskBoard_m(k) = sort_i(lambda_i(k), descending=True)`. | supported as model contract |
| `RQ4.C2.AdvantageBoardDefinition` | Cumulative survival share can be interpreted as match-local advantage. | `RiskAdvantageFramework` | `AdvantageBoard_m(k) = sort_i(pi_i(k), descending=True)`. | supported as model contract |
| `RQ4.C3.BoardIdentifiesRisk` | The board identifies likely next-phase eliminations. | `PhaseTransitionBoardResult`, `BoardBaselineComparisonResult` | LSTM median risk top-1 hit is 0.7004 and beats simple/prior boards on risk top-1 in 13/13 cohorts. | supported with HGB-scoring boundary |
| `RQ4.C4.BoardIdentifiesAdvantage` | The board identifies high final-outcome advantage. | `PhaseTransitionBoardResult`, `BoardBaselineComparisonResult` | LSTM median advantage top-team final top-10 is 0.9490. | supported with HGB-scoring boundary |
| `RQ4.C5.BoardStateChanges` | Board state changes over phases. | `PhaseTransitionBoardResult` | mean leader change rate is 0.4214. | supported descriptive |
| `RQ4.C6.BaselineBoundary` | The board must be compared against simple and prior-work boards. | `BoardBaselineComparisonResult` | comparison complete across 7 board variants. | supported |

## Experiment Objects

| Experiment object | Question | Method | Evidence produced |
| --- | --- | --- | --- |
| `RQ4.E1.RiskAdvantageFramework` | What is a risk board and advantage board? | Define board scores from `lambda_i(k)` and `pi_i(k)`. | `BATTLE_ROYALE_REALTIME_RISK_ADVANTAGE_FRAMEWORK_2026-07-06.md` |
| `RQ4.E2.PhaseTransitionBoardEvaluation` | Does the LSTM readout produce useful phase-transition boards? | Evaluate risk top-k, MRR, advantage top-k, NDCG, and leader changes over phase boards. | `BATTLE_ROYALE_RQ4_PHASE_TRANSITION_RISK_ADVANTAGE_BOARD_RESULT_2026-07-06.md` |
| `RQ4.E3.BoardBaselineComparison` | Does the board beat simple and prior-work board baselines? | Compare random, health/alive-count, zone-distance, Sagong-style, official-raw, pressure-habitat HGB, and LSTM boards. | `BATTLE_ROYALE_RQ4_BOARD_BASELINE_COMPARISON_RESULT_2026-07-06.md` |

## Evidence Objects

| Evidence object | Path | What it supports |
| --- | --- | --- |
| `RiskAdvantageFramework` | `docs/model_design/BATTLE_ROYALE_REALTIME_RISK_ADVANTAGE_FRAMEWORK_2026-07-06.md` | board definitions and metric contract |
| `PhaseTransitionBoardResult` | `docs/experiment_design/BATTLE_ROYALE_RQ4_PHASE_TRANSITION_RISK_ADVANTAGE_BOARD_RESULT_2026-07-06.md` | first LSTM board result |
| `BoardBaselineComparisonProtocol` | `docs/experiment_design/BATTLE_ROYALE_RQ4_BOARD_BASELINE_COMPARISON_PROTOCOL_2026-07-06.md` | frozen baseline comparison protocol |
| `BoardBaselineComparisonResult` | `docs/experiment_design/BATTLE_ROYALE_RQ4_BOARD_BASELINE_COMPARISON_RESULT_2026-07-06.md` | seven-board comparison and HGB boundary |
| `BoardBaselineAggregate` | `reports/rq4_board_baseline_comparison/rq4_board_baseline_aggregate.csv` | aggregate comparison table |
| `BoardBaselineSignCount` | `reports/rq4_board_baseline_comparison/rq4_board_baseline_sign_count.csv` | cohort sign-count comparison |
| `BoardExampleFigure` | `reports/rq4_board_baseline_comparison/rq4_board_baseline_example_board.png` | example board figure |

## Result Objects

### RQ4.R1 Board Definitions

Risk board:

```text
RiskBoard_m(k) = sort_i(lambda_i(k), descending=True)
```

Advantage board:

```text
AdvantageBoard_m(k) = sort_i(pi_i(k), descending=True)
```

### RQ4.R2 LSTM Phase-Transition Board Result

| Metric | Value |
| --- | ---: |
| cohorts | 13 |
| matches | 1,789 |
| LSTM phase boards | 10,616 |
| median risk top-1 hit | 0.7004 |
| median risk top-3 hit | 0.9172 |
| median risk MRR | 0.8121 |
| median advantage top-team final top-5 | 0.7768 |
| median advantage top-team final top-10 | 0.9490 |
| median advantage top-5 contains final top-10 | 1.0000 |
| mean leader change rate | 0.4214 |

### RQ4.R3 Board-Baseline Comparison

| Board | Median risk top-1 | Median risk top-3 | Median risk MRR | Median advantage top-team final top-10 |
| --- | ---: | ---: | ---: | ---: |
| `pressure_habitat_hgb_board` | 0.7188 | 0.9359 | 0.8296 | 0.9514 |
| `lstm_realtime_board` | 0.7004 | 0.9172 | 0.8121 | 0.9490 |
| `official_raw_hgb_board` | 0.6952 | 0.9252 | 0.8113 | 0.9473 |
| `health_alive_count_board` | 0.5094 | 0.8349 | 0.6853 | 0.9184 |
| `sagong_prior_score_hgb_board` | 0.4897 | 0.8385 | 0.6756 | 0.9042 |
| `zone_distance_board` | 0.4240 | 0.7599 | 0.6117 | 0.8926 |
| `random_alive_board` | 0.3868 | 0.7564 | 0.6008 | 0.8742 |

Sign-count summary for LSTM risk top-1:

| Baseline | LSTM wins / ties / losses |
| --- | --- |
| `random_alive_board` | 13 / 0 / 0 |
| `health_alive_count_board` | 13 / 0 / 0 |
| `zone_distance_board` | 13 / 0 / 0 |
| `sagong_prior_score_hgb_board` | 13 / 0 / 0 |
| `official_raw_hgb_board` | 11 / 0 / 2 |
| `pressure_habitat_hgb_board` | 6 / 0 / 7 |

## Boundary Object

Allowed:

```text
The realtime survival readout can be converted into phase-transition boards
that substantially outperform simple and prior-score board baselines.
```

Required boundary:

```text
Pressure-habitat HGB is slightly stronger as a phase-cut board scorer, so the
LSTM board should be framed as the realtime readout form, not the globally
strongest board scorer.
```

Forbidden:

```text
The LSTM board is the strongest board scorer under all comparisons.
```

Forbidden:

```text
This validates a separately supervised 10-second streaming deployment.
```

## Interface To Manuscript Discussion

RQ4 exports the following discussion objects:

```text
validated board form
simple/prior baseline superiority
pressure-habitat HGB scorer boundary
10-second deployment boundary
future calibration / final-outcome head need
```
