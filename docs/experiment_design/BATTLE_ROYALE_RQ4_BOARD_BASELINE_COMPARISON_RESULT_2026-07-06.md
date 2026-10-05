# Battle Royale RQ4 Board-Baseline Comparison Result

Date: 2026-07-06
Class: `ExperimentResult`
Lifecycle: `analysis_complete_bounded`

## Purpose

This experiment closes the blocking RQ4 manuscript gap by comparing the
phase-transition risk/advantage board against simple and prior-work board
baselines.

The question is:

```text
Do phase-transition risk and advantage boards outperform random, simple
survival-capital, zone-distance, Sagong-style, official-raw, and
pressure-habitat boards?
```

## Source Protocol

```text
docs/experiment_design/BATTLE_ROYALE_RQ4_BOARD_BASELINE_COMPARISON_PROTOCOL_2026-07-06.md
```

## Execution

Executable:

```text
tools/compare_rq4_board_baselines.py
```

Command:

```bash
PG_PASSWORD=... python3 tools/compare_rq4_board_baselines.py --refresh-features
```

## Outputs

| Artifact | Path |
| --- | --- |
| score rows | `reports/rq4_board_baseline_comparison/rq4_board_baseline_score_rows.csv` |
| board rows | `reports/rq4_board_baseline_comparison/rq4_board_baseline_board_rows.csv` |
| cohort summary | `reports/rq4_board_baseline_comparison/rq4_board_baseline_by_cohort.csv` |
| aggregate summary | `reports/rq4_board_baseline_comparison/rq4_board_baseline_aggregate.csv` |
| sign-count summary | `reports/rq4_board_baseline_comparison/rq4_board_baseline_sign_count.csv` |
| example figure | `reports/rq4_board_baseline_comparison/rq4_board_baseline_example_board.png` |
| generated report | `reports/rq4_board_baseline_comparison/RQ4_BOARD_BASELINE_COMPARISON_RESULT_2026-07-06.md` |
| manifest | `reports/rq4_board_baseline_comparison/rq4_board_baseline_manifest.json` |

Coverage:

| Field | Value |
| --- | ---: |
| variants | 7 |
| cohorts | 13 |
| matches | 1,789 |
| raw score rows | 735,176 |
| intersected score rows | 727,020 |
| board rows | 74,312 |

## Aggregate Result

| Board | Phase boards | Median risk top-1 | Median risk top-3 | Median risk MRR | Median advantage top-team final top-10 | Median advantage NDCG@5 | Median risk-delta top-1 | Median advantage-delta top-team final top-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pressure_habitat_hgb_board` | 10,616 | 0.7188 | 0.9359 | 0.8296 | 0.9514 | 0.9419 | 0.6172 | 0.9713 |
| `lstm_realtime_board` | 10,616 | 0.7004 | 0.9172 | 0.8121 | 0.9490 | 0.9389 | 0.6254 | 0.9722 |
| `official_raw_hgb_board` | 10,616 | 0.6952 | 0.9252 | 0.8113 | 0.9473 | 0.9422 | 0.6032 | 0.9657 |
| `health_alive_count_board` | 10,616 | 0.5094 | 0.8349 | 0.6853 | 0.9184 | 0.8978 | 0.4687 | 0.9446 |
| `sagong_prior_score_hgb_board` | 10,616 | 0.4897 | 0.8385 | 0.6756 | 0.9042 | 0.9150 | 0.4152 | 0.9497 |
| `zone_distance_board` | 10,616 | 0.4240 | 0.7599 | 0.6117 | 0.8926 | 0.8799 | 0.4588 | 0.9461 |
| `random_alive_board` | 10,616 | 0.3868 | 0.7564 | 0.6008 | 0.8742 | 0.8717 | 0.4150 | 0.9314 |

## Sign-Count Summary

LSTM realtime board versus simple/prior boards:

| Baseline | Risk top-1 | Risk MRR | Advantage final top-10 | Advantage NDCG@5 |
| --- | --- | --- | --- | --- |
| `random_alive_board` | 13-0-0 | 13-0-0 | 13-0-0 | 13-0-0 |
| `health_alive_count_board` | 13-0-0 | 13-0-0 | 12-0-1 | 12-0-1 |
| `zone_distance_board` | 13-0-0 | 13-0-0 | 12-0-1 | 13-0-0 |
| `sagong_prior_score_hgb_board` | 13-0-0 | 13-0-0 | 10-0-3 | 13-0-0 |
| `official_raw_hgb_board` | 11-0-2 | 9-0-4 | 5-0-8 | 4-0-9 |
| `pressure_habitat_hgb_board` | 6-0-7 | 4-0-9 | 3-1-9 | 2-0-11 |

Format:

```text
LSTM wins - ties - LSTM losses across 13 cohorts.
```

## Interpretation

The LSTM realtime board clearly beats the random, health/alive-count,
zone-distance, and Sagong-style adapted boards across the core risk metrics and
most advantage metrics.

The pressure-habitat HGB board is slightly stronger than the LSTM board on
median phase-cut risk scoring and several advantage ranking metrics:

```text
pressure-habitat HGB median risk top-1 = 0.7188
LSTM realtime median risk top-1        = 0.7004
```

Risk top-1/top-3/top-5 hit rates are averaged over phase boards with at least
one next-phase elimination.  All model variants are evaluated on the same
intersected cohort/match/phase/team row universe.

This follows the pre-registered fallback:

```text
Use the LSTM as the realtime readout model, not as the globally strongest
phase-cut board scorer.
```

## Supported Claim

```text
The realtime survival readout can be converted into phase-transition boards
that substantially outperform simple and prior-score board baselines for
identifying next-phase risk and final-outcome advantage.
```

## Claim Boundary

Do not claim:

```text
The LSTM board is the strongest board scorer under all comparisons.
```

The stronger manuscript-safe claim is:

```text
The representation pipeline supports a strong phase-cut board scorer, while
the LSTM path supplies the dynamic realtime readout form.
```

Do not claim:

```text
This validates a separately supervised 10-second streaming deployment.
```

This comparison uses phase-cut board rows and saved phase-hazard predictions.
