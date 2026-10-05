# Battle Royale RQ4 Board-Baseline Comparison Protocol

Date: 2026-07-06
Class: `ExperimentContract`
Lifecycle: `active`

## Purpose

This protocol closes the remaining blocking RQ4 experiment before first
manuscript assembly:

```text
Do phase-transition risk/advantage boards beat simple and prior-work board
baselines at identifying next-phase eliminated teams and final top-k teams?
```

This is a board evaluation, not another model-search run.

## Frozen Board Arms

| Arm | Risk-board score | Advantage-board score | Source |
| --- | --- | --- | --- |
| `random_alive_board` | deterministic random ranking among eligible rows | independent deterministic random ranking | feature rows |
| `health_alive_count_board` | low alive-player / health capital | high alive-player / health capital | `stdb.v_team_phase_feature_long` |
| `zone_distance_board` | high outside-safe / zone-pressure score | low outside-safe / zone-pressure score | `stdb.v_team_phase_feature_long` |
| `sagong_prior_score_hgb_board` | Sagong-style adapted prior-score HGB hazard | cumulative survival share from the same hazard | `a2_prior_scores__hgb` predictions |
| `official_raw_hgb_board` | official-raw HGB hazard | cumulative survival share from the same hazard | `a0_official_raw__hgb` predictions |
| `pressure_habitat_hgb_board` | pressure-habitat HGB hazard | cumulative survival share from the same hazard | `conventional_stdb_network_habitat__hgb` predictions |
| `lstm_realtime_board` | LSTM phase-cut hazard | LSTM dynamic match-local survival share | dynamic trajectory artifact |

No new learner is tuned in this comparison.

## Metrics

Primary risk-board metrics:

```text
risk top-1 hit
risk top-3 hit
risk top-5 hit
risk MRR
```

Primary advantage-board metrics:

```text
advantage top-team final top-5
advantage top-team final top-10
advantage top-5 contains final top-10
advantage NDCG@5 against final top-10
```

Phase-transition delta metrics:

```text
risk delta top-1 hit
risk delta MRR
advantage delta top-team final top-10
advantage delta NDCG@5 against final top-10
```

Delta metrics rank teams by the phase-to-phase change in the board score.  They
answer whether a worsening risk or improving advantage signal at a transition is
also outcome-relevant.

## Data Sources

| Artifact | Role |
| --- | --- |
| `reports/rq4_dynamic_probability_fluctuation/dynamic_probability_trajectories.csv` | proposed LSTM board score rows |
| `model_stgcn/checkpoints/br_rq4_prior_model_zoo_required_v1/*/feature_ablation_predictions.csv` | prior-score, official-raw, and pressure-habitat HGB hazard rows |
| `stdb.v_team_phase_feature_long` | final-rank labels and simple heuristic features |
| `reports/rq4_board_baseline_comparison/board_baseline_feature_rows.csv` | cached feature rows for reproducibility after the first DB-backed run |

## Execution

```bash
PG_PASSWORD=... python3 tools/compare_rq4_board_baselines.py --refresh-features
```

Expected outputs:

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

## Success Criterion

The board claim is manuscript-usable if the LSTM realtime board beats simple
heuristic and prior-score boards in a strong majority of cohorts on the core
risk and advantage metrics.

If the pressure-habitat HGB board beats the LSTM board, the manuscript should
frame the LSTM as the realtime readout model and the pressure-habitat HGB as
the stronger phase-cut board scorer.

## Claim Boundary

This protocol supports phase-cut RQ4 board comparison only.  It does not
validate a separately supervised 10-second streaming deployment, a calibrated
final placement head, or a threat-to-others score.
