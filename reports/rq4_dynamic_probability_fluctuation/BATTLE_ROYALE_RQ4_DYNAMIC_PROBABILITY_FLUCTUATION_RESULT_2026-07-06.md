# Battle Royale RQ4 Dynamic Probability Fluctuation Result

Date: 2026-07-06
Class: `ExperimentResult`
Lifecycle: `analysis_complete_bounded`

## Purpose

This artifact reframes the existing phase-wise hazard predictions as a
dynamic probability model.  It is intended to show what Sagong-style
phase score prediction does not directly show: the probability assigned
to each team changes as the match progresses.

## Model Definition

```text
lambda_i(k) = predicted next-phase elimination hazard
S_i(k)      = product_{j <= k} (1 - lambda_i(j))
pi_i(k)     = S_i(k) / sum_j S_j(k) within the same match and cut phase
```

`pi_i(k)` is a match-local dynamic win-probability proxy, not a fully
calibrated bookmaker-style win probability.

## Source

- run root: `/Users/lunecid/project/PUBG/model_stgcn/checkpoints/br_rq4_sequence_model_zoo_required_v1`
- model variant requested: `snapshot_lstm_no_graph`

## Aggregate

| model_variant | cohorts | median_hazard_auc | mean_hazard_auc | median_latest_top5_auc | median_latest_top10_auc | median_latest_rank_spearman | median_team_win_prob_volatility | mean_leader_changes_per_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| snapshot_lstm_no_graph | 13 | 0.7441 | 0.7379 | 0.9290 | 0.9033 | 0.7914 | 0.0296 | 2.5292 |

## Example Figure

- `/Users/lunecid/project/PUBG/reports/rq4_dynamic_probability_fluctuation/dynamic_probability_example_match.png`
- selected example: cohort `erangel_tpp_squad`, model `snapshot_lstm_no_graph`, match `10691322-1ab3-42be-b896-c9c76f3e10f7`

## Supported Claim

```text
The proposed survival formulation can be rendered as a dynamic probability
trajectory: at each observed phase cut, the model updates next-phase
elimination hazard, cumulative survival, and match-local win-probability
share for every still-at-risk team.
```

This directly separates the thesis from a static phase-score baseline: the
output is not only an eventual survival/win label, but a changing
probability curve over the course of the match.

## Boundary

Do not claim strict 10-second real-time deployment yet.  The current
artifact is dynamic at phase-cut resolution because the stored prediction
rows are phase-cut readouts.  A 10-second streaming version requires an
additional prefix-readout evaluation pass over snapshot sequences.
