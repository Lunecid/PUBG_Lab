# RQ4 Phase-Transition Risk/Advantage Board Result

Date: 2026-07-06
Class: `ExperimentResult`
Lifecycle: `analysis_complete_bounded`

## Purpose

This artifact evaluates the model as a phase-transition board rather than
only as row-level hazard predictions.

## Board Outputs

| Board | Ranking variable | Question |
| --- | --- | --- |
| risk board | `hazard_prob` descending | Which teams are most likely to be eliminated next? |
| advantage board | `dynamic_win_prob` descending | Which teams are in the strongest survival position? |

## Aggregate

| model_variant | cohorts | phase_boards | median_risk_top1_hit | median_risk_top3_hit | median_risk_mrr | median_adv_top_final_top5 | median_adv_top_final_top10 | median_adv_top5_contains_top10 | median_adv_ndcg_top5_top10 | mean_leader_change_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| snapshot_lstm_no_graph | 13 | 10616 | 0.7004 | 0.9172 | 0.8121 | 0.7768 | 0.9490 | 1.0000 | 0.9381 | 0.4214 |

## Cohort Summary

| cohort_slug | model_variant | phase_boards | matches | mean_at_risk_rows | mean_next_phase_eliminations | risk_top1_hit | risk_top3_hit | risk_top5_hit | risk_mrr | risk_precision_at_3 | risk_precision_at_5 | adv_top_final_top5 | adv_top_final_top10 | adv_top5_contains_top5 | adv_top5_contains_top10 | adv_top5_top10_precision | adv_ndcg_top5_top10 | leader_changed_from_prev_phase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| erangel_fpp_squad | snapshot_lstm_no_graph | 242 | 43 | 9.1074 | 2.8471 | 0.6934 | 0.9104 | 0.9623 | 0.8078 | 0.5007 | 0.4393 | 0.6983 | 0.9132 | 0.9793 | 0.9959 | 0.9198 | 0.9185 | 0.4504 |
| erangel_tpp_duo | snapshot_lstm_no_graph | 417 | 65 | 13.8201 | 4.5564 | 0.7462 | 0.9538 | 0.9872 | 0.8519 | 0.5631 | 0.5002 | 0.6427 | 0.8393 | 0.9305 | 0.9952 | 0.8048 | 0.8102 | 0.4916 |
| erangel_tpp_solo | snapshot_lstm_no_graph | 504 | 80 | 23.9385 | 8.5754 | 0.8358 | 0.9705 | 0.9874 | 0.9015 | 0.6601 | 0.6050 | 0.6607 | 0.7778 | 0.8889 | 0.9583 | 0.7361 | 0.7462 | 0.3929 |
| erangel_tpp_squad | snapshot_lstm_no_graph | 3490 | 620 | 9.0625 | 2.8711 | 0.7292 | 0.9189 | 0.9693 | 0.8294 | 0.5150 | 0.4495 | 0.8006 | 0.9539 | 0.9954 | 1.0000 | 0.9362 | 0.9404 | 0.3745 |
| karakin_tpp_squad | snapshot_lstm_no_graph | 179 | 33 | 6.4749 | 2.3855 | 0.7485 | 0.9387 | 0.9693 | 0.8427 | 0.5000 | 0.4426 | 0.8939 | 1.0000 | 1.0000 | 1.0000 | 0.9832 | 0.9858 | 0.3296 |
| miramar_desert_tpp_squad | snapshot_lstm_no_graph | 951 | 154 | 9.5352 | 2.7571 | 0.6832 | 0.9034 | 0.9529 | 0.7978 | 0.4711 | 0.4089 | 0.7129 | 0.9117 | 0.9811 | 1.0000 | 0.9096 | 0.9107 | 0.4543 |
| paramo_tpp_squad | snapshot_lstm_no_graph | 175 | 33 | 5.4171 | 1.7886 | 0.6897 | 0.9172 | 0.9862 | 0.8121 | 0.4371 | 0.3855 | 0.9257 | 0.9943 | 1.0000 | 1.0000 | 0.9977 | 0.9973 | 0.2914 |
| rondo_tpp_squad | snapshot_lstm_no_graph | 626 | 103 | 8.3067 | 2.3978 | 0.6952 | 0.9071 | 0.9647 | 0.8061 | 0.4542 | 0.3882 | 0.8019 | 0.9393 | 0.9888 | 1.0000 | 0.9371 | 0.9381 | 0.4233 |
| sanhok_tpp_squad | snapshot_lstm_no_graph | 585 | 94 | 7.9265 | 2.5692 | 0.7110 | 0.9163 | 0.9753 | 0.8198 | 0.5060 | 0.4354 | 0.7880 | 0.9607 | 0.9983 | 1.0000 | 0.9658 | 0.9658 | 0.4103 |
| taego_fpp_squad | snapshot_lstm_no_graph | 247 | 43 | 8.0769 | 2.6923 | 0.6682 | 0.9409 | 0.9773 | 0.8038 | 0.4811 | 0.4301 | 0.8097 | 0.9595 | 0.9879 | 1.0000 | 0.9530 | 0.9545 | 0.4494 |
| taego_tpp_duo | snapshot_lstm_no_graph | 370 | 57 | 12.4595 | 4.0892 | 0.7507 | 0.9284 | 0.9742 | 0.8418 | 0.5626 | 0.4920 | 0.6432 | 0.8324 | 0.9676 | 0.9973 | 0.8265 | 0.8266 | 0.5054 |
| taego_tpp_squad | snapshot_lstm_no_graph | 2536 | 415 | 8.5982 | 2.5540 | 0.6940 | 0.9048 | 0.9595 | 0.8065 | 0.4727 | 0.4086 | 0.7768 | 0.9543 | 0.9897 | 0.9980 | 0.9466 | 0.9486 | 0.4294 |
| vikendi_tpp_squad | snapshot_lstm_no_graph | 294 | 49 | 9.4490 | 2.8503 | 0.7004 | 0.9134 | 0.9386 | 0.8080 | 0.5102 | 0.4340 | 0.7517 | 0.9490 | 0.9762 | 1.0000 | 0.9143 | 0.9228 | 0.4762 |

## Supported Claim

```text
The dynamic survival readout can be converted into phase-transition boards
that identify the highest next-phase elimination risk and the strongest
match-local survival advantage at each phase cut.
```

## Boundary

This board uses the existing phase-cut trajectory artifact.  It does not
validate a separate 10-second streaming metric.  The risk-board metrics
should be compared against explicit score, health, zone-distance, and
tabular baselines before being treated as a final manuscript result.

## Metric Notes

- Risk hit-rate metrics are averaged over boards with at least one next-phase elimination; boards with no next-phase elimination have undefined risk-hit targets.
- `risk_top5_hit`: at least one next-phase eliminated team appears in the top-5 risk list.
- `risk_mrr`: reciprocal rank of the first actually eliminated team in the risk board.
- `adv_top5_contains_top10`: at least one final top-10 team appears in the top-5 advantage list.
