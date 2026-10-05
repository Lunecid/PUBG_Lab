# RQ4 Board-Baseline Comparison Result

Date: 2026-07-06
Class: `ExperimentResult`
Lifecycle: `analysis_complete_bounded`

## Purpose

This report compares the phase-transition risk/advantage board against the frozen
random, health/alive-count, zone-distance, Sagong-style, official-raw, and
pressure-habitat baselines.

## Aggregate

| model_variant | board_label | cohorts | phase_boards | median_risk_top1_hit | median_risk_top3_hit | median_risk_mrr | median_adv_top_final_top5 | median_adv_top_final_top10 | median_adv_ndcg_top5_top10 | median_risk_delta_top1_hit | median_adv_delta_top_final_top10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pressure_habitat_hgb_board | Pressure-habitat HGB | 13 | 10616 | 0.7188 | 0.9359 | 0.8296 | 0.7834 | 0.9514 | 0.9419 | 0.6172 | 0.9713 |
| lstm_realtime_board | LSTM realtime board | 13 | 10616 | 0.7004 | 0.9172 | 0.8121 | 0.7863 | 0.9490 | 0.9389 | 0.6254 | 0.9722 |
| official_raw_hgb_board | Official-raw HGB | 13 | 10616 | 0.6952 | 0.9252 | 0.8113 | 0.7834 | 0.9473 | 0.9422 | 0.6032 | 0.9657 |
| health_alive_count_board | Health/alive-count | 13 | 10616 | 0.5094 | 0.8349 | 0.6853 | 0.6824 | 0.9184 | 0.8978 | 0.4687 | 0.9446 |
| sagong_prior_score_hgb_board | Sagong-style adapted HGB | 13 | 10616 | 0.4897 | 0.8385 | 0.6756 | 0.7043 | 0.9042 | 0.9150 | 0.4152 | 0.9497 |
| zone_distance_board | Zone-distance | 13 | 10616 | 0.4240 | 0.7599 | 0.6117 | 0.6570 | 0.8926 | 0.8799 | 0.4588 | 0.9461 |
| random_alive_board | Random alive | 13 | 10616 | 0.3868 | 0.7564 | 0.6008 | 0.6261 | 0.8742 | 0.8717 | 0.4150 | 0.9314 |

## Sign Count

Positive values mean the LSTM realtime board is better than the named baseline.

| baseline_label | metric | cohorts_compared | lstm_wins | ties | lstm_losses | median_lstm_minus_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| Health/alive-count | risk_top1_hit | 13 | 13 | 0 | 0 | 0.2205 |
| Health/alive-count | risk_top3_hit | 13 | 13 | 0 | 0 | 0.0859 |
| Health/alive-count | risk_top5_hit | 13 | 13 | 0 | 0 | 0.0380 |
| Health/alive-count | risk_mrr | 13 | 13 | 0 | 0 | 0.1397 |
| Health/alive-count | adv_top_final_top5 | 13 | 12 | 0 | 1 | 0.0552 |
| Health/alive-count | adv_top_final_top10 | 13 | 12 | 0 | 1 | 0.0243 |
| Health/alive-count | adv_top5_contains_top10 | 13 | 6 | 5 | 2 | 0.0000 |
| Health/alive-count | adv_ndcg_top5_top10 | 13 | 13 | 0 | 0 | 0.0243 |
| Health/alive-count | risk_delta_top1_hit | 13 | 13 | 0 | 0 | 0.1848 |
| Health/alive-count | risk_delta_mrr | 13 | 13 | 0 | 0 | 0.1182 |
| Health/alive-count | adv_delta_top_final_top10 | 13 | 13 | 0 | 0 | 0.0137 |
| Health/alive-count | adv_delta_ndcg_top5_top10 | 13 | 13 | 0 | 0 | 0.0245 |
| Official-raw HGB | risk_top1_hit | 13 | 10 | 1 | 2 | 0.0182 |
| Official-raw HGB | risk_top3_hit | 13 | 9 | 0 | 4 | 0.0047 |
| Official-raw HGB | risk_top5_hit | 13 | 7 | 0 | 6 | 0.0056 |
| Official-raw HGB | risk_mrr | 13 | 9 | 0 | 4 | 0.0095 |
| Official-raw HGB | adv_top_final_top5 | 13 | 5 | 0 | 8 | -0.0072 |
| Official-raw HGB | adv_top_final_top10 | 13 | 5 | 1 | 7 | -0.0048 |
| Official-raw HGB | adv_top5_contains_top10 | 13 | 3 | 8 | 2 | 0.0000 |
| Official-raw HGB | adv_ndcg_top5_top10 | 13 | 4 | 0 | 9 | -0.0062 |
| Official-raw HGB | risk_delta_top1_hit | 13 | 11 | 0 | 2 | 0.0430 |
| Official-raw HGB | risk_delta_mrr | 13 | 11 | 0 | 2 | 0.0278 |
| Official-raw HGB | adv_delta_top_final_top10 | 13 | 2 | 3 | 8 | -0.0032 |
| Official-raw HGB | adv_delta_ndcg_top5_top10 | 13 | 4 | 1 | 8 | -0.0026 |
| Pressure-habitat HGB | risk_top1_hit | 13 | 5 | 0 | 8 | -0.0084 |
| Pressure-habitat HGB | risk_top3_hit | 13 | 2 | 1 | 10 | -0.0138 |
| Pressure-habitat HGB | risk_top5_hit | 13 | 2 | 0 | 11 | -0.0098 |
| Pressure-habitat HGB | risk_mrr | 13 | 4 | 0 | 9 | -0.0078 |
| Pressure-habitat HGB | adv_top_final_top5 | 13 | 3 | 0 | 10 | -0.0168 |
| Pressure-habitat HGB | adv_top_final_top10 | 13 | 3 | 1 | 9 | -0.0057 |
| Pressure-habitat HGB | adv_top5_contains_top10 | 13 | 5 | 6 | 2 | 0.0000 |
| Pressure-habitat HGB | adv_ndcg_top5_top10 | 13 | 2 | 0 | 11 | -0.0064 |
| Pressure-habitat HGB | risk_delta_top1_hit | 13 | 10 | 1 | 2 | 0.0136 |
| Pressure-habitat HGB | risk_delta_mrr | 13 | 8 | 0 | 5 | 0.0058 |
| Pressure-habitat HGB | adv_delta_top_final_top10 | 13 | 2 | 2 | 9 | -0.0081 |
| Pressure-habitat HGB | adv_delta_ndcg_top5_top10 | 13 | 1 | 1 | 11 | -0.0047 |
| Random alive | risk_top1_hit | 13 | 13 | 0 | 0 | 0.3341 |
| Random alive | risk_top3_hit | 13 | 13 | 0 | 0 | 0.1686 |
| Random alive | risk_top5_hit | 13 | 13 | 0 | 0 | 0.0669 |
| Random alive | risk_mrr | 13 | 13 | 0 | 0 | 0.2315 |
| Random alive | adv_top_final_top5 | 13 | 13 | 0 | 0 | 0.1350 |
| Random alive | adv_top_final_top10 | 13 | 13 | 0 | 0 | 0.0598 |
| Random alive | adv_top5_contains_top10 | 13 | 8 | 4 | 1 | 0.0023 |
| Random alive | adv_ndcg_top5_top10 | 13 | 13 | 0 | 0 | 0.0542 |
| Random alive | risk_delta_top1_hit | 13 | 13 | 0 | 0 | 0.2390 |
| Random alive | risk_delta_mrr | 13 | 13 | 0 | 0 | 0.1638 |
| Random alive | adv_delta_top_final_top10 | 13 | 11 | 2 | 0 | 0.0389 |
| Random alive | adv_delta_ndcg_top5_top10 | 13 | 13 | 0 | 0 | 0.0345 |
| Sagong-style adapted HGB | risk_top1_hit | 13 | 13 | 0 | 0 | 0.2270 |
| Sagong-style adapted HGB | risk_top3_hit | 13 | 13 | 0 | 0 | 0.0802 |
| Sagong-style adapted HGB | risk_top5_hit | 13 | 12 | 0 | 1 | 0.0247 |
| Sagong-style adapted HGB | risk_mrr | 13 | 13 | 0 | 0 | 0.1442 |
| Sagong-style adapted HGB | adv_top_final_top5 | 13 | 13 | 0 | 0 | 0.0578 |
| Sagong-style adapted HGB | adv_top_final_top10 | 13 | 12 | 0 | 1 | 0.0279 |
| Sagong-style adapted HGB | adv_top5_contains_top10 | 13 | 5 | 6 | 2 | 0.0000 |
| Sagong-style adapted HGB | adv_ndcg_top5_top10 | 13 | 13 | 0 | 0 | 0.0208 |
| Sagong-style adapted HGB | risk_delta_top1_hit | 13 | 13 | 0 | 0 | 0.2155 |
| Sagong-style adapted HGB | risk_delta_mrr | 13 | 13 | 0 | 0 | 0.1465 |
| Sagong-style adapted HGB | adv_delta_top_final_top10 | 13 | 10 | 1 | 2 | 0.0094 |
| Sagong-style adapted HGB | adv_delta_ndcg_top5_top10 | 13 | 12 | 1 | 0 | 0.0197 |
| Zone-distance | risk_top1_hit | 13 | 13 | 0 | 0 | 0.2770 |
| Zone-distance | risk_top3_hit | 13 | 13 | 0 | 0 | 0.1543 |
| Zone-distance | risk_top5_hit | 13 | 13 | 0 | 0 | 0.0779 |
| Zone-distance | risk_mrr | 13 | 13 | 0 | 0 | 0.1963 |
| Zone-distance | adv_top_final_top5 | 13 | 13 | 0 | 0 | 0.1006 |
| Zone-distance | adv_top_final_top10 | 13 | 12 | 0 | 1 | 0.0517 |
| Zone-distance | adv_top5_contains_top10 | 13 | 7 | 5 | 1 | 0.0020 |
| Zone-distance | adv_ndcg_top5_top10 | 13 | 13 | 0 | 0 | 0.0485 |
| Zone-distance | risk_delta_top1_hit | 13 | 13 | 0 | 0 | 0.1810 |
| Zone-distance | risk_delta_mrr | 13 | 13 | 0 | 0 | 0.1198 |
| Zone-distance | adv_delta_top_final_top10 | 13 | 11 | 1 | 1 | 0.0245 |
| Zone-distance | adv_delta_ndcg_top5_top10 | 13 | 12 | 1 | 0 | 0.0285 |

## Cohort Summary

| cohort_slug | model_variant | phase_boards | risk_top1_hit | risk_top3_hit | risk_mrr | adv_top_final_top10 | adv_ndcg_top5_top10 | risk_delta_top1_hit | adv_delta_top_final_top10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| erangel_fpp_squad | health_alive_count_board | 242 | 0.5094 | 0.8349 | 0.6853 | 0.9215 | 0.8939 | 0.5349 | 0.9296 |
| erangel_fpp_squad | lstm_realtime_board | 242 | 0.6934 | 0.9104 | 0.8078 | 0.9050 | 0.9173 | 0.6279 | 0.9397 |
| erangel_fpp_squad | official_raw_hgb_board | 242 | 0.5943 | 0.8915 | 0.7513 | 0.9008 | 0.9245 | 0.5581 | 0.9548 |
| erangel_fpp_squad | pressure_habitat_hgb_board | 242 | 0.7123 | 0.9387 | 0.8254 | 0.9298 | 0.9325 | 0.6163 | 0.9598 |
| erangel_fpp_squad | random_alive_board | 242 | 0.3868 | 0.7358 | 0.5919 | 0.8595 | 0.8693 | 0.4419 | 0.9397 |
| erangel_fpp_squad | sagong_prior_score_hgb_board | 242 | 0.5000 | 0.8019 | 0.6704 | 0.8967 | 0.9008 | 0.4593 | 0.9497 |
| erangel_fpp_squad | zone_distance_board | 242 | 0.5000 | 0.7783 | 0.6637 | 0.8926 | 0.8799 | 0.5523 | 0.9497 |
| erangel_tpp_duo | health_alive_count_board | 417 | 0.5256 | 0.8667 | 0.6994 | 0.7962 | 0.7755 | 0.4801 | 0.8381 |
| erangel_tpp_duo | lstm_realtime_board | 417 | 0.7462 | 0.9538 | 0.8519 | 0.8345 | 0.8100 | 0.6820 | 0.8722 |
| erangel_tpp_duo | official_raw_hgb_board | 417 | 0.7205 | 0.9333 | 0.8302 | 0.8681 | 0.8319 | 0.6239 | 0.8892 |
| erangel_tpp_duo | pressure_habitat_hgb_board | 417 | 0.7513 | 0.9359 | 0.8504 | 0.8225 | 0.8153 | 0.6483 | 0.8892 |
| erangel_tpp_duo | random_alive_board | 417 | 0.4128 | 0.7846 | 0.6191 | 0.7506 | 0.7255 | 0.4220 | 0.8239 |
| erangel_tpp_duo | sagong_prior_score_hgb_board | 417 | 0.4846 | 0.8385 | 0.6729 | 0.7962 | 0.7892 | 0.4434 | 0.8210 |
| erangel_tpp_duo | zone_distance_board | 417 | 0.4154 | 0.7231 | 0.6026 | 0.7530 | 0.7511 | 0.4495 | 0.8239 |
| erangel_tpp_solo | health_alive_count_board | 504 | 0.5916 | 0.8821 | 0.7454 | 0.6567 | 0.6510 | 0.6304 | 0.7288 |
| erangel_tpp_solo | lstm_realtime_board | 504 | 0.8358 | 0.9705 | 0.9015 | 0.7798 | 0.7466 | 0.8152 | 0.8514 |
| erangel_tpp_solo | official_raw_hgb_board | 504 | 0.8147 | 0.9642 | 0.8896 | 0.8115 | 0.7953 | 0.7241 | 0.8679 |
| erangel_tpp_solo | pressure_habitat_hgb_board | 504 | 0.8442 | 0.9768 | 0.9093 | 0.8135 | 0.7737 | 0.8025 | 0.8632 |
| erangel_tpp_solo | random_alive_board | 504 | 0.4211 | 0.8084 | 0.6300 | 0.6131 | 0.6069 | 0.4658 | 0.7099 |
| erangel_tpp_solo | sagong_prior_score_hgb_board | 504 | 0.5579 | 0.9011 | 0.7350 | 0.7778 | 0.7440 | 0.4152 | 0.8420 |
| erangel_tpp_solo | zone_distance_board | 504 | 0.4842 | 0.8211 | 0.6660 | 0.6250 | 0.6286 | 0.5747 | 0.7453 |
| erangel_tpp_squad | health_alive_count_board | 3490 | 0.4972 | 0.8275 | 0.6768 | 0.9049 | 0.8978 | 0.4723 | 0.9446 |
| erangel_tpp_squad | lstm_realtime_board | 3490 | 0.7292 | 0.9189 | 0.8294 | 0.9553 | 0.9399 | 0.6667 | 0.9728 |
| erangel_tpp_squad | official_raw_hgb_board | 3490 | 0.7054 | 0.9252 | 0.8199 | 0.9605 | 0.9422 | 0.6190 | 0.9742 |
| erangel_tpp_squad | pressure_habitat_hgb_board | 3490 | 0.7188 | 0.9393 | 0.8315 | 0.9587 | 0.9419 | 0.6256 | 0.9753 |
| erangel_tpp_squad | random_alive_board | 3490 | 0.3951 | 0.7564 | 0.6008 | 0.8742 | 0.8717 | 0.4150 | 0.9279 |
| erangel_tpp_squad | sagong_prior_score_hgb_board | 3490 | 0.5119 | 0.8472 | 0.6904 | 0.9289 | 0.9150 | 0.3984 | 0.9578 |
| erangel_tpp_squad | zone_distance_board | 3490 | 0.4080 | 0.7599 | 0.6062 | 0.8705 | 0.8713 | 0.4588 | 0.9425 |
| karakin_tpp_squad | health_alive_count_board | 179 | 0.5215 | 0.8528 | 0.7030 | 0.9888 | 0.9685 | 0.4538 | 0.9863 |
| karakin_tpp_squad | lstm_realtime_board | 179 | 0.7485 | 0.9387 | 0.8427 | 1.0000 | 0.9858 | 0.6231 | 1.0000 |
| karakin_tpp_squad | official_raw_hgb_board | 179 | 0.6626 | 0.9325 | 0.8001 | 0.9888 | 0.9855 | 0.5692 | 1.0000 |
| karakin_tpp_squad | pressure_habitat_hgb_board | 179 | 0.7669 | 0.9632 | 0.8598 | 1.0000 | 0.9922 | 0.6923 | 1.0000 |
| karakin_tpp_squad | random_alive_board | 179 | 0.3558 | 0.8896 | 0.6113 | 0.9609 | 0.9495 | 0.4538 | 0.9863 |
| karakin_tpp_squad | sagong_prior_score_hgb_board | 179 | 0.5215 | 0.8834 | 0.7066 | 0.9721 | 0.9651 | 0.5077 | 0.9932 |
| karakin_tpp_squad | zone_distance_board | 179 | 0.4847 | 0.8405 | 0.6732 | 0.9441 | 0.9360 | 0.4615 | 0.9863 |
| miramar_desert_tpp_squad | health_alive_count_board | 951 | 0.4594 | 0.8186 | 0.6518 | 0.8812 | 0.8718 | 0.4288 | 0.9235 |
| miramar_desert_tpp_squad | lstm_realtime_board | 951 | 0.6832 | 0.9034 | 0.7978 | 0.9180 | 0.9127 | 0.6254 | 0.9448 |
| miramar_desert_tpp_squad | official_raw_hgb_board | 951 | 0.6572 | 0.8987 | 0.7858 | 0.9422 | 0.9236 | 0.5869 | 0.9636 |
| miramar_desert_tpp_squad | pressure_habitat_hgb_board | 951 | 0.6761 | 0.9081 | 0.7988 | 0.9401 | 0.9235 | 0.6054 | 0.9624 |
| miramar_desert_tpp_squad | random_alive_board | 951 | 0.3781 | 0.7362 | 0.5789 | 0.8475 | 0.8496 | 0.3932 | 0.9059 |
| miramar_desert_tpp_squad | sagong_prior_score_hgb_board | 951 | 0.4335 | 0.7939 | 0.6330 | 0.8812 | 0.8923 | 0.3832 | 0.9147 |
| miramar_desert_tpp_squad | zone_distance_board | 951 | 0.4087 | 0.7491 | 0.6043 | 0.8749 | 0.8642 | 0.4772 | 0.9134 |
| paramo_tpp_squad | health_alive_count_board | 175 | 0.5448 | 0.9103 | 0.7181 | 0.9829 | 0.9854 | 0.5310 | 0.9930 |
| paramo_tpp_squad | lstm_realtime_board | 175 | 0.6897 | 0.9172 | 0.8121 | 0.9943 | 0.9973 | 0.6372 | 1.0000 |
| paramo_tpp_squad | official_raw_hgb_board | 175 | 0.6966 | 0.9448 | 0.8185 | 0.9886 | 0.9952 | 0.6991 | 1.0000 |
| paramo_tpp_squad | pressure_habitat_hgb_board | 175 | 0.7241 | 0.9310 | 0.8310 | 1.0000 | 0.9972 | 0.6372 | 1.0000 |
| paramo_tpp_squad | random_alive_board | 175 | 0.4000 | 0.8276 | 0.6209 | 0.9829 | 0.9851 | 0.5044 | 1.0000 |
| paramo_tpp_squad | sagong_prior_score_hgb_board | 175 | 0.4897 | 0.8966 | 0.6951 | 1.0000 | 0.9958 | 0.4956 | 1.0000 |
| paramo_tpp_squad | zone_distance_board | 175 | 0.4759 | 0.8414 | 0.6733 | 1.0000 | 0.9867 | 0.4690 | 1.0000 |
| rondo_tpp_squad | health_alive_count_board | 626 | 0.4944 | 0.8216 | 0.6748 | 0.9153 | 0.9145 | 0.4545 | 0.9522 |
| rondo_tpp_squad | lstm_realtime_board | 626 | 0.6952 | 0.9071 | 0.8061 | 0.9425 | 0.9389 | 0.6250 | 0.9751 |
| rondo_tpp_squad | official_raw_hgb_board | 626 | 0.6952 | 0.9052 | 0.8065 | 0.9473 | 0.9451 | 0.5841 | 0.9751 |
| rondo_tpp_squad | pressure_habitat_hgb_board | 626 | 0.6599 | 0.9219 | 0.7943 | 0.9569 | 0.9474 | 0.5932 | 0.9713 |
| rondo_tpp_squad | random_alive_board | 626 | 0.3550 | 0.7193 | 0.5661 | 0.8946 | 0.8901 | 0.3773 | 0.9465 |
| rondo_tpp_squad | sagong_prior_score_hgb_board | 626 | 0.4461 | 0.8048 | 0.6460 | 0.9042 | 0.9180 | 0.3750 | 0.9369 |
| rondo_tpp_squad | zone_distance_board | 626 | 0.4182 | 0.7491 | 0.6109 | 0.9169 | 0.9109 | 0.4545 | 0.9579 |
| sanhok_tpp_squad | health_alive_count_board | 585 | 0.5437 | 0.8536 | 0.7064 | 0.9470 | 0.9303 | 0.4687 | 0.9837 |
| sanhok_tpp_squad | lstm_realtime_board | 585 | 0.7110 | 0.9163 | 0.8198 | 0.9607 | 0.9655 | 0.6218 | 0.9898 |
| sanhok_tpp_squad | official_raw_hgb_board | 585 | 0.7053 | 0.9259 | 0.8183 | 0.9880 | 0.9749 | 0.6032 | 0.9959 |
| sanhok_tpp_squad | pressure_habitat_hgb_board | 585 | 0.7490 | 0.9411 | 0.8499 | 0.9915 | 0.9799 | 0.6172 | 0.9980 |
| sanhok_tpp_squad | random_alive_board | 585 | 0.4030 | 0.7452 | 0.6052 | 0.9009 | 0.9079 | 0.3828 | 0.9633 |
| sanhok_tpp_squad | sagong_prior_score_hgb_board | 585 | 0.5038 | 0.8688 | 0.6923 | 0.9538 | 0.9452 | 0.4200 | 0.9857 |
| sanhok_tpp_squad | zone_distance_board | 585 | 0.4240 | 0.7738 | 0.6229 | 0.9162 | 0.9110 | 0.4408 | 0.9674 |
| taego_fpp_squad | health_alive_count_board | 247 | 0.5273 | 0.8636 | 0.7093 | 0.9393 | 0.9314 | 0.5028 | 0.9657 |
| taego_fpp_squad | lstm_realtime_board | 247 | 0.6682 | 0.9409 | 0.8038 | 0.9636 | 0.9534 | 0.5932 | 0.9706 |
| taego_fpp_squad | official_raw_hgb_board | 247 | 0.6773 | 0.9136 | 0.8039 | 0.9555 | 0.9481 | 0.6215 | 0.9657 |
| taego_fpp_squad | pressure_habitat_hgb_board | 247 | 0.7182 | 0.9409 | 0.8296 | 0.9514 | 0.9520 | 0.6045 | 0.9804 |
| taego_fpp_squad | random_alive_board | 247 | 0.3773 | 0.7636 | 0.5936 | 0.8866 | 0.9093 | 0.4181 | 0.9314 |
| taego_fpp_squad | sagong_prior_score_hgb_board | 247 | 0.5182 | 0.8182 | 0.6905 | 0.9514 | 0.9386 | 0.4407 | 0.9853 |
| taego_fpp_squad | zone_distance_board | 247 | 0.4955 | 0.8318 | 0.6763 | 0.9069 | 0.9135 | 0.4689 | 0.9461 |
| taego_tpp_duo | health_alive_count_board | 370 | 0.5072 | 0.8309 | 0.6812 | 0.8243 | 0.8247 | 0.4315 | 0.8946 |
| taego_tpp_duo | lstm_realtime_board | 370 | 0.7507 | 0.9284 | 0.8418 | 0.8405 | 0.8276 | 0.6507 | 0.9073 |
| taego_tpp_duo | official_raw_hgb_board | 370 | 0.7335 | 0.9026 | 0.8246 | 0.8486 | 0.8424 | 0.5822 | 0.9105 |
| taego_tpp_duo | pressure_habitat_hgb_board | 370 | 0.7221 | 0.9226 | 0.8269 | 0.8514 | 0.8459 | 0.6336 | 0.9137 |
| taego_tpp_duo | random_alive_board | 370 | 0.3983 | 0.7851 | 0.6085 | 0.8081 | 0.7746 | 0.3870 | 0.8435 |
| taego_tpp_duo | sagong_prior_score_hgb_board | 370 | 0.4785 | 0.8481 | 0.6756 | 0.7919 | 0.7953 | 0.3836 | 0.8626 |
| taego_tpp_duo | zone_distance_board | 370 | 0.4040 | 0.7278 | 0.5944 | 0.7946 | 0.7901 | 0.4315 | 0.8435 |
| taego_tpp_squad | health_alive_count_board | 2536 | 0.4938 | 0.8123 | 0.6711 | 0.9464 | 0.9263 | 0.4233 | 0.9623 |
| taego_tpp_squad | lstm_realtime_board | 2536 | 0.6940 | 0.9048 | 0.8065 | 0.9550 | 0.9485 | 0.6246 | 0.9722 |
| taego_tpp_squad | official_raw_hgb_board | 2536 | 0.6757 | 0.9044 | 0.7973 | 0.9550 | 0.9476 | 0.5816 | 0.9769 |
| taego_tpp_squad | pressure_habitat_hgb_board | 2536 | 0.7077 | 0.9257 | 0.8211 | 0.9570 | 0.9497 | 0.6110 | 0.9807 |
| taego_tpp_squad | random_alive_board | 2536 | 0.3599 | 0.7362 | 0.5741 | 0.8971 | 0.8943 | 0.3879 | 0.9392 |
| taego_tpp_squad | sagong_prior_score_hgb_board | 2536 | 0.4693 | 0.8332 | 0.6623 | 0.9255 | 0.9214 | 0.4091 | 0.9628 |
| taego_tpp_squad | zone_distance_board | 2536 | 0.3972 | 0.7451 | 0.5955 | 0.9034 | 0.9042 | 0.4195 | 0.9543 |
| vikendi_tpp_squad | health_alive_count_board | 294 | 0.4585 | 0.7906 | 0.6439 | 0.9184 | 0.8820 | 0.4113 | 0.9388 |
| vikendi_tpp_squad | lstm_realtime_board | 294 | 0.7004 | 0.9134 | 0.8080 | 0.9490 | 0.9213 | 0.6147 | 0.9837 |
| vikendi_tpp_squad | official_raw_hgb_board | 294 | 0.6931 | 0.9314 | 0.8113 | 0.9184 | 0.9261 | 0.6104 | 0.9510 |
| vikendi_tpp_squad | pressure_habitat_hgb_board | 294 | 0.6679 | 0.9242 | 0.7973 | 0.9456 | 0.9256 | 0.5758 | 0.9633 |
| vikendi_tpp_squad | random_alive_board | 294 | 0.3430 | 0.7401 | 0.5621 | 0.8401 | 0.8601 | 0.3636 | 0.8898 |
| vikendi_tpp_squad | sagong_prior_score_hgb_board | 294 | 0.4729 | 0.8267 | 0.6625 | 0.8946 | 0.8900 | 0.4113 | 0.9184 |
| vikendi_tpp_squad | zone_distance_board | 294 | 0.4296 | 0.7545 | 0.6117 | 0.8503 | 0.8537 | 0.4113 | 0.9224 |

## Example Board Figure

- Path: `/Users/lunecid/project/PUBG/reports/rq4_board_baseline_comparison/rq4_board_baseline_example_board.png`
- Cohort: `erangel_fpp_squad`
- Match: `0771a12f-d37f-4525-81aa-edb69245ecac`
- Phase: `1`

## Claim Boundary

Pressure-habitat HGB is stronger on median risk top-1, so the manuscript should frame the LSTM board as the realtime readout form rather than the globally strongest board scorer.

Do not claim that this validates a separate 10-second streaming deployment.
The comparison uses phase-cut board rows and saved phase-hazard predictions.

## Manifest

```json
{
  "lstm_trajectories": "/Users/lunecid/project/PUBG/reports/rq4_dynamic_probability_fluctuation/dynamic_probability_trajectories.csv",
  "prior_root": "/Users/lunecid/project/PUBG/model_stgcn/checkpoints/br_rq4_prior_model_zoo_required_v1",
  "feature_csv": "/Users/lunecid/project/PUBG/reports/rq4_board_baseline_comparison/board_baseline_feature_rows.csv",
  "top_k": 5,
  "random_seed": 20260706,
  "variants": {
    "random_alive_board": "Random alive",
    "health_alive_count_board": "Health/alive-count",
    "zone_distance_board": "Zone-distance",
    "sagong_prior_score_hgb_board": "Sagong-style adapted HGB",
    "official_raw_hgb_board": "Official-raw HGB",
    "pressure_habitat_hgb_board": "Pressure-habitat HGB",
    "lstm_realtime_board": "LSTM realtime board"
  },
  "outputs": {
    "score_rows": "/Users/lunecid/project/PUBG/reports/rq4_board_baseline_comparison/rq4_board_baseline_score_rows.csv",
    "board_rows": "/Users/lunecid/project/PUBG/reports/rq4_board_baseline_comparison/rq4_board_baseline_board_rows.csv",
    "by_cohort": "/Users/lunecid/project/PUBG/reports/rq4_board_baseline_comparison/rq4_board_baseline_by_cohort.csv",
    "aggregate": "/Users/lunecid/project/PUBG/reports/rq4_board_baseline_comparison/rq4_board_baseline_aggregate.csv",
    "sign_count": "/Users/lunecid/project/PUBG/reports/rq4_board_baseline_comparison/rq4_board_baseline_sign_count.csv",
    "example_figure": "/Users/lunecid/project/PUBG/reports/rq4_board_baseline_comparison/rq4_board_baseline_example_board.png",
    "report": "/Users/lunecid/project/PUBG/reports/rq4_board_baseline_comparison/RQ4_BOARD_BASELINE_COMPARISON_RESULT_2026-07-06.md"
  },
  "coverage": {
    "score_rows_raw": 735176,
    "score_rows": 727020,
    "board_rows": 74312,
    "cohorts": 13,
    "matches": 1789,
    "common_universe": {
      "input_rows": 735176,
      "output_rows": 727020,
      "variants": [
        "lstm_realtime_board",
        "health_alive_count_board",
        "official_raw_hgb_board",
        "pressure_habitat_hgb_board",
        "random_alive_board",
        "sagong_prior_score_hgb_board",
        "zone_distance_board"
      ],
      "common_row_keys": 103860,
      "input_board_keys": 10738,
      "output_board_keys": 10616,
      "dropped_rows_by_variant": {
        "lstm_realtime_board": 1631,
        "health_alive_count_board": 260,
        "official_raw_hgb_board": 1915,
        "pressure_habitat_hgb_board": 1915,
        "random_alive_board": 260,
        "sagong_prior_score_hgb_board": 1915,
        "zone_distance_board": 260
      },
      "contract": "all variants evaluated on intersected cohort/match/phase/team rows"
    }
  }
}
```
