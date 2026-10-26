# 패치 재학습 결과 확인표

아래 수치는 완료된 분석 CSV에서 직접 생성했다. 각 행은 선택한 설정의 3시드 지표 평균이다. 전체/사람 팀 평가를 구분한다. 그래프·모집단 대조는 CPU에서 다시 학습한 신경망 참조를 사용하므로 원 튜닝 표와 신경망 수치가 조금 다를 수 있다.

모든 신뢰구간은 경기 단위 paired bootstrap 2,000회로 계산한다. 학습 시드 모집단 전체의 불확실성을 추정하지 않는다. GBDT의 세 시드는 결정적인 적합으로 같은 예측을 낼 수 있다. 추가 대조는 탐색적이며 다중검정 보정된 확증적 결과로 읽지 않는다.

## 동일 시행 횟수 튜닝

| 코호트 | 평가 대상 | 조건 | AUC | 단계 내 AUC | 경기 내 AUC | NLL |
|---|---|---|---:|---:|---:|---:|
| erangel | all | gbdt | 0.8034 | 0.7743 | 0.7833 | 0.4518 |
| erangel | all | krafton | 0.8026 | 0.7783 | 0.7785 | 0.4527 |
| erangel | all | ptr | 0.8015 | 0.7777 | 0.7770 | 0.4538 |
| erangel | human | gbdt | 0.8036 | 0.7683 | 0.7724 | 0.4382 |
| erangel | human | krafton | 0.8031 | 0.7725 | 0.7681 | 0.4395 |
| erangel | human | ptr | 0.8019 | 0.7719 | 0.7670 | 0.4405 |
| taego | all | gbdt | 0.8049 | 0.7778 | 0.7887 | 0.4429 |
| taego | all | krafton | 0.8081 | 0.7831 | 0.7888 | 0.4422 |
| taego | all | ptr | 0.8088 | 0.7846 | 0.7912 | 0.4415 |
| taego | human | gbdt | 0.8048 | 0.7733 | 0.7763 | 0.4306 |
| taego | human | krafton | 0.8078 | 0.7786 | 0.7774 | 0.4306 |
| taego | human | ptr | 0.8083 | 0.7802 | 0.7794 | 0.4295 |

| 코호트 | 평가 대상 | 차이(앞−뒤) | 지표 | 추정치 [95% CI] |
|---|---|---|---|---:|
| erangel | all | ptr_minus_gbdt | auc | -0.0020 [-0.0039, +0.0001] |
| erangel | all | ptr_minus_gbdt | auc_within_match_phase | -0.0062 [-0.0097, -0.0029] |
| erangel | all | ptr_minus_krafton | auc | -0.0011 [-0.0023, +0.0001] |
| erangel | all | ptr_minus_krafton | auc_within_match_phase | -0.0015 [-0.0035, +0.0005] |
| erangel | all | krafton_minus_gbdt | auc | -0.0008 [-0.0028, +0.0012] |
| erangel | all | krafton_minus_gbdt | auc_within_match_phase | -0.0047 [-0.0079, -0.0016] |
| erangel | human | ptr_minus_gbdt | auc | -0.0017 [-0.0038, +0.0006] |
| erangel | human | ptr_minus_gbdt | auc_within_match_phase | -0.0054 [-0.0092, -0.0015] |
| erangel | human | ptr_minus_krafton | auc | -0.0012 [-0.0024, +0.0001] |
| erangel | human | ptr_minus_krafton | auc_within_match_phase | -0.0011 [-0.0032, +0.0011] |
| erangel | human | krafton_minus_gbdt | auc | -0.0005 [-0.0026, +0.0016] |
| erangel | human | krafton_minus_gbdt | auc_within_match_phase | -0.0043 [-0.0081, -0.0006] |
| taego | all | ptr_minus_gbdt | auc | +0.0039 [+0.0019, +0.0058] |
| taego | all | ptr_minus_gbdt | auc_within_match_phase | +0.0026 [-0.0009, +0.0059] |
| taego | all | ptr_minus_krafton | auc | +0.0006 [-0.0004, +0.0017] |
| taego | all | ptr_minus_krafton | auc_within_match_phase | +0.0024 [+0.0005, +0.0043] |
| taego | all | krafton_minus_gbdt | auc | +0.0032 [+0.0012, +0.0052] |
| taego | all | krafton_minus_gbdt | auc_within_match_phase | +0.0001 [-0.0036, +0.0038] |
| taego | human | ptr_minus_gbdt | auc | +0.0035 [+0.0015, +0.0056] |
| taego | human | ptr_minus_gbdt | auc_within_match_phase | +0.0031 [-0.0005, +0.0068] |
| taego | human | ptr_minus_krafton | auc | +0.0005 [-0.0007, +0.0016] |
| taego | human | ptr_minus_krafton | auc_within_match_phase | +0.0020 [+0.0000, +0.0039] |
| taego | human | krafton_minus_gbdt | auc | +0.0030 [+0.0008, +0.0053] |
| taego | human | krafton_minus_gbdt | auc_within_match_phase | +0.0011 [-0.0026, +0.0050] |

## 손실과 예측 구간

| 코호트 | 평가 대상 | 조건 | AUC | 단계 내 AUC | 경기 내 AUC | NLL |
|---|---|---|---:|---:|---:|---:|
| erangel | all | weighted_full | 0.7900 | 0.7664 | 0.7690 | 0.4627 |
| erangel | all | weighted_truncated | 0.7986 | 0.7717 | 0.7784 | 0.4567 |
| erangel | all | cox_full | 0.7863 | 0.7636 | 0.7659 | 0.4658 |
| erangel | all | cox_truncated | 0.7987 | 0.7727 | 0.7782 | 0.4567 |
| erangel | all | bce | 0.8028 | 0.7787 | 0.7790 | 0.4517 |
| erangel | human | weighted_full | 0.7921 | 0.7620 | 0.7599 | 0.4485 |
| erangel | human | weighted_truncated | 0.7976 | 0.7650 | 0.7672 | 0.4452 |
| erangel | human | cox_full | 0.7883 | 0.7591 | 0.7566 | 0.4516 |
| erangel | human | cox_truncated | 0.7979 | 0.7663 | 0.7668 | 0.4451 |
| erangel | human | bce | 0.8028 | 0.7729 | 0.7689 | 0.4392 |
| taego | all | weighted_full | 0.7893 | 0.7669 | 0.7762 | 0.4544 |
| taego | all | weighted_truncated | 0.8027 | 0.7734 | 0.7888 | 0.4454 |
| taego | all | cox_full | 0.7833 | 0.7625 | 0.7725 | 0.4596 |
| taego | all | cox_truncated | 0.8029 | 0.7745 | 0.7871 | 0.4457 |
| taego | all | bce | 0.8064 | 0.7821 | 0.7864 | 0.4412 |
| taego | human | weighted_full | 0.7878 | 0.7609 | 0.7648 | 0.4436 |
| taego | human | weighted_truncated | 0.8004 | 0.7674 | 0.7758 | 0.4360 |
| taego | human | cox_full | 0.7830 | 0.7575 | 0.7623 | 0.4478 |
| taego | human | cox_truncated | 0.8008 | 0.7687 | 0.7743 | 0.4359 |
| taego | human | bce | 0.8057 | 0.7774 | 0.7751 | 0.4304 |

| 코호트 | 평가 대상 | 차이(앞−뒤) | 지표 | 추정치 [95% CI] |
|---|---|---|---|---:|
| erangel | all | weighted_truncated_minus_weighted_full | auc | +0.0086 [+0.0066, +0.0106] |
| erangel | all | weighted_truncated_minus_weighted_full | auc_within_match_phase | +0.0094 [+0.0062, +0.0126] |
| erangel | all | cox_truncated_minus_cox_full | auc | +0.0124 [+0.0101, +0.0146] |
| erangel | all | cox_truncated_minus_cox_full | auc_within_match_phase | +0.0123 [+0.0088, +0.0160] |
| erangel | all | bce_minus_weighted_full | auc | +0.0127 [+0.0108, +0.0147] |
| erangel | all | bce_minus_weighted_full | auc_within_match_phase | +0.0100 [+0.0070, +0.0130] |
| erangel | all | bce_minus_weighted_truncated | auc | +0.0041 [+0.0029, +0.0054] |
| erangel | all | bce_minus_weighted_truncated | auc_within_match_phase | +0.0006 [-0.0014, +0.0025] |
| erangel | all | bce_minus_cox_truncated | auc | +0.0041 [+0.0028, +0.0053] |
| erangel | all | bce_minus_cox_truncated | auc_within_match_phase | +0.0008 [-0.0013, +0.0028] |
| erangel | all | cox_full_minus_weighted_full | auc | -0.0037 [-0.0044, -0.0031] |
| erangel | all | cox_full_minus_weighted_full | auc_within_match_phase | -0.0031 [-0.0044, -0.0018] |
| erangel | human | weighted_truncated_minus_weighted_full | auc | +0.0055 [+0.0033, +0.0077] |
| erangel | human | weighted_truncated_minus_weighted_full | auc_within_match_phase | +0.0073 [+0.0041, +0.0108] |
| erangel | human | cox_truncated_minus_cox_full | auc | +0.0096 [+0.0071, +0.0120] |
| erangel | human | cox_truncated_minus_cox_full | auc_within_match_phase | +0.0101 [+0.0062, +0.0142] |
| erangel | human | bce_minus_weighted_full | auc | +0.0106 [+0.0086, +0.0128] |
| erangel | human | bce_minus_weighted_full | auc_within_match_phase | +0.0090 [+0.0056, +0.0123] |
| erangel | human | bce_minus_weighted_truncated | auc | +0.0051 [+0.0038, +0.0066] |
| erangel | human | bce_minus_weighted_truncated | auc_within_match_phase | +0.0017 [-0.0008, +0.0040] |
| erangel | human | bce_minus_cox_truncated | auc | +0.0049 [+0.0035, +0.0063] |
| erangel | human | bce_minus_cox_truncated | auc_within_match_phase | +0.0021 [-0.0002, +0.0045] |
| erangel | human | cox_full_minus_weighted_full | auc | -0.0038 [-0.0046, -0.0031] |
| erangel | human | cox_full_minus_weighted_full | auc_within_match_phase | -0.0033 [-0.0048, -0.0017] |
| taego | all | weighted_truncated_minus_weighted_full | auc | +0.0134 [+0.0111, +0.0158] |
| taego | all | weighted_truncated_minus_weighted_full | auc_within_match_phase | +0.0126 [+0.0091, +0.0160] |
| taego | all | cox_truncated_minus_cox_full | auc | +0.0196 [+0.0171, +0.0221] |
| taego | all | cox_truncated_minus_cox_full | auc_within_match_phase | +0.0147 [+0.0112, +0.0180] |
| taego | all | bce_minus_weighted_full | auc | +0.0171 [+0.0150, +0.0193] |
| taego | all | bce_minus_weighted_full | auc_within_match_phase | +0.0102 [+0.0063, +0.0142] |
| taego | all | bce_minus_weighted_truncated | auc | +0.0037 [+0.0022, +0.0052] |
| taego | all | bce_minus_weighted_truncated | auc_within_match_phase | -0.0024 [-0.0044, -0.0004] |
| taego | all | bce_minus_cox_truncated | auc | +0.0035 [+0.0021, +0.0049] |
| taego | all | bce_minus_cox_truncated | auc_within_match_phase | -0.0008 [-0.0028, +0.0012] |
| taego | all | cox_full_minus_weighted_full | auc | -0.0060 [-0.0067, -0.0052] |
| taego | all | cox_full_minus_weighted_full | auc_within_match_phase | -0.0037 [-0.0051, -0.0023] |
| taego | human | weighted_truncated_minus_weighted_full | auc | +0.0127 [+0.0101, +0.0152] |
| taego | human | weighted_truncated_minus_weighted_full | auc_within_match_phase | +0.0110 [+0.0075, +0.0149] |
| taego | human | cox_truncated_minus_cox_full | auc | +0.0178 [+0.0151, +0.0203] |
| taego | human | cox_truncated_minus_cox_full | auc_within_match_phase | +0.0121 [+0.0087, +0.0157] |
| taego | human | bce_minus_weighted_full | auc | +0.0179 [+0.0155, +0.0203] |
| taego | human | bce_minus_weighted_full | auc_within_match_phase | +0.0103 [+0.0062, +0.0144] |
| taego | human | bce_minus_weighted_truncated | auc | +0.0052 [+0.0038, +0.0068] |
| taego | human | bce_minus_weighted_truncated | auc_within_match_phase | -0.0007 [-0.0028, +0.0013] |
| taego | human | bce_minus_cox_truncated | auc | +0.0049 [+0.0035, +0.0062] |
| taego | human | bce_minus_cox_truncated | auc_within_match_phase | +0.0007 [-0.0013, +0.0028] |
| taego | human | cox_full_minus_weighted_full | auc | -0.0047 [-0.0055, -0.0039] |
| taego | human | cox_full_minus_weighted_full | auc_within_match_phase | -0.0026 [-0.0039, -0.0012] |

## 과거 요약과 순서

| 코호트 | 평가 대상 | 조건 | AUC | 단계 내 AUC | 경기 내 AUC | NLL |
|---|---|---|---:|---:|---:|---:|
| erangel | all | krafton_current_padded | 0.8041 | 0.7799 | 0.7800 | 0.4523 |
| erangel | all | krafton_history_summary | 0.8008 | 0.7775 | 0.7784 | 0.4554 |
| erangel | all | ptr_ordered_prefix | 0.8032 | 0.7791 | 0.7785 | 0.4530 |
| erangel | all | ptr_shuffled_prefix | 0.8031 | 0.7791 | 0.7782 | 0.4535 |
| erangel | human | krafton_current_padded | 0.8042 | 0.7738 | 0.7693 | 0.4389 |
| erangel | human | krafton_history_summary | 0.8009 | 0.7714 | 0.7679 | 0.4425 |
| erangel | human | ptr_ordered_prefix | 0.8035 | 0.7732 | 0.7688 | 0.4394 |
| erangel | human | ptr_shuffled_prefix | 0.8033 | 0.7732 | 0.7682 | 0.4399 |
| taego | all | krafton_current_padded | 0.8078 | 0.7838 | 0.7894 | 0.4424 |
| taego | all | krafton_history_summary | 0.8049 | 0.7821 | 0.7867 | 0.4444 |
| taego | all | ptr_ordered_prefix | 0.8080 | 0.7837 | 0.7908 | 0.4426 |
| taego | all | ptr_shuffled_prefix | 0.8070 | 0.7828 | 0.7893 | 0.4433 |
| taego | human | krafton_current_padded | 0.8074 | 0.7792 | 0.7772 | 0.4309 |
| taego | human | krafton_history_summary | 0.8038 | 0.7768 | 0.7745 | 0.4332 |
| taego | human | ptr_ordered_prefix | 0.8076 | 0.7793 | 0.7792 | 0.4303 |
| taego | human | ptr_shuffled_prefix | 0.8069 | 0.7786 | 0.7785 | 0.4310 |

| 코호트 | 평가 대상 | 차이(앞−뒤) | 지표 | 추정치 [95% CI] |
|---|---|---|---|---:|
| erangel | all | krafton_history_summary_minus_krafton_current_padded | auc | -0.0033 [-0.0046, -0.0019] |
| erangel | all | krafton_history_summary_minus_krafton_current_padded | auc_within_match_phase | -0.0017 [-0.0040, +0.0007] |
| erangel | all | ptr_ordered_prefix_minus_ptr_shuffled_prefix | auc | +0.0001 [-0.0001, +0.0003] |
| erangel | all | ptr_ordered_prefix_minus_ptr_shuffled_prefix | auc_within_match_phase | +0.0004 [-0.0002, +0.0010] |
| erangel | human | krafton_history_summary_minus_krafton_current_padded | auc | -0.0033 [-0.0048, -0.0017] |
| erangel | human | krafton_history_summary_minus_krafton_current_padded | auc_within_match_phase | -0.0014 [-0.0044, +0.0016] |
| erangel | human | ptr_ordered_prefix_minus_ptr_shuffled_prefix | auc | +0.0001 [-0.0001, +0.0004] |
| erangel | human | ptr_ordered_prefix_minus_ptr_shuffled_prefix | auc_within_match_phase | +0.0006 [-0.0000, +0.0014] |
| taego | all | krafton_history_summary_minus_krafton_current_padded | auc | -0.0028 [-0.0041, -0.0016] |
| taego | all | krafton_history_summary_minus_krafton_current_padded | auc_within_match_phase | -0.0027 [-0.0045, -0.0008] |
| taego | all | ptr_ordered_prefix_minus_ptr_shuffled_prefix | auc | +0.0010 [+0.0006, +0.0013] |
| taego | all | ptr_ordered_prefix_minus_ptr_shuffled_prefix | auc_within_match_phase | +0.0014 [+0.0006, +0.0023] |
| taego | human | krafton_history_summary_minus_krafton_current_padded | auc | -0.0036 [-0.0049, -0.0022] |
| taego | human | krafton_history_summary_minus_krafton_current_padded | auc_within_match_phase | -0.0027 [-0.0049, -0.0003] |
| taego | human | ptr_ordered_prefix_minus_ptr_shuffled_prefix | auc | +0.0007 [+0.0003, +0.0010] |
| taego | human | ptr_ordered_prefix_minus_ptr_shuffled_prefix | auc_within_match_phase | +0.0006 [-0.0003, +0.0015] |

## AI 입력과 사람 팀 손실

| 코호트 | 평가 대상 | 조건 | AUC | 단계 내 AUC | 경기 내 AUC | NLL |
|---|---|---|---:|---:|---:|---:|
| erangel | all | gbdt_mixed | 0.8034 | 0.7743 | 0.7833 | 0.4518 |
| erangel | all | gbdt_no_ai_input | 0.8025 | 0.7727 | 0.7816 | 0.4523 |
| erangel | all | gbdt_human_loss | 0.7975 | 0.7687 | 0.7733 | 0.4559 |
| erangel | all | krafton_mixed | 0.8039 | 0.7792 | 0.7799 | 0.4522 |
| erangel | all | krafton_no_ai_input | 0.8022 | 0.7769 | 0.7784 | 0.4538 |
| erangel | all | krafton_human_loss | 0.7973 | 0.7737 | 0.7737 | 0.4610 |
| erangel | all | ptr_mixed | 0.8014 | 0.7774 | 0.7772 | 0.4538 |
| erangel | all | ptr_no_ai_input | 0.8012 | 0.7773 | 0.7767 | 0.4539 |
| erangel | all | ptr_human_loss | 0.7985 | 0.7765 | 0.7738 | 0.4623 |
| erangel | human | gbdt_mixed | 0.8036 | 0.7683 | 0.7724 | 0.4382 |
| erangel | human | gbdt_no_ai_input | 0.8024 | 0.7671 | 0.7706 | 0.4394 |
| erangel | human | gbdt_human_loss | 0.8042 | 0.7690 | 0.7727 | 0.4372 |
| erangel | human | krafton_mixed | 0.8044 | 0.7734 | 0.7699 | 0.4388 |
| erangel | human | krafton_no_ai_input | 0.8033 | 0.7724 | 0.7688 | 0.4399 |
| erangel | human | krafton_human_loss | 0.8022 | 0.7713 | 0.7680 | 0.4411 |
| erangel | human | ptr_mixed | 0.8015 | 0.7716 | 0.7670 | 0.4408 |
| erangel | human | ptr_no_ai_input | 0.8010 | 0.7713 | 0.7656 | 0.4413 |
| erangel | human | ptr_human_loss | 0.8033 | 0.7729 | 0.7691 | 0.4389 |
| taego | all | gbdt_mixed | 0.8049 | 0.7778 | 0.7887 | 0.4429 |
| taego | all | gbdt_no_ai_input | 0.8050 | 0.7769 | 0.7882 | 0.4428 |
| taego | all | gbdt_human_loss | 0.8007 | 0.7732 | 0.7818 | 0.4455 |
| taego | all | krafton_mixed | 0.8075 | 0.7834 | 0.7889 | 0.4423 |
| taego | all | krafton_no_ai_input | 0.8066 | 0.7814 | 0.7885 | 0.4430 |
| taego | all | krafton_human_loss | 0.8025 | 0.7786 | 0.7824 | 0.4499 |
| taego | all | ptr_mixed | 0.8055 | 0.7817 | 0.7867 | 0.4435 |
| taego | all | ptr_no_ai_input | 0.8068 | 0.7829 | 0.7899 | 0.4427 |
| taego | all | ptr_human_loss | 0.8010 | 0.7790 | 0.7772 | 0.4551 |
| taego | human | gbdt_mixed | 0.8048 | 0.7733 | 0.7763 | 0.4306 |
| taego | human | gbdt_no_ai_input | 0.8048 | 0.7726 | 0.7766 | 0.4311 |
| taego | human | gbdt_human_loss | 0.8050 | 0.7731 | 0.7776 | 0.4303 |
| taego | human | krafton_mixed | 0.8071 | 0.7789 | 0.7776 | 0.4308 |
| taego | human | krafton_no_ai_input | 0.8061 | 0.7773 | 0.7776 | 0.4319 |
| taego | human | krafton_human_loss | 0.8069 | 0.7777 | 0.7776 | 0.4308 |
| taego | human | ptr_mixed | 0.8053 | 0.7775 | 0.7764 | 0.4317 |
| taego | human | ptr_no_ai_input | 0.8064 | 0.7784 | 0.7787 | 0.4309 |
| taego | human | ptr_human_loss | 0.8070 | 0.7787 | 0.7778 | 0.4305 |

| 코호트 | 평가 대상 | 차이(앞−뒤) | 지표 | 추정치 [95% CI] |
|---|---|---|---|---:|
| erangel | all | gbdt_no_ai_input_minus_gbdt_mixed | auc | -0.0010 [-0.0021, +0.0001] |
| erangel | all | gbdt_no_ai_input_minus_gbdt_mixed | auc_within_match_phase | -0.0017 [-0.0040, +0.0006] |
| erangel | all | gbdt_human_loss_minus_gbdt_mixed | auc | -0.0059 [-0.0085, -0.0033] |
| erangel | all | gbdt_human_loss_minus_gbdt_mixed | auc_within_match_phase | -0.0100 [-0.0142, -0.0060] |
| erangel | all | krafton_no_ai_input_minus_krafton_mixed | auc | -0.0017 [-0.0027, -0.0007] |
| erangel | all | krafton_no_ai_input_minus_krafton_mixed | auc_within_match_phase | -0.0015 [-0.0032, +0.0002] |
| erangel | all | krafton_human_loss_minus_krafton_mixed | auc | -0.0066 [-0.0087, -0.0048] |
| erangel | all | krafton_human_loss_minus_krafton_mixed | auc_within_match_phase | -0.0062 [-0.0093, -0.0036] |
| erangel | all | ptr_no_ai_input_minus_ptr_mixed | auc | -0.0002 [-0.0008, +0.0004] |
| erangel | all | ptr_no_ai_input_minus_ptr_mixed | auc_within_match_phase | -0.0005 [-0.0018, +0.0007] |
| erangel | all | ptr_human_loss_minus_ptr_mixed | auc | -0.0030 [-0.0047, -0.0012] |
| erangel | all | ptr_human_loss_minus_ptr_mixed | auc_within_match_phase | -0.0034 [-0.0066, -0.0006] |
| erangel | human | gbdt_no_ai_input_minus_gbdt_mixed | auc | -0.0012 [-0.0022, -0.0002] |
| erangel | human | gbdt_no_ai_input_minus_gbdt_mixed | auc_within_match_phase | -0.0018 [-0.0043, +0.0005] |
| erangel | human | gbdt_human_loss_minus_gbdt_mixed | auc | +0.0006 [-0.0005, +0.0018] |
| erangel | human | gbdt_human_loss_minus_gbdt_mixed | auc_within_match_phase | +0.0003 [-0.0020, +0.0028] |
| erangel | human | krafton_no_ai_input_minus_krafton_mixed | auc | -0.0011 [-0.0019, -0.0003] |
| erangel | human | krafton_no_ai_input_minus_krafton_mixed | auc_within_match_phase | -0.0011 [-0.0028, +0.0004] |
| erangel | human | krafton_human_loss_minus_krafton_mixed | auc | -0.0022 [-0.0029, -0.0013] |
| erangel | human | krafton_human_loss_minus_krafton_mixed | auc_within_match_phase | -0.0019 [-0.0034, -0.0003] |
| erangel | human | ptr_no_ai_input_minus_ptr_mixed | auc | -0.0005 [-0.0010, +0.0000] |
| erangel | human | ptr_no_ai_input_minus_ptr_mixed | auc_within_match_phase | -0.0014 [-0.0026, -0.0003] |
| erangel | human | ptr_human_loss_minus_ptr_mixed | auc | +0.0018 [+0.0009, +0.0028] |
| erangel | human | ptr_human_loss_minus_ptr_mixed | auc_within_match_phase | +0.0021 [+0.0003, +0.0039] |
| taego | all | gbdt_no_ai_input_minus_gbdt_mixed | auc | +0.0001 [-0.0010, +0.0010] |
| taego | all | gbdt_no_ai_input_minus_gbdt_mixed | auc_within_match_phase | -0.0005 [-0.0024, +0.0017] |
| taego | all | gbdt_human_loss_minus_gbdt_mixed | auc | -0.0042 [-0.0065, -0.0021] |
| taego | all | gbdt_human_loss_minus_gbdt_mixed | auc_within_match_phase | -0.0069 [-0.0109, -0.0033] |
| taego | all | krafton_no_ai_input_minus_krafton_mixed | auc | -0.0010 [-0.0018, -0.0001] |
| taego | all | krafton_no_ai_input_minus_krafton_mixed | auc_within_match_phase | -0.0004 [-0.0019, +0.0010] |
| taego | all | krafton_human_loss_minus_krafton_mixed | auc | -0.0050 [-0.0067, -0.0035] |
| taego | all | krafton_human_loss_minus_krafton_mixed | auc_within_match_phase | -0.0066 [-0.0095, -0.0039] |
| taego | all | ptr_no_ai_input_minus_ptr_mixed | auc | +0.0013 [+0.0006, +0.0019] |
| taego | all | ptr_no_ai_input_minus_ptr_mixed | auc_within_match_phase | +0.0032 [+0.0017, +0.0047] |
| taego | all | ptr_human_loss_minus_ptr_mixed | auc | -0.0046 [-0.0063, -0.0030] |
| taego | all | ptr_human_loss_minus_ptr_mixed | auc_within_match_phase | -0.0095 [-0.0129, -0.0065] |
| taego | human | gbdt_no_ai_input_minus_gbdt_mixed | auc | -0.0000 [-0.0009, +0.0009] |
| taego | human | gbdt_no_ai_input_minus_gbdt_mixed | auc_within_match_phase | +0.0003 [-0.0019, +0.0028] |
| taego | human | gbdt_human_loss_minus_gbdt_mixed | auc | +0.0002 [-0.0008, +0.0012] |
| taego | human | gbdt_human_loss_minus_gbdt_mixed | auc_within_match_phase | +0.0013 [-0.0010, +0.0036] |
| taego | human | krafton_no_ai_input_minus_krafton_mixed | auc | -0.0010 [-0.0018, -0.0003] |
| taego | human | krafton_no_ai_input_minus_krafton_mixed | auc_within_match_phase | +0.0000 [-0.0016, +0.0016] |
| taego | human | krafton_human_loss_minus_krafton_mixed | auc | -0.0002 [-0.0007, +0.0004] |
| taego | human | krafton_human_loss_minus_krafton_mixed | auc_within_match_phase | +0.0000 [-0.0010, +0.0011] |
| taego | human | ptr_no_ai_input_minus_ptr_mixed | auc | +0.0010 [+0.0004, +0.0016] |
| taego | human | ptr_no_ai_input_minus_ptr_mixed | auc_within_match_phase | +0.0024 [+0.0010, +0.0036] |
| taego | human | ptr_human_loss_minus_ptr_mixed | auc | +0.0016 [+0.0010, +0.0022] |
| taego | human | ptr_human_loss_minus_ptr_mixed | auc_within_match_phase | +0.0014 [+0.0001, +0.0027] |

## GATv2와 관계 대응

| 코호트 | 평가 대상 | 조건 | AUC | 단계 내 AUC | 경기 내 AUC | NLL |
|---|---|---|---:|---:|---:|---:|
| erangel | all | gatv2 | 0.8048 | 0.7824 | 0.7807 | 0.4511 |
| erangel | all | gatv2_shuffled | 0.8027 | 0.7797 | 0.7787 | 0.4531 |
| erangel | all | gbdt | 0.8034 | 0.7743 | 0.7833 | 0.4518 |
| erangel | all | krafton | 0.8039 | 0.7792 | 0.7799 | 0.4522 |
| erangel | all | ptr | 0.8014 | 0.7774 | 0.7772 | 0.4538 |
| erangel | human | gatv2 | 0.8049 | 0.7764 | 0.7711 | 0.4383 |
| erangel | human | gatv2_shuffled | 0.8028 | 0.7737 | 0.7688 | 0.4402 |
| erangel | human | gbdt | 0.8036 | 0.7683 | 0.7724 | 0.4382 |
| erangel | human | krafton | 0.8044 | 0.7734 | 0.7699 | 0.4388 |
| erangel | human | ptr | 0.8015 | 0.7716 | 0.7670 | 0.4408 |
| taego | all | gatv2 | 0.8079 | 0.7831 | 0.7891 | 0.4410 |
| taego | all | gatv2_shuffled | 0.8068 | 0.7820 | 0.7882 | 0.4423 |
| taego | all | gbdt | 0.8049 | 0.7778 | 0.7887 | 0.4429 |
| taego | all | krafton | 0.8075 | 0.7834 | 0.7889 | 0.4423 |
| taego | all | ptr | 0.8055 | 0.7817 | 0.7867 | 0.4435 |
| taego | human | gatv2 | 0.8071 | 0.7781 | 0.7769 | 0.4302 |
| taego | human | gatv2_shuffled | 0.8058 | 0.7769 | 0.7769 | 0.4314 |
| taego | human | gbdt | 0.8048 | 0.7733 | 0.7763 | 0.4306 |
| taego | human | krafton | 0.8071 | 0.7789 | 0.7776 | 0.4308 |
| taego | human | ptr | 0.8053 | 0.7775 | 0.7764 | 0.4317 |

| 코호트 | 평가 대상 | 차이(앞−뒤) | 지표 | 추정치 [95% CI] |
|---|---|---|---|---:|
| erangel | all | gatv2_minus_gbdt | auc | +0.0013 [-0.0009, +0.0035] |
| erangel | all | gatv2_minus_gbdt | auc_within_match_phase | -0.0026 [-0.0061, +0.0010] |
| erangel | all | gatv2_minus_krafton | auc | +0.0009 [-0.0004, +0.0022] |
| erangel | all | gatv2_minus_krafton | auc_within_match_phase | +0.0008 [-0.0013, +0.0030] |
| erangel | all | gatv2_minus_ptr | auc | +0.0034 [+0.0018, +0.0050] |
| erangel | all | gatv2_minus_ptr | auc_within_match_phase | +0.0035 [+0.0012, +0.0056] |
| erangel | all | gatv2_minus_gatv2_shuffled | auc | +0.0020 [+0.0009, +0.0032] |
| erangel | all | gatv2_minus_gatv2_shuffled | auc_within_match_phase | +0.0019 [+0.0002, +0.0037] |
| erangel | human | gatv2_minus_gbdt | auc | +0.0012 [-0.0010, +0.0036] |
| erangel | human | gatv2_minus_gbdt | auc_within_match_phase | -0.0013 [-0.0053, +0.0027] |
| erangel | human | gatv2_minus_krafton | auc | +0.0005 [-0.0010, +0.0021] |
| erangel | human | gatv2_minus_krafton | auc_within_match_phase | +0.0012 [-0.0013, +0.0039] |
| erangel | human | gatv2_minus_ptr | auc | +0.0033 [+0.0016, +0.0051] |
| erangel | human | gatv2_minus_ptr | auc_within_match_phase | +0.0041 [+0.0015, +0.0068] |
| erangel | human | gatv2_minus_gatv2_shuffled | auc | +0.0021 [+0.0008, +0.0033] |
| erangel | human | gatv2_minus_gatv2_shuffled | auc_within_match_phase | +0.0023 [+0.0000, +0.0044] |
| taego | all | gatv2_minus_gbdt | auc | +0.0029 [+0.0009, +0.0050] |
| taego | all | gatv2_minus_gbdt | auc_within_match_phase | +0.0004 [-0.0033, +0.0040] |
| taego | all | gatv2_minus_krafton | auc | +0.0003 [-0.0008, +0.0015] |
| taego | all | gatv2_minus_krafton | auc_within_match_phase | +0.0001 [-0.0016, +0.0019] |
| taego | all | gatv2_minus_ptr | auc | +0.0023 [+0.0012, +0.0034] |
| taego | all | gatv2_minus_ptr | auc_within_match_phase | +0.0024 [+0.0002, +0.0044] |
| taego | all | gatv2_minus_gatv2_shuffled | auc | +0.0010 [+0.0001, +0.0019] |
| taego | all | gatv2_minus_gatv2_shuffled | auc_within_match_phase | +0.0008 [-0.0008, +0.0024] |
| taego | human | gatv2_minus_gbdt | auc | +0.0023 [+0.0002, +0.0045] |
| taego | human | gatv2_minus_gbdt | auc_within_match_phase | +0.0006 [-0.0033, +0.0043] |
| taego | human | gatv2_minus_krafton | auc | +0.0000 [-0.0012, +0.0012] |
| taego | human | gatv2_minus_krafton | auc_within_match_phase | -0.0007 [-0.0026, +0.0013] |
| taego | human | gatv2_minus_ptr | auc | +0.0018 [+0.0007, +0.0030] |
| taego | human | gatv2_minus_ptr | auc_within_match_phase | +0.0006 [-0.0015, +0.0026] |
| taego | human | gatv2_minus_gatv2_shuffled | auc | +0.0013 [+0.0004, +0.0023] |
| taego | human | gatv2_minus_gatv2_shuffled | auc_within_match_phase | +0.0001 [-0.0017, +0.0018] |

## 선택된 설정과 학습 자원

검증 NLL로 설정을 선택했다. 실행시간은 동시 실행과 장치 차이의 영향을 받으므로 속도 우열을 판정하는 벤치마크가 아니다. 정확한 값과 설정은 `reports/sac2027_patch_v14/tuning/selected_resources.csv`에 있다.

| 코호트 | 모델 | 선택 trial | 검증 NLL | 파라미터 | 16회 학습 누적 초 |
|---|---|---:|---:|---:|---:|
| erangel | gbdt | 4 | 0.44610 | 해당 없음 | 119.6 |
| erangel | krafton | 11 | 0.44630 | 251393 | 1960.0 |
| erangel | ptr | 9 | 0.44533 | 148354 | 3477.5 |
| erangel | gatv2 | 0 | 0.44436 | 369665 | 1680.2 |
| taego | gbdt | 4 | 0.43763 | 해당 없음 | 156.3 |
| taego | krafton | 11 | 0.43806 | 251393 | 2425.9 |
| taego | ptr | 9 | 0.43736 | 148354 | 3493.5 |
| taego | gatv2 | 11 | 0.43642 | 203649 | 1959.8 |

주 분석·구조 비교는 `primary_results_v2/`, 전체/사람 팀 참조 비교는 `control_evaluation/main_population/`, 원 과제 지표는 `native_evaluation/`를 따른다. 그래프 선정과 원본 코드 검증은 `graph_model_selection/SELECTION_KO.md`, 선행연구 교정은 `baseline_audit/AUDIT_KO.md`에 기록했다.
