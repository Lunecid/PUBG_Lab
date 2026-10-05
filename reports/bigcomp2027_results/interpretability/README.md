# PTR-Former 해석 분석 (탐색적·기술적 분석, 인과 주장 아님)
대상: `bigcomp2027_ptr_v1`의 M11R / M11RZ / M11R-shuf(대조), 12 코호트, arm당 16 run(erangel_tpp_squad·taego_tpp_squad는 seed 13/42/2026, 나머지 seed 42). 코드·체크포인트·덤프 읽기 전용, 2 스레드.
combat 가중치 모드는 metrics.json 기준 전 코호트 `log`(덤프가 이미 ln(1+강도) → 추가 log1p 없음). 가중치 통계는 test 덤프 전체 경기에서 에스컬레이션(combat 쌍 prox=0)·대칭화 후 alive 쌍의 비영 값으로 산출(shuf도 동일 값 확인).

## 1. 관계 편향 b[r,h] — `relation_bias_*.csv`, `fig_relation_bias`
- 가중치 median/p90(코호트 평균, 밀도): prox 0.006/0.109(0.90), combat 5.27/6.34(0.06), copress 0.26/0.71(0.04). 유효 편향 = b × median(또는 p90).
- M11R(n=128 = run 16 × 블록 2 × 헤드 4) 유효 편향(median) 평균±sd: prox ≈0(|·|=0.0003), combat_out +0.46±0.68, combat_in +0.61±0.89, copress +0.026±0.033. p90 기준 +0.016 / +0.57 / +0.74 / +0.070. raw b 평균 0.17 / 0.087 / 0.114 / 0.099.
- 부호 일관성(entry, 다수 부호 비율): raw b는 전 채널 양(+) 69–80%(shuf 58–80%), Δb=b−b_init은 88–96%(shuf 79–97%). M11R run 단위 유효 편향은 combat_out 16/16, combat_in 15/16 run이 양. 단 seed 42 초기값을 10개 코호트가 공유하므로 부호 일관성만으로는 약하고, 초기값이 상쇄되는 real−shuf 쌍비교를 우선함(b_init은 init_seed로 정확 복원, corr(b,b_init)=0.3–0.9).
- real vs shuf(코호트 단위 쌍비교 n=12, 정확 부호반전 검정, 다중비교 보정 없음): combat_out 유효 편향 0.39 vs 0.12(차 +0.27, 95% CI 0.19–0.37, 12/12 코호트, p=0.0005), combat_in 0.54 vs 0.08(+0.45, 0.30–0.60, 11/12, p=0.001). 크기 |유효 편향|(combat 두 채널 평균) 0.61 vs 0.32(+0.29, 11/12, p=0.001). prox·copress는 부호 있는 유효 편향에서 차이 없음(p=0.82, 0.45; prox 크기 차는 0.0001 logit). M11RZ도 같은 방향(combat 평균 0.41 vs 0.10, 11/12).
- 주의: combat>copress>prox 순위는 가중치 척도(5.3 vs 0.26 vs 0.006)의 영향이 커서 shuf에서도 유지됨(|유효 편향| 0.40/0.25 vs 0.03 vs 0.0002). 실제 엣지에 특이적인 것은 부호 있는 평균과 shuf 대비 크기 차이다.

## 2. 어텐션 질량 — `attention_lift_*.csv`, `fig_attention_lift`
- 설정: M11R·M11(·M11RZ, CSV만) seed 42, test ≤300경기/코호트(10개 코호트 전량, erangel_tpp_squad 324→300·taego_tpp_squad 415→300은 무작위 표본, seed 20261006). `masked_attention`을 런타임에서 감싸 team-axis softmax를 저장(원본 forward와 hazard_logit 차이 0, 행합 오차 <4e-7; `attention_sanity.csv`).
- lift = (클래스 key에 대한 쿼리 팀의 평균 어텐션) / (1/alive key 수), 쿼리 팀 평균 → 코호트 평균(2블록·4헤드 평균). 클래스: combat(양방향 중 하나), prox-only, copress, no edge. copress-only 쌍은 0개(copress 엣지는 항상 prox와 공존)라 prox∩copress로 표기.
- M11R: combat 2.20(sd 0.74, 1.17–3.33, 12/12 코호트 >1), prox-only 0.90, copress(+prox) 0.89, no edge 0.60. M11(편향 없음): 1.08 / 1.00 / 0.98 / 0.76 → 편향 없는 모델의 학습 어텐션은 combat 쌍에 거의 집중하지 않음.
- M11R−M11 combat 차이 +1.12(95% CI 0.74–1.52, 12/12, p=0.0005). 블록별 L0 2.80 vs 1.12, L1 1.61 vs 1.04. M11R 헤드별 L0 4.5/2.2/2.7/1.8, L1 0.9/2.1/1.2/2.2. target 행만 1.96 vs 0.98, 쌍 가중(ratio of sums) 1.72 vs 1.05.

## 3. Zone 객체(M11RZ seed 42) — `zone_bias_curve*.csv`, `attention_zone_token*.csv`, `fig_zone_bias_curve`
- 팀↔zone 편향(헤드·블록 평균, 나머지 입력은 test 중앙값; 중앙값 d/r 현재 0.70–0.77, 다음 0.88–0.98): d/r 0→1→2에서 현재 zone +0.05→+0.14→+0.23, 다음 zone +0.03→+0.12→+0.22. 12/12 코호트 단조 증가(코호트별 0–2 변화폭 0.05–0.49), 부분의존 평균도 유사(0.06→0.16→0.25).
- 헤드별 부호는 갈림(평균은 상쇄 결과, d/r 0→2): L0 h1 +0.43→+1.68, L0 h2 +0.28→+1.17, L0 h0 −0.16→−0.29, L1 h1 −0.15→−0.56.
- zone 토큰 어텐션 lift(K에 zone 토큰 포함, 1=균일): L0 2.38, L1 0.73. 다음 zone 안 vs 밖 1.40 vs 1.73(차 −0.33, CI −0.62~−0.07, 밖>안 10/12 코호트, p=0.048; L0 p=0.038, L1 p=0.73). d_next 구간별 <1.5에서 ≈1.4, [1.5,2) 1.73, ≥2 2.61. 현재 zone 안 vs 밖 1.37 vs 2.05(p=0.002).

## 요약·한계
- combat 엣지는 유효 편향(≈0.5–0.7 logit)과 어텐션 lift(≈2.2배) 모두 가장 크다. 다만 유효 편향 순위는 척도 효과가 크고, combat lift는 M11에 없고 M11R에서만 나타난다(기술적 관찰).
- 한계: 12 코호트는 독립이 아님(같은 게임, init seed 공유), 일부 코호트 test 20–65경기, 정확검정 p 하한 0.0005(n=12), 어텐션·편향은 설명도 인과도 아님.
- 재현: `python interp.py a1 && python interp.py a23 && python make_figures.py`(이 폴더, pyenv 3.11, 2 스레드). 산출: `relation_bias_*`, `attention_*`, `zone_*` CSV, `fig_*.{pdf,png}`.
