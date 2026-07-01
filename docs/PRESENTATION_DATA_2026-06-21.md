# V3 결과 — 발표/논문 단일 진실 출처 (2026-06-21)

모든 수치는 V3 full-DB, squad, 공유 split(`v3_full_main_causal/split.json`, test 1,521매치 / 77,199 at-risk team-phase 셀), match-level cluster bootstrap(1000 reps) 기준. 예측 단위 = **cut_phase별 trajectory**(매치당 1예측 아님).

## A. Resolution — 스냅샷이 phase 천장을 깬다 (시공간 헤드라인)
| 모델 | 입력 | hazard AUC | rank Spearman |
|---|---|---|---|
| HGB (tabular GBM) | phase 피처 | 0.7864 | — |
| phase_tcn | phase only (graph X) | 0.7872 | 0.6450 |
| phase_rgcn | phase only (graph O) | 0.7926 | 0.6521 |
| **mr_tcn** | **MR 10s 스냅샷** | **0.8349** | 0.7377 |
| **mr_rgcn** | **MR 스냅샷 + graph** | **0.8467** | 0.7596 |

핵심: phase 집계 천장 ≈0.79 (HGB≈phase-GNN) → **MR 스냅샷 +0.054 돌파**.

**match-level 유의성 확정** (MR vs phase paired, 전 cut_phase 유의):
- MR vs phase (둘 다 graph): ΔAUC +0.048~+0.076, ΔSpearman +0.11~+0.16 — **7개 phase 전부 CI 0 제외**
- MR vs phase (둘 다 no-graph): ΔAUC +0.041~+0.090 — **7개 phase 전부 유의**
- 특히 rank Spearman: phase_rgcn 0.51~0.63 → mr_rgcn 0.71~0.76 (스냅샷이 순위 예측에 결정적)
→ "스냅샷이 천장을 깬다"가 hazard·rank 모두 phase별로 통계 확정.

## B. Main 매트릭스 (8모델, hazard AUC, 참고)
mr_gat 0.8496 · mr_rgcn_film 0.8465 · mr_gcn 0.8461 · mr_rgcn 0.8460 · mr_farnet 0.8454 · mr_rgcn_attn_rank 0.8449 · mr_sage 0.8438 · mr_tcn 0.8358

## C. V3 vs Tabular 베이스라인 (전 phase 유의, 어젯밤 확정)
HGB 0.7864 / LR 0.7540 / **V3 mr_gat 0.8496**. ΔAUC(V3−HGB) per cut_phase [95% CI], 전부 유의:
| phase | HGB | V3 | ΔAUC [CI] |
|---|---|---|---|
| 1 | 0.825 | 0.893 | +0.068 [+0.063,+0.073] |
| 2 | 0.748 | 0.835 | +0.087 [+0.078,+0.095] |
| 3 | 0.725 | 0.800 | +0.075 [+0.067,+0.083] |
| 4 | 0.756 | 0.815 | +0.059 [+0.052,+0.066] |
| 5 | 0.742 | 0.805 | +0.063 [+0.053,+0.072] |
| 6 | 0.720 | 0.792 | +0.071 [+0.055,+0.086] |
| 7 | 0.655 | 0.741 | +0.087 [+0.059,+0.113] |
| ALL | 0.786 | 0.850 | **+0.063 [+0.060,+0.067]** |
(vs LR: ALL +0.096). MR-era 보강: MR vs HGB Wilcoxon p=2.5e-144.

## D. phase별 모델 성능 + 95% CI (mr_rgcn_attn_rank, 전부 유의)
| phase | hazard AUC [CI] | rank Spearman [CI] |
|---|---|---|
| 1 | 0.889 [0.884,0.895] | 0.771 [0.762,0.780] |
| 2 | 0.826 [0.819,0.835] | 0.723 [0.712,0.734] |
| 3 | 0.792 [0.784,0.801] | 0.727 [0.717,0.739] |
| 4 | 0.811 [0.802,0.820] | 0.766 [0.754,0.777] |
| 5 | 0.798 [0.786,0.808] | 0.800 [0.788,0.812] |
| 6 | 0.794 [0.780,0.808] | 0.813 [0.797,0.826] |
| 7 | 0.745 [0.715,0.772] | 0.785 [0.758,0.809] |

## E. rank lock-in = decision window (유의 확인)
ΔSpearman(phase_j − phase_i), match-level CI:
- **3→6: +0.085 [+0.070,+0.101] 유의** (최종순위 구조가 이 구간에서 굳음)
- 3→5: +0.073 [+0.059,+0.087] 유의 · 4→6: +0.047 [+0.031,+0.061] 유의
- 5→6: +0.013 [−0.002,+0.025] ns (plateau) · 6→7: −0.028 [−0.053,−0.003] 유의 하락(소수 risk-set)

## F. 그래프 위상 = 신호 없음, 평활화일 뿐 (그래프 헤드라인)
세 null 대조군 모두 row-normalized(위상만 측정). 3-seed 평균 hazard AUC:
- no_graph 0.834 · observed 0.845 · **edge_shuffle 0.847** (observed보다 일관되게 높음, 3/3 시드)
- node_shuffle ≈ observed (팀 신원 무관)
- 채널: proximity 0.8455 · combat 0.837 · copressure 0.837 · prox+combat 0.846

match-level paired (observed − edge_shuffle):
- phase 1–3: observed가 **유의하게 나쁨** (ΔAUC −0.002~−0.004, ΔSpr −0.019~−0.025, CI<0)
- phase 4–7: ns. **어느 phase도 유의하게 좋지 않음** → 엔드게임 위상 가설도 기각
- observed vs no_graph: phase 1–6 유의(+0.008~+0.015) → "그래프(평활화)는 도움, 위상은 아님 = GNN이 low-pass filter"

## G. phase → 실제 게임 시각 (squad ST-DB 실측)
| phase | 시각(시작→끝) | safe 반경 | 생존팀 | e-sports 국면 |
|---|---|---|---|---|
| 1 | 1.5→9.5분 | 5506m(맵전체) | ~21 | 낙하·파밍 |
| 2 | 9.7→13.1분 | 1965m | ~14 | 1차 로테이션 |
| 3 | 13.3→16.1분 | 1099m | ~11 | 중반 포지셔닝 |
| **4** | **16.3→19.1분** | **659m** | **~9** | 후반 구도 진입 |
| **5** | **19.3→21.7분** | **364m** | **~6** | 후반 로테이션 압박 |
| **6** | **21.9→23.6분** | **188m** | **~4** | 엔드게임 setup |
| 7 | 24.0→25.2분 | 98m | ~2.7 | 최종 교전 |
decision window(rank lock-in)와 일치: phase 4–6 = 게임 16–24분.

## 핵심 메시지 (3산출물 공통)
1. **시공간(STDB+MR)**: 10s 스냅샷 다해상도가 phase 집계 천장(0.786)을 +0.06 돌파(0.850), 전 phase 유의, time-shuffle 시 0.51 붕괴.
2. **그래프**: 동적 팀 그래프의 이득은 mean-field 평활화이며 **관측 위상은 무의미**(observed ≤ shuffle, null-model로 분리). honest negative + GNN=low-pass filter 실증.
3. **융합**: 최종 생존 서열은 게임 **16–24분(존 4→6)**에 유의하게 결정됨(rank lock-in). hazard 재정식화가 둘을 묶는 공통 프레임.
