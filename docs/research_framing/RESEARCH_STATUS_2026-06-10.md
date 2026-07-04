# 연구 전체 정리 — 2026-06-10 기준

> 작성: Claude Code 종합 점검 (코드·아티팩트·논문 직접 검증 기반).
> 모든 수치는 산출물 파일에서 재확인한 값이며, 결과마다 기반 데이터 스냅샷을 명시함.

## 1. 연구 한 줄 요약

PUBG 공식 텔레메트리를 **phase-정렬 시공간 데이터베이스(STDB)**로 재구성하고, 그 위에서
**다음 페이즈 팀 탈락 위험(discrete-time hazard)**을 정의한 뒤, **동적 팀-팀 그래프(DynamicSTGNN)**가
강한 tabular baseline 대비 추가 신호를 주는지 검증하는 연구.

- 논문: *Modeling PUBG Telemetry as a Spatio-Temporal Database for Phase-Level Team Elimination Prediction* (ACM 형식, ICE 2027 타깃)
- 핵심 정식화: `h_{i,k} = P(b_k < T_i ≤ b_{k+1} | T_i > b_k, X_{i,≤k})` (Allison 1982; Singer & Willett 1993)

## 2. 연구 질문 (현재 활성 서론 short-en 기준)

- **RQ1.** PUBG 텔레메트리를 이동 팀·동적 자기장·공간 이벤트·페이즈 구조를 담는 STDB로 어떻게 표현하는가
- **RQ2.** 자기장 축소에 따라 팀 궤적과 생존 조건이 페이즈 간 어떻게 변하는가 *(⚠ 현재 대응 실험 미배정 — §10 참조)*
- **RQ3.** 시공간 피처와 동적 팀 간 그래프 관계가 강한 tabular baseline을 넘어 다음 페이즈 탈락 예측에 기여하는가

## 3. 연구 타임라인

| 시기 | 내용 |
|---|---|
| 2026-04/02–04/09 | **레거시 snapshot 시대**: 10초 스냅샷 + 39d 피처 + ArenaSurvivalNet(계층 GNN+GRU). git 커밋 전부 이 구간(47개). 이후 legacy 공식 선언 |
| 2026-04/23–04/30 | **네트워크 분석 시대**: Baltic_Main squad 1,000매치 H1/H2/H3 가설검증, 전략 4유형 분석 |
| 2026-05/04–05/24 | webapp 구축, ICE 2027 가이드 확보(5/18), STDB 연구 제안서(5/23–24) — **서사 전환점** |
| 2026-05/26–06/05 | **phase/STDB 시대 개막**: stdb/ 스키마·ETL, model_stgcn/ DynamicSTGNN, edge/feature ablation, phase-wise 평가 (17,977매치 스냅샷) |
| 2026-06/08 | 필수 보강 실험 3종 완료 (seed 반복·edge sanity·calibration) |
| 2026-06/09–10 | 논문 LaTeX 섹션 일괄 작성 → **full-DB 재실험(21,824매치) 완료** → full-DB edge ablation까지 완료하고 06·07 섹션에 full-DB 수치 반영 |

## 4. 데이터

- 수집: PUBG 공식 API → PostgreSQL `pubg.*` (raw **27,491 매치**, 23개 테이블, damage 이벤트 1.15억+)
- Funnel (paper §3 표와 일치): raw 27,491 → graph-eligible(N_max=128) **21,824**
- Split (match 단위, seed 42): train 17,460 / val 2,182 / test 2,182
- Target rows: 1,092,572 / 139,685 / 134,564 (aligned test 평가 셀 134,245)
- 도메인: TPP squad 5개 맵 — Baltic 6,069 / Tiger 4,051 / Desert 1,495 / Neon 1,024 / Savage 908 (구 스냅샷 기준)
- 양성률(다음 페이즈 탈락): **~0.32**
- `data/graphs/` 레거시 .pt 약 34GB는 구 파이프라인 산출물 (논문과 무관)

## 5. STDB 설계 (stdb/)

PostgreSQL 18 + PostGIS. 텔레메트리를 3계층으로 추상화:

1. **Moving objects** — 플레이어/팀 중심점의 시계열 위치
2. **Dynamic regions** — 시간에 따라 축소되는 안전구역·블루존
3. **Spatial events** — kill/damage/groggy/item (시각·위치·행위자)

ETL: `etl.py`(기본 변환) → `rq1_feature_etl.py`(phase 피처+leakage-safe 라벨) → `phase_spatial_feature_etl.py`(궤적·노출) → `phase_edge_etl.py`(엣지 캐시).
학습 텐서는 파일 캐시가 아니라 `stdb/queries.py`의 SQL에서 직접 공급됨 (`model_stgcn/dataset.py` → `fetch_match_tensors`) — **STDB 프레이밍은 실질적(load-bearing)**.

⚠ 단, 질의 성능·인덱싱 벤치마크는 없음 → 논문에서 DB-시스템 기여가 아니라 "피처 정의의 형식화/재현 인프라"로 주장 수위 유지 필요 (현재 02절 주석의 방침과 일치).

## 6. 예측 과제

- 단위: (match, team, phase). phase k까지 관측 → phase k+1 탈락 여부 예측 (관측-예측 구간 분리로 누수 차단)
- 텐서: T=8 phase, N_max=128 팀, 노드 피처 66d (tabular baseline 입력은 72 피처), 엣지 3채널 (proximity / combat / co-pressure)
- 평가: AUC + AP + BCE + Brier + ECE(10-bin), match 단위 paired Wilcoxon

## 7. 모델

| 모델 | 설명 |
|---|---|
| Tabular HGB | HistGradientBoosting, 동일 피처 테이블, 그래프 무시 — 주 baseline |
| DynamicSTGNN no-graph | 동일 신경망, 엣지 전부 비활성 — neural control |
| DynamicSTGNN full | causal gated temporal conv + 이질 spatial graph conv, 3채널 엣지. 보조 헤드(final-rank, top-k)는 regularizer로만 사용 |
| (tuned h96) | 랭킹 특화 변형 — ECE 0.1236으로 보정 나쁨, 기본 모델로 쓰지 않기로 내부 결정 |

레거시: 루트 `model/arena_survival_net.py` + `train.py` 라인은 PAPER_PREP_RESEARCH_SUMMARY.md에서 legacy/exploratory로 공식 분리.
**레거시 라인에는 same-step 라벨 누수(사망시각 최근접 반올림 + ±5초 입력 윈도우), 해시 파일명 정렬을 chrono split이라 칭하는 문제, person-period 풀링 C-index 오적용(C-index 0.981은 누수성 수치), dmg_exchange 인덱스 버그(dataset.py:356) 등이 있으므로 논문 수치에 사용 금지.**

## 8. 핵심 결과 (검증 완료)

### 8.1 메인 비교 — full DB, aligned 134,245셀 (6/9–10 재실험; 논문 §7 표와 일치)

| 모델 | AUC | AP | Acc | BCE | Brier | ECE |
|---|---|---|---|---|---|---|
| Tabular HGB | 0.7651 | 0.6626 | 0.7483 | 0.5140 | 0.1708 | 0.0075 |
| no-graph | 0.7681 | 0.6695 | 0.7518 | 0.5090 | 0.1692 | 0.0039 |
| proximity | 0.7741 | 0.6762 | 0.7545 | 0.5046 | 0.1674 | 0.0023 |
| proximity+combat | 0.7755 | 0.6779 | 0.7556 | 0.5036 | 0.1670 | 0.0028 |
| **full graph** | **0.7782** | **0.6808** | **0.7568** | **0.5016** | **0.1662** | **0.0015** |

Match-paired Wilcoxon (n=2,182): full vs no-graph AUC +0.0080 (p=1.17e-30), AP +0.0090 (p=3.12e-23), BCE p=1.63e-47; full vs proximity+combat AUC +0.0022 (p=1.69e-05), AP +0.0028 (p=7.71e-06); full vs HGB AUC +0.0113 (p=2.60e-37), AP +0.0154 (p=9.28e-36). **전 지표 유의.**
출처: `model_stgcn/checkpoints/full_db_reexperiment/phase_result_analysis_edges/`, `model_stgcn/checkpoints/full_db_reexperiment/calibration_edges/`, `docs/FULL_DB_REEXPERIMENT_REPORT.md`

### 8.2 구 스냅샷(5,950매치) 결과 — MODEL_RESULTS.md 기준

- HGB AUC 0.7680/AP 0.6458 vs full 0.7693/0.6423 vs tuned 0.7708/0.6469 (graph 이득이 작았음 — full DB에서 +0.013으로 확대)
- Edge ablation: full vs no-graph ΔAUC +0.0075 (p=5.86e-10); full vs HGB +0.0028 (p=0.0205, rank-biserial 0.097)

### 8.3 Feature ablation (squad_feature_crossmap, n=82,662; additive)

| 피처 구성 | AUC | AP |
|---|---|---|
| prior_scores (5) | 0.6691 | 0.4511 |
| prior_raw (12) | 0.7122 | 0.5236 |
| global_state (30) | 0.7270 | 0.5758 |
| conventional (57) | 0.7575 | 0.6293 |
| conventional+stdb_spatial (72) | 0.7605 | 0.6333 |
| full (76) | 0.7601 | 0.6324 |

- 최대 점프: prior_scores→prior_raw(+0.043 AUC), global_state→conventional(+0.031)
- stdb_spatial 추가 증분 +0.0035 AUC (p=2.5e-10) — 소폭 증분 중 가장 큼. trajectory는 최약(+0.0009, p=0.011)
- full vs prior_scores(선행연구식 표현): match-paired ΔAUC +0.088, ΔAP +0.142 (p≈6e-222) — **"표현 재구성 자체가 가장 큰 이득"이라는 논지의 근거**

### 8.4 강건성 (필수 실험 3종, 6/8 완료 — 구 스냅샷 기준)

- **Seed 반복** (3 seeds × 5 maps): full vs none ΔAUC +0.0084±0.0031, **15/15 양수**; ΔAP +0.0120, 14/15 양수
- **Edge sanity** (node_shuffle ×5): 셔플 시 AUC −0.065 / AP −0.109 / BCE +0.065 (Baltic 0.776→~0.719) — 그래프 이득이 regularization이 아님을 입증
- **Calibration** (5,950 기준): full e12 Brier 0.1705/ECE 0.0076 (tuned는 ECE 0.1236으로 탈락)

### 8.5 Phase별·맵별 (domain_runs_expanded_e12, 구 스냅샷)

- Phase별 AUC: p1 0.791 → p5 0.711 → p7 0.610 (후반일수록 어려움; p6–7 표본 희소)
- 그래프 이득은 초·중반에 집중: ΔAUC p1 +0.0117, p3 +0.0115, p6 +0.0024
- 맵별 AUC: Baltic 0.776 > Tiger 0.761 > Desert/Savage 0.746 > Neon 0.724 (Neon은 유일하게 full < proximity_combat)

## 9. 위성 연구 갈래 (논문과 분리된 강의/탐색 산출물)

### 9.1 H1/H2/H3 가설검증 (Baltic_Main squad 1,000매치; network_analysis + webapp)

| 가설 | 판정 | 핵심 근거 |
|---|---|---|
| H1 외곽 파밍 > 핫존 | **지지** | 순위 12.19 vs 16.68, p=4.2e-232, δ=0.318; 생존시간 보정 후에도 유의 |
| H2 고교전 → 좋은 순위 | **기각 (역방향)** | high combat 순위 19.60 vs low 10.72, δ=−0.630; 중간 교전 그룹이 RP 최고 |
| H3 존버 > 이동 | **기각 (mobile 우세)** | 600초 고정창 통제 후 mobile 7.43 vs sedentary 9.23, p=2.0e-60 |

전략 4유형 (FightScore×ActiveScore): **Hunter**(고교전·고활동) 순위 10.6/RP +1.78 최고 ↔ **Sniper**(고교전·저활동) 순위 19.2 최악.
메시지: "교전성 자체가 아니라 **이동성과 결합된 교전성**이 성과를 만든다."

→ 종속변수(final rank)가 논문(hazard)과 달라 결과 통합 불가. **논문에서는 motivation 한 문단 + supplementary(webapp) 링크로 압축 권장.**

### 9.2 webapp

React 18 + Vite + Cytoscape. 1,000매치 중 30개 층화 샘플 시각화 (네트워크/전략/페이즈/가설 페이지). 논문 demo/supplementary 용도.

## 10. 논문 현황 (2026-06-10 12:37 기준)

| 섹션 | 상태 |
|---|---|
| 00 Abstract | 미작성 (의도적 후순위) |
| 01 Introduction | 작성 완료. 단 변형 4종+placeholder 2종 공존 — 1개로 수렴 필요 |
| 02 Related Work | **미작성 (스켈레톤)** — 최대 리스크 |
| 03 Data & STDB | 완료 (funnel 표, 66 피처, 3 엣지) |
| 04 Prediction Task | 완료 (수식, split, 행 수) |
| 05 Model | 완료 (DynamicSTGNN, ablation 설계) |
| 06 Experiments | **완료 (6/10)** — full DB 프로토콜, aligned 평가, paired Wilcoxon |
| 07 Results & Discussion | **완료 (6/10)** — 표 3개 (메인/paired/calibration), full DB 수치, 보수적 클레임 |
| 08 Threats/Limitations | 스켈레톤 (TODO 목록은 구체적: FPP 미포함, p6-7 희소, survivor bias, 비인과) |
| 09 Conclusion | 스켈레톤 |
| references.bib | **8개뿐, 본문 \cite 거의 없음** |

## 11. 남은 작업 (우선순위)

1. **02 Related Work 작성 + bib 보강** — 필수 인용: Katona et al. "Time to Die" (CoG 2019, 과제상 최근접 선행), Hodge et al. (IEEE ToG 2021, 라이브 승패 예측), STGCN(Yu 2018)/DCRNN, EvolveGCN/DySAT/TGN, DeepHit/nnet-survival, Güting et al. (TODS 2000, moving-object DB), Drachen 계열 spatial game analytics
2. **08·09·abstract 작성** (08은 TODO 목록이 이미 구체적이라 빠름)
3. **구 스냅샷 기반 결과의 처리 결정** — full DB edge ablation은 완료되어 메인 표에 반영됨. feature ablation·seed 반복·edge sanity·phase-wise는 아직 17,977/5,950 기준이므로, 재실행하지 않으면 "auxiliary robustness on an earlier snapshot"으로 명시하고 메인 full-DB 표와 섞지 않는다.
4. **RQ2 처리** — 대응 실험이 없음. RQ에서 제거하거나, phase별 기술통계(자기장 축소에 따른 노출·밀도 변화) 절 추가
5. **Introduction 변형 정리** (6파일 → 1), 서론-기여 진술 방식 통일 (RQ식 vs bullet식)
6. 부수: PROJECT_FLOW_SUMMARY.md의 구 수치(17,977) 갱신, MODEL_RESULTS.md에 seed/sanity 결과 반영, 레거시 라인과의 관계를 논문에서 언급하지 않기로 한 방침 유지

## 12. 종합 판단

- **가장 방어 가능한 기여**: phase-level 이산 hazard 과제 정식화 + 그것을 가능케 한 phase-정렬 데이터 재구성 (BR 장르에서 이 조합의 선행 연구 없음)
- **두 번째 기여**: 동적 팀 관계의 신호 존재 검증 (full DB에서 ΔAUC +0.010~0.013, 전 지표 유의, sanity check 통과) — "모델링 우월성"이 아니라 "신호 검증"으로 포지셔닝 (현재 07절이 이미 이 톤)
- **약점으로 내세우지 말 것**: STDB를 DB-시스템 기여로 주장하는 것, trajectory 피처, tuned 모델
- 상태: **실험·본문 핵심은 완성 단계. 남은 것은 related work/마무리 섹션과 스냅샷 일관성 정리** — 새 연구가 아니라 마감 작업
