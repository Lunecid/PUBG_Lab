# PUBG 시공간 그래프 기반 팀 생존 예측 — 통합 설계 문서 (v1)

> **이 문서**: 모델 설계 + 지금까지의 분석 방법론. 결과 수치는 별도 보고용으로 분리.
>
> **연구 제목**: 축소되는 안전구역에서의 동적 시공간 그래프 기반 PUBG 팀 생존 예측
>
> **저자**: 백성은 · 부산대학교 데이터사이언스전문대학원 · 202589713

---

## Context — 왜 이 설계인가

지금까지 STDB 구축 + Phase 단위 trajectory/exposure/graph feature 추출 + archetype clustering 까지 분석 인프라를 완성했다. 1차 모델 설계안의 6가지 누수/정의 문제를 모두 수정한 **v1 설계**다.

**핵심 결정 (수정 사항 포함)**:
1. ⭐ **Hazard target 시점 정의**: `X(phase ≤ t)` → `y(phase t+1 탈락)` — same-phase leakage 제거
2. **N_max = 32** — 26팀 매치 손실 방지 (이전 25)
3. **archetype 은 input 에서 제거** — stratification/해석용으로만 (cut_phase 1/2 누수 방지)
4. **Graph summary feature 를 node feature 에서 제거** — edge ablation 과 중복
5. **Edge 3종 = Proximity + Combat (directed) + Co-pressure** — Trajectory-similarity 는 v2 ablation
6. **Combat edge directed** — damage_dealt vs damage_taken 분리

---

## Part 1 — 지금까지의 분석 (방법론 정리)

### 1.1 데이터 표본
- 100 매치 · Baltic_Main (Erangel) · squad-fpp · 시즌 2026-03~05 (55일)
- 가용 풀: pubg.* 24,294 매치 · 143 GB raw telemetry

### 1.2 STDB 구축
PostgreSQL 18 + PostGIS 3.6.3 (ARM-only 소스 빌드) 위에 `stdb` 스키마 7 테이블:

| 테이블 | 의미 |
|---|---|
| `matches` / `teams` | 매치·팀 메타 |
| `moving_objects` | player + team centroid 시점별 PointZ |
| `dynamic_regions` | safe/poison Polygon + `phase_idx` |
| `spatial_events` | kill / damage / groggy / landing |
| `team_proximity` | 시점별 팀쌍 거리 + encounter 플래그 |

좌표 SRID 0 · cm→m 변환 · GiST(geometry) + BRIN(시간) 인덱스.

### 1.3 Phase 복구
`pubg.telem_phase_changes.elapsed_time` 컬럼이 NULL이라 초기 ETL이 phase_idx를 못 채웠던 문제 → `event_time − match_start_time` 로 elapsed 재계산해 `dynamic_regions.phase_idx` 일괄 UPDATE. 매 매치마다 phase 1~7 시작·종료 시간 확보.

### 1.4 Phase 단위 feature 테이블

PK = `(match_id, team_id, phase_idx)`:

**`stdb.team_phase_trajectory`** — *궤적 모양* (9 feature)
`path_length_m · displacement_m · sinuosity · avg_speed_mps · rotation_timing · safe_alignment · zone_entry_time · outside_safe_duration_s · boundary_distance_slope`

**`stdb.team_phase_exposure`** — *경로 단위 위험 노출* (8 feature)
`mean/min_dist_to_boundary_m · outside_safe_ratio · avg_enemy_density_100m · min_nearest_enemy_m · path_near_enemy_ratio · recent_damage_taken · combat_zone_pass_count`

**`stdb.team_phase_graph`** — *팀 간 관계 측정값* (9 feature)
`avg/max_degree_100m · avg_nearest_enemy_m · proximity_pressure (구 isolation_score) · copress_200m_ratio · combat_edge_count · combat_partner_count · had_combat · survives_next_phase`

> graph 테이블은 **분석/검증용**. **모델 v1 의 node feature 에는 포함하지 않음** (수정 4).

### 1.5 분석/학습용 View
- `stdb.v_team_phase_wide` (1 팀 = 1 row, phase × feature wide pivot)
- `stdb.v_team_truncated_targets` (cut_phase 1~5 별 leakage-safe 라벨)
- 두 view 모두 LightGBM/통계 검정용. ST-GNN 입력은 sequential tensor 별도.

### 1.6 Archetype clustering — 방법론
- 입력: phase 1~3 의 17 trajectory+exposure + 3 graph 메트릭 = 60 컬럼
- 결측 처리: column-median imputation
- 정규화: StandardScaler (학습에 사용 시 train split 에서만 fit)
- K-means k=6 · n_init=20 · random_state=42
- 시각화: PCA 2D (PC1·PC2) 위에 cluster 색·final_rank 색 두 panel
- 산출물: `presentation_build/assets/team_archetypes.csv` (팀당 cluster + archetype 라벨)
- **용도**: stratification 및 해석 용도만. **모델 input 에는 사용 X**

### 1.7 분석 비교 축 (방법론)
- Tier 비교: `final_rank` 4 구간 (top3 / top10 / mid / bot) × phase 별 feature 평균
- Phase 비교: 같은 feature 의 phase 1~6 line chart
- Combat 비교: `had_combat` true/false 두 그룹의 다음 phase 생존 비율
- Archetype × Tier 교차표
- 인터랙티브 시각화: plotly HTML (preview 에 서빙)

---

## Part 2 — 입력 정의 (Input Specification)

### 2.1 시공간 단위 — Phase sequence
- 시간 축: **`T_max = 8`** (phase 1~8 까지 수용; 짧은 끝부분 phase 는 `phase_mask` 로 처리)
- 노드 수: **`N_max = 32`** (26팀 매치 손실 방지)
- 한 (match, phase) 가 한 시점

매우 짧거나 데이터가 거의 없는 마지막 phase (예: phase 7~8) 도 우선 보존. 별도 분석에서 endgame phase 로 병합할지 결정.

### 2.2 Node feature tensor — `X ∈ ℝ^{B × T × N × F}`

**v1 의 F**: graph summary 컬럼 제거. Trajectory + Exposure + basic state 만.

| 그룹 | dim | feature |
|---|---|---|
| trajectory | 8 | path_length · displacement · **log(1+sinuosity)** · avg_speed · rotation_timing · safe_alignment · zone_entry_time · boundary_distance_slope |
| exposure | 8 | mean/min_dist_to_boundary · outside_safe_ratio · avg_enemy_density · min_nearest_enemy · path_near_enemy_ratio · **damage_taken** · **damage_dealt** · path_near_enemy_count |
| basic state | 4 | phase_idx (정규화) · alive_at_phase_start · team_size · vehicle_ratio |

**총 F ≈ 20**. damage_taken/damage_dealt 분리는 combat directed edge 와 정합.

**제외된 것 (의도적)**:
- `archetype` one-hot
- `avg_degree`, `combat_edge_count`, `combat_partner_count` 등 graph summary → 별도 ablation
- `survives_next_phase` (label leakage)

**정규화**: train split 에서만 robust scaler fit → val/test 에 transform 만.
- sinuosity 는 log(1+x) 적용 후 cap
- 절대 거리는 m → /1000 m 스케일

**NULL 처리**: `alive_at_phase_start=False` 인 (phase, team) 은 0 으로 채우고 `node_mask=False`.

### 2.3 Heterogeneous Edge tensor — `A ∈ ℝ^{B × T × E × N × N}`

**v1 의 3종 edge** (`E=3`):

**(1) Proximity edge** — symmetric, 거리 기반
```
weight_ij(t) = exp(− avg_distance_in_phase_t(i,j) / 200m)
```
self-loop 포함, row-normalize per phase.

**(2) Combat edge** — **directed**
- `A_combat_dealt[i,j]` = i 가 j 에게 phase t 동안 가한 누적 damage 의 log(1+x)
- `A_combat_taken[i,j]` = i 가 j 로부터 phase t 동안 받은 누적 damage 의 log(1+x)
- v1: 두 방향성을 두 sub-channel 로 두고 한 GCN 입력에서 concat (E=3 유지)

**(3) Co-pressure edge** — 같은 좁은 영역 압박
```
weight_ij(t) = co-pressure ratio (phase t 내, 200m 안에 i·j 가 동시에 등장한 비율)
```
혹은 같은 안전구역 경계 corridor 에 둘 다 있는 시간 비율.

각 edge channel 은 별도 spatial graph conv pass → 마지막에 학습 가능한 `α_e` 가중 합. v2 에서 `A_traj_sim` 을 추가 ablation 으로 시도.

### 2.4 Masking

| Mask | shape | 의미 |
|---|---|---|
| `node_mask` | (B, T, N) | 매치에 존재 + phase 시작 시점 alive |
| `target_mask` | (B, T−1, N) | 다음 phase hazard 라벨이 정의되는 cell |
| `phase_mask` | (B, T) | phase 가 매치에 실제 존재 |

### 2.5 Sample
- 한 매치 = 한 (X, A, y) 묶음
- Split: **매치 단위** train/val/test (같은 매치 의 팀이 split 가로지르지 않음)

---

## Part 3 — 출력 정의 (Output Specification)

### 3.1 Primary target — Next-phase hazard

```
hazard_next[b, t, i] = P(team i is eliminated during phase t+1
                        | observed through phase t,  alive at phase t end)
```

- Output shape: **(B, T−1, N)**
- X 는 phase ≤ t 만 사용 → same-phase leakage 없음
- `target_mask`: phase t 끝 시점 alive 인 (b, t, i) 만 학습/평가
- 라벨:
  - `last_phase_alive == t` → phase t+1 진입 못 함 → y=1 (탈락)
  - `last_phase_alive >= t+1` → 살아남음 → y=0
  - winner 는 마지막 phase 까지 0 (censored)

손실: BCE-with-logits · `target_mask` 가중 평균.

### 3.2 Auxiliary target #1 — 최종 등수 회귀 `(B, N)`
- L1 loss · `node_mask` 가중
- 라벨 = `final_rank / N_max` (정규화)

### 3.3 Auxiliary target #2 — Top-K 진입 분류 `(B, N, 2)`
- 각 head sigmoid · BCE
- 라벨 = `[rank ≤ 5, rank ≤ 10]`

### 3.4 Multi-task 결합 손실
```
L = α · L_hazard_next  +  β · L_rank_aux  +  γ · L_topK_aux
α=1.0  β=0.3  γ=0.3 (초기값)
```

### 3.5 누수 방지 명시
- 모델은 sequence-to-sequence 이지만 `hazard_next[t]` 계산에 phase>t 정보가 흘러가면 누수
- Temporal conv 의 **causal padding** (kernel=3 → 좌측 padding 2) 강제
- 학습/평가 시 cut_phase 별로도 평가 가능: `cut=k 까지 보고 phase k+1 hazard`

---

## Part 4 — 모델 선택

### 4.1 결정
- **Yu et al. 2018 STGCN 골격 + Dynamic + Heterogeneous A_t**
- 명칭: **Dynamic Spatio-Temporal Graph Neural Network with Heterogeneous Edges** (논문)
- PPT/발표에서는 "ST-GCN" 으로 표기 (Yu 2018 인용)

### 4.2 학술 위치 / 차별점
- Yu STGCN 의 base + (1) 시점별 dynamic A · (2) edge type 3 종 분리 → dynamic + heterogeneous 일반화
- continuous-time graph (TGN/DyRep) 대신 phase 라는 자연 시간 단위 위에서 동작 — 해석성 ↑
- 행동 분류 (Sagong 2025) 와 달리 **공간 환경 + 팀 간 관계 그래프** 중심

### 4.3 Ablation baseline
1. **B1** — Match aggregate (post-match 통계만) → LightGBM
2. **B2** — Phase wide trajectory/exposure (graph 미사용) → LightGBM
3. **B3** — Sequence model, no graph (Temporal Conv only)
4. **B4** — Proximity edge only
5. **B5** — Proximity + Combat
6. **★ Proposed** — Proximity + Combat + Co-pressure
7. **Optional** — Proposed + Trajectory-similarity (v2)

→ Edge 추가에 따른 점진적 기여를 ablation 으로 분리.

---

## Part 5 — 모델 아키텍처

```
Input
  X  : (B, T=8, N=32, F≈20)        # node features
  A  : (B, T, E=3, N, N)            # heterogeneous edges (combat 는 dealt/taken 두 sub-channel)
  masks : node_mask, target_mask, phase_mask

────────────────────────────────────────────────
[1] Node embedding
        X → Linear(F → 64) → ReLU → Dropout(0.1)

[2] ST-Conv Block × 2  (각 block: TGated → HeteroSpatial → TGated)
    
    Block (C_in → C_hid → C_out):
      ├─ Temporal Gated Conv 1D
      │     kernel=3, causal padding (좌측 only)
      │     GLU activation
      │
      ├─ Heterogeneous Spatial GCN (per timestep)
      │     For each edge type e ∈ {prox, combat, copress}:
      │         H_e = A^e_t · X · W_e    (dense, N≤32 라 가볍게)
      │     Combine:  Σ_e softmax(α_e) · H_e  →  LayerNorm + ReLU
      │     α_e ∈ ℝ^E 학습 가능 (해석성)
      │
      └─ Temporal Gated Conv 1D (causal)
    
    Block 1: 64 → 128 → 64
    Block 2: 64 → 128 → 64
    Residual + LayerNorm

────────────────────────────────────────────────
[3] Three heads
    
  (a) hazard_next       (B, T−1, N)   # primary
      h_t = MLP(h^L_t)  for t = 0..T−2
      (hazard_next[t] 는 hidden[t] 만 사용 → causal 보장)
  
  (b) final-rank (aux)  (B, N)
      attention-pool over T → MLP → 1
  
  (c) Top-K (aux)       (B, N, 2)
      같은 pool 결과 → MLP → 2
```

### 5.1 파라미터 추정
- Node embedding: ≈ 1.3 K
- 2 × ST-Conv Block: ≈ 100 K
- 3 heads: ≈ 10 K
- **총 ≈ 0.12 M**

### 5.2 Edge-type 가중 학습
`α_e` 는 학습 후 softmax 값을 보고 **어떤 edge type 이 hazard 예측에 가장 기여**하는지 해석.

---

## Part 6 — 학습 setup

### 6.1 Split
- Match-level split: train 80 / val 10 / test 10 (fixed seed=42)
- Scaler 는 **train 에서만 fit**
- 365 매치 확장 시 비율 유지

### 6.2 Optimizer
- AdamW · lr=1e-3 · weight_decay=1e-4
- Cosine schedule · 50 epoch · early stop on val L_total

### 6.3 Hyperparameter
- `C_emb=64, C_hid=128, num_blocks=2, dropout=0.1, batch=8`
- `α=1.0, β=0.3, γ=0.3`

### 6.4 Cut-phase augmentation (선택)
- 학습 시 매치마다 cut_phase k ∈ {1..T−1} 무작위 → phase ≤ k 의 X 만 보이게 mask
- 데이터 증강 + 누수 분리 명시화
- v1 은 기본 mode 만, v2 에 augmentation

### 6.5 Masking
모든 손실은 `target_mask`(hazard), `node_mask`(rank/topK) 가중 평균.

---

## Part 7 — 평가 (방법론)

### 7.1 평가 카테고리 (5종)
| 카테고리 | 측정 방법 |
|---|---|
| 생존 (hazard) | time-dependent AUC · C-index · Brier · phase-wise calibration |
| 등수 (auxiliary) | Spearman ρ · Kendall τ · MAE of final rank |
| Top-K (auxiliary) | ROC AUC · F1 · Precision@K |
| Ablation | B1~B5 vs Proposed · 같은 split, 같은 metric |
| Explanation | 학습된 `α_e` softmax · feature permutation importance |

### 7.2 통계 검정
- B2(LightGBM wide) vs Proposed 의 paired 비교 → **Wilcoxon signed-rank** at match level
- Archetype 간 hazard 분포 → Kruskal-Wallis
- Tier × Trajectory feature → Mann-Whitney U
- 모든 검정에 **match-level clustering** 또는 **cluster bootstrap**

### 7.3 데이터 누수 명시
모든 평가 표에 **cut_phase k**, **target = hazard_next at phase k+1** 형식으로 라벨 명시.

---

## Part 8 — 데이터 표본 및 한계

| 차원 | 현재 | 1차 확장 | 완전 표본 |
|---|---|---|---|
| 매치 수 | 100 | 365 (Baltic squad-fpp 전체) | 24,194 |
| ETL 예상 시간 | (완료) | ~45분 | ~40시간 |
| 통계 검정력 | 적당 | 매우 ↑ | 메타 robust |

**현 표본의 한계 (방법론 측면)**:
- 단일 맵·단일 모드 (Erangel · 4인 squad · FPP)
- 단일 시즌 (55일)
- 같은 매치 내 팀들은 *독립 표본 아님* → 통계 검정에서 match-level clustering / cluster bootstrap

---

## Part 9 — 구현 단계 + 파일 변경

### 9.1 사전 sanity
- `team_phase_trajectory` vs `team_phase_exposure` row 수 차이 원인 확인
- `dynamic_regions.phase_idx` 미복구 NULL row 원인 확인

### 9.2 신규/수정 파일
| 파일 | 작업 |
|---|---|
| `stdb/queries.py` | **수정** — `fetch_phase_match_tensors(match_id)` 추가 |
| `stdb/queries.py` | **추가** — co-pressure / combat-directed edge 계산 함수 |
| `model_stgcn/dyn_stgnn.py` | **신규** — `DynamicSTGNN` 클래스 |
| `model_stgcn/dataset.py` | **수정** — phase-level Dataset · 3 mask · train scaler |
| `model_stgcn/train.py` | **수정** — multi-task loss |
| `model_stgcn/eval.py` | **수정** — 5 카테고리 metric · cluster-bootstrap |
| `model_stgcn/baselines.py` | **신규** — B1~B5 baseline |
| `model_stgcn/stats.py` | **신규** — Wilcoxon / Mann-Whitney / Kruskal · match-clustered bootstrap |
| `stdb` 컬럼 rename | **수정** — `isolation_score` → `proximity_pressure` |

### 9.3 의존성
- `lightgbm` (B1·B2 baseline)
- `scipy.stats` (이미 있음)
- `lifelines` (선택, time-dependent AUC / C-index 표준)

### 9.4 실행 순서
1. 사전 sanity (9.1) — 데이터 일관성 확인
2. 컬럼 rename — `isolation_score` → `proximity_pressure`
3. Combat directed / co-pressure edge SQL — phase 단위 매트릭스 계산 + 캐시
4. `fetch_phase_match_tensors` — shape 검증 (B=1 매치 sample)
5. `DynamicSTGNN` forward 검증 — 랜덤 입력 → 출력 shape 확인
6. Over-fit smoke test — 5 매치로 train loss → 0
7. Baseline 구현 (B1~B5) — 같은 split 에서 학습
8. Full training (100 매치) + Ablation 표
9. 통계 검정 — match-clustered bootstrap

### 9.5 End-to-End 검증 절차 (방법론적 기준)
- Phase trajectory similarity / co-pressure 매트릭스 sanity: self-pair 와 random-pair 의 값이 직관과 일치
- DynamicSTGNN forward shape: `hazard_next (B, T−1, N)` / `rank (B, N)` / `topK (B, N, 2)`
- Mask: `target_mask=False` 인 cell 의 hazard 손실이 0
- Causal check: shuffle phase>t 의 X → hazard[t] 가 변하지 않는지 (변하면 누수)
- Train scaler: val/test 에 fit_transform 하지 않고 transform 만
- Ablation 정합: 모든 baseline 이 동일 split / 동일 라벨 / 동일 metric

수치적 결과 (AUC / MAE 등) 는 측정 후 별도 보고. 본 설계에는 포함하지 않음.

---

## Part 10 — Quick reference

```
입력  X (B, T=8, N=32, F≈20)   trajectory + exposure + basic state (archetype/graph-summary 제외)
      A (B, T, E=3, N, N)       Proximity / Combat(directed) / Co-pressure
      masks node/target/phase

모델  DynamicSTGNN
      ├─ Node embedding   F→64
      ├─ ST-Conv Block ×2 (Temporal causal Gated → Hetero Spatial GCN → Temporal causal Gated)
      └─ 3 heads:  hazard_next (B,T−1,N) · rank (B,N) · topK (B,N,2)

출력  hazard_next (primary, BCE · X up to phase t → y phase t+1)
      + final rank (auxiliary, L1)
      + top-K (auxiliary, BCE)

학습  AdamW lr=1e-3 · 50 epoch · α=1 β=0.3 γ=0.3 · match-level split · train-only scaler

평가  5 카테고리: 생존·등수·Top-K·Ablation(B1~B5)·Explanation(α_e)
      통계: Wilcoxon (paired) + match-clustered bootstrap
```

---

## Part 11 — 미반영 사항 / 후속

- **archetype 모델 통합**: v1 input 에 미포함. 결과 분석 시 archetype 별 hazard·rank 분포 보고는 유효.
- **Trajectory-similarity edge**: v2 ablation 으로 보존.
- **데이터 확장**: 100→365매치 (단일 맵·모드 전체) 는 v1 학습 dry-run 후 진행.
- **다른 맵/모드 일반화**: v2 이후 단계.

문서 끝.
