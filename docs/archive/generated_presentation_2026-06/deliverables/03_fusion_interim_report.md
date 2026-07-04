# 통합 중간 보고서 — Factorized 다해상도 시공간 표상 + Next-phase Hazard

**작성일: 2026-06-21 (월요일 발표 직전) · 석사 랩세미나 / 지도교수용**
**범위: 그래프 축(C3) + 시공간 축(C1+C2)을 하나의 hazard 프레임으로 융합 (STDB ⊕ 동적 팀 그래프)**

> 모든 헤드라인 수치는 V3 full-DB, squad, 공유 split(test 1,521매치 / 77,199 at-risk team-phase 셀), match-level cluster bootstrap(1000 reps) 기준. 예측 단위 = cut_phase별 trajectory(매치당 1예측 아님). 단일 진실 출처: `docs/PRESENTATION_DATA_2026-06-21.md`. 5장 FUTURE WORK의 예비 수치 출처: `docs/FUTURE_DESIGN_3D_RELATIONAL_2026-06-21.md`.

---

## 1. 현황 한 문단 (2026-06-21, 월요일 발표 직전)

월요일 발표는 **현재 검증 완료된 결과만**으로 진행한다. 핵심 줄기는 세 가지로 수렴했다. (1) **시공간 — 천장 돌파**: phase 집계로 만든 hazard AUC 천장은 약 0.79인데(tabular GBM HGB 0.7864 ≈ phase-GNN phase_rgcn 0.7926), 10초 다해상도(MR) 스냅샷을 넣으면 mr_rgcn 0.8467 / mr_gat 0.8496까지 올라가며, 이 **0.79→0.85 (+0.054)** 차이가 강한 tabular 베이스라인 대비 전 phase에서 유의하다(ALL ΔAUC +0.063 [+0.060,+0.067]). (2) **그래프 — 정직한 음성(honest negative)**: 동적 팀 그래프가 주는 이득은 mean-field 평활화일 뿐이고 관측된 *위상*은 무의미하다(observed 0.845 ≤ edge_shuffle 0.847, 3/3 시드, null-model로 분리). 즉 GNN은 low-pass filter로 작동한다. (3) **융합 — decision window**: 최종 생존 서열은 게임 16–24분(존 phase 4→6)에 유의하게 굳는다(rank lock-in 3→6 +0.085 [+0.070,+0.101]). 이 셋을 묶는 공통 프레임이 **next-phase hazard 재정식화(C1)**다. 아울러 발표 후 future work으로, "3채널 이종 그래프(근접·교전·존압박)의 위상은 무신호였지만 *접촉-게이트 상대 우위*를 엣지로 주면 관계 신호가 살아난다(예비 +0.0033, node-abs 통제 통과)"는 방향을 별도 슬라이드/절로만 제시한다 — 월요일 헤드라인이 아니다.

---

## 2. Thesis — Factorized 다해상도 시공간 표상 + Next-phase Hazard

> **Next-phase 팀 탈락 hazard를 예측하기 위해, phase 집계가 버리는 시간 구조를 복원하는 factorized 다해상도 시공간 표상이 필요하다. 시공간 축(C1 hazard 재정식화 + C2 STDB·MR 다해상도 스냅샷)이 예측의 본질을 담당하고, 그래프 축(C3 동적 팀 그래프 위상의 null-test)은 그 표상 위에서 관계 위상이 무엇을 더하는지(혹은 더하지 않는지)를 정직하게 가른다 — 셋을 하나의 hazard 프레임으로 통합한다.**

배틀로열 생존 예측을 두 개의 직교 축으로 *factorize*한다. 각 축이 hazard에 기여하는 바를 독립적으로 측정하면, 무엇이 진짜 신호이고 무엇이 평활화인지 분리할 수 있다.

- **축 A — 시공간 (C1 hazard + C2 STDB/MR)**
  - **C1 (hazard 재정식화)**: 최종 순위를 사후 통계로 보지 않고, 매 phase 누적되는 탈락 위험(hazard)으로 재정식화. 예측 단위 = **cut_phase별 trajectory**(매치당 1예측 아님). 이 재정식화가 phase 절단점마다 "지금까지 본 정보로 다음 국면 생존이 어떻게 갈리나"를 측정 가능하게 만들고, 그래프 축·시공간 축·rank를 모두 하나의 좌표계에 올린다.
  - **C2 (STDB + 다해상도)**: phase 집계(coarse)와 10초 스냅샷(fine)을 별도 tier로 두는 **factorized multi-resolution(MR)**. STDB(squad 실측 시공간 DB)로 phase↔실제 게임 시각을 정렬해 결과를 e-sports 국면 언어로 해석한다.
- **축 B — 그래프 (C3, null-test)**
  - **C3 (관계 null 검증)**: 동적 팀 그래프(proximity/combat/copressure)를 GNN에 태우되, 그 이득이 **위상(topology)에서 오는지** 아니면 단순 **평활화에서 오는지**를 세 null 대조군(edge_shuffle / node_shuffle / no_graph, 전부 row-normalized로 위상만 측정)으로 분리한다. 이 null-test가 "그래프가 도움이 된다"는 통념을 정직하게 깨는 방법론적 기여다.

요약: **시공간 축이 성능을 책임지고(0.79→0.85), 그래프 축이 정직성을 책임진다(위상=평활화).** 두 축의 결론이 다른 방향(positive / honest-negative)인 것이 오히려 factorization이 작동했다는 증거다.

---

## 3. CURRENT 핵심 발견 (수치)

### (a) MR 다해상도가 phase 천장을 돌파한다 — 전 phase 유의 (그림 1·4)

phase 집계 표상은 모델 종류와 무관하게 약 0.79에서 천장에 닿는다: HGB(tabular GBM) 0.7864, phase_tcn(graph X) 0.7872, phase_rgcn(graph O) 0.7926. 즉 phase 집계 운동장에서는 GNN ≈ GBM이다. **10초 MR 스냅샷이 이 천장을 깬다**: mr_tcn 0.8349, mr_rgcn **0.8467**. phase_rgcn 0.7926 → mr_rgcn 0.8467로 **약 0.79 → 0.85 (+0.054)** 돌파. Main 8-모델 매트릭스 최고치는 mr_gat 0.8496.

| 모델 | 입력 | hazard AUC | rank Spearman |
|---|---|---|---|
| HGB (tabular GBM) | phase 피처 | 0.7864 | — |
| phase_tcn | phase only (graph X) | 0.7872 | 0.6450 |
| phase_rgcn | phase only (graph O) | 0.7926 | 0.6521 |
| **mr_tcn** | **MR 10s 스냅샷** | **0.8349** | 0.7377 |
| **mr_rgcn** | **MR 스냅샷 + graph** | **0.8467** | 0.7596 |

강한 tabular 베이스라인(HGB 0.7864) 대비 V3(mr_gat 0.8496)는 **전 phase에서 유의**하다:

| phase | HGB | V3 | ΔAUC [95% CI] |
|---|---|---|---|
| 1 | 0.825 | 0.893 | +0.068 [+0.063,+0.073] |
| 2 | 0.748 | 0.835 | +0.087 [+0.078,+0.095] |
| 3 | 0.725 | 0.800 | +0.075 [+0.067,+0.083] |
| 4 | 0.756 | 0.815 | +0.059 [+0.052,+0.066] |
| 5 | 0.742 | 0.805 | +0.063 [+0.053,+0.072] |
| 6 | 0.720 | 0.792 | +0.071 [+0.055,+0.086] |
| 7 | 0.655 | 0.741 | +0.087 [+0.059,+0.113] |
| **ALL** | **0.786** | **0.850** | **+0.063 [+0.060,+0.067]** |

vs LR(0.7540): ALL +0.096. MR-era 보강: MR vs HGB Wilcoxon **p=2.5e-144**. **match-level paired 유의 (전 cut_phase)**: MR vs phase(둘 다 graph) ΔAUC +0.048~+0.076, ΔSpearman +0.11~+0.16 — 7개 phase 전부 CI에서 0 제외; graph 없는 쌍에서도 ΔAUC +0.041~+0.090로 7개 phase 전부 유의. rank Spearman이 특히 극적: phase_rgcn 0.51~0.63 → mr_rgcn 0.71~0.76(스냅샷이 순위 예측에 결정적; time-shuffle 시 0.51 붕괴).

→ **그림 1 (`docs/presentation_figures/fig1_resolution_ladder.png`)**: phase 집계 천장(약 0.79) → MR 스냅샷 돌파(0.85)의 해상도 사다리.
→ **그림 4 (`docs/presentation_figures/fig4_hazard_vs_baseline_ci.png`)**: phase별 V3 vs HGB ΔAUC와 95% CI(전 phase 유의).

### (b) 그래프 위상 = 신호 없음, 평활화일 뿐 (honest negative, 그림 2)

세 null 대조군 모두 row-normalized(위상만 측정), 3-seed 평균 hazard AUC:

- no_graph 0.834 · observed 0.845 · **edge_shuffle 0.847** (observed보다 일관되게 높음, 3/3 시드)
- node_shuffle ≈ observed (팀 신원 무관)
- 채널: proximity 0.8455 · combat 0.837 · copressure 0.837 · prox+combat 0.846

match-level paired (observed − edge_shuffle):
- phase 1–3: observed가 **유의하게 나쁨** (ΔAUC −0.002~−0.004, ΔSpr −0.019~−0.025, CI<0)
- phase 4–7: ns — **어느 phase도 유의하게 좋지 않음** → 엔드게임 위상 가설도 기각
- observed vs no_graph: phase 1–6 유의(+0.008~+0.015)

결론: 동적 팀 그래프가 주는 이득은 mean-field **평활화**이며, **관측된 위상 자체는 무의미**하다(observed ≤ shuffle). GNN은 사실상 low-pass filter로 작동한다. 이는 honest negative이자, null-model로 신호와 평활화를 분리해 낸 방법론적 결과다(C3의 정직한 음성 결론).

→ **그림 2 (`docs/presentation_figures/fig2_graph_null.png`)**: no_graph / observed / edge_shuffle / node_shuffle 대조 — observed ≤ shuffle, GNN=low-pass filter.

### (c) Decision window = phase 4–6 = 게임 16–24분 (rank lock-in, 그림 3)

최종 생존 서열은 특정 구간에서 굳는다. ΔSpearman(phase_j − phase_i), match-level CI:

- **3→6: +0.085 [+0.070,+0.101] 유의** (최종순위 구조가 이 구간에서 굳음 — 핵심 융합 결론)
- 3→5: +0.073 [+0.059,+0.087] 유의 · 4→6: +0.047 [+0.031,+0.061] 유의
- 5→6: +0.013 [−0.002,+0.025] ns (plateau) · 6→7: −0.028 [−0.053,−0.003] 유의 하락(소수 risk-set)

이 구간을 squad ST-DB 실측 게임 시각에 사상하면 phase 4–6 = 게임 **16–24분**이다:

| phase | 시각(시작→끝) | safe 반경 | 생존팀 | e-sports 국면 |
|---|---|---|---|---|
| **4** | **16.3→19.1분** | **659m** | **~9** | 후반 구도 진입 |
| **5** | **19.3→21.7분** | **364m** | **~6** | 후반 로테이션 압박 |
| **6** | **21.9→23.6분** | **188m** | **~4** | 엔드게임 setup |

rank Spearman도 이 구간에서 최고치를 찍는다(phase 5 0.800 [0.788,0.812], phase 6 0.813 [0.797,0.826]). 시공간 축(언제 정보가 결정적인가)과 rank 축(언제 서열이 굳는가)이 정확히 같은 구간을 가리킨다. 이것이 두 축을 hazard 프레임으로 묶었을 때 나오는 융합 발견이다.

→ **그림 3 (`docs/presentation_figures/fig3_rank_lockin.png`)**: phase 간 rank lock-in(ΔSpearman)과 16–24분 결정 구간.

---

## 4. 3과목 관점의 융합 (표)

세 과목 관점이 서로 다른 축을 검증하며 하나의 hazard 프레임으로 합쳐진다.

| 과목 / 산출물 | 담당 축 | 핵심 역할 | 헤드라인 발견 (수치) | 참조 figure |
|---|---|---|---|---|
| **그래프데이터분석** | 그래프 축 (C3) | **null 검증 — 정직성** | 동적 팀 그래프 위상 무의미: observed 0.845 ≤ edge_shuffle 0.847(3/3 시드); phase 1–3 observed 유의 열세, phase 4–7 ns → 위상 가설 기각, GNN=low-pass filter | 그림 2 |
| **시공간데이터분석** | 시공간 축 (C1+C2) — 논문 본체 | **성능 논문 — 천장 돌파** | MR 10s 스냅샷이 phase 천장 0.79→0.85; ALL ΔAUC(V3−HGB) +0.063 [+0.060,+0.067], 전 phase 유의; MR vs HGB p=2.5e-144 | 그림 1·4 |
| **랩세미나** | 축 A + 축 B (통합·방법론) | **융합 — decision window** | hazard 재정식화(C1)가 공통 프레임; rank lock-in 3→6 +0.085 [+0.070,+0.101] = 게임 16–24분(phase 4–6) | 그림 3 |

요지: **그래프데이터분석은 "관계 위상은 신호가 아님"을 정직하게 증명(negative), 시공간데이터분석은 "시간 해상도가 본질"을 입증(positive), 랩세미나는 둘을 hazard와 decision window로 통합**한다. 같은 split·같은 평가 프로토콜·같은 hazard 단위를 공유하므로 세 축의 결론이 한 표상 위에서 정합하며, 메시지가 충돌하지 않고 분업한다.

---

## 5. FUTURE WORK 로드맵

> **위치 명확화**: 아래는 전부 **월요일 발표 후 future work**이다. 월요일 헤드라인(MR 0.79→0.85, baseline 전 phase 유의, decision window, 그래프 null=평활화)에는 포함되지 않으며 발표에서는 **별도 "Future Work" 슬라이드/절**로만 제시한다. 상세 설계는 `docs/FUTURE_DESIGN_3D_RELATIONAL_2026-06-21.md` 참조.

### 5.1 핵심 재프레이밍 — "관계의 *종류*가 그래프 정보성을 가른다"

C3의 honest negative는 끝이 아니라 *질문*이다. 3채널 이종 그래프(근접·교전·존압박)의 위상이 무신호였던 이유는, 이 채널들이 모두 *상호작용(존재·강도)*만 담고 *상대적 우위*는 담지 않아 mean-aggregation으로 환원 가능한 관계이기 때문이다. 가설: **접촉-게이트 + 상대-우위(advantage) 엣지**처럼 노드로 환원 불가능한 관계를 주면 그래프 신호가 살아난다.

- **검증 완료 (단 발표엔 future work)**:
  - 모델 수준 유의: kNN observed vs **edge_shuffle ΔAUC +0.0033 [+0.0023,+0.0043]** (phase tier, 8000경기).
  - 모든 null 통과: vs edge_shuffle **+0.0033**, vs node_shuffle **+0.0023**, vs no_graph **+0.0051** (전부 overall CI 0 제외).
  - k-ablation 강건: k∈{3,5,8} = +0.0035 / +0.0033 / +0.0039 전부 유의.
  - **★결정적 통제 — node-abs 통과**: phase 노드에 절대 z·alive·health·zone을 직접 줘도(node_dim 120) 우위엣지가 여전히 유의 **+0.0023 [+0.0013,+0.0033]** (효과 ~70% 잔존). 절대량 노드 추가는 no-graph를 +0.0009밖에 못 올림 → **신호는 노드로 환원 불가능 = irreducibly relational**. 쌍별 "내가 옆 적보다 낮다"는 mean-aggregation이 못 만듦, 단순 z-주입이 아니다.
- **물리 근거 (descriptive)**: 근접 팀 쌍에서 *낮은* 팀이 다음 phase에 더 죽음 — **고지 z=+10.5** (54~56%, 초·중반 강·후반 약). 교전거리 P50=179m vs 무작위쌍 1901m.

### 5.2 정직성 — 무엇이 기여이고 무엇이 아닌가

- ❌ **"2D→3D 확장"은 기여가 아니다** — 원래 3D여야 할 것을 2D로 둔 *설계 결함 수정*(correctness)일 뿐.
- ✅ **진짜 기여 = "관계의 종류가 그래프 정보성을 결정한다"** — 상호작용 관계(근접·교전·존압박)는 평활화, 상대 우위는 노드 환원 불가능한 관계 신호. 그래프ML 일반으로 전이되는 주장.
- 참고(현재 발표의 정직성): **현 MR 헤드라인 결과는 이미 z축(고도)을 본다** — MR 스냅샷 tier에 `pos_z`(SNAP_FEATURE_DIM=13 중 3번째, scale 400)가 포함되므로, 발표에서 "우리 모델은 3D 위치를 본다"고 말해도 정직하다. 누락은 **phase-tier 노드 피처**(z 없음, 2D)와 **그래프 엣지**(등방성 3D 거리만, 상대 고도·고지·LOS 없음) 두 군데뿐 → 이 둘이 future work 대상이다.

### 5.3 보정 설계 (future work 채택안)

- **노드(correctness)**: 116d + 절대 z·alive·health·zone을 *기본 베이스*로(`adv_precompute.py` 기본 node-abs ON, N32).
- **근접 게이트(임의값 폐기)**: **kNN(밀도적응) × 물리 특성간격 ℓ(t)=0.886·R(t)/√N(t)** (Clark-Evans, 수축 점과정) self-tuning. 무차원 d/ℓ. strict(combat)=거리 무관. ablation = k{3,5,8}, β.
- **엣지 = 접촉-게이트 + 상대상태 벡터**, **우위 함수는 edge-conditioned GNN이 학습**(메시지 가중 = MLP(edge_feat)) — "우위"를 손으로 정의하지 않음.
- **접촉 게이트**: strict(combat, 입증된 상호인지) · loose(kNN 지각) · full(원래 null+피처).

### 5.4 재실험 시간 추정 (보정 설계 기준)

- **fast infra**(phase tier, 캐시+N32+MPS): ~1분/모델 → **phase-tier 전체 보정-재실험 ≈ 2~3시간**.
- **MR 헤드라인까지 우위엣지 통합**: 현 인프라 ~2일, **MR을 fast infra로 포팅하면 ~반나절**.
- phase별 분석은 재학습 아님(eval 파생) = 0. MR main/feature/resolution은 *이미 z 보유* → 설계 보정만으로는 재실험 불필요. 재실험 핵심 = **phase-tier + 그래프(우위엣지)**.
- 코드 자산: `model_stgcn/v3/advantage_edges.py`, `model_stgcn/v3/adv_model.py`, `scripts/adv_precompute.py`, `scripts/run_adv_*.sh`, `scripts/highground_signal_test.py`.

### 5.5 액션 아이템 (논문 미커밋 · 헤드라인 재프레이밍)

- **논문 미커밋**: 현 시점 V3 논문 결과·코드는 아직 메인에 commit 전(작업 브랜치). 월요일 발표 전 정본·split·체크포인트를 동결해 재현성을 확보하고, future work는 별도 절로 분리한 상태로 정리.
- **헤드라인 재프레이밍 검토**: 현재 헤드라인 = "MR 천장 돌파(0.79→0.85, ALL ΔAUC +0.063) + 그래프 null=평활화". future work 확정 시 상위 내러티브를 "정보 해상도가 천장을 깬다 + *관계의 종류*가 그래프 정보성을 가른다"로 재프레이밍 가능 — 단, **현재 발표에서는 하지 않는다**(예비 수치이므로). 논문 06·07 섹션 수치를 V3로 동기화 필요.

---

## 부록 — 참조 figure (4개 전부)

- **그림 1** `docs/presentation_figures/fig1_resolution_ladder.png` — Resolution ladder: phase 천장 0.79 → MR 스냅샷 0.85 (§3a).
- **그림 2** `docs/presentation_figures/fig2_graph_null.png` — Graph null: observed 0.845 ≤ edge_shuffle 0.847, GNN=low-pass filter (§3b).
- **그림 3** `docs/presentation_figures/fig3_rank_lockin.png` — Rank lock-in: phase 3→6 +0.085, decision window 16–24분(phase 4–6) (§3c).
- **그림 4** `docs/presentation_figures/fig4_hazard_vs_baseline_ci.png` — Hazard vs tabular baseline, per-phase 95% CI, 전 phase 유의(ALL +0.063) (§3a).

---

*CURRENT 수치는 `docs/PRESENTATION_DATA_2026-06-21.md`에서, FUTURE WORK 예비 수치는 `docs/FUTURE_DESIGN_3D_RELATIONAL_2026-06-21.md`에서만 인용. 시트에 없는 값은 산출하지 않았다.*
