# 미래 설계서 — 3D 표상 & 관계-종류(advantage) 엣지 (2026-06-21)

> **상태**: 월요일 발표는 *현재 결과*로 진행. 이 문서의 재설계/재실험은 **발표 후 future work**.
> 핵심 결론은 검증 완료(아래)이나, 전체 재-베이스가 필요해 이번 발표엔 미반영.

## 0. 가장 중요한 확인 — 현재 발표 결과는 z축이 *이미* 반영돼 있음
- **MR 스냅샷 tier에 `pos_z`(고도)가 들어 있음** (`SNAP_FEATURE_DIM=13` 중 3번째, scale 400). 즉 **현재 헤드라인 결과(MR 0.84, resolution, baseline, decision window)는 팀 고도를 이미 보고 있다.** → 발표에서 "우리 모델은 3D 위치를 본다"고 말해도 정직함(스냅샷 tier 기준).
- 누락은 두 군데뿐: **phase-tier 노드 피처**(z 없음, 2D만)와 **그래프 엣지**(등방성 3D 거리만, 상대 고도·고지·LOS 없음). 이건 future work 대상.

## 1. 발견(검증 완료, 단 발표엔 future work으로) — "관계의 *종류*가 그래프 정보성을 가른다"
- 기존 **proximity 위상 = 무신호(평활화)**: observed ≤ edge_shuffle (V3 graph_relation). GNN이 low-pass filter.
- 새 **접촉-게이트 + 상대-우위 엣지 = 유신호**: observed > 모든 null 유의.
- **검증 4중 + 결정적 통제**:
  1. 모델 수준 유의: kNN observed vs edge_shuffle ΔAUC **+0.0033 [+0.0023,+0.0043]** (phase tier, 8000경기).
  2. 모든 null: vs edge_shuffle +0.0033, vs node_shuffle +0.0023, vs no_graph +0.0051 (전부 overall CI 0 제외).
  3. k-ablation 강건: k∈{3,5,8} = +0.0035/+0.0033/+0.0039 전부 유의.
  4. **★node-abs 통제 통과**: phase 노드에 절대 z·alive·health·zone을 직접 줘도(node_dim 120) 우위엣지 여전히 유의 **+0.0023 [+0.0013,+0.0033]**(효과 ~70% 잔존). 절대량 노드 추가는 no-graph를 +0.0009밖에 안 올림.
  → **신호는 노드로 환원 불가능 = irreducibly relational.** 쌍별 "내가 옆 적보다 낮다"는 mean-aggregation이 못 만듦. z-주입 아님.
- **물리 근거(descriptive)**: 근접 팀 쌍에서 *낮은* 팀이 다음 phase에 더 죽음 — 전체 z=+10.5(54~56%), 초·중반 강·후반 약. 교전거리 P50=179m vs 무작위쌍 1901m.

## 2. 기여 프레이밍 (중요 — 정직성)
- ❌ **"2D→3D 확장"은 기여 아님** — 원래 3D여야 할 걸 2D로 한 *설계 결함 수정*(correctness)일 뿐.
- ✅ **진짜 기여 = "관계의 *종류*가 그래프 정보성을 결정한다"** — 상호작용 관계(근접·교전·존압박, 3채널 이종 그래프)는 평활화, 상대 우위는 노드로 환원 불가능한 관계 신호. 그래프ML 일반에 전이.

## 3. 보정 설계 (future work에서 채택)
- **노드 스펙(correctness)**: 116d + 절대 z·alive·health·zone (node-abs를 *기본 베이스*로). `adv_precompute.py` 기본값 = node-abs ON.
- **근접 게이트(임의값 폐기)**: **kNN(관련성, 밀도적응) × 물리 특성간격 ℓ(t)=0.886·R(t)/√N(t)**(Clark-Evans, 수축 점과정) self-tuning. 무차원 d/ℓ. strict(combat)=거리 무관. ablation = k{3,5,8}, β.
- **엣지 = 접촉-게이트 + 상대상태 벡터**, **우위 함수는 edge-conditioned GNN이 학습**(메시지 가중 = MLP(edge_feat)) — "우위"를 손정의 안 함.
- **접촉 게이트**: strict(combat, 입증된 상호인지)·loose(kNN 지각)·full(원래 null+피처). encounter_recently는 combat 파생이라 strict와 중복.

## 4. 재실험 시간 추정 (보정 설계 기준)
- **fast infra**(phase tier, 캐시+N32+MPS): ~1분/모델. **phase-tier 전체 보정-재실험 ≈ 2~3시간.**
- **MR 헤드라인까지 우위엣지 통합**: 현 인프라 ~2일, **MR을 fast infra로 포팅하면 ~반나절**.
- phase별 분석은 재학습 아님(eval 파생) = 0.
- MR main/feature/resolution은 *이미 z 보유* → 설계 보정으로는 재실험 불필요. 재실험 핵심 = phase-tier + 그래프(우위엣지) 실험.
- **⚠️ 디스크 제약(만성)**: 디스크 99~100%(STDB 500GB+). 2026-06-21 전체 재-베이스가 캐시 .pt 동시 쓰기 중 **디스크 풀로 실패**. 재-베이스 전 디스크 정리 필요 + 캐시를 float16/단일-gate/스트리밍으로 경량화(현재 node-abs 통제 결과 +0.0023는 이미 확보돼 재실험 없이도 핵심 증거 보유).

## 5. 코드 자산 (재현/이어가기)
- `model_stgcn/v3/advantage_edges.py` — 융합 게이트(kNN+ℓ(t)) 추출기, `build_adv_tensor`.
- `model_stgcn/v3/adv_model.py` — edge-conditioned GNN(메시지가중=MLP(edge_feat) 학습) + train/eval.
- `scripts/adv_precompute.py` — phase-tier 디스크 캐시(node-abs 기본 ON, N32).
- `scripts/{run_adv_experiment,run_adv_kablation,run_adv_nodeabs,run_adv_base}.sh`, `adv_analysis.py`.
- 결과: `model_stgcn/checkpoints/adv_runs*`, 캐시 `adv_cache*`. DB: `PG_PASSWORD=...`.
- 교전거리/고지 검정: `scripts/highground_signal_test.py`, 타당성 `advantage_edge_feasibility.py`, 위치감사 `audit_spatial_features.py`.

## 6. 발표에서의 위치 (월요일)
- **현재 결과 헤드라인**: MR resolution(0.79→0.85, 천장 돌파), baseline 전 phase 유의, decision window(rank lock-in phase 4–6), **그래프 null = proximity는 평활화**.
- **future work 슬라이드**: "근접 위상은 무신호였지만, *접촉-게이트 상대 우위*를 엣지로 주면 관계 신호가 살아난다(예비 +0.0033, node-abs 통제 통과) → 관계의 종류가 정보성을 가른다" — 이 문서가 그 설계서.
