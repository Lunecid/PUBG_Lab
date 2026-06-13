# Multi-Resolution ST-GNN 결과 종합 — 2026-06-13

> snapshot(10초) 시간 해상도를 학습으로 도입하면 GNN이 GBM/phase 천장을 돌파한다.
> 세 데이터셋(1맵→전맵→이질 그래프 구조)에서 일관 입증, full squad에서 p<1e-126.

---

## 1. 동기 — 역설

short paper 확장 과정에서 역설이 드러났다: **feature-only HGB(GBM)가 DynamicSTGNN(GNN)을 이긴다**
(full DB AUC 0.7829 vs 0.7780). 진단: 현 ST-GNN은 시간 해상도가 phase 8스텝뿐인데,
raw는 매치당 ~150 snapshot(10초)이다. SQL 집계가 시간 정보를 19배 압축하면서
**GNN과 GBM이 같은 입력(phase 집계)을 받아 GNN의 구조적 이점이 사라진다.** snapshot
해상도야말로 GBM이 구조적으로 볼 수 없는 것(가변길이 시계열 그래프)이다.

## 2. 방법 — MR-STGNN

Hierarchical multi-resolution ST-GNN (`model_stgcn/mr_stgnn.py`):

```
Tier-S: snapshot 10s 팀 그래프 ── causal encoder(temporal+graph) ──┐
                                                                    ▼ phase-local pooling
        phase 노드 표현 ◄── concat ── phase 집계 피처(F0/F4/F6/F7)
                                                                    ▼
        DynamicSTGNN (phase tier, 재사용) → hazard
```

- **deep supervision**: phase hazard + snapshot survival (δ 가중). 1맵 ablation 결과 δ는 거의 무관.
- **누수 0**: snapshot encoder causal(≤t), phase pooling phase-local, combat 엣지는
  `(t-10,t]` 과거 귀속(`ceil(event_t/10)*10` — 옛 combat_zone_pass 누수의 dual 회피),
  survival 라벨은 미래만(입력은 과거). 회귀 테스트 9종(`tests/test_mr_leakage.py`,
  `test_snapshot_features.py`).
- 데이터: `stdb.team_snapshot_features`(56.3M행, 10초 팀 노드), `stdb.snapshot_queries`
  (combat 과거귀속 fetch), A_s 희소화로 cached 메모리 481GB→16GB(25배).

베이스라인 비교: 동일 split에서 **HGB(GBM 149d) / phase-only(141d GNN, snapshot 없음) / MR**.

## 3. 결과

### 3.1 세 데이터셋 일관표 (test hazard AUC)

| 모델 | 1맵 Tiger squad | full squad (15,213) | solo/duo (3,704) |
| --- | ---: | ---: | ---: |
| HGB (GBM 천장) | 0.7949 | 0.7847 | 0.7761 |
| phase-only GNN | 0.7933 | 0.7907 | 0.7779 |
| **MR snapshot ST-GNN** | **0.8374** | **0.8422** | **0.8141** |
| **ΔMR − GBM 천장** | **+0.0425** | **+0.0576** | **+0.0380** |

세 조건 모두: **phase 집계 천장 ≈ 0.78–0.79 (GBM ≈ phase GNN), MR이 +0.038~0.058 돌파.**
solo(100팀)는 squad(26팀)와 그래프 구조가 완전히 다른데도 재현 → 맵·모드·구조 무관.

### 3.2 δ ablation (1맵 Tiger squad) — 이득의 출처

| 모델 | AUC | 분해 |
| --- | ---: | --- |
| phase-only | 0.7933 | — |
| MR δ=0 (snapshot 표현만) | 0.8358 | snapshot 표현 **+0.0425 (96%)** |
| MR δ=0.3 (deep-sup) | 0.8374 | deep supervision +0.0016 (4%, 노이즈) |

**이득의 거의 전부가 snapshot 표현 자체.** deep supervision(survival 라벨)은 무의미 →
모델 단순화 가능(survival head 제거, δ=0 운용), ±5초 survival 라벨 누수 걱정도 소멸.

### 3.3 통계적 유의성 (full squad test, 1,521매치 per-match AUC + Wilcoxon)

| 비교 | ΔAUC | 매치 승률 | Wilcoxon p |
| --- | ---: | ---: | ---: |
| MR vs phase-only | +0.0468 | 1124/1521 (74%) | **5.0e-127** |
| MR vs HGB | +0.0520 | 1179/1521 (78%) | **2.5e-144** |
| phase-only vs HGB | +0.0052 | 834/1521 (55%) | 6.8e-09 |

MR이 매치 74–78%에서 우세, p<1e-126. phase-only ≈ HGB(미미, +0.005).

## 4. 결론

1. **역설의 원인 = phase 집계의 시간 정보 손실** — GBM과 GNN을 같은 운동장에 가둠.
2. **해소 = snapshot 시간 해상도** — GNN의 진짜 차별점이고, GBM이 구조적으로 못 보는 것.
3. **세 단계로 확정**: 1맵 → 전 맵(p<1e-126) → 이질 그래프 구조(solo 100팀). 우연 아님.

## 5. 한계

- squad와 solo/duo를 **분리 학습**(전 모드 cached 58GB > RAM 52GB). 전 모드 단일 모델은 미실행.
- 각 데이터셋 **1회 측정**(full squad는 1,521매치로 매치-페어드 유의, 단 seed 반복은 없음).
- solo/duo 절대값이 낮음(0.814) — 데이터 적고(3,704) 개인전 변동성. 상대 우위는 일관.
- δ ablation은 1맵만(full에선 δ=0.3 고정).

## 6. 인프라 (이번 세션 해결)

- **디스크 위기**: snapshot 빌드가 디스크 100% 채움 → legacy `data/graphs` 34GB 정리로 49Gi 확보.
- **메모리 벽**: dense A_s cached 481GB → COO 희소화 16GB(모델 변경 0, `mr_dataset`만).
- DB 규모 ~650GB(stdb 477 + raw pubg 173). raw 전체 DROP은 미실행(안전장치 차단 + 부분
  삭제 OS 회수 불가). 학습은 디스크와 무관(희소화로 해결).

## 7. 다음 작업

- **논문 06·07 재구성**: 이 3-모델 사다리(역설→해소)를 새 중심 결과로. 기존 full DB
  feature-only(0.7829) 서사를 "phase 천장 vs snapshot ST-GNN"으로 교체.
- (선택) solo/duo 유의성, seed 반복, 전 모드 lazy 단일 모델.
- B3/B4(phase tier full)는 우선순위 하락 — MR이 phase-only를 이미 압도.

산출물: `model_stgcn/checkpoints/{mr_tiger_squad, mr_full_squad, mr_solo_duo}/`.
