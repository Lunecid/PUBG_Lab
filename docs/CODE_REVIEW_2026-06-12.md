# 전체 코드 리뷰 진단 보고서 — 2026-06-12

> 작성: Claude Code 전체 코드 리뷰 (병렬 탐색 에이전트 3개 + 핵심 주장 직접 코드 재검증).
> 범위: 페이퍼 경로(`stdb/` + `model_stgcn/`), 레거시 스택(루트 `model/`·`train.py`·`dataset.py`·`simulate.py` 등), 데이터 파이프라인, `network_analysis/`, `webapp/`, 인프라.
> 표기: ✅ = 코드를 직접 읽어 검증한 항목, ⚠️ = 에이전트 보고 기반(수정 전 재확인 권장).

---

## ✦ 수정 적용 현황 (2026-06-12, 브랜치 `claude/code-review-fixes`)

같은 날 전 항목 수정을 진행했다. 상태 요약:

| 항목 | 상태 | 비고 |
|---|---|---|
| B1 git 추적 | ✔ 완료 | 페이퍼 경로 513파일 커밋, .gitignore 정비 (대용량/산출물 제외) |
| B2 비밀번호 | ✔ 코드 측 완료 | 4곳(진단 3곳 + 추가 발견 `stdb/audit_raw.py`) env 필수화. **DB 비밀번호 실제 교체는 사용자 수행 필요** — 이전 값이 git 이력에 남아 있음 |
| B3 GroupGNN | ✔ 완료 | `_pool_alive_teams` 헬퍼로 글로벌→로컬 정렬 일원화, 제로패딩 핵 제거, 계약 위반 시 명확한 에러. 회귀 테스트 추가 |
| C4 aux 풀링 | ✔ 완료 | `forward(phase_mask=...)` masked mean + 호출부 6곳 전파. **수치 갱신은 full-DB 재학습 필요** |
| C5 combat 방향 | ✔ capability 완료 | E=4 `combat_in` 채널(raw 가중치 기준 정확 정규화), `--combat-in` 플래그, 모드 2종 추가, ckpt 자동 감지. **ablation 학습 3종은 미실행** — 아래 커맨드 참조 |
| C6 forward 방어 | ✔ 완료 | 첫 forward 시 row-stochastic 검증(상시 비용 0), 스모크를 실 분포로 교체 |
| C7 bootstrap CI | ✔ 완료 | `model_stgcn/bootstrap_ci.py` (match-level cluster bootstrap), `eval_phase --bootstrap N` |
| C8 2km 임계값 | ✔ 완료 (opt-in) | **신규 발견: 단위 버그** — V2(39d) 좌표는 0~1 정규화라 `cd<2000`이 사실상 항상 참(비교전 쌍 전부 연결). 기존 체크포인트가 이 분포로 학습됐으므로 기본값은 유지, `team_edge_ratio`+미터 환산은 opt-in |
| D9 legacy 격리 | ✔ 완료 | `legacy/` + README, import/데모 경로 동작 확인 |
| D10 테스트 | ✔ 완료 | `tests/` 18개 (DB 없으면 13개 자동 skip). 라벨 정렬·A 불변식·마스크·forward·B3 회귀 |
| D11 출력 정리 | ✔ 완료 | `outputs/legacy_sim/`, `model_stgcn/config.py` 경로 중앙화. **신규 발견: `phase_wise_eval.py`에 epochs=12 하드코딩** → config로 수정 |
| E12~E17 | ✔ 완료 | requirements `==` 핀 + dev 분리, seed 42 통일+manifest 경고, denom 로깅, max() 주석, ckpt feature_dim 저장/사용, l_encounter 제거, run.py deprecated 표기 |
| E18 인덱스 | ⏸ 보류 | DDL 실행 권한 정책상 미실행 → `stdb/migrations/2026-06-12_phase_indexes.sql` 제공, 실행은 사용자 결정 |

**검증**: `PG_PASSWORD=… python -m pytest tests/` 18 passed · `python -m model_stgcn.dyn_stgnn` 스모크 통과 · 실 DB 초소형 학습 E=3/E=4 각 1 epoch 정상 (E=4에서 combat=0, combat_in 활성 확인) · 인접행렬 검증이 실 데이터에서 통과 (A3 실증).

**진단 정정 (수정 중 확인)**:
- `model_stgcn/splits.py`의 split 중복/오버랩 검증은 **이미 존재** (에이전트 오탐 추가 정정).
- 레거시 ckpt는 `agent_feat_dim`을 config에 **이미 저장** (simulate.py가 사용 중).

**신규 발견 (E20, 데이터 품질)**: `team_phase_labels`에서 타겟의 **2.47%** (33,696/1,366,821)가 `alive_at_phase_start=false`인데 `target_defined=true` — phase 시작 시점 전원 DBNO 또는 스냅샷 공백. 라벨 누수는 아니나 해당 팀은 그 phase에 proximity self-loop을 받지 못함. `tests/test_label_alignment.py::test_alive_start_end_gap_bounded`가 3% 상한 회귀 가드. 근본 해결은 ETL의 시작 시점 정의 보강(직전 phase 종료 상태 forward-fill) — 라벨 테이블 재생성이 필요해 보류.

**C5 ablation 실행 커맨드** (full-DB, 동일 split·seed):
```bash
# 1) 현행 (E=3, 기존 결과 재사용 가능)
# 2) 양방향 E=4
python -m model_stgcn.train_phase --combat-in --edge-mode full \
    --split-json model_stgcn/checkpoints/split_phase_all_seed42.json \
    --checkpoint model_stgcn/checkpoints/ablation_combat_bidir.pt
# 3) 피격 수신만 (combat_out 제거)
python -m model_stgcn.train_phase --combat-in --edge-mode full_combat_in \
    --split-json model_stgcn/checkpoints/split_phase_all_seed42.json \
    --checkpoint model_stgcn/checkpoints/ablation_combat_in_only.pt
# 평가 (95% CI 포함)
python -m model_stgcn.eval_phase --checkpoint <ckpt> --split test --bootstrap 1000
```

---

> 이하는 원본 진단 (2026-06-12 오전, 수정 전 기준).

---

## 0. 요약표

| # | 항목 | 심각도 | 스택 | 위치 | 권장 조치 |
|---|------|--------|------|------|-----------|
| B1 | 페이퍼 경로 전체 git 미추적 | 🔴 Critical | 공통 | `stdb/`, `model_stgcn/`, `docs/`, `paper/` 등 | 즉시 git 추적 + .gitignore 정비 |
| B2 | DB 비밀번호 하드코딩 3곳 | 🔴 Critical | 공통 | `main.py:48` 외 2곳 | env 필수화 + 비밀번호 교체 |
| B3 | GroupGNN 제로 패딩 임시방편 | 🔴 Critical | 레거시 | `model/group_gnn.py:146-158` | compact 인덱스 재매핑으로 근본 수정 |
| C4 | aux 풀링이 phase_mask 무시 | 🟠 Major | 페이퍼 | `model_stgcn/dyn_stgnn.py:193` | masked mean으로 교체 |
| C5 | combat 엣지 방향 의미론 | 🟠 Major | 페이퍼 | `stdb/phase_edge_etl.py` + `dyn_stgnn.py:78` | A^T 채널 추가 또는 transpose ablation |
| C6 | forward에 마스크/정규화 방어 없음 | 🟠 Major | 페이퍼 | `dyn_stgnn.py:176-196` | assertion + 마스크 인자 추가 |
| C7 | bootstrap CI/유의성 검정 부재 | 🟠 Major | 페이퍼 | 평가 모듈 전반 | 95% CI 병기 |
| C8 | 팀 엣지 2km 하드코딩 | 🟠 Major | 레거시 | `dataset.py:397` | 맵 크기 비례 파라미터화 |
| D9 | 3개 데이터 표현 공존 + 중복 파일명 | 🟠 Major | 공통 | 루트 vs `model_stgcn/` vs `network_analysis/` | `legacy/` 격리 (사용자 확정) |
| D10 | 테스트 0개 (leakage-safety 불변식 미보증) | 🟠 Major | 공통 | 전체 | 스모크 테스트 4종 추가 |
| D11 | 출력물 산재 | 🟠 Major | 공통 | repo 루트, `checkpoints/` 2곳 | `outputs/` 규약화 |
| E12~17 | 의존성·CLI·시드·기타 | 🟡 Minor | — | 본문 참조 | 본문 참조 |

**한 줄 결론**: 페이퍼 경로의 그래프·라벨 코어(인과성, 정규화, 누수 방지)는 **건전하다**. 실제 위험은 모델 수식이 아니라 **(1) 버전 관리 부재, (2) 시크릿 노출, (3) 마스킹 사각지대 2곳, (4) 검증 자동화 부재**에 있다.

---

## 1. (A) 검증 결과 "정상"으로 확인된 항목 — 오탐 반박 기록

리뷰 에이전트가 CRITICAL로 보고했으나 직접 검증 결과 **문제없음**으로 판정한 항목들. 향후 리뷰에서 같은 오탐이 반복되지 않도록 근거를 남긴다.

### A1. hazard 라벨 정렬·인과성 — 정상 ✅

- phase는 **1-indexed** (`stdb/queries.py:388-399`, `phase_idx BETWEEN 1 AND 8`).
- 라벨 매핑은 `out_t = p - 1` (`queries.py:533`): 출력 인덱스 k ← cut_phase k+1.
- `hazard_next(p)` = "phase p 종료 시 생존했고 phase p+1에서 탈락" (`stdb/rq1_feature_etl.py:694-696`).
- temporal conv는 left-pad causal (`model_stgcn/dyn_stgnn.py:36`), `hazard_logits = head(htnc[:, :-1])` (`dyn_stgnn.py:190`).
- 종합: hidden(phase k+1까지 관측) → "phase k+2 탈락" 예측. **정렬·인과성 모두 정확**.

### A2. `alive_players_end` 피처 — 누수 아님 ✅

`target_defined = (phase t 종료 시 생존) AND next_phase 존재` (`rq1_feature_etl.py:686-687`). risk set이 "interval 시작 시점 생존자"로 정의되는 표준 discrete-time survival 설계와 일치. phase t 종료 시점 정보로 t+1을 예측하므로 인과적이다.

### A3. 인접행렬 A 정규화 — 존재하며 문서화됨 ✅

`stdb/queries.py:583-595`: edge-type별 독립 row-normalize, proximity만 self-loop(살아있는 노드 한정), combat/copressure는 상호작용 전용으로 zero row 허용(의도 주석 있음). 패딩 노드는 A의 row/col이 0이라 `matmul(A, x)` 경로로 실 노드를 오염시키지 않는다.
참고: row-stochastic(D⁻¹A) 방식이며 symmetric normalization(D^-½AD^-½)이 아닌 것은 **설계 선택**이지 버그가 아니다.

### A4. 팀 순서 라벨 누수 — 이미 수정됨 ✅

`_phase_team_order` (`queries.py:402-419`)가 final_rank 정렬(구버전, 누수) → team_id 정렬로 교체된 이력이 주석으로 남아 있다. `PHASE_N_MAX=128` 쿠션도 같은 이유(`queries.py:60-63`).

### A5. val-fit calibration → test 적용 — 표준 관행 ✅

레거시 `train.py`의 isotonic calibration을 에이전트가 "nested CV leak (CRITICAL)"으로 보고했으나, 검증 셋에서 calibrator를 적합하고 테스트 셋을 평가하는 것은 temperature scaling 등에서 통용되는 표준 절차다. 논문에는 절차만 명시하면 된다.

### A6. AUC 단일클래스 가드 — 주요 평가 모듈 모두 존재 ✅

`baselines.py:143-146`, `phase_wise_eval.py:74-77`, `eval_phase.py:56-61` 모두 try/except ValueError → NaN 처리 확인. 에이전트의 "불일치" 보고는 오탐.

---

## 2. (B) Critical

### B1. 페이퍼 경로 전체가 git 미추적 ✅

**현황**: `git status` 기준 `stdb/`, `model_stgcn/`, `docs/`, `paper/`, `scripts/`, `network_analysis/`, `webapp/`, `checkpoints/`가 전부 **untracked**. git에 커밋된 코드는 4월 snapshot 시대 레거시 스택(47커밋)뿐이다. 즉 **ICE 2027 페이퍼의 21,824매치 full-DB 실험 코드·논문 원고가 버전 관리 0%** 상태다.

**왜 문제인가**: 디스크 사고·실수 삭제 한 번이면 재현 불가. "어느 코드로 어떤 수치가 나왔는가"를 증명할 수 없어 논문 재현성 주장과 정면 충돌. `docs/RESEARCH_STATUS_2026-06-10.md`가 명시한 실험-수치 추적 노력이 코드 레벨에서 무력화됨.

**권장 수정**:
1. `.gitignore`에 대용량/산출물 추가: `data/`, `checkpoints/`, `*.zip`, `*.pptx`, `*.pdf`(필요 시 LFS), `simulation_result_*`, `webapp/node_modules/`, `logs/`.
2. `stdb/`, `model_stgcn/`, `docs/`, `paper/`, `scripts/`, `adapters/` 우선 커밋. `network_analysis/`, `webapp/`도 뒤이어 커밋.
3. 이후 실험은 체크포인트 메타에 commit hash 기록(레거시 `run_meta.py`에 이미 동일 패턴 존재 — 재사용).

**검증 방법**: `git status`에 untracked 소스 디렉토리 0개, `git ls-files | grep model_stgcn` 비어있지 않음.

### B2. DB 비밀번호 하드코딩 3곳 ✅

```
main.py:48                     "password": "104204",        ← git 추적 중 (이력에 이미 노출)
network_analysis/config.py:18  "password": "104204",
stdb/conn.py:26                os.environ.get("PG_PASSWORD", "104204")   ← env 폴백이 시크릿
```

**왜 문제인가**: `main.py`는 이미 git 이력에 들어가 있어, 원격 push 시 즉시 노출된다. B1 권장대로 나머지 디렉토리를 커밋하면 노출 지점이 3곳으로 늘어난다.

**권장 수정**: 세 곳 모두 `os.environ["PG_PASSWORD"]`(폴백 없음, 미설정 시 명확한 에러 메시지)로 통일 → DB 비밀번호 교체 → `.env.example` 추가. 접속 설정은 `stdb/conn.py` 한 곳으로 일원화하고 `main.py`·`network_analysis/config.py`가 이를 import.

**검증 방법**: `grep -rn "104204" --include="*.py"` 결과 0건. env 미설정 상태에서 실행 시 친절한 에러.

### B3. GroupGNN 제로 패딩 임시방편 (커밋 616925b) ✅

`model/group_gnn.py:146-158`:

```python
if len(alive_teams) > 0 and team_h_pooled.shape[0] > n_alive:
    max_idx = max(alive_teams) if alive_teams else -1
    if max_idx >= team_h_pooled.shape[0]:
        # alive_teams에 pooled 범위를 초과하는 팀 ID가 있으면 패딩
        team_h_pooled = torch.cat([team_h_pooled, torch.zeros(pad_size, ...)], dim=0)
    pooled_alive = team_h_pooled[alive_teams]
else:
    pooled_alive = team_h_pooled[:n_alive]
```

**왜 문제인가**:
1. IndexError는 사라졌지만, 범위를 초과한 **살아있는 팀이 조용히 0벡터 임베딩**을 받는다. 크래시가 침묵하는 오답으로 바뀐 것.
2. `else` 분기(`team_h_pooled[:n_alive]`)는 pooled 행 0..n_alive-1이 `alive_teams` 순서와 위치상 일치한다고 **가정**한다. `team_h_pooled.shape[0] == n_alive`인데 팀 ID가 비연속(예: [0,2,5])이면 silent misalignment.
3. 근본 원인: GroupPooling의 출력 인덱스 공간(팀 ID 기반 scatter)과 team_graph의 `alive_teams` 인덱스 공간이 다른 곳에서 각각 정의됨.

**권장 수정**: 그래프 구성 시점(`dataset.py`의 snapshot graph builder)에서 `team_id → 0..n_alive-1` compact 재매핑을 **한 번만** 만들어 player graph(`p_team_idx`)·team graph(`alive_teams`)·pooling이 같은 매핑을 공유하게 한다. GroupGNN 입구에는 `assert team_h_pooled.shape[0] == n_alive` 한 줄만 남기고 패딩 로직 삭제.

**검증 방법**: 비연속 팀 ID(예: [3, 7, 12]) 합성 입력으로 forward → 출력이 팀별로 올바르게 정렬되는지 단위 테스트. 기존 IndexError 재현 케이스 회귀 테스트.

---

## 3. (C) Major — 그래프/모델

### C4. [페이퍼] aux 풀링이 phase_mask를 무시 ✅

`model_stgcn/dyn_stgnn.py:193`:

```python
pooled = htnc.mean(dim=1)          # (B,T,N,C) → T축 단순 평균. phase_mask 미사용
rank_pred = self.rank_head(pooled)
topk_logits = self.topk_head(pooled)
```

**왜 문제인가**: 텐서는 항상 T=8로 패딩되는데, 실제 phase가 8 미만인 매치(상당수)에서 존재하지 않는 phase의 hidden이 평균에 포함된다. 패딩 phase는 X=0이지만 입력 레이어 bias 때문에 hidden은 0이 아니며(`ReLU(W·0+b)`), causal conv가 직전 실 phase 정보를 패딩 위치로 흘려보낸다. rank/topk는 보조 헤드(β=0.3, γ=0.3)지만 **공유 블록으로 그래디언트가 역류**하므로 hazard 표현 학습에도 영향을 준다. `forward`는 phase_mask를 인자로 받지도 않는다(`dyn_stgnn.py:176-181`) — collate가 만들어주는 마스크(`dataset.py`의 `phase_collate`)가 손실 단계에서만 쓰이고 모델 내부에선 버려진다.

**권장 수정**:

```python
# forward(X, A, phase_mask=None, ...)
m = phase_mask.float().unsqueeze(-1).unsqueeze(-1)        # (B,T,1,1)
pooled = (htnc * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
```

**검증 방법**: phase 5짜리 합성 매치에서 패딩 phase hidden에 큰 노이즈를 주입해도 rank_pred 불변임을 단위 테스트. 수정 전후 full-DB 재학습 1회로 hazard AUC/AP 변화 기록(논문 수치 갱신 여부 판단).

### C5. [페이퍼] combat 엣지 방향 의미론 — 피격팀이 공격자 정보를 못 받음 ✅

- ETL: combat 엣지는 **directed, src=공격팀 → dst=피해팀**, weight = damage + 50·groggy (`stdb/phase_edge_etl.py:148-175`).
- 모델: `agg = torch.matmul(ae, xp)` (`dyn_stgnn.py:78`) → `out[i] = Σ_j A[i,j]·x[j]`, 즉 **행(src) 기준 수신**.
- 결과: **공격자가 피해자의 피처를 집계**하고, 피해팀은 공격자에 대한 그래프 메시지를 받지 못한다.

**왜 문제인가**: hazard(다음 페이즈 탈락) 예측에서 직관적으로 더 중요한 신호는 "나를 때리는 팀이 누구이고 얼마나 강한가"인데, 현재 메시지 방향은 그 반대다. 노드 피처의 `damage_taken` 계열이 일부 보상하지만, "공격자의 상태(체력·위치·화력)"는 그래프를 통해서만 전달될 수 있는 정보이고 그 경로가 막혀 있다. 이는 edge ablation에서 combat 엣지 기여도가 낮게 나오는 원인일 수 있다 — **논문의 핵심 주장(그래프 엣지의 기여)과 직결**되므로 확인 가치가 높다.

**권장 수정** (택1, ablation으로 결정):
1. A_combat을 transpose해서 피해자가 수신하도록 변경 (E=3 유지).
2. `combat_in`(피해자 수신)·`combat_out`(공격자 수신) 두 채널로 분리 (E=4) — 정보 손실 없음, 권장.

**검증 방법**: 동일 split·seed로 {현행, transpose, E=4} 3조건 학습 후 hazard AUC/AP, edge weight(softmax) 비교. 기존 `edge_modes.py`/`edge_sanity.py` 인프라 재사용 가능.

### C6. [페이퍼] forward에 마스크·정규화 방어 부재 ✅

- `DynamicSTGNN.forward(X, A, edge_active_mask)` (`dyn_stgnn.py:176-181`)는 node_mask/phase_mask를 받지 않는다 (C4와 동일 뿌리).
- A가 row-stochastic이라는 가정(데이터 측 보장)을 모델이 한 번도 검사하지 않는다. ETL이나 캐시가 바뀌면 조용히 깨진다.
- `__main__` 스모크 테스트(`dyn_stgnn.py:254-262`)는 dense random A를 전 노드(패딩 포함)에 대해 정규화 — 실제 파이프라인(희소, zero-row 허용, alive만 self-loop)과 분포가 달라 스모크로서 대표성이 낮다.

**권장 수정**: forward 첫머리에 디버그 모드 assertion(`(A.sum(-1) - 1).abs() < 1e-4 | A.sum(-1) == 0`), 스모크 테스트를 `fetch_phase_match_tensors` 실데이터 1매치 기반으로 교체.

**검증 방법**: 고의로 정규화 안 된 A 주입 시 assertion 발화.

### C7. [페이퍼] 평가 통계 엄밀성 — bootstrap CI 부재 ⚠️

seed 반복 실험(`seed_repeat_experiments.py`)과 Wilcoxon(`phase_wise_eval.py:22`)은 이미 존재하나, 본 평가 수치(AUC/AP/Brier/ECE)에 match-level bootstrap 95% CI가 없다. 리뷰어가 "0.72 vs 0.70 차이가 유의한가"를 물을 때 방어 수단이 필요하다.

**권장 수정**: match 단위 리샘플링 bootstrap(1,000회) 유틸을 평가 모듈에 추가, 논문 표에 CI 병기.

### C8. [레거시] 팀 엣지 2km 하드코딩 ✅

`dataset.py:397`: `if cd < 2000:  # 2km 이내만` — encounter 없는 팀 쌍의 fallback 연결 임계값이 맵 크기와 무관하게 고정. Sanhok(4×4km)에선 사실상 완전그래프, Erangel(8×8km)에선 희소 그래프가 되어 **맵마다 그래프 밀도 체계가 다른 모델**이 된다.

**권장 수정**: 임계값을 `arena_size × ratio`로 파라미터화하고 config로 노출. (레거시 스택이므로 데모 품질에 영향 있을 때만.)

---

## 4. (D) Major — 아키텍처/파이프라인

### D9. 3개 데이터 표현 공존 + 중복 파일명, deprecation 마커 부재 ✅

| 표현 | 생산자 | 소비자 | 상태 |
|------|--------|--------|------|
| legacy `.pt` snapshot (~34GB, 39d) | `main.py` | 루트 `dataset.py`/`train.py`, `simulate.py`, `visualize.py`, `network_analysis/` | 공식 legacy 선언(RESEARCH_STATUS §3), 단 데모로 활발히 수정 중 (PR #7~#19) |
| STDB phase 테이블 (66d) | `stdb/*_etl.py` | `model_stgcn/` 전체 | **페이퍼 경로 (활성)** |
| webapp JSON 샤드 | `network_analysis/run_v2.py` | `webapp/` | 탐색(H1/H2/H3) 보조 |

여기에 **동명 파일 3중복**: `dataset.py`·`train.py`(루트 vs `model_stgcn/`), `metrics.py`(루트 vs `network_analysis/`). 루트 `metrics.py`는 루트 `train.py`만 import(✅ 확인)하므로 레거시 소속이다. 신규 참여자(또는 미래의 본인)가 잘못된 스택을 수정할 위험이 상시 존재하며, 실제로 이번 리뷰의 에이전트조차 두 스택을 혼동했다.

**권장 수정 (사용자 확정안)**: 루트 레거시 스택(`main.py`, `train.py`, `dataset.py`, `metrics.py`, `simulate.py`, `visualize.py`, `result_viz.py`, `explain.py`, `run_meta.py`, `model/`, `adapters/`)을 `legacy/`로 이동, `legacy/README.md`에 "snapshot 시대(2026-04) 데모 전용, 페이퍼 경로는 `stdb/`+`model_stgcn/`" 명시. import 경로 수정 및 데모 실행 확인까지가 한 세트.

**검증 방법**: `legacy/` 이동 후 `python legacy/simulate.py` 데모 동작 확인, 루트에 모델 코드 잔존 0.

### D10. 테스트 0개 — leakage-safety 불변식이 코드로 보증되지 않음 ✅

`test_*.py` 검색 결과 0건. 논문이 명시적으로 주장하는 안전성(누수 없는 라벨, 정규화된 A, 패딩 비오염)이 전부 "현재 코드가 우연히 그런 상태"일 뿐, ETL·쿼리 수정 한 번이면 조용히 깨진다. 이번 리뷰에서 A1~A4가 정상임을 확인하는 데 수동 검증이 필요했다는 사실 자체가 테스트 부재 비용이다.

**권장 수정** — 최소 스모크 4종 (`tests/`, 실DB 1매치 또는 고정 fixture):
1. `test_label_alignment.py`: out_t↔cut_phase↔hazard_next 정렬 (A1 불변식).
2. `test_adjacency.py`: row-stochastic or zero-row, self-loop 정책, 패딩 row/col=0 (A3).
3. `test_masks.py`: target_mask ⊆ (alive at phase end), node/phase/team mask 상호 일관성.
4. `test_model_forward.py`: DynamicSTGNN shape + 패딩 노드 그래디언트 비전파.

### D11. 출력물 산재 ✅

- `simulation_result_*.json/html`(각 0.6~2MB)이 repo 루트에 존재(일부 git 추적).
- 체크포인트가 루트 `checkpoints/`(Baltic_Main, Tiger_Main)와 `model_stgcn/checkpoints/`(실험 다수)로 이원화.
- 체크포인트 파일명 템플릿(`phase_{edge_mode}_e{epochs}.pt` 등)이 여러 파일에 하드코딩 ⚠️.

**권장 수정**: `outputs/{run_id}/` 규약(체크포인트+config.json+metrics.csv 동봉), 루트 산출물은 `outputs/legacy_sim/`으로 이동 후 `.gitignore`. 경로 템플릿은 `model_stgcn/config.py` 신설로 중앙화.

---

## 5. (E) Minor

| # | 항목 | 위치 | 내용/권장 |
|---|------|------|-----------|
| E12 | 의존성 재현성 | `requirements.txt` | 전부 `>=` 느슨 고정 ✅. `scikit-survival`은 코드에서 사용(`metrics.py:41,108,157`, fallback 있음 ✅)되나 requirements에 없음 — 명시 또는 optional 문서화. `pip freeze` 기반 lock 파일 추가 권장 |
| E13 | seed 기본값 불일치 | `model_stgcn/train.py:223`(0) vs `train_phase.py:312`(42) ✅ | 42로 통일, split manifest의 seed와 CLI seed가 다르면 경고 |
| E14 | 빈 배치 loss=0 | `dyn_stgnn.py:223-224` ✅ | `denom.clamp(min=1)`로 타겟 전무 배치가 loss 0 — 버그는 아니나 학습 곡선 해석 왜곡 가능. denom 로깅 권장 |
| E15 | 중복 엣지 max 집계 | `queries.py:581` ✅ | `A[t,e,i,j] = max(...)` — PK상 중복이 없어야 정상이므로 sum/max 선택 근거를 주석으로 명시 |
| E16 | 체크포인트 로드 검증 부재 | `model_stgcn/infer.py:47-56` ✅ | `in_features=8` 하드코딩 + ckpt의 feature_dim/edge_types 미검증 → 차원 불일치 시 불친절한 에러. 로드 시 shape 검증 추가 |
| E17 | dead code | `model/arena_survival_net.py:279`(l_encounter=0 placeholder, `lambda_encounter` 무효 하이퍼파라미터) ✅, `network_analysis/run.py`(run_v2로 대체) ⚠️ | legacy 격리 시 함께 정리 |
| E18 | DB 인덱스 | `team_phase_trajectory`, `team_phase_exposure` ⚠️ | `(match_id, phase_idx)` 복합 인덱스 — 현재 규모(2.2만 매치)에선 선택, 확장 시 필수 |
| E19 | model_stgcn CLI 난립 | 실행 모듈 18개 ⚠️ | `python -m model_stgcn <subcommand>` 통합은 P3 — 논문 마감 전 필수는 아님 |

---

## 6. (F) 개선 로드맵 — 컨펌 후 별도 세션에서 실행

### P0 — 즉시 (데이터/보안 사고 방지)
1. **B2** 비밀번호 env 필수화 + DB 비밀번호 교체 (커밋 *전에* 수행 — 순서 중요)
2. **B1** `.gitignore` 정비 → 페이퍼 경로 전체 git 커밋
3. **D9** 레거시 스택 `legacy/` 격리

### P1 — 모델 정합성 (페이퍼 수치에 영향 가능)
4. **C4** aux 풀링 phase_mask masked mean + 재학습 1회로 영향 측정
5. **B3** GroupGNN compact 인덱스 재매핑 (레거시지만 데모 정확성)
6. **D10** leakage-safety 스모크 테스트 4종

### P2 — 페이퍼 보강
7. **C5** combat 엣지 방향 ablation (3조건) — 결과에 따라 본문 수치/해석 갱신
8. **C7** match-level bootstrap CI 추가
9. **D11** outputs/ 규약화 + 산출물 이동

### P3 — 정리
10. **E13** seed 통일, **E16** 체크포인트 검증, **E12** requirements lock, **E19** CLI 통합, **E15/E17/E18** 잔여 정리

---

## 부록: 리뷰 방법론

1. 병렬 탐색 에이전트 3개: ① 루트 레거시 스택, ② `model_stgcn/` 스택, ③ 데이터 파이프라인·인프라·문서.
2. 에이전트가 CRITICAL/MAJOR로 보고한 주장 전수에 대해 원본 코드 직접 정독으로 재검증 — 그 결과 4건 반박(A1, A2, A3 일부, A5, A6), 3건 확정(B2, B3, C4 등), 다수 세부 보정.
3. git log·`docs/RESEARCH_STATUS_2026-06-10.md`로 활성/레거시 경로 판별: **git 추적 코드 = 전부 레거시, 페이퍼 경로 = 전부 미추적**이라는 역설이 B1의 근거.
