# Design Spec — "Prediction Targets & Metrics" slides (Graph PPT)
*디자인 설계서 / 빌드는 Claude 디자인(Figma/frontend) 담당. 이 문서는 레이아웃·요소·문구·색을 지정한다. 모든 슬라이드 텍스트는 영어·격식체.*

## 0. 목적 & 덱 내 위치
- **위치**: 모델 슬라이드 다음, *결과(fig5 탈락·fig6 순위) 이전*. 두 예측 타깃을 *어떻게 라벨링했고 어떻게 채점하는지*를 먼저 이해시켜야 결과 그림이 읽힌다.
- **슬라이드 수**: 2장 (A = Labeling, B = Metrics). 밀도가 허용되면 1장으로 합쳐도 됨.
- **검증됨**: 라벨↔figure 일치 코드+DB 100% 확인(team_phase_labels vs phase_metrics). 본 내용은 실데이터에 정확.

## 1. 공유 디자인 시스템 (두 슬라이드 공통)
- **스타일**: modern academic. 밝은 배경(#ffffff), 절제된 색, 얇은 구분선. 차트정크 금지.
- **타이포**: sans-serif(Helvetica/Arial/Inter류). 제목 ~28pt, 본문 ~16pt, 캡션 ~12pt. 좌측 정렬 우선.
- **색 토큰** (figure와 일치 필수):
  - alive/survive = `#2ca02c` (green) · eliminated = `#d62728` (red)
  - observed-window/model-input = `#dbeafe` fill, `#1d4ed8` accent (blue)
  - ranking/award = `#d97706` (amber) accent, `#fef3c7` fill
  - neutral text = `#374151`, muted = `#6b7280`, gridline `#cccccc`
- **아이콘**: 미니멀 라인. zone=concentric circles, elimination=skull/X, ranking=trophy/medal, time=arrow.

---

## 2. Slide A — "Prediction targets: leakage-safe labeling"

### 레이아웃 (3-zone 세로)
```
┌────────────────────────────────────────────────────────────┐
│ TITLE: Prediction targets — leakage-safe phase labeling     │
├────────────────────────────────────────────────────────────┤
│ [CENTER] cut-phase timeline diagram (핵심 시각, 화면 ~60%)   │
├──────────────────────────────┬─────────────────────────────┤
│ [BOTTOM-LEFT] Target 1 card   │ [BOTTOM-RIGHT] Target 2 card │
│  ELIMINATION (hazard)         │  RANKING (final placement)   │
└──────────────────────────────┴─────────────────────────────┘
```

### 핵심 시각 — cut-phase timeline (참조 프로토타입: `docs/presentation_figures/fig0_labeling.png`)
- 가로축 = phases (Phase 1 … 7, 우측 끝 "Match end").
- **2개 예시 팀 행**:
  - *Team A (eliminated)*: cells = alive(green) P1–P4, eliminated(red) P5–P7. final rank badge `#8`.
  - *Team B (winner)*: cells = alive(green) 전 phase. final rank badge `#1 (winner)`.
- **hazard_next 라벨**: 각 alive cut_phase 아래 작은 0/1. Team A = `0 0 0 1`(P4에서 1=다음 phase 탈락), Team B = `0 0 0 0 0 0`. 1은 red, 0은 green.
- **cut-phase 인과 mechanic** (Team A, cut_phase 4에 1개만 강조):
  - P1–P4 위에 옅은 파란 박스 + 라벨 `observed ≤ cut_phase 4 (model input)`.
  - P4→P5로 파란 화살표 + `predict phase 5`.
- **at-risk note** (작게, 우하단): `At-risk mask: only cells with the team alive at end of phase p (and a next phase exists) are scored; eliminated teams are not predicted.`

### Target cards (하단 2개, 좌/우)
- **카드 1 — ELIMINATION** (red accent, skull/X 아이콘):
  - 제목: `Elimination — hazard_next ∈ {0, 1}`
  - 본문: `For a team alive at the end of cut-phase p: is it eliminated during phase p+1?  1 = eliminated, 0 = survives.`
- **카드 2 — RANKING** (amber accent, trophy 아이콘):
  - 제목: `Final placement — final_rank (1 = winner)`
  - 본문: `Each team's final standing, normalized to [0,1]; top-3 / top-5 / top-10 and winner are derived. Broadcast to every cut-phase.`

### 슬라이드 한 줄 takeaway (상단 또는 하단 띠)
`Features are observed through cut-phase p and used to predict phase p+1 — a strictly causal, leakage-safe setup.`

---

## 3. Slide B — "How each target is scored" (지표 설명 — 이게 결과보다 먼저)

### 레이아웃 (2-column)
```
┌────────────────────────────────────────────────────────────┐
│ TITLE: How each prediction is scored                        │
├──────────────────────────────┬─────────────────────────────┤
│ LEFT  — ELIMINATION → AUC     │ RIGHT — RANKING → Spearman ρ │
│  (개념 + 1줄 직관 + 미니막대)  │  (개념 + 4팀 예시표 + 직관)  │
├──────────────────────────────┴─────────────────────────────┤
│ [FOOTER] both computed per cut-phase over at-risk teams      │
└────────────────────────────────────────────────────────────┘
```

### LEFT — Hazard AUC (elimination)
- 제목: `Elimination → Hazard AUC`
- 직관 1줄: `Probability that the model scores a team that IS eliminated higher than one that survives. 0.5 = chance, 1.0 = perfect.`
- 미니 시각: 작은 분포 2개(survive=green, eliminated=red)가 겹치는 정도 → AUC = 분리도. (개념 illust, 데이터 아님)

### RIGHT — Rank Spearman ρ (ranking) — *핵심 설명*
- 제목: `Final placement → rank Spearman (ρ)`
- 직관 1줄: `Correlation between the PREDICTED order and the ACTUAL final-placement order.  +1 = same order (perfect), 0 = random, −1 = reversed.`
- **예시표** (4팀, 반드시 포함 — 이게 이해의 핵심):
  | Team | model score | → predicted order | actual final rank |
  |---|---|---|---|
  | A | 0.1 | 1st | 1st (winner) |
  | B | 0.3 | 2nd | 2nd |
  | C | 0.5 | 3rd | 3rd |
  | D | 0.8 | 4th | 4th |
  - 표 아래 캡션: `Predicted order = actual order  →  ρ = +1.  What matters for ranking is the ORDER (who is above whom), not the exact rank number — so a rank-correlation metric is used.`
- 작은 보조 문장: `(Mean absolute rank error is also tracked, but the figure reports ρ because ranking is fundamentally about order.)`

### FOOTER (양쪽 공통 한 줄)
`Both metrics are computed per cut-phase, over the at-risk (alive) teams at that phase — so late-phase ranking is measured among the surviving (top-placed) teams.`

---

## 4. 빌드 노트 (디자인 도구용)
- 두 슬라이드는 **결과 슬라이드(fig5/fig6)와 색·폰트 일관** 필수 (위 색 토큰 그대로).
- Slide A의 timeline은 `fig0_labeling.png`를 **레이아웃 참조**로 하되, 디자인 도구가 벡터로 재생성(셀·화살표·badge를 깔끔한 컴포넌트로).
- Slide B의 4팀 예시표는 **반드시** 들어가야 함 — rank Spearman 직관의 핵심.
- 언어: 슬라이드 텍스트 전부 영어·격식체. 구어체 금지.
- 데이터 수치는 이 슬라이드들엔 없음(개념 설명 슬라이드). 실수치는 다음 결과 슬라이드(fig5/fig6) 담당.
