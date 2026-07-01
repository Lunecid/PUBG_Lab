# Design Spec — Graph PPT deck (full deck)
*디자인 설계서 / 빌드는 Claude 디자인(Figma/frontend) 담당. 이 문서는 레이아웃·요소·문구·색을 지정한다. 모든 슬라이드 텍스트는 영어·격식체. 모든 수치는 `docs/PRESENTATION_DATA_2026-06-21.md`(메인 결과) 또는 `docs/FUTURE_DESIGN_3D_RELATIONAL_2026-06-21.md`(future-work/advantage)에서 검증됨.*

---

## 0. 덱 개요 & 슬라이드 순서

이 문서는 Graph PPT 덱 전체의 단일 설계서다. 11개 논리 슬라이드를 한 흐름으로 묶는다. **Labeling+Metrics 슬라이드(2장)는 별도 설계서**(`docs/deliverables/design_spec_labeling_metrics.md`)에 이미 완전 설계되어 있으므로 여기서는 위치만 표시하고 참조한다.

| # | Slide | 역할 | 설계 위치 |
|---|---|---|---|
| 1 | **Title** | 표지 + faint dual-lens teaser | 본 문서 §2 |
| 2 | **Worldview** | 발표의 정체성: relations within a shrinking habitat | 본 문서 §3 |
| 3 | **Modeling — Node design** | 노드 = per-team STATE vector(6 feature categories) | 본 문서 §4 |
| 4 | **Modeling — Edges + dynamic adjacency** | 3채널 이종 그래프 + `A[T,E,N,N]` | 본 문서 §5 |
| 5 | **Models — graph-encoder spectrum** | no-graph → GCN/SAGE/GAT → R-GCN(+변형); 결과 군집; advantage=future | 본 문서 §6 |
| — | **Labeling + Metrics (A, B)** | 예측 타깃 라벨링 + 채점 지표 | **참조: `design_spec_labeling_metrics.md`** |
| 6 | **Results — elimination** | `fig5_hazard_by_phase.png`; 인코더 군집, 그래프 = 작은 평활 이득 | 본 문서 §7 |
| 7 | **Results — ranking** | `fig6_rank_by_phase.png`; 더 넓은 스프레드, GAT 우위, late lock-in | 본 문서 §8 |
| 8 | **Interpretation & limitations** | 클라이맥스 진단: 이득=평활화, 관측 위상=무신호 | 본 문서 §9 |
| 9 | **Future work** | contact-gated relative-advantage 엣지 (preliminary, validated) | 본 문서 §10 |
| 10 | **Conclusion** | 관계의 *종류*가 그래프 정보성을 결정한다 | 본 문서 §11 |

**전체 내러티브 가드레일 (모든 슬라이드에 적용):**
- 그래프는 **3채널 이종(multi-relational, R-GCN) 그래프** — `proximity (symmetric)` · `combat (directed)` · `co-pressure (zone-gated)`. 어느 슬라이드도 이를 "단순 근접 그래프(simple proximity)"로 라벨/암시하지 말 것.
- 모든 그래프 인코더는 **한 점에 군집**(MR hazard AUC ≈ 0.844–0.850); **아키텍처는 레버가 아니다**.
- **smoothing / message-passing / attention / low-pass / null-model**의 *메커니즘 설명*은 오직 Slide 8(Interpretation)에서만 등장. 모델·결과 슬라이드는 사실만 전달.
- **advantage(상대 우위) 엣지**와 그 수치(+0.0033 등)는 오직 Slide 9(future work)에서만. preliminary/validated로 프레이밍, 메인 결과와 동급 금지.
- **본 기여 = "관계의 *종류*가 그래프 정보성을 결정한다"**. "2D→3D"는 기여가 아니라 correctness 수정(발표에서 셀링 금지).
- 성능 수치는 결과/해석/future 슬라이드에만. Title·Worldview·Modeling은 정량 결과 없음(개념·정의).

---

## 1. 공유 디자인 시스템 (덱 전체 공통 — 한 번만 정의, 슬라이드별 반복 금지)

- **스타일**: modern academic. 밝은 배경(`#ffffff`), 절제된 색, 얇은 구분선(`#cccccc`). 차트정크 금지.
- **타이포**: sans-serif(Helvetica/Arial/Inter류). 제목 ~28pt, 본문 ~16pt, 캡션 ~12pt. 좌측 정렬 우선. (예외: Slide 1 표지 제목은 표지 관례상 ~40–44pt — 본 문서 §2 빌드 노트.)
- **색 토큰** (덱 전체·모든 figure와 일치 필수):
  - alive/survive = `#2ca02c` (green) · eliminated = `#d62728` (red)
  - observed-window/model-input = `#dbeafe` fill, `#1d4ed8` accent (blue)
  - ranking/award = `#d97706` (amber) accent, `#fef3c7` fill
  - neutral text = `#374151`, muted = `#6b7280`, gridline `#cccccc`
- **relation-channel 전용 토큰** (Slide 4·5 등 — 채널마다 1색, 덱 전체에서 고정):
  - proximity = `#1d4ed8` (blue) · combat = `#d62728` (red) · co-pressure = `#d97706` (amber)
- **인코더 라인/포인트 색** (Slide 5·6·7·8 결과 figure와 1:1 고정):
  - No graph (TCN) = `#7f7f7f` (dashed) · GCN = `#1f77b4` · GraphSAGE = `#ff7f0e` · GAT = `#d62728` · R-GCN = `#2ca02c`
  - R-GCN 변형(FiLM, attention-rank decoder, FARNet) = R-GCN green(`#2ca02c`) 계열 명도 변주(동일 패밀리 표시).
  - **주의(색 충돌 관리)**: 결과 figure 토큰에서 GAT=`#d62728`(red)이고 relation-channel 토큰에서도 combat=`#d62728`. 두 의미는 *서로 다른 슬라이드 문맥*에서만 쓰이므로 한 슬라이드 안에 동시 등장시키지 말 것. Slide 8의 null 차트는 observed=`#1f77b4`, no_graph=`#7f7f7f`, edge_shuffle=`#d62728`(여기서 red=random null 강조).
- **future-work / advantage**: `#6b7280` muted + 점선(아직 메인 매트릭스 아님). root-cause/future 다리만 보라(`#6b21a8` accent, `#f5f3ff`/`#ede9fe` fill).
- **아이콘**: 미니멀 라인. zone = concentric circles(수축), time = arrow, team-node = filled circle, feature-group = small rounded chip, matrix = small grid glyph, elimination = skull/X(표지·관점에서는 미사용 권장), ranking = trophy/medal, high-ground = terrain wedge / elevation ▲, advantage edge = directed arrow (low→high), encoder = 작은 노드-엣지 글리프, no-graph = 노드만 글리프, future = 점선 테두리 배지.

---

## 2. Slide 1 — "Title"

### 레이아웃 (minimal academic title, 4-zone 세로 + faint full-bleed background)
```
┌────────────────────────────────────────────────────────────┐
│ [BG, full-bleed, very faint]  asset V1: real-match team graph │
│ ····· (concentric zone rings + a few node-edge teams) ·····  │
│                                                              │
│ [TOP-LEFT, small] venue / track placeholder (muted)          │
│                                                              │
│ [CENTER-LEFT] TITLE  (2 lines, large)                        │
│   From Proximity to Tactical Advantage:                      │
│   Rethinking Graph Relations for PUBG Survival Prediction    │
│                                                              │
│ [under title] SUBTITLE (1 line, muted accent)                │
│                                                              │
│ ──────────────── thin divider ────────────────              │
│ [BOTTOM-LEFT] Author(s) · Affiliation placeholders           │
│ [BOTTOM-RIGHT, small] date / conference placeholder          │
└────────────────────────────────────────────────────────────┘
```

### 핵심 시각 — faint background teaser (asset V1, *to be generated*)
- **콘텐츠**: 한 실매치(squad)의 한 순간을 추상화 — 옅은 동심원(safe-zone rings, `#cccccc`/`#6b7280` 아주 낮은 불투명도)과 그 안에 흩뿌려진 소수 팀(노드 4–6개 클러스터 + 엣지 몇 개). zone과 graph를 *한 장면*에 겹쳐 Slide 2의 dual-lens를 예고.
- **렌더 규칙**: 전체 화면 full-bleed, **불투명도 8–14%** 정도로 매우 옅게(텍스트 가독성 절대 우선). 색은 무채색 위주 + zone 링에 미세한 amber/blue 힌트만. 차트정크·라벨·축 없음. 순수 분위기용.
- **상태**: 아직 미생성(asset V1). 미생성 시 **플레이스홀더 = 균일 흰 배경**으로 빌드하고, 자리만 잡아둔다(레이아웃이 배경 유무에 안 깨지게).
- **레퍼런스**: 실제 매치 한 프레임의 좌표를 단순화해 생성(좌표 자산은 추후 제공). 정확한 데이터 정합성은 불필요 — *teaser*이므로 시각적 암시면 충분.

### 슬라이드 COPY (전부 영어·격식체)
- **TITLE** (2줄, 대형):
  `From Proximity to Tactical Advantage:`
  `Rethinking Graph Relations for PUBG Survival Prediction`
- **SUBTITLE** (1줄, muted `#6b7280` 또는 blue accent `#1d4ed8`):
  `A dynamic multi-relational team graph and a multi-resolution spatio-temporal model for elimination and final-placement prediction`
- **AUTHOR / AFFILIATION** (플레이스홀더, 좌하단):
  `[Author Name]` · `[Co-authors]`
  `[Department / Laboratory], [University]`
  `[author@institution.edu]`
- **VENUE / DATE** (플레이스홀더, 우하단 small `#6b7280`):
  `[Conference / Track] · [Year]`

### 빌드 노트 (Slide 1)
- 표지 제목은 본문 28pt 기준을 넘겨도 됨 — 표지 관례상 **~40–44pt** 권장, 2줄 균형(콜론에서 줄바꿈). 부제는 ~18–20pt.
- 정렬: 제목·부제·저자 블록은 **좌측 정렬** 유지(덱 일관). 우하단 venue/date만 우측 정렬 허용.
- 배경 asset V1은 *옵션* — 없으면 흰 배경. 있으면 텍스트 영역(좌측 60%)에는 노드/링이 몰리지 않게 배치해 가독성 확보.
- 색은 절제 — 표지에서 인코더 라인색(빨강/초록 등)을 강하게 쓰지 말 것. 무채색 + 미세 accent만.
- 모든 플레이스홀더는 대괄호 `[ ]`로 남겨 디자이너/저자가 채우게 한다.

---

## 3. Slide 2 — "Worldview: a contracting habitat where team relations evolve" (관점 — 발표의 정체성)

### 레이아웃 (split / dual-lens, 2-column + 상단 프레임 문장 + 하단 takeaway 띠)
```
┌────────────────────────────────────────────────────────────┐
│ TITLE: Two lenses on survival — relations and a shrinking    │
│        habitat                                               │
│ [sub] One framing line (the identity of this talk)           │
├──────────────────────────────┬─────────────────────────────┤
│ LEFT LENS (a)                 │ RIGHT LENS (b)               │
│  DYNAMIC TEAM-RELATION GRAPH  │  SHRINKING HABITAT (zone)    │
│  small team graph:            │  concentric contracting      │
│   nodes (teams) + edges       │   rings P1 → P7, outer→inner │
│   (relations evolve over time)│   + surviving-team count     │
│  caption: relations evolve    │   dropping along the rings   │
├──────────────────────────────┴─────────────────────────────┤
│ [FOOTER band] one-line takeaway (worldview statement)        │
└────────────────────────────────────────────────────────────┘
```

### 핵심 시각 — dual-lens (좌: 그래프 / 우: 수축 서식지)
**LEFT — dynamic team-relation graph (lens a):**
- 작은 팀 그래프: **노드 = 팀**(작은 원, neutral `#374151`), **엣지 = 팀 간 관계**. 노드 5–7개로 절제.
- **시간 진화 암시**: 두 개의 옅게 겹친 스냅샷("earlier" → "later")으로 엣지 구성이 *바뀌는* 것을 보여줌 — earlier는 `#cccccc` 옅게(ghost), later는 실선. 화살표/슬라이더 작은 표시 `t` 로 "relations evolve over time"을 암시.
- **관계의 다양성만 암시(디테일 금지)**: 엣지 굵기/스타일을 미세하게 달리해 "여러 종류의 관계"가 있음을 *암시*만 하고, proximity/combat/co-pressure 채널 이름·수식은 **여기서 쓰지 않는다**(뒤 modeling 슬라이드 담당). 절대 "단순 근접"으로 라벨하지 말 것.
- 캡션(작게, `#6b7280`): `Teams are nodes; their relations are edges — and the relations change as the match unfolds.`

**RIGHT — shrinking habitat / contracting play zone (lens b):**
- **동심원 수축**: 바깥(P1, 맵 전체)에서 안(P7)으로 좁아지는 concentric rings. 가장 바깥 = `#cccccc` 옅게, 안쪽으로 갈수록 약간 진하게(또는 amber `#d97706` 힌트 1개). zone 아이콘 = concentric circles 토큰과 일치.
- **생존팀 수 감소 라벨**: 각 ring 옆 또는 안쪽에 작은 숫자로 surviving-team count 추이 — 바깥→안: `~21 → ~14 → ~11 → ~9 → ~6 → ~4 → ~3` (PRESENTATION_DATA §G phase별 생존팀, 반올림값; P7 ~2.7→`~3`). 숫자는 muted, 마지막 `~3`만 살짝 강조.
- **시간·반경 보조**(아주 작게, 옵션): 가장 바깥에 `match start`, 가장 안쪽에 `endgame`. 정밀 분(分)·반경 수치는 여기 넣지 않음(뒤 G표/decision-window 슬라이드 담당) — 관점 슬라이드는 *추세*만.
- 캡션(작게, `#6b7280`): `The playable domain contracts phase by phase; fewer teams remain inside it.`

**두 렌즈의 연결(중앙 또는 하단)**: 좌·우를 잇는 얇은 화살표/브래킷 1개 + 짧은 연결어 `within` — "relations evolve *within* a contracting domain"을 시각적으로 봉합. 차트정크 금지, 얇은 라인 1개로만.

### 슬라이드 COPY (전부 영어·격식체)
- **TITLE** (2줄 가능):
  `Two lenses on survival: relations and a shrinking habitat`
- **SUBTITLE / framing line** (제목 아래, `#1d4ed8` accent 또는 `#374151`):
  `We view a PUBG match as a process unfolding within a contracting spatio-temporal domain, in which the relations among surviving teams continually evolve.`
- **LEFT LENS heading**: `(a) A dynamic team-relation graph`
  - body: `Each team is a node; the relations between teams are edges. These relations are not static — they form, shift, and dissolve as the match progresses.`
  - caption: `Teams are nodes; their relations are edges, and the relations change as the match unfolds.`
- **RIGHT LENS heading**: `(b) A shrinking habitat`
  - body: `The playable area contracts from the full map toward a final point. As the safe zone shrinks, the number of surviving teams falls phase by phase.`
  - caption: `The playable domain contracts phase by phase; fewer teams remain inside it.`
- **FOOTER takeaway band** (한 줄, 강조 — 발표의 정체성 문장):
  `Survival is a process within a contracting domain in which team relations continually evolve — this dual view, relations within a shrinking habitat, is the lens of this talk.`

### 빌드 노트 (Slide 2)
- **좌우 균형**: 두 렌즈는 시각 무게가 대등해야 함(어느 한쪽이 메인이 아님). 가운데 얇은 수직 구분선 `#cccccc` 1개.
- 생존팀 수 `~21 → … → ~3`는 PRESENTATION_DATA §G `phase → 게임 시각` 표의 생존팀 열(반올림). **정밀 수치/분·반경은 넣지 말 것** — 관점 슬라이드는 추세만, 정확한 숫자는 뒤 decision-window 슬라이드.
- **프레이밍 가드레일**(반드시 준수):
  - 좌측 그래프를 "단순 근접 그래프"로 라벨/암시 금지. "relations"(복수, 진화함)로만 표현.
  - proximity/combat/co-pressure, R-GCN, exp(−d/200), 우위 엣지 등 *모델 디테일은 일절 등장 금지* — 전부 뒤 슬라이드 담당.
  - smoothing/message-passing/attention/null-model 언급 금지(이건 interpretation 슬라이드).
  - 성능 수치(AUC/Spearman) 금지 — 관점 슬라이드는 정량 결과 없음.
- 시간 진화 암시는 *은은하게*: earlier/later 두 스냅샷 겹침 또는 작은 `t` 화살표 1개면 충분. 애니메이션 가정 금지(정적 슬라이드로 읽혀야 함).
- 언어: 슬라이드 텍스트 전부 영어·격식체. 구어체 금지.

---

## 4. Slide 3 — "Node design: each node is a surviving team's state vector"

### 레이아웃 (좌우 split, 좌=정의 / 우=한 노드 펼침)
```
┌────────────────────────────────────────────────────────────┐
│ TITLE: Nodes — surviving teams, encoded as per-team state    │
├──────────────────────────────┬─────────────────────────────┤
│ [LEFT ~40%]                   │ [RIGHT ~60%]                 │
│  one mini graph snapshot:     │  ONE node "exploded" into    │
│  N alive-team nodes (green),  │  its feature-category chips  │
│  one node highlighted (blue   │  (핵심 시각):                │
│  ring) → callout line ──────► │   ▸ Trajectory               │
│                               │   ▸ Zone exposure            │
│  caption: nodes appear /      │   ▸ Team state (health…)     │
│  disappear as teams are       │   ▸ Combat                   │
│  eliminated each phase.       │   ▸ Network role             │
│                               │   ▸ Habitat pressure         │
├──────────────────────────────┴─────────────────────────────┤
│ [BOTTOM band] takeaway: a node carries the team's full       │
│ spatial-combat STATE, not a flat aggregate.                  │
└──────────────────────────────────────────────────────────────┘
```

### 핵심 시각 — 한 노드를 feature-category 그룹으로 펼치기 (RIGHT ~60%)
- 중앙 좌측의 하이라이트된 노드(blue ring) 한 개에서 callout line이 우측으로 나가, 6개의 feature-category **chip 묶음**으로 펼쳐진다. 각 chip = 카테고리 이름(굵게) + 2–4개 대표 feature(muted, ~12pt). 한 카테고리당 한 줄 chip, 세로 스택.
  1. **Trajectory** — `path length · displacement · speed · sinuosity · safe-zone alignment`
  2. **Zone exposure** — `distance to boundary · outside-safe ratio · local enemy density`
  3. **Team state (STDB)** — `health · team spread · vehicle ratio · movement state`
  4. **Combat** — `damage dealt / taken · kills · groggies · distinct opponents`
  5. **Network role** — `combat degree / strength · proximity degree · reciprocity`
  6. **Habitat pressure** — `team-level zone deficit · local enemy pressure · accessible safe-area pressure`
- **시각적 규율**: 6 chip 모두 동일 neutral 스타일(중립색 텍스트, 얇은 좌측 액센트 바). *Network role* chip의 좌측 바만 옅게 강조해도 됨(그래프 토폴로지가 노드 안으로도 들어온다는 암시) — 단, 색은 절제. 각 chip은 동일 폭, 동일 라운드.
- **LEFT mini snapshot**: 5–7개 green node를 느슨한 배치로, edge는 옅은 회색 hairline 1–2개만(여기선 node가 주제이므로 edge는 약하게; edge 본론은 Slide 4). 하이라이트 노드 1개에 blue ring. 한 개의 흐릿한(eliminated, red, opacity ~30%) 노드를 가장자리에 두어 "phase마다 노드가 사라진다"를 암시.

### 슬라이드 COPY (영어·격식체)
- **TITLE**: `Nodes — surviving teams, encoded as per-team state`
- **LEFT caption** (snapshot 아래): `Each node is a team that is still alive at the cut-phase. The node set shrinks as teams are eliminated, so the graph is re-instantiated at every phase.`
- **RIGHT intro 한 줄** (펼침 위): `A node is not a flat score — it is a structured state vector grouped into six feature categories:`
- **6 chip 라벨/본문** (위 핵심 시각의 영문 그대로 사용).
- **BOTTOM band takeaway**: `Every node carries the team's full spatial and combat state — trajectory, exposure, health, combat, network role, and the pressure of the shrinking habitat — so the graph reasons over team conditions, not aggregate counts.`
- **우하단 small note** (caption ~12pt): `Match- and phase-level context (safe radius, alive-team count, shrink rate) is supplied as a separate global input, not broadcast into the node, to keep node features strictly team-attributed.`

### 빌드 노트 (Slide 3)
- 6개 chip은 **반드시** 6개 모두 보이게(스크롤·클리핑 금지). 카테고리명은 위 영문 그대로, feature 예시는 ~12pt muted.
- node 색 = alive green(`#2ca02c`), eliminated 잔상 = red(`#d62728`) opacity≈30%, 하이라이트 ring = blue(`#1d4ed8`). 본문/카테고리명 = neutral(`#374151`).
- 이 슬라이드의 edge는 **약하게**(회색 hairline) — edge는 Slide 4가 본론. 여기서 edge를 강조하면 두 슬라이드가 충돌한다.
- 언어 전부 영어·격식체. 구어체 금지.

---

## 5. Slide 4 — "Edges + dynamic adjacency: a 3-channel heterogeneous graph"

### 레이아웃 (상단 multiplex / 하단 3 heatmap + 텐서 표기)
```
┌────────────────────────────────────────────────────────────┐
│ TITLE: Edges — a 3-channel heterogeneous relation graph      │
├────────────────────────────────────────────────────────────┤
│ [TOP ~45%] MULTIPLEX graph: same N nodes, 3 stacked edge     │
│  layers (핵심 시각):                                          │
│    layer 1 proximity (blue, symmetric)                       │
│    layer 2 combat    (red, directed →)                       │
│    layer 3 co-pressure (amber, zone-gated)                   │
├──────────────────────────────────────────────────────────────┤
│ [MID] 3 channel cards, each: name · construction formula ·   │
│  symmetry tag                                                │
├──────────────┬──────────────┬──────────────────────────────┤
│ [BOTTOM-L] adjacency heatmap per channel (3 small panels,    │
│  real-match data asset) │ [BOTTOM-R] tensor + row-norm note  │
│                          │  A[T, E, N, N], row-normalized,   │
│                          │  evolves per phase                │
└──────────────────────────┴──────────────────────────────────┘
```

### 핵심 시각 A — multiplex graph (TOP ~45%)
- **같은 노드 집합** N개(green, Slide 3과 일관)를 **세 개의 반투명 평면(layer)** 위에 동일 좌표로 겹쳐 그린다(약간 3D-stacked / isometric 느낌 또는 세 패널 side-by-side 중 택1; stacked 권장 — "동일 노드, 다른 관계"가 한눈에).
  - **Layer 1 — proximity** (blue `#1d4ed8`): **무방향(symmetric)** 실선, 가까운 팀 쌍을 촘촘히 연결. 굵기 = 가중치(가까울수록 굵게).
  - **Layer 2 — combat** (red `#d62728`): **방향(directed)** 화살표 edge, 소수의 굵은 edge(교전한 쌍만). 비대칭(A→B ≠ B→A) 강조.
  - **Layer 3 — co-pressure** (amber `#d97706`): 같은 존-압박을 공유하는 팀들 사이의 옅은 edge, **zone-pressure-gated**(존 압박이 높을 때만 활성)임을 옅은 amber로.
- **시각 규율**: 세 layer는 색으로만 구분, 각 layer 옆에 미니 범례(색칩 + 채널명 + symmetry 태그: `symmetric` / `directed →` / `zone-gated`). 노드는 layer 간 동일 위치(점선 vertical guide로 "동일 노드" 명시).

### 핵심 시각 B — adjacency-matrix heatmap (BOTTOM-LEFT, 실매치 asset)
- 채널별 **N×N adjacency heatmap 3패널**(small multiples, 동일 색맵, 동일 스케일바). 채널 색조와 일치하도록 단색 sequential map: proximity=blues, combat=reds, co-pressure=ambers/oranges.
  - proximity = 대칭(상·하삼각 대칭), 비교적 dense.
  - combat = 비대칭(상·하삼각 불일치), sparse, 소수 강한 셀.
  - co-pressure = 블록/클러스터 구조(같은 압박 그룹) 암시.
- 각 패널 아래 캡션: `row-normalized`. **실매치 데이터로 생성**(개념 illust 아님) — 빌드 노트의 asset 지시 참조.

### 핵심 시각 C — 동적 adjacency 텐서 (BOTTOM-RIGHT)
- `A[T, E, N, N]` 표기를 큰 모노스페이스로. 각 축을 작은 라벨로:
  - `T` = phases (graph re-built per phase)
  - `E` = 3 relation channels (proximity / combat / co-pressure)
  - `N × N` = team-by-team
- 작은 phase-strip(P1 → P7) 위에 N이 줄어드는 시퀀스(노드 수 감소)로 "evolving per phase"를 시각화. row-normalization을 명시적 배지로: `each row sums to 1`.

### MID — 3 channel cards (construction formulas, 영어·격식체)
3개 카드 가로 배치, 각 카드 = 채널명(채널 색 액센트) + symmetry 태그 + **construction formula** + 1줄 의미.
- **Card 1 — Proximity** (blue, `symmetric`):
  - formula: `w_ij = exp(−d_ij / 200)`
  - meaning: `Spatial closeness between two surviving teams; the weight decays with inter-team distance.`
- **Card 2 — Combat** (red, `directed →`):
  - formula: `w_i→j = ln(1 + damage + 100·kills + 50·groggies)`
  - meaning: `Directed engagement intensity from team i onto team j, dominated by kills and knock-downs.`
- **Card 3 — Co-pressure** (amber, `zone-pressure-gated`):
  - meaning: `Teams forced into the same shrinking habitat; an edge is activated only when the shared zone pressure is high.`

### 슬라이드 COPY (영어·격식체)
- **TITLE**: `Edges — a 3-channel heterogeneous relation graph`
- **상단 한 줄 framing** (multiplex 위): `The same teams are connected by three distinct relation types, so the graph is multi-relational (heterogeneous), not a single proximity graph.`
- **multiplex 범례 태그**: `Proximity — symmetric`, `Combat — directed →`, `Co-pressure — zone-pressure-gated`.
- **3 channel cards COPY** (위 MID의 formula/meaning 그대로 사용).
- **tensor 영역 COPY**: `Dynamic adjacency tensor  A[T, E, N, N]` / `T = phases · E = 3 relation channels · N = surviving teams`. 그 아래: `The graph is re-instantiated and re-normalized at every phase; as teams are eliminated, N shrinks and the relation structure evolves.`
- **row-normalization flag** (명시적, 후반 분석 근거): `Each adjacency row is normalized to sum to one. This makes message passing a weighted mean over neighbours — a fact we return to when interpreting why graph topology, but not relation type, proves informative.`
- **BOTTOM 한 줄 takeaway**: `Three relation channels (proximity, combat, co-pressure) over a per-phase, row-normalized adjacency tensor — consumed directly by the R-GCN encoder, one relation matrix per channel.`

### 빌드 노트 (Slide 4)
- **multiplex graph**: 세 layer는 채널 색(`#1d4ed8` / `#d62728` / `#d97706`)으로만 구분, 노드는 layer 간 **동일 좌표**(점선 guide로 동일 노드 명시). combat layer는 반드시 **화살표(directed)**, proximity는 **무방향 실선(symmetric)** — symmetry 차이가 한눈에 보여야 함.
- **adjacency heatmap 3패널 = 생성할 asset**: 실매치(예: test split의 대표 1매치, 후반 phase) 데이터로 N×N 행렬을 채널별로 추출해 heatmap PNG 생성. proximity=대칭/dense, combat=비대칭/sparse, co-pressure=블록 구조가 드러나는 phase를 고를 것. 동일 색맵·동일 스케일바, 각 패널 `row-normalized` 캡션. 생성 후 이 슬라이드 BOTTOM-LEFT에 배치.
- **row-normalization은 반드시 명시**(텐서 배지 + COPY 문장 둘 다) — 이후 Interpretation 슬라이드(message passing = mean-field = low-pass filter)의 근거 hook.
- **금지**: 여기서 smoothing/attention/GAT을 설명하지 말 것(해석 슬라이드 소관). advantage edges 언급 금지(future work 슬라이드 소관).
- formula는 모노스페이스로 정확히 표기: `exp(−d/200)`, `ln(1 + damage + 100·kills + 50·groggies)`.
- 언어 전부 영어·격식체. 구어체 금지.

### Slide 3·4 공통 빌드 노트
- Slide 3·4는 이후 encoder/결과/해석 슬라이드와 **색·폰트 일관** 필수(§1 토큰 그대로). 특히 relation-channel 색(proximity blue / combat red / co-pressure amber)은 덱 전체에서 고정.
- Slide 3 node 색 = Slide 4 multiplex node 색 = alive green(`#2ca02c`)으로 동일.
- 두 슬라이드 모두 "node 집합이 phase마다 줄어든다(N shrinks)"를 일관되게 시각화 — Slide 3은 잔상 노드, Slide 4는 phase-strip + tensor T축.
- 이 두 장은 *개념·정의* 슬라이드(성능 수치 없음). 실수치는 결과 슬라이드 담당. 단 Slide 4의 adjacency heatmap만 실매치 데이터 asset.

---

## 6. Slide 5 — "We covered the encoder spectrum, not one model"

### 레이아웃 (좌 스펙트럼 리스트 / 우 클러스터 프리뷰, 상단 타이틀·하단 takeaway 띠)
```
┌──────────────────────────────────────────────────────────────────────┐
│ TITLE: We trained the standard graph-encoder spectrum — not one GNN   │
├───────────────────────────────────┬──────────────────────────────────┤
│ [LEFT ~52%] encoder spectrum list  │ [RIGHT ~48%] clustered-AUC preview│
│  No graph (TCN)  ─ baseline        │   hazard AUC, 8 MR models         │
│  GCN                               │   (compact 1-D strip, near-       │
│  GraphSAGE                         │    horizontal cluster)            │
│  GAT            ← best (0.8496)    │                                   │
│  R-GCN (3-channel heterogeneous)   │   0.836 ┊·············┊ 0.850     │
│   ├ FiLM                           │      no-graph     7-encoder cluster│
│   ├ attention-rank decoder         │                                   │
│   └ FARNet                         │   caption: cluster spread         │
│  ─────────────────────────────     │     ≈ 0.006 AUC                   │
│  ⌁ edge-conditioned (advantage)    │                                   │
│     — future work (dashed)         │                                   │
├───────────────────────────────────┴──────────────────────────────────┤
│ [FOOTER band] All encoders cluster — the architecture is not the lever.│
└──────────────────────────────────────────────────────────────────────┘
```
*(주: 우측 strip의 caption은 검증된 `≈ 0.006 AUC`(7 graph encoders 군집 폭)로 표기. 이전 초안의 "spread ≈ 0.011"은 그래프-vs-no-graph 이득 수치를 군집폭으로 오기한 것이므로 폐기.)*

### LEFT — encoder spectrum list (핵심 시각 1)
- **세로 리스트, 진행감 있게 배치** (위 = 단순/그래프 없음 → 아래 = 관계-인식 R-GCN 패밀리). 각 행은 인코더명 + 1줄 역할 설명 + 좌측 컬러 칩(§1 인코더 색 토큰).
- **그룹 구분 (얇은 라벨)**:
  - *Baseline (no graph)* — 1행
  - *Homogeneous graph encoders* — GCN, GraphSAGE, GAT (3행)
  - *Heterogeneous / relational* — R-GCN + 3 변형 (들여쓰기, R-GCN 아래 트리)
  - *Future work (dashed, muted)* — edge-conditioned advantage 1행
- **GAT 행에 best 배지** (`#d62728` 작은 점, "best on this matrix" 캡션). 단, 시각적으로 다른 행보다 크게 두드러지지 않게 — 군집을 강조하는 슬라이드이므로 "겨우 best"임을 전달.
- **R-GCN을 패밀리의 중심으로 명시**: "3-channel heterogeneous (multi-relational): proximity (symmetric) · combat (directed) · co-pressure (zone-gated)" 캡션을 R-GCN 행에 부속.
- 각 행 우측에 회색 작은 AUC 값(아래 COPY의 수치) — 하지만 강조는 우측 strip이 담당하므로 left의 숫자는 muted 처리.
- **advantage 행**: 점선 테두리 + muted 색 + "future work" 태그. 본 매트릭스 8모델과 시각적으로 분리(아래 구분선).

### RIGHT — clustered-AUC preview (핵심 시각 2)
- **1-D 가로 strip plot** (near-horizontal cluster를 보여주는 것이 목적). 가로축 = hazard AUC, 범위 `0.830 ~ 0.852` (좁게 잡아 군집과 no-graph 갭을 동시에 보이되, **데이터 잉크를 늘려 차이를 과장하지 말 것** — 캡션에 spread 수치 명기).
- **8개 포인트**, 각 인코더 색 토큰으로 찍기 (no-graph = `#7f7f7f` 점선 마커로 왼쪽 끝). 7개 그래프 모델은 `0.844–0.850` 구간에 **거의 겹쳐 군집** — 점들이 서로 가깝게 모여 "near-horizontal / one cluster"로 읽혀야 함.
- **두 개의 옅은 가이드**: (a) no-graph 위치에 회색 세로 점선 + 라벨 `no graph 0.836`; (b) 군집 중심에 옅은 음영 밴드(`#dbeafe`, 0.844–0.850) + 라벨 `7 graph encoders`.
- **GAT 포인트만 살짝 라벨** (`GAT 0.850 — best`). 군집 폭이 best−worst(그래프) ≈ 0.006임을 캡션으로.
- 캡션(하단, 12pt): `Cluster spread (7 graph encoders): ≈ 0.006 AUC; the no-graph baseline trails by only ≈ 0.013. Architecture barely moves the number.`
- **금지**: 막대그래프로 8모델을 세워 시각적 차이를 부풀리지 말 것. strip/dot이 군집 메시지에 정직.

### 슬라이드 COPY (영어·격식체 — 빌드 시 그대로 사용)

**Title**
`We trained the standard graph-encoder spectrum — not one GNN`

**Subtitle (제목 아래, muted 16pt)**
`Every encoder was trained on the same multi-resolution (MR) snapshot tier, the same causal split, and the same hazard objective. Only the encoder changes.`

**LEFT list rows (인코더명 — 역할 — AUC):**
- `No graph (TCN)` — `Temporal baseline; nodes only, edges removed.` — `0.8358`
- `GCN` — `Symmetric spectral convolution over the adjacency.` — `0.8461`
- `GraphSAGE` — `Sampling-and-aggregation over neighbours.` — `0.8438`
- `GAT` — `Attention-weighted neighbour aggregation.` — `0.8496  (best on this matrix)`
- `R-GCN` — `Per-relation convolution over the 3-channel heterogeneous graph: proximity (symmetric), combat (directed), co-pressure (zone-gated).` — `0.8460`
- `R-GCN + FiLM` — `Feature-wise modulation of relation messages.` — `0.8465`
- `R-GCN + attention-rank decoder` — `Rank-aware readout head on R-GCN encodings.` — `0.8449`
- `FARNet (R-GCN family)` — `Fused attention-relational variant.` — `0.8454`
- `Edge-conditioned (advantage)` — `Messages weighted by a learned function of pairwise relative state. Future work — preliminary.` — `dashed / not on this matrix`

**RIGHT strip labels & caption (위 핵심 시각 참조):**
- guide labels: `no graph 0.836`, `7 graph encoders 0.844–0.850`, `GAT 0.850 — best`
- caption: `Cluster spread (7 graph encoders): ≈ 0.006 AUC; the no-graph baseline trails by only ≈ 0.013. Architecture barely moves the number.`

**FOOTER band (한 줄 takeaway, 강조 띠):**
`All graph encoders — spectral, sampling, attention, and per-relation — land in the same narrow band. The architecture is not the lever; the relation representation is.`

**Future-work micro-note (advantage 행 옆, 12pt muted):**
`The edge-conditioned model conditions each message on a learned function of pairwise relative state. It is preliminary and is presented later as future work, not part of this main matrix.`

### 빌드 노트 (Slide 5)
- 본 슬라이드는 **결과 슬라이드(fig5/fig6)와 색·폰트 일관** 필수: 인코더 라인/포인트 색을 그대로 재사용해 다음 결과 슬라이드와 시각적 연속성 확보.
- **strip plot은 막대그래프가 아니다** — 군집(near-horizontal/one cluster)을 정직하게 보여주는 1-D dot/strip으로 그릴 것. 축 범위를 과도하게 확대해 0.006 AUC 차이를 시각적으로 부풀리지 말 것.
- GAT의 "best"는 *겨우* best임을 전달해야 함(2위 FiLM 0.8465 대비 +0.003; 그래프 군집 폭 ≈ 0.006). best를 화려하게 강조하지 말 것 — 이 슬라이드의 결론은 "best가 의미 없다"이다.
- R-GCN 행에 3채널 이종(heterogeneous) 그래프임을 반드시 명시 — "simple proximity"가 아니라 proximity(symmetric)·combat(directed)·co-pressure(zone-gated) 3-relation. R-GCN과 3변형은 동일 패밀리로 시각적 그룹핑(green 계열).
- advantage(엣지-조건부) 행은 점선·muted로 본 매트릭스 8모델과 명확히 분리하고 "future work" 태그를 붙일 것. 이 슬라이드에서는 **수치를 제시하지 않는다**(예비 수치는 future work 슬라이드 담당).
- 스무딩·메시지 패싱·어텐션의 *이유*(왜 군집하는가, mean-field/low-pass)는 **이 슬라이드에 넣지 말 것** — 다음 해석/한계 슬라이드 담당. 이 슬라이드는 "스펙트럼을 다 덮었고, 군집한다"는 *사실*만 전달.
- 언어: 슬라이드 텍스트 전부 영어·격식체. 구어체 금지.
- 수치 출처(PRESENTATION_DATA §B main 매트릭스 hazard AUC): mr_gat 0.8496 · mr_rgcn_film 0.8465 · mr_gcn 0.8461 · mr_rgcn 0.8460 · mr_farnet 0.8454 · mr_rgcn_attn_rank 0.8449 · mr_sage 0.8438 · mr_tcn 0.8358. 검증: 7 graph encoders 군집 폭 = 0.8496−0.8438 = 0.0058 ≈ 0.006; no-graph는 best 대비 0.0138 ≈ 0.013 뒤짐.

---

## [참조] Labeling + Metrics 슬라이드 (A, B)

이 두 슬라이드(예측 타깃 라벨링 + 채점 지표)는 **별도 설계서에 완전 설계됨**:
> **`docs/deliverables/design_spec_labeling_metrics.md`**
> - Slide A — "Prediction targets: leakage-safe labeling" (cut-phase timeline, hazard_next ∈ {0,1}, final_rank, at-risk mask)
> - Slide B — "How each target is scored" (Hazard AUC · rank Spearman ρ, 4팀 예시표)

덱 내 위치: **Slide 5(Models) 다음, Slide 6(Results — elimination) 이전**. 색·폰트·시간축 토큰은 본 §1 공유 디자인 시스템과 동일하다(별도 설계서의 §1과 verbatim 일치). 빌드 시 그 문서를 그대로 사용하고, 본 덱 흐름의 이 지점에 배치한다.

---

## 7. Slide 6 — "Results: elimination — all encoders cluster"
*두 결과 슬라이드는 이미 생성된 figure(fig5/fig6)를 소비한다 — 그림은 재생성하지 말고 배치·캡션·메시지 블록만 설계한다.*

### 레이아웃 (좌 figure / 우 메시지, 60/40)
```
┌────────────────────────────────────────────────────────────┐
│ TITLE: Elimination prediction — architecture is not the lever│
├──────────────────────────────────────┬─────────────────────┤
│ [LEFT ~62%]                          │ [RIGHT ~38%]         │
│   fig5_hazard_by_phase.png           │  MESSAGE BLOCK       │
│   (per-phase hazard AUC, 5 encoders, │   • headline stat    │
│    min–max range band, game-time     │   • 3 takeaway lines │
│    axis w/ shrinking-zone bands)     │   • small "what this │
│                                      │     is / isn't" note │
├──────────────────────────────────────┴─────────────────────┤
│ [FOOTER STRIP] one-line bridge to the interpretation slide   │
└────────────────────────────────────────────────────────────┘
```

### 핵심 시각 — fig5_hazard_by_phase.png (이미 생성됨; 배치만)
- 가로축 = **game time (min)**, phase 밴드 P1…P7가 점점 진해지는 파랑으로 깔려 있고 각 밴드 상단에 safe-zone 반경(5,506 m → 98 m)이 인쇄됨. 우측으로 갈수록 zone이 수축 = 난이도 상승을 시각적으로 전달.
- 세로축 = **Next-phase hazard AUC** (~0.725–0.92).
- 5개 인코더 라인(§1 색 토큰) + 회색 **min–max range band**(인코더 스프레드). 핵심 시각 메시지 = 라인들이 거의 겹쳐 한 다발로 내려간다.
- **디자인 강조(오버레이, figure 위 가벼운 주석으로)**: P3 부근(가장 넓은 지점)에 얇은 수직 측정선 + 라벨 `encoder spread ≈ 0.02 (widest)`. P1/끝단의 좁은 곳은 건드리지 말 것. 회색 dashed = No graph가 다발 *아래*에 분리돼 있음을 화살표 없이 라벨로만 표시: `No graph (lower envelope)`.
- figure는 벡터 재생성하지 말고 PNG 그대로 배치; 위 두 라벨만 깔끔한 텍스트 오버레이로 추가.

### 메시지 블록 (우측, exact COPY)
- **HEADLINE(큰 수, ~22pt)**: `Encoder spread per phase: 0.009 – 0.021 AUC`
  - 보조 1줄(muted): `All five encoders fall within a ~0.01–0.02 AUC band at every phase.`
- **TAKEAWAYS (불릿 3개, ~16pt, 각 앞에 인코더 swatch 사용)**:
  1. `Architecture is not the lever. GCN, GraphSAGE, GAT and R-GCN cluster at ~0.84–0.85 hazard — swapping the graph encoder moves the result by at most ~0.02 AUC.`
  2. `The graph itself helps only a little. Averaged over phases, the graph encoders beat the no-graph model by +0.011 AUC (0.798 → 0.809) — a small, smoothing-type gain.`
  3. `Difficulty rises as the zone contracts. Hazard AUC falls from 0.89 at Phase 1 (5,506 m) to 0.74 at Phase 7 (98 m): late-game eliminations are intrinsically harder to anticipate.`
- **WHAT-THIS-IS / ISN'T 미니 노트 (작게, ~12pt, `#6b7280`)**:
  `The small graph gain is a mean-field smoothing effect, not evidence that the observed topology carries elimination signal — that question is settled on the next slide.`

### FOOTER STRIP (한 줄 bridge, exact COPY)
`Because all encoders converge, the question is not which architecture, but whether the relations are informative — examined next.`

### 빌드 노트 (Slide 6)
- figure는 `fig5_hazard_by_phase.png` PNG 그대로; 오버레이는 2개 라벨뿐(spread 측정선, "No graph (lower envelope)"). 추가 차트정크 금지.
- HEADLINE 수치 `0.009–0.021`와 `+0.011 (0.798→0.809)`는 검증값 — 변경 금지. (figure 배열 검산: per-phase spread = [0.011, 0.018, 0.021, 0.017, 0.015, 0.009, 0.011]; graph 평균 0.809 − no-graph 0.798 = +0.011.)
- 인코더 swatch 색은 figure 라인 색과 1:1 동일해야 함(범례 재학습 부담 제거).
- 메시지 블록은 figure 우측에 세로로; 모바일/축소 시 figure 아래로 흐르되 순서 유지(headline→takeaways→note).

---

## 8. Slide 7 — "Results: ranking — wider spread, locks in late"

### 레이아웃 (좌 figure / 우 메시지, 60/40 — Slide 6과 동일 골격)
```
┌────────────────────────────────────────────────────────────┐
│ TITLE: Ranking prediction — graph and GAT help more, late    │
├──────────────────────────────────────┬─────────────────────┤
│ [LEFT ~62%]                          │ [RIGHT ~38%]         │
│   fig6_rank_by_phase.png             │  MESSAGE BLOCK       │
│   (per-phase rank Spearman, 5        │   • headline stat    │
│    encoders, min–max range band,     │   • 3 takeaway lines │
│    game-time axis w/ zone bands)     │   • honest null note │
├──────────────────────────────────────┴─────────────────────┤
│ [FOOTER STRIP] one-line bridge to interpretation/limits      │
└────────────────────────────────────────────────────────────┘
```

### 핵심 시각 — fig6_rank_by_phase.png (이미 생성됨; 배치만)
- 가로축·phase 밴드·zone 반경 = Slide 6과 동일(시각적 일관성 유지, 두 그림을 같은 시간축으로 읽게 함).
- 세로축 = **Final-rank Spearman (ρ)** (~0.61–0.82).
- 5개 인코더 라인 + 회색 **min–max range band**. 세 가지가 한눈에 보여야 함: (a) band가 Slide 6보다 **눈에 띄게 두껍다**(스프레드↑), (b) 빨강(GAT)이 다발 *위*로 분리돼 가장 높다, (c) 라인들이 후반(P5–P6)으로 갈수록 **올라가서 평탄해진다**(lock-in).
- **디자인 강조(오버레이)**: 초반(P1–P2) 가장 두꺼운 band 지점에 측정선 + `spread ≈ 0.10 (early)`, 후반(P5–P6)에 `spread ≈ 0.06 (late)` — 스프레드가 순위에서 크고, 후반에 좁아짐을 대비. GAT 라인 끝에 `GAT — best ranker` 라벨. P3→P6 구간 위에 옅은 amber(`#fef3c7`) 음영 + `rank locks in (Δρ +0.085, 3→6)`.

### 메시지 블록 (우측, exact COPY)
- **HEADLINE(큰 수, ~22pt)**: `Encoder spread per phase: 0.056 – 0.106 ρ`
  - 보조 1줄(muted): `Roughly 5× the elimination spread — encoder choice matters more for ordering teams.`
- **TAKEAWAYS (불릿 3개, ~16pt, swatch 사용)**:
  1. `Here the graph — and especially GAT — helps. GAT leads every phase and beats the no-graph model by +0.075 ρ on average (0.69 → 0.76); the other graph encoders sit between.`
  2. `Ranking locks in toward late game. Spearman climbs and flattens through Phases 3–6 (+0.085 ρ, 3→6), the window where the final standings are effectively decided (game time 16–24 min).`
  3. `Late ranking is measured among the survivors. By Phase 6 only ~4 teams remain, so ρ reflects ordering of the top-placed teams.`
- **HONEST NULL 노트 (작게, ~12pt, `#6b7280` — 반드시 포함)**:
  `The wider spread comes from smoothing plus the rank decoder, not from relational topology: the null test holds for ranking too (observed ≤ edge_shuffle). A better order does not mean the edges encode "who beats whom".`

### FOOTER STRIP (한 줄 bridge, exact COPY)
`Both targets improve with the graph, yet neither improvement traces to the observed topology — the next slide explains why.`

### 빌드 노트 (Slide 7)
- figure는 `fig6_rank_by_phase.png` PNG 그대로; 오버레이 = 2개 spread 측정선, GAT 라벨, P3–P6 amber 음영+lock-in 라벨. 그 외 추가 금지.
- HEADLINE 수치 `0.056–0.106`, takeaway의 `+0.075 (0.69→0.76)`, `+0.085 (3→6)`, `16–24 min`, `~4 teams`는 모두 검증값 — 변경 금지. (figure 배열 검산: per-phase spread = [0.106, 0.097, 0.072, 0.061, 0.056, 0.059, 0.074]; GAT 평균 0.763 − no-graph 평균 0.688 = +0.075; P3→P6 lock-in = +0.085 per PRESENTATION_DATA §E; P6 생존팀 ~4 per §G; game time 16–24분 per §G phase 4–6.)
- **honest null 노트는 필수** — 이 슬라이드가 "GAT가 이겼다 = 위상이 정보적"으로 오독되는 것을 막는 장치. 삭제·완화 금지.
- 색·폰트·시간축은 Slide 6과 픽셀 단위로 일관(두 결과 슬라이드가 자매처럼 읽혀야 함). amber 음영은 ranking/award 토큰(`#fef3c7`)만 사용.

### Slide 6·7 공통 빌드 노트
- 두 슬라이드는 **이전 Labeling/Metrics 슬라이드 및 서로 간** 색·폰트·시간축 일관 필수(§1 토큰 그대로).
- **figure는 재생성하지 말 것** — `fig5_hazard_by_phase.png`, `fig6_rank_by_phase.png`를 그대로 배치하고, 위에 명시한 소수의 텍스트 오버레이만 추가.
- 인코더 라인 색과 슬라이드 텍스트 swatch는 1:1 동일(No graph 회색 dashed, GAT 빨강 등).
- 언어: 슬라이드 텍스트 전부 영어·격식체. 구어체 금지("a lot", "kinda" 등 금지).
- 프레이밍 가드: 어느 슬라이드도 "simple proximity", "2D→3D as a contribution", 또는 "topology is informative"라고 말하지 말 것. 그래프 이득 = 평활화(+rank decoder), 위상 = null. 관계의 *종류*가 정보성을 가른다는 본 기여 메시지는 future-work 슬라이드 소관이며 여기선 selling하지 않는다.
- 두 HEADLINE 수치가 핵심 대비(0.009–0.021 vs 0.056–0.106)이므로 같은 시각 위계(폰트·위치)로.

---

## 9. Slide 8 — "Interpretation & limitations: the gain is smoothing, not topology"

### 덱 내 위치 & 목적
- **위치**: 결과(Slide 6 elimination / Slide 7 ranking) *직후*, future work(Slide 9) *직전*. 이 슬라이드는 덱의 **클라이맥스 진단** — "결과가 왜 이렇게 나왔는가"를 설명한다. Slide 5에서 모든 그래프 인코더(GCN/GraphSAGE/GAT/R-GCN)가 ~0.84–0.85 hazard로 군집했다는 사실의 *원인*을 여기서 푼다.
- **핵심 메시지(한 문장)**: 동적 팀 그래프가 주는 이득(+0.011)은 row-normalized 인접행렬 위의 메시지 전달 = mean-field **평활화(low-pass filtering)** 일 뿐이며, *관측된 위상(topology) 자체는 무신호*다 (observed ≤ edge_shuffle, 3/3 시드). 근본 원인: 3채널이 **상호작용(존재·강도)** 을 부호화할 뿐 **상대적 우위(advantage)** 를 부호화하지 않는다.
- **이 슬라이드가 평활화/메시지전달/어텐션이 등장하는 유일한 곳** — 독립 튜토리얼이 아니라 *결과를 설명하는 한계 진단*으로만 등장한다(framing rule 준수).
- **검증됨(PRESENTATION_DATA §F)**: 세 null 대조군 모두 row-normalized(위상만 측정). 3-seed 평균 hazard AUC. match-level paired CI로 phase 1–3 observed 유의 열세, 4–7 ns 확정.

### 레이아웃 (좌측 figure / 우측 진단 스택, 하단 root-cause 띠)
```
┌────────────────────────────────────────────────────────────────────┐
│ TITLE: Why the observed topology added little — smoothing, not       │
│        relational signal                                             │
├──────────────────────────────────┬─────────────────────────────────┤
│ [LEFT ~52%]                       │ [RIGHT ~48%] DIAGNOSIS (3 steps)│
│  fig2_graph_null.png              │                                 │
│  null-model control bar chart     │  ① The null-model verdict       │
│  (no_graph / observed / shuffle)  │  ② Architecture is not the lever│
│  + small paired-CI callout strip  │  ③ The mechanism: low-pass A    │
├──────────────────────────────────┴─────────────────────────────────┤
│ [BOTTOM BAND] ROOT CAUSE — what the three channels do and do not     │
│  encode  →  hands off to Future work (Slide 9)                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 핵심 시각 — LEFT: null-model control chart (참조: `docs/presentation_figures/fig2_graph_null.png`)
- 디자인 도구는 PNG를 **레이아웃 참조**로 하되 벡터로 재생성. 3개 막대, y축 = "Next-phase hazard AUC", 범위 0.825–0.855 (작은 효과를 정직하게 보이게 하는 zoom된 축 — 축 시작이 0이 아님을 캡션에 명시).
  - `no_graph` = **0.834**, fill `#7f7f7f` (results-encoder 토큰의 "No graph" 색과 일치).
  - `observed (real topology)` = **0.845**, fill `#1f77b4` (model-input/observed blue).
  - `edge_shuffle (random null)` = **0.847**, fill `#d62728` (red, "random ≥ observed"를 강조).
- **빨간 주석(우상단, edge_shuffle 막대 위)**: `random ≥ observed (3/3 seeds)` — `#d62728`, edge_shuffle 막대 꼭대기로 향하는 얇은 화살표.
- **보라/뮤트 콜아웃 박스(observed 막대 옆)**: `Graph helps (+0.011) = mean-field SMOOTHING, not topology` — fill 옅은 보라 `#ede9fe`, 텍스트 `#6b21a8`.
- **막대 아래 작은 paired-CI 스트립(figure 하단 또는 figure 바로 밑 캡션 줄)**:
  - `Match-level paired (observed − edge_shuffle): phase 1–3 observed significantly worse (ΔAUC −0.002 to −0.004, CI < 0); phase 4–7 not significant — no phase is significantly better.`
- **figure 캡션(12pt, muted `#6b7280`)**: `All three controls are row-normalized, so the comparison isolates topology. node_shuffle ≈ observed (team identity carries no signal). y-axis is truncated; the entire effect spans 0.013 AUC.`

### RIGHT — Diagnosis stack (3 numbered steps, 위→아래)
세로로 쌓인 3개 카드/스텝. 각 스텝은 굵은 한 줄 결론 + 한 줄 근거. 좌측에 얇은 세로 accent 바.

- **Step ① — The null-model verdict** (accent `#374151`):
  - 결론: `Randomizing the edges does not hurt — it helps.`
  - 근거: `Across 3 seeds, edge_shuffle (0.847) ≥ observed (0.845) ≥ no_graph (0.834). Identity is also irrelevant: node_shuffle ≈ observed. The real wiring carries no usable signal beyond having a graph at all.`

- **Step ② — Architecture is not the lever** (accent `#374151`):
  - 결론: `Every encoder lands at the same place (~0.84–0.85 hazard).`
  - 근거: `GCN, GraphSAGE, GAT and R-GCN all cluster. Even GAT — which learns to weight neighbours by attention — barely beats plain mean aggregation. If distinguishing neighbours barely helped, the bottleneck is not the encoder; it is the relation being represented.`

- **Step ③ — The mechanism: message passing as a low-pass filter** (accent `#1d4ed8`, model-input blue):
  - 결론: `Message passing over a row-normalized A is mean-field smoothing.`
  - 근거: `A row-normalized adjacency makes each update a weighted average of neighbours — a low-pass filter that denoises a team's hazard toward its local mean. This averaging is what the +0.011 buys; it is indifferent to which specific neighbours are connected, which is exactly why shuffled edges reproduce it.`

### BOTTOM BAND — ROOT CAUSE (핵심 진단, 화면 폭 전체 띠)
- 옅은 보라 배경 `#f5f3ff`, 좌측 굵은 accent 바 `#6b21a8`.
- **헤드라인(굵게)**: `Root cause — the three channels encode interaction, not advantage.`
- **본문**: `The heterogeneous graph records proximity (who is near whom), combat (who engaged whom, and how hard) and co-pressure (who is squeezed together by the zone). Each channel encodes the existence and intensity of an interaction — not the relative state between two teams. Mean aggregation over such edges cannot construct a pairwise comparison such as "I am lower than the enemy beside me", so smoothing is all the topology can offer.`
- **트랜지션 한 줄(band 우측 끝 또는 band 아래, 작게, `#6b7280`)**: `This is an honest negative on topology — and it points directly to the fix: change the type of relation on the edge.  → Future work.`

### 슬라이드 한 줄 takeaway (상단 제목 바로 아래 띠, 또는 슬라이드 하단)
`The dynamic graph helps by smoothing, but its observed topology is uninformative — because the relations encode interaction, not relative advantage.`

### 빌드 노트 (Slide 8)
- **색 일관성 필수**: no_graph 막대 = `#7f7f7f` (results-encoder "No graph" 토큰), observed = `#1f77b4` (model-input blue 계열), edge_shuffle = `#d62728` (red). Step ③ 및 mechanism 강조는 model-input blue `#1d4ed8`. root-cause 띠만 보라(`#6b21a8`/`#f5f3ff`)로 future-work으로의 시각적 다리 역할.
- **figure는 `fig2_graph_null.png`를 레이아웃 참조**로 벡터 재생성. y축 truncation(0.825–0.855)을 *반드시* 캡션에서 명시 — 작은 효과를 과장하지 않는 정직성.
- **수치는 verbatim 고정**(PRESENTATION_DATA 2026-06-21 §F): no_graph 0.834 / observed 0.845 / edge_shuffle 0.847; paired phase 1–3 ΔAUC −0.002~−0.004 (CI<0); phase 4–7 ns; node_shuffle ≈ observed. 임의 변경 금지.
- **평활화/메시지전달/어텐션 용어는 이 슬라이드에만** 등장 — Step ②(어텐션이 평균을 거의 못 이김)·Step ③(row-normalized A = low-pass)·root-cause 띠에 한정. 모델 슬라이드(5)로 끌어올리지 말 것.
- **advantage 엣지(상대 우위)는 여기서 언급하지 않음** — root-cause 띠의 마지막 트랜지션 줄에서 "change the type of relation"으로만 암시하고, 구체 수치(+0.0033, node-abs 통제)는 Slide 9(future work) 담당.
- **3-zone 밀도**: figure가 좌측 ~52%를 차지, 우측 diagnosis 3스텝은 충분한 여백으로 — 텍스트 과밀 금지. 차트정크 없음, 얇은 구분선.
- 언어: 슬라이드 텍스트 전부 영어·격식체. 구어체 금지. 좌측 정렬.

---

## 10. Slide 9 — "Future work: contact-gated relative-advantage relations"

### 레이아웃 (상단 띠 + 좌측 개념도 60% / 우측 증거 패널 40% + 하단 caveat 띠)
```
┌────────────────────────────────────────────────────────────┐
│ TITLE: Future work — relations as contact-gated tactical     │
│        advantage                                             │
│ [SUBTITLE 띠]: Preliminary, validated. Not part of the       │
│                present results.                              │
├──────────────────────────────────┬─────────────────────────┤
│ [LEFT, ~60%]  핵심 시각           │ [RIGHT, ~40%]            │
│  high-ground / advantage-edge     │  EVIDENCE panel          │
│  concept (asset V5)               │  (4 validation chips,    │
│                                   │   stacked, with CIs)     │
│  ▲ high team  ←edge— ▽ low team    │                          │
│  contact-gated directed edge,     │  + node-abs control row  │
│  weighted by relative advantage   │  + descriptive note      │
├──────────────────────────────────┴─────────────────────────┤
│ [BOTTOM 띠] caveat: requires full re-base; effect sizes are  │
│             small and preliminary.                           │
└────────────────────────────────────────────────────────────┘
```

### 핵심 시각 — high-ground / advantage-edge concept (asset **V5**, to be generated)
- **장면**: 단순화된 terrain 단면(side-view) 위에 근접한 두 팀 노드. 하나는 능선/고지(▲, 위쪽), 하나는 저지(▽, 아래쪽). 둘 사이를 잇는 **directed edge(저→고로 향하는 화살표 또는 고→저 우위 방향)**, 화살표 굵기 = 상대 우위 크기.
- **엣지 게이트 시각화**: 두 팀이 **접촉(근접/교전) 범위 안**일 때만 엣지가 그려짐을 점선 원(contact gate, `#1d4ed8` accent)으로 표시. 멀리 떨어진 제3의 팀 쌍은 엣지 없음(게이트 OFF) — "contact-gated" 개념을 한 컷으로.
- **우위 채널 4개 미니 아이콘** (엣지 옆 작은 범례, muted): elevation(▲ 고도), numbers(유닛 수 점 표시), health(얇은 바), zone(동심원 — zone-pressure). caption: `Edge weight = relative advantage across elevation, numbers, health, and zone pressure.`
- **색**: 고지 팀 노드 = neutral `#374151` 외곽 + 옅은 amber `#fef3c7` glow(우위 = 유리), 저지 팀 = 옅은 red `#d62728` 외곽 hint(불리, 다음 phase 더 죽음). terrain = muted gray fill. 절제.
- **핵심 라벨(엣지 위)**: `"I am lower than the adjacent enemy" — a pairwise relation that mean aggregation cannot construct.`
- **대조 미니컷(좌하단, 작게)**: 이전 덱의 proximity 그래프 썸네일(평활화, 무신호) → 화살표 → advantage 엣지. caption: `From interaction topology (smoothing) to relational advantage (signal).`

### EVIDENCE panel (우측 — 4중 검증 + 결정적 통제, 반드시 포함)
- 패널 제목: `Preliminary evidence (phase tier, observed advantage graph)`
- **검증 chip 4개**(stacked, 각 줄에 효과크기 + 95% CI; CI가 0을 제외함을 강조):

  | Control (null) | ΔAUC (observed − null) |
  |---|---|
  | vs **edge_shuffle** (topology) | **+0.0033 [+0.0023, +0.0043]** |
  | vs node_shuffle (team identity) | +0.0023 |
  | vs no_graph | +0.0051 |
  | k-ablation k∈{3,5,8} | +0.0035 / +0.0033 / +0.0039 (all significant) |

- **결정적 통제 row**(별 강조 ★, amber accent box):
  - `Node-absolute control: even when absolute elevation, alive, health, and zone are given directly to the nodes, the advantage edge remains significant: +0.0023 [+0.0013, +0.0033] (~70% of the effect persists).`
  - 한 줄 결론(bold, neutral): `The signal is irreducibly relational — it cannot be reduced to node features.`
- **descriptive note**(작게, muted, 패널 하단): `Descriptive physics: among adjacent team pairs, the lower team is eliminated more often in the next phase (overall z = +10.5; strong early–mid, weaker late). Median engagement distance 179 m vs 1901 m for random pairs.`

### 슬라이드 한 줄 takeaway (하단 caveat 띠 위 또는 부제)
`Proximity topology carried no signal, but contact-gated relative advantage revives a genuinely relational signal — the TYPE of relation, not the GNN architecture, is the lever.`

### 하단 caveat 띠 (정직성 — 반드시)
`These are preliminary results on the phase tier and require a full model re-base before integration; effect sizes are small. Presented as future work, not as a claim of the present study.`

### 빌드 노트 (Slide 9)
- **asset V5(high-ground / advantage-edge concept)는 신규 생성** — terrain 단면 + 근접 두 팀 노드(고/저) + contact-gated directed edge + 4-채널 우위 범례. 벡터로 깔끔하게(노드·엣지·게이트 원·terrain을 컴포넌트로). proximity→advantage 대조 미니컷 포함.
- **수치는 future-work 문서(FUTURE_DESIGN_3D_RELATIONAL_2026-06-21.md) 값만 사용**: edge_shuffle +0.0033 [+0.0023,+0.0043], node_shuffle +0.0023, no_graph +0.0051, k-ablation +0.0035/+0.0033/+0.0039, node-abs control +0.0023 [+0.0013,+0.0033], descriptive z=+10.5, 거리 179m vs 1901m. **다른 출처 수치 금지.**
- **반드시 "preliminary / future work" 띠를 상·하단 둘 다** 두어 메인 결과와 시각적으로 분리(메인 결과보다 절제된 색·작은 강조).
- "2D→3D"를 기여로 쓰지 말 것(correctness일 뿐). advantage 신호는 항상 `(preliminary)` 병기.
- 언어: 슬라이드 텍스트 전부 영어·격식체. 구어체 금지.

---

## 11. Slide 10 — "Conclusion"

### 레이아웃 (중앙 정렬 thesis 1줄 + 2-clause 대조 + 1줄 value statement — 미니멀)
```
┌────────────────────────────────────────────────────────────┐
│                                                              │
│ TITLE (작게, kicker): Conclusion                             │
│                                                              │
│ [CENTER, 큰 thesis 한 줄, 화면 ~40%]                          │
│  The TYPE of relation determines whether graph topology      │
│  is informative.                                             │
│                                                              │
│ [두 칼럼 대조, thesis 아래]                                   │
│  ┌───────────────────────┐   ┌───────────────────────┐      │
│  │ INTERACTION relations │   │ ADVANTAGE relation    │      │
│  │ proximity / combat /  │ → │ relative tactical      │      │
│  │ co-pressure           │   │ advantage             │      │
│  │ → smoothing only      │   │ → irreducibly         │      │
│  │   (no topology signal)│   │   relational signal   │      │
│  │                       │   │   (preliminary)       │      │
│  └───────────────────────┘   └───────────────────────┘      │
│                                                              │
│ [BOTTOM, value statement 1줄]                                │
│  The value lies in designing game-mechanic-aligned           │
│  relations, not in using a GNN per se.                       │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

### 핵심 시각 — minimal 대조 다이어그램
- **중앙 thesis**: 한 줄, ~28–32pt, neutral `#374151`, "TYPE" 만 약하게 강조(weight 또는 `#1d4ed8`). 차트정크 없음 — 텍스트가 주인공.
- **2-clause 대조 카드**(좌/우, 동일 크기, 가운데 화살표 `→`):
  - 좌 카드 — INTERACTION: 헤더 muted `#6b7280`; 본문 "smoothing only". 작은 아이콘 = blurred/low-pass wave(평활화 암시). 색은 muted gray 계열(신호 없음 = 무채색).
  - 우 카드 — ADVANTAGE: 헤더 amber `#d97706` + `#fef3c7` fill; 본문 "irreducibly relational signal". 작은 아이콘 = directed advantage arrow(▲). `(preliminary)` 작게 muted 병기.
- **하단 value statement**: 1줄, ~16–18pt, neutral. bold 없이 차분하게. 강한 마무리.
- 전체적으로 여백 많고 선 적게 — 덱의 마지막 슬라이드답게 "minimal, strong closing".

### 슬라이드 COPY (exact, English·formal)
- **Kicker / 상단 작은 제목**: `Conclusion`
- **THESIS (center)**: `The type of relation determines whether graph topology is informative.`
- **좌 카드 (INTERACTION)**:
  - 헤더: `Interaction relations`
  - 부제: `proximity · combat · co-pressure (3-channel heterogeneous graph)`
  - 결론: `Yield smoothing only — the observed topology is no better than a shuffled one.`
- **우 카드 (ADVANTAGE)**:
  - 헤더: `Relative tactical advantage`
  - 부제: `elevation · numbers · health · zone, contact-gated`
  - 결론: `Shows an irreducibly relational signal that cannot be reduced to node features (preliminary).`
- **VALUE statement (bottom)**: `The value lies in designing game-mechanic-aligned relations, not in using a GNN per se.`

### 슬라이드 한 줄 takeaway (덱 클로징)
`Architecture is not the lever; the relation representation is.`

### 빌드 노트 (Slide 10)
- **텍스트 중심·여백 중심** — 차트·이미지 최소. thesis 한 줄이 시각적 주인공. 2-clause 대조 카드만 보조.
- **정직성**: "2D→3D"를 기여로 쓰지 말 것(correctness일 뿐). advantage 신호는 항상 `(preliminary)` 병기. interaction 관계는 "smoothing only / observed ≤ shuffle"로 정직하게.
- 색·폰트는 §1 공유 시스템 그대로 — 덱 전체와 완전 일관.
- 언어: 슬라이드 텍스트 전부 영어·격식체. 구어체 금지.

---

## 12. 덱 전체 빌드 노트 (디자인 도구용)
- 모든 슬라이드는 **§1 공유 디자인 시스템**(색 토큰·폰트·아이콘)을 verbatim 따른다. 슬라이드별 색 재정의 금지.
- **인코더 라인/포인트 색**(Slide 5·6·7·8)과 **relation-channel 색**(Slide 4·5)은 덱 전체에서 고정. fig5/fig6/fig2의 실제 라인·막대 색과 1:1 일치해야 함.
- **figure 재생성 금지**: `fig5_hazard_by_phase.png`, `fig6_rank_by_phase.png`는 그대로 배치(소수 오버레이만). `fig2_graph_null.png`는 레이아웃 참조로 벡터 재생성(y축 truncation 캡션 명시).
- **생성 필요 asset**: V1(Title faint teaser, 옵션), Slide 4 adjacency heatmap 3패널(실매치 데이터), V5(Slide 9 high-ground/advantage concept).
- **수치 단일 진실 출처**: 메인 결과 = `docs/PRESENTATION_DATA_2026-06-21.md`; future/advantage = `docs/FUTURE_DESIGN_3D_RELATIONAL_2026-06-21.md`. 그 외 출처 금지.
- 언어: 전 슬라이드 텍스트 영어·격식체. 구어체 금지.
