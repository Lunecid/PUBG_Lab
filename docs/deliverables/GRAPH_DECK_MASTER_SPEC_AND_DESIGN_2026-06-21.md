# PUBG Graph-Survival Deck — Master Spec, Design Philosophy & Decision Log

**Author:** Baek Seongeun (백성은) · Pusan National University · *Graph Data Analysis*
**Date:** 2026-06-21
**Primary deliverable:** `docs/deliverables/graph_deck.html` → `graph_deck.pdf` (16:9, 15 slides + title)
**Companion scripts:** `graph_deck_script.tex` (EN), `graph_deck_script_ko.tex` (KO)

This document consolidates **(1)** the research narrative, **(2)** the presentation design philosophy, **(3)** the code-grounded model & data specification, **(4)** the verified result numbers, and **(5)** the full conversation/decision log explaining *why the deck looks the way it does*. It is the single source of truth for the talk; if a number or claim conflicts with an older doc, this file wins.

---

## 0. One-paragraph thesis

> Connecting PUBG teams by how they *interact* — who is near whom, who shoots whom, who is squeezed together by the zone — produces a graph whose **topology carries essentially no usable signal**: an edge-shuffle null matches or beats the real wiring. Ordinary interaction edges behave as **smoothing**, not tactical information. What decides whether topology helps is the **relation type**, not the GNN architecture. This motivates a redesign toward **contact-gated relative-advantage** edges, which show a small but consistent positive signal over the null. The honest negative result *is* the contribution.

The deck **front-loads** this finding rather than hiding it behind a positive-result framing.

---

## 1. Research narrative (the spine of the deck)

1. **Worldview** — A match is not isolated players; it is *teams* moving and fighting inside a *contracting* safe zone. Survival is **relational** and **spatial**.
2. **Objective** — Read each match as a **graph of team relations evolving inside a shrinking habitat**, and predict (a) which team is eliminated next phase and (b) each team's final placement — *and test whether the relational structure itself carries signal*.
3. **Differentiation** — Prior PUBG work models **one player at a time** (aggressiveness scores, play-style clusters). This study takes the **team-relational** view that prior work deliberately set aside.
4. **The test** — Compare observed topology against a **null** that preserves everything except the wiring. If topology mattered, observed should beat the null.
5. **The result** — It does not. The null is equal-or-better; ordinary edges are smoothing.
6. **The redesign** — Edges should carry **relative advantage** (elevation, numbers, health, zone position), gated on contact. Preliminary, small, but directionally right.

This is a **negative-result-as-contribution** narrative. The talk's credibility rests on the rigor of the null-control, so that slide is the climax.

---

## 2. Design philosophy

The deck is a **minimal, researcher-style presentation PPT** — *not* a paper rendered as slides, and *not* a web page. Principles, in priority order:

### 2.1 Governing principles (from the user's direction)
- **Content first.** Every slide earns its place; design serves the message.
- **One message per slide.** Title states the section; the **assertion** (large headline) states the slide's single claim; one **primary evidence object** supports it.
- **Assertion–Evidence headlines.** The big headline is a full *sentence claim* (e.g. "Two time resolutions are fused… then read out into elimination and rank."), not a topic label. Evidence sits below.
- **Tufte restraint.** Small effects shown small; honest truncated axes are disclosed in the caption; no chartjunk.
- **Consistency.** Shared grid, type scale, color roles, and footer across all slides.
- **Whitespace & alignment.** Generous margins; left-aligned text blocks; aligned figure baselines. Conclusion slides must not be mostly empty — the conclusion is *emphasized*, not sparse.
- **Limited color with assigned roles** (see §2.3).
- **Diagrams over text.** Where a relationship can be drawn, draw it. But — **a diagram must represent the *actual designed model*, not a decorative GNN picture** (a hard rule, see §2.4).
- **Minimal animation.** Static, screenshot/print-friendly.

### 2.2 Voice & typography rules (accumulated from feedback)
- **Single author → no "we".** Use "this study", "the model", or passive voice throughout.
- **No semicolons / no code-style fragments in prose.** Captions and readouts are **full presentation sentences**. (Mono is reserved for genuine symbols/tensors, not prose.)
- **Body text must be large** relative to the headline — readable from the back of a room. Early drafts were rejected as "too web-like / body too small vs title".
- **Math is rendered like a paper.** KaTeX/LaTeX, `$…$` / `$$…$$`, never raw ASCII math.
- **Expand every acronym so a non-expert understands it.** This is a *hard* rule added late: even the designer should not have to guess what a letter means. Concretely — **R-GCN is written "Relational-GCN"** and explained ("a separate weight per relation type"); **TCN = temporal convolution**; **FiLM = the global phase context gates every team**. The figure carries the structure; the caption carries the expansions.

### 2.3 Color tokens (roles, not decoration)
| Token | Hex | Role |
|---|---|---|
| Accent blue | `#1D4ED8` | structure, section tags, stage labels, "observed" |
| Ink | `#1f2430` | primary text |
| Grey | `#6b7280` | secondary text, captions, hairlines |
| Hazard / combat red | `#c4322f` / `#d62728` | elimination head, directed combat edges |
| Rank / proximity amber | `#d97706` | rank head, survivor-narrowing band, embeddings |
| Hairline | `#d8dde4` | rules, card borders |

### 2.4 Typography stack
- **Headlines / assertions:** Newsreader (serif) — quiet, academic, paper-like.
- **Body / labels:** IBM Plex Sans.
- **Symbols / tensors:** IBM Plex Mono.
- This pairing (serif display + sans body) is intentional and consistent; it is *not* the generic Inter/Roboto default.

### 2.5 The "diagram must be the real model" rule
The architecture figure went through several rejections. The lesson, stated plainly:

> A figure that merely *looks* like a GNN diagram is wrong. The figure must encode **exactly** what was built — the two input resolutions, the fusion, the stacked spatio-temporal blocks, the two heads — with real dimensions, and with the **swappable** component clearly marked as swappable (not pinning any single encoder). Abstract "bars on nodes" decoration is not acceptable.

---

## 3. Deck structure — 15 slides (title excluded from count)

Mandatory 6-section academic frame: **연구 배경·목적 / 선행연구 / 사용 데이터 / 분석 방법 / 분석 결과 / 한계·기여**.
Slide title = the section; assertion = the slide's claim.

| # | Section | Slide | Primary evidence | Assertion (claim) |
|---|---|---|---|---|
| — | — | **Title** | name · university · course | "From Proximity to Tactical Advantage" |
| 02 | Background | **Worldview** | real Erangel match (P4), 3-channel edges, shrinking zone | survival is relational & spatial |
| 03 | Background | **Objective + perspective** | individual-player vs team-relational contrast | a relational, team-level view |
| 04 | Related work | **Sagong et al. 2025** | their pipeline vs ours | they excluded teams; this study brings them back |
| 05 | Data | **Data + prediction task (merged)** | data strip + labeling math | every alive team–phase carries a leakage-safe label |
| 06 | Methods | **Dynamic heterogeneous graph** | dynamic-axis × heterogeneous-axis decomposition | dynamic (rebuilt per phase + 10 s snapshots) × heterogeneous (3 relation channels) |
| 07 | Methods | **Three relations** | proximity / combat / co-pressure formulas (KaTeX) | each relation is built differently |
| 08 | Methods | **Architecture** | the real multi-resolution model diagram | two resolutions fused → stacked graph+temporal layers → two heads |
| 09 | Results | **Elimination (fig5)** | per-phase hazard AUC, all encoders | encoders cluster within ~0.02 AUC — architecture is not the lever |
| 10 | Results | **Ranking (fig6)** | per-phase rank ρ, all encoders | the rise is **survivor narrowing**, present with no graph |
| 11 | Results | **Performance summary** | graph-encoder lift numbers | a lift is not topology signal |
| 12 | Results | **Null-control (central)** | observed vs edge-shuffle | observed adds no usable signal beyond graph averaging |
| 13 | Results | **Relation redesign (preliminary)** | advantage-edge concept + small gain | relation *type* is the lever |
| 14 | Limits & contrib | **Limitations & contributions** | honest list | smoothing finding, null-control protocol, relation hypothesis, MR pipeline |
| 15 | Conclusion | **Conclusion** | the takeaway, emphasized | relation type — not GNN architecture — decides whether topology helps |

**Compression history:** 17 → 15 by dropping the standalone "Evolution" slide, merging Nodes into Architecture, and merging Data + Labeling. The per-phase result figures (fig5, fig6) were **restored** before the summary slide after they were briefly lost in compression — they are non-negotiable.

---

## 4. Model & data specification (code-grounded)

Source of truth: `docs/MODEL_SPEC_CODE_GROUNDED_2026-06-21.md` + verified-corrections memory. **Use these numbers, not older deck numbers** (older numbers mixed all-mode/unfiltered data and were wrong).

### 4.1 Data
- **15,213 squad matches** (`split.json n_matches`; *not* 16,932, which is the full-DB squad total before filtering).
- **Match-level 80/10/10** train/val/test (val/test ≈ 1,521 matches each). Split **by match** to prevent leakage.
- **≈ 0.86 M labeled team–phase cells** (at-risk, cut-phases 1–7). *Not* 1.37 M.
- **9 maps**, Erangel/Baltic dominant (≈ 39.2%). Steam / PC. Window 2026-03-26 → 2026-06-02.
- Squad distinct teams = 438,014 (the all-mode 760,592 figure is wrong for this dataset).
- Hazard positive rate rises across phases: **P1 ≈ 25.5% → P7 ≈ 42.7%** (test).

### 4.2 N / T semantics (a common question)
- **N = 128** = `PHASE_N_MAX`, a **padding** cushion. Real squad lobby ≈ 27 teams (median), max 60.
- **T = 8** phase slots; **7 scored cut-phases (P1–P7)**; hazard horizon = T−1.
- "Causal" here means **leakage-safe / phase-causal**, *not* causal inference.

### 4.3 Graph construction — three relations (phase tier)
Adjacency `A_phase[T=8, E=3, N=128, N=128]`, row-normalized **per relation**.

| Relation | Weight | Symmetry | Gate |
|---|---|---|---|
| **Proximity** | $w_{ij}=\exp(-d_{ij}/200)$ | symmetric | none |
| **Combat** | $w_{ij}=\ln(1+\text{damage}+100\,\text{kills}+50\,\text{groggies})$ | **directed** (attacker → victim) | none |
| **Co-pressure** | $w_{ij}=\text{ratio}\cdot\exp(-d_{ij}/200)$ | symmetric | **zone-gated**: both teams outside the safe area or ≤ 150 m from its edge, **and** pair ≤ 200 m apart |

Snapshot tier uses only **2 relations** (proximity, combat); co-pressure is phase-level.

### 4.4 Multi-resolution tensors
- **Phase node features:** `X_node[T=8, N=128, 116]` (116-d per team per phase).
- **Global context:** `G[T=8, 25]` (per-phase match state).
- **Snapshot tier (every 10 s):** `X_s[T_s, N, 13]` + `A_snap[T_s, E=2, N, N]`; encoded (temporal + graph conv) and **mean-pooled to the phase**, producing **48-d** per team.

### 4.5 Architecture (what slide 8 must show)
1. **Two input resolutions** in: snapshot tier (10 s, 13 feat, 2 relations) and phase tier (116 feat, 3 relations, + 25-d global context).
2. **Fuse:** concatenate `[X_node 116 ‖ snapshot 48] = 164-d`, then **FiLM** — the global phase context gates each team's features.
3. **Spatio-temporal encoder × 2:** each block = a **causal gated TCN** (kernel = 3, the *fixed* temporal encoder) **+ a graph encoder** (the *only* swapped component). Hidden width **64**. Causal in time, message-passing in space.
4. **Output:** a **64-d embedding per team**, `H ∈ ℝ^{N×64}`.
5. **Two heads:** **hazard head (MLP, 64→32→1)** → next-phase elimination; **rank head (decoder)** → final placement.

**Swappable encoders compared (the only thing that changes across variants):**
`none · GCN · GraphSAGE · GAT · Relational-GCN`.
- **GCN / GraphSAGE / GAT** merge the three relations into **one combined adjacency**.
- **Relational-GCN** ("R" = *Relational*) keeps a **separate weight per relation type** — this is why it appears in the *heterogeneous-graph* discussion. It is **one of five compared encoders, not the headline model.** No projection embeddings (`nn.Embedding`); features enter via Linear projections.

### 4.6 Labeling (leakage-safe) — slide 5 math
- **At-risk mask** $m_{p,i}=1$ iff team $i$ is alive entering phase $p$ (scored only while alive).
- **Hazard target** $y^{\text{haz}}_{p,i}=1$ iff team $i$ is eliminated in phase $p{+}1$.
- **Rank target** $y^{\text{rank}}_{i}=\text{final\_rank}_i$, broadcast over that team's at-risk phases.
- An **exact-name blocklist** keeps any outcome field out of the inputs.

### 4.7 Metrics
- **Elimination:** hazard **AUC** (per phase).
- **Ranking:** **Spearman ρ** (per phase) vs final placement.

---

## 5. Results (verified)

### 5.1 Elimination (fig5 — `fig5_hazard_by_phase.png`)
All encoders track within **≈ 0.02 AUC** of one another, just above the no-graph baseline. Per-phase hazard AUC (P1→P7):

| Encoder | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---|---|---|---|---|---|---|
| No graph | .881 | .817 | .779 | .799 | .790 | .782 | .741 |
| GCN | .890 | .831 | .795 | .810 | .798 | .791 | .752 |
| GraphSAGE | .887 | .827 | .794 | .808 | .798 | .786 | .748 |
| GAT | .893 | .835 | .800 | .815 | .805 | .792 | .741 |
| Relational-GCN | .889 | .829 | .795 | .813 | .800 | .789 | .741 |

**Reading:** the tight cluster already says *architecture is not the lever*. Y-axis is truncated (full span 0.725–0.92) and that truncation is **disclosed** in the caption.

### 5.2 Ranking (fig6 — `fig6_rank_by_phase.png`)
Rank ρ clearly rises P3 → P6 — **but the same rise appears with no graph at all.** GAT is the best *ranker*, but the *shape* of the curve is not a graph effect (see §5.3).

### 5.3 The rank "lock-in" is a **survivor-selection effect** (not a model gain)
- The P3 → P6 rank-ρ rise is **+0.087 for the no-graph baseline**, *larger* than GAT's +0.074.
- The at-risk set shrinks **~7×** (P1 ≈ 23,432 → P6 ≈ 3,382 cells). Fewer survivors are simply easier to order.
- The graph lifts the whole curve a little; it does **not** create the rise. Labeled **"survivor narrowing (P3-6)"** on the figure (previously, and wrongly, "rank lock-in").

---

## 6. Null-control — the central result (slide 12)

This is the climax; its rigor is what the talk's credibility rests on.

- **Headline numbers** (graph-null experiment, `feature_preset=core_habitat`): **no_graph 0.834 · observed 0.845 · edge_shuffle 0.847.** Shuffle ≥ observed ⇒ topology has no signal.
- **Edge-shuffle null (definition):** per **match-phase and per channel**, preserve the **weight multiset** (density + distribution) over alive-team blocks, but **randomly reassign the (src, dst) endpoints** among alive teams, then row-renormalize. This **destroys the actual wiring and degree** while preserving everything else.
- **Node-shuffle null (definition):** preserve graph **structure (degree)** but break the **team-identity ↔ feature** correspondence.
- **Significance:** **match-level cluster bootstrap, 1000 paired reps** (not a 3-seed comparison). Observed is **significantly worse** in phases **1–3**, non-significant in 4–7, and **never significantly better** in any phase.
- **The null-control is CLEAN ✅** — the strongest reviewer objection, defended: the experiment uses the **`core_habitat`** feature preset, which **excludes graph-derived `net_*` features** (degree, reciprocity, network-role). So the no-graph baseline contains **no topology-derived features**; the comparison is not contaminated.
  - *Caveat:* the full feature preset *does* contain `net_*` features; if any slide lists the full 116-d preset, note that the null experiment used the reduced preset.

**Conclusion stated honestly:** the observed topology adds **no usable signal beyond plain graph averaging (smoothing)**.

---

## 7. Relation redesign — advantage edges (slide 13, preliminary)

**Why ordinary edges fail:** proximity / combat / co-pressure encode that an interaction *exists* and *how intense* it is — **not who holds the advantage**. A symmetric distance kernel just averages neighboring states.

**The redesign:** edges that carry **relative advantage** — differences in **elevation, numbers, health, zone position** — **gated on contact**. This relation shows a **small but consistent positive signal over the null controls** (≈ **+0.003 AUC** in the advantage experiment).

**Status:** explicitly **preliminary / hypothesis-generating**, *not* a main claim. The effect is small and early but directionally right. (Future-design detail: `docs/FUTURE_DESIGN_3D_RELATIONAL_2026-06-21.md`.)

---

## 8. Limitations & contributions (slide 14)

**Limitations (stated honestly):**
- Observed topology is essentially **smoothing**.
- Map figures illustrate a **single example match** (the winner marker is for the picture only, never a model input).
- The advantage edge is **preliminary**.

**Contributions:**
1. A **phase-causal, team-level graph formulation** of PUBG survival.
2. A **null-control protocol** (edge/node shuffle + cluster bootstrap) that **exposes the smoothing**.
3. The **relation-design hypothesis** (advantage edges).
4. A **multi-resolution telemetry pipeline** linking snapshots, phases, hazard, and rank.

---

## 9. Related work (slide 4)

**Sagong, Woo, Kim (2025)** — *Entertainment Computing* 55:101040. Per-phase **FightScore** + **ActiveScore**, cluster players into **4 types** (Hunter/Camper etc.), predict with **tree models**, **solo only** (teams deliberately excluded), ≈ 87.9% accuracy. This study **re-introduces the team-relational dimension they set aside** and adds the null-control rigor.

**Zhou et al. (2020)** — GNN survey (`arxiv 1812.08434`, local `~/Downloads/1812.08434v6 (1).pdf`). **Fig. 2** "design pipeline" (Input → GNN layers → Output → Loss/Task, with a zoom-in callout of layer internals) is the visual reference for the architecture slide style.

---

## 10. Conversation & decision log (why the deck is what it is)

Chronological summary of user feedback → response. Each item is a binding precedent for future edits.

1. **Real-map money shot.** Use a real Erangel match (chosen match `f6513861…`, hero phase **P4**) with our 3-channel edges and the contracting zone. The **winner must be inside the current safe zone** (fixed by recovering the *current* circle via least-squares from `dist_blue_boundary_m` at a winner-inside snapshot; earlier the winner sat outside because the data referenced the *next* circle).
2. **Graph too small / hairball** on the map → simplified the on-map graph.
3. **Differentiation = a different perspective** (team-relational vs individual-player), made explicit on slides 3–4.
4. **Readability** — body text was too small vs the title and "too web-like"; rebuilt with presentation-scale type.
5. **Emphasis & less paper-like** — add highlighting; the two-time-resolutions slide → **LaTeX math** + an **architecture-style** layout.
6. **Title page** must carry **BAEK SEONGEUN**, **Pusan National University**, and the course **Graph Data Analysis**.
7. **Single author → remove every "we"** (use "this study"/passive).
8. **Math rendered like a paper** (KaTeX), and the labeling slide expressed as **math with plain-language meanings**, not an image.
9. **Decompose "dynamic heterogeneous graph"** into its two axes (dynamic *and* heterogeneous) on slide 6.
10. **Compress to ≤ 15 slides**, **keep the flow**, **no "Core Question" slide**. (17 → 15.)
11. **Merge Data + Labeling**; remove unclear stats (the "9 maps / 25→43%" mini-panel).
12. **No semicolons / no code-style fragments** — captions become real presentation sentences. Fix the **slide-2 gray overlap** with the caption (reduced map height, removed shadow).
13. **Architecture must not pin R-GCN** (it is one of five swappable encoders) and **must not repeat R-GCN across slides**; make it **pretty + readable**; reference the **GNN survey Fig. 2** style.
14. **(latest)** Architecture had **text overflow**, looked **abstract/decorative**, and **did not represent the actual designed model**; and **"R" in R-GCN was unexplained**. → Redrawn as the **real multi-resolution model** (two tiers → fuse → ST encoder ×2 → two heads) with **no overflow**, the **swappable encoder marked as swappable**, and **R-GCN written "Relational-GCN"** with every acronym expanded in the caption.
15. **Speaker scripts** in **English** and **Korean** (`.tex`), conversational 존댓말 for KO, ~9–11 min, with a Q&A-prep footnote.
16. **23-point adversarial review** applied: `causal → phase-causal`; corrected data numbers (15,213 / 9 maps / N=128 padding / 0.86 M cells); N/T semantics; node-feature contamination addressed (core_habitat); precise edge-shuffle null definition; rank = survivor-selection; softened overclaims.

---

## 11. Deliverables & tooling

### 11.1 Files
| File | Purpose |
|---|---|
| `docs/deliverables/graph_deck.html` | **primary** self-contained deck (HTML/CSS, KaTeX CDN) |
| `docs/deliverables/graph_deck.pdf` | rendered 15-slide PDF (16:9) |
| `docs/deliverables/graph_deck_script.tex` | English speaker script |
| `docs/deliverables/graph_deck_script_ko.tex` | Korean speaker script (XeLaTeX + kotex) |
| `docs/presentation_figures/fig5_hazard_by_phase.png` | per-phase elimination AUC |
| `docs/presentation_figures/fig6_rank_by_phase.png` | per-phase rank ρ |
| `scripts/make_realmap_figure.py` | real-map hero figure generator |
| `scripts/make_phase_summary_figure.py` | fig5 / fig6 generator |
| `docs/MODEL_SPEC_CODE_GROUNDED_2026-06-21.md` | code-grounded model spec |
| `docs/deliverables/design_spec_graph_deck_academic_revised.md` | deck design spec (tokens, layout) |

### 11.2 PDF render recipe
```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=graph_deck.pdf --virtual-time-budget=14000 graph_deck.html
```
- `--virtual-time-budget=14000` is required so **KaTeX finishes rendering** before print.
- **Avoid the gray-page bug** with print CSS: `@media print{html,body{background:#fff}.slide{margin:0;box-shadow:none}.slide:last-child{page-break-after:auto}}`. (A `.slide{margin}` was overflowing the 720 px page and emitting a second dark page per slide.)
- `@page` size 1280 × 720 px → 960 × 540 pt (16:9).
- KaTeX is loaded from CDN with auto-render (`$…$`, `$$…$$`). **KaTeX does not process SVG `<text>`** — use plain text inside SVG, never `$…$`.

### 11.3 Korean speaker script
Compile with **XeLaTeX** (Overleaf: Menu → Compiler → XeLaTeX); requires the `kotex` package.

---

## 12. Open follow-ups (optional)
- Further per-slide design polish on request.
- The advantage-edge experiment is preliminary; a fuller result table would strengthen slide 13.
- If any slide ever lists the full 116-d feature preset, add the **core_habitat caveat** for the null experiment (§6).

---

*This document supersedes earlier scattered notes for the purpose of the talk. When a fact conflicts, prefer the numbers in §4–§6 (code/DB-verified, 2026-06-21).*
