# Revised Design Spec — Graph PPT deck with academic presentation philosophy
*개정 디자인 설계서. 원본 `design_spec_graph_deck.md`의 수치·내러티브를 유지하되, 학술 발표 설계 철학과 실제 프로젝트 검증 결과를 반영한다. 슬라이드에 표시되는 본문은 영어·격식체를 기본으로 한다.*

---

## 0. Revision Goal

이 덱은 "멋있게 보이는 발표"가 아니라 **학술적 주장을 증거로 방어하는 발표**여야 한다. 따라서 모든 슬라이드는 다음 문장으로 평가한다.

> Does this slide make one defensible claim, show the evidence needed to judge it, and remove everything that competes with that judgment?

이번 개정의 핵심은 세 가지다.

1. **Assertion-Evidence 구조 적용**: 제목은 주제가 아니라 한 문장 주장이다. 본문은 bullet list가 아니라 시각 증거다.
2. **인지부하 최소화**: 한 슬라이드 한 판단. 철학·방법·결과·해석을 섞지 않는다.
3. **정직한 데이터 디자인**: 작은 효과는 작게 보이게 하고, null result는 회피하지 않는다.

---

## 1. Reference Platforms, Papers, and Imported Philosophy

### 1.1 Assertion-Evidence Approach — Michael Alley / Penn State

**Source**
- Platform: Assertion-Evidence Approach, Penn State / Michael Alley
- URLs:
  - https://www.assertion-evidence.com/
  - https://www.assertion-evidence.com/research-papers.html
  - https://www.assertion-evidence.org/tutorial.html

**Principle imported into this deck**
- Replace topic headlines with sentence assertions.
- Replace bullet lists with visual evidence.
- Put secondary details in speaker notes or small captions, not in the primary visual field.

**Deck rule**
- Every non-title slide has an **assertion headline**.
- Each slide has one **primary evidence object**: figure, diagram, table, matrix, or paired comparison.
- Bullets are allowed only in side panels with at most three short lines; they must explain the evidence, not duplicate it.

### 1.2 Cognitive Theory of Multimedia Learning — Richard Mayer

**Source**
- Mayer, R. E. (2008). *Applying the science of learning: Evidence-based principles for the design of multimedia instruction*. American Psychologist, 63(8), 760-769.
- CTML principles used here: coherence, signaling, spatial contiguity, segmentation, redundancy control.

**Principle imported into this deck**
- Remove decorative information that does not serve the current learning task.
- Signal the exact part of a figure the audience should inspect.
- Keep labels adjacent to the visual element they explain.
- Segment the argument into digestible scenes.

**Deck rule**
- No slide should require the audience to search across the canvas for legend, annotation, and conclusion.
- A visual annotation is better than a paragraph beside the figure.
- A figure caption must answer "what am I looking at?" in one sentence.

### 1.3 Tufte / Graphical Integrity / Data-Ink Discipline

**Source**
- Edward Tufte, *The Visual Display of Quantitative Information* and related books.
- URL: https://www.edwardtufte.com/books/

**Principle imported into this deck**
- Show the data; reduce non-data ink; do not make small effects look large.
- Avoid chartjunk, decorative 3D, heavy gridlines, and misleading zoom without explicit disclosure.
- Prefer small multiples and direct labeling when comparisons matter.

**Deck rule**
- Small AUC/Spearman differences are shown as dot strips, bands, or paired controls, not as dramatic bars.
- Any truncated axis must be explicitly disclosed in the caption.
- The graph-null result must be visually honest: `edge_shuffle >= observed` is not hidden.

### 1.4 Cleveland & McGill / Graphical Perception

**Source**
- Cleveland, W. S., & McGill, R. (1984). *Graphical perception: Theory, experimentation, and application to the development of graphical methods*. Journal of the American Statistical Association.

**Principle imported into this deck**
- Use encodings that support the audience's comparison task.
- Position on a common scale is easier to compare than angle, area, volume, or decorative color.

**Deck rule**
- Model comparisons use common-axis dot strips or lines.
- Phase trends use aligned time axes.
- Heatmaps are used only for matrix structure, not for precise magnitude comparison.

### 1.5 HBR / Declarative Visualization

**Source**
- Scott Berinato, "Visualizations That Really Work", Harvard Business Review.
- URL: https://hbr.org/2016/06/visualizations-that-really-work

**Principle imported into this deck**
- Before visualizing, decide whether the goal is exploration or communication.
- This deck is declarative: it communicates tested claims, not an exploratory dashboard.

**Deck rule**
- Every result figure must have a single declared readout.
- The slide should not invite open-ended exploration unless the slide is explicitly a modeling-definition slide.

### 1.6 Duarte / Audience-Centered Story Structure

**Source**
- Duarte, *Resonate* / Duarte communication resources.
- URL: https://www.duarte.com/resources/books/resonate/

**Principle imported into this deck**
- The audience is the hero; the presenter is the guide.
- A strong talk moves between the current state, tension, evidence, and a better conceptual frame.

**Deck rule**
- The deck arc is not "we built a model"; it is:
  1. PUBG survival is relational and spatially constrained.
  2. We built a causal multi-resolution representation.
  3. The model works, but the observed interaction topology is an honest negative.
  4. The future direction is relation type: contact-gated relative advantage.

---

## 2. Global Presentation Philosophy

### 2.1 Core Academic Stance

The deck should feel like a rigorous conference talk, not a product pitch.

- **Primary mode**: proof by carefully staged evidence.
- **Tone**: precise, restrained, explicit about limits.
- **Visual character**: quiet academic, white background, thin rules, high legibility.
- **Main virtue**: credibility.

### 2.2 Slide Grammar

Every slide uses this structure unless explicitly exempted:

```text
ASSERTION HEADLINE
One sentence that states the slide's claim.

PRIMARY EVIDENCE AREA
One figure, diagram, timeline, matrix, or paired comparison.

GUIDED READOUT
1-3 short annotations that tell the audience where to look.

CAPTION / LIMIT
One small line that states source, scope, or caveat.
```

### 2.3 What Moves to Speaker Notes

The slide is not the paper. Put these in speaker notes, not on the slide:

- exact training command paths,
- all hyperparameters except those necessary for interpreting the figure,
- secondary ablation values,
- detailed definitions already shown earlier,
- defensive caveats that do not change the current slide's judgment.

### 2.4 Anti-Patterns Banned from This Deck

- Topic-only headings such as `Results`, `Method`, `Graph`.
- Bullet-first slides.
- Decorative network diagrams that do not encode a real distinction.
- Large text blocks beside already labeled figures.
- Huge numeric callouts for tiny effect sizes.
- Any statement implying observed topology is informative when null controls reject it.
- Any framing of "2D to 3D" as the contribution.

---

## 3. Technical Corrections from Project Verification

These corrections must override the original design spec where ambiguous.

1. **Phase graph is 3-channel; snapshot graph is 2-channel.**
   - Phase-level `A[p]`: `proximity`, `combat`, `copressure`.
   - Snapshot-level `A_s[t]`: `proximity`, `combat`.
   - Do not imply the entire MR model uses three relation channels at both tiers.

2. **R-GCN consumes separate relation matrices; homogeneous encoders combine active channels.**
   - R-GCN: per-relation convolution.
   - GCN / GraphSAGE / GAT: combined active adjacency.
   - Therefore "one relation matrix per channel" is R-GCN-specific.

3. **Slide 3 feature groups are presentation groups, not the exhaustive manifest.**
   - Actual node inputs include official/team raw and loot/readiness features too.
   - Phrase Slide 3 as "major feature groups" or "presentation-level grouping".

4. **Global context is separated in V3 model input.**
   - Safe radius, alive-team count, shrink rate, and similar global variables belong to `G[p]`, not `X_node[p,i]`.
   - The legacy tensor path used broadcast context internally, but V3 factorizes it before model input.

---

## 4. Revised Shared Design System

### 4.1 Typography

- Assertion headline: 26-30 pt, sentence case, left aligned.
- Evidence labels: 11-13 pt.
- Side note: 13-15 pt.
- Captions and caveats: 10-12 pt.
- No negative letter spacing. No viewport-scaled type.

### 4.2 Layout

- Use a fixed evidence grid:
  - `60/40` for result slides: figure left, readout right.
  - `50/50` for conceptual contrast.
  - `full-width evidence` for timelines or pipelines.
- Keep the headline region clear.
- Use direct labels instead of detached legends whenever possible.
- Captions sit immediately below the relevant figure.

### 4.3 Color

Keep the original palette, but use it as semantic signal rather than decoration.

- alive/survive: `#2ca02c`
- eliminated: `#d62728`
- model input / observed window: `#dbeafe` fill, `#1d4ed8` accent
- ranking / award / lock-in: `#d97706` accent, `#fef3c7` fill
- neutral text: `#374151`
- muted text: `#6b7280`
- gridline: `#cccccc`

Relation-channel tokens:

- phase proximity: `#1d4ed8`
- phase combat: `#d62728`
- phase co-pressure: `#d97706`

Encoder tokens:

- No graph: `#7f7f7f`, dashed
- GCN: `#1f77b4`
- GraphSAGE: `#ff7f0e`
- GAT: `#d62728`
- R-GCN: `#2ca02c`

### 4.4 Evidence Annotation Rules

- One slide gets no more than three callouts.
- Put callouts on or near the relevant visual feature.
- Use arrows sparingly; prefer bracket, band, or direct text.
- For small effects, annotate exact value and avoid oversized typography.
- For null results, visually emphasize the comparison that falsifies the claim.

---

## 5. Revised Narrative Architecture

### Act I — Frame the Problem

The audience must first accept that PUBG survival is both relational and constrained by a contracting spatial domain.

- Slide 1: Title
- Slide 2: Worldview

### Act II — Define the Representation

The audience should understand what a node, edge, label, and metric mean before seeing results.

- Slide 3: Node design
- Slide 4: Edge design
- Slide 5: Encoder spectrum
- Slide A: Labeling
- Slide B: Metrics

### Act III — Show the Evidence

The audience sees that MR works, graph architecture clusters, and rank has a different behavior than elimination.

- Slide 6: Elimination results
- Slide 7: Ranking results

### Act IV — Interpret Honestly

The talk earns credibility by saying what did not work.

- Slide 8: Smoothing, not topology

### Act V — Convert the Negative Result into a Research Direction

The future work is framed as a relation-design hypothesis, not as a hidden main result.

- Slide 9: Contact-gated relative advantage
- Slide 10: Conclusion

---

## 6. Slide-by-Slide Revised Design

## Slide 1 — Title

**Assertion function**: establish the scope and tone; no result claims.

**Headline / title**
`From Proximity to Tactical Advantage:`
`Rethinking Graph Relations for PUBG Survival Prediction`

**Subtitle**
`A causal multi-resolution spatio-temporal representation for elimination and final-placement prediction`

**Design**
- Minimal academic title slide.
- Optional faint background teaser remains acceptable, but it must be visually subordinate.
- Do not show performance numbers.
- Do not show the future advantage result.

**Philosophy applied**
- Duarte: audience enters through the problem frame.
- Tufte: no ornamental title art competing with the thesis.

---

## Slide 2 — Worldview

**Assertion headline**
`PUBG survival unfolds as team relations inside a contracting habitat.`

**Primary evidence**
Two-lens diagram:

- left: dynamic team-relation graph,
- right: shrinking play zone with survivor count trend,
- center connector: `within`.

**Readout**
- `Teams interact, separate, and re-form local neighborhoods.`
- `The safe zone contracts while the risk set shrinks.`
- `The model must preserve both processes.`

**Caption**
`Survivor counts are rounded phase-level summaries from the squad ST-DB profile.`

**Design changes from original**
- Keep this slide conceptual.
- Do not name relation channels yet.
- Do not include AUC, Spearman, or null controls.

**Philosophy applied**
- CTML segmentation: worldview before model details.
- Assertion-Evidence: headline states the claim; diagram is the evidence.

---

## Slide 3 — Node Design

**Assertion headline**
`Each surviving team is represented as a causal state vector, while global match context is modeled separately.`

**Primary evidence**
One highlighted team node expands into major feature groups.

**Feature groups shown**
- `Trajectory` — path length, displacement, speed, sinuosity, safe-zone alignment
- `Zone exposure` — boundary distance, outside-safe ratio, local enemy density
- `Team state` — alive players, health, spread, vehicle ratio
- `Combat` — damage, kills, groggies, distinct opponents
- `Network role` — combat/proximity degree and strength, reciprocity
- `Local habitat pressure` — zone deficit, local enemy pressure, safe-entry pressure

**Small note**
`Displayed groups are presentation-level summaries. The full manifest also includes official/team raw and loot/readiness features.`

**Global context note**
`Phase context such as safe radius, alive-team count, and shrink rate enters as a separate global vector G[p], not as team-attributed node state.`

**Design changes from original**
- Replace "full spatial-combat STATE" with "major causal team-state groups".
- Keep the node-level/global separation explicit.

**Philosophy applied**
- Tufte: don't overclaim the graphic as exhaustive if it is a summary.
- CTML coherence: one node concept, not full manifest dump.

---

## Slide 4 — Edge Design

**Assertion headline**
`The phase graph separates three interaction relations; the snapshot tier captures high-frequency proximity and combat.`

**Primary evidence**
Two-level graph contract diagram.

**Top band: MR graph contract**

```text
Snapshot tier A_s[t]        Phase tier A[p]
10 s resolution             phase resolution
proximity + combat          proximity + combat + co-pressure
```

**Main visual**
Phase-level multiplex graph with three layers:

- `Proximity` — symmetric distance relation, `w_ij = exp(-d_ij / 200)`
- `Combat` — directed attacker-to-victim relation, `w_i->j = ln(1 + damage + 100*kills + 50*groggies)`
- `Co-pressure` — zone-pressure-gated relation, active for close teams under shared pressure

**Tensor callout**
`A_phase[T, E=3, N, N]`, row-normalized per relation.

**R-GCN-specific note**
`R-GCN consumes one matrix per relation. GCN, GraphSAGE and GAT use a combined active adjacency.`

**Heatmap asset**
- Three real-match phase-level heatmaps.
- Caption: `Phase-level adjacency, row-normalized; selected for structural readability, not for performance claims.`

**Design changes from original**
- Corrects the tier ambiguity.
- Keeps three-channel claim scoped to phase graph.

**Philosophy applied**
- CTML spatial contiguity: formula next to relation visual.
- Tufte graphical integrity: heatmap is labeled as structural evidence, not a result.

---

## Slide 5 — Encoder Spectrum

**Assertion headline**
`Changing the graph encoder barely changes hazard performance.`

**Primary evidence**
One-dimensional dot strip of the 8-model main matrix.

**Exact values**
- No graph (TCN): `0.8358`
- GCN: `0.8461`
- GraphSAGE: `0.8438`
- GAT: `0.8496`
- R-GCN: `0.8460`
- R-GCN + FiLM: `0.8465`
- R-GCN + attention-rank decoder: `0.8449`
- FARNet: `0.8454`

**Readout**
- `Seven graph encoders occupy a 0.0058 AUC band.`
- `The no-graph baseline trails the best model by about 0.014 AUC.`
- `Architecture is not the lever; relation representation is the hypothesis.`

**Future-work row**
Show `Edge-conditioned relative advantage` as dashed, muted, and explicitly outside the main matrix. No future-work numbers here.

**Design**
- No bar chart.
- Dot strip range: `0.830-0.852`.
- Label only `No graph`, `graph encoder band`, and `GAT`.

**Philosophy applied**
- Cleveland/McGill: common scale comparison.
- Tufte: small differences remain visually small.

---

## Slide A — Labeling

**Assertion headline**
`Each prediction is made from a causal cut and scored only while the team is still at risk.`

**Primary evidence**
Cut-phase timeline with two team rows.

**Readout**
- `Input: observed through cut phase p.`
- `Hazard target: eliminated during phase p+1.`
- `At-risk mask: teams already eliminated are not predicted.`

**Design**
- Recreate `fig0_labeling.png` as vector.
- Use blue for observed window, red/green for hazard label.
- Keep explanatory text adjacent to the timeline.

**Philosophy applied**
- Assertion-Evidence: timeline proves leakage-safe labeling.
- CTML signaling: blue observed-window box guides attention.

---

## Slide B — Metrics

**Assertion headline**
`Elimination is evaluated as discrimination; final placement is evaluated as ordering.`

**Primary evidence**
Two metric panels.

**Left**
`Hazard AUC` with small survive/eliminated score distributions.

**Right**
`Rank Spearman rho` with the 4-team predicted-order table.

**Footer**
`Both metrics are computed per cut phase over at-risk teams; late-phase ranking is therefore ranking among survivors.`

**Design**
- Keep formulas minimal.
- Use the 4-team table because it explains why rank correlation, not exact rank error, is central.

**Philosophy applied**
- CTML coherence: explain metric before results.
- HBR declarative visualization: each metric panel answers a decision question.

---

## Slide 6 — Elimination Results

**Assertion headline**
`Elimination prediction changes little across graph encoders.`

**Primary evidence**
Existing `fig5_hazard_by_phase.png`.

**Readout**
- `Encoder spread per phase: 0.009-0.021 AUC.`
- `Graph encoders average 0.809 vs no-graph 0.798 across phases.`
- `AUC declines toward late game as the zone contracts.`

**Deferred interpretation note**
`The next slide pair separates performance gain from topology signal.`

**Design**
- Use existing PNG; do not regenerate.
- Add only:
  - `widest encoder spread ~0.02` near P3,
  - `No graph lower envelope` near the dashed gray line.
- Avoid the word `smoothing` here if possible; reserve mechanism for Slide 8.

**Philosophy applied**
- Tufte: exact range, not exaggerated effect.
- Assertion-Evidence: one claim supported by the phase line figure.

---

## Slide 7 — Ranking Results

**Assertion headline**
`Ranking has a wider encoder spread and locks in during phases 3-6.`

**Primary evidence**
Existing `fig6_rank_by_phase.png`.

**Readout**
- `Encoder spread per phase: 0.056-0.106 rho.`
- `GAT leads the rank forecast, averaging +0.075 rho over no graph.`
- `Rank alignment rises from phase 3 to 6 by +0.085 rho.`

**Mandatory null caveat**
`The ranking gain should not be read as observed topology signal; graph-null controls do not support that interpretation.`

**Design**
- Use existing PNG; do not regenerate.
- Add only:
  - early spread bracket,
  - late spread bracket,
  - P3-P6 lock-in band,
  - `GAT best ranker` direct label.

**Philosophy applied**
- Duarte contrast: ranking behaves differently from elimination.
- Tufte integrity: the caveat prevents overreading the red GAT line.

---

## Slide 8 — Interpretation and Limitation

**Assertion headline**
`The observed topology adds no usable signal beyond graph-based averaging.`

**Primary evidence**
Vector recreation of `fig2_graph_null.png`.

**Bars**
- `no_graph = 0.834`
- `observed = 0.845`
- `edge_shuffle = 0.847`

**Readout**
- `Graph helps relative to no graph.`
- `Observed topology does not beat edge shuffle.`
- `No phase is significantly better for observed topology.`

**Mechanism panel**
`Row-normalized message passing acts as a local weighted average. That can denoise node states, but it does not prove the specific real-world wiring is informative.`

**Root-cause band**
`The three interaction channels encode existence and intensity of interaction, not pairwise relative advantage.`

**Caption**
`All controls are row-normalized. y-axis is truncated to show the small effect; the total span is 0.013 AUC.`

**Design**
- This is the only slide where terms such as `message passing`, `low-pass`, and `smoothing` should appear.
- Make `edge_shuffle >= observed` visually unavoidable.

**Philosophy applied**
- Tufte: disclose axis truncation and show the uncomfortable comparison.
- Assertion-Evidence: null-model chart is the evidence, not an aside.

---

## Slide 9 — Future Work

**Assertion headline**
`Changing the relation type to contact-gated relative advantage revives a preliminary relational signal.`

**Status ribbon**
`Preliminary, validated; not part of the present main results.`

**Primary evidence**
Left: contact-gated relative-advantage concept.

Right: compact evidence panel:

- `observed - edge_shuffle: +0.0033 [+0.0023, +0.0043]`
- `observed - node_shuffle: +0.0023`
- `observed - no_graph: +0.0051`
- `k in {3,5,8}: +0.0035 / +0.0033 / +0.0039`
- `node-absolute control: +0.0023 [+0.0013, +0.0033]`

**Bottom caveat**
`Phase-tier preliminary result; effect sizes are small and full MR integration requires re-base.`

**Design**
- The visual should show the difference between interaction topology and relative-state edge, not just high-ground decoration.
- The edge feature idea must be shown as pairwise relation:
  `src - dst: dz, alive, health, zone`.
- Keep the future-work panel visually muted relative to main results.

**Philosophy applied**
- Duarte: negative result becomes a research direction.
- Tufte: effect sizes remain small and caveated.

---

## Slide 10 — Conclusion

**Assertion headline**
`Relation type, not GNN architecture, determines whether topology is informative.`

**Primary evidence**
Minimal two-clause contrast:

```text
Interaction relations
proximity / combat / co-pressure
-> graph-based averaging, no topology signal

Relative tactical advantage
elevation / numbers / health / zone, contact-gated
-> irreducibly relational signal (preliminary)
```

**Final line**
`The value lies in designing game-mechanic-aligned relations, not in using a GNN per se.`

**Design**
- White space heavy.
- No new numbers.
- `(preliminary)` remains visible beside advantage.

**Philosophy applied**
- Assertion-Evidence: conclusion is one defensible thesis.
- Duarte: closing resolves the tension introduced by the null result.

---

## 7. Revised Asset List

### Existing assets to use

- `docs/presentation_figures/fig0_labeling.png` — vector recreation reference
- `docs/presentation_figures/fig2_graph_null.png` — vector recreation reference
- `docs/presentation_figures/fig5_hazard_by_phase.png` — use PNG as figure
- `docs/presentation_figures/fig6_rank_by_phase.png` — use PNG as figure

### New or revised assets

1. **Title faint teaser**
   - Optional.
   - Must remain below 14% opacity.

2. **Slide 4 phase adjacency heatmaps**
   - Must be phase-level, not snapshot-level, because co-pressure exists at phase tier.
   - Three panels: proximity, combat, co-pressure.
   - Caption must say row-normalized and representative match/phase.

3. **Slide 9 contact-gated advantage concept**
   - Must show edge feature as relative state, not just terrain.
   - Include contact gate ON/OFF.
   - Include `(preliminary)` in the visual field.

---

## 8. Quality Checklist Before Build Handoff

### Argument

- Each slide title is a sentence assertion, except title slide.
- No result slide introduces new methods.
- No modeling slide contains performance numbers.
- The graph-null result is impossible to miss.
- Future work is visibly separated from present results.

### Technical correctness

- Phase graph is described as 3-channel.
- Snapshot graph is described as 2-channel.
- R-GCN is the only encoder described as consuming per-relation matrices directly.
- Node feature groups are described as major groups, not exhaustive manifest.
- Global context is described as `G[p]`.

### Visual design

- One primary evidence object per slide.
- No heavy decorative backgrounds.
- No chartjunk, 3D chart effects, or unnecessary icon clusters.
- Captions disclose axis truncation and preliminary status where relevant.
- All labels are legible at presentation-room distance.

### Cognitive load

- No slide has more than three readout annotations.
- No detached legend if direct labeling is possible.
- No paragraph competes with the figure.
- Any complex diagram is staged through grouping and proximity.

---

## 9. Build Handoff Summary

The revised design philosophy is:

> Treat each slide as an academic evidence claim. The headline states the claim; the visual proves or falsifies it; the caption protects the boundary of the claim.

The resulting deck should feel restrained, exact, and intellectually honest. Its strongest moment is not the model score, but the disciplined transition from a negative topology result to a sharper research hypothesis: relation type determines graph informativeness.

