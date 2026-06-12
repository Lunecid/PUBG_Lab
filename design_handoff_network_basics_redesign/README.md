# Handoff: Network Basics — Page Redesign

## Overview

This bundle contains **three high-fidelity design directions** for the
"Network Basics / 네트워크 기초" page of a PUBG match-network analysis
research site. The page introduces a graph-theoretic vocabulary
(G = (V, E, w), degree, modularity, centralities, community detection) and
connects each definition to three downstream hypotheses (H1·H2·H3) about
team play style.

The three directions explore different editorial voices for the same content:

| ID | Name | Voice | Best when |
|---|---|---|---|
| **A** | Editorial Tufte | Quiet academic paper. Side-margin notes, sparse hairlines, hierarchy via type. | The page should feel scholarly / contemplative. |
| **B** | Pudding Datavis | Magazine-style data journalism. Big hero network, oversized numbers, strong rhythm. | The page should be the *front door* and seduce a non-expert. |
| **C** | Lab Notebook | Field-manual / lab-notebook. Grid paper, monospace, plate numbers, blueprints. | The page should signal "real research artifact, not a marketing site." |

The user has approved the visual direction in this bundle. Pick one (or
hybridize — they share the same data and section structure) and implement
it in the production codebase.

## About the Design Files

The files in this bundle are **design references created in HTML/JSX**. They
are prototypes rendered with React + Babel-standalone in a single HTML host
page; they are **not production code to copy verbatim**.

Your task is to **recreate these designs in the target codebase's existing
environment** — its component library, its CSS / Tailwind / styled-components
conventions, its routing, its data layer — using the patterns that codebase
has already established. If the project doesn't have a frontend yet, choose
the most appropriate framework (Next.js + Tailwind is a reasonable default
for a content-heavy site like this) and implement there.

The HTML in this bundle is just the spec. Open
`Network Basics Redesign.html` in a browser to see the live behavior of all
three directions side-by-side; click the ⤢ icon on any artboard for
fullscreen.

## Fidelity

**High-fidelity (hifi).** Colors, typography, spacing, figure construction,
and content are all final. Implement pixel-perfectly using the codebase's
existing libraries. The only intentionally placeholder element is the
synthetic graph data used by the SVG figures (see *Figures* below) — replace
it with your real telemetry-derived graph.

## Page Structure

All three directions cover the same six conceptual sections, in this order.
Section numbering is **§0 – §5** in A and C, **§1 – §6** in B (it has an
extra hero), but the content alignment is the same.

| § | Topic | What it teaches |
|---|---|---|
| **0 / Hero** | The graph definition | One match → one weighted graph G = (V, E, w). Node = team, edge = combat pair, weight = damage. |
| **1** | Stats overview | \|V\|=26.4, \|E\|=42.6, density=0.131, modularity Q=0.591, avg degree=3.24, clustering=0.398. |
| **2** | Adjacency / matrix | A_ij = w(i,j); D = diag(d_1, …, d_n); how the matrix view follows from the graph view. |
| **3** | Centralities | C_D (degree), C_W (weighted degree), C_B (betweenness — bridges = H2 motif). |
| **4** | Community detection | Louvain partition, modularity Q ≈ 0.59, what "≫ 0.3 random threshold" means. |
| **5 / 6** | Hypothesis map | H1 (외곽 파밍 vs 교전 참여), H2 (다중 교전 중심), H3 (존버 vs 활발 이동) — the formula each hypothesis aggregates from these primitives. |

The page is read top-to-bottom on a fixed 1280px content width. There is no
mobile layout in scope — the consumer is a researcher on a desktop browser.

## Three Directions In Detail

### Option A — Editorial Tufte
*File: `redesign/option-a-tufte.jsx`*

A three-column page (main 720 / gap 60 / margin 320). Body lives in the main
column; **margin notes** live in the right column at varying `top` offsets,
hand-anchored next to the paragraph they annotate. Section headers are
hairline-ruled, numbers are setText with `Newsreader` (serif), captions are
italic Newsreader, all metadata is `JetBrains Mono` 11px uppercase
.08em-tracked.

Key components:
- `MarginNote` — absolutely positioned at `left: main_w + gap`, width `margin_w`. Pass `top` in px to anchor it to the paragraph it annotates.
- `SmallSparkBars` — inline 200×60 bar chart used in the stats table.
- `AdjacencyMatrix` — tiny 18px-cell heatmap; scarcity speaks louder than chrome.
- `SectionHead` — `<num kicker> · <title>` separated by a `1px solid ${rule}` hairline.
- `DefRow` — definition-list row, `120px 1fr` grid, top-bordered.

### Option B — Pudding Datavis
*File: `redesign/option-b-pudding.jsx`*

Magazine spread. **Hero is a full-bleed cream panel** with a 1080×460
network diagram and a labeled "bridge" node (H2 motif). Stats become
**oversized numbers** (`GiantStat`, 84px Newsreader). Section §6 is one
of the visual anchors — a *light cream* full-bleed panel containing three
hypothesis cards with paper-white backgrounds and a 3px accent top-border
each.

Key components:
- `HeroNetwork` — 1080×460. Picks the node with the most cross-community edges and rings it as the "bridge."
- `GiantStat` — 84px tabular numerals on top of an 11px monospace label and 13px italic Newsreader hint.
- `SectionMarkB` — `num` in monospace + accent, hairline divider, then 48px serif title.

### Option C — Lab Notebook
*File: `redesign/option-c-fieldmanual.jsx`*

Document layout on a **grid-paper background** (inline SVG `<pattern>`
14×14 with a 70×70 darker overlay). Monospace-forward (`JetBrains Mono`
in titles). Figures are *plates* with corner ticks and "PL · 01" labels.
Tables have heavy ink borders. Strong "this is a real lab artifact" feel.

Key components:
- `GridPaperBG` — absolutely positioned, two-tier dot/grid pattern.
- `Plate` — figure header with corner-tick frames and `PL · NN` label.
- `FieldNetwork` — same network rendered with blueprint/oxblood/ink palette and node-coordinate ticks.
- `MiniNetwork`, `DegreeHistogram` — shared SVG primitives, also reused by A and B with different color args.

## Design Tokens

### Option A · Editorial Tufte
```js
{
  paper:  '#fbf8ee',   // page background
  ink:    '#1d1a14',   // primary text
  muted:  '#7a715f',   // secondary text, captions
  faint:  '#b8af9a',   // de-emphasized text, hairlines on figures
  rule:   '#dcd2bc',   // table/divider rules
  accent: '#a23232',   // emphasis, highlighted stats
  page_w: 1280,
  main_w: 720,
  margin_w: 320,
  gap: 60,
}
```

### Option B · Pudding Datavis
```js
{
  paper:   '#fefcf7',  // page background
  cream:   '#f3eddd',  // hero + §6 panel background (warm cream)
  ink:     '#15110a',  // primary text
  muted:   '#6b6558',
  faint:   '#cdc5b3',
  rule:    '#d8d1c2',
  accent:  '#b02a2a',  // primary accent (oxblood)
  accent2: '#3a4a5a',  // steel blue — community 2
  accent3: '#6b8e3a',  // olive       — community 3
}
```

### Option C · Lab Notebook
```js
{
  paper:     '#f4efe2',  // grid-paper background base
  ink:       '#0e0c08',  // primary text
  ink2:      '#2c2820',  // (legacy — was used for dark NOTE box; no longer needed)
  muted:     '#5e574a',
  faint:     '#a8a08e',
  rule:      '#c9c0a8',
  accent:    '#7a2018',  // deeper oxblood
  blueprint: '#26405a',  // figure secondary color
}
```

### Shared
- **Body font**: `Inter`, with `Noto Sans KR` for Korean.
- **Display font**: `Newsreader` (serif, weight 400–500, optical-size 18+).
- **Mono font**: `JetBrains Mono` for stats, code, metadata.
- **Math**: KaTeX, loaded from CDN. The `<TeX>` component is a thin wrapper around `katex.render`.
- **Numerals**: always `font-variant-numeric: tabular-nums` for any column of figures.

### Type scale (used across all three)

| Use | Size | Weight | Family |
|---|---|---|---|
| Display title (hero h1) | 56–96px | 500 | Newsreader |
| Section title | 36–56px | 500 | Newsreader |
| Subsection / body lead | 22–28px | 400 | Newsreader (italic for pull-quote) |
| Body | 15–17px | 400 | Inter / Noto Sans KR |
| Caption | 13px | 400 | Newsreader italic |
| Metadata / kicker | 10–11px | 600 | JetBrains Mono, .14–.16em letter-spacing, uppercase |
| Giant stat | 64–84px | 500 | Newsreader, tabular-nums |

### Spacing
8px base unit. Common values used: 8, 12, 14, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 120.
Section-to-section vertical rhythm is 80–120px.

### Figures (shared)

- `makeNetwork(seed, n)` — deterministic 2-community force-laid graph.
  Returns `{ nodes: [{x,y,community}], edges: [{a,b,w}] }`. **Replace
  with your real telemetry-derived graph in production**; the structure
  (nodes with x,y,community + weighted edges) is what the figures expect.
- `MiniNetwork`, `FieldNetwork`, `HeroNetwork` — SVG renderings of
  `makeNetwork` output. They differ only in palette, sizing, and whether
  they highlight a "bridge" node.
- `DegreeHistogram` — 460×200 SVG bar chart. Mean bar uses `accent`.
- `AdjacencyMatrix` / `AdjacencyHeatmapB` — square heatmap, `cell` × `cell` tiles, low-opacity zero cells.

## Interactions & Behavior

The page itself is **a static long-read**, not an app. The only
interactivity in scope:

1. **No interactive widgets in the page body.** All numbers shown are
   means/medians from the actual analysis — they should be rendered from
   data at build time, not client-side.
2. **Math typesetting (KaTeX)** runs once on mount per `<TeX>` element. In
   the production codebase prefer build-time typesetting (e.g. `rehype-katex`
   in MDX) to avoid the FOUC the prototype has.
3. **No animations** — keep transitions for hover state of internal
   navigation links only (200ms color/opacity, ease-out).

The `Network Basics Redesign.html` host file *adds* a fullscreen
"design canvas" with arrow-key navigation and pan/zoom across the three
directions. **None of that chrome is part of the page itself** — it is
purely a side-by-side comparison tool for review. Ignore
`redesign/design-canvas.jsx` when implementing.

## State Management

None. The page is content + a few derived statistics. If implemented in
Next.js, this is an RSC / static page. If the analysis is rerun, the
numbers and the network graph data should be regenerated at build time
and dropped into the page as constants (or read from a JSON file).

## Assets

- **Fonts** — `Inter`, `Newsreader`, `JetBrains Mono`, `Noto Sans KR`. All
  loaded from Google Fonts in the prototype. Use the same families in
  production via `next/font` or the codebase's font convention.
- **KaTeX** — CSS + JS from the official CDN. In production install
  `katex` + `rehype-katex` (or your codebase's equivalent).
- **No images, icons, or logos.** Every figure in the design is drawn
  inline with SVG — copy the SVG components directly.

## Files In This Bundle

| Path | What it is |
|---|---|
| `Network Basics Redesign.html` | Host page. Open in a browser to see all three directions. |
| `redesign/option-a-tufte.jsx` | Direction A source. Pure JSX, no build step. |
| `redesign/option-b-pudding.jsx` | Direction B source. |
| `redesign/option-c-fieldmanual.jsx` | Direction C source. |
| `redesign/design-canvas.jsx` | The review chrome (artboard grid + fullscreen viewer). **Do not port to production.** |

When implementing, the three `option-*.jsx` files are your spec. Ignore the
host HTML's loading shim and the design-canvas wrapper — those exist only
so a non-developer reviewer can see all three options in one tab.

## Implementation Notes

1. **Pick one direction first** — don't try to make a runtime toggle. The
   user has reviewed all three; they will tell you which to ship (or how
   to hybridize). Hybrids are reasonable: A's margin-note system + B's
   hero, for example.
2. **Section headers are the system.** Each direction has its own
   `SectionHead` / `SectionMarkB` / `SectionC` component — keep these as
   their own component in the codebase. They're the pattern that holds
   the page together.
3. **Grids over flex** for the page-level columns (A's three-column layout,
   B's stat strips, C's plate grids). The math is easier and the result
   prints more predictably.
4. **Print stylesheet** would be a nice-to-have — the lab-notebook and
   editorial directions both look like they want to be PDF-able. Out of
   scope for v1 unless the user asks.
5. **Korean + English mixed copy** — the prototype uses both, deliberately.
   Don't translate either way; preserve the bilingual phrasing as-is.
   Make sure the body font stack falls through to `Noto Sans KR` for
   Hangul ranges.
6. **The KaTeX call** in the prototype runs `katex.render` in a
   `useEffect`. In a production React + Next setup, prefer **server-side
   math** via `remark-math` + `rehype-katex` if the math is in MDX, or
   pre-render via `katex.renderToString` at build time and dangerouslySet
   the result. Avoid client-side KaTeX for a static research page.
