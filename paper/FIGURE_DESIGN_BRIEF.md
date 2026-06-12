# PUBG Paper Figure Design Brief

> Purpose: 직접 figure를 다시 그릴 때 사용할 핵심 구조 정리.
> Do not use the current generated PNGs as visual reference. Use this document
> and the paper text as the source of truth.

## 1. Core Pipeline Overview Figure

### Message

The figure should show one continuous transformation:

```text
official telemetry
  -> phase-aligned STDB
  -> team-phase feature rows + dynamic team graphs
  -> DynamicSTGNN / feature-only baselines
  -> next-phase elimination hazard
```

The core conceptual split is:

- **RQ1: representation**
  - raw telemetry is reconstructed as a spatio-temporal database.
  - the database yields team-phase features and dynamic team relations.
- **RQ2: prediction**
  - feature rows define the next-phase hazard task.
  - dynamic graph edges are tested for additional predictive signal.

### Recommended Layout

Use a left-to-right pipeline with two grouped regions.

```text
[Official Telemetry]
       |
       v
[Phase-aligned STDB]
       |
       +---------------------> [Team-phase Features X]
       |
       +---------------------> [Dynamic Team Graphs A]
                                  |
                                  v
                         [DynamicSTGNN]
                                  |
                                  v
                       [Next-phase Hazard]

Bottom braces:
Official Telemetry -> Features/Graphs = RQ1 representation
DynamicSTGNN -> Hazard/Evaluation = RQ2 prediction
```

### Required Blocks and Labels

Use these exact or near-exact English labels:

- `Official telemetry`
  - optional sublabels: `time`, `actor/team`, `location`, `state/event`
- `Phase-aligned STDB`
  - sublabels: `moving teams`, `dynamic zones`, `spatial events`, `phase intervals`
- `Team-phase features`
  - sublabel: `X: 66-d per (match, team, phase)`
- `Dynamic team graphs`
  - sublabels/legend: `proximity`, `combat`, `co-pressure`
- `DynamicSTGNN`
  - sublabel: `2 gated ST blocks`
- `Next-phase hazard`
  - sublabel: `p(eliminated before next phase)`
- Optional dashed auxiliary box:
  - `placement heads`
  - `regularizers only`
- Optional evaluation strip:
  - `HGB feature-only`
  - `no-edge neural`
  - `proximity`
  - `proximity + combat`
  - `full graph`

### Semantics to Preserve

- `Official telemetry` is the source. It is not yet the model input.
- `Phase-aligned STDB` is the representation layer, not a generic data cache.
- `Team-phase features` and `Dynamic team graphs` are parallel outputs of STDB.
- `DynamicSTGNN` receives both `X` and `A`.
- The main output is the next-phase elimination hazard. Final-rank/top-k heads
  are auxiliary only.
- The graph is not just a visualization: it is the model input adjacency tensor.

### What Not to Put in the Figure

- Do not include table/view names such as `stdb.team_phase_labels`.
- Do not include long equations.
- Do not include full metric tables.
- Do not include raw implementation details such as SQL, checkpoint names, or
  file paths.
- Do not imply that phase 0 is used by the model. Model input uses phases 1-8;
  cut phases are 1-7.

### Caption Draft

```text
Pipeline overview. Official telemetry is reconstructed into a phase-aligned
STDB and summarized into team-phase features and dynamic team graphs (RQ1).
DynamicSTGNN and feature-only baselines are then evaluated for next-phase
team elimination hazard (RQ2). Placement heads are auxiliary regularizers.
```

## 2. DynamicSTGNN Architecture Figure

### Message

The architecture figure should show how the model converts a phase sequence of
team features and heterogeneous graph adjacencies into hazard probabilities.

```text
X: B x T x N x F
A: B x T x E x N x N
       |
       v
node embedding
       |
       v
ST block x 2
       |
       +--> hazard head: B x (T-1) x N
       |
       +--> final-rank head / top-k head (auxiliary)
```

### Tensor Definitions

Use these constants:

- `T = 8` phases.
- `cut phases = 1-7`.
- `N <= 128` teams, padded.
- `F = 66` node features.
- `E = 3` relation channels.

Model input:

```text
X: B x T x N x F
A: B x T x E x N x N
```

Model outputs:

```text
hazard logits: B x (T-1) x N
rank prediction: B x N
top-k logits: B x N x 2
```

### Recommended Layout

Use a three-panel or three-column architecture diagram.

```text
(a) Input sequence
    G1, G2, ..., GT
    X and A tensors
    edge legend

(b) DynamicSTGNN network
    Input embedding 66 -> 64
    ST block 1
    ST block 2
    Hazard head
    Auxiliary heads dashed

(c) ST block detail
    gated causal temporal conv
      -> heterogeneous graph conv
      -> gated causal temporal conv
      -> LayerNorm + residual
```

### ST Block Detail

Each ST block should show this exact order:

```text
input H
  -> gated causal temporal convolution
  -> heterogeneous graph convolution
  -> gated causal temporal convolution
  -> LayerNorm
  -> residual addition
```

Inside heterogeneous graph convolution:

```text
proximity branch
combat branch
co-pressure branch
  -> learned softmax channel mix
  -> skip connection
```

Visual convention:

- `proximity`: blue, undirected or symmetric line.
- `combat`: red, directed arrow.
- `co-pressure`: purple dashed line.
- residual path: gray or dark dashed skip arc.
- auxiliary heads: gray dashed boxes.

### Edge Channel Meaning

Use short labels only, but preserve the semantics:

- `proximity`
  - pairwise spatial closeness between teams.
  - symmetric.
  - proximity channel has self-loops.
- `combat`
  - directed attacker-to-victim relation.
  - weight grows with damage/knockdown/kill evidence.
- `co-pressure`
  - teams jointly under safe-zone pressure and spatially close.
  - represents shared spatial constraint.

All relation channels are computed from phase-k observations only and then
row-normalized.

### Hazard Output Meaning

The main head predicts:

```text
P(team i is eliminated before the next phase | observed through cut phase k)
```

Only `target_mask = True` cells are used in training/evaluation:

- team exists in the match.
- team is alive at the cut phase.
- a next phase label exists.

The model may output values for padded or already eliminated slots, but those
are ignored by the mask.

### Caption Draft

```text
DynamicSTGNN architecture. The model receives a sequence of heterogeneous
team graphs, one per phase, with team-phase features X and relation tensors A.
Node states pass through two gated spatio-temporal blocks. Each block combines
causal temporal convolutions with heterogeneous graph aggregation over
proximity, combat, and co-pressure channels. The primary head outputs
next-phase elimination hazard; placement heads are auxiliary regularizers.
```

## 3. Visual Style Guidance

Use a restrained academic diagram style:

- White background.
- Thin gray/blue-gray strokes.
- Pastel fills for block groups.
- Font: Helvetica, Arial, or another clean sans-serif.
- Prefer compact labels over paragraphs.
- Use arrows only where data actually flows.
- Keep all text readable after scaling to `0.86\textwidth` in a two-column ACM
  paper.

Suggested palette:

```text
ink:        #253746
muted text: #8A98A7
feature:    #E8F4EC
graph:      #F0EAF8
model:      #FFF0E4
output:     #E6F0FA
proximity:  #2867B2
combat:     #C0392B
co-pressure:#8E5FB8
```

## 4. Consistency Checklist

Before using the figure in the paper, check:

- Does the pipeline close RQ1 and RQ2 explicitly?
- Does it show both `X` features and `A` graph tensors?
- Does it avoid suggesting that output is final rank rather than hazard?
- Does architecture show two ST blocks?
- Does the ST block show causal temporal processing before future phases?
- Are auxiliary heads visually secondary?
- Are relation colors consistent with the graph example figure?
- Are all labels short enough for two-column scaling?
