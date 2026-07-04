# Battle Royale Framework Validation Plan

Date: 2026-07-02

## Purpose

This document defines the experiments needed to validate the project as a
general Battle Royale framework, not merely a PUBG-specific feature set.

The target claim is:

```text
Battle Royale survival can be represented as a phase-conditioned process in
which contraction or elimination pressure changes the survival environment,
survival capital constrains agent behavior, relations among agents change, and
those changes affect survival outcomes with state feedback across phases.
```

## Research Questions

### RQ1. Environment Change

How does phase contraction or elimination pressure change the survival
environment?

Examples:

- playable space shrinks;
- safe area per alive team decreases;
- boundary pressure increases;
- route conflict increases;
- local enemy density changes.

### RQ2. Behavioral Response

How do agents/teams change behavior as the survival environment changes?

Examples:

- rotation timing changes;
- speed and path shape change;
- contact avoidance or engagement changes;
- resource/vehicle use changes;
- positioning and relative height use change.

### RQ3. Survival Effect

Which feature families explain next-phase survival hazard or rank-related
outcomes, and does their importance change by phase?

### RQ4. Framework Generality

Does the ontology remain meaningful under different observability regimes,
maps, modes, and data sources?

## Hypotheses

| ID | Hypothesis | Evidence required |
| --- | --- | --- |
| H1 | Contraction / elimination pressure measurably changes the survival environment | phase-wise shifts in zone pressure, density, route conflict |
| H2 | Teams adapt behavior to phase-conditioned pressure | phase-wise changes in movement, rotation, contact, resources |
| H3 | Environment and behavior features improve next-phase survival prediction | supervised PUBG ablations |
| H4 | Relation features matter because pressure is multi-agent | graph/relation ablations and null controls |
| H5 | The ontology is general because it survives observability masking and cross-domain projection | observability matrix, mask robustness, Caldera construct validation |

## Ontology Validation Gate

Use the protocol below as the interpretation gate for the experiment suite:

```text
docs/experiment_design/BATTLE_ROYALE_ONTOLOGY_VALIDATION_PROTOCOL_2026-07-03.md
```

The gate separates six evidence types:

- conceptual coverage from official or publisher-controlled genre sources;
- operational availability as legal feature inputs in PUBG;
- phase-mechanism sanity;
- supervised predictive utility;
- robustness across separately trained map / perspective / team-size cohorts;
- Warzone/Caldera construct projection for geometry-rich spatial affordances.

The required PUBG cohorts are not pooled for the main claim.  OM0-OM6 must be
trained and evaluated separately for each required map / perspective / team-size
cohort, then summarized only after cohort-level results exist.

## Experiment Suite

### E1. Ontology Coverage, Corpus Sanity, And Observability

Question:

```text
Which Battle Royale primitives appear across representative genre documents,
and which are observable in each dataset?
```

Method:

- build a 12-15 game mini corpus coverage table from official or
  publisher-controlled sources;
- use `BATTLE_ROYALE_ONTOLOGY_VALIDATION_PROTOCOL_2026-07-03.md` as the
  OV0-OV1 claim boundary;
- build a family-by-dataset observability matrix;
- mark each family as direct, proxy, masked, or future-asset;
- separate input features from labels/masks.

Expected output:

- paper table showing the ontology is not PUBG-only;
- paper table showing PUBG and Caldera have complementary observability;
- proof that the framework is not tied to one exact variable set.

Success criterion:

```text
Representative game documents and available datasets can be represented as
masked projections of the same ontology.  The full corpus saturation protocol is
confirmatory and does not block PUBG prediction.
```

### E2. Phase-Contraction Environment Analysis

Question:

```text
Does the match environment actually change by phase in the way the framework
assumes?
```

PUBG measurements:

- safe radius and radius change;
- safe center shift;
- alive teams per phase;
- safe area per alive team;
- distance to boundary;
- required speed to safe area;
- enemy density and nearest enemy;
- route conflict / co-pressure.

Analysis:

- phase-wise distributions;
- early/mid/late comparisons;
- effect sizes, not only p-values;
- map-stratified checks.

Expected output:

```text
Phase contraction creates measurable habitat compression through O2 x O6 and
changes exposure pressure.
```

### E3. Agent Behavioral Response Analysis

Question:

```text
Do agents change movement, contact, and resource behavior as the environment
changes?
```

PUBG measurements:

- path length;
- displacement;
- speed;
- sinuosity;
- rotation timing;
- zone-entry time;
- outside-safe duration;
- combat/contact rate;
- vehicle/resource use where available;
- vertical/relative-height behavior where available.

Analysis:

- regress behavior family summaries on contraction-pressure variables;
- compare behavior before and after major zone shifts;
- stratify by survival capital and map/mode.

Expected output:

```text
Agents respond to contraction pressure through behavioral response, contact,
and positioning
changes.
```

### E4. Supervised Survival Utility

Question:

```text
Do ontology feature families help predict next-phase survival?
```

Primary data:

- PUBG high-observability axis;
- atomic cohort unit: `map_name x perspective x team_size`;
- validation protocol:
  `docs/experiment_design/BATTLE_ROYALE_ONTOLOGY_VALIDATION_PROTOCOL_2026-07-03.md`;
- cohort contract:
  `docs/data_layer/BATTLE_ROYALE_COHORT_REGISTRY_2026-07-03.md`;
- required cohort set:
  `docs/experiment_design/BATTLE_ROYALE_REQUIRED_COHORT_SET_2026-07-03.md`.

Cohort rule:

```text
OM0-OM6 comparisons must be reportable by registered atomic cohort.
Main cohorts are map / perspective / team-size strata with at least 300
graph-eligible matches.  Group summaries are secondary and must name the
included atomic cohorts.
```

Main training rule:

```text
Train OM0-OM6 separately for each required map / perspective / team-size cohort.
```

Required cohort set:

```text
13 independently trained cohorts covering:
- TPP squad map variation
- same-map FPP/TPP squad perspective contrast
- same-map squad/duo/solo team-size contrast
```

Primary target:

```text
Y_i(k+1) = team i eliminated before or during next phase
given observations through phase k
```

Operational-manifest ablations:

| Model | Included families |
| --- | --- |
| OM0 | conventional baseline / existing core |
| OM1 | OM0 + survival capital |
| OM2 | OM1 + movement/rotation |
| OM3 | OM2 + contraction / zone pressure |
| OM4 | OM3 + interaction / combat / threat |
| OM5 | OM4 + relation/interaction structure |
| OM6 | OM5 + spatial/terrain proxies |
| OM7 | OM6 + resource/mobility where available |

Metrics:

- AUC;
- average precision;
- BCE / Brier;
- calibration by phase;
- phase-wise metrics;
- paired tests at match level.

Success criterion:

```text
Feature families add predictive signal in a phase-conditioned way, not only as
one global aggregate.
```

### E5. Phase-Specific Mechanism Analysis

Question:

```text
Do different feature families matter in different phase regimes?
```

Analysis:

- train/evaluate phase-wise or report per-phase metrics;
- compare family uplift by phase;
- inspect feature importance or ablation deltas by early/mid/late phase;
- test whether zone and relation features increase under habitat compression.

Expected pattern:

| Phase regime | Expected mechanisms |
| --- | --- |
| early | loot, initial movement, first contact |
| mid | rotation, zone pressure, route conflict |
| late | contact pressure, survival capital, relative positioning |

Success criterion:

```text
The framework explains survival as phase-conditioned mechanisms, not a single
static ranking model.
```

### E6. Cross-Map / Cross-Mode Generalization

Question:

```text
Does the ontology generalize beyond one map or mode?
```

Candidate splits:

- train on a subset of PUBG maps, test on held-out map;
- train on TPP squad, test on FPP squad if coverage allows;
- train on common maps, evaluate rare-map robustness;
- compare coordinate-only features against ontology-family features.

Success criterion:

```text
Ontology-family features degrade less than raw coordinate or narrowly tuned
features under map/mode shift.
```

### E7. Observability-Mask Robustness

Question:

```text
Can the framework still operate when some feature families are missing?
```

Method:

- start from PUBG full observability;
- mask families to simulate lower-observability datasets;
- evaluate graceful degradation.

Mask settings:

| Mask | Meaning |
| --- | --- |
| M0 | full PUBG |
| M1 | no combat |
| M2 | no resource |
| M3 | no zone |
| M4 | movement + contact only |
| M5 | movement + terrain-inspired proxy only |
| M6 | Caldera-like: trajectory/spatial only, no label used for supervised claim |

Success criterion:

```text
The framework has explicit masks and can report which claims remain valid under
each observability profile.
```

### E8. Caldera Construct Validation

Question:

```text
Are the spatial/terrain affordance constructs measurable in a geometry-rich
Battle Royale map?
```

Caldera outputs:

- 1-player detailed mesh snapshot;
- 36-match descriptive feature distributions after life segmentation;
- terrain lookup QC;
- tactical-state counts;
- pairwise LOS or local isovist demo;
- optional self-supervised movement/terrain-state task.

Important boundary:

```text
Caldera validates construct measurability, not survival prediction.
```

Success criterion:

```text
The same spatial-tactical ontology has a geometry-native instantiation in
Caldera and a telemetry-proxy instantiation in PUBG.
```

## Minimal Paper-Ready Experiment Set

If time is constrained, prioritize:

1. E1 ontology / observability matrix.
2. E2 phase-contraction environment analysis.
3. E4 supervised PUBG feature-family ablation.
4. E5 phase-specific mechanism analysis.
5. E8 Caldera construct validation as a supporting figure/table.

Optional but valuable:

- E6 cross-map / cross-mode generalization;
- E7 observability-mask robustness.

## Claim Mapping

| Claim | Required experiment |
| --- | --- |
| Battle Royale survival environment changes by phase | E2 |
| Agents change behavior under contraction pressure | E3 |
| Feature ontology has predictive utility | E4 |
| Mechanisms are phase-conditioned | E5 |
| Framework is broader than one PUBG feature set | E1, E6, E7 |
| Terrain affordance is a real computable construct | E8 |
| Ontology validation path is staged and bounded | Ontology validation protocol, E1, E4, E6, E8 |

## Failure Interpretation

If a family does not improve supervised prediction, do not delete it from the
ontology immediately.

Interpretation order:

1. Is the family actually observable in this dataset?
2. Is the implementation a weak proxy?
3. Is the family phase-specific and diluted by global evaluation?
4. Is the label insensitive to this mechanism?
5. Does the family belong in construct validation rather than supervised
   prediction?

This protects the framework from overfitting its ontology to one dataset while
still requiring empirical discipline.
