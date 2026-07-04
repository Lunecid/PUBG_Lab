# Caldera O6 Construct-Validation Artifact Scope

Date: 2026-07-05
Class: `WorldModel`
Lifecycle: `active`

## Purpose

This document limits the Caldera / Warzone O6 artifact after the PUBG O6Z
sensitivity result.

The corrected PUBG O6Z run showed:

```text
PUBG `pos_z`-only proxies do not robustly validate fine-grained spatial or
terrain affordance after OM5.
```

Therefore Caldera should be used as:

```text
geometry-rich construct validation for O6 spatial affordance
```

not as:

```text
supervised phase-wise survival or rank prediction
```

## Scope Decision

Build one compact artifact package:

```text
1 composite figure + 1 descriptive statistics table
```

Do not restart the full-world reconstruction problem.

## Required Figure

The figure should combine existing artifacts:

| Panel | Content | Source |
| --- | --- | --- |
| A | semantic Caldera basemap | existing semantic basemap render/cache |
| B | selected player/team trajectory overlay | existing trajectory JSON / top-down render |
| C | local terrain-experience zoom or player-path panel | existing terrain-experience extraction/render |
| D | small feature callout panel | derived v0 terrain-experience features |

The figure should demonstrate:

```text
geometry-rich data can measure terrain/route/building/environment context that
PUBG supervised telemetry cannot directly observe.
```

It should not imply:

```text
Caldera has valid survival/rank labels for supervised prediction.
```

## Required Table

The table should report descriptive O6 construct features, for example:

| Feature family | Example values |
| --- | --- |
| vertical consistency | `height_above_terrain_m` distribution |
| local terrain shape | slope / relief / roughness proxy |
| built environment | building proximity / building-mesh density |
| route context | road / runway / corridor proximity |
| semantic context | terrain, coast, water, vegetation, road, runway, building |
| tactical state | distribution after life-segmentation guard |

The table is descriptive construct evidence, not label performance.

## Life-Segmentation Guard

Before reporting trajectory-derived statistics, apply or document a guard
against redeploy / teleport-style gaps.

Minimum rule:

```text
new life segment if time gap > 30s and displacement > 500m
```

Rationale:

```text
Without life segmentation, a single trajectory window may mix physically
separate combat episodes.
```

## Optional Stretch

Pairwise LOS is the best optional extension because it bridges O6 and O7:

```text
geometry-rich data can expose pairwise visibility / occlusion relations that
PUBG cannot mirror from basic telemetry.
```

But LOS is not required for the first paper artifact.  It should remain a
stretch goal unless the manuscript draft is already stable.

## Claim Boundary

Allowed:

```text
Caldera demonstrates how the framework's O6 family can be projected into
geometry-derived terrain and environment measurements.
```

Allowed:

```text
This complements PUBG, which is label-rich but geometry-poor.
```

Avoid:

```text
Caldera validates the supervised survival model.
```

Avoid:

```text
The rendered map is a full game-engine reconstruction.
```

## Immediate Next Step

Assemble the composite figure from existing outputs first.  Only add new mesh
or LOS computation after the manuscript sections are underway.

