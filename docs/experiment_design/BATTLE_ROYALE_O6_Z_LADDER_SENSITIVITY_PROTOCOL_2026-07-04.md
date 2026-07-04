# Battle Royale O6 Z-Ladder Sensitivity Protocol

Date: 2026-07-04
Class: `ExperimentContract`
Lifecycle: `active`

Completion note:

```text
The required-cohort OM6Z run was completed on 2026-07-05.  See
docs/experiment_design/BATTLE_ROYALE_O6_Z_LADDER_RESULT_2026-07-05.md.
```

## Purpose

This document pre-registers the next experiment after the paired bootstrap
result:

```text
Does a corrected PUBG O6 spatial-affordance proxy block add predictive signal
after OM5 when it includes the intended Z1-Z3 vertical and relative-height
features?
```

The current `om6_spatial_proxy` result is not a full O6 test.  It adds only:

```text
avg_enemy_density_100m_norm
min_nearest_enemy_norm
path_near_enemy_ratio
```

Those variables overlap with O4 exposure and O7 relation-density projections.
They do not test the planned O6 Z-ladder of vertical position, contact-gated
relative height, and zone-pressure interaction.

## Scope

This is a PUBG supervised sensitivity experiment.  It does not claim true
terrain slope, building cover, line of sight, or mesh occlusion.

The experiment remains separated by exact map / perspective / team-size cohort:

- Erangel TPP squad;
- Taego TPP squad;
- Miramar / Desert TPP squad;
- Rondo TPP squad;
- Sanhok TPP squad;
- Vikendi TPP squad;
- Paramo TPP squad;
- Karakin TPP squad;
- Erangel TPP solo;
- Erangel TPP duo;
- Taego TPP duo;
- Taego FPP squad;
- Erangel FPP squad.

No pooled training is allowed for this experiment.

## Required Pre-Run Gate

This is an ETL-first experiment, not a preset-only change.

Before launching any long run, verify:

| Gate | Required check | Pass condition |
| --- | --- | --- |
| feature existence | `pos_z` / team vertical summaries are present before the cut | non-null, non-degenerate values by cohort and phase |
| tensor exposure | intended Z1-Z3 features appear in the phase tensor feature-name list | `model_stgcn/v3/spec.py` can select them by normalized name |
| split reuse | required-cohort split JSON can be reused | same train/val/test match IDs as `br_om_ladder_required_v1` |
| leakage guard | no post-cut or outcome fields enter Z features | forbidden-input tests pass |
| ownership guard | no duplicate O4/O7 exposure fields are re-added as O6 | OM6Z contains Z features, not copied nearest-enemy density |

If the intended Z columns are absent from `stdb.v_team_phase_feature_v2` or the
V3 tensor feature list, implement the ETL and tensor mapping first.

## Intended Z Blocks

The feature names below are canonical targets for implementation.  The exact
normalized names can change only if the implementation document maps them back
to the same construct.

| Block | Construct | Example phase-level features | Claim role |
| --- | --- | --- | --- |
| `Z0` | no corrected O6 feature beyond OM5 | none | baseline |
| `Z1` | absolute / percentile vertical position | `team_z_mean_norm`, `team_z_percentile_alive`, `team_z_range_norm` | PUBG vertical-position proxy |
| `Z2` | contact-gated relative height | `contact_delta_z_mean_norm`, `contact_delta_z_positive_ratio`, `high_ground_contact_advantage` | high-ground advantage proxy |
| `Z3` | Z2 under contraction pressure | `high_ground_x_boundary_pressure`, `z_percentile_x_hab_crowding_pressure`, `contact_delta_z_x_outside_safe_ratio` | spatial affordance under phase pressure |

Do not add `Z4` route conflict or `Z5` behavior-under-pressure features in this
run.  Those remain sensitivity blocks after the Z1-Z3 result is known.

## Model Protocol

Run a corrected preset:

```text
om6z_spatial_affordance = OM5 relation structure + Z1-Z3
```

The comparison is:

| Comparison | Interpretation |
| --- | --- |
| `OM5 -> om6_old` | already observed old spatial-exposure proxy increment |
| `OM5 -> om6z_spatial_affordance` | corrected Z-ladder increment |
| `om6_old -> om6z_spatial_affordance` | whether the corrected O6 block changes the old OM6 conclusion |

The full OM0-OM5 ladder does not need to be rerun unless ETL changes alter
earlier feature tensors.  If only new Z columns are appended, train only the
new `om6z_spatial_affordance` preset for each required cohort using:

- the same split JSON as `br_om_ladder_required_v1`;
- the same seed as the required OM ladder (`42`);
- the same model family, epoch budget, optimizer settings, and batch settings
  used for the original `om_ladder` run;
- `graph_encoder=none` and `edge_mode=none`, unless a separate O7 edge-stream
  experiment is explicitly launched.

## Confirmatory Targets

Primary confirmatory cells:

| Step | Phase group | Metrics |
| --- | --- | --- |
| `OM5 -> om6z_spatial_affordance` | early `1-2` | `hazard_auc`, `rank_spearman` |
| `OM5 -> om6z_spatial_affordance` | mid `3-4` | `hazard_auc`, `rank_spearman` |

Secondary support:

- `top10_auc` in early `1-2` and mid `3-4`;
- cohort-level comparison against `OM5 -> om6_old`;
- single-feature or sub-block diagnostics for `Z1`, `Z2`, and `Z3` if runtime
  permits.

Phase `7` is not a confirmatory target because the previous phase audit showed
unstable at-risk samples in the final tail.

## Pre-Registered Interpretation

Strong O6 support:

```text
At least one primary confirmatory phase-group/metric cell has positive median
delta, positive deltas in at least 9/13 cohorts, and paired CI support in at
least 5/13 cohorts; and `om6z_spatial_affordance` is directionally stronger
than the old `om6_spatial_proxy` in the same cell.
```

Weak or null O6 support:

```text
Z1-Z3 do not add robust signal after OM5.  This does not falsify the Battle
Royale framework; it bounds PUBG as geometry-poor for O6 and strengthens the
need for Caldera-style construct validation for richer spatial affordance.
```

The experiment is therefore informative in both outcomes:

- if Z-ladder features add signal, PUBG supports a measurable O6 proxy beyond
  pressure/habitat, threat exposure, and relation summaries;
- if they do not add signal, the paper should state that PUBG telemetry supports
  O2 x O6 habitat strongly, but does not by itself validate fine-grained O6
  terrain or high-ground affordance.

## Expected Artifacts

| Artifact | Role |
| --- | --- |
| feature availability audit for Z1-Z3 | confirms whether ETL is needed |
| ETL or tensor-mapping patch if needed | exposes Z1-Z3 to V3 phase tensors |
| `om6z_spatial_affordance` preset | executable corrected O6 block |
| required-cohort `om6z` run root | one new run per exact cohort |
| `OM5 -> om6z` aggregate metrics | macro comparison with old OM6 |
| phase-group paired bootstrap | uncertainty check over early/mid confirmatory cells |
| result document | claim boundary for PUBG O6 spatial-affordance proxy |

## Immediate Next Step

Run the Z1-Z3 feature availability audit:

```text
Check whether phase-level vertical summaries, contact-gated relative-height
summaries, and zone-interaction Z features already exist in the DB views and V3
feature-name surface.
```

If they do not exist, implement ETL before scheduling model runs.
