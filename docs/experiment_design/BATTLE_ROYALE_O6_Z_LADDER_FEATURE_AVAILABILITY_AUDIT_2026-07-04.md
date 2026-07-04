# Battle Royale O6 Z-Ladder Feature Availability Audit

Date: 2026-07-04
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This audit executes the pre-run gate from:

```text
docs/experiment_design/BATTLE_ROYALE_O6_Z_LADDER_SENSITIVITY_PROTOCOL_2026-07-04.md
```

It asks:

```text
Can the corrected O6 Z1-Z3 spatial-affordance features be selected by the
current V3 phase tensor, or does OM6Z require ETL/tensor mapping first?
```

## Short Answer

The result is:

```text
ETL_REQUIRED_BEFORE_OM6Z
```

Follow-up status after implementation on 2026-07-05:

```text
READY_FOR_OM6Z_PRESET
```

The missing ETL/tensor-mapping layer was implemented after this initial audit.
The post-ETL audit found all 9 Z1-Z3 target features exposed to the V3 phase
tensor surface, and the required-cohort ETL completed with zero remaining
matches across all 13 locked cohorts.

The model result is recorded in:

```text
docs/experiment_design/BATTLE_ROYALE_O6_Z_LADDER_RESULT_2026-07-05.md
```

The source vertical signal exists:

```text
stdb.team_snapshot_features.pos_z
```

and it is non-degenerate in all 13 required map / perspective / team-size
cohorts.  However, none of the 9 intended Z1-Z3 target features currently exist
in the V3 phase tensor or phase feature views.

Therefore the next work is not model training.  The next work is ETL/tensor
mapping to expose phase-level vertical, contact-gated relative-height, and
zone-interaction summaries.

## Artifacts

| Artifact | Role |
| --- | --- |
| `tools/audit_o6_z_ladder_availability.py` | DB/tensor-surface audit script |
| `model_stgcn/checkpoints/br_om_ladder_required_v1/o6_z_ladder_availability/o6_z_ladder_availability_summary.json` | summary and verdict |
| `model_stgcn/checkpoints/br_om_ladder_required_v1/o6_z_ladder_availability/o6_z_ladder_feature_availability.csv` | Z1-Z3 target-feature availability table |
| `model_stgcn/checkpoints/br_om_ladder_required_v1/o6_z_ladder_availability/o6_z_ladder_snapshot_pos_z_by_cohort.csv` | cohort-separated source `pos_z` distribution audit |

The audit was read-only with respect to the database.

## Target Feature Availability

Target feature count:

```text
9
```

Ready in V3 phase tensor:

```text
0/9
```

| Block | Feature | V3 phase tensor | Phase views | Snapshot source | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `Z1` | `team_z_mean_norm` | no | no | no | missing |
| `Z1` | `team_z_percentile_alive` | no | no | no | missing |
| `Z1` | `team_z_range_norm` | no | no | no | missing |
| `Z2` | `contact_delta_z_mean_norm` | no | no | no | missing |
| `Z2` | `contact_delta_z_positive_ratio` | no | no | no | missing |
| `Z2` | `high_ground_contact_advantage` | no | no | no | missing |
| `Z3` | `high_ground_x_boundary_pressure` | no | no | no | missing |
| `Z3` | `z_percentile_x_hab_crowding_pressure` | no | no | no | missing |
| `Z3` | `contact_delta_z_x_outside_safe_ratio` | no | no | no | missing |

Spatial-like columns found by table:

| Table | Spatial-like columns |
| --- | --- |
| `stdb.team_snapshot_features` | `pos_z` |
| `stdb.v_team_phase_feature_long` | none |
| `stdb.v_team_phase_feature_v2` | none |
| `stdb.team_phase_edge_cache` | none |
| `stdb.team_phase_network_features` | none |
| `stdb.team_phase_habitat_features` | none |

## Source Signal Audit

`pos_z` is present and non-degenerate for all required cohorts.

| Cohort | Snapshot rows | Matches | Median `pos_z` | Std `pos_z` | Degenerate |
| --- | ---: | ---: | ---: | ---: | --- |
| Erangel TPP squad | 11,481,303 | 6,205 | `11.85` | `44.81` | no |
| Taego TPP squad | 6,970,586 | 4,160 | `24.12` | `55.98` | no |
| Miramar / Desert TPP squad | 2,767,057 | 1,545 | `88.51` | `60.88` | no |
| Rondo TPP squad | 1,957,833 | 1,041 | `15.81` | `50.65` | no |
| Sanhok TPP squad | 1,370,326 | 940 | `13.24` | `38.46` | no |
| Vikendi TPP squad | 1,005,182 | 498 | `50.13` | `42.03` | no |
| Paramo TPP squad | 315,473 | 333 | `-90.62` | `40.00` | no |
| Karakin TPP squad | 289,180 | 333 | `19.59` | `29.48` | no |
| Erangel TPP solo | 4,634,937 | 800 | `11.47` | `32.40` | no |
| Erangel TPP duo | 2,024,760 | 656 | `12.17` | `48.27` | no |
| Taego TPP duo | 1,648,145 | 573 | `23.52` | `48.81` | no |
| Taego FPP squad | 811,681 | 435 | `23.65` | `63.39` | no |
| Erangel FPP squad | 737,538 | 430 | `12.45` | `54.60` | no |

Interpretation:

```text
The data source exists.  The missing layer is phase-level feature construction
and exposure to the V3 feature preset surface.
```

## Implementation Implication

The corrected O6 experiment must implement a new phase-level Z feature source
before any long model run.

Minimum ETL requirements:

| Block | ETL requirement |
| --- | --- |
| `Z1` | aggregate `team_snapshot_features.pos_z` into phase-level team vertical summaries |
| `Z2` | aggregate contact/proximity-gated pairwise `delta_z` into phase-level team relative-height summaries |
| `Z3` | interact Z1/Z2 summaries with existing zone/habitat pressure fields |

Candidate storage options:

| Option | Use |
| --- | --- |
| new table `stdb.team_phase_z_affordance_features` | cleanest ownership and easiest V2-style join |
| extend `stdb.team_phase_habitat_features` | not preferred because habitat is owned by OM3 |
| extend `v_team_phase_feature_long` directly | possible but mixes base and sensitivity concerns |

Preferred option:

```text
Create `stdb.team_phase_z_affordance_features` and add its normalized specs to
`stdb.queries.PHASE_NODE_FEATURE_SPECS_V2` or an equivalent V3 extension path.
```

## Claim Boundary

Allowed:

```text
The corrected O6 Z-ladder cannot yet be run because Z1-Z3 are absent from the
phase tensor, but the underlying vertical source signal exists in all required
cohorts.
```

Avoid:

```text
PUBG has no usable vertical information.
```

Avoid:

```text
The weak old OM6 result proves O6 is not useful.
```

The current result says only that old OM6 did not test the intended Z-ladder.

## Next Step

Implement the phase-level Z-affordance ETL and tensor mapping, then add:

```text
om6z_spatial_affordance
```

as:

```text
OM5 relation structure + Z1-Z3
```

After that, train only the new `om6z_spatial_affordance` preset for each exact
cohort using the locked `br_om_ladder_required_v1` split.
