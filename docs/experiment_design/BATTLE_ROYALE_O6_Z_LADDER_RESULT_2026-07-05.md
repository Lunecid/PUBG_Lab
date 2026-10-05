# Battle Royale O6 Z-Ladder Sensitivity Result

Date: 2026-07-05
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This document records the corrected O6 sensitivity experiment:

```text
Does PUBG telemetry-derived vertical position add predictive signal after
OM5 relation-structure features when represented as the intended Z1-Z3
spatial-affordance proxy block?
```

The experiment follows:

```text
docs/experiment_design/BATTLE_ROYALE_O6_Z_LADDER_SENSITIVITY_PROTOCOL_2026-07-04.md
```

## Short Answer

The result is:

```text
WEAK_OR_NULL_O6Z_SUPPORT_IN_PUBG
```

The corrected O6 Z-ladder ran successfully across all 13 locked map /
perspective / team-size cohorts, but it did not add robust predictive signal
after OM5.

This does not weaken the central pressure-habitat result.  It bounds what can
be claimed from PUBG alone:

```text
PUBG strongly supports the pressure-habitat mechanism, but its basic telemetry
does not validate fine-grained vertical or terrain affordance beyond OM5.
```

The result strengthens the need to present Caldera/Warzone as construct
validation for geometry-rich spatial affordance rather than as supervised
outcome prediction.

## Implemented Feature Block

The corrected preset was:

```text
om6z_spatial_affordance = OM5 relation structure + Z1-Z3
```

It intentionally excludes the old O6 proxy variables:

```text
avg_enemy_density_100m_norm
min_nearest_enemy_norm
path_near_enemy_ratio
```

Those old variables overlap with O4 exposure and O7 relation-density
projections.

The new Z-ladder features are:

| Block | Feature | Meaning |
| --- | --- | --- |
| `Z1` | `team_z_mean_norm` | team mean vertical position in the phase |
| `Z1` | `team_z_percentile_alive` | vertical percentile among alive teams in the same match-phase |
| `Z1` | `team_z_range_norm` | within-team vertical spread |
| `Z2` | `contact_delta_z_mean_norm` | mean relative height against proximity-contact enemies |
| `Z2` | `contact_delta_z_positive_ratio` | share of contact samples where the team is above the contacted enemy |
| `Z2` | `high_ground_contact_advantage` | contact-gated high-ground advantage score |
| `Z3` | `high_ground_x_boundary_pressure` | high-ground advantage under boundary pressure |
| `Z3` | `z_percentile_x_hab_crowding_pressure` | vertical percentile under habitat crowding pressure |
| `Z3` | `contact_delta_z_x_outside_safe_ratio` | relative height under unsafe-zone exposure |

These are PUBG `pos_z` telemetry proxies.  They are not true terrain slope,
cover, building occlusion, line of sight, or mesh-derived spatial affordance.

## Cohort Protocol

No pooled training was used.  One `om6z_spatial_affordance` model was trained
for each locked cohort using the original OM ladder split:

| Cohort | Status |
| --- | --- |
| Erangel TPP squad | complete |
| Taego TPP squad | complete |
| Miramar / Desert TPP squad | complete |
| Rondo TPP squad | complete |
| Sanhok TPP squad | complete |
| Vikendi TPP squad | complete |
| Paramo TPP squad | complete |
| Karakin TPP squad | complete |
| Erangel TPP solo | complete |
| Erangel TPP duo | complete |
| Taego TPP duo | complete |
| Taego FPP squad | complete |
| Erangel FPP squad | complete |

## Artifacts

| Artifact | Role |
| --- | --- |
| `stdb/z_affordance_feature_etl.py` | ETL for `stdb.team_phase_z_affordance_features` |
| `tools/run_o6z_required_cohort_etl.py` | required-cohort ETL runner |
| `tools/audit_o6_z_ladder_availability.py` | pre/post ETL availability audit |
| `scripts/run_br_om6z_required_cohorts.sh` | required-cohort OM6Z runner |
| `model_stgcn/checkpoints/br_om6z_required_v1/om6z_metrics_all_cohorts.csv` | 13-cohort OM6Z metrics |
| `model_stgcn/checkpoints/br_om6z_required_v1/om6z_metrics_summary.json` | collected OM6Z summary |
| `model_stgcn/checkpoints/br_om6z_required_v1/om6z_vs_om5_om6_comparison.csv` | OM5 / old OM6 / OM6Z comparison |
| `model_stgcn/checkpoints/br_om6z_required_v1/om6z_vs_om5_om6_summary.json` | aggregate comparison summary |
| `model_stgcn/checkpoints/br_om6z_required_v1/om6z_phase_deltas_vs_om5_om6.csv` | phase-level deltas |
| `model_stgcn/checkpoints/br_om6z_required_v1/om6z_phase_delta_summary.json` | phase-group summary |

## Overall Result

Sequential increment after OM5:

| Comparison | Metric | Median delta | Positive cohorts | Verdict |
| --- | --- | ---: | ---: | --- |
| `OM5 -> old OM6` | hazard AUC | `+0.0010` | `10/13` | tiny old proxy gain |
| `OM5 -> OM6Z` | hazard AUC | `-0.0012` | `4/13` | no support |
| `old OM6 -> OM6Z` | hazard AUC | `-0.0020` | `2/13` | corrected Z block is weaker overall |
| `OM5 -> old OM6` | rank Spearman | `-0.0014` | `3/13` | no support |
| `OM5 -> OM6Z` | rank Spearman | `-0.0034` | `2/13` | no support |
| `old OM6 -> OM6Z` | rank Spearman | `-0.0011` | `4/13` | no support |
| `OM5 -> old OM6` | top-10 AUC | `-0.0008` | `3/13` | no support |
| `OM5 -> OM6Z` | top-10 AUC | `-0.0007` | `5/13` | no support |
| `old OM6 -> OM6Z` | top-10 AUC | `+0.0005` | `8/13` | negligible |

Calibration did not provide a compensating result:

```text
OM5 -> OM6Z ECE median delta = -0.0004, positive cohorts = 5/13.
```

## Confirmatory Phase-Group Result

The protocol pre-registered early `1-2` and mid `3-4` hazard AUC and rank
Spearman as primary confirmatory targets.

`OM5 -> OM6Z` cohort-mean phase-group deltas:

| Metric | Phase group | Median delta | Positive cohorts | Verdict |
| --- | --- | ---: | ---: | --- |
| hazard AUC | early `1-2` | `+0.0004` | `7/13` | weak trace only |
| hazard AUC | mid `3-4` | `-0.0036` | `1/13` | no support |
| hazard AUC | late `5-6` | `-0.0030` | `3/13` | exploratory, no support |
| rank Spearman | early `1-2` | `-0.0039` | `2/13` | no support |
| rank Spearman | mid `3-4` | `-0.0132` | `2/13` | no support |
| rank Spearman | late `5-6` | `-0.0319` | `4/13` | exploratory, no support |
| top-10 AUC | early `1-2` | `-0.0001` | `6/13` | no support |
| top-10 AUC | mid `3-4` | `-0.0007` | `3/11` | no support |
| top-10 AUC | late `5-6` | `+0.0003` | `4/7` | exploratory, negligible |

The only directionally positive primary cell is early hazard AUC, and it is far
below the pre-registered support threshold:

```text
positive deltas in at least 9/13 cohorts
```

Therefore no phase-group bootstrap is currently warranted for a positive O6Z
claim.  A bootstrap could still be run later as a negative-result robustness
check, but it is no longer on the critical path.

## Interpretation

Allowed claim:

```text
In PUBG, vertical-position telemetry can be operationalized as an O6 proxy, but
the Z1-Z3 block does not robustly improve next-phase survival or rank prediction
after pressure-habitat, threat exposure, and relation-structure features.
```

Allowed paper framing:

```text
This bounds PUBG as label-rich but geometry-poor.  PUBG validates the supervised
pressure-habitat mechanism, while geometry-rich datasets such as Caldera are
needed to demonstrate what finer spatial-affordance construction looks like.
```

Avoid:

```text
O6 spatial affordance is useless.
```

Avoid:

```text
High ground has no tactical value in Battle Royale.
```

Avoid:

```text
PUBG validates terrain affordance.
```

The correct boundary is narrower:

```text
PUBG `pos_z`-only proxies are not enough to validate fine-grained terrain or
high-ground affordance after OM5.
```

## Next Methodological Step

The critical path should return to the RQ1-RQ4 manuscript spine:

1. preserve the pressure-habitat result as the central PUBG supervised
   prediction evidence;
2. report O6Z as a bounded negative sensitivity;
3. use Caldera/Warzone for geometry-rich construct validation rather than
   supervised outcome prediction;
4. keep O7 edge-stream work, if cited, as supplementary null-control evidence.
