# Battle Royale RQ2 PUBG Representation Object

Date: 2026-07-06
Package: research_framing
Lifecycle: active
Responsibility: Define the RQ2 claim, evidence objects, experiments, results, and boundaries.
Depends on: docs/data_layer/PUBG_OFFICIAL_TELEMETRY_OBSERVABILITY_TABLE_2026-07-05.md
Used by: docs/research_framing/BATTLE_ROYALE_RQ_OBJECT_CHAIN_2026-07-06.md

## RQObject

Object id:

```text
RQ2PUBGRepresentation
```

Research question:

```text
How can the Battle Royale formalization be represented in the representative
Battle Royale game PUBG?
```

Role in the paper:

```text
Show that the RQ1 formalization can be instantiated as a measurable,
leakage-guarded official-telemetry representation.
```

## Main Claim

```text
PUBG official telemetry can instantiate the Battle Royale formalization as
leakage-guarded team x phase-cut feature blocks and 10-second snapshot
representations across required map / perspective / team-size cohorts.
```

Manuscript-safe wording:

```text
PUBG is a label-rich supervised instantiation of the Battle Royale
formalization, while its geometry and terrain affordance observations are
explicitly bounded.
```

## Claim Objects

| Claim object | Claim | Evidence object | Result | Status |
| --- | --- | --- | --- | --- |
| `RQ2.C1.ObservabilityContract` | PUBG can instantiate the formalization only if observed, derived, proxy, label, and missing fields are separated. | `PUBGObservabilityTable` | Direct telemetry, derived pre-cut features, proxy-only constructs, label-only outcomes, and missing geometry are separated. | supported |
| `RQ2.C2.ExecutableFeatureSurface` | The RQ1 objects can be represented as executable team-phase and snapshot features. | `OperationalManifest`, `FeatureAvailabilityAudit` | OM0-OM6 features and labels are available for required cohorts. | supported |
| `RQ2.C3.CohortContract` | Claims should not pool incompatible game conditions. | `RequiredCohortSet` | Required cohorts are stratified by map, perspective, and team size. | supported |
| `RQ2.C4.LeakageGuard` | Representation must exclude forbidden future/outcome inputs. | `FeatureAvailabilityAudit`, `SnapshotDataContract` | target-defined rows and causal feature clocks are checked. | supported |
| `RQ2.C5.RepresentationPredictiveRelevance` | The representation is not merely descriptive; mechanism-aligned feature blocks matter for phase survival. | `OMLadderResult`, `PhaseStratifiedResult`, `BootstrapResult` | pressure-habitat gains are robust across required cohorts and phase controls. | supported |
| `RQ2.C6.GeometryBoundary` | PUBG official telemetry is label-rich but geometry-limited. | `PUBGObservabilityTable`, `O6ZResult`, `WarzoneBridge` | true terrain mesh, cover, LOS, and building occlusion are not directly provided by PUBG telemetry. | supported boundary |

## Experiment Objects

| Experiment object | Question | Method | Evidence produced |
| --- | --- | --- | --- |
| `RQ2.E1.PUBGObservabilityAudit` | Which formal objects are directly observed, derived, proxied, label-only, or missing in PUBG? | Build official-telemetry observability table. | `PUBG_OFFICIAL_TELEMETRY_OBSERVABILITY_TABLE_2026-07-05.md` |
| `RQ2.E2.FeatureAvailabilityAudit` | Can the representation be executed for all required cohorts without leakage? | Check feature availability, forbidden inputs, target-defined rows, and cohort readiness. | `BATTLE_ROYALE_FEATURE_AVAILABILITY_AUDIT_2026-07-03.md` |
| `RQ2.E3.RequiredCohortContract` | What cohort strata are used for manuscript claims? | Lock map / perspective / team-size cohort set. | `BATTLE_ROYALE_REQUIRED_COHORT_SET_2026-07-03.md` |
| `RQ2.E4.OMLadderValidation` | Do ontology-derived feature blocks matter for supervised phase survival? | Run OM0-OM6 feature ladder across required cohorts. | `BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md` |
| `RQ2.E5.PhaseAndUncertaintyControls` | Is the central mechanism only pooled phase base-rate encoding? | Run phase-stratified analysis and paired match-cluster bootstrap. | `BATTLE_ROYALE_PHASE_STRATIFIED_OM_RESULT_2026-07-04.md`, `BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md` |
| `RQ2.E6.O6GeometryBoundary` | Can PUBG official telemetry validate fine terrain / vertical affordance? | Run corrected O6Z vertical-proxy sensitivity and document Warzone/Caldera observability boundary. | `BATTLE_ROYALE_O6_Z_LADDER_RESULT_2026-07-05.md`, `PUBG_WARZONE_OBSERVABILITY_GAP_BRIDGE_2026-07-05.md` |

## Evidence Objects

| Evidence object | Path | What it supports |
| --- | --- | --- |
| `PUBGObservabilityTable` | `docs/data_layer/PUBG_OFFICIAL_TELEMETRY_OBSERVABILITY_TABLE_2026-07-05.md` | direct / derived / proxy / label / missing field boundary |
| `OperationalManifest` | `docs/feature_engineering/PUBG_FEATURE_OPERATIONAL_MANIFEST_2026-07-02.md` | formal family to PUBG feature-block mapping |
| `FeatureAvailabilityAudit` | `docs/experiment_design/BATTLE_ROYALE_FEATURE_AVAILABILITY_AUDIT_2026-07-03.md` | executable feature and leakage readiness |
| `RequiredCohortSet` | `docs/experiment_design/BATTLE_ROYALE_REQUIRED_COHORT_SET_2026-07-03.md` | cohort definitions and split discipline |
| `OMLadderResult` | `docs/experiment_design/BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md` | representation block predictive validation |
| `PhaseStratifiedResult` | `docs/experiment_design/BATTLE_ROYALE_PHASE_STRATIFIED_OM_RESULT_2026-07-04.md` | within-phase mechanism check |
| `BootstrapResult` | `docs/experiment_design/BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md` | match-cluster uncertainty support |
| `O6ZResult` | `docs/experiment_design/BATTLE_ROYALE_O6_Z_LADDER_RESULT_2026-07-05.md` | bounded negative terrain-proxy result |
| `WarzoneBridge` | `docs/spatial_world_modeling/PUBG_WARZONE_OBSERVABILITY_GAP_BRIDGE_2026-07-05.md` | supplementary geometry-observability boundary |

## Result Objects

### RQ2.R1 Formal-To-PUBG Mapping

| Formal object | Ontology | PUBG representation | Manuscript role |
| --- | --- | --- | --- |
| `U_m(k)` | O0 | map, mode, team size, phase, elapsed time | context |
| `C_i(k)` | O1 | alive roster, health, damage-derived survival capital | team state |
| `Z_m(k)` | O2 | safe-zone / blue-zone schedule and alive population pressure | pressure schedule |
| `B_i(k)` | O3 | movement, rotation, speed, zone entry timing | behavioral response |
| `H_i(k)` | O2 x O6 | pressure-conditioned habitat, zone gap, crowding | core habitat construct |
| `E_i(k)` | O4 | damage, combat, contact, nearby combat exposure | threat exposure |
| `R_i(k)` | O5 | resources, vehicles, mobility/support proxies | support state |
| `Q_i(k)` | O7 | relation summaries and supplementary edge streams | relation field |
| `Y_i(k+1)` | O8 | next-phase hazard, rank, top-k, winner labels | target |

### RQ2.R2 Central Representation Result

Current central supervised mechanism:

```text
OM2 -> OM3 pressure-habitat improves hazard AUC in 13/13 required cohorts.
```

Manuscript interpretation:

```text
The Battle Royale pressure-habitat representation is an executable and
predictively relevant projection of the RQ1 formalization into PUBG telemetry.
```

### RQ2.R3 Geometry Boundary

PUBG official telemetry supports supervised phase-survival prediction, but it
does not directly provide:

```text
terrain mesh
building occlusion
true line of sight
cover surfaces
full traversable geometry
```

Use Warzone / Caldera only as supplementary geometry-rich construct evidence,
not as supervised PUBG survival validation.

## Boundary Object

Allowed:

```text
PUBG official telemetry instantiates the Battle Royale formalization as a
leakage-guarded, label-rich supervised team-phase and snapshot representation.
```

Forbidden:

```text
PUBG official telemetry fully observes every formal Battle Royale construct.
```

Forbidden:

```text
PUBG official telemetry validates fine terrain, cover, LOS, or building
occlusion effects.
```

## Interface To RQ3

RQ2 exports the following representation objects to RQ3:

```text
required cohorts
team-phase labels
official raw / prior-style baselines
feature-block representations
10-second snapshot tensor surface
leakage and observability constraints
```

RQ3 must test whether these representations can be used for prior-baseline
comparison and realtime elimination-risk readout.
