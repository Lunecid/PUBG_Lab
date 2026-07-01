# ST Feature-Set Result Note (2026-06-23)

## Purpose

This note interprets the map/user-version channel matrix through prior-work
feature-set concepts, not through implementation channel names.

Companion result:

```text
docs/PAPER_ST_DOMAIN_CHANNEL_RESULT_2026-06-23.md
```

Primary artifacts:

```text
model_stgcn/checkpoints/snapshot_domain_channel_primary_tpp_mps300_e1_20260623/domain_channel_metrics_seed42.csv
model_stgcn/checkpoints/snapshot_domain_channel_fpp_aux_mps250_e1_20260623/domain_channel_metrics_seed42.csv
```

## Feature-Set Interpretation Table

| Literature feature-set concept | Battleground/PUBG meaning | Why this is an ST feature set | Fast-pilot result | Interpretation |
| --- | --- | --- | --- | --- |
| Time-varying survival covariates / behavioral telemetry state | Team survival capital: current health, minimum health, alive roster count, and mobility resource | These variables change over game time and describe the team's at-risk survival capacity before a phase cut | Best in all 9 tested map/user-version domains; mean test loss 0.5942 | Dominant short-horizon survival signal. The model first needs to know whether the team still has enough roster/health/mobility capital to survive the next phase |
| Trajectory state / moving-object history | Team movement history: team centroid and speed over the 10-second clock | A team is a moving survival actor; trajectory features describe where it moved and how quickly before the prediction cut | Second-best feature set; mean test loss 0.6178 | Movement is consistently informative, but less immediate than survival capital in this one-epoch triage |
| Multi-agent local interaction field | Local opponent field: nearby teams and spatial exposure relations | The risk around a team depends on other teams occupying nearby or conflicting spaces | Relation variants cluster around 0.6203-0.6206 mean test loss | Local interaction has signal, but it does not dominate the current team-state survival-capital signal |
| Sparse event-interaction field | Combat pressure: directed damage/contact event relations before the cut | Combat events are time-windowed interactions, not post-game totals | Combat-only relation mean test loss 0.6214 | Combat relations are slightly weaker than proximity/local-field relations in this fast run |
| Typed competitive topology | Observed graph structure of proximity/combat relations | ST graph learning asks whether actual edge topology adds signal beyond model capacity or shuffled/contact controls | Observed relation, edge-shuffle, and contact-gate controls are tightly tied | No observed-topology claim is supported at this pilot scale |
| Dynamic spatial constraint / external spatial context | Shrinking safe-zone pressure: position relative to safe-zone center/boundary/target zone | Battle Royale space is dynamic; spatial meaning changes as the safe zone shrinks and moves | Weakest standalone set; mean test loss 0.6411 | Zone pressure is not a strong standalone early signal here; it may require phase context, interaction with trajectory, or longer/full-feature training |
| Mixed high-dimensional node state | All available snapshot team-state families together | Combines trajectory, zone, viability, and resource channels in one model input | Mean test loss 0.6362 in the one-epoch run | One epoch is probably insufficient for the full mixed feature set; this result should not be read as evidence that full features are worse in general |
| Latest-state baseline | Latest causally available snapshot state without temporal modeling | This tests whether temporal sequence modeling adds value over a cut-state snapshot | Mean test loss 0.6311 | Temporal dynamics matter: trajectory and survival-capital sequences beat latest-state-only baseline |

## Map/User-Version Finding

The dominant result is not merely that one implementation variant won. The
reference-level statement is:

```text
Across primary TPP squad maps and auxiliary FPP squad maps, the strongest
feature set is time-varying team survival capital.
```

Domain winners:

| Map | User-version slice | Winning feature-set concept | Test loss |
| --- | --- | --- | ---: |
| `Baltic_Main` | TPP squad | time-varying team survival capital | 0.6038 |
| `Tiger_Main` | TPP squad | time-varying team survival capital | 0.5608 |
| `Desert_Main` | TPP squad | time-varying team survival capital | 0.5770 |
| `Neon_Main` | TPP squad | time-varying team survival capital | 0.5788 |
| `Savage_Main` | TPP squad | time-varying team survival capital | 0.5826 |
| `Baltic_Main` | FPP squad | time-varying team survival capital | 0.5874 |
| `Tiger_Main` | FPP squad | time-varying team survival capital | 0.6236 |
| `Desert_Main` | FPP squad | time-varying team survival capital | 0.6148 |
| `Neon_Main` | FPP squad | time-varying team survival capital | 0.6187 |

## Reference-Aligned Reading

### 1. Survival-capital features are the main Battleground signal.

In dynamic survival modeling, time-varying covariates describe the risk state
of an at-risk subject over time. In this task, the at-risk subject is the team,
and the most effective covariates are roster survival capital: health,
remaining players, and mobility resource.

Paper wording:

```text
The strongest feature family is the time-varying team survival-capital state,
suggesting that short-horizon Battleground elimination risk is primarily
explained by causally observed roster/health/mobility covariates.
```

### 2. Trajectory features are a stable secondary ST signal.

Trajectory-data literature treats movement as a clocked path object rather
than a single coordinate. Our result agrees with that framing: movement history
is consistently useful, but it is secondary to direct survival-capital state in
the fast pilot.

Paper wording:

```text
Team trajectory history provides a consistent secondary signal, supporting the
view that Battle Royale teams should be represented as moving survival actors
rather than static post-game rows.
```

### 3. Zone-only spatial context is not enough by itself.

Shrinking-zone geometry is the domain-specific spatial frame, but the fast
pilot suggests it is weak as a standalone masked feature set. This does not
invalidate the shrinking-arena feature design. It suggests zone pressure may
need to be modeled as an interaction with trajectory, phase horizon, opponent
density, or longer-trained full representations.

Paper wording:

```text
Safe-zone-relative features are conceptually central to the domain, but in the
one-epoch channel triage they do not form a strong standalone predictor. We
therefore treat shrinking-arena pressure as contextual and interactional rather
than as an isolated feature block.
```

### 4. Relation features are not yet topology evidence.

Multi-agent tracking and ST graph learning motivate local opponent fields and
typed relations. The corrected run shows relation features are competitive
with trajectory in some slices, but observed relation topology is not separated
from edge-shuffle or contact-gate controls.

Paper wording:

```text
Local interaction fields carry predictive information, but the present pilot
does not isolate a semantic observed-topology benefit beyond graph capacity or
contact scaffolding controls.
```

## Recommended Result Paragraph

Use this as the paper note:

```text
When the experiment is sliced by map and user-version, the strongest
reference-level feature set is not the graph topology but the time-varying
team survival-capital state. Across nine map/mode slices, the survival-capital
feature set wins every domain, followed by trajectory history. Local opponent
and combat relation fields are competitive but remain tied to graph-null and
contact-gate controls, so they should be interpreted as interaction-field
signals rather than evidence of semantic observed topology. Safe-zone-relative
features remain conceptually central to the Battle Royale domain, but this
pilot suggests they function better as contextual or interactional features
than as a standalone predictor.
```

## Implementation Alias For Reproducibility

The paper should use the feature-set names above. The implementation aliases
are kept only for reproducibility:

| Paper feature-set name | Implementation alias |
| --- | --- |
| time-varying team survival capital | `tcn_viability_resource` |
| team trajectory history | `tcn_trajectory` |
| dynamic safe-zone spatial context | `tcn_zone` |
| trajectory plus safe-zone context | `tcn_trajectory_zone` |
| mixed snapshot team-state features | `tcn_full` |
| local opponent field | `rgcn_proximity` |
| combat event-interaction field | `rgcn_combat` |
| typed observed relation field | `rgcn_full` |
| topology-null relation control | `rgcn_full_edge_shuffle` |
| contact-capacity relation control | `gate_only_full` |
| latest-state baseline | `last_full` |

## Sources

```text
Trajectory data management:
https://arxiv.org/abs/2003.11547

Team-sports spatio-temporal analysis:
https://arxiv.org/abs/1602.06994

ST forecasting with external context:
https://arxiv.org/abs/1701.02543

Spatio-temporal graph convolution:
https://arxiv.org/abs/1709.04875

Behavioral telemetry:
https://arxiv.org/abs/1407.3950

Dynamic survival modeling:
https://arxiv.org/abs/1811.10746
https://arxiv.org/abs/2007.13218
```
