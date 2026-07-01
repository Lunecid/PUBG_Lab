# Input And Model Channel Contract (2026-06-22)

This document is the first design contract to settle before additional model work.

Goal:

> Separate the PUBG survival-prediction problem into explicit input channels, model channels, heads, metrics, preprocessing contracts, and tuning contracts.

If this contract is unclear, model comparisons are not interpretable.

Graph modeling foundation:

```text
docs/GRAPH_DATA_MODELING_FOUNDATION_2026-06-22.md
```

Domain design extension:

```text
docs/PUBG_DOMAIN_DESIGN_LAYER_2026-06-22.md
```

The channel contract defines tensor/input roles. The domain design layer defines
what PUBG survival mechanism each feature, edge, module, ablation, and slide
element represents.

---

## 1. Core Principle

Do not start from "one feature tensor".

Do not start from semantic feature names alone either. The first split must be
the feature's natural update clock.

The project must define channels by:

1. population/domain;
2. feature clock / temporal resolution;
3. spatial meaning;
4. team-internal state;
5. team-to-team relation;
6. global match/phase context;
7. label/head target.

The same channel taxonomy must be used for:

- preprocessing;
- tabular baselines;
- graph models;
- temporal models;
- spatio-temporal graph models;
- ablation tables;
- presentation diagrams.

Clock-aware extension:

```text
docs/FEATURE_CLOCK_CHANNEL_REDESIGN_2026-06-22.md
```

Machine-readable manifest seed:

```text
docs/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv
```

Design rule:

> Every feature must declare both `clock` and `semantic_domain`. Snapshot and
> phase versions of the same concept are different channels, not interchangeable
> variants of one feature group.

---

## 2. Prediction Unit

The canonical prediction unit is:

```text
at-risk team x phase cut
```

The model predicts:

```text
Given information observed up to phase k,
predict whether team i is eliminated before phase k+1.
```

Important:

- Node is team, not player.
- Player telemetry is aggregated into team-level features.
- Team size differences are input features, not separate node counts.
- Main outputs are phase-cut outputs, not every-10-second outputs.

---

## 3. Population And Domain Channels

These are not ordinary node features. They define the dataset slice and domain.

| Channel | Values / Examples | Role |
| --- | --- | --- |
| mode | solo, duo, squad | population structure and label semantics |
| perspective | FPP, TPP, mixed/current | domain split or covariate |
| map | Erangel/Baltic, Miramar/Desert, Taego/Tiger, etc. | geography/domain split |
| team size | 1, 2, 4 or observed roster size | roster context |
| skill/lobby tier | official ranked tier, MMR bucket, or pre-match proxy tier | competition-level split or covariate |
| match id | unique match identifier | split unit; never split rows from same match across train/test |
| time/version | patch or collection period if available | domain shift analysis |

Design rule:

> Mode/map/perspective should be explicit dataset channels or stratification variables. They should not be silently mixed without reporting.

Immediate tasks:

- [ ] Define official map/mode/perspective splits.
- [ ] Decide whether the main result is squad-only or all modes.
- [ ] Report if solo/duo/squad are separate experiments or domain-generalization tests.
- [ ] Audit whether official ranked tier/MMR exists in raw data; otherwise define a leakage-safe proxy tier.

---

## 4. Raw-To-Model Unit Hierarchy

| Level | Example | Is it a model node? | Use |
| --- | --- | --- | --- |
| raw player event | damage, kill, movement, item use | no | preprocessing source |
| player snapshot | player position/health at time t | no | aggregated into team state |
| team snapshot | team centroid/state every 10 s | yes, snapshot lane | `X_s` |
| team phase | team aggregate per phase | yes, phase lane | `X_node` |
| team pair | team i to team j relation | edge, not node | `A_s`, `A`, advantage edges |
| phase/global | safe-zone and match context | global context | `G` |
| label | elimination/rank/top-k | no | target/head |

Design rule:

> Individual/player information may be used, but only after it is explicitly aggregated into a team-level channel unless a future player-node model is created.

---

## 5. Canonical Input Channels

### Required Feature Clock Metadata

Every feature manifest row must include:

| Field | Allowed examples | Meaning |
| --- | --- | --- |
| `clock` | `match_static`, `snapshot_10s`, `phase_cut`, `phase_window`, `label` | when the value updates |
| `entity` | `team`, `pair`, `global`, `label` | what object owns the value |
| `semantic_domain` | `roster`, `movement`, `zone`, `combat`, `resource`, `relation`, `strategy`, `global` | gameplay meaning |
| `causal_window` | `instant`, `trailing_10s`, `trailing_30s`, `phase_to_date`, `phase_window`, `match_static` | what observations are allowed |
| `model_stream` | `snapshot_node`, `snapshot_edge`, `phase_node`, `phase_edge`, `global_context`, `label` | where the model consumes it |

Design rule:

> Do not group a feature by semantic name alone. `combat_last_10s` and
> `combat_phase_total` are different clock channels even though both are combat
> features.

### C0. Domain / Cohort Channel

Purpose:

> Tell the experiment what population this row/match belongs to.

Examples:

```text
map_id
mode
perspective
team_size
official_ranked_tier
lobby_skill_tier
tier_source
match_duration
collection_period
```

Use:

- stratified reporting;
- domain ablation;
- tier-aware training/evaluation;
- optional model conditioning.

Do not use this channel to leak match outcome or final rank. If official
ranked/MMR tier is unavailable, any proxy tier must be computed from pre-match
or historical information only, not from the current match result.

### C1. Snapshot Team-State Channel

Tensor:

```text
X_s: (B, T_s, N, 13)
```

Resolution:

```text
10-second snapshot
```

Current features:

| Subchannel | Features | Meaning |
| --- | --- | --- |
| snapshot map position | `pos_x`, `pos_y`, `pos_z` | where the team is |
| snapshot zone-relative position | `rel_zone_x`, `rel_zone_y`, `dist_blue_boundary_m`, `inside_blue`, `dist_target_center_m` | where the team is relative to shrinking zone |
| snapshot movement | `speed_mps` | short-term mobility |
| snapshot team-internal state | `mean_health`, `min_health`, `alive_n`, `vehicle_ratio` | roster health, surviving members, vehicle use |

Design status:

- This is the minimum-resolution temporal input.
- In the corrected design, this channel must be the primary input.
- In the legacy V3/MR design, this channel was encoded then mean-pooled to phase.

### C2. Snapshot Team-Relation Channel

Tensor:

```text
A_s: (B, T_s, E_s, N, N)
```

Current relations:

| Relation | Meaning | Spatial? |
| --- | --- | --- |
| proximity | teams close in map space at snapshot time | yes |
| combat | recent combat interaction in the past 10-second window | partially; tactical/contact relation |

Future relations:

| Relation | Meaning |
| --- | --- |
| contact-gated relative advantage | pairwise `delta z`, `delta alive`, `delta health`, `delta zone` under contact gate |

Design rule:

> Edge channels must be separated from node feature channels. Relation type and edge features answer a different question from team state features.

### C3. Phase Team-State Channel

Tensor:

```text
X_node: (B, 8, N, 116)
```

Resolution:

```text
phase slot / phase cut
```

Subchannels:

| Subchannel | Examples | Meaning |
| --- | --- | --- |
| roster/team status | `team_size`, `alive_players_start/end/min/avg`, health statistics | team-internal survival state |
| movement/trajectory | path length, displacement, sinuosity, average speed, rotation timing | phase-level mobility |
| zone/spatial exposure | boundary distance, outside safe ratio, enemy density, nearest enemy | spatial pressure |
| combat aggregate | damage dealt/taken, kills, groggy, opponents damaged | phase combat behavior |
| loot/readiness | pickup/equip/use counts, ammo/armor/heal/boost/carepackage | resource state |
| network role | combat/proximity/co-pressure degree/strength, cumulative relations | graph-derived role features |
| habitat pressure | safe-area pressure, crowding, local enemy density, safe-entry congestion | shrinking-space pressure |

Design rule:

> Phase features are allowed for legacy and baseline models, but snapshot-first models must mark them as auxiliary/context, not primary minimum-resolution input.

### C4. Phase Global Context Channel

Tensor:

```text
G: (B, 8, 25)
```

Examples:

| Subchannel | Features | Meaning |
| --- | --- | --- |
| phase clock | phase index, phase duration | where the match is in phase time |
| alive population | alive teams, alive players | remaining competition |
| zone geometry | safe radius, poison radius, center shift, zone gap | global shrinking-zone condition |
| habitat global | safe area, area loss, shrink rate, crowding pressure | match-level spatial pressure |

Design rule:

> Global features are not team-specific. They should be fused or conditioned separately, not repeated silently as node attributes without documentation.

### C5. Phase Team-Relation Channel

Tensor:

```text
A: (B, 8, E_p, N, N)
```

Current relations:

| Relation | Meaning |
| --- | --- |
| proximity | phase-level spatial closeness |
| combat | phase-level combat relation |
| co-pressure | shared zone/habitat pressure |

Optional directed variant:

```text
proximity, combat, co-pressure, combat_in
```

Design rule:

> GCN may combine relation channels; R-GCN must preserve relation channels. Experiments must say which relation treatment was used.

### C6. Individual-Derived Team Aggregate Channel

This channel makes the player/team distinction explicit.

Raw player information can contribute to team features such as:

```text
alive_n
mean_health
min_health
team_spread_avg
team_spread_max
vehicle_ratio
damage/kills/assists aggregated by team
item counts aggregated by team
```

Design rule:

> These are not player nodes. They are team node attributes derived from individual player rows/events.

### C7. Prior-Work Strategy Channel

This channel is for Sagong-style baseline experiments.

Team-level adaptation:

| Group | Features |
| --- | --- |
| Aggressiveness | phase attack count, phase kill count, phase total damage, phase assist count |
| Activeness | phase distance, phase max density, phase item count, carepackage use |
| Scores | FightScore, ActiveScore |
| Strategy type | Hunter, Sniper, Traveler, Camper |
| Prior survival rate | phase/type survival rate from training data only |

Design rule:

> This channel is a prior-work-inspired benchmark channel, not the full STDB representation.

### C8. Label / Mask Channel

Labels:

```text
y_hazard:    (B, 7, N)
y_rank:      (B, N)
y_topk:      (B, N, 2)
target_mask: (B, 7, N)
node_mask:   (B, 8, N)
node_mask_s: (B, T_s, N)
phase_mask:  (B, 8)
```

Design rule:

> Labels and masks are channels, but they are not input features. They must never enter encoders.

---

## 6. Channel Grouping By Concept

| Concept | Channels | Question answered |
| --- | --- | --- |
| time | C1, C3, C4 | how team/match state evolves |
| space | C1, C2, C3, C4, C5 | where teams are and how zone pressure changes |
| situation | C3, C4, C7 | phase context, survival pressure, strategy type |
| team | C1, C3, C6 | team state and roster condition |
| individual-derived | C6 | player-level telemetry aggregated to team |
| relation | C2, C5, future advantage edges | how teams affect each other |
| outcome | C8 | what is predicted and evaluated |

This table should appear in the problem/method design notes.

Domain overlay:

| Domain mechanism | Main channels | Meaning |
| --- | --- | --- |
| D0 population rule-set | C0 | map/mode/perspective/team-size context |
| D1 roster survival capital | C1, C3, C6 | alive count, health, team capacity |
| D2 rotation tempo | C1, C3 | movement, path, zone entry, vehicle use |
| D3 zone / habitat pressure | C1, C3, C4 | shrinking-space pressure and safe-zone migration |
| D4 local opponent field | C2, C3, C5 | enemy density, nearest enemy, proximity exposure |
| D5 combat exchange | C2, C3, C5 | outgoing/incoming damage, kills, combat direction |
| D6 resource readiness | C3, C6 | weapon/ammo/armor/heal/boost/loot readiness |
| D7 strategic behavior | C7 | FightScore, ActiveScore, strategy type baseline |
| D8 relation topology | C2, C5 | whether edges carry information beyond smoothing |
| D9 phase horizon | C4, C8 | phase-conditioned risk and output horizon |
| D10 skill / lobby strength | C0 | official rank/MMR tier or leakage-safe lobby-strength proxy |

Design rule:

> Every feature and ablation should map to both a C-channel and a D-domain mechanism.

---

## 7. Model Channel Families

Every model must declare which input channels it consumes.

### M0. Tabular Phase Baseline

Input:

```text
C3 phase team-state
C4 global context, optionally broadcast or joined
C7 prior-work strategy channel when testing Sagong-style baseline
```

No graph. No snapshot sequence.

Examples:

- Logistic Regression;
- Random Forest;
- XGBoost / HistGradientBoosting;
- MLP.

Purpose:

> Establish feature-only and prior-work baselines.

### M1. Temporal Tabular Model

Input:

```text
phase sequence of C3/C4/C7
```

No explicit graph.

Examples:

- Phase TCN;
- GRU/LSTM;
- Transformer encoder over phases.

Purpose:

> Test whether temporal sequence modeling helps without graph topology.

### M2. Snapshot Temporal Model

Input:

```text
C1 snapshot team-state sequence
optional C4 phase/global context at cut
```

No graph relations.

Examples:

- snapshot TCN without `A_s`;
- GRU/Transformer over 10-second snapshots;
- last-snapshot-only baseline.

Purpose:

> Test the value of high-resolution temporal state without relational message passing.

### M3. Phase Graph Model

Input:

```text
C3 phase team-state
C4 phase global context
C5 phase graph relations
```

Temporal resolution:

```text
8 phase slots
```

Examples:

- GCN;
- GraphSAGE;
- GAT;
- R-GCN.

Purpose:

> Test phase-level graph relation effects and graph null controls.

### M4. Legacy Pooled-MR ST-GNN

Input:

```text
C1 + C2 snapshot lane
C3 + C4 + C5 phase lane
```

Flow:

```text
snapshot ST-GNN -> mean-pool to phase -> phase ST-GNN -> heads
```

Purpose:

> Preserve and interpret existing V3/MR results.

Important caveat:

> This is not a fully snapshot-resolution survival model.

### M5. Snapshot-First ST Survival Model

Input:

```text
C1 snapshot team-state
C2 snapshot team-relation
C8 labels/masks only for supervision
optional strictly causal context channels
```

Flow:

```text
10-second snapshot encoder -> phase-cut representation -> hazard/survival heads
```

Purpose:

> Target design: preserve minimum temporal resolution until phase-cut prediction.

### M6. Edge-Conditioned Relative-Advantage Model

Input:

```text
team node features + contact-gated pairwise relative edge features
```

Edge features:

```text
delta z
delta alive
delta health
delta zone pressure
```

Purpose:

> Test whether relation type creates genuine topology signal beyond smoothing.

Status:

> Preliminary/future-work phase-tier result, not current main result.

---

## 8. Relation Handling Contract

### GCN

GCN receives relation channels but combines them into a single graph.

```text
A_mix = sum or masked combination of relation channels
```

Meaning:

> connected neighbor is treated as one homogeneous relation.

### R-GCN

R-GCN preserves relation channels.

```text
message =
  W_proximity * A_proximity h
+ W_combat * A_combat h
+ W_copressure * A_copressure h
```

Meaning:

> why two teams are connected remains visible to the model.

### Edge-Conditioned GNN

Edge-conditioned GNN uses pairwise edge features.

```text
message weight = MLP(edge_feature_i_to_j)
```

Meaning:

> the edge is not only a relation label; it carries relative state.

---

## 9. Head Contract

### H1. Hazard Head

Target:

```text
next-phase elimination
```

Shape:

```text
(B, 7, N)
```

Primary metric:

- AUC;
- AP;
- BCE;
- Brier;
- ECE.

This is the main head.

### H2. Survival Probability Head

Target:

```text
cumulative survival probability or snapshot auxiliary survival
```

Use:

- explicit survival curve in snapshot-first model;
- auxiliary survival head in legacy MR.

Must state whether this is main or auxiliary.

### H3. Rank Head

Target:

```text
final placement/rank forecast at each cut
```

Metrics:

- Spearman rho;
- normalized rank MAE;
- pairwise rank accuracy.

### H4. Top-K Head

Target:

```text
final top-5 / top-10
```

Metrics:

- top-k AUC;
- top-k AP;
- top-k Brier.

### H5. Calibration / Decision Head

Optional reporting channel:

- ECE;
- reliability curves;
- phase-band calibration;
- decision threshold analysis.

---

## 10. Evaluation Contract

### Primary Unit

```text
at-risk team-phase cell
```

### Primary Metrics

| Task | Metrics |
| --- | --- |
| next-phase elimination | AUC, AP, BCE, Brier, ECE |
| final rank | Spearman rho, rank MAE, pairwise accuracy |
| top-k outcome | AUC, AP, Brier |
| calibration | ECE, reliability by phase |

### Required Slices

Report metrics by:

- phase/cut;
- map;
- mode/team size;
- graph condition;
- feature channel group;
- skill/lobby tier if available or derived;
- seed.

### Required Null Controls

For graph models:

- no_graph;
- edge_shuffle;
- node_shuffle;
- relation drop;
- time shuffle for temporal models.

Design rule:

> Graph improvement is not accepted as topology signal unless real/unshuffled graph beats null controls.

---

## 11. Preprocessing Contract

### P0. Raw Ingestion

Inputs:

- telemetry events;
- player positions;
- combat events;
- item events;
- phase/safe-zone events;
- team rosters and final outcomes.
- player/team ranked tier, MMR, or account-history metadata if available.

### P1. Team Aggregation

Player rows/events are aggregated into team-level features.

Required outputs:

- team snapshot features;
- team phase features;
- team-pair relations;
- labels and masks.

### P2. Snapshot Grid

Rules:

- fixed 10-second grid;
- causal/past-only combat attribution;
- no future events in snapshot features.

### P3. Phase Grid

Rules:

- fixed 8 phase slots for current V3;
- phase features must be computed only from information available up to that phase/cut;
- labels must be separated from inputs.

### P4. Graph Construction

Rules:

- relation type must be named;
- directionality must be documented;
- adjacency row-normalization must be explicit;
- edge controls must preserve the intended null condition.

### P5. Split

Rules:

- split by match, not by row;
- no same-match leakage across train/val/test;
- stratify or report map/mode/tier distributions.

### P6. Tier Construction

Rules:

- prefer official ranked tier or MMR bucket when available;
- if official tier is unavailable, do not treat tier as a main input feature
  unless repeated-player coverage is empirically sufficient;
- simple historical performance averages are not valid tier proxies because
  they do not adjust for lobby strength;
- opponent-adjusted ratings are exploratory fallback only and require a
  repeated-player coverage audit;
- never define tier from current-match final rank, survival time, or win/loss;
- never use the existing `ranked_rp_proxy` as a tier input because it is
  computed from current-match placement/kills/assists and is therefore an
  outcome proxy;
- record `tier_source`, `tier_granularity`, and `tier_policy` in every result;
- report metrics per tier even when the model is trained on pooled data.

Candidate policies:

| Policy | Description | Purpose |
| --- | --- | --- |
| pooled_no_tier | Train all tiers together without tier input | baseline |
| pooled_tier_conditioned | Train one model with tier as C0 context/gate | tests conditioning |
| tier_specialist | Train separate models per tier bucket | tests tier-specific dynamics |
| tier_balanced | Reweight/sample tiers to reduce dominance by large tiers | robustness |
| leave_one_tier_out | Hold out one tier bucket during training | generalization stress test |

Conditional current-data proxy:

```text
rating_same-mode_v1_min3_cov50_k32_s400
```

This is an opponent-adjusted pre-match latent rating, not an official PUBG tier.
It should not be used as a main model channel unless enough repeated players
exist to make the rating graph connected and stable.

### P7. Manifest

Every experiment must output:

- input channel list;
- feature names;
- model family;
- graph relation mode;
- head list;
- metric list;
- tier policy and tier source;
- split file path;
- checkpoint/result path.

---

## 12. Architecture And Tuning Contract

Every model run must declare:

| Category | Required fields |
| --- | --- |
| input channels | C0-C8 consumed |
| temporal resolution | snapshot, phase, or both |
| graph resolution | none, snapshot, phase, both |
| relation handling | none, GCN-combined, R-GCN, edge-conditioned |
| tier policy | pooled_no_tier, pooled_tier_conditioned, tier_specialist, tier_balanced, leave_one_tier_out |
| hidden sizes | snapshot hidden, phase hidden, mid dims |
| depth | number of temporal/graph blocks |
| dropout/weight decay | regularization |
| loss weights | hazard/rank/top-k/survival weights |
| optimizer | optimizer, learning rate, scheduler |
| seed policy | seeds and split path |
| output contract | prediction CSV and metrics JSON/CSV |

Tuning must be separated into:

1. baseline tuning;
2. architecture tuning;
3. representation ablation;
4. final locked evaluation.

Do not tune after looking at test metrics.

---

## 13. Immediate Implementation Order

1. Create a machine-readable feature channel manifest.
2. Add required `clock`, `entity`, `semantic_domain`, `causal_window`, and
   `model_stream` metadata.
3. Add `domain_mechanism` labels from `docs/PUBG_DOMAIN_DESIGN_LAYER_2026-06-22.md`.
4. Add validation that every feature belongs to exactly one clock stream, one
   semantic channel, and at least one domain mechanism.
5. Add result metadata that records consumed clocks, channels, and domain mechanisms for each run.
6. Audit official ranked tier/MMR availability and define a leakage-safe tier policy.
7. Implement Sagong-style C7 prior-work channel.
8. Define model families M0-M6 in experiment matrix.
9. Add head/metric contract to evaluator outputs.
10. Rebuild presentation problem/method slides from this contract.

---

## 14. Domain-Aware Additions Required Before Next Full Run

The next full redesign should incorporate these PUBG-specific requirements:

| Requirement | Why it matters |
| --- | --- |
| map-specific scaling | fixed arena scaling can blur geography across maps |
| roster-normalized features | solo/duo/squad and partial squad survival have different meanings |
| snapshot rotation urgency | survival depends on whether a team can reach safety in time |
| contact-gated relative edges | raw graph topology previously behaved like smoothing |
| third-party/crossfire exposure | battle royale risk can come from nearby fights, not only direct combat |
| phase-band reporting | early looting, mid rotation, and late compression have different mechanics |
| resource readiness ratios | counts alone can confuse event volume with actual survival capacity |
| causal temporal deltas | snapshot-first models need trend, not only instantaneous state |
| skill/lobby tier stratification | survival dynamics and calibration may differ by player skill/lobby strength |

These are design requirements, not yet final empirical claims.

---

## 15. Open Design Questions

- Should the canonical main population be squad-only first?
- Should solo/duo be separate experiments or domain generalization?
- Should map be a conditioning feature, a split axis, or both?
- Should snapshot-first models be allowed to consume phase aggregates as auxiliary context?
- Should relative-advantage edges be phase-tier only for now, or ported to snapshot tier?
- Should rank/top-k heads remain in the corrected snapshot-first design, or be secondary after hazard is stable?
- Should the next manifest require both `channel` and `domain_mechanism` for every feature?
- Should map-specific coordinate normalization replace the current fixed `8160.0` scale before full reruns?
- Should resource-readiness features remain count-based, or should inventory quality be extracted before reruns?
- Does raw data contain official ranked tier/MMR, or must we define an
  opponent-adjusted proxy rating from historical/pre-match information?
- Should the main model be pooled with tier conditioning, trained separately per tier, or both?
