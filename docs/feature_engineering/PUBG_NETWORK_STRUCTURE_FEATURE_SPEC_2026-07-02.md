# PUBG Network Structure Feature Spec

Date: 2026-07-02

Class: `FeatureContract`

Status: design lock candidate

## Purpose

This document designs the network-structure feature layer for the PUBG Battle
Royale framework.  It extends the operational manifest by specifying how graph
metrics such as degree, strength, betweenness, PageRank, component structure,
community structure, and centralization should enter the model.

The central design rule is:

```text
team-level network position -> X_i(k)
match/phase-level network state -> U(k)
pairwise relation state -> E_ij(k)
```

Network features are not a replacement for raw telemetry.  They are a structured
projection of raw telemetry into relation-aware survival features.

Upstream documents:

- `docs/feature_engineering/PUBG_FEATURE_OPERATIONAL_MANIFEST_2026-07-02.md`
- `docs/feature_engineering/BATTLE_ROYALE_FEATURE_ONTOLOGY_2026-07-02.md`
- `docs/feature_engineering/NETWORK_ANALYSIS_FOUNDATION_CONCEPTS_2026-07-01.md`

## Design Thesis

Battle Royale teams are not independent rows.  Their survival environment is
partly created by other teams through proximity, combat, co-pressure, and route
competition.

Network features should therefore answer three different questions:

| Question | Model object | Example |
| --- | --- | --- |
| What relation exists between team `i` and team `j`? | `E_ij(k)` | distance, combat weight, co-pressure, relative height |
| What role does team `i` occupy in the current relation field? | `X_i(k)` | degree, strength, betweenness, PageRank, bridge score |
| What is the overall interaction structure of phase `k`? | `U(k)` | graph density, fragmentation, centralization, modularity |

This split is necessary because centrality is not always global context.  A
team's betweenness is a team feature.  The distribution or concentration of
betweenness across the alive graph is a global feature.

Canonical ownership:

```text
Graph-derived proximity degree, centrality, component, and community features
belong to O7 Relation / Graph Structure.
```

O4 may own realized combat events, and O6 may reference proximity as an
exposure proxy, but graph-derived proximity features such as
`net_prox_degree_100m` are not duplicated into those families.

## Portability Boundary

Network structure is a `BR_GENERAL` construct.  The current edge channels and
weights are PUBG/STDB projections of that construct.

| Network layer | BR_GENERAL meaning | PUBG_NATIVE projection | STDB_IMPLEMENTATION |
| --- | --- | --- | --- |
| combat graph | hostile interaction changes survival pressure | damage, kill, groggy/DBNO event taxonomy | `team_phase_edge_cache`, `net_combat_*` |
| proximity graph | potential contact / spatial exposure | PUBG team centroid distance at snapshots | `team_proximity`, `net_prox_*` |
| co-pressure graph | teams share shrinking-zone pressure | blue/safe-zone boundary pressure overlap | `net_copress_*`, `hab_*` joins |
| advantage graph | pairwise relative state under contact/proximity | relative `pos_z`, alive count, health, zone distance | `advantage_edges.py`, `dz`, `dalive`, `dhealth`, `dzone` |
| global graph state | phase-level interaction structure | PUBG telemetry-observed graph state | `*_null_z`, `*_null_delta` planned features |

Cross-game claims should use the BR_GENERAL terms above.  PUBG-native terms such
as `groggy` are valid only inside the PUBG instantiation.

## Graph Clock Contract

All network features are cut-specific:

```text
G(k) = graph built only from telemetry observed through phase cut k
```

Allowed clocks:

| Clock | Use |
| --- | --- |
| `snapshot_10s` | dynamic proximity/combat edges at snapshot time `t` |
| `trailing_10s` | combat events in `(t-10s, t]` |
| `phase_window` | graph edges observed during completed phase `k` |
| `phase_to_date` | cumulative graph over phases `1..k` |

Forbidden:

```text
G_full_match centrality as model input
```

A full-match graph is allowed only for descriptive figures after outcomes are
not being predicted.

## Graph Families

### NS1. Combat Graph

Meaning:

```text
Directed hostile interaction between teams.
```

Node set:

```text
alive or at-risk teams at phase cut k
```

Edge:

```text
i -> j if team i damaged, knocked, killed, or otherwise pressured team j
```

Recommended edge weight:

```text
log(1 + damage + 100 * kills + 50 * groggy)
```

Current implementation:

- source table: `stdb.team_phase_edge_cache`
- node summary table: `stdb.team_phase_network_features`
- existing columns: `net_combat_*`, `cum_combat_*`, `cum_unique_*`

### NS2. Proximity / Exposure Graph

Meaning:

```text
Spatial exposure or potential encounter among alive teams.
```

Edge:

```text
i -- j if teams are within a chosen distance gate or nearest-neighbor set
```

Recommended variants:

| Variant | Use |
| --- | --- |
| `prox_100m` | immediate local threat |
| `prox_200m` | plausible encounter/contact |
| `prox_500m` | local strategic neighborhood |
| `decay_weighted` | continuous distance-decay exposure |

Current implementation:

- source table: `stdb.team_proximity`
- phase edge cache: `edge_type = proximity`
- existing columns: `net_prox_*`, `cum_prox_close_exposure_200m`

### NS3. Co-Pressure Graph

Meaning:

```text
Teams jointly squeezed by zone pressure, route pressure, or safe-area scarcity.
```

Edge:

```text
i -- j if teams are spatially close while both are under zone pressure
```

Current implementation:

- source table: `stdb.team_phase_edge_cache`
- existing columns: `net_copress_*`, `cum_copress_strength`

Claim boundary:

```text
Co-pressure is analyst-view telemetry.  It models shared survival environment,
not necessarily player-perceived relation.
```

### NS4. Advantage / Relative-State Graph

Meaning:

```text
Pairwise relation features under proximity or combat gates.
```

Current edge feature examples:

```text
d_over_ell
combat_dmg_log
combat_kills
dz
dalive
dhealth
dzone
```

Current implementation:

- code surface: `model_stgcn/v3/advantage_edges.py`
- model object: `E_ij(k)`

This family is the bridge between O7 relation structure and O6
spatial-tactical affordance.

## Feature Layer

### Layer E. Pairwise Edge Features

These features remain on `E_ij(k)` and are consumed by graph models directly.

| Feature | Graph | Status | Notes |
| --- | --- | --- | --- |
| `distance_ij_m` / `d_over_ell` | proximity | available | normalize by phase scale where possible |
| `proximity_gate_ij` | proximity | available | candidate edge mask |
| `combat_weight_ij` | combat | available | directed hostile intensity |
| `combat_kills_ij` | combat | available | directed kill pressure |
| `copressure_weight_ij` | co-pressure | available | shared zone pressure |
| `delta_z_ij` | advantage | available / bridge | O6 high-ground proxy |
| `delta_alive_ij` | advantage | available | relative survival capital |
| `delta_health_ij` | advantage | available | relative damage/capital state |
| `delta_zone_pressure_ij` | advantage | available | relative zone exposure |

Inclusion:

```text
OM5 graph relation stream; OM6 when relative-height bridge is enabled.
```

### Layer X. Team Network-Role Features

These features are team/node features `X_i(k)`.

Already available:

| Feature group | Example columns | Source |
| --- | --- | --- |
| combat degree/strength | `net_combat_out_degree`, `net_combat_in_degree`, `net_combat_total_strength` | `team_phase_network_features` |
| combat asymmetry | `net_combat_balance`, `net_combat_reciprocity_ratio` | `team_phase_network_features` |
| proximity exposure role | `net_prox_degree_100m`, `net_prox_degree_200m`, `net_prox_min_distance_m`, `net_prox_neighbor_entropy` | `team_phase_network_features` |
| co-pressure role | `net_copress_degree`, `net_copress_strength`, `net_copress_mean_ratio` | `team_phase_network_features` |
| cumulative role | `cum_combat_*`, `cum_unique_*`, `cum_prox_*`, `cum_copress_*` | `team_phase_network_features` |

Designed next-extension features:

| Feature | Graph | Meaning | Recommended status |
| --- | --- | --- | --- |
| `net_combat_betweenness` | combat | chain-like hostile interaction position, not a traversable path | interpretive only |
| `net_prox_betweenness` | proximity | bridge across spatially separated groups | sensitivity / interpretive |
| `net_copress_betweenness` | co-pressure | bridge among teams sharing pressure corridors | interpretive first |
| `net_combat_pagerank` | combat | receives or creates influential combat pressure depending direction convention | sensitivity |
| `net_prox_eigenvector` | proximity | embedded in dense high-exposure neighborhood | sensitivity |
| `net_closeness_alive` | proximity | short graph distance to many alive teams | sensitivity |
| `net_kcore_index` | proximity/combat | membership in dense interaction core | sensitivity |
| `net_prox_bridge_score` | proximity | proximity betweenness in excess of local degree | interpretive |
| `net_community_id` | proximity/co-pressure | community assignment for phase graph | figure / categorical |
| `net_within_module_degree` | community graph | role inside community | sensitivity |
| `net_participation_coef` | community graph | cross-community exposure role | sensitivity |

Normalization rule:

```text
centrality_norm = metric normalized within (match_id, phase_idx, graph_type)
```

Raw centrality values should not be compared across phases without alive-team
normalization.

### Layer U. Global Network-State Features

These features summarize the whole phase graph and belong to `U(k)`.

Raw global graph statistics are mechanically coupled to the number of alive
teams `N(k)` and the edge count `M(k)`.  Since `N(k)` is already an O2 headline
feature, global network-state features must enter the headline only as
null-adjusted residuals.

Locked definition:

```text
u_metric_null_delta(k) = observed_metric(k) - E_null[metric | N(k), M(k), graph_type]
u_metric_null_z(k) =
  (observed_metric(k) - E_null[metric | N(k), M(k), graph_type])
  / sd_null[metric | N(k), M(k), graph_type]
```

Preferred null:

```text
edge-shuffle null preserving alive nodes, graph type, edge count, and where
practical the in/out degree sequence
```

The existing edge-shuffle controls are the intended implementation source for
this null.  Raw `U` network metrics may be reported descriptively, but the OM5
global-context extension should use `*_null_delta` or `*_null_z` features.

| Raw feature | Headline feature | Graph | Meaning after null adjustment |
| --- | --- | --- | --- |
| `graph_density` | `graph_density_null_z` | combat/proximity/co-pressure | excess relation density beyond `N,M` expectation |
| `avg_degree_alive` | `avg_degree_alive_null_z` | any | excess connectedness beyond alive-count/edge-count baseline |
| `largest_component_ratio` | `largest_component_ratio_null_z` | any | excess giant-component formation |
| `num_components` | `num_components_null_z` | any | excess fragmentation |
| `network_fragmentation` | `network_fragmentation_null_z` | any | fragmentation beyond `N,M` expectation |
| `reciprocity_ratio` | `reciprocity_ratio_null_z` | combat | excess mutual combat pressure |
| `centralization_degree` | `centralization_degree_null_z` | any | hub concentration beyond null |
| `centralization_betweenness` | `centralization_betweenness_null_z` | any | bridge concentration beyond null |
| `centrality_gini` | `centrality_gini_null_z` | any | role inequality beyond null |
| `mean_betweenness_alive` | `mean_betweenness_alive_null_z` | proximity/co-pressure | average brokerage pressure beyond null |
| `max_betweenness_alive` | `max_betweenness_alive_null_z` | proximity/co-pressure | extreme bridge/control presence beyond null |
| `community_count` | `community_count_null_z` | proximity/co-pressure | mesoscale grouping beyond null |
| `modularity` | `modularity_null_z` | proximity/co-pressure | community separation beyond null |
| `intercommunity_edge_ratio` | `intercommunity_edge_ratio_null_z` | proximity/co-pressure | cross-community exposure beyond null |

Inclusion:

```text
OM5 global-context extension after node/edge network block is stable.
```

Interpretation:

```text
U-network features describe the phase's survival environment.  They do not
describe a specific team's agency unless interacted with X_i(k).
```

Guard:

```text
No raw U-network metric enters a headline OM5 model unless the corresponding
null-adjusted feature is unavailable and the run is explicitly marked
descriptive/sensitivity.
```

## Betweenness Design

Betweenness is useful but easy to misuse.  It should be computed only on
pre-cut graphs.

Definition:

```text
betweenness_i(k) =
  fraction of shortest paths between alive teams that pass through team i
  in G(k)
```

Recommended graph-specific meaning:

| Graph | Meaning of high betweenness |
| --- | --- |
| proximity | team occupies a bridge between spatial clusters |
| combat | interpretive only; combat edges are not traversable routes |
| co-pressure | team connects pressure corridors or contested safe-space groups |

Design choices:

- For proximity graphs, edge distance should be inverse weight or actual
  distance, not raw similarity weight.
- The headline sensitivity metric is proximity betweenness.  Combat
  betweenness is interpretive-only because combat edges do not represent
  traversable paths or physical movement corridors.
- For sparse phases, set betweenness to `0` when there are fewer than three
  alive connected teams and record a validity mask.

Bridge score is locked as:

```text
net_prox_bridge_score =
  percentile_rank(net_prox_betweenness)
  - percentile_rank(net_prox_degree_200m)
```

where percentile ranks are computed within `(match_id, phase_idx, graph_type)`.
Positive values mean the team is more bridge-like than its local degree alone
would imply.  The clipped figure version may use:

```text
net_prox_bridge_score_pos = max(net_prox_bridge_score, 0)
```

## PageRank / Eigenvector Design

These features measure embeddedness in a relation field.

| Metric | Preferred graph | Meaning |
| --- | --- | --- |
| `pagerank` | directed combat | influence/pressure rank under direction convention |
| `eigenvector` | undirected proximity | embeddedness in dense exposure neighborhood |
| `kcore` | proximity/combat | membership in dense interaction core |

Direction convention must be explicitly recorded:

```text
attacker -> victim
```

If PageRank is computed on this direction, high PageRank may mean receiving
pressure from high-pressure attackers.  If reversed, it may mean projecting
pressure onto important targets.  The chosen convention must match the paper
claim.

## Community And Brokerage Design

Community features are mesoscale, not labels of strategy.

Recommended metrics:

| Feature | Level | Meaning |
| --- | --- | --- |
| `community_id` | `X_i(k)` | phase-local graph community assignment |
| `community_size` | `X_i(k)` | size of the team's relation community |
| `within_module_degree` | `X_i(k)` | connectedness inside community |
| `participation_coef` | `X_i(k)` | edges distributed across communities |
| `modularity` | `U(k)` | strength of community separation |
| `intercommunity_edge_ratio` | `U(k)` | cross-community exposure |

Use community identifiers only as categorical/interpretable state.  Community
IDs are arbitrary across matches and phases and should not be treated as stable
semantic labels unless relabeled by role.

## Causal And Leakage Guard

Allowed:

- phase-local graph built from events inside completed phase `k`;
- cumulative graph built from phases `1..k`;
- snapshot graph built from state/events up to snapshot `t`;
- graph metrics normalized within `(match_id, phase_idx, graph_type)`;
- descriptive full-match graph figures outside supervised inputs.

Forbidden:

- full-match centrality as an input to next-phase prediction;
- centrality computed using teams eliminated after the cut as if they were
  still at risk;
- community labels or graph normalizers fit using validation/test outcomes;
- path-based metrics that use future route trajectories beyond phase cut `k`.

Ambiguous / sensitivity only:

- train-split-fitted role categories;
- route-conflict graphs using all-team trajectories;
- community IDs as raw categorical model inputs.

## Implementation Status

| Block | Status | Notes |
| --- | --- | --- |
| edge cache combat/proximity/co-pressure | implemented | `stdb.team_phase_edge_cache` |
| basic network-role node features | implemented | `stdb.team_phase_network_features` |
| cumulative network-role features | implemented | `cum_*` columns through phases `1..k` |
| advantage edge stream | implemented | `model_stgcn/v3/advantage_edges.py` |
| betweenness / closeness / PageRank | not implemented | design extension |
| global network-state features | not implemented as manifest block | design extension |
| community / modularity features | not implemented | interpretive or sensitivity extension |

## Recommended Integration Order

1. Keep current `net_*` and `cum_*` network-role block as the first O7 input.
2. Add global graph-state `U(k)` features only after verifying current O7
   ablation stability.
3. Add betweenness/PageRank/k-core as a sensitivity block, not as the first
   headline network result.
4. Add community/brokerage features for interpretation and figures first.
5. Promote any extension to headline only if phase-wise ablation shows stable
   signal beyond existing relation edges and node summaries.

## Claim Boundary

Allowed claim:

```text
Network features operationalize Battle Royale interdependence by representing
pre-cut proximity, combat, and co-pressure structure at edge, team-role, and
phase-global levels.
```

Disallowed claim:

```text
Centrality computed from a full match explains earlier survival decisions.
```

The paper should say that network features are analyst-view telemetry features
unless a specific feature is proven to be player-available.
