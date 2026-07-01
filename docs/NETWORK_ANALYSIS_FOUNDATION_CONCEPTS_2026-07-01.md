# Network Analysis Foundation Concepts

Date: 2026-07-01

This note preserves the basic network-analysis concepts used across the PUBG
spatio-temporal and graph-modeling work.  It is intentionally conceptual rather
than tied to one generated report or checkpoint.

## Why Network Analysis Was Used

The starting point was:

```text
PUBG teams are not independent rows.
```

Each team's survival condition is affected by:

- nearby opponent teams;
- direct combat events;
- shared safe-zone pressure;
- competition for routes, compounds, and playable space;
- the changing phase structure of the match.

Network analysis gives a vocabulary for representing these dependencies as
nodes, edges, weights, communities, and time-evolving graph structure.

## Core Graph Vocabulary

The minimal graph object is:

```text
G = (V, E)
```

In this project:

| Concept | General meaning | PUBG meaning |
| --- | --- | --- |
| `V` | nodes | players or teams |
| `E` | edges | relations between players/teams |
| node feature | attribute of one node | team state, position, health, alive count |
| edge feature | attribute of a pair | distance, damage, relative advantage |
| edge weight | strength of relation | damage amount, encounter intensity, proximity weight |
| adjacency matrix `A` | graph connectivity | who is connected to whom at a time/cut |
| global context `U` | graph-level state | map, mode, phase, safe zone, alive population |

The project moved from simple graphs:

```text
team --damage--> team
```

toward dynamic, multi-relational graphs:

```text
team_i --proximity/combat/zone-competition--> team_j
```

## Player Graph vs Team Graph

Two graph resolutions appeared in the analysis.

### Player-Level Graph

Early frame-level analysis represented alive players as nodes:

```text
node = player
edge = ally / enemy proximity / combat
```

This was useful for visualization, centrality, community detection, and
per-frame network summaries.

### Team-Level Graph

The research model settled on team nodes:

```text
node = team
edge = team-to-team survival-relevant relation
```

This fits the main targets:

- team survival;
- team final rank;
- next-phase elimination hazard;
- top-k placement.

The team graph is not claiming that teams are similar.  It claims that teams can
change each other's survival conditions.

## Static, Windowed, And Dynamic Graphs

Three temporal forms were used.

### Static Match Graph

All observed combat events in one match are collapsed into one graph:

```text
G_match
```

This is useful for summary statistics such as total degree, PageRank, or
community position, but it can mix early and late information.

### Windowed Graph

Only events up to a fixed time are used:

```text
G_[0,300s]
```

This was used to reduce survival-time bias.  If a team survives longer, it has
more opportunity to accumulate edges; early windows partially control that.

### Dynamic Graph Sequence

The strongest formulation is:

```text
G_1, G_2, ..., G_T
```

or, in PUBG terms:

```text
10-second team graph snapshots -> phase-cut prediction
```

This preserves when relations appeared, not just whether they ever appeared.

## Edge Semantics

Edges should mean survival dependency, not similarity.

### Proximity / Exposure Edge

Meaning:

```text
Two teams are spatially close enough that encounter, observation, or pressure is possible.
```

Possible features:

- distance;
- bearing;
- closing speed;
- local density;
- same sector;
- zone-relative position.

### Combat Edge

Meaning:

```text
One team directly changed another team's survival state through damage or pressure.
```

Combat edges can be directed:

```text
j -> i means team j damaged or pressured team i
```

Possible features:

- damage dealt;
- damage taken;
- knock/kill events;
- time since last damage;
- damage imbalance.

### Zone / Habitat Competition Edge

Meaning:

```text
Two teams compete for the same survivable space, route, chokepoint, compound, or destination.
```

Possible features:

- shared safe-zone entry route;
- path overlap;
- same target sector;
- arrival time gap;
- destination density;
- safe-area pressure similarity.

## Node Features, Edge Features, And Global Context Must Stay Separate

This distinction is one of the most important design rules.

| Object | Example | Where it belongs |
| --- | --- | --- |
| team state | alive players, health, speed | node feature |
| spatial position | centroid `x/y/z`, spread | node feature |
| pairwise distance | distance from team `j` to team `i` | edge feature |
| direct damage | damage `j -> i` | directed edge feature |
| safe-zone phase | current phase, zone radius | global context |
| target label | eliminated before next phase | supervision only |
| alive/valid mask | whether team is scorable | mask only |

If node features and edge features are changed at the same time, the experiment
cannot tell whether the gain came from better team-state representation or
better topology.

## Basic Graph Metrics Used

### Degree

Degree measures how many neighbors a node has.

PUBG interpretation:

```text
How many other teams did this team encounter or fight?
```

Weighted degree sums edge weights:

```text
weighted_degree_i = sum_j w_ij
```

PUBG interpretation:

```text
How much total combat/exposure intensity involved this team?
```

### Closeness Centrality

Closeness measures how near a node is to all others through graph paths.

PUBG interpretation:

```text
A team is structurally close to the broader combat network.
```

When edge weight means stronger relation, distance is usually converted as:

```text
distance_ij = 1 / weight_ij
```

### Betweenness Centrality

Betweenness measures how often a node lies on shortest paths between others.

PUBG interpretation:

```text
A team bridges otherwise separate fight clusters.
```

This was used as a proxy for central, cross-cluster, or multi-front exposure.

### Eigenvector Centrality

Eigenvector centrality gives high score to nodes connected to other important
nodes.

PUBG interpretation:

```text
A team is connected to influential or heavily connected combat participants.
```

### PageRank

PageRank is a random-walk importance score.

PUBG interpretation:

```text
A team is involved with important opponents or central battle regions,
not merely many weak edges.
```

## Global Graph Metrics Used

| Metric | Meaning | PUBG interpretation |
| --- | --- | --- |
| density | fraction of possible edges present | how saturated the match's fight network is |
| average degree | mean number of relations per node | average team exposure/fight count |
| clustering coefficient | neighbor-neighbor closure | triangular or multi-team fight structure |
| connected components | separated graph parts | independent fight regions |
| giant component size | largest connected group | how much of the match is tied into one fight system |
| diameter | longest shortest path in component | graph-scale separation between teams |
| average path length | typical graph distance | small-world-like combat connectivity |

These metrics describe the match or frame as a whole, not one team's individual
quality.

## Community Detection

Community detection asks:

```text
Do nodes form internally dense groups?
```

The project used:

- Louvain community detection;
- spectral clustering;
- modularity `Q`.

PUBG interpretation:

```text
Communities approximate local fight regions, clustered encounter zones, or
groups of teams tied together by combat/exposure.
```

Useful derived features:

| Feature | Meaning |
| --- | --- |
| `community_id` | assigned community |
| `community_size` | number of teams in the community |
| `community_density` | internal edge density |
| `within_deg` | weighted degree inside community |
| `across_deg` | weighted degree crossing communities |
| `community_embeddedness` | how internally embedded a team is |
| `match_modularity_q` | strength of community structure |

Important interpretation:

```text
Community is not an alliance.
```

It is a structural grouping induced by combat, proximity, or exposure.

## Position Labels

For interpretability, teams were sometimes mapped to coarse network positions:

```text
core
mid
peripheral
isolated
```

Typical rule:

- `isolated`: no edges;
- `core`: high betweenness or central network position;
- `peripheral`: low centrality, edge of interaction structure;
- `mid`: between core and peripheral.

These labels are descriptive, not ground-truth roles.

## Strategy Type Scores

The Hunter/Sniper/Traveler/Camper analysis reused network and movement concepts.

Two latent scores were built:

```text
FightScore  = combat/network exposure
ActiveScore = movement/spatial activity
```

Example FightScore components:

- combat participation rate;
- damage event involvement per minute;
- combat-network degree per minute;
- weighted degree per minute;
- PageRank;
- bridge score.

Example ActiveScore components:

- movement per minute;
- active movement frame rate;
- spatial spread;
- landing density;
- nearby landing teams.

Then strategy type was defined by high/low FightScore and ActiveScore:

| Type | FightScore | ActiveScore |
| --- | --- | --- |
| Hunter | high | high |
| Sniper | high | low |
| Traveler | low | high |
| Camper | low | low |

Kills and assists were intentionally kept out of the type score when they were
used as outcomes or ranked-score proxies.

## Spatio-Temporal Link

Network analysis was combined with spatial-temporal analysis through moving
team objects:

```text
team centroid: x, y, z, t
```

Core ST concepts:

- trajectory;
- speed;
- movement efficiency;
- squad spread;
- zone-center distance;
- zone-edge distance;
- nearest enemy distance;
- local enemy density;
- phase-local movement and combat windows.

Network edges are one way of turning ST relations into graph structure:

```text
spatial distance / local density / combat events / shared zone pressure
-> team-team relation
```

## Phase-Aware Analysis

Battle Royale time is not uniform.  Safe-zone phases change the meaning of the
same distance or edge.

The project therefore separates:

```text
snapshot time: high-frequency observation, often 10 seconds
phase time: game-mechanic cut points
```

Important design rule:

```text
Phase is a prediction cut point, not merely a feature aggregation bucket.
```

For survival prediction:

```text
observations <= phase k
-> predict elimination before phase k+1
```

This keeps the setup causal and leakage-safe.

## From Network Analysis To GNN Modeling

Classical network analysis produced interpretable features:

```text
degree
weighted degree
PageRank
betweenness
community position
local clustering
```

GNN modeling generalizes this by learning messages over the graph:

```text
message from team j to team i
  = function(team_i state, team_j state, edge_ij, phase context)
```

This is why the later model design emphasizes:

- no-graph baselines;
- proximity/combat/zone relation ablations;
- relation-specific GNNs;
- edge-conditioned message passing;
- edge-shuffle and density-matched null controls.

The graph claim is valid only if relational information improves prediction
beyond node-only and global-context baselines.

## Guardrails

### Do Not Claim Homophily

PUBG edges are not primarily similarity links.

Preferred wording:

```text
PUBG relations are competitive survival dependencies, not homophilic similarity edges.
```

### Control Exposure-Time Bias

Teams that survive longer can accumulate more events and edges.

Controls:

- fixed early windows;
- phase-local rates;
- per-minute normalization;
- at-risk masks;
- causal cut-phase prediction.

### Avoid Label Leakage

Do not use future combat, future position, or post-cut survival information in
features for a cut-phase prediction.

### Keep Interpretation Structural

Network features show structural association and predictive information.  They
do not automatically prove causal effects.

## Minimal Rebuild Contract

To rebuild the network-analysis layer from raw data, preserve this conceptual
pipeline:

```text
official telemetry / DB rows
-> player/team snapshots
-> team state features
-> team-team edges by relation
-> graph metrics and community features
-> phase-aware ST/network feature table
-> node/global/edge ablations
-> survival/rank prediction
```

Key code entry points:

```text
network_analysis/team_network.py
network_analysis/team_centrality.py
network_analysis/community_detect.py
network_analysis/metrics.py
network_analysis/hypotheses.py
network_analysis/st_hypothesis_analysis.py
network_analysis/strategy_type_network_analysis.py
network_analysis/phase_strategy_type_analysis.py
```

Key design documents:

```text
docs/GRAPH_DATA_MODELING_FOUNDATION_2026-06-22.md
docs/BATTLE_ROYALE_GRAPH_FORMALISM_2026-06-22.md
docs/PUBG_GRAPH_MODELING_RESEARCH_FRAME_2026-06-22.md
docs/SNAPSHOT_PHASE_SURVIVAL_DESIGN.md
```
