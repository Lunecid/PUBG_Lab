# DSACG Method Skeleton (2026-06-22)

## Purpose

This skeleton prevents the paper from shrinking back into "we applied a GNN to
PUBG." The method must keep two levels separate:

```text
General formalism: Dynamic Shrinking-Arena Competitive Graph (DSACG)
Empirical study: PUBG telemetry team-level projection of DSACG
```

Core claim:

```text
We formalize common Battle Royale mechanics as a DSACG and empirically
instantiate the observable team-level projection on PUBG telemetry.
```

Do not claim:

```text
We prove generalization to every Battle Royale title.
```

Safe wording:

```text
We evaluate within-title transfer across maps, modes, phase bands, pressure
levels, and graph null controls. Cross-title validation remains future work.
```

## Research Questions

### RQ1. Genre Formalization

```text
What kind of dynamic survival system is a Battle Royale match?
```

Answer target:

```text
A Battle Royale match can be formalized as a Dynamic Shrinking-Arena
Competitive Graph whose agents, arena, resources, zones, combat events, and
elimination outcomes form time-evolving survival dependencies.
```

### RQ2. Projection Validity

```text
How can the full agent-space-resource-zone-event DSACG be projected into a
team-level graph for PUBG survival prediction?
```

Answer target:

```text
Because the empirical targets are team survival and final team placement, PUBG
telemetry is causally aggregated into team states, team-to-team relations, and
global shrinking-arena context. This projection instantiates the subset of
DSACG relations observable and relevant for team-level hazard prediction.
```

### RQ3. Relational Predictive Validity

```text
Do survival-relevant relations improve phase-wise elimination hazard and final
rank prediction beyond team-only and global-context baselines?
```

Minimum controls:

```text
snapshot_tcn_no_graph
snapshot_rgcn
snapshot_rgcn_edge_shuffle
snapshot_gate_only
snapshot_relative_edge
```

### RQ4. Edge Semantics

```text
Which relation semantics carry survival-relevant signal?
```

Relation families:

- spatial exposure;
- combat interaction;
- zone / habitat competition;
- resource or relative advantage;
- contact-gated pairwise advantage.

### RQ5. Mechanistic And Transfer Validity

```text
Does the model behave in Battle-Royale-like ways across phase bands, maps,
modes, pressure levels, and graph null controls?
```

Required slices:

- phase band;
- map;
- mode / team size;
- zone pressure band;
- local opponent density band;
- alive roster band;
- graph condition.

## Method Structure

### 3.1 Battle Royale As A Survival System

Define the genre-level match object:

```text
M = <A, Omega, T, Z_1:T, R_1:T, C_1:T, S_1:T, Y>
```

| Symbol | Meaning |
| --- | --- |
| `A` | agents or teams |
| `Omega` | arena / map space |
| `T` | time axis |
| `Z_1:T` | shrinking zone / hazard process |
| `R_1:T` | resource field |
| `C_1:T` | combat and contact events |
| `S_1:T` | agent/team state sequence |
| `Y` | elimination time and final ranking |

### 3.2 Dynamic Shrinking-Arena Competitive Graph

Define the general heterogeneous graph:

```text
G_t^BR =
(
  V_t^A,
  V_t^S,
  V_t^R,
  V_t^Z,
  E_t^AA,
  E_t^AS,
  E_t^AR,
  E_t^AZ,
  E_t^SS,
  X_t,
  A_t,
  U_t
)
```

Interpretation:

- `V_t^A`: agents or teams;
- `V_t^S`: spatial regions;
- `V_t^R`: resources;
- `V_t^Z`: zone / hazard nodes;
- `E_t^AA`: agent-agent exposure, combat, competition;
- `E_t^AS`: agent-space occupancy, route, cover;
- `E_t^AR`: agent-resource access;
- `E_t^AZ`: agent-zone exposure;
- `E_t^SS`: spatial adjacency or route structure.

### 3.3 PUBG Team-Level Projection

Define the empirical projection:

```text
Gbar_t^BR =
(
  Vbar_t^team,
  Ebar_t^exposure,
  Ebar_t^combat,
  Ebar_t^habitat,
  Ebar_t^resource,
  Ebar_t^zone,
  Xbar_t,
  Abar_t,
  U_t
)
```

Projection rationale:

```text
The full DSACG is a heterogeneous agent-environment graph. In this paper, the
empirical target is team survival and final team placement. Therefore, player
telemetry is aggregated into team-level states, and observable interactions are
projected into team-to-team or team-context relations. This projection does not
exhaust the full DSACG; it implements the subset needed for team-level hazard
prediction.
```

### 3.4 Relational Survival Learning

Canonical prediction unit:

```text
at-risk team x phase cut
```

Hazard definition:

```text
p_i^{t + Delta} = sigmoid(f_theta(G_1:t^BR, i))
```

Operational target:

```text
Given all 10-second snapshots observed up to phase k, predict whether team i is
eliminated before phase k+1.
```

At-risk semantics:

```text
Because the prediction is made at the phase-k cut after observing phase k, a
team is at risk if it is alive at the end of phase k. This is different from
the phase-start alive mask used by some phase-level node tensors.
```

Outputs:

- next-phase elimination hazard;
- final rank or normalized placement;
- optional top-k placement heads;
- calibration / reliability metrics.

### 3.5 Edge Semantics And Null Controls

Edge semantics:

| Relation | Owner in full DSACG | PUBG team-level projection |
| --- | --- | --- |
| spatial exposure | agent-agent / agent-space | team-team proximity or relative position edge |
| combat interaction | combat event / directed agent-agent | directed team-team combat edge |
| zone pressure | zone node / agent-zone | team node feature plus global context |
| habitat competition | space node / agent-space | team-team habitat or convergence edge |
| resource readiness | resource node / agent-resource | team node feature |
| relative advantage | agent-agent under contact | directed contact-gated pair edge |

Null controls:

- no graph;
- edge shuffle;
- node shuffle;
- gate only;
- relative features without true topology;
- relation drop;
- density-matched random graph.

## Experiment Ladder

Start small before running full model matrices.

| Step | Model | Purpose |
| --- | --- | --- |
| 1 | `snapshot_tcn_no_graph` | temporal signal only |
| 2 | `snapshot_rgcn` | typed relations plus temporal signal |
| 3 | `snapshot_rgcn_edge_shuffle` | topology null control |
| 4 | `snapshot_gate_only` | contact gate without relative pair features |
| 5 | `snapshot_relative_edge` | contact-gated relative advantage |
| 6 | Sagong-style baseline | closest phase-aware PUBG prior-work comparison |

Current validation state:

```text
tools/validate_genre_feature_manifest.py      # manifest/channel contract
tools/smoke_snapshot_phase_contract.py        # tensor + label/mask smoke
model_stgcn/snapshot_baselines.py             # no-graph baseline modules
```

## Required Metrics

Hazard:

- AUC;
- AUPRC;
- Brier score;
- expected calibration error;
- phase-wise AUC.

Ranking:

- Spearman correlation;
- NDCG or placement-aware ranking metric;
- top-5 / top-10 accuracy where relevant.

Validity:

- edge-shuffle gap;
- node-shuffle gap;
- relation-drop delta;
- phase/map/mode/pressure slices.

## Paper Skeleton

```text
1. Introduction
2. Related Work
3. Battle Royale Graph Formalism
4. PUBG Team-Level Projection
5. Experimental Setup
6. Results
7. Discussion
```

Key distinction:

```text
Section 3 proposes the genre-level formalism.
Section 4 defines the PUBG empirical projection.
```

This separation is mandatory.
