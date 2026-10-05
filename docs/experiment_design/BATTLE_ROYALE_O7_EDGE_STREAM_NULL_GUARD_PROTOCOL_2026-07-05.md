# Battle Royale O7 Edge-Stream Null-Guard Protocol

Date: 2026-07-05
Class: `ExperimentContract`
Lifecycle: `active`

## Purpose

This document locks the next O7 relation-structure experiment before any new
graph-relation run is launched.

The question is:

```text
Do observed team-to-team relation edges add predictive signal beyond OM5 node
summaries, or does the graph encoder mostly provide capacity / smoothing?
```

The protocol follows the same discipline as the corrected O6Z sensitivity:
define the support and bounded-negative interpretations before seeing the
result.

## Decision

The headline O7 edge stream is:

```text
contact-gated relative-advantage edges
```

Current code surface:

```text
model_stgcn/v3/advantage_edges.py
model_stgcn/v3/adv_model.py
scripts/adv_precompute.py
```

The advantage edge feature vector is typed and interpretable:

```text
g_combat
g_knn
affinity
d_over_ell
combat_dmg_log
combat_kills
dz
dalive
dhealth
dzone
```

This is preferable as the headline O7 stream because it tests relation type,
not only graph existence.  Proximity, combat, and co-pressure edge streams are
secondary decomposition or sensitivity arms.

## Claim Boundary

Allowed:

```text
Observed advantage-edge topology carries O7 signal if it beats an edge-shuffled
null under the same nodes, model class, edge gate, and training protocol.
```

Avoid:

```text
A graph encoder improves performance, therefore observed relation structure
matters.
```

Reason:

```text
Adding graph edges also changes model capacity and smooths node states.  Only
B-C below isolates observed edge structure from the edge-shuffled null.
```

## Three-Arm Contract

The headline comparison is a three-arm design.

| Arm | Input | Model interpretation | Result role |
| --- | --- | --- | --- |
| `A` | OM5-equivalent node/global features, no edge stream | node-only baseline | reused or reproduced baseline |
| `B` | same nodes + graph encoder + observed advantage edges | observed relation stream | candidate O7 signal |
| `C` | same as `B` + `edge_shuffle` | density/feature null | primary null |

Primary contrast:

```text
B - C
```

Interpretation:

```text
observed structure-specific relation signal
```

Secondary contrast:

```text
B - A
```

Interpretation:

```text
total graph-encoder gain, including capacity and smoothing
```

Do not use `B-A` alone as evidence that relation topology matters.

## Null-Control Rule

The required null is:

```text
edge_shuffle within each match-phase alive-node block
```

It should preserve:

- the active alive-team set for the match-phase;
- the edge density / number of active edges;
- the edge feature-vector distribution where the implementation supports
  feature-vector shuffling;
- the same node features and labels;
- the same split and seed.

Before a full O7 run, perform a short audit:

```text
Verify that edge_shuffle rewires only within the active-node mask and preserves
the intended edge-vector distribution for the advantage-edge implementation.
```

If the audit fails, fix the control before running the required cohorts.

## U-Layer Decision

U-layer global graph summaries are excluded from this run.

Reason:

```text
Raw U metrics are mechanically coupled to alive-team count N(k) and edge count
M(k).  Including them would change the experiment from an edge-stream test into
a node/global feature extension test.
```

Locked rule:

```text
U-layer graph summaries enter only after residualization / null adjustment
against alive-team count and edge count.
```

For this protocol, exclusion is the guard.

## Cohort Rule

Headline training remains separated by exact map / perspective / team-size
cohort.  No pooled training is allowed for headline interpretation.

Required final cohorts:

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

Execution order may prioritize high-power readout:

1. Erangel TPP squad;
2. Taego TPP squad;
3. Erangel TPP solo;
4. Miramar / Desert TPP squad;
5. remaining required cohorts.

Final claims require the 13-cohort set unless the run is explicitly labeled as
pilot or early readout.

## Confirmatory Targets

Primary confirmatory cells:

| Contrast | Phase group | Metrics |
| --- | --- | --- |
| `B - C` | early `1-2` | hazard AUC, rank Spearman |
| `B - C` | mid `3-4` | hazard AUC, rank Spearman |

Secondary / exploratory:

- `B - A` for total graph-encoder gain;
- top-10 AUC in early and mid phase groups;
- late `5-6` and phase `7` relation effects;
- dense or small-map patterns such as Karakin TPP squad and Paramo TPP squad;
- phase `6-7` relation-lift patterns.

Late and small-map effects are useful for interpretation, but they are not the
headline confirmatory evidence unless explicitly re-registered.

## Pre-Registered Interpretation

Strong O7 support:

```text
B-C is positive in at least 9/13 cohorts for at least one primary confirmatory
phase-group / metric cell, and paired match-cluster bootstrap supports the
same cell in enough large cohorts to rule out single-cohort noise.
```

Bounded negative:

```text
B-C does not reach the 9/13 sign threshold.  The observed advantage-edge
topology is not robustly supported as a headline O7 signal in the current PUBG
configuration.
```

Capacity / smoothing outcome:

```text
B-A > 0 but B-C approximately 0.
```

Interpretation:

```text
The graph encoder helps as capacity or smoothing, but observed relation
structure does not carry topology-specific signal.
```

This is an expected and publishable boundary result if it occurs.

## Relationship To O6Z

The corrected O6Z result found that PUBG `pos_z`-only node/window proxies do
not robustly add signal after OM5.

The O7 advantage-edge experiment asks a different question:

```text
Can pairwise relative state under contact carry signal that node/window
summaries do not?
```

If O7 also fails, the paper still supports the central pressure-habitat
mechanism and bounds relation topology as secondary or conditional.

If O7 succeeds, the correct claim is:

```text
Relation type matters: relative-advantage edges can carry observed structure
signal where ordinary interaction topology behaves like smoothing.
```

## Pre-Run Blocker

Before launching a long O7 run, resolve the gen-1 / gen-2 result boundary:

```text
Which older advantage-edge and core_habitat null-control results are reusable
as manuscript evidence, and which must be rerun under the current cohort /
feature / split contract?
```

This is a manuscript blocker because several narrative claims refer to legacy
or preliminary results.  The boundary decision should be a short result
registry, not an open-ended redesign.

## Expected Artifacts

| Artifact | Role |
| --- | --- |
| edge-shuffle audit | proves the null control preserves the intended quantities |
| required-cohort advantage-edge cache or equivalent | DB-free training surface |
| A/B/C run root | one result per exact cohort and arm |
| phase-group delta table | early/mid confirmatory result |
| paired bootstrap if sign threshold passes | uncertainty support |
| result document | claim boundary for O7 |

## Post-Run Status

The gen-1 / gen-2 boundary note is available:

```text
docs/experiment_design/BATTLE_ROYALE_GEN1_GEN2_RESULT_BOUNDARY_2026-07-05.md
```

The edge-shuffle audit and cache/run smoke are available:

```text
docs/experiment_design/BATTLE_ROYALE_O7_EDGE_SHUFFLE_AUDIT_2026-07-05.md
```

The Erangel and four-cohort early readouts are available:

```text
docs/experiment_design/BATTLE_ROYALE_O7_EARLY_READOUT_ERANGEL_2026-07-05.md
docs/experiment_design/BATTLE_ROYALE_O7_EARLY_READOUT_4COHORT_RESULT_2026-07-05.md
```

The full 13-cohort hazard-AUC and hazard-implied rank bridge results are
available:

```text
docs/experiment_design/BATTLE_ROYALE_O7_REQUIRED_COHORT_HAZARD_RESULT_2026-07-05.md
docs/experiment_design/BATTLE_ROYALE_O7_REQUIRED_COHORT_RANK_BRIDGE_RESULT_2026-07-05.md
```

Post-run interpretation:

```text
As supplementary PUBG relation-topology evidence, observed contact-gated
relative-advantage edges clear the edge-shuffle null in hazard AUC and in a
hazard-implied rank bridge.
```

Boundary:

```text
The rank bridge uses Spearman(final_rank_norm, hazard_prob).  The O7
advantage-edge model has no independent rank head.
```

Manuscript boundary:

```text
This is not a core research-question experiment.  Use it as a supplementary
null-control / reviewer-defense result.
```
