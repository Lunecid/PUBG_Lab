# Feature Clock Channel Redesign

Date: 2026-06-22

## Core Correction

The input design must start from the feature's update clock, not only from its
semantic meaning.

Previous grouping:

```text
roster / movement / zone / combat / resource / relation
```

Corrected grouping:

```text
what clock does this information actually change on?
```

Then each feature receives both:

```text
feature_clock + semantic_domain
```

Example:

```text
zone pressure at 10-second tick    -> snapshot clock + zone/habitat domain
zone shrink phase duration         -> phase clock + phase/global domain
damage in last 10 seconds          -> snapshot clock + combat domain
damage aggregated across phase     -> phase clock + combat domain
```

## Feature Clock Taxonomy

| Clock | Tensor / Channel | Update unit | Examples | Model treatment |
| --- | --- | --- | --- | --- |
| match-static | `C0` | match | map, mode, perspective, team_size | conditioning/slice only |
| snapshot-state | `C1` | 10-second tick | position, health, alive count, speed, zone-relative distance | snapshot stream node input |
| snapshot-relation | `C2` | 10-second tick | proximity, recent combat, contact-gated local relation | snapshot stream message passing |
| phase-state | `C3` | phase cut/window | phase trajectory, phase combat totals, phase resource summaries | phase stream node input |
| phase-global | `C4` | phase cut | phase index, phase duration, safe-zone geometry, alive population | context/gating |
| phase-relation | `C5` | phase cut/window | phase proximity, combat, co-pressure | phase stream message passing |
| individual-derived | `C6` | inherited from source window | roster/player aggregates | must declare inherited clock |
| prior-work strategy | `C7` | phase/window | FightScore, ActiveScore, strategy type | baseline/auxiliary |
| label/mask | `C8` | phase cut or snapshot tick | hazard, survival, censoring, masks | supervision only |

## Required Metadata

Every feature must declare:

```text
feature_name
clock: match_static | snapshot_10s | phase_cut | phase_window | label
semantic_domain: roster | movement | zone | combat | resource | relation | global | strategy
entity: team | pair | global | label
causal_window: instant | trailing_10s | trailing_30s | phase_to_date | phase_window | match_static
model_stream: snapshot_node | snapshot_edge | phase_node | phase_edge | global_context | label
```

No feature is allowed to enter a tensor only because it has the same semantic
theme as another feature. Its update clock decides the stream first.

## Why This Matters

If phase features and 10-second features are concatenated too early, the model
does not know whether a value is:

- a tick-level state;
- a phase aggregate;
- a global context variable;
- a pairwise relation;
- or a target/mask.

That makes message passing ambiguous. A 10-second proximity edge and a phase
co-pressure edge should not be treated as the same pooling object simply because
both are "spatial".

## Corrected Model Principle

Message pooling follows the clock:

```text
snapshot clock:
  X_s[t], A_s[t]
  -> snapshot message passing at 10-second resolution
  -> h_s[t, team]

phase clock:
  X_p[k], A_p[k], G[k]
  -> phase message passing at phase-cut resolution
  -> h_p[k, team]

cross-clock fusion:
  h_s[<= cut k] + h_p[k] + G[k]
  -> phase-cut hazard / survival / rank heads
```

The fusion step may summarize snapshot history, but it must do so explicitly:

- last snapshot before cut;
- causal attention over snapshots up to cut;
- learned temporal pooling;
- or phase-local mean pooling as a baseline.

Mean pooling is allowed only when the experiment names it as a baseline, not as
the default meaning of "10-second input".

## Current Legacy Model Interpretation

The legacy MR model partially follows this principle:

```text
X_s, A_s -> snapshot ST-GNN -> mean-pool to phase
X_node, G, A + pooled snapshot embedding -> phase ST-GNN -> heads
```

Therefore it has two input clocks, but the final prediction path is phase-level
after snapshot-to-phase pooling.

It should be described as:

```text
pooled multi-resolution phase-head model
```

not:

```text
fully snapshot-resolution survival model
```

## Required Redesign

Before the next full run:

1. Build a machine-readable feature manifest with `clock` as a required field.
2. Move every current feature into exactly one clock stream.
3. Split semantic groups that currently mix clocks.
4. Add validation that snapshot-first models cannot silently consume phase
   aggregates.
5. Add ablations that compare:
   - phase-only aggregates;
   - last snapshot at cut;
   - 10-second sequence with explicit snapshot message passing;
   - 10-second sequence + phase context;
   - early mean-pool baseline vs late fusion.

## Example Reassignment

| Feature family | Old interpretation | Correct clock-aware interpretation |
| --- | --- | --- |
| position | spatial feature | snapshot-state if per 10s; phase-state if phase aggregate |
| zone distance | zone feature | snapshot-state for current distance; phase-global for zone schedule |
| combat | combat feature | snapshot-relation for recent attack edge; phase-state for phase totals |
| proximity | graph edge | snapshot-relation or phase-relation depending on construction window |
| resource | team feature | phase-state unless inventory snapshots are extracted |
| strategy type | behavior feature | phase/window prior-work channel, not raw snapshot input |

## Presentation Wording

Use:

```text
We separate features by their natural update clock: match-static context,
10-second team state, 10-second relations, phase-level aggregates, phase-global
context, and phase-level relations. Message passing is performed inside each
clock stream before explicit cross-resolution fusion.
```

Avoid:

```text
We use 10-second and phase features together.
```

That wording hides the actual architectural requirement.
