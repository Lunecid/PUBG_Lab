# Caldera Dataset Status Redecision

Date: 2026-07-02

## Decision

Caldera remains a spatial affordance figure and case-study axis, not a second
phase-wise survival / rank prediction dataset.

Use it for:

- object-preserved 3D world reconstruction;
- player trajectory projection into terrain and built-mass context;
- terrain-affordance feature smoke tests;
- limited same-clock pairwise LOS experiments after v1 ray casting exists;
- descriptive statistics across the 36 ingested matches.

Do not use it for:

- phase-wise survival hazard training;
- final-rank prediction;
- safe-zone-constrained route claims;
- cross-title validation of PUBG combat, resource, team-viability, or zone
  mechanisms.

## Local DB Audit Result

Read-only PostgreSQL audit on 2026-07-02:

| Object | Count |
| --- | ---: |
| Caldera matches | 36 |
| Caldera agents | 4,187 |
| Caldera trajectory samples | 598,175 |
| Caldera outcomes rows | 4,187 |
| non-null `final_rank` | 0 |
| non-null `survived` | 0 |
| Caldera `zone_states` | 0 |
| `alive_state = in_play` samples | 594,025 |
| `alive_state = pre_spawn` samples | 4,150 |

Trajectory clock / redeploy audit:

| Metric | Value |
| --- | ---: |
| median in-play sample interval | 3.984 s |
| agents with `dt > 30s` gap | 2,460 |
| total `dt > 30s` gaps | 3,381 |
| median `dt` among `dt > 30s` gaps | 71.712 s |
| p99 `dt` among `dt > 30s` gaps | 450.144 s |
| median XY displacement among `dt > 30s` gaps | 330.962 m |
| p90 XY displacement among `dt > 30s` gaps | 1,192.242 m |
| `dt > 30s` and XY displacement `> 500m` gaps | 1,272 |

Interpretation:

```text
The public Caldera breadcrumb layer contains multi-life / redeploy-like
trajectory discontinuities. Death is not an absorbing state in this source.
```

## Why This Changes The Claim

The previous draft phrase:

```text
time-varying covariates for phase-wise survival and rank prediction
```

is only valid for the PUBG high-observability axis where phase clocks and
outcome labels exist.

For current Caldera, the safe phrase is:

```text
trajectory-grounded terrain-affordance measurement and visualization under
low-observability Battle Royale telemetry.
```

## Required Guardrails

Before any all-agent Caldera feature table:

1. assign `life_segment_id` using a conservative redeploy discontinuity rule;
2. suppress speed, heading-change, and forward-grade features across large
   observation gaps;
3. report gap and segment counts in every summary artifact;
4. calibrate tactical-state thresholds from the full 36-match distribution;
5. keep zone, rank, survival, combat, loot, and team-viability channels masked.

The current v0 extractor uses:

```text
motion gap: dt > 30s
redeploy candidate: dt > 30s and XY displacement > 500m
```

These are engineering guards, not official Warzone life labels.

## v1 Limit

The only recommended v1 expansion before the 2026-08 Tokyo ICE anchor is
pairwise LOS:

```text
same-match same-clock player positions
-> mesh ray cast
-> LOS / occlusion edge feature
```

This is allowed because the Caldera DB audit confirms same-match trajectories
exist on a shared clock. It still must be described as a spatial exposure /
visibility proxy, not as supervised survival modeling.

