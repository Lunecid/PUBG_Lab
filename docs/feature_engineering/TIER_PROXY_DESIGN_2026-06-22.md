# Tier Proxy Design

Date: 2026-06-22

## Status

This document describes the first historical outcome proxy. It is useful for
exploration only, but it is not sufficient as an absolute tier proxy because it
does not adjust for lobby strength and requires repeated player coverage.

Conditional tier-like current-data fallback:

```text
docs/SKILL_RATING_PROXY_DESIGN_2026-06-22.md
stdb/skill_rating_proxy_etl.py
```

## Purpose

The current PUBG telemetry dataset does not contain official ranked tier or MMR.
Instead of discarding the collected data, we add a leakage-safe historical skill
proxy:

```text
player prior match history before target match
  -> player historical strength
  -> team-level skill/lobby proxy
  -> tier-aware slices, conditioning, or specialist models
```

This is not an official PUBG tier. It is a simple outcome-history proxy, should
not be described as absolute player strength, and should not enter the main
model.

## What The Proxy Can And Cannot Claim

Can claim:

- whether model performance/calibration differs across historical skill bands;
- whether tier-conditioned or tier-specialist models help;
- whether survival dynamics differ by lobby strength proxy.

Cannot claim:

- official Bronze/Silver/Gold/etc. rank;
- exact MMR at match start;
- causal effect of official matchmaking tier.
- absolute skill independent of lobby strength.

## Leakage Rule

For target match `m`, proxy features may use only player rows before `m` in the
chronological key `(match_created_at, match_id)`. The target match row itself is
excluded.

Forbidden inputs:

- current-match final rank;
- current-match survival time;
- current-match kills, assists, damage;
- current-match win/top-k;
- existing `ranked_rp_proxy`, because it is made from current-match outcome
  ingredients.

## Implementation

Script:

```text
stdb/skill_proxy_etl.py
```

Tables:

```text
stdb.player_skill_proxy
stdb.team_skill_proxy
```

Default policy:

```text
historical_same-mode_v1_min3
```

Meaning:

- player history is partitioned by `player_id, game_mode`;
- a player contributes only after at least 3 prior matches;
- team proxy is the average prior strength across eligible roster players;
- `team_skill_bucket_q_global` is an exploratory global quantile bucket.

Run:

```bash
PG_PASSWORD=... python3 -m stdb.skill_proxy_etl --mode-scope same-mode --min-prior-matches 3 --refresh
```

Alternative pooled-history policy:

```bash
PG_PASSWORD=... python3 -m stdb.skill_proxy_etl --mode-scope all --min-prior-matches 3 --refresh
```

## Strength Formula

For each player, prior-only summaries are computed with a SQL window ordered by
`(match_created_at, match_id)` and ending at `1 PRECEDING`.

```text
prior_strength =
  0.35 * prior_placement_pct
+ 0.20 * prior_top10_rate
+ 0.20 * normalized_prior_damage
+ 0.15 * normalized_prior_kills
+ 0.10 * prior_survival_pct
```

Damage and kills are normalized with a saturating exponential transform so that
extreme outliers do not dominate the proxy.

## Model Use

Recommended order:

1. Use this only as an exploratory historical-outcome slice.
2. Run repeated-player coverage audit before any tier-like analysis.
3. Prefer opponent-adjusted rating from `skill_rating_proxy_etl.py` only if
   repeated-player coverage is sufficient.
4. Add per-rating-band calibration curves only after coverage is checked.

For final locked evaluation, prefer either:

- continuous `team_prior_strength_raw`; or
- train-split-fitted tier buckets.

The global quantile bucket in the ETL table is useful for exploration and slide
diagnostics, but final experiments should avoid fitting bucket thresholds on
the full evaluation population.
