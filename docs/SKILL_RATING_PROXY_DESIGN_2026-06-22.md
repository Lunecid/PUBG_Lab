# Skill Rating Proxy Design

Date: 2026-06-22

## Why The First Proxy Is Not Enough

Simple historical averages are not sufficient as tier proxies.

Example:

```text
A player who performs well in a weak lobby may not be stronger than a player
who performs moderately in a strong lobby.
```

Therefore, raw prior placement, kills, damage, or survival averages are only
outcome-history features. They do not define absolute skill.

## Status

This design is conditional. It should not be used as a main model feature unless
the dataset has enough repeated `player_id` coverage to make player ratings
stable. If PUBG API sampling is effectively random and repeated players are
rare, official tier/MMR should remain a future data-collection requirement.

## Conditional Current-Data Fallback

If repeated-player coverage is sufficient, use an opponent-adjusted latent
rating instead of simple historical averages.

Coverage audit:

```text
stdb/skill_coverage_audit.py
```

Run:

```bash
PG_PASSWORD=... python3 -m stdb.skill_coverage_audit --pretty
```

```text
chronological match stream
  -> pre-match player ratings
  -> team pre-rating
  -> lobby-adjusted expected placement
  -> rating update from observed placement
  -> next match sees only prior rating
```

Script:

```text
stdb/skill_rating_proxy_etl.py
```

Tables:

```text
stdb.player_skill_rating_proxy
stdb.team_skill_rating_proxy
```

Default policy:

```text
rating_same-mode_v1_min3_cov50_k32_s400
```

## Rating Logic

For each match, all team ratings are computed before any update.

```text
team_pre_rating = mean(pre_rating of roster players)
lobby_pre_rating = mean(team_pre_rating of all teams in the match)
observed_score = (n_teams - final_rank) / (n_teams - 1)
expected_score = mean_j logistic(team_rating_i - team_rating_j)
rating_delta = K * (observed_score - expected_score)
```

The update is applied after the rows for the match are recorded. Thus the
target match never contributes to its own pre-match rating.

## What It Solves

This handles the main objection:

```text
good result in weak lobby != good result in strong lobby
```

A high placement against high-rated teams increases rating more than the same
placement against low-rated teams. A poor result in a strong lobby is penalized
less than a poor result in a weak lobby.

## Remaining Limits

This is still not official PUBG tier/MMR.

It depends on:

- repeated player IDs;
- enough connectivity across matches;
- enough prior observations before each target match;
- the assumption that placement gives a useful noisy signal of skill.

If repeated player coverage is low, use this only as exploratory domain slicing
or calibration analysis.

## Recommended Use

Use the opponent-adjusted rating proxy only after coverage audit passes, for:

- per-rating-band metric reports;
- calibration slices;
- tier-conditioned model experiments;
- tier-specialist model experiments only if coverage is high enough.

Do not describe it as official tier.

Recommended wording:

```text
Because official ranked tier/MMR is unavailable in the collected telemetry, we
do not use tier as a main input channel. As an exploratory extension, we audit
repeated-player coverage and may derive an opponent-adjusted historical rating
from prior matches; this is not an official tier label.
```

## Run

```bash
PG_PASSWORD=... python3 -m stdb.skill_rating_proxy_etl --mode-scope same-mode --min-prior-matches 3 --min-coverage-ratio 0.5 --refresh
```

Alternative pooled-mode rating:

```bash
PG_PASSWORD=... python3 -m stdb.skill_rating_proxy_etl --mode-scope all --min-prior-matches 3 --min-coverage-ratio 0.5 --refresh
```
