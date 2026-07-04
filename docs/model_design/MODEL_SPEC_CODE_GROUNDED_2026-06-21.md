# V3 Model — code-grounded spec (single source for Methods slides)

Verified from code (file:line) by Explore pass 2026-06-21. Use these EXACT facts on Methods slides; do not round/guess.

## Factorized inputs (instances / tensors)
- **Phase tier**: `T=8` (`stdb/queries.py:59`), `N_MAX=128` (`:63`), `E=3` channels `["proximity","combat","copressure"]` (`:64`).
  - `X_node[T=8, N=128, 116]` — node features (`factorized_data.py:90`; node_dim read at build `train.py:77`).
  - `G[T=8, 25]` — global/phase context, separated from node (broadcast cols), `factorized_data.py:67-91`; 25 = 12 base + 13 habitat globals (`spec.py:17-49`).
  - `A_phase[T=8, E=3, N=128, N=128]`, **row-normalized per relation** (`queries.py:748-752`); proximity gets self-loops, combat/copress interaction-only.
- **Snapshot tier (Tier-S)**: resolution **10 s**, `SNAP_T_MAX=200` (`snapshot_queries.py:34`).
  - `X_s[T_s≤200, N=128, 13]` (`SNAP_FEATURE_DIM=13`, `:38-54`).
  - `A_snap[T_s, E=2, N, N]`, channels `["proximity","combat"]` — **NO copressure at snapshot tier** (`:35`), row-normalized.
- Total feature pool: base 66 (`PHASE_FEATURE_DIM=66`) + V2 75 = **141** (`PHASE_FEATURE_DIM_V2=243`); split → 25 global / 116 node.

## Node features (X_node, 116-d) — presentation groups (code: `spec.py:59-86`, `queries.py:72-240`)
- Team state (alive players/health/spread/vehicle), Trajectory (path, displacement, sinuosity, speed, safe-alignment), Zone exposure (boundary dist, outside-safe ratio, local enemy density), Combat as node-attribute (damage/kills/groggies/distinct opponents), Loot/readiness, Network role (`net_*`: combat/prox/copress degree·strength·reciprocity), Local habitat pressure (`hab_*`: shrink rate, speed deficit, edge pressure, local competition).
- Note: groups are PRESENTATION-level; `CONVENTIONAL` preset slice `[42:65]` drops idx65 `lootbox_pickup_count_norm`.

## Edge channels (heterogeneous) — exact construction
- **proximity** (symmetric): `w = exp(-avg_distance_m / 200)` (`phase_edge_etl.py:109`).
- **combat** (directed attacker→victim): `w = ln(1 + max(0,damage) + 100·kills + 50·groggies)` (`:157-159`). Default single-directed; `combat_in` transpose channel OFF by default.
- **co-pressure** (symmetric, zone-gated): active only when both teams `under_pressure` (outside safe OR ≤150 m from boundary) AND pairwise ≤200 m; `w = ratio · exp(-d/200)` (`:188-247`). **Phase tier only.**

## Architecture (no nn.Embedding — continuous + Linear projection)
- **NodeGlobalEncoder** (`models.py:49-88`): node `Linear(116→hidden)→ReLU→Drop`; global `Linear(25→hidden)`; fuse by concat / add / **FiLM** (`h·(1+tanh(γ))+β`). hidden=64, mid=128 (`train.py:359`).
- **Snapshot encoder** (`snapshot_encoder.py:21-52`): `Linear(13→48)` + 2× DynamicSTBlock (causal gated TCN + 2-relation gated R-GCN) → out 48; pooled (masked mean) to phase.
- **Tier fusion = CONCAT**: `X_aug = [X_node ; pooled_snapshot(48)]` (`models.py:425`).
- **Phase encoder**: 2× GraphTemporalBlock = causal gated TCN (Conv2d kernel=3, GLU) → graph layer → TCN; LayerNorm, Dropout0.1, residual ReLU.
- **Graph encoders** (`GRAPH_ENCODERS`): `none` (skip only) / `gcn` / `sage` / `gat` (1-head additive attn) / **`rgcn`** (one Linear per relation + softmax relation gate). GCN/SAGE/GAT use ONE summed+renormalized combined adjacency; **only R-GCN is per-relation** (`models.py:40-46` vs `141-156`).
- Snapshot-tier graph is FIXED 2-relation gated R-GCN regardless of `--graph-encoder` (which swaps only the phase encoder).

## Prediction heads (causal: read `h[:, :-1]`, phases 1..T−1)
- **hazard** (next-phase elimination): MLP `64→32→1` logits → `(B,7,N)` (`models.py:307-312`).
- **rank** (final placement): optional `TeamSetDecoder` (`mlp` passthrough or **`attention`** = 1-layer TransformerEncoder, 4 heads, compares teams within a cut) → MLP `64→32→1` (`:204-318`).
- topk head → `[top5,top10]`; MR survival head `48→24→1` (snapshot deep-supervision).

## Training / labels
- Loss: hazard BCE-with-logits (masked) ·α=1 + rank **L1** ·β=0.3 + topk BCE ·γ=0.3 (+ pairwise rank hinge for attn/farnet ·0.1) + survival BCE ·δ=0.3 (`models.py:438-573`, `train.py:364-370`).
- **At-risk mask** `target_defined = alive_at_phase_end AND next_phase exists` applied to all phase losses (`queries.py:665-686`, `rq1_feature_etl.py:686-696`). hazard idx t ↔ cut_phase t+1.
- AdamW lr1e-3 wd1e-4, CosineAnnealing, clip5, epochs12 batch8 seed42.
- Leakage guard: `FORBIDDEN_INPUT_FEATURES` exact-name blocklist (`spec.py:91-108`), `assert_no_forbidden_input_features`; team order sorted by team_id (outcome-free); snapshot combat bucketed to past window (t−10,t]; eval restricted to at-risk cells per cut phase.

## Model variants (config rows on ONE class `FactorizedMRGNN`; `experiment_matrix.py:41-74`)
mr_tcn (encoder none) · mr_gcn · mr_sage · mr_gat · mr_rgcn · mr_rgcn_film (FiLM global) · mr_rgcn_attn_rank (attention decoder + pairwise0.1) · mr_farnet (FiLM + attention + pairwise). No separate FARNet module — flags only.

## Dynamic Heterogeneous Graph — the decomposition (for the framing slide)
- **Heterogeneous** = E=3 distinct relation types (proximity/combat/co-pressure), each constructed differently, kept as SEPARATE channels → multi-relational; R-GCN convolves each separately.
- **Dynamic** = adjacency rebuilt every phase `A[T=8,E=3,N,N]` + 10 s snapshot tier `A_snap[T_s,E=2,N,N]`; structure evolves over the match (combat→co-pressure); causal gated-TCN temporal encoder reads the evolution.
