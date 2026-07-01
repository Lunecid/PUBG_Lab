# PUBG Domain-Aware Design Layer (2026-06-22)

This document adds a PUBG-specific design layer on top of the input/model
channel contract.

Genre-level formalism:

```text
docs/BATTLE_ROYALE_GRAPH_FORMALISM_2026-06-22.md
```

This PUBG layer should be read as the empirical team-level projection of the
Battle Royale Dynamic Shrinking-Arena Competitive Graph (DSACG), not as the
full genre-level graph schema.

The channel contract answers:

```text
What type of tensor enters the model?
```

This domain layer answers:

```text
What PUBG survival mechanism does that tensor represent?
```

The two must be linked. A feature, edge, model block, ablation, or slide shape
is weak if it cannot be mapped to both an input channel and a domain mechanism.

Related contract:

```text
docs/INPUT_MODEL_CHANNEL_CONTRACT_2026-06-22.md
```

---

## 1. Core Domain Hypothesis

PUBG survival is not only a static ranking problem.

The working hypothesis is:

```text
Team survival is determined by the phase-conditioned interaction of
zone compression, rotation tempo, local contact, combat exchange,
resource readiness, roster integrity, relation topology, and lobby skill level.
```

Therefore, the model should not be described as "a GNN over PUBG teams" only.
It should be described as:

```text
A dynamic team survival model that encodes team state, zone pressure,
contact/combat relations, resource readiness, skill/lobby context, and phase
horizon.
```

---

## 2. Domain Mechanism Dictionary

The following mechanisms are the domain vocabulary that should be used in
feature manifests, ablations, architecture diagrams, and slide narration.

| ID | Domain mechanism | Meaning | Current / candidate signals |
| --- | --- | --- | --- |
| D0 | Population rule-set | The rules and geography that change the meaning of survival | map, mode, perspective, team size |
| D1 | Roster survival capital | How much survivable team capacity remains | alive count, health, alive ratio, knock/death pressure, team spread |
| D2 | Rotation tempo | How the team moves through shrinking phases | path length, speed, displacement, rotation timing, zone entry time, vehicle use |
| D3 | Zone / habitat pressure | How hard the shrinking playable area constrains the team | safe radius, zone gap, outside-safe ratio, boundary distance, crowding, migration deficit |
| D4 | Local opponent field | Who is physically near the team and how dense the neighborhood is | nearest enemy, enemy density, proximity degree, safe-entry congestion |
| D5 | Combat exchange | Whether the team applies or receives direct pressure | damage dealt/taken, kills, groggy, combat initiated, combat in/out edges |
| D6 | Resource readiness | Whether the team has enough equipment to survive future pressure | weapon/ammo/armor/heal/boost/pickup/use/carepackage/lootbox signals |
| D7 | Strategic behavior | Coarse behavioral style relative to prior work | FightScore, ActiveScore, Hunter/Sniper/Traveler/Camper type |
| D8 | Relation topology | Whether team-to-team edges carry useful structure beyond smoothing | proximity, combat, co-pressure, contact-gated relative advantage |
| D9 | Phase horizon | Survival risk changes by phase and cut horizon | phase index, phase duration, early/mid/late/late-game bands |
| D10 | Skill / lobby strength | Competition level changes the meaning of rotation, combat, and survival risk | official ranked tier, MMR bucket, lobby-strength proxy, historical skill bucket |

Design rule:

```text
Every model claim should name the domain mechanism it is testing.
```

For example, "graph helps" is too vague. The stronger claim is:

```text
Contact-gated relative advantage edges improve next-phase hazard prediction
over edge-shuffled null controls, suggesting that D8 relation topology carries
information when the edge semantics are domain-specific.
```

---

## 3. Mapping Domain Mechanisms To Input Channels

The existing C0-C8 channel taxonomy remains the input contract. The D0-D10
mechanisms are an overlay.

| Domain mechanism | Primary input channels | Notes |
| --- | --- | --- |
| D0 population rule-set | C0 | Usually used as slice/stratification first; optional model conditioning later |
| D1 roster survival capital | C1, C3, C6 | Team node state from player aggregation |
| D2 rotation tempo | C1, C3 | Snapshot motion is preferred; phase aggregates are legacy/baseline |
| D3 zone / habitat pressure | C1, C3, C4 | Needs both team-relative and global zone context |
| D4 local opponent field | C2, C3, C5 | Can be node exposure or explicit team-to-team edges |
| D5 combat exchange | C2, C3, C5 | Direction matters: outgoing pressure is not incoming pressure |
| D6 resource readiness | C3, C6 | Currently phase-level; future snapshot inventory quality would be stronger |
| D7 strategic behavior | C7 | Prior-work baseline and interpretability channel |
| D8 relation topology | C2, C5 | Must be tested against graph null controls |
| D9 phase horizon | C4, C8 | Conditions risk and evaluation, not a leakage source |
| D10 skill / lobby strength | C0 | Used for tier slices, optional conditioning, and tier-generalization tests |

Important:

```text
D mechanisms are not extra labels. They are semantic groupings for features,
edges, ablations, and model branches.
```

---

## 4. Feature Redesign Requirements

The following feature changes should be treated as design requirements before
the next full result run.

### FDR-0. Feature Clock Separation

Before semantic feature grouping, every feature must be assigned to its natural
update clock:

- match-static;
- 10-second snapshot state;
- 10-second snapshot relation;
- phase-level state/aggregate;
- phase-global context;
- phase-level relation;
- label/mask.

The same gameplay concept can appear in more than one clock stream, but it must
be named separately.

| Concept | Snapshot-clock version | Phase-clock version |
| --- | --- | --- |
| zone pressure | current distance to boundary at tick `t` | phase shrink duration / phase habitat summary |
| combat | recent damage edge in `(t-10, t]` | phase damage/kills totals |
| proximity | distance graph at tick `t` | phase contact/co-pressure relation |
| movement | instantaneous speed / short trend | phase path length / rotation timing |

Design requirement:

```text
message pooling follows feature clock: snapshot features use snapshot edges;
phase features use phase edges; cross-clock fusion is explicit.
```

### FDR-1. Map-Specific Scaling

Current code often uses a fixed arena scale such as `8160.0`.

Problem:

```text
Different maps have different playable sizes and rotation geometry.
```

Required design:

- store `map_scale_x`, `map_scale_y`, or a normalized playable-area scale;
- report if fixed scaling is retained as a simplification;
- evaluate by map to detect scale-induced artifacts.

Domain mechanisms:

```text
D0, D2, D3
```

### FDR-2. Roster-Normalized Team State

Raw team totals mix skill/action volume with roster size.

Required features:

- alive ratio = alive players / roster size;
- damage dealt/taken per alive player;
- item/heal/boost/resource counts per alive player;
- vehicle capacity or vehicle availability per alive player when possible.

Domain mechanisms:

```text
D1, D5, D6
```

### FDR-3. Rotation Urgency At Snapshot Resolution

The snapshot-first model should not only know where the team is; it should know
whether it can survive the next zone movement.

Required snapshot candidates:

- distance to current safe boundary;
- distance to next target safe center when causally known;
- required speed to reach safe area;
- observed speed;
- migration deficit = required speed - observed speed;
- outside-safe duration so far;
- edge pressure ratio.

Domain mechanisms:

```text
D2, D3, D9
```

### FDR-4. Contact-Gated Relative Edge Features

The earlier graph null result showed that raw topology can behave like
smoothing. The redesign should make edges carry domain meaning.

Required pairwise edge candidates:

```text
delta_position
delta_zone_pressure
delta_alive
delta_health
delta_speed
delta_resource_readiness
delta_recent_damage
delta_required_speed_to_safe
```

Gate:

```text
edge is active only under contact: proximity, recent combat, or safe-entry
competition.
```

Domain mechanisms:

```text
D4, D5, D8
```

### FDR-5. Third-Party And Crossfire Exposure

Battle royale risk is often indirect. A team can die because it rotates through
an active fight, not only because it initiated combat.

Required candidates:

- nearby combat events within radius and time window;
- damage events near path but not involving the team;
- number of distinct fighting pairs near the team;
- safe-entry congestion with active combat.

Domain mechanisms:

```text
D4, D5, D8
```

### FDR-6. Phase-Band Semantics

Early, mid, and late phases have different survival mechanisms.

Required phase bands:

```text
early: landing/looting/first contacts
mid: rotation and zone migration
late: compression, contact density, final placement lock-in
```

Use:

- metric slices;
- optional phase-band embeddings;
- phase-band ablations;
- separate calibration reports.

Domain mechanisms:

```text
D3, D6, D9
```

### FDR-7. Resource Readiness Beyond Counts

Counts are weak proxies. If available, future features should separate resource
quality from event volume.

Candidate upgrades:

- weapon class readiness;
- armor tier;
- heal/boost stock;
- ammo sufficiency;
- throwable availability;
- carepackage/high-value loot indicator.

Domain mechanisms:

```text
D6
```

### FDR-8. Causal Temporal Deltas

The snapshot model should encode short-term trend, not only instantaneous state.

Required candidates:

- delta health over last 10/30/60 seconds;
- delta alive count over last 10/30/60 seconds;
- delta distance to safe boundary;
- acceleration or speed trend;
- recent combat intensity over causal windows.

Domain mechanisms:

```text
D1, D2, D3, D5
```

### FDR-9. Skill-Tier Stratification

Survival patterns may not be stationary across competition level. Higher-tier
lobbies can have different rotation timing, contact density, third-party risk,
and calibration even under the same map/phase.

Required fields:

- `tier_source`: official_ranked, mmr, opponent_adjusted_history,
  historical_proxy_exploratory, unavailable;
- `tier_bucket`: coarse bucket used for split/reporting;
- `tier_policy`: pooled_no_tier, pooled_tier_conditioned, tier_specialist,
  tier_balanced, or leave_one_tier_out.

Leakage rule:

```text
Do not define tier from the current match's final rank, survival time, win/loss,
current-match kills/assists, `ranked_rp_proxy`, or any post-match outcome.
```

Design status:

```text
If official tier/MMR is unavailable, keep tier as future work unless repeated
player coverage is high enough. Raw prior placement/kills/damage do not define
absolute skill because lobby strength is itself variable; opponent-adjusted
ratings are only exploratory until coverage is audited.
```

Domain mechanisms:

```text
D0, D10
```

---

## 5. Domain-Aware Model Design

The target architecture should be described as multi-stream, even if the first
implementation is simpler.

### Proposed Streams

| Stream | Inputs | Domain mechanisms | Role |
| --- | --- | --- | --- |
| Snapshot team stream | C1 at 10-second clock | D1, D2, D3 | Encode tick-level team state |
| Snapshot relation stream | C2 at 10-second clock | D4, D5, D8 | Message passing over tick-level contacts |
| Phase team stream | C3 at phase clock | D1, D2, D3, D6 | Encode phase-level aggregates/context |
| Phase relation stream | C5 at phase clock | D4, D5, D8 | Message passing over phase-level relations |
| Global phase stream | C0/C4 | D0, D3, D9, D10 | Condition risk by map/mode/zone/phase |
| Tier/domain stream | C0 | D0, D10 | Optional conditioning or separate tier-specialist routing |
| Strategy stream | C7 | D7 | Prior-work baseline or auxiliary behavior signal |
| Output heads | C8 supervision | D9 | Predict hazard, rank, top-k, calibration |

### Recommended Flow

```text
10s team snapshots + 10s contact edges
  -> snapshot message-passing stream

phase team aggregates + phase relation edges
  -> phase message-passing stream

global phase and map/mode context
  -> context/gating stream

snapshot stream + phase stream + context stream
  -> phase-cut representation
  -> hazard / survival / rank / top-k heads
```

### Domain Gating

The global phase stream should be allowed to gate team and relation streams.

Examples:

- late phase increases the importance of local opponent density;
- high zone pressure increases the importance of rotation urgency;
- squad mode increases the importance of roster attrition and team spread;
- map/domain context changes the interpretation of movement distance.
- high-skill lobbies may change the expected value of early combat, rotation
  speed, and local contact density.

Design requirement:

```text
If a gate is used, the experiment must state which domain mechanism controls it.
```

---

## 6. Relation Design: What The Graph Means

The graph must not be sold as useful just because a GNN is present.

### Homogeneous GCN Baseline

Meaning:

```text
Neighbors are treated as generic exposure.
```

Use:

- baseline only;
- may capture smoothing but not relation semantics.

### R-GCN

Meaning:

```text
Proximity, combat, and co-pressure have different message weights.
```

Use:

- tests whether relation labels matter.

### Edge-Conditioned Relative-Advantage GNN

Meaning:

```text
The edge carries pairwise state: who is advantaged over whom under contact.
```

Use:

- best-aligned with PUBG domain structure;
- should be tested with edge-shuffle and node-absolute controls.

Acceptance rule:

```text
Topology signal exists only if real/unshuffled edges outperform graph nulls
under the same node features, split, seeds, and metric unit.
```

---

## 7. Domain-Aware Ablation Matrix

The next ablation plan should separate representation gains from architecture
gains.

### A. Domain Channel Ablation

| ID | Input | Question |
| --- | --- | --- |
| DA0 | D1 roster only | Is team health/alive state enough? |
| DA1 | DA0 + D2 rotation | Does movement timing add risk signal? |
| DA2 | DA1 + D3 zone pressure | Does zone/habitat explain survival? |
| DA3 | DA2 + D4 local opponent field | Does nearby opponent context matter? |
| DA4 | DA3 + D5 combat exchange | Does direct combat add signal? |
| DA5 | DA4 + D6 resource readiness | Does loot/readiness add signal? |
| DA6 | DA5 + D7 strategy scores | Does prior-work behavior abstraction add signal? |
| DA7 | DA6 + D8 relation topology | Does graph topology add signal beyond features? |

### B. Spatial Semantics Ablation

| ID | Spatial input | Question |
| --- | --- | --- |
| SP0 | no explicit spatial features | Non-spatial baseline |
| SP1 | absolute map coordinates only | Does raw position help? |
| SP2 | zone-relative features only | Is zone pressure the main spatial signal? |
| SP3 | enemy-relative/local density only | Is opponent field the main spatial signal? |
| SP4 | zone-relative + enemy-relative | Does domain spatial context beat raw coordinates? |
| SP5 | full spatial + map-specific scaling | Does map-aware scaling improve robustness? |

### C. Temporal Resolution Ablation

| ID | Temporal input | Question |
| --- | --- | --- |
| TR0 | phase aggregate only | Legacy/baseline phase resolution |
| TR1 | last snapshot at phase cut | Is the current state enough? |
| TR2 | 30s snapshot sampling | Coarser high-resolution baseline |
| TR3 | 10s snapshot sequence | Target minimum-resolution input |
| TR4 | 10s sequence + causal deltas | Does short-term trend help? |

### D. Relation / Topology Ablation

| ID | Graph input | Question |
| --- | --- | --- |
| RG0 | no graph | Feature-only baseline |
| RG1 | proximity only | Spatial exposure graph |
| RG2 | combat only | Direct interaction graph |
| RG3 | proximity + combat + co-pressure | Legacy relation set |
| RG4 | R-GCN separated relation channels | Do relation labels matter? |
| RG5 | contact-gated relative advantage | Does domain-specific edge meaning help? |
| RG6 | edge-shuffled null | Null topology control |
| RG7 | node-shuffled / node-absolute control | Tests whether edge signal is only node feature leakage |

### E. Domain Generalization Ablation

| ID | Split | Question |
| --- | --- | --- |
| DG0 | squad-only main result | Cleanest team-graph survival setting |
| DG1 | solo/duo/squad separate | Does the model work per rule-set? |
| DG2 | train on squad, test on duo/solo | Does learned structure transfer? |
| DG3 | leave-one-map-out | Does the model generalize across geography? |
| DG4 | FPP vs TPP slices | Does perspective change survival dynamics? |

### F. Head / Objective Ablation

| ID | Heads | Question |
| --- | --- | --- |
| HD0 | hazard only | Main survival objective |
| HD1 | hazard + rank | Does final placement supervision help? |
| HD2 | hazard + top-k | Does coarse placement supervision help? |
| HD3 | hazard + auxiliary survival curve | Does explicit survival regularization help? |
| HD4 | hazard + calibration loss/reporting | Does probability quality improve? |

### G. Tier-Aware Training Ablation

| ID | Tier policy | Question |
| --- | --- | --- |
| TS0 | pooled_no_tier | What is the baseline when all tiers are mixed? |
| TS1 | pooled_tier_conditioned | Does tier as C0 context improve risk estimation? |
| TS2 | tier_specialist | Do separate models per tier learn different dynamics? |
| TS3 | tier_balanced | Does balancing prevent majority-tier dominance? |
| TS4 | leave_one_tier_out | Does the representation generalize to unseen skill levels? |
| TS5 | pooled + per-tier calibration | Is ranking stable but probability calibration tier-dependent? |

---

## 8. Preprocessing Additions

Every preprocessing artifact should record:

- map/mode/perspective/team-size domain metadata;
- map scaling policy;
- phase-band assignment;
- feature window start/end for causal aggregation;
- whether a feature is snapshot or phase resolution;
- whether a feature is team-internal, global, or pairwise;
- whether a graph edge is directed;
- null-control generation policy.
- tier source, tier bucket, and tier construction timestamp/policy.

Leakage checks:

- no future combat events in snapshot features;
- no final rank or final survival columns in model input;
- no current-match final rank, survival time, win/loss, or top-k outcome in tier construction;
- no post-cut phase aggregate in snapshot-first input;
- no test-set survival rates in strategy-type features;
- no same-match rows split across train/validation/test.

---

## 9. Evaluation Additions

Report the existing metrics by these domain slices:

- phase band: early, mid, late;
- map;
- mode/team size;
- perspective if available;
- skill/lobby tier if available or derived;
- alive roster band;
- zone-pressure band;
- local opponent density band;
- relation condition: no graph, real edges, edge shuffle, node shuffle.

Minimum reporting table:

```text
model
input channels C*
domain mechanisms D*
temporal resolution
relation handling
skill/lobby tier policy
head set
overall metrics
phase-band metrics
map/mode slices
graph null-control deltas
```

---

## 10. Presentation Implications

The deck should not introduce the method as just "GNN layers".

Use this storyline:

```text
Problem:
  PUBG survival is a dynamic team risk problem under shrinking-space pressure.

Spatial definition:
  Spatial means map position, zone-relative pressure, local opponent field,
  and team-to-team contact relations.

Model:
  Snapshot-state stream + snapshot-relation stream
  + phase-state stream + phase-relation stream
  + global zone/phase context
  -> phase-cut survival heads.

Contribution:
  The design separates roster, rotation, zone, combat, resource, strategy,
  skill/lobby, and topology mechanisms, then tests each with controlled
  ablations.
```

Every box in the architecture figure should have one of these meanings:

- population/domain context;
- snapshot team-state tensor;
- snapshot relation tensor;
- phase team-state tensor;
- phase relation tensor;
- global zone/phase context;
- trainable encoder;
- fusion/gating operation;
- prediction head;
- loss/metric output.

---

## 11. Immediate Implementation Order

1. Add `domain_mechanism` labels to the machine-readable feature manifest.
2. Validate that every current feature maps to exactly one C channel and at
   least one D mechanism.
3. Add result metadata fields:
   - `input_channels`;
   - `domain_mechanisms`;
   - `temporal_resolution`;
   - `relation_handling`;
   - `head_set`.
4. Audit and extract official tier/MMR if available; otherwise define a
   leakage-safe proxy tier.
5. Add domain-sliced and tier-sliced evaluator outputs.
6. Add snapshot-level rotation urgency and causal temporal delta candidates.
7. Port contact-gated relative advantage edges to the snapshot tier.
8. Run the ablation matrix in this order:
   - feature/channel ablations;
   - temporal-resolution ablations;
   - relation/null-control ablations;
   - tier-aware training ablations;
   - domain-generalization ablations;
   - head/objective ablations.

---

## 12. Current Result Caveat

This document defines the next design.

It does not retroactively change the meaning of existing results.

Current legacy results should still be described as:

```text
10-second snapshots are encoded first, then mean-pooled to phase-level
representations. Main prediction heads operate at phase resolution.
```

The domain-aware snapshot-first design requires new runs before it can be
claimed as the final empirical result.
