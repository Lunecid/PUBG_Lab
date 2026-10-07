# Section III draft v1 (2026-10-06): Telemetry Data and Pipeline

사용자 결정: 데이터 설명을 방법론의 독립된 절로 둔다. 10초 간격의 경기 내 telemetry는 드물다는 것이 이유다.

## 논문 구성 (갱신)

| 장 | 내용 |
|---|---|
| I | Introduction |
| II | Related Work |
| III | Telemetry Data and Pipeline (방법론 1) |
| IV | Formulation and Models (방법론 2): 서식지 손실 hazard, 표현, 참조 모델, 선행 설계 적용, 분해 |
| V | Evaluation Protocol |
| VI | Results |
| VII | Discussion and Conclusion |

## Outline (III)

- A. Telemetry corpus: 무엇이 기록되는지와 왜 드문지, 규모
- B. From logs to samples: 정규화, 시계 정렬, 단계 분할, 인과 컷, 사건 기반 라벨과 감사
- C. Benchmark: 모집단 정제, 코호트, 분할, 텐서와 그래프

## Draft

**III-A · Telemetry corpus (opening + rarity).** PUBG exposes per-match telemetry through a public API [API]. The telemetry logs every player's position at regular ten-second intervals, together with timestamped events: damage, knocks, kills, item pickups, equips and uses, game-state updates of the safe zone, phase changes and landings. In our corpus, 87.7% of consecutive position samples of a player are exactly 10 s apart and 99.94% are at most 11 s apart. Telemetry of this resolution is rarely available in public form. Widely used game datasets aggregate statistics after the match [Kaggle-PUBG], and the in-match logs used by prior battle royale predictors are proprietary [KRAFTON, Sagong25a]. The API also keeps match data only for a limited period [VERIFY: retention window], so a corpus has to be accumulated by continuous collection. We collected matches from 2026-03-26 to 2026-06-09. Of these, 27,491 matches have complete telemetry. Together they hold about 470 million records: 155 million position samples and 314 million events (Table I). The largest event types are damage (130M), item pickups (87M) and item equips (69M), and the stored data occupy 261 GB.

**III-B · From logs to learning samples (design + why).** Four properties of the raw logs prevent their direct use as training samples, and we handle each explicitly.

- *Clock alignment.* Phase-change events carry wall-clock timestamps, whereas positions carry elapsed match time, and the two clocks differ per match. The offset has a median of 5.7 s and a maximum of 41.9 s. We estimate it from game-state events and express all streams on one clock.
- *Causal cut.* For each phase boundary k, every feature and every relation uses only records time-stamped at least 10 s before the boundary on the unified clock. No input can therefore observe the outcome window.
- *Event-based labels.* A team's elimination time is the last kill event among its members, and winners are censored. Knocked players are counted as alive, and players recalled after death are handled by taking each member's last kill event. A naive rule that infers survival from the team's last position snapshot mislabels 11.4% of team-phase rows. Its positive precision is 70.4%, mainly because snapshots that show only knocked teammates are read as eliminations.
- *Coverage.* At each cut, 99.4% of living players have a position sample within the previous 15 s.

**III-C · Benchmark construction.** We remove two populations whose dynamics differ from regular play:

- `airoyale` matches, in which about 92% of accounts are bots;
- the Vikendi cohort, whose logs include about 30 non-player combatant accounts per match.

We keep cohorts with at least 200 matches. A cohort is defined by map, perspective (TPP/FPP) and team size (solo, duo, squad). Bots remain in regular lobbies, so each team's share of AI accounts is kept as an input feature. The resulting benchmark has 13,948 matches in 12 cohorts. Matches are split 80/10/10 per cohort, giving 11,188/1,376/1,384 matches, and the 829,640 risk-set rows are (match, team, phase-boundary) triples. For each cut we store:

- a 150-dimensional team feature vector;
- the team's centroid;
- the current and next safe-zone geometry;
- typed team–team relations: proximity, directed combat, and zone co-pressure.

An optional player layer adds 1.24 million player records with teammate, enemy-proximity and combat edges.

**Table I (draft): Telemetry corpus (27,491 matches, 2026-03-26 to 2026-06-09)**

| Record type | Records |
|---|---:|
| Position samples (10 s) | 155.4M |
| Damage events | 129.6M |
| Item pickups | 87.0M |
| Item equips | 68.5M |
| Item uses | 19.8M |
| Game-state (zone) updates | 4.0M |
| Landings | 2.8M |
| Kills | 2.0M |
| Total | ≈470M (261 GB) |

The figures are planner estimates (pg_class reltuples). Exact counts can be confirmed with COUNT queries before submission if needed.

## Intro P2 축약안 (III장으로 상세를 옮긴 뒤)

> Battle royale games make this setting observable at scale. In PUBG: BATTLEGROUNDS, up to 100 players compete on a large map whose safe zone shrinks in scheduled phases, and the last team standing wins. Its public telemetry records every player's position every ten seconds together with combat, item and zone events. Such in-match logs are rarely available at this resolution: our corpus spans about 470 million records from 27,491 matches (Section III). Turning them into reliable learning samples requires care. Clocks must be aligned, knocks must be separated from eliminations, and no feature may cross a phase boundary.

## Notes

- **[VERIFY] API 데이터 보존 기간**: 공식 문서에서 확인해야 한다(14일로 알려져 있음). 확인하지 못하면 문장을 삭제한다.
- **[Kaggle-PUBG]**: "PUBG Finish Placement Prediction" (2018) 경기 후 집계 데이터. 학위논문 bib에 있다.
- **용어**: 데이터 설명에서는 "knock" 같은 게임 용어가 불가피하므로 첫 등장 시 정의한다.
- **분량**: III장 약 0.9쪽 (표 포함).
