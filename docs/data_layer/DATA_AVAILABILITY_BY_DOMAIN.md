# Data Availability by Perspective, Map, and Team Size

이 문서는 raw fetched match를 논문/재실험에 바로 사용할 수 있는 phase/graph eligible 데이터로 나누기 위해 생성한 availability 요약이다.

## Overall Funnel

| raw_fetched | base_done | base_with_team_positions | phase_feature_done | target_label_done | edge_cache_done | graph_eligible_n128 | not_phase_featured | missing_edges_after_features |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27,491 | 21,833 | 21,830 | 21,830 | 21,824 | 21,830 | 21,824 | 5,661 | 0 |

## Perspective x Team Size

| perspective | team_size | raw_fetched | phase_feature_done | target_label_done | edge_cache_done | graph_eligible_n128 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TPP | squad | 15,220 | 15,219 | 15,213 | 15,219 | 15,213 |
| TPP | duo | 2,273 | 2,272 | 2,272 | 2,272 | 2,272 |
| FPP | squad | 1,712 | 1,712 | 1,712 | 1,712 | 1,712 |
| TPP | solo | 5,520 | 1,432 | 1,432 | 1,432 | 1,432 |
| FPP | duo | 783 | 783 | 783 | 783 | 783 |
| FPP | solo | 412 | 412 | 412 | 412 | 412 |
| TPP | clansolo | 1,571 | 0 | 0 | 0 | 0 |

## Map x Perspective x Team Size: Graph-Eligible Domains

| map_name | perspective | team_size | raw_fetched | phase_feature_done | target_label_done | edge_cache_done | graph_eligible_n128 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baltic_Main | TPP | squad | 6,206 | 6,205 | 6,204 | 6,205 | 6,204 |
| Tiger_Main | TPP | squad | 4,160 | 4,160 | 4,159 | 4,160 | 4,159 |
| Desert_Main | TPP | squad | 1,545 | 1,545 | 1,543 | 1,545 | 1,543 |
| Neon_Main | TPP | squad | 1,041 | 1,041 | 1,039 | 1,041 | 1,039 |
| Savage_Main | TPP | squad | 940 | 940 | 940 | 940 | 940 |
| Baltic_Main | TPP | solo | 801 | 800 | 800 | 800 | 800 |
| Baltic_Main | TPP | duo | 656 | 656 | 656 | 656 | 656 |
| Tiger_Main | TPP | duo | 574 | 573 | 573 | 573 | 573 |
| DihorOtok_Main | TPP | squad | 498 | 498 | 498 | 498 | 498 |
| Tiger_Main | FPP | squad | 435 | 435 | 435 | 435 | 435 |
| Baltic_Main | FPP | squad | 430 | 430 | 430 | 430 | 430 |
| Chimera_Main | TPP | squad | 333 | 333 | 333 | 333 | 333 |
| Summerland_Main | TPP | squad | 333 | 333 | 333 | 333 | 333 |
| Desert_Main | FPP | squad | 288 | 288 | 288 | 288 | 288 |
| Neon_Main | FPP | squad | 264 | 264 | 264 | 264 | 264 |
| Savage_Main | TPP | duo | 243 | 243 | 243 | 243 | 243 |
| Desert_Main | TPP | duo | 241 | 241 | 241 | 241 | 241 |
| Tiger_Main | FPP | duo | 199 | 199 | 199 | 199 | 199 |
| Baltic_Main | FPP | duo | 186 | 186 | 186 | 186 | 186 |
| Kiki_Main | TPP | squad | 164 | 164 | 164 | 164 | 164 |
| Tiger_Main | TPP | solo | 160 | 160 | 160 | 160 | 160 |
| Neon_Main | TPP | duo | 155 | 155 | 155 | 155 | 155 |
| Summerland_Main | TPP | duo | 134 | 134 | 134 | 134 | 134 |
| DihorOtok_Main | TPP | duo | 122 | 122 | 122 | 122 | 122 |
| DihorOtok_Main | FPP | squad | 115 | 115 | 115 | 115 | 115 |
| Chimera_Main | TPP | duo | 106 | 106 | 106 | 106 | 106 |
| Neon_Main | FPP | duo | 105 | 105 | 105 | 105 | 105 |
| Baltic_Main | FPP | solo | 102 | 102 | 102 | 102 | 102 |
| Savage_Main | TPP | solo | 100 | 100 | 100 | 100 | 100 |
| Desert_Main | FPP | duo | 96 | 96 | 96 | 96 | 96 |
| Tiger_Main | FPP | solo | 93 | 93 | 93 | 93 | 93 |
| Desert_Main | TPP | solo | 91 | 91 | 91 | 91 | 91 |
| Neon_Main | TPP | solo | 81 | 81 | 81 | 81 | 81 |
| DihorOtok_Main | FPP | duo | 70 | 70 | 70 | 70 | 70 |
| DihorOtok_Main | TPP | solo | 70 | 70 | 70 | 70 | 70 |
| Summerland_Main | TPP | solo | 62 | 62 | 62 | 62 | 62 |
| Desert_Main | FPP | solo | 56 | 56 | 56 | 56 | 56 |
| Savage_Main | FPP | squad | 56 | 56 | 56 | 56 | 56 |
| Neon_Main | FPP | solo | 52 | 52 | 52 | 52 | 52 |
| Kiki_Main | FPP | squad | 48 | 48 | 48 | 48 | 48 |
| Kiki_Main | TPP | duo | 42 | 42 | 42 | 42 | 42 |
| Savage_Main | FPP | duo | 40 | 40 | 40 | 40 | 40 |
| Chimera_Main | TPP | solo | 39 | 39 | 39 | 39 | 39 |
| Summerland_Main | FPP | squad | 39 | 39 | 39 | 39 | 39 |
| Chimera_Main | FPP | squad | 37 | 37 | 37 | 37 | 37 |
| Chimera_Main | FPP | duo | 33 | 33 | 33 | 33 | 33 |
| DihorOtok_Main | FPP | solo | 31 | 31 | 31 | 31 | 31 |
| Kiki_Main | FPP | duo | 31 | 31 | 31 | 31 | 31 |
| Kiki_Main | TPP | solo | 29 | 29 | 29 | 29 | 29 |
| Savage_Main | FPP | solo | 28 | 28 | 28 | 28 | 28 |
| Summerland_Main | FPP | duo | 23 | 23 | 23 | 23 | 23 |
| Chimera_Main | FPP | solo | 21 | 21 | 21 | 21 | 21 |
| Kiki_Main | FPP | solo | 15 | 15 | 15 | 15 | 15 |
| Summerland_Main | FPP | solo | 14 | 14 | 14 | 14 | 14 |

## Raw Domains Without Current Graph-Eligible Data

| map_name | perspective | team_size | game_mode | raw_fetched | phase_feature_done | graph_eligible_n128 | not_phase_featured |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Range_Main | TPP | clansolo | clansolo | 1,571 | 0 | 0 | 1,571 |
| Range_Main | TPP | solo | solo | 391 | 0 | 0 | 391 |
| Tutorial_Main | TPP | solo | solo | 3,696 | 0 | 0 | 3,696 |

## Interpretation

- `raw_fetched`는 telemetry를 가져온 원천 match 수이다.
- `phase_feature_done`은 phase 단위 feature가 생성된 match 수이다.
- `target_label_done`은 next-phase hazard target이 정의된 match 수이다.
- `edge_cache_done`은 phase-level graph edge cache가 있는 match 수이다.
- `graph_eligible_n128`은 현재 DynamicSTGNN 재실험에 바로 쓸 수 있는 match 수이다.
- 현재 논문용 전체 재실험의 최대 모집단은 raw 27,491건이 아니라 `graph_eligible_n128` 기준 21,824건이다.
