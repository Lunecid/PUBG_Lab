# Research Cohort Flat Manifest

이 문서는 현재 DB의 raw-fetched match를 삭제 없이 연구용 cohort로 평탄화한 결과이다.

## Totals

| raw_fetched | graph_eligible_n128 | excluded |
| ---: | ---: | ---: |
| 27,491 | 21,824 | 5,667 |

## Status Counts

| cohort_status | count |
| --- | ---: |
| graph_eligible | 21,824 |
| raw_only_not_ready | 5,658 |
| feature_incomplete | 6 |
| base_incomplete | 3 |

## Exclusion Reasons

| exclude_reason | count |
| --- | ---: |
| included | 21,824 |
| out_of_scope_map | 5,658 |
| missing_target_labels | 6 |
| missing_team_positions | 3 |

## Map Pivot Summary

연구에 실제로 투입 가능한 `graph_eligible_n128` 기준의 맵/인칭/팀 크기 피벗이다.

| map_label | map_name | fpp_solo | fpp_duo | fpp_squad | fpp_total | tpp_solo | tpp_duo | tpp_squad | tpp_total | graph_eligible_total | raw_fetched | excluded |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Erangel | Baltic_Main | 102 | 186 | 430 | 718 | 800 | 656 | 6,204 | 7,660 | 8,378 | 8,381 | 3 |
| Taego | Tiger_Main | 93 | 199 | 435 | 727 | 160 | 573 | 4,159 | 4,892 | 5,619 | 5,621 | 2 |
| Miramar/Desert | Desert_Main | 56 | 96 | 288 | 440 | 91 | 241 | 1,543 | 1,875 | 2,315 | 2,317 | 2 |
| Rondo | Neon_Main | 52 | 105 | 264 | 421 | 81 | 155 | 1,039 | 1,275 | 1,696 | 1,698 | 2 |
| Sanhok | Savage_Main | 28 | 40 | 56 | 124 | 100 | 243 | 940 | 1,283 | 1,407 | 1,407 | 0 |
| Vikendi | DihorOtok_Main | 31 | 70 | 115 | 216 | 70 | 122 | 498 | 690 | 906 | 906 | 0 |
| Karakin | Summerland_Main | 14 | 23 | 39 | 76 | 62 | 134 | 333 | 529 | 605 | 605 | 0 |
| Paramo | Chimera_Main | 21 | 33 | 37 | 91 | 39 | 106 | 333 | 478 | 569 | 569 | 0 |
| Deston | Kiki_Main | 15 | 31 | 48 | 94 | 29 | 42 | 164 | 235 | 329 | 329 | 0 |
| Tutorial | Tutorial_Main | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3,696 | 3,696 |
| Training Range | Range_Main | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,962 | 1,962 |

## Paper Cohort Contract

논문용 본실험은 아래 master cohort를 공통 모집단으로 고정한다. 모델 variant, ablation, tabular baseline은 동일한 match split을 재사용하고, 맵/인칭/팀 크기 분석은 이 cohort membership의 부분집합으로만 수행한다.

| cohort_id | n_matches | train | val | test | ids_sha1 |
| --- | ---: | ---: | ---: | ---: | --- |
| paper_graph_n128_v1 | 21,824 | 17,460 | 2,182 | 2,182 | 1c729438e9b908edd2101ccb9cfebd39af5aa12b |

- `feature_eligible`가 더 많은 분석도 논문 본표에서는 이 common-denominator cohort로 제한한다.
- 도메인별 실험은 독립 split을 새로 뽑지 않고 master split과 도메인 membership의 교집합을 사용한다.
- 탐색 실험에서 다른 cohort를 쓰면 보조 분석으로 분리하고, 본 주장과 직접 비교하지 않는다.

## Paper Cohort Domain Splits

| map_label | map_name | perspective | team_size | game_mode | train | val | test | total |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Erangel | Baltic_Main | TPP | squad | squad | 4,998 | 568 | 638 | 6,204 |
| Taego | Tiger_Main | TPP | squad | squad | 3,324 | 431 | 404 | 4,159 |
| Miramar/Desert | Desert_Main | TPP | squad | squad | 1,210 | 158 | 175 | 1,543 |
| Rondo | Neon_Main | TPP | squad | squad | 838 | 102 | 99 | 1,039 |
| Sanhok | Savage_Main | TPP | squad | squad | 743 | 109 | 88 | 940 |
| Erangel | Baltic_Main | TPP | solo | solo | 643 | 80 | 77 | 800 |
| Erangel | Baltic_Main | TPP | duo | duo | 529 | 67 | 60 | 656 |
| Taego | Tiger_Main | TPP | duo | duo | 456 | 64 | 53 | 573 |
| Vikendi | DihorOtok_Main | TPP | squad | squad | 398 | 51 | 49 | 498 |
| Taego | Tiger_Main | FPP | squad | squad-fpp | 335 | 58 | 42 | 435 |
| Erangel | Baltic_Main | FPP | squad | squad-fpp | 346 | 36 | 48 | 430 |
| Paramo | Chimera_Main | TPP | squad | squad | 265 | 32 | 36 | 333 |
| Karakin | Summerland_Main | TPP | squad | squad | 270 | 31 | 32 | 333 |
| Miramar/Desert | Desert_Main | FPP | squad | squad-fpp | 231 | 30 | 27 | 288 |
| Rondo | Neon_Main | FPP | squad | squad-fpp | 211 | 25 | 28 | 264 |
| Sanhok | Savage_Main | TPP | duo | duo | 189 | 34 | 20 | 243 |
| Miramar/Desert | Desert_Main | TPP | duo | duo | 193 | 23 | 25 | 241 |
| Taego | Tiger_Main | FPP | duo | duo-fpp | 158 | 17 | 24 | 199 |
| Erangel | Baltic_Main | FPP | duo | duo-fpp | 147 | 17 | 22 | 186 |
| Deston | Kiki_Main | TPP | squad | squad | 128 | 14 | 22 | 164 |
| Taego | Tiger_Main | TPP | solo | solo | 128 | 17 | 15 | 160 |
| Rondo | Neon_Main | TPP | duo | duo | 125 | 16 | 14 | 155 |
| Karakin | Summerland_Main | TPP | duo | duo | 110 | 10 | 14 | 134 |
| Vikendi | DihorOtok_Main | TPP | duo | duo | 103 | 11 | 8 | 122 |
| Vikendi | DihorOtok_Main | FPP | squad | squad-fpp | 89 | 14 | 12 | 115 |
| Paramo | Chimera_Main | TPP | duo | duo | 97 | 2 | 7 | 106 |
| Rondo | Neon_Main | FPP | duo | duo-fpp | 77 | 16 | 12 | 105 |
| Erangel | Baltic_Main | FPP | solo | solo-fpp | 83 | 8 | 11 | 102 |
| Sanhok | Savage_Main | TPP | solo | solo | 78 | 12 | 10 | 100 |
| Miramar/Desert | Desert_Main | FPP | duo | duo-fpp | 80 | 9 | 7 | 96 |
| Taego | Tiger_Main | FPP | solo | solo-fpp | 67 | 18 | 8 | 93 |
| Miramar/Desert | Desert_Main | TPP | solo | solo | 77 | 5 | 9 | 91 |
| Rondo | Neon_Main | TPP | solo | solo | 66 | 7 | 8 | 81 |
| Vikendi | DihorOtok_Main | FPP | duo | duo-fpp | 57 | 5 | 8 | 70 |
| Vikendi | DihorOtok_Main | TPP | solo | solo | 54 | 7 | 9 | 70 |
| Karakin | Summerland_Main | TPP | solo | solo | 49 | 6 | 7 | 62 |
| Miramar/Desert | Desert_Main | FPP | solo | solo-fpp | 45 | 6 | 5 | 56 |
| Sanhok | Savage_Main | FPP | squad | squad-fpp | 51 | 3 | 2 | 56 |
| Rondo | Neon_Main | FPP | solo | solo-fpp | 42 | 6 | 4 | 52 |
| Deston | Kiki_Main | FPP | squad | squad-fpp | 40 | 5 | 3 | 48 |
| Deston | Kiki_Main | TPP | duo | duo | 36 | 4 | 2 | 42 |
| Sanhok | Savage_Main | FPP | duo | duo-fpp | 32 | 4 | 4 | 40 |
| Paramo | Chimera_Main | TPP | solo | solo | 29 | 5 | 5 | 39 |
| Karakin | Summerland_Main | FPP | squad | squad-fpp | 28 | 8 | 3 | 39 |
| Paramo | Chimera_Main | FPP | squad | squad-fpp | 30 | 3 | 4 | 37 |
| Paramo | Chimera_Main | FPP | duo | duo-fpp | 28 | 2 | 3 | 33 |
| Vikendi | DihorOtok_Main | FPP | solo | solo-fpp | 25 | 5 | 1 | 31 |
| Deston | Kiki_Main | FPP | duo | duo-fpp | 25 | 3 | 3 | 31 |
| Deston | Kiki_Main | TPP | solo | solo | 22 | 5 | 2 | 29 |
| Sanhok | Savage_Main | FPP | solo | solo-fpp | 20 | 5 | 3 | 28 |
| Karakin | Summerland_Main | FPP | duo | duo-fpp | 18 | 0 | 5 | 23 |
| Paramo | Chimera_Main | FPP | solo | solo-fpp | 17 | 4 | 0 | 21 |
| Deston | Kiki_Main | FPP | solo | solo-fpp | 8 | 4 | 3 | 15 |
| Karakin | Summerland_Main | FPP | solo | solo-fpp | 12 | 0 | 2 | 14 |

## Domain Summary

| map_name | perspective | team_size | game_mode | raw_fetched | base_done | phase_feature_done | target_label_done | edge_cache_done | graph_eligible_n128 | excluded |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baltic_Main | TPP | squad | squad | 6,206 | 6,206 | 6,205 | 6,204 | 6,205 | 6,204 | 2 |
| Tiger_Main | TPP | squad | squad | 4,160 | 4,160 | 4,160 | 4,159 | 4,160 | 4,159 | 1 |
| Desert_Main | TPP | squad | squad | 1,545 | 1,545 | 1,545 | 1,543 | 1,545 | 1,543 | 2 |
| Neon_Main | TPP | squad | squad | 1,041 | 1,041 | 1,041 | 1,039 | 1,041 | 1,039 | 2 |
| Savage_Main | TPP | squad | squad | 940 | 940 | 940 | 940 | 940 | 940 | 0 |
| Baltic_Main | TPP | solo | solo | 801 | 801 | 800 | 800 | 800 | 800 | 1 |
| Baltic_Main | TPP | duo | duo | 656 | 656 | 656 | 656 | 656 | 656 | 0 |
| Tiger_Main | TPP | duo | duo | 574 | 574 | 573 | 573 | 573 | 573 | 1 |
| DihorOtok_Main | TPP | squad | squad | 498 | 498 | 498 | 498 | 498 | 498 | 0 |
| Tiger_Main | FPP | squad | squad-fpp | 435 | 435 | 435 | 435 | 435 | 435 | 0 |
| Baltic_Main | FPP | squad | squad-fpp | 430 | 430 | 430 | 430 | 430 | 430 | 0 |
| Chimera_Main | TPP | squad | squad | 333 | 333 | 333 | 333 | 333 | 333 | 0 |
| Summerland_Main | TPP | squad | squad | 333 | 333 | 333 | 333 | 333 | 333 | 0 |
| Desert_Main | FPP | squad | squad-fpp | 288 | 288 | 288 | 288 | 288 | 288 | 0 |
| Neon_Main | FPP | squad | squad-fpp | 264 | 264 | 264 | 264 | 264 | 264 | 0 |
| Savage_Main | TPP | duo | duo | 243 | 243 | 243 | 243 | 243 | 243 | 0 |
| Desert_Main | TPP | duo | duo | 241 | 241 | 241 | 241 | 241 | 241 | 0 |
| Tiger_Main | FPP | duo | duo-fpp | 199 | 199 | 199 | 199 | 199 | 199 | 0 |
| Baltic_Main | FPP | duo | duo-fpp | 186 | 186 | 186 | 186 | 186 | 186 | 0 |
| Kiki_Main | TPP | squad | squad | 164 | 164 | 164 | 164 | 164 | 164 | 0 |
| Tiger_Main | TPP | solo | solo | 160 | 160 | 160 | 160 | 160 | 160 | 0 |
| Neon_Main | TPP | duo | duo | 155 | 155 | 155 | 155 | 155 | 155 | 0 |
| Summerland_Main | TPP | duo | duo | 134 | 134 | 134 | 134 | 134 | 134 | 0 |
| DihorOtok_Main | TPP | duo | duo | 122 | 122 | 122 | 122 | 122 | 122 | 0 |
| DihorOtok_Main | FPP | squad | squad-fpp | 115 | 115 | 115 | 115 | 115 | 115 | 0 |
| Chimera_Main | TPP | duo | duo | 106 | 106 | 106 | 106 | 106 | 106 | 0 |
| Neon_Main | FPP | duo | duo-fpp | 105 | 105 | 105 | 105 | 105 | 105 | 0 |
| Baltic_Main | FPP | solo | solo-fpp | 102 | 102 | 102 | 102 | 102 | 102 | 0 |
| Savage_Main | TPP | solo | solo | 100 | 100 | 100 | 100 | 100 | 100 | 0 |
| Desert_Main | FPP | duo | duo-fpp | 96 | 96 | 96 | 96 | 96 | 96 | 0 |
| Tiger_Main | FPP | solo | solo-fpp | 93 | 93 | 93 | 93 | 93 | 93 | 0 |
| Desert_Main | TPP | solo | solo | 91 | 91 | 91 | 91 | 91 | 91 | 0 |
| Neon_Main | TPP | solo | solo | 81 | 81 | 81 | 81 | 81 | 81 | 0 |
| DihorOtok_Main | FPP | duo | duo-fpp | 70 | 70 | 70 | 70 | 70 | 70 | 0 |
| DihorOtok_Main | TPP | solo | solo | 70 | 70 | 70 | 70 | 70 | 70 | 0 |
| Summerland_Main | TPP | solo | solo | 62 | 62 | 62 | 62 | 62 | 62 | 0 |
| Desert_Main | FPP | solo | solo-fpp | 56 | 56 | 56 | 56 | 56 | 56 | 0 |
| Savage_Main | FPP | squad | squad-fpp | 56 | 56 | 56 | 56 | 56 | 56 | 0 |
| Neon_Main | FPP | solo | solo-fpp | 52 | 52 | 52 | 52 | 52 | 52 | 0 |
| Kiki_Main | FPP | squad | squad-fpp | 48 | 48 | 48 | 48 | 48 | 48 | 0 |
| Kiki_Main | TPP | duo | duo | 42 | 42 | 42 | 42 | 42 | 42 | 0 |
| Savage_Main | FPP | duo | duo-fpp | 40 | 40 | 40 | 40 | 40 | 40 | 0 |
| Chimera_Main | TPP | solo | solo | 39 | 39 | 39 | 39 | 39 | 39 | 0 |
| Summerland_Main | FPP | squad | squad-fpp | 39 | 39 | 39 | 39 | 39 | 39 | 0 |
| Chimera_Main | FPP | squad | squad-fpp | 37 | 37 | 37 | 37 | 37 | 37 | 0 |
| Chimera_Main | FPP | duo | duo-fpp | 33 | 33 | 33 | 33 | 33 | 33 | 0 |
| DihorOtok_Main | FPP | solo | solo-fpp | 31 | 31 | 31 | 31 | 31 | 31 | 0 |
| Kiki_Main | FPP | duo | duo-fpp | 31 | 31 | 31 | 31 | 31 | 31 | 0 |
| Kiki_Main | TPP | solo | solo | 29 | 29 | 29 | 29 | 29 | 29 | 0 |
| Savage_Main | FPP | solo | solo-fpp | 28 | 28 | 28 | 28 | 28 | 28 | 0 |
| Summerland_Main | FPP | duo | duo-fpp | 23 | 23 | 23 | 23 | 23 | 23 | 0 |
| Chimera_Main | FPP | solo | solo-fpp | 21 | 21 | 21 | 21 | 21 | 21 | 0 |
| Kiki_Main | FPP | solo | solo-fpp | 15 | 15 | 15 | 15 | 15 | 15 | 0 |
| Summerland_Main | FPP | solo | solo-fpp | 14 | 14 | 14 | 14 | 14 | 14 | 0 |
| Tutorial_Main | TPP | solo | solo | 3,696 | 0 | 0 | 0 | 0 | 0 | 3,696 |
| Range_Main | TPP | clansolo | clansolo | 1,571 | 0 | 0 | 0 | 0 | 0 | 1,571 |
| Range_Main | TPP | solo | solo | 391 | 0 | 0 | 0 | 0 | 0 | 391 |

## Policy

- `pubg.*` raw/normalized layer는 삭제하지 않는다.
- 숫자를 맞출 때는 `graph_eligible_n128 = true`인 match만 연구 cohort에 포함한다.
- 제외 대상은 `exclude_reason`으로 격리한다.
- 삭제가 필요하다면 `stdb.*` 같은 재생성 가능한 파생 layer에서만 수행한다.
