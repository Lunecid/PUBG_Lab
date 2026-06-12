# Experiment Datasets by Raw Domain

이 문서는 raw fetched 전체 모집단을 map / perspective / team size 도메인으로 나누고, 각 도메인에서 실제 실험 가능한 match id와 split manifest를 생성한 결과이다.

- Output root: `model_stgcn/experiment_datasets/raw_domain_population_seed42`
- Minimum graph-eligible threshold for `experiment_ready`: `300` matches

## Overall

| domain_slug | raw_count | feature_eligible_count | graph_eligible_count |
| --- | ---: | ---: | ---: |
| TOTAL | 27,491 | 21,824 | 21,824 |

## Experiment-Ready Domains

| domain_slug | raw_count | feature_eligible_count | graph_eligible_count | graph_train | graph_val | graph_test | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baltic_Main__TPP__squad | 6,206 | 6,204 | 6,204 | 4,964 | 620 | 620 | experiment_ready |
| Tiger_Main__TPP__squad | 4,160 | 4,159 | 4,159 | 3,329 | 415 | 415 | experiment_ready |
| Desert_Main__TPP__squad | 1,545 | 1,543 | 1,543 | 1,235 | 154 | 154 | experiment_ready |
| Neon_Main__TPP__squad | 1,041 | 1,039 | 1,039 | 833 | 103 | 103 | experiment_ready |
| Savage_Main__TPP__squad | 940 | 940 | 940 | 752 | 94 | 94 | experiment_ready |
| Baltic_Main__TPP__solo | 801 | 800 | 800 | 640 | 80 | 80 | experiment_ready |
| Baltic_Main__TPP__duo | 656 | 656 | 656 | 526 | 65 | 65 | experiment_ready |
| Tiger_Main__TPP__duo | 574 | 573 | 573 | 459 | 57 | 57 | experiment_ready |
| DihorOtok_Main__TPP__squad | 498 | 498 | 498 | 400 | 49 | 49 | experiment_ready |
| Tiger_Main__FPP__squad | 435 | 435 | 435 | 349 | 43 | 43 | experiment_ready |
| Baltic_Main__FPP__squad | 430 | 430 | 430 | 344 | 43 | 43 | experiment_ready |
| Chimera_Main__TPP__squad | 333 | 333 | 333 | 267 | 33 | 33 | experiment_ready |
| Summerland_Main__TPP__squad | 333 | 333 | 333 | 267 | 33 | 33 | experiment_ready |

## Small Auxiliary Domains

| domain_slug | raw_count | feature_eligible_count | graph_eligible_count | graph_train | graph_val | graph_test | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Desert_Main__FPP__squad | 288 | 288 | 288 | 232 | 28 | 28 | small_auxiliary |
| Neon_Main__FPP__squad | 264 | 264 | 264 | 212 | 26 | 26 | small_auxiliary |
| Savage_Main__TPP__duo | 243 | 243 | 243 | 195 | 24 | 24 | small_auxiliary |
| Desert_Main__TPP__duo | 241 | 241 | 241 | 193 | 24 | 24 | small_auxiliary |
| Tiger_Main__FPP__duo | 199 | 199 | 199 | 161 | 19 | 19 | small_auxiliary |
| Baltic_Main__FPP__duo | 186 | 186 | 186 | 150 | 18 | 18 | small_auxiliary |
| Kiki_Main__TPP__squad | 164 | 164 | 164 | 132 | 16 | 16 | small_auxiliary |
| Tiger_Main__TPP__solo | 160 | 160 | 160 | 128 | 16 | 16 | small_auxiliary |
| Neon_Main__TPP__duo | 155 | 155 | 155 | 125 | 15 | 15 | small_auxiliary |
| Summerland_Main__TPP__duo | 134 | 134 | 134 | 108 | 13 | 13 | small_auxiliary |
| DihorOtok_Main__TPP__duo | 122 | 122 | 122 | 98 | 12 | 12 | small_auxiliary |
| DihorOtok_Main__FPP__squad | 115 | 115 | 115 | 93 | 11 | 11 | small_auxiliary |
| Chimera_Main__TPP__duo | 106 | 106 | 106 | 86 | 10 | 10 | small_auxiliary |
| Neon_Main__FPP__duo | 105 | 105 | 105 | 85 | 10 | 10 | small_auxiliary |
| Baltic_Main__FPP__solo | 102 | 102 | 102 | 82 | 10 | 10 | small_auxiliary |
| Savage_Main__TPP__solo | 100 | 100 | 100 | 80 | 10 | 10 | small_auxiliary |
| Desert_Main__FPP__duo | 96 | 96 | 96 | 78 | 9 | 9 | small_auxiliary |
| Tiger_Main__FPP__solo | 93 | 93 | 93 | 75 | 9 | 9 | small_auxiliary |
| Desert_Main__TPP__solo | 91 | 91 | 91 | 73 | 9 | 9 | small_auxiliary |
| Neon_Main__TPP__solo | 81 | 81 | 81 | 65 | 8 | 8 | small_auxiliary |
| DihorOtok_Main__FPP__duo | 70 | 70 | 70 | 56 | 7 | 7 | small_auxiliary |
| DihorOtok_Main__TPP__solo | 70 | 70 | 70 | 56 | 7 | 7 | small_auxiliary |
| Summerland_Main__TPP__solo | 62 | 62 | 62 | 50 | 6 | 6 | small_auxiliary |
| Desert_Main__FPP__solo | 56 | 56 | 56 | 46 | 5 | 5 | small_auxiliary |
| Savage_Main__FPP__squad | 56 | 56 | 56 | 46 | 5 | 5 | small_auxiliary |
| Neon_Main__FPP__solo | 52 | 52 | 52 | 42 | 5 | 5 | small_auxiliary |
| Kiki_Main__FPP__squad | 48 | 48 | 48 | 40 | 4 | 4 | small_auxiliary |
| Kiki_Main__TPP__duo | 42 | 42 | 42 | 34 | 4 | 4 | small_auxiliary |
| Savage_Main__FPP__duo | 40 | 40 | 40 | 32 | 4 | 4 | small_auxiliary |
| Chimera_Main__TPP__solo | 39 | 39 | 39 | 33 | 3 | 3 | small_auxiliary |
| Summerland_Main__FPP__squad | 39 | 39 | 39 | 33 | 3 | 3 | small_auxiliary |
| Chimera_Main__FPP__squad | 37 | 37 | 37 | 31 | 3 | 3 | small_auxiliary |
| Chimera_Main__FPP__duo | 33 | 33 | 33 | 27 | 3 | 3 | small_auxiliary |
| DihorOtok_Main__FPP__solo | 31 | 31 | 31 | 25 | 3 | 3 | small_auxiliary |
| Kiki_Main__FPP__duo | 31 | 31 | 31 | 25 | 3 | 3 | small_auxiliary |
| Kiki_Main__TPP__solo | 29 | 29 | 29 | 25 | 2 | 2 | small_auxiliary |
| Savage_Main__FPP__solo | 28 | 28 | 28 | 24 | 2 | 2 | small_auxiliary |
| Summerland_Main__FPP__duo | 23 | 23 | 23 | 19 | 2 | 2 | small_auxiliary |
| Chimera_Main__FPP__solo | 21 | 21 | 21 | 17 | 2 | 2 | small_auxiliary |
| Kiki_Main__FPP__solo | 15 | 15 | 15 | 13 | 1 | 1 | small_auxiliary |
| Summerland_Main__FPP__solo | 14 | 14 | 14 | 12 | 1 | 1 | small_auxiliary |

## Raw-Only Domains Not Ready for Current Experiments

| domain_slug | raw_count | feature_eligible_count | graph_eligible_count | status |
| --- | ---: | ---: | ---: | --- |
| Tutorial_Main__TPP__solo | 3,696 | 0 | 0 | raw_only_not_ready |
| Range_Main__TPP__clansolo | 1,571 | 0 | 0 | raw_only_not_ready |
| Range_Main__TPP__solo | 391 | 0 | 0 | raw_only_not_ready |

## How to Use

- `raw_match_ids.txt`: raw fetched 모집단 ID.
- `feature_match_ids.txt`: feature ablation/tabular experiment에 사용할 수 있는 ID.
- `graph_match_ids.txt`: DynamicSTGNN 학습/평가에 사용할 수 있는 ID.
- `feature_split_seed*.json`: feature-level experiment split.
- `graph_split_seed*.json`: graph/STGNN experiment split.
- 논문용 main 재실험은 `experiment_ready` 도메인 또는 전체 `graph_eligible_count` 기준으로 수행한다.
