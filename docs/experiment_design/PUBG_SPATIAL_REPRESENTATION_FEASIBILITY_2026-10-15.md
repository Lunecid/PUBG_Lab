# PUBG 공간표현 학습 가능성 검토

## Material Passport

- 작성일: 2026-10-15.
- 목적: 안전구역 조건부 경로 Diffusion에 앞서 공간표현 학습의 데이터 가용성과 식별 가능한 목표를 판단한다.
- 확인 완료: 공식 자산 저장소 전체 tree, Erangel 이미지 헤더, 로컬 지도 이미지와 공간 특징 코드, DB 40경기의 읽기 전용 공간 지지영역 검사.
- 미실행: 공간 encoder 학습, 지도 정합 정확도 평가, 지형 통행 가능성 평가, Diffusion 학습.
- 판정: **지도 이미지와 이동 이력을 결합한 거시적 공간표현 학습은 착수 가능하다. 정밀 통행 가능성·충돌·엄폐를 보장하는 PUBG 지형 모델은 현재 자료로 확인되지 않았다.**

공간표현의 효용을 먼저 검증하는 순서는 타당하다. 다만 별도 대형 사전학습 모델이 수학적 필수조건은 아니다. 고정 공간 특징, 선행 학습 후 고정한 encoder, 경로 모델과 함께 학습하는 encoder를 비교하면 된다. 현재 필요한 선행 단계는 공간표현의 정보성과 일반화 검증이다.

## 공식 자료와 현재 자산

공식 `pubg/api-assets`를 commit `32b13b51128b8d8909ae5e77f3b833e01230b24d`에 고정해 recursive tree 934개 항목을 조회했다. 응답은 truncated=false였다. 지도 파일과 지역명 코드는 확인했지만, 이 tree에서 navmesh·heightmap·terrain 이름의 자산은 찾지 못했다. 이것은 확인한 공식 자산 목록의 범위에 대한 결과이며, 다른 경로를 포함한 전 세계의 공개 데이터 부재를 증명하지 않는다.

| 자원 | 확인 내용 | 공간 학습에서의 의미 |
|---|---|---|
| 공식 Erangel 무문자 고해상도 지도 | 8192×8192 PNG; Git LFS 크기 94,717,917 bytes | 지명 글자를 배제한 지도 패치 encoder 입력 후보 |
| 공식 Erangel 무문자 저해상도 지도 | 819×819 PNG | 거친 지역 시각 정보; 세부 건물 구조 학습에는 부적합할 수 있음 |
| 로컬 `baltic_main.jpg` | 4096×4096 | 이미 존재하는 시각화용 지도; 출처·패치 일치 미확정 |
| 로컬 `baltic_main_2k.jpg` | 2048×2048, 지명 텍스트 포함을 시각 확인 | 연구 초기 정합 점검용; 최종 학습은 무문자 버전을 우선 |
| 공식 Location | x/y/z cm, 지도 좌상단 원점, Erangel 범위 816000cm | 좌표와 지도 이미지 연결 가능; 실제 정합 검증 필요 |
| 공식 지역 코드 | `Erangel_Main` 25개 지역명 문자열 | 지역 경계 polygon·도로 연결·출입구가 아님; Baltic_Main과의 대응도 검증 대상 |
| 로컬 고도 특징 | `stdb/z_affordance_feature_etl.py`는 플레이어 팀 중심 고도 proxy라고 명시 | 높이 관측을 terrain elevation·LOS·엄폐량으로 사용하지 않음 |
| 로컬 Caldera mesh 코드 | `scripts/build_caldera_map_mesh.py`는 별도 Caldera USD 지도 대상 | PUBG 지형 확보의 근거가 아님 |

고해상도 이미지의 실제 매체 URL에 Range 요청을 보내 첫 32바이트를 읽었고 HTTP 206과 PNG header를 확인했다. 파일 전체를 다운로드하거나 전체 이미지 무결성을 검증한 것은 아니다. 원본 자산의 LFS hash와 조회 결과는 [자산 검사 JSON](../../reports/spatial_feasibility_2026-10-15/asset_inspection.json)에 보관했다. [전체 tree](../../reports/spatial_feasibility_2026-10-15/official_assets_tree.json)도 보관했다.

8160m 전체 extent가 8192픽셀에 정확히 대응한다면 약 0.996m/pixel이다. 이는 **정합 가정하의 이미지 축척**이며 1m 수준의 지형 정확도가 아니다. 원본 좌표계·border·crop·map version 확인 후 변환을 확정한다.

공식 Update 41.1(2026-04-08)은 Erangel에 destructible terrain 추가를 명시한다. 따라서 정적인 이미지가 모든 경기에서 정확한 지형을 나타낸다고 가정하지 않는다. 본 자료의 경기 시기와 패치 대응은 이번 probe에서 검증하지 않았다. map_name뿐 아니라 경기 시기/패치, 자산 commit, 좌표 변환, 지도 변형을 함께 관리한다.

출처: [공식 지도 자산](https://github.com/pubg/api-assets/tree/32b13b51128b8d8909ae5e77f3b833e01230b24d/Assets/Maps), [공식 텔레메트리 객체](https://documentation.pubg.com/en/telemetry-objects.html), [공식 Update 41.1](https://www.pubg.com/en-asia/news/9926).

## DB에서 확인한 학습 신호

기존 프로파일의 Erangel TPP squad official 후보 171경기에서 match ID의 고정 SHA256 순서로 40경기를 선택했다. 전체 DB의 확률 표본이 아니며 전체 코퍼스 커버리지 추정이 아니다. 32경기를 reference, 8경기를 held-out으로 나눈 진단용 분할이다. 현재 모델의 최종 train/test split을 생성하거나 변경하지 않았다.

`account.` 접두어와 참가자 등록을 만족하는 인간 후보만 사용했다. 최초 착지 이후·최초 kill 이전, 양의 체력, 차량 기록 없음, 공식 좌표 범위 내 조건을 적용했다. 중복 timestamp는 모두 제외하도록 구현했다. 연속 관측 간격 5–15초, 직선 변위/시간 ≤12m/s인 쌍만 이동 연결 후보로 삼았다. 12m/s는 점검용 cutoff이지 게임의 물리 법칙이 아니다. 수영·재배치·관측 사이 차량 탑승은 완전히 제거하지 못했으므로 확정 도보 자료가 아니다.

| 항목 | 실측 |
|---|---:|
| 경기별 인간 참가자 수의 합 | 3,198 player-match, 고유 사람 수 아님 |
| 원시 인간 위치 행 | 276,204 |
| 조건을 통과한 위치 | 173,719 |
| 조건을 통과한 연속 관측 쌍 | 164,222 |
| 관측 간격 중앙값 | 9.999초 |
| 변위 중앙값 | 9.77m |
| 변위 90 / 99 percentile | 41.42 / 63.20m |

각 위치를 25/50/100m 격자에 배정했다. 서로 다른 격자 사이 이동만 아래 지지영역 계산에 사용했다. 같은 격자에 머문 기록으로 지원 비율이 부풀려지는 것을 피했다. 아래 비율은 held-out 이동 기록 단위 micro 평균이며, JSON에는 경기별 평균을 다시 평균한 macro 결과도 함께 보관했다.

| 격자 폭 | Reference 관측 격자 | Held-out 이동 쌍 수 | 양쪽 격자 모두 관측됨 | 동일 방향 연결 관측됨 | 동일 연결이 reference 3경기 이상에서 관측됨 |
|---|---:|---:|---:|---:|---:|
| 25m | 13,847 | 16,517 | 77.0% | 57.4% | 37.4% |
| 50m | 6,357 | 11,114 | 85.7% | 71.4% | 51.7% |
| 100m | 2,373 | 6,122 | 93.9% | 79.5% | 58.9% |

**해석:** 여러 경기에서 반복되는 공간 이용과 이동 연결이 있으므로 학습 신호가 존재한다. 50m 해상도에서는 약 28.6%의 held-out 이동 연결을 reference에서 보지 못했다. 관측 엣지만 허용하는 hard graph는 실제 인간 이동을 배제할 수 있다. 같은 방향의 엣지가 없다는 것은 통행 불가의 증거가 아니다.

100m에서 지지 비율이 높은 것은 집계 해상도를 낮춘 결과이기도 하다. 이를 공간표현 성능 향상이라고 해석하지 않는다. 해상도마다 같은 셀 내부로 사라지는 이동이 달라 분모도 다르다. 그래프가 나타내는 구역 면적을 육지 커버리지로 나누는 계산도 하지 않았다. 해안·바다·건물 등이 포함된 정답 마스크가 아직 없다.

재실행 코드: [audit_spatial_feasibility.py](../../tools/audit_spatial_feasibility.py). [세부 결과](../../reports/spatial_feasibility_2026-10-15/telemetry_support_probe.json). DB는 transaction_read_only=on, statement_timeout=30초로 조회했으며 계정 ID·비밀번호를 결과에 저장하지 않았다.

## 무엇을 학습할 수 있는가

| 목표 | 현재 판단 | 필요한 검증 |
|---|---|---|
| 지역의 시각적 유사성 | 학습 가능 | 한 장의 지도에서 위치를 외운 효과와 분리 |
| 도로·수역·건물 밀집 같은 거친 지도 의미 | 이미지 기반 약한 supervision/수동 표본 라벨로 가능 | 색 분할은 정답이 아니며 지역별 독립 annotation 필요 |
| 인간이 자주 이동하는 구역 간 관계 | 로그에서 학습 가능 | 구역 압박·출발점·이동 모드·표본 빈도와 얽힌 행동 분포임 |
| 일정 시간 뒤 도달하는 지역 분포 | 로그 기반 예측 과제로 가능 | 실제 타임스탬프, 검열, nonself 평가, 시간 범위별 결과 |
| 지형의 절대 최단 이동 시간 | 현재 식별 불가 | 관측 이동 시간은 정지·전투·선호 경로까지 포함 |
| 지형 높이와 경사 전체 복원 | 제한적 | 플레이어 z는 지면 높이가 아님. 실내 층·옥상·점프·다리가 겹침 |
| 출입구·계단·절벽·충돌·엄폐·사격 LOS | 현재 자료만으로 보장 불가 | collision geometry, navmesh, 독립 실측 또는 엔진 검증 필요 |

그래프를 만드는 작업과 공간표현을 학습하는 작업도 구분한다. 격자/이동 연결은 학습의 구조이고, encoder가 그 위에서 지역 간 관계를 설명하는 벡터를 배우는 것이 representation learning이다. 데이터가 없는 구역을 encoder가 일반화할 수 있는지는 실험으로 판단해야 한다.

## 선행 연구의 구체적 설계

첫 목표는 **Erangel의 지역 시각 정보와 관측 이동 관계를 함께 설명하는 공간 embedding**으로 제한한다. 안전구역은 정적 공간 embedding의 정답에 흡수하지 않고 동적 조건으로 분리한다. 공간 encoder의 목적은 변하지 않는 지역 구조를 표현하는 것이며, 이동 예측 head에는 시점별 구역·이동 이력을 별도로 제공한다.

1. 지도 정합과 버전 고정. 원본 이미지 commit, 경기 기간, map_name, world-to-image 변환을 명시한다. 여러 지역의 독립 landmark로 변환을 검증하며, timestamp만으로 패치를 확정할 수 없으면 unverified로 남긴다.
2. 공간 단위 구성. 50m 격자를 기본 pilot로, 25/100m를 민감도로 둔다. 50m 기준 약 27,000개의 전체 bounding-grid 셀이므로 노드 수 자체는 대규모 언어모델 사전학습 수준의 문제가 아니다. 바다·unknown을 임의로 통행 불가 처리하지 않는다.
3. 다중 범위 지도 패치. 각 셀 중심 주변의 작은 범위와 넓은 범위 이미지를 읽어 CNN encoder로 128차원 표현을 만드는 안을 시험한다. 128은 제안값이다. 한 지도에서 나눈 패치는 독립적인 수많은 지도가 아니므로 작은 encoder/고정 특징 기준선을 유지한다.
4. 관측 이동 구조. training 인간 기록에서 방향성 연결·별도 경기 수·관측 시간 간격과 불확실성을 만든다. 공간 인접성과 실제 관측 전이는 별도 edge type이다. 인접하다는 이유로 통행 가능성을 확정하지 않는다.
5. 시간 조건부 이동 예측. 현재 셀·이동 이력·공개 구역을 받아 10/30초 후 셀 분포를 예측한다. full local categorical 또는 명확한 sampled classification으로 학습한다. 선택되지 않은 후보는 '이번에 관측되지 않은 목적지'이며 '통행 불가'가 아니다.
6. 공간 의미 보조 과제. 독립적으로 라벨링한 지도 패치의 수역/도로/건물 밀집 등 거친 의미를 분류하거나, 마스킹한 이미지 특징을 복원한다. 이는 가설이며 거리 기반 shortcut과 지명 글자 shortcut을 점검한다.
7. encoder 평가 후 고정. 학습된 embedding을 고정한 채 작은 이동 예측 head를 비교한다. 이후 Diffusion에 전달하고 frozen versus fine-tuned 비교로 이어간다.

단순 node2vec/GNN 임베딩만 학습하면 같은 지도의 셀 ID와 자주 다니는 경로를 외울 수 있다. 이미지·좌표·그래프를 분리한 기준선과 지역 holdout이 필요하다. 반대로 모든 지역의 static map image를 공개 입력으로 사용하는 것은 이미 알려진 지도라는 과제에서는 가능하지만, 이를 unseen-map 성능으로 부르면 안 된다.

## Diffusion에 들어가기 전 통과해야 할 실험

| 실험 | 기준선과 비교 | 해석 |
|---|---|---|
| 목적지 분포 예측 | 정지, 등속, 좌표/상태 MLP, 평활화 전이 빈도표 | 단순 위치·빈도보다 공간표현에 추가 정보가 있는가 |
| 이미지 효과 | 좌표만 / 이미지+좌표 | 지도 픽셀이 실제로 도움이 되는가 |
| 연결 구조 효과 | 같은 node feature의 연결 없는 encoder / 그래프 encoder | feature 추가와 graph 효과를 분리 |
| 표현 학습 효과 | 고정 특징 / 학습 encoder / frozen encoder+동일 head | 표현 자체의 정보성을 확인 |
| 지역 일반화 | 경기 holdout + 버퍼를 둔 공간 block holdout | 새 경기의 익숙한 위치와 미학습 지역을 구분 |
| 시기 일반화 | 경기 시작시각 기반의 기간 분리와 자산 버전 감사 | 맵 변경·패치·행동 변화에 대한 안정성 |

주지표는 고정된 후보 집합의 목적지 NLL 또는 공간 확률분포 점수로 제안한다. Top-k, 목적지 거리 오차, calibration을 보조 보고한다. 정지/격자 유지가 과제를 지배하지 않도록 전체와 nonself 이동을 각각 보고한다. 직접 전이 빈도표에는 unseen cell/edge fallback을 제공한다. neural 모델도 graph OOD 성능과 이용 가능 범위를 보고한다.

지역 holdout에서는 held-out 지역으로 가거나 그 지역에서 시작하는 training transition을 제외하고, 목표가 교차하는 장기 sequence도 제외한다. 지도 패치 범위가 경계를 넘는 시각 누수를 줄이도록 buffer를 둔다. 해당 지역의 지도 이미지를 학습 중 사용했는지에 따라 '이동 이력 미관측 지역'과 '이미지까지 미관측 지역' 실험을 구분한다.

공간 의미 검증은 도심·도로·교량·해안·산지에서 독립적으로 선정한 소규모 수동 라벨 표본으로 시작한다. 수역과 교량이 겹칠 수 있고 건물 footprint가 실내 통행 불가를 의미하지 않으므로 실제 navigation label과 분리한다. 신뢰구간은 경기 단위 bootstrap, encoder 비교는 같은 split과 seed로 한다. 성능과 정합 오차를 측정하기 전 보편적인 임의 합격률을 주장하지 않는다. 학습 전 최소 유의미한 개선 폭을 validation 계획에 정하고 test를 고정한다.

**진행 기준:** 공간표현이 좌표·전이 빈도 기준선보다 반복적으로 낫고, coverage/OOD 구간과 지역 holdout에서 쓸 수 있는 범위를 확인하면 Diffusion conditioning으로 진행한다. 결과가 같으면 단순 공간 특징을 사용한다. collision-free 경로 생성 주장은 별도 지형 검증 자원이 확보될 때까지 보류한다.

## 스키마에 추가할 항목 제안

아래는 논리적 계약이며 DB 테이블을 생성하지 않았다.

| 논리 테이블 | 주요 컬럼 |
|---|---|
| `map_assets` | asset_id, map_name, patch_scope, source_url, commit, content_hash, width_px, height_px, world_extent_m, transform_id, alignment_status |
| `spatial_cells` | asset_id, resolution_m, cell_id, bbox_world, image_patch_ref, semantic_label_source, semantic_confidence, unknown_mask |
| `spatial_edges` | graph_version, src_cell, dst_cell, edge_type, train_split_hash, distinct_match_count, transition_count, dt_summary, mobility_scope, confidence |
| `spatial_encoder_runs` | run_id, asset_id, split_hash, graph_version, feature_schema, objective, encoder_config, seed, checkpoint_hash |
| `spatial_embeddings` | run_id, cell_id, embedding, support_count, ood_status |
| `spatial_eval` | run_id, evaluation_split, spatial_block, metric, denominator, value, confidence_interval |

`passable=true`는 관측 이동 횟수만으로 생성하지 않는다. 지도 정합, 공간 의미, 관측 이동 지지, 실제 통행 검증 각각의 출처를 구분해 보존한다.

## 관련 연구가 제공하는 근거

- [LaneGCN, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3622_ECCV_2020_paper.php): 지도 연결성과 actor-map 관계를 표현하는 설계의 근거. PUBG에 사용할 lane graph가 이미 있다는 근거는 아니다.
- [Neural Topological SLAM, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Chaplot_Neural_Topological_SLAM_for_Visual_Navigation_CVPR_2020_paper.html): 의미 특징을 가진 노드와 거친 기하 연결로 공간표현을 학습하는 접근의 근거. 해당 연구의 시각 관측·시뮬레이터 자원은 현재 PUBG 텔레메트리와 다르다.

이번 결과는 '공간표현 학습을 위한 관측 자료가 존재한다'는 타당성 근거다. 학습된 공간표현의 예측력, 일반화, 통행 정확도를 검증한 결과는 아니다.
