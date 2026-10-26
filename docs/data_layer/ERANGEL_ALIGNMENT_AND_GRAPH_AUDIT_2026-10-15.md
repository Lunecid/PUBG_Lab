# Erangel 지도 정합과 공간 그래프 검증

2026년 10월 15일 기준으로 공개 원본 지도 두 장을 실제 확보하고, 로컬 PostgreSQL의 Erangel 경기 목록과 위치를 읽기 전용으로 추출했다. **좌표 방향과 범위는 대응하지만 지도 버전은 일치하지 않는다.** 공간 그래프의 시제품은 생성했으며, 현재 산출물은 이동 관측과 역사적 지형을 표현하는 진단 자료다. 같은 패치의 충돌·통행 가능성 정답으로 사용할 수는 없다.

**확보한 원본과 파일 이력**

| 자료 | 실제 파일 | 파일 단위 최종 변경 | 확인 결과 |
|---|---|---|---|
| 커뮤니티 Erangel 높이 래스터 | [height_2019.png](/Users/lunecid/project/PUBG/data/spatial/erangel/source/height_2019.png) | 2019년 12월 19일 | 8192×8192, uint16, 값 범위 22331–65535, 54,046,751 bytes |
| PUBG 공식 무문자 지도 | [official_notext.png](/Users/lunecid/project/PUBG/data/spatial/erangel/source/official_notext.png) | 2024년 10월 28일 | 8192×8192 RGB, 94,717,917 bytes, 공식 Git LFS SHA-256 일치 |

높이 지도는 [cgcostume/pubg-maps 고정 커밋의 파일](https://github.com/cgcostume/pubg-maps/blob/8c0d610dd7ab7a9f0e731b73aead5218e9a34613/erangel/pubg_erangel_height_l16_lod0.png)이다. 저장소를 고정한 커밋은 2020년이지만, **해당 파일이 마지막으로 바뀐 시점은 2019년**이다. [파일 변경 커밋](https://github.com/cgcostume/pubg-maps/commit/b631cd7cbcc49cdd8a4749c1e08dd71c2c4f5573)의 메시지는 `Update to PUBG v6.5.6.3`이다. 이 내부 버전 표기를 공개 패치 6.5로 해석하지 않는다.

공식 이미지는 [pubg/api-assets 고정 커밋](https://github.com/pubg/api-assets/commit/32b13b51128b8d8909ae5e77f3b833e01230b24d)에서 받았다. 커밋 메시지 `31.2 Map image update (#269)`는 저장소의 업데이트 표기이며, Erangel 전체 공간이 정확히 어느 게임 빌드와 같은지를 인증하는 메타데이터는 아니다. [원본 manifest](/Users/lunecid/project/PUBG/data/spatial/erangel/source/manifest.json)에 다운로드 URL, SHA-256, 크기와 파일 날짜를 보관했다.

**우리 텔레메트리의 패치 확인 범위**

DB에 있는 Erangel 8,650경기는 `Baltic_Main`, Steam 자료였다. 시작 이벤트가 있는 경기는 8,381개이고, 시작일 범위는 2026년 3월 26일부터 6월 2일까지다. DB 생성 시각은 수집 시점이므로 경기 패치 추정에는 쓰지 않았다. 조사한 DB 스키마에는 정확한 게임 build나 patch 필드가 없다. 다음 값은 공식 PC 점검 일정과 경기 시작 시각을 결합한 **추정 패치**다. 점검 시간대는 분리했고 핫픽스 빌드까지 판별하지 않는다.

| 추정 패치 또는 상태 | 전체 경기 | official | competitive | airoyale |
|---|---:|---:|---:|---:|
| 40.2 | 2,675 | 1,386 | 225 | 1,064 |
| 41.1 | 4,298 | 1,970 | 422 | 1,906 |
| 41.2 | 1,403 | 749 | 130 | 524 |
| 시작 시각 없음 | 269 | 120 | 34 | 115 |
| 점검 시간 또는 미배정 | 5 | 2 | 0 | 3 |

근거는 [40.2 일정](https://pubg.com/en/news/9809), [41.1 일정](https://www.pubg.com/en-asia/news/9926), [41.2 일정](https://www.pubg.com/en/news/10080)이다. UTC 기준 40.2는 3월 11일 08:30부터 4월 8일 00:00 전, 41.1은 4월 8일 08:30부터 5월 13일 00:00 전으로 배정했다. 마지막 구간의 끝은 이번 관측 자료를 덮는 6월 3일 00:00이며 다음 패치 출시일이라는 뜻은 아니다. 경기별 결과는 [패치 목록 CSV](/Users/lunecid/project/PUBG/reports/erangel_alignment_2026-10-15/erangel_match_patch_inventory.csv)에 있다.

**지도 이후 실제로 있었던 공간 변경**

| 공식 패치 | 관련 변경 | 확보한 자료에 대한 의미 |
|---|---|---|
| [11.2](https://www.pubg.com/en/news/1694), 2021 | Sosnovka 교량 폭·엄폐 배치·보행 통로 변경 | 2019 높이 자료로 현재 교량 구조를 보장할 수 없음 |
| [14.1](https://pubg.com/en/news/1714), 2021 | Pochinki 교회·수로·엄폐물·지형 높이 조정 | 같은 위치에서도 공간과 높이가 달라질 수 있음 |
| [21.1](https://www.pubg.com/en-asia/news/1357), 2022 | 페리 차량 탑승 시설과 부두, Hospital 남쪽 건물·울타리 등 변경 | 지형만으로 이동 연결을 확정할 수 없음 |
| [26.1](https://pubg.com/en/news/6717?category=patch_notes), 2023 | Stalber, Boatyard, Ruins, Mylta, Georgopol의 구조·지형 변경 | 2019 높이 자료와 2024 공식 이미지 사이에도 차이 존재 |
| [39.1](https://pubg.com/en/news/9466), 2025 | Subzero 지역, 부분 붕괴 교량, 남서쪽 평지, 얼어붙은 물 등 | 2024 공식 이미지 이후의 변경 |
| [40.1](https://pubg.com/en/news/9690), 2026 | 추위가 사라진 뒤에도 Subzero 흔적이 남음, 페리 복귀, 파괴 가능한 가스 탱크 추가 | 예전 상태로 완전히 복구됐다고 간주할 수 없음 |
| [40.2](https://pubg.com/en/news/9809), 2026 | 기념 장식·오브젝트 등 | 이번 정합 표본의 추정 패치. 변경 사항이 적다는 이유로 과거 지도와 동일하다고 인증하지 않음 |
| [41.1](https://www.pubg.com/en-asia/news/9926), 2026 | Erangel에 파괴 가능한 지형 도입 | 이후 경기는 경기 중 공간 변화도 고려해야 함 |

이 표는 버전 불일치를 입증하는 주요 변경 이력이다. 모든 핫픽스와 오브젝트 변경의 완전한 목록이나, 변경 없는 지역을 인증하는 목록은 아니다. 40.1에서 어느 Subzero 구조가 얼마나 남았는지도 패치 문구만으로 좌표별 확정하지 않는다. 따라서 전체 자료를 단일한 정적 지도에 묶지 않고, 이번 시제품은 지형 파괴 도입 전인 **40.2 추정 일반 TPP 스쿼드 40경기**로 제한했다.

**좌표 정합과 독립 검증**

[공식 Location 문서](https://documentation.pubg.com/en/telemetry-objects.html)에 따라 센티미터 좌표를 사용한다. Erangel의 816,000cm 범위를 8192px 래스터에 대응시켰다. 원점은 좌상단, x는 오른쪽, y는 아래쪽이다.

```text
x_m = x_cm / 100
y_m = y_cm / 100
u_pixel = x_cm × 8192 / 816000
v_pixel = y_cm × 8192 / 816000
```

8192픽셀을 무조건 8km로 처리하지 않는다. 위 변환에서 한 픽셀은 약 0.9961m다. x/y 교환 및 좌우·상하 반전도 별도로 대조했다.

40.2 추정, 일반, Steam, `squad`, 이벤트 모드 아님 조건의 939경기 중 경기 ID의 고정 SHA-256 순서로 40경기를 선택했다. 32경기는 정합과 관측 그래프 구성, 8경기는 독립 검증으로 분리했다. 선택 경기의 시작 시각은 한국 시간 3월 26일 21:23:34부터 4월 7일 20:36:41까지다.

`account.` 접두사와 참가자 목록으로 인간 후보를 정하고, 착지 후부터 첫 사망 전까지 생존 상태, 기록된 차량 없음, 유효 XYZ, 중복 없는 시각 조건을 적용했다. 인간 후보 원시 위치 273,151개 중 170,304개가 남았다. 연속 시점 간 5–15초, 수평 평균속도 12m/s 이하 조건을 만족한 연결은 160,991개다. 위치 필터만으로 도보·지면 접촉을 확정할 수 없으며, 수영·일부 특수 이동이 섞일 수 있다. 보간 경로는 만들지 않았다.

수직 스케일은 `player_z ≈ a × (raw_height − 32768) + b`로 추정했다. 검증 8경기와 보류한 500m 공간 블록을 정합 학습에서 제외했다. 알려진 변경 지역의 근사 범위와 RGB 수역 후보도 제외하고, 25m 셀당 한 점을 사용해 5,707점으로 robust fitting했다.

| 항목 | 측정값 | 해석 |
|---|---:|---|
| 수직 스케일 a | 0.00779437 m/raw | 인간 고도에 맞춘 추정값, 공식 지형 변환값 아님 |
| 고도 오프셋 b | 1.10210m | 플레이어 기준점·자료 오차가 섞인 값 |
| 미세 정합 dx, dy | −0.0489m, −0.4340m | 좌표의 절대 정확도를 나타내지 않음 |
| 수평 스케일 | 0.99994255 | 공식 기본 대응과 매우 가까움 |
| 독립 8경기 33,980점의 절대 고도 잔차 | 중앙값 0.458m, P90 7.133m | 지면 측량 오차가 아닌 플레이어 고도 잔차 |
| 수평 미세 조정 없는 기본 변환 | 중앙값 0.463m, P90 7.124m | 미세 조정의 실질 이득이 작아 기본 좌표식을 권장 |
| 보류 공간 블록의 독립 경기 5,933점 | 중앙값 0.545m, P90 8.177m | 변경 범위·수역 후보 밖만 포함한 추가 진단 |
| 잘못된 축 방향 7가지 | 중앙값 7.64–8.74m | 기본 방향이 더 타당함을 지지 |
| 높이 육지 후보와 RGB 육지 후보의 IoU | 0.920 | 두 휴리스틱의 일치도이며 정답 해안선 정확도 아님 |

낮은 중앙 잔차는 지도 버전 일치의 증거가 아니다. 건물·교량의 층위, 관측 시점의 이동 상태, 지형 변화가 모두 잔차에 영향을 준다. 변경 범위 안팎의 차이도 표본 구성이 달라 변경 효과로 해석하지 않았다. 독립 랜드마크 측량이 없으므로 XY 정합 RMSE를 보고하지 않는다.

![Erangel 정합 결과](/Users/lunecid/project/PUBG/reports/erangel_alignment_2026-10-15/overview.png)

왼쪽은 공식 지도 위 위치 표본과 높이 지도 육지 후보의 경계, 가운데는 역사적 원시 높이의 색상 표현, 오른쪽은 3경기 이상에서 지지된 관측 연결이다. 주황 원은 공식 변경 폴리곤이 아닌 분석자가 지정한 근사 범위다. 원 밖도 변경 없음이 검증된 곳은 아니다. 지명 중심은 기존 수집한 pubg.gs POI 자료를 사용했고, 교량·남서 평지는 근사 좌표를 지정했다.

**실제로 생성한 두 공간 그래프**

| 그래프 | 노드 | 엣지 | 의미 |
|---|---:|---:|---|
| 역사적 지형 인접 후보 | 12,624 | 무방향 24,264 | 50m 셀 중심의 raw height ≥32768을 육지 후보로 두고 상하좌우를 연결 |
| 관측 이동 | 6,182 | 방향 14,042 | 구성용 32경기에서 실제로 관측된 셀 간 연속 위치 연결 |

지형 그래프의 엣지에는 높이 차이와 수평 거리의 비율을 저장했다. 임의의 기울기 임계값으로 통행을 허용·차단하지 않았다. 육지 후보 임계값도 검증된 수면·해안 라벨이 아니다. 지형 래스터에는 건물 내부와 교량 상판이 없으므로 이 그래프는 navmesh가 아니다.

관측 그래프에서 3경기 이상이 지지하는 엣지는 2,433개다. 독립 8경기의 비자기 연결 10,921개 중 두 끝 셀이 모두 구성 그래프에 존재하는 비율은 83.52%, 같은 방향 엣지가 존재하는 비율은 70.55%다. 이는 관측 범위의 재현 정도이며 경로 예측 정확도나 통행 가능성 정확도가 아니다. 인접 관측 사이를 잇는 직선이 벽을 통과하지 않는다는 보장도 없다.

![School 주변 두 그래프 비교](/Users/lunecid/project/PUBG/reports/erangel_alignment_2026-10-15/graph_detail.png)

School 주변 확대에서 왼쪽 격자는 지형 인접 관계, 오른쪽 연결은 실제 플레이어의 관측 전이 관계다. 두 그래프가 다른 정보를 제공한다는 점과, 건물의 세부 통행 구조를 표현하지 못한다는 점을 함께 확인할 수 있다.

**Diffusion 연구에 적용할 범위**

현재 자료로 시작할 수 있는 것은 **관측 이동 그래프의 공간 임베딩과 역사적 높이 보조 특성**이다. 노드에 정규화 XY, 역사적 높이, 관측 빈도, 지지 경기 수, 고도 잔차, 버전 불일치·변경 범위 표시를 붙이고, 관측 전이와 지형 인접 관계는 서로 다른 엣지 유형으로 유지한다. 그래프는 학습 경기만으로 구축해야 하며 검증·시험 경기에서 새 엣지를 추가하지 않는다.

생성 모델은 현재 위치와 과거 이동, 현재 공개된 안전구역 조건, 지역 그래프 임베딩을 받아 미래 좌표열을 생성하는 방식으로 확장할 수 있다. 이번 단계에서 모델 학습이나 안전구역 조건부 생성 실험을 실행한 것은 아니다. 인간과 봇 비교에서는 어느 집단의 궤적으로 그래프를 만들었는지 고정해야 한다. 이번 인간 후보 기반 그래프를 중립적인 통행 정답으로 취급하면 봇 평가에 편향이 생긴다.

현재 자산을 근거로 벽 관통 손실, 절대 통행 마스크, 충돌 없는 경로의 성공률을 정의해서는 안 된다. 다음 단계는 40.2에 대응하는 지형·충돌 자료를 확보하거나, 검증 가능한 구역을 수작업·게임 관측으로 정하고 관측 그래프의 주장 범위를 제한하는 것이다. 41.1 이후 자료는 파괴 가능한 지형 때문에 초기 지도만으로 경기 전체의 지형 상태를 고정할 수 없다.

**결과 파일과 재현**

- [인터랙티브 뷰어](/Users/lunecid/project/PUBG/reports/erangel_alignment_2026-10-15/viewer.html): 자체 포함 HTML. 지도·높이 전환, 두 그래프, 지지 경기 수, 지역 확대, 노드 수치 조회. 외부 데이터 전송 없음.
- [정합 수치](/Users/lunecid/project/PUBG/reports/erangel_alignment_2026-10-15/alignment_metrics.json), [추출 정의](/Users/lunecid/project/PUBG/reports/erangel_alignment_2026-10-15/extraction.json), [고도 잔차 그림](/Users/lunecid/project/PUBG/reports/erangel_alignment_2026-10-15/alignment_diagnostics.png).
- [관측 노드](/Users/lunecid/project/PUBG/reports/erangel_alignment_2026-10-15/nodes.csv), [관측 엣지](/Users/lunecid/project/PUBG/reports/erangel_alignment_2026-10-15/edges.csv), [지형 노드](/Users/lunecid/project/PUBG/reports/erangel_alignment_2026-10-15/terrain_nodes.csv), [지형 엣지](/Users/lunecid/project/PUBG/reports/erangel_alignment_2026-10-15/terrain_edges.csv).
- `telemetry_probe.npz`: `points[170304,5]`는 경기 인덱스·경기 경과 초·x_m·y_m·player_z_m, `pairs[160991,4]`는 원본 점 인덱스 두 개·dt_s·displacement_m. 계정 ID는 내보내지 않았다. 경기 ID와 분할은 extraction.json에 보관했다.

재현 코드는 `tools/erangel_spatial_extract.py`, `tools/erangel_spatial_patch_inventory.py`, `tools/erangel_spatial_build.py`, `tools/erangel_spatial_visualize.py`와 `tools/erangel_spatial_viewer.html.template`이다. numpy, scipy, Pillow, matplotlib, psycopg2-binary가 필요하다. 원본 URL·해시는 manifest에 있고, 높이 지도와 공식 지도 파일별 GitHub 이력도 결과 폴더에 저장했다. build의 근사 변경 범위는 `reports/spatial_source_inventory_2026-10-15/pubggs_erangel_embedded_map.json`에 의존한다.

```bash
python tools/erangel_spatial_extract.py  # DB 비밀번호는 숨김 입력, 연결은 읽기 전용
python tools/erangel_spatial_patch_inventory.py
python tools/erangel_spatial_build.py
python tools/erangel_spatial_visualize.py
```

이미 추출된 자료가 있으면 DB 단계 없이 나머지 세 단계를 실행할 수 있다. 이번 실행은 기존 프로젝트 환경을 변경하지 않고 별도 `/private/tmp/pubg-spatial-venv`에서 수행했다. 생성 데이터·보고서 폴더와 대용량 지도는 기존 gitignore 정책에 따라 로컬 산출물로 남아 있다. 향후 외부 배포 시 원본 자산 재배포 범위는 별도 확인해야 한다.
