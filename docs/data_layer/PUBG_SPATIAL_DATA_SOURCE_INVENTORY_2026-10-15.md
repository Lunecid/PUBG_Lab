# PUBG 공간 데이터 출처 조사와 확보 가능성

조사일: 2026-10-15. 대상은 **PC PUBG: BATTLEGROUNDS**이다. PUBG Mobile, Game for Peace, 다른 게임으로 재현한 Erangel은 원본 PC 맵의 공간 정답에서 제외한다.

## 1. 결론과 기존 검토의 갱신

**PUBG 공간표현을 학습할 실제 공개 자료가 존재한다.** 공식 지도 이미지·텔레메트리 외에 수치형 높이 지도, 법선 지도, UGC 건물 메쉬와 배치 표현, 영상·입력 데이터셋을 확인했다. 따라서 연구 입력을 지도 이미지와 이동 로그만으로 제한할 필요가 없다.

다만 **현행 배틀로얄 맵 전체의 건물 내부·충돌·통행 가능성을 보증하는 공개 navmesh 데이터셋은 이번 조사에서 확인하지 못했다.** 높이 지도는 지표면의 2.5D 표현이고, 렌더링 메쉬는 게임 충돌체와 다르다. UGC의 배치 정보도 원래 Erangel 전체의 건물 배치를 제공하는 것은 아니다.

이 문서는 [앞선 공간표현 가능성 검토](../experiment_design/PUBG_SPATIAL_REPRESENTATION_FEASIBILITY_2026-10-15.md)의 **외부 공간 자산 가용성 판단을 확장**한다. 이전 DB 실측과 검증 파일은 그대로 보존한다. 이번에는 모델을 학습하거나 게임에서 추출기를 실행하지 않았다.

## 2. 조사 범위와 확인 수준

- 공식 API 문서, 공식 GitHub 자산, 공식 패치·UGC·Playgrounds 공지를 조사했다.
- PUBG GitHub 조직 목록 51개, 높이 지도 저장소의 fork 목록 34개를 조회하고 관련 후보를 선별했다. 모든 저장소의 모든 파일을 읽었다는 뜻은 아니다.
- 주요 GitHub 저장소 9개의 기본 브랜치 recursive tree를 저장했다. 응답은 모두 `truncated=false`였다. 커밋과 Git tree 식별자도 별도로 확인했다.
- cgcostume의 높이·법선 PNG 14개, mortarkiller의 높이 후보 PNG 9개에 대해 실제 헤더를 읽었다. PNG 전체 픽셀의 무결성·높이 정확도를 검증한 것은 아니다.
- UGC 공개 메쉬 샘플 2개는 gzip 해제 후 전체 바이너리 구조와 유한 좌표를 검증했다. 기본 레벨 GLB는 헤더와 JSON chunk를 읽었다.
- 커뮤니티 지도, 리플레이 도구, UGC 공유, 클라이언트 자산 추출, 런타임 충돌체 추출, 공개 연구 데이터셋까지 경로별로 조사했다.

**확인 수준:** `실물 확인`은 파일·응답 구조를 직접 읽은 것, `문서 확인`은 제작자나 공식 문서의 설명, `미검증`은 현재 클라이언트 실행·정확도·다운로드 등이 남은 것이다. 인터넷의 비공개 자료와 모든 개인 저장소까지 존재 여부를 증명하는 전수조사는 아니며, 아래는 조사 시점의 공개 경로별 목록이다.

## 3. 출처별 목록

| 경로 | 얻을 수 있는 자료 | 이번 확인 | 연구에서의 위치와 한계 |
|---|---|---|---|
| [PUBG 공식 API](https://documentation.pubg.com/en/telemetry-objects.html) | 플레이어/이벤트 x,y,z, 시간, 안전구역 상태 | 공식 스키마 확인; 기존 DB probe 별도 보존 | 행동·동적 조건. 지형 메쉬, 지면 높이 정답, navmesh가 아님 |
| [pubg/api-assets](https://github.com/pubg/api-assets) | 지도 PNG, 지역명 및 enum | 전체 tree, Erangel 실제 PNG 헤더 | 공식 배경 영상. 확인한 tree에서 지형·충돌·navmesh 자산 미발견 |
| [cgcostume/pubg-maps](https://github.com/cgcostume/pubg-maps) | 7종 맵/버전의 16비트 높이 및 법선 지도 | 14개 PNG 헤더와 추출 소스 | 가장 명확한 수치 지형 후보. 오래된 자산, 물리 단위와 패치 정합 필요 |
| [MingDMeng/mortarkiller](https://github.com/MingDMeng/mortarkiller) | 9개 맵의 `_hm.png` | 9개 헤더와 픽셀 사용 코드 | 신맵 보완 후보. 실제 파일은 8비트 RGB/RGBA, 정밀 DEM으로 간주 불가 |
| [PlzDntKilMeh/pugc-editor](https://github.com/PlzDntKilMeh/pugc-editor) | UGC 오브젝트 카탈로그, 렌더링 메쉬, 좌표 변환, `.pugc` 입출력 소스 | 소스·카탈로그·메쉬 2개·GLB 메타데이터 | 건물 단위 표현과 통제된 UGC 환경에 유용. BR 전체 맵과 충돌체는 별도 |
| [SomunsMo/pubg-ugc-map-cf](https://github.com/SomunsMo/pubg-ugc-map-cf) | 사용자 제작 `.pugc` 3개 | 파일 tree와 크기 | 실제 UGC 맵 파일 존재. 이번에 파일 decode·게임 load 미실행 |
| [공식 Playgrounds/CurseForge 안내](https://pubg.com/en/events/notice/10543) | 게임 내 제작·저장·공유 경로 | 공식 공지와 창작자 배포 페이지 | 직접 설계한 맵에서 공간·이동 정답 수집 가능성을 시험할 경로 |
| [pubg.gs](https://pubg.gs/maps/erangel/) | 지점·구역 좌표와 지도 레이어 | HTML 내 좌표 및 레이어 메타데이터 추출 | 원본 전체 레이어 JSON 요청은 403. 모든 지점을 확보했다고 보고하지 않음 |
| [PUBGMAPS.gg](https://pubgmaps.gg/) | 2D/3D 지형, POI, API 기반 리플레이 뷰어 | 공개 페이지와 제작자 설명 | 높이 지도 사용 사례. 학습용 원본 export와 버전·라이선스 미확인 |
| [Gamermaps](https://www.gamermaps.net/map/pubg/erangel) | 지형 표현 지도 타일, 게임 지점 | 페이지·공개 JS의 타일 경로 | 지형색·등고 표현 이미지와 수치 높이 배열은 구별해야 함 |
| [FModel 호환 목록](https://github.com/FModel/Game-Compatibility/blob/master/VERIFIED_WORKING.md), [UModel](https://www.gildor.org/smf/index.php?topic=3229.75) | 설치된 게임 자산을 읽는 후보 도구 | PUBG 지원 문서·버전 이력 | 현재 빌드의 완전한 레벨/충돌 추출 성공은 미검증 |
| [EpicKitten replay 문서](https://github.com/EpicKitten/PUBG-Resources/wiki/Replay-Files-Documentation), [numinit parser](https://gist.github.com/numinit/35364ee3b943b5886a86780b10dd599d) | 리플레이 구조·이벤트 해석 | 공개 소스·설명 | 오래된 도구. 리플레이 자체가 맵 geometry 패키지는 아님 |
| [D2E-Original](https://huggingface.co/datasets/open-world-agents/D2E-Original) | PUBG 영상·오디오·키보드·마우스 입력 | 데이터 카드와 PUBG 파일 목록 | 시각·행동 사전학습 보조. world pose/depth/navmesh 정답은 확인되지 않음 |
| [PhysxPUBG](https://github.com/Jackjun724/PhysxPUBG) | 런타임 PhysX geometry 접근을 설명하는 소스 | 제작자 README 수준 | 공개 완성 데이터셋이 아님. 실행 중 메모리·특수 장치 기반이며 이번 연구에서 미사용 |
| [PUBG Ally 논문](https://arxiv.org/html/2609.29837v1) | 개발사 내부 게임 에이전트 연구 사례 | 논문 | 일반 연구자가 바로 쓰는 공개 공간 SDK/맵 데이터 배포의 근거로 보지 않음 |

## 4. 수치 높이 지도: 지금 가장 실용적인 출발점

### cgcostume/pubg-maps

고정 커밋은 `8c0d610dd7ab7a9f0e731b73aead5218e9a34613`이며 커밋 날짜는 **2020-09-09**이다. 저장소 `pushed_at`은 2021년이지만 이를 자산의 제작 연도로 바꾸어 해석하지 않는다. `o2e/pubg-maps`도 확인한 기본 브랜치가 같은 커밋이므로 별도의 최신 높이 자료로 세지 않는다.

| 폴더/대상 | 실제 높이 PNG | 높이 encoding | 함께 확인한 법선 PNG |
|---|---:|---|---|
| erangel | 8192×8192 | 16-bit grayscale | 8192×8192, 8-bit RGB |
| erangelclassic | 8192×8192 | 16-bit grayscale | 같은 해상도, 8-bit RGB |
| miramar | 8192×8192 | 16-bit grayscale | 같은 해상도, 8-bit RGB |
| sanhok | 4096×4096 | 16-bit grayscale | 같은 해상도, 8-bit RGB |
| vikendi | 6144×6144 | 16-bit grayscale | 같은 해상도, 8-bit RGB |
| karakin | 2048×2048 | 16-bit grayscale | 같은 해상도, 8-bit RGB |
| jackal | 4096×4096 | 16-bit grayscale | 같은 해상도, 8-bit RGB |

Erangel의 파일은 `erangel/pubg_erangel_height_l16_lod0.png`이다. README에 보이는 512픽셀 8비트 preview 대신 원본 `lod0`를 사용해야 한다. 실제 파일 주소·헤더는 [heightmap_headers.json](../../reports/spatial_source_inventory_2026-10-15/heightmap_headers.json)에 기록했다. 자료 및 추출 설명: [고정 버전 저장소](https://github.com/cgcostume/pubg-maps/tree/8c0d610dd7ab7a9f0e731b73aead5218e9a34613).

**16비트라는 사실은 미터 단위 고도가 검증됐다는 뜻이 아니다.** 이번에 읽은 추출 경로는 높이 채널을 저장하며, 연구에 필요한 Landscape의 수직 scale/offset을 확정하지 못했다. `z=a*h+b` 같은 변환과 좌표 방향, crop, 원점, 지도 extent를 검증해야 한다. `a,b`를 공식 엔진/자산 정보로 얻거나 독립적인 지면 기준점으로 보정한다. 플레이어 z 전체를 지면 정답으로 사용하면 옥상·다리·점프가 섞인다.

Camp Jackal의 4096픽셀을 현행 공식 Range extent와 바로 등치하지 않는다. 맵 리메이크와 오래된 Vikendi의 크기 차이도 확인 대상이다. 실제 물리 경사는 수평·수직 축척이 확정된 뒤 계산한다.

### MingDMeng/mortarkiller

커밋 `22b3f2a6aeede0fc5d411d825e48f511e0583418`, 2024-10-19. Erangel, Miramar, Sanhok, Vikendi, Karakin, Paramo, Taego, Deston, Rondo의 파일을 확인했다. **9개 모두 8192×8192이지만 16비트 지형이 아니라 8비트 RGB/RGBA PNG**이다. 코드에서는 첫 색 채널의 차이에 `2.08`을 곱한다. 이는 작성자의 변환 가정이지 독립적으로 검증한 물리 높이 정답이 아니다. 모든 맵의 이미지 크기가 같으므로 픽셀당 거리를 일괄 고정해서도 안 된다. [소스](https://github.com/MingDMeng/mortarkiller/blob/22b3f2a6aeede0fc5d411d825e48f511e0583418/mortarkiller.py), [헤더 검사](../../reports/spatial_source_inventory_2026-10-15/mortarkiller_heightmap_headers.json).

README는 이 도구를 outdated로 표시하고 [후속 desktop 도구](https://github.com/Ashimdude/mortarkiller_desktop)를 안내한다. 후속 저장소의 기본 tree 23개 항목에서는 새 높이 PNG를 확인하지 못했다. 후속 도구의 최신 커밋만으로 지형 자료가 최신화되었다고 판단하지 않는다. 이 조사에서는 계산기나 overlay를 실행하지 않았다.

## 5. UGC는 별도의 유망한 3D 데이터 경로

[공식 UGC 설명](https://pubg.com/en/news/8865?category=notice)과 [2026 Playgrounds·CurseForge 공지](https://pubg.com/en/events/notice/10543)에서 게임 내 제작·공유 기능을 확인했다. 후자의 대회는 조사 시점에 종료된 행사이며, 현재 진행 중인 지원 프로그램으로 소개하지 않는다. [창작자의 배포 페이지](https://www.curseforge.com/pubg-battlegrounds/game-mods/device-usage-guide)는 `.pugc` 파일과 `%localappdata%\TslGame\Saved\UGC` 저장 경로를 안내한다. 실행·호스팅 조건은 실제 사용할 클라이언트 빌드에서 재확인한다.

`pugc-editor`의 고정 커밋은 `3bd69a10c3ad9ebe8f9aa44964a36784b7bce29b`, 2026-08-14이다. 소스에서 다음을 확인했다.

- `.pugc`를 decode한 JSON의 `objects` 또는 `modObjects` 배열을 읽는다. 이번에는 실제 `.pugc` decode와 round-trip 저장은 실행하지 않았다.
- 객체 카탈로그에서 `objectId`, `objectClass`, 분류와 게임 자산 경로를 연결한다.
- 좌표 변환 코드는 UE 좌표를 100으로 나누고 y/z 축을 바꿔 Three.js에 전달한다. 이것은 해당 편집기의 구현을 확인한 것이며, BR 텔레메트리와 UGC 원점이 동일하다는 검증은 아니다.
- 공개 바이너리 포맷에 vertex positions, normals, UV, material sections, instance matrices가 있다.

직접 확인한 수치는 다음과 같다.

| 검사항목 | 결과 | 해석 |
|---|---|---|
| `objects.json` | 실제 최상위 항목 **698개** | `meta.json`의 645라는 수치와 불일치. 메타데이터만 신뢰하지 않음 |
| `dump/bp-mesh/*.bin.gz` | **741개 파일**, blob hash 중복 제거 시 710개 | 741개 고유 건물이라는 뜻은 아님. 장식·복합 오브젝트 포함 |
| 작은 객체 `1.bin.gz` | mesh 1개, instance 1개, 구조 전체 parse 성공 | 실제 geometry 파일 존재 확인 |
| 건물 `10009.bin.gz` | mesh 132개, instance 3,050개, 구조 전체 parse 성공 | 해당 prefab 내부 구성. Erangel 전체 건물 수가 아님 |
| `cache/mod_main_level.glb` | GLB v2, 11,579,416 bytes; JSON에 node 17개, mesh 10개 | UGC 기본 레벨 후보. 전체 binary buffer와 게임 충돌은 미검증 |

출처: [mesh decoder](https://github.com/PlzDntKilMeh/pugc-editor/blob/3bd69a10c3ad9ebe8f9aa44964a36784b7bce29b/scene-editor/mesh-assets.js), [좌표 변환](https://github.com/PlzDntKilMeh/pugc-editor/blob/3bd69a10c3ad9ebe8f9aa44964a36784b7bce29b/scene-editor/coords.js), [객체 카탈로그](https://github.com/PlzDntKilMeh/pugc-editor/blob/3bd69a10c3ad9ebe8f9aa44964a36784b7bce29b/data/catalog/objects.json), [실물 메쉬 검사](../../reports/spatial_source_inventory_2026-10-15/pugc_mesh_validation.json).

**연구 활용:** 문·계단·복도·우회 경로를 통제한 UGC 맵을 만들고, 알고 있는 객체 배치와 실제 게임 이동을 비교하는 검증 환경으로 쓸 수 있다. mesh로 계산한 길과 게임에서 실제 통과한 길이 일치하는지 먼저 시험한다. 맵 구조를 바꾼 held-out 환경을 구성하면 단순 위치 암기를 분리하기 쉽다.

UGC에 PUBG 건물 자산이 있다는 것과 정규 BR 맵의 모든 건물·층·문 상태를 복원할 수 있다는 것은 다르다. UGC에서 학습한 표현을 BR 데이터에 적용할 때에도 객체 구성·이동 규칙·지도 분포 차이를 평가해야 한다.

## 6. 커뮤니티 지도와 지점 좌표

### pubg.gs

[사이트 제작자의 출처 설명](https://pubg.gs/about/)은 게임 Level actor와 DataTable에서 좌표를 추출한다고 밝힌다. 사이트가 표시한 기준은 Update 43.2, Steam build 25901960, 추출일 2026-10-14이다. 이는 **게시자의 출처 표시**이며 원본 게임 파일과 대조한 결과는 아니다.

Erangel 공개 HTML에 포함된 JSON에서 실제로 `extent=816000`, POI 26개, zone entry 57개와 x/y/r 등의 필드를 추출했다. 레이어 counts 합은 27,275이다. 이 수치는 고유 건물 수가 아니며 item/vehicle/zone 등 여러 범주의 합이다. `zones` 배열 길이 57과 count의 `zones=43`도 같은 모집단으로 가정하지 않는다. HTML은 `/layers/baltic.json`을 전체 레이어 경로로 지정하지만 요청이 **HTTP 403**이어서 전체 지점은 확보하지 못했다. [추출한 HTML 내 지도 객체](../../reports/spatial_source_inventory_2026-10-15/pubggs_erangel_embedded_map.json), [Erangel 표](https://pubg.gs/maps/erangel/tables/).

구역 중심·반경과 초기화 규칙은 특정 경기에서 실제 결정된 미래 안전구역과 구별한다. 공간 표현에는 정적 지점 정보로, 안전구역 조건에는 해당 시점에 관측 가능한 경기 상태로 사용한다.

### PUBGMAPS.gg와 Gamermaps

[PUBGMAPS 제작자](https://www.reddit.com/r/threejs/comments/1uyw8z8/i_built_a_pubg_map_explorer_and_match_replay/)는 heightmap과 KTX2 texture 기반 3D 뷰어라고 설명한다. [다른 제작자 설명](https://www.reddit.com/r/PUBATTLEGROUNDS/comments/1uz3390/i_built_a_free_interactive_pubg_map_site_3d_view/)에는 minimap 가공에 AI·OCR·inpainting을 쓴다고 되어 있다. 따라서 화면이 원시 지형 정답이라는 보장은 없다. landing HTML은 읽었지만 linked JS 요청은 403이었다. 원본 terrain export나 학습용 데이터 제공은 확인하지 못했다.

[Gamermaps](https://www.gamermaps.net/map/pubg/erangel)의 공개 JS에서 terrain JPG tile 주소를 확인했다. 이는 배경 raster의 존재를 뒷받침하며, 높이 수치·building polygon·충돌 정보가 공개되었다는 뜻은 아니다. 두 사이트 모두 시각 정합과 수동 검수 보조로 우선 사용한다.

## 7. 게임 안에서 직접 얻는 경로

| 방법 | 직접 관측/저장 가능한 것 | 미해결 부분 |
|---|---|---|
| 일반 플레이·리플레이 자유 카메라 | RGB 화면, landmark, 문/계단/장애물, 이동 성공 사례 | depth·정밀 camera pose·collision 정답이 자동으로 따라오지 않음 |
| Custom Match에서 통제된 이동 | 출발/도착·경과시간·통과 성공/실패, 동기화 영상 | 경로 실패 원인이 지형인지 조작/문 상태인지 구별해야 함 |
| UGC 제작·저장 | 설계한 객체 배치, 장치·게임 규칙, 공유 mode file | 현행 버전 parser, export 완전성, 이동/텔레메트리 기록 지원 검증 필요 |
| 설치된 클라이언트 자산 읽기 | Landscape·StaticMesh·Level actor·DataTable 등의 추출 후보 | 빌드별 파일 형식, instance transform, collision, 동적 상태를 함께 해결해야 함 |
| 영상 기반 SfM/NeRF/3DGS | 관측된 표면의 시각적 재구성 | 보이지 않는 내부·미터 축척·충돌·통행 가능성은 별도 검증 |

[공식 리플레이 조작 안내](https://support.pubg.com/hc/en-us/articles/115004615754-Are-there-any-hotkeys-for-the-replay-mode)는 자유 카메라 등 관찰 수단의 근거다. 오래된 replay parser가 현재 빌드의 모든 actor transform을 복원한다고 주장하지 않는다. 리플레이를 3D로 재생할 때 사용하는 지도는 설치된 게임 자산에서 오므로 replay 파일만으로 3D 맵을 얻을 수 있다고 단정할 수 없다.

FModel 호환 표는 **데이터 파싱 호환성**이며 ModelViewer 등의 모든 기능을 보증하지 않는다고 명시한다. UModel 포럼의 확인 가능한 PUBG 지원 이력도 특정 과거 빌드를 대상으로 한다. 따라서 도구 지원 목록만으로 2026-10 클라이언트의 전체 지형 추출 성공을 선언하지 않는다. 이번 호스트의 기본 Steam 경로에서는 PUBG manifest와 `common/PUBG`를 발견하지 못했다. 다른 디스크·원격 PC까지 설치 부재를 확인한 것은 아니다.

## 8. 추가 학습 데이터와 제외한 후보

**D2E:** [데이터 카드](https://huggingface.co/datasets/open-world-agents/D2E-Original)는 PUBG 4.9시간, 영상/음성 및 동기화 키보드·마우스 기록을 설명한다. [PUBG 폴더](https://huggingface.co/datasets/open-world-agents/D2E-Original/tree/main/PUBG)에 `.mcap`/`.mkv` 쌍이 있고 총 43.1GB로 표시된다. CC BY-NC 4.0 표기가 있다. 이번에는 카드·파일 목록을 확인했으며 대용량 영상을 다운로드하거나 world pose 정답을 발견한 것은 아니다. 시각적 장소 표현·행동 사전학습에는 후보지만, 기존 DB의 match ID와 자동 join되는 데이터로 간주하지 않는다.

| 후보 | 판단 근거 |
|---|---|
| `ambienthaze/PUBG-Heightmaps-35.1` | 확인한 기본 브랜치 tree는 64바이트 README 하나뿐. 이름에 35.1이 있어도 높이 파일을 확보한 것이 아님 |
| `o2e/pubg-maps` | 원본과 같은 기본 브랜치 커밋. 새로운 자산으로 중복 집계하지 않음 |
| `Ashimdude/mortarkiller_desktop` | 최신 도구 코드가 있어도 확인한 tree에 새 높이 PNG 없음 |
| Sketchfab PUBG model 모음 | 검색 후보는 있으나 페이지 조회 403. 개별 맵의 출처·다운로드·정확도 검증 전 채택하지 않음 |
| PUBG Mobile/Game for Peace UGC·navmesh | 게임·맵 버전과 런타임이 다름. PC PUBG 공간 정답으로 합치지 않음 |
| 다른 게임에 재현한 Erangel, AI 생성 3D | 유사한 외형과 실제 PUBG geometry를 구별. 정확한 지형 데이터로 제외 |
| match 통계·death-coordinate 데이터 | 행동 데이터는 될 수 있지만 terrain/충돌체 원본은 아님 |

검색에서 특정 키워드 결과가 0건이었다는 사실은 공개 navmesh가 어디에도 없다는 증명이 아니다. `PhysxPUBG` 같은 runtime 접근 소스도 발견했으므로 기술적 추출 가능성과 바로 사용할 수 있는 공개 데이터셋을 분리해서 판단한다.

## 9. 공간표현·그래프·Diffusion으로 연결하는 실행안

### 1단계: Erangel 외부 공간 입력 정합

공식 무문자 지도, cgcostume 높이/법선, 기존 로그를 소규모 영역에서 먼저 맞춘다. 오래된 heightmap을 최근 경기와 무조건 결합하지 않는다. 경기 기간과 자산 버전을 비교하고 변하지 않은 영역 또는 대응 가능한 과거 자료를 선택한다. 랜드마크를 여러 지형·지도 구역에 나누어 학습용 보정점과 독립 검증점으로 분리한다.

저장 항목은 `map_key`, `game_build`, `asset_version`, `world_crs`, `world_to_raster`, `horizontal_scale`, `vertical_scale`, `vertical_offset`, `unknown_mask`, `alignment_error_m`, `source_url`, `source_hash`, `license_status`이다. 해시와 버전이 있어도 지형 정확성이 보증되는 것은 아니다.

공식 [Update 41.1](https://www.pubg.com/en-asia/news/9926)은 Erangel의 파괴 가능한 지형을 명시한다. 따라서 정적 자산과 특정 경기 시점의 지형 상태를 분리한다. 원래 지형에서 계산한 공간 특징을 사용했다면 그 범위를 논문에 밝힌다.

### 2단계: 2.5D 공간 encoder와 확률적 이동 그래프

지역 입력은 지도 RGB, 보정된 높이, 경사, 데이터 신뢰도/결측 mask를 기본으로 한다. 수역·도로·건물 마스크는 이미지 추론 라벨과 수동 검증 라벨을 구분한다. 인접 지형 그래프와 로그에서 확인한 이동 연결은 별도 edge type으로 둔다. 관측되지 않은 이동을 통행 불가로 표시하지 않는다.

출력은 지역 embedding `Z∈R^(N×d)`와 조건부 이동 비용/연결 확률이다. 이는 설계 표기이며 이번에 runtime shape나 모델 성능을 측정한 결과가 아니다. 기준 비교는 좌표만, RGB만, RGB+높이, RGB+높이+이동 그래프로 둔다. 구역 정보는 정적 지형의 정답에 흡수하지 않고 시점별 조건으로 넣는다.

공간표현 평가는 위치 보간뿐 아니라 미관측 지역의 이동 시간/목적지 분포, 인간과 봇에 공통으로 적용 가능한 지형 특징, 지역 분리 일반화를 포함한다. train 데이터에서 만든 이동 그래프에 test 궤적을 섞지 않는다. 일부 교량·다층 건물·절벽은 heightmap 한 층으로 표현할 수 없으므로 unknown 또는 별도 다층 노드로 처리한다.

### 3단계: UGC를 통한 국소 통행 검증

공개 prefab 또는 직접 배치한 UGC의 geometry를 연구용 graph로 바꾸고, 게임 내 통과 실험과 비교한다. 시각 메쉬에 자체 경로 탐색을 적용해 얻은 navmesh는 **연구자가 만든 근사 navmesh**로 표기한다. PUBG 봇이 실제 사용하는 공식 navmesh라고 부르지 않는다. 문·창문·계단·난간·점프·낙하·차량 등 이동 모드별로 성공/실패를 따로 기록한다.

### 4단계: 안전구역 조건부 경로 생성

Diffusion에는 과거 궤적, 현재 위치/이동 모드, 관측 시점의 안전구역, 지역 embedding 및 graph 정보를 전달한다. 전체 경로를 직접 생성하거나 상위 구역 경로와 하위 연속 궤적을 나누는 방식을 비교할 수 있다. 그래프는 불확실성을 가진 공간 조건으로 시작하고, 검증된 통행 제약만 강제한다. 시점에 공개되지 않은 미래 circle은 입력에서 제외한다.

수업 논문에서는 **정합된 지형 정보가 RGB·좌표·로그만 쓴 모델보다 이동 예측/생성에 도움이 되는가**를 검증한다. FDG 확장은 **지형과 이동 습관을 분리한 표현이 인간·봇 비교 및 미관측 공간에서의 생성 타당성을 개선하는가**로 잡을 수 있다. UGC 실험은 통행 타당성 검증을 보강하지만, BR 경기 분포에서의 평가를 대체하지 않는다.

## 10. 채택 우선순위와 남은 조건

1. **우선 확보·정합:** cgcostume 16비트 높이/법선 + 공식 지도 + 기존 로그. 현재 가장 직접적인 지형 학습 경로다.
2. **병행 소규모 검증:** pugc-editor의 건물 자산과 직접 설계 UGC. 국소 구조와 실제 통행 검증에 적합하다.
3. **보완 후보:** mortarkiller의 신맵 높이, pubg.gs 좌표, D2E 영상. 해상도·출처·버전·결합 가능성을 각각 검증한다.
4. **후속 조사:** 현행 클라이언트에서 Landscape+instance transforms+collision의 완전한 추출 가능성. 데이터가 부족한 경우에 이 경로의 비용을 따로 판단한다.

cgcostume, mortarkiller, pugc-editor는 확인한 GitHub 메타데이터에서 라이선스가 식별되지 않았다. 공개 접근 가능성과 논문 부속 데이터의 재배포 가능성을 같게 보지 않는다. 데이터셋 배포 설계에는 원본 출처·해시·변환 코드와 권리 확인 상태를 포함한다. UGC 예제 저장소의 MIT 표기도 포함된 모든 게임 자산의 권리를 일괄 해결하는 근거로 사용하지 않는다.

현재 확보 수준으로 **거시적인 지형 조건부 공간표현 연구는 진행 가능**, **UGC 기반 국소 3D 검증은 유망하지만 게임 실험이 필요**, **현행 BR 전체의 정밀 충돌·navmesh 정답은 미확보**로 판정한다.

## 11. 재현 자료

- [저장소 버전 목록](../../reports/spatial_source_inventory_2026-10-15/repository_versions.json): 9개 저장소의 commit/tree/date.
- [높이·법선 헤더](../../reports/spatial_source_inventory_2026-10-15/heightmap_headers.json): 14개 원본 URL 및 PNG 형식.
- [추가 높이 후보 헤더](../../reports/spatial_source_inventory_2026-10-15/mortarkiller_heightmap_headers.json): 9개 신맵 포함 후보.
- [UGC 메쉬 구조 검사](../../reports/spatial_source_inventory_2026-10-15/pugc_mesh_validation.json).
- [UGC 기본 GLB 메타데이터](../../reports/spatial_source_inventory_2026-10-15/pugc_base_level_glb_metadata.json).
- [공개 HTML에서 읽은 Erangel 좌표 객체](../../reports/spatial_source_inventory_2026-10-15/pubggs_erangel_embedded_map.json).
- [검증 기록과 파일 해시](../../reports/spatial_source_inventory_2026-10-15/validation.json).

조사 중 외부 코드는 읽기만 했으며 실행하지 않았다. 직접 작성한 구조 검사로 공개 파일을 읽었다. 게임 실행·클라이언트 자산 추출·런타임 메모리 접근·계정 로그인·학습은 수행하지 않았다.
