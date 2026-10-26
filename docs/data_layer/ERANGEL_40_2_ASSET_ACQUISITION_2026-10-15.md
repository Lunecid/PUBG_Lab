# Erangel 40.2 공간 자산 확보 경로 검증

2026년 10월 15일 조사. 사용자는 당시 게임 설치 폴더 백업과 리플레이를 보유하지 않는다고 확인했다. **40.2와 일치한다고 검증된 공개 높이·건물·교량 데이터 묶음은 찾지 못했다. 가장 구체적인 확보 경로는 당시 Steam Content depot의 manifest를 식별해 배포본을 확보한 뒤, 지형과 오브젝트 배치를 추출하는 것이다. 리플레이만으로 과거의 정적 공간 원본을 대체할 수는 없다.**

이 문서는 공개 자료 및 소스 코드 검사 결과와 실행 계획을 구분한다. 이번 조사에서 새 40.2 게임 자산을 다운로드하거나 게임 패키지 추출에 성공한 것은 아니다. 공개 저장소 6개의 커밋·파일 목록과 관련 소스 13개를 고정해 읽었다. AI를 이용해 자료 검색, 코드 검토, 근거 정리를 수행했다.

**공개 자료의 실제 상태**

| 후보 | 이번 확인 | 40.2 자료로 사용할 수 있는가 |
|---|---|---|
| [PUBG 공식 api-assets](https://github.com/pubg/api-assets) | 앞선 파일 이력 검사에서 Erangel 공식 이미지가 2024년 자료임을 확인 | 40.2 높이·건물 메시·충돌 자료가 아님 |
| [cgcostume/pubg-maps](https://github.com/cgcostume/pubg-maps) | 높이 추출 스크립트와 역사적 래스터, 앞선 실측 자산의 파일 날짜는 2019년 | 추출 방식의 선례. 40.2 정답으로 사용 불가 |
| [ambienthaze/PUBG-Heightmaps-35.1](https://github.com/ambienthaze/PUBG-Heightmaps-35.1) | 고정 커밋 df0410d…의 tree는 README 1개. main 외 브랜치 없음, release·tag 없음 | 이름만 보고 실제 높이 지도를 확보했다고 판단할 수 없음. 버전도 35.1 |
| [PUBGMAPS](https://pubgmaps.gg/updates) | 2026년 7월 6일 출시, 7월 15일 42.2 자산 업데이트를 공지 | 3D 지형 서비스는 있으나 공개 변경 이력에서 40.2 원본·manifest·다운로드 묶음은 확인하지 못함 |
| [pubg.gs](https://pubg.gs/about/) | 게시자는 game Level actor·DataTable 추출이라고 설명. 현재 표기는 Update 43.2, Steam build 25901960 | 추출 사례와 문의 후보. 현재 데이터는 40.2가 아님 |
| [pubg.gs 변경 이력](https://pubg.gs/changelog/) | 공개 비교 이력은 최근 빌드 3개로 표시 | 공개 페이지에서 40.2 보관본 확인 안 됨. 운영자의 비공개 보관 여부는 모름 |

검색 범위는 PUBG PC의 공식 API 자산, 공개 GitHub 높이 지도·지도 추출 도구, 3D 지도 서비스, Steam 배포 메타데이터, Epic 리플레이 문서와 PUBG 리플레이 관련 공식 공지다. PUBG Mobile, AI 생성 모델, 지명만 있는 지도, 빌드 출처 없는 팬 제작 메시를 버전 일치 원본으로 채택하지 않았다. 검색 부재는 세상에 자료가 없다는 증명이 아니다.

**경로 1 당시 Steam 배포본 확보**

[SteamDB의 앱 depot 목록](https://steamdb.info/app/578080/depots/)에서 다음 식별자를 확인했다.

| 식별자 | 값 | 역할 |
|---|---|---|
| Steam App ID | 578080 | 일반 PUBG PC 게임. Dev/Test/Esports 서버와 구분 |
| Content depot | 578081 | 지도와 자산 패키지 확보의 우선 대상 |
| Binaries depot | 578082 | 대응 실행 파일·버전 조사 후보. 메시 추출과 게임 실행은 다른 작업 |

현재 [Content 파일 목록](https://steamdb.info/depot/578081/manifests/)에는 `TslGame/Content/Paks`, `PakList.json`, `*_asset_registry.pak`, `*_maps.pak`, `*_assets_building.pak`, `*_assets_world.pak` 등이 있다. 이는 최근 배포본에서 확인한 파일 패턴이다. **40.2의 실제 파일 목록과 이름은 해당 manifest를 받아 확인해야 한다.** 여러 pakchunk와 업데이트 패키지 사이에 의존성이 있으므로 이름에 maps가 들어간 파일 하나만 받으면 충분하다고 가정하지 않는다.

이번 정합에 사용한 40경기의 시기는 한국 시간 2026년 3월 26일~4월 7일이다. [40.2 공식 PC 일정](https://pubg.com/en/news/9809)과 [41.1 일정](https://www.pubg.com/en-asia/news/9926)으로 패치를 추정했으나 정확한 서버·클라이언트 빌드까지 확정한 것은 아니다. 같은 40.2 기간에도 핫픽스 manifest가 여럿일 수 있다. 지도 관련 파일의 해시가 동일한지 비교해 하나의 공간 버전으로 합칠 수 있는지 판단해야 한다.

확보 순서는 다음과 같다.

1. SteamDB에서 App 578080의 public 배포 이력과 depot 578081의 3월~4월 manifest를 확인한다. Dev Server 또는 Esports Server의 같은 날짜 배포본을 일반 경기 자료에 대입하지 않는다.
2. 경기 시작 시각 전후의 배포 시점, build ID, manifest ID를 표로 만든다. 배포 시각은 서버가 모든 큐에 같은 상태를 적용했다는 보증이 아니므로 공식 패치와도 대조한다.
3. [DepotDownloader](https://github.com/SteamRE/DepotDownloader)의 `-manifest-only`로 파일 목록부터 확인한다. 과거 manifest를 지정하는 기능은 구현돼 있지만, 특정 40.2 manifest가 지금 다운로드 가능한지는 Steam의 접근 권한과 보존 상태를 실제로 시험해야 한다.
4. 파일 목록에 근거한 `-filelist`로 관련 콘텐츠를 별도 디렉터리에 받는다. 높이·레벨·건물/교량 메시·공유 종속 자산·registry·patch overlay를 포함하도록 목록을 구성한다. 읽는 패키지와 참조 종속성이 모두 해결되는지 확인한다.
5. manifest와 원본 파일 해시를 보관한다. 서로 다른 날짜의 콘텐츠를 임의로 섞지 않는다.

아래는 **manifest가 확인된 뒤 사용할 명령 형식**이다. 아직 실행해 성공한 명령이 아니며, `<...>`는 실제 값으로 대체해야 한다. 전체 게임을 받기 전에 목록만 확인하는 단계다.

```text
DepotDownloader -app 578080 -depot 578081 -manifest <CONFIRMED_MANIFEST_ID> -os windows -manifest-only -username <STEAM_USERNAME> -dir <AUDIT_DIRECTORY>
```

도구 문서상 비밀번호 인자를 생략하면 대화형 입력을 사용한다. DB 비밀번호와 Steam 로그인은 전혀 별개다. 과거 콘텐츠 접근 성공을 확인한 뒤에만 파일 목록을 지정한 다운로드로 넘어간다.

**현재 미해결 지점:** 공개로 읽은 SteamDB manifest 페이지는 최근 10개까지만 표시하고, 3월~4월에 도달하려면 Steam 로그인을 요구한다. 따라서 이번 조사에서는 40.2의 정확한 build/manifest 번호를 확보하지 못했다. SteamDB 로그인은 이력 열람이고, Steam 콘텐츠 다운로드 권한 확인은 별도 단계다. 번호를 안다는 사실만으로 파일 다운로드가 보장되지는 않는다.

**경로 2 게임 콘텐츠에서 지형과 오브젝트 배치 추출**

FModel 호환 목록에는 PUBG의 `GAME_PlayerUnknownsBattlegrounds` 프로필이 있다. 그러나 해당 목록은 추가 당시의 데이터 파싱 호환성을 뜻하며, Model Viewer 등의 모든 기능을 보장하지 않는다고 명시한다. 이번에 고정한 호환 목록 저장소의 마지막 커밋은 2024년 5월 30일이므로 이를 2026년 40.2 검증으로 해석하지 않는다. [고정 호환 목록](https://github.com/FModel/Game-Compatibility/blob/3365976f8677f21e8d2d64a7b5ce20d7e4617a9e/VERIFIED_WORKING.md)

CUE4Parse는 `0415fe26eeba66ac0d814c2547b098219682066f`로 고정해 아래 구현을 직접 읽었다.

| 필요한 데이터 | 확인한 구현 | 판단 |
|---|---|---|
| 지형 타일의 높이와 배치 | `ULandscapeComponent`: SectionBaseX/Y, HeightmapScaleBias, component size. `LandscapeDataAccess`: RG에서 16비트 높이 조합 | 읽을 수 있는 패키지를 확보하면 높이 복원 경로가 있음 |
| 지형 메시·높이 출력 | `LandscapeMeshExporter`: Landscape DTO에서 메시와 heightmap PNG 출력 | 일반 구현은 있음. PUBG 40.2 샘플 실행은 미검증 |
| 레벨의 오브젝트 목록 | `UWorld`, `ULevel.Actors` | 건물 메시와 별도로 인스턴스의 배치 정보를 추출할 기반 |
| 반복 배치된 건물·시설 | `UInstancedStaticMeshComponent`, BlenderUMap의 HISM 처리 | 메시 하나만 내보내는 것보다 인스턴스별 transform 처리가 필요 |
| PUBG 특수 구성요소 | `TslGame.cs`의 문·유리·DoorFrame 등 StaticMeshComponent 계열 등록 | PUBG 특수 클래스 지원 일부 존재. 게임 로직 완전 복원이라는 뜻은 아님 |
| 건물 물리 데이터 | `UBodySetup`의 AggGeom·CookedFormatData | 물리 데이터 접근 기반. 실제 40.2 충돌 형태와 동작의 동등성 미검증 |
| 지형의 cooked collision | `ULandscapeHeightfieldCollisionComponent`의 PUBG 분기 | bulk data를 생성해 읽지만 반환·보관하지 않음. 이 구현만으로 충돌 지형 완성 불가 |

핵심 근거: [Landscape exporter](https://github.com/FabianFG/CUE4Parse/blob/0415fe26eeba66ac0d814c2547b098219682066f/CUE4Parse-Conversion/Exporters/LandscapeMeshExporter.cs), [높이 디코딩](https://github.com/FabianFG/CUE4Parse/blob/0415fe26eeba66ac0d814c2547b098219682066f/CUE4Parse-Conversion/Dto/LandscapeDataAccess.cs), [PUBG 충돌 분기 28–29행](https://github.com/FabianFG/CUE4Parse/blob/0415fe26eeba66ac0d814c2547b098219682066f/CUE4Parse/UE4/Assets/Exports/Component/Landscape/ULandscapeHeightfieldCollisionComponent.cs#L28), [UBodySetup](https://github.com/FabianFG/CUE4Parse/blob/0415fe26eeba66ac0d814c2547b098219682066f/CUE4Parse/UE4/Objects/PhysicsEngine/UBodySetup.cs).

따라서 목표를 세 층으로 나누는 것이 정확하다. **지형 높이**, **건물·교량의 시각적 메시와 배치**, **게임과 동등한 충돌·통행 구조**는 각각 검증해야 한다. 앞의 두 가지에는 구체적인 공개 구현이 있고, 마지막 단계에는 추가 파서·물리 검증 또는 제작사 제공 자료가 필요할 가능성이 크다.

추출 후에는 다음 처리가 필요하다.

- 지형: 타일과 component offset, scale, world transform을 적용한다. 소스의 local 높이 변환은 `(raw-32768)/128`이지만 world scale과 좌표 단위를 적용하기 전 값이다. 지난 텔레메트리 fitting의 0.00779를 새 파일에 무조건 재사용하지 않는다.
- 배치: actor → component → parent attach → instance의 transform을 합성한다. 메시 경로, 위치, 회전, 스케일, 인스턴스 ID, 레벨, 원본 해시를 함께 저장한다.
- 맵 전체성: persistent level뿐 아니라 해당 모드에 사용된 하위 레벨을 포함한다. [BlenderUMap의 확인한 진입점](https://github.com/TrevTV/BlenderUMap/blob/564e89640ecc173fc4eb5413c9a51874f7204bdb/Source/Program.cs)은 PersistentLevel.Actors를 순회하고 일부 actor 유형만 처리하므로 한 번 실행하면 Erangel 전체가 완성된다고 가정할 수 없다.
- 교량과 건물: 바닥·상판·지붕·다층 구조를 별도 표면으로 유지한다. 단일 `height(x,y)`로 합치면 교량 아래와 위, 건물 층이 사라진다. 공간 그래프는 `(x,y,z,layer)` 또는 다층 표면으로 만든다.
- 충돌: 시각 메시와 collision proxy를 분리하고 문·창문·장애물의 동적 상태는 초기 지형과 별도로 표시한다. 시각 메시 기반 ray casting 결과를 실제 게임의 통행 정답이라고 명명하지 않는다.

PUBG 고유 패키지 포맷, 암호화·이름 매핑, 해당 빌드에 맞는 파서 설정 때문에 파일 보유만으로 성공이 보장되지는 않는다. 이번에는 보호 우회나 실행 중 게임에 대한 접근을 수행하지 않았다. 조사 목적에 필요한 것은 보존한 파일의 오프라인 읽기와 검증이다.

**경로 3 리플레이에서 얻을 수 있는 것**

[Epic 공식 Replay 설명](https://dev.epicgames.com/documentation/unreal-engine/using-the-replay-system-in-unreal-engine)은 DemoNetDriver가 replication 데이터를 기록하고 재생한다고 설명한다. 이는 게임 사건과 상태를 복원하는 시스템이다. 문서의 UE 일반 구조와 [PUBG replay 파일 문서](https://github.com/EpicKitten/PUBG-Resources/wiki/Replay-Files-Documentation)를 종합하면, 리플레이가 정적 건물 메시·지형 높이·충돌 파일 전체를 담은 독립 3D 공간 묶음이라고 볼 근거는 없다. 렌더링에 필요한 설치 자산은 별도로 필요하다.

| 용도 | 가능 범위 |
|---|---|
| 날짜·맵·경기 ID 확인 | replay 메타데이터를 조사할 가치 있음. 필드는 실제 버전의 파일에서 확인해야 함 |
| 플레이어·차량·문 등 동적 상태 | 기록 범위와 복제 항목, 파서 호환성에 따라 일부 추출 가능성 있음 |
| 높이 지도·건물 메시·교량 전체 복원 | 리플레이 단독 확보 경로로 채택할 수 없음 |
| 게임 내 자유 카메라 관찰 | 호환 클라이언트에서 가능한 표본 검증 수단. 자동으로 정확한 3D 데이터가 출력되는 것은 아님 |
| 현재 API의 telemetry JSON으로 과거 .demo 생성 | 문서화된 변환 경로 없음. API 이벤트와 엔진 replay는 다른 형식 |

EpicKitten 문서는 2018년 자료이며, 거기에 기재된 `Changelist`는 당시 항상 0이었다. 리플레이가 있으면 정확한 2026년 build를 무조건 알 수 있다는 주장도 피해야 한다. PUBG는 [공식 12.2 공지](https://www.pubg.com/en/news/1740)에서 리플레이 업데이트 후 이전 파일을 사용할 수 없다고 명시한 사례가 있다. 이는 현재의 모든 버전 쌍을 시험한 결과는 아니지만 과거 리플레이 호환을 가정할 수 없다는 근거다.

우리 DB의 자료는 [공식 API에서 설명하는 이벤트 JSON](https://documentation.pubg.com/en/telemetry.html)이며, 로컬 `PUBG.demo` 보관본이 아니다. 사용자는 당시 replay도 없다고 확인했으므로, 이번 확보 작업에서 리플레이는 주 경로가 아니다.

**외부 제공을 요청한다면 필요한 범위**

Steam 과거 배포본 접근이 실패하면 KRAFTON의 연구용 제공 또는 공개 지도 제작자의 보관본 제공을 확인하는 경로가 남는다. 현재 추출물을 만드는 [pubg.gs 연락 창구](https://pubg.gs/contact/)와 PUBGMAPS 제작자가 후보지만, 이들이 40.2를 보유하고 있다는 증거는 없다. 연락을 보내지는 않았다.

요청할 데이터는 다음과 같다.

1. 2026년 3월 26일~4월 7일 일반 Steam 경기의 Erangel/Baltic에 대응하는 build·manifest 또는 map revision.
2. 지형 높이 타일과 component/world transform, holes·water surface 정보.
3. 건물·교량 static mesh, 각 배치 인스턴스의 transform과 레벨 소속. 가능하면 collision geometry 및 collision 설정.
4. 해당 기간 핫픽스 간 지도·배치·충돌 파일 변경 여부와 해시 목록.
5. 학술 분석, 파생 그래프 공개, 논문 그림, 재현 코드·파생 수치 배포에 허용되는 범위.

문의 초안: “Erangel의 안전구역 조건부 이동 생성과 인간·봇 이동 비교 연구를 위해, 2026년 3월 26일~4월 7일 Steam 일반 경기에서 사용된 40.2 계열의 정적 공간 자료를 찾고 있습니다. 텍스처나 상업적 게임 재현이 목적은 아니며, 높이·건물/교량 인스턴스·충돌 또는 통행 구조와 빌드 출처가 필요합니다. 해당 시점 추출물의 보관 여부, 제공 가능한 항목, 학술 이용과 파생 그래프 배포 조건을 안내받을 수 있을까요?”

기술적 추출 가능성과 이용·재배포 허용은 별도다. [PUBG Steam EULA](https://store.steampowered.com/eula/578080_eula_1)는 역공학·파생물·보호 조치 우회·재배포를 제한하는 조항을 둔다. 학술 목적이라는 이유만으로 포괄 허용됐다고 판단하지 않으며, 연구용 데이터 제공이나 범위를 특정한 허가가 가장 명확한 경로다. 이 문서는 적용 법률의 예외 여부를 판단하지 않는다.

**실행 순서와 통과 기준**

아래는 제안한 후속 검증 순서이며 아직 실행 결과가 아니다.

1. **원본 접근 확인:** 40.2의 manifest ID 확보 → 목록 조회 성공 → 지도 관련 파일 다운로드 및 해시 확인. 이 단계가 실패하면 추출기 구현보다 제공자 문의를 우선한다.
2. **작은 공간 추출:** 지형 타일 하나, School 건물 구역, 서쪽 교량에서 메시·배치·단위를 확인한다. 대규모 처리 전에 빌드 불일치와 파서 누락을 찾는다.
3. **지형 검증:** 타일 경계 연속성, RG 16비트 복원, 원점·방향·스케일, holes, 랜드마크를 검사한다. 텔레메트리의 player z는 보조 비교로만 사용한다.
4. **건물·교량 검증:** 인스턴스 수·종속 파일 해소율·부모 transform·LOD·교량 상하 표면 분리를 기록한다. 시각 확인만으로 충돌 검증을 통과시키지 않는다.
5. **연구 그래프 생성:** 지형·표면·관측 이동 엣지를 분리한다. 아직 추출하지 못한 충돌이나 동적 상태는 unknown으로 유지한다.

당장 바뀌지 않는 판단은 **40.2 원본 확보가 첫 번째 관문**이라는 점이다. 오래된 높이 지도에 최신 건물 메시를 붙이는 작업으로 이 관문을 대체할 수 없다. 원본 확보가 끝내 불가능하면 기존 40.2 연구에서는 관측 이동 그래프의 주장 범위를 유지하고, 정밀 공간 조건 연구는 공간 자료와 텔레메트리를 함께 보존할 수 있는 새 패치에서 별도로 설계해야 한다.

근거 파일: [저장소 고정 정보](/Users/lunecid/project/PUBG/reports/erangel_40_2_acquisition_2026-10-15/repositories.json), [읽은 소스 URL 목록](/Users/lunecid/project/PUBG/reports/erangel_40_2_acquisition_2026-10-15/source_files.json), [확보 상태와 다음 단계](/Users/lunecid/project/PUBG/reports/erangel_40_2_acquisition_2026-10-15/acquisition_status.json).
