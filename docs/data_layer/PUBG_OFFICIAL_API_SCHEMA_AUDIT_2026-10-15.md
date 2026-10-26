# PUBG 공식 API 스키마와 이동 연구 데이터의 대응

확인일: 2026-10-15. 스키마의 기준은 **PUBG 공식 홈페이지의 OpenAPI 정의와 Telemetry Events/Objects**다. 로컬 PostgreSQL은 그중 일부를 변환·선택 저장한 구현이다. 공식 정의, 로컬 보존 상태, 연구에서 파생한 변수의 의미를 구분한다.

## 1. 공식 출처와 확인 범위

| 출처 | 확인 대상 |
|---|---|
| [Matches endpoint](https://documentation.pubg.com/en/matches-endpoint.html) 및 [matches.yml](https://documentation.pubg.com/en/_static/swagger/en/matches.yml) | match, roster, participant, asset의 관계 |
| [match.yml](https://documentation.pubg.com/en/_static/swagger/en/schemas/match.yml) | 경기 메타데이터·열거값 |
| [participant.yml](https://documentation.pubg.com/en/_static/swagger/en/schemas/participant.yml) | 계정 연결·경기 통계·단위 |
| [roster.yml](https://documentation.pubg.com/en/_static/swagger/en/schemas/roster.yml), [asset.yml](https://documentation.pubg.com/en/_static/swagger/en/schemas/asset.yml) | 팀 참조·텔레메트리 URL |
| [Telemetry](https://documentation.pubg.com/en/telemetry.html) | 다운로드·event array 구조 |
| [Telemetry Events](https://documentation.pubg.com/en/telemetry-events.html) | 사건별 필드와 중첩 객체 |
| [Telemetry Objects](https://documentation.pubg.com/en/telemetry-objects.html) | Character, Common, Location, GameState, Vehicle 등 |
| [Known Issues](https://documentation.pubg.com/en/known-issues.html), [Changelog](https://documentation.pubg.com/en/changelog/changelog.html) | 기존 오탈자·거리 통계 문제·이벤트 변경 |
| [공식 api-assets](https://github.com/pubg/api-assets) | 별도 사전·enum의 역할과 대소문자 유지 규칙 |

공식 서버의 YAML/HTML 원본 10개와 SHA-256은 [공식 원문 manifest](../../reports/movement_schema_2026-10-15/official_sources/manifest.json)에 보존했다. 모든 PUBG endpoint를 다룬 사전이 아니라 **이번 경기·이동 연구의 source schema 감사**다. 홈페이지의 changelog/enum 목록이 모든 2026 수집 값을 포함한다고 가정하지 않는다.

## 2. API 응답과 텔레메트리는 별도 구조다

```text
GET /shards/{platform}/matches/{match_id}
  data: match
    relationships.rosters.data[] → included의 roster 객체
    relationships.assets.data[]  → included의 asset 객체
  included[]:
    roster.relationships.participants.data[] → participant.id
    participant.attributes.stats.playerId    → 플레이어 계정 ID
    asset.attributes.URL                    → telemetry 파일

telemetry 파일: event 객체의 배열
  event._T              → event 종류
  event._D              → timestamp
  event.common.isGame   → 게임 상태 값
  event.character / gameState / victim / ... → event별 payload
```

`match_id`는 다운로드 맥락에서 각 event에 붙이는 관계키다. 모든 공식 event 공통 필드에 `match_id`가 있다고 가정하지 않는다. `participant.id`는 경기 응답 객체 참조 ID이고 `stats.playerId`가 계정 ID다. 따라서 `participant.id == character.accountId`로 조인하면 안 된다. [공식 participant 정의](https://documentation.pubg.com/en/_static/swagger/en/schemas/participant.yml), [공식 telemetry 구조](https://documentation.pubg.com/en/telemetry.html)

## 3. Match API에서 로컬 DB로의 대응

| 공식 경로 | 공식 타입·의미 | 로컬 컬럼 / 변환 |
|---|---|---|
| `match.id` | string | `matches.match_id` |
| `match.attributes.createdAt` | dateTime string, API에 match object가 저장된 시각 | `matches.created_at`; 실제 경기 시작시각으로 쓰지 않음 |
| `match.attributes.duration` | integer, 초 | `matches.duration` |
| `match.attributes.gameMode` | string | `matches.game_mode` |
| `match.attributes.mapName` | string | `matches.map_name` |
| `match.attributes.matchType` | string | `matches.match_type` |
| `match.attributes.isCustomMatch` | boolean | `matches.is_custom` |
| `roster.id` | string, 응답 내부 참조 | `rosters.roster_id` |
| `roster.attributes.stats.teamId` | integer | `rosters.team_id`; 경기 안에서 사용 |
| `roster.attributes.won` | **string** | `rosters.won` boolean으로 파싱 |
| `participant.id` | string, 응답 내부 참조 | `participants.participant_id` |
| `participant.attributes.stats.playerId` | string, 계정 ID | `participants.player_id` |
| `participant.attributes.stats.timeSurvived` | number, 초 | `participants.time_survived`; 사후 통계 |
| `participant.attributes.stats.walkDistance/rideDistance/swimDistance` | number, **미터** | 동명 snake_case 컬럼; 좌표처럼 다시 /100 하지 않음 |
| `asset.attributes.URL` | string | `matches.telemetry_url` |

위 내용은 각 공식 YAML의 구조를 로컬 스키마와 대조한 것이다. 로스터-참가자 연결은 `roster.relationships.participants`를 통해 구성한다. 경기 최종 통계는 시간창의 과거 입력으로 사용할 수 없다.

다운로드한 `match.yml`의 enum에는 로컬에서 관측한 `competitive`와 일부 새 map 값이 포함되지 않는다. 따라서 이 정적 목록만으로 실제 레코드를 삭제하지 않고 `unknown_enum`으로 보존한 뒤 공식 사전·원시 응답·수집 버전을 대조한다. `createdAt` 기반 분할은 **API 저장일 기준**이라고 명명하고 실제 경기 시간순 분할에는 match-start timestamp를 우선한다.

## 4. 이동·구역 필드의 공식 경로

| 공식 event / 경로 | 타입 | 현재 로컬 대응 |
|---|---|---|
| 공통 `_D`, `_T`, `common.isGame` | string, string, number | event_time, 테이블명에 반영된 종류, is_game |
| `LogPlayerPosition.character.accountId` | string | telem_positions.account_id |
| `.character.teamId`, `.character.health` | int, number | team_id, health |
| `.character.location.{x,y,z}` | number | pos_x/y/z |
| `.elapsedTime` | **number** | elapsed_time **integer**: 소수 보존 여부 미확인 |
| `.numAlivePlayers` | int | num_alive |
| `.vehicle.vehicleType`, `.vehicle.vehicleId` | string | vehicle_type, vehicle_id |
| `.vehicle.velocity` | number | vehicle_speed; 단위는 이 페이지에 명시되지 않음 |
| `LogGameStatePeriodic.gameState.elapsedTime` | int | telem_game_states.elapsed_time |
| `.gameState.safetyZonePosition`, `.safetyZoneRadius` | Location, number | safe_zone_x/y/z, safe_zone_radius |
| `.gameState.poisonGasWarningPosition`, `.poisonGasWarningRadius` | Location, number | poison_zone_x/y/z, poison_zone_radius |
| `LogPhaseChange.phase`, `.elapsedTime` | int, number | phase numeric, elapsed_time integer |
| `LogMatchStart.cameraViewBehaviour` | string | camera_view_type: 로컬 사본의 추출 키 불일치 아래 참조 |
| `LogMatchStart.blueZoneCustomOptions` | 문자열로 직렬화된 배열 | blue_zone_options jsonb; JSON string이 다시 저장될 수 있음 |

근거: [공식 event](https://documentation.pubg.com/en/telemetry-events.html), [공식 object](https://documentation.pubg.com/en/telemetry-objects.html). 연구용 `blue_current`, `white_announced`는 공식 필드명이 아닌 별칭이다.

## 5. 공식 사실과 구현 해석의 경계

| 항목 | 공식 문서에서 확인한 범위 | 연구 규칙 |
|---|---|---|
| 위치 단위 | Location은 cm, 원점은 지도 좌상단 | x/y/z를 m로 변환 |
| 반경 단위 | 반경은 number로 표기; 해당 객체 페이지는 단위를 별도로 설명하지 않음 | 기존 ETL의 /100 변환은 로컬 관례. 원시 위치와 반경의 기하 일치 점검 후 고정 |
| 구역 의미 | 두 구역의 필드 경로는 명시됨 | ‘현재 blue/공개 white’는 기존 코드 해석. 원시 trace와 구역 전환으로 검증 |
| isGame | 0.1은 비행기, 1.0/1.5는 첫 원 등장/축소 상태 등 | phase index 파생은 1 이상에서만; 정수 elapsedTime 또는 생존 bool이 아님 |
| phase-change 사건의 전환 전/후 상태 | Common 정의만으로 해당 이벤트의 전후 의미가 보장되지는 않음 | 기존 코드의 ‘종료된 상태’ 규칙은 trace 근거로 별도 검증 |
| 봇 구분 | 확인한 Character 정의에 isBot 없음 | ai./account. 접두어는 프로젝트 라벨 규칙; 공식 보증으로 쓰지 않음 |
| 위치 주기 | 확인한 event 정의에 고정 10초 보장 없음 | 10초는 로컬 표본 실측값 |
| 누락·null | 기본 event/object 개요는 완전한 required/nullable 계약이 아님 | required 여부를 임의 생성하지 않고 결측을 기록 |

공식 문서의 좌표 범위에는 Erangel 등 816,000cm, Sanhok 408,000cm가 나온다. 연구의 map_extent를 단순히 8,000/4,000m로 놓지 않는다. 지도 버전과 image-to-world 변환도 확인한다. [Location/Common 정의](https://documentation.pubg.com/en/telemetry-objects.html)

## 6. 공식 API에는 있으나 현재 정규화 DB에서 빠진 이동 사건

| 공식 사건 | 연구 활용 | 현재 DB 상태 |
|---|---|---|
| LogPlayerRedeploy, LogPlayerRedeployBRStart | 재배치 episode 구분 | 전용 테이블 없음 |
| LogPlayerRevive | 기절 이후 소생 구분 | 전용 테이블 없음 |
| LogSwimStart, LogSwimEnd | 도보·수영 구간 구분 | 전용 테이블 없음 |
| LogVehicleRide, LogVehicleLeave | 관측 사이의 탑승·하차 구분 | 전용 테이블 없음 |
| LogEmPickupLiftOff | 비상 수송 시작 | 전용 테이블 없음 |
| LogVaultStart | 지형 이동 단서 | 전용 테이블 없음 |

모두 [공식 이벤트 목록](https://documentation.pubg.com/en/telemetry-events.html)에 정의되어 있다. 표준 API에 존재해도 과거 모든 경기에서 발생·기록되었다는 뜻은 아니다. raw telemetry JSON의 보관 및 해당 event 존재를 확인하면 재추출할 수 있다. 따라서 **원본 복구 가능성을 먼저 확인한 뒤**, 불가능한 구간에만 휴리스틱 분절을 적용한다.

현재 positions의 Character 구역 flags와 Vehicle의 탑승석·고도·수중 여부 등도 일부만 보존되어 있다. 이 값들의 추가 보존이 이동 모드 판별에 도움이 되는지는 원자료와 함께 확인한다.

## 7. 수집 코드 사본과 공식 스키마의 불일치

확인 파일: `.claude/worktrees/brave-carson-776d45/tools/pubg_collector/03_fetch_telemetry.py`. **이 파일이 현재 DB를 만든 실행본이라는 provenance는 확인하지 않았다.** 아래는 source-level 불일치이며 기존 DB 데이터 손상을 확정한 결과가 아니다. 해당 사본·기존 수집 코드는 수정하지 않았다.

| 대상 | 공식 정의 | 코드 사본 | 확인할 영향 |
|---|---|---|---|
| Match start camera | cameraViewBehaviour | cameraViewType을 읽음 | camera_view_type 누락 가능성 |
| KillV2 피해 정보 | killerDamageInfo, finishDamageInfo, dBNODamageInfo 역할별 객체 | 단일 damageInfo를 읽음 | damage_reason/type/causer 등의 결측·역할 손실 가능성 |
| Groggy 피해 정보 | event 최상위 damageReason, damageTypeCategory 등 | damageInfo 내부를 읽음 | 동일한 결측 가능성 |
| TakeDamage distance | 확인한 event 스키마에는 최상위 distance가 열거되지 않음 | event.distance를 읽음 | 로컬 distance에 값이 항상 있다고 가정 금지 |

KillV2에서는 killer·finisher·DBNO maker가 다를 수 있으므로 여러 DamageInfo를 무조건 하나로 합치지 않는다. 이 구분은 사망 시각 기반 episode 분절과 피해원인별 분석에도 필요하다. [공식 KillV2/Groggy 정의](https://documentation.pubg.com/en/telemetry-events.html)

## 8. 기존 설계에 적용한 수정과 검증 순서

1. source-of-truth를 공식 API 정의로 명시하고 DB 사전은 구현 대응표로 둔다.
2. `created_at`을 실제 시작시각과 구분한다. 시간 원점은 LogMatchStart의 _D이다.
3. 원본 number elapsedTime은 신규 canonical schema에서 float64로 보존하고 local integer 값을 별도로 표시한다.
4. 미보존 이동 사건의 raw JSON 복구를 휴리스틱보다 먼저 조사한다.
5. blue/white 별칭·반경 단위·phase-change 시점 의미는 검증 상태를 붙인다. 모델 차원은 유지하되 구역 입력의 의미 검증 없이 학습을 시작하지 않는다.
6. 공식 Known Issues의 거리 통계 불일치를 반영해 participant 누적거리만으로 궤적 길이 정답을 검증하지 않는다. 공식 안내는 값이 다를 때 GameResult를 우선하도록 설명한다. 이것도 per-window 실제 경로의 정답을 제공하는 것은 아니다. [Known Issues](https://documentation.pubg.com/en/known-issues.html)
7. 공식 문서의 오탈자·deprecated 필드를 임의로 교정하지 않는다. raw 키를 유지하고 canonical 이름은 매핑층에서 바꾼다. 알 수 없는 enum/필드는 원문을 보존한다. [공식 사전 안내](https://github.com/pubg/api-assets)

이 문서로 공식 정의와 로컬 구현 대응을 정리했다. raw payload와 현재 수집 실행본의 대조, DB 누락률·오파싱 여부 검증은 별도 작업으로 남는다.
