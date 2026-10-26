# PUBG Lab

PUBG 텔레메트리의 공간·개인·팀·경기 구조를 연구하는 저장소다.
현재 목표는 **수업 논문 → FDG 2027**이며 ToG는 후속 확장이다.

## 먼저 읽기

- [프로젝트 지침](AGENTS.md)과 [현재 적용 범위](docs/guidelines/PUBG_APPLICATION.md)
- [버전 목록과 재현 경로](versions/README.md)
- [문서 색인](docs/README.md)과 [데이터 품질 색인](docs/data_layer/CORPUS_QUALITY_INDEX.md)
- [main 병합·로컬 전환 기록](docs/repository/cleanup_2026-10-26/MAIN_MERGE_KO.md)
- [이번 폴더 정리 기록](docs/repository/cleanup_2026-10-26/REPORT_KO_v2.md)

## 폴더 안내

| 경로 | 내용 |
|---|---|
| `world_model/` | 패치 간 전처리, 개인·팀·공간 인코더, G1–G4 구현 |
| `movement_diffusion/`, `tools/movement_diffusion/` | 이동 생성 V1–V4의 보존 코드와 실행 도구 |
| `model_stgcn/`, `stdb/` | 이전 BR·PTR 실험과 STDB 처리 코드 |
| `legacy/` | 초기 snapshot 데모의 보존 코드 |
| `tools/pubg_collector/` | 기존 main에서 병합한 공식 API 수집기. 실행은 별도 승인 범위 |
| `tools/`, `scripts/`, `tests/` | 검증·분석·실행 도구와 시험 |
| `docs/` | 지침, 연구 계약, 설계, 문헌 근거, 변경 기록 |
| `paper/` | BigComp·SAC 등 원고별 소스와 과거 버전 |
| `versions/` | 연구 계열 색인과 해시로 고정한 저장소 릴리스 |
| `reports/`, `data/`, `model_stgcn/checkpoints/` | 기존 위치에 보존하는 로컬 증거·자료·모델. Git에 일부만 포함 |
| `network_analysis/`, `webapp/`, `presentation_build/` | 기존 분석·시각화·발표 도구 |
| `third_party/` | 출처·라이선스를 보존하는 참조 구현 |

코드의 V3, 전처리의 v3, 원고의 v20은 서로 다른 버전 축이다.
[registry.json](versions/registry.json)은 이 구분을 유지한다.
`VERSION`의 `2026.10.26-r3`은 정리한 저장소의 릴리스이며 새 모델 성능 결과를 뜻하지 않는다.

## 모듈화 상태

버전 색인과 기존 패키지는 갖췄지만, 동일 계열의 실행·보고·검증 도구를 부품별 폴더로 모으는 작업은 아직 끝나지 않았다.
`tools/`의 World model·SAC 도구와 설치 가능한 프로젝트 메타데이터는 후속 구조 변경 대상이다.
[실제 구조 점검과 미완료 항목](docs/repository/cleanup_2026-10-26/MODULARITY_STATUS_KO.md)을 참고한다.

기존 main의 [논문·발표 자료 목록](docs/PAPERS_AND_TALKS.md)과 [병합 전 안내](docs/archive/main_before_cleanup_merge_2026-10-26/README.md)를 보존했다.
수집기와 레거시 그래프 생성기는 서로 다른 실행 경로다. 레거시 진입점은 `legacy/main.py`다.
레거시 DB 연결은 `PG_*`와 `PUBG_DB_*` 환경변수를 모두 받으며, 같은 설정이 있으면 `PG_*`를 우선한다.
수집기 실행은 이 병합 과정에서 수행하지 않았다.

## 보존과 공개 범위

완료된 실행·실패 기록·원본·원고와 재현 코드는 덮어쓰지 않는다.
데이터, 가중치, 대용량 예측 행, 비밀 설정, 보호 분할 모듈은 새 공개 업로드 대상에서 제외한다.
기존 실행 경로는 유지하고 [로컬 산출물 목록](docs/repository/cleanup_2026-10-26/local_artifacts.json)으로 연결한다.
저장소를 클론하는 것만으로 로컬 데이터·비밀 모듈·완료된 실행까지 복원되지는 않는다.

## 릴리스 확인

```sh
python3 tools/repository_release.py verify --release 2026.10.26-r3
```

새 정리 릴리스는 `VERSION`과 변경 기록을 갱신하고, 시험 뒤 새로운 이름으로 생성한다.
이미 생성한 `versions/releases/<버전>/manifest.json`은 수정하지 않는다.

```sh
python3 tools/repository_release.py snapshot --release YYYY.MM.DD-rN
```

모델 학습·DB 조회·서비스 시작은 이 명령에 포함되지 않는다.
실험 실행은 각 계열의 연구 계약과 별도 승인 범위를 따른다.

최상위 프로젝트 라이선스는 아직 지정되지 않았다. `third_party/`의 자료는 각 원출처 라이선스를 따른다.
