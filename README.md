# PUBG Lab

PUBG 텔레메트리의 공간·개인·팀·경기 구조를 연구하는 모노레포다.
현재 목표는 **수업 논문 → FDG 2027**이며 ToG는 후속 확장이다.
저장소 릴리스는 **2026.10.26-r4**다. 모델 성능의 새 판정을 뜻하지 않는다.

## 먼저 읽기

- [작업 지침](AGENTS.md)과 [현재 연구 승인 범위](docs/guidelines/PUBG_APPLICATION.md)
- [코드 구조·양식 기준](docs/repository/CODEBASE_STANDARD.md)과 [재편 결과](docs/repository/refactor_2026-10-26/REPORT_KO.md)
- [버전 목록](versions/README.md), [문서 색인](docs/README.md), [데이터 품질 색인](docs/data_layer/CORPUS_QUALITY_INDEX.md)

## 코드 구조

```text
src/pubg_lab/
  common/                 # 저장소 경로·호환 실행 공통 부품
  data/                   # STDB·품질 규칙·전처리
  world_model/            # 입력·인코더·사전학습 부품과 cli/
  movement/               # 개인·팀 이동 생성 부품과 cli/
  survival/               # BR·PTR·생존 모델 및 특징·평가
  spatial/                # erangel/·caldera/ 공간 처리
  analysis/network/       # 네트워크 분석
  collectors/pubg/        # 명시적으로 실행하는 API 수집기
  projects/               # sac2027/·battle_royale/·figures/ 응용
  repository/             # 릴리스·구조 검증
tests/                    # 연구 불변식·구조·호환성 검증
configs/                  # 코드 소유·의존 경계 설정
docs/                     # 지침·계약·설계·변경 기록
paper/                    # 원고별 소스와 동결 과거 버전
webapp/                   # 별도 프런트엔드 애플리케이션
presentation_build/       # 별도 발표 제작 애플리케이션
```

실제 Python 구현은 `src/pubg_lab/` 한곳에 있다. 각 폴더 README가 책임·부품·의존 방향을 설명한다.
`world_model/`, `movement_diffusion/`, `model_stgcn/`, `stdb/`, `network_analysis/`,
`tools/`와 `scripts/*.py`는 과거 import·명령을 위한 호환 경로다. 새 구현을 이 경로에 추가하지 않는다.
`scripts/*.sh`는 과거 배치 실행 명령을 보존한 경로이며, `legacy/`와 `third_party/`는 기존 범위대로 보존한다.
보호 분할 구현은 공개 패키지 밖의 기존 위치에 남겨 둔다.

## 실행 도구 찾기

소스 checkout에서는 설치 없이 다음 명령으로 실행 도구를 확인한다. 목록 조회는 앱을 import하거나 실행하지 않는다.

```sh
PYTHONPATH=src python3 -m pubg_lab list
PYTHONPATH=src python3 -m pubg_lab run repository.check-structure
PYTHONPATH=src python3 -m pubg_lab run repository.repository-release verify --release 2026.10.26-r4
```

분리된 개발 환경에 패키지를 설치했다면 같은 기능을 `pubg-lab list`, `pubg-lab run <명령>`으로 사용할 수 있다.
기존 `python tools/<파일>.py` 명령과 기존 Python import도 호환 진입점이 연결한다.
연구 실행은 해당 계약과 승인 범위를 따르며, 이 재편에서 새 학습·DB·API 실행은 하지 않았다.

## 개발·검증

```sh
make structure format-check lint test-architecture
make test-synthetic
make wheel
```

학습 환경에 개발 도구를 설치하지 않는다. 별도 환경의 버전은 [requirements-quality.txt](requirements-quality.txt)에 고정했다.
[기여·개발 안내](CONTRIBUTING.md)에 준비 방법, 의존 경계, 검증 범위와 호환 경로 정책을 적었다.

## 자료와 재현성

완료된 실행·실패 기록·원본·원고·참조 코드는 원래 위치에 보존한다.
데이터·가중치·대용량 예측 행은 저장소의 공개 패키지에 넣지 않는다.
[기존 로컬 산출물 목록](docs/repository/cleanup_2026-10-26/local_artifacts.json)은 원 경로·크기·해시를 연결한다.
과거 실행의 정확한 코드 재현은 해당 Git 태그와 실행의 source snapshot을 사용한다.
현재 호환 코드는 새 정리 버전의 구현을 연결하므로 과거 코드 해시와 동일하다고 주장하지 않는다.

배포 wheel은 코드만 담는다. 설치된 코드에서 로컬 자료가 필요한 경우
`PUBG_REPOSITORY_ROOT`로 실제 작업 폴더를 지정한다. 비밀 모듈·완료된 실행은 클론이나 wheel만으로 복원되지 않는다.

[공개 논문·발표 자료](docs/PAPERS_AND_TALKS.md)와 과거 원고의 수치는 각 시점의 계약·자료를 따른다.
최상위 프로젝트 라이선스는 아직 지정되지 않았다. 참조 자료는 각 원출처 라이선스를 따른다.
