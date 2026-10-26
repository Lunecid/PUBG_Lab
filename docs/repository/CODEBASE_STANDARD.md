# 코드베이스 구조와 양식

## 실제 디렉터리와 의존 방향

```text
apps/                    사용자용 웹앱·발표 생성
projects/                연구 과제별 실행 프로젝트
  world_model/           data · training · evaluation · reporting · common
  movement/              data · training · evaluation · reporting · release
  battle_royale/          pipelines · evaluation · analysis · reporting · launchers
  sac2027/               training · evaluation · reporting · release
  krafton_baseline/      data · models · training · evaluation · reporting · pipelines
  spatial/               erangel · caldera
src/pubg_lab/            재사용 라이브러리
  survival/              models · data · features · training · evaluation
                         baselines · experiments · inference · ptr · v3
  world_model/           data · features · models · training
  movement/              data · models · evaluation
  data/stdb/             connection · queries · temporal · pipelines · schemas · audits
  analysis/network/      data · features · reporting · config · legacy
  common/                저장소 경로
  repository/            정적 구조·릴리스 검사
tests/                   합성 회귀 시험과 별도 실제 자료 시험
configs/                 환경 핀과 코드 소유 경계
archive/                 과거 소스와 설계 기록
versions/                프로젝트 버전 색인과 불변 릴리스 매니페스트
```

`projects → pubg_lab`, `domain → common` 방향을 따른다. 공통 라이브러리는 실행 프로젝트를 import하지 않는다.
여러 프로젝트가 같이 쓰는 구현은 해당 도메인 라이브러리에 둔다. 프로젝트 내부 공통 부품은 그 프로젝트의 common에 둔다.
같은 분야의 모델·데이터·평가·실험을 한 폴더에 평평하게 나열하지 않는다. 역할별 폴더를 만든다.
모듈 경로와 파일 경로·DB 스키마·연구 preset 이름은 구별한다. 구조 변경으로 연구 값을 바꾸지 않는다.
루트 호환 .py 파일은 만들지 않는다. 호출 코드·테스트·명령 목록을 실제 경로로 바꾼다.

## 실행·양식·자동 검사

저장소 루트에서 `PYTHONPATH=src:. python3 -m pubg_lab list`로 정식 모듈 목록을 확인한다.
명시적 실행은 `PYTHONPATH=src:. python3 -m pubg_lab run <command> ...` 또는 목록의 모듈을 `python -m`으로 호출한다.
외부 위치에 설치한 wheel은 자료 경로가 필요할 때 `PUBG_REPOSITORY_ROOT`를 설정한다.
목록 탐색은 모델·자료를 import하거나 서비스·DB를 시작하지 않는다.

Python 3.11, UTF-8, 공백 4칸, snake_case 파일·함수, PascalCase 클래스를 사용한다.
Ruff 0.14.1 format, 행 길이 100, double quotes, LF를 적용한다. 명명과 연구 상수는 이동 과정에서 임의로 바꾸지 않는다.
전역 thread/determinism 설정의 순서를 보존하기 위해 import 자동 정렬을 사용하지 않는다.
`make structure format-check lint test-architecture`가 소유 경계·폴더 배치·양식·명령 경로를 검사한다.
`configs/environments/quality.txt`의 검사 도구는 학습 환경과 분리해 설치한다.
GitHub Actions도 같은 구조·양식·아키텍처 검사를 실행한다. 합성 수치 시험의 전체 결과는 릴리스 보고서에 기록한다.

## 버전과 보존 경계

패키지 코드는 릴리스 태그로, 연구 계열·모델 버전·전처리·실행·원고는 `versions/registry.json`의 별도 축으로 관리한다.
G1–G4, movement v1–v4, PTR·v3 등 기존 결과의 정의와 명칭을 동결한다. 새 저장소 릴리스는 새 연구 결과를 뜻하지 않는다.
완료된 실행·원시 자료·원고·third_party는 정리 과정에서 내용을 바꾸거나 삭제하지 않는다.
`archive/legacy_snapshot`은 원 legacy 코드의 바이트를 보존한 이동이다.
`model_stgcn`은 기존 실험 자료와 제한 분할, `tools`는 제한 분할 경로만 남기는 명시적 예외다.
이 두 제한 구현은 읽기·해시·배포 대상에서 제외한다. 예외를 새 호환 코드의 근거로 확대하지 않는다.

보호 분할은 `survival/data/split_access.py`에서 허용된 공개 함수만 연결한다.
보존 모듈이 참조하는 공개 provenance의 옛 이름을 그 경계에서만 공급하며 루트 파일은 만들지 않는다.
이 경계는 분할 생성·본문 조회·자료 접근을 실행하지 않는다. 과거 문서의 옛 경로는 이동표와 Git 태그로 해석한다.

설치된 wheel에서 제한 분할을 쓰는 명령은 원 checkout의 제한 구현과 접근 권한도 필요하다.
이 경우 원 checkout을 PYTHONPATH에 명시한다. wheel만으로 보호 구현을 복제하거나 복구하지 않는다.
