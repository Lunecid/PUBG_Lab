# PUBG Lab

PUBG 텔레메트리의 공간·개인·팀·경기 구조를 연구하는 모노레포다.
현재 목표는 **수업 논문 → FDG 2027**, ToG는 후속 확장이다.
저장소 릴리스 **2026.10.26-r5**는 코드 구조 버전이며 새 모델 결과를 뜻하지 않는다.

## 시작 위치

```text
src/pubg_lab/   재사용 모델·자료·특징·평가 부품
projects/       실제 연구 실행 프로젝트
apps/           웹앱·발표 생성
tests/         합성 회귀 시험·별도 실제 자료 시험
configs/        환경 핀·구조 경계
archive/        과거 소스·설계 기록
versions/       모델 계열 색인·릴리스 매니페스트
```

[프로젝트 목록](projects/README.md)에서 과제를, [라이브러리 목록](src/pubg_lab/README.md)에서 재사용 부품을 찾는다.
ST-GNN 구현은 `src/pubg_lab/survival/{models,data,training,evaluation,features,experiments,baselines,inference}`로 실제 분리했다.
PTR·v3도 내부에서 모델·데이터·학습·평가·실험을 나눈다. 같은 폴더에 여러 역할의 파일을 평평하게 쌓지 않는다.
기존 루트 전달용 Python 파일은 제거하고 호출 코드·시험·명령을 실제 경로에 연결했다.

`data/`, `reports/`, `paper/`, `output/`, `third_party/`는 기존 자료·결과·참조 코드 보존 위치다.
`model_stgcn/`은 완료된 실험 자료와 제한 분할, `tools/`는 제한 분할 경로를 보존하는 예외다.
과거 코드 자체는 Git 태그와 실행별 source snapshot으로 식별한다.

## 지침과 실행

작업 전 [AGENTS](AGENTS.md), [작업 규칙](docs/guidelines/WORKING_RULES.md), [현재 승인 범위](docs/guidelines/PUBG_APPLICATION.md)를 읽는다.
연구 실행은 해당 계약·자료 접근 범위를 따른다. 명령 목록 조회는 앱·자료를 import하거나 서비스를 시작하지 않는다.

```sh
PYTHONPATH=src:. python3 -m pubg_lab list
PYTHONPATH=src:. python3 -m pubg_lab run repository.check-structure
PYTHONPATH=src:. python3 -m pubg_lab run repository.repository-release verify --release 2026.10.26-r5
```

설치된 패키지는 `pubg-lab list`, `pubg-lab run <명령>`으로 호출한다.
로컬 자료가 필요한 설치 환경은 `PUBG_REPOSITORY_ROOT`로 실제 저장소를 지정한다.

## 개발과 버전

```sh
make structure format-check lint test-architecture
make test-synthetic
make wheel
```

도구는 학습 환경과 분리한다. 의존성은 [configs/environments](configs/environments/)에 연구·개발·품질·공간 도구별로 고정한다.
[개발 안내](CONTRIBUTING.md), [구조·양식 기준](docs/repository/CODEBASE_STANDARD.md), [실제 이동·검증 보고서](docs/repository/layout_2026-10-26/REPORT_KO.md)를 따른다.
모델 버전·전처리·실행·원고는 [버전 색인](versions/README.md)과 [registry](versions/registry.json)의 별도 축으로 관리한다.
원시 자료·보호 분할·대용량 예측은 새 공개 코드 패키지에 포함하지 않는다.

[데이터 품질 색인](docs/data_layer/CORPUS_QUALITY_INDEX.md), [전체 문서](docs/README.md), [공개 논문·발표](docs/PAPERS_AND_TALKS.md).
최상위 라이선스는 아직 지정되지 않았다. 참조 코드는 원출처 라이선스를 따른다.
