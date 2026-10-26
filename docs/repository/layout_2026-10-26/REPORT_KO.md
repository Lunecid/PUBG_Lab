# r5 실제 프로젝트·모듈 분리 결과

r4는 루트에 전달용 Python 파일을 남기고 survival 구현도 평평하게 두었다. 사용자가 지적한 문제를 해결하지 못한 상태였다.
이번에는 탐색기에 보이는 실제 소스 위치와 모든 활성 호출을 바꿨다. 에디터 숨김 설정으로 처리하지 않았다.

## 바뀐 위치

```text
src/pubg_lab/survival/
  models/       STGCN·DynamicSTGNN·snapshot 모델
  data/         dataset·dump·phase_v2 조립·제한 분할 공개 경계
  training/     train·train_mr·train_phase·train_snapshot_phase
  evaluation/   eval·calibration·bootstrap
  features/     edge·manifest 부품
  experiments/  ablation·중요도·분석
  baselines/    tabular·Sagong 기준선
  inference/    추론
  ptr/          data·models·training·evaluation·experiments·common
  v3/           data·models·features·training·evaluation·experiments
```

사진에 있던 39개 파일의 평평한 나열 대신, survival 직접 루트에는 __init__.py·config.py·provenance.py만 남는다.
world model·movement·STDB·network 라이브러리도 책임별 폴더로 나눴다.
실행 코드는 실제 루트 projects/ 아래 과제별 패키지로, 웹앱·발표는 apps/, 과거 소스·시안은 archive/로 옮겼다.
[프로젝트 안내](../../../projects/README.md), [현재 구조 기준](../CODEBASE_STANDARD.md), [전체 이동표](path_map.json)를 따른다.

## 부품과 실행 경계

World model의 해시·tensor 해시·private memory sampler를 common/resources.py로 추출했다. 원 정의 3개의 AST가 동일함을 확인했다.
모든 관련 검증·학습·보고 호출은 공통 부품을 사용한다. 라이브러리가 프로젝트 실행 코드를 import하는 역의존은 없다.
KRAFTON의 고정 third_party 참조 바이트는 유지했다. 현재 로컬 어댑터 22개는 projects/krafton_baseline의 역할별 폴더에서 정식 모듈을 호출한다.
루트 전달용 파일 374개와 기존 호환 실행 인프라를 제거했다. 호출 코드·테스트·302개 명령 색인·42개 shell launcher를 갱신했다.
웹앱의 필수 진입 HTML과 src/lib 소스 9개도 기존 광범위 ignore에서 복구해 함께 버전 관리한다.

보호 분할은 기존 두 경로에 그대로 보존한다. 본문을 직접 열거나 해시·수정하지 않았다.
변경할 수 없는 분할 모듈의 공개 provenance 의존 이름은 data/split_access.py 한곳에서 연결하고 승인된 공개 함수만 재노출한다.
이 경계는 루트 .py 호환 파일을 만들지 않고 분할 생성·자료 조회를 실행하지 않는다.
model_stgcn에는 기존 실험 자료와 제한 분할, tools에는 제한 분할 경로만 남는 보존 예외가 있다.

원고의 기존 표 생성 소스도 그대로 보존하고 현재 생성 코드는 projects/battle_royale/reporting으로 분리했다. 함수 정의 AST는 원본과 동일하다.
실행 계획의 코드 snapshot은 실제 src/projects와 고정 참조 소스를 포함하도록 고쳤으며 기존 합성 fixture도 같은 구조로 갱신했다.

## 검증

- 합성·아키텍처·수치 회귀: **377 통과, 12 건너뜀, 실제 상태 시험 1개 제외**, 실패 0. 12개는 pyarrow 미설치에 따른 Parquet 시험이며 실제 자료 시험으로 대체하지 않았다.
- 구조·로컬 import 대상·역의존·도메인 순환 검사, Ruff format과 선택 lint, 아키텍처 8개 시험 통과.
- 학습·평가 명령 6개의 --help를 저장소 밖에서 확인. 연구 실행은 하지 않았다.
- shell 42개 bash -n, 발표 JS 3개 node --check 통과. 웹앱 TypeScript 검사·Vite 빌드 통과.
- 웹앱 빌드는 publicDir=false로 수행해 로컬 연구 payload를 복사하지 않았고 build:data·서비스 시작은 하지 않았다.
- wheel에 Python 507개를 포함하고 원 소스 경로가 없는 격리 import 통과. 비밀·보호 분할·학습 자료는 wheel에 없다.
- 기존 소스/시험 458개의 import·경로 정규화 후 AST 비교: 438개 동일, 20개 차이를 개별 검토했다. 차이는 구조 검사·경로·코드 식별자·공통 부품 추출 변경으로 검토했다.
- KRAFTON 어댑터 22개 중 18개는 정규화 후 동일, 4개는 소스 fingerprint 경로·subprocess PYTHONPATH 변경이다. 모델·손실·평가 정의는 바꾸지 않았다.

[시험 기록](test_results.json), [원본 보존](preservation_check.json), [소스 비교](source_semantics.json), [wheel 확인](wheel_validation.json)에 근거를 남겼다.
기존 data·reports·paper·third_party·output와 r1–r4 manifest/감사 기록에는 추적 파일 내용 변경이 없다.
legacy 원 소스는 바이트를 유지해 archive로 이동했고 로컬 앱·viewer 자료는 이동 전후 inode·크기로 확인했다.
새 모델 학습·DB/API 호출·보호 경기/최종 자료 조회·재평가는 수행하지 않았다.

## 버전·게시

저장소 버전은 2026.10.26-r5, 패키지는 2026.10.26.post5다. 이전 태그·원 실행·보고서를 보존한다.
이 작업의 commit·main 병합·push·로컬 main 전환은 사용자의 기존 명시적 요청에 따른다.
[릴리스 manifest](../../../versions/releases/2026.10.26-r5/manifest.json)와 별도 감사 SHA256 목록으로 내용을 식별한다.
원격 CI는 PR/수동 workflow이며 로컬 검증을 원격 CI 실행 성공으로 표기하지 않았다.
