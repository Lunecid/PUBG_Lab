# r4 코드베이스 구조·양식 재편 결과

날짜: 2026-10-26. 기준 커밋: f09bed03f669bfb6d51fac9dbc3e17d02a649993.
저장소 릴리스: **2026.10.26-r4**.

## 결과

Python 실제 구현 **374개**를 `src/pubg_lab/`에 책임별로 재배치했다.
현재 canonical Python 파일은 패키지 경계·새 공통 인프라·CLI·검사기를 포함해 **396개**다.
데이터·World model·이동·생존·공간·분석·수집기·프로젝트 응용·저장소 도구가 각각 담당 폴더를 소유한다.
각 패키지 README와 configs/codebase.json이 책임·구성 부품·의존 방향을 연결한다.

기존 Python 경로에는 구현을 복제하지 않는 호환 진입점 374개를 남겼다.
정상 import는 canonical 모듈을 공유하고, importlib spec로 보유하는 옛 모듈 객체에도 속성을 연결한다.
실행 가능했던 스크립트 47개의 shebang도 유지했다.
새 구현은 root 호환 경로에 추가하지 않는다. 과거 shell 배치·legacy·원고·참조 코드는 그대로 보존했다.

공통화한 부품은 저장소 경로와 호환 실행이다. 과제별 손실·메트릭·표본 선택을 이름만 같다는 이유로 합치지 않았다.
프로젝트별 실행 도구는 담당 패키지에 모았고, 명령 **284개**를 목록으로 찾을 수 있다. 목록 조회는 앱을 import하지 않는다.

## 표준 개발 형식

- pyproject.toml: src 패키징·공통 CLI·시험 경로·Ruff 설정.
- Ruff 0.14.1: 공백 4칸, 줄 길이 100, double quote, LF. 복수 import 58곳은 순서를 유지해 분리했다.
- 구조 검사: 코드 소유, 폴더 README, canonical import, 라이브러리→응용 역의존과 도메인 순환을 검사한다.
- Makefile·CONTRIBUTING·고정 quality 의존성·PR/수동 GitHub workflow를 추가했다.
- wheel은 코드·필수 소스 자원만 포함한다. 비밀 설정·보호 분할·원시 데이터·실행 결과는 패키지 밖에 둔다.

학습 환경은 변경하지 않았다. 개발·빌드 도구는 `/private/tmp/pubg-code-quality-20261026`에만 준비했고 artifact hash를 기록했다.
특정 기업의 비공개 형식과 같다고 주장하지 않는다. PyPA src layout과 Google Research Scenic의 라이브러리/프로젝트 분리 원칙을 참고한 연구 모노레포다.

## 연구 보존

- 이동 전 공개 소스 374개의 해시를 기준 Git 커밋과 대조했다.
- import·저장소 경로·새 실행의 소스 위치만 변환했다. 연산·손실·마스크·표본·시드·지표·상수는 변경하지 않았다.
- 포맷과 순서 보존 import 분리를 정규화한 AST로 대조했다. 설명되지 않은 변경은 없다.
- G2의 동적 source snapshot 목록은 새 canonical 코드와 공통 경로·빌드 메타데이터를 가리키도록 별도 교정했다. 이 metadata 블록을 명시적 예외로 기록했다.
- data·reports·paper·third_party·legacy와 r1–r3 릴리스 파일은 이번 재편에서 변경하지 않았다. 보호 분할 본문은 읽거나 이동하지 않았다.
- 실제 DB·API·서비스·훈련·보호/최종 테스트를 실행하지 않았다.

과거 실행의 정확한 코드 재현은 그 실행의 source snapshot과 원 Git 태그를 사용한다.
현재 호환 모듈은 r4 구현을 연결하므로 과거 소스 해시와 동일한 코드라고 부르지 않는다.

## 검증

**246개 통과, 12개 제외(pyarrow 미설치), 기존 경고 4개**다.
기존 연구 합성 시험에 구조·호환 검증 7개와 추가 생존·그래프 불변식을 포함했다.
포맷·핵심 lint는 canonical 구현·호환 코드·구조 시험 **771개**에서 통과했다.
396개 canonical 파일의 정적 소유·의존 방향·순환 검사도 통과했다.

wheel 빌드와 격리 import를 통과했다. 저장소를 sys.path에서 제거한 별도 폴더에서
WorldEncoder, FormationRegressor, hazard_bce를 wheel의 모듈로 불러왔다.
로컬 자료가 필요한 배포 코드는 PUBG_REPOSITORY_ROOT를 명시한다.

초기 spec-loader 호환 실패 9건, 구조 시험의 줄바꿈 의존 1건과 양식 검사 중간 발견은
[실패·수정 기록](failed_attempts.json)에 보존했다. 수정 후 해당 시험과 전체 검증을 통과했다.
자체 점검이며 독립 모델 검토로 부르지 않는다. 원격 GitHub workflow는 아직 실행하지 않았다.

## 범위와 한계

모든 명령 284개의 실제 데이터 실행을 검증한 것은 아니다.
보호 분할을 쓰는 과거 생존 파이프라인은 승인된 원 작업 폴더의 private 모듈이 필요하다.
과거 checkpoint 전체를 로드해 완전 호환이나 bit 재현을 검증하지 않았다.
프런트엔드·발표 앱은 기존 애플리케이션 폴더를 유지했고 새로 빌드하지 않았다.
완전한 저장소 전체 타입 검사와 미사용 import 전면 제거를 수행한 것으로 주장하지 않는다.

## 증거

[사전 설계](PLAN_KO.md), [이동표](move_plan.json), [변환 기록](migration_receipt.json),
[원 소스 대조](baseline_identity.json), [AST 검증](format_ast_validation.json),
[동결 경로 보존](frozen_paths_check.json), [G2 source snapshot](snapshot_path_validation.json),
[시험 결과](final_tests.json), [wheel 격리](wheel_validation.json),
[도구 고정](toolchain_artifacts.json), [graphify AST](graphify_validation.json).

발행 전 검사에서 기존 ignore 규칙의 부수 효과로 비추적 참조 소스 5개가 스테이징에 드러난 후보를 제외했다.
참조 파일은 원위치에 보존했고, 발행되지 않은 후보 manifest는 내용 변경 없이 [실패 후보 기록](rejected_candidate.json)에 남겼다.
최종 규칙은 기존 data/ 경계를 유지하면서 canonical source data 폴더만 예외로 둔다.
