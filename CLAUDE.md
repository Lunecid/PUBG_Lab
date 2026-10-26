# PUBG 프로젝트 공통 지침

작업을 시작하기 전에 [AGENTS.md](AGENTS.md), [작업 지침](docs/guidelines/WORKING_RULES.md), [PUBG 적용 범위](docs/guidelines/PUBG_APPLICATION.md)를 읽고 따른다.

에이전트별로 규칙 사본을 따로 관리하지 않는다. 새 규칙과 저자 확인은 공통 문서에 반영한다. 현재 투고 경로는 수업 논문 → FDG 2027이며 ToG는 후속 확장이다.


## 저장소 정리와 버전

[버전 색인](versions/README.md)과 [정리 정책](docs/repository/cleanup_2026-10-26/PLAN_KO.md)을 따른다.
기존 실행·원고·재현 코드는 보존하며, 새 릴리스는 새 경로·매니페스트로 기록한다.
2026-10-26 사용자가 요청한 이번 정리 결과의 commit·push는 승인되어 있다.


## 현재 코드 위치와 개발 기준

실제 Python 구현은 `src/pubg_lab/`에 둔다. root 기존 Python 경로는 호환용이다.
[코드 구조·양식 기준](docs/repository/CODEBASE_STANDARD.md)과 `make structure format-check lint test-architecture`를 따른다.
연구 상수·분할·평가·동결 증거를 구조 변경과 함께 바꾸지 않는다.
