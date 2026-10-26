# 정리 브랜치의 main 통합과 로컬 전환

날짜: 2026-10-26. 저장소 릴리스: **2026.10.26-r3**.
사용자 승인: 정리 브랜치를 main에 병합·push하고 현재 로컬 폴더에도 반영한다.

## 병합 범위

- 정리 전 main: `eae79539031657ab290e697d52a290250a24867e`.
- 정리 r2: `2c62414a741c6a79a7cd4c23982eff6ce9dabf5c`.
- 정리 브랜치에 원격 main을 통합해 충돌·시험을 해결한 뒤, 로컬 main은 검증된 결과로 fast-forward한다.
- 원격 main·정리 브랜치와 새 `repo-2026.10.26-r3` 태그를 같은 결과로 push한다. 기존 이력을 강제로 덮어쓰지 않는다.
- 현재 로컬 작업 폴더는 `/Users/lunecid/project/PUBG`이며 이 폴더의 활성 브랜치를 main으로 바꾼다. 다른 worktree는 변경하지 않는다.

## 충돌 해결

| 파일 | 해결 원칙 |
|---|---|
| README.md | 현재 연구·버전 안내 유지, main의 수집기·자료 목록 연결. 당시 안내 원문은 archive 보존 |
| docs/README.md | 문서 객체 모델 유지, 기존 논문·발표 목록을 별도 문서로 연결 |
| .gitignore | 대용량·개별 예측 제외와 소규모 증거 allowlist를 유지하고 main의 CSV·SQL·영상 제외 및 공개 자료 예외를 통합 |
| .env.example | 내용을 열거나 출력하지 않고 기존 PG_* 예제를 보존. 비밀 설정은 변경하지 않음 |
| legacy/main.py | 기존 그래프 구현 유지, main의 환경변수 사용 목적을 통합. PG_* 우선, PUBG_DB_* 별칭 지원, 실제 연결 요청 때만 dotenv 로드 |
| requirements.txt | 기존 수치·모델 패키지 고정 버전 유지. main이 추가한 dotenv 의존성을 기존 로컬 버전 1.2.2로 명시. 설치 미실행 |

[충돌 해결 기록](main_merge_resolution.json), [main 자료 보존](main_assets_preservation.json).
원고·수식·분할·목표·평가·학습 설정은 변경하지 않았다. 레거시에서는 접속 설정 함수 두 개 외의 그래프 변환 함수가 동일함을 AST로 확인했다.

## 검증

- 139개 통과, pyarrow 미설치로 12개 제외, 기존 라이브러리 경고 3개.
- 새 DB 연결 회귀 시험 6개는 가짜 환경변수·dotenv·connect를 사용했다. 실제 DB·비밀 파일 접근 없음.
- main의 수집기·공개 논문·발표 파일 9개는 Git blob이 동일하다.
- 소스 문법 확인과 보호·대용량 제외 점검을 수행한다.
- 기존 r1·r2 릴리스 매니페스트와 정리 감사 원문은 보존한다. 통합 파일 집합은 r3로 새로 고정한다.

[시험 영수증](main_merge_tests.json), [소스 문법 검사](main_merge_syntax.json).
main에서 병합한 수집기를 실행하거나 모델을 다시 학습한 결과가 아니다.

## 모듈화 판정

**모듈화·프로젝트화·동일 모듈의 부품별 폴더 집중은 전체 완료되지 않았다.**
이전 작업은 버전 관리·캐시 정리·기록 보존이었으며, 이번 병합도 그 이상을 구현했다고 보고하지 않는다.
`tools/`의 World model 18개·SAC 44개 도구가 남아 있고, 설치 가능한 pyproject.toml도 없다.
[실제 적용 상태와 남은 구조 변경](MODULARITY_STATUS_KO.md)에 구체적으로 남겼다.

공백 점검: 기존 main의 `tools/pubg_collector/03_fetch_telemetry.py`에 EOF 빈 줄 경고 1건이 있다.
이 파일은 원 main blob과 동일하게 보존했다. 이번에 작성·수정한 병합 파일의 공백 검사는 통과했다.
