# PUBG 정리 패치 사전 제시 (2026-10-26)

사용자 승인: LOL TEAMFIGHT 원칙으로 폴더 정리, 버전 관리, 정크 삭제, Lunecid GitHub commit/push.

## 적용할 패치

1. `README.md`, `VERSION`, `CHANGELOG.md`: 첫 공개 정리 릴리스 `2026.10.26-r1`, 연구 세대와 원고 버전 입구.
2. `versions/registry.json`, `versions/README.md`: snapshot legacy / BR·PTR / movement V1–V4 / world-model V3·G1–G4 / BigComp / SAC 원고 버전 구분. 코드·완료된 산출물 위치는 그대로 보존.
3. `tools/repository_release.py`: 표준 라이브러리만 사용하는 파일 목록·SHA256 검증 CLI. 비밀·분할 모듈은 열지 않음. 새 릴리스 경로는 덮어쓰지 않음.
4. `.gitignore` 뒤에 LaTeX 보조물·graphify 캐시·미추적 raw prediction/row CSV·비밀 split_v3 모듈 무시 규칙 추가. 기존 추적 증거는 유지.
5. `.idea/`의 이미 추적된 6개 설정은 로컬 보존하고 Git 색인에서만 제외.
6. `docs/repository/cleanup_2026-10-26/`: 기존 미커밋 상태, 삭제 사전 목록·크기·이유·해시, 삭제 영수증, 로컬 대용량 산출물 매니페스트, 검증 결과 기록.
7. `docs/guidelines/WORKING_RULES.md`, `PUBG_APPLICATION.md`, 루트 지침 진입점: 당일 정리·삭제·commit/push 승인 범위와 버전 관리 정책 연결. 공통 정책은 보존.

## 실제 파일 삭제 범위

보호 경로를 제외한 `.DS_Store`, Python `__pycache__`와 `.pytest_cache`, LaTeX `.aux/.blg/.out/.toc/.fls/.fdb_latexmk/.synctex.gz` 중 증거 문서가 직접 인용하지 않는 파일만 삭제. `.log`/실패 기록/원본/모델/데이터/원고/과거 재현 코드/참조 코드/worktree/venv는 보존.

## 검증

실제 모델·DB·서비스를 호출하지 않는 로컬 AST 분석, 기존 합성 시험, 파일 보존 해시, 릴리스 매니페스트 검증, Git staged 파일의 대용량·비밀 경로 점검. 새 학습·실험·평가 판정 변경 없음.
