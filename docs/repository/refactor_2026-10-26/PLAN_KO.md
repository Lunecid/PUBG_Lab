# 코드베이스 구조·양식 재편 사전 설계

날짜: 2026-10-26. 저자 요청: 코드 양식과 전체 구조를 대규모 코드베이스처럼 체계화한다.
B3·B7에 따라 아래 파일 이동·변환 패치를 먼저 제시하고 요청 범위 안에서 적용한다.

## 실제 적용 구조

```text
pyproject.toml                # 패키징·시험·양식 설정
src/pubg_lab/
  common/                     # 저장소 경로와 호환 진입점
  data/{stdb,quality,preparation}/
  world_model/{기존 부품,cli}/
  movement/{기존 부품,cli}/
  survival/{ptr,phase_v2,sagong2025,v3,...}/
  spatial/{erangel,caldera}/
  analysis/network/
  collectors/pubg/
  projects/{sac2027,battle_royale,figures}/
  repository/
tests/                       # 기존 연구 검증 + 구조·호환·패키징 검증
configs/                     # 코드 품질·소유 경계; 동결 연구 상수는 이동하지 않음
docs/                        # 설계·계약·기록
tools/, scripts/, 기존 패키지/ # 이전 호출 경로를 위한 호환 코드
```

[기계 판독 이동표](move_plan.json)에 원 경로·새 경로·크기·사전 해시를 기록했다.
일반적인 src 패키지 형식과, 공통 라이브러리/프로젝트 코드의 분리를 채택한다.
특정 기업의 비공개 내부 코드베이스와 동일하다고 주장하지 않는다.

## 적용 패치

1. 실제 Python 구현을 책임별 패키지로 옮긴다. 공개 API·모델 수식·분할·평가·시드·연구 상수는 유지한다.
2. 내부 import와 저장소 루트 계산, 새 실행의 소스 경로 참조를 기계적으로 갱신한다.
3. 기존 Python 경로에는 구현을 복제하지 않는 호환 진입점만 둔다. 원본 자료·과거 코드 snapshot·동결 원고·legacy·참조 코드는 수정하지 않는다.
4. pyproject·README·공개 API 안내·폴더 책임·개발 명령·품질 점검과 합성 CI를 구성한다.
5. 공통 경로 처리와 호환 실행을 작은 부품으로 묶는다. 연구별 수식·손실·평가는 이름만 같다는 이유로 공통화하지 않는다.
6. Ruff 버전을 작업용 임시 환경에만 고정하고 양식을 통일한다. import 실행 순서는 자원·결정성에 영향을 주므로 자동 재정렬하지 않는다.

## 검증과 제한

- 기존 139개 합성 시험을 재실행하고, import 동일성·과거 명령·경로·배포 패키지 격리 시험을 추가한다.
- 변환 후와 포맷 후 AST를 대조해 양식 변경이 로직을 바꾸지 않았음을 확인한다.
- 경로·import·bootstrap 외 연구 함수·상수 변화를 별도로 점검한다.
- 보호 분할 모듈·비밀 설정은 읽거나 이동하지 않는다. 과거 실행을 새 코드로 덮어쓰지 않는다.
- 실제 DB·API·서비스·훈련·보호/최종 테스트를 실행하지 않는다.
- r1–r3는 보존하고 완료한 구조를 새 릴리스로 기록한다.

근거: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
https://github.com/google-research/scenic#scenic-component-design
https://docs.astral.sh/ruff/configuration/
