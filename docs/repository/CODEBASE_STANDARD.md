# 코드베이스 구조·양식 기준

날짜: 2026-10-26. Lifecycle: active. 책임: 현재 소스 구조와 팀 개발 기준을 정의한다.

## 구조

정식 Python 구현은 src/pubg_lab 안에 둔다. common은 표준 라이브러리 기반 경로·호환 실행만 소유한다.
데이터·World model·이동·생존·공간·분석은 재사용 부품을 소유하고, projects와 collectors는 이를 조합하는 응용이다.
같은 부품 계열의 실제 구현·CLI·README를 같은 담당 패키지 아래 둔다.
시험은 tests의 해당 계열에서 검증하고, 코드 소유는 configs/codebase.json에 둔다.

기존 root 모듈과 tools/scripts Python 경로는 호환 코드다. 실제 구현을 복제하지 않으며 정상 import에서는 같은 canonical 모듈 객체를 반환한다.
importlib spec로 보유한 예전 모듈 객체에도 속성과 공개 export를 연결한다.
보호 분할 구현은 비밀·접근 제한 정책 때문에 기존 위치를 유지하는 명시적 예외다.
과거 shell 배치·원고·완료된 실행·참조 코드는 보존한다.

## 의존과 코드 양식

- common → 표준 라이브러리. 라이브러리 → 공통·필요한 도메인 부품. 응용 → 라이브러리.
- 라이브러리에서 응용 프로젝트를 import하지 않는다. 도메인 순환과 호환 경로 역참조를 정적으로 검사한다.
- 클래스·함수는 책임 단위로 모은다. 공유 여부가 확인되지 않은 수식을 이름만 같다는 이유로 합치지 않는다.
- 새 공개 API에는 타입·단위·마스크·예외를 설명한다. 논문에서 온 연산은 원 문헌 근거를 유지한다.
- Ruff 포맷, 명시한 핵심 lint, 구조 검사, 합성 회귀와 wheel 격리를 사용한다.
- import 순서·연구 상수·산술 연산·표본 선택은 양식 변경과 구분해 검증한다.

## 근거와 범위

PyPA의 [src layout 설명](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)과
Google Research Scenic의 [공통 라이브러리·프로젝트 분리](https://github.com/google-research/scenic#scenic-component-design)를 참고했다.
양식 설정은 [Ruff 공식 문서](https://docs.astral.sh/ruff/configuration/)를 따른다.
특정 기업의 비공개 코드베이스와 동일하거나 모든 기업이 같은 형식을 쓴다고 주장하지 않는다.
외부 참고 프로젝트의 연구 설계·라이선스를 우리 연구의 승인으로 옮겨오지 않는다.
