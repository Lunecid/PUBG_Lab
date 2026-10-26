# 개발 안내

작업 전에 AGENTS.md, WORKING_RULES.md, PUBG_APPLICATION.md와 해당 연구 계약을 읽는다.
모집단·분할·시드·시간 경계·손실·평가 기준을 구조 변경이나 양식 변경으로 바꾸지 않는다.

## 개발 환경

학습 환경과 분리된 환경에서 개발 도구를 준비한다.

```sh
python3 -m venv /tmp/pubg-lab-dev
/tmp/pubg-lab-dev/bin/python -m pip install -r requirements-quality.txt
/tmp/pubg-lab-dev/bin/python -m pip install --no-deps --no-build-isolation -e .
make structure format-check lint test-architecture PYTHON=/tmp/pubg-lab-dev/bin/python RUFF=/tmp/pubg-lab-dev/bin/ruff
```

수치·모델 합성 시험에는 별도로 고정된 연구 패키지가 필요하다. `make test-synthetic`은 데이터·DB·서비스를 사용하는 연구 실행을 대체하지 않는다.
새 의존성이 필요하면 프로젝트 설치 규칙과 별도 환경·잠금·시험 절차를 따른다.

## 코드 소유와 API

- 새 코드는 `src/pubg_lab/<담당 모듈>/`에 둔다. 기존 root 호환 경로에는 구현을 추가하지 않는다.
- 모델·데이터·평가 부품은 이름과 책임이 일치하는 파일로 나눈다. 한 함수·클래스마다 별도 파일을 만들지는 않는다.
- 공통 부품은 실제로 여러 모듈이 공유할 때만 common으로 옮긴다. 연구별 수식은 해당 소유 모듈이 유지한다.
- 프로젝트 응용은 공통·도메인 부품을 조합한다. 라이브러리는 projects·collectors·repository에 의존하지 않는다.
- 공개 새 함수에는 타입과 역할·입출력·단위를 적는다. 입력과 라벨, 결측과 0을 구분한다.
- 명령 진입점은 같은 담당 패키지에 두고 CLI와 재사용 기능을 구분한다. 등록된 명령은 pubg-lab list로 찾는다.

## 양식·검증·기록

Ruff 0.14.1, Python 3.11 기준, 공백 4칸, 줄 길이 100, double quote, LF를 사용한다.
복수 import는 순서를 보존해 분리한다. 수치 라이브러리 전 환경 설정이 있으므로 import 자동 정렬은 하지 않는다.
lint는 복수 import·문법·정의되지 않은 이름 등 현재 고정한 핵심 규칙을 검사한다. 저장소 전체의 미사용 import 제거·완전한 정적 타입 검사를 완료했다고 주장하지 않는다.

구조 변경에는 파일 이동표, 기존 호출 호환, 합성 불변식, 패키지 격리 검증을 함께 남긴다.
과거 릴리스·run·코드 snapshot·실패 기록은 수정하지 않는다. 새 릴리스는 새 manifest와 태그로 기록한다.
커밋·push·학습·데이터 전송은 사용자 승인 범위를 따른다.

GitHub workflow는 PR 또는 수동 실행용이다. 현재 로컬 검증 성공을 원격 CI 성공으로 부르지 않는다.
