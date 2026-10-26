# 사용자용 애플리케이션

- `webapp/`: 네트워크 연구 결과를 탐색하는 React/Vite 앱. `src/components`, `src/routes`, `src/lib`를 함께 버전 관리한다.
- `presentations/`: 발표 자료 생성 코드. 기존 PPTX 자료는 원위치에 보존한다.

로컬 생성 데이터·node_modules·dist는 Git에서 제외한다. 웹앱의 `build:data`는 실제 연구 자료를 읽으므로 구조 검증에 포함하지 않는다.
