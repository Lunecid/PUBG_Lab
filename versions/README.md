# 버전별 관리

현재 저장소 릴리스는 **2026.10.26-r5**다. 모델·자료·실행·원고 버전과 별도로 관리한다.

| 계열 | 현재 코드 위치 | 보존하는 버전과 해석 |
|---|---|---|
| 초기 Snapshot | [archive/legacy_snapshot](../archive/legacy_snapshot/) | 초기 데모, 내용 그대로 보존 |
| BR·PTR | [survival](../src/pubg_lab/survival/), [battle_royale](../projects/battle_royale/) | PTR·v3의 모델·자료·학습·평가·실험을 역할별 분리 |
| 이동 생성 | [movement 라이브러리](../src/pubg_lab/movement/), [실행 프로젝트](../projects/movement/) | data/models에서 v1·v3·v4 코드를 구분, 기존 버전 삭제 안 함 |
| World model | [라이브러리](../src/pubg_lab/world_model/), [실행 프로젝트](../projects/world_model/) | 전처리 v3·G1–G4의 계약·결과 분리 |
| KRAFTON 비교 | [로컬 어댑터](../projects/krafton_baseline/), third_party 참조 | 고정 참조 코드의 내용 보존, 현재 호출은 정식 모듈 |
| BigComp 2027 | paper/bigcomp2027 | en_v12·overleaf 등 원고 이력 |
| SAC 2027 | [실행 프로젝트](../projects/sac2027/), paper/sac2027 | v13–v20 원고 이력, 최신 번호가 투고·게재 완료를 뜻하지 않음 |

[registry.json](registry.json)은 과거 경로와 현재 정식 경로를 구분한다.
이동 생성 V4와 World model G4의 기존 실패·한계·판정은 원 보고서를 따른다.
이번 정리는 재학습·실자료 재평가를 수행하지 않았다.

## 릴리스 검증

[r5 매니페스트](releases/2026.10.26-r5/manifest.json)는 공개 파일의 경로·크기·SHA-256을 고정한다.
비밀 설정·보호 분할은 읽거나 해시하지 않는다. 해시는 파일 동일성만 증명한다.

```sh
PYTHONPATH=src:. python3 -m pubg_lab.repository.repository_release verify --release 2026.10.26-r5
```

r1–r4 매니페스트·태그·당시 보고서는 수정하지 않는다. 과거 manifest 검증은 해당 태그 checkout에서 수행한다.
현재 코드 위치는 [구조 기준](../docs/repository/CODEBASE_STANDARD.md), 이동·삭제 내역은 [r5 기록](../docs/repository/layout_2026-10-26/)을 따른다.
