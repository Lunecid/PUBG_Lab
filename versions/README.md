# 버전별 관리

저장소 릴리스: **2026.10.26-r2**. 이 번호는 모델·데이터·원고 버전과 별개다.
기존 import, 실행 명령, 증거 링크를 보존하기 위해 코드는 현재 경로에 두고,
아래 계열과 [기계 판독 색인](registry.json)으로 버전을 구분한다.

| 계열 | 기존 버전과 보관 위치 | 상태·해석 |
|---|---|---|
| Snapshot | [legacy](../legacy/README.md) | 초기 데모 보존, 현재 주 연구 아님 |
| BR·PTR | `model_stgcn/`, `stdb/` | 과거 실험·평가·재현 코드 보존 |
| 이동 생성 V1–V2 | `movement_diffusion/{bank,dataset,model}.py` | 초기 계약·실험의 재현용 |
| 이동 생성 V3 | `movement_diffusion/*_v3.py` | 재진입·공동 경로 비교의 기존 개발 실험 |
| 이동 생성 V4 | `movement_diffusion/model_v4.py` | [실패·한계를 포함한 개발 결과](../docs/experiment_design/HETEROGRAPH_DIFFUSION_V4_FORMATION_AUDIT_2026-10-15.md) 보존. 최종 개선안으로 채택되지 않음 |
| World model | [world_model](../world_model/README.md), V3 전처리·G1–G4 | 현재 연구 계열. 각 단계의 계약·결과를 별도로 보존 |
| BigComp 2027 | `paper/bigcomp2027/`, `en_v12/`, `overleaf/` | 원고 버전 보존 |
| SAC 2027 | `paper/sac2027/v13_*` … `v20_submission/` | 서로 다른 원고 릴리스 보존. 최신 번호가 투고·게재 완료를 뜻하지 않음 |

World model G4는 제한 CPU 비교이며, 개선 판정 미달·희귀 수송 한계가 기록돼 있다.
과거 문서에 등장하는 G3/G4 계획 명칭과 실제 후속 단계는 각 실행 계약으로 구분한다.
이번 정리에서는 모델을 다시 학습하거나 기존 결과를 재평가하지 않았다.

## 고정한 파일 확인

각 [릴리스 폴더](releases/2026.10.26-r2/manifest.json)는 공개 대상 파일의 경로·크기·SHA-256을 고정한다.
비밀 설정·보호 분할 모듈은 읽거나 해시하지 않으며, 대용량 로컬 자료는 별도 목록에서 관리한다.
해시는 파일 동일성의 근거이며 성능·재현 성공을 증명하지 않는다.

```sh
python3 tools/repository_release.py verify --release 2026.10.26-r2
```

새 변경은 새 브랜치·새 릴리스에 기록하고 과거 manifest와 원 실행은 수정하지 않는다.
기존 태그는 옮기지 않으며 강제 push로 과거 이력을 덮어쓰지 않는다.
