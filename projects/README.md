# 실행 프로젝트

각 폴더가 실제 Python 프로젝트 패키지다. 같은 프로젝트의 입력 준비·학습·평가·보고 부품을 역할별 하위 폴더에 둔다.
재사용 모델·자료 계약은 `src/pubg_lab/`에서 가져온다. 라이브러리는 프로젝트 실행 코드를 import하지 않는다.

| 프로젝트 | 범위 |
|---|---|
| `world_model/` | 공간·경기 표현의 준비, 단계별 학습, 검증, 보고 |
| `movement/` | 개인·팀 경로 생성의 버전별 준비, 학습, 진단, 보고 |
| `battle_royale/` | BR/PTR 실험 구성, ablation, 비교, 기존 실행 launchers |
| `sac2027/` | SAC 원고에 연결된 학습, 통제 비교, 통계, 보고·전달 |
| `krafton_baseline/` | 고정된 KRAFTON 참조 구현을 호출하는 로컬 어댑터 |
| `spatial/` | Erangel/Caldera의 수집, 공간 구성, 검증, 시각화 |
| `network_analysis/` | 네트워크 연구별 분석, 실행, 보고 |
| `data_preparation/` | 명시적으로 실행하는 자료 준비 |
| `collector/` | 명시적으로 실행하는 수집기 |
| `figures/` | 그림 렌더링 |

목록 확인: 저장소 루트에서 `PYTHONPATH=src:. python3 -m pubg_lab list`.
새 명령을 실행할 때는 각 연구 계약의 승인·자료 접근 범위를 따른다. 구조 검사나 명령 목록은 연구 실행이 아니다.
