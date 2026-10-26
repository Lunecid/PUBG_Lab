# PUBG 개인·팀 이동 생성 연구 코드

현재 비교 실험은 **V3**다. `bank_v3.py`는 생애 episode와 이동 모드를 복원하고, `dataset_v3.py`는 관측자·서버 조건을 분리하며, `model_v3.py`는 같은 입력의 공동 회귀와 Diffusion을 구현한다. 이전 데이터셋 v1과 모델 v2는 재현을 위해 그대로 보존했다.

## V3 실행과 계약

- 데이터: `data/processed/heterograph_diffusion_v3`, 학습657 / 내부 검증188 / 잠금 테스트 후보94경기.
- 잠금 후보94경기는 이번 학습·공간 사전·어휘·평가에서 제외하지만, 이전 V2 학습 이력이 있다. 완전히 새로운 외부 테스트는 아직 없다.
- 재진입 항공기·긴급 호출 수송·지상 차량·수상 차량·글라이더·고정 탑승을 구분한다. 전용 원본 이벤트가 없는 재진입과 항공기 이탈 상태는 명시적으로 추정한다.
- 입력은 과거 기록만 사용한다. 목표 좌표의 라벨용 보간은 같은 생애·같은 모드 안에서만 허용한다. 사망 뒤의 다른 생애를 같은 미래 경로로 연결하지 않는다.
- 현재 두 원의 팀원별 상대 좌표·반경·경계 거리를 직접 조건으로 주며, 두 모델은 같은 입력과 공동 20/40/60초 지점 표현을 쓴다. Diffusion은 팀원들의 noisy 잔차를 함께 집계한다.
- 학습 설정과 결과: [V3 비교 보고서](../reports/heterograph_diffusion_experiment_v3/REPORT_KO.md), [데이터 계약](../data/processed/heterograph_diffusion_v3/contract.json), [잠금 명세](../data/processed/heterograph_diffusion_v3/split_lock.json).

```sh
OPENBLAS_NUM_THREADS=1 /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/build_v3.py
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v3.py prepare
# 시드41/42/43 각각 실행
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v3.py train-seed --seed 41
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v3.py train-ablation
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v3.py evaluate
```

관측자 입력은 정확한 자기 상태 텔레메트리와 팀 HUD의 근사다. 실제 화면에서 지각 가능한 값만 검증한 모델은 아니다. 안전구역의 실제 HUD 공개 시각도 미검증이다. 테스트 로더는 현재 의도적으로 막혀 있다.

## 보존된 데이터셋 v1 / 모델 v2 기록

아래 수치와 첫 사망 검열 정책은 이전 버전의 계약이며 V3에 적용하지 않는다.

## 데이터 범위와 생성 결과

40.2 추정 기간의 Steam 공식 Erangel TPP squad **939경기 전부**를 로컬 PostgreSQL에서 읽기 전용으로 추출했다. 다른 맵, FPP, competitive, airoyale는 이번 학습 코호트에 포함하지 않았다. 정확한 클라이언트 빌드 번호와 동시기 공간 형상은 미검증이다.

| 분할 | 경기 | 현재 시점 조건을 만족하는 창 | 본인 미래 정답이 하나 이상 있는 창 | 고정 대상 팀원 전원의 60초 정답이 있는 창 |
|---|---:|---:|---:|---:|
| 학습 | 751 | 1,475,648 | 1,447,837 | 1,220,424 |
| 검증 | 188 | 367,206 | 360,017 | 303,597 |

기존 `reports/erangel_100m_40_2_2026-10-15/match_split.csv`의 경기 분할을 유지했다. 새 독립 test split은 만들지 않았다. 183만 개 이상의 창은 서로 겹치는 반복 관측이며 독립 표본 수가 아니다. 추론·통계의 기본 단위는 경기다. 학습 실행기는 경기 선택 후 그 경기 안의 창을 선택하여 긴 경기의 과도한 가중을 줄인다.

현재 버전은 **각 참가자의 첫 사망 이전 생존 구간**을 복원한다. 리콜·재등장 이후 이동은 대상에서 제외한다. 첫 사망 이후 양수 체력 위치가 기록된 행은 학습 188,401행, 검증 45,469행이다. 원본에는 보존했지만 상태 복원에는 쓰지 않았다. 서버용 역시 이 복원 계약 안에서의 정보 확장판이며, 게임 엔진의 완전한 정답이나 리콜을 모두 처리한 전지적 상태가 아니다.

## 한 학습 표본

- 예측 기준 시각 `t0`: 30초 간격. 현재 위치가 유효한 참가자가 관측자 역할을 한다.
- 과거 문맥: `t0−30, −20, −10, 0초`의 그래프 4개.
- 미래 정답: `t0+10, +20, +30, +40, +50, +60초`의 개인별 XY.
- 타깃 개인: 현재 시점에 위치가 유효한 같은 로스터 멤버. 관측자를 첫 슬롯에 두고 최대 4명을 고정한다. 미착지·오래된 위치·첫 사망 이후 인원은 타깃 멤버에 포함하지 않는다. 로스터의 나머지 개인 노드는 상태 마스크와 함께 문맥에 남는다.
- 미래에 누군가 사망하거나 기록이 빠져도 다른 사람으로 슬롯을 교체하지 않는다. 타깃 유효성 마스크로 해당 개인·시점의 손실을 제외한다.
- 팀 중심 정답: 고정 대상 멤버의 미래 위치를 **동일 가중 평균**. 해당 미래 시점에 전원의 정답이 있을 때만 팀 손실·평가가 유효하다. 현재 관측 팀 중심은 표본 나이로 가중하지만 미래 정답 중심은 고정된 동일 가중이다.
- 모든 XY는 `t0` 관측자의 마지막 유효 위치를 원점으로 하고 1,000m로 나눈다. 시간 이력도 같은 현재 원점에 정렬한다.
- 인간·봇 구분은 계정 접두어 기반 분석용 메타데이터다. 모델 특징이나 임베딩에는 넣지 않는다.

정답은 목표 시각에 가장 가까운 실제 위치 표본(±5초)을 사용한다. 미래 좌표 보간은 하지 않는다. 정답의 실제 기록 시각과 목표 시각 차이를 `time_error_s`로 보존한다. 첫 사망 경계를 넘는 라벨, 좌표 범위 밖, 동일 시각 충돌 위치, 15초 이내 인접 표본 사이 120m/s를 넘는 위치 도약은 유효 위치에서 제외한다. 120m/s는 이상치 제거 휴리스틱이지 게임의 공식 최대 속도가 아니다.

## 같은 정답을 공유하는 두 입력

| 정보 | `observer` | `server` |
|---|---|---|
| 본인 위치·체력·과거 속도 | 포함 | 포함 |
| 같은 팀의 위치·체력·기절/사망 근거 | HUD 대리값 가정으로 포함 | 포함 |
| 본인 장비·차량·피격·아이템 획득/사용 사건 요약 | 포함 | 포함 |
| 동료의 비공개 장비·차종·피해 | 제외 | 포함 |
| 상대 팀의 개인·팀 노드와 상태 | 제외 | 전체 참가자 범위에서 포함 |
| 실제 가해 피해량·최근 공격자 위치 | 제외 | 과거 30초 사건으로 포함 |
| 전역 생존 팀 수 | 제외 | 포함 |
| raw `safe_zone` / `poison_zone`, 생존자 수·단계 | 동일한 과거 기록 | 동일한 과거 기록 |
| 미래 안전구역·미래 위치·최종 순위 | 제외 | 제외 |

`observer`는 실제 플레이 화면을 검증한 strict perception 모델이 아니라 **관측 가능 정보의 텔레메트리 대리값 모델**이다. 팀 HUD의 정밀도·지연·공유 상태와 안전구역의 화면 발표 시점은 미확인이다. `server`는 수집된 정규화 텔레메트리 범위의 확장판이며, 보이지 않는 아이템·빈 차량·시야·사운드·완전한 인벤토리까지 복원하지 않는다.

두 로더는 동일한 `sample_id`, 타깃 멤버 순서, 정답, 정답 마스크를 공유한다. `server`에서 추가되는 노드·비공개 속성·관계만 달라진다. 두 모델의 파라미터 수는 같고, 입력 그래프의 크기는 다르다. 성능 차이는 복원·가용성 계약을 포함한 정보 접근의 효과이며 인간이 실제로 얻을 수 있는 개선량으로 해석하면 안 된다.

## 공간·개인·팀 관계

노드 타입은 cell, region, player, team, vehicle_evidence, contact_evidence, zone, game, equipment의 9종이다. equipment는 개인에게 귀속된 장비 토큰을 그래프 노드로 펼친 것이다. 같은 차종·장비 종류라는 이유로 두 개인의 물체를 하나로 합치지 않는다.

- 전역 400m 지역 **441개**를 매 그래프에 유지한다.
- 100m 상세 셀은 해당 역사 시점의 자기 팀 유효 위치 주변 ±2셀에서 선택한다. 두 입력 버전의 상세 셀 선택 규칙은 같고 미래 경로를 보지 않는다. 서버의 먼 상대는 전역 지역에 연결한다.
- 역사적 높이·육수 비율·변경 주의 영역은 기존 지도 자산의 대리값이다. 40.2 건물·교량·보행 가능성은 보증하지 않는다.
- 관측된 비차량/차량 이동 관계를 구분한다. 근거가 없는 연결은 역사적 육지 비율 ≥0.6, 높이 차/100m ≤0.3인 4방향 이웃에 한해 추정 후보로 만든다. 수치는 탐색용 규칙이며 통행 가능성 판정기가 아니다.
- 이동 그래프는 이 코호트의 학습 **751경기만** 사용해 다시 집계했다. 관측된 방향·이동 유형별 연결은 116,607개다. 학습 표본을 읽을 때는 **그 경기 자체의 이동 횟수·경기 지지 수를 차감**한다. 검증 표본은 학습 그래프만 읽는다. 자기 미래 경로가 정적 prior에 들어가는 문제도 줄였다.
- 팀 노드는 현재 멤버 위치 집합에서 중심·메도이드·반경·최대/평균 거리·유효 인원·사망 근거를 요약한다. 강제 응집 손실을 쓰지 않는다.
- 피격 증거는 익명이다. 관측자용 입력에 적 위치·정체·가시성 정답을 만들지 않는다. 피해 없음은 적 없음이 아니다.

모든 관계는 순방향/역방향을 별도 타입으로 구분한다. 14개 기본 관계 × 2 = **28개 관계 타입**이다. 지역 격자 인접·추정 연결은 이동 가능성 보증이 아니다. 상세 셀 crop 밖의 경로는 전역 지역 표현을 읽을 수 있지만, 현재 데모 모델이 이를 잘 활용하는지는 학습 후 검증해야 한다.

## 실제 텐서와 로더

`data/processed/heterograph_diffusion_v1/`에는 939개 경기별 압축 원본, 939개 상태 bank, 두 분할의 공통 표본 인덱스, 학습에서만 만든 어휘 사전·공간 prior·경기별 차감 자료가 있다. 현재 약 745MiB이다. 수백만 창마다 같은 경기 상태를 복제하지 않고, 공유 bank에서 시점별 그래프를 구성한다.

```python
from torch.utils.data import DataLoader
from movement_diffusion.dataset import TrajectoryDataset, collate

observed = TrajectoryDataset(split="train", view="observer")
server = TrajectoryDataset(split="train", view="server")
loader = DataLoader(observed, batch_size=2, collate_fn=collate, num_workers=0)
batch = next(iter(loader))
conditions = batch["conditions"]   # 이것만 모델의 조건 입력으로 전달
labels = batch["targets"]         # 손실·평가 및 누락 라벨의 과거 기반 대체용
metadata = batch["meta"]          # 경기/인간·봇 분석용; 입력 임베딩 금지
```

| 텐서 | shape / 의미 |
|---|---|
| `conditions.x`, `feature_mask` | `[전체 노드 수, 32]`; 4시점 × B개의 가변 크기 그래프를 묶음 |
| `node_type`, `category` | `[전체 노드 수]`; 범주 0=padding/토큰 없음, 1=학습 사전에 없는 값 |
| `edge_index` | `[2, 전체 방향 연결 수]` |
| `edge_type` | `[전체 방향 연결 수]` |
| `edge_attr`, `edge_mask` | `[전체 방향 연결 수, 5]`; 지지 요약·불확실성 표시·사건 나이·서버 공격자 XY. 관계 타입별 의미를 구분 |
| `target_nodes` | `[B, 4개 과거 시점, 4개 개인 슬롯]`; padding −1 |
| `current_presence` | `[B, 4]`; 현재 시점에 고정한 타깃 멤버 슬롯 |
| `initial_xy`, `velocity` | `[B, 4, 2]`; 현재 위치·과거 차분 속도 |
| `targets.trajectory` | `[B, 4, 6, 2]` |
| `trajectory_valid`, `mode_valid` | `[B, 4, 6]` |
| `team_center`, `team_valid` | `[B, 6, 2]`, `[B, 6]` |
| `mode` | `[B, 4, 6]`; 정답 위치와 같은 원본 표본의 차량 기록 여부. 주변 차량 존재 여부가 아님 |
| `time_error_s` | `[B, 4, 6]`; 실제 정답 표본의 시간 오차 |
| 생성 경로 | `[B, K, 4, 6, 2]`; 같은 K번째 시나리오 안에서 팀원 경로를 함께 해석 |

실제 B=2 배치에서 observer는 **3,898노드 / 23,628연결**, server는 **6,355노드 / 29,468연결**이었다. 이는 특정 두 표본의 런타임 측정값이며 고정 shape가 아니다. `graph_schema.json`과 `contract.json`에 실제 필드 이름·스케일·마스크 규칙을 저장했다.

## Diffusion 학습과 비교 실행

현재 `model.py`는 초기 연결 시험을 교체한 v2 모델이다. 타입별 graph message passing 2층 → 각 팀원별 4시점 GRU → 팀원 간 attention → 공동 잔차 궤적 v predictor로 구성한다. 과거 안전구역 노드가 조건에 포함된다. 모든 예측은 과거 정보로 만든 등속도 경로 주위의 잔차이며, 좌표 단위는 km, 확산 잔차 scale은 200 m다. 현재 위치 표본의 `position_age_s`도 조건과 등속도 기준에 반영한다.

두 정보 조건 각각 3개 시드, 각 2,500회 update를 실제 실행했다. 파라미터는 각각 204,044개다. 같은 표본·초기 난수·학습량을 사용했고 검증 188경기에서 총 1,504개 창을 평가했다. 원래 3-step smoke 결과는 과거 연결 시험으로 보존했다. 최종 수치와 제약은 [실제 학습 및 감사 보고서](../reports/heterograph_diffusion_experiment_v2/REPORT_KO.md)에 있다.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v2.py --output reports/heterograph_diffusion_experiment_v2
```

완료 checkpoint가 같은 출력 경로에 있으면 재사용하므로, 완전한 재학습에는 새 경로를 지정한다. 모델 선택은 고정 예산의 마지막 EMA다. `train.py`는 한 경기당 한 창의 빠른 실행용으로 남겨 두었으며, 실제 비교에는 `experiment_v2.py`를 쓴다.

미래 정답 mask는 loss·평가와 누락 잔차의 0 보충에만 쓰며 encoder나 attention mask에는 제공하지 않는다. 미래 차량 mode 라벨은 준비했지만 예측 head는 아직 없다. 생성 경로를 지도 간선에 강제로 투영하거나 collision을 막는 sampler도 없다.

평가는 개인·팀 중심의 평균경로/기대/최소 ADE, 고정 60초 FDE, joint ADE, 공동 Energy score, 팀원 간 거리 MAE를 포함한다. 등속도와 현재 위치 유지 기준을 함께 평가하고 경기 단위 bootstrap을 제공한다. 최소 ADE는 정답을 본 뒤 선택하는 oracle 지표이므로 이것만으로 우위를 주장하지 않는다.

예제 배치는 `data/processed/heterograph_diffusion_v1/examples/`에 저장했다. `conditions_observer.npz`, `conditions_server.npz`와 공통 `shared_targets.npz`를 물리적으로 분리해 입력·정답의 경계를 확인할 수 있다. `metadata.json`에는 두 버전의 동일 sample ID와 좌표 원점이 있다.

## 검증과 알려진 한계

- [전체 bank 감사](../reports/heterograph_dataset_v1/full_audit.json): 939경기 모두에서 유한값·마스크·과거 시각·정답 시간 허용오차·경기 분할을 점검했다.
- [조건부 Diffusion 연결 검사](../reports/heterograph_dataset_v1/smoke.json): 실제 두 표본, 두 입력 버전의 학습/생성 shape와 finite loss·gradient를 확인했다.
- [대응 비교 실행](../reports/heterograph_dataset_v1/paired_run_check/results.json): 두 모델의 학습 표본 선택·검증 sample ID가 대응하는 실행 경로를 확인했다. 이 파일의 점수는 3-step 시험 값이며 성능 결과로 인용하지 않는다.
- 자동 검사는 과거 prefix 복원 불변성, 비공개 특징 변경 시 observer 불변성, paired label 동일성, 자기 경기 공간 prior 차감, 고정 멤버 팀 정답, 동일 시각 장비 순서 불변성, 완전 예측의 평가값 0, 미래 라벨 변경 시 조건 불변성을 포함한다.
- ±5초 라벨 정합의 평균 절대 시간 오차: 학습 2.497초, 검증 2.482초. 현재 입력도 최대 15초 오래된 표본의 대리값이다.
- 유효 지상 상태 중 장비 동일 시각 충돌이 남아 있는 비율: 학습 **22.41%**, 검증 **21.93%**. 충돌한 아이템은 확정 토큰에서 제외하고 ambiguity count를 남긴다. 12개 장비 토큰 제한에 의해 잘린 토큰은 0개였다.
- 처음부터 완전한 장비 이력이 있다는 보장은 없다. 이벤트가 존재하거나 equip 이후 해제가 없다는 것은 완전한 인벤토리·실제 손에 든 무기·잔탄의 증거가 아니다.
- 완전한 60초 공동 정답 비율은 학습 82.70%, 검증 82.68%이다. 완전 창만 평가할 경우 생존 선택 편향이 생기므로 부분 창 마스크 결과와 함께 보고한다.
- 같은 계정이 다른 경기 분할에 등장하는지 검증하지 않았다. account-disjoint 평가라고 부르지 않는다.
- 인간·봇 비교에는 팀 규모, 차량, 경기 단계, 장비 불확실성, 표본 나이 등을 함께 맞춰야 한다. 관측 결측이나 초기 장비 로깅 차이를 행동 차이로 해석하지 않는다.

## 재현 파일

- `tools/movement_diffusion/extract.py`: 읽기 전용 DB 추출. 비밀번호는 getpass로 받고 저장하지 않음.
- `tools/movement_diffusion/build.py`, `movement_diffusion/bank.py`: 어휘·공간 prior·bank·인덱스 생성.
- `movement_diffusion/dataset.py`: 관측 범위 분리, 시간별 이종 그래프, PyTorch collate.
- `movement_diffusion/model.py`: 실행 가능한 최소 공동 경로 Diffusion.
- `movement_diffusion/evaluation.py`, `tools/movement_diffusion/train.py`: 대응 비교 학습·평가.
- `tools/movement_diffusion/audit.py`, `tests/movement_diffusion/test_contract.py`: 전체 검사와 계약 테스트.
- `data/processed/heterograph_diffusion_v1/bank_sha256.json`: bank 파일별 해시.

실행 환경의 버전은 `reports/heterograph_dataset_v1/runtime_versions.json`에 기록했다. 임시 가상환경이 사라지면 해당 버전으로 다시 설치한다. DB 재접속 없이 저장된 원본에서 bank·표본을 재생성할 수 있다.
