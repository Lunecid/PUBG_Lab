# G2 전체 입력 검사와 소규모 사전학습

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run (사용자가 코드 구현·수정·자체 피드백·학습을 직접 승인)
- Origin Date: 2026-10-20
- Verification Status: 실행 전 계약
- Version Label: g2_cpu_v1

## 승인·범위와 적용 전 패치

저자 지시: “전체 입력 검사하고 사전학습 손실까지 검증한 다음 소규모 CPU 학습을 해봐.
자기 피드백 반복하면서 결과와 시각화까지 보고하고, 보호 경기는 계속 잠가둬.”
원천 DB·동결 V3 데이터·기존 완료 실행은 변경하지 않는다.
추가 파일: 공간 조회 인덱스, 전수 감사 도구, 사전학습 heads/losses,
toy 시험, 소규모 CPU runner, 보고/그림 도구. 필요 수정은 반례와 새 실행에 기록한다.
무거운 결과는 승인된 `/Users/lunecid/PUBG_Data/runs/20261020_world_model_g2_cpu_v1`,
요약은 `reports/world_model_g2_2026-10-20`에 저장한다.

## G2a 검사

`full.json`의 train/validation ID만 상세 열람한다. metadata의 protected/reserved 등과
교집합이 있으면 즉시 실패한다. 분할 모듈 본문은 읽지 않는다.
모든 bank의 모든 입력 배열·모든 anchor를 벡터화하여 검사한다:
finite/mask, source time≤각 frame, 좌표/팀/아이템 인덱스, anchor 현재 유효성,
future horizon의 episode/segment 경계와 label support.
현재 유효 팀원으로 고정한 완전 60s 공동 정답 및 부분 정답 분포를
개인-anchor와 중복 제거한 team-time 양쪽 분모로 집계한다.
모든 비어 있지 않은 경기에서 첫 anchor의 observer/server **실제 어댑터 조립**도 검사한다.
이는 모든 창을 Python adapter로 생성한다는 뜻은 아니다. 실행 범위를 구분해 보고한다.
오류는 match ID와 연결하고 학습 전에 hard error 0을 요구한다.

공간 조회 가속은 source-cell 인덱스를 만들되 기존 loader와 숫자/순서 동일성을 시험한다.
현재 경기 제외는 G1과 동일하다. 4 CPU worker까지 사용, 합산 private memory 48GiB 미만.

## G2b 과제와 검산

| 과제 | 선택·근거·출력 | 손실·반증 |
|---|---|---|
| P1 | GraphMAE/ForecastMAE의 masked reconstruction 원리. focal actor의 체력 채널을 전체 과거에서 가리고 현재 관측 체력/100 복원 | 선택된 관측값만 MSE. 입력의 원래 값 변조가 가린 입력에 남으면 실패 |
| P2 | V3의 공동 GMM. 동일 team component K=4, 각 현재 팀원의 10–60s 상대 이동 평균/대각 표준편차. mode 10개 범주 | 관측 좌표 marginal joint NLL/유효 좌표 수, mode CE. 혼합성분/개인 순서 변경에 불변 |
| P3 | 팀원 쌍의 대칭 latent 및 현재 거리→미래 거리/100. 두 개인과 팀 표현 유지 | 관측 pair MSE. solo는 미정의로 제외. 가까이 모이도록 강제하지 않음 |

P4/P6, 위치 복원, 장비 복원, occupancy, Diffusion은 이번 제한 실행에서 끈다.
P1은 허용된 정보의 체력만 가린다. 공간·속도를 가리는 우회 누수 문제는 만들지 않는다.
P2는 log std [-4,4]로 한정한다. joint NLL은 component 내부에서 관측 차원을 합한 후
logsumexp를 계산하며, 독립 개인 mixture로 바꾸지 않는다.
총 목적함수: `L = Lhealth + Lmotion + 0.1 Lmode + 0.1 Lpair`.
가중치는 소규모 실행의 고정 engineering 설정이며 최적값/논문 검증값이 아니다.
각 손실의 support와 gradient 규모를 기록한다. 빈 mask는 해당 과제에서 제외한다.
손계산, NumPy, torch 값을 비교하고 gradient/마스크 반례 시험 후 실제 학습한다.

## G2c 미리 고정하는 제한 실행

- 선택: 감사 통과·anchor 존재 경기에서 SHA256(match_id)의 오름차순으로
  train 최대 32경기, development 최대 16경기. map을 선택 조건으로 쓰지 않으며 실제 분포 보고.
- 팀 중복을 줄이기 위해 같은 (team,time)의 최소 actor ID를 대표 ego로 삼는다.
  팀별·시간별 고정 해시 순서로 경기당 train 최대 8, development 최대 4 anchor를 보유한다.
  미래 정답 유무로 현재 팀원을 줄이지 않는다. 무정답 sample은 해당 과제만 skip.
- 학습 추출은 **경기→팀→시각** 균등 추출. observer/server는 동일 anchor를 각각 학습한다.
- 인코더는 G1 d=128. CPU 1 thread, seed=20261020, MHA fastpath off,
  deterministic algorithms, AdamW lr=3e-4, weight_decay=1e-4, foreach=False,
  gradient norm clip=1.0. 한 update에 2개 sample의 평균.
- 사전학습 256 update. 선택된 첫 8 train sample의 고정 loss를 시작/종료에 검사.
  validation을 보며 best checkpoint/epoch/가중치를 선택하지 않는다.
- 표현 비교: 사전학습 encoder를 동결하고 새로운 동일 초기화 선형 경로 head를 128 update.
  같은 초기 난수 encoder를 동결한 대조군도 동일 head·sample 순서·128 update.
  이는 **frozen representation probe**이며 end-to-end scratch 대비가 아니다.
- 추가 기준선: 현재 관측 속도의 CV, 속도 결측이면 0을 쓰며 fallback 빈도 보고.
- 한 view 학습 wall time 600초 상한. nonfinite/메모리 초과/학습 경계 위반이면 실패 기록.
  실패 수정은 원 실행을 보존하고 원인·수정 후 새 실행으로 한다(저자의 자기 피드백 승인).
- 전체 소규모 학습을 별도 프로세스에서 한 번 재실행하여 state/예측 hash 비교.
  자동으로 큰 학습이나 추가 seed 탐색을 이어가지 않는다.

## 판정과 한계

주 비교: pretrained-frozen probe − random-frozen probe의 개발 **ADE(m), 10–60s**.
팀→시각 평균 후 경기 평균을 사용한다. 같은 경기의 paired 차이를 공개한다.
보조: FDE(60s), 관측 가능한 팀원 pair 거리 MAE, P1/P2/P3 학습곡선·support,
map/팀 크기/인간·봇/이동 모드 분포. 정답 없는 metric은 null, 0으로 채우지 않는다.
시드 1개의 feasibility 실행이므로 일반적 우월성·비열등·유의성 주장을 하지 않는다.
경기 bootstrap 구간을 보일 경우 단일 seed·선택 코호트에 조건부임을 명시한다.
학습 loss 감소와 새 패치 성능 개선은 별개다. 개선이 없으면 그대로 보고한다.
geometry 정확도·승리 정책·시야 복원은 검증하지 않는다.
41.1은 노출 이력이 있는 개발 자료, 보호/41.2는 미사용이다.

ARS 실행 규칙 중 “사용자가 준 스크립트만 실행/수정 금지”는 이번 사용자의 명시적인
구현·검증·학습·자기 피드백 요청 범위에 맞춰 적용한다. 다른 모델 위임 없이 자체 검토하며
독립 리뷰로 표시하지 않는다.
