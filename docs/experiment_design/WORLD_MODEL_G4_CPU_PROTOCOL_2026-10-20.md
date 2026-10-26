# G4 시간 제한 공간 채널 통합과 제한 CPU 비교

사용자 승인: 기존 공간 채널 시간 제한·중복 방지 통합, 누수/gradient 검사,
혼합비 결정 후 소규모 CPU 비교·자기 피드백·시각화. 보호 유지.
설계와 적용 패치 범위를 실행 전 고정한다. G1–G3 실행 및 V3 전처리는 보존한다.

## 직관과 구조

공간 근거는 한 조회기로만 받는다. 과거 학습 경기가 끝난 뒤에야 알 수 있는 이동을
그보다 앞선 경기의 입력으로 주지 않는다. 동일 관측을 old observed와 historical 양쪽에
합산하지 않는다. 미관측 인접 연결은 별도 가설이며 통행 가능성 정답은 아니다.

패치: 새 `world_model/temporal_prior.py`의 Dataset/입력/공간 블록,
기존 WorldEncoder에 공간 부분을 재정의하는 작은 hook, G4 시험·실행·보고 도구.
기존 frozen preprocess/dataset/patches 본문은 변경하지 않는다.
기존 어댑터의 전체 fit spatial 조회와 자기 경기 빼기 경로를 우회한다.
빈 공간 지원에서 현재 crop만 만든 후 새 past-only 쿼리로 지원·가설을 재구성한다.
cell feature의 지원 누적값도 0으로 두어 같은 관측량을 edge와 node에서 이중 입력하지 않는다.

근거: G3 설계·64 crop 검사, G1 typed encoder (HGT-inspired; Hu et al.,
arXiv:2003.01332; 재현 구현 아님), 기존 G2 pretraining heads.
새 문헌 주장 없이 승인된 G3 인터페이스를 최소 구현한다.
시간 비용 O(crop 조회행 + E*d), CPU 1 thread, private memory 48GiB 상한.
유한 손실/gradient, 역할·시간·지도 차단, 동일 seed 재현 실패 시 실행 중단.

## 자료와 비교 고정

- 허용 일반+경쟁 BR의 40.2 train / 41.1 development만 사용.
- 결과와 무관한 SHA 순서로 일반 train 32경기×최대4시각, dev 16경기×최대4시각.
  경기 내 team/frame 중복 제외. 경기·팀을 균등 추출하는 제한 자연 표본이며 전체 코퍼스
  빈도 분포의 불편 추정치라고 부르지 않는다.
- 희귀 표본: 긴급호출 first30, 직접재진입 first30, 실제 긴급호출 mode4,
  실제 재진입 mode5 네 층에서 train 각16, dev 각8, 각 층의 경기 중복을 피한다.
  층간 동일 키를 제외한다. 추정 재진입은 이번 작은 직접근거 비교에 포함하지 않는다.
- 실제 fit ID는 위 train 표본의 경기 합집합으로 축소한다.
  prior donor·장비 어휘·조건 어휘도 이 집합만 사용한다. 자연-only arm과 mixed arm은
  이 고정 fit 자원을 공유하며 optimizer sampling의 효과만 비교한다.
  자연-only도 동일 fit corpus의 공간 사전정보를 가진 대조군임을 명시한다.
- natural-only batch=4 일반. mixed batch=3 일반+1 희귀, 희귀 네 층 순환.
  각 층 안 경기→팀→구간 균등. 비율은 소규모 기전 확인용 설계값이며 최적 비율이 아니다.
- seed 20261020, 20261021, 20261022. 관측자와 서버 별도 학습.
- d=64, heads=4, 사전학습 192 optimizer steps, 동결 표현 probe 96 steps.
  AdamW lr=3e-4, wd=1e-4, clip=1. dropout=0, CPU deterministic, fastpath off.
  G2의 health/motion/mode/pair 목적함수·가중치 유지.
- probe는 두 표현 모두 동일 75:25 schedule로 학습해 사전학습 혼합 효과를 분리한다.
  random frozen encoder probe와 등속도 기준도 동일 개발 입력/정답으로 평가한다.
- H0=대상과 같은 source patch만; H1=모든 허용 과거 patch.
  모든 arm에서 donor 종료 < 대상 시작, donor != target, actual fit만 유지.
  40.2 train의 H0/H1은 같으므로 재학습 차이로 주장하지 않는다.
  동일 checkpoint를 H0/H1로 평가하는 정보 가용성 비교다.
- 이전 G2와는 모집단·모델 폭·표본수가 달라 수치 개선을 직접 주장하지 않는다.
- 단계별 1200초/seed-view 상한. 실패 시 기록 후 중단, 자동 예산 확대 금지.

## 지표·판정

주 비교: observer H1 mixed-pretrained 대 natural-pretrained,
일반 개발 표본의 30초 team-joint ADE(m), 현재 팀원을 고정하고 유효 정답만 집계.
표본→팀→경기 평균 후 seed paired difference, 같은 경기 bootstrap 2000회 95% CI.
CI가 0을 포함하면 미결. CI 상한<0이고 세 seed 차이가 모두 음수일 때만
이 제한 표본에서 개선이라고 부른다. 비열등/전체 코퍼스 우월성 주장은 하지 않는다.
보조: first30 pair-distance MAE, 지원된 60초 ADE/FDE, 희귀 네 층,
H0/H1 정보 효과, server/observer, motion mode4/5 정확도.
단일 주 비교만 판정하며 보조 결과로 주 결론을 교체하지 않는다.
짧은 수송은 30초 완전 정답을 요구하지 않고 유효 horizon으로 별도 평가한다.
가장 잘 나온 그림을 고르지 않고 고정 첫 개발 표본들을 시각화한다.
한 seed의 전체 실행을 새 프로세스에서 반복해 state/prediction tensor hash 비교.

## 검산

donor end=10, target start=50인 7회 관측과 end=100인 99회 관측이면 지원은 7.
edge support는 한 채널에만 7, cell support 0, observed legacy는 빈 배열.
batch 4에서 일반3+희귀1을 192회면 일반576+희귀192 노출, 희귀 각층48회.
이 수치를 시험으로 확인하고 실제 sampler 로그와 비교한다.
gate는 sigmoid 출처·연령·변경 특징이며 보정된 통행 확률이 아니다.
