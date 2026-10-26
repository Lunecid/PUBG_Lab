# G4 검산 — 시간 경계, 중복 회계, 혼합비

실험 프로토콜의 식은 `n(e,target)=Σ_d n(d,e) 1[d∈actual_fit, d≠target,
end(d)<start(target), map(d)=map(target)]`이다.
종료 시각이 같아도 아직 알려진 자료로 취급하지 않기 위해 엄격한 부등호를 쓴다.
40.2→41.1일 때 H0는 source patch 일치까지 요구하며 H1은 과거 patch를 허용한다.

작은 사례: target start=50, donor A end=10/count=7,
donor B end=100/count=99이면 edge 지원은 7이다.
`log1p(7)=2.0794415416798357`.
old observed는 빈 배열, cell의 지원 feature는 0이다.
따라서 과거 관측 7이 old edge·cell·historical에 반복 전달되지 않는다.
공간 블록의 가중치 공유 반복은 메시지 전달 단계이며 원천 관측의 중복 집계와는 다르다.

batch=4, steps=192에서 mixed는 일반576·희귀192 노출이다.
희귀 네 층을 순환하면 각48회다. 이는 고유 표본 수가 아니라 optimizer 노출 수다.
데이터가 적은 층도 같은 노출을 받으므로 자연 분포의 유병률/발생률을 추정하는 학습 분포는 아니다.

직접 해보기: end=50인 donor count=11을 추가하면?
답: `50<50`이 거짓이므로 여전히 7.
일반-only는 192×4=768회 일반 노출이다. 두 arm의 총 sample exposure는 같다.

숫자는 `reports/world_model_g4_2026-10-20/math_check.json`으로 NumPy 검산했고
시간·중복은 `test_g4.py`, strict boundary는 `test_g3.py`,
혼합비는 `test_locked_mixture_schedule`에서 시험한다.

추가 반례: 학습 feature x=0이면 선형 계수 w에 대한 gradient는
`dL/dw=(dL/dy)*x=0`이다. 따라서 개발에서 x=1일 때 임의 초기값 w가 그대로 작용한다.
G4 v2는 train에서 항상0인 changed/verified 두 열의 초기 계수를 0으로 고정해 시작한다.
AdamW decay도 `0*(1-lr*wd)=0`이므로 학습되지 않은 임의 효과가 생기지 않는다.
이 feature의 효과를 학습했다는 뜻은 아니다. flag 변조에도 출력이 같음을 시험한다.
