# G1 연산 검산

이 문서는 학습 결과가 아니라 입력·연산 계약의 작은 사례다.

1. **공간 자기 경기 제외**: 동일한 방향·모드의 연결을 현재 경기 A가 3번,
   다른 fit 경기 B가 99번 제공하면 전체 관측 수는 102, 지원 경기 수는 2다.
   A의 입력에서는 `(102−3, 2−1)=(99,1)`만 사용한다.
   A의 관측 수를 999999로 바꾸어도 A 입력의 값은 `(99,1)`이다.
   B가 fit 밖이면 잔여 관측은 0이며, 해당 연결을 실제 통행 정답으로 유지하지 않는다.
   `test_own_match_future_spatial_support_is_excluded`가 이 사례를 실행한다.

2. **팀 구성과 미래 정답**: t0 유효 구성원이 5명이면 encoder target index는 5개다.
   미래 mask를 모두 False로 바꾸어도 그 5개 index는 유지된다.
   `joint_complete_60s`는 정답 쪽 코호트 변수이고 입력 변수가 아니다.

3. **팀 순서**: 개인 집합의 순열 Π에 대해 개인 연산은 `f(ΠX)=Πf(X)`,
   팀 mean 및 상대 벡터 집합 집계는 `g(ΠX)=g(X)`를 만족하도록 공유 가중치를 쓴다.
   부동소수점 합산 순서 차이는 허용 오차 안에서 판단한다.
   상대 벡터는 ordered pair를 사용하여 구성원 ID 순서에 따라 부호를 고르지 않는다.
   singleton에서는 pair 집계가 없고 `has_pair=0`이므로 거리 0과 구별된다.

4. **관측 단절**: history index 6이 결측, 7부터 다시 유효라면 현재 summary의
   history valid는 0…6에서 False다. 7의 속도는 이전 위치와 잇지 않는다.
   30초 rolling 피해/사용 특징도 7…9에서 가리고 index 10부터 허용한다.
   숨긴 bank episode ID를 사용하지 않고 공개된 마스크·체력·허용 episode age로만 판단한다.

5. **안전구역 상대값**: 좌표 단위가 100m일 때 player=(0,0), zone center=(3,4),
   radius=6이면 중심거리 5, signed boundary distance=−1이다.
   이는 원 안쪽 100m를 의미한다. 이 값을 입력으로 넣지만 원 중심으로 이동하라는
   정답/손실은 추가하지 않는다. zone 변형 시험은 배선 민감도만 확인한다.

구현과 시험: `world_model/inputs.py`, `world_model/encoder.py`,
`tests/world_model/test_encoder.py`. CPU 최종 증거: `reports/world_model_g1_2026-10-20/`.
