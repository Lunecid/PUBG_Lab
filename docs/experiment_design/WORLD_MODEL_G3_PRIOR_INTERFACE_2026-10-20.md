# G3 historical prior 입력 인터페이스와 학습 전 검증

## 직관과 적용 상태

예전 패치에서 여러 사람이 이동했던 연결은 새 패치에서도 참고할 수 있다.
하지만 연결의 과거 존재가 현재의 통행 가능성·높이·충돌을 보증하지는 않는다.
따라서 기존 현재 관측 연결에 합산하지 않고 `historical_behavior` 관계로 전달한다.
현재 구현은 출처가 보존된 저장소와 누수 방지 조회까지이며 인코더 구조 변경·가중치 학습은 미실행이다.

## 자료 경계

target = `(match_id, map, patch, match_start, current_crop)`.
`current_crop`은 t0에서 관측 가능한 자기 팀 위치의 100m 셀 주변이다.
행 수·crop을 정하는 데 미래 사건 위치나 실제 미래 경로를 쓰지 않는다.

donor의 필수 조건:

1. 실제 fit ID 집합에 포함되고 일반·경쟁 BR에 해당.
2. target과 동일 map key.
3. donor 종료 시각 < target 시작 시각.
4. donor ID ≠ target ID.
5. 공간 연결 모드 1·2·3만 허용. 수상/육상/비차량 관계 구분 유지.

fit 전체 SQLite에 행이 있어도 inner-fold와 ablation은 `actual_fit_ids`로 다시 제한한다.
모델 어휘도 `CohortDataset`에서 해당 manifest의 train ID만 사용한다.
보호·보류·validation의 상세 기록은 prior fit에 읽어 들이지 않는다.
학습 중도 학습 경기 이전에 종료된 donor만 조회하는 temporal prefix 방식이다.

## 텐서 계약 — 제안

| 필드 | 모양 | 의미 |
|---|---|---|
| historical_edge_index | [2, E_hist] | 현재 crop의 source/destination cell index |
| historical_mode | [E_hist] | 비차량/육상/수상 relation type |
| historical_numeric | [E_hist, 5] | log1p 관측수, log1p 기여 경기수, 경과 일수/30, known_change, geometry_verified(현재 0) |
| source_patch / target_patch | [E_hist] / [1] | fit 어휘; 새 patch는 UNK 처리 |
| provenance | 별도 manifest | donor ID 목록/정책/해시. learned identity embedding으로 사용하지 않음 |

`historical_edge_index`는 기존 `observed`와 `inferred`와 별도다.
같은 셀 쌍에 여러 이동 모드가 있으면 mode별 행을 유지한다.
0개 연결은 관측 지원이 없음을 뜻하며 불통 label로 쓰지 않는다.

제안 연산:

`m_hist(v) = Σ_(u,v) gate(e, current_state) · W_mode · h_u`

`h'_v = h_v + λ · LayerNorm(m_hist(v))`

gate는 출처·연령·지원 규모를 사용하는 제한된 attention이다.
gate 값은 이동 성공 확률이나 지형 신뢰도의 보정된 추정치가 아니다.
λ와 gate 구조는 후속 비교 전에 고정한다. 현 구현에 임의 신뢰도 확률을 넣지 않았다.
추론 때 prior로 경로를 강제로 자르거나 최단경로 정답을 만들지 않는다.
부품 요구: 새 패치의 prior 지원 확보; 근거: G2 개발 observed edge 0,
공식 41.1 지형 변화; 비용: crop 안 historical edge 수에 비례.
반증: 출처 누수, 학습 비용 과다, 기본 표현보다 일관된 악화면 사용 범위를 축소한다.

## 새 패치에서 학습하지 못한 정보

40.2만 학습하면 41.1의 변화 flag와 patch token은 새 조건이다.
따라서 flag를 주었다고 변화 대응을 학습했다고 주장할 수 없다.
시간 prefix는 시간 노후화의 대체 실험일 뿐 실제 지형 변경의 재현이 아니다.
공식 변경이 있는 구간에서 geometry hard constraint는 계속 금지한다.
known_change=0도 지형 불변을 뜻하지 않는다. 이후 여러 과거 패치가 확보되면
실제 이전→다음 패치 쌍에서 이 부품을 학습·검증해야 한다.

## 후속 비교를 위한 고정 후보

- H0: historical 채널 off. exact-patch prior에도 donor 종료 < 대상 시작 제한을 적용.
- H1: historical 채널 on, 출처·연령·known_change 사용. 동일 donor 관측을 기존 observed 채널에 중복 합산하지 않는다.
- H2: 동일 연결, 연령·변화 feature 제거. 출처/시간 제한은 그대로.
- H3: 과거 연결의 공간 배치를 섞는 대조군. 실제 효과가 support 양 때문인지 구분.
- 각 조건은 동일 primary fit/개발 경기, 동일 자연 분포 평가, 동일 seed 묶음.
- rare challenge는 자연 분포 결과와 따로 보고. validation을 prior fit이나 confidence fit에 사용하지 않는다.
- 이 문서는 실행 전 설계이며 H0–H3 학습을 실행하지 않았다.

### 통합 시 지켜야 할 추가 조건 — 자체 피드백

현재 `CohortDataset`은 기존 `IndexedWorldDataset`의 exact-patch support 경로를 재사용한다.
이 경로는 실제 fit subset과 자기 경기 제외를 지키지만, 같은 패치의 다른 fit 경기 종료 시각을
대상 경기 시작으로 제한하지 않는다. 따라서 현재 어댑터 검사 결과를 전체 파이프라인의
엄격한 chronological-prefix 검증으로 해석하면 안 된다. validation/protected를 prior fit에
사용했다는 의미는 아니며, 기존 offline fit 방식과 이번에 제안한 더 엄격한 시간 경계의 차이다.

후속 통합은 새 조회기의 출력으로 공간 관측 지원 경로를 교체한다.
H0에서는 그 출력 중 source patch == target patch만 사용한다.
H1에서는 허용된 모든 과거 source patch를 historical 채널로 전달하고 같은 근거의 기존
observed 채널 합산은 끈다. 기존의 관측 없는 이웃 연결은 inferred 채널로 별도 유지한다.
이렇게 해야 40.2 학습에서도 historical 메시지를 학습할 수 있고,
41.1에서는 과거 40.2 지원을 받으며, 같은 근거가 두 번 들어가지 않는다.
현재 결과는 저장소·조회기·입력 자료 검증까지이며 이 채널 교체를 실행했다고 주장하지 않는다.
통합 후 과거 donor 변조/미래 donor 변조, 같은 관계 중복, H0/H1 입력 차이와
그래디언트 연결 시험을 통과해야 다음 학습을 시작한다.

## 희귀 표본의 두 층

주 집합: 사건 episode를 유지하면서 최초 10/20/30초 정답이 연속 유효한 대표 구간.
추가 집합: 동일 episode 안에 해당 수송 mode의 미래 관측점이 하나 이상인 짧은 구간.
두 번째 집합은 30초 조건을 통과했다고 표시하지 않으며 유효 horizon mask를 그대로 저장한다.
최대 수송 mode 관측점, 30초 지원, 이른 시각 순으로 고른다.
입력에는 선택 이유·미래 사건·미래 mask를 넣지 않는다.
후속 학습에서 자연 분포와 희귀 집합의 혼합비를 먼저 정하고,
증강된 분포의 NLL을 자연 분포 확률 보정의 근거로 삼지 않는다.
동일 (match,frame,actor)는 중복 제거하고 관련 event ID들은 함께 보존한다.
