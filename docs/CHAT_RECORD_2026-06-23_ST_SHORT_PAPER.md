# 2026-06-23 Conversation Record: ST Short Paper

## Context

- Project: `/Users/lunecid/project/PUBG`
- Paper folder: `paper/dsacg/st_representation_short`
- Overleaf project: `https://www.overleaf.com/project/6a278b8a1b180286c4967710`
- Main task: DSACG full-paper 방향 중 일부로, PUBG squad telemetry에서 ST representation이 battle royale genre의 생존/순위 예측 문제를 어떻게 정의하고 설명하는지 보여주는 short paper 작성.
- Current submission constraint: 과제 제출 마감이 가까우므로 새 실험을 추가하지 않고, 현재 실험의 주장 범위를 정확히 낮추는 방향으로 수정.

## Core Paper Position

오늘 정리된 최종 포지셔닝은 다음과 같다.

> 본 논문은 공개 PUBG telemetry 전체 feature를 모두 사용한 최종 성능 비교가 아니라, ST 관점에서 추출한 13개 snapshot feature와 관계 입력을 분해하여 어떤 시공간 구성요소가 먼저 작동하는지 확인한 preliminary ST feature block ablation이다.

따라서 현재 논문은 다음을 주장하지 않는다.

- PUBG 공개 telemetry 전체 feature보다 STDB 표현이 더 좋다고 주장하지 않는다.
- `v_team_phase_feature_v2`의 187개 전체 feature를 모두 사용했다고 쓰지 않는다.
- 원본 JSON telemetry를 그대로 baseline으로 넣었다고 쓰지 않는다.
- graph structure가 고유하게 효과 있었다고 주장하지 않는다.
- safe zone feature가 현재 실험에서 긍정적 개선을 만들었다고 주장하지 않는다.

현재 논문이 말할 수 있는 것은 다음이다.

- PUBG telemetry를 final-rank row가 아니라 phase-level survival hazard 문제로 재구성할 수 있다.
- 현재 snapshot ST representation 내부에서는 직접 생존 상태와 이동 이력이 먼저 안정적인 신호로 나타났다.
- 생존 상태가 강한 것은 전투 생존 게임에서 자연스러운 직접 상태 변수의 효과다.
- 이동 이력 조건은 좌표/속도만으로 next-phase elimination을 예측한 조건이며, 최신 상태만 사용하는 압축 조건보다 낮은 손실을 보였다.
- safe zone과 graph relation은 단순 추가 방식으로는 충분히 분리되지 않았고, 후속 표현 설계가 필요하다.

## Terminology Decisions

### Battle Royale Wording

- `Battle Royal`은 잘못된 표기.
- 장르 일반명은 `battle royale genre`, 한국어 논문에서는 `배틀로얄 장르`, `배틀로얄 게임`.
- `Battle Royale`을 독립 주어처럼 쓰지 않는다.
- 안전한 주어:
  - `PUBG: Battlegrounds와 같은 배틀로얄 장르의 게임`
  - `PUBG 스쿼드 매치`
  - `배틀로얄 장르의 게임에서`

### Baseline Wording

현재 원고에서 baseline이라는 단어를 조심해서 써야 한다.

- `latest-state baseline`: 전체 연구 baseline이 아니라 `시간 이력 제거 조건`.
- `no-graph base`: official telemetry baseline이 아니라 `ST snapshot 기본 조건`.
- 아직 없는 baseline: `official telemetry-derived full baseline`.

정확한 설명:

> 본 실험은 official telemetry-derived full baseline과 STDB 전체 표현을 비교한 것이 아니라, ST 관점에서 구성한 snapshot feature block 내부의 ablation이다.

### Raw Telemetry Wording

우리 DB는 원본 telemetry JSON 자체가 아니다.

층위:

1. PUBG 공개 원본 telemetry: API가 제공하는 JSON event log.
2. `pubg.*` raw layer: 원본 telemetry를 event type별 관계형 테이블로 flatten한 저장 layer.
3. official telemetry-derived baseline: `pubg.*` raw layer에서 phase cut 이전 field만 집계한 baseline. 현재 short paper에는 아직 없음.
4. `stdb.*` STDB layer: moving object, dynamic region, spatial event, dynamic relation, risk set 등으로 재구성한 layer.
5. current snapshot/ST ablation: `stdb.team_snapshot_features`의 13개 model-facing \(X_s\) feature와 \(A_s\) relation을 사용한 실험.

## Current Experiment Interpretation

### What Was Actually Ablated

현재 실험은 전체 DB feature를 사용한 것이 아니다.

사용한 snapshot node feature \(X_s\)는 13개:

- Movement:
  - `pos_x`
  - `pos_y`
  - `pos_z`
  - `speed_mps`
- Zone context:
  - `rel_zone_x`
  - `rel_zone_y`
  - `dist_blue_boundary_m`
  - `inside_blue`
  - `dist_target_center_m`
- Survival state:
  - `mean_health`
  - `min_health`
  - `alive_n`
- Mobility resource:
  - `vehicle_ratio`

관계 입력:

- Primitive relation tensor \(A_s\):
  - `proximity`
  - `combat`
- Latest relative-edge follow-up \(E_s\):
  - contact gate
  - proximity/combat weights
  - \(X_s,A_s\)에서 파생한 상대 상태 차이

### Meaning of Each Condition

- 팀 생존 여력:
  - `mean_health`, `min_health`, `alive_n`, `vehicle_ratio`
  - 전투 생존 게임에서 다음 phase를 버틸 수 있는지 직접 나타내는 상태 변수.
  - 가장 좋은 성능은 자연스럽고, 새로운 발견처럼 과장하면 안 됨.

- 이동 이력:
  - `pos_x`, `pos_y`, `pos_z`, `speed_mps`
  - 체력, 생존 인원, 차량, 안전구역, graph relation을 제외하고, 팀 중심 좌표와 속도의 10초 이력만으로 다음 phase 탈락을 예측한 조건.
  - 이 조건이 최신 상태 기준보다 낮은 손실을 보였다는 점이 ST representation 관점에서 중요.

- 안전구역 맥락:
  - safe/blue zone 상대 위치와 경계 거리 중심.
  - 단독 조건도 약했고, 이동 이력과의 단순 결합도 이동 이력만 사용한 조건보다 개선되지 않음.
  - 해석: safe zone이 중요하지 않다는 뜻이 아니라 현재 feature construction이 zone pressure를 충분히 사용하지 못한 것.

- 기본 표현(no graph):
  - 13개 \(X_s\) 전체의 10초 이력.
  - official telemetry baseline이 아님.
  - ST snapshot 기본 조건.

- graph 조건:
  - graph만으로 예측한 것이 아님.
  - 13개 \(X_s\) 전체 이력 위에 \(A_s\) relation을 결합한 조건.
  - observed \(A_s\) graph, edge-shuffle control, contact-gate control 손실이 거의 같으므로 observed topology-specific signal은 아직 분리되지 않음.

## Main Text Changes Made Today

### Methods

Methods에서 verbose한 feature 설명을 줄이고 표 중심으로 수정.

추가/정리된 표:

1. Raw telemetry에서 구성한 모델 입력 feature block
2. Feature block을 분해·조합한 학습 조건

두 번째 표에서 `구현 alias` 열은 제거했다. 논문 독자에게 `tcn_trajectory`, `rgcn_full` 같은 코드 alias는 불필요하고 노이즈가 된다고 판단했다.

현재 표 구조:

- 학습 조건
- 사용한 \(X_s\) 조합
- 관계 입력

### Abstract

초록에 다음 해석을 반영했다.

- 팀 생존 여력이 가장 강한 것은 전투 생존 게임에서 직접 상태 변수가 강하게 작동한 결과.
- 더 중요한 비교는 직접 생존 상태를 제외한 조건에서도 이동 이력이 의미를 갖는가.
- 이동 이력 조건은 팀 중심 좌표와 속도의 10초 이력만으로 다음 phase 탈락을 예측한 조건.

### Results

결과 섹션에서 다음을 명시했다.

- 팀 생존 여력의 성능은 주된 기여로 해석하지 않는다.
- 평균 체력, 최소 체력, 생존 인원, 차량 이용 여부는 다음 phase 생존 가능성을 직접 나타낸다.
- 이동 이력 조건은 survival state, zone, graph를 제외하고 좌표와 속도만 사용한다.
- 따라서 이동 이력이 latest-state compression보다 나았다는 점이 ST representation 관점에서 더 중요하다.

### Discussion and Conclusion

논의/결론에서 다음을 명시했다.

- 직접 생존 상태 변수가 강한 것은 자연스럽다.
- 논문의 시공간 표현 질문은 그 다음에 온다.
- 현재 실험은 official telemetry-derived full baseline과 STDB 전체 표현을 비교한 것이 아니다.
- 현재 결과는 ST snapshot feature block ablation으로 해석해야 한다.

## Current Overleaf Status

Overleaf에 반영한 주요 파일:

- `00-abstract.tex`
- `01-introduction.tex`
- `02-background.tex`
- `03-methods.tex`
- `04-experiments-results.tex`
- `05-limitations-conclusion.tex`

마지막 확인 상태:

- PDF compile 성공.
- 페이지 수: 7 pages.
- fatal error 없음.
- 남은 warning:
  - ACM template 관련 warning.
  - BibTeX field warning.
  - Underfull vbox warning.
  - Package balance warning.

## Generated Supporting Artifact

공개 telemetry와 로컬 DB feature 전체 목록을 따로 생성했다.

- File: `docs/TELEMETRY_DB_FEATURE_INVENTORY_2026-06-23.md`
- 내용:
  - Official PUBG telemetry event schemas: 53 event types.
  - Official telemetry object schemas: 13 object types.
  - Local DB schemas `pubg.*` and `stdb.*`: 52 tables/views, 1,172 columns.
  - Current short-paper model-facing snapshot feature list.

공식 문서 출처:

- https://documentation.pubg.com/en/telemetry.html
- https://documentation.pubg.com/en/telemetry-events.html
- https://documentation.pubg.com/en/telemetry-objects.html

## Important Warnings for Future Writing

마감 전 추가 실험 없이 제출한다면, 다음 표현은 피해야 한다.

- "STDB representation outperforms official telemetry baseline"
- "graph topology is effective"
- "safe zone improves trajectory modeling"
- "we use all public telemetry features"
- "raw telemetry was directly used as model input"

대신 다음처럼 써야 한다.

- "본 실험은 ST 관점에서 구성한 snapshot feature block ablation이다."
- "팀 생존 여력은 직접 상태 변수라 강하게 작동한 것이 자연스럽다."
- "이동 이력 조건은 좌표와 속도만으로 next-phase elimination을 예측한 조건이다."
- "이동 이력이 최신 상태 기준보다 나은 점은 ST 이력 보존의 필요성을 보여준다."
- "safe zone과 graph relation은 단순 추가 방식으로는 충분히 분리되지 않았다."

## Recommended Final Submission Framing

제출용 최종 프레임:

> PUBG 스쿼드 매치 telemetry를 시공간 데이터베이스 표현으로 재구성하고, ST 관점에서 추출한 snapshot feature block이 다음 phase 탈락 예측에서 어떤 역할을 하는지 예비적으로 비교했다. 가장 강한 신호는 직접 생존 상태였지만 이는 전투 생존 게임에서 자연스러운 결과다. 더 중요한 점은 생존 상태를 제외한 이동 이력 조건이 최신 상태만 사용하는 압축 기준보다 낮은 손실을 보였다는 점이다. 반면 safe zone과 graph relation은 단순 feature 추가만으로는 충분히 활용되지 않았고, 후속 연구에서 interaction-based ST representation으로 재설계해야 한다.

## Late Revision: Remove Graph Results

마지막 수정에서 graph 결과는 본문에서 제거했다. 이유는 이 short paper의 핵심을
다음 문장으로 좁히기 위해서다.

> 생존 예측에서 중요한 것은 마지막 순간의 상태 하나만 보는 것이 아니라, 그 상태에
> 도달하기까지의 10초 간격 이동 이력을 보존하는 것이다.

수정 내용:

- Abstract에서 graph result 문장 제거.
- RQ3을 graph relation 질문이 아니라, 직접 생존 상태, 이동 이력, 안전구역 맥락의
  feature block ablation 질문으로 변경.
- Related Work에서 GNN/graph relation subsection 제거.
- Methods에서 \(A_s\), \(E_s\), edge-shuffle, contact-gate control 설명 제거.
- Results에서 graph relation table 및 observed/control 비교 해석 제거.
- Discussion/Conclusion에서 graph-specific signal, observed topology 해석 제거.
- Overleaf 재컴파일 결과: 6 pages, fatal error 없음.

최신 상태 기준과 이동 이력 조건의 정확한 의미:

- 최신 상태 기준:
  - 각 phase cut에서 마지막으로 관측된 snapshot만 사용한다.
  - 코드상 `SnapshotLastOnlyHazard`가 `gather_phase_cut_embeddings`를 통해 cut phase별
    마지막 snapshot embedding을 가져와 다음 phase hazard를 예측한다.
- 이동 이력:
  - `pos_x`, `pos_y`, `pos_z`, `speed_mps`만 사용한다.
  - 체력, 생존 인원, 차량, 안전구역 변수는 제외된다.
  - 10초 snapshot을 phase 단위 평균으로 먼저 뭉뚱그리는 것이 아니라, 10초 sequence에
    causal temporal convolution을 적용한 뒤 phase cut 위치의 embedding으로 다음 phase
    탈락 위험을 예측한다.
  - 따라서 “phase 내 평균 feature”가 아니라 “phase cut까지의 10초 이동 이력 보존”
    조건으로 설명해야 한다.

현재 제출본에서 graph는 DSACG full paper 또는 후속 연구 항목으로 남기는 것이 맞다.

## Late Revision: Explicit Phase-Cut Prediction Example

사용자가 다시 확인한 핵심 오해 가능성:

- "페이즈 1과 2의 10초 snapshot을 모두 뭉쳐서 페이즈 2 생존을 예측하는가?"

정확한 답:

- 아니다. 예측 시점은 항상 한 페이즈 앞선다.
- 페이즈 \(k\) 종료 시점까지 생존한 팀에 대해, 페이즈 \(k\) 종료 이전에 관측된
  10초 snapshot 이력만 사용하여 페이즈 \(k+1\)에서 탈락할지를 예측한다.
- 예: 페이즈 1 종료 이전 이력으로 페이즈 2 탈락 여부를 예측한다.
- 페이즈 2 안에서 발생한 이동, 피해, 탈락 사건은 페이즈 2 예측 입력에 들어가지
  않는다. 이것이 leakage를 막는 핵심 설정이다.
- 이동 이력 조건은 10초 snapshot을 평균이나 합계로 먼저 압축하지 않는다. 좌표와
  속도 sequence를 causal temporal convolution으로 인코딩한 뒤 phase cut 위치의
  표현으로 다음 페이즈 탈락 위험을 예측한다.

원고 반영:

- `sections/03-methods.tex`: 페이즈 1 종료 이전 이력으로 페이즈 2 탈락 여부를
  예측한다는 예시 문단 추가.
- `sections/03-methods.tex`: feature condition table의 시간 처리 열을 `phase cut 이전
  10초 스냅샷 시퀀스`로 구체화.
- `sections/04-results.tex`: 이동 이력 결과가 같은 페이즈의 사후 정보를 사용한 것이
  아니라, 페이즈 \(k\) 종료 이전 이력으로 페이즈 \(k+1\) 탈락 여부를 예측한 결과임을
  명시.
- Overleaf 반영 및 재컴파일 완료: output.pdf 6 pages, fatal error 없음.

## Late Revision: Metric Definition

사용자가 지적한 문제:

- 결과 표에서 "손실"만 제시하면, 이것이 accuracy의 반대인지, AUC 대신 무엇을
  계산한 것인지, 어떤 단위로 평균낸 것인지 불명확하다.

정리한 정의:

- 모델은 각 유효한 팀-페이즈 셀 \((i,k)\)에 대해 다음 페이즈 탈락 확률
  \(\hat{p}_{i,k}=\sigma(z_{i,k})\)를 출력한다.
- \(y_{i,k}=1\)이면 팀 \(i\)가 페이즈 \(k+1\)에서 탈락한 경우이고, 0이면 그렇지 않은
  경우다.
- \(m_{i,k}=1\)인 셀, 즉 페이즈 \(k\) 종료 시점까지 생존해 다음 페이즈 라벨이 정의된
  위험집합 셀만 평가에 포함한다.
- 테스트 손실은 masked binary cross-entropy 평균이다.
- 낮은 손실은 모델이 실제 탈락 팀에는 높은 탈락 확률을, 생존 팀에는 낮은 탈락
  확률을 부여했다는 뜻이다. 이것은 accuracy의 반대말이 아니라 확률 예측의 평균
  로그 손실이다.

원고 반영:

- `sections/03-methods.tex`: BCE 식과 mask 정의 추가.
- `sections/04-results.tex`: 표의 평균 손실이 9개 맵/플레이 모드 조건별 테스트 BCE의
  평균이고, 표준편차가 조건 간 변동임을 명시.
- Overleaf 반영 및 재컴파일 완료: output.pdf 6 pages, fatal error 없음.
