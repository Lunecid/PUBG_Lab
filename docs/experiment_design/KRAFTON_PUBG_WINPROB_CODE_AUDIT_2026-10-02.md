# KRAFTON PUBG 승리확률 공개 코드 분석

분석일: 2026-10-02 KST  
문서 성격: 공개 구현의 근거 검증과 후속 실험 설계 자료  
대상: `krafton-ai/pubg-winprob`  
고정 커밋: `99ed9d1fa5f1df0cb242bb42ed8a511e6c972896`  
커밋 시각: 2026-08-26 06:34:26 UTC, 15:34:26 KST

이 저장소는 현재 경기 상태에서 팀별 우승 가능성을 추정하는 직접적인 비교 대상이다. 팀별 누적 행동 통계와 현재 위치·장비·자기장 상태를 한 시점에 모아 팀 간 self-attention을 수행한다. 시간 순서를 보존한 과거 궤적을 backbone에 입력하지는 않는다. 그러나 누적 행동과 주변 적 수가 포함되므로 과거 정보나 관계 정보가 전혀 없는 모델로 설명해서도 안 된다.

공개 구현을 후속 실험의 기준 모델로 사용하려면 먼저 점수 방향, 학습·추론의 입력 스키마, attention mask, 생존 지표의 의미를 정리해야 한다. 합성 입력으로 점수 방향 불일치와 입력 차원 오류 등을 재현했다. 이 결과는 공개 코드의 동작에 대한 증거이며, 원 논문의 성능이나 산업 시스템의 성능을 판정하는 결과는 아니다. 원본 데이터·분할·학습 가중치가 없기 때문이다.

근거 수준은 **코드 확인**, **합성 실행 확인**, **설계를 위한 해석**, **미확인**으로 구분한다. AI 도구를 이용해 소스를 분석하고 검증 사례를 작성했다.

## 분석 대상과 재현 범위

공식 저장소는 “Real-Time Win Probability Prediction in Battle Royale Games via Survival Analysis”의 구현이라고 소개한다. [저장소](https://github.com/krafton-ai/pubg-winprob), [고정 커밋](https://github.com/krafton-ai/pubg-winprob/commit/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896)

GitHub API로 당시 `main`의 커밋을 확인하고 해당 커밋의 ZIP을 내려받았다. 검색 결과에는 이전 README의 50개 시점·72개 feature 설명도 노출됐지만, 분석에는 현재 커밋의 소스를 사용했다. ZIP에는 45개 파일이 있으며, 주요 dataset·backbone·head·loss·training·evaluation·calibration·inference 경로와 데이터 생성의 관련 부분을 확인했다. 6개 notebook에는 저장된 실행 결과가 없었다. [현재 README](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/README.md)

| 항목 | 확인 결과 | 해석 범위 |
| --- | --- | --- |
| 라이선스 | Apache 2.0 | 출처와 라이선스를 보존하며 코드 재사용 가능 |
| 학습·평가 코드 | 공개 | 자체 데이터에서 구현을 재현·이식할 수 있음 |
| 원본 경기 데이터 | ZIP에 없음 | 원 논문의 데이터 규모·분포를 재현할 수 없음 |
| 실제 분할 목록 | ZIP에 없음 | 경기 중복, 시간 분할, 팀 중복 여부를 확인할 수 없음 |
| pretrained checkpoint | ZIP에 없음 | 원 모델 성능을 직접 검증할 수 없음 |
| 원 논문 본문·DOI·서지정보 | README/ZIP에서 확인되지 않음 | 저장소를 정식 출판 논문으로 단정하지 않음 |
| 정식 논문 검색 | 제목 검색 및 arXiv/KRAFTON 공식 도메인 검색에서 해당 서지 기록을 식별하지 못함 | 검색 범위의 한계이며 논문 부재의 증명이 아님 |

조회 질의는 `site:github.com KRAFTON pubg-winprob`, 논문 제목 전체, `"pubg-winprob" paper dataset`, `site:arxiv.org "Real-Time Win Probability" "Battle Royale"`, `site:ai.krafton.com "Win Probability"`였다. 검색 스니펫은 위치를 찾는 데만 사용하고 구현 관련 판단은 내려받은 소스로 검증했다.

검증 자료는 프로젝트의 `reports/krafton_pubg_winprob_audit_2026-10-02/`에 저장했다.

| 자료 | 내용 |
| --- | --- |
| `upstream_99ed9d1.zip` | 원본 소스와 LICENSE를 포함한 고정 커밋 ZIP |
| `source_manifest.json` | 커밋·출처·파일별 SHA256 |
| `audit_checks.py` | 원본 ZIP을 임시 폴더에 풀어 합성 검증을 수행하는 코드 |
| `audit_results.json` | 실행 환경과 검증 결과 |

## 데이터에서 예측까지

```text
경기 로그
  → 시점 t까지의 플레이어 행동과 현재 상태 추출
  → 팀 단위 집계와 팀별 CSV 행 생성
  → 같은 (match_id, time_point)의 모든 팀을 한 sample로 묶음
  → 연속 feature MLP + 플레이어 위치 Fourier encoding
    + blue/white zone embedding + map embedding
  → 팀 간 Transformer self-attention 또는 팀별 MLP
  → 선택한 loss에 대응하는 하나의 scalar head
  → 평가·추론의 masked softmax와 temperature scaling
```

데이터 로더의 sample 단위는 경기의 시간열 전체가 아니라 **한 경기의 한 시점**이다. `PGCDataset`은 `(match_id, time_point)`로 묶고, backbone의 입력은 `(batch, squads, features)`다. Dataset V2는 팀 번호를 정렬한다. 시간별 sample은 독립적으로 학습 로더에 들어가며, 시간 상태를 이어받는 recurrent state나 temporal attention은 없다. [Dataset](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/data/dataset.py#L94-L125), [Dataset V2](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/data/dataset_v2.py#L275-L285), [Backbone](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/models/backbone.py#L159-L203)

### 데이터 모집단은 공개 코드만으로 확정할 수 없다

README는 esports 승리확률을 연구 대상으로 소개한다. 그러나 데이터 생성 경로에는 esports post-match JSON 처리뿐 아니라 내부 game-log S3를 이용한 경쟁전 처리도 있다. 경쟁전 script는 PC·AS·squad-FPP·4개 맵을 설정하고, 고랭크 경기 목록으로 필터링하며, 출력 경로를 `train_v2.1`로 지정한다. 목록을 읽지 못하면 필터 없이 진행하는 fallback도 있다. 코드의 날짜와 경로는 예시 설정이므로 원 실험의 최종 데이터 기간으로 단정할 수 없다. [경쟁전 생성 경로](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/data_generation/generate_competitive_dataset/generate_pgc_competitive_dataset.py#L53-L59), [경쟁전 필터 설정](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/data_generation/generate_competitive_dataset/generate_pgc_competitive_dataset.py#L411-L450), [post-match 생성 경로](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/data_generation/generate_postmatch_dataset/generate_pgc_postmatch_dataset.py#L1-L14)

따라서 “KRAFTON은 esports 데이터만 학습했고 우리는 일반 경기만 학습했다”는 차별점은 아직 확정할 수 없다. 확인된 차이는 공개 구현의 입력 구조와 목표 정의다. 우리의 public API telemetry로 이식하려면 필요한 이벤트와 feature의 가용성을 별도로 대조해야 한다.

또한 공개 경로는 사후 경기 로그로 시점별 예측을 구성한다. 경기 중 시점 t의 예측을 계산할 수 있다는 설명과 실제 운영 중 실시간 데이터 피드가 제공된다는 설명은 구별해야 한다. 모든 상대 팀의 상태를 사용하는 입력은 관전자 또는 경기 전체 상태를 사용할 수 있는 환경을 전제한다.

## 입력 정보와 시간 처리

### 실제 연속 입력은 55개이고 기본 V2는 57개다

`CONTINUOUS_FEATURES`의 실제 항목 수를 세면 55개다. 주석에 적힌 각 블록의 수와 검색 스니펫의 수를 그대로 사용하면 틀릴 수 있다. V2는 zone 거리 두 항목을 추가 계산해 뒤에 붙이므로 57차원이 된다. 추가한 두 이름은 이미 기본 목록에도 들어 있다. 따라서 V2는 CSV의 표준화된 두 값과 다시 계산한 비표준화 두 값을 함께 사용한다. 이것은 같은 의미의 두 feature가 서로 다른 처리로 반복되는 구조이며, 실제 값이 완전히 같다는 뜻은 아니다. [Feature 목록](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/data/continuous_features.py#L5-L74), [V2 계산과 결합](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/data/dataset_v2.py#L328-L349)

| 입력 블록 | 담기는 정보 | 비교 실험에서 통제할 내용 |
| --- | --- | --- |
| 누적 행동 | kill·damage·피해·heal·pickup/drop·차량 탑승 | 현재 상태 모델에도 이미 과거 요약이 있음 |
| 팀 상태 | 생존 인원·체력 합계·비다운 인원·zone 밖 인원 | 탈락 위험의 강한 현재 정보 |
| 장비·소모품 | 방어구·가방·투척물·회복제 | 우리 입력에서 누락하면 정보량이 달라짐 |
| 공간 요약 | 주변 아군·적 수, zone 거리 | 단순 snapshot도 관계를 일부 요약함 |
| 플레이어 좌표 | 팀원 최대 4명의 3D 위치 | Fourier 변환 후 생존 플레이어 embedding을 평균 |
| 경기 공통 상태 | blue/white zone 중심·반경, map | 현재 위치와 공간 압박을 함께 제공 |

행동 통계는 플레이어별 누적값을 팀에서 합산한다. 장비 수준은 생존자의 평균, item count는 생존자의 합계, 체력과 생존 인원은 별도 집계한다. 좌표는 팀 중심 하나를 Fourier 변환하는 방식이 아니라 **각 플레이어의 Fourier embedding을 평균**하는 방식이다. [팀별 집계](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/data_generation/feature_engineering/extracting_pgc_features.py#L420-L518), [좌표 pooling](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/models/backbone.py#L122-L157)

**설계 해석:** 이 모델과 비교할 때 temporal model의 feature를 더 풍부하게 주면 시간 순서의 효과와 정보량의 효과를 구별할 수 없다. 우리의 history 효과는 같은 현재 상태와 누적 통계를 제공한 뒤 추가로 검증해야 한다.

### 192개 시점은 시간열 입력 길이가 아니다

생성기는 각 phase의 범위에서 uniform random sampling을 수행한다. 178개 random point와 phase 전환 전후 14개 anchor를 합쳐 192개 시점을 만든다. 약 60초·20초·10초·5초는 평균적인 밀도를 나타내며, 고정 간격으로 생성되는 시간열은 아니다. anchor는 7개 전환에만 배치된다. [Sampling 설정](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/data_generation/feature_engineering/extracting_pgc_features.py#L63-L91), [Sampling 실행](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/data_generation/feature_engineering/extracting_pgc_features.py#L520-L549)

| Phase | 고정 범위 초 | Random 시점 | Anchor 포함 시점 |
| --- | --- | ---: | ---: |
| 1 | 0–600 | 10 | 11 |
| 2 | 600–810 | 11 | 13 |
| 3 | 810–990 | 9 | 11 |
| 4 | 990–1170 | 18 | 20 |
| 5 | 1170–1350 | 18 | 20 |
| 6 | 1350–1530 | 18 | 20 |
| 7 | 1530–1680 | 30 | 32 |
| 8 | 1680–1800 | 24 | 25 |
| 9 | 1800–1970 | 34 | 34 |
| 10 | 1970–2000 | 6 | 6 |
| 전체 | 0–2000 | 178 | 192 |

Phase 구분도 실제 telemetry의 phase event를 읽는 평가 방식이 아니라 고정 시간 경계로 계산된다. `time_point`는 metadata와 평가에 사용되지만, backbone의 forward에 전달되지 않으며 현재 연속 feature 목록에도 포함되지 않는다. phase embedding도 없다. zone와 누적 행동이 시간 진행을 간접적으로 담을 수는 있다. [시간별 phase 함수](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/data/utils.py#L113-L126), [학습 forward](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/trainer.py#L153-L164)

**설계 해석:** 이것은 phase-aware sampling/evaluation이며, phase 구간별 탈락확률을 직접 학습하는 phase-conditioned hazard와 같지 않다. 맵별 phase 길이가 다른 우리의 cohort에 고정 경계를 그대로 복사하면 안 된다.

## 모델과 학습 목표

### 팀 간 attention과 선택형 head

팀별 token은 연속 feature MLP, 좌표 encoding, 두 zone embedding, map embedding을 더해 구성한다. Transformer는 같은 시점의 팀 token 사이를 연결한다. 순서형 squad positional embedding은 없으며, 명시적인 proximity/combat typed edge도 없다. 그래프가 없다는 사실은 관계 학습이 없다는 뜻이 아니다. dense attention은 다른 팀의 현재 상태를 참조할 수 있다. [Backbone 구성](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/models/backbone.py#L59-L117)

`--encoder mlp`는 팀 간 attention 없이 팀별 MLP로 처리하는 ablation이다. 이는 같은 feature에서 팀 간 정보 결합을 비교할 출발점이지만 두 encoder의 계산량과 파라미터 수가 자동으로 같아지는 것은 아니다.

Regression·hazard·classification head를 동시에 학습하는 multitask model은 아니다. `loss_type`에 따라 **하나의 head와 하나의 loss**를 선택한다. [Head 선택](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/models/heads.py#L140-L173), [Trainer에서의 선택](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/trainer.py#L623-L640)

| Loss 선택 | 학습 의미 | 출력 점수의 의미 |
| --- | --- | --- |
| `mse` | 최종 팀 death time 회귀 | 높을수록 오래 생존 |
| `ce`, `survival_ce` | max death time 팀을 winner class로 학습 | 높을수록 우승 쪽 |
| `cox` | 미래 탈락 순서의 risk set과 survivor penalty | 높을수록 탈락 쪽 |
| `rank_cox` | 위 loss에 생존 순위 위반 penalty 추가 | 높을수록 탈락 쪽 |
| `weighted_cox` | 탈락 event와 survivor penalty를 시간 차이로 가중 | 높을수록 탈락 쪽 |
| `concordance` | winner와 다른 팀의 점수 차이에 시간 차이 가중 | classification 계열 점수 |

README의 재현 설정은 embedding 64, heads 8, layers 8, FFN 256, dropout 0.1, batch 512, AdamW learning rate/weight decay 각각 1e-4, 최대 10 epoch, patience 4, seed 42, `weighted_cox`, Fourier frequency 8개다. CLI 기본값은 이 설정과 다르므로 기본 명령만 실행해 원 설정을 재현했다고 주장하면 안 된다. [재현 설정](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/README.md)

### 기본 target과 Cox 계열 loss의 실제 의미

기본 target은 `squad_death_time / 2000`이다. 생성기는 사망 시간이 없는 플레이어를 2000초로 채우고 팀원별 death time의 최대값을 팀 label로 만든다. 기본 neural trainer는 remaining time이나 다음 phase 탈락 label을 사용하지 않는다. 생성기에 다음 sampled point까지의 생존 label이 별도로 있어도 기본 학습 target으로 선택되지는 않는다. [Label 생성](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/data_generation/generate_postmatch_dataset/generate_pgc_postmatch_dataset.py#L174-L192), [Dataset target](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/data/dataset_v2.py#L359-L363), [별도 구간 label](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/data_generation/feature_engineering/extracting_pgc_features.py#L584-L615)

현재 sample에 유효한 팀들 가운데 미래 탈락시간이 T_i인 팀 i의 risk set을 `R_i = {j : T_j >= T_i}`로 만든다. 현재 snapshot에서 얻은 점수 g_j로 각 미래 event에 대해 `q_ij = exp(g_j) / sum_{k in R_i} exp(g_k)`를 계산한다. Loss는 탈락 팀의 `-log(q_ii)`뿐 아니라 다른 팀의 `-log(1-q_ij)`도 더한다. 최장 생존 팀은 탈락 event에서 제외한다. `weighted_cox`는 이 event별 loss에 최장 생존시간과의 차이를 가중한다. 코드상 분모는 스케일된 최장 시간과 1 중 큰 값이다. [Cox loss](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/losses.py#L60-L165), [Weighted loss](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/losses.py#L226-L301)

**설계 해석:** 이것은 현재 snapshot으로 이후 탈락 순서를 학습하는 변형 목적함수다. 일반 Cox partial likelihood만을 사용하는 구현으로 기술하거나, baseline hazard를 추정해 시간별 survival curve를 생성한다고 설명하면 부정확하다. 기본 경로에는 event/censoring column을 별도로 전달하지 않으며, 다음 phase의 절대 탈락확률 head도 없다. 미래 label을 loss에 사용하는 것은 정상적인 supervised learning이며 그 자체는 입력 누수가 아니다.

## 평가와 baseline의 비교 가능성

공개 평가 함수는 우승 accuracy, 경기 내부 팀 쌍의 C-index, `IBS`, top-label ECE, winner log loss를 계산한다. 실제 winner는 `squad_win`을 직접 읽기보다 target death time의 argmax로 정한다. C-index는 target이 다른 팀 쌍을 비교하며 prediction tie에는 0.5를 부여한다. [Metric 구현](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/metrics.py)

`IBS` 구현은 각 snapshot의 살아 있는 팀들에 대한 우승확률과 one-hot winner의 제곱 오차를 합산하고 snapshot 전체에서 평균한다. 함수에 시간 간격이나 survival-at-horizon target이 입력되지 않는다. 따라서 이 지표는 **sampled snapshot별 multiclass winner Brier score의 평균**으로 해석해야 한다. 전통적인 시간별 생존확률의 IBS와 동일하게 읽거나 우리의 hazard Brier와 직접 비교하면 안 된다. 비균일 sampling이므로 전체 평균은 후반 시점에 더 큰 비중을 둔다. [Brier 구현](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/metrics.py#L179-L216), [평가 전체 평균](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/evaluation.py#L239-L276)

Temperature scaling은 validation의 15-bin top-label ECE를 최소화하며, tau 0.1–2.0, 간격 0.01의 탐색과 추가 refinement를 사용한다. 모든 팀별 확률이나 모든 phase의 calibration을 보장하는 검증은 아니다. 학습 직후 test evaluation은 calibration을 적용하기 전 raw score 경로이고, calibration은 별도 명령이다. [Calibration](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/calibration.py#L106-L179), [학습 직후 평가](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/trainer.py#L692-L719)

Baseline은 RandomGuess, RuleBased, LogReg, LogHaz, RFClf, LGBMReg, LGBMClf, CoxPH, DeepSurv의 9개다. 파일 단위 split과 최종 metric 함수는 공유한다. 그러나 정보 표면과 mask는 완전히 같지 않다. 기본 tabular baseline은 55개 scalar feature를 쓰고 일부 classifier/regressor에는 time_point를 추가한다. Transformer는 별도의 player position·zone·map embedding과 V2의 추가 2개 feature를 쓴다. Survival baseline은 `squad_death_time - time_point`와 winner censoring을 사용하며, CoxPH/DeepSurv score는 부호를 반전해 승리 쪽으로 변환한다. Baseline mask는 alive count 기준이고 neural V2 mask는 유효 좌표 기준이다. [Baseline 입력·target](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/baselines.py#L102-L134), [Baseline score 부호](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/baselines.py#L555-L567), [Baseline mask](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/scripts/run_baselines.py#L95-L126)

**설계 해석:** 같은 split·metric은 좋은 출발점이지만 같은 입력·target·risk set을 뜻하지 않는다. 우리 비교에서는 이를 명시적으로 통제해야 한다.

## 재현을 막거나 해석을 바꾸는 구현 문제

아래 결과는 `audit_checks.py`에서 원본 ZIP의 코드를 실행해 확인했다. 코드를 수정해 원 결과를 재평가한 실험은 아니다.

### Cox score의 학습 방향과 평가 방향이 반대다

Loss는 g가 큰 팀을 빨리 탈락하는 팀으로 학습한다. 하지만 평가의 accuracy/C-index는 g가 클수록 오래 살아 우승하는 것으로 읽고, 추론은 `softmax(g/tau)`를 우승확률로 저장한다. 이 경로는 `loss_type`에 따른 부호 변환을 수행하지 않는다. [Loss 방향](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/losses.py#L24-L27), [Metric 방향](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/metrics.py#L21-L60), [추론 변환](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/inference.py#L188-L199)

합성 사례의 target은 `[0.2, 0.5, 1.0]`, hazard score는 `[2, 0, -2]`다. 가장 일찍 탈락하는 팀의 hazard가 가장 높다.

| 확인 항목 | 원 공개 경로 | 부호를 반전한 진단 |
| --- | ---: | ---: |
| Winner accuracy | 0 | 1 |
| C-index | 0 | 1 |
| Winner Brier | 1.7336 | 0.0318 |

이 hazard ordering의 weighted loss는 0.3539이고, 반대 ordering은 7.1539다. Loss가 선호하는 방향을 평가가 뒤집어 읽는다는 증거다. 원 공개 softmax는 가장 먼저 탈락하는 팀에 약 86.68%를 준다.

부호 반전은 순위 방향을 확인하는 진단일 뿐이다. `softmax(-g)`가 Cox 모형에서 이론적으로 도출된 정확한 우승확률이라는 결론은 별도 검증 없이 내릴 수 없다. 양수 temperature scaling은 순위를 뒤집지 않으므로 이 불일치를 해결하지 못한다. 이 문제는 Cox 계열 선택에 해당하며 CE/MSE 경로까지 같은 이유로 오류라고 판단하지 않는다.

### 기본 학습 Dataset V2와 표준 CSV 추론이 맞지 않는다

Trainer의 기본 V2 입력은 57차원인데 `run_inference_on_csv`는 V1 `PGCDataset`을 고정 사용해 55차원을 만든다. 합성 CSV를 V1로 읽어 V2 구조의 backbone에 전달하자 `mat1 and mat2 shapes cannot be multiplied (2x55 and 57x16)`이 발생했다. 오류의 16은 검증용 소형 embedding 크기다. [학습 차원 결정](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/trainer.py#L607-L624), [표준 추론 Dataset](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/inference.py#L158-L167)

표준 추론은 padding 없이 실제 팀 수로 forward하고 학습은 최대 16개로 padding한다. 다음 항목의 attention mask 문제 때문에 차원만 고쳐도 학습·추론의 dummy-token 조건이 달라질 수 있다. JSON demo 경로와 표준 CSV 추론 경로도 같은 것으로 간주하면 안 된다.

### Padding과 탈락 팀이 attention에서 제외되지 않는다

Loss와 metric에는 valid mask가 있으나 backbone의 `self.transformer(tokens)`에는 key padding mask가 없다. 따라서 padding token과 이미 탈락한 팀도 살아 있는 팀의 representation 계산에 들어갈 수 있다. 탈락 팀의 역사적 통계를 문맥으로 남기려면 의도와 처리를 별도로 정의할 수 있지만, 가짜 padding token까지 팀처럼 사용하는 것은 분리해야 한다. [Backbone forward](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/models/backbone.py#L178-L201), [Loss mask](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/trainer.py#L166-L191)

동일한 두 실제 팀을 2개 token으로 처리한 경우와 16개로 padding한 경우, 살아 있는 팀 embedding의 최대 절댓값 차이가 0.1715였다. invalid token의 feature만 바꾸면 차이는 1.6137이었다. 이는 seed 42, dropout 0, 학습되지 않은 작은 backbone에서 측정한 **동작 진단값**이다. 실제 데이터의 성능 효과 크기를 뜻하지 않는다.

### V1의 생존 mask는 표준화된 체력에 의존한다

V1은 feature를 표준화한 뒤 `squad_total_health > 0`을 검사한다. 이때 0은 원 체력의 0이 아니라 학습 평균을 뜻한다. 합성 사례에서 원 체력 50과 150인 두 팀은 모두 생존하지만 mean 100/std 10을 적용한 V1 mask는 `[False, True]`였다. 기본 V2는 이 health 조건을 주석 처리하고 위치 유효성만 사용하므로 `[True, True]`를 유지했다. 이 문제를 기본 V2 학습 문제로 혼동하면 안 되지만 표준 CSV inference의 V1 경로에는 해당한다. [V1 scaling](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/data/dataset.py#L188-L201), [V1 mask](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/data/dataset.py#L236-L251), [V2 mask](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/data/dataset_v2.py#L303-L318)

### 16개보다 많은 팀은 조용히 잘린다

Collate는 `min(n_squads, 16)`개만 복사한다. 20개 팀의 합성 sample은 16개 slot으로 줄어들지만 `num_squads` metadata는 20으로 남았다. Solo·duo·일반 squad에서 팀 수가 이 제한을 넘는 sample은 그대로 이식하면 안 된다. 이는 데이터 로더의 제한이며 self-attention 모델이 본질적으로 16개만 처리할 수 있다는 뜻은 아니다. [Collate](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/data/dataset.py#L63-L76)

### 비어 있는 risk set을 제외하는 방어가 부족하다

유효 팀이 없는 합성 sample에서 metric 함수는 accuracy 1, Brier 0을 반환했고 weighted loss는 finite하지 않았다. Baseline runner는 alive team이 없는 sample을 건너뛰지만 neural evaluation은 같은 조건을 명시적으로 건너뛰지 않는다. 원본 데이터에 이런 sample이 얼마나 있었는지는 모른다. 또한 sampling generator는 2000초까지 고정 시점을 생성하며 경기 종료 시점으로 truncation하는 처리가 해당 sampling 함수에는 없다. 종료 후 sample과 단독 생존 팀 sample을 확인하고 평가에서 다룰 규칙이 필요하다. [Metric](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/metrics.py#L21-L60), [Neural evaluation](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/src/training/evaluation.py#L209-L237), [Baseline skip](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/scripts/run_baselines.py#L98-L101)

### 최초 zone 관측 이전에는 미래 zone을 반환한다

`_get_zone_center_at_timepoint`는 t 이전 관측이 하나도 없으면 전체 zone table의 첫 행을 사용한다. 첫 관측이 30초인 합성 table에 t=10초를 질의하면 30초의 zone 값이 반환됐다. 해당 조건에서는 미래 관측 fallback이 발생한다. 원본 데이터에서의 빈도와 예측 영향은 미확인이다. 전처리 전체를 미래 누수가 있는 것으로 일반화하는 대신, **관측 전 fallback을 unknown/default로 처리하고 prefix replay와 full-log extraction을 대조**해야 한다. [Zone fallback](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/data_generation/feature_engineering/extracting_pgc_features.py#L301-L339)

### 부활 처리와 death time 기반 feature가 일치하지 않는다

위치 계산의 생존 판정은 KILLED/REVIVED 이력을 읽는다. 반면 주변 인원·일부 zone-distance snapshot 함수는 `death_time <= t`면 사망으로 처리한다. 합성 사례에서 100초 사망, 150초 부활, 170초 질의를 주면 revival-aware alive는 True인데 주변 적 feature는 사망 sentinel -1이었다. [부활 생존 판정](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/data_generation/feature_engineering/extracting_pgc_features.py#L158-L197), [주변 적 판정](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/data_generation/feature_engineering/extracting_pgc_features.py#L834-L857)

Label parser도 플레이어별 첫 KillV2만 남겨 death time으로 사용한다. 이를 팀에서 max 집계해도 부활 뒤의 최종 팀 탈락시간이 복원되지는 않는다. 부활이 존재하는 데이터에 대해서는 event 정의를 재검토해야 한다. 부활이 없는 경기의 label까지 잘못됐다는 뜻은 아니다. [첫 death time 선택](https://github.com/krafton-ai/pubg-winprob/blob/99ed9d1fa5f1df0cb242bb42ed8a511e6c972896/data_generation/feature_engineering/parsing_match_log.py#L340-L349)

## 붙여넣은 설명에서 유지하거나 수정할 부분

| 기존 설명 | 판정 | 사용할 표현 |
| --- | --- | --- |
| Transformer의 sequence dimension은 squad | 확인 | 한 시점의 팀 간 attention |
| 과거가 없는 snapshot 모델 | 수정 필요 | 과거 누적 통계는 포함하지만 순서형 이력은 직접 입력하지 않음 |
| 192개 phase-aware sampled point | 확인 | 178개 random + 14개 anchor를 개별 snapshot으로 학습 |
| Survival regression + Cox + winner classification 출력 | 오해 가능 | Loss별로 한 head를 선택하며 동시 multitask 학습은 아님 |
| KRAFTON의 데이터는 esports에 한정 | 미확인 | esports 목적을 표방하지만 경쟁전 생성 경로도 공개돼 있음 |
| Cox/IBS로 일반적인 survival probability 평가 | 수정 필요 | 변형 탈락 순서 loss와 snapshot별 winner Brier를 구현 |
| Phase-aware sampling은 phase-conditioned hazard | 동일시하면 안 됨 | Sampling/evaluation의 phase 구분과 예측 event 구간 정의를 분리 |
| History를 더하면 차별점이 입증됨 | 아직 가설 | 누적 통계·summary·order-destroyed history를 통제한 뒤 검증 |
| Typed graph는 기존 모델에 없는 관계 정보 | 과도한 표현 | 기존 attention/주변 적 feature 대비 명시적 관계 정의의 추가 효과를 검증 |

붙여넣은 글의 우리 연구 성능 수치는 이번 공개 저장소 audit의 검증 대상이 아니다. 우리 프로젝트의 기존 `BATTLE_ROYALE_RQ4_SAME_CONDITION_SEQUENCE_RESULT_2026-07-05.md`는 history summary HGB가 강한 비교군임을 보여 준다. 따라서 sequence 모델이 약한 last-only neural baseline을 이긴 결과만으로 **시간 순서 자체의 효과**를 주장하지 않아야 한다. 기존 결과의 전체 수치와 provenance 재검증은 새 실험 설계 단계에서 따로 수행한다.

## 새 실험으로 옮기기 전에 정할 계약

다음은 코드에서 자동으로 따라오는 결론이 아니라, 이번 audit에 근거한 **설계 제안**이다. 모델을 추가하기 전에 입력·예측 사건·관측 시점·risk set·평가 단위를 고정하는 것이 우선이다.

### 서로 다른 두 재현 목표를 구별한다

| 목표 | 유지할 내용 | 결과의 명칭 |
| --- | --- | --- |
| 공개 연구 목표의 자체 데이터 재현 | 팀별 snapshot, 원형 loss, winner/death-order target | KRAFTON 구현을 자체 데이터에서 재현한 결과 |
| 우리 next-phase hazard로 이식 | 같은 snapshot encoder, 우리 event/구간과 binary hazard head | KRAFTON-style encoder의 task adaptation |

현재의 데이터 부재로 원 논문의 성능표 자체를 재현할 수는 없다. 공개 코드의 점수 방향과 padding 등을 고치는 경우 변경 내용을 기록해야 하며 수정된 구현을 원형 코드와 동일하다고 부르면 안 된다. Hazard로 이식한 결과는 원 논문의 win probability와 직접 성능 비교할 수 없다.

### 시간 정보의 비교는 네 단계로 구성한다

| 비교군 | 허용할 정보 | 확인하려는 질문 |
| --- | --- | --- |
| 현재 상태 | 체력·위치·장비·현재 zone·경기 공통 상태 | 현재 상태만의 설명력 |
| 현재 상태와 누적 통계 | 위 입력 + KRAFTON에 대응하는 행동 누적 | 이미 요약된 과거의 설명력 |
| 현재 상태와 history summary | 위 입력 + 동일한 window의 요약·변화량 | 순서 없는 또는 제한적으로 요약된 이력의 추가 설명력 |
| Ordered history | 동일한 측정 변수·window·현재 상태와 시간 순서 | 순서와 경로의 추가 설명력 |

모든 군에 필요한 현재 누적 통계와 현재 zone 정보를 동일하게 제공하고, 비교 가능한 encoder/head·학습 예산을 사용한다. History window 길이, sampling interval, missingness, 마지막 관측 시점도 맞춘다. 순서의 효과는 단순 history 유무 외에 time-order control로 확인하되, shuffle이 불가능한 궤적을 만드는 정도와 입력 분포 변화를 함께 점검한다. 현재 상태 feature·누적 통계·summary의 통제 없이 ordered history의 독립 효과라고 결론내리지 않는다.

### 팀 간 구조는 시간 정보와 별도로 비교한다

같은 팀 표현에서 독립 MLP, 전체 팀 attention, typed dynamic graph를 비교한다. Typed graph는 실제 관계·shuffled 관계·관계 없는 조건을 두고, 관계에서 사용하는 상대 정보를 node/global 입력에도 주는 통제를 포함한다. 이 비교는 “관계의 유무”보다 **현재 팀 전체의 정보 결합 방식과 명시적 관계 정의의 효과**를 검증한다. Graph의 이득이 작으면 이를 주기여로 미리 정하지 않는다.

### Event와 평가 모집단을 먼저 정의한다

1. 예측 시점 t에 실제로 관측 가능한 로그만 입력에 쓴다. 전체 사후 로그에서 계산한 feature가 prefix replay와 일치하는지 확인한다.
2. 사망, 다운, 부활, 최종 팀 탈락을 구별하고 phase 전환 전후의 event 포함 규칙을 정한다.
3. 실제 경기 phase event를 기준으로 horizon을 정한다. 가변 phase 길이와 phase 안의 예측 시점에서 남은 시간을 함께 처리한다.
4. Winner probability는 살아 있는 팀 간 합이 1인 출력이고, next-phase 팀별 탈락확률은 여러 팀이 함께 탈락할 수 있어 합이 1일 필요가 없다. 둘의 head와 loss를 구별한다.
5. 최소 경기 단위로 분할하고 시간·반복 플레이어/팀·patch의 분포 이동을 추가로 점검한다. 현재 원 저장소의 실제 split은 미확인이다.
6. 모든 팀 수를 보존하고 padding·탈락·관측 없음의 mask를 분리한다. Unknown map을 Erangel로 조용히 매핑하지 않는다.
7. 종료 후·빈 risk set·단독 생존 팀 sample의 포함 규칙을 고정한다. 시간별, phase별, 경기별 평균의 가중 방식을 함께 보고한다.
8. Hazard에는 AUC/AP와 Brier/log loss 및 phase별 calibration을, winner에는 accuracy와 multiclass probability 지표를 사용한다. Survival curve를 주장하려면 곡선과 censoring을 다루는 평가를 별도로 정의한다.
9. 신뢰구간은 같은 경기의 반복 시점을 독립 표본으로 처리하지 않고 경기 단위로 구성한다. Seed 반복과 hyperparameter 선택 예산을 맞춘다.

이 단계에서 중심 가설은 “전체 팀의 현재 상태와 누적 행동을 이미 알고 있을 때, 명시적인 시간 이력이 다음 phase 탈락 예측에 추가 정보를 주는가?”로 둘 수 있다. **시간 순서가 유용하다는 것은 검증할 가설이며 아직 이번 audit의 결과가 아니다.**

## 실행 검증과 남은 한계

검증은 Python 3.11.5, PyTorch 2.11.0, NumPy 1.26.1, pandas 2.1.1에서 CPU로 실행했다. 원본 학습이나 새 예측 모델 학습은 수행하지 않았다. 설치된 runtime을 사용했으며 requirements 전체를 새로 설치하지 않았다.

프로젝트 루트에서 다음 명령으로 합성 검증을 다시 실행할 수 있다.

```sh
python3 reports/krafton_pubg_winprob_audit_2026-10-02/audit_checks.py \
  --archive reports/krafton_pubg_winprob_audit_2026-10-02/upstream_99ed9d1.zip \
  --output reports/krafton_pubg_winprob_audit_2026-10-02/audit_results.json
```

저장된 ZIP의 SHA256은 `7cdfc8fade1e38538f264ab0f096de65a76420dfeab43ecb2798916522df7b30`이다. 재현 사례의 assertion은 모두 통과했다. 통과의 의미는 원 공개 구현의 예상된 문제 동작이 재현됐다는 뜻이며 저장소 전체가 정상이라는 뜻은 아니다.

원 논문 본문, 실제 원본 로그, 분할, 체크포인트가 없어 원 성능표·결과의 유의성·실제 누수 빈도·각 구현 문제의 성능 영향을 확인하지 못했다. 다른 비공개 버전이나 운영 시스템에 같은 문제가 있다고 단정할 수 없다. 기존 우리 원고와 결과 파일은 수정하지 않았다. 이번 문서는 후속 실험 계약을 확정할 근거이며 최종 실험 프로토콜은 별도 단계에서 작성한다.
