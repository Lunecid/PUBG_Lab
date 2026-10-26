# v15 → v16 변경 사항 (2026-10-08)

v15 리뷰(점수 79)의 1–4번 항목과 첫 리뷰(`ACM_SAC2027_MLA_Review_and_Revision_KO.md`)의 우선순위 3개를 반영했다(사용자 선택). 새로 학습한 모델은 full-feature logistic 기준 하나다. 나머지 새 분석은 저장된 시험 예측만 썼다.

## 1. 매치 오프셋 분해 (피드백 4번, 신규 분석)
- `tools/sac2027_match_offset.py`: 목적함수 대조 5종 × 시드 3개 × 두 코호트에 적용했다.
  - BCE − truncated Cox의 단계 안 격차: 에랑겔 +0.0060, 타이고 +0.0076.
  - 매치·단계 평균 logit을 빼면(중심화) +0.0015 [−0.0003, 0.0033], +0.0008 [−0.0009, 0.0024]로 줄어든다.
  - 매치·단계 평균 logit만으로 매치 간 AUC를 재면 BCE가 +0.016 [0.013, 0.020], +0.019 [0.015, 0.022] 높다.
  - full과 truncated Cox 사이에서는 이 매치 수준 성분이 바뀌지 않는다(+0.002, −0.004).
  - 같은 매치 쌍은 단계 안 쌍의 0.2%다.
- `tools/sac2027_match_offset_structure.py`: 11개 코호트의 팀 축 주효과와 PTR−GBDT에 적용했다(시드 42, DerSimonian–Laird).
  - PTR−GBDT: 단계 안 +0.0058이 중심화 후 +0.0011 [−0.0022, 0.0044]가 된다. 매치 평균 성분은 +0.0034 [0.0016, 0.0051]이다.
  - 팀 축: 단계 안 +0.0109 가운데 +0.0049 [0.0022, 0.0076]가 매치 수준이다. 중심화 후에도 +0.0043 [0.0020, 0.0067]이 남는다(약 절반만 매치 수준).
- 원고 반영 위치:
  - 5.1절: 분해 정의
  - 6.3절: 목적함수와 구조 결과
  - 7.1절: Cox 위험집합 정규화의 불변성 설명과 조건부 우도 비유(Chamberlain 1980 인용 추가)
  - 초록·기여·결론

## 2. 감사 기록 문체와 단서 정리 (피드백 2번)
- 엔지니어링 기록을 `ARTIFACT_README.md`로 옮겼다.
  - 원고에서 빠진 것: 테스트 개수, 점검 범위 단서, 유지보수 창 경기, KRAFTON 통합 수정, IBS, CPU 대조, GATv2 그래프 밀도, 연산 비교 불가 문장.
  - 원고에는 결과 첫 단락에 재구성 검증 결과 두 문장만 남겼다.
- 흩어진 단서를 한계 단락 한 곳에 모았다. 한계 단락은 위험집합 정의, 후향 분할, 기준선 근사, 관측 연관, 튜닝 범위를 다룬다.
- 그 밖의 압축:
  - GATv2는 두 문장으로, 이력·모집단 통제는 한 단락으로 줄였다.
  - 예시 소절은 3.4절에 합쳤다.
  - 서론 2단락과 관련 연구를 압축했다.
- 본문의 "not"은 48개에서 25개로, "does not"은 11개에서 2개로 줄었다.
- 쪽수: 10쪽을 유지했다.
  - 본문은 9쪽 오른쪽 단 중간에서 끝나고, 참고문헌(ACM 형식, 40개)이 약 2.45단을 차지한다.
  - 9쪽에 맞추려면 본문을 정확히 1쪽(2단) 더 빼야 해서 보류했다. SAC 한도는 8+2쪽이다.

## 3. 분석 이력 표현 (피드백 3번)
- "inherited", "earlier analyses", "original results", "older split"을 모두 지웠다.
- 대신 이렇게 썼다: "결정 규칙, SESOI, 다섯 주요 비교는 같은 경기의 탐색적 무작위 분할 분석에서 패치 분할을 정하기 전에 고정했고, 패치 프로토콜은 그 결과를 본 뒤 재학습 전에 기록했다."
- 기준 레시피는 41.2 경기가 학습에 들어 있던 무작위 분할 단계에서 정했다. 반면 동일 횟수 튜닝(표 8)은 41.1 검증만으로 선택했다. 그래서 표 8을 가장 깨끗한 구조 비교로 명시했다.

## 4. 초록 (피드백 4번 항목 중 초록)
- 다음을 넣었다: RQ2 비교 수치(위상 보정 KRAFTON 0.747, Sagong 0.675, 사람 팀 격차 +0.020), PTR-Former 정의(phase–team attention Transformer), 입력 비교(KRAFTON 입력이 경기 안에서 우세, 합집합 GBDT 0.786), 매치 수준 메커니즘.
- 끝의 단서 문장은 한계 단락으로 옮겼다. 초록은 195단어다.

## 5. 첫 리뷰 우선순위 3개 (사용자 선택: 기준+모드+활용)

### 5-1. Full-feature 정규화 logistic 기준 (P0-2)
- `tools/sac2027_logistic_baseline.py`: 단계 지시변수 6개와 GBDT의 151개 입력에 L2 logistic을 맞췄다. 입력은 학습 집합 평균·표준편차로 표준화하고 상수 열은 뺐다.
  - 벌점 C는 {0.001, 0.01, 0.1, 1, 10} 가운데 41.1 검증 log loss로 골랐다(선택값 0.001–0.1).
  - 학습 40.2, 검증 41.1, 시험 41.2로 다른 설계와 같은 행·분할을 쓴다.
- `tools/sac2027_logistic_summary.py`: 표 4의 다른 행을 만든 코드(`sac2027_evaluate_controls.analyze`와 native 평가 도우미)로 지표를 계산했다.
  - 같은 실행에서 GBDT를 다시 계산했고, 저장된 표 4 입력값과 1e-10 안에서 일치함을 확인했다.
- 표 4 새 행(11개 코호트 산술 평균): AUC 0.759, 사람 팀 0.752, 단계 안 0.737, 경기 안 0.729, log loss 0.504, Brier 0.166, 단계 보정 오차 0.019, 승자 정확도 0.366, C-index 0.646.
- 통합 대조(DerSimonian–Laird, `tools/sac2027_extra_evaluation.py`):
  - GBDT 대비: AUC −0.0098 [−0.0162, −0.0034], 경기 안 −0.0178 [−0.0298, −0.0058].
  - PTR 대비: AUC −0.0104 [−0.0170, −0.0039].
  - 단계 보정 KRAFTON 대비: AUC +0.0103 [0.0009, 0.0196], 경기 안 −0.0088 [−0.0203, 0.0028].
- 해석(7.1절): 선형 모델로도 GBDT의 KRAFTON 대비 전체 이득의 약 절반을 얻는다. 경기 안 이득은 비선형 적합에서만 나타난다.

### 5-2. Normal/ranked별 평가 (리뷰 표 5번)
- 시험 경기가 20개 이상인 층만 평가했다.
  - normal: 10개 코호트(론도는 18경기라 제외).
  - ranked: 6개 코호트(26–108경기). 듀오 ranked(8경기, 13경기)는 제외.
- 단계 보정 KRAFTON 대비 통합 AUC 이득(모두 0을 포함하지 않음):
  - PTR: normal +0.0166 [0.0116, 0.0216], ranked +0.0246 [0.0135, 0.0356].
  - GBDT: normal +0.0157 [0.0105, 0.0209], ranked +0.0315 [0.0222, 0.0407].
- PTR − GBDT: normal +0.0010 [−0.0023, 0.0043], ranked −0.0058 [−0.0131, 0.0014]. 둘 다 불확실하다.
- 앞서 실수로 지웠던 한계 문장("normal/ranked 미시험")을 이 결과로 대체했다. ranked 층의 크기(6개 코호트, 26–108경기)는 한계 단락에 남겼다.

### 5-3. Top-3 탈락 포착률과 보정 기울기·절편 (리뷰 표 6번, P1-4)
- Top-3 포착률: 경기의 각 경계에서 위험이 가장 높은 3팀에 든 다음 단계 탈락의 비율이다.
  - 동점은 무작위 배분의 기댓값으로 계산했다. 탈락이 없는 경계는 뺐다.
  - 구간은 경기 단위 부트스트랩(2,000회)으로 냈다.
- 11개 코호트 평균: GBDT 51.7%, PTR 51.6%, 단계 보정 KRAFTON 51.1%, logistic 50.6%, 입력 결합 GBDT 53.3%, Sagong 42.4%.
  - 기준: 무작위 34.7%(단계만 쓰는 모델은 경계 안에서 모두 동점이라 무작위와 같다), 남은 인원 최소 규칙 45.1%.
- 통합 대조:
  - GBDT − 인원 규칙: +0.068 [0.062, 0.073], 11/11 코호트에서 같은 부호.
  - 결합 − GBDT: +0.0146 [0.0091, 0.0200].
  - GBDT − 단계 보정 KRAFTON: +0.0057 [0.0001, 0.0114].
  - PTR − GBDT: −0.0018 [−0.0061, 0.0026].
- 보정 기울기와 calibration-in-the-large, 코호트 평균:
  - 정의: 결과를 예측 log-odds에 logistic 회귀한 계수, 그리고 그 계수를 1로 고정했을 때의 절편.
  - GBDT 0.86 / −0.09, PTR 0.95 / −0.03, 단계 보정 KRAFTON 0.95 / +0.03, logistic 1.00 / −0.08, 결합 0.87 / −0.10, Sagong 0.59 / −0.08.
  - GBDT는 41.2에서 예측이 약간 넓게 퍼지고, 평균적으로 조금 과대 예측한다.
- 인용 추가: Van Calster et al. 2016 (J. Clin. Epidemiol. 74:167–176). 서지는 Crossref로 확인했다.
- 독립 점검: 론도 GBDT의 top-3 포착률을 pandas 순위로 따로 계산했고, 0.616490으로 일치했다.

### 원고 반영 위치
- 기여 문단: logistic 기준 언급
- 4.2절: logistic 기준 정의 한 문장
- 5.1절: top-3 포착률, 보정 기울기·절편 정의
- 6.2절: logistic 결과, normal/ranked 결과, 검토 효용 단락
- 표 4: logistic 행 (`tables/main_results.tex`에 직접 넣음; 값 출처는 `reports/sac2027_patch_v14/baseline_logistic/table_metrics_means.csv`)
- 7.1절: 선형 대 비선형 해석 한 문장
- 7.3절 한계: "top-k 미측정"을 "top-3 포착률은 대리 지표"로, "logistic·MLP 미시험"을 "MLP 미시험"으로 바꿨다. ranked 층 크기를 추가했다.
- 쪽수: 10쪽 유지, overfull 0, 참고문헌 41개. 10쪽 오른쪽 단은 y≈582pt에서 끝난다(약 13줄 여유).

## 6. 그림 수정 (사용자 요청)
- 그림 1 (`figures/fig_pipeline_clean.tex`):
  - RQ2 "Our design" 상자에 "regularized logistic"을 넣었다(표 4 행 이름과 같다).
  - RQ3 상자에 "equal-trial tuning incl. GATv2 graph" 줄을 넣었다. 그래프 모델 GATv2가 그림에 처음 나온다.
  - 패널 C 띠: "reference recipes · human / history / objective / graph controls". 16회 튜닝은 RQ3 상자로 옮겼고, GATv2의 관계 재표지 대조를 "graph controls"로 넣었다.
  - 지표 상자에 "calibration slope", "top-3 capture"를 넣었다. 한 줄로는 상자 폭(3.23 cm)을 넘어서(3.44 cm) 두 줄로 나눴다.
  - 서식지 상자의 "movement residual"을 4.1절 정의 이름 "movement deficit"으로 바꿨다.
  - 배치: 아래 영역(패널 C·D)만 0.46 cm 아래로 늘렸고, 위 영역 좌표는 그대로다. 그림 높이는 8.97 cm에서 9.43 cm가 됐다.
- 그림 2 (`figures/fig_rq1_effects.pdf`): y축 라벨 "Movement residual"을 "Movement deficit"으로, "Nearby enemies"를 "Local competition"으로 바꿨다(4.2절 RQ1 변수 이름).
  - `tools/sac2027_plot_rq1.py`로 다시 그렸다. 바꾸기 전 이 스크립트가 기존 그림을 픽셀 단위로 그대로 재현함을 확인했다. 스크립트는 이제 출력 폴더를 인자로 받는다(기본값은 예전대로 v14 폴더).
- 본문: 초록, 4.2절 강건성 문장, 결론의 "movement residual"을 "movement deficit"으로 바꿨다. 정의 문장의 "speed residual"과 6.1절의 "negative-residual fractions"는 설명어라 그대로 뒀다.
- 그림 3은 새 분석과 무관해 그대로다.
- 이전 그림 파일은 `figures/v16_pre_logistic_gatv2/`에 보존했다.
- 쪽수: 10쪽, overfull 0.

## 7. v16 리뷰(82점) 반영 (2026-10-08)
- 3번 "cannot learn": 초록, 기여, 결론을 "match-level risk offset에 대해 훈련 신호가 없다"로 바꿨다.
  - 6.3절에 truncated Cox의 매치 평균 logit AUC 수준을 넣었다: 에랑겔 0.586, 타이고 0.597 (BCE 0.602, 0.616). 팀 입력이 매치 상태를 담기 때문에 0.5보다 높다.
  - 7.1절: 매치 정보는 팀 수준 적합의 부산물로만 Cox 점수에 들어간다고 썼다.
- 4번 LTR 연결: 7.1절에 한 문장을 넣었다. 동점이 없을 때 Cox 부분우도는 ListMLE가 최대화하는 Plackett–Luce 우도와 같고 [Xia08], listwise 점수는 쿼리 간에 보정되지 않으며 pointwise sigmoid CE를 더하면 보정된다 [Bai23]. 기여 범위는 "landmark 생존 예측에서 이 성분을 matched control로 측정한 것"으로 적었다.
  - 결론의 future work에 Cox와 CE 결합 목적함수를 넣었다.
  - 참고문헌 2개는 Crossref로 서지를 확인했다.
- 5번: 7.1절 끝 구절을 "next-phase 학습의 경기 안 우위는 비선형 학습기에서만 나타난다"로 바꿨다.
- 6번: 이름을 "movement residual"로 통일했다(4.1절 정의, 표 2, 4.2절 변수 목록과 민감도 문장, 초록, 결론, 그림 1·2). 바로 전 턴에 deficit으로 통일했던 것을 되돌렸다. 정의 문장은 "speed difference"로 써서 이름과 겹치지 않게 했다.
- 7번: 초록 문장 순서를 바꿔 "In the two largest cohorts"가 튜닝과 truncation을 함께 덮게 했다. 초록은 200단어다.
- 9번: 6.2절 top-3 문단에 "over this rule"을 붙이고, 학습된 세 설계가 0.6pp 안에 있으며 이는 규칙 대비 이득(6.0–6.6pp)의 약 10분의 1이라는 문장을 넣었다.
  - SESOI 환산(`tools/sac2027_sesoi_top3.py`): 9개 대조 × 11개 코호트(99점)에서 top-3 포착률은 경기 안 AUC 1당 0.66 [0.54, 0.78] 변한다. 따라서 경기 안 AUC 차이 0.005는 약 0.33pp [0.27, 0.39]이고, 경계 약 90개당 탈락 1건을 더 포착하는 정도다(코호트 평균 경계당 탈락 3.34건). δ 자체는 전체 AUC 기준이므로, 주요 비교에 대해서는 아래의 직접 측정이 근거다.
  - Table 7 비교의 top-3 변화: 단계 축 −0.36pp [−0.52, −0.19], 팀 축 −0.07pp, 관계 정렬 +0.05pp, 존 토큰 +0.05pp (팀 축 순위상관은 팀 축과 같은 arm).
  - 위치: 6.3절 "We retain δ=0.005" 뒤. 5.2절에는 참조만 두었다(방법 절에 결과를 쓰지 않는 원칙).
- 10번(선택):
  - RQ2 중심화(`tools/sac2027_match_offset_rq2.py`): PTR과 GBDT의 phase 보정 KRAFTON 대비 단계 안 이득 +0.0271, +0.0188이 중심화 후 +0.0093 [0.0044, 0.0143], +0.0073 [0.0020, 0.0126]로 줄어 경기 안 이득(+0.0071, +0.0100)과 비슷해진다. 매치 평균 logit만으로는 +0.048, +0.043. 6.2절에 한 문장, 결론에 한 구절을 넣었다.
  - 퍼짐 표준화: 팀 축의 중심화 후 잔여 +0.0044 [0.0020, 0.0067]가 match–phase SD로 나눠도 +0.0034 [0.0007, 0.0062]로 남는다. 리뷰가 제안한 퍼짐 가설은 지지되지 않아 원고에는 넣지 않았다.
- 11번:
  - 중심화 분해에 "post hoc diagnostic" 라벨을 붙였다.
  - RQ1 예측구간(Higgins et al. 2009): 경쟁 [0.08, 0.49], 인원 [−0.95, −0.24]. 이질성이 커도 새 코호트에서 부호가 유지된다.
  - 제목의 "Team Context"를 "Habitat Context"로 바꿨다.
  - "Against CPU references"를 지웠다(통제 문단 압축 때 함께).
  - 철자: "centring"을 "centering"으로, "re-evaluation"을 "reevaluation"으로 바꿨다.
  - 표 8 각주에 기준 레시피 AUC를 넣었다(GBDT 0.803/0.803, PTR-Former 0.799/0.807, KRAFTON+CE 0.800/0.801).
  - 3.2절에 검증 분할이 가장 큰 이유를 넣었다: 표본 경기가 패치 41.1에서 35일, 40.2에서 13일, 41.2에서 16일에 걸친다(`split_manifest.csv`의 생성 시각).
- 8번 쪽수:
  - 결과: 10쪽 유지, 마지막 단 여백 195pt에서 323pt(약 1/4쪽)로 늘었다. camera-ready 모의본(저자 3명과 긴 소속, ACM 저작권 블록, ACM Reference Format, 140자 사사)도 10쪽에 들어가고 마지막 단에 약 76pt가 남는다.
  - 그림 1: 구역 라벨과 구분 점선을 없애 높이를 9.43 cm에서 8.98 cm로 줄였다. 리뷰가 제안한 20%는 글자를 7pt 아래로 줄이거나 내용을 빼야 해서 하지 않았다.
  - 6.3절 통제 문단을 두 문장으로 줄였다.
  - 표 1(선행 설계 비교): 셀을 줄여 모든 행을 한 줄로 만들고, 4.3절 본문과 겹치는 "Our adaptation" 행을 지웠다. 표 2의 선행 설계 특징 목록, 표 3과 표 4의 각주도 줄였다.
  - 학습 설정은 핵심만 남기고 전체는 `ARTIFACT_README.md`로 옮겼다.
  - 줄이거나 지운 문장: 관련 연구의 농구 사례(Cervone16 인용이 빠짐)와 PUBG 설계 나열 중복, 생태학 문장, 3.1절 업데이트·윤리 문장, 4장 도입 로드맵, 5.1절 예시 문장, Shapley 식, 평가 절 도입, KRAFTON 목적함수 불변성 문장, 검증 적합 쌍둥이(방법과 결과), 서식지 63% 문장, KRAFTON 경로 수치 나열(그림 3과 표 5에 있음), weighted truncation 문장, "Their sum is exact", δ 민감도, 튜닝의 human-only 문장, 결합 입력 0.786 중복, RQ1 이동·단계·lite 문단 일부, 7.2절, 한계의 서론 중복 문장, 결론의 "Structural effects" 문장.
- 1번 코드 링크: 공개용 저장소를 점검했다(`REVIEW_RESPONSE_KO.md`). 링크는 저장소를 만든 뒤 교체해야 한다.

## 8. v16 리뷰 후속(문장 수준, 2026-10-09)
- 결론: "most of the gain"을 "most of the within-phase gain"으로 좁혔다. 중심화로 줄어드는 몫은 PTR 66%, GBDT 61%이고, 경기 안 이득에는 정의상 매치 수준 성분이 없다.
- SESOI: 6.3절 문장을 "a within-match AUC difference of 0.005 corresponds to about 0.33 pp"로 고쳤다. 5.2절 참조도 "relates it to top-3 capture"로 바꿨다.
- 6.3절 튜닝 문단에 표 8 각주의 의미를 두 문장으로 넣었다.
  - 41.2 경기가 학습에 들어 있던 단계에서 정한 기준 레시피와 41.1로만 튜닝한 값(153열)의 차이는 GBDT와 PTR에서 최대 0.002다. 따라서 기준 결과가 그 노출로 부풀었다는 근거는 없다.
  - KRAFTON 백본은 0.003/0.007 오르고, 그 결과 타이고의 기준 격차(PTR − KRAFTON +0.006)가 +0.0006으로 사라진다. 표 5 (e) 단계의 +0.004도 이 레시피 차이 범위 안이다.
- 6.3절 Cox 문장에 "BCE가 우연 수준을 넘는 매치 판별력의 약 84%"를 넣었다((0.586−0.5)/(0.602−0.5)=0.84, (0.597−0.5)/(0.616−0.5)=0.84).
- 7.1절: "knows"를 "faces"로 바꾸고, Bai et al.은 "one remedy"로 낮췄다.
- 6.2절: 매치 평균 격차에 CI를 붙였다(+0.048 [0.036, 0.060], +0.043 [0.032, 0.054]).
- 초록: SAC 한도(100–200단어) 여유를 위해 생존 적응 문장을 줄여 197단어로 맞췄다.
- 쪽수: 10쪽, 마지막 단 여백 259pt(약 1/5쪽). 리뷰 권고대로 camera-ready 때 1/3쪽을 더 줄인다.

## 9. 외부 리뷰(Borderline–Weak Accept, 2026-10-09) 반영
- 3.3 BCE 일차 조건(최우선): 7.1절의 "각 경기–단계에서 기대·관측 탈락 수를 일치시킨다"를 고쳤다. 공통 오프셋 $c_{m,k}$에 대한 BCE 미분 $\sum_i(\hat p-Y)$를 적고, "모델이 오프셋을 표현할 수 있으면 매치 수준 위험의 학습 신호를 준다"로 낮췄다. Cox 쪽 문장도 "so … so" 이중 구문을 두 문장으로 나눴다.
- 중심화 진단의 해석 범위: 5.1절에 "neither is an additive decomposition of AUC into causes"를 넣었다. 6.3절과 7.1절의 "decomposition"은 "centering diagnostic"으로 바꿨다.
- 3.1 기여 초점: 기여 문단 첫 문장을 "matched-control measurement"로 바꿨다(horizon 정렬로 경기 안 격차가 닫히고, 남은 단계 안 격차와 KRAFTON 대비 단계 안 이득 대부분이 매치 수준 오프셋). 나머지는 이를 받치는 기여로 정리했다.
- 3.4 구조 결론 범위: 결론에 "in two cohorts … under the tested search budget"를 넣었다.
- 그림 1:
  - 패널 D: 종합 상자를 "Synthesis (random effects) / 11 cohorts; controls: 2 largest"로 바꿨다. 지표 상자에 "match–phase centering"을 넣고 "calibration slope · top-3 capture"를 한 줄로 합쳤다(상자 폭 3.65 cm).
  - RQ1 화살표 라벨을 "coefficients, clustered SEs"로 바꿨다(4.2절의 match-clustered SE).
  - RQ3 상자: "inputs · horizon · objective · phase baseline" / "structure: equal-trial tuning, GATv2 graph".
  - 패널 B: "observed alive status", 라벨 상자 "Retrospective labels"(risk set $T_{m,i}>b_{m,k}$, $Y=1$ if $T_{m,i}\le b_{m,k+1}$), 시간축 기호 $a_{m,k}$, $b_{m,k}$, $b_{m,k+1}$.
- 그림 3 재구성(`tools/sac2027_figure3.py`, 저장된 결과만 사용):
  - (a) 단계별 보정은 그대로다.
  - (b) horizon 정렬: 두 코호트에서 BCE − full Cox와 BCE − next-phase Cox를 경기 안·단계 안 AUC로 보인다.
  - (c) 중심화 진단: BCE − next-phase Cox(두 코호트), 팀 축, PTR − GBDT, PTR − KRAFTON+phase(11개 코호트 통합)의 단계 안 차이를 중심화 전후로 보이고, 경기 안 차이를 나란히 둔다.
  - 기존 경로 폭포도는 `figures/v16_pre_fig3_redesign/`에 보존했다. 경로 변화는 표 5(통합)에 그대로 있다. 6.3절 첫 문단은 표 5의 통합값(서식지 +0.046, 나머지 입력 +0.028, 단계 기준선 +0.020, BCE 레시피 +0.016)으로 바꿨다.
  - 본문의 해당 결과 문장에 Fig. 3b, 3c 참조를 넣었다.
- 그림 2: 리뷰가 그대로 두라고 해서 변경하지 않았다.
- 반영하지 않은 것: 미접촉 경기로 표 6 비교 재확인(새 데이터 수집과 재학습이 필요하다), MLP, Cox 점수 + 학습 오프셋 통제(새 실험), 결과 절 순서 재배치(그림 3이 핵심 증거를 7쪽으로 앞당겨 일부 대신한다).
- 쪽수: 10쪽, overfull 0, 마지막 단 여백 195pt(약 1/6쪽).

## 10. 그림 마무리와 코드 공개 범위 (2026-10-09)
- 그림 3: (c)의 팀 축 행을 "Team axis: on − off (pooled)"로 바꿨다. 캡션은 "two largest cohorts"로 쓰고, "Horizontal bars indicate 95% confidence intervals; pooled rows use random-effects estimates across 11 cohorts"로 구체화했다.
- 7.1절: "centering removes it"를 "centering substantially attenuates it, leaving no clear residual difference in the matched Cox–BCE comparison"으로 완화했다.
- 그림 1: 범례 아래에 기호 설명 두 줄을 넣었다(m, i, k, Y; $a_{m,k}=b_{m,k}-10$ s, $b_{m,k}$, $b_{m,k+1}$, $T_{m,i}$). "기호: 설명" 형식으로 쓰고 항목 사이를 넓혔다.
- 공개 저장소(`Lunecid/nextphase-elimination-code`, 커밋 0b87e36, 파일 166개): 사용자 지시에 따라 학습·평가·그림 코드와 선행 설계 적응만 남겼다.
  - 뺀 것: DB 적재(`stdb/`), 표본 생성기(`phase_v2/`의 생존 규칙 `alive.py`만 남김), dump 작성기, 원자료 재구축·검증 스크립트, 오케스트레이터(`rerun_v2`, `patch_run`), BigComp용 표·숫자 스크립트, 옛 실험 모듈, 이들에 의존하던 테스트.
  - 끊긴 import 0, 테스트 675개 통과, 노출 스캔 0건.
  - README에 범위를 밝히고, `README_ADAPTATION.md`에 빠진 파일 안내를 붙였다.
- 원고: 3.1절과 Code availability를 "training, evaluation, and figure(-generation) code"로 맞췄다.
- 익명 저장소: https://anonymous.4open.science/r/nextphase-elimination-code-D1D9/ (파일 166개, 만료 2027-04-09). `\artifacturl`을 이 주소로 바꿨다. 10쪽, overfull 0, 마지막 단 여백 92pt.

## 11. 평가 서술과 외부 리뷰 4.2–4.7 (2026-10-09)
- 사용자 결정: 평가는 패치 홀드아웃(40.2 학습, 41.1 선택, 41.2 평가)으로만 서술한다. 3.2절, 5.2절, 6.3절에서 이전 무작위 분할 분석 언급을 뺐다.
  - 5.2절: "판정 규칙, SESOI, 주요 비교 5개는 패치 분할 모델 학습 전에 고정했다", "패치 프로토콜은 학습 전에 기록했다".
  - 6.3절: 표 8 기준 레시피 비교는 수치만 남겼다.
  - 한계의 "no prospective or new-rule test"는 남겼다. 결론의 "reevaluation"은 "evaluation"으로 바꿨다.
- 4.2: 기여 문단 첫 문장에 구체 패턴과 크기를 넣었다. horizon을 맞추면 경기 안 격차가 0.001 미만, 단계 안 격차 0.006–0.008이 남고 중심화 후 0.002 이하. KRAFTON 대비 단계 안 이득 대부분도 같은 양상이다.
- 4.3: "has a structural cause", "lies in", "it is match-level"을 "is consistent with", "the diagnostics place it at the match level"로 낮췄다(초록, 기여, 6.3절, 7.1절, 결론).
- 4.4: 한계 단락에 "학습된 설계는 규칙보다 top-3 포착률이 약 6pp 높지만 서로는 0.6pp 이내라, 리뷰 시스템 선택보다 평가 방법에 대한 연구"를 넣었다.
- 4.6: 한계에 "적응 품질이 비교에 영향을 줄 수 있음", "텔레메트리를 재배포하지 않아 제3자는 코드는 볼 수 있지만 같은 표본을 다시 돌릴 수는 없음"을 넣었다.
- 4.7: 기여 문단에서 서식지 연관은 "방향보다 단계 의존성과 AI 민감성이 중요한" 보조 결과로 표현했다.
- 4.5: 원고가 이미 두 코호트와 탐색 예산으로 범위를 제한하고 있어 수정하지 않았다.
- 10쪽, overfull 0, 초록 198단어, 마지막 단 여백 76pt.

## 그 밖
- 코드 공개 문장을 되살렸다. 익명 저장소 링크는 `\artifacturl` 자리표시자이며, 투고 전에 사용자가 올린 주소로 바꿔야 한다.
- AI 사용 고지: 사용자 결정(2026-10-08)으로 "원고 편집" 구절을 뺐다. 지금은 분석·그림 코드 작성과 수정만 적혀 있다.
- v15의 QA 파일은 `qa/from_v15/`로 옮겼다. v16 검증 결과가 아니다.
- 반영하지 않은 항목(사용자 선택): 5번 RQ1 단위 계수 재통합, 7번 작은 항목(Kaufman 위치, 제목 용어, SESOI 근거, 검증 집합 크기, 표 8 기준 행, EPV 동기).
