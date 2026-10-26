# v17 변경 기록 (2026-10-09, 평가 단위 중심 재구성)

v16(`../v16_revision/`)을 바탕으로 원고 축을 평가 단위로 옮겼다. v16 폴더는 그대로 두었다.

## 1. 축과 제목
- 제목: "Ranking Teams or Matches? Evaluation Units for Next-Phase Elimination Prediction in Battle Royale".
- 핵심 주장: 단계 안 AUC의 쌍 중 같은 경기 쌍은 0.15–2.8%(가장 큰 두 코호트 0.20%, 0.15%)뿐이라, 이 지표는 거의 경기 간 순위를 매긴다. 그래서 경기 수준 오프셋을 학습하는 목적함수가 점수를 받는다. 경기 안 AUC와 경기–단계 중심화 진단을 별도 평가 단위로 제시한다.
- "알려진 성질의 통제 측정"과 "새 위험 추정기가 아님" 문장은 지웠다.
- 관련 연구에 van Klaveren et al. (2014, 군집 자료의 군집 안 판별)을 추가했다. Crossref로 서지를 확인했다.

## 2. RQ 재배치
- RQ1: 목적함수 차이 중 예측 기간 정렬과 경기 수준 오프셋이 각각 얼마인가(매칭 통제).
- RQ2: 우리 설계와 선행 설계 적용판을 세 평가 단위로 비교.
- RQ3: 목표와 평가 단위를 맞춘 뒤 입력과 구조의 효과.
- 서식지 생태학 틀은 4.1절 한 문단으로 줄였다. 회귀 결과는 한 문장으로만 남겼다.

## 3. 개입 실험 (Cox + BCE)
- next-phase Cox + 1.0 × 점별 BCE. 가중치와 예측 세 개는 실행 전에 `reports/sac2027_patch_v14/COXBCE_PLAN_KO_2026-10-09.md`에 기록했다(sha256 03def9d5…, 실행 로그 첫 줄).
- 결과는 두 코호트 모두 예측 세 개를 충족했다.
  - 단계 안 AUC는 Cox보다 +0.0035 [0.0015, 0.0056](Erangel), +0.0054 [0.0040, 0.0068](Taego) 높다.
  - BCE 대비 격차의 58%, 71%를 닫았다. 남은 차이 +0.0025, +0.0022는 90% CI가 ±0.005 안이지만 95% CI는 0을 포함하지 않는다.
  - 경기 안 AUC는 Cox와 실질 동등하다(+0.0015, +0.0007).
  - 회복분은 경기 수준이다. 경기–단계 평균만의 AUC는 +0.011, +0.012이고, 중심화 후 변화는 +0.0009, +0.0010이다.
- 기존 객관 통제 수치는 새 조건을 넣고 다시 계산해도 바이트 단위로 같았다(대조 128행·120행, 차이 0).
- 표 4(객관 통제)에 "Cox + BCE, next phase" 행을 추가했다. 그림 2(a)에도 Cox+BCE 행을 넣었다.

## 4. 압축과 부록
- 본문 10쪽 → 9쪽(본문은 8쪽에서 끝나고 참고문헌이 9쪽 절반).
- 보충 노트 `supplement.pdf`(3쪽)로 옮긴 것: 서식지 위험 모형(방법, 결과, 옛 그림 2), 단계별 경로 분해(옛 표 5), 목적함수×매핑 요인 분석, 동일 시도 튜닝 표(옛 표 8), 인구·이력·GATv2 통제. 새로 넣은 것: Cox+BCE 사전 계획과 90% 구간 표, 코호트별 같은 경기 쌍 비율 표.
- 옛 그림 3의 (a) 단계별 보정 패널은 뺐다. 단계 보정 오차는 표 5의 열과 본문 수치로 남는다.
- 그림 1: 맨 위 상자를 "Matched objective controls"(RQ1)로 바꾸고, 지표 상자에서 경기 안 AUC와 중심화를 강조했다.

## 5. 초록 수치
- 초록 수치를 9개에서 7개로 줄였다(3%, 13,948, 0.001, 0.006–0.008, 0.002, 58–71%, 0.6). 195단어.

## 6. 코드
- `tools/sac2027_objective_controls.py`: `cox_truncated_bce` 목적함수와 `--objectives` 인자.
- `tools/sac2027_match_offset.py`, `tools/sac2027_evaluate_controls.py`: 새 조건과 대비, 90% 구간 열(`ci90_low`, `ci90_high`).
- `tools/sac2027_paper_tables.py`: Cox+BCE 행, `--out` 인자.
- 새 도구: `tools/sac2027_figure_eval_units.py`(그림 2), `tools/sac2027_pair_share.py`(쌍 비율).
- 새 시험: `tests/test_sac2027_objective_controls.py`의 오프셋 기울기 시험(Cox 0, BCE는 기대−관측 탈락).

## 7. 익명 저장소 (2026-10-09 push: d8e3462, 미러 갱신 확인)
- 위 코드와 보충 노트를 반영했다. README 제목, 표·절 번호, 실행 순서도 고쳤다. `sac2027_figure3.py`는 지웠다.
- 신원 점검 결과
  - 공개 파일에 실명, 학교, 계정명, 메일, 절대 경로는 없다.
  - 로컬 `.git`에는 원격 URL(계정명)과 reflog(학교 메일)가 있지만, 둘 다 미러로 나가지 않는다.
  - 정리한 것: 기본 경로의 버전명(v14_review, v16_revision), `.gitignore`의 논문 폴더 구조, ADAPTATION.patch 헤더의 현지 시각.

## 8. 부록 없이 다시 (v17b, 2026-10-09)
- 사용자 지적: SAC 규정에 부록 허용이 명시되어 있지 않다. 저장소로 옮긴 내용은 심사에서 빠지는 셈이고, 본문 주장이 저장소에 기대면 분량 우회로 읽힌다. 그래서 "옮기기"를 "잘라내기"로 바꿨다.
- 보충 노트(`supplement.pdf`)는 익명 저장소에서 지웠다. 원고에서 "supplementary notes" 참조도 모두 없앴다. 파일은 `unused_supplement/`에 보관만 한다.
- 본문으로 되살린 것(압축형)
  - 서식지 위험 모형: 방법 두 문장(4.1절), 결과 세 문장(경쟁 +0.284, 남은 인원 −0.594, 깊이·경쟁의 단계 의존). 민감도 분석은 한 문장. 그림 3으로 되살렸다.
  - 경로 분해: 입력 행 (a)(b)(g)만 남긴 표 6(`tables/input_results.tex`, `sac2027_paper_tables.py --table inputs`). 목적함수·구조 단계 행은 잘랐다.
  - 목적함수×매핑 요인 분석: RQ1 결과에 한 문장(매핑 0.0108, 단계 안 0. 목적함수 0.0250, 단계 안 0.0137).
  - 인구·이력·GATv2: 방법 한 문장(4.3절), 결과 한 문장(인구·이력 최대 0.0033, GATv2 효과 전부 0.005 미만).
  - 튜닝: 표 없이 본문 수치만.
- 그림 1 압축: C·D 패널 아래 띠 제거, 범례를 위로 올림, 기호 설명을 한 줄로(Y는 패널 B 상자에 정의). 높이 9.78 → 8.96cm. 반 단을 줄이려면 패널을 덜어내거나 한 단 그림으로 다시 그려야 해서, 여기까지만 했다.
- 산문 압축: DeepHit 문장 합침, 일반·랭크 수치 표기 압축, 한계의 중복 구절, 7.1절 첫 문장, 7.2절 일반화 문장, 초록 한 문장. 4.2절의 낡은 "decomposition step" 구절과 "our-input path"도 고쳤다.
- 결과: 9쪽, 마지막 단 여백 약 148pt(13줄), 초록 191단어, overfull 0. camera-ready에서는 저작권 상자, ACM 참고 형식, 저자, 사사가 들어가 약 25줄이 더 필요하다. 그때 10쪽이 될 수 있다.

## 9. 외부 리뷰 3건 중 골라서 반영 (v17c, 2026-10-09)
### 새 분석 (저장된 예측, 재학습 없음)
- 단계 가중치를 맞춘 경기 안 AUC: 단계별 경기 안 AUC를 단계 안 AUC와 같은 가중치로 평균했다.
  - 목적함수 통제: BCE − next-phase Cox +0.0008, 0.0000 → 결론 유지.
  - 설계 비교: 우리 설계 − KRAFTON+phase가 +0.0031, +0.0027(구간이 0 포함). 쌍 가중 0.007–0.010 이득은 팀이 많은 초반 페이즈에 몰려 있었다.
- KRAFTON 원점수 중심화: KRAFTON의 cloglog 보정은 logit에서 점수와 비선형이다. 원점수 s로 중심화하면 중심화 후 이득이 0.0074, 0.0052로 더 작아진다 → 결론 유지.
- 격차 해소율의 결합 붓스트랩 CI: 58% [30, 84], 71% [55, 92]. 시드별 BCE 우위는 모든 시드에서 양수다.
- 사람 팀만 본 경기 안 이득: +0.0077, +0.0105(AI 지표 효과 아님).
- 1차 구조 비교 다섯 개의 경기 안 결과: 최대 0.0018(단계 가중 0.0034).
- 서식지 블록 제거의 경기 안 결과: 0.0159(11/11) → 서식지 기여 유지.
- top-3 짝 비교 CI: GBDT·PTR − KRAFTON+phase가 0.57 [0.01, 1.13], 0.40 [−0.09, 0.89]pp.
### 새 실험 (실행 전 예측 기록)
- 오프셋 전용 통제(`OFFSET_ONLY_PLAN_KO_2026-10-09.md`): Cox 점수를 고정하고 경기–단계 공통 항만 로지스틱으로 학습했다. 경기 안 순위는 그대로다.
  - 단계 안 +0.0027, +0.0053으로 격차의 45%, 70%를 닫았다.
  - BCE와 실질 동등 예측은 Taego에서만 맞고 Erangel에서는 틀렸다(90% 상한 0.0056).
- 9개 코호트 반복(`OBJECTIVE_REPLICATION_PLAN_KO_2026-10-09.md`): 예측 7개가 모두 맞았다.
  - BCE − Cox: 경기 안 +0.0015, 단계 안 +0.0086
  - Cox+BCE: +0.0062, 격차의 약 72%
  - truncation: 경기 안 +0.0120
- 체크포인트를 각 목적함수의 검증 손실로 고른 결과(11개 코호트): BCE − Cox가 경기 안 −0.0002, 단계 안 +0.0090(단계 안 AUC 기준 +0.0077보다 큼), Cox+BCE는 단계 안 +0.0067이다. 선택 기준이 격차를 부풀리지 않았다.
### 원고
- 4.3절: 결합 손실 정규화 식, 위험집합은 경기–단계 안에서 구성, 오프셋 전용 통제 식, 선택 기준 두 가지, 9개 코호트 반복.
- 5.1절: 경기 안 AUC의 쌍 가중 명시와 단계 가중 점검, A_B 표기 정리, top-3에서 생존 팀 3팀 이하일 때 전원 포착되는 점 명시.
- 표기·정의
  - 식 (2)는 개념형이고 별도 절편이 아님, 우승 정확도·C-index 정의, GBDT 구현(scikit-learn, 31 leaves)
  - 표 2 "scheduled phase length"(실제로 예정 길이였다. 누수 아님), d=0, required speed
  - 밴드 SD 척도 두 가지, 표 4 KRAFTON 이름 통일, RE와 산술평균 차이 각주, 그림 2(b) †로 합성 행 표시
- 참고문헌: GAUC(Zhou et al. 2018) 추가, Xenopoulos21 DOI·쪽 추가(2021 학회본, arXiv 2207은 이후 게시라 정상).
- 1차 비교 시점 문구를 5.2절과 같게 통일했다("patch-split 모델 학습 전").
- 일반화 범위를 줄였다(7.2절 "다른 분야에서 목적함수 비교가 뒤집히는지는 미검증").
- 한계: 다음 구역 미포함 정보 집합, 코호트 간 비독립, 가설과 검증이 같은 테스트 패치.
- 지면: 표 3(선행 설계 비교표)과 목적함수×매핑 요인 분석을 뺐다. 표 2 행을 합치고 여러 문장을 압축했다. 9쪽.
### 반영하지 않은 것과 이유
- 새 패치 수집·다음 구역 입력: 데이터 재수집·재구축이 필요해 마감 전 불가. 한계로 적었다.
- GLMM·conditional logit·frailty, DeepHit, 단독 MLP 기준선: 핵심 주장과 거리가 있고 지면이 없다.
- 관측 기반 자격 민감도: DB 재구축이 필요하다.
- 표본별 예측·라벨 공개: 데이터 비공개 원칙(사용자 결정).
- 다른 백본, 결합 가중치 스윕, 계층 붓스트랩: 9개 코호트 반복과 시드별 범위로 일부 대체했다.
- 사전 기록의 커밋 해시: 소급할 수 없다.
- 서식지 기여 강등: 경기 안에서도 0.0159 기여가 있어 유지했다.
- 초록을 숫자 2개로 줄이는 것: 사용자가 정한 초록 구조를 유지했다.

## 10. 재리뷰(N1–N5) 반영 (v17d, 2026-10-09)
- N1: 경기 안 AUC의 주 추정량을 쌍 풀링으로 정하고 근거를 적었다(경계에 팀이 많을수록 비교할 쌍이 많다). 경계별·단계 가중은 점검용이다. 설계 비교는 "주 단위에서는 gain, 초반 페이즈에 몰리고, 경계별·단계 가중으로는 판정 불가"로 고쳤다. "largely vanish"는 지웠다.
- N2: KRAFTON 중심화를 원점수 s 기준으로 주 결과에 올렸다. 손실이 s 이동에 불변이고 cloglog p = γ_k + s이기 때문이다(그림 2(b)도 원점수 기준).
  - 표준화(SD로 나눔) 중심화를 추가했다: +0.0028, +0.0023. 단계 가중 경기 안 이득과 같아서, 중심화 뒤 남던 몫은 경기별 점수 분산이다.
  - "removes any match-level offset"은 "removes any additive offset on that scale"로 좁혔다.
- N3: 예측 7개 표(표 4)를 넣었다. 기준, 추정치, 90% CI, 판정이 들어간다. 오프셋 전용 통제는 두 코호트에서만 돌렸다고 4.3·6.1·7.1절과 결론에 나눠 적었다.
- N4
  - 초록과 결론에 "nine further cohorts of the same test patch"로 범위를 붙였다.
  - 7.2절 "reverses"를 "changes"로 바꿨다.
  - 1절 "pooled" 주장에 근거를 달았다(Hodge21, Katona19, Marshall22). KRAFTON은 전 구간 경기 안 일치도를 보고한다고 구분했다.
  - 결론의 "usual"을 지웠다.
- 초록: 다시 썼다. Cox truncation을 풀어 설명하고 숫자를 3개 안팎으로 줄였다(200단어).
- 표 6(1차 결정) 캡션에 AUC 종류를 적었다.
- 지면: 표 5(입력 대비 3행)를 본문 문장으로 옮겼다(숫자 유지). 6장 첫머리의 AUC 해석 문장 등 몇 문장을 압축했다. 9쪽, 여백 약 4줄.
- 표 번호: 1 코호트, 2 입력, 3 목적함수 통제, 4 예측 7개, 5 주 결과, 6 1차 결정.
- 넣지 않음
  - N5(개발 단계 데이터 사용 공개): 무작위 분할을 언급하지 않는다는 사용자 결정 때문.
  - entry congestion·20초 규칙 근거, 3팀 이하 경계 제외 top-3, 설계 비교 예측구간: 지면 부족.

## 11. 그림 2 재설계 (v17e, 2026-10-09)
- 사용자 지적: 평가 단위를 색·모양·세로 엇갈림에 맡김, 패널 간 x축 범위·뺄셈 방향 불일치, 질문 세 개와 모집단 두 개가 한 패널에 섞임, SESOI와 단계 가중 within-match가 없음.
- 새 그림: 열 = 평가 단위(단계 안, 중심화, 경기 안, 단계 가중 경기 안), 행 = 대비. 색 하나, 범례 없음, 회색 띠 ±SESOI, 네 열 모두 x축 범위가 같다. 크기는 원본과 같다(17.8 × 5.84cm).
  - 위 그룹: BCE − next-phase Cox(Erangel, Taego, 나머지 9개 합성).
  - 아래 그룹: PTR·GBDT − KRAFTON+phase(11개 합성, KRAFTON은 원점수 중심화).
- 빈 점은 없다. 두 코호트의 단계 가중 CI, 9개 코호트의 중심화·단계 가중 CI를 모두 부트스트랩·합성 결과로 채웠다.
- 뺀 행: horizon(full Cox), Cox+BCE(표 3·표 4), team axis와 PTR − GBDT(표 6·본문).
- 캡션은 제안 문안을 따랐다. 본문의 "그림 2a/2b" 참조는 "표 3", "그림 2 위·아래"로 바꿨다.
- 그림 읽는 법: 캡션에 네 열의 뜻(단계 안, 경기–단계 평균 제거, 경기 안, 단계 가중 경기 안)과 "왼쪽에서 띠 밖이던 점이 오른쪽으로 가며 띠 안에 들어오면 경기 수준 차이"를 넣었다. 본문 6.1절에서 그림을 처음 가리키는 곳에도 "열이 오른쪽으로 갈수록 같은 경기의 팀끼리 비교에 가까워진다"를 넣었다.
- 지면: 그림과 겹치는 CI 일부, 7.2절 문장, GATv2 언급(효과가 모두 SESOI 미만인 사후 탐색), 3.1절 패치 설명을 줄였다. 9쪽.


## 12. 다듬기 리뷰 반영 (v17f, 2026-10-09)
- 초록(200단어 유지)
  - 둘째 문장은 제안 문안을 따랐다. 한도 때문에 "within a phase fewer than 3% of its pairs"만 "fewer than 3% of its within-phase pairs"로 줄였다.
  - top-3 문장도 제안 문안을 따르되 "amounts to"를 "is"로 줄였다.
  - 나머지 문장에서 몇 단어를 덜었다("When the Cox loss is restricted" → "Restricted", "game patch" → "patch").
- 7.2절: "mostly compares groups" → "mostly compares members of different groups".
- 5.1절: "Within-phase AUC therefore compares almost entirely teams from different matches."로 고쳤다. 점검 추정량 문장은 "as checks we average AUCs per boundary and, with equal phase weights as in within-phase AUC, per phase"로 바꿨다.
- 그림 2
  - 네 열 모두에 영점선(실선)을 넣었다. 0 위치의 격자선은 빼서 두 선이 겹쳐 보이지 않게 했다.
  - 넷째 열 제목: "Within match, phase-weighted" → "Within match, equal phase weights".
  - 같은 표현을 5.1절, 6.1절("also with equal phase weights"), 6.3절("with equal phase weights"), 캡션, 대체 텍스트에도 맞췄다.
  - 눈금은 이미 0, 0.01, 0.02, 0.03이라 그대로 뒀다.
- 표 4: 판정 규칙(>0은 95% CI 기준)과 표의 90% CI 열이 어긋나던 문제를 노트로 풀었다. ">0: 95% CI above zero (in row order, [0.0056, 0.0115], [0.0042, 0.0082], [0.0071, 0.0169]); ≈0: 90% CI within ±0.005". 표의 모든 값은 `objective_all_cohorts/pooled.csv`(nine_new, within_phase_auc 선택)와 대조했다.
- 소소한 것
  - 표 6 음수 부호: 하이픈 → `$-$`.
  - 표 2: 첫 열을 왼쪽 정렬(`>{\raggedright\arraybackslash}p{1.45cm}`)로 바꿔 "Habitat con-text" 줄바꿈을 없앴다.
  - 표 1·2 캡션을 다른 표처럼 문장형 대소문자로 바꿨다.
  - 7.1절 BCE 오프셋 미분: p̂ → λ̂(식 (1)–(2)와 같게).
  - entry congestion 정의를 표 2에 넣었다: "teams in the target zone at entry".
  - 20초 부활 규칙 근거를 3.3절에 넣었다: "a record just after a kill can carry the last health; observed revives come at least 34 s after it". 근거는 `model_stgcn/phase_v2/alive.py` 주석이다. 처치→체력 기록 66,741쌍 중 1건이 처치 1.6초 뒤 체력 10을 보였고, 실제 부활은 최소 34초 뒤였다.
  - GATv2: 원고에는 넣지 않았다. 익명 저장소 NOTES.md에 결과를 남겼다(모든 효과가 SESOI 미만이고, 참조 결과를 본 뒤 추가한 분석이라는 점 포함). README에는 "not reported in the paper"로 표시했다.
- 지면
  - 2.1절의 적응 문장은 1절과 4.2절에 같은 내용이 있어 뺐다. Sagong25a 인용은 1절과 3.1절에 남아 있다.
  - 7.3절 "by at most 0.6 pp"는 7.2절에 같은 수치가 있어 "differ little"로 줄였다.
  - 표 4 노트를 두 줄로 줄였다.
  - 결과: 9쪽, overfull 0.
- 그림 스크립트(`tools/sac2027_figure_eval_units.py`)는 익명 저장소에도 복사했다(신원 단어 검사 통과).

## 13. 앞부분 재구성: 평가 논문 틀 (v18, 2026-10-09, `v18_revision/`)
v17f는 `v17_revision/`에 그대로 두고, v18은 새 폴더에서 고쳤다. 증거와 결론 문장은 그대로 두고 순서와 틀만 바꿨다.

- 사용자 지적 네 가지와 대응
  1. 메커니즘이 토론(7.1절)에 있었다. → 새 2절 "Why the Evaluation Unit Can Decide the Comparison"으로 옮겨 데이터·방법보다 앞에 뒀다.
  2. 지표 쪽 불변성이 문장으로 없었다. → 1절 셋째 문단과 2절 "Metrics" 문단에 문장으로 썼다. 경기 안 AUC는 경기 상수를 무시하고, 단계 안 AUC는 그 상수가 경기 위험을 따라가면 보상한다.
  3. 세 방향의 검증이 흩어져 있었다. → 2절 "Predictions" 문단에서 셋을 묶었다(지표 쪽 중심화, 손실 쪽 Cox+BCE, 학습 뒤 오프셋 전용). 6.1절도 같은 순서로 썼다("On the metric side / On the loss side / After training").
  4. 모델 논문처럼 읽혔다.
     - "our designs"를 "reference designs"로 바꿨다(GBDT, PTR-Former, 정규화 로지스틱).
     - 표 6 행 이름: "Ours, ..." → "Reference ...".
     - 그림 1: 'ours' 강조색과 범례를 지웠다. "Our design"을 "Reference designs"로, 서식지 상자를 "Habitat and team inputs"로 바꿨다. RQ3 꼬리표를 지우고, 아래 상자는 "Design contrasts"(RQ2 쌍 아래 괄호)로 바꿨다. 목적함수 통제 상자에 offset-only를 넣었다.
- 새 절 구성: 1 서론(용도, π, 손실–지표 맞물림 한 문단, 기여) → 2 원리(관련 문헌을 먼저 쓰고, 쌍 분해·손실 미분·지표 불변성 세 문단, 단위별 예측 표 1, 예측과 사전 기록 범위) → 3 데이터 → 4 설계와 예측별 통제 → 5 평가 단위·통계 → 6.1 예측 검증 → 6.2 단위별 설계 비교 → 7 함의와 한계 → 8 결론.
- RQ는 둘로 줄였다(사용자 문안 그대로).
  - RQ1: 평가 단위가 risk-set과 cross-entropy 목적함수 비교를 정하는가, risk-set 손실의 이동 불변성이 그것을 설명하는가.
  - RQ2: 목적함수와 입력이 다른 설계 비교도 단위에 따라 같은 식으로 바뀌는가.
- 표 1(새 표): 단위별로 경기 상수 c를 보상하는지, Cox와 BCE가 c를 배우는지, BCE − Cox_next의 예측 부호를 적었다. 그림 2 위쪽 행은 이 표의 확인이고, 표 5(예측 7개)는 사전 기록의 증거다.
- 관련 문헌 절(옛 2절)은 없앴다.
  - 조건부 우도, ListMLE, Bai et al., DeepHit, Heagerty–Zheng, van Klaveren, GAUC는 2절 첫 문단으로 옮겼다. "재료는 알려져 있고, 우리는 그것들이 평가에서 맞물리는 방식을 시험한다"로 시작한다.
  - e스포츠 예측 문헌은 1절로, 이산시간 생존 모형은 3.4절로, 다중 에이전트 구조는 4.2절로 옮겼다.
- 옛 RQ3 정리
  - team axis(단계 안 +0.0108, 경기 안 +0.0011)는 6.2절 RQ2 사례로 옮겼다.
  - 입력 대비는 단계 안·경기 안 수치로 바꿔 "입력 차이는 두 단위에서 같다"는 대조 사례로 썼다. Sagong 입력 → 우리 입력 +0.106/+0.101, 서식지 제거 0.0151/0.0159, KRAFTON 입력 대비 −0.007/−0.012, 결합 +0.019/+0.020. CI는 `primary_results_v2` 결과와 대조했다.
  - 서식지 연관 모델(옛 4.1절 끝 문장, 옛 6.3절 문단, 그림 3)은 지웠다. 부록을 쓰지 않는다는 원칙에 따라 옮기지 않았다. 한계의 "Habitat associations are observational."도 지웠다.
- 사전 기록 범위(과장 방지)
  - 2절 끝: 설명은 두 최대 코호트에서 BCE의 단계 안 우위와 horizon 효과를 본 뒤 세웠다. 사전 기록은 결합 목적함수, 오프셋 전용, 9개 코호트 예측에만 해당한다.
  - 6.1절 첫 문장: "Two results in the two largest cohorts came before the explanation." 그 코호트의 중심화는 "a diagnostic in these two cohorts"로 적었다.
  - 오프셋 전용의 두 예측도 실행 전에 기록했다(계획 해시와 로그 첫 줄 일치, launch 2026-10-08T19:27:17Z). 결과는 그대로 적었다: 이득은 두 코호트 모두 성립, BCE와의 동등성은 Taego만 성립.
  - 2절에서 RQ2 함의는 "examine descriptively"로 적어 사전 예측처럼 보이지 않게 했다.
- 초록: 원리 문장으로 시작한다(risk-set 손실은 집단 상수를 못 배우고, 집단 안 AUC는 그 위험을 보상하지 못하며, 집단을 합친 AUC는 보상한다). 설계 비교 문장에 "입력 이득은 경기 안에서도 유지"를 넣었다. 200단어.
- 지면: 9쪽, overfull 0. 마지막 쪽 오른쪽 단에 여백이 있다.
- 익명 저장소 README의 코드–원고 대응 열을 v18 번호로 바꿨다(표 3–7, 그림 2, 2절). 서식지 위험 모형 도구는 "not reported in the paper"로 표시했다. 커밋은 하지 않았다.

## 14. v18 골격 유지, 입력 설계 복원, 결과 재배치 (v19, 2026-10-09, `v19_revision/`)
v18(3acm)을 골격으로 두고 v17f(2acm)에서 입력 설계 설명을 되살렸다. v17과 v18 폴더는 그대로 있다.

- 초록: 과제로 시작한다(경기 검토, 같은 경기의 팀 비교). 이어서 같은 경기 쌍이 3% 미만이라 경기 간 위험 차이가 경기 안 판별과 함께 섞인다고 쓰고, 손실 성질로 넘어간다.
  - 결합 손실은 "within-match AUC stays practically equivalent"로 썼다.
  - 서식지 입력 이득은 경기 안에서도 유지되고, team attention 이득은 0.011에서 0.001로 줄어든다.
  - top-3는 약 0.5pp. 197단어.
- 서론
  - 1문단: 과제와 사용 상황.
  - 2문단: 단계 안 AUC는 경기 안 판별과 경기 간 위험 차이를 함께 담는다. "경기만 순위화한다" 식 표현은 쓰지 않았다.
  - 3문단: 시간 범위와 학습 신호를 나란히 쓰고, 시간 범위를 먼저 맞춰야 한다고 썼다.
  - 4문단: 통제 논리와 공통 입력(구역 노출·이동·국소 경쟁 + 팀 상태) 예고.
  - RQ1: 입력·백본을 고정하고 시간 범위를 맞춘 뒤, risk-set과 BCE의 차이가 평가 단위에 따라 어떻게 달라지고 경기 공통 점수 항이 얼마나 설명하는가.
  - RQ2: 입력·구조·설계 전체의 이득이 경기 안 평가에서도 남는가, 검토 지표로는 얼마나 큰가.
  - 기여에서 "derive"를 뺐다.
- 2절 "Evaluation Units and Learning Signals"
  - 순서: 비교 대상 정의 → 지표 성질 → 손실 성질 → 시간 범위(상대 위험 대 구간 확률, 2acm 2.2절 핵심) → 조건부 가설 H1–H4 → 연구 순서.
  - 표 1은 공통 이동 c에 대한 성질 표로 바꿨다(경기 안 AUC 불변, 단계 안 AUC는 다른 경기 쌍으로 변함, 중심화 후 불변, risk-set 손실 ∂ℓ/∂c=0, BCE ∂ℓ/∂c=Σ(λ̂−Y)).
  - 예측 부호는 표에서 빼고 가설 문장으로 옮겼다. "성질만으로 부호가 정해지지 않는다"를 명시했고, RQ2는 방향 예측 없이 묻는다.
- 4절 "Input Representations and Reference Designs"
  - 4.1 "Domain-Informed Input Representations"는 동기 → 정의 → 실험상 역할 순서다.
    - 같은 인원·체력·피해 이력의 두 팀이 구역 위치·이동·주변 상대에서 다를 수 있다는 과제 논리로 시작한다.
    - 선행 입력에도 구역 거리와 주변 적이 있다고 밝혔다(최초 도입이라고 주장하지 않음).
    - 생태학은 차원을 고른 동기로만 둔다.
  - 2acm 정의를 복원했다.
    - outside/inside depth.
    - movement residual: 사후적 속도 차이이자 경험적 대리 변수. τ_k는 남은 시간이 아니고 ℓ은 우회를 포함하며, d=0이면 움직인 구역 안 팀은 음수다. target zone은 cutoff 전에 이미 보인 구역이다.
    - 국소 경쟁: 100/200/500m, 3-D 중심점 거리, 마지막 생존 격자점(코드로 확인).
    - 생존 인원·체력은 team status 블록이다.
  - 역할: 서식지 블록은 팀별 상태와 경기 환경을 함께 담는다. RQ1은 같은 입력을 주고, RQ2는 블록을 넣고 뺀다. 블록 제거는 44개 변수의 제거이지 모든 공간 정보의 제거가 아니다.
  - 표 3 서식지 행을 exposure / movement / competition으로 정리했다.
  - 4.2 "Reference and Adapted Designs": reference designs를 입력·구조 대비의 실험 도구로 위치 지었다.
- 5절 "Experimental Design and Evaluation Protocol"
  - 5.1 Matched Comparisons: 옛 4.3절. 오프셋 전용의 a에는 제약이 없지만 코드가 a>0을 검사하고, 여섯 번 적합 값은 0.99–1.16이라 경기 안 순위가 Cox와 정확히 같다.
  - 5.2 Metrics, Aggregation, and Diagnostics: 평가 단위는 단계 안·경기 안 두 비교 집합이다. 중심화는 점수 변환(진단), 동일 단계 가중은 집계 점검으로 구분했다.
  - 5.3 Statistical Decisions and Model Selection: 옛 Training 절을 합쳤다.
- 결과
  - 6.1(RQ1) 순서: 시간 범위 정렬 → 두 단위 비교(동일 단계 가중 점검을 바로 옆에, H1·H2) → 중심화는 진단이며 원인을 확정하지 않는다(H3) → 결합 손실(전체 학습 변경, 경기 안 AUC 실용적 동등)과 오프셋 전용(고정 점수, 경기 안 순위 보존; Taego만 BCE와 동등)(H4) → 9개 코호트와 선택 기준 민감도(표 5).
  - 6.2(RQ2) 입력·구조 대비: 새 표 6(입력 4행, 구조 2행, 95% CI)을 넣고 입력을 먼저 쓴다.
    - 서식지 제거 0.0151/0.0159.
    - Sagong 입력에 서식지만 더하면 overall AUC +0.046. overall임을 밝혔고 몫으로 나누지 않았다.
    - KRAFTON 입력보다 낫지 않다(−0.007/−0.012). 결합하면 상보적이다.
    - 다음으로 team attention, PTR − GBDT(−0.0009 [−0.0057, 0.0039]).
    - 1차 결정은 사전에 정한 규칙 그대로 두고, 경기 안 수치는 "별도의 이후 분석"으로 구분했다.
  - 6.3(RQ2) 설계 전체와 검토 효용: 표 8을 여기로 옮겼다.
    - top-3는 "최대 0.6pp라 같다"가 아니라 크기와 불확실성으로 썼다(GBDT 0.57 [0.01, 1.13], PTR 0.40 [−0.09, 0.89]).
    - 보정과 우승 예측은 "목표에 따라 강점이 다르다"로 묶었다.
- 그림
  - 그림 1: 입력 상자는 "Habitat context + team state"로 두 줄을 나눴고, 괄호 아래 상자는 "Input and structure contrasts"다.
  - 그림 2 캡션: 첫째·셋째 열은 평가 단위, 둘째는 진단용 점수 변환, 넷째는 집계 점검이다. "왼쪽에서 오른쪽으로 원인이 제거된다" 식 설명은 지웠다.
- 7–8절
  - 7.1: "평가 단위 맞추기와 도메인 상태 표현은 별도 점검"을 넣었다. top-3는 0.4–0.6pp로 썼다.
  - 7.2: 서식지 블록 제거는 44개 변수의 조건부 가치이지 생태 메커니즘이나 변수별 가치가 아니라고 적었다.
  - 결론을 줄였다.
- 넣지 않은 것과 이유
  - 서식지 연관 모형과 옛 그림 3: 리뷰는 보충자료를 권했지만, 부록을 쓰지 않는 원칙에 따라 원고에는 넣지 않았다. 익명 저장소 `NOTES.md`에 전체 결과를 남겼다(경쟁·인원뿐 아니라 깊이의 단계 의존성, AI 보정 후 이동 잔차 약화 포함, 예측용 정규화 로지스틱과 구분).
  - 피처 정의 부록: 같은 이유로 넣지 않았다. 원고 4.1절에 핵심 정의를 두고, 전체 정의는 저장소의 피처 manifest가 담당한다.
- 지면: 9쪽, overfull 0.
  - 줄인 것: 표 8 recipe 두 행, 기여·결론 압축, 그림 1·2 캡션 압축, 3.2절 검증 집합 크기 이유, 정상·랭크 경기 문장.
- 익명 저장소: README 대응 열을 v19 번호로 바꿨다(표 6–8, 5.2절, 6.3절). `NOTES.md`에 서식지 연관 모형 항목을 추가했다. 커밋하지 않았다.

## 15. 해석·보고 범위 정리 (v19b, 2026-10-09)
v19a(4acm)는 `archive/main_v19a.tex`, `archive/main_v19a.pdf`에 보관했다. 목차는 그대로다.

- 2절 가설: 기록된 규칙에 맞춰 H0–H4 다섯 개로 다시 썼다.
  - H0 horizon, H1 경기 안 동등, H2 단계 안 우위, H3 중심화 후 동등, H4 결합 손실·오프셋 전용이 BCE와 실용적 동등(경기 안 AUC는 Cox_next와 동등).
  - "H0–H3은 두 최대 코호트에서 관측한 뒤 세웠다. 그곳의 H4 예측과 9개 코호트의 다섯 예측은 실행 전에 기록했다"로 순서를 밝혔다.
  - 표 5에 H 열을 넣고 행을 H0→H4 순서로 바꿨다(각주의 95% CI 순서도 맞춤).
- 사실 수정
  - 1절 넷째 문단: "All designs share inputs" → "The reference designs and objective controls share inputs".
  - 6.3절: +0.0252는 winner accuracy, +0.0065는 C-index 차이다(v14 원고로 확인, 지표 이름을 복원).
- 기여 1: "두 알려진 불변성을 엮어 평가 단위가 목적함수 비교를 정하는 조건을 시험 가능한 설명으로 만들었다"로 바꿨다(within-phase AUC는 "can reward").
- RQ2와 메커니즘 연결: 7.1절에 team-axis attention은 match–phase의 팀을 모아 경기 공통 항으로 가는 통로가 된다는 해석을 넣었다(예측하지 않았음을 명시).
  - 입력 대비는 팀별 상태를 더해 두 단위에서 유지된다.
  - 권고는 "그룹을 가로질러 모으는 구성 요소의 이득은 그룹 안에서 확인하라"이다. 결론도 같은 취지로 바꿨다.
- 해석 정리(두 번째 리뷰)
  - 5.2절에 "중심화는 진단용 점수 변환이지 AUC의 가산 분해가 아니다"를 복원했다.
  - team attention의 0.0049는 "within-phase 이득 중 경기 수준 몫"이 아니라 별도 진단(match–phase 평균 점수만의 AUC, `structure_match_offset`의 `match_mean_between_match`)으로 바꿔 썼다. 중심화 후 값은 0.0043이다.
  - 6.1절: "sits in the match–phase score level" → "depends on match–phase score levels". 결합 손실의 "recovered gain lies in the means" → "gain appears in the AUC of match–phase mean scores".
  - 그림 2 캡션: "다른 경기 쌍에서 온 차이"라는 단정을 "열 간 차이는 비교 집합·변환·집계를 반영하며, 그것만으로 원인을 가리지 않는다"로 바꿨다.
  - 보정 권고는 "pointwise 항이나 사후 그룹 보정은 확률 정확도·보정 지표로 평가할 후보"로 낮췄다. risk-set 손실의 신호 주장은 2절과 같은 범위(독립 경기 공통 항, 직접 신호)로 맞췄다.
  - "9개 코호트에서 성립" → "9개 코호트 합성 추정치가 일곱 예측을 지지"(초록·기여·6.1절·결론·표 5 캡션).
  - 용어 구분
    - practical gain: 단측 p<0.05이고 점추정치 ≥ SESOI.
    - positive improvement: 95% CI가 0 위.
    - practical equivalence: 90% CI가 ±SESOI 안.
    - 오프셋 전용: Erangel의 Cox 대비 개선은 양수지만 practical-gain 기준 미만이라고 적었다. 결합 손실은 "두 코호트 모두 양의 개선, practical gain은 Taego만"이다.
  - 표 1: within-phase AUC "changes" → "can change".
  - top-3 범위: "complete designs 약 0.5pp" → "reference designs가 phase-adjusted KRAFTON보다 0.4–0.6pp"(초록·기여·결론).
- 오프셋 전용 정보 집합(두 번째 리뷰 5절)
  - 확인 결과: 요약값 z(평균 점수, 위험집합 크기, 평균 입력)는 b 시점의 후향적 위험집합 R 위에서 계산된다.
  - PTR과 KRAFTON 백본의 attention 키는 cutoff 시점에 살아 있던 팀이다(`model_stgcn/ptr/data.py` key_mask, adaptation `key_mask_of`).
  - R의 크기가 cutoff 생존 팀 수보다 작은 match–phase가 두 코호트에서 19.5–22.4%다(10초 창 안의 탈락).
  - 사후 민감도(`tools/sac2027_offset_only_cutoff.py`, 결과 `control_evaluation/offset_only_cutoff/`):
    - cutoff 생존 팀 위의 입력 평균 + 그 수 + 원 Cox 점수: Erangel +0.0021 [−0.0004, 0.0043], Taego +0.0044 [0.0031, 0.0059]. 격차의 34%, 58%(기록된 통제는 45%, 70%).
    - 위험집합 크기만 빼면 +0.0022, +0.0047. 경기 안 상수 입력 25개만 쓰면 +0.0004, +0.0008.
    - 6.1절과 기여에 사후 분석으로 한 문장씩 넣었다. Erangel에서는 cutoff 정보만으로 양의 개선이 지지되지 않는다.
- 그림 2: 아래 그룹 이름을 "Reference design − KRAFTON+phase"(두 줄)로 바꿨다.
- 지면(9쪽 유지)
  - 줄인 것: 튜닝 문장, 한계의 서식지 문장(4.1·6.2절과 중복), Sagong CI, 보정 문장, 그림 2 캡션 끝 문장, 사람 팀 전체 AUC 문장, combined-input 문장, 식 (2) 설명 문장, 7.3절 6pp 문장.
  - 인용 6개를 뺐다: Zheng15, LeeLee25, Xenopoulos21, Cawley10, DynamicDeepHit20, Gensheimer19.
- 확인만 하고 고치지 않은 것
  - 오프셋 전용 사전 기록: 계획 파일은 2026-10-09 04:26:18 KST에 만들어진 뒤 수정되지 않았다. 로그는 04:27:17에 만들어졌고 첫 줄에 계획 파일과 같은 sha256과 launch UTC가 있다. 첫 결과는 04:29:27에 나왔다. git 커밋은 없다(reports/는 .gitignore). Cox+BCE와 9개 코호트 계획도 같은 방식이고 역시 커밋은 없다. 같은 근거라 문장을 유지했다.
  - N5(패치 분할 이전 개발): 10/9 사용자 결정(무작위 분할 언급 안 함)을 유지했다.
  - 예측·라벨 산출물 공개: 코드만 공개한다는 원칙 때문에 하지 않았다.
- 익명 저장소: 새 도구와 그림 스크립트를 복사했다. README에 행을 하나 추가했고, NOTES에 민감도 항목을 추가했다. 커밋하지 않았다.

## 16. 5acm 리뷰 두 건 반영 (v19c, 2026-10-09)
v19b는 `archive/main_v19b.tex`, `archive/main_v19b.pdf`에 보관했다.

- offset-only 정보 집합
  - 6.1절의 "Its summaries omit..." 문장을 바꿨다. 요약값이 R 위에서 계산돼 cutoff 뒤 10초 동안의 탈락 정보를 쓰고, 이는 식 (1)의 조건 집합 밖이며 표 4의 다른 조건은 받지 않는 정보라고 썼다.
  - cutoff 정보만 쓰는 변형: 34%와 58%. Erangel은 95% CI가 0을 포함해 positive-improvement 규칙에 실패한다. BCE 대비 90% 상한은 0.0061(실패), 0.0048(Taego만 동등).
  - 형태와 정보를 나누려고 사후 변형을 하나 더 돌렸다(`raw_risk_set`: 같은 원점수 형태 + R 요약). 결과는 40%와 60%이므로, 45–70%에서 34–58%로 줄어든 것은 형태(별도 평균 점수 항)와 정보 집합 둘 다의 몫이다.
  - 5.1절에 변형의 정의를 넣었다: 원점수 s를 쓰고 별도 평균 점수 항이 없다(10초 창 안에 탈락한 팀의 점수는 저장돼 있지 않음). 요약은 cutoff에 관측된 생존 팀의 수(log)와 입력 평균이고, 적합은 검증 자료로 한다. 점수 가중치는 모두 양수(0.98–1.16)다.
  - 기여 2: cutoff 정보 기준(34–58%)을 앞에 두고, 45–70%는 "기록된 형태이며 예측 시점까지의 생존 정보도 씀"으로 괄호에 넣었다.
- top-3 표현: 초록·기여 3·결론을 "0.4–0.6pp 위이며 PTR-Former의 95% 구간은 0을 포함"으로 바꿨다(제안 문안). 6.3절의 "about a tenth"는 "under a tenth"로 바꿨다(0.57/6.8, 0.40/6.5).
- 가설: H4를 H4a(Cox+BCE)와 H4b(offset-only)로 나눴다. "9개 코호트에서 H0–H3과 H4a"로 범위를 바로잡았다. 표 5의 H4는 H4a로, 6.1절은 "(H4a, H4b)", "H4b holds in Taego only"로 바꿨다.
- 사소한 것
  - 식 (2) 설명을 복원했다(두 학습기가 phase index를 입력으로 받아 α_k를 표현하고, 별도의 단계별 절편은 없다).
  - 그림 2 캡션에 "Pooled rows use random effects."를 되살렸다.
  - 6.3절: practical-gain 규칙은 충족하지만 PTR의 95% 하한(0.0030)이 SESOI 아래라는 구절을 붙였다.
  - 사람 팀 평가 문장을 "AI와 사람 대상 팀의 구분만으로 설명되지 않는다"로 좁혔다.
- top-3 정의(두 번째 리뷰 4.3): 5.2절에 식을 넣었다. 탈락이 있는 경계들에서 상위 세 팀이 잡은 탈락 수의 합 ÷ 전체 탈락 수의 합이고, 탈락 수로 가중한 비율이다(코드 `TopK`로 확인).
- 결론: 결합 손실(전체 재학습, 58–71%)과 고정 점수에 대한 cutoff 정보 공통 항(34–58%, Taego만 지지)을 나눠 쓰고 "함께 경기 공통 점수 정보의 역할을 지지한다"로 묶었다.
- N5: 리뷰 규칙(쓰였으면 7.2절에 한 줄)에 따라 7.2절에 공개했다. BigComp 단계에서 같은 수집분을 경기 단위 8:1:1 무작위 분할로 썼고, 41.2 경기가 포함됐다. 문장: 입력과 구조는 41.2 경기를 포함한 이전 무작위 분할에서 개발돼 설계가 시험 패치 자료를 반영했을 수 있으며, 패치 분할·판정 규칙·1차 비교는 보고한 모델 학습 전에 고정했다. 10/9의 "언급 안 함" 결정은 이 지시로 대체했다.
- 지면(9쪽)
  - 그림 2 높이: 5.84 → 5.2cm.
  - 표 8에서 combined-input 행을 뺐다(표 6에 같은 대비가 있음).
  - 문장 압축: cutoff 예시, 튜닝 설명, 7.1절 반복 수치, 표 3 각주 한 문장, 보정 문장, 결론 첫 문장, 모집단·이력 통제 설명 합침, 체크포인트 민감도 괄호화, 6.3절 combined 53.3%.
  - 인용 2개를 더 뺐다: KagglePUBG, Bergstra12.
- 익명 저장소: 도구(변형 4개)와 그림 스크립트를 복사했다. NOTES와 README를 갱신했다. 커밋하지 않았다.
- (같은 날 추가) N5 문장 삭제: 사용자 요청으로 7.2절의 무작위 분할 문장을 지웠다. 평가는 다시 패치 홀드아웃으로만 서술한다. 원고, 표, 익명 저장소 README·NOTES에 무작위 분할 언급이 없음을 확인했다. 9쪽, overfull 0.

## 17. 6acm 리뷰 두 건 반영 (v19d, 2026-10-09)
v19c는 `archive/main_v19c.*`에 보관했다.

- 입력 결합 설계 복원
  - 표 8에 "Both input sets combined (more information)" 블록(GBDT, ours + KRAFTON inputs)을 되살렸다.
  - 6.3절에 top-3 문장을 넣었다: 53.3%, reference GBDT 대비 +1.46 [0.92, 2.00]pp, 10/11(`sesoi_top3/pooled.csv`의 Union_minus_GBDT). 공통 입력에서는 모델을 바꿔도 capture가 거의 움직이지 않고, 입력 집합을 넓히면 오른다고 썼다.
  - 7.2절의 "learned designs"는 "reference designs using the shared reference inputs"로 좁혔다.
- 결론 첫 문장 복원: "Within-phase AUC on samples pooled over matches mixes within-match discrimination with between-match risk."
- 형태·정보 효과를 수치로: "같은 축소 형태에서 위험집합 요약은 40%, 60%를 메운다. 점추정치로 보면 정보를 cutoff로 제한하면 6·2포인트, 평균 점수 항을 빼면 5·10포인트 낮아진다." 기여 2에 "reduced form"을 넣었다.
- 초록
  - 경기별 위험집합 조건을 복원했다("When risk sets are formed within each match–phase, Cox-type losses receive no direct training signal ...").
  - 서식지 설명을 세 차원으로 되돌렸다.
  - 단어 수 맞춤: "under 3%", "between-match risk mixes with within-match discrimination", "per match–phase", "above a phase-adjusted survival design's". 199단어.
- 6.2절 통제 문장에 prefix shuffle을 되살렸다(0.0033 안).
- N5: 사용자 결정대로 넣지 않았다.
- 선택 사항(cutoff 생존 팀으로 평균 점수 항까지 계산한 변형)은 하지 않았다.
- 지면(9쪽)
  - 그림 2 높이: 5.2 → 4.95cm.
  - 압축: 학습 설정, 1절 "because zones, team counts, and fights differ", offset-only 판정 문장, 5.2절 within-match 정의의 중복 인용, 6장 머리말, 6.3절 SESOI 문장의 구조 대비 구절.
  - 인용 Ngiam22를 뺐다(이로써 뺀 인용은 모두 9개).
- 익명 저장소: 그림 스크립트를 복사했다. 커밋하지 않았다.

## 18. 최종 교정과 저장소 푸시 (v19e, 2026-10-09)
v19d는 `archive/main_v19d.*`에 보관했다. 사용자가 10쪽을 허용했다.

- 6.3절: "on shared inputs model changes move capture little, whereas a larger input set raises it"를 수치 문장으로 바꿨다. 입력 결합 +1.46 [0.92, 2.00]pp에 대비해, 공통 입력에서 1차 구조 대비가 capture를 움직인 폭은 최대 0.36pp다.
- 5.3절 하이퍼파라미터 문장을 v6대로 복원했다(learning rate, leaves, heads, blocks, AdamW 학습률).
- 4.2절 Scene Transformer(Ngiam22) 인용을 복원했다.
- 7.2절: "On shared inputs, the reference designs differ little in top-3 capture"로 바꿨다.
- top-3 비교 대상: 초록·기여 3·결론을 "reference GBDT and PTR-Former"로 명시했다. 초록 200단어("with PTR-Former's 95% interval including zero").
- 10쪽을 허용했으므로 지면 때문에만 뺐던 인용 8개(Zheng15, KagglePUBG, LeeLee25, Xenopoulos21, Bergstra12, Cawley10, DynamicDeepHit20, Gensheimer19)와 3.3절 cutoff 예시 문장을 되살렸다.
- 결과: 10쪽(마지막 쪽은 참고문헌 일부), overfull 0, 참고문헌 39개.
- 익명 저장소: README 실행 순서에 `tools/sac2027_offset_only_cutoff.py`를 넣었다.
  - 오프라인 테스트: 676 passed, 14 skipped.
  - 신원 검사 통과.
  - Anonymous로 커밋(8928de5, 2026-10-09T03:25Z)하고 main에 푸시했다. 저장소는 PRIVATE를 유지한다.
  - anonymous.4open.science 미러는 사용자가 새로고침해야 반영된다.

## 19. 인용 위치와 5.1절 가독성 (v19f, 2026-10-09)
v19e는 `archive/main_v19e.*`에 보관했다.

- 인용 위치
  - Zheng15: "spectator-view telemetry"에서 빼서, 표 3 "Movement, space" 행의 "tortuosity" 뒤로 옮겼다.
  - Xenopoulos21: relation-type bias 문장에서 빼서, 식 (2) 아래 𝒢 정의("a graph representation of game state")로 옮겼다.
  - LeeLee25: "PUBG prediction began with ... [KagglePUBG], which recent work still studies [LeeLee25]"로 바꿨다.
- 5.1절: 목적함수 통제와 고정 점수 보정 통제("Calibration controls on fixed scores (RQ1)")를 다른 문단으로 나눴다. 결합 손실은 식 (3), offset-only는 식 (4)로 표시 수식으로 뺐다. 내용 변경은 없다.
- AI 고지: 사용자 결정으로 그대로 뒀다.
- 결과: 10쪽, overfull 0, 참고문헌 39개.

## 20. top-3 capture → recall@3 (v19g, 2026-10-09)
v19f는 `archive/main_v19f.*`에 보관했다.

- "top-3 capture"는 최종 순위 3위 안으로 오해될 수 있어 모두 "recall@3"로 바꿨다.
  - 5.2절 정의: 각 경계에서 설계가 가장 위험하다고 본 세 팀 안에 든 다음 단계 탈락의 비율이다. 탈락이 있는 경계를 모두 합쳐 계산하고, 식은 표시 수식으로 뺐다.
  - "The three are ranked by predicted risk, not by final placement."를 명시했다.
- 초록·기여 3·결론은 지표 이름 대신 "place ... more (next-phase) eliminations among their three riskiest teams"로 풀어 썼다. 초록 200단어를 맞추려고 "which of the teams"를 "which teams"로, "shrinks from 0.011 to 0.001 AUC"를 "falls from 0.011 to 0.001"로 줄였다.
- 5.3·6.3·7.2절의 나머지 언급과 그림 1 지표 상자도 recall@3로 바꿨다.
- 결과: 10쪽, overfull 0. 원고에 "top-3"가 더는 남아 있지 않다. 코드 식별자(`top3_capture`)는 그대로다.

## 21. 용어 풀이와 Widom 구성 점검 (v19h, 2026-10-09)
v19g는 `archive/main_v19g.*`에 보관했다. 10쪽은 SAC 2027 규정 안이다(저자 키트와 MLA 트랙 페이지: "Full papers are limited to 8 pages with the option for up to 2 additional pages"; 2026년 기준 추가 페이지 비용은 쪽당 80달러, 2027년 금액은 미게시).

- 정의 전 사용 금지(메모리 규칙)
  - 서론 2문단에 evaluation unit, within-phase AUC, within-match AUC의 정의를 넣었다.
  - 서론 3문단에 risk set(같은 경기에서 아직 살아 있는 팀)과 match-common term(경기 전체 위험 수준)의 풀이를 넣었다.
  - 서론 4문단에 공통 과제(landmark), cutoff 10초, 패치(게임 버전), next-phase Cox, 중심화, 결합 손실, 고정 점수 보정의 풀이를 넣었다.
  - 서론 5문단을 새로 두고 reference designs(GBDT, PTR-Former), team attention, habitat-context, phase-adjusted KRAFTON, cohort의 풀이를 넣었다.
  - 3.4절 landmark, 3.1절 patch, 5.2·기여의 "fewest-remaining-members rule"(살아 있는 인원이 가장 적은 팀을 고르는 규칙), 6.3절 I²에 풀이를 붙였다. 5.3절은 SESOI를 첫 사용 전에 정의하도록 순서를 바꿨다.
  - 초록은 쉬운 말로 바꿨다: patch-split, cohorts, team attention, centering, landmark 같은 표현을 일반어로 바꿨다. 198단어.
- 방법 절의 수치를 결과로 옮김(메모리 규칙)
  - 같은 경기 쌍 비율(0.15–2.8%)은 6장 머리말로 옮겼다.
  - offset-only 기울기 범위(0.98–1.16)는 6.1절로 옮겼다.
  - 패치별 경기 수와 AI 팀 비율은 표 2와 표 2 각주로 옮겼다.
- Widom 구성
  - 기여를 목록(itemize)으로 바꿨다.
  - 서론 끝에 절 안내와 한계 예고(한 게임, 한 시험 버전, 목적함수 통제는 한 네트워크)를 넣었다.
  - 2절에 쉬운 예시를 넣었다(빠르게 좁혀지는 구역의 경기 팀을 모두 위에 두면, 같은 경기 안 순서를 모르고도 다른 경기 쌍을 대부분 맞힌다).
  - 5.1절 offset-only에 식 앞에서 직관 설명을 붙였다.
  - 결론 앞에 "Related Work" 절(8절)을 새로 두었다(e스포츠·배틀로얄 예측, 생존 예측과 평가).
  - 문단 하나에 메시지 하나가 되도록 6.1절 조작 문단과 7.1절을 둘로 나눴다.
- 결과: 10쪽, overfull 0, 참고문헌 39개.

## 22. 관련 연구를 2절로 (v19i, 2026-10-09)
v19h는 `archive/main_v19h.*`에 보관했다.

- 사용자 요청으로 Related Work를 서론 바로 뒤(2절)로 옮겼다. 절 번호가 하나씩 밀렸다(원리 3절, 데이터 4절, 설계 5절, 프로토콜 6절, 결과 7절, 함의·한계 8절, 결론 9절).
- 겹치던 인용 문장을 정리했다.
  - 서론의 PUBG 순위 예측 문장은 2절로만 남겼다.
  - 3절의 within-group 평가 인용 세 개(Heagerty–Zheng, van Klaveren, GAUC)는 2절로 넘기고, 3절은 "Section 2의 그룹 안 평가와 같다"고만 쓴다.
  - DeepHit은 2절에만 남겼다.
- 2절에 선행 설계들이 경기를 합친 지표나 전체 기간 지표로 보고하고 목표도 서로 다르다는 한계를 썼다. 우리는 두 설계를 한 과제로 옮겨 쓴다.
- 서론 끝 절 안내를 새 순서로 고쳤다.
- 결과: 10쪽, overfull 0, 참고문헌 39개.
- 익명 저장소 README의 원고 대응 열을 새 절 번호로 고쳤다(14행). 커밋과 푸시는 하지 않았다.

## 23. 제목 변경 (v19j, 2026-10-09)
- 질문형 제목을 ACM SAC식 서술형으로 바꿨다(사용자 선택): "Evaluation Units for Next-Phase Elimination Risk Prediction in Battle-Royale Games". 머리글용 짧은 제목은 그대로다.
- 익명 저장소 README 첫 줄의 제목도 바꿨다(로컬). 커밋·푸시는 하지 않았다.

## 24. 경기 간 위험 차이의 근거와 세 분석의 이유 (v19k, 2026-10-09)
v19j는 `archive/main_v19j.*`에 보관했다.

- 3절 예시가 틀렸다. 구역 일정(단계 길이, 수축 속도)은 맵마다 고정이라 "빨리 좁혀지는 구역"은 사실이 아니다. "많은 팀이 몰린 경기 vs 적은 팀이 흩어진 경기"로 바꿨다.
- 서론 2문단에 같은 단계에서도 경기마다 위험이 다른 이유를 넣었다: 일정은 고정이지만 살아남은 팀 수, 밀집, 교전, 무작위 구역 위치가 다르다.
- 서론 4문단에 세 분석을 하는 이유를 넣었다. 각 분석이 무엇을 묻고, 결과에 따라 무엇을 뜻하는지다.
  - 중심화: 경기 수준을 지워도 격차가 남는가(진단).
  - 결합 손실: Cox에 없는 학습 신호를 주면 경기 안 AUC를 유지한 채 격차가 닫히는가.
  - 고정 점수 보정: 학습 뒤 경기 수준 항만 더해도 닫히는가.
- 3절 가설 문단에 H3, H4a, H4b가 각각 무엇을 묻는지 한 문장을 넣었다.
- 결과: 10쪽, overfull 0.

## 25. 기여를 서론만 읽고 이해되게 (v19l, 2026-10-09)
v19k는 `archive/main_v19k.*`에 보관했다.

- 기여 세 항목을 서론에서 이미 소개한 개념만으로 다시 썼다. 각 항목은 한 줄 요지(기울임)로 시작한다.
  1. 평가 단위가 목적함수 비교를 정하는 조건: 생존 손실은 경기 전체 위험 수준에 대한 직접 신호가 없고 경기 안 AUC는 그것을 보상하지 못하며, 교차 엔트로피는 배우고 단계 안 AUC는 보상한다.
  2. 그 설명의 통제된 시험: 경기 안에서는 차이가 없고 단계 안에서만 0.006–0.008. 세 분석이 모두 경기 수준을 가리킨다(중심화로 대부분 사라짐, 결합 손실 58–71%, cutoff 정보의 경기 수준 이동 34–58%, 한 코호트는 불확실). 사전 예측은 9개 코호트 합성 추정치가 지지했다.
  3. 경기 안에서 남는 이득: 서식지 입력은 두 단위에서 비슷하게 움직이고, team attention과 PTR-Former의 GBDT 대비 이득은 경기 안에서 거의 0이 된다. 검토 지표로는 0.4–0.6pp(PTR-Former는 불확실)이며, 단순 규칙 대비 이득의 10분의 1 미만이다.
- 빼서 결과 절에만 남긴 것: "reduced form/recorded form" 구분, "testable account/invariance" 같은 용어, recall@3 이름.
- 결과: 10쪽, overfull 0.

## 26. 그림 점검 (v19m, 2026-10-09)
v19l은 `archive/main_v19l.*`에 보관했다.

- BCE 약어가 정의되지 않은 채 그림 1·2와 본문에 쓰이고 있었다. 서론 3문단에 "binary cross-entropy (BCE)"로 정의했다.
- 그림 1
  - "Patch split"을 "Game-version split"으로 바꿨다(게임 용어 대신 본문 표현).
  - "Metrics" 상자 제목을 "Metrics and diagnostics"로 바꿨다. 중심화는 지표가 아니라 진단이라는 6.2절 표현과 맞춘 것이다.
- 그림 2: 세로를 4.95cm에서 5.6cm로 되돌렸다. 두 줄 그룹 이름이 위아래 행과 붙던 문제를 없앴다.
- 그대로 둔 것: 그림 1의 패널 순서(RQ별), 기호 범례, 그림 2의 열 구성과 캡션.
- 결과: 10쪽, overfull 0. 그림 스크립트를 익명 저장소에 복사했다(로컬, 미푸시).
