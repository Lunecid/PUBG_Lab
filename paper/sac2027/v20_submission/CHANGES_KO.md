# SAC 2027 v20 변경 기록

v19까지의 기록은 `../v19_revision/CHANGES_KO.md`(1–26절)에 있다.

## 27. 공식 템플릿 적용 (v20a, 2026-10-09)
- `ACM_SAC_2027_Article_Template.zip`을 내려받아 `acmart.cls`(2026/06/27 v2.19)와 `ACM-Reference-Format.bst`를 그대로 썼다. v19가 쓰던 파일과 sha256이 같다.
- 전문(preamble)을 템플릿 구조로 바꿨다: `\documentclass[sigconf,anonymous]{acmart}`, 권리 블록(`\setcopyright{acmlicensed}`, `\copyrightyear`, `\acmYear`, `\acmDOI`, `\acmConference[SAC'27]`, `\acmISBN`)은 템플릿의 예시값, `\renewcommand{\shorttitle}`, 익명 저자 블록.
- 뺀 것: `nonacm`, `\settopmatter{printacmref=false,printfolios=true}`, `\setcopyright{none}`, 빈 `\acmDOI{}`/`\acmISBN{}`. 첫 쪽에 ACM Reference Format 줄과 저작권 문단(예시값)이 찍힌다.
- 참고문헌 스타일: `ACM-Reference-Format-citation-order` → 템플릿의 `ACM-Reference-Format`(알파벳순). 본문은 v19m과 같다. 결과 11쪽(참고문헌 약 17줄 넘침).

## 28. 1차 축약 (v20b, 2026-10-09)
- 11쪽 → 10쪽. 본문·표에 중복된 수치와 괄호 설명을 줄였다.
  - 서론 끝에 한 줄 요지를 넣었다.
  - 4절: 데이터 수집 문단을 합쳤고("past 24 h" 삭제), 격자 시간 정의를 한 문장으로 썼다.
  - 5.1: 이동 잔차 설명에서 반복 문구를 줄였다.
  - 6.1: 결합 손실의 사전 예측 세 개와 9코호트 예측 수는 3절 가설·표 5·7.1절에 있으므로 6.1에서는 뺐다. 오프셋 전용의 사전 예측 두 개는 다시 넣었다(29절).
  - 7.1: BCE−Cox 경기 안 95% CI, 평균 점수 AUC CI 일부를 뺐다(표 4·5와 그림 2에 있음).
  - 7.2·7.3: 표 6·8에 있는 값(0.0151/0.0159의 CI, 평균 AUC 0.768/0.769/0.720/0.747/0.675/0.759)을 본문에서 뺐다.
- 용어: "estimand" → "prediction target"(4.4절 제목 포함). 평균 점수의 경기 간 AUC에 "mean-score AUC"라는 이름을 붙여 반복 설명을 없앴다.

## 29. 가독성 피드백 반영 (v20b, 2026-10-09)
사용자 피드백(중심 메시지, 용어 먼저 풀기, 추상명사→행동, 결론→직관→수치, 통계 절차명은 방법 절에만). 피드백의 일부 항목(estimand, relative-risk adaptation, 71%, 단계 a1–g, Table IV, 12 코호트, Fig. 3(a))은 BigComp 원고 것이라 이 원고에는 없다.
- 초록: 문장을 나누고 쉬운 말로 바꿨다. 위험집합 조건은 "compare each eliminated team only with survivors of its match"로 남겼다. 200단어.
- 서론: habitat context는 "a team's position relative to the zone, its movement, and the opponents nearby"로 먼저 풀고 이름을 붙였다. 기여 뒤의 요지 문장은 수치로 썼다: 시간 범위와 입력은 경기 안 AUC를 0.01 이상 움직였고, 정렬된 손실 간 선택·팀 attention·PTR-Former 대 GBDT는 약 0.001.
- 5.1: 생태학 언급은 이름의 유래 한 문장으로 줄였다("the analogy only organizes the variables").
- 5.2: Platt scaling을 "logistic function fit on validation data"로 먼저 설명.
- 6.1: "Breslow ties"는 괄호로, "administrative censoring"은 "censors teams there for Cox"로. "validation NLL" → "validation log loss".
- 6.2: 중심화 설명에서 "additive decomposition" 표현을 "a way to split AUC into match-level and team-level parts"로.
- 6.3: 랜덤효과 모형을 "lets the true effect differ between cohorts"로 풀고 "pooled estimate"를 정의했다. SESOI는 "the smallest difference we treat as meaningful".
- 7절: 각 문단 첫 문장을 결론으로 바꿨다. 7.2의 "also under REML/Hartung–Knapp" → "also under the alternative random-effects estimator", "Holm p" → "adjusted p". 7.3 첫 문단은 표 8 수치 대신 "both reference designs beat the adapted prior designs; the largest gap is to Sagong; the advantage over phase-adjusted KRAFTON is smaller but consistent"로. "Within matches, these gains shrink by half to three quarters"(0.0187→0.0100, 0.0271→0.0071). recall@3 문단은 "For a reviewer, the reference designs and phase-adjusted KRAFTON differ little"로 시작. 단계 기준선은 "lets KRAFTON's probabilities follow each phase's elimination rate".
- 8.1: 긴 문장을 나눴다. "pair composition" → "the same mix of pairs". 결론 첫 문장의 "within-match discrimination"은 "how well a model ranks the teams of one match with how well it ranks matches"로.
- 주장 드리프트 점검(v19m 대비, 서브에이전트) 뒤 되돌린 것: 결합 손실이 "학습 전체를 바꾼다"는 단서, 오프셋 전용의 Erangel "실용 이득 미달" 괄호, 결론의 "supported by pooled estimates"와 오프셋 전용은 "두 최대 코호트에서"라는 범위, 6.1 "The same check applies to them", 7.1 "no other condition in Table 4 receives", 기여 2의 "were supported". 초록·결론의 "best designs place at most 0.6 pp"는 "GBDT and PTR-Former place 0.4–0.6 pp … PTR-Former's interval includes zero"로 되돌렸다.
- 2차 드리프트 점검(평이화 뒤) 뒤 고친 것: 초록 "each match's mean score" → "each match--phase's mean score"(단위 오기), "practically unchanged" → "practically equivalent"(정의된 판정어); 서론 5문단 "share the same inputs" → "share, alongside team state, a block of inputs"(PTR-Former는 중심점·관계 입력을 더 쓰므로); RQ 머리말은 "what changes when models are evaluated within matches rather than within phases"; "Giving Cox the missing signal works" → "recovers most of the gap"; "advantage sits in" → "depends on"; "its gains appear between matches, not within them" 삭제(경기 안 CI가 0을 포함); "equal tuning budgets" → "equal-trial tuning"(계산량은 동일하지 않음); "The pattern repeats in the nine cohorts" → "holds when pooled over"; "smaller but consistent" → "smaller"(부호 수 없음, I²=0.85); "these gains shrink" → "the gains over phase-adjusted KRAFTON shrink"; "More information helps more than more structure" → "Adding information helped more than changing structure here"; "survival objective" → "survival designs"(KRAFTON 대 PTR-Former는 입력·사상도 다름). 6.1의 사전 예측 열거(결합 손실 3개, 9코호트 7개)와 "positive-improvement rule fails there", 그리고 표에 없는 CI 세 개(+0.046, 평균 점수 AUC +0.051/+0.046, Cox+BCE +0.011/+0.012)를 복원했다. 요지 문장은 "BCE versus next-phase Cox, attention across teams, and PTR-Former versus GBDT differed within matches by about 0.001, with intervals that include zero"로 범위를 명시했다.
- 결과: 10쪽(마지막 쪽 오른쪽 단 약 6줄 여유), 마지막 패스 overfull 0, 인용 39개 그대로, 수치 변경 없음, 초록 200단어.

## 30. 구조 변경: 중심 질문 먼저, 결과 개요 먼저 (v20c, 2026-10-09)
v20b는 `archive/main_v20c_pre_restructure.*`에 보관했다(이름은 변경 전 판이라는 뜻).

- 사용자 지적: 피드백의 구조 문제(개념을 먼저 쌓고 RQ가 뒤에 옴, 결과가 실험 순서대로 읽힘)는 SAC 원고도 같다.
- 서론을 다시 썼다(절 수는 같음).
  1. 과제(그대로).
  2. 중심 질문과 답: 기존 설계는 시간 범위·손실·입력·구조가 다르고 경기 합산 AUC로 비교된다. 검토자가 하는 비교(한 경기 안)에 무엇이 중요한가. 답은 채점 방식에 달렸다: 단계 안에서는 BCE가 Cox형 손실을 0.006–0.009, 팀 attention이 0.011 앞서지만, 경기 안에서는 둘 다 약 0.001로 떨어지고 시간 범위 정렬과 입력은 0.01 이상 움직인다.
  3. 이유(평이하게): 평가 단위, 단계 안 AUC는 다른 경기 쌍이 대부분(3% 미만 같은 경기), 경기 간 위험 차이의 근거(고정 일정, 팀 수·밀집·교전·구역 위치), 경기 안 AUC, 위험집합·match-common term, 시간 범위 정렬의 필요.
  4. 한 일: 공통 과제, 패치 분할, 코호트, next-phase Cox, 세 가지 탐침(중심화·결합 손실·오프셋 전용, 각각 무엇을 뜻하는지 한 절씩), 참조 설계 GBDT·PTR-Former와 habitat context, phase-adjusted KRAFTON.
  - RQ 두 개는 목록 대신 한 문단으로. 기여 세 항목은 2문단과 겹치는 수치를 줄여 짧게 했다. "In short" 문장은 2문단으로 흡수돼 삭제.
- 결과 7절 머리에 "7.1 Overview"를 두었다: 같은 경기 쌍 비율(0.15–2.8%) → 그림 2 → 경기 안 AUC를 0.01 이상 움직인 선택(시간 범위 +0.012~+0.015, habitat 블록 +0.016, KRAFTON 입력 추가 +0.020)과 약 0.001인 선택(BCE 대 next-phase Cox, 팀 attention; PTR−GBDT −0.001), 단계 안에서는 후자가 0.006–0.011로 보임 → 이후 소절이 각 수치의 통제를 설명. 소절은 7.2 목적함수, 7.3 입력·구조, 7.4 설계 전체·검토 효용으로 번호가 밀렸다.
- cutoff 재구성 검사(453/450, 817,766행 중 2행)는 결과 머리에서 4.3절 끝으로 옮겼다(자료 무결성 검사이므로).
- 중복 설명 삭제: 3절 첫 문단 마지막 문장(경기 안 AUC 재설명), 3절 Horizon 문단 축약, 8.1의 수치 반복(0.019–0.027 → 0.007–0.010은 7.4에 있음), 기여 2·3 축약.
- 분량: 10쪽 그대로(본문은 9쪽 오른쪽 단 중간에서 끝나고 참고문헌이 약 1.5단). 9쪽으로 가려면 표 4(두 코호트 절대 AUC)나 표 7(사전 지정 판정) 중 하나를 빼야 한다. 뺀 것 없음.
- 결론을 핵심만 남겨 12줄 → 7줄로 줄였다(사용자 지적). 남긴 것: 첫 문장(단계 안 AUC가 두 가지를 섞는다는 명제), 손실·attention 차이가 단계 안에만 있고 경기 안에서는 약 0.001, Cox+BCE가 격차 대부분을 메움(사전 예측, 9코호트 합산 지지)과 match-common 정보의 역할, 경기 안 AUC를 움직인 것은 시간 범위·입력(0.01 이상)이지 손실·구조(약 0.001, 구간 0 포함)가 아님, 집단 안 평가 권고. 뺀 것: 오프셋 전용 수치, recall@3 수치, 향후 과제 문장(모두 7절·초록·기여에 남아 있음).
- 새 서론·개요 점검(서브에이전트) 뒤 고친 것: "about 0.001" → "0.002 or less"(9코호트 합산 BCE−Cox 경기 안 +0.0015 포함; 서론·개요·결론), "aligning the horizon" → "aligning the Cox horizon"(가중 손실은 Erangel 경기 안 +0.009), "changing the inputs" → "the main input contrasts"(AI 입력 제거 등 통제는 ≤0.0033), "The reason is" → "The explanation we test concerns", "These known properties predict" → "suggest", 기여 1·2 제목에서 "predicts"·"recorded in advance" 완화(가설은 두 코호트를 본 뒤 세움, cutoff 변형은 사후), 기여 3의 "do not" → "shrink to about 0.001 … intervals that include zero", 개요의 PTR−GBDT에 "uncertain", AUC 첫 사용에 풀이, match–phase·practically equivalent 첫 사용에 풀이, match-common term 정의를 "such a constant"로 명시, 개요에 목적함수 수치 범위의 출처(두 최대 코호트와 나머지 9개 합산) 명시.

## 31. 서론 무수치화와 설명 순서 재배치 (v20d, 2026-10-09)
v20c는 `archive/main_v20d_pre_nonum_intro.*`에 보관했다. 근거: 사용자 지적("서론에서 숫자가 나오면 안되지")과 종합 평가(ChatGPT 공유본, 접근성 5/10).

- 초록: 한 문장에 판단 하나씩. "With inputs, network, and horizon matched, both losses rank the teams of a match equally well, but cross-entropy wins when AUC is pooled within phases." 199단어. 위험집합 조건, 사전 예측, 0.4–0.6pp와 PTR-Former 구간은 유지.
- 서론: 결과 수치 전부 삭제(남은 숫자는 게임 사실 100명과 cutoff 10초뿐). 순서: 과제 → 중심 질문(평가 단위가 비교 결론을 바꾸는가)과 정성적 답 → 설명(단계 안 AUC는 다른 경기 쌍이 대부분, 경기 간 위험 차이, 경기 안 AUC, 위험집합과 match-common term, 시간 범위) → 한 일(세 탐침; 오프셋 전용은 "after training and without retraining") → RQ1→RQ2 연결 문단("경기 수준을 보상하는 지표라면 팀 간 정보를 모으는 구조도 경기 안 순위 개선 없이 이득을 볼 수 있다") → RQ1(왜)·RQ2(무엇이) → 기여 3개(평가 단위가 중심, 설명 검증, 검토 함의) → 절 안내.
- 관련 연구: 차별점 한 문장("These studies treat the grouping as a property of the metric. We add the loss side …").
- 3절: 두 경기 숫자 예시(A 0.60/0.40, B 0.50/0.30, A에 +0.30 → 경기 안 AUC 불변, 합산 AUC 변화)를 수식보다 먼저. 그다음 쌍 분해식, 같은 경기 쌍 비율 0.15–2.8%, 성질(표 1), 손실, 시간 범위. 가설 H0–H4b는 6.1절로 옮겼다(평가의 "보통" 항목; 독자가 truncation·결합 손실·오프셋 전용을 안 뒤 읽도록). 6.1의 사전 기록 범위 문장도 가설 문단에 합쳤다(두 코호트 뒤 가설 수립, H4a 세 예측·H4b 두 예측·9코호트 일곱 예측은 실행 전 기록).
- 6.3: practical equivalence에 "a stronger statement than the absence of a significant difference" 풀이.
- 7.1: "A statistically positive difference need not meet the pre-specified threshold for a practical gain" 문장 추가. 7.2: 그림 2에 있는 H1·H2 CI는 본문에서 뺌.
- 결론 끝: "the evaluation unit is part of the claim, because it decides which design differences count."
- 분량 조정(11쪽→10쪽): 기여 1 축약, 3절 crowded 문장 축약, 6.1의 H3/H4 "asks" 문장 삭제(서론 4문단이 같은 내용), 오프셋 전용 형태·정보 분해 문장 축약(40%/60%만), 6.2 경기 안 AUC 재정의 삭제, 8.1 보정 지표 문구 축약, 표 3 글꼴 footnotesize→scriptsize. 10쪽, 마지막 쪽 오른쪽 단 약 1/3 여유.
- 재작성 부분 점검(서브에이전트) 뒤 고친 것: 초록 "rank the teams of a match equally well" → "reach comparable within-match AUC"(판정은 실용적 동등), "supported in nine" → "supported when pooled over nine"; 서론 "changing the inputs still help" → "still matters"(KRAFTON→ours는 −0.012), "the second comparison" → "the within-match comparison", "To test this" → "To test this explanation", "gain there" → "gain within phases", 시간 범위 정렬에 한 줄 풀이, RQ2에 "which combine inputs, loss, and probability mapping", 5문단에 "although we recorded no predicted direction for it"; 기여 1 "comparisons of inputs do not" → "change little"(최대 0.005 이동), 기여 2 "three probes point to" 뒤에 "(the third in one of two cohorts)", 기여 3 "which teams they flag" → "how many of the next eliminations they catch among their riskiest teams"(팀 겹침은 측정하지 않음), "simple rule" → "a rule that flags the teams with the fewest living members", "phase-adjusted survival design" → "phase-adjusted KRAFTON", "within-group/within-stratum AUC" → "within-match next to within-phase AUC". 표 5 캡션의 가설 참조를 6.1절로.

## 32. 참고문헌 번호를 인용 순서로 (v20d, 2026-10-09)
- 사용자 요청. `../v19_revision/ACM-Reference-Format-citation-order.bst`(템플릿 bst에서 정렬 키만 첫 인용 순서로 바꾼 로컬 파생본)를 복사하고 `\bibliographystyle`을 바꿨다. 항목 서식은 템플릿과 같다. 템플릿 기본은 알파벳순(BibTeX 경로와 biblatex acmnumeric 경로 모두)이므로 README에 되돌리는 방법을 적었다.

## 33. 외부 리뷰 3차(Borderline) 핵심 반영 (v20e, 2026-10-09)
v20d는 `archive/main_v20e_pre_review3.tex`에 보관했다. 사용자 지시: 게재율을 올리는 핵심만.

### 사실 확인
- W6 후반(attention key 누수)은 코드상 해당 없음: PTR과 KRAFTON 백본 모두 key = cutoff 시점 생존 팀(`model_stgcn/ptr/data.py:412` `key_mask`=`node_mask`, `third_party/krafton_pubg_winprob/adaptation/extract.py` `KEY`), 손실 행만 R. 4.3절에 한 문장으로 밝혔다.
- 학습 1회 65–73초라 S1(새 손실·가중 스윕)은 비용이 작다.

### 본문 수정(A군)
- 초록: "nine further match groups of the same held-out game version"(같은 시험 패치임을 명시).
- 서론: 기여 1을 데이터 크기로 축소("small within-phase gains of components that pool information across the teams of a match … shrink to near zero within matches …, whereas the larger effects of the prediction horizon and of the inputs appear on both units"); 2문단에 "small to begin with".
- 관련 연구(W2): Sagong은 경기 합산 정확도, KRAFTON 저장소는 winner accuracy와 시간 가중 C-index를 평가함을 명시. 7.4에 지표별 설계 순위가 다른 예(winner accuracy는 KRAFTON, overall AUC는 PTR-Former, 경기 안은 GBDT=PTR-Former).
- 6.1 가설 문단(Q5): 가설 수립 시점에 11개 코호트의 완성 설계 결과는 이미 있었고, 9코호트 재현은 짝지은 목적함수 통제만 새로 본 것임을 명시.
- 6.3(W4): practical gain은 점추정 규칙임을 명시, 최소효과검정(90% 하한 > SESOI) 정의. 7.4: GBDT는 통과(95% 하한 0.0071), PTR-Former는 불통과(0.0030).
- 6.3·7.3·7.4(S2): "단위에 따라 다르다"를 검정하는 phase-minus-match 차이(같은 재표집, 단계 동일 가중 경기 안 AUC와 짝) 정의와 수치: 팀 attention +0.0087 [0.0067, 0.0107](10/11), PTR−GBDT +0.0054 [0.0013, 0.0096](9/11), PTR−KRAFTON+phase +0.0212 [0.0139, 0.0285](11/11), GBDT−KRAFTON+phase +0.0145 [0.0075, 0.0216](10/11). 도구 `tools/sac2027_did_rq2.py` → `control_evaluation/did_rq2/`.
- W7: 표 8 굵은 글씨 제거, 주석에 열 간 비교 금지(m2)와 proxy 입력 명시; 8.2에 γ_k 적합 패치 비대칭.
- S7(지면): 표 7(1차 판정)을 본문 한 문장으로 접음(추정치·90% CI 그대로), 오프셋 전용 "위험집합 요약 유지 변형 40/60%" 삭제, 동일 시도 튜닝 세부 수치 축약, SD 스케일 중심화 삭제, 체크포인트 검사 축약, 4.1 수집 횟수 삭제, 7.4 보정 기울기 삭제.

### 새 실험(S1, 사전 등록)
- 계획 `reports/sac2027_patch_v14/CONDLOGIT_PLAN_KO_2026-10-09.md`(sha256 ba18b400…, 실행 2026-10-09T08:19:02Z, 로그 `objective_controls_condlogit.log`).
- `cond_logit`(경기–단계 탈락 수 조건부 로지스틱, ESP 로그 DP; `tools/sac2027_objective_controls.py`에 구현, 완전 열거와 일치·이동 불변·공통 이동 기울기 0 검증)과 Cox + w·BCE, w ∈ {0.25, 0.5, 2, 4}. 두 최대 코호트 1차, 9개 코호트 합산 재현. 평가 `tools/sac2027_condlogit_eval.py` → `control_evaluation/condlogit/`.
- 9개 코호트 합산(예측 6·7 성립, `control_evaluation/condlogit/pooled.csv`): BCE − cond_logit 단계 안 +0.0097 [0.0063, 0.0132](9/9), 경기 안 +0.0026(90% [0.0006, 0.0046], 실질 동등이나 0은 배제); cond_logit − Cox 단계 안 −0.0017, 경기 안 −0.0014(90% CI ±0.004 안); 회복률 합산 0.39/0.49/0.63/0.78/0.92(w=0.25–4); w=4에서 BCE와의 차이 −0.0003 [−0.0013, 0.0006]. 7.2 끝에 한 문장으로 넣었다. 학습 165회 전부 성공(로그 `all done 2026-10-09T09:18:35Z`).

## 34. 재심(Weak Accept) N1–N10과 3차 재평가 반영 (v20f, 2026-10-09)
v20e는 `archive/main_v20f_pre_rereview.tex`에 보관했다.
- N1: "after a first review round" 삭제 → "after the main analyses"(6.1, 표 3 캡션).
- N2: 전제를 landmark 평가 관행으로 옮김. 초록 "Landmark predictors are scored by AUC over all teams at risk, pooled over matches"; 서론 2문단 "Landmark risk predictors are conventionally scored by AUC over everyone at risk at a landmark time~\cite{vanHouwelingen07,HeagertyZheng05}; in a battle royale the teams at risk at a phase boundary are the union of the teams of many matches …"; 관련 연구 "KRAFTON's metrics compare the teams of one match but over its whole elimination order, Sagong's pool matches".
- N3: 초록 "in proportion to its weight" → "increasingly with its weight". 초록·결론 모두 조건부 로짓("practically equivalent to Cox")을 중심에.
- N4: 회복률 추정량 통일(풀링 차이의 비율): 9코호트 0.39/0.57/0.73/0.92/1.01(w=0.25–4); "about 72%" → "about 73%".
- N5: 표 4(사전 예측)에 C1–C5 행 추가(BCE−CLogit 단계 안 >0, 경기 안 ≈0; CLogit−Cox 두 단위 ≈0; BCE−Cox+4BCE ∋0), 캡션·주석에 규칙과 95% CI. 6.1은 "five … and three … rows C1–C5".
- N6: 최소효과검정 90% 하한으로 통일(GBDT 0.0076 통과, PTR-Former 0.0037 불통과; `rq2_match_offset/pooled.csv`).
- N7: 서론 4문단에 조건부 로짓을 "sharpest" 탐침으로 추가, 그림 2에 "BCE − conditional logit" 행 3개(Erangel·Taego·9코호트 합산; `tools/sac2027_figure_eval_units.py`, 높이 5.6→7.4 cm), 7.2 순서를 H0 → H1/H2+PMM → 중심화 → 조건부 로짓 → 결합 손실+가중 스윕 → 오프셋(보조 진단) → 9코호트로 재배치, 결론을 조건부 로짓 중심으로 다시 쓰고 끝 문장을 "whether such small gains count"로 좁힘.
- N8: 기여 1과 8.1을 "components that let scores carry a match-common level (a pointwise loss term, attention across teams)"로.
- N9: team attention의 SD 정규화 잔차 +0.0034 [0.0007, 0.0062] 보고("level and spread carry about two thirds; the remainder is unexplained").
- N10: "All losses were designed …", 7.4 "these", 조건부 로짓 분모는 ESP 로그 재귀로 정확 계산(6.1)과 탈락 0/전원 집합 무기여 명시, 9코호트 BCE−CLogit 경기 안 +0.0026 [0.0003, 0.0050](SESOI 미만이나 0 배제; 공통 항으로는 설명 불가, 미설명으로 남김), BCE−Cox 9코호트 PMM +0.0079 [0.0040, 0.0118], Δ=10초 근거(위치 기록 간격) 4.3절.
- 3차 재평가 추가분: 6.3에 SESOI 근거(recall@3 약 0.33pp), 표 6 각주 가중 설명(단계 동일 가중 평균 대 위험집합 쌍 풀링), Yan et al. 인용 유지.
- 지면: 표 2(입력 블록)와 표 5(입력·구조 대비)를 본문 수치로 접음(모든 CI 본문에 유지), 표 3의 weighted 두 행 삭제, 그림 1 0.86·그림 2 0.88 배율, 7.1 중복 문장·7.3 사후 분석 문장·8.2 튜닝 문장 삭제, 7.4 target 문단 축약. 10쪽, 마지막 쪽 오른쪽 단 약 15줄 여유, 최종 패스 overfull 0, 초록 199단어.
- 표 번호(최종): 1 코호트, 2 목적함수 통제, 3 사전 예측(H0–H4a, C1–C5), 4 주 결과.

## 35. 4차 재심(Weak Accept, V1–V4)과 main(7) 재평가 반영 (v20g, 2026-10-09)
v20f는 `archive/main_v20g_pre_v1v4.tex`에 보관했다.
- V1(SESOI 시간 순서): 6.3은 "fixed before training"만 남기고, recall@3 0.33pp 환산은 7.4에서 "Post hoc"로만 제시. "differ little"은 "compared with their common gain over a simple rule"(6.8pp 대비)로 명시.
- V2(C5): 규칙은 그대로 두고 표 3 C5에 ‡("90% CI also within ±0.005"), 7.2에 "includes zero and is practically equivalent" 병기.
- V3: 6.1 기록 범위를 "각 두 최대 코호트: C1–C5 + 가중에 따라 오르는 회복률 + 모든 가중에서 경기 안 동등; 9코호트 합산: C1–C5"로 정정(계획 예측 1–5와 6–7에 대응). 표 3 캡션에 기록 범위, ≈0 행 중 90% CI가 0을 배제하는 넷(H3, H4a 둘째, C2, C4)에 †.
- V4: 그림 2에 세 번째 그룹 "Inputs or structure"(habitat 블록 with−without, GBDT; team attention on−off, PTR-Former; `rq2_match_offset/pooled.csv`의 hgb_cut_nohab, team_main) 추가(17.8×8.6 cm). 7.3이 이 그룹을 가리킴. 지면은 오프셋 식 (4)를 인라인으로 접어 확보.
- main(7) 재평가: 7.3 "about two thirds" → "shrinks the gain from 0.0108 to 0.0034; … a diagnostic, not an additive attribution"; 그림 1 목적함수 상자에 CLogit·Cox+wBCE 추가(`figures/fig_pipeline_clean.tex`, 점 구분자를 좁혀 상자 안에 맞춤); 7.2 문단마다 run-in 제목(Horizon (H0), Aligned losses (H1, H2), Centering (H3), Conditional logistic loss, Combined loss and weight (H4a), Offset (H4b, auxiliary), Nine further cohorts); "conventionally scored" → "A common evaluation … is"; "reproduces Cox" → "is practically equivalent to Cox"(서론 기여·결론), 7.2 "yields a deficit as large as Cox's"; CLogit 우도를 6.1에 인라인 식으로.
- 자잘한 것: 초록 "no direct signal", "gains little within matches", "Landmark predictors are often scored"; 서론 "four ways"; 7.1에 "Erangel and Taego TPP squad" 복원; 4.3에 다음 구역 제외 이유("every design predicts from the same pre-reveal state"); 7.2에 KRAFTON 가중 손실 결과 한 문장(next-phase에서 경기 안 0.778/0.789 vs BCE 0.779/0.786, 단계 안 0.772/0.773 vs 0.779/0.782); 7.4 가중 생존 대 BCE에 "(both intervals exclude zero)" 복원; 6.3에 scikit-learn 구현명 복원.
- 회복률 w=0.25 값 확인: 풀링 차이의 비율 0.3862, 코호트별 비율 풀링 0.3883으로 둘 다 0.39(우연 일치, 재계산 누락 아님).
- 지면: 5.2 적응 문단·7.3 입력 대비 CI·8.2 정보 집합 문장 축약. 10쪽, 마지막 쪽 오른쪽 단 약 7줄 여유, 초록 200단어, overfull 0.
- 안 한 것(리뷰가 저널판으로 미뤄도 된다고 한 것 포함): γ_k 동일 패치 재적합, 정규화 recall@k·표 4 recall@3 열, 표 2 넷째 자리, Δ 민감도, 플레이어 군집 부트스트랩, LTR 기준선.
- 코드 정보 집합 감사(서브에이전트, 읽기 전용): 미래 정보 유입 없음(모든 입력 기록 < a, `features.py:691`, `grid.py:23-28`; 구역은 a 이전 마지막 상태; prefix replay 453/453 일치). attention key = cutoff 생존 팀(`node_mask`/`KEY`; PTR M00–M11R, 목적함수 통제 백본, KRAFTON-tf(-h0) 모두). 원고보다 약한 두 곳을 고쳤다: 생존 팀 수는 a 직전 마지막 게임 상태 기록(시험 덤프에서 `node_mask` 합과 92.4% 일치, 나머지는 1–3 많음), 관계 요약은 단계 안 이전 상호작용(a 이전에 탈락한 팀 포함). R은 학습 쪽에서만 쓰임(comeback 질의 토큰 학습 2/252,334행·검증·시험 0행, 학습 표준화, 셔플 대조 순열 영역, KRAFTON 학습 표본 선택); KRAFTON의 손실은 자체 관례대로 cutoff 시점 위험집합(T > a). 4.3 문장을 "Final elimination times define the labels, the scored risk set, and the training losses (KRAFTON's with its native risk sets at the cutoff); no input of a scored row depends on them."으로 바꿨다.

## 36. 가독성 정리: 결과 표 하나, 짧은 문장 (v20h, 2026-10-09)
v20g는 `archive/main_v20h_pre_readability.*`에 보관했다. 사용자 요청: 독자가 무리 없이 읽고, 주장이 드러나고, 논리를 따라가기 쉽게.
- 표 2를 수준값 표에서 두 최대 코호트의 대비 표로 바꿨다(`tables/rq1_contrasts.tex`, 생성기 `tools/sac2027_rq1_table.py`, 값 `tables/rq1_contrasts_values.csv`): H0, BCE−Cox(단계 안·경기 안·phase−match·중심화), CLogit−Cox·BCE−CLogit(두 단위), Cox+BCE(w=1) 대비·나머지·회복률, w=4 회복률·경기 안·나머지, 오프셋(cutoff 정보, 사후) 대비·회복률. 각 칸은 3시드 평균과 95% 경기 붓스트랩 구간. 주석에 phase−match 정의와 사후 추가 행 표시, 캡션에 "H0–H3는 이 결과 뒤 수립, H4a는 실행 전 기록".
- 7.2–7.4를 "주장 → 핵심 수치 1–2개 → 표·그림 참조"로 다시 썼다. 구간은 표 2·3·4와 그림 2로 옮기고 본문에는 판정에 필요한 것만 남겼다(본문 100단어당 소수: 7.2 12.8→7.7).
- 45단어 넘는 문장을 21개에서 대부분 쪼갰다(서론 5→1, 6.1 4→2, 결과 0). 초록은 문장 평균 약 21단어, 결론은 43→23단어.
- 판정 용어 하나 줄임: "positive improvement"를 "95% CI above zero"로 풀어 씀.
- 드리프트 점검(서브에이전트) 뒤 복원: "With everything else matched", "in these data", "descriptive slope", 7.1 "tests the explanation", 7.3 "the structure changes we tested shrink to near zero there", BCE 평균 점수 AUC CI, team attention 경기 안 CI, 중심화만의 +0.0043 [0.0020, 0.0067]과 SD 나눗셈 정의, PTR 중심화 CI, "90% CIs within ±0.005" 두 곳, 예측 범위 문장 분리(조건부 로짓 C1–C4 / 스윕: 오르는 회복률·C5·모든 가중 경기 안 동등, C5만 9코호트), "as ratios of pooled differences", 결론의 주어 "the evaluation unit".
- 10쪽, 초록 200단어, 최종 패스 overfull 0, 수치 변경 없음(반올림만), 인용 39개.

## 37. 5차 재심(Accept)과 acm12 재평가 반영 (v20i, 2026-10-09)
- X1: 다음 구역 제외 이유를 "we define risk from the teams' state before the next zone is drawn, not after its random placement is known"로(동일 조건은 이유가 아님).
- X2: 초록·결론의 사전 기록 범위를 실제와 맞춤: "The predictions recorded for both losses held when pooled over nine further match groups"; 가중에 따른 단조 증가는 결론에서 "in the two largest cohorts"로 따로 적음.
- X3: (1) "cutof"는 원본에 없음(PDF 텍스트 추출 시 ff 합자 문제로 보임). (2) KRAFTON의 cutoff 위험집합은 KRAFTON 적응판에만 해당하고, 목적함수 통제(가중 손실 포함)를 비롯한 모든 짝지은 비교는 R을 쓴다고 4.3에 명시(코드 감사와 일치: `tools/sac2027_objective_controls.py`는 target_mask=R). (3) 7.3에 "while a residual remains"(+0.0034 [0.0007, 0.0062]) 복원.
- W8: 기존 예측으로 R-precision(경계별 탈락 수만큼 표시, 천장 없음)과 달성 가능 최대치로 나눈 recall@3를 계산(`tools/sac2027_recall_norm.py` → `control_evaluation/recall_norm/`; recall@3는 기존 값 51.7/51.6/51.1/45.1/53.3과 일치). 표 4에 "Review (%)" 열 두 개(recall@3, R-prec.) 추가, 7.4에 R-precision 문장(60.0/59.9/59.4 대 규칙 51.2). 정규화 recall@3(69.9/69.7/69.1 대 60.2)도 결론 같음(차이는 규칙 대비 이득의 10분의 1 미만).
- W7(γ_k): 코드 확인 결과 본문에 보고하는 phase-adjusted KRAFTON(KRAFTON-tf-h0)은 γ_k를 검증 위험집합에서만 적합. 학습 위험집합 적합은 본문에 없는 변형(KRAFTON 백본 + 우리 입력, `KRAFTON-tf-ourfeat-h0`)이므로 5.2 문장을 "γ_k is fit on validation risk sets"로 고치고 8.2의 한계 문장을 삭제.
- acm12 재평가: 표 3을 [b] 배치로 바꿔 7→8쪽에서 핵심 문장이 표로 끊기지 않게 함; 표 2 글꼴 scriptsize → footnotesize(구간만 scriptsize, 단위 표기 phase/match로 축약, 캡션에 정의); 서론 4문단 "is what BCE learns" → "the gap is consistent with BCE learning the level rather than …"; 관련 연구 KRAFTON 지표를 winner accuracy(최상위 팀 우승 여부)와 경기 전체 순서의 C-index로 구분; 결론에 "a small within-match edge of BCE over the conditional loss remains unexplained".
- 지면: 표 2에서 오프셋(사후) 두 행을 빼고 그 구간은 본문으로, 7.2 머리 문장·가중 손실 문장·체크포인트 문장, 5.1·7.3·7.4 문장 축약. 10쪽, 마지막 쪽 약 3줄 여유, 최종 패스 overfull 0, 초록 199단어.
- 익명 저장소 점검: 작업 트리에 저자명·소속·로컬 경로·이메일 0건, 커밋 4개 모두 Anonymous. 로컬 git 설정 user.name은 실명 계정이라 커밋할 때 반드시 `-c user.name=Anonymous -c user.email=anonymous@example.org`. README 원고 대응 열을 현재 표·절 번호로 갱신하고 새 도구 2개(`sac2027_rq1_table.py`, `sac2027_recall_norm.py`) 행 추가(미커밋).

## 38. 6차 지적 4건과 acm13 재평가 반영 (v20j, 2026-10-09)
- "cutof": 원본 철자는 맞지만 4.3의 "cutoff);"에서만 PDF 텍스트 층이 ff 합자 뒤 ")"를 만나 "cutof)"로 추출됨(화면 렌더링은 정상, pdftotext -raw로 재현; 다른 "cutoff" 14곳은 정상). 괄호를 풀어 "...training losses; no input of a scored row depends on them. All matched comparisons use that risk set; only the KRAFTON adaptation keeps its native risk sets at the cutoff."로 바꿈. 재추출에서 "cutof" 0건.
- 표 2 미인용: 7.2 첫 문단(H0) 끝에 "(Table~\ref{tab:objective})".
- R-precision: 6.2 지표 목록에 사후 추가임을 밝혀 정의("Post hoc, we add R-precision, ... which unlike recall@3 can reach 100% at every boundary; it needs the realized E_g, so it is not a rule a reviewer could apply."). acm13 P0의 "no ceiling"(100% 상한은 있음)과 P1(실제 탈락 수 필요)을 같은 문장에서 해결. 표 4 주석 "R-precision (added post hoc)". 7.4는 "R-precision (Section 6.2) gives the same picture"로 줄임.
- R-precision 구간: `tools/sac2027_recall_norm.py`에 recall@3와 같은 경기 부트스트랩 표본(extra_v16 시드, 2,000회)과 DerSimonian–Laird 통합을 추가(`control_evaluation/recall_norm/contrasts_per_cohort.csv`, `pooled.csv`; 기존 per_cohort·arm_means 값은 2.7e-15 이내 동일). recall@3 대조는 저장된 extra_v16 값을 그대로 재현(GBDT−KRAFTON+phase 0.005736 [0.000105, 0.011367] 등). R-precision: GBDT−KRAFTON+phase +0.56 [0.16, 0.95]pp(9/11), GBDT−규칙 +8.8 [7.8, 9.8]pp(11/11), PTR−KRAFTON+phase +0.38 [0.004, 0.75]pp(8/11, 본문 미기재), PTR−GBDT −0.17 [−0.49, 0.16], 합집합−GBDT +2.06 [1.54, 2.58]. 7.4에 "GBDT exceeds phase-adjusted KRAFTON by 0.56 [0.16, 0.95] pp, under a tenth of its 8.8 [7.8, 9.8] pp gain over the rule."
- 같이 고친 것: 7.4의 "and PTR-Former by 0.40"이 "GBDT가 PTR-Former를 0.40 앞선다"로 읽혀(값은 PTR−KRAFTON+phase, sesoi_top3) "and PTR-Former exceeds it by 0.40"; 4.3 "before the next zone is drawn" → "revealed"(같은 문단의 reveal과 용어 통일, acm13 P2).
- 지면(11쪽으로 넘친 참고문헌 2개 회수, 마지막 줄 1–2단어인 문단만 손봄): H0 "With all else matched ... truncating Cox", 3절 "we truncate Cox", 4.2 "Karakin (six)", 5.2 "Neither reproduces original performance", 7.2 가중 문단 "C5 held", 공동 축소율 목록 콜론, 7.4 "(both CIs exclude zero)", 표 3 캡션 "recorded for the two largest cohorts and the pooled nine". 10쪽, 마지막 쪽 오른쪽 단 약 4줄 여유, overfull 0, 그림·표 배치 v20i와 동일.
- acm13 P0의 KRAFTON γ_k 검증 적합은 37절 W7에서 코드로 확인함(보고판 KRAFTON-tf-h0는 검증 위험집합에서만 적합).

## 39. acm14 심사 반영 (v20k, 2026-10-10)
- W2(P0): 초록 "so the metric mostly rewards separating matches by risk" → "so the metric is sensitive to match-level risk differences"(쌍 비율이 보여 주는 범위로 한정).
- W5·W6(P0) 사후 점검 두 가지, 저장된 목적함수 통제 예측으로 계산(재학습 없음; `tools/sac2027_count_error.py` → `control_evaluation/count_error/`; 목적함수 분석과 같은 경기 재표집 시드, 3시드 평균, 9코호트 DL 통합). 7.2 끝에 "Post hoc checks" 문단 추가.
  - 사람 팀만(team_ai_frac==0): BCE−다음 단계 Cox 단계 안 +0.0066 [0.0042, 0.0088](Erangel), +0.0087 [0.0064, 0.0109](Taego), 9코호트 +0.0088 [0.0061, 0.0115]; 경기 안 +0.0021, +0.0007, +0.0017 [0.0002, 0.0032].
  - 경계별 탈락 수 오차 |Σp̂ − ΣY|의 평균: BCE 0.841/0.795, 다음 단계 Cox 0.869/0.820, 조건부 로짓 0.861/0.816, Cox+BCE(w=1) 0.849/0.808, w=4 0.843/0.803. BCE−Cox −0.028 [−0.036, −0.020], −0.025 [−0.033, −0.016], 9코호트 −0.029 [−0.044, −0.014](9/9); BCE−조건부 로짓 −0.021, −0.021, 9코호트 −0.034 [−0.050, −0.018]; 조건부 로짓−Cox −0.007 [−0.013, −0.002], −0.004 [−0.008, 0.000], 9코호트 +0.004 [−0.002, 0.010]. 경계당 실제 탈락 수 평균 2.47/2.14.
- W6(P1): 6.2에 경기 안 AUC 대상 경계 비율 "(90% of 15,285 boundaries; the rest have no elimination)"(13,808/15,285; 모든 팀이 탈락한 경계 0).
- W4: 7.2 9코호트 문단에 "(same test patch)".
- 5.1의 151개와 6.1의 153개 입력: 153 = 151개 특징 + 팀 위치 2개(`model_stgcn/ptr/data.py`의 pos_in_token) → "the same 153 inputs (the 151 features plus team position)".
- 반영 안 함: W1(관련 연구·서론이 이미 "알려진 성질을 통제 비교로 함께 검정"으로 서술), W3 명시적 경기 공통 항 모형(새 학습 필요), SESOI 0.005의 사전 근거(설계 문서에 값만 있고 근거 기록이 없어 만들지 않음; 7.4의 사후 환산이 정직한 답), Q4 cutoff–경계 사이 탈락 팀 수(점수 대상 제외 규칙은 4.3·4.4에 있음), 그림 글꼴.
- 지면(11쪽 → 10쪽): 초록 끝 문장 "should also be evaluated within groups"(196단어, 전폭 한 줄 회수), 3절 "Cox ranks", 4.4 "We drop boundaries ... over 60 s", 6.2 "Within-match AUC primarily pools", 7.2 "There, BCE keeps", 7.3 "concentrate in early phases", 7.4 R-precision의 절 참조 삭제·"0.005 of within-match AUC", 8.2 "affect them"·"others", 코드 공개 "Training, evaluation, and figure code". 10쪽, 마지막 쪽 오른쪽 단 약 3줄 여유, overfull 0, 그림·표 배치 v20j와 동일.

## 40. 초록을 평가 단위 원리부터 (v20l, 2026-10-10)
- 사용자가 가져온 재구성 계획(ChatGPT)과 원고를 대조: 서론(검토 과제 → 평가 단위 불일치·중심 질문 → 단위가 보상하는 것과 학습 신호 → 통제 검정 → 입력·구조 → RQ1/RQ2 → 기여 3개), 3절(평가 단위 정의 → 두 경기 예시 → π 분해와 0.15–2.8% → Metrics·Losses·Horizon → 가설로 연결), 7절 순서(단위에 따른 변화 → RQ1 원인 → 입력·구조 → 전체 설계·검토 효용), 결론이 이미 계획의 구조와 같음. 바꾼 것은 초록과 7.1 제목뿐.
- 초록(196단어; research-paper-writing 스킬의 과제 → 어려움 → 통찰 → 기여 → 결과 → 원칙 순서): "What an AUC rewards depends on which pairs it compares."로 시작해 검토 과제와 경기 간 통합 평가의 불일치, 3% 미만 쌍 비율, Cox형 손실과 교차 엔트로피의 학습 신호 차이, 통제 비교 결과(단계 안 0.006–0.009, 경기 안 실질적 동등), 조건부 로짓·결합 손실, 9코호트(같은 보류 버전), 설계 판단 변화(팀 어텐션 0.011 대 0.001, 서식지 입력 경기 안 0.016), 원칙 순. recall@3 문장은 200단어 제한으로 뺌(기여 목록·7.4에 있음).
- 계획에서 따르지 않은 것: 13,948경기와 시험 데이터의 쌍 비율을 한 문장에 섞기, 서론에 쌍 비율 수치(서론 무수치 원칙), "closes most of the gap"(가중치에 따라 닫힘으로 유지), "사라진다/vanish", 학습 신호 문장 삭제, 그림 1 개념도 추가(3절 두 경기 예시가 같은 역할, 지면), 관련 연구 순서 뒤집기(현재 순서가 차별점 문장으로 끝남).
- 7.1 제목 "Overview" → "What the Evaluation Unit Changes". 10쪽, overfull 0, 그림·표 배치 동일, 마지막 쪽 약 3줄 여유.

## 41. A_W·A_B를 결과에 직접 제시, 그림 2 재배치 (v20m, 2026-10-11)
- P0-1: 3절 식 π A_W + (1−π) A_B의 두 항을 결과에서 수치로 제시. 저장 산출물(`objective_match_offset/contrasts.csv`, `rq2_match_offset/pooled.csv`)의 `original_between_match`(A_B, 단계 평균)와 `within_match_phase_mean`(A_W, 같은 단계 가중)을 사용, 새 계산 없음. 7.1: "Within phases the same two gain 0.006–0.011, on different-match pairs: averaged over phases, BCE's advantage over next-phase Cox in the two largest cohorts is +0.0060 and +0.0076 on A_B but +0.0008 and 0.0000 on A_W, team attention's is +0.0109 against +0.0026, and the habitat block's is +0.0150 and +0.0155."(값: BCE−Cox A_B 0.006038/0.007608, A_W 0.000825/−0.000021; 팀 어텐션 A_B 0.010918, A_W 0.002612 [−0.0003, 0.0055]; 서식지 A_B 0.015005, A_W 0.015474.)
- P0-3 그림 2: 열 순서를 within phase | within match | within match, equal phase weights | within phase, centered로 바꿔 두 평가 단위를 나란히 둠. 열 머리를 굵은 이름 + 회색 설명 두 줄로(mostly different-match pairs / pairs pooled over boundaries / equal phase weights / match-phase mean removed). 캡션에 "The first column is close to the different-match AUC A_B and the third is the same-match AUC A_W, averaged over phases (Section 3)." `tools/sac2027_figure_eval_units.py`(UNITS 순서, HEADS), 이전 그림은 `archive/fig_eval_units_v20l.pdf`. 대체 텍스트 순서도 맞춤.
- P1 SESOI: 6.3에 "no external standard sets this margin, and Section 7.4 translates it into review terms post hoc"(사전 근거는 기록이 없어 만들지 않음).
- 그대로 둔 것: 서론의 평가 단위 불일치는 이미 둘째 문단, 사전 기록·추가 코호트 관계는 6.1·표 3, 회고적 위험집합은 4.3·8.2에 있음. 코호트 이질성(I²)·시드 변동과 집단 간 위험 기술 통계는 지면상 생략.
- 지면: 7.1의 중복 경고 문장("A statistically positive difference need not be a practical gain …") 삭제, 7.1 안내 문장 축약, 그림 2 캡션 "in the two largest cohorts". 표 2 `\input`을 7.1 앞으로 옮겨 표 2가 6쪽에 남게 함(옮기지 않으면 8쪽으로 밀림). 10쪽, overfull 0, 표 2·3·4 위치 v20l과 같음, 마지막 쪽 약 3줄 여유.

## 42. 분석 시점 서술 제거, 역할로 서술 (v20n, 2026-10-11)
- 사용자 지시: "결과를 관측한 후 가설을 설정했다", "그 시점에서 테스트되었다", "주요 분석 이후 추가되었다" 같은 시점 서술은 인과가 어색하므로 뺀다. 같은 정보는 코호트의 역할로 서술(두 큰 코호트 = H0–H3의 개발 코호트, 나머지 9개 = 검정 코호트, 예측은 해당 실행 전에 기록).
- 6.1 목적함수 통제: "After the main analyses we added, in all 11 cohorts, …" → "In all 11 cohorts we also train …".
- 6.1 가설: "We formed these hypotheses after observing H0–H3 in the two largest cohorts; by then the complete designs … had been trained and tested in all 11 cohorts, so the nine-cohort replication tests the matched objective controls, not a first look at the test patch." → "The two largest cohorts are the development cohorts for H0–H3 and the nine further cohorts their test cohorts, new for the objective controls but not for the test patch, on which the complete designs of Section 7.4 are also evaluated."(시험 패치가 처음 보는 자료가 아니라는 공개는 유지).
- 6.2: "in the two largest cohorts, centering was added after the objective results were examined" 삭제(두 큰 코호트는 개발 코호트, 중심화는 진단으로 이미 규정).
- 8.2: "In the two largest cohorts the explanation and the centering diagnostic followed the objective results" → "H0–H3 are tested only in the nine further cohorts".
- 7.2·표 3: "the nine cohorts not used to form the explanation" → "the nine test cohorts"; 표 3 캡션 C 설명 "added after the main analyses" 삭제("also recorded for each of the two largest cohorts").
- 표 2(`tools/sac2027_rq1_table.py`로 재생성, 값 동일): 캡션 "H0–H3 were formed after these results, and H4a was recorded before its runs" → "(H0–H3 are tested in Table 3; H4a was recorded before these runs)"; 각주 b "Added after the main analyses, with predictions recorded before running" → "Predictions recorded before running".
- 유지: 사전 기록 문장("recorded before running/training"), "post hoc" 표시(R-precision, 사후 점검, 오프셋 변형, SESOI 환산).
- 10쪽, 최종 패스 overfull 0(첫 패스에만 1.6pt), 표 위치 동일(표 2는 6쪽), 마지막 쪽 약 6줄 여유.

## 43. 인과 순서 전면 정리 (v20o, 2026-10-11)
- 원칙: 질문 → 이론에서 나온 기대 → 설계·사전 기록 → 결과. 우리 수치와 점검 결과는 결과 절·표 주석·초록·기여 목록에만.
- 서론 2문단: 결과 미리보기("it changes the picture … both advantages … shrink to near zero …")를 지우고 "For a match review, the within-match comparison is the one that matters."로 끝냄. 3문단 첫 문장 "The explanation we test starts from …" → "The two units can reward different things."(설명할 결과를 먼저 보이지 않음). 4문단 "To test this explanation" → "To test this expectation", "the remaining within-phase gap" → "any within-phase gap".
- 3절: "in our test data, same-match pairs are 0.15–2.8%"를 7.1 첫 문장으로 옮기고, 3절은 "With M matches of similar size, π≈1/M, so with many matches within-phase AUC is decided by different-match pairs."(이론만).
- 4.3: 입력 재구성 점검을 절차로만("We check this by rebuilding … and with a builder that ignores the cutoff."), 결과(453개 전부 재현, 차단 없는 빌더는 450개 변경)는 7.1 끝으로.
- 5.2: "Neither reproduces original performance" → "Neither is meant to reproduce the original results"(재현 실패라는 결과처럼 읽히지 않게).
- 6.1 가설: 개발·검정 코호트 문장을 지우고 사전 기록 내용만("Predictions were recorded before the corresponding runs: for H4a in the two largest cohorts …; for H4b there …; and for the nine further cohorts, seven predictions for H0–H3 and H4a").
- 6.2: "(90% of 15,285 boundaries …)"를 표 4 주석으로("within match: pairs pooled over the risk sets with both classes (90% of the 15,285 test boundaries; the rest have no elimination)").
- 7.2·표 3: "nine test cohorts" → "nine further cohorts".
- 유지: 표 2 캡션 "(H0–H3 are tested in Table 3; H4a was recorded before these runs)", 8.2 "H0–H3 are tested only in the nine further cohorts", 사전 기록·post hoc 표시, 8.1 "although we did not predict it"(RQ2 무방향과 일치).
- 10쪽, overfull 0, 표 위치 동일, 마지막 쪽 약 12줄 여유.

## 44. 결론을 서론 질문의 답으로 (v20p, 2026-10-11)
- 사용자 지시: "뭘 했다"의 나열이 아니라 서론의 질문을 증거와 함께 완결하는 결론.
- 구조(research-paper-writing 결론 가이드): 질문 → 답("We asked whether conclusions from AUC comparisons … depend on the evaluation unit, and in these data they do.") → 단위별 증거(단계 안 0.006–0.009·0.011 대 경기 안 0.002 이하, 호라이즌·주요 입력은 0.01 이상) → 원인(RQ1: 다른 경기 쌍에서만, 조건부 로짓 ≈ Cox, 가중치에 따라 닫힘, 9코호트 사전 기록 충족, 설명 안 되는 작은 경기 안 우위) → 복기 효용(RQ2: 최대 0.6%p, 규칙 대비 이득의 10분의 1 미만) → 통찰("A pooled AUC can therefore credit a component for separating groups rather than for ranking the cases a decision compares.") → 권고와 범위(다른 집단 문제는 미검증, 한 게임·한 보류 버전).
- 274단어, 두 문단. 10쪽, 최종 패스 overfull 0(첫 패스 2.1pt), 표 위치 동일, 마지막 쪽 약 3줄 여유.
- (v20q, 같은 날) 사용자: 너무 길다, 핵심 질문에 핵심 답만. 109단어 한 문단으로: 질문 → 답 → 단위별 증거 한 문장 → 원인 한 문장(조건부 로짓 ≈ Cox) → 권고. 복기 효용·9코호트·잔여 우위·범위 문장은 본문(7.2·7.4·8.2)에 있으므로 결론에서 뺌. 긴 판은 `archive/main_v20p_long_conclusion.tex`. 10쪽, overfull 0.

## 45. Jennifer Widom 문체로 전면 재작성 (v21, 2026-10-11)
- 사용자 지시: 핵심 질문을 중심으로 처음부터 다시, "누더기를 짜깁기한 느낌" 제거.
- 구조: 1 서론(Widom 다섯 질문 순: 문제 → 왜 중요한가 → 왜 어려운가(세 가지 성질) → 기존 평가가 왜 못 푸는가 → 우리 접근(두 단위·손실의 학습 신호 기대) → 검정 방법 → 입력·구조·전체 설계, RQ1/RQ2, 기여 3개, 개요) / 2 평가 단위와 학습 신호(두 경기 예시 → π 분해 → Metrics·Losses·Horizon → Hypotheses H0–H4b: 이론에서 바로 가설) / 3 데이터·과제 / 4 입력·설계 / 5 실험 방법(5.1 목적함수 통제·결합 손실 식에 가중 w·조건부 로짓·오프셋·사전 기록 내용, 5.2 입력·구조 대조, 5.3 지표, 5.4 통계) / 6 결과(6.1 단위가 바꾸는 것, 6.2 호라이즌·목적함수 통제, 6.3 설계 이득의 경기 안 유지, 6.4 전체 설계·복기 효용) / 7 관련 연구(Widom 권고대로 뒤로) / 8 함의·한계 / 9 결론(109단어 유지).
- 수치·표·그림은 그대로(드리프트 점검: 숫자 누락·변경 0, 금지 표현 0, 참조 모두 해석). 표 2·3 캡션의 가설 참조를 2절로(`tools/sac2027_rq1_table.py`로 표 2 재생성).
- 시점 서술 두 문장 추가 삭제(이전 정리에서 남은 "These two-cohort results came before …", "All losses were designed in the two largest cohorts …").
- 서브에이전트 드리프트 점검 반영: 서론 "compared by AUC" 단정 삭제, "predict" → "suggest", 신규성 주장은 인용 연구 범위로("those studies treat the grouping as a property of the metric. We add the loss side …"), 어텐션 조건문 복원, "often small"·"well before" 삭제, "can arise", AUC·winner accuracy·C-index·horizon을 첫 사용에서 정의, 기여 목록에 "(within a pre-set margin)" 복원, 6.2 제목을 중립으로, 서론의 사전 기록 문장을 정확히("last three probes … nine further cohorts, centering included").
- 참고문헌 번호는 첫 인용 순이라 관련 연구 이동으로 일부 바뀜(예: Yan22가 [39]).
- 10쪽, 최종 패스 overfull 0(첫 패스 2.0pt), 그림·표: 표 1 p3, 그림 1 p4, 표 2 p6, 그림 2 p7, 표 3 p8, 표 4 p9. 마지막 쪽 약 2줄 여유. v20q는 `archive/main_v20q_pre_widom.*`.

## 46. 본문 자체의 인과 논리 점검 (v21a, 2026-10-11)
- 기준: 실제 실험 시점과 무관하게, 글 안에서 원인 → 결과, 전제 → 결론이 맞는지.
- 고친 것 7개:
  1. 서론 어려움 3: "inputs must end … before the outcome" → "before the prediction time"(입력은 정의상 결과보다 앞섬; 차단 기준은 예측 시점).
  2. 서론 6문단: 사전 기록 문장이 중심화를 빼는 이유가 없었음 → "Centering needs no new fit; we recorded predictions for the other three probes before running them, and for nine further cohorts before training."
  3. 2절 H4b: 오프셋 보정은 구성상 Cox의 경기 안 순위를 유지하므로 "within-match AUC stays practically equivalent"는 반증 불가 → H4a(경기 안 동등 포함)와 H4b(구성상 순위 유지 명시)로 분리.
  4. 6.2 중심화: 첫 문장 "depends on the match–phase score levels"(인과)와 끝 문장 "centering alone does not establish the cause"가 모순 → "Most of BCE's within-phase advantage goes with the match–phase score levels"; "because team inputs carry match state" → "as team inputs can carry match state".
  5. 8.1: "although we did not predict it"가 서론의 어텐션 조건부 기대와 충돌 → 삭제.
  6. 8.1: "inputs keep gains, so group-level components should be checked"(비약) → "it is the components that let scores carry a group-level term whose gains need checking within groups".
  7. 5.1 오프셋: 사후 변형이 왜 필요한지 빠져 있었음 → "Because R depends on survival to b, these summaries use eliminations in the 10 s after the cutoff; a post hoc variant …".
- 지면(11쪽 → 10쪽): 6.1 분해 문장 축약, 중심화 문단 참조 삭제, 8.2 마지막 문장을 코드 공개 문단과 합침(중복 제거), 8.1 "at a prediction time" 삭제·"for probabilities compared across groups", 관련 연구의 이전 평가 지표 반복 축약, 6.3 제목 "Design Gains Within Matches (RQ2)"(두 줄 → 한 줄), "Other targets" 두 문장을 한 문장으로.
- 10쪽, 최종 패스 overfull 0, 그림·표 위치 동일, 마지막 쪽 약 7줄 여유. 이전 판 `archive/main_v21_pre_causalaudit.*`.

## 47. 과한 실험·결과 정리, 그림 1 교체, 일반화 논의 (v21b, 2026-10-11)
- 재점검 후 적용(논증 사슬: 쌍 비율·A_B/A_W → 손실 신호 이론 → 통제 비교 → 조건부 로짓·결합 손실 → 9코호트 확인 → 서식지 대 어텐션 → 복기 효용).
- 삭제: 평균 점수 AUC(정의·수치; 역할은 중심화·A_B/A_W·탈락 수 오차가 대신), 구조 비교 점추정 3개(판정만 유지), 어텐션 표준편차 진단(+0.0034), Sagong 입력 관련 수치(+0.046, +0.106/0.101)와 KRAFTON 입력 단독(0.007/0.012)·"complementary" 주장, 가중치 스윕 통합 회복률 5개(코호트별 10개는 단조 증가라는 사전 기록 예측의 근거라 유지), "Other targets" 문단(위상 보정 0.068→0.012는 Overall로), 표 4의 Human AUC·log loss·Brier·phase cal. 열(12열 → 7열), 5.3의 log loss·Brier·calibration slope(보고 안 함).
- 축소: 오프셋 보정(식 삭제, 정의·정보 범위·사후 변형만), 5.2 입력 대조 목록(보고하는 둘만).
- 재점검으로 유지로 바꾼 것: KRAFTON 가중 손실 수치(5.1에 조건으로 있어 수치 없으면 근거 없는 주장).
- 그림 1: 파이프라인 그림(3–5절 반복, 두 단 약 7 cm) → 2절 실행 예시 개념도(한 단; A 같은/다른 경기 쌍, B 경기 A에 0.30 더하기, C 위험집합 손실과 교차 엔트로피가 그 이동을 보는가; `tools/sac2027_figure_concept.py`, `figures/fig_concept.pdf`). 본문 참조 Fig. 1A–C, 옛 Fig. 1B/1D 참조는 본문 서술로.
- 그림 2를 0.88\textwidth → \textwidth(글자 7 pt 그대로 인쇄).
- 8.1 "Where the units should be separated": 집단 내·집단 간 AUC를 나눠야 할 상황을 인용과 함께(병동 조기경보 Churpek16, 군집 자료 판별 vanKlaveren14·Riley16, 사용자별 AUC Zhou18, 질의별 평가 Jarvelin02와 질의 간 보정 Yan22·Bai23, 자율주행 위험 평가 Lefevre14), 집단 수준 항을 실을 수 있는 구성요소(점별 손실, 집단 구성원 간 어텐션, 집단 수준 입력)는 집단 내 AUC 병기, 집단 간 결정이면 집단 간 성분과 보정, 손실 비교의 일반화는 미검증. 새 참고문헌 4개는 Crossref·검색으로 서지 확인.
- 숫자 점검: 지워진 수치는 위 의도 목록뿐. 10쪽, overfull 0, BibTeX 경고 0, 그림·표: 그림 1 p2, 표 1 p3, 표 2 p6, 그림 2·표 3 p7, 표 4 p9, 마지막 쪽 약 20줄 여유. 이전 판 `archive/main_v21a_pre_trim.*`.

## 48. H0–H3의 형성 경위 서술 삭제 (v21c, 2026-10-11)
- 사용자 지시: H0–H3가 두 큰 코호트 결과를 보고 만든 설명이라는 서술을 없앤다.
- 8.2 "H0–H3 are tested only in the nine further cohorts" 삭제(→ "All tests share one test patch, and the objective controls use one backbone."). 표 2 캡션 "(H0–H3 are tested in Table 3; H4a was recorded before these runs)" → "H4a: hypothesis of Section 2, with predictions recorded before these runs."
- 같이 한 것: 경위 서술이 없어지면 두 큰 코호트 결과의 H0–H3 표시가 가설 검정처럼 읽히므로(사후 가설을 검정으로 제시하는 셈) 표 2 행과 6.2 소제목에서 H0–H3 표시를 뺌("Horizon.", "Aligned losses.", "Centering."). H0–H3는 2절 가설 목록, 5.1 사전 기록(9코호트), 표 3에만 남음. H4a는 두 큰 코호트에서도 실행 전에 기록했으므로 표시 유지. 표 2 수치 동일(`tools/sac2027_rq1_table.py` 재생성).
- 10쪽, overfull 0, 그림·표 위치 동일. 이전 판 `archive/main_v21b_pre_h03.tex`.

## 49. 제목 변경 (v21d, 2026-10-11)
- "Evaluation Units for Next-Phase Elimination Risk Prediction in Battle-Royale Games" → "Which Pairs Does AUC Compare? The Evaluation Unit Changes Design Conclusions in Battle-Royale Elimination Prediction"(초록 첫 문장과 같은 질문 + 발견 + 적용 범위). 쪽 머리 제목 "Which Pairs Does AUC Compare?". PDF 메타데이터 제목 자동 갱신. 익명 저장소 README 제목도 맞춤(미커밋). 10쪽, overfull 0. 이전 판 `archive/main_v21c_pre_title.tex`.

## 50. 15차 리뷰(SAC Weak Accept 경계선) 선별 반영 (v21e, 2026-10-11)
- 사실 확인: 인용 제안 4건 서지 확인(Crossref: van Oirbeek & Lesaffre 2010 Stat Med 29(30):3160–3171; Sculley 2010 KDD 979–988; Janes & Pepe 2009 Biometrika 96(2):371–382; Sagong & Kim BigComp 2025 105–106). PUBG API 문서 "The data retention period is 14 days". 리뷰의 SAC 일정(10/2 마감)은 옛 정보, 공식 사이트에 10/16(EST)로 연장 공지.
- 반영
  - M1: 2절 분해식에 van Oirbeek & Lesaffre 인용, 손실 성질에 Sculley(점별+순위 손실 결합) 인용, 관련 연구에 공변량 보정 ROC(Janes & Pepe)·클러스터 C-index 분해 추가, Sagong & Kim(BigComp 2025) 추가. 서론 4문단·관련 연구 끝을 "두 성질은 알려져 있고, 기여는 한 과제에서 둘의 상호작용을 통제 비교로 잰 것"으로.
  - M2: 사후 마진 민감도(기존 90% CI로 계산): 0.0025에서는 표 3 동등 판정 7개 중 5개가 불확정, 0.01에서는 두 큰 코호트의 BCE 단계 안 우위도 동등(90% CI [0.0041, 0.0078], [0.0058, 0.0094]); 단위 의존성은 마진과 무관한 phase-minus-match 차이로 뒷받침.
  - M4: 2절에 "경기 안 AUC는 match–phase 안 임의의 증가 변환에 불변 → 점수 퍼짐 차이도 다른 경기 쌍에만 작용, 중심화는 이동만 제거" 추가; 6.3 어텐션 표준편차 진단(+0.0034 [0.0007, 0.0062]) 복원.
  - M6: 초록 "predictors are usually scored by AUC per phase" → "the usual landmark evaluation computes AUC per phase"(배틀로얄 선행은 그렇지 않음). 8.1에 경기 간 용도 예(기대 대 실제 탈락).
  - M7·M8·M3: 8.2에 "one configuration for every loss, so loss-specific learning rates are untested", "omit seed variance (three seeds)", 코드 공개 문단에 "the public API keeps match data for only 14 days". 5.4에 학습 설정 명시(AdamW lr 1e-3, wd 1e-4, 배치 8경기, 최대 12 epoch, 검증 log loss 조기 종료 — RQ2 체크포인트가 단계 안 AUC로 선택된 것이 아님).
  - Minor: reviewer → analyst(4곳), fewest-remaining → fewest-living, 6.1 A_B/A_W 추정 방식 명시(중심화 분석의 재표집 경기에서 쌍별 직접 계산), MAE 두 값과 약 3%, Spearman 정의(순위 출력과 최종 순위, match–phase 평균), 그림 1 캡션 "logit scale", 서식지 생태 비유 문장과 Fahrig·Nathan 인용 삭제.
- 반영 안 함: A3(두 코호트 발견 → 9개 확인 서술; 사용자 지시와 충돌), E1–E4(재학습·새 데이터·데이터 공개 불가), B1–B4·recall@K·사람 팀 공동 1차 지표(과함), 8.1 축소(사용자가 방금 확장 요청), Sagong 각주화.
- 10쪽, 최종 패스 overfull 0, 초록 196단어, 그림·표 위치 동일, 마지막 쪽 약 2줄 여유. 이전 판 `archive/main_v21d_pre_review15.*`.

## 51. acm17 리뷰 1–6 반영, 익명 저장소 계획 파일 의존 제거 (v21f, 2026-10-11)
- 원고 반영
  - P0-C: 6.4 "Which design ranks first depends on the metric" → "Among designs on one input set, the leader depends on the metric: ..."(표 4의 "GBDT, ours + KRAFTON inputs"가 세 AUC 모두 1위라 범위 한정).
  - P0-B: 5.1 CLogit "no within-phase ordering" → "no use of elimination order", 6.2 "The within-phase ordering of eliminations" → "Elimination order"(평가 단위 이름과 겹치지 않게; 뜻은 다음 단계 안 탈락 순서).
  - P0-A: 6.1 "within-phase AUC is decided by different-match pairs" → "different-match pairs dominate within-phase AUC". 8.1 경기 간 용도 예 "expected against observed eliminations"(개수·보정 문제라 AUC가 재지 않음) → "choosing which match to monitor".
  - P1-A: 6.2 CLogit 문단 "removing the match--phase level, which also drops match--phases with no or only eliminations, yields a deficit as large as Cox's, supporting the match-level account without isolating it."
  - P1-D: 8.2 코드 공개 문장에 "database loading and sample construction are not included" 추가.
  - P1-E: 표 3 각주에 "No family-wise control across rows." 추가(†‡ 문구 축약), 6.4 recall@3 기울기 "corresponds to" → "over nine design contrasts ... went with".
- 분량 맞추기: 6.2 CLogit 문단 "(90% CIs within ±0.005)" 삭제(practically equivalent의 정의 반복), 6.4 순위 문장 축약, 6.1 "(Section 2)" 중복 삭제와 A_B/A_W 계산 방식 "Computed directly from pairs"로 축약. 1–4쪽에서 줄인 문장은 절 끝 빈칸에 흡수돼 효과가 없어 되돌림.
- P1-F는 이미 반영됨(지표 설명에 가중 방식; 초록 0.001은 attention만, 결론 "0.002 or less"는 9코호트 BCE−Cox +0.0015 포함).
- 반영 안 함: P1-C 이력 표(시점 서술 원칙), 무작위 분할 언급, 자료 공개, P1-B 재학습, 8.1 축소, 초록 대체안, 제목 축약, 첫 쪽 Anonymous Author(s)·임시 DOI/ISBN(템플릿 기본값 유지), 표 글씨 확대(쪽수).
- 익명 저장소(미커밋): `tools/sac2027_condlogit_eval.py`가 공개하지 않은 계획 파일의 SHA-256을 무조건 계산 → 파일이 있을 때만(없으면 `complete.json`에 null). 미공개 계획 파일 참조 삭제(condlogit_eval, objective_controls, figure_eval_units, objective_all_cohorts, README). "nine cohorts not used to form the explanation" → "nine further cohorts"(3곳), "original rule" → "primary rule". 메인 저장소에는 코드 수정만(주석의 비공개 경로는 유지). 테스트 676 passed, 14 skipped.
- 10쪽, overfull 0, 그림·표 위치 동일, 마지막 쪽 약 1줄 여유, 초록 196단어(변경 없음). 이전 판 `archive/main_v21e_pre_acm17.*`(+ `prediction_results_v21e.tex`).

## 52. 참고문헌 점검 반영 (v21g, 2026-10-12)
- 7절: DeepHit·Dynamic-DeepHit을 Cox의 심층 변형에서 분리("the Cox model learns relative risks and DeepHit learns the distribution of event times directly").
- 8.1: 사용자별 AUC "weighted by impressions"[Zhou18]; 병동·교통 예는 "can rank"/"can compare"로 적용 예시임을 표시.
- 서지: Marshall22 쪽수 1–10, 저자 "Mavromoustakos-Blom"; PUBGBlueZone2026 제목 "Blue Zone Revamp: Dev Letter"(링크는 개발자 편지 하나); Ngiam22 제목 최종 PDF 제목 "...Predicting Multiple Agent Trajectories".
- 분량: 6.1 A_B/A_W 문장의 "is" 두 개 삭제. 10쪽, overfull 0. 이전 판 `archive/main_v21f_pre_refcheck.*`, `refs_v21f.bib`.

## 53. 참고문헌 재점검: 44개 서지·인용 문장 대조 (v21h, 2026-10-12)
- 방법: 4개 묶음을 Crossref·출판사 페이지·arXiv·PUBG 페이지로 대조, Sagong et al. 2025는 원문 PDF로, KRAFTON 지표는 벤더 코드(`metrics.py`, `evaluation.py`)로 확인.
- 본문
  - [1] "spectator-view telemetry" → "match telemetry"; 10초 간격 위치 기록은 API 문서에 없어 Sagong et al.(4.1절 "location data is recorded every 10 s")로 인용.
  - [2] 랜드마크 예측기[vanHouwelingen07]와 위험집합 AUC[HeagertyZheng05]로 인용을 나눔.
  - [4] Sagong 설계는 RF·LR·XGBoost라 "network structure" → "model structure".
  - [5] KRAFTON C-index는 가중치 없이 각 시점 생존 스쿼드 쌍을 비교: 서론 "over a match's whole elimination order" → "over the survival order of the teams alive at sampled times", 7절 "time-weighted C-index ..." → "a C-index over the teams alive at sampled times"(README의 시간 간격 가중은 손실 옵션).
  - [7] "van Oirbeek" → "Van Oirbeek". [9] Xia08은 ListMLE만 정의 → "has the Plackett--Luce form that ListMLE maximizes". [11]/[40] "miscalibration across queries" → "uncalibrated ranking scores", "scale calibration across queries" → "score calibration", Yan22 "the same" → "a related shift invariance".
  - [14] 41.1 "Blue Zone protection" → "added Jammer Pack tiers that absorb Blue Zone damage". [15] "before the Blue Zone revamp announced in June 2026".
  - [41] "(e.g., [Churpek16])", [44] "Collision-risk assessment ... scores each surrounding vehicle, so the vehicles of one scene can be compared".
- 서지: PUBGAPI 제목 "PUBG API Documentation"; PUBGBlueZone2026 "[Dev Letter] Blue Zone Revamp", URL /en/news/10280; Ying21 인쇄본 제목 "Perform Bad"(Ngiam22와 같은 기준); KagglePUBG "(Kernels Only)"; Bergstra12·Cawley10 JMLR URL 삭제(분량).
- 확인만: Sagong25a 서술(0.5단계 간격 솔로 생존, 80/20 분할 정확도, "The data that has been used is confidential") 모두 일치; KRAFTON 인용 제목은 GitHub 저장소 설명; Riley16 정정(그림 라벨)은 인용 주장과 무관; Zhou18 사용자 AUC는 노출 수 가중.
- 분량: CLogit 문단 "(rows C1--C4)" 삭제와 마지막 문장 축약, 6.1 "directly" 삭제, 8.1 마지막 문장 축약. 10쪽, overfull 0, 마지막 쪽 여유 거의 없음. 이전 판 `archive/main_v21g_pre_refcheck2.*`, `refs_v21g.bib`.
