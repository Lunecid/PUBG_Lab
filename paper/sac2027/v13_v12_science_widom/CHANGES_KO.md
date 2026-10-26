# v13: v12 과학적 내용 + Widom 설명 구조

작성일: 2026-10-08. 영문 원고, ACM SAC 2027 MLA 투고를 염두에 둔 익명 검토용 편집본.

## 편집 기준

과학적 내용의 기준은 `BIGCOMP2027_EN_v12_2026-10-07.tex`와 해당 PDF다. `BIGCOMP2027_Widom_revised.pdf`에서는 설명 순서와 독자 안내 방식을 채택했다. 원본 파일을 수정하지 않았고, 새 실험이나 결과 재계산을 수행하지 않았다. 첨부 문서의 문장은 편집 자료로만 취급했다.

원고 제목은 **Next-Phase Elimination Risk in Battle Royale: Separating the Contributions of Inputs, Objectives, and Model Structure**로 바꿨다. SAC의 ML 독자가 논문의 핵심을 예측 목표·입력·모형 구조의 조건부 비교로 파악하도록 했다. Habitat 관점과 RQ1의 연관성 분석은 유지했다.

## 구성과 표현

| 부분 | 이번 수정 |
|---|---|
| 초록 | 186단어. 예측 대상 → 자료·설계 → 핵심 결과 → 적용 범위 순으로 정리했다. |
| 서론 | 문제의 정의, 활용 가치, 세 가지 어려움, 선행 설계의 다른 예측 목표, 접근법, RQ·기여 순서로 썼다. |
| 3절 | 데이터·cohort·분할 → 구체적 시간 예시 → cutoff → risk set·label·estimand 순서로 구성했다. |
| 4절 | Habitat 입력 정의 → RQ1 해석 모형 → RQ2 예측 설계 → RQ3 조건부 분해 순서를 유지했다. |
| 5절 | 예측 지표, 통계적 비교, 학습·강건성 설정을 독립된 평가 절로 모았다. 학습 설정 각주는 본문으로 옮겼다. |
| 6절 | RQ별 결과를 유지하면서 AI 팀 민감도, phase 차이, lite model, 분해 순서, 구조 효과, patch holdout, 입력 상보성을 명시했다. |
| 7–8절 | 예측 목표를 먼저 정렬해야 한다는 실증적 기여를 중심으로 논의·결론을 정리했다. 관측 연관성, adaptation, 현재 학습 조건이라는 한계를 유지했다. |

`70.6%`는 AUC 자체의 비율이 아니라 **within-phase AUC − 0.5의 보존 비율**임을 명시했다. 네 habitat constructs, RQ1 회귀변수, 네 변수 lite model도 구분했다.

성능표·그림의 cohort 산술평균과 본문·분해표의 random-effects pooled estimate가 서로 다른 집계임을 밝혔다. Sagong 경로의 GBDT 시작점 0.699와 XGBoost adaptation의 0.686도 구분했다. 약 65%·94%는 특정 경로에 따른 기여이며 독립적인 인과적 기여율이 아님을 설명했다.

구조적 이득에 대해서는 within-phase 이득과 within-match의 작은 효과를 구분했다. Habitat block의 유용성이 생태학 이론의 독립적 검증을 뜻하지 않는다고 명시했다. Sagong의 retrospective 변형 결과도 원 논문의 누출을 확정하는 증거로 서술하지 않았다.

## v12에서 반드시 유지한 내용

| 항목 | 유지 내용 |
|---|---|
| 자료·분할 | 13,948 matches, 817,766 samples, 12 cohorts; 원래 수집 범위와 match-level split |
| 정의 | 10초 cutoff, 새로 공개되는 zone 제외, event-based label, 위험집합, 세 가지 관계 정의 |
| 핵심 수식 | 번호 수식 3개를 v12 LaTeX와 그대로 유지 |
| 표 | 5개 표의 데이터 행·수치·부호·정밀도·강조를 유지 |
| 그림 | 원본 벡터 PDF 3개를 바이트 단위로 그대로 사용 |
| 인용 | v12의 32개 인용 키와 원본 `.bib` 유지 |
| AI 팀 검증 | AI indicator, 반경 변화, ranked-only 점검과 movement deficit의 약화 |
| 시간 분할 | patches 40.2–41.1 → 41.2 검증, 동일 학습량 random split 비교, pooled/within-phase 판정 차이 |
| 역순 분해 | objective를 먼저 바꿨을 때 +0.034 [0.029, 0.040]이라는 결과를 각주에서 본문으로 이동 |
| 원래 과제 | KRAFTON의 winner accuracy·C-index, Sagong solo survival 및 retrospective 비교 |
| 구조 검증 | SESOI·Holm·REML/Hartung–Knapp, 초기화·epoch·ranking-loss 점검 |

Widom 버전에서 누락됐던 Heagerty–Zheng, Naeini et al., Schuirmann 인용을 v12 기준으로 포함했다. KRAFTON score는 v12처럼 **cutoff a에서의 값**으로 유지했다.

## 조판

공식 SAC 2027 배포본의 `acmart.cls`와 `ACM-Reference-Format.bst`를 사용했다. `sigconf,anonymous,nonacm` 설정의 **9쪽** 영문 편집본이다. `nonacm`은 임의의 DOI·ISBN·권리 문구를 넣지 않기 위한 검토용 설정이며, 최종 출판 설정은 아니다.

표 4는 판별력·확률 정확도·원래 과제 성능을 열 묶음으로 구분했다. 모든 설계와 지표를 한 표에서 비교할 수 있게 했다. 그림 번호와 본문 참조를 등장 순서에 맞춰 정리했다. 본문 글자 크기나 여백을 임의로 축소하지 않았다.

현재 9쪽이므로 기본 8쪽 분량을 1쪽 넘는다. [SAC 2027 Author Kit](https://www.sigapp.org/sac/sac2027/authorkit.php)의 full paper 안내에는 추가 비용으로 최대 2쪽을 더 허용한다고 되어 있다. 따라서 이 파일을 기본 8쪽 원고라고 표시하지 않는다. 최종 업로드 단계의 트랙 지침과 출판 설정 확인은 별도다.

## 검증 결과와 범위

- Tectonic으로 실제 LaTeX·BibTeX 컴파일을 완료했다.
- 정적·PDF 무결성 검사 70개를 통과했다. 원본 입력 파일의 SHA-256도 변하지 않았다.
- 5개 표의 데이터 행, 번호 수식 3개, 결과 절의 모든 inline math 출현, 32개 인용 키를 v12와 대조했다.
- PDF 내부 링크 84개가 모두 유효한 대상에 연결된다. 미정의 인용·상호 참조 및 가로 넘침은 없다.
- 전체 9쪽을 이미지로 렌더링해 표·그림·수식·본문의 잘림과 겹침을 확인했다. 모든 PDF 글꼴은 임베딩되어 있다.
- 마지막 참고문헌 페이지의 단 균형 처리에서 1.14pt의 내부 vertical-box 경고가 남는다. 렌더링상 페이지 밖으로 나가거나 잘린 내용은 없다.
- ZIP을 별도 임시 폴더에 풀어 재컴파일하고, 제공 PDF와 페이지 수·추출 텍스트·페이지 렌더링을 비교했다.

이는 **원고 보존·컴파일·표시 검증**이다. 원자료나 학습 결과를 독립적으로 재현한 검증은 아니며, 참고문헌 32개의 서지사항 전체를 외부 데이터베이스와 재대조한 작업도 아니다. 상세 기록은 작업 폴더의 `qa/validation.json`에 있다.

## 저자 확인이 남는 두 가지 표현

1. v12의 SESOI 민감도 문장에서 겹쳐 보이는 범위를, 보고된 반올림 기준값을 유지하여 `0.0026과 0.0036 사이: inconclusive; 0.0026 이하: gain`으로 풀어 썼다. 판정값을 재계산한 것은 아니므로 정확한 경계값은 원래 분석 기록으로 최종 확인해야 한다.
2. AI 고지는 v12의 Claude 사용·저자 검토 진술과 이번 Codex 편집 사실을 시간적으로 구분했다. 사용 모델·버전과 이번 수정본에 대한 저자 최종 검토 완료 여부는 제출 전에 확인해야 한다. 이번 편집에서 저자가 새 원고 전체를 이미 검토했다고 가정하지 않았다.

구체적으로 정의되지 않았던 movement deficit의 이동거리 누적 구간도 추정하여 추가하지 않았다. 추후 재현성 보강 시 구현을 확인해 명시할 수 있다.

설명 구조의 참고: Jennifer Widom, [Tips for Writing Technical Papers](https://cs.stanford.edu/people/widom/paper-writing.html). Widom 지침은 학회 조판 템플릿과 별개의 글쓰기 기준으로 적용했다.
