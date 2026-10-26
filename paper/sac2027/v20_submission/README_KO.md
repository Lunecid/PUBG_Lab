# SAC 2027 v20 제출본 (공식 템플릿 + 평이화)

- 원고: `main.tex` → `main.pdf`(10쪽). 빌드: `tectonic -X compile main.tex`.
- v19m 본문을 SAC 2027 공식 LaTeX 템플릿에 옮기고(v20a), 가독성 피드백에 따라 표현을 평이하게 고쳤다(v20b). 이전 판은 `../v19_revision/`에 그대로 있다. 변경 내용은 `CHANGES_KO.md`.

## 규정 점검 기록 (2026-10-09 확인)
- 대상: ACM SAC 2027, MLA 트랙, 정규 논문, 심사용 제출(이중 맹검).
- 공식 출처
  - 저자 키트 페이지: https://www.sigapp.org/sac/sac2027/authorkit.php
  - 저자 키트 PDF: https://www.sigapp.org/sac/sac2027/authorkit/author-kit-2027-July11.pdf
  - 템플릿: `ACM_SAC_2027_Article_Template.zip`(sha256 `cabdde54…7af4`, 1,061,630 B). zip 안의 `acmart.cls`(2026/06/27 v2.19)와 `ACM-Reference-Format.bst`를 그대로 복사했다(수정 없음, sha256 동일). 실제 빌드는 인용 순서 파생본 bst를 쓴다(아래).
  - 제출 페이지: https://www.sigapp.org/sac/sac2027/submission.php (이중 맹검, 4쪽 미만은 데스크 리젝 가능, EasyChair).
- 규정과 현재 상태
  - 분량: 정규 논문 8쪽 무료 + 추가 2쪽 유료(최대 10쪽) → 10쪽.
  - 용지·여백: letter(8.5×11"), 좌우 0.75", 상하 1" → 템플릿 기본 기하 그대로(본문 x 범위 52.3–559.8pt).
  - 폰트 임베드: 18개 폰트 모두 임베드(`pdffonts`).
  - 초록 100–200단어 → 200단어. 키워드 5개 이하 → 5개. CCS 개념 포함.
  - 쪽 번호: 템플릿 기본(심사용 `anonymous`는 쪽 번호 없음, 인쇄소가 넣는다).
- 심사용 설정: `\documentclass[sigconf,anonymous]{acmart}`. 템플릿의 `natbib=false`(biblatex 경로)는 쓰지 않고, 템플릿이 허용하는 BibTeX 경로를 쓴다. 참고문헌 번호는 첫 인용 순서(`ACM-Reference-Format-citation-order.bst`: 템플릿 zip의 `ACM-Reference-Format.bst`에서 정렬 키만 인용 순서로 바꾼 로컬 파생본; 항목 서식은 동일). 템플릿 기본(알파벳순)으로 되돌리려면 `\bibliographystyle{ACM-Reference-Format}`. 템플릿의 biblatex 경로(acmnumeric)도 기본이 `sorting=nty`(알파벳순)라 인용 순서는 어느 경로든 설정 변경이 필요하다.
- 카메라 레디에서 할 일: `anonymous` 제거, 실제 저자·ORCID·소속 입력, `\renewcommand{\shortauthors}` 설정, 저작권 블록(`\setcopyright`, `\acmDOI`, `\acmISBN`)을 ACM eRights 메일의 값으로 교체, 추가 2쪽 비용 등록. 키트는 LaTeX 사용자에게 "strict"가 아닌 "Alternate" 스타일로 저작권 문단을 찍으라고 한다.

## 절 구성
1 서론(과제 → 중심 질문과 정성적 답 → 설명 → 한 일 → RQ1→RQ2 연결 → RQ·기여; 결과 수치 없음) → 2 관련 연구(차별점 문장) → 3 Evaluation Units and Learning Signals(두 경기 숫자 예시 → 쌍 분해 → 성질 표 1 → 손실 → 시간 범위; 가설 없음) → 4 데이터와 과제 → 5 입력 표현과 참조 설계 → 6 실험 설계와 평가 규약(6.1에 가설 H0–H4b와 사전 기록 범위) → 7 결과(7.1 개요, 7.2 RQ1 목적함수, 7.3 입력·구조 대비, 7.4 설계 전체와 검토 효용) → 8 함의와 한계 → 9 결론 → AI 고지. 변경 경위는 `CHANGES_KO.md` 30–31절.

## 그림·표(v20f 기준)
- 그림 1은 `../v19_revision/`과 같다(배율 0.86). 그림 2는 `python tools/sac2027_figure_eval_units.py paper/sac2027/v20_submission/figures`로 다시 그렸다(가운데 그룹 "BCE − conditional logit" 추가, 17.8×7.4 cm, 배율 0.88).
- 표 번호(v20f): 1 코호트(`main.tex`), 2 목적함수 통제(`tables/objective_results.tex`; 생성기 출력에서 weighted 두 행을 빼고 Cox + 4 BCE·Conditional logistic 두 행을 손으로 더함, 값은 `control_evaluation/condlogit/contrasts_per_cohort.csv`의 `level:` 행), 3 사전 예측(`tables/prediction_results.tex`; H0–H4a 7행 + C1–C5 5행, 값은 `objective_all_cohorts/pooled.csv`와 `condlogit/pooled.csv`), 4 주 결과(`tables/main_results.tex`, 굵은 글씨 없음).
- 접은 표: 옛 성질 표(3절 본문), 1차 판정 표(7.3 본문), 입력 블록 표(5.1 본문), 입력·구조 대비 표(7.3 본문, CI 포함).
- 새 도구: `tools/sac2027_condlogit_eval.py`, `tools/sac2027_did_rq2.py`, 수정 `tools/sac2027_objective_controls.py`(cond_logit, w 스윕), `tools/sac2027_figure_eval_units.py`. 익명 저장소에 복사했다(미커밋).

## 보관본
- `archive/main_v20a_template_only.*`: 템플릿만 바꾼 판(본문은 v19m과 같음, 11쪽).
- `archive/main_v20b_pre_plain.tex`: 1차 축약 뒤, 평이화 전.
- `archive/main_v20c_pre_restructure.*`: 평이화 뒤, 서론·결과 개요 구조 변경 전.
- `archive/main_v20d_pre_nonum_intro.*`: 서론 무수치화·3절 예시·가설 이동 전.
- `archive/main_v20e_pre_review3.tex`: 3차 리뷰 반영 전. `archive/main_v20f_pre_rereview.tex`: 재심 N1–N10 반영 전.
- `archive/main_v20g_pre_v1v4.tex`: 4차 재심 V1–V4 반영 전. `archive/main_v20h_pre_readability.*`: 가독성 정리 전. `archive/main_v20i_pre_acm13.*`(+ `main_results_v20i_pre_acm13.tex`): 6차 지적·acm13 반영 전. `archive/main_v20j_pre_acm14.*`: acm14 반영 전. `archive/main_v20k_pre_abstract.*`: 초록 재작성 전. `archive/main_v20l_pre_pairtypes.*`(+ `fig_eval_units_v20l.pdf`): A_W·A_B 제시와 그림 2 재배치 전. `archive/main_v20m_pre_timeline.*`(+ `rq1_contrasts_v20m.tex`, `prediction_results_v20m.tex`): 시점 서술 제거 전. `archive/main_v20n_pre_causal.*`(+ `main_results_v20n.tex`, `prediction_results_v20n.tex`): 인과 순서 정리 전. `archive/main_v20o_pre_conclusion.*`: 결론 재작성 전. `archive/main_v20q_pre_widom.*`(+ `rq1_contrasts_v20q.tex`, `prediction_results_v20q.tex`): Widom 문체 전면 재작성 전. `archive/main_v21a_pre_trim.*`(+ `main_results_v21a.tex`, `refs_v21a.bib`): 과한 결과 정리·그림 1 교체 전. `archive/main_v21d_pre_review15.*`(+ `refs_v21d.bib`): 15차 리뷰 반영 전. `archive/main_v21e_pre_acm17.*`(+ `prediction_results_v21e.tex`): acm17 리뷰 반영 전.
