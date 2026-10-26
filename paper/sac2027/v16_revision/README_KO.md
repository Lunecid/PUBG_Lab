# SAC 2027 MLA v16 — v15 리뷰 1–4번 + 첫 리뷰 우선순위 3개 반영본

v15를 기준으로 v15 리뷰 1–4번과 첫 리뷰의 우선순위 3개(logistic 기준, normal/ranked 평가, top-3 포착률·보정 기울기)를 반영한 영문 원고다. 새로 학습한 모델은 full-feature logistic 기준 하나다. 매치 오프셋 분해, 모드별 평가, top-3 포착률은 저장된 시험 예측으로 계산했다. 이어서 v16 리뷰(82점)의 3–11번을 반영했다. 참고문헌 포함 10쪽(마지막 단 약 1/4쪽 여유), 표 8개, 그림 3개, 참고문헌 43개다.

- `main.tex`: 편집 시작점. `\artifacturl` = https://anonymous.4open.science/r/nextphase-elimination-code-D1D9/
- `CHANGES_KO.md`: v15→v16 변경 사항과 새 분석 수치.
- `REVIEW_RESPONSE_KO.md`: 리뷰 항목별 반영 상태.
- `ARTIFACT_README.md`: 원고에서 옮긴 엔지니어링·검증 기록(코드 공개본에 함께 넣을 문서).
- 그림: 그림 1에 logistic 기준, GATv2, 새 지표를 넣었고, 그림 1·2의 이동 변수 라벨을 "movement deficit"으로 맞췄다. 이전 그림은 `figures/v16_pre_logistic_gatv2/`.
- 새 분석 결과: `reports/sac2027_patch_v14/control_evaluation/objective_match_offset/`, `.../structure_match_offset/`, `.../extra_v16/`, `reports/sac2027_patch_v14/baseline_logistic/`.

컴파일: `tectonic -X compile main.tex` (XeLaTeX + BibTeX). v12–v15 원본은 보존했다.
