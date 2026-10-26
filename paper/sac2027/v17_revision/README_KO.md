# SAC 2027 v17 원고 (평가 단위 중심)

- 원고: `main.tex` → `main.pdf`(9쪽). 빌드: `tectonic -X compile main.tex`.
- 보충 노트는 쓰지 않는다(SAC 부록 규정이 불명확해 보수적으로 잘라냈다). 옛 파일은 `unused_supplement/`에 보관만 한다.
- 그림
  - 그림 1: `figures/fig_pipeline_clean.tex`(TikZ, `tectonic -X compile`).
  - 그림 2: `python tools/sac2027_figure_eval_units.py paper/sac2027/v17_revision/figures`(값은 `fig_eval_units_values.csv`).
  - 그림 3(서식지): `figures/fig_rq1_effects.pdf`(v16 그림 2와 같다).
- 표 3(목적함수 통제): `python tools/sac2027_paper_tables.py --table objective --out paper/sac2027/v17_revision/tables`. 표 4(예측 7개)는 손으로 쓴 표이고 값은 `control_evaluation/objective_all_cohorts/pooled.csv`에서 왔다. 입력 대비 표(`--table inputs`)는 원고에 넣지 않고 본문 문장으로 옮겼다. 다른 표는 v16과 같다.
- 쌍 비율: `python tools/sac2027_pair_share.py` → `reports/sac2027_patch_v14/control_evaluation/pair_share/pair_share.csv`.
- Cox+BCE
  - 사전 계획: `reports/sac2027_patch_v14/COXBCE_PLAN_KO_2026-10-09.md`.
  - 학습 로그: `reports/sac2027_patch_v14/objective_controls_coxbce.log`.
  - 평가 로그: `reports/sac2027_patch_v14/analysis_coxbce.log`.
- 변경 내용: `CHANGES_KO.md`. 리뷰 대응: `REVIEW_RESPONSE_KO.md`.
- 조립 스크립트: `build_scripts/`(v16 본문에서 옮긴 블록과 저장소 갱신 절차; 저장소 갱신은 1회용이라 다시 돌리면 실패한다).
- v17c 추가 분석
  - 사전 계획: `reports/sac2027_patch_v14/OFFSET_ONLY_PLAN_KO_2026-10-09.md`, `OBJECTIVE_REPLICATION_PLAN_KO_2026-10-09.md`(각 로그 첫 줄에 sha256과 실행 시각).
  - 결과: `control_evaluation/offset_only/`, `control_evaluation/objective_all_cohorts/`(`predictions.csv`), `objective_controls_ownloss/`.
  - 단계 가중 경기 안 AUC와 원점수 중심화: `control_evaluation/objective_match_offset/`, `control_evaluation/rq2_match_offset/`.

