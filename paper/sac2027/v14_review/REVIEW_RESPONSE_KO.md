# 리뷰 반영 기록 — 패치 재학습 완료

이 문서는 검토 의견과 실제 수정의 대응표다. 새 패치 분할의 참조 실험, 동일 시행 횟수 탐색, 추가 대조군과 GATv2 학습·분석을 완료했다. 모든 선택은 고정된 탐색공간의 검증 NLL와 대조했고, 집계에 사용한 예측 파일 해시를 확인했다.

| 검토 항목 | 반영 내용 | 근거 위치 |
|---|---|---|
| 인용 순서 | ACM의 참고문헌 항목 형식을 유지하면서 첫 등장 순으로 정렬. 39개 인용의 aux/bbl 순서 대조 | `qa/citation_order.json` |
| 패치 분할 | 40.2 학습, 41.1 검증, 41.2 시험. 모든 원시 특징과 전처리를 다시 구성. 학습 50경기 미만 Karakin은 별도 탐색 결과로 분리 | `reports/sac2027_patch_v14/protocol.json`, `cohort_counts.csv` |
| AI 모집단 | 주 RQ1을 AI 보정판으로 변경. 전체/사람 팀 AUC를 분리. AI 입력 제거와 사람 팀 손실 전용 재학습, 같은 CPU의 혼합 모집단 참조 실행 | `rq1_controls/`, `population_controls/`, `control_evaluation/` |
| 이동 지표 | cutoff 위치·보이는 목표 구역·예정 단계 길이·음수 허용을 명시. 미래 이동 부담이라고 해석하지 않음. AI/사람 분포, 양의 거리 비율, required/observed speed 분해 확인 | `rq1_controls/movement_diagnostics_train.csv`, `tidy_movement_components_ai.csv` |
| 동일 예산 튜닝 | 두 큰 코호트에서 GBDT/KRAFTON 파생 구조/PTR에 각각 16개 설정. 같은 153개 입력·BCE 목표·검증 NLL 선택. 선택한 설정을 3시드로 반복 | `tuning/search_space.json`, 각 `selection.json` |
| 손실과 예측 구간 | upstream weighted full/truncated, 표준 Cox full/truncated, BCE를 동일 설정·입력·검증 규칙으로 비교 | `objective_controls/` |
| 순서 의존 분해 | 명시적인 objective × global/phase-aware mapping의 2×2 조건을 만들고 두 순서의 평균 기여 계산. 경기 부트스트랩에서도 합산 일치 확인 | `mapping_factorial/` |
| 이력 정보 | 현재 입력과 과거 요약을 같은 입력 폭으로 비교. PTR 과거 prefix의 순서만 섞고 현재 토큰과 예측 cutoff를 고정 | `history_controls/` |
| 그래프 모델 추가 요청 | 주요 그래프 모델 계열을 조사하고 학습 그래프의 높은 연결 밀도를 확인해 edge-feature GATv2 선정. 공식 PyG 소스 대조 및 6개 검사 후 16설정·3시드 + 관계 섞기 대조 실행 | `graph_model_selection/SELECTION_KO.md`, `tuning/*/gatv2/` |
| 선행연구 코드 | 고정된 KRAFTON 원본 ZIP 대조, 출력·기울기·패딩 검사, 실제 경기 prefix replay. Sagong 논문 수식 대조 및 어시스트/캐시 수정 후 재적합 | `baseline_audit/AUDIT_KO.md` |
| 추론과 보고 | 경기 단위 paired bootstrap, 코호트 이질성, 범위가 제한된 구조 효과, 과거 자료를 다시 쓴 회고적 평가임을 명시 | `primary_results_v2/`, `control_evaluation/` |
| 관련 문헌 | Kaufman, DeepHit, Dynamic-DeepHit, 무작위 탐색·선택 편향, Shapley 분해 문헌 추가 | `refs.bib` |
| AI 사용 고지 | 각 도구가 이 연구에서 수행한 일을 기술. Codex가 새 실험을 수행하지 않았다는 문구 삭제 | `main.tex` |

리뷰의 일부 전제는 그대로 채택하지 않았다. AI·반경·랭크전, 기존 패치 holdout, 역순 분해 등은 v12에도 있던 분석으로 확인되므로 v13에 새로 생겼다고 기록하지 않는다. PTR의 최종 순위 보조 손실은 DeepHit의 event-time ranking loss와 동일하지 않다. 이전 원고의 반올림된 서로 다른 pooled 효과를 더해 '목적함수 76%'로 계산하지 않고, 공통 조건의 네 셀과 공통 부트스트랩으로 다시 계산한다. 이동 변수의 정의를 사후적으로 유리한 변수로 바꾸는 대신 원래 정의를 유지하고 구성 요소 분석과 해석 제한을 추가한다.

AME 관련 수치는 미분값과 +1 단위의 유한 차분을 구분했다. 이번 원고의 물리 단위 효과는 적합 모형에서 한 단위 증가시킨 평균 확률 차이다. 이는 관측적 모형의 요약이며 개입 효과가 아니다.

기존 v12/v13 원본은 보존한다. 새 실험 결과로 바뀌는 결론은 과거 주장에 맞추지 않고 실제 결과에 따라 갱신한다.
