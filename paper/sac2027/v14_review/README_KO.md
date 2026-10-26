# SAC 2027 MLA v14 패치 재학습본

v12의 예측 과제와 변수 정의를 기반으로 설명 구조를 다시 짜고, 리뷰 및 후속 요청에 따라 새 패치 분할로 실험을 다시 수행한 영문 원고다. 원본 v12/v13은 보존했다. 참고문헌을 포함한 10쪽이며, 표 7개·그림 3개·참고문헌 39개를 담았다. 변경 사항은 `CHANGES_KO.md`, 리뷰 대응은 `REVIEW_RESPONSE_KO.md`, 추가 실험 수치와 신뢰구간은 `RESULTS_KO.md`를 참조한다.

## 편집 및 컴파일

소스 ZIP은 원고에 필요한 파일을 포함한다. `main.tex`가 시작점이고, `tables/*.tex`는 분석 결과에서 생성한 표, `figures/*.pdf`는 벡터 그림이다. 참고문헌 항목의 ACM 형식은 유지하되 `ACM-Reference-Format-citation-order.bst`로 첫 인용 순서대로 번호를 붙인다.

Overleaf에서 ZIP을 새 프로젝트로 가져오고 main document를 `main.tex`, compiler를 **XeLaTeX**, 참고문헌 처리를 BibTeX로 설정한다. 로컬에서는 아래 중 하나로 빌드한다.

```sh
tectonic --keep-logs --keep-intermediates main.tex
```

```sh
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

제공본은 Tectonic으로 컴파일한다. Overleaf와 별도 TeX Live 버전의 실제 빌드는 검증하지 않았으며, 엔진·패키지 차이에 따라 줄바꿈이 달라질 수 있다. 공식 템플릿 안내와 라이선스는 `TEMPLATE_README.txt`에 있다.

## 분석 재현 경로

이 ZIP은 원고를 재컴파일하는 소스 패키지다. 대규모 원시 텔레메트리나 학습 체크포인트를 포함하지 않는다. 저장소에서 실험 프로토콜과 분할은 `reports/sac2027_patch_v14/protocol.json`, `split_manifest.csv`에 있고, 검증 NLL 선택 결과는 `tuning/*/*/selection.json`에 있다.

- 데이터 재구성·검증·참조 실험: `tools/sac2027_patch_rebuild.py`, `sac2027_patch_validate.py`, `sac2027_patch_run.py`
- 동일 시행 횟수 탐색: `tools/sac2027_fair_tune.py`
- 손실·이력·모집단·매핑: `tools/sac2027_objective_controls.py`, `sac2027_history_controls.py`, `sac2027_population_controls.py`, `sac2027_mapping_factorial.py`
- GATv2: `model_stgcn/ptr/gatv2.py`, `tools/sac2027_graph_run.py`
- 집계·원고 표·그림: `tools/sac2027_evaluate_controls.py`, `sac2027_paper_tables.py`, `sac2027_result_figures.py`
- 완료·선택·예측 해시 확인: `tools/sac2027_verify_results.py`
- 원고/PDF 검증: `tools/sac2027_verify_paper.py`

원시 데이터 재구성에는 별도 읽기 전용 데이터베이스 접근과 기존 프로젝트 의존성이 필요하다. 비밀 정보나 데이터베이스 인증 정보는 이 패키지에 포함하지 않는다.

## 투고 형식

익명 검토용이며 저자·DOI·ISBN을 임의로 만들지 않았다. SAC 2027 공개 지침은 기본 8쪽과 추가 최대 2쪽을 안내한다: https://www.sigapp.org/sac/sac2027/authorkit.php . 추가 페이지 비용 및 실제 업로드 조건은 적용되는 안내를 따른다. 저자들은 제출 전에 과학적 해석과 AI 사용 고지의 도구·버전을 확인해야 한다.
