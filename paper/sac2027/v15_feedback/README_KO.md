# SAC 2027 MLA v15 — 첨부 리뷰 선택 반영본

패치 재학습을 완료한 v14를 기준으로, 현재 증거로 가능한 수정만 적용한 영문 원고다. 모델·분할·예측값을 다시 학습하거나 바꾸지 않았다. 참고문헌 포함 10쪽, 표 8개, 그림 3개, 참고문헌 39개다.

- `main.tex`: 편집 시작점.
- `REVIEW_RESPONSE_KO.md`: 첨부 리뷰별 반영 여부와 미해결 과제.
- `CHANGES_KO.md`: 이번 수정 요약.
- `RESULTS_KO.md`: v14에서 완료된 추가 실험의 전체 결과; 이번 수정으로 수치를 바꾸지 않음.
- `METHODS_TRACE_KO.md`: 변수·시점·통계 규칙의 코드 대응.
- `qa/revision_evidence.json`: 수정에 사용한 파일과 해시.

## 컴파일

이 ZIP은 원고를 재컴파일하는 소스 패키지이며, 전체 실험 재현 패키지가 아니다. 원시 텔레메트리·데이터베이스 인증·학습 체크포인트는 포함하지 않는다.

검증한 명령은 `tectonic -C --keep-logs --keep-intermediates main.tex`다. Tectonic의 캐시가 없는 환경에서는 `-C`를 빼고 필요한 TeX 리소스를 설치한다.

Overleaf에서는 `main.tex`와 XeLaTeX/BibTeX를 선택할 수 있다. 해당 환경 자체에서 빌드를 실행한 것은 아니므로 엔진·패키지에 따라 줄바꿈 차이가 있을 수 있다. 로컬 대안은 `xelatex main.tex`, `bibtex main`, `xelatex main.tex` 2회다.

`ACM-Reference-Format-citation-order.bst`는 ACM 항목 형식을 유지하며 본문 첫 인용 순서로 번호를 붙인다. 공식 템플릿 안내는 `TEMPLATE_README.txt`에 있다.

## 해석과 제출 범위

v12/v13/v14 원본은 보존했다. 기존 사후 위험집합을 사용하므로 온라인 자격 판정의 전수 검증을 주장하지 않는다. 제출자는 AI 사용 고지의 도구·버전과 최종 저자 진술을 확인해야 한다. 현재 ZIP 제공은 공개 익명 실험 저장소의 개설이나 투고 완료를 뜻하지 않는다.
