# KRAFTON baseline 로컬 어댑터

`data/`, `models/`, `training/`, `evaluation/`, `reporting/`, `pipelines/`, `common/`으로 부품을 나눈 실행 프로젝트다.
`third_party/krafton_pubg_winprob/adaptation/`의 기존 로컬 어댑터에서 시작해 현재 모듈 경로로 연결했다.
참조 폴더와 완료된 실험의 원본 코드는 그대로 보존한다. 신규 호출은 이 프로젝트의 정식 모듈을 쓴다.

Upstream: krafton-ai/pubg-winprob `99ed9d1fa5f1df0cb242bb42ed8a511e6c972896`, Apache-2.0.
기존 변경 내역·NOTICE·LICENSE·ADAPTATION.patch는 보존된 참조 폴더에 있다.
이 어댑터의 결과를 KRAFTON 원논문의 결과라고 부르지 않는다. 원 모델·특징·손실·표본 정의는 변경하지 않았다.
추출 전후 해시는 `docs/repository/layout_2026-10-26/krafton_adapter_plan.json`에 기록한다.
