# 기존 실험 자료·제한 소스 보존 경로

개발 코드는 `src/pubg_lab/survival/`의 역할별 패키지와 `projects/`에 있다.
이 폴더에는 완료된 실험의 checkpoints·dumps·experiment_datasets·etl_runs와 접근 제한된 splits.py만 보존한다.
분할 구현은 이번 정리에서 읽거나 이동하지 않았다. 루트 호환 실행 코드는 제거했다.

현재 호출은 `pubg_lab.survival.data.split_access`의 공개 함수 경계를 사용한다. 이 경계가 기존 공개 provenance 의존 이름을 연결한다.
