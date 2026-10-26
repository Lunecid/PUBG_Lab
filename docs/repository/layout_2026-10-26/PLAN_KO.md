# 실제 폴더·프로젝트 분리 수정안

사용자 확인: r4의 루트 호환 파일과 역할별로 분리되지 않은 파일 나열은 요청한 정리가 아니다.
현재 요청에 따라 실제 import·명령·시험을 옮기고 r4가 생성한 루트 호환 파일을 제거한다.

```text
projects/
  world_model/{data,training,evaluation,reporting,common}/
  movement/{data,training,evaluation,reporting,release}/
  battle_royale/{pipelines,evaluation,analysis,reporting,launchers}/
  sac2027/{training,evaluation,reporting,release}/
  spatial/{erangel,caldera}/
  network_analysis/{pipelines,studies,docs}/
  collector/
apps/{webapp,presentations}/
src/pubg_lab/
  survival/{models,data,training,evaluation,features,experiments,baselines,ptr,v3}/
  world_model/{data,models,features,training}/
  movement/{data,models,evaluation}/
  data/stdb/{connection,queries,temporal,pipelines,schemas}/
  analysis/network/{data,features,reporting,config,legacy}/
  common/
archive/{legacy_snapshot,design_notes}/
```

[전체 파일 이동·삭제 사전 목록](plan.json)을 따른다.
삭제 대상 374개는 r4가 만든 전달용 파일임을 AST로 확인했다. 원 모델 구현은 삭제하지 않는다.
기존 호출 코드와 시험은 새 정식 경로로 갱신하며, 루트에 호환 .py 파일을 다시 만들지 않는다.
World model 검증·학습이 함께 쓰는 자원/해시 도우미는 프로젝트 common 부품으로 추출한다.
모델·손실·표본·분할·평가 정의는 유지한다. 원본·완료된 실험 자료는 원위치에 보존한다.
따라서 model_stgcn에는 보존 실험 자료와 보호 분할만, tools에는 보호 분할만 남을 수 있다.
보호 분할 본문은 읽거나 옮기지 않는다. 이는 개발 코드가 아니라 기존 접근 제한 경로의 예외다.
원 legacy 코드는 내용 변경 없이 archive로 옮기며 원 경로·새 경로·해시를 기록한다.
