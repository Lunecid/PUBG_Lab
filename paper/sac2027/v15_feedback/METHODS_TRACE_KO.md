# 방법 정의의 구현 대응

이 문서는 v15의 설명을 기존 코드와 대조한 기록이다. 코드나 데이터 정의를 바꾸지 않았다. 경로는 PUBG 저장소 기준이며, 파일 해시는 `qa/revision_evidence.json`에 기록한다.

| 정의 | 구현 | 확인 내용 |
|---|---|---|
| 특징 cutoff | `model_stgcn/phase_v2/features.py:compute_cut_features` | 원자료를 먼저 w<a로 자르고 단계별 관측 창을 생성한다. |
| 위치 격자 | `model_stgcn/phase_v2/grid.py:nearest_sample`, `last_grid_time` | 10초 격자, ±5초 최근접 관측, 동률이면 앞선 관측. t+5<a인 격자만 사용한다. 선형 보간이 아니다. |
| 팀 중심점 | `grid.py:build_team_grid` | 해당 격자에서 관측된 health>0 또는 missing-health 멤버의 좌표 평균. 전원이 비생존이면 전체 관측 평균을 저장하지만, 생존 격자 필터를 통과하지 못해 현재 이동/경쟁 계산에서 제외된다. |
| 이동 경로 ℓ | `features.py:team_grid_features` | 단계 시작부터 cutoff까지 사용 가능한 alive 격자의 중심점 수평 변위 합. 멤버 구성 변화와 관측 간격의 영향이 있을 수 있어 개인별 실제 경로와 동일하지 않다. |
| movement-deficit proxy | 같은 함수 | max(마지막 중심점–알려진 목표 원 중심 거리−반지름,0)/예정 단계 길이 − ℓ/예정 단계 길이. 예정 길이는 학습 패치 추정치다. |
| 경쟁 밀도 | `grid.py:distance_cube`, `features.py:team_grid_features` | 마지막 alive 격자의 3차원 중심점 거리. 다른 팀마다 1회, <200m. 100/500m 민감도는 같은 구현. |
| 특징의 생존 인원 | `features.py:compute_cut_features`, `alive.py` | cutoff까지의 kill/건강한 위치 재등장 타임라인을 사용한다. 격자의 health 판정과 서로 다른 구현임을 구분한다. |
| 사후 위험집합 | `labels.py:build_labels` | T>b. 유한 T이고 T≤다음 경계이면 label=1. 최종 구간은 match end, winner는 무한 T로 검열한다. |
| 로그 단절 제외 | 같은 함수 | match end 미상 또는 모든 플레이어의 합친 position stream에서 >60초 간격이 outcome window와 겹치면 해당 경계를 제외. 개별 플레이어 로그의 완전성을 증명하는 규칙은 아니다. |
| 전역 확률 변환 | `third_party/krafton_pubg_winprob/adaptation/train.py:fit_platt` | 표준화한 score에 C=1e8 logistic을 적합한 뒤 원 scale의 A/B로 되돌린다. p=σ(As+B); validation 사용. |
| phase baseline | `third_party/krafton_pubg_winprob/adaptation/h0_fit.py` | 본문 수식과 원본 입력/우리 입력 경로의 fit split 차이를 유지했다. |
| 판정·Holm | `model_stgcn/ptr/aggregate.py:pool_contrasts`, `verdict` | gain은 Holm 단측 p<.05와 점추정≥SESOI, 그다음 90% CI의 ±SESOI 내 포함 여부, 나머지 inconclusive. 등호 포함. 동등성 CI는 다중검정 보정하지 않는다. |

## 검증의 범위

기존 prefix replay는 cutoff에서 재생성한 특징·관계 등에 미래 기록이 영향을 주는지 확인한다. 이번 리뷰 대응에서는 이 검사를 새로 실행하지 않았으며, 결과는 기존 완료 기록을 인용한다. 학습 전처리의 train-only provenance와 모델 마스크 검사는 별도다. 이들 검사를 합쳐도 b까지의 온라인 상태와 사후 T>b를 모든 후보에서 대조한 전수 감사를 대신하지 못한다.

첨부 리뷰가 제안한 redeploy 이벤트 완전성 확인, 온라인 포함/제외 교차표, 잘못된 표본에 대한 재학습은 미수행으로 남는다. 해당 감사에서 실제 정의 오류가 확인되면 그때 데이터와 관련 실험을 다시 생성해야 한다.
