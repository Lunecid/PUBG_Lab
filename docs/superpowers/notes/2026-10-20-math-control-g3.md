# G3 작은 검산: 과거 prior와 부분 정답

대상 경기 시작이 50초라 하자. 같은 지도 donor A는 10초에 종료하고 연결을 7번 관측했다.
다른 donor B는 100초에 종료해 999,999번 관측했다. 자기 경기 C와 inner-heldout D도 7번씩 관측했다.
그러면 A만 남는다: count=7, donor_matches=1, age_days=(50−10)/86400=0.000462962963.
log1p(count)=2.07944154168, log1p(donor_matches)=0.69314718056.
B가 매우 큰 수여도 미래 donor이므로 기여는 0이다. 자기 경기와 fit 밖의 D도 0이다.
source-cell 조회와 이동 모드는 그대로 보존한다. 이 수치는 정확한 통행 성공 확률이 아니다.

정답 마스크 [1,0,1,1,1,1]은 유효 점이 5개여도 최초 30초 연속 지원이 아니다.
첫 세 원소의 AND가 0이기 때문이다. 짧은 수송 집합에는 사용할 수 있으나 30초 완전 표본으로 세지 않는다.
다음 재진입 episode의 유효 경로는 앞 재진입 사건을 보완하는 정답으로 사용할 수 없다.

검산: tests/world_model/test_g3.py의 시간·자기 경기·fold·episode 반례와
보고서의 math_check.json. 정답 시간 50을 10으로 바꾸면 strict donor_end<target_start 조건에서 A도 빠진다.
