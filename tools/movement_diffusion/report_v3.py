"""Generate measured V3 report with explicit retrospective-test limitation."""
import json,sys,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/heterograph_diffusion_experiment_v3';DATA=ROOT/'data/processed/heterograph_diffusion_v3'
def read(p):return json.loads(p.read_text())
def main():
 s=read(OUT/'comparison_summary.json');o=s['overall']['models'];verify=read(OUT/'final_verification.json');quality=read(DATA/'summary.json');audit=read(OUT/'leakage_audit.json');ablation=read(OUT/'zone_ablation.json');sensitivity=read(OUT/'condition_sensitivity.json');names={'constant_velocity':'등속도','stationary':'현재 위치 유지','observer_regression':'관측자 회귀','observer_diffusion':'관측자 Diffusion','server_regression':'서버 회귀','server_diffusion':'서버 Diffusion'};keys=['ego_meanpathADE_m','team_meanpathADE_m','joint_energy_score_m','ego_expectedADE_m','pair_distance_MAE_m'];rows=[]
 for k,n in names.items():rows.append('| '+n+' | '+' | '.join(f"{o[k][m]['mean']:.2f}" for m in keys)+' |')
 paired=[]
 for pair,values in s['overall']['contrasts'].items():
  for metric in keys[:3]:
   v=values[metric];ci=v['ci95'];adj=v['familywise95_12comparisons'];paired.append(f"| {pair} | {metric} | {v['difference_m']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | [{adj[0]:.2f}, {adj[1]:.2f}] |")
 abl=[]
 for kind in ['regression','diffusion']:
  full=read(OUT/f'evaluation_observer_{kind}_41.json')['match_macro'];no=ablation['models'][kind]['match_macro']
  for metric in keys[:3]:
   a=full[metric]['mean_over_matches'];b=no[metric]['mean_over_matches'];abl.append(f'| {kind} | {metric} | {a:.2f} | {b:.2f} | {b-a:+.2f} |')
 sen=[]
 for model in list(names)[2:]:
  a=[sensitivity['models'][f'{model}_{seed}'] for seed in [41,42,43]];sen.append(f"| {names[model]} | {np.mean([v['mean_shift_m'] for v in a]):.3f} | {np.mean([v['masked_minus_full_ego_ADE_m'] for v in a]):+.3f} |")
 sub=[]
 for group,x in s['subgroups'].items():sub.append('| '+group+f" | {x['windows']} | "+' | '.join(f"{x['models'][m]['ego_meanpathADE_m']['mean']:.2f}" for m in names)+' |')
 best=min(names,key=lambda m:o[m]['ego_meanpathADE_m']['mean']);q=quality['quality_totals'];cases=read(OUT/'visualization_cases.json')
 transport=read(OUT/'transport_audit_results.json') if (OUT/'transport_audit_results.json').exists() else None
 transport_section=''
 if transport:
  transport_rows=[]
  for group,values in transport['summary'].items():
   for model,n in names.items():
    values_m=values['models'][model]
    transport_rows.append('| '+group+' | '+n+' | '+' | '.join(f"{values_m[k]['mean']:.2f}" for k in keys[:3])+' |')
  transport_section='''## 직접 관측된 수송 모드의 보조 평가

주 검증 표본에는 긴급 호출 수송 8개, 재진입 항공기 1개만 들어 있었다. 검증 예측을 보기 전에 별도 계획을 고정해, 해당 모드가 현재 직접 관측되고 미래 개인 라벨이 하나 이상 있는 창에서 경기·모드당 최대 2개를 무작위 추출했다. 긴급 호출 수송 130경기 255개 창, 재진입 항공기 84경기 116개 창이다. 주 평가와 중복될 수 있으며, 독립된 새 경기 테스트가 아니다. 모델 재학습·선택·튜닝에는 쓰지 않았다.

아래는 모드 안에서 경기 평균 후 세 학습 시드 평균한 탐색적 결과다. 일반 이동 성능과 합쳐 전체 성능을 주장하지 않는다. 학습 표본에서 재진입 항공기 탑승 anchor는 시드별 5/9/5개, 긴급 호출 수송 anchor는 75/67/68개에 불과했다. 복원 자료의 총량과 실제 학습 노출량을 구분해야 한다. 현재 모드가 해당 수송 상태인 경우의 향후 경로 오차이며, 모든 미래 시점이 계속 탑승 상태라는 뜻은 아니다.

| 현재 모드 | 모델 | 개인 평균경로 ADE | 팀 중심 평균경로 ADE | 공동 Energy score |
|---|---|---:|---:|---:|
'''+chr(10).join(transport_rows)+'''

[사전 고정 보조 평가 계획](transport_audit_plan.json) · [모드별 실측값과 표본별 결과](transport_audit_results.json) · [주 학습·평가 표본의 모드 구성](sampling_coverage.json)

'''

 if transport:
  pickup=transport['summary']['emergency_pickup']['models'];redeploy=transport['summary']['redeploy_aircraft']['models']
  transport_section+=f"긴급 호출 수송의 관측자 Diffusion 개인 ADE는 {pickup['observer_diffusion']['ego_meanpathADE_m']['mean']:.1f}m로 등속도 {pickup['constant_velocity']['ego_meanpathADE_m']['mean']:.1f}m보다 낮다. 그러나 재진입 항공기에서도 관측자 Diffusion ADE가 {redeploy['observer_diffusion']['ego_meanpathADE_m']['mean']:.1f}m로 크므로, 항공 수송 경로를 정밀하게 생성한다고 주장할 수 없다. 수송 모드 복원과 해당 모드의 예측 정확도는 별개다. 적은 직접 학습 노출, 장거리 이동, 탑승·이탈 전이가 한계 후보이며 원인을 분해한 실험은 아직 하지 않았다.\n\n"
 content=f'''# V3 재진입·이동 모드 복원과 안전구역 조건부 경로 생성

## Material Passport

- 상태: 실제 학습·내부 검증·누수 감사 완료. 최종 테스트 추론은 수행하지 않음.
- 원자료: 날짜로 40.2를 추정한 Steam 공식 TPP squad Erangel 텔레메트리. 실행파일 build 확정 자료는 없음.
- 이번 데이터: 학습 657경기, 내부 검증 188경기, 잠금 테스트 후보 94경기.
- 학습: 정보 조건 2개 × 모델 2개 × 시드 3개 = 주 모델 12개. 안전구역 제거 재학습 2개를 추가해 총 14개.
- 각 모델: 2,000 updates × batch 8 = 16,000 표본 제시. 주 비교 총 192,000회, 제거 실험 32,000회. 전체 데이터셋을 한 번 순회한 실험은 아님.
- 시드41/42/43의 직접 학습 경기 수는 652/657/657, 서로 다른 학습 창 수는 15,889/15,886/15,903이다. 공간 사전은 학습 분할 657경기 전체로 구성한다.
- 주 평가: 같은 검증 188경기의 1,504개 창, 미래 10–60초. 회귀 1개 경로, Diffusion 20개 공동 시나리오.
- 수송 모드 보조 평가: 직접 관측된 긴급 호출·재진입 항공기 371개 창. 같은 검증 경기 풀에서 별도로 선정하고 주 결과와 분리.
- 재현: 주 모델 12개의 첫 검증 표본을 저장 checkpoint로 재생성해 float32 좌표가 정확히 일치. 전체 학습 재실행은 하지 않음.

**최종 테스트에 관한 중요한 제한:** 잠근 94경기는 이전 V2 학습에 사용된 이력이 있다. 이번 V3에서는 bank/index 자체를 만들지 않았고, 어휘·공간 사전·학습·모델 선택·평가에서 제외했다. 그렇더라도 과거의 연구 노출을 되돌릴 수는 없다. 따라서 이 집합은 ‘V3에서 잠근 후향적 홀드아웃’이며, 논문에서 완전히 독립적인 최종 테스트라고 주장하면 안 된다. 읽기 전용 DB 조회 결과 같은 40.2 조건의 추가 미사용 경기 ID는 0개였다. 진정한 외부 최종 테스트는 새 경기 확보가 필요하다. [잠금 명세](../../data/processed/heterograph_diffusion_v3/split_lock.json)

## 복원한 상태와 아직 추정인 상태

공식 문서는 `LogPlayerRedeploy`, `LogPlayerRedeployBRStart`, `LogEmPickupLiftOff`, `LogSwimStart/End`, 차량 탑승·하차 이벤트를 정의한다. 현재 DB에는 전용 저장 테이블이 없고, 기존 CDN URL 한 건의 재확인도 403으로 실패했다. 그러므로 전용 이벤트를 확보했다고 주장하지 않는다. 차량 분류·사망·위치·체력·착지 기록으로 가능한 범위를 복원했다. [공식 이벤트 스키마](https://documentation.pubg.com/en/telemetry-events.html)

| 상태 | 복원 근거와 처리 |
|---|---|
| 사망 | kill 시각부터 현재 생애 종료, 장비 상태 초기화, 속도 연결 끊기 |
| 재진입 항공기 | 사망 뒤 양의 체력과 `RedeployAircraft` 차량 기록이 관측된 시점에 새 episode 시작. 항공기 증거로 표시 |
| 재진입 추정 | 사망 뒤 새 착지+양의 체력, 또는 일정 간격의 연속 양의 체력·이동 증거. 확인 시점에서만 새 episode 시작하며 첫 후보 시점으로 소급하지 않음 |
| 긴급 호출 수송 | `EmergencyPickup` 차량 유형·클래스의 직접 기록 |
| 최초 수송기 / 재진입 수송기 | 차량 클래스 구분. 최초 착지 전 ego 표본은 학습 anchor에서 제외 |
| 항공기 이탈 이후 | 수송기·긴급 호출·글라이더 기록 이후 차량이 없어졌으면 공중 이탈 상태로 추정. 착지 또는 지상/수상 차량·고정 탑승 증거가 나오면 해제. 실제 비행 여부 확정이 아님 |
| 지상 차량 / 수상 차량 / 글라이더 / 고정 탑승 | 차량 클래스별 분리. 박격포 탑승을 지상 이동 차량으로 간주하지 않음 |
| 차량 없는 착지 이후 | 비차량 후보. 보행·수영을 확정 구분할 수 없으므로 ‘보행 정답’으로 쓰지 않음 |

845개 학습·내부 검증 경기에서 항공기 증거 재진입 **{q['direct_aircraft_reentries']:,}건**, 추정 재진입 **{q['inferred_reentries']:,}건**을 복원했다. 복귀 이후 유효 위치 기록은 {q['valid_post_reentry_positions']:,}개이며, 재진입 episode에서 시작하는 후보 창은 {q['post_reentry_anchors']:,}개다. 이 숫자는 엔진 정답과 대조한 precision/recall이 아니라 규칙으로 복원한 건수다.

![원자료 상태 복원 예시](state_reconstruction.png)

위 그림은 학습 경기에서 차량 종류로 고른 사례이며 복원 정확도의 통계적 추정이 아니다. [그림의 원자료 증거](state_reconstruction_evidence.json)에서 기록별 시각·차량 클래스·생애를 확인할 수 있다.

현재 세계 그래프에는 복귀한 팀원·적을 다시 포함한다. 예측 대상은 기준 시점에 유효한 팀원이며, 각자의 현재 생애에 한정한다. 미래에 죽었다가 다시 등장한 위치를 하나의 연속 경로 정답으로 연결하지 않는다. ‘죽은 플레이어의 미래 재진입 시각과 착지 위치까지 생성’하는 모델은 아니다.

목표 좌표는 같은 생애·같은 이동 모드의 인접 기록 간 간격이 15초 이하일 때만 **라벨용 선형 보간**을 쓴다. 사망·복귀 전이와 모드 전이를 건너지 않는다. 그 외에는 ±5초 최근접 좌표를 제한적으로 쓰고 실제 시각 차이를 보존한다. 검증 라벨의 평균 절대 시각 차이는 **{verify['target_mean_abs_clock_error_s']:.3f}초**, 목표 시각과 일치하는 비율은 {verify['label_exact_time_fraction']*100:.1f}%다. 보간도 실제 좌표 정답은 아니며 급회전·가림 구간의 오차가 남는다. 입력은 보간하지 않고 과거 15초 이내 마지막 기록만 쓴다.

공간 사전은 학습 657경기에서 새로 만들었다. 관측 연결 {quality['spatial_edges']:,}개에 재진입 경계·모드 전환·항공 수송·글라이더·박격포 이동을 섞지 않았다. 학습 시 자기 경기 기여도 추가로 차감한다. 지상·수상 차량의 연결은 여전히 같은 vehicle relation 안에 있으므로 물리적 지상 도로망으로 해석하면 안 된다. 지형은 역사적 높이·지표면 사전이며 40.2 충돌 메시는 아니다.

## 모델의 실제 변경

두 모델 모두 이종 그래프 message passing, 팀원별 과거 이력, 팀원 attention을 사용한다. 동일한 입력에는 위치·속도·관측 나이, 무장·피해 증거, 허용된 이동 모드·재진입 증거, 현재 기록된 두 안전구역 원이 포함된다. 관측자 모델은 자기 상태와 팀 HUD의 근사이고, 서버 모델은 모든 플레이어의 추출된 상태를 쓴다. 팀원의 비공개 이동 모드·장비는 관측자 입력에서 가렸다. 정확한 자기 좌표·고도 등 텔레메트리를 사용하는 **자기 상태 oracle을 포함한 관측자 근사**이며, 실제 화면에서 지각한 값만을 사용한다고 주장하지 않는다.

안전구역을 전역 그래프 평균에만 맡기지 않고 팀원마다 현재 원 중심의 상대 XY, 반경, 경계까지의 부호 있는 거리, 중심 방향 속도, 안/밖, 유효성 표시를 직접 전달한다. 두 원 × 7개 값으로 `[B,4,14]`이며, 학습된 그래프 조건과 함께 회귀·denoising 단계에 들어간다. 미래에 발표되는 원은 쓰지 않는다. 실제 HUD 공개 시각은 아직 검증되지 않았다.

| 구성 | 동일 입력 회귀 | Diffusion |
|---|---|---|
| 조건 encoder·팀원 attention | 동일 구조, 독립 학습 | 동일 구조, 독립 학습 |
| 평균 경로 | 20·40·60초의 상대 위치 3개를 예측 | 같은 평균 경로 head |
| 시간 표현 | 시작점과 세 지점 사이 선형 보간으로 10초 간격 출력 | 팀원들의 현재 noisy 잔차를 함께 집계해 세 지점의 잔차를 공동 확산 후 보간 |
| 손실 | Huber 경로 + 0.1 팀 중심 + 0.05 이동 모드 + 0.05 현재 원에 대한 거리 오차 | 같은 손실 + v-prediction 잔차 denoising |
| 생성 | 결정론적 공동 경로 1개 | cosine 32단계 DDIM, 공동 시나리오 20개 |

실제 train 표본으로 텐서 shape를 확인했다. 팀원 조건은 `[B,4,110]`, 회귀 경로는 `[B,1,4,6,2]`, Diffusion 경로는 `[B,20,4,6,2]`다. 파라미터 수는 회귀 124,794개, Diffusion 143,088개다. 그래프 노드/엣지 수는 표본별 가변이며, 이 shape 확인용 초기화 모델의 출력은 성능 평가에 사용하지 않았다. [실행 확인 텐서 계약](runtime_tensor_contract.json)

원에 대한 거리 보조 손실은 정답 경로와 같은 경계 거리를 학습하게 한다. 모든 사람을 무조건 원 안으로 밀거나 미래 안전구역에 투영하지 않는다. 곡선 시작점은 마지막 위치를 과거 속도 × 관측 경과 시간만큼 현재 시점으로 외삽한 값이다. 미래 라벨로 시작점을 보정하지 않는다. 20초 간격의 지점 보간은 이전의 10초마다 흔들리는 궤적을 줄이지만 급회전 표현력을 제한한다. 경로의 충돌·도로·교량 통행을 보장하는 물리 모델은 아니다. 회귀와 Diffusion의 입력·시간 표현은 같지만 Diffusion에는 추가 denoiser 파라미터가 있으므로 동일 파라미터 수 비교는 아니다.

학습은 시드 41·42·43, AdamW 5e-4→5e-5, 100회 warmup, EMA .99, 마지막 update의 EMA를 사용했다. 검증 점수로 checkpoint를 고르지 않았다. 학습 초기 차량 사전 감사에서 글라이더·박격포 분류 문제를 찾아 기존 시도를 중단하고 별도 폴더에 보존한 뒤, 데이터를 다시 만들고 모든 모델을 처음부터 학습했다. 초기 분류 오류 실행의 결과나 가중치는 사용하지 않았다. 이후 확산 잔차의 팀원 간 결합과, 공중 이탈 뒤 지상/수상/고정 탑승 증거가 나올 때 상태를 해제하는 규칙도 수정했다. 시간순 원자료 그림과 테스트로 확인한 뒤 모든 최종 비교 모델을 다시 처음부터 학습했다. 중간에 보존했던 회귀 가중치도 최종 비교에는 쓰지 않았다. 모든 수정은 검증 예측을 보기 전에 했다.

## 실제 내부 검증 결과

단위 m, 낮을수록 좋다. 경기 안에서 창 평균 → 시드 평균 → 경기 평균 순서이며 모든 경기에 같은 가중치를 준다. 현재 개인 평균경로 ADE가 가장 낮은 모델은 **{names[best]}**다. 이 수치는 새 데이터 정의와 학습 분할의 결과여서 V2 수치와 직접 빼서 개선량으로 주장하지 않는다.

| 모델 | 개인 평균경로 ADE | 팀 중심 평균경로 ADE | 공동 Energy score | 개인 기대 ADE | 팀원 간 거리 MAE |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

지표별 유효 경기 수는 개인 ADE {o['constant_velocity']['ego_meanpathADE_m']['matches']}, 팀 중심 ADE {o['constant_velocity']['team_meanpathADE_m']['matches']}, 공동 Energy score {o['constant_velocity']['joint_energy_score_m']['matches']}, 팀원 간 거리 MAE {o['constant_velocity']['pair_distance_MAE_m']['matches']}이다. 모든 미래 60초에 팀원 전체의 정답이 있는 창은 {verify['complete_joint_60s_windows']:,}개다. 그 외에는 각 시점의 유효 라벨 mask에 따라 평가한다.

평균 경로는 생성 좌표의 평균이며 실행 가능한 한 경로가 아닐 수 있다. 기대 ADE는 생성된 경로 하나하나의 오차를 평균한다. 팀 중심은 기준 시점의 팀원을 고정하고 이들 모두에게 유효한 미래 라벨이 있을 때만 계산한다. 최소 ADE는 oracle 선택 지표이므로 우위 판단의 주 지표로 쓰지 않았다. 공동 Energy score는 표본 간 거리에서 자기 자신과의 쌍을 제외한다.

![검증 성능](metrics_comparison.png)

| 대응 비교 | 지표 | 차이 m | 95% 경기 bootstrap CI | 12개 주 비교 보정 구간 |
|---|---|---:|---|---|
{chr(10).join(paired)}

이 구간은 학습한 세 시드를 고정한 경기 표본 불확실성이다. 새 시드의 전체 불확실성이나 서로 다른 경기의 동일 계정 의존성을 포함하지 않는다.

**이번 실험의 판정:** 관측자·서버 두 조건 모두 회귀가 개인·팀 중심 평균 경로 ADE에서 우세하고, Diffusion이 공동 Energy score에서 우세했다. 이 세 차이는 각각 12개 주 비교 보정 구간에서도 0을 포함하지 않았다. 반면 서버−관측자 차이는 세 주 지표 모두 보정 구간에 0을 포함하므로, 서버 정보의 추가 이점을 확인하지 못했다. 이는 이번 구조·학습 예산의 결과이며 서버 정보가 원래 쓸모없다는 결론이 아니다.

관측자 Diffusion의 개별 생성 시나리오 기대 ADE는 {o['observer_diffusion']['ego_expectedADE_m']['mean']:.2f}m, 팀원 간 거리 MAE는 {o['observer_diffusion']['pair_distance_MAE_m']['mean']:.2f}m다. 회귀의 {o['observer_regression']['ego_expectedADE_m']['mean']:.2f}m, {o['observer_regression']['pair_distance_MAE_m']['mean']:.2f}m보다 크다. Energy score 하나의 개선을 근거로 개인 경로·팀 대형·확률 보정이 모두 좋아졌다고 주장하면 안 된다. 현재 단일 경로 예측의 기준은 회귀로 두고, Diffusion은 공동 분포 점수의 이점과 시나리오 품질의 약점을 함께 연구하는 단계다.


## 안전구역을 실제로 사용하는가

첫째, 안전구역 수치와 현재 원 거리 보조 손실을 제거한 관측자 회귀·Diffusion을 같은 seed41, 표본 순서, 학습 예산으로 **별도 재학습**했다. 양의 차이는 안전구역을 뺀 모델의 오차가 더 크다는 뜻이다. 단일 시드 탐색 결과이며 확증적 기여 검증으로 확대하지 않는다.

| 모델 | 지표 | 전체 조건 | 안전구역 수치 제거 재학습 | 제거 − 전체 |
|---|---|---:|---:|---:|
{chr(10).join(abl)}

둘째, 각 검증 경기에서 오차와 무관하게 뽑은 1개 창에 동일 난수를 사용하고 안전구역 수치를 가렸다. 아래는 세 시드 평균이다. 가림은 학습 분포 밖 조작이므로 조건 반응의 진단이며, 위 재학습 실험과 구분해야 한다. 관계 구조·게임 시간 등 간접 단서는 두 검사에 남는다.

| 모델 | 가림으로 평균 경로가 변한 거리 m | 가림 − 전체 개인 ADE m |
|---|---:|---:|
{chr(10).join(sen)}

이번 seed41 제거 재학습에서는 두 모델의 세 주 지표가 모두 악화됐다. 개인 ADE는 회귀 **3.99m**, Diffusion **6.27m** 증가했고, 팀 중심 ADE는 각각 **3.69m**, **6.59m** 증가했다. 따라서 이번 단일 시드 실험에는 현재 안전구역 수치·거리 보조 손실의 예측 기여를 뒷받침하는 결과가 있다. 조건 가림으로 평균 경로도 약 70–83m 변했으므로 모델이 조건을 수치적으로 사용한다는 점과 연결된다.

다만 수치 반응 자체가 성능 기여의 증거는 아니다. 가림은 분포 밖 진단이고, 재학습 비교는 입력과 거리 보조 손실을 함께 제거했다. 여러 시드의 제거 재학습과 입력/손실을 분리한 실험 없이는 일반적인 효과 크기나 각 요소의 독립 기여를 확정할 수 없다. 인간 의사결정의 인과 효과도 아니다.

## 상황별 비교

아래는 개인 평균경로 ADE다. 재진입 여부와 현재 모드는 기준 시점까지의 정보로 분류했다. `aerial`은 직접 항공 수송·글라이더 증거와 항공기 이탈 이후 추정 상태를 함께 포함한다. 인간·봇은 계정 prefix의 분석 metadata이며 학습 입력이 아니다.

| 집단 | 창 수 | 등속도 | 위치 유지 | 관측자 회귀 | 관측자 Diffusion | 서버 회귀 | 서버 Diffusion |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(sub)}

집단 간 상황 분포가 다르므로 인간·봇의 본질적 이동 차이나 능력의 인과 비교로 해석하지 않는다.

{transport_section}## 검증 경기의 개인·팀 경로

[인터랙티브 뷰어](viewer.html)에서 전체 1,504개 검증 창을 확인할 수 있다. 회귀·Diffusion의 평균 경로와 실제 하나의 생성 시나리오를 구분한다. 다음 사례는 미리 정한 집단에서 중간 오차 표본을 고른 설명용 그림이다.

'''
 for c in cases:content+=f"**{c['label']}** — `{c['sample_id']}`\n\n![{c['label']}]({c['figure']})\n\n"
 content+=f'''## 누수와 복원 감사

- 학습·검증·잠금 후보 경기 ID 교집합 0. 테스트 로더는 명시적으로 실패하고 테스트 bank/index 파일이 없다.
- 학습 657경기만으로 어휘와 공간 연결을 다시 구성했다. 검증·테스트의 지형 관측 통계는 쓰지 않았다. 역사적 외부 높이·지표면 자료만 이전 정적 파일에서 재사용했다.
- 입력 유효 위치 {audit['source_asof_valid_positions']:,}개에서 기록 시각이 입력 시점 이후인 사례 0개. 현재 생애 시작보다 이전 위치를 현 위치로 사용하는 사례도 없다.
- 5경기 × 300·600·900초의 원자료 prefix 재구성에서 위치·상태·장비·이동 모드·생애·현재 원이 전체 자료의 해당 시점과 일치했다.
- 미래 target 변경이 조건 입력을 바꾸지 않는 검사, 관측자에게 가린 적·팀원 비공개 정보를 바꿔도 입력이 변하지 않는 검사, 모터글라이더/박격포 분리, 생애 경계 속도 차단 등을 포함한 13개 새 테스트와 기존 13개 테스트, 총 26개를 통과했다.
- 공간 연결 {quality['spatial_edges']:,}개의 count·support가 학습 경기별 파일 합산과 정확히 일치했다.
- 학습 설정과 평가 창을 사전에 저장했다. 모델·학습량·checkpoint를 검증 점수에 맞춰 바꾸지 않았다. 주 비교와 제거 재학습 모두 최종 update EMA다.
- 주 모델 12개의 체크포인트·표본 계획을 확인하고 첫 검증 표본 좌표를 정확히 재생성했다. 전체 학습 재실행 재현은 하지 않았다.

분할 단위는 경기 ID이며 날짜순 외삽 검증이 아니다. 학습 경기에서 얻은 공간 사전은 같은 패치 기간의 사후 학습 자료이므로, 각 검증 경기의 실제 개최일 당시 이미 확보된 지도였다는 뜻도 아니다.

검사한 경로에서 미래 입력·검증 경기 혼입은 발견하지 못했다. 하지만 이전 연구 노출, 실제 HUD 관측성, 계정 단위 중복, 추정 재진입의 오탐·미탐, 누락·사망 라벨에 따른 선택 편향은 남는다. 특히 사망한 사람의 재진입 경로를 예측하지 않으므로 장기 전체 경기 정책 생성으로 확대하면 안 된다.

## 연구 주장과 다음 검증

현재는 상태 복원 규칙과 모델 조건 사용을 개선한 고정 예산 예비 비교다. 재진입 증거의 정확도는 전체 원본 이벤트/리플레이 대조가 없으므로 알려져 있지 않다. 관측자와 서버는 정보 조건 비교이며 실제 사람의 시각·청각 인지 재현 모델이 아니다. 공간 연결의 불확실성 표지는 보정된 통행 확률이 아니다.

최종 논문 전에 필요한 것은 신규 미사용 테스트 확보, 재진입·이동 모드 수동/원본 이벤트 대조 표본, 세 시드 이상의 안전구역 제거 재학습, 같은 입력의 no-graph 비교, 20초 지점 보간 표현의 민감도, 지상·수상·공중별 실행 가능성 평가다. 추가 튜닝은 내부 검증에서만 하고, 테스트 공개는 설계 동결 뒤 한 번으로 제한해야 한다.

통계 점검: 상황별 집계로 Simpson 가능성을 드러내며 통제했다고 주장하지 않는다. 경기 평균을 개인 능력으로 확대하지 않는다(ecological fallacy). 미래 라벨 유무·생존 선택으로 Berkson/collider/survivorship 위험이 남는다. 집단별 창 수를 보고해 base-rate를 숨기지 않는다. 극단 집단의 전후 개선 처치가 아니므로 regression-to-mean 해석을 적용하지 않는다. 모든 사전 지정 주 비교와 12개 비교 보정 구간을 남겨 look-elsewhere/forking-paths 위험을 줄인다. 관찰 예측 연구이므로 상관을 인과 또는 역인과 결론으로 바꾸지 않는다. 외부 사전등록 연구는 아니다.

## 재실행

```bash
# 기존 잠금을 보존한 상태에서 데이터 재생성. 테스트 raw/bank는 읽거나 만들지 않음.
OPENBLAS_NUM_THREADS=1 /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/build_v3.py
OPENBLAS_NUM_THREADS=1 /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/audit_v3.py
/private/tmp/pubg-spatial-venv/bin/python -m pytest tests/movement_diffusion/test_v3.py -q
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v3.py prepare
# 각 seed41,42,43에 대해 실행. 완료 checkpoint는 동일 명세 확인 후 재사용.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v3.py train-seed --seed 41
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v3.py train-ablation
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v3.py evaluate
MPLCONFIGDIR=/private/tmp/pubg-mpl /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/analyze_v3.py
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/diagnose_v3.py
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/coverage_v3.py
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/transport_audit_v3.py evaluate
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/report_v3.py
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/finalize_v3.py
```

실험 경로는 고정돼 있다. 전체 학습 재실행은 별도 프로젝트 복제 또는 명시적인 새 출력 경로를 적용한 새 명세에서 해야 하며, 보존된 완료 checkpoint를 삭제해서 덮어쓰지 않는다. 프로세스 DataLoader에는 로컬 공유 메모리 권한이 필요하다.

- `experiment_spec.json`, `analysis_protocol.json`: 학습·평가·비교 규칙과 코드 해시
- `*_seed` 대신 `observer_regression_41.pt` 등 12개 주 checkpoint와 `*_nozone.pt` 2개
- `predictions.npz`, `evaluation_targets.npz`: 검증 생성 좌표와 라벨·mask
- `comparison_summary.json`, `zone_ablation.json`, `condition_sensitivity.json`: 실측 비교와 조건 사용 진단
- `leakage_audit.json`, `final_verification.json`, `delivery_verification.json`: 상태·시간·분할·추론 재현·최종 산출물 검증
- `runtime_environment.json`, `artifact_manifest.json`: 실행 환경과 최종 산출물 SHA-256
- `aborted_initial_mobility_classification/`: 초기 분류 오류로 중단한 실행 기록, 최종 결과에서 제외
- `pre_joint_residual_fix/`, `pre_ground_transport_reset/`: 검증 전 구현 감사에서 중단·교체한 소스와 실행 기록. 최종 비교의 가중치로 사용하지 않음
'''
 (OUT/'REPORT_KO.md').write_text(content)
 doc=ROOT/'docs/experiment_design/HETEROGRAPH_DIFFUSION_V3_REENTRY_ZONE_COMPARISON_2026-10-15.md';doc.write_text(re.sub(r'\]\((?!https?://)([^)]+)\)',lambda m:']('+str((OUT/m.group(1)).resolve())+')',content))
 print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
