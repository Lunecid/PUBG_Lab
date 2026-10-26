"""Render the Korean experiment record from measured artifacts, never placeholders."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import json
import sys
import hashlib
import platform
import re
from pathlib import Path
import numpy as np

ROOT = REPO_ROOT
OUT = ROOT / "reports/heterograph_diffusion_experiment_v2"


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    s = read("comparison_summary.json")
    o = s["overall"]
    spec = read("experiment_spec.json")
    audit = read("leakage_audit.json")
    runtime = read("runtime_model_verification.json")
    diag = read("evaluation_data_diagnostics.json")["summary"]
    cases = read("visualization_cases.json")
    rows = []
    names = {
        "constant_velocity": "등속도",
        "stationary": "현재 위치 유지",
        "observer": "관측자 Diffusion",
        "server": "서버 Diffusion",
    }
    keys = [
        "ego_meanpathADE_m",
        "team_meanpathADE_m",
        "joint_energy_score_m",
        "ego_expectedADE_m",
        "pair_distance_MAE_m",
        "ego_minADE_m",
    ]
    for model in names:
        rows.append(
            "| "
            + names[model]
            + " | "
            + " | ".join(f"{o['models'][model][k]['mean']:.2f}" for k in keys)
            + " |"
        )
    differences = []
    for pair, label in [
        ("observer_minus_constant_velocity", "관측자 − 등속도"),
        ("server_minus_constant_velocity", "서버 − 등속도"),
        ("server_minus_observer", "서버 − 관측자"),
    ]:
        for k, metric in zip(
            keys[:3], ["개인 평균경로 ADE", "팀 중심 평균경로 ADE", "공동 Energy score"]
        ):
            r = o["paired_differences"][pair][k]
            ci = r["ci95_paired_match_bootstrap"]
            family = r["ci_familywise95_bonferroni_9"]
            differences.append(
                f"| {label} | {metric} | {r['difference_m']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | [{family[0]:.2f}, {family[1]:.2f}] |"
            )
    groups = []
    for group, label in [
        ("human", "인간"),
        ("bot", "봇"),
        ("vehicle", "차량 기록 있음"),
        ("on_foot", "차량 기록 없음"),
        ("multi_member", "현재 팀원 2명 이상"),
        ("complete60s", "현재 팀원 전원 60초 라벨 완전"),
        ("before_post_death_positive_evidence", "경기 내 사망 후 양의 체력 재기록 전"),
        ("after_post_death_positive_evidence", "경기 내 사망 후 양의 체력 재기록 후"),
    ]:
        g = s["subgroups"][group]
        groups.append(
            "| "
            + label
            + f" | {g['windows']} | "
            + " | ".join(
                f"{g['models'][m]['ego_meanpathADE_m']['mean']:.2f}"
                for m in ["constant_velocity", "stationary", "observer", "server"]
            )
            + " |"
        )
    seedlines = []
    for m in ["observer", "server"]:
        seedlines.append(
            f"- {names[m]} 개인 평균경로 ADE의 시드별 경기 평균: "
            + ", ".join(f"{v:.2f} m" for v in o["models"][m]["ego_meanpathADE_m"]["seed_means"])
            + "."
        )
    lower = lambda m: o["models"][m]["ego_meanpathADE_m"]["mean"]
    result = f"개인 평균경로 ADE는 등속도 {lower('constant_velocity'):.2f} m, 현재 위치 유지 {lower('stationary'):.2f} m, 관측자 {lower('observer'):.2f} m, 서버 {lower('server'):.2f} m였다."
    generation = read("generation_diagnostics.json")["models"]
    genrows = []
    for model in names:
        variants = (
            [model]
            if model in ("constant_velocity", "stationary")
            else [f"{model}_{seed}" for seed in [41, 42, 43]]
        )

        def avg(key):
            return float(np.mean([generation[v][key]["match_macro"] for v in variants]))

        coverage = (
            f"{avg('ego_endpoint_radial_80pct_coverage') * 100:.1f}%"
            if len(variants) > 1
            else "해당 없음"
        )
        radius = (
            f"{avg('ego_endpoint_radial_80pct_radius_m'):.1f}" if len(variants) > 1 else "해당 없음"
        )
        genrows.append(
            f"| {names[model]} | {coverage} | {radius} | {avg('mean_path_length_error_complete_member_m'):.1f} | {avg('outside_map_coordinate_fraction') * 100:.3f}% |"
        )
    condition = read("condition_diagnostics.json")["results"]
    condrows = []
    for model in ["observer", "server"]:
        cs = condition[model]["summary"]
        condrows.append(
            "| "
            + names[model]
            + " | "
            + " | ".join(
                f"{cs[k]['mean']:.3f}"
                for k in [
                    "zero_learned_condition_meanpath_shift_m",
                    "masked_zone_features_meanpath_shift_m",
                    "generated_spread_mean_m",
                ]
            )
            + " |"
        )
    mobility = read("mobility_diagnostics.json")["summary"]
    content = f"""# PUBG 관측자와 서버 이종 그래프 Diffusion 학습 비교

**이번 설정에서는 두 Diffusion 모두 개인 경로와 팀원 간 거리 오차에서 기준 모델을 넘지 못했다. 서버 추가 정보의 실질적인 이점도 확인되지 않았다.** 학습과 평가가 실행되는 것은 확인했지만, 안전구역과 팀 이동을 제대로 학습한 설계로 인정할 근거는 부족하다.

{result} 이 결과는 검증 188경기의 1,504개 창과 3개 학습 시드에 대한 **초기 고정 예산 실험**이다. 모델별 2,500회 업데이트를 실제로 완료했으며 체크포인트와 생성 좌표를 저장했다. 수렴이나 논문 수준의 최종 우위를 입증한 결과로 해석하지 않는다.

## Material Passport

- 작업 유형: 실제 모델 학습, 경기 단위 검증, 데이터 누수 및 설계 감사
- 상태: 학습 및 평가 완료, 구현 검증 통과, 관측 정보 의미와 공간 정밀도에는 미해결 제약 존재
- 원자료: 로컬 40.2 추정 Erangel 공식 TPP squad 텔레메트리 939경기
- 자료 범위: 기존 match-disjoint 분할 유지. 독립적인 최종 test split은 없음
- 검증 수준: 원자료 무결성 및 인과적 상태 재구성 검사, 모델 단위 테스트, 고정 시드 추론 재현 검사. 전체 학습 재실행 재현성은 미검증
- 생성일: 2026-10-15

## 실제 학습 조건

| 항목 | 설정 |
|---|---|
| 정보 조건 | 자신의 비공개 상태 + 팀 HUD 근사 / 추출된 전체 서버 텔레메트리 |
| 학습·검증 경기 | 751 / 188, 교집합 0 |
| 시드 | 41, 42, 43 |
| 각 모델 학습 | 2,500 updates × batch 8 = 20,000 windows drawn per seed |
| 전체 학습 규모 | 2 views × 3 seeds = 6 models, 총 120,000 표본 제시. 세 시드를 합쳐 학습 751경기 모두 방문 |
| 표본 추출 | 경기 균등 추출 후 경기 내 창 추출. 두 정보 조건에 같은 표본 순서 적용 |
| 한 시드의 고유 창 | {", ".join(str(spec["plans"][str(v)]["unique_windows"]) for v in [41, 42, 43])}; 데이터셋 전체 1회 순회가 아님 |
| 입력·출력 시간 | 과거 -30, -20, -10, 0초 / 미래 10, 20, 30, 40, 50, 60초 |
| 파라미터 수 | {runtime["model_parameters"]:,}, 두 정보 조건 동일 |
| 최적화 | AdamW, weight decay 1e-4, 100회 warmup, 3e-4 → 3e-5 cosine LR, gradient norm 1 |
| 추론 모델 | 마지막 update의 EMA(decay .995), 검증 점수로 checkpoint 선택 안 함 |
| 확산 | cosine noise schedule, v prediction, 32 DDIM steps, K=20 공동 시나리오 |
| 실행 환경 | CPU float32, deterministic algorithms, PyTorch 2.14.1 |

시드별 표본 계획과 소스 SHA-256은 [experiment_spec.json](experiment_spec.json)에 있다. 최초 시도는 샌드박스의 PyTorch 공유 메모리 차단으로 업데이트 전에 중단됐으며, 권한을 허용한 동일 설정 실행으로 완료했다. 오류 기록도 남겼다.

## 설계 검토 후 수정한 내용

1. **팀원별 입력과 출력의 대응**: 기존 smoke 모델은 ego·팀·전역 요약만 압축해 팀원별 경로에 전달했다. 현재는 각 팀원의 노드 이력을 별도 GRU로 처리하고 현재 위치·속도·관측 경과 시간을 직접 제공한다. 공유 가중치와 팀원 attention을 사용하며, 팀원 슬롯을 바꿔도 결과가 같은 순서로 바뀌는 permutation equivariance를 검사했다.
2. **확산 종단 분포**: 기존 32단계 선형 스케줄은 끝에 원자료 신호가 상당히 남았다. cosine 스케줄의 마지막 alpha-bar는 {runtime["terminal_alpha_bar"]:.8f}이며, v prediction으로 작은 alpha를 나누는 불안정한 복원을 피한다. [Improved DDPM 원문](https://arxiv.org/abs/2102.09672)과 [Progressive Distillation 원문](https://arxiv.org/abs/2202.00512)의 구성 요소를 참고했다. 해당 논문의 성능이 여기서 재현됐다는 의미는 아니다.
3. **좌표 표현**: 등속도 경로 주위의 잔차를 200 m 단위로 확산한다. 등속도는 마지막 기록 위치에서 미래 horizon과 이미 경과한 관측 시간을 합쳐 이동시킨다. 예를 들어 5초 전 위치와 10 m/s 속도가 있으면 10초 뒤 예측은 그 위치에서 150 m 이동한 지점이다.
4. **노드 유형별 집계**: 지도의 441개 region 노드가 플레이어나 안전구역 정보를 희석하지 않도록 player, zone, game, cell, region별로 평균 집계한 뒤 결합한다.
5. **공동 이동**: 네 팀원의 잔차 궤적을 함께 denoise하고 팀 중심 복원 오차를 보조 loss로 쓴다. 팀원을 억지로 모으는 거리에 대한 정규화는 넣지 않았다. 팀원 간 거리 MAE를 별도 평가한다.
6. **공정한 평가**: best-of-20만 보지 않고 평균 경로 ADE, 생성 표본 전체의 기대 ADE, 공동 Energy score를 함께 쓴다. Energy score의 표본 간 거리는 대각선을 제외한 U-statistic으로 계산한다. 간단하지만 중요한 현재 위치 유지 기준도 추가했다.

실제 확인한 텐서 크기는 graph node features `[N, 32]`와 같은 크기의 관측 mask, 팀원별 조건 `[B, 4, 64]`, 라벨 `[B, 4, 6, 2]`, 생성 결과 `[B, 20, 4, 6, 2]`다. `B=2` 런타임 검증은 [runtime_model_verification.json](runtime_model_verification.json)에 기록했다. 존재하는 노드·관계 유형으로 gradient가 흐르는지도 검사했지만, gradient가 있다는 사실만으로 그 feature가 성능에 기여한다고 주장하지 않는다.

## 검증 경기에서의 성능

모든 거리는 m이며 작을수록 좋다. 각 경기 안에서 창별 점수를 평균하고, 학습 시드별 점수를 평균한 다음 경기에 같은 가중치를 준다. 평균 경로는 20개 생성 경로의 좌표 평균이므로 실행 가능한 하나의 경로라는 보장은 없다.

| 모델 | 개인 평균경로 ADE | 팀 중심 평균경로 ADE | 공동 Energy score | 개인 기대 ADE | 팀원 간 거리 MAE | 개인 최소 ADE |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

개인 지표는 관측자 P1을 뜻한다. 나머지 팀원을 포함하는 joint ADE는 전체 수치 파일에 별도로 있다. 팀 중심은 예측 기준 시점에 선택된 팀원을 고정한 평균이다. 미래에 살아 있는 사람만 다시 모아 중심을 만드는 방식은 쓰지 않았다. 특정 미래 시점에 고정된 팀원의 위치가 모두 유효할 때만 팀 지표를 계산한다.

기대 ADE는 각 생성 경로의 오차를 평균한다. 최소 ADE는 정답을 본 뒤 가장 가까운 생성 경로를 고른 값이므로 K=1 등속도와 이것만 비교해 생성기의 우위를 말할 수 없다. Energy score는 공동 궤적 벡터에 대한 예측–정답 거리에서 표본 간 거리의 절반을 뺀 값이며, 유효한 actor-time 수의 제곱근으로 정규화했다.

![검증 성능 비교](metrics_comparison.png)

### 결과가 의미하는 것

관측자 모델의 개인 평균경로 ADE는 등속도보다 6.36 m 높았다(경기 대응 95% CI 4.74–7.96 m). 서버와 관측자의 차이는 약 0.02 m였고 구간에 0이 포함됐다. 이는 현재 표현·모델·학습 예산에서 추가 정보를 활용한 이점이 확인되지 않았다는 뜻이며, 서버 정보가 원래 무용하다는 결론은 아니다.

생성한 경로 하나하나의 기대 ADE는 약 256 m이고 팀원 간 거리 MAE는 약 228 m다. 경로 길이 MAE도 약 698 m로, 평균 경로의 오차보다 생성 궤적의 과도한 흔들림과 팀 배치 실패가 더 뚜렷하다. 팀 중심 오차가 작다는 이유만으로 공동 이동을 잘 생성했다고 판단하면 안 된다. Energy score는 등속도보다 낮지만 현재 위치 유지 기준보다 높다. 최소 ADE만 골라 보여주면 이 실패를 가릴 수 있다.

고정 예산에서 얻은 부정적 예비 결과로 남긴다. 학습 부족만이 원인이라고 단정하지 않으며, 데이터 상태 복원·이동 모드·조건 활용·시간적 연속성을 함께 개선해야 한다.

## 같은 경기에서의 대응 차이

음수는 표의 앞 모델이 더 낮은 오차를 보였다는 뜻이다. 95% CI는 5,000회 경기 단위 paired bootstrap이다. 마지막 열은 주 지표 3개 × 모델 비교 3개의 총 9개 비교에 대한 Bonferroni 보정 구간이다. 이는 학습 시드 3개를 고정한 경기 표본 불확실성이며, 모든 미래 학습 시드에 대한 불확실성을 포함하지 않는다. p-value나 행동의 인과 효과는 주장하지 않는다.

| 비교 | 지표 | 차이 m | 대응 95% CI | 9개 비교 보정 95% 구간 |
|---|---|---:|---|---|
{chr(10).join(differences)}

{chr(10).join(seedlines)}

## 인간과 봇 및 상황별 진단

아래는 개인 평균경로 ADE다. 인간·봇은 계정 prefix 기반 분석용 metadata이며 모델 입력이나 학습 조건으로 쓰지 않았다. 이 표는 서로 다른 상황의 오차를 기술하며, 인간과 봇의 이동 능력이나 본질적인 행동 차이를 입증하지 않는다. 차량·게임 단계·교전·팀원 수의 분포가 다르다.

| 집단 | 창 수 | 등속도 | 현재 위치 유지 | 관측자 | 서버 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(groups)}

검증 창 중 {diag["post_death_positive_evidence_seen_windows"]}개는 경기 안에서 사망 후 양의 체력이 다시 기록된 이후다. 이는 재진입 또는 기록 불일치의 징후이며 재진입의 확정 라벨은 아니다. 현재 상태 bank는 최초 사망 뒤 위치를 검열하므로, 이 구간의 ‘서버’ 입력은 실제 전체 세계 상태와 특히 거리가 있다. 앞뒤 집단 비교는 이 제약을 드러내기 위한 진단이다.

## 생성 분포와 기본 좌표 진단

아래 80% 반경은 20개 생성 도착점의 평균을 중심으로 각 도착점까지의 거리에서 80% 분위수를 취한 값이다. 실제 도착점이 이 원 안에 든 비율을 함께 계산했다. 이는 출력 분포의 기술적 보정 진단이며, 지도 자체의 신뢰도나 통계적 신뢰구간과 다르다.

| 모델 | 60초 실제 도착점의 80% 반경 포함률 | 평균 80% 반경 m | 경로 길이 MAE m | 지도 좌표 범위 밖 비율 |
|---|---:|---:|---:|---:|
{chr(10).join(genrows)}

경로 길이는 여섯 미래 좌표가 모두 있는 개인만 평가한다. 좌표 범위 검사는 [0, 8160) m 사각형을 벗어났는지만 검사하며 물·벽·건물 충돌 여부를 뜻하지 않는다. 세부 수치는 [generation_diagnostics.json](generation_diagnostics.json)에 있다.

## 사후 조건 민감도와 실패 사례 확인

주 평가를 끝낸 뒤 조건을 실제로 쓰는지 별도 진단했다. 각 검증 경기에서 오차와 무관한 난수로 1개 창씩 뽑아 188개 창 × 3개 시드를 검사했다. 등속도 참조 경로와 생성 난수는 고정하고, 학습된 조건 벡터 전체를 0으로 만들거나 안전구역 노드의 수치·관측 mask만 가렸다. 아래는 원래 출력과 달라진 평균 경로의 거리다.

| 모델 | 전체 학습 조건 제거 시 변화 m | 안전구역 수치 가림 시 변화 m | 원래 생성 표본의 평균 경로 주위 분산 거리 m |
|---|---:|---:|---:|
{chr(10).join(condrows)}

전체 조건에는 반응하지만 안전구역 수치 변화에 대한 반응은 약 4 cm에 그쳤다. 현재 모델이 안전구역 조건을 충분히 활용한다고 주장할 근거가 없다. 다만 이 조작은 학습 분포 밖 입력이며, 안전구역 노드·관계의 존재와 시간 등의 간접 정보는 남는다. **재학습 ablation이나 인과적 feature importance가 아니다.** 처음의 94개 창 탐색은 정렬된 표본에서 초반 창을 과대표집해 별도 보존했고, 보고한 수치는 경기별 무작위 188개 창으로 다시 계산했다. 진단 뒤 모델 선택·재학습은 하지 않았다. [조건 진단 원본](condition_diagnostics.json)

가장 큰 오차 사례를 원자료와 대조한 결과, 450초 기준 시점 직후 452.121초부터 `EmergencyPickup`이 기록됐다. 팀 전체가 공중 수송으로 약 3 km 이동한 사례이며 단순한 잘못된 좌표로 삭제하면 안 된다. 검증 창에서 현재 팀의 최근 긴급 호출 수송 기록이 있는 창은 {mobility["current_evidence_windows"]}개, 미래 65초에 기록되는 창은 {mobility["future_evidence_windows"]}개, 그중 현재 기록이 없는 창은 {mobility["future_without_current_evidence_windows"]}개였다. 이는 사후 설명용 라벨이고 입력이나 평가 제외 조건으로 쓰지 않았다. [이동 모드 원자료 진단](mobility_diagnostics.json)

현재 `landed`는 최초 착지 이후라는 뜻이며 계속 지상에 있다는 뜻이 아니다. 차량 기록 없음도 보행 확정 라벨이 아니다. 지상 격자 제약을 모든 경로에 강제하면 이와 같은 합법적인 공중 이동을 막게 된다. 이동 모드 전환을 별도로 표현하고, 공중 수송·낙하·수상·지상 이동의 제약을 나누는 것이 다음 설계의 필수 과제다.

## 실제 경로와 생성 경로

[인터랙티브 뷰어](viewer.html)에서 평가한 1,504개 창을 모두 선택할 수 있다. 시드를 바꾸고, 20개 평균 경로와 정답을 사용하지 않고 선택한 하나의 공동 시나리오를 전환할 수 있다. 그림은 비교 가능하도록 같은 사례의 세 모델에 동일한 지도 범위를 사용한다.

"""
    for c in cases:
        content += f"**{c['label']}** — `{c['sample_id']}`. 선택 규칙: {c['selection']}.\n\n![{c['label']}]({c['figure']})\n\n"
    content += f"""## 누수 감사 결과

- **경기 분리**: train 751 / validation 188, 교집합 0. 세 시드의 실제 학습 표본과 평가 표본도 분리됐다.
- **원자료 유지**: 939개 bank의 SHA-256이 기존 추출 결과와 같았다. 위치 입력 {audit["checks"]["bank_integrity"]["position_asof_checks"]:,}개에서 기록 시각이 입력 시점 이후인 경우가 없었다.
- **어휘 사전과 노드 존재 시점**: 학습 원본 751경기에서 아이템 85종·차량 52종의 사전을 다시 만들었고 기존 사전과 ID까지 일치했다. 939개 bank의 해시 중복은 0개였다. 91,254개 경기-참가자 기록 전부에서 최초 이벤트가 가장 이른 입력 frame보다 앞섰다. 미래에만 처음 등장하는 참가자 노드를 미리 넣은 사례는 이 코호트에서 없었다. UI 로스터 표시와 팀 배정 시각을 별도로 검증했다는 뜻은 아니다.
- **미래 이벤트 차단**: 서로 다른 학습 5경기의 90·300·600초에서 원시 이벤트를 잘라 만든 bank와 전체 원자료의 해당 시점 입력이 같았다. 미래 target 값을 바꿔도 조건 그래프가 바뀌지 않는 검사를 통과했다.
- **관측자·서버 분리**: 적 위치, 타인의 무장·차량·피해, outgoing damage 및 생존 팀 수를 변경해도 observer 조건은 변하지 않고 server 조건은 달라지는 검사를 통과했다. 이는 지정한 정보 정책에 대한 검사이며 실제 화면 관측 가능성을 보증하지 않는다.
- **공간 사전**: 116,607개 방향·이동유형별 연결의 count와 match support를 751개 학습 경기 파일에서 다시 합산해 정확히 일치함을 확인했다. 검증 경기 파일은 없다. 학습 표본에는 자기 경기의 연결 집계도 빼서 적용한다.
- **미래 라벨 분리**: future-valid mask는 loss와 평가 및 누락 잔차의 0 보충에만 사용한다. graph encoder, attention mask, sampling API에는 들어가지 않는다. 누락 궤적 보충은 학습상의 가정으로 남는다.
- **모델 선택**: 검증 결과를 보고 학습량·시드·checkpoint를 바꾸지 않았다. 최종 update EMA만 평가했다. 표본 선택과 실험 설정을 평가 전에 저장했다.
- **재현 및 수치 검사**: 13개 계약·모델 테스트, 유한 loss/gradient 검사, 고정 시드 추론의 bitwise 재현 검사. 전체 학습을 다시 돌린 재현 검증은 하지 않았다.

따라서 **검사한 코드 경로에서는 직접적인 미래 입력·검증 경기 학습 혼입을 찾지 못했다**. ‘누수가 절대 없다’고 확대할 수는 없다. 실제 관측 화면과 UI 발표 시각, 계정 단위 중복, 재진입 후 세계 상태는 아직 검증되지 않았다. 세부 증거는 [leakage_audit.json](leakage_audit.json)에 있다.

## 논문화할 때 남는 설계 문제

**현재 모델은 이종 그래프를 조건으로 연속 좌표를 생성하는 실험 모델이다.** 생성 좌표를 그래프의 합법적인 간선 위로 제한하거나 건물·물·교량 충돌을 검사하는 planner는 아니다. 관측 간선, 추정 간선, support, 불확실성 표지를 다른 입력으로 전달했지만, 경로의 공간적 실행 가능성을 검증한 결과는 아직 없다.

1. **관측 정책 검증**: observer는 telemetry/HUD proxy다. 안전구역 원이 실제 언제 표시됐는지, 무피해 시각·청각으로 적을 인지했는지 알 수 없다. 피해 없음은 적 인지 없음이라는 라벨이 아니다. UI 동기화 자료가 없으면 관측자 모델의 명칭과 주장 범위를 이 수준으로 제한해야 한다.
2. **재진입 상태 재구성**: 최초 사망 전 목표 경로와 세계 상태의 재진입 처리를 분리해야 한다. 현재처럼 모든 플레이어를 영구 검열하면 다시 살아난 팀원·적의 상태가 잘못 표현될 수 있다. 생애 episode 단위 상태를 복원한 뒤 같은 실험을 재실행해야 한다.
3. **목표 시각 정합과 검열**: 검증 목표 위치의 평균 절대 시각 차이는 {diag["mean_abs_target_clock_error_s"]:.3f}초다. 고속 차량에서 이 오차가 커진다. label-only 보간·허용 간격 민감도 평가가 필요하다. 위치 누락과 사망 후 경로가 빠지므로 결과는 관측 가능한 미래 이동에 대한 조건부 평가다.
4. **공간 정확도**: 높이는 2019년 자료이고 배경은 과거 공식 지도다. 40.2 지형의 정답이나 collision mesh가 아니다. 100 m 격자는 작은 건물·출입구·교량 층위를 표현하지 못한다. 미관측 간선은 불가능한 간선이 아니다. graph adherence를 넣더라도 불확실성을 반영한 soft constraint부터 검증해야 한다. 현재 불확실성은 연결 유형·지지 수·미관측 여부·변경 주의 영역의 표지이며 보정된 통행 가능 확률이 아니다. 역방향 message-passing 간선도 역방향 통행 가능성의 증거가 아니다.
5. **모델 기여 분리**: 현재 비교는 정보량 비교다. 그래프가 유효한지, 안전구역 조건이 유효한지, Diffusion이 유효한지는 각각 재학습 ablation이 필요하다. 같은 특징을 쓰는 회귀/GRU 또는 GMM 기준, no-graph, no-zone, inferred-edge 제거, uncertainty 제거를 비교해야 한다.
6. **공동 이동 검증**: 중심 오차가 작더라도 팀원이 비현실적으로 벌어지거나 모일 수 있다. 현재 pair-distance MAE를 함께 보고했으며, 다음에는 분리·재집결 이벤트, 방향 정렬, 공동 coverage를 검증해야 한다. 현재 표본의 팀원 간 최대 거리는 최고 {diag["current_member_span_m_quantiles"][-1]:.0f} m여서 팀 중심이 통행 가능한 실제 위치라고 볼 수 없다.
7. **학습·일반화 범위**: 수렴은 미확인이고 데이터셋 전체 epoch를 돌린 실험이 아니다. match-disjoint일 뿐 account-disjoint·시간 순 분할은 아니다. 학습 곡선은 아래와 같다. 하이퍼파라미터를 선택할 내부 dev와 최종 untouched test를 확보해야 한다. 그때 공간 사전과 vocab도 fit split에 맞춰 다시 분리해야 한다.

수정 우선순위는 (1) 재진입 episode와 목표 시각 정합 복원, (2) 이동 모드 전환 및 현재 수송 증거 표현, (3) 동일 조건의 결정론적 공동 경로 회귀 기준 수립, (4) 시간적 연속성을 갖는 denoiser와 안전구역 조건 전달 검증, (5) 내부 dev에서 no-graph·no-zone 등 재학습 비교, (6) 잠근 test에서 최종 평가다. 관측자 정보 정책의 실제 HUD 검증은 이 과정과 병행해야 한다. 지금의 모델에 단순히 학습량만 늘리면 해결된다고 보지 않는다.

![학습 곡선](training_curves.png)

수업 논문에서는 ‘시간 누수를 통제한 이종 그래프 조건부 이동 생성의 예비 실험’으로 범위를 정할 수 있다. FDG 확장에서는 실제 인지 정보의 범위, 재진입, 공간 실행 가능성, 그래프·안전구역·생성모형의 기여를 각각 검증해야 한다. 현재 수치만으로 안전구역을 이해한다거나 인간다운 경로를 생성한다고 주장할 수 없다.

## 통계적 해석 점검

11개 항목을 점검했다. Simpson 역전은 인간·봇·차량 등의 상황별 기술표를 함께 제시하되 통제 분석을 했다고 주장하지 않는다. Ecological fallacy는 경기 평균을 개인의 능력으로 확대하지 않아 피한다. Berkson·collider·survivorship 문제는 생존/좌표 기록이 있는 미래만 평가하는 선택으로 남는다. Base-rate neglect는 평가 집단의 인간·봇 및 상황별 수를 함께 보고한다. Regression to the mean은 처치 전후 극단 집단 개선 실험이 아니므로 해당하지 않는다. Look-elsewhere와 forking paths는 모든 사전 지정 주 지표, 시드, 비교를 남기고 9개 주 비교의 보정 구간을 추가해 완화한다. Correlation/causation과 reverse causality는 행동의 원인에 대한 주장을 하지 않는다. 외부 사전등록 또는 독립 test를 갖춘 확증 연구는 아니다.

## 재실행과 산출물

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v2.py --output reports/heterograph_diffusion_experiment_v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/audit_experiment_v2.py
MPLCONFIGDIR=/private/tmp/pubg-mpl XDG_CACHE_HOME=/private/tmp/pubg-cache /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/analyze_experiment_v2.py
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/generation_diagnostics_v2.py
OPENBLAS_NUM_THREADS=1 /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/condition_diagnostics_v2.py
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/mobility_diagnostics_v2.py
/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/report_experiment_v2.py
```

첫 명령은 같은 설정의 완료 checkpoint가 있으면 해당 학습을 재사용하고 평가를 다시 수행한다. 완전한 재학습은 새 output 경로를 지정해야 한다. 분석·보고서 스크립트는 현재 실험 경로를 사용한다. 다중 프로세스 DataLoader에는 macOS 공유 메모리 실행 권한이 필요하다.

- `observer_seed41.pt`부터 `server_seed43.pt`까지 6개 checkpoint: EMA·마지막 raw weights·optimizer·설정·학습 이력
- `training_indices_seed*.npy`, `evaluation_indices.npy`: 실제 학습·평가 표본
- `predictions.npz`: 모든 평가 표본의 각 모델·시드별 공동 생성 좌표
- `evaluation_targets.npz`: mask와 실제 경로, 시작 위치
- `comparison_summary.json`, `metrics_table.csv`: 전체·상황별 경기 평균과 대응 구간
- `validation_case_*.png/pdf`, `metrics_comparison.png/pdf`, `training_curves.png/pdf`: 재사용 가능한 정적 그림
- `viewer.html`, `viewer_samples/`: 전 표본의 로컬 탐색 시각화
- `leakage_audit.json`, `runtime_model_verification.json`, `experiment_spec.json`: 검증·설정·출처 증거
"""
    (OUT / "REPORT_KO.md").write_text(content)
    doc = ROOT / "docs/experiment_design/HETEROGRAPH_DIFFUSION_TRAINING_AUDIT_2026-10-15.md"
    doc.write_text(
        re.sub(
            r"\]\((?!https?://)([^)]+)\)",
            r"](../../reports/heterograph_diffusion_experiment_v2/\1)",
            content,
        )
    )
    readme = ROOT / "movement_diffusion/README.md"
    old = readme.read_text()
    heading = (
        "## Diffusion 학습과 비교 실행"
        if "## Diffusion 학습과 비교 실행" in old
        else "## Diffusion 연결과 비교 실행"
    )
    start = old.index(heading)
    end = old.index("예제 배치는", start)
    new = """## Diffusion 학습과 비교 실행

현재 `model.py`는 초기 연결 시험을 교체한 v2 모델이다. 타입별 graph message passing 2층 → 각 팀원별 4시점 GRU → 팀원 간 attention → 공동 잔차 궤적 v predictor로 구성한다. 과거 안전구역 노드가 조건에 포함된다. 모든 예측은 과거 정보로 만든 등속도 경로 주위의 잔차이며, 좌표 단위는 km, 확산 잔차 scale은 200 m다. 현재 위치 표본의 `position_age_s`도 조건과 등속도 기준에 반영한다.

두 정보 조건 각각 3개 시드, 각 2,500회 update를 실제 실행했다. 파라미터는 각각 204,044개다. 같은 표본·초기 난수·학습량을 사용했고 검증 188경기에서 총 1,504개 창을 평가했다. 원래 3-step smoke 결과는 과거 연결 시험으로 보존했다. 최종 수치와 제약은 [실제 학습 및 감사 보고서](../reports/heterograph_diffusion_experiment_v2/REPORT_KO.md)에 있다.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v2.py --output reports/heterograph_diffusion_experiment_v2
```

완료 checkpoint가 같은 출력 경로에 있으면 재사용하므로, 완전한 재학습에는 새 경로를 지정한다. 모델 선택은 고정 예산의 마지막 EMA다. `train.py`는 한 경기당 한 창의 빠른 실행용으로 남겨 두었으며, 실제 비교에는 `experiment_v2.py`를 쓴다.

미래 정답 mask는 loss·평가와 누락 잔차의 0 보충에만 쓰며 encoder나 attention mask에는 제공하지 않는다. 미래 차량 mode 라벨은 준비했지만 예측 head는 아직 없다. 생성 경로를 지도 간선에 강제로 투영하거나 collision을 막는 sampler도 없다.

평가는 개인·팀 중심의 평균경로/기대/최소 ADE, 고정 60초 FDE, joint ADE, 공동 Energy score, 팀원 간 거리 MAE를 포함한다. 등속도와 현재 위치 유지 기준을 함께 평가하고 경기 단위 bootstrap을 제공한다. 최소 ADE는 정답을 본 뒤 선택하는 oracle 지표이므로 이것만으로 우위를 주장하지 않는다.

"""
    readme.write_text(old[:start] + new + old[end:])
    print("Report written", OUT / "REPORT_KO.md")


if __name__ == "__main__":
    main()
