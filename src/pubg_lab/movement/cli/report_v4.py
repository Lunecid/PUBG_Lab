"""Evidence-linked Korean experiment record, including unsuccessful outcomes."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import json
from pathlib import Path

ROOT = REPO_ROOT
OUT = ROOT / "reports/heterograph_diffusion_experiment_v4"


def read(n):
    return json.loads((OUT / n).read_text())


def main():
    d = read("comparison_summary.json")
    diag = read("v3_error_diagnosis.json")
    coverage = read("sampling_coverage.json")
    audit = read("leakage_audit.json")
    gen = read("generation_diagnostics.json")
    overall = d["overall"]
    keys = [
        "ego_meanpathADE_m",
        "team_meanpathADE_m",
        "pair_distance_MAE_m",
        "joint_energy_score_m",
    ]
    labels = {
        "v3_observer_regression": "V3 관측 회귀",
        "v3_observer_diffusion": "V3 관측 Diffusion",
        "v4_observer_regression": "V4 관측 회귀",
        "v4_observer_diffusion": "V4 관측 팀 대형 Diffusion",
        "v4_observer_independent": "V4 관측 독립 잔차 대조군",
        "v3_server_regression": "V3 서버 회귀",
        "v3_server_diffusion": "V3 서버 Diffusion",
        "v4_server_regression": "V4 서버 회귀",
        "v4_server_diffusion": "V4 서버 팀 대형 Diffusion",
        "constant_velocity": "등속도",
        "stationary": "현재 위치 유지",
    }

    def table(models, names=None):
        names = names or list(models)
        return "\n".join(
            [
                "| 모델 | 개인 평균 경로 ADE | 팀 중심 ADE | 생성 경로의 팀원 거리 MAE | Energy score |",
                "|---|---:|---:|---:|---:|",
            ]
            + [
                "| "
                + labels.get(m, m)
                + " | "
                + " | ".join(f"{models[m][k]['mean']:.2f}" for k in keys)
                + " |"
                for m in names
            ]
        )

    def change(a, b, key):
        x = overall[a][key]["mean"]
        y = overall[b][key]["mean"]
        return f"{y:.2f} → {x:.2f}m ({(x / y - 1) * 100:+.1f}%)"

    lines = [
        "# V4: 팀 대형과 희소 수송 이동의 재학습 실험",
        "",
        "이 문서는 V3에서 관측된 오차를 분석한 뒤, V4의 학습 계획을 고정하고 재학습·평가한 개발 실험 기록이다. V3와 동일한 검증 188경기, 1,504개 구간, 10–60초 예측, 시드 41·42·43, Diffusion 생성 수 20, 32단계 DDIM, 기존 정답 마스크와 경기별 평균 지표를 사용했다. 테스트 94경기는 열지 않았다. 검증 자료를 반복 사용했으므로 독립적인 최종 성능 검증이나 사전등록 실험으로 해석하지 않는다.",
        "",
        "## 판단: 전체 경로 정확도 개선에는 실패",
        "",
        "V4는 관측자 Diffusion의 팀원 거리 오차를 163.03→82.24m로 줄였지만, 개인 평균 경로 오차를 120.71→131.15m, 팀 중심 오차를 103.55→113.44m로 악화시켰다. 같은 입력의 V4 회귀는 110.10/94.85/59.68m로 세 오차 모두 더 작다. 따라서 현재 V4 계층형 Diffusion을 경로 정확도와 대형 유지의 최종 개선안으로 채택하지 않는다.",
        "",
        "독립 잔차 대조군의 거리 오차 68.60m가 계층형의 82.24m보다 낮았다. 계층형−대조군 차이는 +13.64m이며 경기 bootstrap 기술적 95% 구간은 [11.52,15.91]m다. 제안한 구조+거리 손실 묶음이 대형을 더 잘 유지한다는 주장은 이 결과가 지지하지 않는다. 반면 계층형은 대조군보다 Energy score가 낮고 best-of-20도 좋아, 더 다양한 대형과 단일 정답 유사도 사이의 절충이 있다. V3 대비 Energy score의 감소 115.28→112.04m는 차이 구간 [-7.79,1.24]m가 0을 포함하므로 개선이 확실하다고 말하지 않는다.",
        "",
        "희소 구간은 별도로 개선됐다. 긴급 수송 개인 경로 오차는 725.77→506.38m, 직접 재진입 항공기는 903.60→713.66m였다. 하지만 재진입 항공기의 팀원 거리 오차는 519.21→530.85m로 악화됐고, 주 검증의 재진입 이후 에피소드 전체 ADE도 128.71→141.95m로 악화됐다. 따라서 재진입 전반을 해결했다고 해석하지 않는다.",
        "",
        "## 실제 결과",
        "",
        table(overall, list(labels)),
        "",
        "단위는 m이며 모두 낮을수록 좋다. **개인·팀 중심 ADE는 생성 20개를 평균한 경로의 오차**, **팀원 거리 MAE는 각각의 생성 경로에서 계산한 팀원 간 거리 오차**다. 회귀는 생성 1개인 퇴화분포로 평가했다. Energy score는 경로 전체의 분포 지표이며, ADE와 함께 판단해야 한다. 표는 시드별 경기 평균을 다시 평균한 값이다.",
        "",
        f"관측자 Diffusion의 V3→V4 변화: 개인 경로 {change('v4_observer_diffusion', 'v3_observer_diffusion', keys[0])}, 팀 중심 {change('v4_observer_diffusion', 'v3_observer_diffusion', keys[1])}, 팀원 거리 {change('v4_observer_diffusion', 'v3_observer_diffusion', keys[2])}.",
        "",
        "![동일 검증 조건 비교](metrics_comparison.png)",
        "",
        "가능한 미래를 여러 개 생성하는 모델은 하나의 중심 경로를 예측하는 회귀보다 개별 경로의 평균 오차가 커질 수 있다. 이 차이만으로 생성 분포가 잘못됐다고 판단할 수는 없다. 그러나 같은 K=20 평균 경로까지 악화됐는지, 대형이 얼마나 과도하게 흔들리는지, 분포 점수가 개선되는지는 각각 확인할 수 있다. 이 실험은 세 질문을 분리한다.",
        "",
        "## V3 오차의 원인: 확인된 사실과 해석",
        "",
        "- V3 코드에서 팀원별 초기 잡음은 독립이었다. 잡음의 팀 평균을 함께 입력했지만, 팀 중심과 상대 위치를 분리하지 않았고 생성된 경로의 팀원 간 거리에 직접 손실을 주지 않았다. 팀 중심 손실은 평균 예측 헤드에만 적용됐다. 따라서 중심이 맞아도 팀원들의 상대 위치는 흩어질 수 있었다.",
        "- 관측자 V3의 결정론적 예측 헤드 ADE는 109.50m, 같은 헤드의 팀원 거리 오차는 59.93m였다. 생성 20개의 평균 경로는 개인 ADE 120.71m, 팀원 거리 오차 70.33m였고, 개별 생성 경로의 거리 오차는 163.03m였다. 회귀의 108.17m/59.79m와 비교하면, 평균 예측 헤드보다 잔차 생성 단계에서 악화가 컸다.",
        "- 생성 평균과 평균 예측 헤드의 경로 차이는 약 41m였다. 이는 전부 체계적 편향을 뜻하지 않는다. 유효 지점 가중 RMS로 계산하면 전체 차이 약 52–54m 중 K=20의 유한 표본 평균 오차로 예상되는 RMS가 약 42m였다. 남은 제곱 차이 추정치는 약 964–1,170m²였다. 이 분해는 iid 생성 가정의 기술 통계이며 ADE의 가법 분해나 원인의 인과적 증명이 아니다.",
        "- 공통으로 관측되는 팀 미래 시점에서, 시점별 상대 위치의 생성 분산 비율은 경기별 평균 약 절반이었다. 팀 중심의 불확실성과 별도로 대형이 크게 흔들렸다.",
        "- V3 16,000회 표집에서 직접 재진입 항공기는 시드별 5/9/5회, 긴급 호출 수송은 75/67/68회였다. 표본 부족은 학습 계획으로 확인된다. 이것이 모든 수송 오차를 설명한다는 주장은 대조 실험 없이 할 수 없다.",
        "",
        "[오차 분해 JSON](v3_error_diagnosis.json), [유한 생성 수 진단](v3_monte_carlo_diagnosis.json), [실제 학습 노출량](leakage_audit.json).",
        "",
        "## 구현한 모델",
        "",
        "회귀 모델은 기존 이종 그래프·이동 이력·현재 자기장·현재 이동 상태 인코더를 유지한다. 공유 팀 중심 변위와 팀원별 상대 변위를 별도 헤드로 예측하며, 상대 변위의 팀 평균은 0으로 투영한다. 현재 관측 위치와 관측 시점 나이를 반영한 기준점에 이를 더해 20·40·60초의 3개 좌표 매듭을 만들고, 10초 간격 6개 예측으로 선형 보간한다.",
        "",
        "Diffusion은 완성된 해당 회귀 모델을 **그대로 고정**한다. 회귀와 정확히 동일한 그래프·현재 자기장·상태·이력 정보를 사용하며, 정답과 회귀 경로의 차이만 추가로 학습한다. 이렇게 하면 잔차 손실이 회귀 인코더를 바꾸어 점 예측을 악화시키는 경로를 차단한다. 다만 Diffusion의 20개 평균이 회귀 경로와 같도록 사후 보정하지 않았으므로, 생성 편향과 유한 표본 오차는 평가에 남아 있다.",
        "",
        r"각 미래 매듭에서 잔차를 $r_p=c+\delta_p$, $\sum_{p\in\mathcal{P}}\delta_p=0$로 분리한다. 팀 중심 잡음 1개를 공유하고, 팀원 상대 잡음은 합이 0이 되도록 투영한다. 중심과 상대 잔차의 RMS 척도는 학습 캐시에서만 추정한다. 마스킹된 팀원은 평균·손실·투영에서 제외한다.",
        "",
        "```mermaid",
        "flowchart LR",
        "  A[현재 이종 그래프·자기장·상태·이력] --> B[팀 중심·상대 변위 회귀]",
        "  B --> C[고정 평균 경로와 조건 특성]",
        "  N[팀 중심 잡음 1개와 합이 0인 상대 잡음] --> D[조건부 x0 잡음 제거 32회]",
        "  C --> D",
        "  D --> R[팀 중심 잔차와 상대 위치 잔차]",
        "  R --> Y[회귀 경로에 더한 공동 경로 20개]",
        "  C --> Y",
        "```",
        "",
        "조건 텐서는 실제 실행으로 확인했다: 그래프 노드 수는 가변, 노드 수치 피처 폭 32; 자기장 기하 `[B,4,14]`; 상태+관측 마스크 `[B,4,14]`; 이력 `[B,4,4,6]`; 고정 인코더와 회귀 변위를 합친 조건 `[B,4,116]`; 중심+상대 잠재변수 `[B,5,3,2]`; 출력 `[B,20,4,6,2]`. 자세한 실제 B=2 출력은 [텐서 계약](runtime_tensor_contract.json)에 있다.",
        "",
        "회귀 손실은 좌표 Huber + 0.1×팀 중심 + 0.05×이동 모드 CE + 0.05×현재 자기장에 대한 경계 거리 + 0.2×미래 팀원 거리 Huber다. Diffusion은 정규화 잔차의 x0 예측 MSE + 0.2×복원된 매듭의 실제 미래 팀원 거리 Huber를 사용한다. 대형을 고정하거나 팀원을 무조건 모으는 손실이 아니다. 실제로 팀이 벌어지면 벌어지는 거리를 학습한다. 사망·재진입 때문에 현재 팀원 전체의 정답을 알 수 없는 미래 매듭은 잔차 학습에서 제외한다. 이는 입력 정보가 아니라 학습 손실의 마스크다.",
        "",
        "잔차의 학습·정규화에는 회귀가 학습한 동일 표본의 잔차를 사용했다. Out-of-fold 잔차나 별도 보정 집합이 아니므로 예측 분산을 과소 추정할 가능성이 있다. 이 한계 때문에 생성 포함률을 함께 공개하며 보정된 확률분포라고 주장하지 않는다.",
        "",
        "회귀는 2,000업데이트×8표본, 잔차는 고정 특성 캐시에서 4,000업데이트×32표본으로 학습했다. Diffusion은 회귀 학습에 추가 계산을 사용하므로 총 계산량이 같은 비교는 아니다. 잔차 대조군끼리는 같은 특성·표본·업데이트·최적화 계획을 사용한다. 학습률·손실 가중치·표집 계획은 V4 검증 결과 전에 고정했고 마지막 EMA만 평가했다.",
        "",
        "## 희소 이동 표집과 대조 실험",
        "",
        "미니배치 8개 중 6개는 기존 경기 균등 표집을 보존하고, 1개는 긴급 수송/재진입 항공기를 번갈아, 1개는 그 두 모드 이외의 재진입 생명 에피소드를 뽑았다. 각 희소 계층 안에서는 경기 먼저, 해당 경기 구간을 다음으로 균등 표집한다. 657개 학습 경기 모두에서 표집했지만 전체 약 136만 학습 구간을 한 차례씩 순회한 것은 아니다. 자연 발생 빈도로 되돌리는 중요도 가중은 적용하지 않았으므로 학습 분포가 의도적으로 바뀌었다. 검증 분포는 그대로다.",
        "",
        "| 시드 | 긴급 수송 제시 횟수 | 재진입 항공기 제시 횟수 | 재진입 에피소드 제시 횟수 | 전체 고유 구간 |",
        "|---|---:|---:|---:|---:|",
    ]
    for s, r in coverage["plans"].items():
        lines.append(
            f"| {s} | {r['mode_counts']['4']} | {r['mode_counts']['5']} | {r['reentry_draws']} | {r['unique_windows']} |"
        )
    lines += [
        "",
        "학습 풀에는 긴급 수송 6,123구간/471경기, 재진입 항공기 462구간/296경기, 두 수송 모드 이외 재진입 에피소드 52,171구간/617경기가 있었다. 제시 횟수 증가는 독립 사건 수 증가가 아니다. 잔차 학습은 캐시에서 재표집하므로 더 자주 보지만 새로운 경기·구간을 추가하지 않는다. 정확한 고유 구간·경기 수와 잔차 노출 횟수는 누수 감사에 기록했다.",
        "",
        "독립 잔차 대조군은 동일한 V4 회귀를 고정하고 x0 잔차를 학습하되, 별도 팀 중심/합이 0인 상대 잔차 구조와 생성 거리 손실을 빼었다. 대조군도 팀 평균 잡음과 공통 조건을 입력받으므로 생성 결과의 완전한 통계적 독립성을 가정하지 않는다. 따라서 이 대조는 **계층 구조+거리 손실 묶음**의 효과를 본다. 두 요소 각각의 독립 효과를 분리한 것은 아니다. 두 잔차 모델의 학습 가능한 파라미터 수도 다르므로 동일 용량의 구조 제거 실험으로 해석하지 않는다. 또한 V3→V4는 표집·회귀 구조·거리 손실·고정 백본·예측 목적·학습량이 함께 바뀌었으므로 개선 전체를 하나의 요소 때문이라고 해석할 수 없다.",
        "",
        "기존 표집을 사용한 V4 회귀 시드41 대조군도 별도로 학습했다. 같은 구조·손실·2,000업데이트에서 표집만 바뀐다. 단일 시드의 탐색적 대조다.",
        "",
        "| 지표 | 기존 표집 V4 회귀(41) | 균형 표집 V4 회귀(41) | 균형−기존, 경기 bootstrap 95% 구간 |",
        "|---|---:|---:|---|",
    ]
    for k in keys:
        r = d["sampling_control"]
        q = r["balanced_minus_uniform"][k]
        lines.append(
            f"| {k} | {r['uniform'][k]['mean']:.2f} | {r['balanced'][k]['mean']:.2f} | {q['difference_m']:+.2f} [{q['descriptive_ci95'][0]:+.2f}, {q['descriptive_ci95'][1]:+.2f}] |"
        )
    for group, title in [
        ("emergency_pickup", "긴급 호출 수송"),
        ("redeploy_aircraft", "직접 확인된 재진입 항공기"),
    ]:
        r = d["transport"][group]
        lines += [
            "",
            f"### {title}: 기존과 같은 {r['counts']['windows']}구간 / {r['counts']['matches']}경기",
            "",
            table(
                r["models"],
                [
                    "v3_observer_regression",
                    "v3_observer_diffusion",
                    "v4_observer_regression",
                    "v4_observer_diffusion",
                    "v4_server_diffusion",
                ],
            ),
            "",
            "주 평가와 일부 겹치는 모드별 보조 검증이다. 독립 검증 집합이나 실제 발생 비율을 반영한 전체 평균으로 취급하지 않는다.",
        ]
    lines += [
        "",
        "## 인간·봇 및 재진입 이후 구간",
        "",
        "| 집단 | 구간 수 | 관측 회귀 ADE | 관측 Diffusion ADE | 관측 회귀 거리 MAE | 관측 Diffusion 거리 MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g in ["human", "bot", "post_reentry"]:
        r = d["subgroups"][g]
        a = r["models"]["v4_observer_regression"]
        b = r["models"]["v4_observer_diffusion"]
        lines.append(
            f"| {g} | {r['windows']} | {a[keys[0]]['mean']:.2f} | {b[keys[0]]['mean']:.2f} | {a[keys[2]]['mean']:.2f} | {b[keys[2]]['mean']:.2f} |"
        )
    lines += [
        "",
        "인간·봇 표시는 평가 집단 구분에 쓰며 입력에 주입하지 않았다. 플레이어 유형 효과와 시작 위치·생존 시점·이동 모드의 차이는 아직 통제하지 않았다.",
        "",
        "## 생성 다양성과 한계",
        "",
        "| 생성 모델 | 평균 좌표별 10–90% 밴드 폭(m) | 좌표별 정답 포함률 |",
        "|---|---:|---:|",
    ]
    for version, kind in [("v3", "diffusion"), ("v4", "diffusion"), ("v4", "independent")]:
        rs = [gen[version][f"observer_{kind}_{s}"] for s in [41, 42, 43]]
        width = sum(r["coordinate_10_90_width_m"]["mean_over_matches"] for r in rs) / 3
        cov = sum(r["coordinate_10_90_coverage"]["mean_over_matches"] for r in rs) / 3
        lines.append(f"| {version} 관측 {kind} | {width:.2f} | {cov:.1%} |")
    lines += [
        "",
        "거리 오차 감소가 불확실성의 정확한 학습을 자동으로 뜻하지 않는다. 대형 손실은 관측된 한 미래 대형에 맞추므로 가능한 여러 대형의 분산을 줄이는 유인이 있다. 분산을 줄여 평균 주변에 모으면 ADE/거리 오차가 좋아질 수 있으므로, 밴드 폭·포함률·Energy score·best-of-20를 함께 공개했다. 위 포함률은 좌표별 주변 포함률이며, 전체 경로 동시 포함률이나 보장된 신뢰구간이 아니다. best-of-20는 실제 선택 가능한 예측 성능이 아니라 oracle 진단이다. 세 시드 결과와 경기 단위 차이 구간은 [전체 비교 JSON](comparison_summary.json)에 있다. 구간은 고정 시드에 조건부이며 다중 비교 보정이나 계정 간 의존성 보정은 하지 않았다.",
        "",
        "## V4 실패의 후속 진단과 다음 실험 경계",
        "",
        "동일한 검증 시점에서 관측자 생성의 상대 위치 분산은 약 15,088→5,963m²로 감소했지만, 팀 중심 분산은 약 17,665→40,427m²로 증가했다. 대형이 덜 흔들려도 팀 전체가 잘못 이동할 수 있다는 문제를 보여준다. 또한 고정 회귀 경로와 생성 평균의 유효 지점 가중 RMS 차이는 시드별 약 130–180m였고, K=20 유한 표본 효과로 예상한 RMS는 약 49–55m였다. 생성 수만 늘려 해결될 문제로 보기 어렵다. 이 수치는 사후 기술 진단이며 모델을 다시 맞추거나 주 평가를 변경하지 않았다.",
        "",
        "가능한 원인은 희소 이동을 늘린 학습 혼합과 정규화 잔차 x0 MSE의 결합, 강건한 Huber 점 예측과 MSE 기반 잔차 중심의 목적 차이, 반복 잡음 제거 과정의 평균 편향이다. 이들을 분리하는 추가 대조 실험은 수행하지 않았으므로 확정 원인이라고 쓰지 않는다. 생성 폭의 축소와 포함률 87.0%→52.6% 하락도 함께 남아 있어 보정된 불확실성을 학습했다는 주장도 보류한다.",
        "",
        "현재 점 경로와 팀 대형 오차를 기준으로는 회귀를 기준 예측기로 유지한다. 후속 실험은 회귀 중심과 확률적 잔차 중심을 별도로 추정·검증하고, 학습 경기 내 교차 적합 잔차와 별도 보정 자료로 생성 분포를 보정하는 설계를 검토할 필요가 있다. 같은 추가 학습 예산의 결정론적 잔차 모델, 구조와 거리 손실을 각각 제거한 대조, 이동 모드별 중심/상대 분산 보정이 추가로 필요하다. 이번 결과를 본 뒤 검증에 맞춰 스케일을 줄이거나 생성 경로를 재중심화하지 않았다.",
        "",
        "[예측 시간별 오차](horizon_diagnostics.png), [후속 평균 이동 진단](v4_residual_drift_diagnosis.json).",
        "",
        "## 누수·정보 범위·테스트 잠금",
        "",
        "- 학습 657 / 내부 검증 188 / 잠금 94경기의 ID 교집합은 없다. V3의 845개 학습·검증 bank SHA-256, 검증 인덱스·정답·수송 계획·기존 시각화 사례가 그대로인지 확인했다.",
        "- 공간 사전·어휘·복원 입력을 V3에서 변경하지 않았다. V3의 as-of 복원·학습 경기 공간 사전 검증을 이어 쓰며, 현재 시점 이후 정답 수정이 입력에 영향을 주지 않는 검사와 관측자에게 다른 행위자의 비공개 상태가 들어가지 않는 검사를 포함해 테스트를 실행했다.",
        "- 미래 모드·미래 원 위치·미래 팀원 생존 마스크는 조건에 들어가지 않는다. 현재 알려진 팀원으로 예측 대상을 고정한다. 현재 팀원이 죽었다가 재진입하는 경계를 넘어가는 경로는 학습 정답에서 제외한다.",
        "- 관측자 모델의 자기 무장·상태는 서버 텔레메트리로 복원한 자기 상태 oracle이고 팀원 위치는 HUD 근사다. 실제 시야·소리·적 인식 로그가 아니다. 피해 이벤트는 교전 증거일 뿐 모든 적 인식을 복원하지 않는다. 자기장 발표 시점과 실제 HUD 지연까지 검증한 상태가 아니다.",
        "- 출력은 현재 팀원의 이동 경로이며, 미래 사망·복귀 시점이나 새 팀원 출현을 생성하는 모델은 아니다. 미래 생존/관측 가능 마스크에 따라 평가 대상이 달라지므로 생존 과정 전체의 공동 모델로 해석하지 않는다. 20·40·60초의 세 매듭을 보간하는 공통 표현은 급격한 착지·선회 궤적을 제한한다. 미래 모드 분류는 보조 손실이고 경로를 모드별 동역학으로 전환하는 하이브리드 생성기는 아니다.",
        "- 직접 재진입 항공기와 긴급 수송 클래스는 확인된 모드다. 항공기 하차 후 mode7은 불확실한 추론이며, 리플레이를 통한 탑승·착지 정밀 정답은 없다.",
        "- 테스트 loader 차단, 테스트 bank/index/공간 파일 부재, 테스트 예측 0건을 유지했다. **잠근 94경기는 과거 V2 학습에 사용된 이력이 있어 역사적으로 미관측한 최종 테스트가 아니다.** 논문 최종 평가에는 이후 수집한 새로운 경기가 필요하다.",
        "- 100m 공간 그래프는 관측·추정 이동 연결의 근사다. 건물 충돌·지형 높이·교량 연결·통행 가능성에 대한 물리적 정확도를 이번 실험으로 입증하지 않았다.",
        "",
        "## 검증 사례와 재현",
        "",
        "기존 V3 네 사례와, V4 결과 전에 고정한 수송 두 사례를 같은 좌표 범위로 시각화했다. 지형 배경은 참고 이미지다. 실선은 정답, 점선은 평균 경로, 하단은 팀원 쌍별 거리이며 생성 10–90% 밴드는 신뢰구간이 아니다.",
        "",
    ]
    for i in range(1, 7):
        lines.append(f"- [검증 사례 {i}](validation_case_{i}.png)")
    lines += [
        "",
        "[인터랙티브 뷰어](viewer.html)에서 정보 범위·시드·개별 생성 경로를 바꿀 수 있다.",
        "",
        "```sh",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v4.py train-seed --seed 41",
        "# 동일하게 --seed 42, --seed 43; 완료 체크포인트는 재사용한다.",
        "/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v4.py train-uniform",
        "/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/experiment_v4.py evaluate",
        "/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/analyze_v4.py",
        "/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/visualize_v4.py",
        "/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/report_v4.py",
        "/private/tmp/pubg-spatial-venv/bin/python tools/movement_diffusion/finalize_v4.py",
        "```",
        "",
        "고정 계획은 [experiment_spec.json](experiment_spec.json), 학습 노출량은 [sampling_coverage.json](sampling_coverage.json), 누수 감사는 [leakage_audit.json](leakage_audit.json), 검증된 체크포인트·추론 재현·환경은 [delivery_verification.json](delivery_verification.json), 해시 목록은 [artifact_manifest.json](artifact_manifest.json)에 저장했다. 학습 전체를 다시 실행한 재현 실험은 하지 않았으며, 저장된 모든 모델의 첫 검증 예측을 동일 난수로 재생해 확인했다.",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "REPORT_KO.md").write_text(text)
    dest = ROOT / "docs/experiment_design/HETEROGRAPH_DIFFUSION_V4_FORMATION_AUDIT_2026-10-15.md"
    dest.write_text(text.replace("](", "](../../reports/heterograph_diffusion_experiment_v4/"))
    print("REPORT_WRITTEN", flush=True)


if __name__ == "__main__":
    main()
