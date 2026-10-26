"""Readable preprocessing receipt and offline notebook; no model training."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(root, report):
    report.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "summary.json").read_text())
    prep = json.loads((root / "preparation_summary.json").read_text())
    audit = json.loads((report / "validation.json").read_text())
    with (root / "matches.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (root / "ablation_summary.csv").open() as f:
        ablations = list(csv.DictReader(f))
    db = sqlite3.connect(f"file:{root / 'catalog.sqlite'}?mode=ro", uri=True)
    labels = db.execute("""SELECT m.role,count(*),sum(o.rank_mask),sum(o.winner_mask),
                         sum(o.terminal_time_mask) FROM outcomes o JOIN matches m USING(match_id)
                         GROUP BY m.role""").fetchall()
    issues = db.execute(
        "SELECT code,count(*) FROM issues GROUP BY code ORDER BY count(*) DESC"
    ).fetchall()
    maps = db.execute(
        "SELECT map,role,count(*) FROM matches GROUP BY map,role ORDER BY map,role"
    ).fetchall()
    db.close()
    labels_table = "\n".join(
        f"| {r} | {n:,} | {rank:,} | {win:,} | {terminal:,} |"
        for r, n, rank, win, terminal in labels
    )
    issue_table = "\n".join(f"| {code} | {n:,} |" for code, n in issues)
    coverage_table = "\n".join(
        f"| {role} | {v['anchors']:,} | {v['ego_any_future']:,} | {v['ego_complete_60s']:,} |"
        for role, v in audit["coverage_by_role"].items()
    )
    ready = sum(a["status"] == "ready_manifest" for a in ablations)
    patches = ["40.2", "41.1", "41.2", "unknown"]
    roles = [
        "train",
        "validation",
        "protected",
        "reserved_patch",
        "excluded_prior_validation",
        "quarantine",
    ]
    colors = ["#147d64", "#3677bb", "#8b5867", "#b29b55", "#a3a8ad", "#d98141"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(patches))
    for role, color in zip(roles, colors):
        values = np.array(
            [sum(r["patch"] == p and r["role"] == role for r in rows) for p in patches]
        )
        ax.bar(patches, values, bottom=bottom, label=role, color=color)
        bottom += values
    for i, n in enumerate(bottom):
        ax.text(i, n + 150, f"{int(n):,}", ha="center", fontsize=10)
    ax.set_ylim(0, float(bottom.max()) * 1.15)
    ax.set_ylabel("Matches (inventory, before task-quality filtering)")
    ax.set_title("Patch-based split: 40.2 training → 41.1 development")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(report / "patch_split.png", dpi=180)
    plt.close(fig)
    content = f"""# 패치 단위 월드모델 전처리 결과

모델 학습과 성능 ablation은 실행하지 않았다. 아래는 실제 전처리와 검증 결과다.

## 처리 범위

- 전체 메타데이터: {prep["inventory"]:,}경기.
- 일정 기반 패치 추정: 40.2 {prep["patch_counts"]["40.2"]:,}, 41.1 {prep["patch_counts"]["41.1"]:,},
  41.2 {prep["patch_counts"]["41.2"]:,}, 미확정 {prep["patch_counts"]["unknown"]:,}경기.
- **학습 40.2 {prep["eligible_by_role"]["train"]:,}경기 / 개발 검증 41.1 {prep["eligible_by_role"]["validation"]:,}경기**를 전처리했다.
- 해당 역할에 배정된 경기 중 필수 원천·참가자 결합 조건을 만족하지 못한
  {prep["roles"]["train"] - prep["eligible_by_role"]["train"]:,}개 학습 경기와
  {prep["roles"]["validation"] - prep["eligible_by_role"]["validation"]:,}개 검증 경기는 목록에만 보존했다.
- 41.2는 보류했고, 과거 테스트 지정 경기와 경계 미확정 경기의 상세 전처리를 하지 않았다.
- 이전 검증으로 지정된 40.2 경기는 학습으로 옮기지 않았다.
- 분할 충돌 이력 {prep["historical_conflicts"]:,}경기는 `exposure_registry.csv`에 남겼다.
  과거 테스트 지정의 합집합은 {prep["roles"]["protected"]:,}경기다.

![패치별 분할](patch_split.png)

시점 경계는 공식 PC 점검 일정에 근거한 **추정 패치**다. 정확한 실행 빌드·핫픽스까지 검증하지 않았다.
최종 테스트로 새롭게 미관측한 경기가 확보된 것은 아니다.

## 생성된 데이터

- 처리 완료 {summary["preprocessed_matches"]:,}경기.
- 개인별 anchor {summary["totals"]["anchors"]:,}개. 30초 간격이며 같은 경기의 anchor는 독립 표본이 아니다.
- 유효 as-of 상태 {summary["totals"]["valid_states"]:,}개, 정답 시점 상태 {summary["totals"]["target_states"]:,}개.
  이 수치는 길이 60초가 모두 관측된 경로 수와 다르다.
- 원천 좌표의 cm→m 변환, 이벤트 시각 정렬, 13프레임 과거/6시점 미래,
  생명·연결 구간, 이동 모드, 장비 기록, 개인·팀·차량 증거와 자기장 상태를 저장했다.
- 팀원 수는 가변이며 4명으로 자르지 않는다.
- 문제 구간은 SQLite `issues`의 경기·개인·시작/종료 시각으로 찾을 수 있다.
- 각 경기의 원천 추출물과 상태 bank에 SHA-256을 저장했다.
- 완전한 탄약·인벤토리, 실제 적 인식, 차량 고유 인스턴스, 정밀 지형은 확보했다고 주장하지 않는다.
- 공식 종료 이후 위치 {audit["post_end_rows_excluded"].get("positions", 0):,}행을 상태·정답·공간 연결에서 제외했다.

| 역할 | 전체 anchor | 개인 미래 정답 일부 존재 | 개인 60초 전체 정답 존재 |
|---|---:|---:|---:|
{coverage_table}

입력 이력·현재 유효 상태 조건을 만족하는 anchor가 없는 경기는 학습
{audit["coverage_by_role"]["train"]["matches_without_anchors"]:,}개, 검증
{audit["coverage_by_role"]["validation"]["matches_without_anchors"]:,}개다.
이들의 원천·품질 기록은 유지하지만 경로 학습 표본 수에는 포함하지 않는다.

미래 정답이 없는 anchor도 과거 복원·검증된 결과 라벨 등 사용 가능한 과제에는 남길 수 있다.
경로 손실은 해당 미래 마스크를 반드시 사용해야 한다.

## 최종 결과 라벨

| 역할 | 팀 결과 행 | 순위 사용 마스크 | 우승 사용 마스크 | 최종 탈락 시각 마스크 |
|---|---:|---:|---:|---:|
{labels_table}

순위/우승 마스크는 원천 필드 간 일관성을 확인한 **후보 라벨**이다.
개인·팀 순위가 충돌하면 원본을 보존하고 해당 손실을 끈다.
마지막 사망을 최종 팀 탈락으로 확정하지 않았으므로 terminal_time_mask는 현재 꺼져 있다.
결과 라벨이 불확실해도 정상적인 이동 구간은 다른 과제에 보존한다.

## 구간별 표시

| 문제·근거 코드 | 기록 수 |
|---|---:|
{issue_table}

직접 재진입 증거, 재진입 추정, 속도 검토 후보, 관측 공백, 무효 좌표는 같은 오류로 취급하지 않는다.
`review_not_clip`은 좌표를 임의로 지도 안에 밀어 넣지 않았다는 뜻이다.

## Ablation 준비

총 {len(ablations)}개 목록 중 학습·검증 표본이 모두 있는 목록은 {ready}개다.
빈 도메인은 `unavailable_empty_domain`으로 남겼다.
맵·인칭·팀 규모, 공간 제거/관측 연결만, 팀 노드 제거, 결과 손실 제거,
짧은 이력, 사전학습 없음, 관측자/서버, 조건 변수 제거 목록을 생성했다.
이는 **학습·성능 비교 완료가 아니다**.

모든 목록의 학습은 40.2이고 검증은 41.1이다.
공간 지지·어휘는 해당 ablation의 정확한 학습 ID로 제한한다.
정규화 통계는 적합하지 않았으며 후속 학습에서도 해당 fit 자료만 써야 한다.
41.1 공간 연결을 학습 prior에 넣지 않았다. 따라서 검증의 관측 연결 지지는 비어 있고
100m 이웃 인접성은 불확실한 가설 연결로 남는다.

## 검증

- 누수·정보 가림·재진입·패치 경계·종료 후 기록 차단·가변 팀 크기·ablation 로더 단위 시험 13개.
- 실제 도메인 그룹 {len(audit["domain_prefix_checks"])}개를 점검했다.
  이 중 {sum(bool(x.get("prefix_equal")) for x in audit["domain_prefix_checks"])}개는
  prefix 재구성·정보 범위 검사를 통과했고 나머지는 이력이 짧아 해당 검사를 건너뛰었다.
- 실제 경기 {len(audit["full_bank_rebuild_checks"])}개는 원천에서 다시 구성한 bank의 SHA-256이 기존 파일과 일치했다.
- ablation 목록 {len(audit["ablation_checks"])}개에서 학습/검증/보호 교집합과 패치 경계를 검사했다.
- 실제 검증 그래프 {len(audit["runtime_graph_examples"])}개를 로드해 노드·연결·정답 형식을 확인했다.
- 테스트/보류 경기 상세 파일 생성: 0.

아직 등장하지 않은 참가자의 생명 confidence에는 저장용 초기값이 있을 수 있다.
실제 입력은 known 마스크로 이를 제거한다. 원시 배열 전체 동일성 대신 이 마스크를 적용한
confidence와 관측자·서버 입력 각각의 prefix 동일성을 검사했다.

관측자 모델은 텔레메트리 기반 자기 상태·팀 HUD 근사다.
시야·소리·발표 지연을 실제로 관측한 데이터와 동일하지 않다.
공간 그래프는 이동 증거이며 건물·교량·충돌·통행 가능성 정답이 아니다.

## 경로

- 데이터 루트: `{root}`
- 목록: `matches.csv`, `exposure_registry.csv`, `ablation_summary.csv`, `ablations/`
- 상태: `banks/`, 원천 추출: `raw/`, 품질/결과/해시: `meta/`
- 검색: `catalog.sqlite` (`matches`, `issues`, `outcomes`, `spatial_support`)
- 재현 계약: `preprocess_spec.json`, `source_snapshot_at_prepare/`
- 검증: `validation.json`, `unit_tests.txt`, `PREPROCESS_REVIEW.ipynb`

후속 학습 전에 결과 라벨 정책과 낮은 관측 지지 조건의 평가 기준을 확정한다.
이번 실행에서는 어떠한 모델 가중치도 학습하지 않았다.
"""
    (report / "REPORT_KO.md").write_text(content)
    receipt = {
        "data_root": str(root),
        "summary": summary,
        "preparation": prep,
        "labels_by_role": labels,
        "issues": issues,
        "maps": maps,
        "ablation_ready": ready,
    }
    (report / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 월드모델 전처리 검토\n",
                "저장된 결과를 읽는 오프라인 검토입니다. 학습·DB 원본 변경을 수행하지 않습니다.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\nimport json, sqlite3\n",
                f"root = Path({str(root)!r})\n",
                "summary = json.loads((root / 'summary.json').read_text())\n",
                "{k: v for k, v in summary.items() if k != 'runtime_examples'}",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "db = sqlite3.connect(f'file:{root / \"catalog.sqlite\"}?mode=ro', uri=True)\n",
                "db.execute('SELECT role, count(*), sum(anchors) FROM matches GROUP BY role').fetchall()",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "db.execute('SELECT code, count(*) FROM issues GROUP BY code ORDER BY count(*) DESC').fetchall()"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "db.execute('SELECT sum(rank_mask), sum(winner_mask), sum(terminal_time_mask) FROM outcomes').fetchone()"
            ],
        },
    ]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "cells": cells,
    }
    (report / "PREPROCESS_REVIEW.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"report": str(report / "REPORT_KO.md"), "ablation_ready": ready}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.report)
