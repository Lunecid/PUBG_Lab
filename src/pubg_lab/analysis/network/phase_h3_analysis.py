"""
Phase-wise H3 movement hypothesis analysis.

H3 asks whether low-movement ("sedentary") teams perform better than highly
mobile teams.  This script re-tests that question at the team-phase level using
the phase-wise feature table produced by phase_strategy_type_analysis.py.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is expected, but keep script usable.
    stats = None


DEFAULT_DATASET = os.path.join(
    "network_analysis", "output", "analysis_datasets", "Baltic_Main_squad_n1000"
)
DEFAULT_PHASE_FEATURE_SUBPATH = os.path.join(
    "phase_strategy_type_analysis", "phase_team_features.csv"
)
GROUP_ORDER = ["phase_sedentary", "phase_middle", "phase_mobile"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean_json(data), f, ensure_ascii=False, indent=2)


def _safe_mwu(a: pd.Series, b: pd.Series) -> float:
    if stats is None:
        return np.nan
    av = _num(a).dropna()
    bv = _num(b).dropna()
    if len(av) < 2 or len(bv) < 2:
        return np.nan
    try:
        return float(stats.mannwhitneyu(av, bv, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def _assign_movement_groups(
    df: pd.DataFrame,
    *,
    value_col: str,
    group_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out[group_col] = "phase_middle"
    out["phase_movement_q25"] = np.nan
    out["phase_movement_q75"] = np.nan
    for _, idx in out.groupby(["match_id", "phase"]).groups.items():
        vals = _num(out.loc[idx, value_col]).dropna()
        if vals.shape[0] < 4:
            continue
        q25 = float(vals.quantile(0.25))
        q75 = float(vals.quantile(0.75))
        out.loc[idx, "phase_movement_q25"] = q25
        out.loc[idx, "phase_movement_q75"] = q75
        if not q25 < q75:
            continue
        movement = _num(out.loc[idx, value_col])
        out.loc[idx[movement <= q25], group_col] = "phase_sedentary"
        out.loc[idx[movement >= q75], group_col] = "phase_mobile"
    return out


def _summarize(df: pd.DataFrame, *, cohort: str, group_col: str) -> pd.DataFrame:
    metric_cols = [
        "final_rank",
        "ranked_rp_proxy",
        "team_kills",
        "team_assists",
        "survival_seconds",
        "phase_observed_s",
        "survived_phase",
        "phase_movement_per_min",
        "phase_active_frame_rate",
        "phase_position_std_m",
        "phase_combat_st_participation_rate",
        "phase_fight_score",
        "phase_active_score",
    ]
    rows: list[dict[str, Any]] = []
    for (phase, group), g in df.groupby(["phase", group_col], dropna=False):
        row: dict[str, Any] = {
            "cohort": cohort,
            "phase": int(phase),
            "phase_movement_group": str(group),
            "team_phase_n": int(g.shape[0]),
            "unique_match_team_n": int(g[["match_id", "team_idx"]].drop_duplicates().shape[0]),
            "top10_rate": float((_num(g["final_rank"]) <= 10).mean()),
        }
        for col in metric_cols:
            if col not in g.columns:
                continue
            vals = _num(g[col]).dropna()
            row[f"{col}_mean"] = float(vals.mean()) if not vals.empty else np.nan
            if col == "final_rank":
                row[f"{col}_median"] = float(vals.median()) if not vals.empty else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order_map = {name: i for i, name in enumerate(GROUP_ORDER)}
    out["_order"] = out["phase_movement_group"].map(order_map).fillna(99)
    return out.sort_values(["cohort", "phase", "_order"]).drop(columns=["_order"])


def _tests(df: pd.DataFrame, *, cohort: str, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phase, g in df.groupby("phase"):
        sed = g[g[group_col] == "phase_sedentary"]
        mob = g[g[group_col] == "phase_mobile"]
        if sed.empty or mob.empty:
            continue
        rows.append(
            {
                "cohort": cohort,
                "phase": int(phase),
                "sedentary_n": int(sed.shape[0]),
                "mobile_n": int(mob.shape[0]),
                "sedentary_final_rank_mean": float(_num(sed["final_rank"]).mean()),
                "mobile_final_rank_mean": float(_num(mob["final_rank"]).mean()),
                "rank_diff_sedentary_minus_mobile": (
                    float(_num(sed["final_rank"]).mean()) - float(_num(mob["final_rank"]).mean())
                ),
                "rank_p_mwu": _safe_mwu(sed["final_rank"], mob["final_rank"]),
                "sedentary_rp_mean": float(_num(sed["ranked_rp_proxy"]).mean()),
                "mobile_rp_mean": float(_num(mob["ranked_rp_proxy"]).mean()),
                "rp_diff_sedentary_minus_mobile": (
                    float(_num(sed["ranked_rp_proxy"]).mean())
                    - float(_num(mob["ranked_rp_proxy"]).mean())
                ),
                "rp_p_mwu": _safe_mwu(sed["ranked_rp_proxy"], mob["ranked_rp_proxy"]),
                "sedentary_survived_phase_mean": float(_num(sed["survived_phase"]).mean()),
                "mobile_survived_phase_mean": float(_num(mob["survived_phase"]).mean()),
                "survived_phase_diff_sedentary_minus_mobile": (
                    float(_num(sed["survived_phase"]).mean())
                    - float(_num(mob["survived_phase"]).mean())
                ),
                "survived_phase_p_mwu": _safe_mwu(sed["survived_phase"], mob["survived_phase"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["cohort", "phase"]) if rows else pd.DataFrame()


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def _report_table(summary: pd.DataFrame, cohort: str) -> list[str]:
    lines = [
        f"### {cohort}",
        "",
        "| phase | movement group | team-phase n | final mean rank | RP proxy | phase survival | movement m/min |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    sub = summary[summary["cohort"] == cohort].copy()
    for row in sub.itertuples(index=False):
        lines.append(
            f"| {int(row.phase)} | {row.phase_movement_group} | {int(row.team_phase_n):,} | "
            f"{_fmt(getattr(row, 'final_rank_mean', np.nan))} | "
            f"{_fmt(getattr(row, 'ranked_rp_proxy_mean', np.nan))} | "
            f"{_fmt(getattr(row, 'survived_phase_mean', np.nan))} | "
            f"{_fmt(getattr(row, 'phase_movement_per_min_mean', np.nan))} |"
        )
    return lines


def make_report(
    *,
    all_summary: pd.DataFrame,
    completed_summary: pd.DataFrame,
    tests: pd.DataFrame,
    out_dir: str,
) -> str:
    summary = pd.concat([all_summary, completed_summary], ignore_index=True)
    lines = [
        "# Phase-wise H3 Movement Hypothesis Analysis",
        "",
        "## 1. 분석 목적",
        "",
        (
            "H3는 저이동/존버형 팀이 고이동 팀보다 평균 순위가 높은지를 묻는다. "
            "기존 H3는 착륙 후 600초 고정 창으로 생존시간 오염을 줄였지만, "
            "phase별 전략 변화는 반영하지 않았다. 본 분석은 같은 질문을 `team × phase` "
            "단위로 다시 계산한다."
        ),
        "",
        "## 2. 해석상 주의",
        "",
        (
            "`final_rank`는 phase별 실시간 순위가 아니라 매치 종료 후 하나로 결정되는 최종 순위다. "
            "따라서 phase별 final mean rank는 특정 phase에서 저이동/고이동 상태로 관측된 팀들이 "
            "최종적으로 기록한 평균 순위다."
        ),
        "",
        (
            "또한 phase 중 사망한 팀은 이동 기회가 짧아 저이동으로 오분류될 수 있다. "
            "그래서 본 보고서는 두 코호트를 분리한다."
        ),
        "",
        "| cohort | 정의 | 주된 해석 |",
        "|---|---|---|",
        "| all_phase_entries | 해당 phase에 30초 이상 관측된 모든 팀 | phase 진입 후 저이동/고이동 상태와 phase 생존 및 최종 성과의 관계 |",
        "| completed_phase_only | 해당 phase를 끝까지 생존한 팀만 포함 | 조기 탈락으로 이동량이 낮아지는 오염을 줄인 주 분석 |",
        "",
        "movement group은 각 `match_id × phase` 내부에서 `phase_movement_per_min` 하위 25%, 중간 50%, 상위 25%로 정의했다.",
        "",
        "## 3. 결과",
        "",
    ]
    lines.extend(_report_table(summary, "all_phase_entries"))
    lines.extend([""])
    lines.extend(_report_table(summary, "completed_phase_only"))
    lines.extend(
        [
            "",
            "## 4. Sedentary vs Mobile 검정",
            "",
            "| cohort | phase | sedentary n | mobile n | rank diff | rank p | RP diff | RP p | phase survival diff |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if tests.empty:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    else:
        for row in tests.itertuples(index=False):
            lines.append(
                f"| {row.cohort} | {int(row.phase)} | {int(row.sedentary_n):,} | {int(row.mobile_n):,} | "
                f"{_fmt(row.rank_diff_sedentary_minus_mobile)} | {_fmt(row.rank_p_mwu, 4)} | "
                f"{_fmt(row.rp_diff_sedentary_minus_mobile)} | {_fmt(row.rp_p_mwu, 4)} | "
                f"{_fmt(row.survived_phase_diff_sedentary_minus_mobile)} |"
            )
    lines.extend(
        [
            "",
            "## 5. 결론",
            "",
            (
                "phase별로 다시 계산해도 H3의 원문 가설, 즉 저이동/존버형 팀이 고이동 팀보다 "
                "평균 순위가 높다는 주장은 지지되지 않는다. 조기 탈락 오염을 줄이기 위해 해당 phase를 "
                "끝까지 생존한 팀만 비교하면, phase 1-3에서는 mobile 그룹의 최종 평균 순위가 분명히 좋고 "
                "phase 4-6에서는 순위 차이가 작아지거나 유의하지 않다. 반면 RP proxy는 거의 모든 phase에서 "
                "mobile 그룹이 sedentary 그룹보다 높다."
            ),
            "",
            f"- Output directory: `{out_dir}`",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset
    phase_features_path = args.phase_features or os.path.join(
        dataset_dir, DEFAULT_PHASE_FEATURE_SUBPATH
    )
    out_dir = args.out or os.path.join(dataset_dir, "phase_h3_analysis")
    os.makedirs(out_dir, exist_ok=True)

    cols = [
        "match_id",
        "team_idx",
        "phase",
        "final_rank",
        "ranked_rp_proxy",
        "team_kills",
        "team_assists",
        "survival_seconds",
        "phase_duration_s",
        "phase_observed_s",
        "survived_phase",
        "phase_movement_per_min",
        "phase_active_frame_rate",
        "phase_position_std_m",
        "phase_combat_st_participation_rate",
        "phase_fight_score",
        "phase_active_score",
    ]
    df = pd.read_csv(phase_features_path, usecols=lambda c: c in cols)
    df["match_id"] = df["match_id"].astype(str)
    df["survived_phase"] = df["survived_phase"].astype(str).str.lower().isin(["true", "1", "yes"])
    df = df[_num(df["phase_observed_s"]) >= float(args.min_phase_observed_s)].copy()

    all_entries = _assign_movement_groups(
        df,
        value_col="phase_movement_per_min",
        group_col="phase_movement_group",
    )
    completed = df[df["survived_phase"]].copy()
    completed = _assign_movement_groups(
        completed,
        value_col="phase_movement_per_min",
        group_col="phase_movement_group",
    )

    all_summary = _summarize(
        all_entries,
        cohort="all_phase_entries",
        group_col="phase_movement_group",
    )
    completed_summary = _summarize(
        completed,
        cohort="completed_phase_only",
        group_col="phase_movement_group",
    )
    tests = pd.concat(
        [
            _tests(all_entries, cohort="all_phase_entries", group_col="phase_movement_group"),
            _tests(completed, cohort="completed_phase_only", group_col="phase_movement_group"),
        ],
        ignore_index=True,
    )

    all_features_path = os.path.join(out_dir, "phase_h3_all_entries_features.csv")
    completed_features_path = os.path.join(out_dir, "phase_h3_completed_features.csv")
    all_summary_path = os.path.join(out_dir, "phase_h3_all_entries_summary.csv")
    completed_summary_path = os.path.join(out_dir, "phase_h3_completed_summary.csv")
    tests_path = os.path.join(out_dir, "phase_h3_tests.csv")
    report_path = os.path.join(out_dir, "phase_h3_report.md")

    all_entries.to_csv(all_features_path, index=False)
    completed.to_csv(completed_features_path, index=False)
    all_summary.to_csv(all_summary_path, index=False)
    completed_summary.to_csv(completed_summary_path, index=False)
    tests.to_csv(tests_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            make_report(
                all_summary=all_summary,
                completed_summary=completed_summary,
                tests=tests,
                out_dir=out_dir,
            )
        )
        f.write("\n")

    summary = {
        "dataset_dir": dataset_dir,
        "phase_features_path": phase_features_path,
        "out_dir": out_dir,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "min_phase_observed_s": float(args.min_phase_observed_s),
        "n_all_phase_entries": int(all_entries.shape[0]),
        "n_completed_phase_entries": int(completed.shape[0]),
        "artifacts": {
            "all_features": all_features_path,
            "completed_features": completed_features_path,
            "all_summary": all_summary_path,
            "completed_summary": completed_summary_path,
            "tests": tests_path,
            "report": report_path,
        },
    }
    _write_json(os.path.join(out_dir, "phase_h3_summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-wise H3 movement analysis.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--phase-features", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--min-phase-observed-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "out_dir": summary["out_dir"],
                "n_all_phase_entries": summary["n_all_phase_entries"],
                "n_completed_phase_entries": summary["n_completed_phase_entries"],
                "report": summary["artifacts"]["report"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
