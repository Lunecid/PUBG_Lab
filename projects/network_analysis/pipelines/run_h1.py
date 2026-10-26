"""
H1 CLI · Bias-controlled combat engagement vs placement
========================================================
N 개 매치에 대해 팀 네트워크를 빌드하고 네 축(raw / intensity / early / partial)
모두에서 degree ↔ rank 관계를 검증한 뒤 HTML 리포트를 생성.

Usage:
  python3 -m network_analysis.run_h1
  python3 -m network_analysis.run_h1 --n 10
  python3 -m network_analysis.run_h1 --n 5 --window-sec 240
  python3 -m network_analysis.run_h1 --n 5 --skip-early
"""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import os
import sys
import time
import argparse
from datetime import datetime

from pubg_lab.analysis.network.config.config import (
    DEFAULT_MAP,
    DEFAULT_MODE,
    DEFAULT_N_MATCHES,
    OUTPUT_ROOT,
)
from pubg_lab.analysis.network.data.data_loader import list_matches, build_match_data
from pubg_lab.analysis.network.features.team_centrality import MEASURES
from pubg_lab.analysis.network.legacy.h1_analysis import (
    collect_all_matches,
    correlations_for_scatter,
    fourway_correlation_table,
    quartile_group_compare,
    generate_h1_conclusion,
    FOURWAY_FEATURES,
)
from projects.network_analysis.reporting.h1_report import write_h1_report


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _print_fourway(table, verdict):
    def _cell(r, p=None):
        if r is None or r != r:  # NaN
            return "   —   "
        star = ""
        if p is not None and p == p and p < 0.05:
            star = "*" if p >= 0.01 else ("**" if p >= 0.001 else "***")
        return f"{r:+.3f}{star:<4}"

    print(f"\n{'feature':<22} {'raw':>10} {'intensity':>10} {'early':>10} {'partial':>10}  n")
    print("-" * 76)
    for row in table:
        raw = _cell(row["raw_r"], row["raw_p"])
        inte = _cell(row["int_r"], row["int_p"])
        early = _cell(row["early_r"], row["early_p"])
        part = _cell(row["partial_r"])
        print(
            f"{row['label'][:21]:<22} {raw:>10} {inte:>10} {early:>10} {part:>10}  {row['raw_n']}"
        )
    print(f"\nverdict: {verdict.upper()}")


def main():
    ap = argparse.ArgumentParser(
        description="H1: bias-controlled combat engagement vs placement",
    )
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--mode", default=DEFAULT_MODE)
    ap.add_argument("--n", type=int, default=DEFAULT_N_MATCHES, help="number of matches to analyze")
    ap.add_argument(
        "--weight", default="dmg", choices=["dmg", "count", "binary"], help="edge weight mode"
    )
    ap.add_argument(
        "--window-sec", type=int, default=300, help="early-game window in seconds (default 300)"
    )
    ap.add_argument("--skip-early", action="store_true", help="skip early-game analysis (faster)")
    ap.add_argument("--out", default=None, help="report HTML output path")
    args = ap.parse_args()

    run_id = _ts()
    out_path = args.out or os.path.join(
        OUTPUT_ROOT,
        f"h1_report_{args.map}_{args.mode}_{run_id}.html",
    )
    early_window = None if args.skip_early else args.window_sec

    print(
        f"[h1] map={args.map} mode={args.mode} n={args.n} "
        f"weight={args.weight} early_window={early_window}s"
    )
    print(f"[h1] out: {out_path}\n")

    pts = list_matches(args.map, args.mode, limit=args.n)
    if not pts:
        print(f"[!] no .pt under data/graphs/{args.map}/{args.mode}")
        sys.exit(1)
    print(f"[h1] matches found: {len(pts)}")

    t0 = time.time()
    collected = collect_all_matches(
        pts,
        build_match_data_fn=build_match_data,
        weight_mode=args.weight,
        early_window_sec=early_window,
        verbose=True,
    )
    print(
        f"\n[h1] collection done: {time.time() - t0:.1f}s  "
        f"(rows={len(collected['rows'])}, "
        f"matches={len(collected['match_ids'])})"
    )

    # Four-way correlation
    fourway_table = fourway_correlation_table(collected["rows"])

    # Quartile groupwise (raw features)
    quartile_keys = ["degree", "combat_events", "dmg_dealt"] + [f"c_{m}" for m in MEASURES]
    quartile_results = [quartile_group_compare(collected["rows"], k) for k in quartile_keys]

    # Scatter correlations (subset for plots)
    scatter_keys = [
        "degree",
        "combat_events",
        "dmg_dealt",
        "c_degree",
        "c_pagerank",
        "c_betweenness",
    ]
    corr_scatter = correlations_for_scatter(collected["rows"], scatter_keys)

    verdict, conclusion_lines, hero = generate_h1_conclusion(
        fourway_table,
        quartile_results,
        n_matches=len(collected["match_ids"]),
        n_teams_total=len(collected["rows"]),
        early_window_sec=early_window,
    )

    _print_fourway(fourway_table, verdict)
    print("\nFindings:")
    for line in conclusion_lines:
        print(f"  · {line}")

    analysis = {
        "fourway_table": fourway_table,
        "corr_for_scatter": corr_scatter,
        "quartile_results": quartile_results,
        "conclusion_lines": conclusion_lines,
        "verdict": verdict,
        "hero": hero,
    }
    write_h1_report(collected, analysis, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n[h1] report: {out_path} ({size_mb:.2f} MB)")
    return out_path


if __name__ == "__main__":
    main()
