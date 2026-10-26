"""
Network Analysis 오케스트레이터 (v2)
====================================
⚠ DEPRECATED — canonical 경로는 run_v2.py (H1/H2/H3 분석 + webapp 데이터 export).
이 모듈은 초기 per-match HTML 탐색용으로만 남겨둔다 (CODE_REVIEW E17).

여러 매치를 순회하며 per-match 인터랙티브 HTML 생성.
교전 엣지는 DB telem_damage로부터 attacker→victim 로 빌드.

사용:
  cd /Users/lunecid/project/PUBG
  python3 -m network_analysis.run              # 기본 3 매치
  python3 -m network_analysis.run --n 5
  python3 -m network_analysis.run --stride 2   # 스냅샷 절반
  python3 -m network_analysis.run --no-db      # DB 스킵 (combat 엣지 없음)
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
    TOP_K,
    NARRATIVE_ENABLED,
    NARRATIVE_TREND_WINDOW,
    VIEW_MODE_DEFAULT,
)
from pubg_lab.analysis.network.data.data_loader import list_matches, build_match_data
from pubg_lab.analysis.network.features.metrics import compute_all_frames
from pubg_lab.analysis.network.reporting.visualize_match import write_match_html


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    ap = argparse.ArgumentParser(description="PUBG Network Analysis v2 (combat edges from DB)")
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--mode", default=DEFAULT_MODE)
    ap.add_argument("--n", type=int, default=DEFAULT_N_MATCHES)
    ap.add_argument("--stride", type=int, default=1, help="스냅샷 샘플링 간격 (1=10초마다, 2=20초)")
    ap.add_argument(
        "--no-db", action="store_true", help="DB에서 damage 이벤트 로드 스킵 (combat 엣지 없음)"
    )
    ap.add_argument(
        "--measure",
        default="pagerank",
        choices=["degree", "closeness", "betweenness", "eigenvector", "pagerank"],
    )
    ap.add_argument(
        "--view-mode",
        default=VIEW_MODE_DEFAULT,
        choices=["team", "louvain", "spectral", "centrality"],
        help="HTML 초기 노드 색상 기준",
    )
    ap.add_argument("--no-narrative", action="store_true", help="프레임별 자연어 해설 생성 스킵")
    ap.add_argument(
        "--trend-window",
        type=int,
        default=NARRATIVE_TREND_WINDOW,
        help="해설의 추세 판단에 쓸 lookback 프레임 수",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_id = _ts()
    out_dir = args.out or os.path.join(OUTPUT_ROOT, f"{args.map}_{args.mode}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[run] map={args.map}  mode={args.mode}  n={args.n}  "
        f"stride={args.stride}  db={'off' if args.no_db else 'on'}"
    )
    print(f"[run] output: {out_dir}")

    pts = list_matches(args.map, args.mode, limit=args.n)
    if not pts:
        print(f"[!] no .pt under data/graphs/{args.map}/{args.mode}")
        sys.exit(1)
    print(f"[run] {len(pts)} matches\n")

    per_match_htmls = []
    for i, pt in enumerate(pts, 1):
        t0 = time.time()
        name = os.path.basename(pt)
        print(f"[{i}/{len(pts)}] {name}")

        t1 = time.time()
        md = build_match_data(pt, fetch_damage=not args.no_db, stride=args.stride)
        print(
            f"    load+build: {time.time() - t1:.1f}s  "
            f"(frames={len(md['frames'])}, players={md['static']['n_players']}, "
            f"teams={md['static']['n_teams']})"
        )

        t2 = time.time()
        compute_all_frames(
            md,
            top_k=TOP_K,
            narrative_enabled=(NARRATIVE_ENABLED and not args.no_narrative),
            trend_window=args.trend_window,
        )
        print(f"    metrics:    {time.time() - t2:.1f}s")

        out_html = os.path.join(out_dir, f"match_{md['static']['match_id'][:12]}.html")
        write_match_html(
            md,
            out_html,
            default_measure=args.measure,
            top_k=TOP_K,
            default_view_mode=args.view_mode,
        )
        size_mb = os.path.getsize(out_html) / 1e6
        print(
            f"    → {os.path.basename(out_html)}  ({size_mb:.2f} MB, total {time.time() - t0:.1f}s)\n"
        )
        per_match_htmls.append(out_html)

    # index.html
    idx_path = os.path.join(out_dir, "index.html")
    items = "\n".join(
        f'<li><a href="{os.path.basename(p)}">{os.path.basename(p)}</a></li>'
        for p in per_match_htmls
    )
    with open(idx_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Network Analysis Index</title>
<style>body{{font-family:-apple-system,sans-serif;padding:22px;background:#f3f4f6;}}
.card{{background:white;border-radius:10px;padding:16px 20px;max-width:900px;margin:0 auto;
box-shadow:0 2px 8px rgba(0,0,0,0.06);}}
h1{{margin:0 0 10px;}}a{{color:#2c3e50;}}li{{margin:4px 0;}}</style></head>
<body><div class="card">
<h1>🕸️ PUBG Network Analysis Index</h1>
<p>{args.map} / {args.mode}, {len(pts)} 매치 · combat edges = {"OFF" if args.no_db else "ON (telem_damage)"}</p>
<h3>매치별 인터랙티브 HTML</h3>
<ul>{items}</ul>
</div></body></html>
""")
    print(f"[run] index → {idx_path}")
    print("\n완료.")
    return out_dir


if __name__ == "__main__":
    main()
