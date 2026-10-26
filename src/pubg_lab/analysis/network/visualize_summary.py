"""
여러 매치 크로스 분석 HTML
===========================
- Centrality ↔ final_rank / time_survived Spearman 상관
- 매치별 평균 centrality 분포 (box/violin)
- 매치간 degree distribution 비교 (log-log)
- 시계열 지표 평균±표준편차 밴드

단일 HTML 파일로 출력.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html
from scipy import stats as scipy_stats

from pubg_lab.analysis.network.config import CENTRALITY_MEASURES


_MEASURE_LABEL = {
    "degree": "Degree",
    "closeness": "Closeness",
    "betweenness": "Betweenness",
    "eigenvector": "Eigenvector",
    "pagerank": "PageRank",
}


# ============================================================
# 1. 플레이어 × 매치 데이터 집계
# ============================================================


def aggregate_player_centrality(match_meta, match_result):
    """
    한 매치 내 플레이어별 centrality를 "생존 구간 평균" 으로 요약.

    생존 구간 평균: 해당 플레이어가 노드로 등장한 프레임들만 평균.

    Returns:
        list of dict: {account_id, team_id, final_rank, time_survived, survived,
                       <measure>_mean, <measure>_max}
    """
    death_times = match_meta["death_times"]
    player_team = match_meta["player_team"]
    team_rank = match_meta["team_rank"]

    # account_id → list of per-frame centrality values (per measure)
    per_player = {}  # aid -> measure -> [values]

    for fr in match_result["frames"]:
        aids = fr["account_ids"]
        cmap = fr["metrics"]["centrality"]
        for i, aid in enumerate(aids):
            if aid not in per_player:
                per_player[aid] = {m: [] for m in CENTRALITY_MEASURES}
            for m in CENTRALITY_MEASURES:
                v = cmap.get(m, {}).get(i, None)
                if v is not None:
                    per_player[aid][m].append(float(v))

    # match_duration: 마지막 스냅샷 시각
    match_duration = match_meta["snapshot_times"][-1] if match_meta["snapshot_times"] else 1800.0

    rows = []
    for aid, mvals in per_player.items():
        tid = player_team.get(aid)
        rank = team_rank.get(tid) if tid is not None else None
        elim = death_times.get(aid, None)
        time_survived = float(elim) if elim is not None else float(match_duration)

        row = {
            "match_id": match_meta["match_id"],
            "account_id": aid,
            "team_id": tid,
            "final_rank": rank,
            "time_survived": time_survived,
            "survived": elim is None,
        }
        for m in CENTRALITY_MEASURES:
            vs = mvals[m]
            row[f"{m}_mean"] = float(np.mean(vs)) if vs else 0.0
            row[f"{m}_max"] = float(np.max(vs)) if vs else 0.0
        rows.append(row)
    return rows


# ============================================================
# 2. Spearman 상관
# ============================================================


def compute_correlations(player_rows):
    """
    측정치 × 결과변수 Spearman 상관.

    결과변수: final_rank (낮을수록 좋음 → 음의 상관 기대),
              time_survived (길수록 좋음 → 양의 상관 기대).

    Returns:
        dict: {measure: {"final_rank": (rho, p), "time_survived": (rho, p)}}
    """
    out = {}
    for m in CENTRALITY_MEASURES:
        xs = np.array([r[f"{m}_mean"] for r in player_rows], dtype=float)
        ranks = np.array(
            [r["final_rank"] for r in player_rows if r["final_rank"] is not None], dtype=float
        )
        ts = np.array([r["time_survived"] for r in player_rows], dtype=float)
        xs_for_rank = np.array(
            [r[f"{m}_mean"] for r in player_rows if r["final_rank"] is not None], dtype=float
        )

        if len(ranks) > 3 and xs_for_rank.std() > 0 and ranks.std() > 0:
            rho_r, p_r = scipy_stats.spearmanr(xs_for_rank, ranks)
        else:
            rho_r, p_r = np.nan, np.nan
        if len(ts) > 3 and xs.std() > 0 and ts.std() > 0:
            rho_t, p_t = scipy_stats.spearmanr(xs, ts)
        else:
            rho_t, p_t = np.nan, np.nan

        out[m] = {
            "final_rank": (
                float(rho_r) if rho_r == rho_r else np.nan,
                float(p_r) if p_r == p_r else np.nan,
            ),
            "time_survived": (
                float(rho_t) if rho_t == rho_t else np.nan,
                float(p_t) if p_t == p_t else np.nan,
            ),
        }
    return out


# ============================================================
# 3. Figure 조립
# ============================================================


def _corr_table_figure(corr):
    """매치 통합 상관표를 Plotly Table 로."""
    rows_m = []
    rows_vs_rank = []
    rows_vs_time = []
    for m in CENTRALITY_MEASURES:
        rr, pr = corr[m]["final_rank"]
        rt, pt = corr[m]["time_survived"]
        rows_m.append(_MEASURE_LABEL[m])
        rows_vs_rank.append("—" if not np.isfinite(rr) else f"{rr:+.3f} (p={pr:.3g})")
        rows_vs_time.append("—" if not np.isfinite(rt) else f"{rt:+.3f} (p={pt:.3g})")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Centrality",
                        "ρ vs final_rank<br><sub>(낮을수록 좋음)</sub>",
                        "ρ vs time_survived<br><sub>(길수록 좋음)</sub>",
                    ],
                    fill_color="#2c3e50",
                    font=dict(color="white", size=13),
                    align="left",
                    height=40,
                ),
                cells=dict(
                    values=[rows_m, rows_vs_rank, rows_vs_time],
                    fill_color="#f8f8fa",
                    align="left",
                    font=dict(size=12),
                    height=26,
                ),
            )
        ]
    )
    fig.update_layout(
        title="Centrality ↔ 결과 Spearman 상관 (모든 매치 · 플레이어 통합)",
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def _scatter_rank_vs_centrality(player_rows, measure):
    """final_rank ↔ centrality 산점도."""
    xs = [r[f"{measure}_mean"] for r in player_rows if r["final_rank"] is not None]
    ys = [r["final_rank"] for r in player_rows if r["final_rank"] is not None]
    mids = [r["match_id"][:6] for r in player_rows if r["final_rank"] is not None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=7,
                opacity=0.55,
                color=ys,
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="final_rank"),
            ),
            text=mids,
            hovertemplate=f"{_MEASURE_LABEL[measure]}=%{{x:.3f}}<br>rank=%{{y}}<br>match=%{{text}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{_MEASURE_LABEL[measure]} (mean over alive frames) vs final_rank",
        xaxis_title=f"{_MEASURE_LABEL[measure]} — player mean",
        yaxis_title="final_rank (1=win)",
        yaxis_autorange="reversed",
        height=420,
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def _degree_dist_comparison(all_frames_by_match):
    """매치별 degree distribution 을 log-log로 overlay."""
    fig = go.Figure()
    # 각 매치에 대해 중간 스냅샷 1개씩
    for i, (mid, frames) in enumerate(all_frames_by_match.items()):
        if not frames:
            continue
        mid_frame = frames[len(frames) // 2]
        dd = mid_frame["metrics"]["degree_dist"]
        if not dd["ks"]:
            continue
        fig.add_trace(
            go.Scatter(
                x=dd["ks"],
                y=dd["probs"],
                mode="markers+lines",
                name=f"match {mid[:8]} (mid)",
                line=dict(width=1.2),
                marker=dict(size=7, opacity=0.75),
            )
        )
    fig.update_layout(
        title="Degree Distribution P(k) — 매치별 중간 스냅샷 비교",
        xaxis=dict(type="log", title="degree k", gridcolor="#eee"),
        yaxis=dict(type="log", title="P(k)", gridcolor="#eee"),
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def _timeseries_band(all_timeseries, key, label):
    """매치별 시계열을 mean ± std 로 밴드."""
    # 길이가 다르므로 최소 길이 기준 정렬
    if not all_timeseries:
        return go.Figure()
    min_len = min(len(ts[key]) for ts in all_timeseries)
    if min_len == 0:
        return go.Figure()
    arr = np.array([ts[key][:min_len] for ts in all_timeseries], dtype=float)
    elapsed = all_timeseries[0]["elapsed"][:min_len]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=elapsed,
            y=mean + std,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=elapsed,
            y=mean - std,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(44, 130, 201, 0.18)",
            line=dict(width=0),
            name="±1σ",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=elapsed,
            y=mean,
            mode="lines",
            line=dict(color="rgb(44, 130, 201)", width=2.2),
            name="mean",
        )
    )
    fig.update_layout(
        title=f"{label} — 매치 간 평균 ± 표준편차",
        xaxis_title="elapsed (sec)",
        yaxis_title=label,
        template="plotly_white",
        height=320,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


# ============================================================
# 4. HTML 저장
# ============================================================


def write_summary_html(all_match_meta, all_match_result, out_path):
    """
    Parameters:
        all_match_meta: list of match_meta dict
        all_match_result: list of match_result dict (compute_match_metrics 출력)
    """
    # 1) 플레이어 통합 집계
    player_rows_all = []
    for mm, mr in zip(all_match_meta, all_match_result):
        player_rows_all.extend(aggregate_player_centrality(mm, mr))

    # 2) 상관
    corr = compute_correlations(player_rows_all)

    # 3) figure들
    fig_corr = _corr_table_figure(corr)
    fig_scatters = {m: _scatter_rank_vs_centrality(player_rows_all, m) for m in CENTRALITY_MEASURES}
    fig_degdist = _degree_dist_comparison(
        {mm["match_id"]: mr["frames"] for mm, mr in zip(all_match_meta, all_match_result)}
    )
    fig_ts_cc = _timeseries_band(
        [mr["timeseries"] for mr in all_match_result],
        "avg_clustering",
        "Avg clustering coefficient C̄",
    )
    fig_ts_apl = _timeseries_band(
        [mr["timeseries"] for mr in all_match_result],
        "avg_path_length",
        "Avg shortest-path length",
    )
    fig_ts_dens = _timeseries_band(
        [mr["timeseries"] for mr in all_match_result],
        "density",
        "Density",
    )

    # 4) HTML
    def embed(fig, div_id, first=False):
        return to_html(
            fig, include_plotlyjs=("cdn" if first else False), full_html=False, div_id=div_id
        )

    match_list_html = "\n".join(
        f"<li><code>{mm['match_id']}</code> &nbsp;·&nbsp; rank_count={len(mm['team_rank'])} "
        f"&nbsp;·&nbsp; snapshots={len(mm['snapshot_times'])}</li>"
        for mm in all_match_meta
    )

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>PUBG Network Analysis — Summary</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f3f4f6; margin: 0; padding: 18px; color: #222; }}
  .wrap {{ max-width: 1200px; margin: 0 auto; }}
  .card {{ background: white; border-radius: 10px; padding: 16px 18px;
           margin: 0 0 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  h1 {{ margin: 0 0 4px; font-size: 22px; }}
  h2 {{ margin: 18px 0 8px; font-size: 16px; color: #333; }}
  p, li  {{ color: #555; font-size: 13px; }}
  ul {{ margin: 4px 0 0 18px; padding: 0; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  .tabs {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
  .tab-btn {{ padding: 6px 12px; border: 1px solid #ccc; background: white;
              cursor: pointer; border-radius: 6px; font-size: 12px; }}
  .tab-btn.active {{ background: #2c3e50; color: white; border-color: #2c3e50; }}
  .sc-panel {{ display: none; }}
  .sc-panel.active {{ display: block; }}
</style>
</head><body>
<div class="wrap">
  <div class="card">
    <h1>📊 PUBG Network Analysis — Cross-Match Summary</h1>
    <p>Chapter 3 지표(degree / closeness / betweenness / eigenvector / PageRank)를
       매치별로 계산하고, <b>centrality ↔ 생존 결과</b> 상관을 집계합니다.</p>
    <p><b>매치 {len(all_match_meta)}건 · 플레이어-매치 샘플 {len(player_rows_all)}건</b></p>
    <ul>{match_list_html}</ul>
  </div>

  <div class="card">
    <h2>1. Centrality ↔ 결과 Spearman 상관</h2>
    {embed(fig_corr, "corr-table", first=True)}
    <p style="font-size:12px; color:#888">
      가설: centrality(사회적/공간적 중심성)가 높은 플레이어는 교전 노출이 많아
      <em>일찍 죽을 수도</em>(부호 -), <em>오래 살아남을 수도</em>(부호 +) 있음.
      실데이터의 부호와 p-value를 보고 해석.
    </p>
  </div>

  <div class="card">
    <h2>2. final_rank vs centrality — 산점도</h2>
    <div class="tabs">
"""
    for i, m in enumerate(CENTRALITY_MEASURES):
        cls = "tab-btn active" if i == 0 else "tab-btn"
        html += f'<button class="{cls}" onclick="showScatter({i})">{_MEASURE_LABEL[m]}</button>'
    html += "</div>\n"
    for i, m in enumerate(CENTRALITY_MEASURES):
        cls = "sc-panel active" if i == 0 else "sc-panel"
        html += f'<div class="{cls}" id="sc-panel-{i}">'
        html += embed(fig_scatters[m], f"scatter-{i}")
        html += "</div>"

    html += f"""
  </div>

  <div class="card">
    <h2>3. Degree Distribution P(k) 비교</h2>
    {embed(fig_degdist, "degdist")}
    <p style="font-size:12px; color:#888">
      기울기가 완만한 선형(log-log) 형태라면 scale-free (배틀로열의 소수 지배 구조),
      급격히 꺾이면 homogeneous.
    </p>
  </div>

  <div class="card">
    <h2>4. 시계열 — 매치 간 평균 ± 표준편차</h2>
    <div class="grid2">
      <div>{embed(fig_ts_cc, "ts-cc")}</div>
      <div>{embed(fig_ts_dens, "ts-dens")}</div>
    </div>
    <div>{embed(fig_ts_apl, "ts-apl")}</div>
    <p style="font-size:12px; color:#888">
      자기장이 좁아지면 density↑ · avg path length↓ 가 자연스러움.
      실제 데이터와 일치하는지 확인.
    </p>
  </div>
</div>
<script>
function showScatter(i) {{
  document.querySelectorAll('.tab-btn').forEach((b, bi) => {{
    b.classList.toggle('active', bi === i);
  }});
  document.querySelectorAll('.sc-panel').forEach((p, pi) => {{
    p.classList.toggle('active', pi === i);
  }});
  const target = document.querySelector('#sc-panel-' + i + ' .plotly-graph-div');
  if (target && window.Plotly) {{ window.Plotly.Plots.resize(target); }}
}}
</script>
</body></html>
"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
