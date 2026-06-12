"""
Unified network analysis report (H1 · H2 · H3)
===============================================
세 가설을 좌측 메뉴 네비게이션을 갖춘 하나의 HTML 페이지에 통합.
각 지표마다 PUBG 도메인 해석 박스를 붙여 "숫자 나열" 을 피한다.

섹션 구성:
  · 개요 (세 가설 판정 badge + 한 줄 요약)
  · 네트워크 기초 (그래프 구성 + 지표 정의 + 해석)
  · 팀 중심성 (5종 수식 + 의미 + 사례)
  · 가설 1 · 교전 참여도 ↔ 순위
  · 가설 2 · 교전 밀집도 ↔ 순위
  · 가설 3 · 외곽 vs 중심 생존율
"""

from __future__ import annotations
import base64
import html as html_mod
import io
import os
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json as _json

from network_analysis.team_centrality import MEASURES
from network_analysis.hypotheses import (
    density_interpretation, clustering_interpretation,
    diameter_interpretation, modularity_interpretation,
    pagerank_interpretation, betweenness_interpretation,
    position_label_kr,
)
from network_analysis.legacy.h2_analysis import SIZE_BUCKET_LABEL_KR
from network_analysis.legacy.h3_analysis import POSITION_ORDER, POSITION_LABEL_KR


# ============================================================
# Plot palette
# ============================================================

_PAPER_BG  = "#fefcf7"
_INK       = "#1a1a1a"
_MUTED     = "#6b6558"
_RULE      = "#d8d1c2"
_ACCENT    = "#b02a2a"
_POSITIVE  = "#2d5f3f"
_GRID      = "#ebe4d3"

plt.rcParams.update({
    "axes.facecolor":   _PAPER_BG,
    "figure.facecolor": _PAPER_BG,
    "axes.edgecolor":   _RULE,
    "axes.labelcolor":  _INK,
    "text.color":       _INK,
    "xtick.color":      _MUTED,
    "ytick.color":      _MUTED,
    "grid.color":       _GRID,
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.titleweight": "regular",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _fig_to_b64(fig, dpi: int = 140) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=_PAPER_BG)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _h(s) -> str:
    return html_mod.escape(str(s))


def _fmt(v, d=3):
    if v is None: return "—"
    try:
        vf = float(v)
        if not np.isfinite(vf): return "—"
        return f"{vf:.{d}f}"
    except Exception:
        return "—"


def _fmt_r(v, d=3):
    if v is None: return "—"
    try:
        vf = float(v)
        if not np.isfinite(vf): return "—"
        return f"{vf:+.{d}f}"
    except Exception:
        return "—"


def _fmt_p(v):
    if v is None: return "—"
    try:
        vf = float(v)
        if not np.isfinite(vf): return "—"
        return "<0.0001" if vf < 0.0001 else f"{vf:.4f}"
    except Exception:
        return "—"


def _sig_star(p):
    try:
        pf = float(p)
    except Exception:
        return ""
    if not np.isfinite(pf): return ""
    return "***" if pf < 0.001 else "**" if pf < 0.01 else "*" if pf < 0.05 else ""


# ============================================================
# Charts — shared
# ============================================================

def chart_adjacency(A: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(A, cmap="pink_r", aspect="auto")
    ax.set_xlabel("team index"); ax.set_ylabel("team index")
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("cumulative damage", color=_MUTED)
    cb.outline.set_edgecolor(_RULE)
    return _fig_to_b64(fig)


def chart_pk(all_pks) -> str:
    cnt = Counter(); total = 0
    for ks, probs in all_pks:
        for k, p in zip(ks, probs):
            cnt[k] += p
        total += 1
    if total == 0 or not cnt:
        cnt[0] = 1; total = 1
    ks = sorted(cnt.keys())
    avg_p = [cnt[k] / total for k in ks]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.bar(ks, avg_p, color=_INK, width=0.7)
    ax.set_xlabel("degree  k")
    ax.set_ylabel("P(k)  ·  average")
    ax.grid(axis="y", alpha=0.4)
    return _fig_to_b64(fig)


def chart_globals_box(globals_list) -> str:
    dens = [g["density"] for g in globals_list]
    cc = [g["avg_clustering"] for g in globals_list]
    apl = [g["avg_path_length"] for g in globals_list]
    dm = [g["diameter"] for g in globals_list]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.4))
    data = [dens, cc, apl, dm]
    labels = ["density", "avg C̄", "avg path", "diameter"]
    for ax, d, lbl in zip(axes, data, labels):
        ax.boxplot(d, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#f0e6cc", edgecolor=_INK, linewidth=0.8),
                   medianprops=dict(color=_ACCENT, linewidth=1.4),
                   whiskerprops=dict(color=_INK, linewidth=0.8),
                   capprops=dict(color=_INK, linewidth=0.8),
                   flierprops=dict(markerfacecolor=_MUTED,
                                   markeredgecolor=_MUTED, markersize=3))
        ax.set_title(lbl)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.35)
    return _fig_to_b64(fig)


def chart_scatter(rows, feature, label, corr) -> str:
    xs, ys = [], []
    for r in rows:
        x = r.get(feature); y = r.get("final_rank")
        if x is None or y is None: continue
        xs.append(float(x)); ys.append(float(y))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.scatter(xs, ys, s=12, alpha=0.45, color=_INK, edgecolor="none")
    if np.isfinite(corr.get("lin_slope", np.nan)):
        xr = np.array([min(xs), max(xs)])
        yr = corr["lin_intercept"] + corr["lin_slope"] * xr
        ax.plot(xr, yr, color=_ACCENT, linewidth=1.0,
                label=f"R² = {corr['lin_r2']:.3f}")
        ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_xlabel(label); ax.set_ylabel("final_rank  (lower is better)")
    r = corr["pearson_r"]; p = corr["pearson_p"]
    r_txt = "—" if not np.isfinite(r) else f"{r:+.3f}"
    ax.set_title(f"{label}  ·  r = {r_txt}  ·  p = {_fmt_p(p)}")
    ax.grid(alpha=0.35); ax.invert_yaxis()
    return _fig_to_b64(fig)


def chart_louvain_Q(communities) -> str:
    qs = [c["louvain"]["modularity_q"] for c in communities if c.get("louvain")]
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    if qs:
        ax.hist(qs, bins=min(10, max(3, len(qs)//2)),
                color="#3a3a3a", edgecolor=_INK, linewidth=0.5)
    ax.axvline(0.3, color=_ACCENT, linewidth=1.2, linestyle="--",
               label="Q = 0.3 threshold")
    ax.set_xlabel("modularity Q (Louvain)")
    ax.set_ylabel("# matches")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.35)
    return _fig_to_b64(fig)


def chart_bucket_bar(labels, means, ns, title, ylabel) -> str:
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=_INK, width=0.55)
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f" n={n}", ha="center", va="bottom",
                fontsize=8, color=_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.35)
    return _fig_to_b64(fig)


# ============================================================
# Shared UI helpers
# ============================================================

def interp_box(body_html: str, *, tone: str = "default") -> str:
    cls = "interp" if tone == "default" else f"interp {tone}"
    return (f'<div class="{cls}">'
            f'<div class="interp-label">이 수치가 말하는 것</div>'
            f'<div class="interp-body">{body_html}</div></div>')


def verdict_badge(verdict: str) -> str:
    mapping = {
        "confirmed":       ("가설 지지", "ok"),
        "rejected":        ("기각 — 편향 기인", "bad"),
        "inconclusive":    ("결론 유보", "mid"),
        "core_better":     ("중심 팀 우위", "ok"),
        "periphery_better":("외곽 팀 우위", "ok"),
        "isolated_better": ("파밍형 우위", "ok"),
        "dense_better":    ("진영 참여형 우위", "ok"),
        "no_difference":   ("차이 없음", "mid"),
    }
    label, cls = mapping.get(verdict, ("결론 유보", "mid"))
    return f'<span class="badge {cls}">{label}</span>'


# ============================================================
# Section renderers
# ============================================================

# ============================================================
# Match explorer (interactive D3 force-directed graph)
# ============================================================

def render_explorer(matches: list[dict]) -> str:
    """매치 선택 + 3가지 뷰 (degree/community/position) 인터랙티브 그래프."""
    if not matches:
        return ""

    # 드롭다운 옵션
    options = "\n".join(
        f'<option value="{i}">매치 {m["match_id"]} · '
        f'|V|={m["n_nodes"]} · |E|={m["n_edges"]} · Q={m["modularity"]:.2f}</option>'
        for i, m in enumerate(matches)
    )

    matches_json = _json.dumps(matches, separators=(",", ":"),
                                ensure_ascii=False)

    interp_html = interp_box(
        '<b>degree</b> · 원 크기 = 이 팀이 교전한 상대 팀 수. 선 두께 = '
        '교전 이벤트 수. <b>community</b> · 같은 Louvain 커뮤니티는 같은 '
        '색 — 지리적으로 같은 전투 클러스터에 속했음을 뜻함. '
        '<b>position</b> · 색 = 네트워크 위치. '
        '<span class="sw pos-core"></span>core (중개자) · '
        '<span class="sw pos-mid"></span>mid · '
        '<span class="sw pos-peri"></span>peripheral (외곽) · '
        '<span class="sw pos-iso"></span>isolated (교전 0회, 파밍형 추정).'
    )

    return f"""
<section id="explorer">
  <div class="section-label">탐색기</div>
  <h2>매치 탐색기</h2>
  <p class="lede">
    매치를 선택하고 뷰 모드를 바꿔가며 팀 간 교전 구조를 살펴보세요.
    원을 드래그하면 레이아웃이 다시 안정화됩니다. 스크롤/제스처로
    확대·축소, 배경을 드래그해 이동할 수 있습니다.
  </p>

  {interp_html}

  <div class="explorer-ctrl">
    <label class="ctrl-group">
      <span class="ctrl-label">매치</span>
      <select id="match-select">
        {options}
      </select>
    </label>
    <div class="view-buttons" role="tablist">
      <button data-view="degree" class="active">degree</button>
      <button data-view="community">community</button>
      <button data-view="position">position</button>
    </div>
    <div class="level-buttons" role="tablist">
      <button data-level="team" class="active">Team view</button>
      <button data-level="player">Player view</button>
    </div>
    <div class="explorer-stats" id="xp-stats"></div>
  </div>

  <div class="explorer-legend" id="xp-legend"></div>

  <div class="explorer-canvas">
    <svg id="xp-graph" viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid meet">
      <defs>
        <clipPath id="xp-clip"><rect x="0" y="0" width="1000" height="1000"/></clipPath>
        <marker id="xp-arrow" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#b02a2a"/>
        </marker>
      </defs>
      <rect id="xp-bg" x="0" y="0" width="1000" height="1000" fill="#13161b"/>
      <g id="xp-world" clip-path="url(#xp-clip)">
        <image id="xp-map" x="0" y="0" width="1000" height="1000"
               preserveAspectRatio="none" opacity="0.62"/>
        <rect id="xp-map-tint" x="0" y="0" width="1000" height="1000"
              fill="#0f1115" opacity="0.18" pointer-events="none"/>
        <g id="xp-grid"></g>
        <g id="xp-zone"></g>
        <g id="xp-hulls"></g>
        <g id="xp-ally"></g>
        <g id="xp-combat"></g>
        <g id="xp-nodes"></g>
        <g id="xp-labels-team"></g>
        <g id="xp-labels-deg"></g>
      </g>
    </svg>
    <div class="xp-tooltip" id="xp-tip" style="display:none"></div>
    <div class="xp-zoom-ctrl">
      <button id="xp-zoom-in"  title="확대">+</button>
      <button id="xp-zoom-out" title="축소">−</button>
      <button id="xp-zoom-reset" title="초기화">⌂</button>
    </div>
  </div>

  <div class="xp-timeline">
    <button id="xp-play" aria-label="play/pause">▶</button>
    <input type="range" id="xp-slider" min="0" max="100" step="1" value="100"/>
    <span id="xp-time" class="xp-time">00:00 / 00:00</span>
    <label class="xp-speed">
      <span>속도</span>
      <select id="xp-speed">
        <option value="0.5">0.5×</option>
        <option value="1">1×</option>
        <option value="2" selected>2×</option>
        <option value="4">4×</option>
        <option value="8">8×</option>
      </select>
    </label>
    <span class="xp-frameinfo" id="xp-frameinfo"></span>
  </div>
</section>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const XP_MATCHES = {matches_json};
const XP_CPAL = ["#b02a2a","#1f4e8c","#2d5f3f","#a05a00","#6f4a88",
                 "#8a6a00","#007a8a","#7a3a5a","#3a6a3a","#8a3a3a",
                 "#5a5a7a","#7a5a3a","#3a5a7a","#5a7a3a","#7a3a7a"];
const XP_POSC = {{
  core:       "#b02a2a",
  mid:        "#3a3a3a",
  peripheral: "#b8b0a0",
  isolated:   "#d8d1c2"
}};

// 공용 map base64 / URL fallback — 첫 매치에만 embed, 이후 매치는 공유
const XP_SHARED_MAP = (XP_MATCHES[0] && XP_MATCHES[0].map_b64) || null;
const XP_SHARED_URLS = (XP_MATCHES[0] && XP_MATCHES[0].map_urls) || [];

let xpCurrentIdx  = 0;
let xpCurrentView = "degree";
let xpCurrentLevel = "team";   // "team" | "player"
let xpCurrentT    = 0;
let xpPlaying     = false;
let xpSpeed       = 2;
let xpRAF         = null;
let xpLastTs      = 0;
let xpNodeSel, xpAllySel, xpCombatSel, xpLabelTeamSel, xpLabelDegSel;
let xpNodeById    = new Map();   // current level: id -> node datum

const xpSvg = d3.select("#xp-graph");
const xpWorld = d3.select("#xp-world");
const xpMap = d3.select("#xp-map");
const xpMapTint = d3.select("#xp-map-tint");

const XP_VB = 1000;      // viewBox side
let xpArenaM = 8160;     // 업데이트: 매치 로드시
const xpAx = x => (x / xpArenaM) * XP_VB;
const xpAy = y => (y / xpArenaM) * XP_VB;

const xpZoom = d3.zoom()
  .scaleExtent([0.5, 12])
  .on("zoom", (e) => xpWorld.attr("transform", e.transform));
xpSvg.call(xpZoom);
xpSvg.on("dblclick.zoom", null);
xpSvg.on("dblclick", () => {{
  xpSvg.transition().duration(300).call(xpZoom.transform, d3.zoomIdentity);
}});

document.getElementById("xp-zoom-in").onclick  = () =>
  xpSvg.transition().duration(150).call(xpZoom.scaleBy, 1.4);
document.getElementById("xp-zoom-out").onclick = () =>
  xpSvg.transition().duration(150).call(xpZoom.scaleBy, 1/1.4);
document.getElementById("xp-zoom-reset").onclick = () =>
  xpSvg.transition().duration(250).call(xpZoom.transform, d3.zoomIdentity);

// 맵 이미지 fallback (b64 / URL chain)
function xpShowMap(href) {{
  xpMap.attr("href", href).style("display", "");
  xpMapTint.style("display", "");
}}
function xpHideMap() {{
  xpMap.style("display", "none");
  xpMapTint.style("display", "none");
}}
function xpTryMapUrls(queue) {{
  if (!queue.length) {{ xpHideMap(); return; }}
  const url = queue.shift();
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload  = () => xpShowMap(url);
  img.onerror = () => xpTryMapUrls(queue);
  img.src = url;
}}

// 격자 + km 라벨 — 맵 이미지 없을 때도 좌표 공간 감을 남김
function xpRenderGrid() {{
  const g = d3.select("#xp-grid");
  g.selectAll("*").remove();
  const km = Math.floor(xpArenaM / 1000);
  for (let i = 0; i <= km; i++) {{
    const p = xpAx(i * 1000);
    g.append("line")
      .attr("x1", p).attr("y1", 0).attr("x2", p).attr("y2", 1000)
      .attr("stroke", "#2b313a").attr("stroke-width", 0.6).attr("opacity", 0.85);
    g.append("line")
      .attr("y1", p).attr("x1", 0).attr("y2", p).attr("x2", 1000)
      .attr("stroke", "#2b313a").attr("stroke-width", 0.6).attr("opacity", 0.85);
    if (i > 0 && i < km) {{
      g.append("text")
        .attr("x", p + 3).attr("y", 12)
        .text(i + " km")
        .attr("fill", "#5a6270").attr("font-size", 9)
        .attr("font-family", "IBM Plex Mono, SF Mono, monospace");
    }}
  }}
}}

const xpTip = document.getElementById("xp-tip");
function xpShowTip(html, evt) {{
  xpTip.innerHTML = html;
  xpTip.style.display = "block";
  const b = xpSvg.node().getBoundingClientRect();
  xpTip.style.left = (evt.clientX - b.left + 12) + "px";
  xpTip.style.top  = (evt.clientY - b.top + 12) + "px";
}}
function xpHideTip() {{ xpTip.style.display = "none"; }}

// degree 기반 노드 크기 — 모든 뷰에서 동일.
// min 9 (degree=0 / 고립), max 26.
const XP_R_MIN = 9, XP_R_MAX = 26;
function xpRadius(d) {{
  if (!d.alive) return 5;
  const total = (d.out || 0) + (d.in || 0);
  const r = XP_R_MIN + Math.sqrt(total) * 4.0;
  return Math.min(XP_R_MAX, r);
}}
function xpColor(d) {{
  if (!d.alive) return "#c8c1b2";
  if (xpCurrentView === "degree") return "#1a1a1a";
  if (xpCurrentView === "community") {{
    if (d.community == null || d.community < 0) return "#d8d1c2";
    return XP_CPAL[d.community % XP_CPAL.length];
  }}
  if (xpCurrentView === "position") {{
    return XP_POSC[d.position] || "#d8d1c2";
  }}
  return "#1a1a1a";
}}
function xpOpacity(d) {{
  if (!d.alive) return 0.22;
  return xpCurrentView === "degree" ? 0.85 : 0.9;
}}

function xpTooltipHtml(d) {{
  const rank = (d.rank == null) ? "—" : `${{d.rank}}등`;
  const surv = (d.survival/60).toFixed(1);
  const alive = d.alive ? "생존" : "탈락";
  return `
    <b>팀 T${{d.team}}</b> <span style="color:#b02a2a">${{alive}}</span><br/>
    최종 순위 ${{rank}} · 총 생존 ${{surv}}분<br/>
    현 시점 degree ${{d.degree || 0}}<br/>
    community C${{d.community == null ? '—' : d.community}}<br/>
    위치 <i>${{d.position || '—'}}</i>
  `;
}}

// ── 매치 로드 ──
function xpLoadMatch(idx) {{
  const match = XP_MATCHES[idx];
  if (!match) return;
  xpCurrentIdx = idx;
  xpArenaM = match.arena_m || 8160;
  // 새 매치 로드 시 player view 데이터가 없으면 team 으로 폴백
  if (xpCurrentLevel === "player" && !(match.players && match.players.length)) {{
    xpCurrentLevel = "team";
    document.querySelectorAll("#explorer .level-buttons button")
      .forEach(b => b.classList.toggle("active", b.dataset.level === "team"));
  }}

  // 맵 이미지: base64 우선 → URL chain fallback (이미지 로드 시도)
  const mapSrc = match.map_b64 || XP_SHARED_MAP;
  const urls = (match.map_urls && match.map_urls.length)
                ? match.map_urls : XP_SHARED_URLS;
  if (mapSrc) {{
    xpShowMap(mapSrc);
  }} else if (urls && urls.length) {{
    xpTryMapUrls(urls.slice());
  }} else {{
    xpHideMap();
  }}
  xpRenderGrid();

  // 노드 datum (level 별)
  xpNodeById = new Map();
  let data;
  if (xpCurrentLevel === "player" && match.players && match.players.length) {{
    data = match.players.map(p => {{
      const d = {{
        id: p.id, team: p.team, team_id: p.team_id,
        short: p.short, survival: p.survival, kind: "player",
        x: 0, y: 0,
        out: 0, in: 0, community: null, position: "mid", alive: true,
      }};
      xpNodeById.set(p.id, d);
      return d;
    }});
  }} else {{
    data = match.nodes.map(n => {{
      const d = {{
        id: n.id, team: n.team, rank: n.rank,
        survival: n.survival, kind: "team",
        x: 0, y: 0,
        out: 0, in: 0, community: null, position: "mid", alive: true,
      }};
      xpNodeById.set(n.id, d);
      return d;
    }});
  }}

  // nodes 레이어
  xpNodeSel = d3.select("#xp-nodes").selectAll("circle")
    .data(data, d => d.id).join(
      enter => enter.append("circle")
        .attr("stroke", "#1a1a1a").attr("stroke-width", 0.8)
        .style("cursor", "pointer")
        .on("mousemove", (e, d) => xpShowTip(xpTooltipHtml(d), e))
        .on("mouseleave", xpHideTip),
      update => update,
      exit => exit.remove()
    );

  // 팀명 라벨 (노드 위쪽, 작은 폰트)
  xpLabelTeamSel = d3.select("#xp-labels-team").selectAll("text")
    .data(data, d => d.id).join(
      enter => enter.append("text")
        .text(d => xpCurrentLevel === "player"
                    ? ((d.short && d.short.length) ? d.short : "P" + d.id) + " · T" + d.team
                    : "T" + d.team)
        .attr("font-size", xpCurrentLevel === "player" ? 7 : 8.5)
        .attr("font-family", "IBM Plex Mono, SF Mono, monospace")
        .attr("text-anchor", "middle")
        .attr("fill", "#fefcf7")
        .attr("stroke", "#0f1115")
        .attr("stroke-width", 1.6)
        .attr("paint-order", "stroke")
        .style("pointer-events", "none")
        .style("opacity", 0.92),
      update => update.text(d => xpCurrentLevel === "player"
                  ? ((d.short && d.short.length) ? d.short : "P" + d.id) + " · T" + d.team
                  : "T" + d.team)
        .attr("font-size", xpCurrentLevel === "player" ? 7 : 8.5),
      exit => exit.remove()
    );

  // degree 숫자 라벨 (노드 안, 큰 폰트, 가독성 우선)
  xpLabelDegSel = d3.select("#xp-labels-deg").selectAll("text")
    .data(data, d => d.id).join(
      enter => enter.append("text")
        .attr("font-size", 12)
        .attr("font-weight", 700)
        .attr("font-family", "IBM Plex Mono, SF Mono, monospace")
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", "#fefcf7")
        .attr("stroke", "#0f1115")
        .attr("stroke-width", 1.4)
        .attr("paint-order", "stroke")
        .style("pointer-events", "none"),
      update => update,
      exit => exit.remove()
    );

  // 타임라인 슬라이더
  const slider = document.getElementById("xp-slider");
  slider.min = 0;
  slider.max = match.duration_s;
  slider.step = 1;
  slider.value = Math.floor(match.duration_s / 2);   // 기본은 매치 중반

  // zoom 리셋 (매치 바뀌면 맵 전체가 보이도록)
  xpSvg.call(xpZoom.transform, d3.zoomIdentity);

  xpApplyTime(parseFloat(slider.value));
}}

// ── 현재 시각 t → 맵 상 노드 + 두 종류 엣지 + 지표 업데이트 ──
function xpApplyTime(t) {{
  const match = XP_MATCHES[xpCurrentIdx];
  if (!match) return;
  xpCurrentT = t;

  // 1) 이전/다음 프레임 찾기 (선형 보간)
  const frames = match.frames;
  if (!frames.length) return;
  let fi = 0;
  for (let i = 0; i < frames.length; i++) {{
    if (frames[i].t <= t) fi = i; else break;
  }}
  const f0 = frames[fi];
  const f1 = frames[Math.min(fi + 1, frames.length - 1)];
  const span = Math.max(f1.t - f0.t, 1);
  const u = Math.max(0, Math.min(1, (t - f0.t) / span));

  const isPlayer = (xpCurrentLevel === "player");

  // 2) 노드 속성 업데이트 — 레벨 분기
  let alive, ioMap, commMap, posMap, pos0, pos1;
  if (isPlayer) {{
    alive   = new Set(f0.p_alive || []);
    ioMap   = f0.p_io || {{}};
    commMap = f0.comm || {{}};   // player level community 는 미구현, team 의 것을 자기 팀 기준으로 차용
    posMap  = f0.pos  || {{}};
    pos0    = f0.p_pos || {{}};
    pos1    = f1.p_pos || {{}};
  }} else {{
    alive   = new Set(f0.alive || []);
    ioMap   = f0.team_io || {{}};
    commMap = f0.comm || {{}};
    posMap  = f0.pos  || {{}};
    pos0    = f0.cx ? Object.fromEntries(Object.keys(f0.cx).map(
                  k => [k, [f0.cx[k], f0.cy[k]]])) : {{}};
    pos1    = f1.cx ? Object.fromEntries(Object.keys(f1.cx).map(
                  k => [k, [f1.cx[k], f1.cy[k]]])) : {{}};
  }}

  xpNodeById.forEach((n, id) => {{
    const sid = String(id);
    const p0 = pos0[sid];
    const p1 = pos1[sid];
    if (p0 && p1) {{
      n.x = p0[0] + (p1[0] - p0[0]) * u;
      n.y = p0[1] + (p1[1] - p0[1]) * u;
    }} else if (p0) {{
      n.x = p0[0]; n.y = p0[1];
    }}
    const io = ioMap[sid] || ioMap[id] || {{out:0, in:0}};
    n.out = io.out || 0;
    n.in  = io.in  || 0;
    // player view 에서 community/position 은 그 player 가 속한 팀 기준
    const colorKey = isPlayer ? String(n.team) : sid;
    n.community = (commMap[colorKey] != null) ? commMap[colorKey] : null;
    n.position  = posMap[colorKey] || "mid";
    n.alive     = alive.has(id);
  }});

  // 3) Combat 엣지 (directed, 누적 elapsed ≤ t)
  const events = isPlayer ? (match.p_events || []) : match.events;
  // (src,dst) 단방향 키
  const dirKey = (a, b) => a + "→" + b;
  const combatMap = new Map();
  for (const ev of events) {{
    if (ev[2] > t) break;
    const k = dirKey(ev[0], ev[1]);
    let ent = combatMap.get(k);
    if (!ent) {{
      ent = {{ src: ev[0], dst: ev[1], events: 0, dmg: 0 }};
      combatMap.set(k, ent);
    }}
    ent.events += 1;
    ent.dmg    += ev[3];
  }}
  const combatLinks = Array.from(combatMap.values())
    .filter(e => e.src !== e.dst)
    .map(e => {{
      const s = xpNodeById.get(e.src);
      const d_ = xpNodeById.get(e.dst);
      // team view 에서 src/dst 가 같은 팀이면 (= player 단위 attacker/victim 이 같은 팀) 무시
      // 이미 payload 에서 팀킬 제외됐으나 안전.
      if (!s || !d_) return null;
      return {{
        src: e.src, dst: e.dst, events: e.events, dmg: e.dmg,
        sx: s.x, sy: s.y, dx: d_.x, dy: d_.y,
        both_alive: s.alive && d_.alive,
      }};
    }})
    .filter(Boolean);

  // 4) Ally 엣지 (player level 만, undirected, proximity 기반 — frame 마다 다름)
  let allyLinks = [];
  if (isPlayer) {{
    const allyPairs = f0.ally || [];
    for (const pr of allyPairs) {{
      const a = xpNodeById.get(pr[0]);
      const b = xpNodeById.get(pr[1]);
      if (!a || !b) continue;
      allyLinks.push({{
        a: pr[0], b: pr[1],
        ax: a.x, ay: a.y, bx: b.x, by: b.y,
        both_alive: a.alive && b.alive,
      }});
    }}
  }}

  // 5) 엣지 렌더 — combat (directed, arrow)
  xpCombatSel = d3.select("#xp-combat").selectAll("line")
    .data(combatLinks, d => dirKey(d.src, d.dst))
    .join(
      enter => enter.append("line")
        .attr("stroke", "#b02a2a")
        .attr("stroke-opacity", 0)
        .attr("stroke-linecap", "round")
        .attr("marker-end", "url(#xp-arrow)")
        .call(e => e.transition().duration(220).attr("stroke-opacity", 0.6)),
      update => update,
      exit => exit.call(e => e.transition().duration(180)
                              .attr("stroke-opacity", 0).remove())
    )
    .attr("x1", d => xpAx(d.sx))
    .attr("y1", d => xpAy(d.sy))
    .attr("x2", d => xpAx(d.dx))
    .attr("y2", d => xpAy(d.dy))
    .attr("stroke-width", d => Math.min(0.5 + Math.sqrt(d.events) * 0.22, 2.2))
    .attr("stroke-opacity", d => d.both_alive ? 0.65 : 0.22);

  // ally (undirected, dashed)
  xpAllySel = d3.select("#xp-ally").selectAll("line")
    .data(allyLinks, d => Math.min(d.a, d.b) + "-" + Math.max(d.a, d.b))
    .join(
      enter => enter.append("line")
        .attr("stroke", "#14b8a6")
        .attr("stroke-opacity", 0)
        .attr("stroke-width", 0.9)
        .call(e => e.transition().duration(180).attr("stroke-opacity", 0.5)),
      update => update,
      exit => exit.call(e => e.transition().duration(150)
                              .attr("stroke-opacity", 0).remove())
    )
    .attr("x1", d => xpAx(d.ax))
    .attr("y1", d => xpAy(d.ay))
    .attr("x2", d => xpAx(d.bx))
    .attr("y2", d => xpAy(d.by))
    .attr("stroke-opacity", d => d.both_alive ? 0.55 : 0.18);

  // 6) 노드 위치 + 색
  xpNodeSel
    .attr("cx", d => xpAx(d.x))
    .attr("cy", d => xpAy(d.y))
    .attr("r",  d => xpRadius(d))
    .attr("fill", d => xpColor(d))
    .attr("fill-opacity", d => xpOpacity(d));

  // 팀명 라벨
  xpLabelTeamSel
    .attr("x", d => xpAx(d.x))
    .attr("y", d => xpAy(d.y) - xpRadius(d) - 3)
    .attr("fill-opacity", d => d.alive ? 0.92 : 0.35);

  // in/out degree 라벨 — 노드 안. 형식: "out↑ · in↓" 짧은 표기
  xpLabelDegSel
    .attr("x", d => xpAx(d.x))
    .attr("y", d => xpAy(d.y))
    .text(d => {{
      if (!d.alive) return "";
      const o = d.out || 0, i = d.in || 0;
      return o + "↑" + i + "↓";
    }})
    .style("font-size", d => {{
      const r = xpRadius(d);
      return Math.max(7, Math.min(11, r * 0.55)) + "px";
    }});

  // 7) Zone (마지막 frame 의 zone 만 정적 표시)
  xpRenderZone(match);

  // 8) Community hull
  xpRenderCommunityHulls();

  // 9) HUD
  xpUpdateStats(match, f0, combatLinks, allyLinks);
  xpUpdateLegend(match, f0);
  xpUpdateTimeReadout(t, match.duration_s, f0);
}}

// ── Zone 그리기 (현 시점 zone 정보가 frame 에 있으면 그것을 사용) ──
function xpRenderZone(match) {{
  const frames = match.frames;
  // 가장 가까운 zone 정보가 있는 frame
  let zone = null;
  for (let i = frames.length - 1; i >= 0; i--) {{
    if (frames[i].t > xpCurrentT) continue;
    if (frames[i].zone) {{ zone = frames[i].zone; break; }}
  }}
  const g = d3.select("#xp-zone");
  g.selectAll("*").remove();
  if (!zone) return;
  if (zone.pr > 0) {{
    g.append("circle")
      .attr("cx", xpAx(zone.px)).attr("cy", xpAy(zone.py))
      .attr("r", (zone.pr / xpArenaM) * 1000);
  }}
  if (zone.sr > 0) {{
    g.append("circle")
      .attr("cx", xpAx(zone.sx)).attr("cy", xpAy(zone.sy))
      .attr("r", (zone.sr / xpArenaM) * 1000)
      .attr("stroke", "rgba(255,255,255,0.7)")
      .attr("fill", "none");
  }}
}}

// ── 커뮤니티별 convex hull (반투명 영역) ──
function xpRenderCommunityHulls() {{
  const aliveNodes = [];
  xpNodeById.forEach(n => {{ if (n.alive) aliveNodes.push(n); }});
  const groups = d3.group(aliveNodes,
    d => (d.community != null ? d.community : "_none"));
  const hulls = [];

  for (const [cid, members] of groups) {{
    if (cid === "_none") continue;
    if (members.length === 1) {{
      const m = members[0];
      hulls.push({{
        cid,
        d: `M ${{xpAx(m.x) - 32}} ${{xpAy(m.y)}}
            a 32 32 0 1 0 64 0
            a 32 32 0 1 0 -64 0 Z`,
      }});
      continue;
    }}
    if (members.length === 2) {{
      // 두 노드를 잇는 캡슐 모양: 직접 path 로
      const a = members[0], b = members[1];
      const ax = xpAx(a.x), ay = xpAy(a.y);
      const bx = xpAx(b.x), by = xpAy(b.y);
      const dx = bx - ax, dy = by - ay;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len;
      const nx = -uy, ny = ux;
      const pad = 30;
      const ax1 = ax - ux * pad, ay1 = ay - uy * pad;
      const bx1 = bx + ux * pad, by1 = by + uy * pad;
      const p1 = [ax1 + nx * pad, ay1 + ny * pad];
      const p2 = [bx1 + nx * pad, by1 + ny * pad];
      const p3 = [bx1 - nx * pad, by1 - ny * pad];
      const p4 = [ax1 - nx * pad, ay1 - ny * pad];
      hulls.push({{
        cid,
        d: `M${{p1[0]}},${{p1[1]}} L${{p2[0]}},${{p2[1]}} ` +
           `L${{p3[0]}},${{p3[1]}} L${{p4[0]}},${{p4[1]}} Z`,
      }});
      continue;
    }}
    // 3개 이상: convex hull + smooth + outward padding
    const pts = members.map(m => [xpAx(m.x), xpAy(m.y)]);
    const hull = d3.polygonHull(pts);
    if (!hull) continue;
    const cx = d3.mean(hull, p => p[0]);
    const cy = d3.mean(hull, p => p[1]);
    const PAD = 24;
    const inflated = hull.map(([x, y]) => {{
      const dx = x - cx, dy = y - cy;
      const len = Math.hypot(dx, dy) || 1;
      return [x + (dx / len) * PAD, y + (dy / len) * PAD];
    }});
    const lineGen = d3.line().curve(d3.curveCatmullRomClosed.alpha(0.7));
    hulls.push({{ cid, d: lineGen(inflated) }});
  }}

  const baseAlpha = (xpCurrentView === "community") ? 0.22 : 0.10;
  const strokeAlpha = (xpCurrentView === "community") ? 0.55 : 0.25;

  d3.select("#xp-hulls").selectAll("path")
    .data(hulls, d => d.cid)
    .join(
      enter => enter.append("path")
        .attr("fill-opacity", 0)
        .attr("stroke-opacity", 0)
        .call(e => e.transition().duration(300)
                    .attr("fill-opacity", baseAlpha)
                    .attr("stroke-opacity", strokeAlpha)),
      update => update.transition().duration(220)
        .attr("fill-opacity", baseAlpha)
        .attr("stroke-opacity", strokeAlpha),
      exit => exit.call(e => e.transition().duration(200)
                              .attr("fill-opacity", 0)
                              .attr("stroke-opacity", 0).remove())
    )
    .attr("d", d => d.d)
    .attr("fill", d => XP_CPAL[(+d.cid) % XP_CPAL.length])
    .attr("stroke", d => XP_CPAL[(+d.cid) % XP_CPAL.length])
    .attr("stroke-width", 1.2)
    .attr("stroke-linejoin", "round")
    .style("pointer-events", "none");
}}

function xpFmtT(sec) {{
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${{String(m).padStart(2,"0")}}:${{String(s).padStart(2,"0")}}`;
}}

function xpUpdateStats(match, frame, combatLinks, allyLinks) {{
  const isPlayer = (xpCurrentLevel === "player");
  const aliveN = frame
    ? (isPlayer ? (frame.p_alive || []).length : (frame.alive || []).length)
    : 0;
  const total = isPlayer ? (match.n_players || match.n_nodes) : match.n_nodes;
  const ce = combatLinks ? combatLinks.length : 0;
  const ae = allyLinks ? allyLinks.length : 0;
  const allyTxt = isPlayer ? ` · ally <b>${{ae}}</b>` : "";
  document.getElementById("xp-stats").innerHTML =
    `생존 <b>${{aliveN}}/${{total}}</b> · ` +
    `combat <b>${{ce}}</b>${{allyTxt}} · 최종 Q <b>${{match.modularity.toFixed(2)}}</b>`;
}}

function xpUpdateLegend(match, frame) {{
  const host = document.getElementById("xp-legend");
  if (xpCurrentView === "degree") {{
    host.innerHTML =
      `<span class="legend-hint">원 크기 = 현 시점 degree · ` +
      `선 두께 = 누적 교전 이벤트 수 · 투명 원 = 탈락 팀</span>`;
  }} else if (xpCurrentView === "community") {{
    const seen = new Set();
    (frame ? Object.values(frame.comm) : []).forEach(v => seen.add(v));
    const comms = Array.from(seen).sort((a,b)=>a-b);
    const items = comms.map(c =>
      `<span class="legend-item"><span class="sw" style="background:` +
      `${{XP_CPAL[c % XP_CPAL.length]}}"></span>C${{c}}</span>`
    ).join("");
    host.innerHTML = `<span class="legend-hint">Louvain 커뮤니티 (현 시점, anchor=최소 team_idx): </span>${{items}}`;
  }} else {{
    host.innerHTML =
      `<span class="legend-hint">네트워크 위치 (현 시점): </span>` +
      `<span class="legend-item"><span class="sw" style="background:${{XP_POSC.core}}"></span>core</span>` +
      `<span class="legend-item"><span class="sw" style="background:${{XP_POSC.mid}}"></span>mid</span>` +
      `<span class="legend-item"><span class="sw" style="background:${{XP_POSC.peripheral}}"></span>peripheral</span>` +
      `<span class="legend-item"><span class="sw" style="background:${{XP_POSC.isolated}}"></span>isolated</span>`;
  }}
}}

function xpUpdateTimeReadout(t, dur, frame) {{
  document.getElementById("xp-time").textContent =
    `${{xpFmtT(t)}} / ${{xpFmtT(dur)}}`;
  const fInfo = document.getElementById("xp-frameinfo");
  if (frame) {{
    fInfo.textContent = `frame t=${{frame.t}}s · edges=${{frame.n_edges}}`;
  }} else {{
    fInfo.textContent = "";
  }}
}}

// ── Playback ──
function xpStartPlay() {{
  xpPlaying = true;
  document.getElementById("xp-play").textContent = "❚❚";
  xpLastTs = 0;
  xpRAF = requestAnimationFrame(xpTick);
}}
function xpStopPlay() {{
  xpPlaying = false;
  document.getElementById("xp-play").textContent = "▶";
  if (xpRAF) cancelAnimationFrame(xpRAF);
}}
function xpTick(ts) {{
  if (!xpPlaying) return;
  if (!xpLastTs) xpLastTs = ts;
  const dtMs = ts - xpLastTs;
  xpLastTs = ts;
  // 실제 매치 시간 속도 = xpSpeed × 게임초/실초 (기본 2×: 1초에 2초)
  // 너무 느리면 지루하므로 ×10 기본 가속
  const gameSecPerRealMs = (xpSpeed * 10) / 1000;
  const dt = dtMs * gameSecPerRealMs;
  const match = XP_MATCHES[xpCurrentIdx];
  let nextT = xpCurrentT + dt;
  if (nextT >= match.duration_s) {{
    nextT = match.duration_s;
    xpStopPlay();
  }}
  xpApplyTime(nextT);
  document.getElementById("xp-slider").value = nextT;
  if (xpPlaying) xpRAF = requestAnimationFrame(xpTick);
}}

// ── Event bindings ──
document.getElementById("match-select").addEventListener("change", (e) => {{
  xpStopPlay();
  xpLoadMatch(parseInt(e.target.value, 10));
}});
document.querySelectorAll("#explorer .view-buttons button").forEach(b => {{
  b.addEventListener("click", () => {{
    xpCurrentView = b.dataset.view;
    document.querySelectorAll("#explorer .view-buttons button")
      .forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    xpApplyTime(xpCurrentT);
  }});
}});
document.querySelectorAll("#explorer .level-buttons button").forEach(b => {{
  b.addEventListener("click", () => {{
    if (b.classList.contains("active")) return;
    xpStopPlay();
    xpCurrentLevel = b.dataset.level;
    document.querySelectorAll("#explorer .level-buttons button")
      .forEach(x => x.classList.toggle("active", x === b));
    // 레벨이 바뀌면 노드 셋이 통째로 바뀌므로 매치 재로드
    const savedT = xpCurrentT;
    xpLoadMatch(xpCurrentIdx);
    xpApplyTime(savedT);
    document.getElementById("xp-slider").value = savedT;
  }});
}});
document.getElementById("xp-play").addEventListener("click", () => {{
  if (xpPlaying) {{ xpStopPlay(); return; }}
  const match = XP_MATCHES[xpCurrentIdx];
  if (xpCurrentT >= match.duration_s) {{
    xpApplyTime(0);
    document.getElementById("xp-slider").value = 0;
  }}
  xpStartPlay();
}});
document.getElementById("xp-slider").addEventListener("input", (e) => {{
  xpStopPlay();
  xpApplyTime(parseFloat(e.target.value));
}});
document.getElementById("xp-speed").addEventListener("change", (e) => {{
  xpSpeed = parseFloat(e.target.value);
}});

xpLoadMatch(0);
</script>
"""


def render_methods() -> str:
    """Methods reference — 모든 지표가 어떤 조건일 때 어떻게 정의되는가."""
    return """
<section id="methods">
  <div class="section-label">methods reference</div>
  <h2>지표 · 노드 · 엣지 정의</h2>
  <p class="lede">
    이 분석에서 쓰인 모든 지표가 <b>어떤 조건일 때 어떻게 부여되는가</b> 를
    한 곳에서 정리한다. 코드의 함수가 아니라 원리 중심.
  </p>

  <h3>1. 노드 — 두 레벨</h3>
  <div class="m-grid">
    <div class="m-card">
      <h4>Player node</h4>
      <p>1 노드 = 매치에 참가한 개인 플레이어. 매치당 약 100명.
         위치는 매 30초 프레임에서의 (x, y) 미터 좌표 (PUBG 정규화 좌표
         × arena 크기). <b>alive</b> 의 조건은 그 플레이어의 사망 시각이
         t 이후이거나 매치 끝까지 생존한 경우.</p>
    </div>
    <div class="m-card">
      <h4>Team node</h4>
      <p>1 노드 = squad team. 매치당 약 25팀. 위치는 그 시점의 alive 멤버
         평균 좌표(team centroid). <b>alive</b> 의 조건은 팀의 마지막 멤버
         사망 시각 ≥ t — 누구라도 살아 있으면 팀은 살아 있다.</p>
    </div>
  </div>
  <p class="note">explorer 헤더의 <b>Team view</b> / <b>Player view</b>
    토글로 전환. 어느 view 에서든 ally/combat 두 엣지 종류는 그대로 보인다.</p>

  <h3>2. 엣지 — ally(양방향) + combat(단방향)</h3>
  <div class="m-grid">
    <div class="m-card ally">
      <h4>Ally edge — 양방향, 청록색</h4>
      <p>두 플레이어가 (a) 같은 squad team 이고, (b) 그 시점에 서로 일정
         거리 안에 있을 때 연결된다 (proximity 기반). 멤버가 흩어지면
         사라진다 — 시점마다 다시 계산. <b>degree 에는 포함하지 않는다.</b></p>
    </div>
    <div class="m-card combat">
      <h4>Combat edge — 단방향(→), 빨강</h4>
      <p>telem_damage 로그에서 attacker A 가 victim B 에게 데미지를 입힌
         이벤트가 발생했을 때 <b>A → B</b> 로 그어진다. 시간이 흐를수록
         누적되며, 시간을 되감으면 사라진다. 두께는 누적 이벤트 수의
         √ 비례, 두 노드 모두 alive 면 진하게(0.65), 한쪽 탈락이면
         흐리게(0.22).</p>
    </div>
  </div>

  <h3>3. Degree — 데미지 이벤트 기준, in / out 분리</h3>
  <table class="stat">
    <tr><th>이름</th><th>어떤 조건일 때 +1 되는가</th><th>의미</th></tr>
    <tr>
      <td><b>out_degree(i)</b></td>
      <td>i 가 attacker 였던 데미지 이벤트에서 새로운 victim 이 등장했을 때</td>
      <td>i 가 공격한 서로 다른 적의 수</td>
    </tr>
    <tr>
      <td><b>in_degree(i)</b></td>
      <td>i 가 victim 이었던 데미지 이벤트에서 새로운 attacker 가 등장했을 때</td>
      <td>i 를 공격한 서로 다른 적의 수</td>
    </tr>
  </table>
  <p class="note">같은 적을 5번 때려도 out_degree 는 +1 (그래프 이론의
    표준 degree). 노드 안에는 <code>out↑ in↓</code> 형식으로 두 값을
    함께 표시. 노드 크기는 <code>√(out + in)</code> 비례, 최소 9 ~ 최대 26.</p>

  <h3>4. 좌표계와 맵</h3>
  <p>원점 (0, 0) = 맵 이미지 좌상단. x 양의 방향 = 우, y 양의 방향 = 아래
     (이미지 좌표계와 동일, flip 불필요). arena 한 변 = 매치 메타에서 가져옴
     (Baltic_Main = 8160m). SVG viewBox 1000 × 1000 에 한 변이 정확히
     매핑되도록 \\( \\text{screen}_x = x / \\text{arena} \\times 1000 \\)
     변환. 회전 0°.</p>

  <h3>5. 전역 그래프 지표 (combat directed graph)</h3>
  <table class="stat">
    <tr><th>지표</th><th>어떤 조건/원리</th><th>PUBG 의미</th></tr>
    <tr><td>density</td>
        <td>가능한 directed pair 중 실제 combat edge 가 있는 비율</td>
        <td>가능한 적 쌍 중 실제 데미지 발생 비율</td></tr>
    <tr><td>avg clustering \\( \\bar{C} \\)</td>
        <td>"내가 공격한 두 적이, 그 둘끼리도 공격했나" 의 평균</td>
        <td>한 지역에 셋 이상이 동시 교전한 빈도</td></tr>
    <tr><td>diameter / avg path</td>
        <td>giant component 안에서 두 노드 간 최단경로의 max / mean</td>
        <td>매치가 small-world 구조인지</td></tr>
    <tr><td>n_components / giant size</td>
        <td>약하게 연결된 성분 수와 최대 크기</td>
        <td>매치가 한 덩어리인지, 분리된 진영인지</td></tr>
  </table>

  <h3>6. 중심성 5종</h3>
  <table class="stat">
    <tr><th>지표</th><th>어떤 원리로 / 어떤 조건일 때 큰가</th><th>PUBG 의미</th></tr>
    <tr><td>Degree</td>
        <td>(out_degree + in_degree) / (n−1)</td>
        <td>가장 단순한 활동성</td></tr>
    <tr><td>Closeness</td>
        <td>다른 노드까지의 directed 평균 거리의 역수</td>
        <td>어디로든 짧게 닿는 위치</td></tr>
    <tr><td>Betweenness</td>
        <td>모든 directed pair 의 최단경로가 이 노드를 지나는 비율 합</td>
        <td>다른 진영을 잇는 중개자</td></tr>
    <tr><td>Eigenvector</td>
        <td>중요한 노드와 연결될수록 본인도 중요해지는 재귀적 정의</td>
        <td>강팀의 표적이 된 정도 (in 방향)</td></tr>
    <tr><td>PageRank</td>
        <td>random teleport 보강된 정상상태 분포</td>
        <td>중심 교전지에 있던 노드</td></tr>
  </table>
  <p class="note">거리 기반 지표(Closeness, Betweenness)는 가중치가 클수록
    가깝게 해석하기 위해 <code>distance = 1/events</code> 변환을 거친다.</p>

  <h3>7. 클러스터(커뮤니티) — 언제 같이 묶이나</h3>
  <p>모듈러리티 \\( Q \\) 를 가장 크게 만드는 분할(Louvain)에서 같은 그룹에
     들어갈 때 같은 클러스터로 본다. \\( Q \\) 의 정의:</p>
  <p class="eq">\\[
    Q = \\frac{1}{2m} \\sum_{i,j}
       \\left( A_{ij} - \\frac{k_i\\,k_j}{2m} \\right)
       \\delta(c_i, c_j)
  \\]</p>
  <p>직관: <b>"그룹 내부 엣지 합" 이 같은 degree 분포의 무작위 그래프
     기댓값보다 충분히 클 때</b> 한 클러스터가 된다. 즉 자기들끼리
     집중적으로 싸운 묶음. <b>Q ≤ 0.3</b> = 무작위와 다름없음, <b>Q > 0.3</b>
     = 의미 있는 진영 구조 (Newman 2004), <b>Q > 0.5</b> = 매우 강함.</p>
  <p class="note">색 안정화: 한 클러스터의 ID = 멤버 중 최소 인덱스(anchor).
     멤버가 그대로면 색 일정, 합쳐지면 anchor 가 더 작은 쪽으로 점프하면서
     색이 의미 있게 바뀐다.</p>

  <h3>8. 외곽 vs 중심 — Position 라벨</h3>
  <table class="stat">
    <tr><th>조건</th><th>라벨</th><th>의미</th></tr>
    <tr><td>out + in = 0</td>
        <td><b>isolated</b></td>
        <td>매치 내내 데미지 주고받음 0 (파밍/존 외곽)</td></tr>
    <tr><td>betweenness ≥ 상위 25%</td>
        <td><b>core</b></td>
        <td>다른 진영을 잇는 중개자</td></tr>
    <tr><td>betweenness ≤ 하위 25%</td>
        <td><b>peripheral</b></td>
        <td>연결은 있으나 주변부</td></tr>
    <tr><td>그 외</td>
        <td><b>mid</b></td>
        <td>중간</td></tr>
  </table>

  <h3>9. 시간당 강도 (rate-normalized)</h3>
  <p>누적 값을 그 노드의 생존 분(分) 으로 나눠 단위 시간당 강도로 변환한다.
     분모: <code>survival_seconds / 60</code>. 짧게 살고 격렬했던 노드는
     큰 값, 오래 살고 조용했던 노드는 작은 값 → "오래 살아서 자연히 큰 값"
     의 효과를 산술적으로 분리.</p>

  <h3>10. Early-game window</h3>
  <p>매치 시작 후 처음 N초(기본 300초) 의 combat 이벤트만으로 그래프를
     다시 구성. 이 구간에는 거의 모든 노드가 살아있어 생존 편향이 0 에
     가깝다 — "초반에 적극적인 노드가 결국 최종 순위가 좋은가" 를 직접
     묻는 형태.</p>

  <h3>11. 편상관 — 생존시간 통제</h3>
  <p>두 변수 X, Y 의 상관에서 공통 원인 Z(=생존시간)의 영향을 산술적으로
     제거한다.</p>
  <p class="eq">\\[
    r_{XY \\cdot Z} \\;=\\;
    \\frac{r_{XY} - r_{XZ}\\,r_{YZ}}{\\sqrt{(1 - r_{XZ}^{2})(1 - r_{YZ}^{2})}}
  \\]</p>
  <p>직관: <b>"Z 가 같다고 가정했을 때 X 와 Y 가 함께 움직이는 정도"</b>.
     원시 상관이 모두 Z 때문이었다면 이 값은 0 부근.</p>

  <h3>12. 가설 자동 판정 규칙</h3>
  <table class="stat">
    <tr><th>가설</th><th>지지/기각/유보 조건</th></tr>
    <tr><td>H1 — degree ↔ 순위</td>
        <td>원시 r ≤ −0.2 + (강도·초반·편상관) 셋 다 |r| < 0.15 → <b>기각</b> /
            셋 중 둘 이상 r ≤ −0.15, 중앙값 ≤ −0.2 → <b>지지</b> /
            그 외 <b>결론 유보</b></td></tr>
    <tr><td>H2 — 커뮤니티 크기 ↔ 순위</td>
        <td>고립(size=1) vs 대규모(size≥6) 평균 순위 차이 ≥ 1.5등 + p<0.05 →
            우위 방향에 따라 <b>파밍형</b> / <b>진영 참여형</b> 우위.
            그 외 <b>차이 없음</b></td></tr>
    <tr><td>H3 — 외곽 vs 중심</td>
        <td>core vs peripheral 평균 순위 차이 ≥ 1.0등 + p<0.05 → 우위
            방향에 따라 <b>중심</b> / <b>외곽</b> 우위. 그 외 <b>유보</b></td></tr>
  </table>

  <h3>13. 통계 검정</h3>
  <table class="stat">
    <tr><th>검정</th><th>어떤 질문에 쓰는가</th></tr>
    <tr><td>Pearson r</td><td>두 연속 변수의 선형 관계</td></tr>
    <tr><td>Spearman ρ</td><td>순위 기반 단조 관계 (비모수)</td></tr>
    <tr><td>Welch t</td><td>두 그룹 평균 차이 (분산 다름 허용)</td></tr>
    <tr><td>Mann-Whitney U</td><td>두 그룹 분포 차이 (비모수)</td></tr>
    <tr><td>Kruskal-Wallis H</td><td>세 그룹 이상 동등성 (비모수)</td></tr>
    <tr><td>Linear regression</td><td>scatter 위 회귀선 + R²</td></tr>
  </table>
  <p class="note">유의 표시: <code>*</code> p&lt;0.05 / <code>**</code>
     p&lt;0.01 / <code>***</code> p&lt;0.001.</p>
</section>
"""


def render_nav(sections: list[tuple[str, str]]) -> str:
    """좌측 네비게이션. sections = [(anchor, label), ...]"""
    items = "\n".join(
        f'<a class="nav-item" href="#{a}" data-sec="{a}">{_h(lbl)}</a>'
        for a, lbl in sections
    )
    return f"""
<nav class="sidenav">
  <div class="nav-title">팀 교전 네트워크 분석</div>
  <div class="nav-sub">PUBG 스쿼드 매치</div>
  <div class="nav-list">{items}</div>
  <div class="nav-foot">
    가설 1 · 2 · 3 을 한 페이지에 통합.<br/>
    좌측 메뉴로 이동하거나 스크롤.
  </div>
</nav>
"""


def render_overview(hero, h2, h3, n_matches, n_teams, early_sec) -> str:
    raw = hero.get("raw_r")
    rank_delta = hero.get("rank_delta")
    part = hero.get("partial_r")
    h1_badge = verdict_badge(hero.get("verdict", "inconclusive"))
    h2_badge = verdict_badge(h2.get("verdict", "inconclusive"))
    h3_badge = verdict_badge(h3.get("verdict", "inconclusive"))
    h1_line = (hero_summary(hero))
    h2_line = (h2["conclusion_lines"][0] if h2.get("conclusion_lines")
               else "결과 없음")
    h3_line = (h3["conclusion_lines"][0] if h3.get("conclusion_lines")
               else "결과 없음")

    early_str = f"초반 윈도우 {int(early_sec)}초" if early_sec else "—"

    return f"""
<section id="overview">
  <div class="section-label">개요</div>
  <h1 class="title">PUBG 스쿼드 매치에서 네트워크 지표는 순위를 예측하는가?</h1>
  <p class="lede">
    팀 간 데미지 이벤트로 구성한 교전 네트워크 위에서 세 가지 가설을
    동시 검증한다. 각 지표가 단순히 "숫자" 가 아니라 PUBG 매치에서 어떤
    상황을 의미하는지 함께 설명한다.
  </p>

  <div class="meta-row">
    <span>매치 <b>{n_matches}</b>건</span>
    <span>팀 데이터 <b>{n_teams}</b>행</span>
    <span>{early_str}</span>
  </div>

  <div class="cards">
    <a class="card" href="#h1">
      <div class="card-badge">{h1_badge}</div>
      <h3>가설 1 · 교전 참여도 ↔ 순위</h3>
      <p class="card-line">{_h(h1_line)}</p>
    </a>
    <a class="card" href="#h2">
      <div class="card-badge">{h2_badge}</div>
      <h3>가설 2 · 교전 밀집도 ↔ 순위</h3>
      <p class="card-line">{_h(h2_line)}</p>
    </a>
    <a class="card" href="#h3">
      <div class="card-badge">{h3_badge}</div>
      <h3>가설 3 · 외곽 vs 중심 생존율</h3>
      <p class="card-line">{_h(h3_line)}</p>
    </a>
  </div>
</section>
"""


def hero_summary(hero) -> str:
    verdict = hero.get("verdict", "inconclusive")
    raw = hero.get("raw_r"); part = hero.get("partial_r")
    delta = hero.get("rank_delta")
    parts = []
    if raw is not None and np.isfinite(raw):
        parts.append(f"원시 r={raw:+.3f}")
    if part is not None and np.isfinite(part):
        parts.append(f"편상관 r={part:+.3f}")
    if delta is not None and np.isfinite(delta):
        parts.append(f"상·하위 25% 순위차 {delta:.1f}등")
    if verdict == "confirmed":
        verdict_kr = "가설 지지"
    elif verdict == "rejected":
        verdict_kr = "기각 (생존 편향 기인)"
    else:
        verdict_kr = "결론 유보"
    return f"{verdict_kr}. " + " · ".join(parts) + "."


def render_basics(adjs, pks, globals_list) -> str:
    A, _ = adjs[0]
    adj_b64 = chart_adjacency(A)
    pk_b64 = chart_pk(pks)
    glb_b64 = chart_globals_box(globals_list)

    # 평균 값 + 해석
    def _avg(k):
        vs = [g[k] for g in globals_list if g.get(k) is not None]
        return float(np.mean(vs)) if vs else 0.0
    def _med(k):
        vs = [g[k] for g in globals_list if g.get(k) is not None]
        return float(np.median(vs)) if vs else 0.0

    avg_n = _avg("n_nodes")
    avg_density = _avg("density")
    avg_cc = _avg("avg_clustering")
    avg_apl = _avg("avg_path_length")
    med_diam = _med("diameter")

    density_row = f"""<tr>
      <td class='lbl'>밀도 (density)</td>
      <td>{avg_density:.3f}</td>
      <td class='interp-inline'>{density_interpretation(avg_density, avg_n)}</td>
    </tr>"""
    cc_row = f"""<tr>
      <td class='lbl'>평균 클러스터링 \\( \\bar{{C}} \\)</td>
      <td>{avg_cc:.3f}</td>
      <td class='interp-inline'>{clustering_interpretation(avg_cc)}</td>
    </tr>"""
    diam_row = f"""<tr>
      <td class='lbl'>지름 (중앙값) / 평균 경로</td>
      <td>{med_diam:.1f} / {avg_apl:.2f}</td>
      <td class='interp-inline'>{diameter_interpretation(med_diam, avg_apl)}</td>
    </tr>"""

    return f"""
<section id="basics">
  <div class="section-label">방법 · 네트워크 기초</div>
  <h2>네트워크 구성과 기본 지표</h2>

  <p>
    각 매치를 무향 가중 그래프 \\( G = (V, E) \\) 로 모델링한다.
    \\( V \\) 는 팀 집합, 두 팀 사이에 데미지 이벤트가 한 번이라도 발생하면
    엣지가 생성되고, 가중치는 누적 데미지이다. 이 표현에서 인접행렬 A 와
    차수행렬 D 가 이후 모든 분석의 기반이 된다.
  </p>
  <p class="eq">
    \\[
      \\text{{density}} = \\frac{{2\\,|E|}}{{|V|\\,(|V|-1)}}
      \\qquad
      C_i = \\frac{{2\\,e_i}}{{k_i\\,(k_i - 1)}}
    \\]
  </p>

  <figure>
    <img src="data:image/png;base64,{adj_b64}"/>
    <figcaption>대표 매치의 인접행렬 A (\\( |V| = {A.shape[0]} \\)).
      밝을수록 두 팀의 누적 데미지가 큼.</figcaption>
  </figure>
  <figure>
    <img src="data:image/png;base64,{pk_b64}"/>
    <figcaption>차수 분포 \\( P(k) \\) — 매치 전체 평균.</figcaption>
  </figure>

  <h3>매치 전체 평균 · 해석과 함께</h3>
  <table class="stat interp-table">
    <tr><th>지표</th><th>값</th><th>PUBG 매치에서의 의미</th></tr>
    {density_row}
    {cc_row}
    {diam_row}
  </table>

  <figure>
    <img src="data:image/png;base64,{glb_b64}"/>
    <figcaption>매치별 global 지표 분포 (boxplot).</figcaption>
  </figure>
</section>
"""


_CENTR_DEF = {
    "degree":      r"\( C_D(i) = \dfrac{k(i)}{|N| - 1} \)",
    "closeness":   r"\( C_C(i) = \dfrac{|N| - 1}{\sum_{j} d(i, j)} \)",
    "betweenness": r"\( C_B(i) = \sum_{j,\,k} \dfrac{p_{jk}(i)}{p_{jk}} \)",
    "eigenvector": r"\( \lambda\, c = A\, c \)",
    "pagerank":    r"\( r_j = \beta \sum_{i \to j} \dfrac{r_i}{d_i} + \dfrac{1 - \beta}{N} \)",
}
_CENTR_LABEL_KR = {
    "degree":      "연결중심성 (degree)",
    "closeness":   "근접중심성 (closeness)",
    "betweenness": "매개중심성 (betweenness)",
    "eigenvector": "고유벡터중심성 (eigenvector)",
    "pagerank":    "페이지랭크 (PageRank)",
}
_CENTR_PUBG = {
    "degree":      "이 팀이 직접 맞붙은 상대 팀의 수. 가장 단순한 '교전 참여도' 지표.",
    "closeness":   "모든 다른 팀과의 거리가 짧다 = 맵 곳곳의 팀들과 연쇄적으로 엮인 팀.",
    "betweenness": "서로 다른 교전 클러스터를 잇는 '중개자' 팀. 맵 중앙 교전지 혹은 경계선에서 여러 방향과 싸운 팀일 가능성.",
    "eigenvector": "단순히 많이 싸운 게 아니라 '강팀과 싸웠는가'. 매치의 중심 교전지에 있었는지를 나타냄.",
    "pagerank":    "eigenvector 의 확장. 고립된 팀에도 최소값이 보장되어 실제 매치 분석에 더 안정적.",
}


def render_centrality_defs() -> str:
    rows = "\n".join(
        f"<tr><td class='lbl'>{_CENTR_LABEL_KR[m]}</td>"
        f"<td class='math'>{_CENTR_DEF[m]}</td>"
        f"<td class='interp-inline'>{_CENTR_PUBG[m]}</td></tr>"
        for m in MEASURES
    )
    return f"""
<section id="centrality-defs">
  <div class="section-label">방법 · 중심성</div>
  <h2>팀 중심성 5종과 PUBG 해석</h2>
  <p>
    거리 기반 지표(closeness, betweenness)에서는 가중치가 클수록 "가깝다" 로
    해석하기 위해 \\( \\text{{distance}} = 1 / \\text{{weight}} \\) 변환을 적용.
    고유벡터 중심성은 Perron–Frobenius 정리로 보장되는 유일한 양의 고유벡터를
    power iteration 으로 구한다.
  </p>
  <table class="stat interp-table">
    <tr><th>지표</th><th>정의</th><th>PUBG 매치에서의 의미</th></tr>
    {rows}
  </table>
</section>
"""


def render_community_section(communities) -> str:
    qs = [c["louvain"]["modularity_q"] for c in communities if c.get("louvain")]
    ns = [c["louvain"]["n_communities"] for c in communities if c.get("louvain")]
    chart_b64 = chart_louvain_Q(communities)
    avg_q = float(np.mean(qs)) if qs else 0.0
    avg_n = float(np.mean(ns)) if ns else 0.0
    n_sig = sum(1 for q in qs if q > 0.3)

    return f"""
<section id="community">
  <div class="section-label">방법 · 커뮤니티</div>
  <h2>교전 커뮤니티 구조</h2>
  <p>
    가중 팀 그래프에 Louvain 알고리즘을 적용하면 서로 집중적으로 교전한 팀
    군집이 탐지된다. 모듈러리티 \\( Q \\) 는 이 분할이 랜덤보다 얼마나 뚜렷한지를
    수치화한다.
  </p>
  <p class="eq">
    \\[
      Q = \\frac{{1}}{{2m}} \\sum_{{i,\\,j}}
        \\left( A_{{ij}} - \\frac{{k_i\\,k_j}}{{2m}} \\right)
        \\delta(c_i, c_j)
    \\]
  </p>
  <table class="stat interp-table">
    <tr><th>통계</th><th>값</th><th>PUBG 매치에서의 의미</th></tr>
    <tr>
      <td class="lbl">평균 \\( Q \\)</td>
      <td>{avg_q:.3f}</td>
      <td class="interp-inline">{modularity_interpretation(avg_q, avg_n)}</td>
    </tr>
    <tr>
      <td class="lbl">\\( Q > 0.3 \\) 매치 수</td>
      <td>{n_sig} / {len(qs)}</td>
      <td class="interp-inline">
        Newman (2004) 기준을 넘는 매치 비율. 대부분의 squad 매치가 지리적 진영을
        형성한다는 뜻.
      </td>
    </tr>
  </table>
  <figure>
    <img src="data:image/png;base64,{chart_b64}"/>
    <figcaption>매치별 Louvain \\( Q \\) 분포.</figcaption>
  </figure>
</section>
"""


def render_h1(rows, h1) -> str:
    fourway = h1["fourway_table"]
    scatter = h1["corr_for_scatter"]
    q_results = h1["quartile_results"]
    lines = h1["conclusion_lines"]
    verdict = h1["verdict"]

    # 4-way table
    def _cell(r, p=None):
        if r is None or not np.isfinite(r): return '<td class="na">—</td>'
        star = _sig_star(p) if p is not None else ""
        return f'<td>{r:+.3f}<span class="star">{star}</span></td>'

    four_rows = []
    for row in fourway:
        four_rows.append(
            f"<tr><td class='lbl'>{_h(row['label'])}</td>"
            f"{_cell(row['raw_r'], row['raw_p'])}"
            f"{_cell(row['int_r'], row['int_p'])}"
            f"{_cell(row['early_r'], row['early_p'])}"
            f"{_cell(row['partial_r'], None)}"
            f"<td class='n'>{row['raw_n']}</td></tr>"
        )
    four_html = "\n".join(four_rows)

    scatter_imgs = []
    for c in scatter:
        if not np.isfinite(c.get("pearson_r", np.nan)): continue
        feat = c["feature"]
        label = feat.replace("c_", "centrality · ").replace("_", " ")
        b64 = chart_scatter(rows, feat, label, c)
        scatter_imgs.append(f'<figure><img src="data:image/png;base64,{b64}"/></figure>')
    scatter_html = "\n".join(scatter_imgs)

    q_rows = []
    for q in q_results:
        if q.get("insufficient"): continue
        f = q["feature"].replace("c_", "")
        q_rows.append(
            f"<tr><td class='lbl'>{f}</td>"
            f"<td>{q['n_top']}</td>"
            f"<td>{q['top_mean_rank']:.2f}</td>"
            f"<td>{q['bot_mean_rank']:.2f}</td>"
            f"<td>{q['top_mean_rank'] - q['bot_mean_rank']:+.2f}</td>"
            f"<td>{q['welch_t']:+.2f}</td>"
            f"<td>{_fmt_p(q['welch_p'])}<span class='star'>{_sig_star(q['welch_p'])}</span></td>"
            f"</tr>"
        )
    q_html = "\n".join(q_rows) or '<tr><td colspan="7" class="na">—</td></tr>'

    lines_html = "".join(f"<li>{_h(l)}</li>" for l in lines)

    return f"""
<section id="h1">
  <div class="section-label">가설 1</div>
  <h2>교전 참여도 ↔ 최종 순위</h2>
  <p class="lede">
    "공격형 플레이가 유리한가, 파밍형이 유리한가?" 를 네 가지 방법으로 동시
    검증한다. 단순 교전 횟수(degree) 는 생존 시간과 얽혀 있으므로,
    분당 강도 · 초반 5분 구간 · 편상관 세 가지 방식으로 생존 편향을 통제한다.
  </p>

  <aside class="abstract {verdict}">
    <h3>요약 {verdict_badge(verdict)}</h3>
    <ol>{lines_html}</ol>
  </aside>

  <h3>4-축 상관 비교</h3>
  {interp_box(
      "<b>원시 r</b> 은 생존 편향을 포함. <b>강도 r</b> 은 분당 교전 "
      "(생존 길이에 독립). <b>초반 r</b> 은 매치 첫 5분 구간만 사용 (모든 "
      "팀이 살아있는 시점). <b>편상관</b> 은 생존시간을 회귀로 제거한 순수 "
      "효과. 네 축이 일치하면 진짜 신호, 후자 세 축이 0 부근이면 원시 상관은 "
      "생존 편향의 산물."
  )}
  <p class="eq">
    \\[
      r_{{xy \\cdot z}} =
        \\frac{{r_{{xy}} - r_{{xz}}\\, r_{{yz}}}}
             {{\\sqrt{{(1 - r_{{xz}}^{{2}})\\,(1 - r_{{yz}}^{{2}})}}}}
    \\]
  </p>
  <table class="four">
    <thead><tr>
      <th>지표</th>
      <th>원시 r</th>
      <th>강도 r<br/><span class="sub">per min</span></th>
      <th>초반 r<br/><span class="sub">\\( t \\le \\) 윈도우</span></th>
      <th>편상관 r<br/><span class="sub">생존시간 통제</span></th>
      <th>n</th>
    </tr></thead>
    <tbody>{four_html}</tbody>
  </table>

  <h3>상위 25% vs 하위 25%</h3>
  <table class="stat">
    <thead><tr>
      <th>지표</th><th>n/그룹</th>
      <th>상위 평균순위</th><th>하위 평균순위</th><th>\\( \\Delta \\)</th>
      <th>Welch \\( t \\)</th><th>Welch \\( p \\)</th>
    </tr></thead>
    <tbody>{q_html}</tbody>
  </table>

  <h3>산점도 · 회귀선 (원시)</h3>
  <div class="scatter-grid">{scatter_html}</div>
</section>
"""


def render_h2(rows, h2) -> str:
    verdict = h2.get("verdict", "inconclusive")
    lines = h2.get("conclusion_lines", [])
    corr_rows = h2["corr_rows"]
    group_stats = h2["group_stats"]
    kw = h2["kruskal"]
    pairwise = h2["pairwise"]

    # 상관 테이블
    corr_html = "\n".join(
        f"<tr><td class='lbl'>{_h(c['label'])}</td>"
        f"<td>{_fmt_r(c['raw_r'])}<span class='star'>{_sig_star(c['raw_p'])}</span></td>"
        f"<td>{_fmt_p(c['raw_p'])}</td>"
        f"<td>{_fmt_r(c['partial_r'])}</td>"
        f"<td class='n'>{c['raw_n']}</td></tr>"
        for c in corr_rows
    )

    # 그룹 비교 — 차트 label 은 영문 bucket 키 (한글 폰트 경고 회피)
    means = [g["mean_rank"] if np.isfinite(g["mean_rank"]) else 0
             for g in group_stats]
    ns = [g["n"] for g in group_stats]
    bucket_chart = chart_bucket_bar(
        [g["bucket"] for g in group_stats],
        means, ns,
        "mean final_rank by community size bucket",
        "mean final_rank  (lower = better)",
    )

    grp_html = "\n".join(
        f"<tr><td class='lbl'>{_h(g['label_kr'])}</td>"
        f"<td class='n'>{g['n']}</td>"
        f"<td>{_fmt(g['mean_rank'], 2)}</td>"
        f"<td>{_fmt(g['median_rank'], 1)}</td>"
        f"<td>{_fmt(g['mean_surv_s']/60.0 if np.isfinite(g['mean_surv_s']) else float('nan'), 1)}</td>"
        f"</tr>"
        for g in group_stats
    )

    # pairwise rows
    pw_rows = []
    for key, pw in pairwise.items():
        a, _, b = key.partition("_vs_")
        pw_rows.append(
            f"<tr><td class='lbl'>{_h(a)} vs {_h(b)}</td>"
            f"<td>{pw.get('n_a', '—')} / {pw.get('n_b', '—')}</td>"
            f"<td>{_fmt(pw.get('mean_a'), 2)} / {_fmt(pw.get('mean_b'), 2)}</td>"
            f"<td>{_fmt(pw.get('diff'), 2)}</td>"
            f"<td>{_fmt_p(pw.get('mw_p'))}<span class='star'>{_sig_star(pw.get('mw_p'))}</span></td>"
            f"</tr>"
        )
    pw_html = "\n".join(pw_rows) or '<tr><td colspan="5" class="na">—</td></tr>'

    lines_html = "".join(f"<li>{_h(l)}</li>" for l in lines)

    return f"""
<section id="h2">
  <div class="section-label">가설 2</div>
  <h2>교전 밀집도 ↔ 최종 순위</h2>
  <p class="lede">
    "파밍만 한 팀(고립)과 대규모 교전지에 섞인 팀 중 누가 순위가 좋은가?"
    팀이 속한 Louvain 커뮤니티의 크기와 밀도로 분류해 평균 순위를 비교한다.
  </p>

  {interp_box(
      "<b>community_size = 1</b> (고립) 은 <i>파밍형</i>의 시그니처 — "
      "매치 내내 다른 팀과 교전이 없었던 팀. <b>community_size ≥ 6</b> 은 "
      "큰 지리적 교전지에 얽혀 있던 팀. 두 집단의 평균 순위를 직접 비교하면 "
      "공격형과 파밍형 중 어느 전략이 실제로 더 유리한지 판정할 수 있다."
  )}

  <aside class="abstract {verdict}">
    <h3>요약 {verdict_badge(verdict)}</h3>
    <ol>{lines_html}</ol>
  </aside>

  <h3>커뮤니티 지표 ↔ final_rank 상관</h3>
  <table class="stat">
    <thead><tr>
      <th>지표</th><th>Pearson r</th><th>p</th>
      <th>편상관 r<br/><span class="sub">생존시간 통제</span></th><th>n</th>
    </tr></thead>
    <tbody>{corr_html}</tbody>
  </table>

  <h3>커뮤니티 크기 버킷별 평균</h3>
  <figure>
    <img src="data:image/png;base64,{bucket_chart}"/>
    <figcaption>버킷별 평균 final_rank. 낮을수록 좋은 등수.</figcaption>
  </figure>
  <table class="stat">
    <thead><tr>
      <th>버킷</th><th>n</th><th>평균 순위</th>
      <th>중앙값 순위</th><th>평균 생존 (분)</th>
    </tr></thead>
    <tbody>{grp_html}</tbody>
  </table>
  <p class="note">
    Kruskal–Wallis (그룹 전체 차이): H = {_fmt(kw['statistic'], 2)},
    p = {_fmt_p(kw['p'])}
  </p>

  <h3>주요 쌍별 비교 (Mann–Whitney)</h3>
  <table class="stat">
    <thead><tr>
      <th>비교</th><th>n (a/b)</th>
      <th>평균순위 (a/b)</th><th>\\( \\Delta \\)</th><th>p</th>
    </tr></thead>
    <tbody>{pw_html}</tbody>
  </table>
</section>
"""


def render_h3(rows, h3) -> str:
    verdict = h3.get("verdict", "inconclusive")
    lines = h3.get("conclusion_lines", [])
    group_stats = h3["group_stats"]
    kw_rank = h3["kruskal_rank"]
    kw_surv = h3["kruskal_surv"]
    pairwise = h3["pairwise"]

    # group table
    grp_html = "\n".join(
        f"<tr><td class='lbl'>{_h(g['label_kr'])}</td>"
        f"<td class='n'>{g['n']}</td>"
        f"<td>{_fmt(g['mean_rank'], 2)}</td>"
        f"<td>{_fmt(g['median_rank'], 1)}</td>"
        f"<td>{_fmt(g['mean_surv_s']/60.0 if np.isfinite(g['mean_surv_s']) else float('nan'), 1)}</td>"
        f"</tr>"
        for g in group_stats
    )

    # bar charts
    labels = [g["pos"] for g in group_stats]
    rank_means = [g["mean_rank"] if np.isfinite(g["mean_rank"]) else 0
                  for g in group_stats]
    surv_means = [g["mean_surv_s"]/60.0 if np.isfinite(g["mean_surv_s"]) else 0
                  for g in group_stats]
    ns = [g["n"] for g in group_stats]
    chart_rank = chart_bucket_bar(labels, rank_means, ns,
                                   "mean final_rank by position",
                                   "mean final_rank")
    chart_surv = chart_bucket_bar(labels, surv_means, ns,
                                   "mean survival time by position",
                                   "mean survival (min)")

    # pairwise
    pw_rows = []
    for key, pw in pairwise.items():
        if "n_a" not in pw or pw.get("n_a", 0) < 3:
            continue
        a, _, b = key.partition("_vs_")
        pw_rows.append(
            f"<tr><td class='lbl'>{_h(a)} vs {_h(b)}</td>"
            f"<td>{pw.get('n_a', '—')} / {pw.get('n_b', '—')}</td>"
            f"<td>{_fmt(pw.get('rank_mean_a'), 2)} / {_fmt(pw.get('rank_mean_b'), 2)}</td>"
            f"<td>{_fmt_p(pw.get('rank_mw_p'))}<span class='star'>{_sig_star(pw.get('rank_mw_p'))}</span></td>"
            f"<td>{_fmt((pw.get('surv_mean_a') or 0)/60, 1)} / {_fmt((pw.get('surv_mean_b') or 0)/60, 1)}</td>"
            f"<td>{_fmt_p(pw.get('surv_mw_p'))}<span class='star'>{_sig_star(pw.get('surv_mw_p'))}</span></td>"
            f"</tr>"
        )
    pw_html = "\n".join(pw_rows) or '<tr><td colspan="6" class="na">—</td></tr>'

    lines_html = "".join(f"<li>{_h(l)}</li>" for l in lines)

    return f"""
<section id="h3">
  <div class="section-label">가설 3</div>
  <h2>외곽 노드 vs 중심 노드의 생존</h2>
  <p class="lede">
    교전 네트워크에서 중심에 있던 팀(core)과 주변부(peripheral) 혹은 아예
    교전이 없던 팀(isolated) 의 평균 순위와 생존 시간을 비교한다. 순위와
    생존시간은 서로 다른 이야기를 할 수 있다 — 오래 살아도 순위가 나쁠 수 있고
    반대도 가능.
  </p>

  {interp_box(
      "<b>core</b> = betweenness 상위 25% — 매치의 교전 중심지를 거쳐간 팀. "
      "<b>peripheral</b> = 하위 25% — 교전이 있었지만 주변부에 머문 팀. "
      "<b>isolated</b> = degree 0 — 매치 내내 다른 팀과 직접 교전이 없었던 팀 "
      "(파밍/존 밖 플레이)."
  )}

  <aside class="abstract {verdict}">
    <h3>요약 {verdict_badge(verdict)}</h3>
    <ol>{lines_html}</ol>
  </aside>

  <h3>포지션별 통계</h3>
  <table class="stat">
    <thead><tr>
      <th>포지션</th><th>n</th>
      <th>평균 순위</th><th>중앙값 순위</th><th>평균 생존 (분)</th>
    </tr></thead>
    <tbody>{grp_html}</tbody>
  </table>
  <p class="note">
    Kruskal–Wallis (순위): H = {_fmt(kw_rank['statistic'], 2)},
    p = {_fmt_p(kw_rank['p'])}  ·  (생존):
    H = {_fmt(kw_surv['statistic'], 2)}, p = {_fmt_p(kw_surv['p'])}
  </p>

  <div class="two-charts">
    <figure><img src="data:image/png;base64,{chart_rank}"/></figure>
    <figure><img src="data:image/png;base64,{chart_surv}"/></figure>
  </div>

  <h3>주요 쌍별 비교 (Mann–Whitney)</h3>
  <table class="stat">
    <thead><tr>
      <th>비교</th><th>n (a/b)</th>
      <th>평균 순위 (a/b)</th><th>순위 p</th>
      <th>평균 생존 분 (a/b)</th><th>생존 p</th>
    </tr></thead>
    <tbody>{pw_html}</tbody>
  </table>
</section>
"""


# ============================================================
# HTML template
# ============================================================

_HTML = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>팀 교전 네트워크 분석 · PUBG 스쿼드 매치</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Noto+Serif+KR:wght@400;500;600&family=Inter:wght@400;500;600&family=Noto+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<script>
  window.MathJax = {
    tex: { inlineMath: [['\\(','\\)']], displayMath: [['\\[','\\]']],
           processEscapes: true },
    svg: { fontCache: 'global' }
  };
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
:root {
  --bg:       #f5f2ec;
  --surface:  #ffffff;
  --paper:    #fefcf7;
  --ink:      #1a1a1a;
  --muted:    #6b6558;
  --rule:     #d8d1c2;
  --accent:   #b02a2a;
  --positive: #2d5f3f;
  --nav-bg:   #ece7da;
}
* { box-sizing: border-box; }
html, body { background: var(--bg); color: var(--ink); }
body {
  margin: 0;
  font-family: "Inter", "Noto Sans KR", -apple-system,
               "Apple SD Gothic Neo", sans-serif;
  font-size: 15px;
  line-height: 1.75;
  -webkit-font-smoothing: antialiased;
  word-break: keep-all;
}

/* Layout: sidebar + content */
.layout { display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  min-height: 100vh;
}
.sidenav {
  position: sticky; top: 0;
  height: 100vh; overflow-y: auto;
  background: var(--nav-bg);
  border-right: 1px solid var(--rule);
  padding: 36px 22px;
}
.nav-title {
  font-family: "Lora", "Noto Serif KR", serif;
  font-size: 18px; font-weight: 600;
  color: var(--ink); line-height: 1.2;
}
.nav-sub {
  font-size: 12px; color: var(--muted);
  margin-top: 4px; margin-bottom: 28px;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.nav-list { display: flex; flex-direction: column; gap: 2px; }
.nav-item {
  display: block;
  padding: 8px 12px;
  color: var(--muted);
  text-decoration: none;
  border-left: 2px solid transparent;
  font-size: 13.5px;
  transition: color 0.15s, border-color 0.15s;
}
.nav-item:hover { color: var(--ink); }
.nav-item.active {
  color: var(--ink); border-left-color: var(--accent);
  font-weight: 500;
}
.nav-foot {
  margin-top: 36px;
  padding-top: 18px;
  border-top: 1px solid var(--rule);
  font-size: 11.5px;
  color: var(--muted);
  line-height: 1.5;
}

.content {
  max-width: 880px;
  padding: 64px 48px 140px;
  margin: 0 auto;
}

/* Typography */
h1, h2, h3 {
  font-family: "Lora", "Noto Serif KR", "Charter", Georgia, serif;
  color: var(--ink);
  font-weight: 600;
  line-height: 1.25;
  letter-spacing: -0.01em;
}
h1.title { font-size: 38px; margin: 6px 0 16px; letter-spacing: -0.015em; }
h2 { font-size: 26px; margin: 12px 0 10px; }
h3 { font-size: 17px; margin: 24px 0 8px; }
p { margin: 10px 0; }
code {
  font-family: "IBM Plex Mono", "SF Mono", Menlo, monospace;
  font-size: 0.92em;
  background: var(--paper);
  padding: 1px 6px;
  border: 1px solid var(--rule);
  border-radius: 3px;
}
.lede {
  font-family: "Lora", "Noto Serif KR", serif;
  font-size: 18px; font-style: italic;
  color: var(--muted);
  line-height: 1.6;
  margin: 14px 0 24px;
  max-width: 680px;
}

/* Sections */
section {
  margin: 72px 0;
  padding-top: 40px;
  border-top: 1px solid var(--rule);
}
section:first-of-type { border-top: none; padding-top: 0; margin-top: 0; }
.section-label {
  font-family: "Inter", sans-serif;
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
}

/* Meta row (overview) */
.meta-row {
  display: flex; gap: 28px; flex-wrap: wrap;
  font-size: 13px; color: var(--muted);
  margin: 16px 0 36px;
  padding: 16px 0;
  border-top: 2px solid var(--ink);
  border-bottom: 1px solid var(--rule);
}
.meta-row b { color: var(--ink); font-weight: 600;
              font-variant-numeric: tabular-nums; }

/* Cards */
.cards {
  display: grid;
  grid-template-columns: 1fr;
  gap: 14px;
  margin: 20px 0;
}
.card {
  display: block;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 20px 24px;
  text-decoration: none;
  color: var(--ink);
  transition: border-color 0.15s, transform 0.15s;
}
.card:hover {
  border-color: var(--ink);
  transform: translateY(-1px);
}
.card-badge { margin-bottom: 8px; }
.card h3 { margin: 4px 0 8px; font-size: 18px; }
.card-line { margin: 0; color: var(--muted); font-size: 13.5px; }

/* Badges */
.badge {
  display: inline-block;
  font-family: "Inter", sans-serif;
  font-size: 11px;
  letter-spacing: 0.06em;
  padding: 3px 10px;
  border-radius: 11px;
  font-weight: 500;
  border: 1px solid;
}
.badge.ok  { color: var(--positive); border-color: var(--positive); }
.badge.bad { color: var(--accent);   border-color: var(--accent); }
.badge.mid { color: var(--muted);    border-color: var(--muted); }

/* Figures */
figure {
  margin: 20px 0;
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 2px;
  padding: 18px;
}
figure img { display: block; width: 100%; height: auto; }
figcaption {
  margin-top: 10px;
  color: var(--muted);
  font-size: 13px;
  font-style: italic;
}
.two-charts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

/* Tables */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 14px 0;
  font-variant-numeric: tabular-nums;
}
table th, table td {
  padding: 8px 12px;
  text-align: left;
  border-bottom: 1px solid var(--rule);
  font-size: 13.5px;
}
table th {
  color: var(--muted);
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 10.5px;
  border-bottom: 1.5px solid var(--ink);
}
table th .sub {
  font-weight: 400;
  text-transform: none;
  letter-spacing: 0;
}
table td.lbl { font-weight: 500; }
table td.n { color: var(--muted); }
table td.na { color: var(--muted); }
table td.math { font-family: "Lora", serif; }
table.four td { text-align: right; }
table.four td.lbl { text-align: left; }
table.stat td.interp-inline {
  color: var(--muted);
  font-size: 12.5px;
  line-height: 1.55;
  font-style: italic;
  max-width: 420px;
}
table.interp-table th:last-child,
table.interp-table td:last-child { width: 45%; }
.star { color: var(--accent); margin-left: 2px; }

/* Interpretation boxes */
.interp {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--accent);
  border-radius: 2px;
  padding: 14px 20px;
  margin: 16px 0;
}
.interp-label {
  font-size: 10.5px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 500;
  margin-bottom: 6px;
}
.interp-body { font-size: 13.5px; color: var(--ink); line-height: 1.7; }
.interp-body b { color: var(--ink); }
.interp-body i { color: var(--muted); }

/* Abstract */
.abstract {
  border-left: 3px solid var(--muted);
  padding: 20px 28px;
  margin: 20px 0 30px;
  background: var(--surface);
  border-radius: 2px;
}
.abstract.confirmed, .abstract.core_better,
.abstract.isolated_better, .abstract.periphery_better,
.abstract.dense_better { border-left-color: var(--positive); }
.abstract.rejected { border-left-color: var(--accent); }
.abstract h3 { margin: 0 0 10px; font-size: 16px; }
.abstract ol { margin: 0; padding-left: 20px; }
.abstract li { margin: 6px 0; }

/* Notes / equation block */
.note { font-size: 13px; color: var(--muted); font-style: italic; }
.eq {
  background: var(--paper);
  border: 1px solid var(--rule);
  padding: 12px 18px;
  border-radius: 2px;
  font-size: 14.5px;
  color: var(--ink);
  margin: 14px 0;
  overflow-x: auto;
  text-align: center;
}
mjx-container[display="true"] { margin: 6px 0 !important; }
table td.math mjx-container { margin: 0 !important; }

.scatter-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

/* Explorer (interactive match graph) */
.explorer-ctrl {
  display: flex;
  gap: 18px;
  margin: 18px 0 10px;
  flex-wrap: wrap;
  align-items: center;
}
.ctrl-group {
  display: flex; align-items: center; gap: 8px;
  font-size: 12px; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.08em;
}
.ctrl-label { font-weight: 500; }
#match-select {
  background: var(--surface);
  border: 1px solid var(--rule);
  color: var(--ink);
  font-family: inherit;
  font-size: 13px;
  padding: 7px 12px;
  border-radius: 2px;
  cursor: pointer;
  min-width: 280px;
}
#match-select:focus { outline: 2px solid var(--accent); outline-offset: -1px; }

.view-buttons {
  display: inline-flex;
  border: 1px solid var(--rule);
  border-radius: 2px;
  overflow: hidden;
}
.view-buttons button {
  background: transparent;
  border: 0;
  border-right: 1px solid var(--rule);
  color: var(--muted);
  padding: 7px 16px;
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
  letter-spacing: 0.04em;
  transition: background 0.15s, color 0.15s;
}
.view-buttons button:last-child { border-right: 0; }
.view-buttons button:hover { color: var(--ink); }
.view-buttons button.active {
  background: var(--ink);
  color: var(--paper);
}

.explorer-stats {
  font-size: 12.5px;
  color: var(--muted);
  margin-left: auto;
  font-variant-numeric: tabular-nums;
}
.explorer-stats b { color: var(--ink); font-weight: 600; }

.explorer-legend {
  font-size: 12px;
  color: var(--muted);
  margin: 6px 0 10px;
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
}
.legend-hint { letter-spacing: 0.02em; }
.legend-item {
  display: inline-flex; align-items: center; gap: 4px;
  color: var(--ink);
}
.sw {
  display: inline-block; width: 10px; height: 10px;
  border-radius: 50%; border: 1px solid rgba(0,0,0,0.15);
  vertical-align: middle;
}
.sw.pos-core { background: #b02a2a; }
.sw.pos-mid  { background: #6b6558; }
.sw.pos-peri { background: #b8b0a0; }
.sw.pos-iso  { background: #d8d1c2; }

.explorer-canvas {
  position: relative;
  background: #0f1115;
  border: 1px solid var(--rule);
  border-radius: 2px;
  margin: 8px 0 0;
  overflow: hidden;
}
#xp-graph {
  width: 100%;
  height: 620px;
  display: block;
  cursor: grab;
  background: #0f1115;
}
#xp-graph:active { cursor: grabbing; }

.xp-zoom-ctrl {
  position: absolute;
  top: 10px; left: 10px;
  display: flex; flex-direction: column; gap: 4px;
  z-index: 3;
}
.xp-zoom-ctrl button {
  width: 32px; height: 32px;
  background: rgba(254, 252, 247, 0.95);
  border: 1px solid var(--rule);
  color: var(--ink);
  font-family: inherit;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  border-radius: 2px;
  padding: 0;
  line-height: 1;
  transition: background 0.12s;
}
.xp-zoom-ctrl button:hover {
  background: var(--accent);
  color: var(--paper);
  border-color: var(--accent);
}

.m-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
  margin: 12px 0 18px;
}
.m-card {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--muted);
  padding: 14px 18px;
  border-radius: 2px;
}
.m-card h4 { margin: 0 0 6px; font-size: 14px;
             font-family: "Lora", "Noto Serif KR", serif; }
.m-card p { margin: 0; font-size: 13px; line-height: 1.6; }
.m-card.ally   { border-left-color: #14b8a6; }
.m-card.combat { border-left-color: var(--accent); }
@media (max-width: 720px) { .m-grid { grid-template-columns: 1fr; } }

.level-buttons {
  display: inline-flex;
  border: 1px solid var(--rule);
  border-radius: 2px;
  overflow: hidden;
  margin-left: 8px;
}
.level-buttons button {
  background: transparent;
  border: 0;
  border-left: 1px solid var(--rule);
  padding: 4px 12px;
  font: 500 11.5px "Inter", sans-serif;
  color: var(--muted);
  cursor: pointer;
  letter-spacing: 0.04em;
}
.level-buttons button:first-child { border-left: 0; }
.level-buttons button.active {
  background: var(--ink);
  color: var(--paper);
}
#xp-grid line { stroke: #2b313a; opacity: 0.6; }
#xp-grid text { fill: #5a6270; }
#xp-zone circle { fill: rgba(96,165,250,0.08); stroke: rgba(96,165,250,0.5);
                  stroke-width: 1.2; stroke-dasharray: 5 4; }
#xp-ally line { stroke: #14b8a6; stroke-opacity: 0.55; stroke-width: 0.9; }
#xp-combat line { stroke: #b02a2a; stroke-linecap: round; }
#xp-hulls path { transition: fill-opacity 0.25s ease, stroke-opacity 0.25s ease; }
#xp-edges line { transition: stroke-opacity 0.25s ease; }
#xp-nodes circle { transition: r 0.18s ease, fill 0.2s ease, fill-opacity 0.2s ease; }

.xp-tooltip {
  position: absolute;
  background: var(--surface);
  border: 1px solid var(--ink);
  border-radius: 2px;
  padding: 8px 12px;
  font-size: 12px;
  line-height: 1.5;
  color: var(--ink);
  pointer-events: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  z-index: 4;
  max-width: 240px;
}
.xp-tooltip b { color: var(--ink); }
.xp-tooltip i { color: var(--accent); font-style: normal; }

/* Timeline controls */
.xp-timeline {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 14px 0 0;
  padding: 10px 14px;
  border: 1px solid var(--rule);
  border-radius: 2px;
  background: var(--surface);
  flex-wrap: wrap;
}
.xp-timeline #xp-play {
  background: var(--ink); color: var(--paper);
  border: 0; border-radius: 2px;
  padding: 6px 14px; cursor: pointer;
  font-size: 13px; font-family: inherit;
  min-width: 44px;
}
.xp-timeline #xp-play:hover { background: var(--accent); }
.xp-timeline #xp-slider {
  flex: 1; min-width: 240px;
  -webkit-appearance: none; appearance: none;
  height: 3px; background: var(--rule); border-radius: 2px;
  outline: none; cursor: pointer;
}
.xp-timeline #xp-slider::-webkit-slider-thumb {
  -webkit-appearance: none; width: 14px; height: 14px;
  background: var(--accent); border-radius: 50%;
  cursor: pointer; box-shadow: 0 0 0 3px rgba(176,42,42,0.15);
}
.xp-timeline #xp-slider::-moz-range-thumb {
  width: 14px; height: 14px;
  background: var(--accent); border-radius: 50%; border: 0;
  cursor: pointer;
}
.xp-time {
  font-family: "IBM Plex Mono", "SF Mono", monospace;
  font-size: 12px;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
  min-width: 110px;
  text-align: center;
}
.xp-speed {
  display: flex; align-items: center; gap: 6px;
  font-size: 11px; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em;
}
.xp-speed select {
  background: transparent; border: 1px solid var(--rule);
  font-family: inherit; font-size: 12px;
  padding: 3px 6px; border-radius: 2px; cursor: pointer;
}
.xp-frameinfo {
  font-family: "IBM Plex Mono", "SF Mono", monospace;
  font-size: 11px; color: var(--muted);
  font-variant-numeric: tabular-nums;
}

.colophon {
  margin-top: 96px;
  padding-top: 20px;
  border-top: 1px solid var(--rule);
  font-size: 11.5px;
  color: var(--muted);
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  letter-spacing: 0.04em;
}

@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .sidenav { position: static; height: auto;
             border-right: none; border-bottom: 1px solid var(--rule); }
  .content { padding: 32px 22px 100px; }
  .scatter-grid, .two-charts { grid-template-columns: 1fr; }
}
</style></head>
<body>
<div class="layout">
  __NAV__
  <main class="content">
    __OVERVIEW__
    __METHODS__
    __EXPLORER__
    __BASICS__
    __CENTR_DEFS__
    __COMMUNITY__
    __H1__
    __H2__
    __H3__
    <footer class="colophon">
      <span>생성 __GEN__</span>
      <span>매치 __N_MATCHES__건 · 팀 __N_TEAMS__행 · __EARLY__</span>
    </footer>
  </main>
</div>

<script>
// scroll-spy: 현재 보이는 섹션의 nav item 활성화
(function() {
  const sections = document.querySelectorAll('main.content section[id]');
  const links = document.querySelectorAll('.nav-item');
  const map = {};
  links.forEach(a => { map[a.dataset.sec] = a; });

  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(a => a.classList.remove('active'));
        const a = map[e.target.id];
        if (a) a.classList.add('active');
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px', threshold: 0 });
  sections.forEach(s => io.observe(s));
})();
</script>
</body></html>
"""


# ============================================================
# Public entry
# ============================================================

def write_report(collected: dict, analyses: dict, out_path: str) -> str:
    rows = collected["rows"]
    globals_list = collected["globals"]
    adjs = collected["adjs"]
    pks = collected["pks"]
    communities = collected["communities"]
    n_matches = len(collected["match_ids"])
    n_teams = len(rows)
    early_sec = collected.get("early_window_sec")
    early_str = f"초반 윈도우 {int(early_sec)}초" if early_sec else "초반 분석 생략"

    h1 = analyses["h1"]
    h2 = analyses["h2"]
    h3 = analyses["h3"]

    nav = render_nav([
        ("overview",        "개요"),
        ("methods",         "Methods reference"),
        ("explorer",        "매치 탐색기"),
        ("basics",          "네트워크 기초"),
        ("centrality-defs", "중심성 정의"),
        ("community",       "커뮤니티 구조"),
        ("h1",              "가설 1 · 교전 참여도"),
        ("h2",              "가설 2 · 교전 밀집도"),
        ("h3",              "가설 3 · 외곽 vs 중심"),
    ])

    overview = render_overview(
        hero=h1["hero"], h2=h2, h3=h3,
        n_matches=n_matches, n_teams=n_teams, early_sec=early_sec,
    )
    explorer_html = render_explorer(analyses.get("explorer_matches", []))
    methods_html  = render_methods()
    basics = render_basics(adjs, pks, globals_list)
    centr_defs = render_centrality_defs()
    comm = render_community_section(communities)
    h1_html = render_h1(rows, h1)
    h2_html = render_h2(rows, h2)
    h3_html = render_h3(rows, h3)

    gen_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = _HTML
    html = html.replace("__NAV__", nav)
    html = html.replace("__OVERVIEW__", overview)
    html = html.replace("__METHODS__",  methods_html)
    html = html.replace("__EXPLORER__", explorer_html)
    html = html.replace("__BASICS__", basics)
    html = html.replace("__CENTR_DEFS__", centr_defs)
    html = html.replace("__COMMUNITY__", comm)
    html = html.replace("__H1__", h1_html)
    html = html.replace("__H2__", h2_html)
    html = html.replace("__H3__", h3_html)
    html = html.replace("__GEN__", gen_at)
    html = html.replace("__N_MATCHES__", str(n_matches))
    html = html.replace("__N_TEAMS__", str(n_teams))
    html = html.replace("__EARLY__", early_str)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
