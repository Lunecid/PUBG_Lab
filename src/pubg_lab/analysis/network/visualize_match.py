"""
단일 매치 인터랙티브 HTML 시각화 (v4 · 노드 안정화 + 팀 식별 + 맵 + Pan/Zoom/Rotate)
=====================================================================================
v3 대비 변경:
  1) 노드 크기를 매치 전체 centrality p05/p95 기준으로 normalization → 팀 움직임에 흔들리지 않음
  2) Team Roster 패널 + 상시 centroid 라벨로 팀 식별 명확화
  3) PUBG 맵 이미지(assets/Baltic_Main.jpg) base64 인라인 임베드, 없으면 gracefully 스킵
  4) Pan/Zoom/Rotate — wheel(zoom-to-cursor), drag(pan), Shift+drag(회전), 버튼(±90°, Reset)
"""

import os
import json
import base64
import numpy as np

from pubg_lab.analysis.network.config import (
    CENTRALITY_MEASURES,
    TOP_K,
    MAP_IMAGE_PATH,
    MAP_IMAGE_URLS,
    MAP_ROTATE_DEG,
    MAP_OPACITY_DEFAULT,
)


_TEAM_PALETTE = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#ecf0f1",
    "#16a085",
    "#d35400",
    "#c0392b",
    "#2980b9",
    "#27ae60",
    "#f1c40f",
    "#8e44ad",
    "#95a5a6",
    "#ff6b9d",
    "#5cdb95",
    "#f9ca24",
    "#a29bfe",
    "#00a8cc",
    "#60a5fa",
    "#fb923c",
    "#c084fc",
    "#34d399",
]


def _team_color(team_idx):
    return _TEAM_PALETTE[team_idx % len(_TEAM_PALETTE)]


def _compute_cnorm(frames, measures):
    """각 measure 의 p05/p95 를 전체 프레임에 걸쳐 계산."""
    out = {}
    for m in measures:
        vals = []
        for fr in frames:
            arr = fr["metrics"]["centrality"].get(m, [])
            for v in arr:
                if v is not None:
                    vals.append(float(v))
        if len(vals) >= 2:
            lo, hi = np.percentile(vals, [5, 95])
        else:
            lo, hi = 0.0, 1.0
        out[m] = {"lo": float(lo), "hi": float(hi)}
    return out


def _load_map_base64(path=MAP_IMAGE_PATH):
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(
        ext.lstrip("."), "jpeg"
    )
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def _build_payload(match_data):
    static = match_data["static"]
    players = [
        {
            "idx": p["idx"],
            "aid": p["account_id"],
            "s": p["short"],
            "ti": p["team_idx"],
            "tid": str(p["team_id"]),
        }
        for p in static["players"]
    ]

    teams = []
    for t in static["teams"]:
        teams.append(
            {
                "idx": t["idx"],
                "tid": str(t["team_id"]),
                "rank": t["rank"],
                "color": _team_color(t["idx"]),
                "members": t["members"],
                "size": len(t["members"]),
            }
        )

    frames = []
    for fr in match_data["frames"]:
        metr = fr["metrics"]
        centr_c = {}
        for k, arr in metr["centrality"].items():
            centr_c[k] = [None if v is None else round(float(v), 4) for v in arr]
        top_c = {
            k: [[int(i), round(float(v), 4)] for i, v in lst] for k, lst in metr["top_rank"].items()
        }

        frames.append(
            {
                "t": fr["t"],
                "am": fr["alive_mask"],
                "pos": [[round(x, 1), round(y, 1)] for x, y in fr["positions"]],
                "hp": [round(v, 2) for v in fr["hp"]],
                "tac": fr["team_alive_count"],
                "tc": [
                    None if c is None else [round(c[0], 1), round(c[1], 1)]
                    for c in fr["team_centroid"]
                ],
                "ea": fr["edges"]["ally"],
                "ee": fr["edges"]["enemy"],
                "ec": [[int(s), int(d), round(float(w), 1)] for s, d, w in fr["edges"]["combat"]],
                "zn": {k: round(v, 1) for k, v in fr["zone"].items()},
                "mg": {
                    "nn": metr["global"]["n_nodes"],
                    "ne": metr["global"]["n_edges"],
                    "dn": round(metr["global"]["density"], 4),
                    "cc": round(metr["global"]["avg_clustering"], 4),
                    "ad": round(metr["global"]["avg_degree"], 3),
                    "dm": metr["global"]["diameter"],
                    "pl": round(metr["global"]["avg_path_length"], 3),
                    "gs": metr["global"]["giant_size"],
                    "nc": metr["global"]["n_components"],
                },
                "dd": metr["degree_dist"],
                "ce": centr_c,
                "tr": top_c,
                "cb": {
                    "in": metr["combat"]["combat_in_degree"],
                    "out": metr["combat"]["combat_out_degree"],
                    "dd": [round(v, 1) for v in metr["combat"]["dmg_dealt_window"]],
                    "dt": [round(v, 1) for v in metr["combat"]["dmg_taken_window"]],
                },
                # Community (CH5): 노드별 할당 + 요약
                "cm": _build_community_entry(metr.get("community", {})),
                # Narrative: 프레임별 해설
                "nr": fr.get("narrative", {}),
            }
        )

    cnorm = _compute_cnorm(match_data["frames"], CENTRALITY_MEASURES)
    map_b64 = _load_map_base64()  # 로컬 파일 있으면 base64 (완전 오프라인용)

    # Match-level timeseries (compute_all_frames 가 채워둠)
    ts = match_data.get("timeseries", {})

    return {
        "meta": {
            "match_id": static["match_id"],
            "map": static["map_name"],
            "mode": static["game_mode"],
            "arena": static["arena_size_m"],
            "duration": static["duration_s"],
            "n_players": static["n_players"],
            "n_teams": static["n_teams"],
            "map_opacity_default": MAP_OPACITY_DEFAULT,
            "map_rotate": MAP_ROTATE_DEG,
        },
        "players": players,
        "teams": teams,
        "frames": frames,
        "cnorm": cnorm,
        "ts": ts,  # 매치 레벨 시계열
        "map_b64": map_b64,  # 로컬 base64 (없으면 None)
        "map_urls": list(MAP_IMAGE_URLS),  # CDN fallback chain
    }


def _build_community_entry(community: dict) -> dict:
    """frame["metrics"]["community"] → HTML payload 의 짧은 키 구조."""
    lv = community.get("louvain", {}) if community else {}
    sp = community.get("spectral", {}) if community else {}
    return {
        "l": {
            "a": lv.get("assignments", []),
            "s": lv.get("sizes", []),
            "n": int(lv.get("n_communities", 0)),
            "q": round(float(lv.get("modularity_q", 0.0)), 4),
        },
        "s": {
            "a": sp.get("assignments", []),
            "s": sp.get("sizes", []),
            "n": int(sp.get("n_communities", 0)),
            "q": round(float(sp.get("modularity_q", 0.0)), 4),
            "k": int(sp.get("k_used", 0)),
        },
    }


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>PUBG Network Analysis — __MATCH_SHORT__</title>
<style>
:root {
  --bg: #1c1f24; --panel: rgba(28,31,36,0.92); --panel-solid: #23272e;
  --border: #3a3f47; --text: #e5e7eb; --muted: #9aa1ab;
  --accent: #60a5fa; --accent2: #3b82f6;
  --red: #ef4444; --amber: #f59e0b; --green: #34d399; --purple: #a78bfa;
  --arena: #0f1115;
  --inv-s: 1;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { width: 100%; height: 100%; }
body {
  background: var(--bg); color: var(--text);
  font-family: "SF Mono", Menlo, "Fira Code", monospace;
  font-size: 12px; overflow: hidden;
}
.panel {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 14px;
  backdrop-filter: blur(8px);
  line-height: 1.65; box-shadow: 0 4px 20px rgba(0,0,0,0.35);
}
h3.sec { color: var(--accent); font-size: 11px; margin: 9px 0 5px;
         border-bottom: 1px solid var(--border); padding-bottom: 3px;
         letter-spacing: 0.04em; text-transform: uppercase; font-weight: 600; }
h3.sec:first-child { margin-top: 0; }

/* ── LEFT: HUD ── */
#hud { position: absolute; top: 12px; left: 12px; width: 210px; }
#hud .row { display: flex; justify-content: space-between; gap: 14px; }
#hud .lbl { color: var(--muted); font-size: 11px; }
#hud .val { font-weight: 600; color: var(--text); }
#hud .big { font-size: 22px; font-weight: 700; color: var(--accent);
            line-height: 1.1; margin: 2px 0 6px; }

/* ── RIGHT: merged control+metrics panel ── */
#side { position: absolute; top: 12px; right: 12px; bottom: 260px;
        width: 300px; overflow-y: auto; padding: 12px 14px; }
#side::-webkit-scrollbar { width: 6px; }
#side::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
.btn {
  background: #2a2e36; border: 1px solid var(--border);
  color: var(--muted); border-radius: 5px;
  padding: 4px 9px; cursor: pointer; font-size: 10.5px;
  font-family: inherit; margin: 0 3px 4px 0; transition: all 0.12s;
}
.btn:hover { color: var(--text); border-color: var(--accent); }
.btn.on { background: var(--accent2); border-color: var(--accent2); color: #fff; font-weight: 600; }
.mrow { display: flex; justify-content: space-between; padding: 2px 0; font-size: 11px; }
.ml { color: var(--muted); }
.mv { font-weight: 600; font-variant-numeric: tabular-nums; }
.topk-row, .roster-row { display: flex; justify-content: space-between; padding: 3px 2px;
            border-bottom: 1px dashed #2d323a; font-size: 10.5px; cursor: pointer; gap: 6px; }
.topk-row:last-child, .roster-row:last-child { border-bottom: none; }
.topk-row .rk, .roster-row .rk { color: var(--muted); width: 18px; flex: none; }
.topk-row .nm, .roster-row .nm { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.topk-row .vl, .roster-row .vl { font-weight: 600; font-variant-numeric: tabular-nums; flex: none; }
.roster-row.on { background: rgba(96,165,250,0.15); }
.roster-row.dead { opacity: 0.35; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
       margin-right: 5px; vertical-align: middle; border: 1px solid rgba(0,0,0,0.3); }
.select-m {
  background: #2a2e36; border: 1px solid var(--border); color: var(--text);
  font-family: inherit; font-size: 10.5px; padding: 2px 5px;
  border-radius: 4px; margin-left: 4px;
}

/* ── CENTER: arena ── */
#arena-wrap { position: absolute; top: 12px; left: 232px; right: 322px; bottom: 260px;
              background: var(--arena); border-radius: 10px; overflow: hidden;
              border: 1px solid var(--border); box-shadow: 0 4px 20px rgba(0,0,0,0.35); }
#arena { width: 100%; height: 100%; display: block; cursor: grab; user-select: none;
         touch-action: none; }
#arena.dragging { cursor: grabbing; }

/* label font size counter-scales with world */
#g-labels text { font-size: calc(10px * var(--inv-s)); }
#g-labels .team-label { font-size: calc(9px * var(--inv-s)); }

/* ── BOTTOM controls (two rows) ── */
#controls { position: absolute; bottom: 0; left: 0; right: 0;
            display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
            padding: 8px 14px; background: var(--panel-solid);
            border-top: 1px solid var(--border); min-height: 62px; }
#controls button, #controls-row2 button {
  background: #2a2e36; color: var(--muted);
  border: 1px solid var(--border); border-radius: 5px;
  padding: 4px 10px; cursor: pointer; font-size: 11px;
  font-family: inherit; transition: all 0.12s;
}
#controls button:hover { background: #353a44; color: var(--text); }
#controls button.on { background: var(--accent2); border-color: var(--accent2); color: #fff; }
#timeline { flex: 1; min-width: 180px; height: 5px; -webkit-appearance: none; appearance: none;
            background: #2d323a; border-radius: 3px; cursor: pointer; }
#timeline::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px;
                                  background: var(--accent); border-radius: 50%;
                                  cursor: pointer; box-shadow: 0 0 4px rgba(96,165,250,0.6); }
#tstamp { min-width: 96px; text-align: center; font-size: 11px; color: var(--muted);
          font-variant-numeric: tabular-nums; }
.ctrl-sep { width: 1px; height: 18px; background: var(--border); margin: 0 2px; }
.ctrl-group { display: flex; align-items: center; gap: 6px; }
#op-slider { width: 80px; vertical-align: middle; }

/* ── Tooltip ── */
#tip { position: absolute; pointer-events: none;
       background: rgba(15,17,21,0.96); color: var(--text);
       padding: 7px 11px; border-radius: 6px; font-size: 10.5px;
       display: none; z-index: 99; white-space: nowrap;
       border: 1px solid var(--border); box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
#tip b { color: var(--accent); }

#legend-inline { position: absolute; left: 14px; bottom: 14px;
                 color: #cbd5e1; font-size: 10px;
                 background: rgba(15,17,21,0.75); padding: 7px 10px;
                 border-radius: 6px; line-height: 1.7;
                 border: 1px solid rgba(255,255,255,0.08); pointer-events: none; }
#legend-inline .lg { display: flex; align-items: center; gap: 6px; }
#legend-inline .ll { width: 22px; height: 2px; border-radius: 1px; display: inline-block; }
#legend-inline .ld { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

#subtitle { position: absolute; right: 14px; bottom: 14px; color: #94a3b8;
            font-size: 10px; text-align: right;
            background: rgba(15,17,21,0.75); padding: 5px 9px; border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.08); pointer-events: none; }

#header { position: absolute; left: 232px; right: 322px; top: 0;
          text-align: center; padding: 4px 10px;
          color: var(--muted); font-size: 10.5px;
          pointer-events: none; }
#header b { color: var(--text); }

/* ── BOTTOM: time-series charts (매치 전체 추이) ── */
#timeseries { position: absolute; left: 12px; right: 12px; bottom: 72px;
              height: 176px;
              display: grid; grid-template-columns: repeat(6, 1fr); gap: 6px;
              padding: 8px 10px;
              background: var(--panel-solid);
              border: 1px solid var(--border); border-radius: 8px;
              box-shadow: 0 -2px 10px rgba(0,0,0,0.25); }
.ts-chart { background: #1a1d22; border: 1px solid var(--border);
            border-radius: 6px; padding: 4px 6px 6px; overflow: hidden;
            display: flex; flex-direction: column; }
.ts-chart .ts-title { font-size: 10px; color: var(--muted);
                      margin-bottom: 2px; white-space: nowrap;
                      overflow: hidden; text-overflow: ellipsis;
                      border-bottom: 1px solid #2d323a; padding-bottom: 2px; }
.ts-chart svg { width: 100%; flex: 1; display: block; min-height: 0; }
.ts-line { fill: none; stroke-width: 1.4; }
.ts-now  { stroke: #fca5a5; stroke-width: 1.3; stroke-dasharray: 3 3; }
.ts-axis { stroke: #2d323a; stroke-width: 0.5; }
.ts-val  { font-size: 9px; fill: var(--muted);
           font-variant-numeric: tabular-nums; }
.ts-legend { display: flex; gap: 6px; font-size: 9px; color: var(--muted);
             padding: 1px 0 0; flex-wrap: wrap; }
.ts-legend span { display: inline-flex; align-items: center; gap: 3px; }
.ts-legend i { width: 10px; height: 2px; display: inline-block;
               border-radius: 1px; }

/* ── Narrative panel (프레임별 해설) ── */
.nr-panel { font-size: 10.5px; line-height: 1.55; }
.nr-summary { font-size: 11.5px; font-weight: 700; color: var(--accent);
              padding: 5px 0; border-bottom: 1px solid var(--border);
              margin-bottom: 6px; font-variant-numeric: tabular-nums; }
.nr-section { margin: 7px 0; }
.nr-section h4 { font-size: 10px; color: var(--muted);
                 margin-bottom: 3px; text-transform: uppercase;
                 letter-spacing: 0.04em; font-weight: 600; }
.nr-section ul { list-style: none; padding: 0; margin: 0; }
.nr-section li { padding: 2px 0 2px 8px;
                 border-left: 2px solid #2d323a;
                 margin-bottom: 2px; color: var(--text); }
.nr-section li.nr-good { border-left-color: var(--green); }
.nr-section li.nr-warn { border-left-color: var(--amber); }
.nr-section li.nr-note { border-left-color: var(--purple); }

/* ── Explainer tooltips on metric labels ── */
.ml[data-tip], .lbl[data-tip], h3.sec[data-tip] {
  cursor: help;
  border-bottom: 1px dotted rgba(148,163,184,0.4);
}
</style>
</head>
<body>

<div class="panel" id="hud">
  <div class="lbl">Match time</div>
  <div class="big" id="h-t">00:00</div>
  <div class="row"><span class="lbl">Frame</span><span class="val" id="h-f">0/0</span></div>
  <div class="row"><span class="lbl">Alive</span><span class="val" id="h-a">0</span></div>
  <div class="row"><span class="lbl">Teams alive</span><span class="val" id="h-tm">0</span></div>
  <div class="row"><span class="lbl">Combat pairs</span><span class="val" id="h-c">0</span></div>
  <div class="row"><span class="lbl">Dmg (30s win)</span><span class="val" id="h-dw">0</span></div>
  <div class="row"><span class="lbl">Safe R</span><span class="val" id="h-sr">0m</span></div>
  <div class="row"><span class="lbl">Poison R</span><span class="val" id="h-pr">0m</span></div>
  <div class="row" style="margin-top:4px"><span class="lbl">Zoom</span><span class="val" id="h-zoom">1.0×</span></div>
  <div class="row"><span class="lbl">Rotation</span><span class="val" id="h-rot">0°</span></div>
</div>

<div id="header"></div>

<div id="arena-wrap">
  <svg id="arena" viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid meet">
    <defs>
      <marker id="arrow-combat" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#ef4444"/>
      </marker>
    </defs>
    <g id="world">
      <image id="map-bg" x="0" y="0" width="1000" height="1000"
             preserveAspectRatio="none" opacity="0.55" style="display:none"/>
      <rect  id="map-tint" x="0" y="0" width="1000" height="1000"
             fill="#000" opacity="0.20" style="display:none"/>
      <g id="g-zone"></g>
      <g id="g-edges-enemy"></g>
      <g id="g-edges-ally"></g>
      <g id="g-edges-combat"></g>
      <g id="g-teams"></g>
      <g id="g-nodes"></g>
      <g id="g-labels" pointer-events="none"></g>
    </g>
  </svg>
  <div id="legend-inline">
    <div class="lg"><div class="ll" style="background:#3b82f6"></div>Ally</div>
    <div class="lg"><div class="ll" style="background:rgba(251,191,36,0.4)"></div>Enemy proximity</div>
    <div class="lg"><div class="ll" style="background:#ef4444"></div>Combat (attacker→victim)</div>
    <div class="lg"><div class="ld" style="background:#e5e7eb"></div>Player · size ∝ centrality</div>
    <div class="lg"><div class="ld" style="background:transparent;border:1px dashed #94a3b8"></div>Team centroid</div>
  </div>
  <div id="subtitle">wheel=zoom · drag=pan · shift+drag=rotate · dblclick=reset</div>
</div>

<div class="panel" id="side">
  <h3 class="sec" data-tip="노드 색상 기준을 선택">View mode</h3>
  <div>
    <button class="btn vm on" data-vm="team" title="팀 기준 색상">Team</button>
    <button class="btn vm" data-vm="centrality" title="Centrality 값 = 빨강/하양 heatmap">Centrality</button>
    <button class="btn vm" data-vm="combat" title="Combat dmg 유무로 하이라이트">Combat</button>
    <button class="btn vm" data-vm="louvain" title="Louvain 커뮤니티 기준 색상 (CH5)">Louvain</button>
    <button class="btn vm" data-vm="spectral" title="Spectral 클러스터 기준 색상 (CH5)">Spectral</button>
  </div>
  <div style="margin-top:6px">
    <span class="ml" style="margin-right:4px">edges:</span>
    <button class="btn et on" data-et="ally">Ally</button>
    <button class="btn et on" data-et="enemy">Enemy</button>
    <button class="btn et on" data-et="combat">Combat</button>
  </div>
  <div style="margin-top:6px">
    <span class="ml" style="margin-right:4px">measure:</span>
    <select id="topk-sel" class="select-m">
      __MEASURE_OPTIONS__
    </select>
  </div>

  <h3 class="sec">Team roster (<span id="roster-count">0</span>)</h3>
  <div id="roster-body"></div>

  <h3 class="sec" data-tip="현재 프레임의 전역 그래프 지표 (CH3)">Global (this frame)</h3>
  <div class="mrow"><span class="ml" data-tip="생존자 수 = 현재 프레임 노드 수">n_nodes (alive)</span><span class="mv" id="g-nn">-</span></div>
  <div class="mrow"><span class="ml" data-tip="ally + enemy + combat 통합 그래프의 엣지 수">n_edges</span><span class="mv" id="g-ne">-</span></div>
  <div class="mrow"><span class="ml" data-tip="density = 실제 엣지 / 가능한 엣지. 0=고립, 1=완전 연결">density</span><span class="mv" id="g-dn">-</span></div>
  <div class="mrow"><span class="ml" data-tip="C̄ = 평균 클러스터링 계수. 이웃끼리 서로 연결될 확률 (삼각형 밀도)">avg clustering C̄</span><span class="mv" id="g-cc">-</span></div>
  <div class="mrow"><span class="ml" data-tip="노드당 평균 연결 수 = 2·|E| / |V|">avg degree</span><span class="mv" id="g-ad">-</span></div>
  <div class="mrow"><span class="ml" data-tip="모든 쌍의 평균 최단경로. 작을수록 small-world">avg path length</span><span class="mv" id="g-pl">-</span></div>
  <div class="mrow"><span class="ml" data-tip="모든 최단경로 중 가장 긴 것">diameter</span><span class="mv" id="g-dm">-</span></div>
  <div class="mrow"><span class="ml" data-tip="giant = 최대 연결 성분 크기 / 전체 연결 성분 수">giant / comp</span><span class="mv" id="g-gs">-</span></div>

  <h3 class="sec" data-tip="CH5 Community Detection 결과 (Louvain / Spectral)">Communities (CH5)</h3>
  <div class="mrow"><span class="ml" data-tip="Louvain 분할의 modularity Q. Q>0.3 = 유의 (Newman 2004)">Louvain Q</span><span class="mv" id="cm-lq">-</span></div>
  <div class="mrow"><span class="ml">Louvain n</span><span class="mv" id="cm-ln">-</span></div>
  <div class="mrow"><span class="ml" data-tip="Louvain 커뮤니티 크기 리스트 (내림차순)">Louvain sizes</span><span class="mv" id="cm-ls">-</span></div>
  <div class="mrow"><span class="ml" data-tip="Spectral Clustering modularity Q">Spectral Q</span><span class="mv" id="cm-sq">-</span></div>
  <div class="mrow"><span class="ml" data-tip="Spectral n_communities (k_used)">Spectral n (k)</span><span class="mv" id="cm-sn">-</span></div>

  <h3 class="sec" data-tip="중심성 지표 Top-K 순위">Top-__TOP_K__ (<span id="measure-label">-</span>)</h3>
  <div id="topk-body"></div>

  <h3 class="sec" data-tip="P(k) = 노드 중 degree = k 일 확률 분포">Degree distribution P(k)</h3>
  <svg id="pk-chart" width="100%" height="90" viewBox="0 0 280 90"
       preserveAspectRatio="none"
       style="background:#161a20;border-radius:4px;border:1px solid var(--border);"></svg>

  <h3 class="sec">Selected player</h3>
  <div id="sel-info"><span style="color:var(--muted)">click a node</span></div>

  <h3 class="sec" data-tip="이 프레임에서 네트워크가 어떤 상태인가 — 자동 해설">📝 Narrative</h3>
  <div class="nr-panel" id="nr">
    <div class="nr-summary" id="nr-summary">-</div>
    <div class="nr-section"><h4>전역 구조</h4><ul id="nr-global"></ul></div>
    <div class="nr-section"><h4>중심성</h4><ul id="nr-centrality"></ul></div>
    <div class="nr-section"><h4>커뮤니티</h4><ul id="nr-community"></ul></div>
    <div class="nr-section"><h4>추세 (최근 프레임 대비)</h4><ul id="nr-trend"></ul></div>
  </div>
</div>

<div id="timeseries"></div>

<div id="controls">
  <button id="btn-rewind" title="First frame">⏮</button>
  <button id="btn-prev" title="Prev frame">◀</button>
  <button id="btn-play">▶ Play</button>
  <button id="btn-next" title="Next frame">▶</button>
  <button id="btn-end" title="Last frame">⏭</button>
  <input type="range" id="timeline" min="0" max="100" value="0" step="0.01">
  <span id="tstamp">00:00 / 00:00</span>
  <div class="ctrl-sep"></div>
  <label style="font-size:10.5px;color:var(--muted)">speed
    <select id="speed-sel" class="select-m">
      <option value="0.5">0.5×</option>
      <option value="1">1×</option>
      <option value="2" selected>2×</option>
      <option value="4">4×</option>
      <option value="8">8×</option>
    </select>
  </label>
  <label style="font-size:10.5px;color:var(--muted)">
    <input type="checkbox" id="cb-interp" checked> smooth
  </label>
  <div class="ctrl-sep"></div>
  <label style="font-size:10.5px;color:var(--muted)">
    <input type="checkbox" id="cb-map" checked> map
  </label>
  <input type="range" id="op-slider" min="0" max="1" step="0.01" value="__MAP_OP__">
  <div class="ctrl-sep"></div>
  <div class="ctrl-group">
    <button id="btn-rot-ccw" title="Rotate −90°">⟲</button>
    <button id="btn-rot-cw" title="Rotate +90°">⟳</button>
    <button id="btn-view-reset">Reset view</button>
  </div>
</div>

<div id="tip"></div>

<script>
const DATA = __DATA_JSON__;
const MEASURES = __MEASURES_LIST__;

// ── Playback state ──
let frameIdx = 0;
let virtualT = 0;
let playing = false;
let rafId = null;
let lastTs = 0;
let speedMul = 2;
let smoothInterp = true;
let viewMode = "team";
let measure = "__DEFAULT_MEASURE__";

// Community 색상 팔레트 (Team 팔레트와 구분되는 pastel 톤)
const COMMUNITY_PALETTE = [
  "#22d3ee", "#a78bfa", "#fbbf24", "#4ade80", "#f87171",
  "#38bdf8", "#fb923c", "#c084fc", "#34d399", "#f472b6",
  "#eab308", "#22c55e", "#06b6d4", "#8b5cf6", "#ef4444",
  "#14b8a6", "#f59e0b", "#d946ef", "#10b981", "#3b82f6"
];

// 시계열 차트 설정 (6개). 순서는 grid 배치 순서.
const TS_CHARTS = [
  {id:"density",  title:"density · avg_clustering",
   series:[{key:"density", color:"#60a5fa", label:"density"},
           {key:"avg_clustering", color:"#f59e0b", label:"C̄"}]},
  {id:"path",     title:"diameter · avg_path",
   series:[{key:"diameter", color:"#a78bfa", label:"diam"},
           {key:"avg_path_length", color:"#34d399", label:"avg path"}]},
  {id:"edges",    title:"n_edges · alive",
   series:[{key:"n_edges", color:"#f472b6", label:"edges"},
           {key:"alive", color:"#94a3b8", label:"alive"}]},
  {id:"modQ",     title:"modularity Q (Louvain/Spectral)",
   series:[{key:"modularity_q_louvain",  color:"#10b981", label:"Louv"},
           {key:"modularity_q_spectral", color:"#06b6d4", label:"Spec"}]},
  {id:"ncomm",    title:"# communities",
   series:[{key:"n_communities_louvain",  color:"#10b981", label:"Louv"},
           {key:"n_communities_spectral", color:"#06b6d4", label:"Spec"}]},
  {id:"combat",   title:"combat edges · dmg",
   series:[{key:"combat_edges", color:"#ef4444", label:"pairs"},
           {key:"combat_dmg_total", color:"#fb923c", label:"dmg"}]},
];
let edgeOn = { ally: true, enemy: true, combat: true };
let selectedPlayer = null;
let hoveredTeam = null;
let pinnedTeam = null;   // clicked team in roster → stays highlighted

const NFRAMES = DATA.frames.length;
const ARENA = DATA.meta.arena;
const SVG = document.getElementById("arena");
const VB = 1000;
const ax = (x) => (x / ARENA) * VB;
const ay = (y) => (y / ARENA) * VB;

const TCOLOR = {};
DATA.teams.forEach(t => TCOLOR[t.idx] = t.color);
const teamColor = (ti) => TCOLOR[ti] || "#888";

const el = (id) => document.getElementById(id);
const gZone = el("g-zone"), gEA = el("g-edges-ally"),
      gEE = el("g-edges-enemy"), gEC = el("g-edges-combat"),
      gTeams = el("g-teams"), gNodes = el("g-nodes"), gLabels = el("g-labels"),
      worldG = el("world"), mapBg = el("map-bg"), mapTint = el("map-tint");

// ── Map (local base64 first, then CDN fallback chain like visualize.py) ──
(function initMap() {
  const urls = DATA.map_urls || [];
  let urlIdx = 0;
  function showMap(href) {
    mapBg.setAttribute("href", href);
    mapBg.setAttributeNS("http://www.w3.org/1999/xlink", "xlink:href", href);
    mapBg.style.display = "";
    mapTint.style.display = "";
    if (DATA.meta.map_rotate) {
      mapBg.setAttribute("transform", `rotate(${DATA.meta.map_rotate} 500 500)`);
    }
  }
  function hideMap() {
    mapBg.style.display = "none";
    mapTint.style.display = "none";
  }
  function tryNext() {
    if (urlIdx >= urls.length) { hideMap(); return; }
    showMap(urls[urlIdx]);
    urlIdx++;
  }
  // SVGImageElement: listen for error → advance to next URL
  mapBg.addEventListener("error", () => { tryNext(); });
  if (DATA.map_b64) {
    showMap(DATA.map_b64);
  } else if (urls.length > 0) {
    tryNext();
  } else {
    hideMap();
  }
})();

// ── View (pan/zoom/rotate) ──
const view = { s: 1, tx: 0, ty: 0, rot: 0 };
function applyView() {
  worldG.setAttribute("transform",
    `translate(${view.tx.toFixed(2)},${view.ty.toFixed(2)}) ` +
    `rotate(${view.rot.toFixed(2)} 500 500) scale(${view.s.toFixed(4)})`);
  document.documentElement.style.setProperty("--inv-s", (1/view.s).toFixed(4));
  el("h-zoom").textContent = view.s.toFixed(2) + "×";
  el("h-rot").textContent = Math.round(((view.rot % 360) + 360) % 360) + "°";
}

// Convert screen (mouse) coordinates to SVG viewBox coords
function screenToSvg(clientX, clientY) {
  const pt = SVG.createSVGPoint();
  pt.x = clientX; pt.y = clientY;
  const m = SVG.getScreenCTM();
  if (!m) return { x: 500, y: 500 };
  return pt.matrixTransform(m.inverse());
}

SVG.addEventListener("wheel", (e) => {
  e.preventDefault();
  const factor = Math.exp(-e.deltaY * 0.0015);
  const newS = Math.max(0.3, Math.min(8, view.s * factor));
  if (Math.abs(newS - view.s) < 1e-6) return;
  // zoom-to-cursor (rot=0), else center-zoom
  if (Math.abs(view.rot % 360) < 0.1) {
    const p = screenToSvg(e.clientX, e.clientY);
    view.tx = p.x - (p.x - view.tx) * (newS / view.s);
    view.ty = p.y - (p.y - view.ty) * (newS / view.s);
  }
  view.s = newS;
  applyView();
}, { passive: false });

// drag / shift-drag rotate
let drag = null;
SVG.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  if (e.target.classList && e.target.classList.contains("p-node")) return; // let node click pass
  SVG.setPointerCapture(e.pointerId);
  const p = screenToSvg(e.clientX, e.clientY);
  drag = {
    mode: e.shiftKey ? "rot" : "pan",
    startX: e.clientX, startY: e.clientY,
    startTx: view.tx, startTy: view.ty,
    startRot: view.rot,
    startAngle: Math.atan2(p.y - 500, p.x - 500) * 180 / Math.PI,
    moved: false,
  };
  SVG.classList.add("dragging");
});
SVG.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.startX, dy = e.clientY - drag.startY;
  if (!drag.moved && Math.hypot(dx, dy) < 3) return;
  drag.moved = true;
  if (drag.mode === "pan") {
    const ctm = SVG.getScreenCTM();
    const scaleX = ctm ? ctm.a : 1;  // screen units per svg unit
    view.tx = drag.startTx + dx / Math.max(scaleX, 1e-6);
    view.ty = drag.startTy + dy / Math.max(scaleX, 1e-6);
  } else {
    const p = screenToSvg(e.clientX, e.clientY);
    const a = Math.atan2(p.y - 500, p.x - 500) * 180 / Math.PI;
    view.rot = drag.startRot + (a - drag.startAngle);
  }
  applyView();
});
SVG.addEventListener("pointerup", (e) => {
  if (drag) { SVG.releasePointerCapture(e.pointerId); }
  drag = null;
  SVG.classList.remove("dragging");
});
SVG.addEventListener("dblclick", () => { resetView(); });

function resetView() { view.s = 1; view.tx = 0; view.ty = 0; view.rot = 0; applyView(); }
function rotateBy(deg) { view.rot = (view.rot + deg) % 360; applyView(); }
el("btn-rot-ccw").onclick = () => rotateBy(-90);
el("btn-rot-cw").onclick  = () => rotateBy(+90);
el("btn-view-reset").onclick = resetView;

// map toggle / opacity
el("cb-map").onchange = (e) => {
  const hasAny = !!(DATA.map_b64) || (DATA.map_urls && DATA.map_urls.length);
  const show = e.target.checked && hasAny;
  mapBg.style.display = show ? "" : "none";
  mapTint.style.display = show ? "" : "none";
};
el("op-slider").oninput = (e) => {
  mapBg.setAttribute("opacity", e.target.value);
};

// ── Header ──
el("header").innerHTML = `<b>${DATA.meta.match_id.slice(0,16)}</b> &nbsp;·&nbsp; ${DATA.meta.map} / ${DATA.meta.mode} &nbsp;·&nbsp; ${DATA.meta.n_players} players · ${DATA.meta.n_teams} teams · ${NFRAMES} snapshots`;

// ── Timeline ──
const timeline = el("timeline");
timeline.max = NFRAMES - 1;
timeline.oninput = () => { virtualT = parseFloat(timeline.value); frameIdx = Math.floor(virtualT); render(); };

el("btn-play").onclick = togglePlay;
el("btn-prev").onclick = () => { stopPlay(); virtualT = Math.max(0, Math.floor(virtualT) - 1); frameIdx = virtualT; render(); };
el("btn-next").onclick = () => { stopPlay(); virtualT = Math.min(NFRAMES - 1, Math.floor(virtualT) + 1); frameIdx = virtualT; render(); };
el("btn-rewind").onclick = () => { stopPlay(); virtualT = 0; frameIdx = 0; render(); };
el("btn-end").onclick = () => { stopPlay(); virtualT = NFRAMES - 1; frameIdx = NFRAMES - 1; render(); };
el("speed-sel").onchange = (e) => { speedMul = parseFloat(e.target.value); };
el("cb-interp").onchange = (e) => { smoothInterp = e.target.checked; render(); };

function togglePlay() { playing ? stopPlay() : startPlay(); }
function startPlay() {
  playing = true;
  el("btn-play").textContent = "⏸ Pause";
  el("btn-play").classList.add("on");
  if (virtualT >= NFRAMES - 1) virtualT = 0;
  lastTs = 0;
  rafId = requestAnimationFrame(tick);
}
function stopPlay() {
  playing = false;
  el("btn-play").textContent = "▶ Play";
  el("btn-play").classList.remove("on");
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
}
function tick(ts) {
  if (!playing) return;
  if (lastTs === 0) lastTs = ts;
  const dt = (ts - lastTs) / 1000;
  lastTs = ts;
  virtualT += dt * speedMul;
  if (virtualT >= NFRAMES - 1) {
    virtualT = NFRAMES - 1; frameIdx = NFRAMES - 1;
    render(); stopPlay(); return;
  }
  frameIdx = Math.floor(virtualT);
  render();
  rafId = requestAnimationFrame(tick);
}

// ── Controls (view/edge/measure) ──
document.querySelectorAll(".vm").forEach(b => b.onclick = () => {
  document.querySelectorAll(".vm").forEach(x => x.classList.remove("on"));
  b.classList.add("on"); viewMode = b.dataset.vm; render();
});
document.querySelectorAll(".et").forEach(b => b.onclick = () => {
  b.classList.toggle("on"); edgeOn[b.dataset.et] = b.classList.contains("on"); render();
});
el("topk-sel").onchange = (e) => { measure = e.target.value; render(); };

const tip = el("tip");
function showTip(html, evt) {
  tip.innerHTML = html;
  tip.style.display = "block";
  tip.style.left = (evt.clientX + 14) + "px";
  tip.style.top  = (evt.clientY + 14) + "px";
}
function hideTip() { tip.style.display = "none"; }

function fmtT(s) {
  s = Math.max(0, Math.round(s));
  const m = Math.floor(s / 60), r = s % 60;
  return String(m).padStart(2,'0') + ":" + String(r).padStart(2,'0');
}
function fmtNum(v, d = 3) {
  if (v === null || v === undefined) return "-";
  if (typeof v !== "number") return v;
  return (Math.abs(v) < 1e-5 && v !== 0) ? v.toExponential(2) : v.toFixed(d);
}

// ── Interpolation ──
function interpPos(pIdx) {
  const i0 = frameIdx;
  const i1 = Math.min(frameIdx + 1, NFRAMES - 1);
  const f0 = DATA.frames[i0], f1 = DATA.frames[i1];
  const alive0 = f0.am[pIdx], alive1 = f1.am[pIdx];
  if (!alive0 && !alive1) return null;
  const p0 = f0.pos[pIdx], p1 = f1.pos[pIdx];
  if (!smoothInterp || i0 === i1) return [p0[0], p0[1]];
  const u = Math.max(0, Math.min(1, virtualT - i0));
  if (alive0 && alive1) return [p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u];
  return alive0 ? [p0[0], p0[1]] : [p1[0], p1[1]];
}
function interpTeamCentroid(ti) {
  const i0 = frameIdx;
  const i1 = Math.min(frameIdx + 1, NFRAMES - 1);
  const c0 = DATA.frames[i0].tc[ti];
  const c1 = DATA.frames[i1].tc[ti];
  if (!c0) return c1 || null;
  if (!c1) return c0;
  if (!smoothInterp || i0 === i1) return c0;
  const u = Math.max(0, Math.min(1, virtualT - i0));
  return [c0[0] + (c1[0] - c0[0]) * u, c0[1] + (c1[1] - c0[1]) * u];
}
function interpZone() {
  const i0 = frameIdx, i1 = Math.min(frameIdx + 1, NFRAMES - 1);
  const z0 = DATA.frames[i0].zn, z1 = DATA.frames[i1].zn;
  if (!smoothInterp || i0 === i1) return z0;
  const u = Math.max(0, Math.min(1, virtualT - i0));
  const L = (a, b) => a + (b - a) * u;
  return {
    sx: L(z0.sx, z1.sx), sy: L(z0.sy, z1.sy), sr: L(z0.sr, z1.sr),
    px: L(z0.px, z1.px), py: L(z0.py, z1.py), pr: L(z0.pr, z1.pr),
  };
}
function interpTime() {
  const i0 = frameIdx, i1 = Math.min(frameIdx + 1, NFRAMES - 1);
  const t0 = DATA.frames[i0].t, t1 = DATA.frames[i1].t;
  if (!smoothInterp || i0 === i1) return t0;
  const u = Math.max(0, Math.min(1, virtualT - i0));
  return t0 + (t1 - t0) * u;
}

// ── Node sizing / coloring (match-level p05/p95) ──
function nodeSize(fr, pIdx) {
  const v = fr.ce[measure] ? fr.ce[measure][pIdx] : null;
  if (v === null || v === undefined) return 5;
  const n = DATA.cnorm[measure] || {lo:0, hi:1};
  const u = Math.max(0, Math.min(1, (v - n.lo) / Math.max(n.hi - n.lo, 1e-9)));
  return 5 + u * 11;
}
function nodeColor(fr, p, pIdx) {
  if (viewMode === "centrality") {
    const v = fr.ce[measure] ? fr.ce[measure][pIdx] : null;
    if (v === null || v === undefined) return "#4b5563";
    const n = DATA.cnorm[measure] || {lo:0, hi:1};
    return heatColor(Math.max(0, Math.min(1, (v - n.lo) / Math.max(n.hi - n.lo, 1e-9))));
  } else if (viewMode === "combat") {
    const inD = fr.cb.in[pIdx] || 0, outD = fr.cb.out[pIdx] || 0;
    if (inD + outD === 0) return "rgba(229,231,235,0.22)";
    if (outD > inD) return "#ef4444";
    if (inD > outD) return "#f59e0b";
    return "#a78bfa";
  } else if (viewMode === "louvain" || viewMode === "spectral") {
    const key = viewMode === "louvain" ? "l" : "s";
    const cm = fr.cm ? fr.cm[key] : null;
    if (!cm || !cm.a) return teamColor(p.ti);
    const cid = cm.a[pIdx];
    if (cid === null || cid === undefined) return "rgba(229,231,235,0.22)";
    return COMMUNITY_PALETTE[cid % COMMUNITY_PALETTE.length];
  }
  return teamColor(p.ti);
}
function heatColor(u) {
  u = Math.max(0, Math.min(1, u));
  if (u < 0.5) {
    const t = u / 0.5;
    return `rgb(${Math.round(37 + (251-37)*t)},${Math.round(99 + (191-99)*t)},${Math.round(235 + (36-235)*t)})`;
  } else {
    const t = (u - 0.5) / 0.5;
    return `rgb(${Math.round(251 + (239-251)*t)},${Math.round(191 + (68-191)*t)},${Math.round(36 + (68-36)*t)})`;
  }
}

function shortenSegment(x1, y1, x2, y2, rStart, rEnd) {
  const dx = x2 - x1, dy = y2 - y1;
  const d = Math.hypot(dx, dy);
  if (d < 1e-3) return [x1, y1, x2, y2];
  const ux = dx / d, uy = dy / d;
  return [x1 + ux * rStart, y1 + uy * rStart, x2 - ux * rEnd, y2 - uy * rEnd];
}

// Which team to highlight (pinned from roster OR hovered from arena)
function activeTeam() { return pinnedTeam !== null ? pinnedTeam : hoveredTeam; }

// ── Roster panel (render once, update per-frame) ──
function renderRoster(fr) {
  const host = el("roster-body");
  const active = activeTeam();
  // Sort by rank asc (best first); unknown rank → end
  const sorted = DATA.teams.slice().sort((a, b) =>
    ((a.rank ?? 999) - (b.rank ?? 999)));
  el("roster-count").textContent = sorted.length;
  host.innerHTML = sorted.map(t => {
    const alive = fr.tac[t.idx];
    const dead = alive === 0;
    const aliveStr = dead ? `—/${t.size}` : `${alive}/${t.size}`;
    const rankStr = (t.rank !== null && t.rank !== undefined) ? `#${t.rank}` : "—";
    const isOn = active === t.idx;
    return `<div class="roster-row ${dead ? 'dead':''} ${isOn?'on':''}" data-ti="${t.idx}">
      <span class="nm"><span class="dot" style="background:${t.color}"></span>T${t.tid}</span>
      <span class="rk">${rankStr}</span>
      <span class="vl">${aliveStr}</span>
    </div>`;
  }).join("");
  host.querySelectorAll(".roster-row").forEach(r => {
    r.onclick = () => {
      const ti = parseInt(r.dataset.ti);
      pinnedTeam = (pinnedTeam === ti) ? null : ti;
      render();
    };
  });
}

// ── Render ──
function render() {
  const fr = DATA.frames[frameIdx];
  if (!fr) return;
  timeline.value = virtualT;

  el("h-t").textContent  = fmtT(interpTime());
  el("h-f").textContent  = `${frameIdx + 1}/${NFRAMES}`;
  el("h-a").textContent  = fr.mg.nn;
  el("h-tm").textContent = fr.tac.filter(c => c > 0).length;
  el("h-c").textContent  = fr.ec.length;
  const dmgSum = fr.ec.reduce((s, e) => s + e[2], 0);
  el("h-dw").textContent = Math.round(dmgSum);
  const zi = interpZone();
  el("h-sr").textContent = Math.round(zi.sr) + "m";
  el("h-pr").textContent = Math.round(zi.pr) + "m";
  el("tstamp").textContent = fmtT(interpTime()) + " / " + fmtT(DATA.meta.duration);

  el("g-nn").textContent = fr.mg.nn;
  el("g-ne").textContent = fr.mg.ne;
  el("g-dn").textContent = fmtNum(fr.mg.dn, 4);
  el("g-cc").textContent = fmtNum(fr.mg.cc, 4);
  el("g-ad").textContent = fmtNum(fr.mg.ad, 3);
  el("g-pl").textContent = fmtNum(fr.mg.pl, 3);
  el("g-dm").textContent = fr.mg.dm;
  el("g-gs").textContent = `${fr.mg.gs} / ${fr.mg.nc}`;
  el("measure-label").textContent = measure;

  // Communities (CH5)
  const cm = fr.cm || {l:{}, s:{}};
  el("cm-lq").textContent = fmtNum(cm.l.q, 3);
  el("cm-ln").textContent = cm.l.n ?? "-";
  el("cm-ls").textContent = Array.isArray(cm.l.s) ? JSON.stringify(cm.l.s) : "-";
  el("cm-sq").textContent = fmtNum(cm.s.q, 3);
  el("cm-sn").textContent = (cm.s.n ?? "-") + (cm.s.k ? ` (k=${cm.s.k})` : "");

  gZone.innerHTML = "";
  if (zi.pr > 0) {
    gZone.innerHTML += `<circle cx="${ax(zi.px)}" cy="${ay(zi.py)}" r="${ax(zi.pr)}"
      fill="rgba(59,130,246,0.12)" stroke="rgba(59,130,246,0.5)" stroke-width="1.3"
      vector-effect="non-scaling-stroke"/>`;
  }
  if (zi.sr > 0) {
    gZone.innerHTML += `<circle cx="${ax(zi.sx)}" cy="${ay(zi.sy)}" r="${ax(zi.sr)}"
      fill="none" stroke="rgba(255,255,255,0.8)" stroke-width="1.2" stroke-dasharray="5 4"
      vector-effect="non-scaling-stroke"/>`;
  }

  const livePos = [];
  const liveSize = [];
  for (let i = 0; i < DATA.players.length; i++) {
    if (fr.am[i]) {
      livePos[i] = interpPos(i);
      liveSize[i] = nodeSize(fr, i);
    } else {
      livePos[i] = null;
      liveSize[i] = 0;
    }
  }

  const active = activeTeam();

  // Edges
  gEA.innerHTML = ""; gEE.innerHTML = ""; gEC.innerHTML = "";
  if (edgeOn.ally) {
    const parts = [];
    for (const [s, d] of fr.ea) {
      const ps = livePos[s], pd = livePos[d]; if (!ps || !pd) continue;
      const [x1, y1, x2, y2] = shortenSegment(ax(ps[0]), ay(ps[1]), ax(pd[0]), ay(pd[1]),
                                              liveSize[s] * 0.5, liveSize[d] * 0.5);
      parts.push(`<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"
          stroke="#3b82f6" stroke-opacity="0.65" stroke-width="1.4" vector-effect="non-scaling-stroke"/>`);
    }
    gEA.innerHTML = parts.join("");
  }
  if (edgeOn.enemy) {
    const parts = [];
    for (const [s, d] of fr.ee) {
      const ps = livePos[s], pd = livePos[d]; if (!ps || !pd) continue;
      const [x1, y1, x2, y2] = shortenSegment(ax(ps[0]), ay(ps[1]), ax(pd[0]), ay(pd[1]),
                                              liveSize[s] * 0.5, liveSize[d] * 0.5);
      parts.push(`<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"
          stroke="rgba(251,191,36,0.22)" stroke-width="0.7" stroke-dasharray="3 3"
          vector-effect="non-scaling-stroke"/>`);
    }
    gEE.innerHTML = parts.join("");
  }
  if (edgeOn.combat) {
    const maxD = Math.max(1, ...fr.ec.map(e => e[2]));
    const parts = [];
    for (const [s, d, w] of fr.ec) {
      const ps = livePos[s], pd = livePos[d]; if (!ps || !pd) continue;
      const tk = 1.0 + 2.2 * (w / maxD);
      const op = 0.55 + 0.45 * (w / maxD);
      const [x1, y1, x2, y2] = shortenSegment(ax(ps[0]), ay(ps[1]), ax(pd[0]), ay(pd[1]),
                                              liveSize[s] * 0.5 + 0.5,
                                              liveSize[d] * 0.5 + 2.2);
      parts.push(`<circle cx="${x1.toFixed(1)}" cy="${y1.toFixed(1)}" r="${(tk*0.9).toFixed(1)}"
          fill="#ef4444" fill-opacity="${(op*0.9).toFixed(2)}"/>`);
      parts.push(`<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"
          stroke="#ef4444" stroke-opacity="${op.toFixed(2)}" stroke-width="${tk.toFixed(2)}"
          stroke-linecap="round" marker-end="url(#arrow-combat)" vector-effect="non-scaling-stroke"/>`);
    }
    gEC.innerHTML = parts.join("");
  }

  // Team centroid rings (alive ≥ 3 always, plus active team)
  const teamParts = [];
  DATA.teams.forEach(t => {
    const c = interpTeamCentroid(t.idx);
    const alive = fr.tac[t.idx];
    const isActive = active === t.idx;
    const show = (alive >= 3) || isActive;
    if (!c || !show || alive === 0) return;
    const r = 7 + alive * 2.2;
    const w = isActive ? 2.0 : 1.0;
    const op = isActive ? 0.95 : 0.55;
    teamParts.push(`<circle cx="${ax(c[0])}" cy="${ay(c[1])}" r="${r}"
      fill="none" stroke="${t.color}" stroke-opacity="${op}" stroke-width="${w}"
      stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>`);
  });
  gTeams.innerHTML = teamParts.join("");

  // Player nodes (highlight active team members)
  const nodeParts = [];
  DATA.players.forEach(p => {
    if (!fr.am[p.idx]) return;
    const pos = livePos[p.idx]; if (!pos) return;
    const sz = liveSize[p.idx];
    const col = nodeColor(fr, p, p.idx);
    const isSel = selectedPlayer === p.idx;
    const isActiveMember = active === p.ti;
    const stroke = isSel ? "#fff" : (isActiveMember ? "#fff" : "rgba(255,255,255,0.35)");
    const sw = isSel ? 2.2 : (isActiveMember ? 1.5 : 0.7);
    nodeParts.push(`<circle cx="${ax(pos[0]).toFixed(1)}" cy="${ay(pos[1]).toFixed(1)}"
       r="${sz.toFixed(2)}" fill="${col}" stroke="${stroke}" stroke-width="${sw}"
       data-pidx="${p.idx}" class="p-node" style="cursor:pointer"
       vector-effect="non-scaling-stroke"/>`);
  });
  gNodes.innerHTML = nodeParts.join("");
  document.querySelectorAll(".p-node").forEach(n => {
    n.addEventListener("mousemove", onNodeHover);
    n.addEventListener("mouseleave", onNodeLeave);
    n.addEventListener("click", onNodeClick);
  });

  // Labels: always-on team centroids for alive≥2 teams + selected player name
  const labelParts = [];
  DATA.teams.forEach(t => {
    const alive = fr.tac[t.idx];
    if (alive < 2) return;
    const c = interpTeamCentroid(t.idx); if (!c) return;
    const isActive = active === t.idx;
    const op = isActive ? 1.0 : 0.55;
    const fs = isActive ? 12 : 9;  // ignored by CSS; use attr for emphasis via size
    labelParts.push(`<text class="team-label" x="${ax(c[0]).toFixed(1)}"
       y="${(ay(c[1]) - 11).toFixed(1)}" fill="${t.color}" text-anchor="middle"
       opacity="${op}" font-weight="700"
       style="paint-order:stroke;stroke:#000;stroke-width:2.5;stroke-linejoin:round"
       >T${t.tid}</text>`);
  });
  if (selectedPlayer !== null && fr.am[selectedPlayer]) {
    const sp = DATA.players[selectedPlayer];
    const pos = livePos[selectedPlayer];
    if (pos) {
      const sz = liveSize[selectedPlayer];
      labelParts.push(`<text x="${ax(pos[0]).toFixed(1)}" y="${(ay(pos[1]) - sz - 4).toFixed(1)}"
         fill="#fff" text-anchor="middle"
         style="paint-order:stroke;stroke:#000;stroke-width:2.2;stroke-linejoin:round"
         >${sp.s}</text>`);
    }
  }
  gLabels.innerHTML = labelParts.join("");

  renderTopK(fr);
  renderPkChart(fr);
  renderSelected(fr);
  renderRoster(fr);
  renderNarrative(fr);
  renderTSCursors(frameIdx);
}

function onNodeHover(evt) {
  const pIdx = parseInt(evt.target.dataset.pidx);
  const fr = DATA.frames[frameIdx];
  const p = DATA.players[pIdx];
  const cv = fr.ce[measure] ? fr.ce[measure][pIdx] : null;
  hoveredTeam = p.ti;
  render();
  showTip(
    `<b>${p.s}</b> · team T${p.tid} · rank ${DATA.teams[p.ti].rank ?? "—"}<br>` +
    `HP ${fr.hp[pIdx]} · ${measure}=${fmtNum(cv, 3)}<br>` +
    `combat in=${fr.cb.in[pIdx]} out=${fr.cb.out[pIdx]}<br>dmg dealt=${fr.cb.dd[pIdx]} taken=${fr.cb.dt[pIdx]}`,
    evt
  );
}
function onNodeLeave() { hoveredTeam = null; hideTip(); render(); }
function onNodeClick(evt) {
  selectedPlayer = parseInt(evt.target.dataset.pidx); render();
}

function renderTopK(fr) {
  const lst = fr.tr[measure] || [];
  const html = lst.map(([pIdx, v], i) => {
    const p = DATA.players[pIdx];
    return `<div class="topk-row" data-pidx="${pIdx}">
      <span class="rk">${i + 1}</span>
      <span class="nm"><span class="dot" style="background:${teamColor(p.ti)}"></span>${p.s}</span>
      <span class="vl">${fmtNum(v, 4)}</span>
    </div>`;
  }).join("");
  el("topk-body").innerHTML = html || '<span style="color:var(--muted)">-</span>';
  document.querySelectorAll("#topk-body .topk-row").forEach(r =>
    r.onclick = () => { selectedPlayer = parseInt(r.dataset.pidx); render(); });
}

function renderPkChart(fr) {
  const dd = fr.dd;
  const svg = el("pk-chart");
  svg.innerHTML = "";
  if (!dd.ks || dd.ks.length === 0) return;
  const W = 280, H = 90, pad = 16;
  const kmax = Math.max(...dd.ks, 1);
  const pmax = Math.max(...dd.probs, 0.01);
  const xsc = (k) => pad + (k / kmax) * (W - pad - 6);
  const ysc = (p) => H - pad + 2 - (p / pmax) * (H - pad - 6);
  let html = `<line x1="${pad}" y1="${H-pad+2}" x2="${W-3}" y2="${H-pad+2}" stroke="#3a3f47"/>
              <line x1="${pad}" y1="4" x2="${pad}" y2="${H-pad+2}" stroke="#3a3f47"/>
              <text x="${W-4}" y="${H-pad+11}" font-size="8" text-anchor="end" fill="#9aa1ab">k</text>
              <text x="${pad-4}" y="9" font-size="8" text-anchor="end" fill="#9aa1ab">P(k)</text>`;
  const bw = (W - pad - 8) / Math.max(dd.ks.length, 1) - 1;
  dd.ks.forEach((k, i) => {
    const x = xsc(k), y = ysc(dd.probs[i]);
    html += `<rect x="${(x-bw/2).toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(2,bw).toFixed(1)}"
             height="${(H-pad+2-y).toFixed(1)}" fill="#60a5fa" opacity="0.8"/>`;
  });
  svg.innerHTML = html;
}

function renderSelected(fr) {
  if (selectedPlayer === null) {
    el("sel-info").innerHTML = '<span style="color:var(--muted)">click a node or Top-K row</span>';
    return;
  }
  const p = DATA.players[selectedPlayer];
  const i = selectedPlayer;
  if (!fr.am[i]) {
    el("sel-info").innerHTML =
      `<div class="mrow"><span class="ml">Player</span><span class="mv">${p.s}</span></div>
       <div class="mrow"><span class="ml">Team</span><span class="mv" style="color:${teamColor(p.ti)}">T${p.tid}</span></div>
       <div class="mrow"><span class="ml">Status</span><span class="mv" style="color:var(--red)">ELIMINATED</span></div>`;
    return;
  }
  let centr = "";
  MEASURES.forEach(m => {
    const v = fr.ce[m] ? fr.ce[m][i] : null;
    centr += `<div class="mrow"><span class="ml">${m}</span><span class="mv">${fmtNum(v, 4)}</span></div>`;
  });
  el("sel-info").innerHTML =
    `<div class="mrow"><span class="ml">Player</span><span class="mv">${p.s}</span></div>
     <div class="mrow"><span class="ml">Team</span><span class="mv" style="color:${teamColor(p.ti)}">T${p.tid}</span></div>
     <div class="mrow"><span class="ml">HP</span><span class="mv">${fr.hp[i]}</span></div>
     <div class="mrow"><span class="ml">Pos (m)</span><span class="mv">${Math.round(fr.pos[i][0])}, ${Math.round(fr.pos[i][1])}</span></div>
     <div class="mrow"><span class="ml">combat in/out</span><span class="mv">${fr.cb.in[i]} / ${fr.cb.out[i]}</span></div>
     <div class="mrow"><span class="ml">dmg dealt/taken</span><span class="mv">${fr.cb.dd[i]} / ${fr.cb.dt[i]}</span></div>
     <div style="height:1px;background:var(--border);margin:6px 0"></div>
     ${centr}`;
}

// ============================================================
// Narrative panel
// ============================================================
function renderNarrative(fr) {
  const nr = fr.nr || {};
  el("nr-summary").textContent = nr.summary || "-";
  const fill = (sectionId, lines, cls) => {
    const ul = el(sectionId);
    if (!ul) return;
    if (!lines || !lines.length) {
      ul.innerHTML = '<li style="color:var(--muted)">-</li>';
      return;
    }
    ul.innerHTML = lines.map(s => {
      const esc = String(s).replace(/[&<>]/g, c =>
        ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
      return `<li class="${cls || ''}">${esc}</li>`;
    }).join("");
  };
  fill("nr-global",     nr.global,     "");
  fill("nr-centrality", nr.centrality, "nr-note");
  fill("nr-community",  nr.community,  "nr-good");
  fill("nr-trend",      nr.trend,      "nr-warn");
}

// ============================================================
// Time-series charts (매치 전체 추이)
// ============================================================
const TS_W = 280, TS_H = 84, TS_PAD_L = 6, TS_PAD_R = 6,
      TS_PAD_T = 6, TS_PAD_B = 14;

function _tsXScale(len, xMax) {
  return (i) => TS_PAD_L + (i / Math.max(len - 1, 1)) *
                 (TS_W - TS_PAD_L - TS_PAD_R);
}
function _tsYScale(vmin, vmax) {
  const range = Math.max(vmax - vmin, 1e-9);
  return (v) => (TS_H - TS_PAD_B) -
                ((v - vmin) / range) * (TS_H - TS_PAD_T - TS_PAD_B);
}

function buildTSCharts() {
  const host = el("timeseries");
  if (!host) return;
  const ts = DATA.ts || {};
  const N = (ts.t || []).length;
  if (N === 0) {
    host.innerHTML = '<div style="color:var(--muted);padding:8px">No timeseries</div>';
    return;
  }
  const xs = _tsXScale(N);
  const parts = TS_CHARTS.map(cfg => {
    // y range from all series
    let vmin = Infinity, vmax = -Infinity;
    cfg.series.forEach(s => {
      const arr = ts[s.key] || [];
      arr.forEach(v => {
        if (v === null || v === undefined) return;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      });
    });
    if (!isFinite(vmin)) { vmin = 0; vmax = 1; }
    if (vmin === vmax) { vmin -= 0.5; vmax += 0.5; }
    const ysc = _tsYScale(vmin, vmax);

    // lines
    const lines = cfg.series.map(s => {
      const arr = ts[s.key] || [];
      const pts = arr.map((v, i) => {
        if (v === null || v === undefined) return null;
        return `${xs(i).toFixed(1)},${ysc(v).toFixed(1)}`;
      }).filter(Boolean).join(" ");
      return `<polyline class="ts-line" stroke="${s.color}" points="${pts}"/>`;
    }).join("");

    // y labels (min/max)
    const yTicks = `
      <text class="ts-val" x="${TS_W-3}" y="9" text-anchor="end">${fmtNum(vmax,2)}</text>
      <text class="ts-val" x="${TS_W-3}" y="${TS_H-TS_PAD_B+10}"
            text-anchor="end">${fmtNum(vmin,2)}</text>
    `;
    // axis
    const axis = `
      <line class="ts-axis" x1="${TS_PAD_L}" y1="${TS_H-TS_PAD_B}"
            x2="${TS_W-TS_PAD_R}" y2="${TS_H-TS_PAD_B}"/>
    `;
    // current frame cursor (initial: 0)
    const cursor = `<line class="ts-now" id="tsc-${cfg.id}"
                      x1="${xs(0).toFixed(1)}" y1="${TS_PAD_T}"
                      x2="${xs(0).toFixed(1)}" y2="${TS_H-TS_PAD_B}"/>`;

    // legend
    const legend = cfg.series.map(s =>
      `<span><i style="background:${s.color}"></i>${s.label}</span>`
    ).join("");

    return `<div class="ts-chart" data-id="${cfg.id}">
      <div class="ts-title">${cfg.title}</div>
      <svg viewBox="0 0 ${TS_W} ${TS_H}" preserveAspectRatio="none">
        ${axis}${lines}${cursor}${yTicks}
      </svg>
      <div class="ts-legend">${legend}</div>
    </div>`;
  });
  host.innerHTML = parts.join("");
}

function renderTSCursors(fi) {
  const ts = DATA.ts || {};
  const N = (ts.t || []).length;
  if (N === 0) return;
  const xs = _tsXScale(N);
  const x = xs(Math.min(fi, N - 1)).toFixed(1);
  TS_CHARTS.forEach(cfg => {
    const ln = el(`tsc-${cfg.id}`);
    if (ln) { ln.setAttribute("x1", x); ln.setAttribute("x2", x); }
  });
}

// ============================================================
// Hover tooltip for [data-tip] labels (reuse existing #tip)
// ============================================================
function attachTipHandlers() {
  document.querySelectorAll("[data-tip]").forEach(elm => {
    elm.addEventListener("mousemove", (e) => {
      const msg = elm.getAttribute("data-tip");
      showTip(msg, e);
    });
    elm.addEventListener("mouseleave", hideTip);
  });
}

// Initial default view mode (run.py 가 __DEFAULT_VIEW_MODE__ 로 주입)
viewMode = "__DEFAULT_VIEW_MODE__" || "team";
document.querySelectorAll(".vm").forEach(b => {
  b.classList.toggle("on", b.dataset.vm === viewMode);
});

buildTSCharts();
attachTipHandlers();
applyView();
render();
</script>
</body>
</html>
"""


def write_match_html(
    match_data, out_path, default_measure="pagerank", top_k=TOP_K, default_view_mode="team"
):
    payload = _build_payload(match_data)

    measure_opts = "\n      ".join(
        f'<option value="{m}"{" selected" if m == default_measure else ""}>{m}</option>'
        for m in CENTRALITY_MEASURES
    )
    measures_list_js = "[" + ",".join(f'"{m}"' for m in CENTRALITY_MEASURES) + "]"

    short = payload["meta"]["match_id"][:12]
    html = _HTML_TEMPLATE
    html = html.replace("__MATCH_SHORT__", short)
    html = html.replace("__TOP_K__", str(top_k))
    html = html.replace("__MEASURE_OPTIONS__", measure_opts)
    html = html.replace("__DEFAULT_MEASURE__", default_measure)
    html = html.replace("__MEASURES_LIST__", measures_list_js)
    html = html.replace("__MAP_OP__", f"{MAP_OPACITY_DEFAULT:.2f}")
    html = html.replace("__DEFAULT_VIEW_MODE__", default_view_mode)

    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    html = html.replace("__DATA_JSON__", payload_json)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
