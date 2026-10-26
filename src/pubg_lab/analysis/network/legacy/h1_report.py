"""
H1 · Report writer
==================
Bias-controlled combat engagement vs placement 분석 결과를 academic-style 단일
페이지 HTML 리포트로 렌더링.
"""

from __future__ import annotations
import base64
import html as html_mod
import io
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pubg_lab.analysis.network.team_centrality import MEASURES


# ============================================================
# Plot style — light paper palette
# ============================================================

_PAPER_BG = "#fefcf7"
_INK = "#1a1a1a"
_MUTED = "#6b6558"
_RULE = "#d8d1c2"
_ACCENT = "#b02a2a"
_POSITIVE = "#2d5f3f"
_GRID = "#ebe4d3"

plt.rcParams.update(
    {
        "axes.facecolor": _PAPER_BG,
        "figure.facecolor": _PAPER_BG,
        "axes.edgecolor": _RULE,
        "axes.labelcolor": _INK,
        "text.color": _INK,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "grid.color": _GRID,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "regular",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _fig_to_b64(fig, dpi: int = 140) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=_PAPER_BG)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _h(s) -> str:
    return html_mod.escape(str(s))


def _fmt_r(v: float, d: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:+.{d}f}"


def _fmt_p(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    if v < 0.0001:
        return "<0.0001"
    return f"{v:.4f}"


def _sig_mark(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ============================================================
# Charts
# ============================================================


def chart_adjacency(A: np.ndarray, nodes: list[int]) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(A, cmap="pink_r", aspect="auto")
    ax.set_xlabel("team index")
    ax.set_ylabel("team index")
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("cumulative damage", color=_MUTED)
    cb.outline.set_edgecolor(_RULE)
    return _fig_to_b64(fig)


def chart_pk_aggregate(all_pks: list[tuple[list[int], list[float]]]) -> str:
    cnt = Counter()
    total = 0
    for ks, probs in all_pks:
        for k, p in zip(ks, probs):
            cnt[k] += p
        total += 1
    if total == 0 or not cnt:
        cnt[0] = 1
        total = 1
    ks = sorted(cnt.keys())
    avg_p = [cnt[k] / total for k in ks]

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.bar(ks, avg_p, color=_INK, width=0.7)
    ax.set_xlabel("degree  k")
    ax.set_ylabel("P(k)  ·  average across matches")
    ax.grid(axis="y", alpha=0.4)
    return _fig_to_b64(fig)


def chart_global_distribution(globals_list: list[dict]) -> str:
    dens = [g["density"] for g in globals_list]
    cc = [g["avg_clustering"] for g in globals_list]
    apl = [g["avg_path_length"] for g in globals_list]
    dm = [g["diameter"] for g in globals_list]

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.4))
    data = [dens, cc, apl, dm]
    labels = ["density", "avg C̄", "avg path", "diameter"]
    for ax, d, lbl in zip(axes, data, labels):
        ax.boxplot(
            d,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="#f0e6cc", edgecolor=_INK, linewidth=0.8),
            medianprops=dict(color=_ACCENT, linewidth=1.4),
            whiskerprops=dict(color=_INK, linewidth=0.8),
            capprops=dict(color=_INK, linewidth=0.8),
            flierprops=dict(markerfacecolor=_MUTED, markeredgecolor=_MUTED, markersize=3),
        )
        ax.set_title(lbl)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.35)
    return _fig_to_b64(fig)


def chart_scatter(
    rows: list[dict],
    feature: str,
    label: str,
    corr: dict,
) -> str:
    xs, ys = [], []
    for r in rows:
        x = r.get(feature)
        y = r.get("final_rank")
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.scatter(xs, ys, s=12, alpha=0.45, color=_INK, edgecolor="none")

    if np.isfinite(corr.get("lin_slope", np.nan)):
        xr = np.array([min(xs), max(xs)])
        yr = corr["lin_intercept"] + corr["lin_slope"] * xr
        ax.plot(xr, yr, color=_ACCENT, linewidth=1.0, label=f"R² = {corr['lin_r2']:.3f}")
        ax.legend(loc="best", fontsize=8, frameon=False)

    ax.set_xlabel(label)
    ax.set_ylabel("final_rank  (lower is better)")
    r = corr["pearson_r"]
    p = corr["pearson_p"]
    r_txt = "—" if not np.isfinite(r) else f"{r:+.3f}"
    ax.set_title(f"{label}  ·  r = {r_txt}  ·  p = {_fmt_p(p)}")
    ax.grid(alpha=0.35)
    ax.invert_yaxis()
    return _fig_to_b64(fig)


def chart_louvain_Q(communities: list[dict]) -> str:
    qs = [c["louvain"]["modularity_q"] for c in communities if c.get("louvain")]
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    if qs:
        ax.hist(
            qs, bins=min(10, max(3, len(qs) // 2)), color="#3a3a3a", edgecolor=_INK, linewidth=0.5
        )
    ax.axvline(0.3, color=_ACCENT, linewidth=1.2, linestyle="--", label="Q = 0.3 threshold")
    ax.set_xlabel("modularity Q (Louvain)")
    ax.set_ylabel("# matches")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.35)
    return _fig_to_b64(fig)


# ============================================================
# Section renderers
# ============================================================

_CENTR_DEF = {
    "degree": r"\( C_D(i) = \dfrac{k(i)}{|N| - 1} \)",
    "closeness": r"\( C_C(i) = \dfrac{|N| - 1}{\sum_{j} d(i, j)} \)",
    "betweenness": r"\( C_B(i) = \sum_{j,\,k} \dfrac{p_{jk}(i)}{p_{jk}} \)",
    "eigenvector": r"\( \lambda\, c = A\, c \)",
    "pagerank": r"\( r_j = \beta \sum_{i \to j} \dfrac{r_i}{d_i} + \dfrac{1 - \beta}{N} \)",
}

_CENTR_MEANING = {
    "degree": "교전한 상대 팀 수 (정규화)",
    "closeness": "다른 모든 팀까지의 평균 거리의 역수",
    "betweenness": "다른 팀 쌍들의 최단 경로가 이 팀을 지나는 빈도 — 중개자",
    "eigenvector": "중요한 팀과 연결될수록 본인도 중요해지는 재귀적 지표",
    "pagerank": "random teleport 을 더한 eigenvector 확장 — 검색엔진 랭킹 원형",
}

_CENTR_LABEL_KR = {
    "degree": "연결중심성 (degree)",
    "closeness": "근접중심성 (closeness)",
    "betweenness": "매개중심성 (betweenness)",
    "eigenvector": "고유벡터중심성 (eigenvector)",
    "pagerank": "페이지랭크 (PageRank)",
}


def render_hero(hero: dict, conclusion_lines: list[str]) -> str:
    verdict = hero.get("verdict", "inconclusive")
    verdict_label = {
        "confirmed": "가설 지지",
        "rejected": "기각 — 편향 기인",
        "inconclusive": "결론 유보",
    }.get(verdict, "결론 유보")
    verdict_color = {
        "confirmed": _POSITIVE,
        "rejected": _ACCENT,
        "inconclusive": _MUTED,
    }.get(verdict, _MUTED)

    raw = hero.get("raw_r")
    rank_delta = hero.get("rank_delta")
    intensity = hero.get("int_r")
    early = hero.get("early_r")
    partial = hero.get("partial_r")

    def _big(v, d=2):
        if v is None or not np.isfinite(v):
            return "—"
        return f"{v:+.{d}f}" if isinstance(v, float) and d > 0 else f"{v:.{d}f}"

    stat_delta = "—" if rank_delta is None or not np.isfinite(rank_delta) else f"{rank_delta:.1f}"

    first_line = conclusion_lines[0] if conclusion_lines else ""

    return f"""
<header class="hero">
  <div class="eyebrow">팀 교전 네트워크 분석</div>
  <h1 class="title">PUBG 스쿼드 매치에서 교전을 많이 한 팀이 정말 순위가 높을까?</h1>
  <p class="lede">
    원시 상관 · 시간당 강도 · 초반 5분 구간 · 편상관 — 네 가지 방법으로
    동시에 검증하여, 교전 참여도가 실제로 순위를 끌어올리는지 혹은
    생존 편향(survival bias)의 부산물에 불과한지를 가린다.
  </p>

  <div class="hero-grid">
    <div class="stat">
      <div class="stat-num">{_big(raw)}</div>
      <div class="stat-label">원시 Pearson r</div>
    </div>
    <div class="stat">
      <div class="stat-num">{stat_delta}</div>
      <div class="stat-label">상위 25% vs 하위 25% 평균순위 차이</div>
    </div>
    <div class="stat">
      <div class="stat-num" style="color:{verdict_color}">{verdict_label}</div>
      <div class="stat-label">편향 통제 후 판정</div>
    </div>
  </div>

  <div class="hero-ctrl">
    <span>시간당 강도 r = <b>{_big(intensity)}</b></span>
    <span>초반 구간 r = <b>{_big(early)}</b></span>
    <span>편상관 r (생존시간 통제) = <b>{_big(partial)}</b></span>
  </div>

  <p class="first-line">{_h(first_line)}</p>
</header>
"""


def render_section_network(adjs, pks) -> str:
    A, nodes = adjs[0]
    adj_b64 = chart_adjacency(A, nodes)
    pk_b64 = chart_pk_aggregate(pks)
    return f"""
<section>
  <div class="section-label">방법 · 네트워크 구성</div>
  <h2>네트워크 구성</h2>
  <p>
    각 매치를 무향 가중 그래프 \\( G = (V, E) \\) 로 모델링한다.
    \\( V \\) 는 팀 집합, 두 팀 \\( t_a, t_b \\) 사이에 데미지 이벤트가
    한 번이라도 기록되면 엣지 \\( (t_a, t_b) \\) 가 생성된다. 엣지 가중치는
    두 팀 사이의 누적 데미지 (<code>telem_damage</code> 집계)이다.
  </p>
  <p>
    이 표현은 인접행렬 \\( A \\) 와 차수행렬
    \\( D = \\mathrm{{diag}}(A\\mathbf{{1}}) \\) 로 요약되며, 이 두 행렬이
    이후 모든 지표(밀도 · 중심성 · 커뮤니티) 계산의 출발점이다.
  </p>
  <p class="eq">
    \\[
      \\text{{density}} = \\frac{{2\\,|E|}}{{|V|\\,(|V|-1)}}
      \\qquad
      C_i = \\frac{{2\\,e_i}}{{k_i\\,(k_i - 1)}}
    \\]
  </p>
  <figure>
    <img src="data:image/png;base64,{adj_b64}" alt="adjacency matrix"/>
    <figcaption>대표 매치의 인접행렬 \\( A \\)
      (\\( |V| = {A.shape[0]} \\)). 밝을수록 두 팀 사이 누적 데미지가 큼.</figcaption>
  </figure>
  <figure>
    <img src="data:image/png;base64,{pk_b64}" alt="P(k)"/>
    <figcaption>차수 분포 \\( P(k) \\), 분석된 매치 전체 평균.</figcaption>
  </figure>
</section>
"""


def render_section_structural(globals_list: list[dict]) -> str:
    if not globals_list:
        return ""

    def _avg(key):
        vals = [g[key] for g in globals_list if g.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    def _med(key):
        vals = [g[key] for g in globals_list if g.get(key) is not None]
        return float(np.median(vals)) if vals else 0.0

    dens = _avg("density")
    cc = _avg("avg_clustering")
    apl = _avg("avg_path_length")
    diam = _med("diameter")
    small_world = apl < 3.0 and cc > 0.3
    verdict = (
        "짧은 평균 경로와 상대적으로 높은 클러스터링 — small-world 경향."
        if small_world
        else "팀 레벨에서는 뚜렷한 small-world 신호가 나타나지 않음."
    )
    chart_b64 = chart_global_distribution(globals_list)
    return f"""
<section>
  <div class="section-label">방법 · 구조적 특성</div>
  <h2>구조적 특성</h2>
  <p>
    매치별 팀 그래프의 전역 지표. 밀도와 평균 클러스터링 \\( \\bar{{C}} \\)
    은 평균, 지름(diameter)은 매치 간 중앙값을 사용한다.
    클러스터링 계수는 \\( C_i = 2\\,e_i / (k_i(k_i - 1)) \\) 로 정의된다.
  </p>
  <table class="stat">
    <tr><th>지표</th><th>값</th></tr>
    <tr><td>밀도 (density)</td><td>{dens:.3f}</td></tr>
    <tr><td>평균 클러스터링 \\( \\bar{{C}} \\)</td><td>{cc:.3f}</td></tr>
    <tr><td>지름 (중앙값)</td><td>{diam:.1f}</td></tr>
    <tr><td>평균 최단경로 길이</td><td>{apl:.2f}</td></tr>
  </table>
  <p class="note">{verdict}</p>
  <figure><img src="data:image/png;base64,{chart_b64}" alt="global"/></figure>
</section>
"""


def render_section_centrality() -> str:
    rows_tbl = "\n".join(
        f"<tr><td class='lbl'>{_CENTR_LABEL_KR[m]}</td>"
        f"<td class='math'>{_CENTR_DEF[m]}</td>"
        f"<td>{_CENTR_MEANING[m]}</td></tr>"
        for m in MEASURES
    )
    return f"""
<section>
  <div class="section-label">방법 · 중심성</div>
  <h2>팀 중심성</h2>
  <p>
    각 팀 노드에 대해 다섯 가지 중심성 지표를 계산한다. 거리 기반 지표
    (closeness, betweenness) 는 가중치가 클수록 "가깝다" 로 해석하기 위해
    \\( \\text{{distance}} = 1 / \\text{{weight}} \\) 변환을 적용한다.
    고유벡터 중심성은 Perron–Frobenius 정리로 보장되는 유일한 양의
    고유벡터를 power iteration 으로 구한다.
  </p>
  <table class="stat">
    <tr><th>지표</th><th>정의</th><th>의미</th></tr>
    {rows_tbl}
  </table>
</section>
"""


def render_section_community(communities: list[dict]) -> str:
    qs = [c["louvain"]["modularity_q"] for c in communities if c.get("louvain")]
    ns = [c["louvain"]["n_communities"] for c in communities if c.get("louvain")]
    chart_b64 = chart_louvain_Q(communities)
    avg_q = float(np.mean(qs)) if qs else 0.0
    avg_n = float(np.mean(ns)) if ns else 0.0
    n_sig = sum(1 for q in qs if q > 0.3)
    verdict = (
        f"{len(qs)}개 매치 중 {n_sig}개에서 \\( Q > 0.3 \\) — "
        "팀 단위의 뚜렷한 교전 연합(community) 구조가 존재."
    )
    if not qs:
        verdict = "데이터 부족"
    return f"""
<section>
  <div class="section-label">방법 · 커뮤니티</div>
  <h2>커뮤니티 구조</h2>
  <p>
    가중 팀 그래프에 Louvain 알고리즘을 적용하면 서로 집중적으로 교전한
    팀들의 군집 — 사실상의 교전 연합(coalition) — 이 탐지된다. 분할의
    유의성은 모듈러리티 \\( Q \\) 로 측정한다.
  </p>
  <p class="eq">
    \\[
      Q = \\frac{{1}}{{2m}} \\sum_{{i,\\,j}}
        \\left( A_{{ij}} - \\frac{{k_i\\,k_j}}{{2m}} \\right)
        \\delta(c_i, c_j)
    \\]
  </p>
  <p class="note">
    \\( A_{{ij}} \\) 는 두 노드 사이의 가중치, \\( k_i \\) 는 노드 \\( i \\) 의
    가중 차수, \\( m \\) 은 총 엣지 가중치의 합, \\( \\delta(c_i, c_j) \\) 는
    같은 커뮤니티에 속할 때 1 아니면 0. Newman (2004) 은 \\( Q > 0.3 \\) 을
    유의한 커뮤니티 구조의 기준으로 제시.
  </p>
  <table class="stat">
    <tr><th>통계</th><th>값</th></tr>
    <tr><td>평균 \\( Q \\) (Louvain)</td><td>{avg_q:.3f}</td></tr>
    <tr><td>평균 커뮤니티 수</td><td>{avg_n:.1f}</td></tr>
    <tr><td>\\( Q > 0.3 \\) 인 매치 수</td><td>{n_sig} / {len(qs)}</td></tr>
  </table>
  <p class="note">{verdict}</p>
  <figure><img src="data:image/png;base64,{chart_b64}" alt="Q histogram"/></figure>
</section>
"""


def render_section_result(
    rows: list[dict],
    fourway_table: list[dict],
    corr_for_scatter: list[dict],
    quartile_results: list[dict],
    conclusion_lines: list[str],
    verdict: str,
) -> str:
    # conclusion 박스
    verdict_cls = verdict  # "confirmed" | "rejected" | "inconclusive"
    lines_html = "".join(f"<li>{_h(line)}</li>" for line in conclusion_lines)

    # 4-way table
    def _cell(r, p):
        if not np.isfinite(r):
            return '<td class="na">—</td>'
        star = _sig_mark(p) if p is not None else ""
        return f'<td>{r:+.3f}<span class="star">{star}</span></td>'

    fourway_rows = []
    for row in fourway_table:
        fourway_rows.append(
            f"<tr><td class='lbl'>{_h(row['label'])}</td>"
            f"{_cell(row['raw_r'], row['raw_p'])}"
            f"{_cell(row['int_r'], row['int_p'])}"
            f"{_cell(row['early_r'], row['early_p'])}"
            f"{_cell(row['partial_r'], None)}"
            f"<td class='n'>{row['raw_n']}</td></tr>"
        )
    fourway_html = "\n".join(fourway_rows)

    # Scatter plots
    scatter_imgs = []
    for c in corr_for_scatter:
        if not np.isfinite(c.get("pearson_r", np.nan)):
            continue
        feat = c["feature"]
        label = feat.replace("c_", "centrality · ").replace("_", " ")
        b64 = chart_scatter(rows, feat, label, c)
        scatter_imgs.append(f'<figure><img src="data:image/png;base64,{b64}"/></figure>')
    scatter_grid = "\n".join(scatter_imgs)

    # Quartile
    q_rows = []
    for q in quartile_results:
        if q.get("insufficient"):
            continue
        f = q["feature"].replace("c_", "")
        q_rows.append(
            f"<tr><td class='lbl'>{f}</td>"
            f"<td>{q['n_top']}</td>"
            f"<td>{q['top_mean_rank']:.2f}</td>"
            f"<td>{q['bot_mean_rank']:.2f}</td>"
            f"<td>{q['top_mean_rank'] - q['bot_mean_rank']:+.2f}</td>"
            f"<td>{q['welch_t']:+.2f}</td>"
            f"<td>{_fmt_p(q['welch_p'])}<span class='star'>{_sig_mark(q['welch_p'])}</span></td>"
            f"<td>{_fmt_p(q['mannwhitney_p'])}<span class='star'>{_sig_mark(q['mannwhitney_p'])}</span></td></tr>"
        )
    q_html = "\n".join(q_rows) or '<tr><td colspan="8" class="na">insufficient data</td></tr>'

    return f"""
<section>
  <div class="section-label">결과 · 가설 검증</div>
  <h2>교전 참여도와 순위의 관계</h2>

  <aside class="abstract {verdict_cls}">
    <h3>요약</h3>
    <ol>{lines_html}</ol>
  </aside>

  <h3>4-축 상관 비교표</h3>
  <p class="note">
    후보 지표 \\( X \\) 에 대해 \\( r(X, \\text{{rank}}) \\) 를 네 가지
    조건에서 계산한다. 유의 표시 (<code>*</code> p&lt;0.05, <code>**</code>
    p&lt;0.01, <code>***</code> p&lt;0.001) 는 원시 · 강도 · 초반 열에
    적용되며, 편상관은 계수 자체만 표시한다. 편상관은 다음 공식을 따른다:
  </p>
  <p class="eq">
    \\[
      r_{{xy \\cdot z}} =
        \\frac{{r_{{xy}} - r_{{xz}}\\, r_{{yz}}}}
             {{\\sqrt{{(1 - r_{{xz}}^{{2}})\\,(1 - r_{{yz}}^{{2}})}}}}
    \\]
  </p>
  <table class="four">
    <thead>
      <tr><th>지표</th>
          <th>원시 r</th>
          <th>강도 r<br/><span class="sub">per min</span></th>
          <th>초반 r<br/><span class="sub">\\( t \\le \\) 윈도우</span></th>
          <th>편상관 r<br/><span class="sub">생존시간 통제</span></th>
          <th>n</th></tr>
    </thead>
    <tbody>{fourway_html}</tbody>
  </table>

  <h3>산점도와 회귀선 (원시 값)</h3>
  <div class="scatter-grid">{scatter_grid}</div>

  <h3>상위 25% 팀 vs 하위 25% 팀</h3>
  <p class="note">
    각 지표를 기준으로 팀을 상위 25% / 하위 25% 로 나눠 평균 순위를 비교.
    Welch \\( t \\) 검정과 Mann–Whitney \\( U \\) 검정의 \\( p \\) 값을
    함께 제시한다.
  </p>
  <table class="stat">
    <thead>
      <tr><th>지표</th><th>n/그룹</th>
          <th>상위 평균순위</th><th>하위 평균순위</th><th>\\( \\Delta \\)</th>
          <th>Welch \\( t \\)</th><th>Welch \\( p \\)</th><th>Mann–Whitney \\( p \\)</th></tr>
    </thead>
    <tbody>{q_html}</tbody>
  </table>
</section>
"""


# ============================================================
# Full HTML template
# ============================================================

_HTML = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>팀 교전 네트워크 분석 · PUBG 스쿼드 매치</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Noto+Serif+KR:wght@400;500;600&family=Inter:wght@400;500;600&family=Noto+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<script>
  window.MathJax = {
    tex: {
      inlineMath: [['\\(', '\\)']],
      displayMath: [['\\[', '\\]']],
      processEscapes: true
    },
    svg: { fontCache: 'global' },
    chtml: { scale: 1.02 }
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
}
* { box-sizing: border-box; }
html, body { background: var(--bg); color: var(--ink); }
body {
  margin: 0;
  font-family: "Inter", "Noto Sans KR", -apple-system,
               "Apple SD Gothic Neo", "Helvetica Neue", sans-serif;
  font-size: 15px;
  line-height: 1.75;
  -webkit-font-smoothing: antialiased;
  word-break: keep-all;
}
.wrap {
  max-width: 760px;
  margin: 0 auto;
  padding: 72px 24px 120px;
}

/* Typography */
h1, h2, h3 {
  font-family: "Lora", "Noto Serif KR", "Charter", Georgia,
               "Apple SD Gothic Neo", serif;
  line-height: 1.25;
  color: var(--ink);
  font-weight: 600;
  letter-spacing: -0.01em;
}
h1.title {
  font-size: 40px;
  margin: 6px 0 14px;
  letter-spacing: -0.015em;
}
h2 {
  font-size: 26px;
  margin: 12px 0 10px;
}
h3 {
  font-size: 18px;
  margin: 28px 0 8px;
  font-weight: 600;
}
p { margin: 10px 0; }
code {
  font-family: "IBM Plex Mono", "SF Mono", Menlo, monospace;
  font-size: 0.92em;
  background: var(--paper);
  padding: 1px 6px;
  border: 1px solid var(--rule);
  border-radius: 3px;
  color: var(--accent);
}
a { color: var(--accent); }

/* Layout */
section {
  margin: 64px 0;
  padding-top: 36px;
  border-top: 1px solid var(--rule);
}
section:first-of-type { border-top: none; padding-top: 0; }

.section-label {
  font-family: "Inter", sans-serif;
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}

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

/* Hero */
.hero {
  margin-bottom: 84px;
}
.eyebrow {
  font-family: "Inter", sans-serif;
  font-size: 11px;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 4px;
}
.lede {
  font-family: "Lora", serif;
  font-size: 19px;
  font-style: italic;
  color: var(--muted);
  line-height: 1.55;
  margin: 16px 0 28px;
  max-width: 620px;
}
.hero-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin: 24px 0 16px;
  padding: 28px 0;
  border-top: 2px solid var(--ink);
  border-bottom: 1px solid var(--rule);
}
.stat { text-align: left; }
.stat-num {
  font-family: "Lora", serif;
  font-size: 44px;
  line-height: 1.05;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--ink);
  letter-spacing: -0.01em;
}
.stat-label {
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 6px;
}
.hero-ctrl {
  display: flex;
  gap: 24px;
  font-size: 13px;
  color: var(--muted);
  flex-wrap: wrap;
  font-variant-numeric: tabular-nums;
}
.hero-ctrl b { color: var(--ink); font-weight: 600; }
.first-line {
  margin-top: 16px;
  font-size: 13px;
  color: var(--muted);
  padding-top: 14px;
  border-top: 1px solid var(--rule);
}

/* Tables */
table { border-collapse: collapse; width: 100%; margin: 14px 0 6px;
        font-variant-numeric: tabular-nums; }
table.stat th, table.stat td,
table.four  th, table.four  td {
  padding: 7px 12px;
  text-align: left;
  border-bottom: 1px solid var(--rule);
  font-size: 13.5px;
}
table th {
  background: transparent;
  color: var(--muted);
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 10.5px;
  border-bottom: 1.5px solid var(--ink);
}
table th .sub { font-weight: 400; color: var(--muted);
                text-transform: none; letter-spacing: 0; }
table.four td { text-align: right; }
table.four td.lbl, table.stat td.lbl { text-align: left; }
table.four td.n, table.stat td.n { color: var(--muted); }
table .na { color: var(--muted); }
.star { color: var(--accent); margin-left: 2px; font-weight: 500; }

/* Abstract / findings */
.abstract {
  border-left: 3px solid var(--muted);
  padding: 20px 28px;
  margin: 16px 0 30px;
  background: var(--surface);
}
.abstract.confirmed    { border-left-color: var(--positive); }
.abstract.rejected     { border-left-color: var(--accent); }
.abstract.inconclusive { border-left-color: var(--muted); }
.abstract h3 {
  margin: 0 0 8px;
  font-size: 16px;
}
.abstract ol { margin: 0; padding-left: 20px; }
.abstract li { margin: 6px 0; }

/* Notes */
.note {
  font-size: 13px;
  color: var(--muted);
  font-style: italic;
}
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
mjx-container { outline: none; }
mjx-container[display="true"] { margin: 6px 0 !important; }
table.stat td.math, table.stat th.math { text-align: left; }
table.stat td.math mjx-container { margin: 0 !important; }

/* Scatter grid */
.scatter-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin: 12px 0;
}

/* Colophon */
.colophon {
  margin-top: 96px;
  padding-top: 20px;
  border-top: 1px solid var(--rule);
  font-size: 11.5px;
  color: var(--muted);
  display: flex;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  letter-spacing: 0.04em;
}
</style></head>
<body><div class="wrap">
__HERO__
__SECTION_NETWORK__
__SECTION_STRUCT__
__SECTION_CENTR__
__SECTION_COMM__
__SECTION_RESULT__
<footer class="colophon">
  <span>생성 __GENERATED_AT__</span>
  <span>매치 __N_MATCHES__건 · 팀 데이터 __N_TEAMS__행 · 초반 윈도우 __EARLY_SEC__</span>
</footer>
</div></body></html>
"""


# ============================================================
# Public API
# ============================================================


def write_h1_report(collected: dict, analysis: dict, out_path: str) -> str:
    rows = collected["rows"]
    globals_list = collected["globals"]
    adjs = collected["adjs"]
    pks = collected["pks"]
    communities = collected["communities"]
    n_matches = len(collected["match_ids"])
    n_teams = len(rows)
    early_sec = collected.get("early_window_sec")

    fourway_table = analysis["fourway_table"]
    corr_scatter = analysis["corr_for_scatter"]
    quartile = analysis["quartile_results"]
    conclusion = analysis["conclusion_lines"]
    verdict = analysis["verdict"]
    hero = analysis["hero"]

    gen_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    early_str = f"{int(early_sec)} s" if early_sec else "—"

    html = _HTML
    html = html.replace("__HERO__", render_hero(hero, conclusion))
    html = html.replace("__SECTION_NETWORK__", render_section_network(adjs, pks))
    html = html.replace("__SECTION_STRUCT__", render_section_structural(globals_list))
    html = html.replace("__SECTION_CENTR__", render_section_centrality())
    html = html.replace("__SECTION_COMM__", render_section_community(communities))
    html = html.replace(
        "__SECTION_RESULT__",
        render_section_result(rows, fourway_table, corr_scatter, quartile, conclusion, verdict),
    )
    html = html.replace("__GENERATED_AT__", gen_at)
    html = html.replace("__N_MATCHES__", str(n_matches))
    html = html.replace("__N_TEAMS__", str(n_teams))
    html = html.replace("__EARLY_SEC__", early_str)

    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
