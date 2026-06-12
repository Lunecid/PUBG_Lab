"""
Result Visualization — 논문 핵심 5종 (CLAUDE.md §7)
=====================================================
1. 팀별 Survival Curve        ★가장 중요
2. Species-Area Validation
3. Team-Team Competition Heatmap
4. Spatial Hazard Map
5. Rank Evolution Plot

모든 그림은 matplotlib 기반 정적 출력 (논문 삽입용).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from collections import defaultdict


# ── 공통 스타일 ──
STYLE = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}
plt.rcParams.update(STYLE)

# 팀 색상 팔레트 (최대 25팀)
TEAM_COLORS = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors[:5])


# ============================================================
# 1. 팀별 Survival Curve ★
# ============================================================

def plot_survival_curves(
    team_survival_probs,
    team_ids=None,
    time_points=None,
    highlight_teams=None,
    event_annotations=None,
    title="Team Survival Curves",
    save_path=None,
):
    """
    팀별 생존 곡선.

    Parameters:
        team_survival_probs: dict {team_id: np.array [T]} — S_i(t) 시계열
        team_ids: list — 표시할 팀 (None=전체)
        time_points: np.array [T] — 시간축 (초)
        highlight_teams: list — 강조 표시할 팀
        event_annotations: list of dict — [{team_id, time, label, type}]
            type: "zone" (자기장), "combat" (교전), "elimination" (탈락)
        save_path: str — 파일 저장 경로
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    teams = team_ids or sorted(team_survival_probs.keys())
    n_teams = len(teams)

    if time_points is None:
        first_key = next(iter(team_survival_probs))
        time_points = np.arange(len(team_survival_probs[first_key])) * 10

    for i, tid in enumerate(teams):
        if tid not in team_survival_probs:
            continue

        s_t = team_survival_probs[tid]
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        is_highlight = highlight_teams and tid in highlight_teams
        lw = 2.5 if is_highlight else 0.8
        alpha = 1.0 if is_highlight else 0.3
        zorder = 10 if is_highlight else 1

        label = f"Team {tid}" if is_highlight or n_teams <= 10 else None
        ax.plot(time_points[:len(s_t)], s_t, color=color,
                linewidth=lw, alpha=alpha, zorder=zorder, label=label)

    # 이벤트 주석
    if event_annotations:
        markers = {"zone": "v", "combat": "x", "elimination": "D"}
        marker_colors = {"zone": "#2196F3", "combat": "#F44336", "elimination": "#000000"}
        for ann in event_annotations:
            tid = ann["team_id"]
            t = ann["time"]
            if tid in team_survival_probs:
                s_vals = team_survival_probs[tid]
                t_idx = np.searchsorted(time_points, t)
                t_idx = min(t_idx, len(s_vals) - 1)
                s_val = s_vals[t_idx]
                mtype = ann.get("type", "combat")
                ax.scatter(t, s_val, marker=markers.get(mtype, "o"),
                          color=marker_colors.get(mtype, "red"),
                          s=80, zorder=20, edgecolors="white", linewidths=0.5)
                if "label" in ann:
                    ax.annotate(ann["label"], (t, s_val),
                              textcoords="offset points", xytext=(5, 8),
                              fontsize=7, color=marker_colors.get(mtype, "red"))

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Survival Probability S(t)")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(time_points[0], time_points[-1])

    if n_teams <= 10 or highlight_teams:
        ax.legend(loc="lower left", framealpha=0.8)

    # 이벤트 타입 범례
    if event_annotations:
        legend_elements = [
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#2196F3",
                   label="Zone pressure", markersize=8),
            Line2D([0], [0], marker="x", color="w", markerfacecolor="#F44336",
                   label="Combat", markersize=8, markeredgecolor="#F44336"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#000000",
                   label="Elimination", markersize=6),
        ]
        ax.legend(handles=legend_elements, loc="upper right", framealpha=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ============================================================
# 2. Species-Area Validation
# ============================================================

def plot_species_area(
    empirical_points,
    predicted_points=None,
    z_empirical=None,
    z_predicted=None,
    c_empirical=None,
    c_predicted=None,
    r2_empirical=None,
    title="Species-Area Relationship",
    save_path=None,
):
    """
    log(alive_teams) vs log(zone_area) — 실제 vs 모델 vs 이론 곡선.

    Parameters:
        empirical_points: list of (log_area, log_teams) — 실측
        predicted_points: list of (log_area, log_teams) — 모델 예측 (optional)
        z_empirical, c_empirical: S = c * A^z fitting 결과
        z_predicted, c_predicted: 모델 예측 fitting 결과
        r2_empirical: R² (실측)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 실측 데이터
    if empirical_points:
        ex = [p[0] for p in empirical_points]
        ey = [p[1] for p in empirical_points]
        ax.scatter(ex, ey, c="#1976D2", alpha=0.4, s=15, label="Empirical", zorder=2)

        # fitting 곡선
        if z_empirical is not None and c_empirical is not None:
            x_fit = np.linspace(min(ex), max(ex), 100)
            y_fit = z_empirical * x_fit + np.log(c_empirical)
            r2_str = f" (R²={r2_empirical:.3f})" if r2_empirical is not None else ""
            ax.plot(x_fit, y_fit, color="#1976D2", linewidth=2,
                    label=f"Empirical: z={z_empirical:.3f}{r2_str}", zorder=3)

    # 모델 예측
    if predicted_points:
        px = [p[0] for p in predicted_points]
        py = [p[1] for p in predicted_points]
        ax.scatter(px, py, c="#E53935", alpha=0.3, s=15, marker="^",
                  label="Predicted", zorder=2)

        if z_predicted is not None and c_predicted is not None:
            x_fit = np.linspace(min(px), max(px), 100)
            y_fit = z_predicted * x_fit + np.log(c_predicted)
            ax.plot(x_fit, y_fit, color="#E53935", linewidth=2, linestyle="--",
                    label=f"Predicted: z={z_predicted:.3f}", zorder=3)

    # 생태학 참고 구간
    if empirical_points:
        x_range = np.linspace(min(ex), max(ex), 100)
        for z_ref, label, color in [
            (0.15, "Continental (z=0.15)", "#A5D6A7"),
            (0.25, "Island (z=0.25)", "#FFF59D"),
            (0.35, "Isolated (z=0.35)", "#FFAB91"),
        ]:
            y_ref = z_ref * x_range + np.log(c_empirical or 1)
            ax.plot(x_range, y_ref, color=color, linewidth=1,
                    linestyle=":", alpha=0.6, label=label)

    ax.set_xlabel("log(Zone Area) [m²]")
    ax.set_ylabel("log(Alive Teams)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ============================================================
# 3. Team-Team Competition Heatmap
# ============================================================

def plot_competition_heatmap(
    alpha_matrices,
    phase_labels=None,
    team_labels=None,
    title="Team-Team Competition Intensity (α_ij)",
    save_path=None,
):
    """
    Phase별 α_ij 경쟁 계수 행렬 히트맵.

    Parameters:
        alpha_matrices: list of np.array [n_teams, n_teams] — 각 phase의 α_ij
        phase_labels: list of str — phase 이름
        team_labels: list of str — 팀 이름
    """
    n_phases = len(alpha_matrices)
    n_cols = min(n_phases, 4)
    n_rows = (n_phases + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                              squeeze=False)

    vmax = max(m.max() for m in alpha_matrices if m.size > 0)
    vmin = 0

    for i, alpha in enumerate(alpha_matrices):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]

        n_teams = alpha.shape[0]
        im = ax.imshow(alpha, cmap="YlOrRd", aspect="auto",
                       vmin=vmin, vmax=max(vmax, 0.01))

        phase_label = phase_labels[i] if phase_labels else f"Phase {i+1}"
        ax.set_title(phase_label, fontsize=10)

        if team_labels and n_teams <= 15:
            ax.set_xticks(range(n_teams))
            ax.set_xticklabels(team_labels[:n_teams], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(n_teams))
            ax.set_yticklabels(team_labels[:n_teams], fontsize=7)
        else:
            ax.set_xlabel("Target team")
            ax.set_ylabel("Source team")

    # 빈 subplot 제거
    for i in range(n_phases, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r][c].axis("off")

    fig.colorbar(im, ax=axes, label="Competition Intensity (α)", shrink=0.6)
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ============================================================
# 4. Spatial Hazard Map
# ============================================================

def plot_spatial_hazard(
    player_positions,
    hazard_values,
    zone_safe=None,
    zone_poison=None,
    map_size=8160,
    grid_resolution=50,
    title="Spatial Hazard Map",
    save_path=None,
    background_img=None,
):
    """
    맵 위 hazard intensity overlay.

    Parameters:
        player_positions: np.array [N, 2] — (x, y) meters
        hazard_values: np.array [N] — hazard 확률 (0~1)
        zone_safe: dict {center_x, center_y, radius} — 흰 원
        zone_poison: dict {center_x, center_y, radius} — 파란 원
        map_size: float — 맵 크기 (m)
        grid_resolution: int — hazard grid 해상도
        background_img: np.array — 맵 배경 이미지 (optional)
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 배경
    if background_img is not None:
        ax.imshow(background_img, extent=[0, map_size, 0, map_size],
                 alpha=0.3, origin="lower")

    # Hazard grid (KDE-like interpolation)
    grid = np.zeros((grid_resolution, grid_resolution))
    counts = np.zeros((grid_resolution, grid_resolution))
    cell = map_size / grid_resolution

    for pos, hz in zip(player_positions, hazard_values):
        gx = min(int(pos[0] / cell), grid_resolution - 1)
        gy = min(int(pos[1] / cell), grid_resolution - 1)
        gx = max(0, gx)
        gy = max(0, gy)
        grid[gy, gx] += hz
        counts[gy, gx] += 1

    # 스무딩
    from scipy.ndimage import gaussian_filter
    grid_mean = np.where(counts > 0, grid / counts, 0)
    grid_smooth = gaussian_filter(grid_mean, sigma=1.5)

    im = ax.imshow(grid_smooth, extent=[0, map_size, 0, map_size],
                   origin="lower", cmap="hot_r", alpha=0.7, vmin=0, vmax=1)

    # 플레이어 위치 (hazard 기준 색상)
    scatter = ax.scatter(player_positions[:, 0], player_positions[:, 1],
                        c=hazard_values, cmap="RdYlGn_r", s=20, alpha=0.8,
                        edgecolors="white", linewidths=0.3, vmin=0, vmax=1,
                        zorder=5)

    # Zone 표시
    if zone_safe:
        circle = plt.Circle(
            (zone_safe["center_x"], zone_safe["center_y"]),
            zone_safe["radius"], fill=False, color="white",
            linewidth=2, linestyle="--", label="Safe zone", zorder=10
        )
        ax.add_patch(circle)

    if zone_poison:
        circle = plt.Circle(
            (zone_poison["center_x"], zone_poison["center_y"]),
            zone_poison["radius"], fill=False, color="#2196F3",
            linewidth=2, linestyle="-", label="Poison zone", zorder=10
        )
        ax.add_patch(circle)

    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(title)
    ax.set_aspect("equal")

    fig.colorbar(scatter, ax=ax, label="Hazard Probability", shrink=0.7)
    if zone_safe or zone_poison:
        ax.legend(loc="upper right", framealpha=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ============================================================
# 5. Rank Evolution Plot
# ============================================================

def plot_rank_evolution(
    team_rank_trajectories,
    time_points=None,
    highlight_teams=None,
    true_final_ranks=None,
    title="Predicted Rank Evolution",
    save_path=None,
):
    """
    시간별 예측 기대 순위 변화.

    Parameters:
        team_rank_trajectories: dict {team_id: np.array [T]} — 예측 순위 시계열
        time_points: np.array [T]
        highlight_teams: list — 강조할 팀
        true_final_ranks: dict {team_id: int} — 실제 최종 순위
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    teams = sorted(team_rank_trajectories.keys())
    n_teams = len(teams)

    if time_points is None:
        first_key = next(iter(team_rank_trajectories))
        time_points = np.arange(len(team_rank_trajectories[first_key])) * 10

    for i, tid in enumerate(teams):
        ranks = team_rank_trajectories[tid]
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        is_highlight = highlight_teams and tid in highlight_teams
        lw = 2.5 if is_highlight else 0.6
        alpha = 1.0 if is_highlight else 0.2
        zorder = 10 if is_highlight else 1

        label = None
        if is_highlight and true_final_ranks and tid in true_final_ranks:
            label = f"Team {tid} (final: #{true_final_ranks[tid]})"
        elif is_highlight:
            label = f"Team {tid}"

        ax.plot(time_points[:len(ranks)], ranks, color=color,
                linewidth=lw, alpha=alpha, zorder=zorder, label=label)

        # 최종 순위 마커
        if true_final_ranks and tid in true_final_ranks and is_highlight:
            ax.scatter(time_points[min(len(ranks)-1, len(time_points)-1)],
                      ranks[-1], color=color, s=60, zorder=15,
                      edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Predicted Expected Rank")
    ax.set_title(title)
    ax.invert_yaxis()  # 1위가 위

    if highlight_teams:
        ax.legend(loc="upper left", framealpha=0.8)

    # y축: 1부터 팀 수까지
    if n_teams <= 30:
        ax.set_yticks(range(1, n_teams + 1, max(1, n_teams // 10)))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ============================================================
# 6. 보조 시각화
# ============================================================

def plot_training_history(
    history,
    title="Training History",
    save_path=None,
):
    """학습 히스토리 (loss + metrics)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=1.5)
    if history.get("best_epoch"):
        axes[0].axvline(history["best_epoch"], color="gray", linestyle="--",
                        alpha=0.5, label=f"Best (epoch {history['best_epoch']})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    # C-index
    axes[1].plot(epochs, history["val_c_index"], "g-", linewidth=1.5)
    axes[1].axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random (0.5)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("C-index")
    axes[1].set_title("Validation C-index")
    axes[1].legend()

    # Spearman
    axes[2].plot(epochs, history["val_spearman"], "m-", linewidth=1.5)
    axes[2].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Spearman ρ")
    axes[2].set_title("Rank Correlation")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


def plot_phase_calibration(
    phase_calibration,
    title="Phase-wise Calibration",
    save_path=None,
):
    """Phase별 calibration plot (predicted vs observed)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    phases = sorted(phase_calibration.keys())
    predicted = [phase_calibration[p]["predicted"] for p in phases]
    observed = [phase_calibration[p]["observed"] for p in phases]
    n_samples = [phase_calibration[p]["n_samples"] for p in phases]

    # perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    # phase별 점
    scatter = ax.scatter(predicted, observed, c=phases, cmap="viridis",
                        s=[max(20, min(n * 0.5, 200)) for n in n_samples],
                        edgecolors="black", linewidths=0.5, zorder=5)

    for i, p in enumerate(phases):
        ax.annotate(f"P{p}", (predicted[i], observed[i]),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Predicted Hazard Probability")
    ax.set_ylabel("Observed Event Rate")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.legend()

    fig.colorbar(scatter, ax=ax, label="Zone Phase")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ============================================================
# 7. 통합 보고서 생성
# ============================================================

def generate_all_figures(
    eval_results,
    match_data=None,
    model_outputs=None,
    output_dir="figures",
    prefix="",
):
    """
    전체 논문 그림 일괄 생성.

    Parameters:
        eval_results: evaluate_survival_model() 결과
        match_data: dict with team_survival_probs, species_area, etc.
        model_outputs: dict with hazard_logits, alphas, rank_trajectories
        output_dir: str
        prefix: str — 파일명 접두사
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # (1) Survival Curves
    if match_data and "team_survival_probs" in match_data:
        fig = plot_survival_curves(
            team_survival_probs=match_data["team_survival_probs"],
            time_points=match_data.get("time_points"),
            highlight_teams=match_data.get("highlight_teams"),
            event_annotations=match_data.get("event_annotations"),
            title="Team Survival Curves",
            save_path=os.path.join(output_dir, f"{prefix}survival_curves.png"),
        )
        figures["survival_curves"] = fig
        plt.close(fig)

    # (2) Species-Area
    if match_data and "species_area" in match_data:
        sa = match_data["species_area"]
        fig = plot_species_area(
            empirical_points=sa.get("empirical_points", []),
            predicted_points=sa.get("predicted_points"),
            z_empirical=sa.get("z_empirical"),
            z_predicted=sa.get("z_predicted"),
            c_empirical=sa.get("c_empirical"),
            c_predicted=sa.get("c_predicted"),
            r2_empirical=sa.get("r2_empirical"),
            save_path=os.path.join(output_dir, f"{prefix}species_area.png"),
        )
        figures["species_area"] = fig
        plt.close(fig)

    # (3) Competition Heatmap
    if model_outputs and "alpha_matrices" in model_outputs:
        fig = plot_competition_heatmap(
            alpha_matrices=model_outputs["alpha_matrices"],
            phase_labels=model_outputs.get("phase_labels"),
            team_labels=model_outputs.get("team_labels"),
            save_path=os.path.join(output_dir, f"{prefix}competition_heatmap.png"),
        )
        figures["competition_heatmap"] = fig
        plt.close(fig)

    # (4) Spatial Hazard
    if model_outputs and "spatial_data" in model_outputs:
        sp = model_outputs["spatial_data"]
        fig = plot_spatial_hazard(
            player_positions=sp["positions"],
            hazard_values=sp["hazards"],
            zone_safe=sp.get("zone_safe"),
            zone_poison=sp.get("zone_poison"),
            save_path=os.path.join(output_dir, f"{prefix}spatial_hazard.png"),
        )
        figures["spatial_hazard"] = fig
        plt.close(fig)

    # (5) Rank Evolution
    if model_outputs and "rank_trajectories" in model_outputs:
        fig = plot_rank_evolution(
            team_rank_trajectories=model_outputs["rank_trajectories"],
            time_points=model_outputs.get("time_points"),
            highlight_teams=model_outputs.get("highlight_teams"),
            true_final_ranks=model_outputs.get("true_final_ranks"),
            save_path=os.path.join(output_dir, f"{prefix}rank_evolution.png"),
        )
        figures["rank_evolution"] = fig
        plt.close(fig)

    # (6) Phase Calibration
    if eval_results and "phase_calibration" in eval_results:
        fig = plot_phase_calibration(
            phase_calibration=eval_results["phase_calibration"],
            save_path=os.path.join(output_dir, f"{prefix}phase_calibration.png"),
        )
        figures["phase_calibration"] = fig
        plt.close(fig)

    print(f"Generated {len(figures)} figures in {output_dir}/")
    return figures
