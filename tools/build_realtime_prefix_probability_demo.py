"""Build a 10-second prefix probability demo from a snapshot LSTM checkpoint.

This is a visualization/readout tool, not a new supervised training protocol.
The checkpoint was trained with phase-cut hazard labels.  For a snapshot time
tau inside phase k, the script reads the LSTM hidden state at tau, applies the
trained hazard head, and combines that prefix hazard with survival accumulated
through prior completed phase cuts:

    S_i(tau)  = S_i(k-1) * (1 - lambda_i(k | x_{<=tau}))
    pi_i(tau) = S_i(tau) / sum_j S_j(tau)

The output demonstrates a real-time-style probability trajectory at the native
10-second snapshot clock.  Metric claims still belong to the phase-cut
evaluation unless a dedicated prefix-label protocol is added.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model_stgcn.snapshot_phase_dataset import fetch_snapshot_phase
from model_stgcn.train_snapshot_phase import (
    _device,
    _forward_model,
    _make_model,
    _mask_snapshot_features,
)
from tools.eval_snapshot_survival_bridge import _namespace_from_checkpoint


DEFAULT_CHECKPOINT = Path(
    "model_stgcn/checkpoints/br_rq4_sequence_model_zoo_required_v1/"
    "erangel_tpp_squad/snapshot_lstm_no_graph/model.pt"
)
DEFAULT_MATCH_ID = "10691322-1ab3-42be-b896-c9c76f3e10f7"
DEFAULT_OUT = Path("reports/rq4_dynamic_probability_fluctuation")


def _cli_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(args.checkpoint),
        device=args.device,
        batch=None,
        eval_workers=None,
        prefetch_factor=None,
        cached=False,
        snapshot_cache_dir=None,
    )


def _single_item_batch(item: dict[str, Any]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        elif key in {"match_id", "team_ids"}:
            batch[key] = [value]
        else:
            batch[key] = value
    return batch


def _load_model(checkpoint: Path, args: argparse.Namespace) -> tuple[torch.nn.Module, SimpleNamespace, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    ns = _namespace_from_checkpoint(ckpt, _cli_namespace(args))
    if ns.model_variant != "snapshot_lstm_no_graph":
        raise SystemExit(
            "This demo currently supports snapshot_lstm_no_graph only, "
            f"got {ns.model_variant!r}."
        )
    device = _device(ns.device)
    model = _make_model(ns).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ns, ckpt


def _prefix_hazard(model: torch.nn.Module, batch: dict[str, Any], args: SimpleNamespace) -> torch.Tensor:
    X_s = _mask_snapshot_features(batch["X_s"], args)
    node_mask_s = batch["node_mask_s"].bool()
    mask = node_mask_s.bool()
    x = model.input(X_s.float()) * mask.unsqueeze(-1).to(X_s.dtype)
    bsz, steps, nodes, _hidden = x.shape
    seq = x.permute(0, 2, 1, 3).reshape(bsz * nodes, steps, -1)
    h, _ = model.lstm(seq)
    h = h.reshape(bsz, nodes, steps, -1).permute(0, 2, 1, 3)
    h = model.dropout(model.norm(h)) * mask.unsqueeze(-1).to(h.dtype)
    return torch.sigmoid(model.hazard_head(h).squeeze(-1))


def _prior_phase_survival(
    phase_prob: np.ndarray,
    valid_cut: np.ndarray,
) -> np.ndarray:
    phases, nodes = phase_prob.shape
    base = np.ones((phases, nodes), dtype=np.float64)
    running = np.ones((nodes,), dtype=np.float64)
    for phase_idx in range(phases):
        base[phase_idx] = running
        valid = valid_cut[phase_idx]
        running[valid] *= 1.0 - phase_prob[phase_idx, valid]
        running = np.clip(running, 0.0, 1.0)
    return base


def build_prefix_rows(
    checkpoint: Path,
    match_id: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model, ns, ckpt = _load_model(checkpoint, args)
    device = _device(ns.device)
    item = fetch_snapshot_phase(match_id, include_edges=False)
    if item is None:
        raise SystemExit(f"match {match_id} is not snapshot-phase-ready")
    batch = _single_item_batch(item)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    with torch.no_grad():
        prefix_prob = _prefix_hazard(model, batch, ns).detach().cpu().numpy()[0]
        phase_logits, cut_mask = _forward_model(model, batch, ns)
        phase_prob = torch.sigmoid(phase_logits).detach().cpu().numpy()[0]
        cut_valid = (
            batch["target_mask"].bool() & cut_mask.bool().unsqueeze(-1)
        ).detach().cpu().numpy()[0]

    prior_survival = _prior_phase_survival(phase_prob, cut_valid)
    phase_of_snap = batch["phase_of_snap"].detach().cpu().numpy()[0].astype(int)
    snapshot_ts = batch["snapshot_ts"].detach().cpu().numpy()[0].astype(float)
    node_mask = batch["node_mask_s"].detach().cpu().numpy()[0].astype(bool)
    y_rank = batch["y_rank"].detach().cpu().numpy()[0].astype(float)
    y_topk = batch["y_topk"].detach().cpu().numpy()[0].astype(int)
    team_ids = list(item["team_ids"])

    rows: list[dict[str, Any]] = []
    max_phase = phase_prob.shape[0]
    for snap_idx, phase in enumerate(phase_of_snap):
        if phase < 1 or phase > max_phase:
            continue
        live_idx = np.flatnonzero(node_mask[snap_idx])
        if live_idx.size == 0:
            continue
        scores = np.clip(
            prior_survival[phase - 1, live_idx] * (1.0 - prefix_prob[snap_idx, live_idx]),
            0.0,
            1.0,
        )
        denom = float(scores.sum())
        shares = scores / denom if denom > 0.0 else np.zeros_like(scores)
        for offset, team_idx in enumerate(live_idx):
            team_id = int(team_ids[team_idx]) if team_idx < len(team_ids) else 0
            if team_id == 0:
                continue
            rows.append(
                {
                    "model_variant": ns.model_variant,
                    "checkpoint": str(checkpoint),
                    "match_id": match_id,
                    "snapshot_idx": int(snap_idx),
                    "elapsed_s": float(snapshot_ts[snap_idx]),
                    "phase_slot": int(phase),
                    "team_idx": int(team_idx),
                    "team_id": team_id,
                    "prefix_hazard_prob": float(prefix_prob[snap_idx, team_idx]),
                    "prior_phase_survival": float(prior_survival[phase - 1, team_idx]),
                    "dynamic_survival_score": float(
                        prior_survival[phase - 1, team_idx]
                        * (1.0 - prefix_prob[snap_idx, team_idx])
                    ),
                    "dynamic_survival_share": float(shares[offset]),
                    "final_rank_norm": float(y_rank[team_idx]),
                    "final_top5": int(y_topk[team_idx, 0]),
                    "final_top10": int(y_topk[team_idx, 1]),
                }
            )

    df = pd.DataFrame(rows)
    meta = {
        "checkpoint": str(checkpoint),
        "model_variant": ns.model_variant,
        "match_id": match_id,
        "snapshots_total": int(len(snapshot_ts)),
        "prefix_rows": int(len(df)),
        "test_match": bool(match_id in [str(mid) for mid in (ckpt.get("test_ids") or [])]),
        "readout_contract": "10-second prefix readout from phase-cut-trained LSTM; descriptive demo, not separately supervised prefix metric.",
    }
    return df, meta


def plot_prefix(df: pd.DataFrame, out: Path, *, top_n: int) -> None:
    latest = (
        df.sort_values("elapsed_s")
        .groupby("team_id", as_index=False)
        .tail(1)
        .sort_values("dynamic_survival_share", ascending=False)
        .head(top_n)
    )
    team_order = latest["team_id"].astype(int).tolist()
    plot_df = df[df["team_id"].isin(team_order)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(team_order))))
    for color, team_id in zip(colors, team_order):
        team = plot_df[plot_df["team_id"] == team_id].sort_values("elapsed_s")
        x = team["elapsed_s"] / 60.0
        axes[0].plot(x, team["dynamic_survival_share"], color=color, linewidth=1.9, label=f"Team {team_id}")
        axes[1].plot(x, team["prefix_hazard_prob"], color=color, linewidth=1.6, label=f"Team {team_id}")
    phase_changes = (
        df.sort_values("elapsed_s")
        .drop_duplicates("phase_slot")[["elapsed_s", "phase_slot"]]
        .sort_values("elapsed_s")
    )
    for _, row in phase_changes.iterrows():
        x = float(row["elapsed_s"]) / 60.0
        for ax in axes:
            ax.axvline(x, color="0.80", linewidth=0.8, alpha=0.65)
    axes[0].set_title("10-second prefix probability readout")
    axes[0].set_ylabel("Match-local survival share")
    axes[1].set_ylabel("Prefix hazard")
    axes[1].set_xlabel("Elapsed minutes")
    for ax in axes:
        ax.grid(True, alpha=0.20)
        ax.set_ylim(bottom=0.0)
    axes[0].legend(loc="upper right", ncols=2, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_note(path: Path, meta: dict[str, Any], csv_path: Path, fig_path: Path) -> None:
    lines = [
        "# RQ4 Realtime Prefix Probability Demo",
        "",
        "Date: 2026-07-06",
        "Class: `ExperimentArtifact`",
        "Lifecycle: `descriptive_demo_bounded`",
        "",
        "## Purpose",
        "",
        "This artifact demonstrates that the snapshot LSTM checkpoint can emit a",
        "changing probability readout on the native 10-second snapshot clock.",
        "",
        "## Contract",
        "",
        "```text",
        "S_i(tau)  = S_i(k-1) * (1 - lambda_i(k | x_{<=tau}))",
        "pi_i(tau) = S_i(tau) / sum_j S_j(tau)",
        "```",
        "",
        "`tau` is a 10-second snapshot time inside phase `k`.",
        "",
        "## Outputs",
        "",
        f"- csv: `{csv_path}`",
        f"- figure: `{fig_path}`",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(meta, indent=2),
        "```",
        "",
        "## Boundary",
        "",
        "This is a descriptive prefix-readout demo from a phase-cut-trained model.",
        "Do not report it as a separately validated 10-second supervised metric until",
        "a prefix-label evaluation protocol is added.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df, meta = build_prefix_rows(args.checkpoint.resolve(), args.match_id, args)
    if df.empty:
        raise SystemExit("No prefix rows were produced.")
    csv_path = out_dir / "realtime_prefix_probability_trajectory.csv"
    fig_path = out_dir / "realtime_prefix_probability_example_match.png"
    note_path = out_dir / "RQ4_REALTIME_PREFIX_PROBABILITY_DEMO_2026-07-06.md"
    manifest_path = out_dir / "realtime_prefix_probability_manifest.json"
    df.to_csv(csv_path, index=False)
    plot_prefix(df, fig_path, top_n=args.top_n)
    meta.update(
        {
            "outputs": {
                "csv": str(csv_path),
                "figure": str(fig_path),
                "note": str(note_path),
                "manifest": str(manifest_path),
            }
        }
    )
    write_note(note_path, meta, csv_path, fig_path)
    manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--match-id", default=DEFAULT_MATCH_ID)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
