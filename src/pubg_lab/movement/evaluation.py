"""Masked joint-scenario metrics in meters; matched sample IDs, match-level averages."""

from collections import defaultdict
import numpy as np


def metrics(pred, target, presence):
    # pred: [K,P,F,2], normalized by 1000m. One scenario is shared across all members.
    pred = np.asarray(pred) * 1000
    y = target["trajectory"] * 1000
    v = target["trajectory_valid"] & presence[:, None]
    result = {}
    err = np.linalg.norm(pred - y[None], axis=-1)
    if v.any():
        result["joint_minADE_m"] = float((err * v[None]).sum((1, 2)).min() / v.sum())
        result["joint_expectedADE_m"] = float((err * v[None]).sum((1, 2)).mean() / v.sum())
        result["joint_meanpathADE_m"] = float(
            (np.linalg.norm(pred.mean(0) - y, axis=-1) * v).sum() / v.sum()
        )
        difference = ((pred - y[None]) * v[None, :, :, None]).reshape(len(pred), -1)
        first = np.linalg.norm(difference, axis=1).mean() / np.sqrt(v.sum())
        delta = (pred[:, None] - pred[None, :]) * v[None, None, :, :, None]
        second = (
            np.linalg.norm(delta.reshape(len(pred), len(pred), -1), axis=-1).sum()
            / max(len(pred) * (len(pred) - 1), 1)
            / np.sqrt(v.sum())
        )
        result["joint_energy_score_m"] = float(first - 0.5 * second)
    if v[0].any():
        result["ego_minADE_m"] = float((err[:, 0] * v[0]).sum(1).min() / v[0].sum())
        result["ego_expectedADE_m"] = float((err[:, 0] * v[0]).sum(1).mean() / v[0].sum())
        result["ego_meanpathADE_m"] = float(
            np.linalg.norm(pred.mean(0)[0] - y[0], axis=-1)[v[0]].mean()
        )
    if v[0, -1]:
        result["ego_minFDE_60s_m"] = float(err[:, 0, -1].min())
        result["ego_expectedFDE_60s_m"] = float(err[:, 0, -1].mean())
        result["ego_meanpathFDE_60s_m"] = float(np.linalg.norm(pred.mean(0)[0, -1] - y[0, -1]))
    tm = target["team_valid"]
    center = pred[:, presence].mean(1)
    if tm.any():
        te = np.linalg.norm(center - target["team_center"][None] * 1000, axis=-1)[:, tm]
        result["team_minADE_m"] = float(te.mean(1).min())
        result["team_expectedADE_m"] = float(te.mean())
        result["team_meanpathADE_m"] = float(
            np.linalg.norm(center.mean(0) - target["team_center"] * 1000, axis=-1)[tm].mean()
        )
    pair = []
    for a in range(len(presence)):
        for b in range(a + 1, len(presence)):
            ok = v[a] & v[b]
            if ok.any():
                pair.extend(
                    np.abs(
                        np.linalg.norm(pred[:, a] - pred[:, b], axis=-1)[:, ok]
                        - np.linalg.norm(y[a] - y[b], axis=-1)[ok][None]
                    )
                    .ravel()
                    .tolist()
                )
    if pair:
        result["pair_distance_MAE_m"] = float(np.mean(pair))
    return result


def match_macro(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        for key, value in r["metrics"].items():
            grouped[r["match_id"]][key].append(value)
    keys = sorted({k for values in grouped.values() for k in values})
    return {
        k: {
            "mean_over_matches": float(
                np.mean([np.mean(g[k]) for g in grouped.values() if k in g])
            ),
            "matches": sum(k in g for g in grouped.values()),
        }
        for k in keys
    }
