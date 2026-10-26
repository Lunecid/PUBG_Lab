"""Compare the phase-tier dumps v1 and v2 row by row (read-only; both dump trees are only loaded).

For every cohort / split present in BOTH trees the matches are aligned by match id and the rows by
(cut slot, team id).  Reported (JSON to ``--out`` and a short table on stdout):

* rows / positives / terminal rows per cut for v1 (labels v2 = the event-based labels the PTR loader reads) and v2,
  rows that exist only in v1 or only in v2;
* per feature: mean |dX|, share of rows with |dX| > ``--tol``, correlation, on the rows alive in both dumps;
* the changed-row share (a row changes if any feature differs by more than ``--tol``) on the rows that are in both
  risk sets.

    python -m tools.compare_phase_dumps_v1_v2 --v1 model_stgcn/dumps/bigcomp2027_phase_v1 \
        --v2 model_stgcn/dumps/bigcomp2027_phase_v2 --out model_stgcn/dumps/bigcomp2027_phase_v2/v1_v2_comparison.json
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SPLITS = ("train", "val", "test")


def feature_groups(names: list[str]) -> dict[str, list[int]]:
    """Column groups of the report: the replaced alive counts, the scheduled-duration inputs, the event (combat / loot)
    inputs that were uncapped in v1, the network features, and the rest."""
    from pubg_lab.survival.phase_v2 import features as FT

    idx = {n: i for i, n in enumerate(names)}
    spec_col = {s[0]: s[1] for s in FT.SPECS}
    combat = [n for n in names if spec_col[n] in (
        "damage_dealt", "damage_taken", "damage_events_dealt", "damage_events_taken", "groggy_dealt", "groggy_taken",
        "kills_dealt", "kills_taken", "distinct_opponents_damaged", "distinct_opponents_damaged_by", "combat_initiated")]  # fmt: skip
    expo_ev = [n for n in names if spec_col[n] in ("recent_damage_taken", "combat_zone_pass_count")]
    loot = [n for n in names if spec_col[n] in (
        "pickup_count", "equip_count", "use_count", "unique_item_count", "weapon_pickup_count", "weapon_equip_count",
        "ammo_pickup_count", "armor_pickup_count", "throwable_pickup_count", "heal_use_count", "boost_use_count",
        "carepackage_pickup_count", "lootbox_pickup_count")]  # fmt: skip
    network = [n for n in names if spec_col[n].startswith(("net_", "cum_"))]
    groups = {
        "alive_counts(4 replaced)": list(FT.REPLACED_FEATURES),
        "phase_duration_10min": ["phase_duration_10min"],
        "scheduled_duration_inputs(11)": list(FT.DURATION_FEATURES),
        "combat_events(11)": combat,
        "exposure_events(2)": expo_ev,
        "loot_events": loot,
        "network(net_*,cum_*)": network,
    }
    used = {n for g in groups.values() for n in g}
    groups["all_other"] = [n for n in names if n not in used]
    return {g: [idx[n] for n in cols] for g, cols in groups.items()}


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def compare_cohort(v1_dir: Path, v2_dir: Path, slug: str, tol: float, acc: dict) -> None:
    for split in SPLITS:
        p1, p2 = v1_dir / slug / f"{split}.pt", v2_dir / slug / f"{split}.pt"
        if not (p1.exists() and p2.exists()):
            continue
        d1, d2 = _load(p1), _load(p2)
        names = list(d2["meta"]["feature_names"])
        assert names == list(d1["meta"]["feature_names"])
        by1 = {m["match_id"]: m for m in d1["matches"]}
        for m2 in d2["matches"]:
            m1 = by1.get(m2["match_id"])
            if m1 is None:
                acc["matches_only_v2"] += 1
                continue
            acc["matches_common"] += 1
            t1, t2 = m1["target_mask_v2"].numpy(), m2["target_mask_v2"].numpy()
            y1, y2 = m1["y_hazard_v2"].numpy() > 0.5, m2["y_hazard_v2"].numpy() > 0.5
            ids1, ids2 = m1["team_ids"].numpy(), m2["team_ids"].numpy()
            assert (ids1 == ids2).all()
            X1, X2 = m1["X"].numpy(), m2["X"].numpy()
            alive1, alive2 = m1["node_mask"].numpy(), m2["node_mask"].numpy()
            for k in range(7):
                acc["rows_v1"][k] += int(t1[k].sum())
                acc["rows_v2"][k] += int(t2[k].sum())
                acc["pos_v1"][k] += int((y1[k] & t1[k]).sum())
                acc["pos_v2"][k] += int((y2[k] & t2[k]).sum())
                both = t1[k] & t2[k]
                acc["rows_both"][k] += int(both.sum())
                acc["rows_only_v1"][k] += int((t1[k] & ~t2[k]).sum())
                acc["rows_only_v2"][k] += int((t2[k] & ~t1[k]).sum())
                # rows that v2 has and v1 lacks because phase k+1 had no grid snapshot in v1 (D1)
                if not bool(m1["phase_mask"][k + 1]) and bool(m2["phase_mask"][k + 1]):
                    acc["rows_v2_terminal_added"][k] += int((t2[k] & ~t1[k]).sum())
                acc["label_flip"][k] += int((both & (y1[k] != y2[k])).sum())
                # feature differences on the tokens alive in both dumps (the model input rows)
                al = alive1[k] & alive2[k]
                if al.any():
                    d = np.abs(X1[k][al].astype(np.float64) - X2[k][al].astype(np.float64))
                    acc["diff_sum"] += d.sum(0)
                    acc["diff_n"] += int(al.sum())
                    acc["changed_cells"] += (d > tol).sum(0)
                    acc["changed_rows_alive"] += int((d.max(1) > tol).sum())
                    acc["sx"] += X1[k][al].astype(np.float64).sum(0)
                    acc["sy"] += X2[k][al].astype(np.float64).sum(0)
                    acc["sxx"] += (X1[k][al].astype(np.float64) ** 2).sum(0)
                    acc["syy"] += (X2[k][al].astype(np.float64) ** 2).sum(0)
                    acc["sxy"] += (X1[k][al].astype(np.float64) * X2[k][al].astype(np.float64)).sum(
                        0
                    )
                if both.any():
                    d = np.abs(X1[k][both].astype(np.float64) - X2[k][both].astype(np.float64))
                    acc["risk_rows"] += int(both.sum())
                    acc["risk_rows_changed"] += int((d.max(1) > tol).sum())
                    acc["risk_diff_sum"] += d.sum(0)
                    for g, cols in acc["groups"].items():
                        acc["g_rows_changed"][g] += int((d[:, cols].max(1) > tol).sum())
                        acc["g_diff_sum"][g] += float(d[:, cols].sum())
        acc["n_splits"] += 1
    acc["names"] = names if "names" in locals() else acc.get("names")


def new_acc(n_feat: int = 150, names: list[str] | None = None) -> dict:
    z7 = lambda: np.zeros(7, dtype=np.int64)  # noqa: E731
    acc: dict = defaultdict(int)
    acc["groups"] = feature_groups(names) if names else {}
    acc["g_rows_changed"] = defaultdict(int)
    acc["g_diff_sum"] = defaultdict(float)
    for key in (
        "rows_v1",
        "rows_v2",
        "pos_v1",
        "pos_v2",
        "rows_both",
        "rows_only_v1",
        "rows_only_v2",
        "rows_v2_terminal_added",
        "label_flip",
    ):
        acc[key] = z7()
    for key in ("diff_sum", "changed_cells", "sx", "sy", "sxx", "syy", "sxy", "risk_diff_sum"):
        acc[key] = np.zeros(n_feat, dtype=np.float64)
    return acc


def summarise(acc: dict) -> dict:
    n = max(int(acc["diff_n"]), 1)
    names = acc["names"]
    mad = acc["diff_sum"] / n
    chg = acc["changed_cells"] / n
    mx, my = acc["sx"] / n, acc["sy"] / n
    vx = acc["sxx"] / n - mx**2
    vy = acc["syy"] / n - my**2
    cov = acc["sxy"] / n - mx * my
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.sqrt(vx * vy)
    feats = {
        nm: {"mean_abs_diff": float(mad[i]), "share_rows_changed": float(chg[i]),
             "corr": None if not np.isfinite(corr[i]) else float(corr[i]),
             "mean_v1": float(mx[i]), "mean_v2": float(my[i])}
        for i, nm in enumerate(names)
    }  # fmt: skip
    rb = max(int(acc["risk_rows"]), 1)
    group_stats = {
        g: {
            "n_features": len(cols),
            "mean_abs_diff_per_cell": float(acc["g_diff_sum"][g] / (rb * len(cols))),
            "share_risk_rows_changed": float(acc["g_rows_changed"][g] / rb),
        }
        for g, cols in acc["groups"].items()
    }
    return {
        "groups": group_stats,
        "rows_by_cut": {
            "v1": acc["rows_v1"].tolist(),
            "v2": acc["rows_v2"].tolist(),
            "positives_v1": acc["pos_v1"].tolist(),
            "positives_v2": acc["pos_v2"].tolist(),
            "in_both": acc["rows_both"].tolist(),
            "only_v1": acc["rows_only_v1"].tolist(),
            "only_v2": acc["rows_only_v2"].tolist(),
            "terminal_rows_added": acc["rows_v2_terminal_added"].tolist(),
            "label_flips_in_both": acc["label_flip"].tolist(),
        },
        "alive_token_rows": int(acc["diff_n"]),
        "alive_token_rows_changed": int(acc["changed_rows_alive"]),
        "risk_rows_in_both": int(acc["risk_rows"]),
        "risk_rows_in_both_changed": int(acc["risk_rows_changed"]),
        "risk_rows_changed_share": float(acc["risk_rows_changed"] / rb),
        "features": feats,
        "matches_common": int(acc["matches_common"]),
        "matches_only_v2": int(acc["matches_only_v2"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--v1", type=Path, default=Path("model_stgcn/dumps/bigcomp2027_phase_v1"))
    ap.add_argument("--v2", type=Path, default=Path("model_stgcn/dumps/bigcomp2027_phase_v2"))
    ap.add_argument("--cohort", action="append", default=None)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cohorts = args.cohort or sorted(p.name for p in args.v2.iterdir() if p.is_dir())
    names = list(
        torch.load(next(args.v2.glob("*/test.pt")), map_location="cpu", weights_only=False)["meta"][
            "feature_names"
        ]
    )
    acc = new_acc(names=names)
    per_cohort = {}
    for slug in cohorts:
        a = new_acc(names=names)
        compare_cohort(args.v1, args.v2, slug, args.tol, a)
        if a["n_splits"]:
            per_cohort[slug] = summarise(a)
            for key, val in a.items():
                if key in ("names", "groups"):
                    acc[key] = val
                elif key in ("g_rows_changed", "g_diff_sum"):
                    for g, v in val.items():
                        acc[key][g] += v
                else:
                    acc[key] = acc[key] + val
    total = summarise(acc)
    out = {
        "total": total,
        "per_cohort": {
            k: {kk: vv for kk, vv in v.items() if kk != "features"} for k, v in per_cohort.items()
        },
    }  # noqa: E501
    if args.out:
        args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("rows by cut", json.dumps(total["rows_by_cut"]))
    for g, st in total["groups"].items():
        print(
            f"group {g:34s} n={st['n_features']:3d} mean|dX|/cell={st['mean_abs_diff_per_cell']:.5f} "
            f"risk rows changed={st['share_risk_rows_changed']:.3f}"
        )
    print(
        "risk rows in both",
        total["risk_rows_in_both"],
        "changed",
        total["risk_rows_in_both_changed"],
        f"({total['risk_rows_changed_share']:.3f})",
    )
    top = sorted(total["features"].items(), key=lambda kv: -kv[1]["mean_abs_diff"])[:25]
    for name, f in top:
        print(
            f"{name:45s} mad={f['mean_abs_diff']:.4f} chg={f['share_rows_changed']:.3f} corr={f['corr']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
