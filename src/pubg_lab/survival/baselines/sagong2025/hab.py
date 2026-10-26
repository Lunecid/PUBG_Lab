"""Zone + habitat columns of the PTR dump per risk-set row, for the ``SAGONG-hgb-hab`` control (needs torch, no DB).

The 44 columns are exactly what the ``*-noHab`` arms remove: ``data.removed_columns(meta, NOHAB_GROUPS, NOHAB_KEEP)``
(13 ``zone`` + 31 ``habitat`` columns of ``meta['feature_groups']``; the alive-count variables sit in their own group and
stay in noHab).  Per cohort this stage writes ``<plan-dir>/<cohort>/hab44.parquet`` (``split, match_id, team_id, phase, y``
and the 44 raw columns, float32, as ``HGB-cut`` reads them) and ``hab44_audit.json``.  The rows come from
``ptr.hgb.build_rows(dump, 'HGB-cut', labels)``, i.e. the HGB-cut rows themselves (labels v2, winner censoring).  The column
set is asserted equal to the ``feature_removal`` block of the cohort's ``HGB-noHab`` ``metrics.json`` when that run exists.

``attach_hab`` joins those columns to the Sagong rows of ``fit`` by (match_id, team_id, phase) and asserts a one-to-one match
with identical labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HAB_FILE = "hab44.parquet"
AUDIT_FILE = "hab44_audit.json"
N_HAB = 44  # 13 zone + 31 habitat columns


def build_hab_table(
    dump_dir, cohort: str, plan_dir, runs_ref=None, labels: str | None = None
) -> dict:
    import pubg_lab.survival.ptr.data.data as D
    import pubg_lab.survival.ptr.models.hgb as H
    from pubg_lab.survival.ptr.models.arms import NOHAB_GROUPS, NOHAB_KEEP
    from pubg_lab.survival.ptr.common.util import dump_fingerprint

    labels = labels or D.DEFAULT_LABELS
    splits = D.load_cohort(dump_dir, cohort, labels=labels)
    meta = splits["train"]["meta"]
    cols = D.removed_columns(meta, NOHAB_GROUPS, NOHAB_KEEP)
    names = [str(meta["feature_names"][int(c)]) for c in cols]
    if len(cols) != N_HAB:
        raise AssertionError(f"{cohort}: noHab removes {len(cols)} columns, expected {N_HAB}")
    audit = {
        "cohort": cohort,
        "labels": labels,
        "n_columns": int(len(cols)),
        "names": names,
        "indices": [int(c) for c in cols],
        "groups": list(NOHAB_GROUPS),
        "kept_by_exception": list(NOHAB_KEEP),
        "dump_version": D.dump_version(splits["train"]),
        "dump_fingerprint": dump_fingerprint(dump_dir, cohort),
    }
    if runs_ref is not None:
        p = Path(runs_ref) / cohort / "HGB-noHab" / "seed42" / "metrics.json"
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found: the removed-column set cannot be asserted")
        rec = json.loads(p.read_text(encoding="utf-8"))["feature_removal"]
        if (
            rec["removed_columns"] != names
            or [int(i) for i in rec["removed_indices"]] != audit["indices"]
        ):
            raise AssertionError(
                f"{cohort}: the 44 columns differ from the HGB-noHab run's feature_removal"
            )
        audit["equals_hgb_noHab_feature_removal"] = True
    frames = []
    for s in D.SPLITS:
        rows = H.build_rows(
            splits[s], "HGB-cut", labels
        )  # the HGB-cut rows: [X (150 dump columns) | team_ai_frac]
        assert rows.feature_names[: len(meta["feature_names"])] == list(meta["feature_names"])
        df = rows.keys.copy()
        df.insert(0, "split", s)
        df["y"] = rows.y.astype(np.float64)
        block = np.asarray(rows.X[:, cols], dtype=np.float32)
        df = pd.concat([df, pd.DataFrame(block, columns=names)], axis=1)
        audit.setdefault("n_rows", {})[s] = int(len(df))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    base = Path(plan_dir) / cohort
    base.mkdir(parents=True, exist_ok=True)
    out.to_parquet(base / HAB_FILE, index=False)
    audit["n_nan"] = int(out[names].isna().sum().sum())
    (base / AUDIT_FILE).write_text(json.dumps(audit, indent=1) + "\n")
    return audit


def attach_hab(
    plan_dir, cohort: str, rows: dict, X: dict, version: str | None = None
) -> tuple[dict, list[str], dict]:
    """Append the 44 columns to the Sagong inputs ``X[split]`` (aligned with ``rows[split]``); one-to-one, labels asserted.

    ``version`` = the dump version of the plan: a habitat table written from another dump version (no ``dump_version`` in the
    audit = v1) raises ``DumpVersionError``."""
    base = Path(plan_dir) / cohort
    hab = pd.read_parquet(base / HAB_FILE)
    audit = json.loads((base / AUDIT_FILE).read_text())
    if version is not None:
        from pubg_lab.survival.ptr.data.data import check_same_version

        check_same_version(
            str(audit.get("dump_version", "v1")),
            version,
            what=f"the habitat table {base / HAB_FILE}",
            against="the plan",
        )
    names = list(audit["names"])
    out = {}
    for s, r in rows.items():
        h = hab[hab["split"] == s].drop(columns="split").rename(columns={"y": "y_dump"})
        m = r[["match_id", "team_id", "phase", "y"]].merge(
            h,
            on=["match_id", "team_id", "phase"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if len(m) != len(r) or not (m["_merge"] == "both").all():
            raise AssertionError(
                f"{cohort}/{s}: the habitat table does not cover exactly the risk-set rows"
            )
        if not np.array_equal(
            m["y"].to_numpy(dtype=np.float64), m["y_dump"].to_numpy(dtype=np.float64)
        ):
            raise AssertionError(
                f"{cohort}/{s}: labels differ between the Sagong rows and the dump rows"
            )
        if len(h) != len(r):
            raise AssertionError(f"{cohort}/{s}: {len(h)} dump rows vs {len(r)} Sagong rows")
        out[s] = np.hstack([X[s], m[names].to_numpy(dtype=np.float64)])
    info = {
        "n_columns": len(names),
        "names": names,
        "indices": audit["indices"],
        "equals_hgb_noHab_feature_removal": audit.get("equals_hgb_noHab_feature_removal"),
        "dump_fingerprint": audit["dump_fingerprint"],
        "hab44_file": [
            base.joinpath(HAB_FILE).stat().st_size,
            base.joinpath(HAB_FILE).stat().st_mtime_ns,
        ],
    }
    return out, names, info
