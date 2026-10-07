"""Feature parity table of the reconstruction (definitional status + descriptive statistics of the train rows).

    python -m adaptation.parity --data-dir model_stgcn/dumps/krafton_winprob_v2 --out third_party/krafton_pubg_winprob/PARITY_TABLE.csv

``status``: exact = same definition from the same event type of this database; proxy = approximated from a coarser /
different signal (see ``note``); zero = not buildable, the column is 0 for every row.  The statistics are over the
alive-squad rows of the train samples (pooled over the cohorts, <= 200k random rows per cohort); ``nonzero_share`` is
the share of rows whose value is not 0 (for the distance / count features that use -1 as 'unknown', the share of rows
equal to -1 is given as ``unknown_share``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptation import schema as S  # noqa: E402
from adaptation.data import _load_shards  # noqa: E402

BLOCKS = [
    ("n_pickup", "action cumulative"), ("n_drop", "action cumulative"), ("n_kill", "action cumulative"), ("n_finish", "action cumulative"),
    ("n_groggy", "action cumulative"), ("n_dmg", "action cumulative"), ("sum_dmg", "action cumulative"), ("n_take_dmg", "action cumulative"),
    ("sum_take_dmg", "action cumulative"), ("n_heal", "action cumulative"), ("sum_heal_amount", "action cumulative"), ("n_vehicle_ride", "action cumulative"),
    ("n_teammates", "location snapshot"), ("n_enemies", "location snapshot"), ("dist_from_bluezone", "location snapshot"), ("dist_from_whitezone", "location snapshot"),
    ("dist_from_bluezone_v2", "location snapshot"), ("dist_from_whitezone_v2", "location snapshot"), ("is_vehicle_count", "vehicle snapshot"),
    ("item_helmet_level", "armor / backpack"), ("item_vest_level", "armor / backpack"), ("item_backpack_level", "armor / backpack"),
    ("squad_total_health", "team status"), ("squad_alive_count", "team status"), ("squad_non_groggy_count", "team status"), ("squad_out_bluezone_count", "team status"),
]  # fmt: skip


def block_of(name: str) -> str:
    for n, b in BLOCKS:
        if n == name:
            return b
    if name.startswith(("n_dmg_by_", "sum_dmg_by_")):
        return "action cumulative"
    if name.startswith("n_heal_"):
        return "action cumulative"
    if name.startswith("item_") and name.endswith("_count"):
        return "throwables" if any(k in name for k in ("throwables", "grenade", "smoke", "flashbang", "molotov")) else "aids"
    return "other"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rng = np.random.RandomState(0)
    chunks = []
    for c in sorted(p.name for p in Path(a.data_dir).iterdir() if (p / "shards" / "train").is_dir()):
        arrays, _ = _load_shards(Path(a.data_dir) / c / "shards" / "train")
        X = arrays["X"]
        idx = rng.choice(len(X), size=min(200_000, len(X)), replace=False)
        chunks.append(X[idx])
    X = np.concatenate(chunks)
    rows = []
    for j, name in enumerate(S.FEATURES):
        st, src, note = S.PARITY[name]
        x = X[:, j].astype(np.float64)
        rows.append({
            "feature": name, "block": block_of(name), "status": st, "source": src, "note": note, "mean": x.mean(), "std": x.std(),
            "p50": np.percentile(x, 50), "p99": np.percentile(x, 99), "nonzero_share": float((x != 0).mean()), "unknown_share": float((x == -1).mean()),
        })  # fmt: skip
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, float_format="%.6g")
    print(df.groupby("status").size().to_string())
    print(f"rows used: {len(X)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
