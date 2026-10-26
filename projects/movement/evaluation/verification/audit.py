"""Full-bank/source integrity and sample-index audit without re-querying the DB."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.data.dataset import DEFAULT


def main():
    rows = list(csv.DictReader((DEFAULT / "matches.csv").open()))
    result = {}
    hashes = {}
    indices = {s: np.load(DEFAULT / f"{s}_index.npy") for s in ("train", "validation")}
    assert set(indices["train"][:, 0]).isdisjoint(indices["validation"][:, 0])
    total = Counter()
    for row in rows:
        path = DEFAULT / "banks" / f"{row['match_id']}.npz"
        hashes[row["match_id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as b:
            assert np.isfinite(b["x"]).all() and np.isfinite(b["target"]).all()
            assert not b["x"][~b["mask"]].any()
            tt = np.broadcast_to(b["times"][:, None], b["source_time"].shape)
            has = b["source_time"] >= 0
            assert (b["source_time"][has] <= tt[has]).all()
            assert ((tt - b["source_time"])[b["valid"]] <= 15).all()
            assert (abs(b["target_time"][b["target_valid"]] - tt[b["target_valid"]]) <= 5).all()
            assert (b["items"] >= 0).all()
            c = result.setdefault(row["split"], Counter())
            c["valid_ground_actor_frames"] += int(b["valid"].sum())
            c["valid_target_actor_frames"] += int(b["target_valid"].sum())
            c["equipment_truncated_tokens"] += int(b["quality"][0])
            c["valid_actor_frames_with_ambiguous_equipment"] += int(
                ((b["x"][:, :, 17] > 0) & b["valid"]).sum()
            )
            c["valid_actor_frames_without_equip_history"] += int(
                (~b["item_seen"] & b["valid"]).sum()
            )
            c["unknown_vocab_item_tokens"] += int((b["items"] == 1).sum())
            c["post_first_death_positive_position_rows_excluded"] += int(b["quality"][2])
            errors = abs(b["target_time"][b["target_valid"]] - tt[b["target_valid"]])
            c["target_time_abs_error_sum_s"] += float(errors.sum())
            c["target_time_count"] += len(errors)
            index = indices[row["split"]]
            ix = index[index[:, 0] == int(row["match_index"])]
            k = ix[:, 1]
            ego = ix[:, 2]
            assert b["valid"][k, ego].all()
            assert (k >= 3).all()
            c["anchors_all_4_ego_history_positions_valid"] += int(
                b["valid"][k[:, None] - np.arange(3, -1, -1), ego[:, None]].all(1).sum()
            )
            for kind, label in [(1, "human"), (2, "bot")]:
                sel = ix[:, 3] == kind
                c[label + "_anchors"] += int(sel.sum())
                c[label + "_complete_joint_60s"] += int(ix[sel, 5].sum())
    total["banks_checked"] = len(rows)
    total["split_disjoint"] = True
    report = {
        "checks": dict(total),
        "by_split": {s: dict(c) for s, c in result.items()},
        "limitations": [
            "first-death censoring excludes recall episodes",
            "raw safe/poison fields approximate publicly available zone information; actual UI announcement timing unverified",
            "full server view means available normalized telemetry, not complete engine state",
            "overlapping windows are not independent experimental units",
        ],
    }
    (ROOT / "reports/heterograph_dataset_v1/full_audit.json").write_text(
        json.dumps(report, indent=2)
    )
    (DEFAULT / "bank_sha256.json").write_text(json.dumps(hashes, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
