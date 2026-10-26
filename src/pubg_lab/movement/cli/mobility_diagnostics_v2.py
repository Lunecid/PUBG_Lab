"""Post-hoc raw-event check of aerial transport in the fixed evaluation cohort."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import json
import sys
from pathlib import Path
import numpy as np

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.dataset import TrajectoryDataset
from pubg_lab.movement.bank import read_raw

OUT = ROOT / "reports/heterograph_diffusion_experiment_v2"


def main():
    ds = TrajectoryDataset(split="validation")
    ids = np.load(OUT / "evaluation_indices.npy")
    groups = {}
    for number, i in enumerate(ids):
        groups.setdefault(int(ds.index[i, 0]), []).append((number, int(i)))
    records = []
    for mi, entries in groups.items():
        mid = ds.rows[mi]["match_id"]
        raw = read_raw(ds.root / "raw" / f"{mid}.json.gz")
        rows = raw["tables"]["positions"]
        b = ds.bank(mid)
        emergency = [
            r
            for r in rows
            if "EmergencyPickup" in str(r.get("vehicle_type", ""))
            or "EmergencyPickup" in str(r.get("vehicle_class", ""))
        ]
        for number, i in entries:
            _, k, ego, *_ = ds.index[i]
            t = float(b["times"][k])
            team = b["team"][ego]
            members = set(np.flatnonzero((b["team"] == team) & b["valid"][k]).tolist())
            current = [r for r in emergency if r["actor"] in members and t - 15 <= r["t"] <= t]
            future = [r for r in emergency if r["actor"] in members and t < r["t"] <= t + 65]
            if current or future:
                records.append(
                    {
                        "evaluation_position": number,
                        "match_id": mid,
                        "anchor_s": t,
                        "ego_actor": int(ego),
                        "current_team_emergency_pickup_evidence": bool(current),
                        "future_team_emergency_pickup_evidence": bool(future),
                        "future_first_evidence_s": min([r["t"] for r in future], default=None),
                    }
                )
    case_mid = "50f8ad89-163e-47cd-b6b1-e2967dad040a"
    raw = read_raw(ds.root / "raw" / f"{case_mid}.json.gz")
    team = raw["actors"][29]["team"]
    members = [i for i, a in enumerate(raw["actors"]) if a["team"] == team]
    evidence = [
        r for r in raw["tables"]["positions"] if r["actor"] in members and 430 <= r["t"] <= 520
    ]
    result = {
        "status": "POST_HOC_DESCRIPTIVE_RAW_EVENT_CHECK",
        "meaning": "Raw EmergencyPickup vehicle evidence in current 15s or future 65s, fixed current valid team. Future flags are diagnostic labels only, never model inputs or evaluation exclusion rules. Current evidence is a recent-event proxy, not guaranteed current attachment. No main metrics or model selection changed.",
        "summary": {
            "current_evidence_windows": sum(
                r["current_team_emergency_pickup_evidence"] for r in records
            ),
            "future_evidence_windows": sum(
                r["future_team_emergency_pickup_evidence"] for r in records
            ),
            "future_without_current_evidence_windows": sum(
                r["future_team_emergency_pickup_evidence"]
                and not r["current_team_emergency_pickup_evidence"]
                for r in records
            ),
        },
        "records": records,
        "largest_error_case_raw_evidence": evidence,
    }
    (OUT / "mobility_diagnostics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
