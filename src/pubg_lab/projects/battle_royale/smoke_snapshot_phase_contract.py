"""Smoke-test the snapshot-first phase-survival tensor contract.

This intentionally does not train a full model. It checks that:

* snapshot tensors have the canonical 10s shapes;
* phase labels/masks have the canonical phase-cut shapes;
* every labeled hazard target is at-risk at the corresponding cut phase;
* the snapshot model can run a tiny forward pass on fetched tensors.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import json
from dataclasses import asdict, dataclass

import torch

from pubg_lab.survival.snapshot_phase_survival import SnapshotPhaseSurvivalGNN
from pubg_lab.data.stdb.conn import connect
from pubg_lab.data.stdb.queries import (
    PHASE_N_MAX,
    PHASE_T_MAX,
    fetch_phase_match_tensors,
    list_phase_eligible_match_ids,
)
from pubg_lab.data.stdb.snapshot_queries import (
    SNAP_EDGE_TYPES,
    SNAP_FEATURE_DIM,
    SNAP_N_MAX,
    SNAP_T_MAX,
    fetch_snapshot_match_tensors,
)


@dataclass
class MatchSmokeReport:
    match_id: str
    snapshot_shape: tuple[int, ...]
    edge_shape: tuple[int, ...]
    hazard_shape: tuple[int, ...]
    target_count: int
    at_risk_target_violations: int
    team_mask_violations: int
    phase_start_mask_disagreements: int
    model_forward_ok: bool


def _label_rows(match_id: str) -> dict[tuple[int, int], dict[str, object]]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT team_id, cut_phase, alive_at_phase_start, alive_at_phase_end,
                       target_defined, next_phase_idx
                  FROM stdb.team_phase_labels
                 WHERE match_id = %s
                """,
                (match_id,),
            )
            rows: dict[tuple[int, int], dict[str, object]] = {}
            for (
                team_id,
                cut_phase,
                alive_start,
                alive_end,
                target_defined,
                next_phase_idx,
            ) in cur.fetchall():
                rows[(int(team_id), int(cut_phase))] = {
                    "alive_at_phase_start": bool(alive_start),
                    "alive_at_phase_end": bool(alive_end),
                    "target_defined": bool(target_defined),
                    "next_phase_idx": next_phase_idx,
                }
            return rows


def smoke_match(match_id: str, *, run_forward: bool) -> MatchSmokeReport:
    phase = fetch_phase_match_tensors(match_id)
    if phase is None:
        raise RuntimeError(f"{match_id}: phase tensors unavailable")
    snap = fetch_snapshot_match_tensors(match_id)
    if snap is None:
        raise RuntimeError(f"{match_id}: snapshot tensors unavailable")

    assert snap.X_s.shape == (SNAP_T_MAX, SNAP_N_MAX, SNAP_FEATURE_DIM), snap.X_s.shape
    assert snap.A_s.shape == (SNAP_T_MAX, len(SNAP_EDGE_TYPES), SNAP_N_MAX, SNAP_N_MAX), (
        snap.A_s.shape
    )
    assert phase.y_hazard.shape == (PHASE_T_MAX - 1, PHASE_N_MAX), phase.y_hazard.shape
    assert phase.target_mask.shape == (PHASE_T_MAX - 1, PHASE_N_MAX), phase.target_mask.shape
    assert phase.node_mask.shape == (PHASE_T_MAX, PHASE_N_MAX), phase.node_mask.shape
    assert phase.team_mask.shape == (PHASE_N_MAX,), phase.team_mask.shape

    label_rows = _label_rows(match_id)
    team_ids = [int(t) for t in phase.team_ids]
    at_risk_violations = 0
    team_violations = 0
    phase_start_disagreements = 0
    for out_t in range(PHASE_T_MAX - 1):
        cut_phase = out_t + 1
        for i, labeled in enumerate(phase.target_mask[out_t]):
            if not labeled:
                continue
            team_id = team_ids[i]
            row = label_rows.get((team_id, cut_phase))
            if row is None:
                at_risk_violations += 1
                continue
            # The prediction is made at the phase-k cut after observing phase k.
            # Therefore the at-risk condition is alive_at_phase_end, not
            # alive_at_phase_start.
            if not row["alive_at_phase_end"] or row["next_phase_idx"] is None:
                at_risk_violations += 1
            if not row["alive_at_phase_start"]:
                phase_start_disagreements += 1
            if not phase.team_mask[i]:
                team_violations += 1

    forward_ok = False
    if run_forward:
        model = SnapshotPhaseSurvivalGNN(
            SNAP_FEATURE_DIM,
            snap_hidden=8,
            snap_mid=12,
            snap_blocks=1,
            snap_edge_types=len(SNAP_EDGE_TYPES),
            t_phase=PHASE_T_MAX,
            dropout=0.0,
        ).eval()
        with torch.no_grad():
            hazard, cut_mask = model(
                torch.from_numpy(snap.X_s).unsqueeze(0).float(),
                torch.from_numpy(snap.A_s).unsqueeze(0).float(),
                torch.from_numpy(snap.phase_of_snap).unsqueeze(0),
                torch.from_numpy(snap.node_mask).unsqueeze(0),
            )
        assert hazard.shape == (1, PHASE_T_MAX - 1, PHASE_N_MAX), hazard.shape
        assert cut_mask.shape == (1, PHASE_T_MAX - 1), cut_mask.shape
        forward_ok = True

    return MatchSmokeReport(
        match_id=match_id,
        snapshot_shape=tuple(int(x) for x in snap.X_s.shape),
        edge_shape=tuple(int(x) for x in snap.A_s.shape),
        hazard_shape=tuple(int(x) for x in phase.y_hazard.shape),
        target_count=int(phase.target_mask.sum()),
        at_risk_target_violations=at_risk_violations,
        team_mask_violations=team_violations,
        phase_start_mask_disagreements=phase_start_disagreements,
        model_forward_ok=forward_ok,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", action="append", default=[])
    parser.add_argument("--map", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--run-forward", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    match_ids = list(args.match_id)
    if not match_ids:
        match_ids = list_phase_eligible_match_ids(args.map, args.mode, args.limit)
    if not match_ids:
        raise SystemExit("no match ids available")

    reports: list[MatchSmokeReport] = []
    failures: list[str] = []
    for match_id in match_ids:
        try:
            report = smoke_match(match_id, run_forward=args.run_forward)
            reports.append(report)
            if report.at_risk_target_violations or report.team_mask_violations:
                failures.append(
                    f"{match_id}: at_risk_target_violations={report.at_risk_target_violations} "
                    f"team_mask_violations={report.team_mask_violations}"
                )
        except Exception as exc:  # noqa: BLE001 - smoke report should continue across matches
            failures.append(f"{match_id}: {exc}")

    payload = {
        "checked": len(reports),
        "requested": len(match_ids),
        "failures": failures,
        "reports": [asdict(r) for r in reports],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"requested={payload['requested']} checked={payload['checked']} failures={len(failures)}"
        )
        for report in reports:
            print(
                f"{report.match_id}: X_s={report.snapshot_shape} A_s={report.edge_shape} "
                f"y={report.hazard_shape} targets={report.target_count} "
                f"forward={report.model_forward_ok} "
                f"phase_start_disagreements={report.phase_start_mask_disagreements}"
            )
        for failure in failures:
            print(f"FAIL: {failure}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
