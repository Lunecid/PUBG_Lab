"""Audit the O7 advantage-edge edge-shuffle null control.

This script does not train a model. It verifies on real cohort matches that the
advantage-edge shuffle used by ``model_stgcn.v3.adv_model`` is constrained to
each match-phase alive-node block and preserves the paired mask / edge-feature
multiset inside that block.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model_stgcn.splits import load_or_create_split
from pubg_lab.survival.v3.adv_model import shuffle_adj
from pubg_lab.survival.v3.advantage_edges import build_adv_tensor, fetch_raw
from pubg_lab.survival.v3.cohorts import resolve_cohort
from pubg_lab.survival.v3.factorized_data import fetch_factorized_phase_match
from pubg_lab.data.stdb.queries import list_phase_eligible_match_ids


DEFAULT_ROOT = Path("model_stgcn/checkpoints/br_o7_advantage_edge_required_v1")
OM_ROOT = Path("model_stgcn/checkpoints/br_om_ladder_required_v1")


def _canonical_block(feat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return lexicographically sorted mask+feature rows for multiset checks."""
    flat = np.concatenate([mask[..., None].astype(np.float32), feat.astype(np.float32)], axis=-1)
    flat = np.round(flat.reshape(-1, flat.shape[-1]), 6)
    order = np.lexsort(tuple(flat[:, i] for i in range(flat.shape[1] - 1, -1, -1)))
    return flat[order]


def _outside_alive_sum(arr: np.ndarray, alive: np.ndarray) -> float:
    outside = (~alive[:, None]) | (~alive[None, :])
    return float(np.abs(arr[outside]).sum())


def audit_match(
    match_id: str,
    *,
    gate: str,
    k: int,
    beta: float,
    seed: int,
) -> dict[str, Any]:
    match = fetch_factorized_phase_match(match_id)
    if match is None:
        return {"match_id": match_id, "status": "skipped", "reason": "missing factorized match"}

    raw = fetch_raw(match_id)
    a_feat, a_mask = build_adv_tensor(
        match_id,
        match.team_ids,
        match.phase_indices,
        match.node_mask,
        gate=gate,
        k=k,
        beta=beta,
        raw=raw,
    )
    in_feat = torch.from_numpy(a_feat[None].astype(np.float32))
    in_mask = torch.from_numpy(a_mask[None].astype(np.float32))
    nmask = torch.from_numpy(match.node_mask[None].astype(np.bool_))
    out_feat_t, out_mask_t = shuffle_adj(
        in_feat,
        in_mask,
        nmask,
        "edge_shuffle",
        torch.Generator().manual_seed(seed),
    )
    out_feat = out_feat_t.numpy()[0]
    out_mask = out_mask_t.numpy()[0]

    failures: list[dict[str, Any]] = []
    phase_blocks = 0
    edge_blocks = 0
    rewired_blocks = 0
    total_edges = 0
    for t, phase in enumerate(match.phase_indices):
        alive = match.node_mask[t].astype(bool)
        if int(alive.sum()) <= 1:
            continue
        phase_blocks += 1
        in_edges = int(a_mask[t][alive][:, alive].sum())
        out_edges = int(out_mask[t][alive][:, alive].sum())
        total_edges += in_edges
        if in_edges > 0:
            edge_blocks += 1

        outside_mask_sum = _outside_alive_sum(out_mask[t], alive)
        outside_feat_sum = _outside_alive_sum(np.abs(out_feat[t]).sum(axis=-1), alive)
        if outside_mask_sum != 0.0 or outside_feat_sum != 0.0:
            failures.append(
                {
                    "phase": int(phase),
                    "kind": "outside_alive_block_nonzero",
                    "outside_mask_sum": outside_mask_sum,
                    "outside_feat_sum": outside_feat_sum,
                }
            )

        if in_edges != out_edges:
            failures.append(
                {
                    "phase": int(phase),
                    "kind": "edge_count_changed",
                    "before": in_edges,
                    "after": out_edges,
                }
            )

        src_feat = a_feat[t][alive][:, alive]
        dst_feat = out_feat[t][alive][:, alive]
        src_mask = a_mask[t][alive][:, alive]
        dst_mask = out_mask[t][alive][:, alive]
        if not np.array_equal(
            _canonical_block(src_feat, src_mask), _canonical_block(dst_feat, dst_mask)
        ):
            failures.append(
                {
                    "phase": int(phase),
                    "kind": "mask_feature_multiset_changed",
                    "before_edges": in_edges,
                    "after_edges": out_edges,
                }
            )

        if in_edges > 1 and not np.array_equal(src_mask, dst_mask):
            rewired_blocks += 1

    return {
        "match_id": match_id,
        "status": "failed" if failures else "passed",
        "phase_blocks": phase_blocks,
        "edge_blocks": edge_blocks,
        "rewired_blocks": rewired_blocks,
        "total_edges": total_edges,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="Erangel TPP squad")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-matches", type=int, default=8)
    parser.add_argument("--gate", choices=("knn", "strict", "full"), default="knn")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cohort = resolve_cohort(args.cohort)
    split_json = Path(args.split_json) if args.split_json else OM_ROOT / cohort.slug / "split.json"
    out_path = (
        Path(args.out) if args.out else DEFAULT_ROOT / cohort.slug / "o7_edge_shuffle_audit.json"
    )

    ids = list_phase_eligible_match_ids(cohort.map_name, cohort.game_mode)
    train_ids, val_ids, test_ids = load_or_create_split(
        ids,
        split_json=str(split_json),
        seed=args.seed,
        val_frac=0.1,
        test_frac=0.1,
    )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    selected = split_ids[: args.max_matches]

    rows = [
        audit_match(match_id, gate=args.gate, k=args.k, beta=args.beta, seed=args.seed + i)
        for i, match_id in enumerate(selected)
    ]
    failures = [failure for row in rows for failure in row.get("failures", [])]
    summary = {
        "status": "failed" if failures else "passed",
        "cohort": cohort.label,
        "cohort_slug": cohort.slug,
        "map": cohort.map_name,
        "mode": cohort.game_mode,
        "split_json": str(split_json),
        "split": args.split,
        "gate": args.gate,
        "k": args.k,
        "beta": args.beta,
        "seed": args.seed,
        "matches_requested": args.max_matches,
        "matches_audited": sum(1 for row in rows if row["status"] != "skipped"),
        "phase_blocks": sum(int(row.get("phase_blocks", 0)) for row in rows),
        "edge_blocks": sum(int(row.get("edge_blocks", 0)) for row in rows),
        "rewired_blocks": sum(int(row.get("rewired_blocks", 0)) for row in rows),
        "total_edges": sum(int(row.get("total_edges", 0)) for row in rows),
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
