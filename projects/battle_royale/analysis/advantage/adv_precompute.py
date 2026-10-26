"""Precompute & cache phase-tier tensors + fused-gate advantage edges to disk.
One-time DB pass so training never touches the DB (fast). Phase tier only,
trimmed to N_MAX teams (squad <=26) for small dense N x N ops on MPS.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))
import argparse
import time
import logging
import numpy as np
import torch

from pubg_lab.survival.data.split_access import load_or_create_split
from pubg_lab.survival.v3.data.factorized_data import fetch_factorized_phase_match
from pubg_lab.survival.v3.features.advantage_edges import (
    fetch_raw,
    build_adv_tensor,
    EDGE_FEATURE_DIM,
    Z_SCALE,
    ALIVE_SCALE,
    HEALTH_SCALE,
    ZONE_SCALE,
)
from pubg_lab.survival.v3.data.cohorts import resolve_map_mode
from pubg_lab.survival.v3.data.spec import FEATURE_PRESETS
from pubg_lab.data.stdb.queries.queries import list_phase_eligible_match_ids
from pubg_lab.data.stdb.connection.conn import connect

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("adv_precompute")
GATES = ("knn", "strict", "full")


def fetch_radius_batch(mids):
    out = {}
    with connect() as conn:
        cur = conn.cursor()
        for i in range(0, len(mids), 1500):
            chunk = mids[i : i + 1500]
            cur.execute(
                "SELECT match_id, phase_idx, max(safe_radius_start_m) "
                "FROM stdb.v_team_phase_feature_v2 WHERE match_id = ANY(%s) GROUP BY 1,2",
                (chunk,),
            )
            for mid, ph, r in cur.fetchall():
                if r is not None:
                    out.setdefault(mid, {})[int(ph)] = float(r)
        cur.close()
    return out


def node_abs_features(m, raw, N):
    """Per-(phase,team) absolute [z, alive, health, zone] from snapshot state,
    same scales as the edge relative-state features. The CONTROL: give nodes the
    absolutes so the advantage EDGE can only add the RELATIONAL (pairwise) part."""
    T = m.X_node.shape[0]
    nb = np.zeros((T, N, 4), dtype=np.float32)
    for t, p in enumerate(m.phase_indices):
        p = int(p)
        if p <= 0:
            continue
        for i, team in enumerate(m.team_ids[:N]):
            st = raw.state.get((int(team), p)) if team else None
            if st:
                nb[t, i] = [
                    st["z"] / Z_SCALE,
                    st["alive"] / ALIVE_SCALE,
                    st["health"] / HEALTH_SCALE,
                    st["zone"] / ZONE_SCALE,
                ]
    return nb


def build_split(
    ids,
    radius_all,
    n_max,
    k,
    beta,
    gates=GATES,
    node_abs=False,
    feature_preset="om5_relation_structure",
):
    items = []
    t0 = time.time()
    for n, mid in enumerate(ids):
        m = fetch_factorized_phase_match(mid, feature_preset=feature_preset)
        if m is None:
            continue
        T = m.X_node.shape[0]
        N = min(n_max, m.X_node.shape[1])
        try:
            raw = fetch_raw(mid)
        except Exception:
            continue
        rad = radius_all.get(mid, raw.radius)
        Xn = m.X_node[:, :N].astype(np.float32)
        if node_abs:
            Xn = np.concatenate([Xn, node_abs_features(m, raw, N)], axis=-1)
        rec = {
            "match_id": mid,
            "X_node": Xn,
            "G": m.G.astype(np.float32),
            "y_hazard": m.y_hazard[:, :N].astype(np.float32),
            "target_mask": m.target_mask[:, :N].astype(np.bool_),
            "node_mask": m.node_mask[:, :N].astype(np.bool_),
            "phase_indices": np.asarray(m.phase_indices, dtype=np.int64),
        }
        for g in gates:
            A_feat, A_mask = build_adv_tensor(
                mid,
                m.team_ids[:N],
                m.phase_indices,
                m.node_mask[:, :N],
                gate=g,
                k=k,
                beta=beta,
                raw=raw,
                radius_by_phase=rad,
            )
            rec[f"A_feat_{g}"] = A_feat.astype(np.float16)
            rec[f"A_mask_{g}"] = A_mask.astype(np.bool_)
        items.append(rec)
        if (n + 1) % 500 == 0:
            log.info("  %d/%d (%.0fs)", n + 1, len(ids), time.time() - t0)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split-json", default="model_stgcn/checkpoints/v3_full_main_causal/split.json"
    )
    ap.add_argument("--out-dir", default="model_stgcn/checkpoints/adv_cache")
    ap.add_argument(
        "--cohort", default=None, help="named required cohort, e.g. 'Erangel TPP squad'"
    )
    ap.add_argument("--map", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--feature-preset", choices=FEATURE_PRESETS, default="om5_relation_structure")
    ap.add_argument("--n-max", type=int, default=32)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--max-val", type=int, default=1521)
    ap.add_argument("--max-test", type=int, default=1521)
    ap.add_argument("--gates", default=",".join(GATES))
    # node absolutes (z/alive/health/zone) are part of the CORRECT node spec -> ON by default.
    ap.add_argument("--node-abs", dest="node_abs", action="store_true", default=True)
    ap.add_argument(
        "--no-node-abs",
        dest="node_abs",
        action="store_false",
        help="(diagnostic only) deficient nodes without elevation/state",
    )
    args = ap.parse_args()
    gates = tuple(g for g in args.gates.split(",") if g)
    if args.cohort is None and args.map is None and args.mode is None:
        args.mode = "squad"
    try:
        map_filter, mode_filter, cohort_spec = resolve_map_mode(
            cohort=args.cohort,
            map_name=args.map,
            mode=args.mode,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.map = map_filter
    args.mode = mode_filter
    args.cohort = cohort_spec.label if cohort_spec else args.cohort
    args.cohort_slug = cohort_spec.slug if cohort_spec else None

    import os

    os.makedirs(args.out_dir, exist_ok=True)
    ids = list_phase_eligible_match_ids(args.map, args.mode, None)
    tr, va, te = load_or_create_split(
        ids,
        split_json=args.split_json,
        seed=42,
        val_frac=0.1,
        test_frac=0.1,
        metadata={
            "source": "projects.battle_royale.analysis.advantage.adv_precompute",
            "cohort": args.cohort,
            "cohort_slug": args.cohort_slug,
            "map": args.map,
            "mode": args.mode,
            "feature_preset": args.feature_preset,
            "node_abs": args.node_abs,
            "gates": gates,
        },
    )
    tr, va, te = tr[: args.max_train], va[: args.max_val], te[: args.max_test]
    log.info(
        "cohort=%s map=%s mode=%s feature_preset=%s node_abs=%s",
        args.cohort,
        args.map,
        args.mode,
        args.feature_preset,
        args.node_abs,
    )
    log.info(
        "split sizes: train=%d val=%d test=%d | n_max=%d k=%d beta=%.1f F=%d",
        len(tr),
        len(va),
        len(te),
        args.n_max,
        args.k,
        args.beta,
        EDGE_FEATURE_DIM,
    )

    radius_all = fetch_radius_batch(list(set(tr + va + te)))
    log.info("fetched radius for %d matches", len(radius_all))

    for name, split in (("test", te), ("val", va), ("train", tr)):
        log.info("building %s ...", name)
        items = build_split(
            split,
            radius_all,
            args.n_max,
            args.k,
            args.beta,
            gates=gates,
            node_abs=args.node_abs,
            feature_preset=args.feature_preset,
        )
        path = os.path.join(args.out_dir, f"{name}.pt")
        torch.save(
            {
                "items": items,
                "n_max": args.n_max,
                "k": args.k,
                "beta": args.beta,
                "edge_dim": EDGE_FEATURE_DIM,
                "gates": gates,
                "cohort": args.cohort,
                "cohort_slug": args.cohort_slug,
                "map": args.map,
                "mode": args.mode,
                "feature_preset": args.feature_preset,
                "node_abs": args.node_abs,
            },
            path,
        )
        log.info("saved %s: %d matches -> %s", name, len(items), path)
    log.info("PRECOMPUTE DONE")


if __name__ == "__main__":
    main()
