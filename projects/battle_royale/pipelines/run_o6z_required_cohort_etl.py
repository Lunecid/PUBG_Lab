"""Build O6Z Z-affordance features for the required PUBG cohorts.

This runner keeps the user's experimental unit intact: each map / perspective /
team-size cohort is handled separately.  `--shards` only partitions match IDs
inside each cohort so multiple workers can populate the same feature table
without mixing cohorts for model training.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import json
import logging
from pathlib import Path

from pubg_lab.survival.v3.data.cohorts import REQUIRED_COHORTS, resolve_cohort
from pubg_lab.data.stdb.pipelines.z_affordance_feature_etl import (
    apply_schema,
    build_batch,
    select_match_ids,
)

logger = logging.getLogger("projects.battle_royale.pipelines.run_o6z_required_cohort_etl")

DEFAULT_OUT_DIR = Path("model_stgcn/checkpoints/br_om_ladder_required_v1/o6z_required_cohort_etl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", action="append", default=None, help="Exact cohort label; repeatable"
    )
    parser.add_argument("--refresh", action="store_true", help="Rebuild even if rows already exist")
    parser.add_argument("--limit-per-cohort", type=int)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.cohort:
        cohorts = tuple(resolve_cohort(value) for value in args.cohort)
    else:
        cohorts = REQUIRED_COHORTS

    apply_schema()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for spec in cohorts:
        match_ids = select_match_ids(
            limit=args.limit_per_cohort,
            refresh=args.refresh,
            match_id=None,
            map_name=spec.map_name,
            mode=spec.game_mode,
            shards=args.shards,
            shard_index=args.shard_index,
        )
        logger.info(
            "cohort=%s map=%s mode=%s shard=%s/%s selected=%s",
            spec.label,
            spec.map_name,
            spec.game_mode,
            args.shard_index,
            args.shards,
            len(match_ids),
        )
        result = build_batch(match_ids)
        summary = {
            "cohort": spec.label,
            "cohort_slug": spec.slug,
            "map_name": spec.map_name,
            "game_mode": spec.game_mode,
            "shards": args.shards,
            "shard_index": args.shard_index,
            "selected_matches": len(match_ids),
            **result,
        }
        summaries.append(summary)
        path = args.out_dir / f"{spec.slug}__shard_{args.shard_index}_of_{args.shards}.json"
        path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)

    aggregate = {
        "shards": args.shards,
        "shard_index": args.shard_index,
        "cohort_count": len(summaries),
        "selected_matches": sum(int(s["selected_matches"]) for s in summaries),
        "ok": sum(int(s["ok"]) for s in summaries),
        "failed": sum(int(s["failed"]) for s in summaries),
        "summaries": summaries,
    }
    aggregate_path = args.out_dir / f"aggregate__shard_{args.shard_index}_of_{args.shards}.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, default=str), flush=True)
    return 0 if aggregate["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
