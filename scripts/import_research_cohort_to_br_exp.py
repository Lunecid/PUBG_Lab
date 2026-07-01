"""Import the existing paper cohort manifest into br_exp.

This links the current paper_graph_n128_v1 split to br_ground.matches without
changing the source PUBG telemetry or derived stdb feature tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from psycopg2.extras import Json, execute_batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stdb.conn import apply_sql_file, connect


DEFAULT_COHORT_DIR = ROOT / "model_stgcn" / "experiment_datasets" / "research_cohort_flat"
DEFAULT_MEMBERSHIP = DEFAULT_COHORT_DIR / "paper_graph_n128_v1_membership.csv"
DEFAULT_MANIFEST = DEFAULT_COHORT_DIR / "paper_graph_n128_v1_manifest.json"
DEFAULT_COHORT_ID = "paper_graph_n128_v1"
FEATURE_SET_ID = "pubg_phase_graph_n128_v1"


def sha1_ids(ids: list[str]) -> str:
    return hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()


def load_membership(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Membership CSV not found: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def import_cohort(args: argparse.Namespace) -> None:
    if args.apply_schema:
        apply_sql_file(ROOT / "stdb" / "schema_br_layers.sql")

    rows = load_membership(args.membership)
    manifest = load_manifest(args.manifest)
    ids = [row["match_id"] for row in rows]
    ids_hash = sha1_ids(ids)
    canonical_ids_hash = manifest.get("ids_sha1") or ids_hash

    member_rows = []
    for row in rows:
        domain_key = "__".join(
            [
                row.get("map_name") or "unknown_map",
                row.get("perspective") or "unknown_perspective",
                row.get("team_size") or "unknown_team_size",
            ]
        )
        member_rows.append(
            (
                args.cohort_id,
                "pubg:" + row["match_id"],
                domain_key,
                row.get("split"),
                Json(
                    {
                        "map_label": row.get("map_label"),
                        "map_name": row.get("map_name"),
                        "perspective": row.get("perspective"),
                        "team_size": row.get("team_size"),
                        "game_mode": row.get("game_mode"),
                        "created_at": row.get("created_at"),
                        "duration": row.get("duration"),
                        "num_teams": row.get("num_teams"),
                        "num_players": row.get("num_players"),
                    }
                ),
            )
        )

    definition = {
        "source": "model_stgcn/experiment_datasets/research_cohort_flat",
        "membership_csv": str(args.membership.relative_to(ROOT) if args.membership.is_relative_to(ROOT) else args.membership),
        "manifest_json": str(args.manifest.relative_to(ROOT) if args.manifest.is_relative_to(ROOT) else args.manifest),
        "ids_sha1_recomputed": ids_hash,
        "ids_sha1_canonical": canonical_ids_hash,
        "manifest": manifest,
    }

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO br_repr.feature_sets (
                    feature_set_id, description, source_id, manifest_path, metadata
                ) VALUES (
                    %s, %s, 'pubg_current_l1', %s, %s
                )
                ON CONFLICT (feature_set_id) DO UPDATE
                SET description = EXCLUDED.description,
                    manifest_path = EXCLUDED.manifest_path,
                    metadata = EXCLUDED.metadata
                """,
                (
                    FEATURE_SET_ID,
                    "PUBG phase graph N<=128 research representation used by paper_graph_n128_v1.",
                    str(args.manifest),
                    Json({"cohort_id": args.cohort_id}),
                ),
            )
            cur.execute(
                """
                INSERT INTO br_exp.cohorts (
                    cohort_id, source_scope, feature_set_id, definition,
                    n_matches, ids_sha1
                ) VALUES (
                    %s, 'pubg_current_l1', %s, %s, %s, %s
                )
                ON CONFLICT (cohort_id) DO UPDATE
                SET feature_set_id = EXCLUDED.feature_set_id,
                    definition = EXCLUDED.definition,
                    n_matches = EXCLUDED.n_matches,
                    ids_sha1 = EXCLUDED.ids_sha1
                """,
                (args.cohort_id, FEATURE_SET_ID, Json(definition), len(rows), canonical_ids_hash),
            )
            execute_batch(
                cur,
                """
                INSERT INTO br_exp.cohort_members (
                    cohort_id, br_match_id, domain_key, split_name, metadata
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (cohort_id, br_match_id) DO UPDATE
                SET domain_key = EXCLUDED.domain_key,
                    split_name = EXCLUDED.split_name,
                    metadata = EXCLUDED.metadata
                """,
                member_rows,
                page_size=1000,
            )
        conn.commit()

    print(f"imported cohort: {args.cohort_id}")
    print(f"members: {len(rows)}")
    print(f"ids_sha1: {canonical_ids_hash}")
    if canonical_ids_hash != ids_hash:
        print(f"ids_sha1_recomputed: {ids_hash}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--cohort-id", default=DEFAULT_COHORT_ID)
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    import_cohort(args)


if __name__ == "__main__":
    main()
