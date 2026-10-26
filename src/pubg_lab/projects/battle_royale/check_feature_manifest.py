"""Check a feature-family manifest against an STDB feature view.

The script is intentionally read-only. It reports which manifest columns are
currently available in a database view and which columns still need ETL/view
work before the V2 ablation can run.
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

import psycopg2

from pubg_lab.data.stdb.conn import DB_CONFIG, require_pg_password


DEFAULT_MANIFEST = Path("model_stgcn/feature_manifests/pubg_representation_v2.json")


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _split_view_name(view_name: str) -> tuple[str, str]:
    parts = view_name.split(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("view name must be schema.table, e.g. stdb.v_team_phase_feature_long")
    return parts[0], parts[1]


def _view_columns(view_name: str) -> set[str]:
    schema, table = _split_view_name(view_name)
    with psycopg2.connect(password=require_pg_password(), **DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_schema = %s
                   AND table_name = %s
                 ORDER BY ordinal_position
                """,
                (schema, table),
            )
            return {str(row[0]) for row in cur.fetchall()}


def build_report(manifest: dict[str, Any], view_name: str) -> dict[str, Any]:
    columns = _view_columns(view_name)
    forbidden = set(manifest.get("forbidden_input_columns", []))
    family_reports = {}
    for family_name, family in manifest["families"].items():
        wanted = list(family.get("columns", []))
        present = [col for col in wanted if col in columns]
        missing = [col for col in wanted if col not in columns]
        forbidden_hits = [col for col in wanted if col in forbidden]
        family_reports[family_name] = {
            "status": family.get("status"),
            "wanted": len(wanted),
            "present": len(present),
            "missing": len(missing),
            "forbidden_hits": forbidden_hits,
            "present_columns": present,
            "missing_columns": missing,
        }
    return {
        "manifest": manifest.get("name"),
        "view": view_name,
        "view_columns": len(columns),
        "families": family_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--view",
        default=None,
        help=(
            "Feature view/table to audit. Defaults to manifest source_view_target "
            "when present, otherwise stdb.v_team_phase_feature_long."
        ),
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    view_name = args.view or manifest.get("source_view_target") or "stdb.v_team_phase_feature_long"
    report = build_report(manifest, view_name)
    indent = 2 if args.pretty else None
    print(json.dumps(report, indent=indent, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
