"""Catalog Caldera OpenUSD geometry metadata into the BR ground layer.

This script keeps the heavy OpenUSD scene files under data/br_l0/caldera and
stores only a scene-graph catalog in PostgreSQL: prim paths, types, references,
payloads, variants, light bbox metadata, and provenance.  It is intentionally
metadata-first so the Caldera geometry can become a spatial analysis substrate
without duplicating billions of points in the database.

Examples:

    .venv/bin/python scripts/caldera_ingest_geometry_catalog.py --dry-run --limit 2000
    PG_PASSWORD=... .venv/bin/python scripts/caldera_ingest_geometry_catalog.py --apply-schema
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg2.extras
from psycopg2.extras import Json
from pxr import Kind, Usd, UsdGeom


ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))

from pubg_lab.data.stdb.connection.conn import apply_sql_file, connect


SOURCE_ID = "activision_caldera"
ENVIRONMENT_ID = "caldera_map_source_v1"
SOURCE_GAME = "Call of Duty: Warzone Caldera"
FEATURE_SET_ID = "caldera_environment_catalog_v1"
CHECKOUT_ARTIFACT_ID = "caldera_source_sparse_checkout"
STAGE_ARTIFACT_ID = "caldera_map_source_mp_wz_island_usd"
DEFAULT_STAGE = ROOT / "data" / "br_l0" / "caldera" / "repo" / "map_source" / "mp_wz_island.usd"


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def ensure_source_and_feature_set(
    cur, args: argparse.Namespace, prim_count: int | None = None
) -> None:
    cur.execute(
        """
        INSERT INTO br_raw.sources (
            source_id, source_game, provider, canonical_url, license_name,
            license_url, observability, notes
        ) VALUES (
            %s, %s, 'Activision', 'https://github.com/Activision/caldera',
            'Activision Non-Commercial Use License Agreement',
            'https://github.com/Activision/caldera/blob/main/LICENSE.md',
            'geometry_plus_anonymized_player_point_clouds',
            %s
        )
        ON CONFLICT (source_id) DO UPDATE
        SET notes = br_raw.sources.notes || EXCLUDED.notes,
            observability = EXCLUDED.observability
        """,
        (
            SOURCE_ID,
            SOURCE_GAME,
            Json(
                {
                    "geometry_catalog": {
                        "environment_id": args.environment_id,
                        "stage": relative(args.stage),
                    }
                }
            ),
        ),
    )

    checkout = args.stage.parents[1]
    if (checkout / ".git").exists():
        checkout_bytes = tree_size(checkout)
        cur.execute(
            """
            INSERT INTO br_raw.artifacts (
                artifact_id, source_id, layer_name, artifact_kind, uri,
                local_path, content_format, size_bytes, git_commit, metadata
            ) VALUES (
                %s, %s, 'L0', 'git_sparse_checkout',
                'https://github.com/Activision/caldera', %s, 'git/openusd',
                %s, %s, %s
            )
            ON CONFLICT (artifact_id) DO UPDATE
            SET local_path = EXCLUDED.local_path,
                size_bytes = EXCLUDED.size_bytes,
                git_commit = EXCLUDED.git_commit,
                metadata = br_raw.artifacts.metadata || EXCLUDED.metadata
            """,
            (
                CHECKOUT_ARTIFACT_ID,
                SOURCE_ID,
                relative(checkout),
                checkout_bytes,
                git_commit(checkout),
                Json({"profile": "source", "raw_storage_policy": "external_filesystem"}),
            ),
        )

    cur.execute(
        """
        INSERT INTO br_raw.artifacts (
            artifact_id, source_id, layer_name, artifact_kind, uri,
            local_path, content_format, size_bytes, sha256, metadata
        ) VALUES (
            %s, %s, 'L0', 'openusd_stage',
            'https://github.com/Activision/caldera/blob/main/map_source/mp_wz_island.usd',
            %s, 'usd', %s, %s, %s
        )
        ON CONFLICT (artifact_id) DO UPDATE
        SET local_path = EXCLUDED.local_path,
            size_bytes = EXCLUDED.size_bytes,
            sha256 = EXCLUDED.sha256,
            metadata = br_raw.artifacts.metadata || EXCLUDED.metadata
        """,
        (
            STAGE_ARTIFACT_ID,
            SOURCE_ID,
            relative(args.stage),
            args.stage.stat().st_size,
            sha256_file(args.stage),
            Json(
                {
                    "environment_id": args.environment_id,
                    "load_policy": args.load_policy,
                    "cataloged_prim_count": prim_count,
                }
            ),
        ),
    )

    cur.execute(
        """
        INSERT INTO br_repr.feature_sets (
            feature_set_id, description, source_id, manifest_path, metadata
        ) VALUES (
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (feature_set_id) DO UPDATE
        SET description = EXCLUDED.description,
            manifest_path = EXCLUDED.manifest_path,
            metadata = br_repr.feature_sets.metadata || EXCLUDED.metadata
        """,
        (
            FEATURE_SET_ID,
            "Caldera OpenUSD environment scene-graph catalog for spatial substrate features.",
            SOURCE_ID,
            relative(args.stage),
            Json(
                {
                    "environment_id": args.environment_id,
                    "representation_role": "geometry_metadata_catalog",
                    "raw_geometry_policy": "external_l0_openusd",
                }
            ),
        ),
    )


def list_op_items(value: Any) -> list[Any]:
    if value is None:
        return []
    items: list[Any] = []
    for method_name in (
        "GetAddedOrExplicitItems",
        "GetExplicitItems",
        "GetPrependedItems",
        "GetAppendedItems",
    ):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                items.extend(method())
            except Exception:
                pass
    for attr_name in ("explicitItems", "prependedItems", "appendedItems", "addedItems"):
        try:
            attr_items = getattr(value, attr_name)
        except Exception:
            continue
        if attr_items:
            items.extend(attr_items)

    deduped = []
    seen = set()
    for item in items:
        key = str(item)
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def authored_asset_paths(prim, metadata_name: str) -> list[str]:
    values = []
    for spec in prim.GetPrimStack():
        try:
            list_op = spec.GetInfo(metadata_name)
        except Exception:
            list_op = None
        for item in list_op_items(list_op):
            asset_path = getattr(item, "assetPath", None)
            prim_path = getattr(item, "primPath", None)
            if asset_path:
                suffix = f"<{prim_path}>" if prim_path else ""
                values.append(f"{asset_path}{suffix}")
            else:
                values.append(str(item))
    return sorted(set(values))


def variant_sets_for_prim(prim) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    try:
        variant_sets = prim.GetVariantSets()
        names = variant_sets.GetNames()
    except Exception:
        return result
    for name in names:
        try:
            variant_set = variant_sets.GetVariantSet(name)
            result[name] = {
                "selection": variant_set.GetVariantSelection(),
                "variants": list(variant_set.GetVariantNames()),
            }
        except Exception as exc:
            result[name] = {"error": str(exc)}
    return result


def prim_kind(prim) -> str | None:
    try:
        model = Usd.ModelAPI(prim)
        kind = model.GetKind()
    except Exception:
        return None
    return kind or None


def attr_names(prim, max_attrs: int) -> list[str]:
    names = []
    try:
        attrs = prim.GetAttributes()
    except Exception:
        return names
    for attr in attrs[:max_attrs]:
        names.append(attr.GetName())
    return names


def bbox_for_prim(
    prim, bbox_cache, scale: float
) -> tuple[float, float, float, float, float, float] | None:
    if bbox_cache is None:
        return None
    try:
        bbox = bbox_cache.ComputeWorldBound(prim)
        box_range = bbox.ComputeAlignedBox()
        if box_range.IsEmpty():
            return None
        mn = box_range.GetMin()
        mx = box_range.GetMax()
    except Exception:
        return None
    return (
        float(mn[0]) * scale,
        float(mn[1]) * scale,
        float(mn[2]) * scale,
        float(mx[0]) * scale,
        float(mx[1]) * scale,
        float(mx[2]) * scale,
    )


def open_stage(path: Path, load_policy: str) -> Usd.Stage:
    load = Usd.Stage.LoadNone if load_policy == "none" else Usd.Stage.LoadAll
    stage = Usd.Stage.Open(str(path), load)
    if stage is None:
        raise RuntimeError(f"Could not open stage: {path}")
    return stage


def iter_prims(stage: Usd.Stage, max_depth: int | None):
    stack = [(child, 1) for child in reversed(list(stage.GetPseudoRoot().GetChildren()))]
    while stack:
        prim, depth = stack.pop()
        yield prim, depth
        if max_depth is not None and depth >= max_depth:
            continue
        children = list(prim.GetChildren())
        for child in reversed(children):
            stack.append((child, depth + 1))


def catalog_rows(
    stage: Usd.Stage, args: argparse.Namespace
) -> tuple[list[tuple], Counter, Counter, Counter]:
    scale = float(stage.GetMetadata("metersPerUnit") or 1.0)
    bbox_cache = None
    if not args.no_bbox:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.guide,
            ],
            useExtentsHint=True,
        )

    rows: list[tuple] = []
    type_counts: Counter = Counter()
    reference_assets: Counter = Counter()
    payload_assets: Counter = Counter()

    for idx, (prim, depth) in enumerate(iter_prims(stage, args.max_depth), start=1):
        if args.limit is not None and idx > args.limit:
            break
        path = str(prim.GetPath())
        parent_path = str(prim.GetParent().GetPath()) if prim.GetParent() else None
        prim_type = prim.GetTypeName() or None
        kind = prim_kind(prim)
        type_counts[prim_type or "<none>"] += 1

        refs = authored_asset_paths(prim, "references")
        payloads = authored_asset_paths(prim, "payload")
        reference_assets.update(refs)
        payload_assets.update(payloads)

        imageable = UsdGeom.Imageable(prim)
        purpose = None
        visibility = None
        if imageable:
            try:
                purpose = imageable.GetPurposeAttr().Get()
            except Exception:
                purpose = None
            try:
                visibility = imageable.GetVisibilityAttr().Get()
            except Exception:
                visibility = None

        bounds = bbox_for_prim(prim, bbox_cache, scale)
        if bounds is None:
            bbox_args = (None, None, None, None, None, None, None, None, None, None)
        else:
            min_x, min_y, min_z, max_x, max_y, max_z = bounds
            centroid_x = (min_x + max_x) / 2.0
            centroid_y = (min_y + max_y) / 2.0
            centroid_z = (min_z + max_z) / 2.0
            bbox_args = (
                min_x,
                min_y,
                min_z,
                max_x,
                max_y,
                max_z,
                centroid_x,
                centroid_x,
                centroid_y,
                centroid_z,
            )

        rows.append(
            (
                SOURCE_ID,
                args.environment_id,
                path,
                args.source_artifact_id,
                relative(args.stage),
                parent_path,
                prim.GetName(),
                prim_type,
                kind,
                str(purpose) if purpose else None,
                str(visibility) if visibility else None,
                prim.IsActive(),
                prim.IsLoaded(),
                prim.IsInstance(),
                prim.IsInstanceable(),
                Kind.Registry.IsA(kind or "", Kind.Tokens.model),
                prim.HasAuthoredReferences(),
                prim.HasAuthoredPayloads(),
                refs,
                payloads,
                Json(variant_sets_for_prim(prim)),
                attr_names(prim, args.max_attrs_per_prim),
                *bbox_args,
                Json(
                    {
                        "meters_per_unit": scale,
                        "has_children": bool(list(prim.GetChildren())),
                        "is_pseudo_root": prim.IsPseudoRoot(),
                        "depth": depth,
                    }
                ),
            )
        )

    return rows, type_counts, reference_assets, payload_assets


def chunked(values: Iterable[tuple], size: int) -> Iterable[list[tuple]]:
    chunk: list[tuple] = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def ingest(
    rows: list[tuple],
    type_counts: Counter,
    reference_assets: Counter,
    payload_assets: Counter,
    args: argparse.Namespace,
) -> None:
    insert_sql = """
        INSERT INTO br_ground.environment_prims (
            source_id, environment_id, prim_path, source_artifact_id, stage_path,
            parent_path, prim_name, prim_type, kind, purpose, visibility,
            is_active, is_loaded, is_instance, is_instanceable, is_model,
            has_authored_references, has_authored_payloads,
            authored_reference_assets, authored_payload_assets,
            variant_sets, attribute_names,
            bbox_min_x_m, bbox_min_y_m, bbox_min_z_m,
            bbox_max_x_m, bbox_max_y_m, bbox_max_z_m,
            centroid, metadata
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            CASE WHEN %s IS NULL THEN NULL ELSE ST_MakePoint(%s, %s, %s)::geometry(PointZ, 0) END,
            %s
        )
        ON CONFLICT (source_id, environment_id, prim_path) DO UPDATE
        SET source_artifact_id = EXCLUDED.source_artifact_id,
            stage_path = EXCLUDED.stage_path,
            parent_path = EXCLUDED.parent_path,
            prim_name = EXCLUDED.prim_name,
            prim_type = EXCLUDED.prim_type,
            kind = EXCLUDED.kind,
            purpose = EXCLUDED.purpose,
            visibility = EXCLUDED.visibility,
            is_active = EXCLUDED.is_active,
            is_loaded = EXCLUDED.is_loaded,
            is_instance = EXCLUDED.is_instance,
            is_instanceable = EXCLUDED.is_instanceable,
            is_model = EXCLUDED.is_model,
            has_authored_references = EXCLUDED.has_authored_references,
            has_authored_payloads = EXCLUDED.has_authored_payloads,
            authored_reference_assets = EXCLUDED.authored_reference_assets,
            authored_payload_assets = EXCLUDED.authored_payload_assets,
            variant_sets = EXCLUDED.variant_sets,
            attribute_names = EXCLUDED.attribute_names,
            bbox_min_x_m = EXCLUDED.bbox_min_x_m,
            bbox_min_y_m = EXCLUDED.bbox_min_y_m,
            bbox_min_z_m = EXCLUDED.bbox_min_z_m,
            bbox_max_x_m = EXCLUDED.bbox_max_x_m,
            bbox_max_y_m = EXCLUDED.bbox_max_y_m,
            bbox_max_z_m = EXCLUDED.bbox_max_z_m,
            centroid = EXCLUDED.centroid,
            metadata = EXCLUDED.metadata,
            cataloged_at = now()
    """

    catalog_run_id = f"{args.environment_id}_{args.load_policy}_{utc_now_compact()}"
    with connect() as conn:
        with conn.cursor() as cur:
            ensure_source_and_feature_set(cur, args, len(rows))
            for batch in chunked(rows, args.batch_size):
                psycopg2.extras.execute_batch(cur, insert_sql, batch, page_size=args.batch_size)
            cur.execute(
                """
                INSERT INTO br_repr.environment_catalog_runs (
                    catalog_run_id, source_id, environment_id, stage_path,
                    load_policy, prim_count, type_counts, reference_asset_count,
                    payload_asset_count, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (catalog_run_id) DO UPDATE
                SET prim_count = EXCLUDED.prim_count,
                    type_counts = EXCLUDED.type_counts,
                    reference_asset_count = EXCLUDED.reference_asset_count,
                    payload_asset_count = EXCLUDED.payload_asset_count,
                    metadata = EXCLUDED.metadata
                """,
                (
                    catalog_run_id,
                    SOURCE_ID,
                    args.environment_id,
                    relative(args.stage),
                    args.load_policy,
                    len(rows),
                    Json(dict(type_counts)),
                    len(reference_assets),
                    len(payload_assets),
                    Json(
                        {
                            "source_artifact_id": args.source_artifact_id,
                            "limit": args.limit,
                            "no_bbox": args.no_bbox,
                            "max_attrs_per_prim": args.max_attrs_per_prim,
                        }
                    ),
                ),
            )
        conn.commit()
    print(f"catalog_run_id={catalog_run_id}")


def print_summary(
    rows: list[tuple],
    type_counts: Counter,
    reference_assets: Counter,
    payload_assets: Counter,
    args: argparse.Namespace,
) -> None:
    bbox_count = sum(1 for row in rows if row[22] is not None)
    print("summary")
    print(f"  stage={relative(args.stage)}")
    print(f"  environment_id={args.environment_id}")
    print(f"  load_policy={args.load_policy}")
    print(f"  max_depth={args.max_depth}")
    print(f"  prim_rows={len(rows)}")
    print(f"  bbox_rows={bbox_count}")
    print(f"  unique_reference_assets={len(reference_assets)}")
    print(f"  unique_payload_assets={len(payload_assets)}")
    print("  top_types=" + json.dumps(type_counts.most_common(20), ensure_ascii=False))
    if reference_assets:
        print(
            "  sample_reference_assets="
            + json.dumps(list(reference_assets.keys())[:10], ensure_ascii=False)
        )
    if payload_assets:
        print(
            "  sample_payload_assets="
            + json.dumps(list(payload_assets.keys())[:10], ensure_ascii=False)
        )
    print(f"  dry_run={args.dry_run}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--environment-id", default=ENVIRONMENT_ID)
    parser.add_argument("--source-artifact-id", default=STAGE_ARTIFACT_ID)
    parser.add_argument("--load-policy", choices=["none", "all"], default="none")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-attrs-per-prim", type=int, default=80)
    parser.add_argument("--no-bbox", action="store_true")
    args = parser.parse_args()

    args.stage = args.stage.resolve()
    if not args.stage.exists():
        raise FileNotFoundError(args.stage)

    if args.apply_schema and not args.dry_run:
        apply_sql_file(ROOT / "src/pubg_lab/data/stdb/schemas/schema_br_layers.sql")

    stage = open_stage(args.stage, args.load_policy)
    rows, type_counts, reference_assets, payload_assets = catalog_rows(stage, args)
    print_summary(rows, type_counts, reference_assets, payload_assets, args)

    if not args.dry_run:
        ingest(rows, type_counts, reference_assets, payload_assets, args)


if __name__ == "__main__":
    main()
