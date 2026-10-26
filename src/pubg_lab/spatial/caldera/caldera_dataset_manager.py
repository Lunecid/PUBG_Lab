"""Manage Activision Caldera as an external BR L0 data source.

Default usage is metadata-first and non-destructive:

    python3 scripts/caldera_dataset_manager.py manifest
    python3 scripts/caldera_dataset_manager.py register-db --apply-schema

The large data checkout is explicit:

    python3 scripts/caldera_dataset_manager.py fetch --profile full --yes

The script deliberately keeps raw Caldera files under data/br_l0/caldera
(ignored by git) and only registers provenance / thin metadata in PostgreSQL.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT

import argparse
import json
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = REPO_ROOT
DEFAULT_DATA_DIR = ROOT / "data" / "br_l0" / "caldera"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "manifest" / "caldera_github_manifest.json"

SOURCE_ID = "activision_caldera"
REPO_OWNER = "Activision"
REPO_NAME = "caldera"
REPO_HTML_URL = "https://github.com/Activision/caldera"
REPO_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
CONTENTS_API_URL = f"{REPO_API_URL}/contents"
RELEASES_API_URL = f"{REPO_API_URL}/releases"

MINIMAL_PLAYER_PATHS = [
    "README.md",
    "LICENSE.md",
    "caldera.py",
    "caldera.usda",
    "layers/breadcrumbs.usd",
    "layers/endpoints.usd",
    "layers/cameras.usd",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return value / 1024**3


def read_json_url(url: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "pubg-br-caldera-ingest/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def read_bytes_url(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "pubg-br-caldera-ingest/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sum_tree(tree: dict[str, Any]) -> tuple[int, int]:
    total = 0
    files = 0
    for item in tree.get("tree", []):
        if item.get("type") == "blob":
            total += int(item.get("size") or 0)
            files += 1
    return total, files


def fetch_manifest() -> dict[str, Any]:
    repo = read_json_url(REPO_API_URL)
    contents = read_json_url(CONTENTS_API_URL)
    releases = read_json_url(RELEASES_API_URL)

    top_level: list[dict[str, Any]] = []
    total_bytes = 0
    total_files = 0

    for item in contents:
        name = item["name"]
        item_type = item["type"]
        html_url = item.get("html_url")

        if item_type == "file":
            size = int(item.get("size") or 0)
            files = 1
            truncated = False
        elif item_type == "dir":
            tree_url = f"{item['git_url']}?recursive=1"
            tree = read_json_url(tree_url)
            size, files = sum_tree(tree)
            truncated = bool(tree.get("truncated"))
        else:
            size = 0
            files = 0
            truncated = False

        total_bytes += size
        total_files += files
        top_level.append(
            {
                "name": name,
                "path": item["path"],
                "type": item_type,
                "html_url": html_url,
                "size_bytes": size,
                "size_gib": bytes_to_gib(size),
                "file_count": files,
                "tree_truncated": truncated,
            }
        )

    release_summary = [
        {
            "tag_name": rel.get("tag_name"),
            "name": rel.get("name"),
            "html_url": rel.get("html_url"),
            "published_at": rel.get("published_at"),
            "asset_count": len(rel.get("assets") or []),
            "asset_bytes": sum(int(asset.get("size") or 0) for asset in rel.get("assets") or []),
        }
        for rel in releases
    ]

    return {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "fetched_at": utc_now(),
        "repo": {
            "owner": REPO_OWNER,
            "name": REPO_NAME,
            "html_url": repo.get("html_url"),
            "clone_url": repo.get("clone_url"),
            "default_branch": repo.get("default_branch"),
            "size_kb_github_api": repo.get("size"),
            "size_bytes_from_repo_kb": int(repo.get("size") or 0) * 1024,
            "pushed_at": repo.get("pushed_at"),
            "updated_at": repo.get("updated_at"),
            "license": repo.get("license"),
        },
        "official_blog": {
            "url": "https://blog.activision.com/activision/2024/activision-releases-call-of-duty-warzone-caldera-data-set",
            "geometry_size_text": "approximately 4 GB",
            "player_data_text": "one million endpoint points and time-sampled player breadcrumbs in the OpenUSD scene",
        },
        "tree": {
            "total_size_bytes": total_bytes,
            "total_size_gib": bytes_to_gib(total_bytes),
            "total_file_count": total_files,
            "top_level": top_level,
        },
        "releases": release_summary,
        "recommended_local_budget": {
            "manifest_only_gib": 0.1,
            "metadata_sparse_checkout_gib": 1.0,
            "full_checkout_min_free_gib": 12.0,
            "note": "Keep Caldera raw files external to PostgreSQL; load only manifest and extracted canonical samples.",
        },
    }


def write_manifest(manifest: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def print_manifest_summary(manifest: dict[str, Any]) -> None:
    repo = manifest["repo"]
    tree = manifest["tree"]
    print(f"source_id: {manifest['source_id']}")
    print(f"repo: {repo['html_url']}")
    print(
        "github_api_repo_size: "
        f"{repo['size_kb_github_api']} KB "
        f"({bytes_to_gib(repo['size_bytes_from_repo_kb']):.3f} GiB)"
    )
    print(
        f"expanded_tree_size: {tree['total_size_bytes']} bytes ({tree['total_size_gib']:.3f} GiB)"
    )
    print("top_level:")
    for item in sorted(tree["top_level"], key=lambda row: row["size_bytes"], reverse=True):
        print(
            f"  {item['path']:<16} {item['size_bytes']:>12} bytes "
            f"{item['size_gib'] or 0.0:>7.3f} GiB files={item['file_count']}"
        )
    release_assets = sum(rel["asset_count"] for rel in manifest["releases"])
    print(f"release_assets: {release_assets}")


def disk_free_gib(path: Path) -> float:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024**3


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def sparse_patterns(profile: str) -> list[str]:
    if profile == "metadata":
        return [
            "/README.md",
            "/LICENSE.md",
            "/caldera.py",
            "/caldera.usda",
            "/layers/**",
            "/images/**",
        ]
    if profile == "source":
        return [
            "/README.md",
            "/LICENSE.md",
            "/caldera.py",
            "/caldera.usda",
            "/layers/**",
            "/images/**",
            "/map_source/**",
        ]
    if profile == "full":
        return ["/*"]
    raise ValueError(f"unknown profile: {profile}")


def fetch_repo(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    repo_dir = data_dir / "repo"
    free_gib = disk_free_gib(data_dir)

    if free_gib < args.min_free_gib:
        raise SystemExit(
            f"Refusing checkout: free disk is {free_gib:.1f} GiB, "
            f"below --min-free-gib={args.min_free_gib:.1f}."
        )
    if not args.yes:
        raise SystemExit(
            "This command can download several GiB. Re-run with --yes after "
            "checking the target path and free disk."
        )
    if repo_dir.exists() and not (repo_dir / ".git").exists():
        raise SystemExit(f"Target exists but is not a git repo: {repo_dir}")
    if repo_dir.exists() and not args.reuse:
        raise SystemExit(f"Target already exists: {repo_dir}. Use --reuse to keep using it.")

    data_dir.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--depth",
                "1",
                REPO_HTML_URL + ".git",
                str(repo_dir),
            ]
        )

    run(["git", "-C", str(repo_dir), "sparse-checkout", "init", "--no-cone"])
    sparse_file = repo_dir / ".git" / "info" / "sparse-checkout"
    sparse_file.write_text("\n".join(sparse_patterns(args.profile)) + "\n")
    run(["git", "-C", str(repo_dir), "checkout", args.ref])
    run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"])
    print(f"checked_out: {repo_dir}")


def content_entry(path: str) -> dict[str, Any]:
    return read_json_url(f"{CONTENTS_API_URL}/{path}")


def fetch_minimal_players(args: argparse.Namespace) -> None:
    """Download only README/LICENSE/scripts + player USD layers.

    This path is intentionally independent of git clone because it is the safest
    first move on a nearly full disk.
    """
    data_dir = args.data_dir.resolve()
    raw_dir = data_dir / "raw_minimal_players"
    manifest_dir = data_dir / "manifest"
    free_gib = disk_free_gib(data_dir)
    if free_gib < args.min_free_gib:
        raise SystemExit(
            f"Refusing download: free disk is {free_gib:.1f} GiB, "
            f"below --min-free-gib={args.min_free_gib:.1f}."
        )

    entries = [content_entry(path) for path in MINIMAL_PLAYER_PATHS]
    planned_bytes = sum(int(entry.get("size") or 0) for entry in entries)
    print(f"planned_files: {len(entries)}")
    print(f"planned_bytes: {planned_bytes} ({bytes_to_gib(planned_bytes):.3f} GiB)")
    for entry in entries:
        print(f"  {entry['path']:<24} {int(entry.get('size') or 0):>12} bytes")

    if args.dry_run:
        print("dry_run: no files downloaded")
        return
    if not args.yes:
        raise SystemExit("Re-run with --yes to download the minimal player files.")

    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    for entry in entries:
        rel_path = Path(entry["path"])
        out_path = raw_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            out_path.exists()
            and out_path.stat().st_size == int(entry.get("size") or 0)
            and not args.force
        ):
            status = "reused"
        else:
            data = read_bytes_url(entry["download_url"])
            out_path.write_bytes(data)
            status = "downloaded"
        digest = sha256_file(out_path)
        row = {
            "path": entry["path"],
            "status": status,
            "html_url": entry["html_url"],
            "download_url": entry["download_url"],
            "github_sha": entry["sha"],
            "size_bytes": out_path.stat().st_size,
            "sha256": digest,
            "local_path": str(out_path),
        }
        downloaded.append(row)
        print(f"{status}: {entry['path']} sha256={digest[:16]}...")

    mini_manifest = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "profile": "minimal_players",
        "fetched_at": utc_now(),
        "repo": REPO_HTML_URL,
        "files": downloaded,
        "total_size_bytes": sum(item["size_bytes"] for item in downloaded),
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = manifest_dir / "caldera_minimal_players_manifest.json"
    out_manifest.write_text(json.dumps(mini_manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote: {out_manifest}")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}. Run the manifest command first.")
    return json.loads(path.read_text())


def register_db(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)

    sys.path.insert(0, str(ROOT))
    from psycopg2.extras import Json
    from pubg_lab.data.stdb.conn import apply_sql_file, connect

    if args.apply_schema:
        apply_sql_file(ROOT / "stdb" / "schema_br_layers.sql")

    repo = manifest["repo"]
    tree = manifest["tree"]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO br_raw.sources (
                    source_id, source_game, provider, canonical_url,
                    license_name, license_url, observability, notes
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (source_id) DO UPDATE
                SET canonical_url = EXCLUDED.canonical_url,
                    observability = EXCLUDED.observability,
                    notes = br_raw.sources.notes || EXCLUDED.notes
                """,
                (
                    SOURCE_ID,
                    "Call of Duty: Warzone Caldera",
                    "Activision",
                    REPO_HTML_URL,
                    "Activision Non-Commercial Use License Agreement",
                    "https://github.com/Activision/caldera/blob/main/LICENSE.md",
                    "geometry_plus_anonymized_player_point_clouds",
                    Json(
                        {
                            "manifest_path": str(args.manifest),
                            "official_blog": manifest["official_blog"]["url"],
                            "fetched_at": manifest["fetched_at"],
                        }
                    ),
                ),
            )

            artifacts = [
                {
                    "artifact_id": "caldera_github_repo_main",
                    "artifact_kind": "github_repository",
                    "uri": repo["html_url"],
                    "local_path": str((args.data_dir / "repo").resolve()),
                    "content_format": "git/openusd",
                    "size_bytes": repo["size_bytes_from_repo_kb"],
                    "git_ref": repo["default_branch"],
                    "metadata": {
                        "github_api_size_kb": repo["size_kb_github_api"],
                        "expanded_tree_size_bytes": tree["total_size_bytes"],
                        "pushed_at": repo["pushed_at"],
                        "updated_at": repo["updated_at"],
                    },
                },
                {
                    "artifact_id": "caldera_github_manifest",
                    "artifact_kind": "manifest",
                    "uri": str(args.manifest),
                    "local_path": str(args.manifest.resolve()),
                    "content_format": "json",
                    "size_bytes": args.manifest.stat().st_size,
                    "git_ref": repo["default_branch"],
                    "metadata": {"schema_version": manifest["schema_version"]},
                },
            ]
            minimal_manifest = args.data_dir / "manifest" / "caldera_minimal_players_manifest.json"
            if minimal_manifest.exists():
                mini = json.loads(minimal_manifest.read_text())
                artifacts.append(
                    {
                        "artifact_id": "caldera_minimal_players_manifest",
                        "artifact_kind": "manifest",
                        "uri": str(minimal_manifest),
                        "local_path": str(minimal_manifest.resolve()),
                        "content_format": "json",
                        "size_bytes": minimal_manifest.stat().st_size,
                        "git_ref": repo["default_branch"],
                        "metadata": {
                            "profile": "minimal_players",
                            "total_size_bytes": mini.get("total_size_bytes"),
                        },
                    }
                )
                for item in mini.get("files", []):
                    artifacts.append(
                        {
                            "artifact_id": "caldera_minimal_"
                            + item["path"].replace("/", "_").replace(".", "_"),
                            "artifact_kind": "minimal_player_file",
                            "uri": item["download_url"],
                            "local_path": item["local_path"],
                            "content_format": "openusd_or_repo_file",
                            "size_bytes": item["size_bytes"],
                            "git_ref": repo["default_branch"],
                            "metadata": {
                                "path": item["path"],
                                "github_sha": item["github_sha"],
                                "sha256": item["sha256"],
                            },
                        }
                    )
            for item in tree["top_level"]:
                artifacts.append(
                    {
                        "artifact_id": f"caldera_top_{item['path'].replace('/', '_')}",
                        "artifact_kind": f"top_level_{item['type']}",
                        "uri": item["html_url"],
                        "local_path": str((args.data_dir / "repo" / item["path"]).resolve()),
                        "content_format": "openusd_or_repo_file",
                        "size_bytes": item["size_bytes"],
                        "git_ref": repo["default_branch"],
                        "metadata": {
                            "file_count": item["file_count"],
                            "tree_truncated": item["tree_truncated"],
                        },
                    }
                )

            for artifact in artifacts:
                cur.execute(
                    """
                    INSERT INTO br_raw.artifacts (
                        artifact_id, source_id, layer_name, artifact_kind, uri,
                        local_path, content_format, size_bytes, git_ref, metadata
                    ) VALUES (
                        %s, %s, 'L0', %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (artifact_id) DO UPDATE
                    SET uri = EXCLUDED.uri,
                        local_path = EXCLUDED.local_path,
                        content_format = EXCLUDED.content_format,
                        size_bytes = EXCLUDED.size_bytes,
                        git_ref = EXCLUDED.git_ref,
                        metadata = EXCLUDED.metadata,
                        registered_at = now()
                    """,
                    (
                        artifact["artifact_id"],
                        SOURCE_ID,
                        artifact["artifact_kind"],
                        artifact["uri"],
                        artifact["local_path"],
                        artifact["content_format"],
                        artifact["size_bytes"],
                        artifact["git_ref"],
                        Json(artifact["metadata"]),
                    ),
                )
        conn.commit()
    print(f"registered Caldera manifest in br_raw.* from {args.manifest}")


def inspect_usd(args: argparse.Namespace) -> None:
    try:
        from pxr import Usd  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "USD Python bindings are not installed. Install the optional "
            "usd-core package before inspecting Caldera USD files."
        ) from exc

    stage = Usd.Stage.Open(str(args.usd_path))
    if stage is None:
        raise SystemExit(f"Could not open USD stage: {args.usd_path}")

    rows = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        lowered = path.lower()
        if "player" not in lowered and "breadcrumb" not in lowered and "endpoint" not in lowered:
            continue
        rows.append(
            {
                "path": path,
                "type_name": prim.GetTypeName(),
                "attributes": [attr.GetName() for attr in prim.GetAttributes()],
            }
        )
        if args.limit and len(rows) >= args.limit:
            break

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(row["path"])
            print(f"  type={row['type_name']}")
            print(f"  attrs={', '.join(row['attributes'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_manifest = sub.add_parser(
        "manifest", help="Fetch small GitHub metadata and write an L0 manifest."
    )
    p_manifest.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)

    p_fetch = sub.add_parser("fetch", help="Sparse/partial checkout of the Caldera repo.")
    p_fetch.add_argument("--profile", choices=["metadata", "source", "full"], default="metadata")
    p_fetch.add_argument("--ref", default="main")
    p_fetch.add_argument("--min-free-gib", type=float, default=12.0)
    p_fetch.add_argument("--yes", action="store_true")
    p_fetch.add_argument("--reuse", action="store_true")

    p_min = sub.add_parser(
        "fetch-minimal-players",
        help="Direct-download README/LICENSE/caldera.py/caldera.usda and player USD layers only.",
    )
    p_min.add_argument("--min-free-gib", type=float, default=2.0)
    p_min.add_argument("--dry-run", action="store_true")
    p_min.add_argument("--yes", action="store_true")
    p_min.add_argument("--force", action="store_true")

    p_register = sub.add_parser("register-db", help="Register manifest artifacts in br_raw.*.")
    p_register.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p_register.add_argument("--apply-schema", action="store_true")

    p_usd = sub.add_parser("inspect-usd", help="List player/breadcrumb prims in a local USD stage.")
    p_usd.add_argument("usd_path", type=Path)
    p_usd.add_argument("--limit", type=int, default=100)
    p_usd.add_argument("--json", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "manifest":
        manifest = fetch_manifest()
        write_manifest(manifest, args.out)
        print_manifest_summary(manifest)
        print(f"wrote: {args.out}")
    elif args.cmd == "fetch":
        fetch_repo(args)
    elif args.cmd == "fetch-minimal-players":
        fetch_minimal_players(args)
    elif args.cmd == "register-db":
        register_db(args)
    elif args.cmd == "inspect-usd":
        inspect_usd(args)
    else:
        parser.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
