"""Validate the DSACG genre-primitive feature manifest.

This is a pre-experiment contract check. It validates that each declared
feature/family has an owner, clock, model stream, and leakage status, and can
optionally check coverage against the legacy JSON feature-family manifest.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("docs/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv")
DEFAULT_LEGACY = Path("model_stgcn/feature_manifests/pubg_representation_v2.json")

REQUIRED_COLUMNS = [
    "pubg_feature_name",
    "genre_primitive",
    "entity_owner",
    "tensor_owner",
    "clock",
    "causal_window",
    "model_stream",
    "domain_mechanism",
    "leakage_risk",
    "allowed_models",
    "notes",
]

ENTITY_OWNERS = {"match", "team", "pair", "global", "label"}
TENSOR_OWNERS = {
    "X_s",
    "A_s",
    "E_s",
    "X_phase",
    "A_phase",
    "E_phase",
    "U",
    "Y",
    "M",
}
CLOCKS = {
    "match_static",
    "snapshot_10s",
    "phase_cut",
    "phase_window",
    "phase_to_date",
    "label",
}
MODEL_STREAMS = {
    "snapshot_node",
    "snapshot_edge",
    "phase_node",
    "phase_edge",
    "global_context",
    "label",
    "split_only",
}
LEAKAGE_RISKS = {"none", "suspicious", "forbidden"}
ALLOWED_MODEL_TOKENS = {
    "all_models",
    "baseline",
    "phase_baseline",
    "snapshot_first",
    "graph_model",
    "phase_graph_model",
    "sagong_baseline",
    "supervision_only",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
        return list(reader.fieldnames or []), rows


def _tokens(value: str) -> set[str]:
    return {p.strip() for p in value.split("|") if p.strip()}


def _legacy_columns(path: Path) -> set[str]:
    data: dict[str, Any] = json.loads(path.read_text())
    cols: set[str] = set()
    cols.update(data.get("label_columns", []))
    cols.update(data.get("key_columns", []))
    cols.update(data.get("categorical_columns", []))
    cols.update(data.get("forbidden_input_columns", []))
    for family in data.get("families", {}).values():
        cols.update(family.get("columns", []))
    return cols


def _is_covered(feature_name: str, manifest_names: list[str]) -> bool:
    for pattern in manifest_names:
        if pattern == feature_name:
            return True
        if "*" in pattern and fnmatch.fnmatchcase(feature_name, pattern):
            return True
    return False


def validate_manifest(
    path: Path,
    *,
    legacy_manifest: Path | None = None,
) -> dict[str, Any]:
    fieldnames, rows = _read_csv(path)
    errors: list[str] = []
    warnings: list[str] = []

    if fieldnames != REQUIRED_COLUMNS:
        errors.append(f"header mismatch: got {fieldnames}, expected {REQUIRED_COLUMNS}")

    seen_keys: set[tuple[str, str, str, str]] = set()
    for lineno, row in enumerate(rows, start=2):
        for col in REQUIRED_COLUMNS:
            if col not in row:
                errors.append(f"line {lineno}: missing column {col}")
                continue
            if col != "notes" and not row[col]:
                errors.append(f"line {lineno}: empty required value {col}")

        key = (
            row.get("pubg_feature_name", ""),
            row.get("tensor_owner", ""),
            row.get("clock", ""),
            row.get("model_stream", ""),
        )
        if key in seen_keys:
            errors.append(f"line {lineno}: duplicate manifest key {key}")
        seen_keys.add(key)

        if row.get("entity_owner") not in ENTITY_OWNERS:
            errors.append(f"line {lineno}: invalid entity_owner {row.get('entity_owner')}")
        if row.get("tensor_owner") not in TENSOR_OWNERS:
            errors.append(f"line {lineno}: invalid tensor_owner {row.get('tensor_owner')}")
        if row.get("clock") not in CLOCKS:
            errors.append(f"line {lineno}: invalid clock {row.get('clock')}")
        if row.get("model_stream") not in MODEL_STREAMS:
            errors.append(f"line {lineno}: invalid model_stream {row.get('model_stream')}")
        if row.get("leakage_risk") not in LEAKAGE_RISKS:
            errors.append(f"line {lineno}: invalid leakage_risk {row.get('leakage_risk')}")

        model_tokens = _tokens(row.get("allowed_models", ""))
        unknown_tokens = model_tokens - ALLOWED_MODEL_TOKENS
        if unknown_tokens:
            errors.append(f"line {lineno}: invalid allowed_models tokens {sorted(unknown_tokens)}")

        tensor_owner = row.get("tensor_owner")
        model_stream = row.get("model_stream")
        leakage_risk = row.get("leakage_risk")
        if tensor_owner in {"Y", "M"}:
            if leakage_risk != "forbidden" and model_stream == "label":
                errors.append(f"line {lineno}: label/mask stream must be forbidden")
            if "supervision_only" not in model_tokens and model_stream == "label":
                errors.append(f"line {lineno}: label/mask stream must be supervision_only")

        if "snapshot_first" in model_tokens and row.get("clock") == "phase_window":
            errors.append(
                f"line {lineno}: snapshot_first cannot silently consume phase_window feature "
                f"{row.get('pubg_feature_name')}"
            )
        if model_stream.startswith("snapshot") and row.get("clock") != "snapshot_10s":
            errors.append(f"line {lineno}: snapshot stream must use snapshot_10s clock")
        if model_stream.startswith("phase") and row.get("clock") not in {"phase_cut", "phase_window", "phase_to_date"}:
            errors.append(f"line {lineno}: phase stream has incompatible clock {row.get('clock')}")
        if model_stream == "global_context" and tensor_owner != "U":
            errors.append(f"line {lineno}: global_context must use tensor_owner U")

        if "*" in row.get("pubg_feature_name", ""):
            warnings.append(f"line {lineno}: wildcard row {row['pubg_feature_name']} requires later physical-column expansion")

    missing_legacy: list[str] = []
    if legacy_manifest is not None:
        legacy_cols = _legacy_columns(legacy_manifest)
        names = [row["pubg_feature_name"] for row in rows]
        missing_legacy = sorted(c for c in legacy_cols if not _is_covered(c, names))
        for c in missing_legacy:
            errors.append(f"legacy column not covered by genre manifest: {c}")

    return {
        "manifest": str(path),
        "rows": len(rows),
        "errors": errors,
        "warnings": warnings,
        "legacy_manifest": str(legacy_manifest) if legacy_manifest else None,
        "missing_legacy_columns": missing_legacy,
        "ok": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--legacy-manifest", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--no-legacy-coverage", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    legacy = None if args.no_legacy_coverage else args.legacy_manifest
    report = validate_manifest(args.manifest, legacy_manifest=legacy)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"manifest={report['manifest']}")
        print(f"rows={report['rows']}")
        print(f"warnings={len(report['warnings'])}")
        print(f"errors={len(report['errors'])}")
        for warning in report["warnings"][:20]:
            print(f"WARNING: {warning}")
        if len(report["warnings"]) > 20:
            print(f"WARNING: ... {len(report['warnings']) - 20} more")
        for error in report["errors"]:
            print(f"ERROR: {error}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
