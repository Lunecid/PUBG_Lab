from __future__ import annotations

import json

import pytest

from model_stgcn.splits import load_or_create_split


def test_require_existing_split_refuses_silent_creation(tmp_path) -> None:
    split_path = tmp_path / "missing_split.json"

    with pytest.raises(FileNotFoundError):
        load_or_create_split(
            ["m1", "m2", "m3", "m4"],
            split_json=str(split_path),
            require_existing=True,
        )

    assert not split_path.exists()


def test_written_split_manifest_contains_provenance(tmp_path) -> None:
    split_path = tmp_path / "split.json"

    load_or_create_split(
        ["m1", "m2", "m3", "m4", "m5"],
        split_json=str(split_path),
        seed=7,
    )

    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 7
    assert "provenance" in manifest
    assert "git_commit" in manifest["provenance"]
    assert "python" in manifest["provenance"]
