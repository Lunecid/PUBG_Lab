from __future__ import annotations

import re
from pathlib import Path

from pubg_lab.survival.v3.data.spec import FORBIDDEN_INPUT_FEATURES


REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL_MANIFEST = (
    REPO_ROOT
    / "docs"
    / "feature_engineering"
    / "PUBG_FEATURE_OPERATIONAL_MANIFEST_2026-07-02.md"
)


def _manifest_forbidden_inputs() -> tuple[str, ...]:
    text = OPERATIONAL_MANIFEST.read_text(encoding="utf-8")
    section = text.split("## Forbidden Inputs", 1)[1].split("\n## ", 1)[0]
    match = re.search(r"```text\n(?P<body>.*?)\n```", section, flags=re.S)
    assert match, "Forbidden Inputs section must contain a text code block"
    return tuple(
        line.strip()
        for line in match.group("body").splitlines()
        if line.strip()
    )


def test_operational_manifest_forbidden_inputs_match_v3_spec():
    manifest_features = _manifest_forbidden_inputs()
    assert len(manifest_features) == len(set(manifest_features))
    assert set(manifest_features) == set(FORBIDDEN_INPUT_FEATURES)
