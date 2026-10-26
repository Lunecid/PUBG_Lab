"""Compatibility entry point; implementation: src/pubg_lab/movement/cli/finalize_v4.py."""

import sys as _compat_sys
from pathlib import Path as _CompatPath

_source = _CompatPath(__file__).resolve().parents[2] / "src"
if str(_source) not in _compat_sys.path:
    _compat_sys.path.insert(0, str(_source))
from pubg_lab.common.compat import redirect as _redirect

_redirect(globals(), "pubg_lab.movement.cli.finalize_v4", package=False)
