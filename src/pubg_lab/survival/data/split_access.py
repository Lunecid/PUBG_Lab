"""The public API boundary to the unchanged, access-restricted split module.

No source inspection, split creation, manifest writing, or dataset access occurs
here. The legacy dependency name is supplied only because that private module
must remain at its original path and cannot be patched during layout changes.
"""

from importlib import import_module
import sys
from typing import TYPE_CHECKING

from pubg_lab.survival import provenance

if TYPE_CHECKING:
    from model_stgcn.splits import load_or_create_split, split_ids, write_split_manifest

__all__ = ["load_or_create_split", "split_ids", "write_split_manifest"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    # Preserve its old dependency without recreating root forwarding files.
    sys.modules.setdefault("model_stgcn.provenance", provenance)
    private = import_module("model_stgcn.splits")
    return getattr(private, name)
