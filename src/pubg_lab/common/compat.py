"""Compatibility forwarding with one canonical implementation per Python module."""

import importlib
import runpy
import sys
from typing import Any


def redirect(namespace: dict[str, Any], target: str, *, package: bool = False) -> None:
    """Forward an old import or script path without copying implementation code."""
    name = namespace["__name__"]
    if name == "__main__":
        runpy.run_module(target, run_name="__main__")
        return
    module = importlib.import_module(target)
    # importlib callers can retain the original module_from_spec object.
    # Forward its attributes too, while normal imports share the canonical object.
    namespace["__getattr__"] = lambda attribute: getattr(module, attribute)
    namespace["__dir__"] = lambda: dir(module)
    namespace["__all__"] = getattr(
        module, "__all__", [key for key in vars(module) if not key.startswith("_")]
    )
    if package:
        return
    else:
        sys.modules[name] = module
