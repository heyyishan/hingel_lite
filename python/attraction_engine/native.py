"""Loader for the compiled C++26 / C23 core.

The extension is built by CMake into ``attraction_engine/_native``. If it is
missing (e.g. a wheel-only deploy, or a machine without a C++ toolchain) the
import fails loudly rather than silently re-implementing the numerics in Python:
the API layer exposes ``NATIVE_AVAILABLE`` so the UI can show the degradation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load() -> ModuleType | None:
    here = Path(__file__).resolve().parent
    native_dir = here / "_native"
    candidates = list(native_dir.glob("ae_python*.so")) + list(native_dir.glob("ae_core*.so"))
    candidates += list(native_dir.glob("ae_python*.pyd")) + list(native_dir.glob("ae_core*.pyd"))
    for path in candidates:
        spec = importlib.util.spec_from_file_location("ae_core", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("ae_core", module)
        spec.loader.exec_module(module)
        return module
    return None


core: ModuleType | None = _load()
NATIVE_AVAILABLE: bool = core is not None
_IMPORT_HINT = (
    "The native core is not built. Run scripts/build_native.sh "
    "(cmake --build cpp/build --target ae_python)."
)


def require() -> ModuleType:
    if core is None:
        raise RuntimeError(_IMPORT_HINT)
    return core


def build_info() -> dict:
    if core is None:
        return {"native_available": False, "hint": _IMPORT_HINT}
    info = dict(core.build_info())
    info.update(core.cpu_info())
    info["native_available"] = True
    return info
