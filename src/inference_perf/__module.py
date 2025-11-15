"""Module metadata: name and path references."""

import importlib.util
from pathlib import Path

# Module name
MODULE_NAME = "inference_perf"

# Path to the project root directory (triton/inference-perf)
# Found by traversing up from the module location until we find pyproject.toml
_spec = importlib.util.find_spec(MODULE_NAME)
if _spec is None or _spec.origin is None:
    raise ImportError(f"Could not find module {MODULE_NAME}")
_module_path = Path(_spec.origin).parent

# Path to the module directory (src/inference_perf)
MODULE_PATH = _module_path

# Traverse up to find the project root (directory containing pyproject.toml)
_current = _module_path
while _current != _current.parent:
    if (_current / "pyproject.toml").exists():
        PROJECT_PATH = _current
        break
    _current = _current.parent
else:
    # Fallback to module path if pyproject.toml not found
    PROJECT_PATH = _module_path


__all__ = ["MODULE_NAME", "PROJECT_PATH", "MODULE_PATH"]

