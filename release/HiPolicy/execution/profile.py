"""Select one benchmark per process before importing a policy or environment."""
import os

_selected = None


def benchmark():
    global _selected
    value = os.environ.get("HIPOLICY_BENCHMARK")
    if value not in ("robotwin_v1", "robotwin_v2"):
        raise RuntimeError("Select --benchmark robotwin_v1|robotwin_v2, or set HIPOLICY_BENCHMARK before importing policies.")
    if _selected is not None and value != _selected:
        raise RuntimeError("Use a separate process for each benchmark.")
    _selected = value
    return value
