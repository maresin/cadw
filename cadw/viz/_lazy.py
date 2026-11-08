"""Lazy-import helpers for viz adapters (v0.9.133 DEC-054).

Prior to v0.9.133 the `_plt()` helper was duplicated (identical body,
different error-message string) in seaborn_adapter + matplotlib_adapter.
"""
from __future__ import annotations


def lazy_plt(module_name: str):
    """Lazy import of `matplotlib.pyplot` with actionable error.

    Parameters
    ----------
    module_name : str
        Adapter name for the error message ("cadw.viz.<adapter>").
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            f"matplotlib is required for {module_name}; "
            "install with: uv sync --group viz"
        ) from e
    return plt
