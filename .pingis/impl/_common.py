"""Common helper for reference impls under .pingis/impl/.

Each impl lives outside the cadw package. When imported as a script or
run via `python .pingis/impl/<Sec>/<file>.py`, this helper ensures the
repo root is on sys.path so `import cadw` works.

The reference impls never modify cadw; they only import it read-only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_cadw_on_path() -> None:
    here = Path(__file__).resolve()
    # .pingis/impl/_common.py -> repo root is parents[2]
    root = here.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_cadw_on_path()

__all__ = []
