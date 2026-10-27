"""Shared whitelist parser for gate tests (G1-G4).

Whitelist lives in .pingis/plan_v0.9.87_refactor.md as a machine-
readable block:

    <!-- gate-whitelist
    G1: cadw/ml/gbdt.py::fit_gbdt_multiclass
    G3: cadw/ml/sub_pca.py::_prepare_windows
    G4: A.*
    -->

Refactor removes an entry; when all entries gone, refactor complete.
"""
from __future__ import annotations

import re
from pathlib import Path

_PLAN = Path(__file__).parent.parent / ".pingis" / "plan_v0.9.87_refactor.md"
_RE = re.compile(r"<!-- gate-whitelist\n(.*?)-->", re.DOTALL)

_EMPTY = {"G1": set(), "G2": set(), "G3": set(), "G4": set()}


def load_whitelist() -> dict[str, set[str]]:
    """Parse whitelist block. Missing file/block -> empty sets."""
    if not _PLAN.exists():
        return _EMPTY
    m = _RE.search(_PLAN.read_text())
    if not m:
        return _EMPTY
    result = {k: set() for k in _EMPTY}
    for line in m.group(1).strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        gate, _, loc = line.partition(":")
        gate = gate.strip()
        loc = loc.strip()
        if gate in result and loc:
            result[gate].add(loc)
    return result


def is_whitelisted(wl_set: set[str], key: str) -> bool:
    """Exact match or trailing-* prefix match."""
    if key in wl_set:
        return True
    for entry in wl_set:
        if entry.endswith("*") and key.startswith(entry[:-1]):
            return True
    return False
