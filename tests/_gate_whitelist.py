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
    """Return True if key matches any whitelist entry.

    Supported forms:
      "E1.IForest"   exact match
      "D3.1.*"       subtype wildcard: matches "D3.1.<model>", "D3.1.*"
      "E.*"          section wildcard: matches "E1.<m>", "E2.<m>",
                     "E10.<m>" (single-letter section covers all
                     subtypes of that section); also "E1.*".
      "cadw/ml/*.py::f"  trailing-* prefix match (legacy).

    v0.9.87-P6c-β-fix1: previous impl used `entry[:-1]` for `E.*`
    which yielded `E.` — no `E1.<m>` key starts with `E.`. Section
    wildcard (plan §whitelist: "`E.*` = all E subtype cards") was
    documented but not implemented. This fix restores the intent.
    """
    if key in wl_set:
        return True
    for entry in wl_set:
        if not entry.endswith("*"):
            continue
        if entry.endswith(".*"):
            prefix = entry[:-2]  # strip ".*"
        else:
            prefix = entry[:-1]  # strip trailing "*"
        # Rule 1: prefix is subtype-level (has dot, e.g. "D3.1") OR
        # prefix is section-level but key literally starts prefix+"."
        # (e.g. entry "D1.*", key "D1.NaiveForecaster").
        if key.startswith(prefix + "."):
            return True
        # Rule 2: section wildcard where prefix has no dot (e.g. "E").
        # Match "E<digit>..." keys like "E1.x", "E2.x", "E10.x", "E1.*".
        # Do NOT match "EE1.x" (next char must be digit).
        if "." not in prefix and key.startswith(prefix):
            rest = key[len(prefix):]
            if rest and rest[0].isdigit():
                return True
    return False
