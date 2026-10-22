"""v0.9.25d-a: whitelist + diagnostic classifier invariant tests.

Diagnostic found 83 to_pylist sites in cadw/. v0.9.25d-a whitelists
29 legitimate O(1)/O(K)/data-structure sites + 3 `_ColumnProxy`
check_state sites. Remaining (v0.9.25d-b targets): 23 unguarded +
12 low_factor. This file pins the v0.9.25d-a scope only.
"""
import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "service" / "diagnose_unguarded_to_pylist.py"
)


def _load_diagnostic():
    spec = importlib.util.spec_from_file_location("diag_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_classifier_self_test_passes():
    """v0.9.25d-a: 7/7 classifier cases (guarded, unguarded,
    low_factor, check_batch-after, variable-factor, method-factor,
    other-guard). Fix classifier REGRESSION here BEFORE anything else.
    """
    mod = _load_diagnostic()
    rc = mod._run_self_tests()
    assert rc == 0, "classifier self-test failed (see stdout above)"


def test_no_unknown_sites():
    """v0.9.25d-a: 0 `unknown` (non-literal extra_factor)."""
    mod = _load_diagnostic()
    results = mod.scan()
    assert not results["unknown"], (
        f"unknown (non-literal extra_factor) sites: "
        f"{results['unknown']}. Fix: make extra_factor a literal or "
        f"whitelist with rationale."
    )


def test_no_other_guard_sites_after_whitelist():
    """v0.9.25d-a: 0 `other-guard` (check_state but not check_batch).

    Was 1 (CadwDataFrame._ColumnProxy.to_list) — whitelisted with
    reason "check_state stricter than check_batch".
    """
    mod = _load_diagnostic()
    results = mod.scan()
    assert not results["other-guard"], (
        f"other-guard sites: {results['other-guard']}. Either convert "
        f"to check_batch, or whitelist with rationale (check_state "
        f"limit=0.5*avail vs check_batch limit=avail — check_state "
        f"is stricter, legitimate)."
    )


def test_whitelist_has_minimum_30_entries():
    """v0.9.25d-a: whitelist >= 30 entries.

    Guards against accidental whitelist deletion. v0.9.25d-a scope =
    8 original + 22 new = 30. Regression below 30 = lost entries.
    """
    mod = _load_diagnostic()
    n = len(mod.WHITELIST)
    assert n >= 30, (
        f"whitelist has {n} entries, expected >= 30. If you "
        f"legitimately removed entries, update this test."
    )


def test_all_sites_have_whitelist_reason():
    """Every WHITELIST key has non-empty rationale."""
    mod = _load_diagnostic()
    for key, reason in mod.WHITELIST.items():
        assert isinstance(key, str) and ":" in key, (
            f"malformed whitelist key: {key!r}"
        )
        assert isinstance(reason, str) and reason, (
            f"whitelist key {key!r} has empty rationale"
        )


def test_categories_present_sanity():
    """At least guarded / whitelisted non-empty.

    v0.9.25d-b1 closed low_factor (Tier B).
    v0.9.25d-b2 closed unguarded (Tier C: 4 inserted + 19 whitelisted).
    Remaining non-empty categories: guarded (27 + 4 inserted = 31)
    and whitelisted (33 + 19 = 52).

    For explicit assertion unguarded==0 and low_factor==0, see
    test_no_unguarded_or_low_factor_to_pylist.
    """
    mod = _load_diagnostic()
    results = mod.scan()
    for cat in ("guarded", "whitelisted"):
        assert results.get(cat), (
            f"category {cat!r} unexpectedly empty — likely a classifier "
            f"bug or all sites fixed (check version)"
        )


def test_low_factor_zero_after_bumps():
    """v0.9.25d-b1: all low_factor sites bumped to >= 3.0.

    After v0.9.25d-b1, no site has check_batch with factor < 3.0
    guarding to_pylist. Remaining unguarded (Tier C) — v0.9.25d-b2.
    """
    mod = _load_diagnostic()
    results = mod.scan()
    lf = results.get("low_factor", [])
    sites_str = ", ".join(f"{s['file']}:{s['line']}" for s in lf[:5])
    assert not lf, (
        f"low_factor sites remain: {len(lf)}. If you legitimately "
        f"want factor < 3.0 at one site, add it to WHITELIST with "
        f"rationale. Sites: {sites_str}"
    )


def test_no_unguarded_or_low_factor_to_pylist():
    """v0.9.25d-b2: Tier C closed — unguarded=0, low_factor=0.

    Combined with whitelist (v0.9.25d-a + b2), all to_pylist sites in
    cadw/ are guarded (check_batch extra_factor>=3.0) or whitelisted.
    No unguarded or low-factor sites remain.
    """
    mod = _load_diagnostic()
    results = mod.scan()
    ung = results.get("unguarded", [])
    lf = results.get("low_factor", [])
    if ung or lf:
        details = []
        for s in ung:
            details.append(
                f"UNGUARDED {s['file']}:{s['line']} {s['qualname']}"
            )
        for s in lf:
            details.append(
                f"LOW_FACTOR {s['file']}:{s['line']} {s['qualname']}"
            )
        raise AssertionError(
            f"unguarded={len(ung)}, low_factor={len(lf)}, "
            f"expected both 0. Sites:\n" + "\n".join(details)
        )
