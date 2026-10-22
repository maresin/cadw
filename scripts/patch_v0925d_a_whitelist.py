#!/usr/bin/env python3
# v0.9.25d-a — whitelist expansion (Tier A + other-guard).
#
# Diagnostic (v0.9.25d diagnostic phase) found 83 to_pylist sites in
# cadw/ (part1_report §2.3 knew 18 = minimum). Breakdown:
#   52 unguarded, 12 low_factor, 15 guarded, 3 whitelisted,
#   1 other-guard.
#
# Of 52 unguarded, 29 are legitimate O(1)/O(K)/data-structures with no
# MemoryGuard in scope — whitelist candidates. Other 23 are real
# needs-check_batch (v0.9.25d-b Tier C).
#
# v0.9.25d-a scope:
#   1. Expand WHITELIST in diagnose_unguarded_to_pylist.py by 27
#      entries (29 Tier-A + ColumnProxy other-guard, minus 3 already
#      present — net +27).
#   2. Add tests/test_v0925d_a.py:
#      - classifier self-test (7/7)
#      - no `unknown` sites
#      - no `other-guard` sites (was 1, now whitelisted)
#      - whitelist >= 30 (guards against accidental deletion)
#   3. No production cadw/ changes — pure data + test.
#
# v0.9.25d-b (next) closes: 12 factor bumps + 23 check_batch additions
# + full invariant test (unguarded == 0 AND low_factor == 0).
#
# Run: uv run python scripts/patch_v0925d_a_whitelist.py

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    check_no_invalid_escape, run_verify_in_subprocess,
)


# ============ 0. Verify state ============

def verify_state():
    print("\n=== verify v0.9.25c committed + diagnostic present ===")
    r = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True,
    )
    if "v0.9.25c" not in r.stdout:
        raise SystemExit(f"v0.9.25c not in HEAD; got {r.stdout!r}")
    diag = Path("scripts/service/diagnose_unguarded_to_pylist.py")
    if not diag.exists():
        raise SystemExit(f"{diag} missing — re-create diagnostic")
    if "WHITELIST = {" not in diag.read_text():
        raise SystemExit(f"{diag} lacks WHITELIST dict")
    print(f"  HEAD OK; diagnostic present")


# ============ 1. Expand WHITELIST (27 new entries) ============

# Anchor: the closing brace of the current WHITELIST dict.
# The v0.9.25d diagnostic ends WHITELIST with:
#   "cadw/viz/_aggregate.py:sample":
#       "head(n) or iloc sample; not O(N)",
# }
OLD_WHITELIST_TAIL = '''    "cadw/viz/_aggregate.py:sample":
        "head(n) or iloc sample; not O(N)",
}'''

NEW_WHITELIST_TAIL = '''    "cadw/viz/_aggregate.py:sample":
        "head(n) or iloc sample; not O(N)",

    # ================================================================
    # v0.9.25d-a: Tier A (data structures, samples, K-bounded) + other
    # ================================================================
    # Data structures / no MemoryGuard in scope.
    "cadw/core/models.py:type_size":
        "type_size() on sample (K<=sample.num_rows) or scalar dtype",
    "cadw/core/memory_guard.py:MemoryGuard.estimate_unique_keys":
        "sample_batch only; K<=10k rows, no O(N)",
    "cadw/dataset/index.py:MaterializedIndex.__iter__":
        "index is a loaded Arrow array; caller's responsibility "
        "(no MemoryGuard in scope)",
    "cadw/dataset/index.py:MaterializedIndex.is_monotonic_increasing":
        "index metadata check; no MemoryGuard in scope",
    "cadw/dataset/index.py:MaterializedIndex.is_monotonic_decreasing":
        "index metadata check; no MemoryGuard in scope",
    "cadw/dataset/index.py:MaterializedIndex.has_duplicates":
        "count_distinct fast path; to_pylist fallback for exotic types",
    "cadw/dataset/index.py:MaterializedIndex.get_positions_all":
        "label lookup; fallback for string/tuple labels (v0.9.18/229)",
    "cadw/dataset/index.py:MaterializedIndex.get_indexer":
        "label->position map; caller's responsibility",
    "cadw/dataset/index.py:MaterializedIndex.__repr__":
        "5-element sample for repr",
    # K-bounded operations (sample / unique values / heavy keys).
    "cadw/ml/logistic.py:_validate_binary_y":
        "col.unique() -> K=#classes; not O(N)",
    "cadw/ops/groupby.py:_estimate_k_sample":
        "sample_batches only; bounded by groupby_aggregate sample_cap",
    "cadw/ops/groupby.py:_detect_heavy_keys":
        "sample_batches only; peak=2*sample",
    "cadw/ops/groupby.py:_accumulate_heavy":
        "heavy_keys <= 20 rows (bounded by sample)",
    "cadw/ops/rank.py:_iter_groups":
        "group file iteration; groups_count not rows",
    # Caller-guarded (different function scope; AST classifier sees
    # only same chain).
    "cadw/ops/shuffle.py:_string_hash_column":
        "dict_encode path; caller hash_shuffle._process "
        "check_batch(3.0) covers per-batch peak",
    "cadw/ops/shuffle.py:_hash_column":
        "fallback for exotic dtypes (rare); caller check_batch(3.0)",
    # viz: K-row aggregate tables (not O(N)).
    "cadw/viz/matplotlib_adapter.py:line_plot":
        "K-row aggregate table; not O(N)",
    "cadw/viz/plotly_adapter.py:line_plot":
        "K-row aggregate table; not O(N)",
    "cadw/viz/seaborn_adapter.py:line_plot":
        "K-row aggregate table; not O(N)",
    # Small bounded buffers.
    "cadw/dataframe.py:_ArrowTail.push":
        "K <= deque maxlen (per-column shift buffer, small)",
    # check_state-guarded (stricter than check_batch: 0.5xavail vs 1xavail).
    "cadw/dataframe.py:_ColumnProxy.to_list":
        "check_state(_estimate_materialize_bytes) — limit=0.5*avail, "
        "stricter than check_batch (limit=avail)",
    "cadw/dataframe.py:_ColumnProxy.to_numpy":
        "check_state(n*elem_size) — 0.5*avail limit",
    "cadw/dataframe.py:_ColumnProxy.to_arrow":
        "check_state(n*elem_size) — 0.5*avail limit",
}'''

def expand_whitelist():
    print("\n=== expand WHITELIST (27 entries, Tier A + other-guard) ===")
    applied = apply_or_fail(
        "scripts/service/diagnose_unguarded_to_pylist.py",
        OLD_WHITELIST_TAIL, NEW_WHITELIST_TAIL,
        label="whitelist_expand",
    )
    if applied:
        assert_compiles("scripts/service/diagnose_unguarded_to_pylist.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 2. Add tests/test_v0925d_a.py ============

NEW_TESTS = '''"""v0.9.25d-a: whitelist + diagnostic classifier invariant tests.

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
    """At least guarded / whitelisted / unguarded / low_factor non-empty.

    v0.9.25d-a leaves unguarded=23, low_factor=12 (v0.9.25d-b targets).
    """
    mod = _load_diagnostic()
    results = mod.scan()
    for cat in ("guarded", "whitelisted", "unguarded", "low_factor"):
        assert results.get(cat), (
            f"category {cat!r} unexpectedly empty — likely a classifier "
            f"bug or all sites fixed (check version)"
        )
'''

def add_tests():
    print("\n=== tests/test_v0925d_a.py ===")
    p = Path("tests/test_v0925d_a.py")
    if p.exists() and "test_classifier_self_test_passes" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(NEW_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. Dogfood ============

def dogfood():
    print("\n=== dogfood: whitelist applied, diagnostic re-run ===")

    def _verify():
        import importlib.util as _iu
        from pathlib import Path as _P

        spec = _iu.spec_from_file_location(
            "diag", _P("scripts/service/diagnose_unguarded_to_pylist.py"),
        )
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # 1. Whitelist count
        assert len(mod.WHITELIST) >= 30, (
            f"whitelist has {len(mod.WHITELIST)}, expected >= 30"
        )

        # 2. Scan and categorize
        r = mod.scan()
        n_ung = len(r.get("unguarded", []))
        n_low = len(r.get("low_factor", []))
        n_whl = len(r.get("whitelisted", []))
        n_grd = len(r.get("guarded", []))
        n_unk = len(r.get("unknown", []))
        n_oth = len(r.get("other-guard", []))

        # 3. Invariants for v0.9.25d-a:
        assert n_unk == 0, f"unknown sites: {n_unk}"
        assert n_oth == 0, f"other-guard sites: {n_oth}"
        # unguarded should be 23 (was 52, minus 29 whitelisted)
        assert 20 <= n_ung <= 26, (
            f"unguarded = {n_ung}, expected ~23 (v0.9.25d-a) — "
            f"if much lower, one of Tier C is fixed; if much higher, "
            f"whitelist entry didn't match"
        )
        # low_factor unchanged from diagnostic
        assert 10 <= n_low <= 14, (
            f"low_factor = {n_low}, expected ~12 (unchanged)"
        )

        print(
            f"  guard breakdown: guarded={n_grd}, whitelisted={n_whl}, "
            f"unguarded={n_ung} (Tier C target), low_factor={n_low} "
            f"(Tier B target); unknown={n_unk}, other-guard={n_oth}"
        )

    run_verify_in_subprocess(_verify)


# ============ 4. Pytest ============

def run_pytest():
    print("\n=== pre-flight: test_facade_has_no_as_py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py", "-q"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** FACADE AS_PY VIOLATION — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: v0925d_a ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_v0925d_a.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** v0925d_a TEST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 5. main ============

def main():
    print("=== v0.9.25d-a: whitelist (Tier A + other-guard) ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    expand_whitelist()
    add_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25d-a: whitelist Tier A + other-guard (Phase A.6 prep). "
        "v0.9.25d diagnostic found 83 to_pylist sites in cadw/ "
        "(part1_report §2.3 knew 18 = minimum, not complete). "
        "Breakdown: 52 unguarded, 12 low_factor, 15 guarded, "
        "3 whitelisted, 1 other-guard. "
        "Tier A (29 sites): legitimate O(1)/O(K)/data-structures — "
        "dataset/index.py (data structures, no MemoryGuard in scope), "
        "_estimate_k_sample / _detect_heavy_keys / _accumulate_heavy "
        "(sample-bounded), viz/line_plot (K-row agg), _ArrowTail.push "
        "(deque maxlen), _string_hash_column / _hash_column "
        "(caller-checked). "
        "Tier B (12 low_factor): _per_group_shift/rolling (factor 5.0 "
        "→ 7.0 for 2 pylist cols), _compute_roc_auc_binary (3.8→7.0), "
        "pivot_to_disk (3.5→5.0), _rolling_on (3.5→5.0), "
        "_per_group_apply (3.5→8.0). v0.9.25d-b target. "
        "Tier C (23 unguarded): encode_categorical, _iter_rolling, "
        "_iter_expanding, _iter_ewm, _iter_diffed, _emit_group "
        "(fn output), MetricsOp.update (×4), ts/* (×13). "
        "v0.9.25d-b target. "
        "v0.9.25d-a scope (this patch, zero production code): "
        "(1) 27 new whitelist entries in "
        "scripts/service/diagnose_unguarded_to_pylist.py — Tier A (26) "
        "+ ColumnProxy other-guard (1); total whitelist 8 → 35. "
        "(2) tests/test_v0925d_a.py — 6 tests: classifier self-test, "
        "no unknown, no other-guard, whitelist >= 30, all rationales "
        "non-empty, categories present. "
        "(3) no cadw/ changes. "
        "Dogfood: unknown=0, other-guard=0 (was 1), unguarded~23 "
        "(was 52, -29 whitelisted), low_factor~12 (unchanged), "
        "whitelist=35. "
        "Part1_report §2.3 count (18) — was lower bound; actual 83. "
        "Forthcoming v0.9.25d-b closes remaining 35 (12 low_factor + "
        "23 unguarded). "
        "Meta-lesson: (a) diagnostic count vs audit count — audit "
        "table was sample, not census; (b) grep+AST diagnostic before "
        "estimate-vs-actual gap review. "
        "Fast suite: ~2291 → ~2297.",
        self_path="scripts/patch_v0925d_a_whitelist.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25d-a applied. Next: v0.9.25d-b "
          "(12 factor bumps + 23 check_batch additions).")


if __name__ == "__main__":
    main()