#!/usr/bin/env python3
"""Read-only scanner for A.6 (v0.9.25d diagnostic phase).

Scans cadw/ for `.to_pylist()` calls; for each call, checks whether a
preceding `check_batch(extra_factor >= 3.0)` exists in the same or
enclosing function scope (AST-based, not substring — §13.4/252).

Also checks for preceding `check_state(est)` — treats as "other-guard"
(note in output; may be legitimate cover, e.g. `ColumnProxy.to_list`
computes est from n_rows explicitly).

Categories:
  guarded     — check_batch with extra_factor literal >= 3.0 precedes
  low_factor  — check_batch exists, but extra_factor < 3.0 (e.g. 1.0)
  unknown     — check_batch exists, extra_factor is non-literal
                (variable or method call); needs manual review
  other-guard — preceding check_state (without check_batch); manual
                verify that est scales with n_rows
  unguarded   — no preceding guard at all
  whitelisted — in WHITELIST (rationale required)

Exit codes:
  0 — all non-whitelisted sites guarded
  1 — unguarded or low_factor sites found (actionable)
  2 — unknown or other-guard sites exist (manual review)

Usage:
    uv run python scripts/service/diagnose_unguarded_to_pylist.py
    uv run python scripts/service/diagnose_unguarded_to_pylist.py --verbose
    uv run python scripts/service/diagnose_unguarded_to_pylist.py --self-test
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


CADW_ROOT = Path("cadw")

# Whitelist: sites where `to_pylist` is bounded by O(1) or O(K) where K
# is small and independent of N. Key = "{path}:{qualname}". Rationale
# must be explicit — trust = documented bound.
WHITELIST = {
    "cadw/ops/quantile.py:_stream_values_at_positions":
        "K = #targets (<= 2*len(q)); not O(N)",
    "cadw/ops/sort.py:_merge_keys":
        "K <= max_k=256 (fan-in); not O(N)",
    "cadw/dataframe.py:CadwDataFrame._repr_html_":
        "20-row sample via head(20); O(1)",
    "cadw/viz/_aggregate.py:line_agg":
        "K-row aggregate; not O(N)",
    "cadw/viz/matplotlib_adapter.py:_sample_pair":
        "n-bounded sample (n=10k default); not O(N)",
    "cadw/viz/plotly_adapter.py:_sample_pair":
        "n-bounded sample; not O(N)",
    "cadw/viz/seaborn_adapter.py:_sample_to_pandas":
        "n-bounded sample; not O(N)",
    "cadw/viz/_aggregate.py:sample":
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
    "cadw/dataframe.py:CadwDataFrame.encode_categorical":
        "pc.value_counts per batch; K per batch (unique values), not N; per-batch check_state(|uniques| * 180)",
    "cadw/dataframe.py:CadwDataFrame._per_group_apply._emit_group":
        "per-group fn output; group bounded by _flush_group (0.3*avail); fn's own alloc = user responsibility (§3.9)",
    "cadw/ops/metrics.py:MetricsOp.update":
        "guarded via reduce().check_batch(extra_factor=batch_memory_factor()); MetricsOp overrides to 8.0 for classification (K+1 pylist cols)",
    "cadw/ts/backtest.py:backtest_forecaster":
        "guarded via _check(guard, batch, name, factor) helper (cadw/ts/features.py) or equivalent per-batch pattern",
    "cadw/ts/features.py:_shift_stream.gen":
        "guarded via _check(guard, batch, name, factor) helper (cadw/ts/features.py) or equivalent per-batch pattern",
    "cadw/ts/features.py:diff_features.gen":
        "guarded via _check(guard, batch, name, factor) helper (cadw/ts/features.py) or equivalent per-batch pattern",
    "cadw/ts/models.py:_extract_series":
        "guarded via _check(guard, batch, name, factor) helper (cadw/ts/features.py) or equivalent per-batch pattern",
    "cadw/ts/stats.py:rolling_stats.gen":
        "guarded via _check(guard, batch, name, factor) helper (cadw/ts/features.py) or equivalent per-batch pattern",
    "cadw/ts/stats.py:rolling_crosscorr.gen":
        "guarded via _check(guard, batch, name, factor) helper (cadw/ts/features.py) or equivalent per-batch pattern",
    "cadw/ts/thf.py:TemporalHierarchy.fit":
        "per-batch group discovery (batch_size=10_000 fixed); result K bounded by unique groups, not N",
}


# ============ AST helpers (adapted from test_arrow_call_memory_guard) ====

def _iter_functions(tree):
    """Yield (qualname, func_node, enclosing_chain). Same shape as
    test_arrow_call_memory_guard._iter_functions; duplicated here to
    keep this tool independent of tests/."""
    def visit(node, qual, chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                yield from visit(child, qual + [child.name], chain)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join(qual + [child.name])
                yield qualname, child, tuple(chain)
                yield from visit(
                    child, qual + [child.name], chain + [child],
                )
            else:
                yield from visit(child, qual, chain)
    yield from visit(tree, [], [])


def _find_to_pylist_calls(func):
    """Yield Call nodes where attr == 'to_pylist', skipping nested
    defs (their calls belong to their own qualname)."""
    stack: list = [func]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                 ast.Lambda),
            ):
                continue
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "to_pylist"):
                yield child
            stack.append(child)


def _extract_extra_factor(call: ast.Call):
    """Return ('literal', 4.0) | ('unknown', None) | ('absent', None)."""
    for kw in call.keywords:
        if kw.arg == "extra_factor":
            if isinstance(kw.value, ast.Constant) and isinstance(
                kw.value.value, (int, float),
            ):
                return ("literal", float(kw.value.value))
            return ("unknown", None)
    return ("absent", None)


def _classify_guards(chain: tuple, call_line: int) -> dict:
    """Scan for guards preceding `call_line` in fn-chain (outer→inner).

    Returns {'check_batch': [(lineno, kind, value), ...],
             'check_state': [lineno, ...]}.
    """
    cb: list = []
    cs: list = []
    for fn in chain:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not isinstance(f, ast.Attribute):
                continue
            if node.lineno >= call_line:
                continue
            if f.attr == "check_batch":
                kind, val = _extract_extra_factor(node)
                cb.append((node.lineno, kind, val))
            elif f.attr == "check_state":
                cs.append(node.lineno)
    return {"check_batch": cb, "check_state": cs}


def _classify_site(chain: tuple, call_node: ast.Call) -> str:
    """Return one of: guarded, low_factor, unknown, other-guard,
    unguarded."""
    g = _classify_guards(chain, call_node.lineno)
    cb = g["check_batch"]
    if cb:
        # Use the LATEST preceding check_batch.
        _, kind, val = max(cb, key=lambda t: t[0])
        if kind == "literal":
            return "guarded" if val >= 3.0 else "low_factor"
        if kind == "unknown":
            return "unknown"
        if kind == "absent":
            # check_batch(batch, name=...) — default factor=1.0.
            return "low_factor"
    if g["check_state"]:
        return "other-guard"
    return "unguarded"


# ============ Diagnostic ============

def scan(verbose: bool = False) -> dict:
    """Scan cadw/ for to_pylist sites. Returns grouped results."""
    results: dict = {
        "guarded": [], "low_factor": [], "unknown": [],
        "other-guard": [], "unguarded": [], "whitelisted": [],
    }
    for py in sorted(CADW_ROOT.rglob("*.py")):
        if py.name == "__init__.py":
            continue
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        rel = py.as_posix()
        for qualname, func, chain in _iter_functions(tree):
            key = f"{rel}:{qualname}"
            wl = key in WHITELIST
            for call in _find_to_pylist_calls(func):
                if wl:
                    cat = "whitelisted"
                else:
                    cat = _classify_site((func, *chain), call)
                results[cat].append({
                    "file": rel, "line": call.lineno,
                    "qualname": qualname,
                    "whitelist_reason": WHITELIST.get(key, ""),
                })
    return results


def report(results: dict, verbose: bool) -> int:
    total = sum(len(v) for v in results.values())
    print(f"\n=== to_pylist diagnostic: {total} sites in cadw/ ===\n")

    actionable: list = []

    for cat, label in [
        ("unguarded", "UNGUARDED (no guard before to_pylist)"),
        ("low_factor", "LOW_FACTOR (check_batch with extra_factor < 3.0)"),
        ("unknown", "UNKNOWN (extra_factor is non-literal)"),
        ("other-guard", "OTHER-GUARD (check_state but not check_batch)"),
        ("whitelisted", "WHITELISTED (documented rationale)"),
        ("guarded", "GUARDED (check_batch extra_factor >= 3.0)"),
    ]:
        rows = results.get(cat, [])
        if not rows:
            continue
        print(f"--- {label}: {len(rows)} ---")
        for r in rows:
            print(f"  {r['file']}:{r['line']:<5} {r['qualname']}")
            if cat == "whitelisted" and r.get("whitelist_reason"):
                print(f"      reason: {r['whitelist_reason']}")
        if cat in ("unguarded", "low_factor", "unknown", "other-guard"):
            actionable.extend(rows)
        print()

    # Exit code
    if any(len(results.get(c, [])) for c in
           ("unguarded", "low_factor")):
        print("=== ACTION REQUIRED: unguarded or low_factor sites ===\n")
        print(f"Total actionable: {len(results.get('unguarded', [])) + len(results.get('low_factor', []))}")
        print(f"Manual review (unknown/other-guard): "
              f"{len(results.get('unknown', [])) + len(results.get('other-guard', []))}")
        return 1
    if results.get("unknown") or results.get("other-guard"):
        print("=== MANUAL REVIEW: unknown/other-guard sites ===\n")
        return 2
    print("=== ALL CLEAR: all non-whitelisted sites guarded ===\n")
    return 0


# ============ Self-test: AST-classification sanity ============

def _run_self_tests() -> int:
    """3 fake-source tests. Confirms classifier correctness — one per
    bug-class that burned v0.9.25c (substring, default-factor, before-
    check). Not pytest — this is a read-only diagnostic tool."""
    cases = [
        # (label, source, expected_category_per_call)
        (
            "guarded_check_batch_4.0",
            (
                "def f(g, reader):\n"
                "    for b in reader:\n"
                "        g.check_batch(b, name='x', extra_factor=4.0)\n"
                "        v = b.column('c').to_pylist()\n"
            ),
            ["guarded"],
        ),
        (
            "unguarded_no_check",
            (
                "def f(reader):\n"
                "    for b in reader:\n"
                "        v = b.column('c').to_pylist()\n"
            ),
            ["unguarded"],
        ),
        (
            "low_factor_default",
            (
                "def f(g, reader):\n"
                "    for b in reader:\n"
                "        g.check_batch(b, name='x')\n"
                "        v = b.column('c').to_pylist()\n"
            ),
            ["low_factor"],
        ),
        (
            "check_batch_after_pylist",
            (
                "def f(g, reader):\n"
                "    for b in reader:\n"
                "        v = b.column('c').to_pylist()\n"
                "        g.check_batch(b, extra_factor=4.0)\n"
            ),
            ["unguarded"],
        ),
        (
            "variable_extra_factor",
            (
                "def f(g, reader, ef):\n"
                "    for b in reader:\n"
                "        g.check_batch(b, extra_factor=ef)\n"
                "        v = b.column('c').to_pylist()\n"
            ),
            ["unknown"],
        ),
        (
            "method_extra_factor",
            (
                "def f(g, reader, op):\n"
                "    for b in reader:\n"
                "        g.check_batch(b, extra_factor=op.batch_memory_factor())\n"
                "        v = b.column('c').to_pylist()\n"
            ),
            ["unknown"],
        ),
        (
            "other_guard_check_state",
            (
                "def f(g, reader):\n"
                "    for b in reader:\n"
                "        g.check_state(1000, name='x')\n"
                "        v = b.column('c').to_pylist()\n"
            ),
            ["other-guard"],
        ),
    ]
    failures = 0
    for label, src, expected in cases:
        tree = ast.parse(src)
        got = []
        for qualname, func, chain in _iter_functions(tree):
            for call in _find_to_pylist_calls(func):
                got.append(_classify_site((func, *chain), call))
        ok = got == expected
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {label}: expected={expected}, got={got}")
        if not ok:
            failures += 1
    if failures:
        print(f"\nSELF-TEST FAILED: {failures}/{len(cases)} cases")
        return 1
    print(f"\nSELF-TEST PASSED: {len(cases)}/{len(cases)} cases")
    return 0


# ============ main ============

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Read-only scanner for A.6 to_pylist sites in cadw/",
    )
    ap.add_argument("--verbose", action="store_true",
                    help="also list guarded sites")
    ap.add_argument("--self-test", action="store_true",
                    help="run classifier self-tests (3 fake cases)")
    args = ap.parse_args()

    if not CADW_ROOT.exists():
        print(f"ERROR: {CADW_ROOT} not found (run from repo root)",
              file=sys.stderr)
        return 2

    if args.self_test:
        print("=== classifier self-test ===")
        return _run_self_tests()

    results = scan(verbose=args.verbose)
    return report(results, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())