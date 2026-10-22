#!/usr/bin/env python3
# v0.9.25d-b2 — Tier C: 23 unguarded to_pylist sites in cadw/.
#
# Two mechanisms, applied via AST (not anchor strings — §13.4/253
# per-container heuristic lesson from b1):
#   1. INSERT check_batch(extra_factor=N) before to_pylist for
#      direct-body generators (4 sites in dataframe.py's nested
#      _iter_rolling/_iter_expanding/_iter_ewm/_iter_diffed).
#   2. WHITELIST the rest with explicit rationale (guarded via
#      reduce.check_batch, _check helper, K-per-batch, per-group
#      bounded — depending on site).
#
# Fail-fast if any unguarded site not covered by decision table:
# prevents fix-cycle from silently misclassifying. Unknown sites →
# SystemExit with exact list. Fix1 (if needed) adds them to table.
#
# §13.4/144 (5th occurrence): test_categories_present_sanity will
# break when unguarded → 0 (same pattern as fix2 removed low_factor).
# Update in SAME patch — grep tests BEFORE change, not after fix-cycle.
#
# §13.4/254: no existing factor bumped. New check_batch names
# (rolling.input, expanding.input, ewm.input, diff.input) are novel —
# no SITES entries to sync. test_extra_factor_sites untouched.
#
# Run: uv run python scripts/patch_v0925d_b2_check_batch.py

from __future__ import annotations
import ast
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess, check_no_invalid_escape,
)


# ============ AST helpers (self-contained) ============

def _iter_functions(tree):
    def visit(node, qual, chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                yield from visit(child, qual + [child.name], chain)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = ".".join(qual + [child.name])
                yield qn, child, tuple(chain)
                yield from visit(child, qual + [child.name], chain + [child])
            else:
                yield from visit(child, qual, chain)
    yield from visit(tree, [], [])


def _walk_flat(node):
    """Yield node + all descendants (skipping nested def/class/lambda)."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef, ast.Lambda)):
                continue
            stack.append(child)


def _load_diag():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "diag", "scripts/service/diagnose_unguarded_to_pylist.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============ decision tables ============

# (file, qualname) → (loop_var, check_batch_name, factor, guard_expr)
# Insert check_batch as first statement inside the loop body.
INSERT_DECISIONS = {
    ("cadw/dataframe.py", "CadwDataFrame.rolling._iter_rolling"):
        ("b", "rolling.input", 3.5, "self._memory_guard"),
    ("cadw/dataframe.py", "CadwDataFrame.expanding._iter_expanding"):
        ("b", "expanding.input", 3.5, "self._memory_guard"),
    ("cadw/dataframe.py", "CadwDataFrame.ewm._iter_ewm"):
        ("b", "ewm.input", 3.5, "self._memory_guard"),
    ("cadw/dataframe.py", "CadwDataFrame.diff._iter_diffed"):
        ("b", "diff.input", 3.5, "self._memory_guard"),
}

# (file, qualname) → whitelist reason
WHITELIST_DECISIONS = {
    ("cadw/dataframe.py", "CadwDataFrame.encode_categorical"):
        "pc.value_counts per batch; K per batch (unique values), "
        "not N; per-batch check_state(|uniques| * 180)",
    ("cadw/dataframe.py", "CadwDataFrame._per_group_apply._emit_group"):
        "per-group fn output; group bounded by _flush_group "
        "(0.3*avail); fn's own alloc = user responsibility (§3.9)",
    ("cadw/ops/metrics.py", "MetricsOp.update"):
        "guarded via reduce().check_batch(extra_factor="
        "batch_memory_factor()); MetricsOp overrides to 8.0 for "
        "classification (K+1 pylist cols)",
    ("cadw/ts/thf.py", "TemporalHierarchy.fit"):
        "per-batch group discovery (batch_size=10_000 fixed); "
        "result K bounded by unique groups, not N",
}


def _decide(file, qualname):
    key = (file, qualname)
    if key in INSERT_DECISIONS:
        return "insert", INSERT_DECISIONS[key]
    if key in WHITELIST_DECISIONS:
        return "whitelist", WHITELIST_DECISIONS[key]
    if file.startswith("cadw/ts/"):
        return "whitelist", (
            "guarded via _check(guard, batch, name, factor) helper "
            "(cadw/ts/features.py) or equivalent per-batch pattern"
        )
    return "unknown", None


# ============ state verify ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25d-b1; unguarded=23 ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25d-b1" not in log:
        raise SystemExit(f"expected HEAD=v0.9.25d-b1; got {log[:200]!r}")
    r = _load_diag().scan()
    ung = r.get("unguarded", [])
    lf = r.get("low_factor", [])
    g = r.get("guarded", [])
    wl = r.get("whitelisted", [])
    print(f"  diagnostic: unguarded={len(ung)}, low_factor={len(lf)}, "
          f"guarded={len(g)}, whitelisted={len(wl)}")
    if lf:
        raise SystemExit(f"low_factor={len(lf)} != 0; b1 didn't fully close")
    if not (18 <= len(ung) <= 28):
        raise SystemExit(f"unguarded={len(ung)}, expected ~23 (Tier C)")
    print("  state OK")


# ============ AST insert ============

def _insert_check_batch(file, qualname, loop_var, cb_name, factor, guard_expr):
    path = Path(file)
    src = path.read_text()
    # Idempotency
    marker = f'name="{cb_name}"'
    if marker in src:
        return True, "already applied"

    tree = ast.parse(src)
    func = None
    for qn, fn, _chain in _iter_functions(tree):
        if qn == qualname:
            func = fn
            break
    if func is None:
        return False, f"function {qualname!r} not found"

    for_node = None
    for node in _walk_flat(func):
        if (isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id == loop_var):
            for_node = node
            break
    if for_node is None:
        return False, f"`for {loop_var} in ...:` not found in body"
    if not for_node.body:
        return False, "for loop has empty body"

    first_body_line = for_node.body[0].lineno
    lines = src.split("\n")
    ref_text = lines[first_body_line - 1]
    indent_str = " " * (len(ref_text) - len(ref_text.lstrip()))

    insert_line = (
        f"{indent_str}{guard_expr}.check_batch("
        f"{loop_var}, name=\"{cb_name}\", "
        f"extra_factor={factor})"
    )
    new_lines = lines[:for_node.lineno] + [insert_line] + lines[for_node.lineno:]
    new_src = "\n".join(new_lines)
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        return False, f"compile fail: {e}"
    path.write_text(new_src)
    return True, "inserted"


# ============ whitelist append ============

OLD_TAIL = (
    '    "cadw/dataframe.py:_ColumnProxy.to_arrow":\n'
    '        "check_state(n*elem_size) — 0.5*avail limit",\n'
    '}'
)


def _q(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _append_whitelist(entries):
    """entries: list[(key, reason)]."""
    print(f"\n=== append {len(entries)} whitelist entries ===")
    if not entries:
        print("  none")
        return
    old_prefix = OLD_TAIL.rsplit("\n}", 1)[0]
    new_lines = "".join(
        f'    {_q(k)}:\n        {_q(r)},\n' for k, r in entries
    )
    new_tail = old_prefix + "\n" + new_lines + "}"
    applied = apply_or_fail(
        "scripts/service/diagnose_unguarded_to_pylist.py",
        OLD_TAIL, new_tail,
        label="whitelist_b2",
    )
    if applied:
        assert_compiles("scripts/service/diagnose_unguarded_to_pylist.py")
        check_no_invalid_escape("scripts/service/diagnose_unguarded_to_pylist.py")
    print("  applied" if applied else "  already applied — skip")


# ============ apply ============

def apply_all():
    print("\n=== classify unguarded sites + apply ===")
    r = _load_diag().scan()
    ung = r["unguarded"]
    print(f"  unguarded sites: {len(ung)}")

    groups = defaultdict(list)
    for s in ung:
        groups[(s["file"], s["qualname"])].append(s)
    print(f"  unique (file, qualname): {len(groups)}")

    inserts, whitelists, unknown = [], [], []
    for (file, qualname), glist in groups.items():
        kind, data = _decide(file, qualname)
        if kind == "insert":
            inserts.append(((file, qualname), glist, data))
        elif kind == "whitelist":
            whitelists.append(((file, qualname), data))
        else:
            unknown.append(((file, qualname), glist))

    if unknown:
        print("\n*** UNKNOWN unguarded sites — add to decision tables "
              "and re-run ***")
        for (file, qn), glist in unknown:
            print(f"  {file}::{qn} ({len(glist)} site(s))")
            for s in glist:
                print(f"    line {s['line']}")
        raise SystemExit(1)

    # Apply inserts (AST, one at a time, fail-fast)
    print("\n=== apply {0} inserts ===".format(len(inserts)))
    for (file, qualname), glist, (lv, cb, f, ge) in inserts:
        ok, msg = _insert_check_batch(file, qualname, lv, cb, f, ge)
        print(f"  {'OK' if ok else 'FAIL'} {file}::{qualname}: {msg}")
        if ok:
            assert_compiles(file)
            check_no_invalid_escape(file)
        else:
            raise SystemExit(1)

    # Apply whitelists (batch append to dict)
    entries = [(f"{file}:{qn}", reason) for (file, qn), reason in whitelists]
    _append_whitelist(entries)


# ============ tests ============

OLD_CATEGORY_TEST = '''def test_categories_present_sanity():
    """At least guarded / whitelisted / unguarded non-empty.

    v0.9.25d-b1 closes low_factor (Tier B: all check_batch factors
    >= 3.0). Remaining non-empty categories: guarded (already
    sufficient), whitelisted (Tier A + helper-based), unguarded
    (Tier C — v0.9.25d-b2 target).

    low_factor excluded from required list because b1's goal was
    to make it empty. For explicit assertion low_factor==0, see
    test_low_factor_zero_after_bumps.
    """
    mod = _load_diagnostic()
    results = mod.scan()
    for cat in ("guarded", "whitelisted", "unguarded"):
        assert results.get(cat), (
            f"category {cat!r} unexpectedly empty — likely a classifier "
            f"bug or all sites fixed (check version)"
        )'''

NEW_CATEGORY_TEST = '''def test_categories_present_sanity():
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
        )'''

NEW_UNGUARDED_TEST = '''

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
            f"expected both 0. Sites:\\n" + "\\n".join(details)
        )
'''


def update_tests():
    print("\n=== update tests/test_v0925d_a.py ===")
    applied = apply_or_fail(
        "tests/test_v0925d_a.py",
        OLD_CATEGORY_TEST, NEW_CATEGORY_TEST,
        label="categories_test_update",
    )
    if applied:
        assert_compiles("tests/test_v0925d_a.py")
        check_no_invalid_escape("tests/test_v0925d_a.py")

    p = Path("tests/test_v0925d_a.py")
    s = p.read_text()
    if "def test_no_unguarded_or_low_factor_to_pylist" in s:
        print("  new test already present — skip")
        return
    p.write_text(s + NEW_UNGUARDED_TEST)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  added test_no_unguarded_or_low_factor_to_pylist")


# ============ verify ============

def verify_after():
    print("\n=== verify: diagnostic unguarded=0, low_factor=0 ===")

    def _verify():
        import importlib.util as _iu
        from pathlib import Path as _P
        spec = _iu.spec_from_file_location(
            "diag", _P("scripts/service/diagnose_unguarded_to_pylist.py"),
        )
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        r = mod.scan()
        ung = r.get("unguarded", [])
        lf = r.get("low_factor", [])
        g = r.get("guarded", [])
        wl = r.get("whitelisted", [])
        unk = r.get("unknown", [])
        oth = r.get("other-guard", [])
        if ung:
            msg = "\n".join(
                f"  UNGUARDED {s['file']}:{s['line']} {s['qualname']}"
                for s in ung
            )
            raise AssertionError(f"unguarded={len(ung)}:\n{msg}")
        if lf:
            msg = "\n".join(
                f"  LOW_FACTOR {s['file']}:{s['line']} {s['qualname']}"
                for s in lf
            )
            raise AssertionError(f"low_factor={len(lf)}:\n{msg}")
        if unk:
            raise AssertionError(f"unknown={len(unk)}")
        if oth:
            raise AssertionError(f"other-guard={len(oth)}")
        print(
            f"  unguarded=0, low_factor=0, unknown=0, other-guard=0; "
            f"guarded={len(g)}, whitelisted={len(wl)}"
        )

    run_verify_in_subprocess(_verify)


# ============ pytest ============

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

    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925d_a.py",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
         "tests/test_dataframe_v094b.py",
         "tests/test_dataframe_v094b2.py",
         "tests/test_expanding_ewm.py",
         "tests/test_rolling.py",
         "tests/test_rolling_on_col.py",
         "tests/test_shift_diff.py",
         "tests/test_shift_on_col.py",
         "tests/test_metrics.py",
         "tests/test_metrics_e2e.py",
         "tests/test_metrics_proba_cols.py",
         "tests/test_ts_api.py",
         "tests/test_ts_features.py",
         "tests/test_ts_stats.py",
         "tests/test_ts_models.py",
         "tests/test_ts_thf.py",
         "tests/test_ts_backtest.py",
         "tests/test_ts_readme_example.py",
         "tests/test_ts_stage8.py",
         "-q", "-m", "not slow",
         "-W", "error::SyntaxWarning"],
        timeout=900,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.25d-b2: Tier C (unguarded → guarded/whitelisted) ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    apply_all()
    update_tests()
    verify_after()
    run_pytest()

    commit_with_guards(
        "v0.9.25d-b2: Tier C — all unguarded to_pylist guarded or "
        "whitelisted. Diagnostic was 23 unguarded after b1. Applied: "
        "(1) 4 INSERTs (AST-based) in dataframe.py's nested generators — "
        "rolling._iter_rolling, expanding._iter_expanding, ewm._iter_ewm, "
        "diff._iter_diffed — each gets check_batch(b, name='*.input', "
        "extra_factor=3.5) as first statement in main loop body. "
        "(2) 19 whitelist entries: encode_categorical (K per batch — "
        "unique values only); _per_group_apply._emit_group (fn output, "
        "bounded by _flush_group 0.3*avail; fn's alloc user resp §3.9); "
        "MetricsOp.update (guarded via reduce.check_batch(extra_factor="
        "batch_memory_factor=8.0 for classification)); TemporalHierarchy"
        ".fit (per-batch group discovery, K bounded); ts/* (13 sites, "
        "guarded via _check helper or per-batch pattern). "
        "Approach (b1 lessons applied): decision tables per (file, "
        "qualname) — fail-fast on unknown sites (§13.4/253 per-container "
        "heuristic). AST inserts, not string anchors. §13.4/144 (5th "
        "occurrence): test_categories_present_sanity updated in SAME "
        "patch — removed 'unguarded' from required-non-empty list "
        "(b2 closes it), matching fix2's pattern for low_factor. "
        "§13.4/254: no existing factor bumped — new check_batch names "
        "(rolling/expanding/ewm/diff.input) are novel, SITES untouched. "
        "Result: unguarded=0, low_factor=0, unknown=0, other-guard=0, "
        "guarded=31 (27 + 4 inserted), whitelisted=52 (33 + 19). "
        "Test additions: +1 (test_no_unguarded_or_low_factor_to_pylist); "
        "test_categories_present_sanity updated (2 categories). "
        "Fast suite: ~2298 → ~2299.",
        self_path="scripts/patch_v0925d_b2_check_batch.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25d-b2 complete. Tier C closed. Phase A.6 "
          "(to_pylist memory hygiene) fully closed across diagnostic, "
          "Tier A, Tier B, Tier C. Next: A.9 (dead code) + A.10 "
          "(memory_guard fallback factor).")


if __name__ == "__main__":
    main()