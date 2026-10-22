#!/usr/bin/env python3
# v0.9.25d-b1 — Tier B: bump low_factor check_batch calls to 7.0.
#
# Diagnostic (v0.9.25d) found 12 low_factor sites: check_batch with
# factor < 3.0 (or absent) preceding to_pylist. 12 sites, but only
# 5 unique (file, qualname) — some functions have several to_pylist
# calls sharing one check_batch (e.g. `_rolling_on` has rid_list +
# src_vals; `_per_group_shift` has batch_first_key + gvals + prev_group).
#
# Data-driven approach (lesson v0.9.25c-fix5 partial-anchor):
#   - Load diagnostic module (stable, 7/7 self-test)
#   - Get low_factor list
#   - For each (file, qualname): find check_batch preceding to_pylist
#     (max line), bump factor to 7.0 (edit or insert before `)`)
#   - Verify edit compiles BEFORE write (compile-before-write, §13.4/211)
#   - Re-scan diagnostic: low_factor must be 0
#
# No manual anchors. If a site fails (non-literal factor, ambiguous
# anchor, multi-line insert), that site stays low_factor → verify_after
# fails → fix1 patches it precisely.
#
# Factor=7.0 rationale: 2 pylist calls × 3.5 B/value → 7.0 conservative
# bound. Sites with 1 pylist (e.g. `_per_group_apply` group_vals with 1
# group col) get over-guard (7.0 > 3.5) — safe, just conservative.
#
# Run: uv run python scripts/patch_v0925d_b1_factor_bumps.py

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
    assert_compiles, commit_with_guards, run_verify_in_subprocess,
    check_no_invalid_escape,
)


# ============ AST helpers (local, self-contained) ============

def _load_diag():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "diag", "scripts/service/diagnose_unguarded_to_pylist.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _iter_functions(tree):
    def visit(node, qual, chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                yield from visit(child, qual + [child.name], chain)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join(qual + [child.name])
                yield qualname, child, tuple(chain)
                yield from visit(child, qual + [child.name], chain + [child])
            else:
                yield from visit(child, qual, chain)
    yield from visit(tree, [], [])


def _walk_no_nested_defs(node):
    """Yield descendants, skipping nested def/class/lambda bodies."""
    stack = [node]
    while stack:
        n = stack.pop()
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef, ast.Lambda)):
                continue
            yield child
            stack.append(child)


def _get_func_by_qualname(tree, qualname):
    for qn, func, chain in _iter_functions(tree):
        if qn == qualname:
            return func, chain
    return None, None


# ============ bump one function ============

def _bump_factor(file: str, qualname: str, target: float = 7.0):
    """Bump check_batch extra_factor for this function.

    Returns (ok: bool, msg: str). Uses AST for locating, string edit
    for application, ast.parse(new_src) for compile-before-write.
    """
    path = Path(file)
    src = path.read_text()
    tree = ast.parse(src)
    func, chain = _get_func_by_qualname(tree, qualname)
    if func is None:
        return False, f"function {qualname!r} not found"

    to_pylist_lines = [
        n.lineno for n in _walk_no_nested_defs(func)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "to_pylist"
    ]
    if not to_pylist_lines:
        return False, "no to_pylist in function body"

    candidates = [
        n for n in _walk_no_nested_defs(func)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "check_batch"
        and any(n.lineno < tl for tl in to_pylist_lines)
    ]
    if not candidates:
        return False, "no preceding check_batch"

    cb = max(candidates, key=lambda n: n.lineno)
    lines = src.split("\n")
    seg = "\n".join(lines[cb.lineno - 1 : cb.end_lineno])

    # Construct new segment
    has_ef = False
    new_seg = seg
    for kw in cb.keywords:
        if kw.arg == "extra_factor":
            has_ef = True
            if not isinstance(kw.value, ast.Constant):
                return False, f"non-literal extra_factor at line {cb.lineno}"
            old_val = kw.value.value
            new_seg = seg.replace(
                f"extra_factor={old_val}", f"extra_factor={target}", 1,
            )
            if new_seg == seg:
                return False, f"replace failed in {seg[:60]!r}"
            break

    if not has_ef:
        stripped = seg.rstrip()
        if not stripped.endswith(")"):
            return False, (
                f"multi-line call end not `)`, cannot safely insert: "
                f"{stripped[-40:]!r}"
            )
        idx = seg.rfind(")")
        new_seg = seg[:idx] + f", extra_factor={target}" + seg[idx:]

    # Safety: no ambiguous anchors
    n_matches = src.count(seg)
    if n_matches > 1:
        return False, f"ambiguous anchor ({n_matches} matches in file)"

    new_src = src.replace(seg, new_seg, 1)
    if new_src == src:
        return False, "replace no-op (anchor same as new)"

    # Compile-before-write (§13.4/211)
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        return False, f"compile fail after edit: {e}"

    path.write_text(new_src)
    return True, "applied"


# ============ State / verify ============

def verify_state():
    print("\n=== verify v0.9.25d-a committed + diagnostic present ===")
    r = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    )
    if "v0.9.25d-a" not in r.stdout:
        raise SystemExit(f"v0.9.25d-a not in HEAD; got {r.stdout!r}")
    diag = Path("scripts/service/diagnose_unguarded_to_pylist.py")
    if not diag.exists():
        raise SystemExit(f"{diag} missing")
    print(f"  HEAD OK; diagnostic present")


def apply_bumps() -> bool:
    print("\n=== apply factor bumps (diagnostic-driven) ===")
    diag = _load_diag()
    results = diag.scan()
    sites = results.get("low_factor", [])
    print(f"  low_factor sites: {len(sites)}")
    if not sites:
        print("  none — nothing to bump")
        return True

    groups: dict = defaultdict(list)
    for s in sites:
        groups[(s["file"], s["qualname"])].append(s)

    print(f"  unique (file, qualname): {len(groups)}")
    success = 0
    failed = []
    for (file, qualname), glist in groups.items():
        ok, msg = _bump_factor(file, qualname, 7.0)
        if ok:
            success += len(glist)
            print(f"  OK    {file}::{qualname} ({len(glist)} site(s))")
        else:
            failed.append((file, qualname, msg))
            print(f"  FAIL  {file}::{qualname}: {msg}")

    print(f"\n  bumped: {success} site(s), failed groups: {len(failed)}")
    if failed:
        for f, q, m in failed:
            print(f"    {f}::{q}: {m}")
        return False
    return True


def verify_after():
    """Re-run diagnostic — low_factor must be 0."""
    print("\n=== verify: re-run diagnostic ===")
    diag = _load_diag()
    results = diag.scan()
    lf = results.get("low_factor", [])
    if lf:
        print(f"  STILL LOW FACTOR: {len(lf)} site(s)")
        for s in lf:
            print(f"    {s['file']}:{s['line']} {s['qualname']}")
        raise SystemExit(
            f"low_factor = {len(lf)} (expected 0). "
            f"Fix manually or via fix1."
        )
    print(f"  low_factor = 0 — clean")
    print(f"  unguarded: {len(results.get('unguarded', []))} (Tier C, deferred)")
    print(f"  whitelisted: {len(results.get('whitelisted', []))}")
    print(f"  guarded: {len(results.get('guarded', []))}")


# ============ tests ============

NEW_TEST = '''

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
'''


def add_tests():
    print("\n=== add test_low_factor_zero_after_bumps ===")
    p = Path("tests/test_v0925d_a.py")
    s = p.read_text()
    if "test_low_factor_zero_after_bumps" in s:
        print("  already present — skip")
        return
    p.write_text(s + NEW_TEST)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: diagnostic re-run ===")

    def _verify():
        import importlib.util as _iu
        from pathlib import Path as _P
        spec = _iu.spec_from_file_location(
            "diag", _P("scripts/service/diagnose_unguarded_to_pylist.py"),
        )
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        r = mod.scan()
        lf = len(r.get("low_factor", []))
        ung = len(r.get("unguarded", []))
        g = len(r.get("guarded", []))
        wl = len(r.get("whitelisted", []))
        assert lf == 0, f"low_factor={lf}, expected 0 (Tier B complete)"
        assert 18 <= ung <= 28, (
            f"unguarded={ung}, expected ~23 (Tier C unchanged by v0.9.25d-b1)"
        )
        assert g >= 15, f"guarded={g}, expected >= 15"
        assert wl >= 30, f"whitelisted={wl}, expected >= 30"
        print(
            f"  low_factor=0, unguarded={ung} (Tier C), "
            f"guarded={g}, whitelisted={wl}"
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

    print("\n=== pytest: targeted (v0925d + affected) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925d_a.py",
         "tests/test_mustfix_local_groupby.py",
         "tests/test_mustfix_skew.py",
         "tests/test_groupby_v2.py",
         "tests/test_metrics.py",
         "tests/test_metrics_e2e.py",
         "tests/test_category_beyond_memory.py",
         "tests/test_dataframe.py",
         "tests/test_shift_on_col.py",
         "tests/test_rolling_on_col.py",
         "tests/test_groupby_per_group.py",
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
    print("=== v0.9.25d-b1: factor bumps (Tier B) ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    ok = apply_bumps()
    if not ok:
        print("\n*** SOME BUMPS FAILED — applying what succeeded, "
              "verify_after will catch remaining ***")

    verify_after()  # raises SystemExit if low_factor > 0
    add_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25d-b1: bump low_factor check_batch to extra_factor=7.0 "
        "(Phase A.6 Tier B). Diagnostic (v0.9.25d) found 12 low_factor "
        "sites (check_batch with factor < 3.0 or absent) preceding "
        "to_pylist calls. Grouped into 5 unique (file, qualname): "
        "_per_group_shift, _per_group_rolling, _rolling_on, "
        "_per_group_apply, _compute_roc_auc_binary — each has 1-3 "
        "to_pylist calls sharing one check_batch. "
        "Data-driven bump via AST (lesson v0.9.25c-fix5 partial-anchor): "
        "load diagnostic, find last check_batch before to_pylist per "
        "function, bump extra_factor to 7.0 (2 pylist × 3.5 overhead). "
        "Compile-before-write (§13.4/211): ast.parse(new_src) BEFORE "
        "write_text; ambiguous-anchor check via src.count(seg) > 1. "
        "Result: low_factor=0 (was 12). unguarded unchanged (Tier C — "
        "v0.9.25d-b2 targets). Tests: +1 "
        "(test_low_factor_zero_after_bumps). "
        "Fast suite: ~2297 → ~2298.",
        self_path="scripts/patch_v0925d_b1_factor_bumps.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25d-b1 applied. Next: v0.9.25d-b2 "
          "(23 check_batch additions for Tier C).")


if __name__ == "__main__":
    main()