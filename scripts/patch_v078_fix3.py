#!/usr/bin/env python3
"""v0.7.8-fix3: fit_gbdt_classifier dispatch preserves binary error messages.

Bugs (v0.7.8 regression):
    test_fit_gbdt_classifier_non_binary_y_raises — y={2,3}, expected "binary",
      got "0-indexed" (from _validate_multiclass_y).
    test_fit_gbdt_classifier_single_class_raises — y={1}, expected "both
      classes", got "0-indexed".

Fix:
    In dispatch, count distinct first. Route:
        <= 2 distinct (or non-int)  -> binary path (which raises the
                                       informative "binary (0/1)" or
                                       "both classes" error).
        >= 3 distinct               -> multiclass path (which raises
                                       "0-indexed" error for non-0-indexed).

    Preserves backward compat for old error messages; new "0-indexed"
    message only for {0,1,5} and similar (intended multiclass, bad values).

Also:
    - Add AST guard `test_all_exports_importable` (from §13.4/115):
      every name in cadw.__all__ / cadw.ml.__all__ actually imports.
      Catches the class of silent no-op bugs (v0.7.8, v0.7.8-fix).
    - CHANGELOG [0.7.8] fix note.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:25]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


# ============ 1. Dispatch fix ============

def fix_dispatch():
    print("\n=== cadw/ml/gbdt.py: dispatch counts distinct ===")
    p = Path("cadw/ml/gbdt.py")
    s = p.read_text()

    old = '''    if n_classes is None:
        K, _ = _validate_multiclass_y(df, y_col)
    else:
        K = int(n_classes)
'''

    new = '''    if n_classes is None:
        # v0.7.8-fix3: count distinct BEFORE full validation.
        # Backward compat: single-class / 2-non-0-indexed cases route to
        # binary path, which raises the historical "both classes" /
        # "binary (0/1)" error. Multiclass path (with stricter "0-indexed"
        # error) only for intended-K>=3 cases.
        distinct: set[int] = set()
        has_non_int = False
        for b in df._iter_filtered_batches():
            y = _column_to_f64(b.column(y_col))
            for v in y[~np.isnan(y)]:
                fv = float(v)
                if fv != int(fv):
                    has_non_int = True
                    break
                distinct.add(int(fv))
            if has_non_int:
                break
        if has_non_int or len(distinct) <= 2:
            K = 2  # route to binary; _validate_binary_y raises proper error
        else:
            K = len(distinct)  # route to multiclass; _validate_multiclass_y
    else:
        K = int(n_classes)
'''

    s = replace_exact(s, old, new, tag="dispatch-fix")
    p.write_text(s)
    compile_check(p)
    print("  OK: dispatch preserves backward-compat error messages")


# ============ 2. AST export guard ============

GUARD_SRC = '''

# ---------- v0.7.8-fix3: every name in __all__ actually imports ----------
#
# Catches silent no-op patch bugs (§13.4/44, 111, 114, 115): a patched
# __init__.py could add a name to `__all__` but fail to add the import
# line, leaving ImportError for downstream users while test collection
# passes (until someone does `from cadw import X`).
#
# Three levels of verify:
#   1. AST: is the name in an ImportFrom / Assign?
#   2. __all__: is every name present in module?
#   3. Runtime: can we actually `from pkg import name`?

import ast as _ast
import inspect as _inspect
import importlib as _importlib


def test_cadw_all_exports_importable():
    """Every name in cadw.__all__ must be importable from cadw."""
    import cadw
    missing = []
    for name in cadw.__all__:
        if not hasattr(cadw, name):
            missing.append(name)
    assert not missing, (
        f"cadw.__all__ references names not importable from cadw: "
        f"{missing}. Likely __all__ updated without import line (§13.4/115)."
    )


def test_cadw_ml_all_exports_importable():
    """Every name in cadw.ml.__all__ must be importable from cadw.ml."""
    import cadw.ml
    missing = []
    for name in cadw.ml.__all__:
        if not hasattr(cadw.ml, name):
            missing.append(name)
    assert not missing, (
        f"cadw.ml.__all__ references names not importable from cadw.ml: "
        f"{missing}"
    )


def test_cadw_ast_all_matches_imports():
    """AST: every name in `cadw.__all__` is either imported or assigned
    at module level. Catches the specific bug where __all__ line was
    added but import line was missed.
    """
    import cadw
    src = _inspect.getsource(cadw)
    tree = _ast.parse(src)

    imported: set[str] = set()
    assigned: set[str] = set()
    for node in tree.body:
        if isinstance(node, _ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
        elif isinstance(node, _ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, _ast.Assign):
            for target in node.targets:
                if isinstance(target, _ast.Name):
                    assigned.add(target.id)

    all_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, _ast.Assign):
            for target in node.targets:
                if (isinstance(target, _ast.Name)
                        and target.id == "__all__"
                        and isinstance(node.value, _ast.List)):
                    for elt in node.value.elts:
                        if isinstance(elt, _ast.Constant):
                            all_names.add(elt.value)

    missing = all_names - imported - assigned - {"__all__"}
    # Allow names re-exported from other modules (they appear in ImportFrom
    # and thus would be in `imported` already). If anything is missing —
    # the __all__ line was added without the import.
    assert not missing, (
        f"cadw/__init__.py: names in __all__ but not imported/assigned: "
        f"{missing}. Add `from ... import name` line (§13.4/115)."
    )
'''


def add_guard():
    print("\n=== tests/test_api_availability.py: +export guard ===")
    p = Path("tests/test_api_availability.py")
    s = p.read_text()
    if "test_cadw_all_exports_importable" in s:
        print("  already present — skip")
        return
    s = s.rstrip() + GUARD_SRC + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: 3 AST/runtime export guards")


# ============ 3. CHANGELOG fix note ============

def patch_changelog():
    print("\n=== CHANGELOG.md: [0.7.8] fix note ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "Dispatch preserves backward-compat error messages" in s:
        print("  already present — skip")
        return

    # Find [0.7.8] block and insert note under it
    old = "## [0.7.8] — 2026-10-19\n"
    if old not in s:
        print("  WARNING: [0.7.8] not found — skipping")
        return

    note = (
        "\n### Исправлено в v0.7.8-fix3\n"
        "- **Dispatch preserves backward-compat error messages.**\n"
        "  `fit_gbdt_classifier(n_classes=None)` считает distinct **перед**\n"
        "  `_validate_multiclass_y`. 2 distinct (или non-integer) → binary\n"
        "  path (`_validate_binary_y` даёт исторический `\"binary (0/1)\"` /\n"
        "  `\"both classes\"`); >2 distinct → multiclass path (`\"0-indexed\"`).\n"
        "  Без этого `{2, 3}` и `{1}` давали `\"0-indexed\"` вместо ожидаемых\n"
        "  сообщений — ломалось `test_gbdt_classification.py`.\n"
        "- **+AST export guard** (`tests/test_api_availability.py`):\n"
        "  `test_cadw_all_exports_importable`,\n"
        "  `test_cadw_ml_all_exports_importable`,\n"
        "  `test_cadw_ast_all_matches_imports`. Ловит класс багов\n"
        "  «__all__ обновлён, import забыт» (§13.4/115).\n"
    )
    s = s.replace(old, old + note, 1)
    p.write_text(s)
    print("  OK: [0.7.8] fix note")


# ============ main ============

def main():
    print("=== v0.7.8-fix3 ===")
    fix_dispatch()
    add_guard()
    patch_changelog()

    print("\n=== pytest tests/test_gbdt_classification.py -v (regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_gbdt_classification.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.8-fix3 (regression)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_gbdt_multiclass.py (new, unchanged) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_gbdt_multiclass.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** MULTICLASS FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.8-fix3 (multiclass)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_api_availability.py (export guard) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_api_availability.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** EXPORT GUARD FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.8-fix3 (exports)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.8-fix3: dispatch preserves binary error messages (distinct "
        "count); +AST export guard (§13.4/115)"])

    print("\n==> v0.7.8-fix3 complete. Готов squash → tag v0.7.8.")


if __name__ == "__main__":
    main()