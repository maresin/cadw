#!/usr/bin/env python3
# v0.9.25f-a-fix4 — stale whitelist paths after extract to new file.
#
# STATE: HEAD = v0.9.25f-diagnostic-fix1. v0.9.25f-a uncommitted
# (extract + fix1b + fix2 + fix3 applied on disk). Fast-suite (2311
# passed) caught 2 stale whitelist assertions:
#
#   test_verify_whitelist_real_cadw (test_patch_utils.py):
#     orphan keys in WHITELISTED_FUNCS (test_arrow_call_memory_guard.py):
#       "cadw/dataframe.py:_concat_batches"
#       "cadw/dataframe.py:_convert_batch"
#     Both symbols moved to cadw/dataframe_helpers.py. `_collect_all_
#     qualnames(rglob)` sees them at new path; keys at old path =
#     orphan (dead whitelist → no guard applied).
#
#   test_no_unguarded_or_low_factor_to_pylist (test_v0925d_a.py):
#     unguarded=1: cadw/dataframe_helpers.py:70 _ArrowTail.push
#     WHITELIST in diagnose_unguarded_to_pylist.py has key
#     "cadw/dataframe.py:_ArrowTail.push" — old path. Diagnostic scans
#     rglob, so it sees _ArrowTail.push at dataframe_helpers.py;
#     key at old path doesn't match → classified unguarded.
#
# §13.4/144 (8th+9th occurrence this session): stale-test-on-refactor.
# §13.4/261 (NEW): move-file refactor requires grep-all-whitelist-
# paths. Any symbol moved to new file → grep WHITELISTED_FUNCS +
# WHITELIST + WHITELIST_DECISIONS in diagnostic + AST guard + any
# patch tables. Rule: after extract, rglob for path references in
# tests/scripts; update to new path in SAME patch that moves them.
#
# Detection worked correctly BOTH ways: WHITELISTED_FUNCS detects
# orphan via _collect_all_qualnames(rglob); diagnostic classifies
# unguarded because key doesn't match. Both are "whitelist path
# stale" — symptom differs, root identical.
#
# Fix scope: 3 path updates, all apply_or_fail. Zero production
# changes. Fast suite 2311 → ~2313 (+2 fix from whitelist test
# passing).
#
# Fix-cycle count v0.9.25f-a: extract → fix1 (verify bug) → fix1b
# (mutation, pre-flight abort) → fix2 (test replace, dogfood abort)
# → fix3 (finish, add synthetic) → fix4 (whitelist paths). 6 attempts,
# budget 2 (3× over). Zero bad-commits: each attempt caught at next
# gate (pytest after extract; verify_state after fix1; pre-flight
# after fix1b; dogfood after fix2; fast-suite after fix3). Churn 100%
# tooling + whitelist paths, NOT in extract logic (extract, numpy,
# symtable guard correct since their respective patches).
#
# Add meta-guard in tests/test_v0925f_a.py:
#   test_all_whitelist_paths_valid — parses both WHITELISTED_FUNCS
#   and WHITELIST, verifies every key's `path:qualname` path exists
#   in current cadw tree with that qualname defined. Prevents future
#   path drift from being caught only on fast-suite.
#
# Run: uv run python scripts/patch_v0925f_a_fix4_whitelist_paths.py

from __future__ import annotations
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    check_no_invalid_escape,
)


TESTS_WHITELIST_PATH = "tests/test_arrow_call_memory_guard.py"
DIAG_PATH = "scripts/service/diagnose_unguarded_to_pylist.py"
NEW_TESTS_PATH = "tests/test_v0925f_a.py"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25f-diagnostic-fix1; extract applied; "
          "2 stale whitelists present ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25f-diagnostic-fix1" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25f-diagnostic-fix1; got {log[:200]!r}"
        )
    if "v0.9.25f-a" in log:
        raise SystemExit("v0.9.25f-a already committed — fix4 misplaced")

    # helpers.py exists + 10 helpers
    hp = Path("cadw/dataframe_helpers.py")
    if not hp.exists():
        raise SystemExit("cadw/dataframe_helpers.py missing")
    h_tree = ast.parse(hp.read_text())
    h_defs = {
        n.name for n in h_tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef,
                          ast.AsyncFunctionDef))
    }
    expected = {
        "_ArrowTail", "_RollState", "_ExpandingState", "_EwmState",
        "_group_boundaries_arrow", "_convert_batch", "_concat_batches",
        "_empty_value_counts_table", "_reservoir_positions",
        "_weighted_reservoir_positions",
    }
    missing = expected - h_defs
    if missing:
        raise SystemExit(f"helpers.py missing {sorted(missing)}")

    # dataframe.py no longer defines moved symbols
    df_tree = ast.parse(Path("cadw/dataframe.py").read_text())
    df_defs = {
        n.name for n in df_tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef,
                          ast.AsyncFunctionDef))
    }
    still_in_facade = expected & df_defs
    if still_in_facade:
        raise SystemExit(
            f"helpers still in dataframe.py: {sorted(still_in_facade)} — "
            f"extract not applied?"
        )

    # Stale whitelists present (the bugs we fix)
    tw = Path(TESTS_WHITELIST_PATH).read_text()
    if '"cadw/dataframe.py:_concat_batches"' not in tw:
        raise SystemExit(
            "expected stale `_concat_batches` key — already updated?"
        )
    if '"cadw/dataframe.py:_convert_batch"' not in tw:
        raise SystemExit(
            "expected stale `_convert_batch` key — already updated?"
        )
    diag = Path(DIAG_PATH).read_text()
    if '"cadw/dataframe.py:_ArrowTail.push"' not in diag:
        raise SystemExit(
            "expected stale `_ArrowTail.push` key in diagnostic — "
            "already updated?"
        )

    # New tests file exists with 7 tests + no meta-guard yet
    nt = Path(NEW_TESTS_PATH).read_text()
    if "test_all_whitelist_paths_valid" in nt:
        raise SystemExit(
            "test_all_whitelist_paths_valid already present — "
            "fix4 add_meta_guard misplaced"
        )

    print("  state OK: extract applied, 3 stale whitelist paths, "
          "no meta-guard yet")


# ============ 1. update whitelist paths ============

def update_paths():
    print("\n=== update whitelist paths ===")
    updates = [
        (TESTS_WHITELIST_PATH,
         '"cadw/dataframe.py:_concat_batches"',
         '"cadw/dataframe_helpers.py:_concat_batches"',
         "test_wl_concat"),
        (TESTS_WHITELIST_PATH,
         '"cadw/dataframe.py:_convert_batch"',
         '"cadw/dataframe_helpers.py:_convert_batch"',
         "test_wl_convert"),
        (DIAG_PATH,
         '"cadw/dataframe.py:_ArrowTail.push"',
         '"cadw/dataframe_helpers.py:_ArrowTail.push"',
         "diag_wl_arrowtail"),
    ]
    for path, old, new, label in updates:
        applied = apply_or_fail(path, old, new, label=label)
        if applied:
            assert_compiles(path)
            check_no_invalid_escape(path)
        print(f"  {label}: {'applied' if applied else 'skip'}")


# ============ 2. add meta-guard ============

META_GUARD = '''

def test_all_whitelist_paths_valid():
    """v0.9.25f-a-fix4 (§13.4/261): every whitelist key's file path
    exists and defines the named qualname.

    Prevents stale-path class that broke v0.9.25f-a (extract moved
    `_concat_batches` / `_convert_batch` / `_ArrowTail.push` to
    cadw/dataframe_helpers.py; whitelist keys still referenced
    cadw/dataframe.py → orphan in WHITELISTED_FUNCS (dead guard) +
    unguarded classification in diagnostic).

    Two whitelists checked:
      1. tests/test_arrow_call_memory_guard.py::WHITELISTED_FUNCS
      2. scripts/service/diagnose_unguarded_to_pylist.py::WHITELIST

    Both are keyed by `{path}:{qualname}`; qualname may be dotted for
    methods (Class.method). Uses AST qualname collection (same as
    verify_whitelist + _collect_all_qualnames).
    """
    import ast as _ast
    import importlib as _importlib
    import importlib.util as _importlib_util
    from pathlib import Path as _P

    cadw_root = _P("cadw")
    actual: set = set()
    for py in cadw_root.rglob("*.py"):
        try:
            tree = _ast.parse(py.read_text())
        except (SyntaxError, OSError):
            continue
        try:
            rel = py.relative_to(_P(".")).as_posix()
        except ValueError:
            rel = py.as_posix()

        def _walk(node, prefix):
            body = getattr(node, "body", None)
            if body is None:
                return
            for child in body:
                if isinstance(child, _ast.FunctionDef):
                    qn = prefix + child.name
                    actual.add(rel + ":" + qn)
                    _walk(child, qn + ".")
                elif isinstance(child, _ast.ClassDef):
                    cp = prefix + child.name + "."
                    for item in child.body:
                        if isinstance(item, _ast.FunctionDef):
                            qn = cp + item.name
                            actual.add(rel + ":" + qn)
                            _walk(item, qn + ".")
        _walk(tree, "")

    # --- Whitelist 1: WHITELISTED_FUNCS in test_arrow_call_memory_guard
    wl1 = _importlib.import_module(
        "tests.test_arrow_call_memory_guard"
    ).WHITELISTED_FUNCS
    stale1 = [k for k in wl1 if k not in actual]

    # --- Whitelist 2: WHITELIST in diagnostic
    spec = _importlib_util.spec_from_file_location(
        "diag_mod",
        _P("scripts/service/diagnose_unguarded_to_pylist.py"),
    )
    diag = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(diag)
    wl2 = diag.WHITELIST
    stale2 = [k for k in wl2 if k not in actual]

    msg_parts = []
    if stale1:
        msg_parts.append(
            f"WHITELISTED_FUNCS orphan keys: {sorted(stale1)}"
        )
    if stale2:
        msg_parts.append(
            f"diagnostic WHITELIST orphan keys: {sorted(stale2)}"
        )

    assert not msg_parts, (
        "§13.4/261: stale whitelist path(s) after refactor:\\n"
        + "\\n".join(msg_parts)
        + "\\nUpdate to new path in SAME patch that moves the "
        + "symbol."
    )
'''


def add_meta_guard():
    print("\n=== add test_all_whitelist_paths_valid ===")
    p = Path(NEW_TESTS_PATH)
    s = p.read_text()
    if "test_all_whitelist_paths_valid" in s:
        print("  already present — skip")
        return
    p.write_text(s + META_GUARD)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. pytest ============

def run_pytest():
    print("\n=== targeted: verify_whitelist + diagnostic + v0925f_a ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py::test_verify_whitelist_real_cadw",
         "tests/test_v0925d_a.py",
         "tests/test_v0925f_a.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** STALE WHITELIST NOT FULLY FIXED — ABORT ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 4. main ============

def main():
    print("=== v0.9.25f-a-fix4: whitelist path updates ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    update_paths()
    add_meta_guard()
    run_pytest()

    commit_with_guards(
        "v0.9.25f-a: partial extract — 10 pure helpers → "
        "cadw/dataframe_helpers.py + fix1b (numpy) + fix2 (symtable) "
        "+ fix3 (finish) + fix4 (whitelist paths). "
        "Phase B diagnostic (v0.9.25f) found 48 AST-guard tests "
        "scanning inspect.getsource(cadw.dataframe) — HARD blocker "
        "for full split. Full split DEFERRED to v0.9.26. "
        "Extract: 10 helpers (433 L) with no CadwDataFrame "
        "cross-refs — 4 leaf state classes (_ArrowTail, _RollState, "
        "_ExpandingState, _EwmState) + 6 pure functions "
        "(_group_boundaries_arrow, _convert_batch, _concat_batches, "
        "_empty_value_counts_table, _reservoir_positions, "
        "_weighted_reservoir_positions). dataframe.py 7172 → ~6740 L. "
        "Backward compat via re-export. AST guards intact. "
        "fix1b: missing module-level numpy in extract header (37 tests "
        "NameError). §13.4/257: extracted module needs runtime smoke of "
        "every symbol. "
        "fix2: my own guard test used ast.walk + manual local-collection "
        "→ false-positived on comprehension/except-as/AnnAssign. "
        "Replaced with symtable. §13.4/259: ast.walk is not a scope "
        "analyzer. "
        "fix3: fix2's dogfood `_verify` had triple-quoted 0-indent "
        "templates → textwrap.dedent no-op → IndentationError. Skipped "
        "subprocess dogfood; added test_symtable_guard_no_false_"
        "positives (4 cases) as permanent pytest. §13.4/260: textwrap."
        "dedent fails with 0-indent triple-quoted content. "
        "fix4: 3 stale whitelist paths after extract to new file. "
        "(1) test_arrow_call_memory_guard.py::WHITELISTED_FUNCS "
        "orphan `cadw/dataframe.py:_concat_batches` + `_convert_batch` "
        "→ both moved to dataframe_helpers.py. "
        "(2) diagnose_unguarded_to_pylist.py::WHITELIST orphan "
        "`cadw/dataframe.py:_ArrowTail.push` → moved. "
        "Both detected correctly: WHITELISTED_FUNCS via "
        "_collect_all_qualnames(rglob); diagnostic via _ArrowTail.push "
        "classified unguarded because key doesn't match. "
        "§13.4/261 (NEW): move-file refactor requires grep-all-"
        "whitelist-paths. After extract, rglob for path references in "
        "tests/scripts; update to new path in SAME patch that moves "
        "them. Add meta-guard test_all_whitelist_paths_valid — parses "
        "both whitelists, verifies every key's `path:qualname` exists "
        "in current cadw tree. Prevents future path drift from being "
        "caught only on fast-suite. "
        "Fix-cycle count v0.9.25f-a: extract → fix1 → fix1b → fix2 → "
        "fix3 → fix4 = 6 attempts, budget 2 (3× over). Zero bad-"
        "commits — each caught at next gate (pytest / verify_state / "
        "pre-flight / dogfood / fast-suite). Churn 100% in test-"
        "tooling + whitelist infrastructure, NOT in extract logic. "
        "Lesson for v0.9.26 full split: b1..b7 sequence with path-"
        "update step in same patch that moves symbols (§13.4/238 "
        "multi-feature cascade). "
        "New tests/test_v0925f_a.py: 8 (4 original + 2 fix1b "
        "smoke/guard + 1 fix3 synthetic + 1 fix4 meta-guard). "
        "Next: Phase C — Pandas P0 (20 features, HIGH user value). "
        "Fast suite: ~2306 → ~2313 (+7).",
        self_path="scripts/patch_v0925f_a_fix4_whitelist_paths.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25f-a COMPLETE (extract + fix1b + fix2 + fix3 + "
          "fix4). Phase B deferred to v0.9.26. Next: Phase C — "
          "Pandas P0.")


if __name__ == "__main__":
    main()