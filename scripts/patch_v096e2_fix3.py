#!/usr/bin/env python3
# v0.9.6e2-fix3 — test_patch_utils.py: ast missing + meta-guard.
#
# Same class as the helper exists for (§13.4/192) — ironic repeat of
# §13.4/178 (scaffold needs its own protection). fix2's dogfood ran
# on 5 test files but NOT on test_patch_utils.py itself → missed
# that test_parse_method_source_indented references `ast` without
# import. Runtime NameError, caught by pytest.
#
# fix3:
#   1. Scan ALL tests/test_*.py via assert_test_imports_defined;
#      print report.
#   2. Add `import ast` module-level to test_patch_utils.py.
#   3. If other files also fail → print + abort (follow-up fix).
#   4. Add meta-guard `test_all_tests_have_bindings` — runs helper
#      over every tests/test_*.py; prevents recurrence.
#
# Run: uv run python scripts/patch_v096e2_fix3.py

from __future__ import annotations
import ast
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import (
    assert_compiles, commit_with_guards,
    assert_test_imports_defined,
)


# ============ 1. scan ============

def scan_all_tests():
    print("\n=== scan all tests/test_*.py via assert_test_imports_defined ===")
    failures: list[tuple[str, str]] = []
    for py in sorted(Path("tests").glob("test_*.py")):
        src = py.read_text()
        try:
            assert_test_imports_defined(src, fn_name=py.name)
        except AssertionError as e:
            failures.append((py.name, str(e)))
    if not failures:
        print("  all clean — no missing bare-Call bindings")
    else:
        print(f"  {len(failures)} file(s) with issues:")
        for name, msg in failures:
            # Compact: just file + names line
            print(f"    {name}: {msg.split(': ', 1)[-1]}")
    return failures


# ============ 2. fix test_patch_utils.py: add import ast ============

def fix_test_patch_utils_ast_import():
    print("\n=== test_patch_utils.py: add `import ast` module-level ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "\nimport ast\n" in s or s.startswith("import ast\n"):
        print("  already imported — skip")
        return

    # Anchor: first top-level import line (the file starts with
    # `"""v0.9.1a: ..."""` docstring then `import pytest`).
    # Add `import ast` before `import pytest` at module level.
    old = "import pytest\n"
    idx = s.find(old)
    if idx < 0:
        raise SystemExit("'import pytest' anchor not found at top level")
    # Check nothing before (like another stdlib import) that would
    # make this weird.
    new = "import ast\nimport pytest\n"
    s = s[:idx] + new + s[idx + len(old):]
    p.write_text(s)
    assert_compiles(p)
    print("  added `import ast`")


# ============ 3. add meta-guard test ============

META_GUARD_LINES = [
    "",
    "",
    "# ============ v0.9.6e2-fix3: meta-guard — all tests bound ============",
    "",
    "",
    "def test_all_tests_have_bindings():",
    '    """Meta-guard (§13.4/192/196): every tests/test_*.py has all',
    '    bare-Call targets bound (import / local / module / builtin).',
    "",
    "    Runs assert_test_imports_defined over every test file. Catches",
    "    future regressions of the class that produced v0.9.6e2-fix3",
    "    (ast missing in test_parse_method_source_indented — helper's",
    "    own bug class, in the helper's own test file).",
    '    """',
    "    from cadw.patch_utils import assert_test_imports_defined",
    "    from pathlib import Path as _P",
    "    root = _P(__file__).parent",
    "    errors: list[str] = []",
    "    for py in sorted(root.glob('test_*.py')):",
    "        try:",
    "            assert_test_imports_defined(",
    "                py.read_text(), fn_name=py.name,",
    "            )",
    "        except AssertionError as e:",
    "            errors.append(str(e))",
    "    assert not errors, (",
    "        'bare-Call target(s) not bound in test file(s):\\n'",
    "        + '\\n'.join(errors)",
    "    )",
]


def add_meta_guard():
    print("\n=== test_patch_utils.py: add meta-guard ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_all_tests_have_bindings" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(META_GUARD_LINES))
    assert_compiles(p)
    print("  OK")


# ============ 4. re-scan + fail-if-remaining ============

def rescan_and_verify():
    print("\n=== re-scan after fixes ===")
    failures = scan_all_tests()
    if failures:
        print("\n*** remaining failures — fix4 needed for:***")
        for name, msg in failures:
            print(f"  {name}: {msg.split(': ', 1)[-1]}")
        print("\n(Not committing; next patch fixes the list above.)")
        sys.exit(1)
    print("  all test files clean")


# ============ main ============

def main():
    print("=== v0.9.6e2-fix3 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    # 1. Scan BEFORE — for diagnostic.
    scan_all_tests()

    # 2. Fix the known case: test_patch_utils.py (ast missing).
    fix_test_patch_utils_ast_import()

    # 3. Re-scan; abort if anything else (follow-up fix will handle).
    rescan_and_verify()

    # 4. Add meta-guard.
    add_meta_guard()

    # 5. Run pytest guards.
    print("\n=== pytest: patch_utils + guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PATCH_UTILS/GUARDS FAILED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    commit_with_guards(
        "v0.9.6e2-fix3: test_parse_method_source_indented — `import "
        "ast` missing. Same class the helper catches (§13.4/192), "
        "in the helper's own test file (§13.4/178 repeat: scaffold "
        "needs its own protection; fix2 dogfood excluded "
        "test_patch_utils.py). "
        "Fixes: (1) add `import ast` module-level to "
        "test_patch_utils.py; (2) new meta-guard test_all_tests_have_"
        "bindings — runs assert_test_imports_defined over EVERY "
        "tests/test_*.py, prevents recurrence; (3) fix3 scanned all "
        "tests first, aborts if other failures remain (follow-up fix "
        "would handle). v0.9.6e2 complete.",
        self_path="scripts/patch_v096e2_fix3.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6e2 complete (fix1 + fix2 + fix3).")


if __name__ == "__main__":
    main()
