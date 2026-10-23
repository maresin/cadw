#!/usr/bin/env python3
# v0.9.26w-fix1: test_helper_is_defined inspects _helpers not _core.
# §13.4/327: discovery pattern must match actual test code form.
# Bootstrap-safe.

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

NEW_MOD = "cadw/dataframe/_helpers.py"
CORE = "cadw/dataframe/_core.py"
TEST_PATH = "tests/test_filter_compat.py"


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26v":
        raise SystemExit(f"expected HEAD=v0.9.26v, got {head!r}")
    print("HEAD: " + head)

    # _helpers.py present with all 5 symbols
    if not Path(NEW_MOD).exists():
        raise SystemExit(f"{NEW_MOD} missing")
    mod_src = Path(NEW_MOD).read_text()
    for name in ("_apply_filter_with_positions", "_detect_mode",
                 "_normalize_scalar", "_infer_arrow_type_from_value",
                 "_build_out_schema"):
        if f"def {name}(" not in mod_src:
            raise SystemExit(f"{name} not in {NEW_MOD}")
    ast.parse(mod_src, filename=NEW_MOD)
    print(f"  {NEW_MOD}: OK ({len(mod_src.splitlines())} lines)")

    # _core.py has re-export
    core_src = Path(CORE).read_text()
    if "from ._helpers import" not in core_src:
        raise SystemExit("_core.py missing _helpers re-export")
    print(f"  {CORE}: re-export OK ({len(core_src.splitlines())} lines)")

    # Test still broken
    test_src = Path(TEST_PATH).read_text()
    if "inspect.getsource(df_mod._core)" not in test_src:
        raise SystemExit("test already fixed — inspect manually")
    if "inspect.getsource(df_mod._helpers)" in test_src:
        raise SystemExit("test already references _helpers")
    print(f"  {TEST_PATH}: needs fix")


def fix_test():
    print("\n--- fix test_filter_compat.py ---")
    p = Path(TEST_PATH)
    src = p.read_text()

    old = "inspect.getsource(df_mod._core)"
    new = "inspect.getsource(df_mod._helpers)"

    if new in src:
        print("  already fixed")
        return 0
    if old not in src:
        raise SystemExit(f"anchor {old!r} not found")

    src = src.replace(old, new)
    p.write_text(src)
    ast.parse(src, filename=TEST_PATH)
    print(f"  {old!r} -> {new!r}")
    return 1


def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26w-fix1: test target ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()
    n_fix = fix_test()
    run_tests()

    print("\n--- verify_whitelist ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.patch_utils import verify_whitelist; "
         "verify_whitelist(raise_on_orphan=True); print('WL OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("whitelist failed")
    print(r.stdout.strip())

    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26w: extract _helpers (B.1 step 17) + fix1 (test). "
        "_apply_filter_with_positions/_detect_mode/_normalize_scalar/"
        "_infer_arrow_type_from_value/_build_out_schema (~142 LOC, "
        "5 helpers) moved verbatim from _core.py to _helpers.py. "
        "_core.py re-exports (§13.4/325). fix1 (§13.4/327): "
        "test_helper_is_defined_not_just_called inspects "
        "df_mod._helpers (not _core) — actual pattern was "
        "`from cadw import dataframe; df_mod._core` (my v0.9.26w "
        "anchor used different form). _core.py 853 → 711 LOC. "
        "0 logic change; tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26w COMPLETE (fix1)")


if __name__ == "__main__":
    main()