#!/usr/bin/env python3
# v0.9.25j-fix2v2 — state recovery + script patch + re-run.
#
# SITUATION: v0.9.25j applied methods+tests to working tree, pytest
# failed 10/144, commit_with_guards not reached → no commit. Then
# working tree was reverted (cause unknown: git reset/checkout/stash/
# IDE). HEAD still v0.9.25i. fix2's verify_state check #2 correctly
# failed with "def cut missing". Data is gone; script is the source
# of truth.
#
# §13.4/273 (NEW): failed patches must leave a recovery path.
#   apply_or_fail is idempotent ONLY when anchor is unique in file.
#   v0.9.25j's anchor `# __getitem__` appears as SUFFIX of
#   METHODS_BLOCK after first apply → re-run without revert DUPLICATES
#   the block (would then hit "def cut already in file" downstream).
#   Correct: git checkout -- file before re-run. multi-block inserts
#   are NOT natively idempotent; explicit revert required.
#
# STRATEGY: patch the SCRIPT (patch_v0925j_cut_qcut_duplicated.py)
# constants METHOD_BLOCK and NEW_TESTS inline. Then revert working
# tree and re-run the script. Produces correct methods+tests in one
# clean apply.
#
# §13.4/272 (REPL): reproduced — nullable=False on Dataset.from_batches
# field → ArrowInvalid; nullable=True → OK. Verify #6.
#
# §13.4/264 (REPL-first): 6 checks + mini-repro.
# §13.4/270: HEAD subject-prefix, not substring.
# §13.4/271: f-string single-literal (fix1 applied).
# §13.4/267: mirror — cut/qcut already nullable=True; only
#   duplicated was wrong.
# §13.4/271-fixup: fix1's inner-compile false-positive (§13.4/271
#   companion) — patch METHODS_BLOCK has trailing `def __getitem__`
#   with no body (body comes from original file). Append `pass`
#   sentinel for compile check only.
#
# Usage: uv run python scripts/patch_v0925j_fix2v2_nullable.py

from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, assert_compiles, check_no_invalid_escape,
)


DATAFRAME_PATH = "cadw/dataframe.py"
TESTS_PATH = "tests/test_v0925j.py"
SCRIPT_PATH = "scripts/patch_v0925j_cut_qcut_duplicated.py"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25i; script has bug markers; "
          "working tree state to be recovered ===")

    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    m = re.match(r'^[a-f0-9]+ (v\d+\.\d+\.\d+\w*):', log)
    head_version = m.group(1) if m else None
    if head_version != "v0.9.25i":
        raise SystemExit(
            f"expected HEAD version v0.9.25i; got {head_version!r}"
        )
    print(f"  (1) HEAD subject-prefix = v0.9.25i ✓")

    status = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True, text=True,
    ).stdout
    print(f"  git status --short:\n{status if status.strip() else '    (clean)'}")

    df_src = Path(DATAFRAME_PATH).read_text()
    has_cut = "def cut(self" in df_src
    has_nf = 'pa.field("duplicated", pa.bool_(), nullable=False)' in df_src
    print(f"  (2) dataframe.py: has_cut={has_cut} "
          f"has_nullable_false={has_nf} (recovery handles either) ✓")

    script_src = Path(SCRIPT_PATH).read_text()
    if "_first_or_empty" not in script_src:
        raise SystemExit(
            f"script missing f-string fix (_first_or_empty). "
            f"Run patch_v0925j_fix1_fstring.py first."
        )
    print("  (3) script has f-string fix ✓")

    if 'pa.field("duplicated", pa.bool_(), nullable=False)' not in script_src:
        raise SystemExit(
            'SCRIPT BUG MARKER MISSING: nullable=False not found in '
            'script — script already patched (nullable fix applied).'
        )
    print("  (4) script METHODS_BLOCK has nullable=False ✓")

    if 'match="at least 2"' not in script_src:
        raise SystemExit(
            'SCRIPT TEST BUG MARKER MISSING: "at least 2" not found in '
            'script — script already patched.'
        )
    print("  (5) script NEW_TESTS has 'at least 2' ✓")

    import tempfile
    import pyarrow as pa
    from cadw.dataset import Dataset

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bad_schema = pa.schema([
            pa.field("x", pa.int64(), nullable=True),
            pa.field("dup", pa.bool_(), nullable=False),
        ])
        def bad_iter():
            yield pa.RecordBatch.from_arrays(
                [pa.array([1, 2]), pa.array([False, True])],
                schema=bad_schema,
            )
        try:
            Dataset.from_batches(bad_schema, bad_iter(), td / "bad")
            raise SystemExit(
                "REPL: nullable=False should have raised ArrowInvalid."
            )
        except (pa.lib.ArrowInvalid, pa.lib.ArrowTypeError):
            print("  (6) REPL nullable=False → ArrowInvalid ✓")

    print("  state verified")


# ============ 1. revert working tree ============

def revert_working_tree():
    print("\n=== revert working tree (dataframe.py tracked, "
          "test file untracked) ===")
    r = subprocess.run(
        ["git", "checkout", "--", DATAFRAME_PATH],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        print(f"  reverted {DATAFRAME_PATH} → HEAD (v0.9.25i)")
    else:
        print(f"  git checkout {DATAFRAME_PATH}: "
              f"rc={r.returncode} stderr={r.stderr.strip()[:120]}")

    tests_p = Path(TESTS_PATH)
    if tests_p.exists():
        tests_p.unlink()
        print(f"  removed {TESTS_PATH}")
    else:
        print(f"  {TESTS_PATH} not present (noop)")


# ============ 2. patch script METHODS_BLOCK: nullable ============

def patch_script_nullable():
    print("\n=== patch script METHODS_BLOCK: nullable=False → "
          "nullable=True ===")
    applied = apply_or_fail(
        SCRIPT_PATH,
        'pa.field("duplicated", pa.bool_(), nullable=False)',
        'pa.field("duplicated", pa.bool_(), nullable=True)',
        label="script_nullable",
    )
    if applied:
        assert_compiles(SCRIPT_PATH)
        check_no_invalid_escape(SCRIPT_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 3. patch script NEW_TESTS: regex ============

def patch_script_test_regex():
    print("\n=== patch script NEW_TESTS: 'at least 2' → '>= 2' ===")
    applied = apply_or_fail(
        SCRIPT_PATH,
        'match="at least 2"',
        'match=">= 2"',
        label="script_test_regex",
    )
    if applied:
        assert_compiles(SCRIPT_PATH)
        check_no_invalid_escape(SCRIPT_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 4. verify_inner_compiles with sentinel ============

def verify_inner_compiles():
    """Compile METHODS_BLOCK from patched script; append `pass`
    sentinel to satisfy trailing `def __getitem__` (body comes from
    original file, not METHODS_BLOCK)."""
    print("\n=== inner-compile: METHODS_BLOCK + sentinel `pass` ===")
    import importlib.util
    import textwrap
    spec = importlib.util.spec_from_file_location(
        "_vj_fix2v2_check", SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    block = mod.METHODS_BLOCK
    dedented = textwrap.dedent(block)
    wrapped = dedented + "\n        pass\n"
    try:
        compile(wrapped, "<METHODS_BLOCK+sentinel>", "exec")
    except SyntaxError as e:
        print(
            f"  STILL BROKEN after sentinel: {e}\n"
            f"  line {e.lineno}: {e.text!r}"
        )
        raise SystemExit(1)

    if "nullable=False" in dedented:
        raise SystemExit(
            "nullable=False still in METHODS_BLOCK — patch failed"
        )
    if "nullable=True" not in dedented:
        raise SystemExit(
            "nullable=True not in METHODS_BLOCK — patch failed"
        )
    print(f"  compiles clean ({len(dedented.splitlines())} lines) ✓")
    print("  nullable=True present, nullable=False absent ✓")


# ============ 5. re-run v0.9.25j ============

def rerun_v925j():
    print("\n=== re-run v0.9.25j (fresh apply + pytest + commit) ===")
    r = subprocess.run(
        ["uv", "run", "python", SCRIPT_PATH],
        timeout=2400,
    )
    if r.returncode != 0:
        print(
            f"\n*** v0.9.25j re-run failed (rc={r.returncode}). "
            f"See output above. ***"
        )
        sys.exit(r.returncode)
    print("\n  v0.9.25j re-run: SUCCESS (methods + tests + commit)")


# ============ main ============

def main():
    print("=== v0.9.25j-fix2v2: state recovery + script patch + re-run ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    revert_working_tree()
    patch_script_nullable()
    patch_script_test_regex()
    verify_inner_compiles()
    rerun_v925j()


if __name__ == "__main__":
    main()