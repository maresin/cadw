#!/usr/bin/env python3
# v0.9.25j-fix3 — stale docstring in METHODS_BLOCK + fix2v2 check too coarse.
#
# STATE: HEAD=v0.9.25i. dataframe.py reverted to HEAD (no cut/qcut/
# duplicated). tests/test_v0925j.py removed. Target script
# patch_v0925j_cut_qcut_duplicated.py state:
#   - f-string fix applied (fix1)
#   - pa.field("duplicated", ...) → nullable=True applied (fix2v2)
#   - match=">= 2" applied (fix2v2)
#   - docstring STILL says "(nullable=False)" (STALE + trips fix2v2's
#     coarse check)
#
# BUG: fix2v2's verify_inner_compiles did
#   if "nullable=False" in dedented:
#       raise SystemExit("nullable=False still in METHODS_BLOCK")
# That's a BARE SUBSTRING check on the whole 440-line block. But the
# duplicated method's docstring legitimately contains "nullable=False"
# (documentation of prior state). Field was patched correctly; docstring
# was left stale. Check saw stale docstring → false-failed → fix2v2
# aborted before rerun_v925j.
#
# §13.4/274 (NEW): substring guards on METHOD DOCSTRINGS false-positive
# when docstring legitimately mentions the anti-pattern (example,
# warning, or — this case — stale documentation). Same family as
# §13.4/265 ("wait, X" comment vs stale assert), §13.4/270 ("vX.Y.Z"
# in log vs HEAD), §13.4/271 (f-string brace across literals). Rule:
# match full SYNTAX UNIT (field string, call expression), not bare
# token. Docstrings mention patterns legitimately.
#
# COROLLARY: stale docs are their own bug. When code changes, docstring
# must change too — this patch fixes both problems at once.
#
# FIX:
#   (a) Docstring: `bool column (nullable=False).` → `bool column
#       (nullable=True).` in the duplicated docstring. Both correct
#       documentation AND eliminates the fix2v2 false-positive trigger.
#   (b) verify_inner_compiles in THIS script uses PRECISE check:
#       full field string absent + full field string (nullable=True)
#       present. No bare-token substring check.
#
# §13.4/264 (REPL-first): verify_state does 5 checks:
#   (1) HEAD subject-prefix == v0.9.25i (§13.4/270)
#   (2) dataframe.py clean (no cut/qcut/duplicated) — post-fix2v2
#   (3) script field nullable=True present
#   (4) script test regex ">= 2" present
#   (5) script docstring "bool column (nullable=False)" present (STALE marker)
#
# §13.4/266 (anchor check): unique anchor `bool column (nullable=False).
# Row order` — one occurrence in the duplicated docstring.
#
# §13.4/267 (mirror-check): after docstring fix, all three new methods
# (cut, qcut, duplicated) consistently document nullable=True for
# appended bin/dup columns. grep for "nullable=False" across script
# should return 0 in METHODS_BLOCK.
#
# §13.4/272 (REPL, verified in fix2v2): nullable=False → ArrowInvalid;
# nullable=True → OK. fix3 inherits that fact; still confirms via
# mini-repro for defense.
#
# §13.4/273 (working-tree): fix2v2 reverted before its own abort —
# working tree is clean. fix3 does not need to revert; but fix3
# defensively checks that dataframe.py has no new methods before
# re-running v0.9.25j.
#
# §13.4/260/261/262/265/269/271 preserved: no subprocess dogfood,
# no symbol moves, typed scalars, no copy-pasted asserts, no new
# low_factor sites (fix3 adds no to_pylist), f-strings single-literal.
#
# Usage:
#   uv run python scripts/patch_v0925j_fix3_docstring.py

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


SCRIPT_PATH = "scripts/patch_v0925j_cut_qcut_duplicated.py"
DATAFRAME_PATH = "cadw/dataframe.py"
TESTS_PATH = "tests/test_v0925j.py"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25i; script has field+regex+stale "
          "docstring; dataframe.py clean ===")

    # (1) HEAD subject-prefix (§13.4/270)
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

    # (2) dataframe.py clean (fix2v2 reverted, v0.9.25j not re-applied)
    df_src = Path(DATAFRAME_PATH).read_text()
    for marker in ("def cut(self", "def qcut(self", "def duplicated(self"):
        if marker in df_src:
            raise SystemExit(
                f"UNEXPECTED: {marker!r} in dataframe.py. fix2v2 "
                f"reverted; if present, another process partially "
                f"applied v0.9.25j. Revert: git checkout -- "
                f"{DATAFRAME_PATH}"
            )
    print("  (2) dataframe.py clean (no cut/qcut/duplicated) ✓")

    # (3) script field nullable=True present
    script_src = Path(SCRIPT_PATH).read_text()
    if 'pa.field("duplicated", pa.bool_(), nullable=True)' not in script_src:
        raise SystemExit(
            'field `pa.field("duplicated", pa.bool_(), nullable=True)` '
            'not in script — fix2v2 field patch failed or script '
            'reverted. Re-run patch_v0925j_fix2v2_nullable.py from '
            'clean HEAD.'
        )
    print("  (3) script field nullable=True ✓")

    # (4) script test regex ">= 2" present
    if 'match=">= 2"' not in script_src:
        raise SystemExit(
            'test regex `match=">= 2"` not in script — fix2v2 test '
            'patch failed.'
        )
    print('  (4) script test regex ">= 2" ✓')

    # (5) script docstring stale marker present
    if 'bool column (nullable=False)' not in script_src:
        if 'bool column (nullable=True)' in script_src:
            raise SystemExit(
                "docstring already patched — fix3 misplaced"
            )
        raise SystemExit(
            "stale docstring marker `bool column (nullable=False)` "
            "not found — script structure changed unexpectedly."
        )
    print('  (5) script stale docstring `bool column (nullable=False)` ✓')

    # (6) working tree: tests file removed (fix2v2 did this; defensive)
    tests_p = Path(TESTS_PATH)
    if tests_p.exists():
        print(f"  WARN: {TESTS_PATH} exists — will remove before re-run")
    else:
        print(f"  (6) {TESTS_PATH} absent ✓")

    # (7) REPL confirmation (§13.4/272): full-field-string distinction
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
                "REPL: nullable=False should raise ArrowInvalid"
            )
        except (pa.lib.ArrowInvalid, pa.lib.ArrowTypeError):
            print("  (7) REPL nullable=False → ArrowInvalid ✓")

    print("  state verified")


# ============ 1. fix stale docstring ============

def fix_docstring():
    print("\n=== fix stale docstring: `(nullable=False)` → "
          "`(nullable=True)` ===")
    applied = apply_or_fail(
        SCRIPT_PATH,
        "bool column (nullable=False). Row order",
        "bool column (nullable=True). Row order",
        label="docstring_nullable",
    )
    if applied:
        assert_compiles(SCRIPT_PATH)
        check_no_invalid_escape(SCRIPT_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. verify_inner_compiles (PRECISE) ============

def verify_inner_compiles():
    """v0.9.25j-fix3: PRECISE check. Full field string, not bare
    `nullable=False` (§13.4/274). Docstring may legitimately mention
    `nullable` — that's documentation, not a bug marker.
    """
    print("\n=== inner-compile: METHODS_BLOCK + sentinel `pass` ===")
    import importlib.util
    import textwrap
    spec = importlib.util.spec_from_file_location(
        "_vj_fix3_check", SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    block = mod.METHODS_BLOCK
    dedented = textwrap.dedent(block)

    # Sentinel `pass` so trailing `def __getitem__(self, key):` (whose
    # body lives in the original file, not the block) compiles.
    wrapped = dedented + "\n        pass\n"
    try:
        compile(wrapped, "<METHODS_BLOCK+sentinel>", "exec")
    except SyntaxError as e:
        print(
            f"  STILL BROKEN: {e}\n"
            f"  line {e.lineno}: {e.text!r}"
        )
        raise SystemExit(1)

    # PRECISE field checks — full syntax unit, not bare token
    if 'pa.field("duplicated", pa.bool_(), nullable=False)' in dedented:
        raise SystemExit(
            "FIELD still nullable=False — fix2v2 field patch reverted?"
        )
    if 'pa.field("duplicated", pa.bool_(), nullable=True)' not in dedented:
        raise SystemExit(
            "FIELD not nullable=True — fix2v2 field patch missing."
        )

    # WARN-only: bare token occurrence (docstring mention, comment).
    # fix3's docstring patch should have eliminated it, but if there's
    # a future comment mentioning `nullable=False`, only WARN (doc is
    # not a bug).
    if "nullable=False" in dedented:
        for i, line in enumerate(dedented.splitlines(), 1):
            if "nullable=False" in line:
                print(
                    f"  WARN: bare 'nullable=False' at line {i}: "
                    f"{line.strip()}"
                )
        print(
            "  (not fatal — only full field string is authoritative "
            "per §13.4/274)"
        )
    else:
        print("  no bare 'nullable=False' in block ✓")

    print(f"  compiles clean ({len(dedented.splitlines())} lines) ✓")
    print("  field nullable=True present, nullable=False absent ✓")


# ============ 3. re-run v0.9.25j ============

def rerun_v925j():
    print("\n=== re-run v0.9.25j (fresh apply + pytest + commit) ===")
    tests_p = Path(TESTS_PATH)
    if tests_p.exists():
        tests_p.unlink()
        print(f"  removed {TESTS_PATH} (fresh apply)")

    r = subprocess.run(
        ["uv", "run", "python", SCRIPT_PATH],
        timeout=2400,
    )
    if r.returncode != 0:
        print(
            f"\n*** v0.9.25j re-run failed (rc={r.returncode}) — see "
            f"output above. ***"
        )
        sys.exit(r.returncode)
    print("\n  v0.9.25j re-run: SUCCESS")


# ============ main ============

def main():
    print("=== v0.9.25j-fix3: stale docstring + precise check + re-run ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    fix_docstring()
    verify_inner_compiles()
    rerun_v925j()


if __name__ == "__main__":
    main()