#!/usr/bin/env python3
# v0.9.25j-fix2 — duplicated nullable=False → ArrowInvalid at runtime.
#
# STATE: HEAD=v0.9.25i. v0.9.25j methods applied to dataframe.py, tests
# created. 10 failed / 134 passed (targeted):
#   - 9 × duplicated tests: ArrowInvalid "Tried to write record batch
#     with different schema"
#   - 1 × test_cut_invalid_bins_list: regex mismatch
#
# BUG 1 (production, §13.4/169 + /272): `duplicated` method's
# `out_schema` has `pa.field("duplicated", pa.bool_(), nullable=False)`.
# Dataset.from_batches at dataset.py:110:
#   single = pa.RecordBatch.from_arrays([rb.column(field.name)],
#                                       names=[field.name])
# `from_arrays(names=...)` without schema ALWAYS produces nullable=True
# on that field. Writer initialized with nullable=False → batch's
# nullable=True → mismatch → ArrowInvalid.
#
# §13.4/169 convention (documented in cv.py::_attach_column): ALL
# output fields routed through Dataset.from_batches MUST be
# nullable=True. `cut`/`qcut` already do this for `_bin`. `duplicated`
# was the only new column with nullable=False — my mistake.
#
# BUG 2 (test): production message "...must have >= 2 edges...",
# test regex "at least 2". re.search mismatch. Consistency: my other
# errors use >= (`k must be int >= 1`). Fix regex to ">= 2".
#
# fix1 verification false-positive (documented, not fixed here):
# fix1's `verify_inner_compiles` did `compile(textwrap.dedent(
# METHODS_BLOCK))` and failed on trailing `def __getitem__(self, key):`
# — that line's BODY comes from the original code after the anchor,
# not from METHODS_BLOCK. So METHODS_BLOCK alone is intentionally
# incomplete. Lesson: inner-compile checks on anchor-replacement blocks
# (with trailing incomplete def) must append a sentinel body or check
# only the CHANGED region, not the whole block. fix1 was not blocking
# (user ran both commands; v0.9.25j applied anyway).
#
# §13.4/264 (REPL-first): verify_state does 6 checks:
#   (1) HEAD subject-prefix == v0.9.25i (§13.4/270)
#   (2) dataframe.py has new methods (v0.9.25j uncommitted)
#   (3) `nullable=False` for duplicated present (bug marker)
#   (4) `nullable=True` for duplicated ABSENT (fix not yet applied)
#   (5) test regex "at least 2" present (bug marker)
#   (6) test regex ">= 2" ABSENT (fix not yet applied)
#   (7) REPL: mini-repro — Dataset.from_batches with nullable=False
#       fails; with nullable=True succeeds. Confirms §13.4/169 + /272.
#
# §13.4/266 (anchor check): two disjoint anchors — production snippet
# unique to `duplicated`; test block unique to `test_cut_invalid_bins_list`.
#
# §13.4/267 (mirror-check): `cut`/`qcut` already use nullable=True for
# their `_bin` field. Only `duplicated` was wrong — grep for
# `nullable=False` in dataframe.py returns 1 hit (bug site). Post-fix:
# 0 hits.
#
# §13.4/268/269/271 preserved: no bfill fills all (unchanged), no new
# to_pylist low_factor sites, f-strings single-literal.
#
# §13.4/260/261/262/265 preserved: no subprocess dogfood, no symbol
# moves, typed scalars, no copy-pasted asserts.
#
# Usage:
#   uv run python scripts/patch_v0925j_fix2_nullable.py

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
    commit_with_guards,
)


DATAFRAME_PATH = "cadw/dataframe.py"
TESTS_PATH = "tests/test_v0925j.py"


# ============ 0. verify state (REPL-first, §13.4/264 + /272) ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25i; v0.9.25j uncommitted; "
          "nullable=False + test regex both present ===")

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

    # (2) methods present
    df_src = Path(DATAFRAME_PATH).read_text()
    for marker in ("def cut(self", "def qcut(self", "def duplicated(self"):
        if marker not in df_src:
            raise SystemExit(
                f"UNEXPECTED: {marker!r} missing — v0.9.25j didn't "
                f"apply? Confirm apply output."
            )
    print("  (2) cut/qcut/duplicated methods present ✓")

    # (3) bug marker present: nullable=False for duplicated
    if 'pa.field("duplicated", pa.bool_(), nullable=False)' not in df_src:
        raise SystemExit(
            'BUG MARKER MISSING: `pa.field("duplicated", pa.bool_(), '
            'nullable=False)` not found — fix2 misplaced or already '
            'applied.'
        )
    print('  (3) nullable=False bug marker for "duplicated" present ✓')

    # (4) fixed form absent
    if 'pa.field("duplicated", pa.bool_(), nullable=True)' in df_src:
        raise SystemExit(
            "FIX ALREADY APPLIED: nullable=True for duplicated present."
        )
    print("  (4) nullable=True absent (fix not yet applied) ✓")

    # (5) test regex bug marker
    tests_src = Path(TESTS_PATH).read_text()
    if 'match="at least 2"' not in tests_src:
        raise SystemExit(
            'TEST BUG MARKER MISSING: `match="at least 2"` not found '
            '— already fixed or absent.'
        )
    print('  (5) test regex "at least 2" present ✓')

    # (6) fixed test regex absent
    if 'match=">= 2"' in tests_src:
        raise SystemExit("FIX ALREADY APPLIED: `match=\">= 2\"` present.")
    print('  (6) test regex ">= 2" absent (fix not yet applied) ✓')

    # (7) REPL mini-repro: Dataset.from_batches nullable=False fails,
    #     nullable=True succeeds.
    import tempfile
    import pyarrow as pa
    from cadw.dataset import Dataset

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # Case A: nullable=False on `dup` → should raise ArrowInvalid
        bad_schema = pa.schema([
            pa.field("x", pa.int64(), nullable=True),
            pa.field("dup", pa.bool_(), nullable=False),
        ])
        def bad_iter():
            yield pa.RecordBatch.from_arrays(
                [pa.array([1, 2]), pa.array([False, True])],
                schema=bad_schema,
            )
        bad_raised = False
        try:
            Dataset.from_batches(bad_schema, bad_iter(), td / "bad")
        except (pa.lib.ArrowInvalid, pa.lib.ArrowTypeError) as e:
            bad_raised = True
            print(f"  (7a) nullable=False → ArrowInvalid: "
                  f"{str(e)[:80]}... ✓")
        if not bad_raised:
            raise SystemExit(
                "REPL: nullable=False should have raised ArrowInvalid "
                "in Dataset.from_batches but didn't. PyArrow version "
                "may have changed behavior."
            )

        # Case B: nullable=True → succeeds
        good_schema = pa.schema([
            pa.field("x", pa.int64(), nullable=True),
            pa.field("dup", pa.bool_(), nullable=True),
        ])
        def good_iter():
            yield pa.RecordBatch.from_arrays(
                [pa.array([1, 2]), pa.array([False, True])],
                schema=good_schema,
            )
        ds = Dataset.from_batches(good_schema, good_iter(), td / "good")
        if ds.total_rows != 2:
            raise SystemExit(
                f"REPL: nullable=True dataset total_rows={ds.total_rows}, "
                f"expected 2"
            )
        print("  (7b) nullable=True → Dataset.from_batches OK ✓")

    print("  state + REPL verified")


# ============ 1. production fix: nullable=True for duplicated ============

NULLABLE_OLD = '''        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field("duplicated", pa.bool_(), nullable=False))
        out_schema = pa.schema(out_fields)'''


NULLABLE_NEW = '''        # v0.9.25j-fix2 (§13.4/169 + /272): nullable=True, not False.
        # Dataset.from_batches singleton rebuild at dataset.py:110 uses
        # pa.RecordBatch.from_arrays([col], names=[name]) — always
        # nullable=True. Any out_schema field with nullable=False
        # → writer schema mismatch → ArrowInvalid at runtime (NOT
        # caught by assert_compiles). Same convention as cv.py::
        # _attach_column and every other method that appends new
        # columns (pct_change, ffill, bfill, cut, qcut).
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field("duplicated", pa.bool_(), nullable=True))
        out_schema = pa.schema(out_fields)'''


def fix_nullable():
    print("\n=== fix duplicated: nullable=False → nullable=True ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, NULLABLE_OLD, NULLABLE_NEW,
        label="duplicated_nullable",
    )
    if applied:
        assert_compiles(DATAFRAME_PATH)
        check_no_invalid_escape(DATAFRAME_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. test fix: regex "at least 2" → ">= 2" ============

TEST_OLD = '''def test_cut_invalid_bins_list():
    df = _mk({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="at least 2"):
        df.cut("x", bins=[5])
    with pytest.raises(ValueError, match="strictly ascending"):
        df.cut("x", bins=[5, 5])
    with pytest.raises(ValueError, match="strictly ascending"):
        df.cut("x", bins=[5, 3])'''


TEST_NEW = '''def test_cut_invalid_bins_list():
    """v0.9.25j-fix2: production message uses `>= 2` for consistency
    with other errors (`k must be int >= 1` etc.). Prior regex
    `at least 2` (test bug, from before production message
    harmonization) didn't match; correct to `>= 2`.
    """
    df = _mk({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match=">= 2"):
        df.cut("x", bins=[5])
    with pytest.raises(ValueError, match="strictly ascending"):
        df.cut("x", bins=[5, 5])
    with pytest.raises(ValueError, match="strictly ascending"):
        df.cut("x", bins=[5, 3])'''


def fix_test():
    print('\n=== fix test_cut_invalid_bins_list: "at least 2" → ">= 2" ===')
    applied = apply_or_fail(
        TESTS_PATH, TEST_OLD, TEST_NEW,
        label="cut_invalid_bins_test",
    )
    if applied:
        assert_compiles(TESTS_PATH)
        check_no_invalid_escape(TESTS_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 3. pytest ============

def run_pytest():
    print("\n=== pre-flight: facade AST guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py",
         "tests/test_dataframe.py::test_facade_has_no_row_loops",
         "tests/test_dataframe_filters.py::test_facade_has_no_as_py_ast",
         "tests/test_dataframe_filters.py::test_facade_has_no_row_loop_ast",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** FACADE AST GUARD FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== pre-flight: diagnostic meta-guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_v0925d_a.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** DIAGNOSTIC META-GUARDS FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: v0925j + related (10 fixed + regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925j.py",
         "tests/test_v0925i.py",
         "tests/test_v0925h.py",
         "tests/test_v0925d_a.py",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
         "tests/test_v0925g.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=600,
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


# ============ 4. main ============

def main():
    print("=== v0.9.25j-fix2: duplicated nullable + test regex ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    fix_nullable()
    fix_test()
    run_pytest()

    commit_with_guards(
        "v0.9.25j: Phase C batch C-3b — cut + qcut + duplicated "
        "+ fix2 (duplicated nullable + test regex). "
        "3 of 20 Pandas P0 features (C-1 v0.9.25g isin/between/"
        "select_dtypes; C-2 v0.9.25h idxmax/idxmin/pct_change; C-3a "
        "v0.9.25i ffill/bfill/reindex; C-3b here). "
        "All pure compositions of existing primitives + Arrow kernels. "
        "No symbol moves, no whitelist changes, no new modules, no "
        "new memory models. "
        "Features: "
        "(1) `CadwDataFrame.cut(column, bins, *, right=True, "
        "include_lowest=True) -> (df, edges)` — bin numeric column. "
        "bins=int → k equal-width bins [min, max]; bins=list[float] "
        "→ explicit ascending edges. Appends `{column}_bin` int32 "
        "nullable. Returns edges list[float]. Semantics: right=True → "
        "(lo, hi] (pandas default); include_lowest=True (CADW default, "
        "DIFFERS from pandas) → min goes to bin 0 (not null), "
        "implemented via np.nextafter(min, -inf). Out-of-range → null; "
        "null input → null. MinMaxOp ReduceOp + np.digitize. "
        "(2) `CadwDataFrame.qcut(column, q, *, right=True, "
        "include_lowest=True) -> (df, edges)` — quantile-based "
        "binning. q=int k → k equal-frequency bins; q=list[float] in "
        "(0,1) → interior boundaries; edges = [min] + quantile(q) + "
        "[max]. Same label/schema/boundary semantics as cut "
        "(mirror-symmetry §13.4/267). Tied quantile edges deduped; "
        "fewer bins than requested, warning. All-same → ValueError. "
        "Uses df.quantile() (external_sort). "
        "(3) `CadwDataFrame.duplicated(subset=None, keep='first') "
        "-> augmented df` — original cols + `duplicated` bool column "
        "(nullable=True after fix2, §13.4/169). keep ∈ {'first', "
        "'last', False}; same pandas semantics. NULL in subset → own "
        "group. Implementation: (a) hash_shuffle + _local_dedup on "
        "(subset, __rid) via existing _ops_drop_duplicates → kept "
        "rid set; (b) per-batch mask = NOT pc.is_in(rid, kept_set) "
        "via _iter_filtered_batches_with_positions. Result order = "
        "filtered view order. Users filter via df2[df2['duplicated'] "
        "== True]. "
        "fix2 (2 bugs from v0.9.25j initial, 10 failed / 134 passed): "
        "(a) PRODUCTION BUG (§13.4/169 + /272): `duplicated` field "
        "declared `nullable=False` in out_schema. Dataset.from_batches "
        "singleton rebuild at dataset.py:110 uses pa.RecordBatch."
        "from_arrays([col], names=[name]) — always nullable=True. "
        "Writer schema (nullable=False) vs batch schema "
        "(nullable=True) → ArrowInvalid 'Tried to write record batch "
        "with different schema'. NOT caught by assert_compiles (runtime "
        "error). Same convention as cv.py::_attach_column: ALL fields "
        "routed through Dataset.from_batches MUST be nullable=True. "
        "cut/qcut already correct (nullable=True for _bin); "
        "duplicated was the only new column with nullable=False. "
        "Fix: nullable=True. "
        "(b) TEST BUG: test_cut_invalid_bins_list regex `at least 2` "
        "vs production message `>= 2`. re.search mismatch. Fix: "
        "match \">= 2\" (consistency with other errors using >=). "
        "fix1 verification false-positive (documented, not fixed): "
        "fix1's `verify_inner_compiles` did compile(dedent("
        "METHODS_BLOCK)) and failed on trailing `def __getitem__("
        "self, key):` — that line's BODY comes from original code "
        "after anchor, not from METHODS_BLOCK. So METHODS_BLOCK alone "
        "is intentionally incomplete. Lesson: inner-compile checks on "
        "anchor-replacement blocks (with trailing incomplete def) "
        "must append a sentinel body or check only the CHANGED "
        "region. fix1 was not blocking — user ran both commands; "
        "v0.9.25j applied anyway. "
        "§13.4/272 (NEW): nullable=False on any Dataset.from_batches-"
        "routed field is runtime ArrowInvalid, not compile error. "
        "Pre-write checklist: new columns for Dataset.from_batches "
        "path MUST be nullable=True (§13.4/169). Verify via mini-"
        "repro in REPL (verify_state #7 here). "
        "§13.4/270 preserved: HEAD check uses subject-prefix regex, "
        "not substring. §13.4/271 preserved: f-string single-literal "
        "(fix1 corrected qcut error message). §13.4/267 preserved: "
        "cut/qcut share label_fn logic; dup keep modes share "
        "_ops_drop_duplicates. §13.4/269 preserved: cut/qcut use "
        "np.digitize (no to_pylist); duplicated uses pc.is_in "
        "(Arrow-native). §13.4/264 (REPL-first): 7 checks before edit "
        "— HEAD regex, methods present, bug markers x2, fixed-forms "
        "absent x2, mini-repro (nullable=False fails, nullable=True "
        "succeeds). All PASS. "
        "Tests (tests/test_v0925j.py): 30 total — cut (12), qcut (9), "
        "duplicated (9). "
        "Fast suite: 2404 → ~2434 (+30). "
        "Next: Phase C C-4 (str extras: str.slice/cat/count).",
        self_path="scripts/patch_v0925j_fix2_nullable.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25j COMPLETE (3 features + fix2). "
          "Phase C: 12/20. Next: C-4 (str extras).")


if __name__ == "__main__":
    main()