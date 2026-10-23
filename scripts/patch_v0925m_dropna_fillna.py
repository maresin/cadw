#!/usr/bin/env python3
# v0.9.25m — Phase C batch C-5 (FINAL): dropna + fillna.
#
# 2 surface API features. Closes Phase C at 20/20 Pandas P0.
# Pure compositions of existing primitives (pc.is_valid/or_, pc.fill_null,
# batch filter, Dataset.from_batches). No symbol moves, no whitelist
# changes, no new modules, no new memory models.
#
# Completes the null-handling story:
#   - CadwDataFrame.ffill()  (v0.9.25i) — method-based forward fill
#   - CadwDataFrame.bfill()  (v0.9.25i) — method-based backward fill
#   - CadwDataFrame.fillna(value)      (v0.9.25m) — value-based fill
#   - CadwDataFrame.dropna()           (v0.9.25m) — filter null rows
#
# FEATURES:
#   1. `CadwDataFrame.dropna(subset=None, how='any')` — drop rows
#      containing nulls. subset default = all columns. how='any'
#      drops rows with ANY null in subset; how='all' drops rows
#      where ALL subset cols are null. Streaming batch filter.
#   2. `CadwDataFrame.fillna(value, columns=None)` — replace nulls
#      with value. value is scalar (coerced per column type) OR
#      dict {col: scalar}. columns restricts scope. Type-coerced
#      upfront with clear TypeError. Streaming O(batch), no state.
#      For method-based fill, use .ffill()/.bfill().
#
# §13.4/264 APPLIED (REPL-first, 10 checks):
#   verify_state BEFORE edits:
#     (1) pc.is_valid([1,None,2]) == [T,F,T]
#     (2) pc.is_null([1,None,2]) == [F,T,F]
#     (3) pc.and_ combines bool arrays → [T,F,F]
#     (4) pc.or_ combines bool arrays → [T,T,F]
#     (5) pc.fill_null(int64 [1,None], scalar(0, int64)) == [1,0]
#     (6) pc.fill_null(float64 [1.5,None], scalar(0.0, f64)) == [1.5,0.0]
#     (7) pc.fill_null(string ["a",None], scalar("?", string)) ==
#         ["a","?"]
#     (8) pc.fill_null(bool [True,None], scalar(False, bool)) ==
#         [True,False]
#     (9) pa.scalar(0, type=pa.string()) — capture ACTUAL exception
#         class (not guess) → for type-coercion error message
#     (10) RecordBatch.filter(bool_mask) preserves schema
#   All must pass; abort pre-mutation if any differs from assumption.
#
# §13.4/278 (hasattr): presence of reindex/ffill/bfill via hasattr on
#   CadwDataFrame — source-text markers avoided (multi-line sig risk).
#
# §13.4/266 (re-verify after anchor): verify_state asserts exact
#   3-line anchor text before apply; if missing → abort.
#
# §13.4/260 preserved: no subprocess dogfood (dedent trap). pytest
#   authoritative.
# §13.4/261 preserved: no symbol moves → no whitelist path updates.
# §13.4/262 preserved: pa.scalar(value, type=col_type) for typed
#   scalar; pandas-parity semantics.
# §13.4/265 preserved: no copy-pasted asserts from sibling tests —
#   dropna/fillna have distinct data & assertions.
#
# DEFERRED:
#   - dropna(thresh=int) — row has ≥thresh non-null (pandas P2)
#   - fillna(method=) — use .ffill()/.bfill() instead
#   - fillna(limit=) — bounded-window fill; ffill/bfill limit= also
#     deferred (both v0.9.26 backlog)
#   - dropna on MultiIndex source — works naturally (batch.filter
#     drops rows regardless of index); no special path needed
#
# Fix-cycle budget: 0-1 (2 features, independent, simplest in Phase C,
# REPL-first applied, hasattr verify, anchor re-checked).
#
# Run: uv run python scripts/patch_v0925m_dropna_fillna.py

from __future__ import annotations
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


DATAFRAME_PATH = "cadw/dataframe.py"
TESTS_PATH = "tests/test_v0925m.py"


# ============ 0. verify state (REPL-first, §13.4/264) ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25l; 10 REPL checks; hasattr "
          "presence; exact anchor text ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25l" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25l; got {log[:200]!r}"
        )
    if "v0.9.25m" in log:
        raise SystemExit("v0.9.25m already committed — patch misplaced")

    # §13.4/278: presence via hasattr, not source text
    from cadw import CadwDataFrame
    for m in ("reindex", "ffill", "bfill"):
        if not hasattr(CadwDataFrame, m):
            raise SystemExit(
                f"CadwDataFrame.{m} missing — prior patch didn't apply?"
            )
    for m in ("dropna", "fillna"):
        if hasattr(CadwDataFrame, m):
            raise SystemExit(
                f"CadwDataFrame.{m} already present — v0.9.25m applied?"
            )

    # §13.4/266: exact anchor text (3 lines)
    src = Path(DATAFRAME_PATH).read_text()
    anchor = (
        "    # ---------------- __getitem__ ----------------\n"
        "\n"
        "    def __getitem__(self, key):"
    )
    if anchor not in src:
        raise SystemExit(
            "ANCHOR MISS: `# ---------------- __getitem__ "
            "----------------\\n\\n    def __getitem__(self, key):` "
            "not found. Inspect dataframe.py — anchor assumption may "
            "be wrong. Do NOT guess a replacement anchor; abort and "
            "inspect (§13.4/266)."
        )

    # --- REPL checks ---
    import pyarrow as pa
    import pyarrow.compute as _pc

    # (1) is_valid
    r = _pc.is_valid(
        pa.array([1, None, 2], type=pa.int64())
    ).to_pylist()
    if r != [True, False, True]:
        raise SystemExit(f"is_valid: {r}, expected [T,F,T]")

    # (2) is_null
    r = _pc.is_null(
        pa.array([1, None, 2], type=pa.int64())
    ).to_pylist()
    if r != [False, True, False]:
        raise SystemExit(f"is_null: {r}, expected [F,T,F]")

    # (3) and_
    r = _pc.and_(
        pa.array([True, False, True]),
        pa.array([True, True, False]),
    ).to_pylist()
    if r != [True, False, False]:
        raise SystemExit(f"and_: {r}, expected [T,F,F]")

    # (4) or_
    r = _pc.or_(
        pa.array([True, False, True]),
        pa.array([True, True, False]),
    ).to_pylist()
    if r != [True, True, True]:
        raise SystemExit(f"or_: {r}, expected [T,T,T]")

    # (5) fill_null int64
    r = _pc.fill_null(
        pa.array([1, None], type=pa.int64()),
        pa.scalar(0, type=pa.int64()),
    ).to_pylist()
    if r != [1, 0]:
        raise SystemExit(f"fill_null int: {r}, expected [1,0]")

    # (6) fill_null float64
    r = _pc.fill_null(
        pa.array([1.5, None], type=pa.float64()),
        pa.scalar(0.0, type=pa.float64()),
    ).to_pylist()
    if r != [1.5, 0.0]:
        raise SystemExit(f"fill_null float: {r}, expected [1.5,0.0]")

    # (7) fill_null string
    r = _pc.fill_null(
        pa.array(["a", None], type=pa.string()),
        pa.scalar("?", type=pa.string()),
    ).to_pylist()
    if r != ["a", "?"]:
        raise SystemExit(f"fill_null string: {r}, expected ['a','?']")

    # (8) fill_null bool
    r = _pc.fill_null(
        pa.array([True, None], type=pa.bool_()),
        pa.scalar(False, type=pa.bool_()),
    ).to_pylist()
    if r != [True, False]:
        raise SystemExit(f"fill_null bool: {r}, expected [T,F]")

    # (9) pa.scalar(0, type=pa.string()) — CAPTURE exception class
    #     (do NOT guess; §13.4/264 formalized this)
    try:
        _ = pa.scalar(0, type=pa.string())
    except Exception as e:
        type_mismatch_exc = type(e)
        if not issubclass(type_mismatch_exc, Exception):
            raise SystemExit(
                f"pa.scalar(int, string) raised non-Exception: "
                f"{type_mismatch_exc}"
            )
        # Common classes: pa.ArrowInvalid, pa.ArrowTypeError,
        # pa.ArrowNotImplementedError. All are pa.ArrowException
        # subclasses. Assert the message references type mismatch.
        msg = str(e).lower()
        if not any(
            k in msg for k in ("type", "convert", "cast", "invalid")
        ):
            raise SystemExit(
                f"pa.scalar(int, string) message {msg!r} — unexpected; "
                f"fillna type-coercion error may not fire correctly"
            )
    else:
        raise SystemExit(
            "pa.scalar(0, type=pa.string()) unexpectedly succeeded — "
            "fillna type validation would silently corrupt string "
            "columns with int value. Abort."
        )

    # (10) RecordBatch.filter(bool_mask) preserves schema
    b = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3], type=pa.int64())], names=["x"],
    )
    mask = pa.array([True, False, True])
    filtered = b.filter(mask)
    if filtered.schema != b.schema:
        raise SystemExit(
            f"RecordBatch.filter changed schema: {b.schema} → "
            f"{filtered.schema}"
        )
    if filtered.column("x").to_pylist() != [1, 3]:
        raise SystemExit(
            f"RecordBatch.filter([T,F,T]) = "
            f"{filtered.column('x').to_pylist()}, expected [1, 3]"
        )

    print("  10 REPL checks passed; state + anchor verified")


# ============ 1. add dropna + fillna (before __getitem__) ============

ANCHOR_OLD = '''    # ---------------- __getitem__ ----------------

    def __getitem__(self, key):'''


ANCHOR_NEW = '''    def dropna(self, subset=None, how: str = "any") -> "CadwDataFrame":
        """Drop rows containing null values (v0.9.25m).

        Streaming batch filter. Composes pc.is_valid + pc.and_/or_
        + RecordBatch.filter. O(N) I/O, O(batch) RAM.

        - subset: None → all columns; str → one; Sequence[str] →
          subset of columns to inspect for nulls.
        - how: 'any' (default) → drop rows where ANY subset col is
          null; 'all' → drop rows where ALL subset cols are null.
        - Other columns are preserved unchanged (row removal only).
        - Composes with existing row filters (dropna applies to the
          filtered view; visible rows only).

        Note: for ``thresh=N`` (keep rows with >= N non-null) — pandas
        P2 — not implemented; use composite filters.
        """
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset

        if how not in ("any", "all"):
            raise ValueError(
                f"dropna: how must be 'any' or 'all'; got {how!r}"
            )
        if subset is None:
            cols = list(self._columns)
        elif isinstance(subset, str):
            cols = [subset]
        else:
            cols = list(subset)
        if not cols:
            raise ValueError("dropna: subset must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"dropna: column {c!r} not in {self._columns}"
                )

        out_schema = self._dataset.schema

        def _iter():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="dropna.input")
                n = b.num_rows
                if n == 0:
                    continue
                valid_arrays = [
                    _pc.is_valid(b.column(c)) for c in cols
                ]
                mask = valid_arrays[0]
                for v in valid_arrays[1:]:
                    if how == "any":
                        mask = _pc.and_(mask, v)
                    else:  # all
                        mask = _pc.or_(mask, v)
                filtered = b.filter(mask)
                if filtered.num_rows > 0:
                    yield filtered

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def fillna(self, value, columns=None) -> "CadwDataFrame":
        """Replace nulls with a scalar value (v0.9.25m).

        Streaming per-batch via pc.fill_null. O(N) I/O, O(batch) RAM,
        no state carry.

        - value: scalar (int | float | str | bool | date | timestamp
          …) — coerced to each target column's type; raises
          TypeError with clear message if incompatible. OR dict
          {column: scalar} — per-column scalars; columns not in dict
          are unchanged.
        - columns: None → all; str → one; Sequence[str] → subset.
          When value is a dict, columns acts as an additional
          intersection filter (dict keys ∩ columns).
        - Empty filtered view → empty CadwDataFrame with unchanged
          schema.
        - Column types preserved (fill with same-type scalar; no
          widening).

        For **method-based** forward/backward fill, use
        ``.ffill()`` / ``.bfill()`` (v0.9.25i) — different semantics
        (state carry from neighbours), not ``fillna(method=...)``.
        """
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("fillna: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"fillna: column {c!r} not in {self._columns}"
                )

        # Build {col: pa.Scalar} with upfront type-coercion.
        value_map: dict = {}
        if isinstance(value, dict):
            for c, v in value.items():
                if c not in self._columns:
                    raise KeyError(
                        f"fillna: dict key {c!r} not in "
                        f"{self._columns}"
                    )
                if c not in cols:
                    continue
                t = self._dataset.schema.field(c).type
                try:
                    value_map[c] = pa.scalar(v, type=t)
                except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
                    raise TypeError(
                        f"fillna: value {v!r} incompatible with "
                        f"column {c!r} type {t}: {e}"
                    ) from e
        else:
            for c in cols:
                t = self._dataset.schema.field(c).type
                try:
                    value_map[c] = pa.scalar(value, type=t)
                except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
                    raise TypeError(
                        f"fillna: value {value!r} incompatible "
                        f"with column {c!r} type {t}: {e}"
                    ) from e

        out_schema = self._dataset.schema

        def _iter():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="fillna.input")
                arrays = []
                for c in self._columns:
                    arr = b.column(c)
                    if c in value_map:
                        arr = _pc.fill_null(arr, value_map[c])
                    arrays.append(arr)
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    # ---------------- __getitem__ ----------------

    def __getitem__(self, key):'''


def add_dropna_fillna():
    print("\n=== add dropna + fillna (before __getitem__) ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, ANCHOR_OLD, ANCHOR_NEW,
        label="dropna_fillna",
    )
    if applied:
        assert_compiles(DATAFRAME_PATH)
        check_no_invalid_escape(DATAFRAME_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. tests ============

NEW_TESTS = '''"""v0.9.25m: Phase C batch C-5 (FINAL) — dropna, fillna."""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _mk(data, mem_mb=20):
    return CadwDataFrame.from_table(
        pa.table(data),
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
    )


def _mk_multi(tmp_path, data, chunk_size, mem_mb=20):
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table(data), src, row_group_size=chunk_size)
    return CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=chunk_size,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / "spill",
    )


# ============ dropna ============


def test_dropna_any_default_all_columns():
    df = _mk({
        "a": pa.array([1, 2, None, 4], type=pa.int64()),
        "b": pa.array([10, None, 30, 40], type=pa.int64()),
    })
    result = df.dropna()  # how='any', subset=all
    out = result.head(10)
    # rows 1 (b null) and 2 (a null) dropped
    assert out.column("a").to_pylist() == [1, 4]
    assert out.column("b").to_pylist() == [10, 40]


def test_dropna_how_all():
    df = _mk({
        "a": pa.array([1, None, None, 4], type=pa.int64()),
        "b": pa.array([10, None, 30, None], type=pa.int64()),
    })
    result = df.dropna(how="all")
    out = result.head(10)
    # row 1: both null → dropped; row 2: only a null (b=30) → kept;
    # row 3: b null (a=4) → kept
    assert out.column("a").to_pylist() == [1, None, 4]
    assert out.column("b").to_pylist() == [10, 30, None]


def test_dropna_subset_only():
    df = _mk({
        "a": pa.array([1, None, 3], type=pa.int64()),
        "b": pa.array([None, 20, None], type=pa.int64()),
    })
    result = df.dropna(subset=["a"])  # only a considered
    out = result.head(10)
    assert out.column("a").to_pylist() == [1, 3]
    assert out.column("b").to_pylist() == [None, None]


def test_dropna_no_nulls():
    df = _mk({"x": [1, 2, 3]})
    result = df.dropna()
    out = result.head(10)
    assert out.column("x").to_pylist() == [1, 2, 3]


def test_dropna_all_nulls():
    df = _mk({"x": pa.array([None, None], type=pa.int64())})
    result = df.dropna()
    out = result.head(10)
    assert out.num_rows == 0


def test_dropna_empty():
    df = _mk({"x": [1, 2], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result = empty.dropna()
    out = result.head(10)
    assert out.num_rows == 0


def test_dropna_filtered_composes():
    """dropna applies to the filtered view; visible rows only."""
    df = _mk({
        "x": pa.array([1, None, 3, None, 5], type=pa.int64()),
        "v": [10, 20, 30, 40, 50],
    })
    filtered = df[df["v"] > 15]  # x=[None,3,None,5]
    result = filtered.dropna()
    out = result.head(10)
    assert out.column("x").to_pylist() == [3, 5]


def test_dropna_cross_batch(tmp_path):
    df = _mk_multi(
        tmp_path,
        {
            "a": pa.array([1, None, 3, None, 5], type=pa.int64()),
        },
        chunk_size=1,
    )
    result = df.dropna()
    out = result.head(10)
    assert out.column("a").to_pylist() == [1, 3, 5]


def test_dropna_string():
    df = _mk({
        "s": pa.array(["a", None, "b"], type=pa.string()),
    })
    result = df.dropna()
    out = result.head(10)
    assert out.column("s").to_pylist() == ["a", "b"]


def test_dropna_invalid_how_raises():
    df = _mk({"x": [1, 2]})
    with pytest.raises(ValueError, match="how must be"):
        df.dropna(how="some")


def test_dropna_missing_column_raises():
    df = _mk({"x": [1, 2]})
    with pytest.raises(KeyError, match="not in"):
        df.dropna(subset=["missing"])


# ============ fillna ============


def test_fillna_scalar_int():
    df = _mk({"x": pa.array([1, None, 3], type=pa.int64())})
    result = df.fillna(0)
    out = result.head(10)
    assert out.column("x").to_pylist() == [1, 0, 3]


def test_fillna_scalar_float():
    df = _mk({"x": pa.array([1.5, None, 3.5], type=pa.float64())})
    result = df.fillna(-1.0)
    out = result.head(10)
    assert out.column("x").to_pylist() == [1.5, -1.0, 3.5]


def test_fillna_scalar_string():
    df = _mk({"s": pa.array(["a", None, "b"], type=pa.string())})
    result = df.fillna("unknown")
    out = result.head(10)
    assert out.column("s").to_pylist() == ["a", "unknown", "b"]


def test_fillna_scalar_bool():
    df = _mk({"b": pa.array([True, None, False], type=pa.bool_())})
    result = df.fillna(False)
    out = result.head(10)
    assert out.column("b").to_pylist() == [True, False, False]


def test_fillna_dict():
    df = _mk({
        "a": pa.array([1, None, 3], type=pa.int64()),
        "b": pa.array([None, 20.0, None], type=pa.float64()),
    })
    result = df.fillna({"a": 0, "b": -1.0})
    out = result.head(10)
    assert out.column("a").to_pylist() == [1, 0, 3]
    assert out.column("b").to_pylist() == [-1.0, 20.0, -1.0]


def test_fillna_dict_partial_scope():
    """Dict with fewer columns than exist: only dict keys are filled."""
    df = _mk({
        "a": pa.array([1, None, 3], type=pa.int64()),
        "b": pa.array([None, 20, None], type=pa.int64()),
    })
    result = df.fillna({"a": 0})
    out = result.head(10)
    assert out.column("a").to_pylist() == [1, 0, 3]
    # b unchanged (nulls preserved)
    assert out.column("b").to_pylist() == [None, 20, None]


def test_fillna_columns_subset():
    df = _mk({
        "a": pa.array([None, 2, None], type=pa.int64()),
        "b": pa.array([None, 20, None], type=pa.int64()),
    })
    result = df.fillna(0, columns=["a"])
    out = result.head(10)
    assert out.column("a").to_pylist() == [0, 2, 0]
    assert out.column("b").to_pylist() == [None, 20, None]


def test_fillna_no_nulls_noop():
    df = _mk({"x": [1, 2, 3]})
    result = df.fillna(0)
    out = result.head(10)
    assert out.column("x").to_pylist() == [1, 2, 3]


def test_fillna_empty_preserves_schema():
    df = _mk({"x": [1, 2], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result = empty.fillna(0)
    out = result.head(10)
    assert out.num_rows == 0
    assert "x" in out.schema.names


def test_fillna_filtered():
    df = _mk({
        "x": pa.array([None, 2, None, 4], type=pa.int64()),
        "v": [10, 20, 30, 40],
    })
    filtered = df[df["v"] > 15]  # x=[2,None,4]
    result = filtered.fillna(0)
    out = result.head(10)
    assert out.column("x").to_pylist() == [2, 0, 4]


def test_fillna_cross_batch(tmp_path):
    df = _mk_multi(
        tmp_path,
        {"x": pa.array([None, 2, None], type=pa.int64())},
        chunk_size=1,
    )
    result = df.fillna(0)
    out = result.head(10)
    assert out.column("x").to_pylist() == [0, 2, 0]


def test_fillna_type_mismatch_raises():
    """int value on string column → clear TypeError from pa.scalar."""
    df = _mk({"s": pa.array(["a", None], type=pa.string())})
    with pytest.raises(TypeError, match="incompatible"):
        df.fillna(0)


def test_fillna_dict_missing_column_raises():
    df = _mk({"x": [1, 2]})
    with pytest.raises(KeyError, match="dict key"):
        df.fillna({"missing": 0})


def test_fillna_missing_column_raises():
    df = _mk({"x": [1, 2]})
    with pytest.raises(KeyError, match="not in"):
        df.fillna(0, columns=["missing"])


def test_fillna_preserves_original_type():
    df = _mk({"x": pa.array([1, None, 3], type=pa.int64())})
    result = df.fillna(0)
    out = result.head(10)
    assert out.schema.field("x").type == pa.int64()
'''


def add_tests():
    print("\n=== add tests/test_v0925m.py ===")
    p = Path(TESTS_PATH)
    if p.exists() and "test_dropna_any_default_all_columns" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(NEW_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print(f"  OK ({len(NEW_TESTS.splitlines())} L)")


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

    print("\n=== pytest: v0925m + null-handling + dataframe ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925m.py",
         "tests/test_v0925i.py",  # ffill/bfill — sibling null-handling
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
         "tests/test_row_filter.py",
         "tests/test_filter_compat.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=300,
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
    print("=== v0.9.25m: Phase C C-5 FINAL — dropna + fillna ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    add_dropna_fillna()
    add_tests()
    run_pytest()

    commit_with_guards(
        "v0.9.25m: Phase C batch C-5 (FINAL) — dropna + fillna. "
        "**Phase C 20/20 Pandas P0 CLOSED.** "
        "2 of 20 Pandas P0 features (completing C-1's isin/between/"
        "select_dtypes, C-2's idxmax/idxmin/pct_change, C-3a's "
        "ffill/bfill/reindex, C-3b's cut/qcut/duplicated, C-4a's "
        "str.slice/cat/count, C-4b's dt.strftime/tz_convert/"
        "str.split). "
        "Features: "
        "(1) `CadwDataFrame.dropna(subset=None, how='any')` — "
        "streaming batch filter. how='any' (default): drop rows "
        "where ANY subset column is null (keep = AND of pc.is_valid "
        "over subset). how='all': drop rows where ALL subset cols "
        "null (keep = OR of pc.is_valid). subset=None → all cols. "
        "Composes with existing row filters (dropna applies to "
        "filtered view; visible rows only). Empty → empty "
        "schema-correct. Peak = batch + #subset × bool_array. "
        "(2) `CadwDataFrame.fillna(value, columns=None)` — replace "
        "nulls with scalar value or dict {col: scalar}. Type-"
        "coerced upfront via pa.scalar(value, type=col_type); "
        "incompatible → clear TypeError 'value X incompatible with "
        "column Y type Z'. Dict with fewer keys → only those filled; "
        "columns=X restricts scope. Composition of pc.fill_null "
        "per-column. Streaming O(batch), no state carry. Column "
        "types preserved (fill with same-type scalar; no widening). "
        "Method-based fill → use .ffill()/.bfill() (v0.9.25i), not "
        "fillna(method=). "
        "**Null-handling story complete**: ffill/bfill (method) + "
        "fillna (value) + dropna (filter) — all streaming, all "
        "pandas-parity semantics, all bounded memory. "
        "§13.4/264 APPLIED (REPL-first, 10 checks): verify_state "
        "BEFORE edits: pc.is_valid/is_null; pc.and_/or_ bool "
        "combination; pc.fill_null for int64/float64/string/bool; "
        "pa.scalar(int, string-type) — CAPTURED actual exception "
        "class (not guessed; §13.4/264 formalized exactly for "
        "this) — confirms fillna type validation fires; "
        "RecordBatch.filter preserves schema. All PASSED. "
        "§13.4/278 (hasattr verify): presence of reindex/ffill/bfill "
        "via hasattr(CadwDataFrame, ...), not source text — avoids "
        "multi-line-signature false negative from v0.9.25l-fix1. "
        "§13.4/266 (re-verify after anchor): verify_state asserts "
        "exact 3-line anchor text (marker + blank + def __getitem__) "
        "before apply. "
        "§13.4/260 preserved: no subprocess dogfood. §13.4/261 "
        "preserved: no symbol moves → no whitelist path updates. "
        "§13.4/262 preserved: pa.scalar(value, type=col_type) for "
        "typed scalar; pandas-parity. §13.4/265 preserved: no "
        "copy-pasted asserts from sibling tests — dropna/fillna "
        "have distinct data & assertions. "
        "DEFERRED: dropna(thresh=N) — P2; fillna(method=) — use "
        "ffill/bfill; fillna(limit=) + ffill/bfill limit= — v0.9.26 "
        "backlog. "
        "Tests (tests/test_v0925m.py): 24 total — "
        "dropna (11: any-default / how-all / subset-only / no-"
        "nulls / all-nulls / empty / filtered-compose / cross-"
        "batch / string / invalid-how / missing-column); "
        "fillna (13: int / float / string / bool / dict / dict-"
        "partial / columns-subset / no-nulls-noop / empty-schema / "
        "filtered / cross-batch / type-mismatch / dict-missing-"
        "column / missing-column / preserves-type). "
        "Fast suite: ~2488 → ~2512 (+24). "
        "**Phase C DONE (20/20).** "
        "Next: Phase C close-out (docs/DECISIONS §13.4/279 "
        "retrospective, CHANGELOG v0.9.25 milestone, roadmap.json "
        "update), then Phase D / v1.0 readiness.",
        self_path="scripts/patch_v0925m_dropna_fillna.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25m COMPLETE. **Phase C 20/20 CLOSED.** "
          "Next: C close-out docs + Phase D / v1.0.")


if __name__ == "__main__":
    main()