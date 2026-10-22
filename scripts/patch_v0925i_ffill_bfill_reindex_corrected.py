#!/usr/bin/env python3
# v0.9.25i-corrected — C-3a ffill + bfill + reindex (corrected anchor
#                     + reindex filtered-RangeIndex fix).
#
# PREVIOUS v0.9.25i attempt aborted at add_ffill_bfill_reindex:
#   ANCHOR MISS: "def corr(self, columns=None, *, method='pearson'):"
# Reason: after pct_change (v0.9.25h, inserted after select_dtypes),
# next is `# ---------------- __getitem__ ----------------`, NOT
# def corr. corr is far down the file. Anchor was based on wrong
# assumption about file structure.
#
# §13.4/160 fail-fast (apply_or_fail) worked correctly: 0 mutations.
# HEAD stayed v0.9.25h. This is not a fix-cycle — it's the safety net.
#
# During re-anchoring (manual reasoning), found latent reindex bug:
#   `reindex` for `_index_kind == "range"` mapped labels [start,
#   n_view) against FILTERED-VIEW positions. But filtered RangeIndex
#   has labels = ORIGINAL positional values (positions + start), not
#   filtered enumeration. E.g., filtered rows at raw [1, 3, 5] have
#   index labels [1, 3, 5]; reindex([3]) should return the row with
#   label 3 (filtered pos 1), not "label 3 doesn't exist because
#   filtered view has only 3 rows [0,1,2]".
#
#   Fix: for filtered RangeIndex (and materialized case), use
#   `self.index` (returns MaterializedIndex(original_positions +
#   start) when filtered; MaterializedIndex(_index_values) when
#   not). Both have get_positions_all returning filtered-view
#   positions (which is what iloc[] expects).
#
# Also fixed: all-missing-labels case (`k_valid = 0`) — take on
# empty table with null indices would raise ArrowIndexError; added
# null-fill shortcut before take.
#
# Also efficiency: materialized lookup was O(K × N) per label via
# get_positions_all (linear scan each call). Now build label→pos
# dict once → O(N + K).
#
# §13.4/266 (NEW): apply_or_fail anchor miss = 0-mutation safety,
# but manual reasoning during fix must check ALL affected call paths.
# Re-verify each feature's assumptions after anchor correction.
#
# §13.4/264 (REPL-first): verify_state has 8 checks (same as before
# + 1 new: pc.fill_null_forward on strings). Added during re-anchor
# to cover string ffill path.
#
# §13.4/260 preserved: no subprocess dogfood.
# §13.4/261 preserved: no symbol moves → no whitelist path updates.
# §13.4/262 preserved: pa.scalar typed reconstruction.
#
# Run: uv run python scripts/patch_v0925i_ffill_bfill_reindex_corrected.py

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
TESTS_PATH = "tests/test_v0925i.py"


# ============ 0. verify state (REPL-first, §13.4/264) ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25h; REPL-check 9 assumptions; "
          "no v0.9.25i features yet ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25h" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25h; got {log[:200]!r}"
        )
    if "v0.9.25i" in log:
        raise SystemExit("v0.9.25i already committed — patch misplaced")

    src = Path(DATAFRAME_PATH).read_text()
    if "def ffill(self" in src:
        raise SystemExit("ffill already present")
    if "def bfill(self" in src:
        raise SystemExit("bfill already present")
    if "def reindex(self" in src:
        raise SystemExit("reindex already present")

    # Verify anchor exists (the correction vs prior aborted attempt)
    if "    # ---------------- __getitem__ ----------------\n\n    def __getitem__(self, key):" not in src:
        raise SystemExit(
            "ANCHOR MISS: `# ---------------- __getitem__ ----------------`"
            " followed by `def __getitem__(self, key):` not found. "
            "Inspect dataframe.py structure — my anchor assumption "
            "may be wrong again."
        )

    # --- REPL checks ---
    import pyarrow as pa
    import pyarrow.compute as _pc

    # (1) fill_null_forward exists
    if not hasattr(_pc, "fill_null_forward"):
        raise SystemExit(
            "pc.fill_null_forward missing — PyArrow version drift"
        )

    # (2) int
    r = _pc.fill_null_forward(
        pa.array([1, None, None, 2, None], type=pa.int64())
    ).to_pylist()
    if r != [1, 1, 1, 2, 2]:
        raise SystemExit(
            f"fill_null_forward int: {r}, expected [1,1,1,2,2]"
        )

    # (3) leading nulls stay
    r = _pc.fill_null_forward(
        pa.array([None, 1, None], type=pa.int64())
    ).to_pylist()
    if r != [None, 1, 1]:
        raise SystemExit(
            f"fill_null_forward leading: {r}, expected [None,1,1]"
        )

    # (4) string (NEW — v0.9.25i-corrected)
    r = _pc.fill_null_forward(
        pa.array(["a", None, "b", None], type=pa.string())
    ).to_pylist()
    if r != ["a", "a", "b", "b"]:
        raise SystemExit(
            f"fill_null_forward string: {r}, expected "
            f"['a','a','b','b'] — string ffill path may break"
        )

    # (5) Table.take with null indices
    tbl = pa.table({"x": [10, 20, 30]})
    try:
        r = tbl.take(pa.array([0, None, 2], type=pa.int64()))
        r_list = r.column("x").to_pylist()
        if r_list != [10, None, 30]:
            raise SystemExit(
                f"take with null: {r_list}, expected [10,None,30]"
            )
    except (pa.ArrowInvalid, pa.ArrowIndexError) as e:
        raise SystemExit(
            f"take with null fails: {type(e).__name__}: {e}. "
            f"reindex needs scatter fallback."
        )

    # (6) get_positions_all missing/dup
    from cadw.dataset.index import MaterializedIndex
    idx = MaterializedIndex(pa.array(["a", "b", "c"]))
    if idx.get_positions_all("z") != []:
        raise SystemExit("get_positions_all missing != []")
    if idx.get_positions_all("b") != [1]:
        raise SystemExit("get_positions_all single != [1]")
    idx_dup = MaterializedIndex(pa.array(["a", "b", "a"]))
    if idx_dup.get_positions_all("a") != [0, 2]:
        raise SystemExit(
            f"get_positions_all dup: {idx_dup.get_positions_all('a')}"
        )

    # (7) pa.array with None
    r = pa.array([1, 2, None], type=pa.int64()).to_pylist()
    if r != [1, 2, None]:
        raise SystemExit(f"pa.array with None: {r}")

    # (8) external_sort signature
    import inspect
    from cadw.ops.sort import external_sort
    sig = inspect.signature(external_sort)
    for p in ("value_columns", "key_columns"):
        if p not in sig.parameters:
            raise SystemExit(f"external_sort missing {p!r}")

    # (9) RangeIndex for unfiltered (sanity, no actual df needed)
    from cadw.dataset.index import RangeIndex
    ri = RangeIndex(0, 10)
    if len(ri) != 10:
        raise SystemExit(f"RangeIndex(0,10) length {len(ri)}, expected 10")

    print("  9 REPL checks passed; state + anchor verified")


# ============ 1. add ffill + bfill + reindex (before # __getitem__) ============

FFILL_ANCHOR_OLD = '''    # ---------------- __getitem__ ----------------

    def __getitem__(self, key):'''


FFILL_ANCHOR_NEW = '''    def ffill(self, columns=None) -> "CadwDataFrame":
        """Forward-fill nulls per column (v0.9.25i).

        Streaming per-batch via pc.fill_null_forward + carry for
        leading-null cross-batch fix. O(#cols) state, O(N) I/O.

        - columns: None → all columns; str → one; Sequence[str] →
          subset. Non-fillable columns pass through unchanged.
        - Leading nulls stay null (no prior value).
        - Null-safe across batches: last non-null carries forward.
        - Original column types preserved (nullable=True forced).
        - Int cols stay int (fill doesn't widen types).

        Peak RAM = batch + #cols × 8 B (carry buffers).
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
            raise ValueError("ffill: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"ffill: column {c!r} not in {self._columns}"
                )

        fill_set = set(cols)
        out_fields = []
        for c in self._columns:
            orig = self._dataset.schema.field(c)
            if c in fill_set:
                out_fields.append(
                    pa.field(c, orig.type, nullable=True)
                )
            else:
                out_fields.append(orig)
        out_schema = pa.schema(out_fields)

        carry: dict = {c: None for c in cols}

        def _iter():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="ffill.input")
                n = b.num_rows
                if n == 0:
                    yield b.select(self._columns)
                    continue
                arrays = []
                for c in self._columns:
                    arr = b.column(c)
                    if c not in fill_set:
                        arrays.append(arr)
                        continue
                    car = carry[c]
                    if car is not None:
                        extended = pa.concat_arrays([car, arr])
                        filled = _pc.fill_null_forward(extended).slice(1)
                    else:
                        filled = _pc.fill_null_forward(arr)
                    if len(filled) > 0:
                        last_slice = filled.slice(len(filled) - 1, 1)
                        if _pc.is_valid(last_slice).to_pylist()[0]:
                            carry[c] = last_slice
                    arrays.append(filled)
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

    def bfill(self, columns=None) -> "CadwDataFrame":
        """Backward-fill nulls per column (v0.9.25i).

        Implementation: reverse-sort (__orig_rid descending) +
        ffill-internal + re-sort (__orig_rid ascending). 2×
        external_sort — O(N log N) I/O, O(batch) RAM. **Heavier than
        ffill** (O(N) I/O). Documented.

        - columns: None → all; str → one; Sequence[str] → subset.
        - Trailing nulls stay null (no later value).
        - Original order preserved (2× sort round-trip).
        """
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset
        from cadw.io.batch_reader import ArrowIPCBatchReader
        from cadw.io.batch_writer import ArrowIPCBatchWriter
        from cadw.ops.sort import external_sort

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("bfill: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"bfill: column {c!r} not in {self._columns}"
                )

        fill_set = set(cols)
        out_fields = []
        for c in self._columns:
            orig = self._dataset.schema.field(c)
            if c in fill_set:
                out_fields.append(
                    pa.field(c, orig.type, nullable=True)
                )
            else:
                out_fields.append(orig)
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        # Step 1: add __orig_rid, external_sort descending
        def _iter_with_rid():
            offset = 0
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="bfill.step1")
                n = b.num_rows
                rid = pa.array(
                    range(offset, offset + n), type=pa.uint64(),
                )
                offset += n
                yield b.append_column("__orig_rid", rid)

        sorted_desc = external_sort(
            _iter_with_rid(),
            key_columns=["__orig_rid"],
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "bfill" / "sort_desc",
            ascending=False,
            log=False,
        )

        # Step 2: ffill pass
        filled_path = self._spill_dir / "bfill" / "filled.arrow"
        filled_path.parent.mkdir(parents=True, exist_ok=True)
        filled_schema = pa.schema(
            [pa.field("__orig_rid", pa.uint64())]
            + [
                pa.field(c, self._dataset.schema.field(c).type,
                         nullable=True)
                for c in self._columns
            ]
        )
        writer = ArrowIPCBatchWriter(str(filled_path), filled_schema)
        carry: dict = {c: None for c in self._columns}
        try:
            for b in ArrowIPCBatchReader(str(sorted_desc), None):
                self._memory_guard.check_batch(
                    b, name="bfill.step2.read",
                )
                n = b.num_rows
                if n == 0:
                    continue
                arrays = [b.column("__orig_rid")]
                for c in self._columns:
                    arr = b.column(c)
                    car = carry[c]
                    if car is not None:
                        extended = pa.concat_arrays([car, arr])
                        filled = _pc.fill_null_forward(extended).slice(1)
                    else:
                        filled = _pc.fill_null_forward(arr)
                    if len(filled) > 0:
                        last_slice = filled.slice(len(filled) - 1, 1)
                        if _pc.is_valid(last_slice).to_pylist()[0]:
                            carry[c] = last_slice
                    arrays.append(filled)
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        arrays, schema=filled_schema,
                    )
                )
        finally:
            writer.close()

        # Step 3: external_sort ascending by __orig_rid
        def _iter_filled():
            for b in ArrowIPCBatchReader(str(filled_path), None):
                yield b

        sorted_asc = external_sort(
            _iter_filled(),
            key_columns=["__orig_rid"],
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "bfill" / "sort_asc",
            ascending=True,
            log=False,
        )

        def _iter_out():
            for b in ArrowIPCBatchReader(str(sorted_asc), None):
                yield b.select(list(self._columns))

        new_ds = Dataset.from_batches(
            out_schema, _iter_out(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def reindex(self, labels) -> "CadwDataFrame":
        """Row alignment to target labels (v0.9.25i).

        For each label in `labels`:
          - found in source index → source row
          - not found → null-filled row

        - labels: iterable of scalars.
        - Returns CadwDataFrame with rows in `labels` order.
        - Duplicate source labels: first occurrence wins.
        - MultiIndex source: NotImplementedError.
        - RangeIndex source: labels interpreted as ORIGINAL positional
          ints (0-based). Filtered RangeIndex: labels are ORIGINAL
          positions, not filtered-view enumeration.
        - Filtered source: labels matched against filtered view's index
          (via self.index property — MaterializedIndex with original
          labels).

        Memory: O(K × row_size) where K = len(labels). Requires
        K × row_size ≤ state_limit.

        Peak RAM = K × row_size + batch.
        """
        from cadw.dataset import Dataset
        from cadw.dataset.index import MaterializedIndex
        from cadw.core.memory_guard import MemoryLimitExceeded

        labels = list(labels)
        if not labels:
            raise ValueError("reindex: labels must be non-empty")

        if self._index_kind == "multi":
            raise NotImplementedError(
                "reindex: MultiIndex source not supported; use "
                ".reset_index() first"
            )

        # Resolve each label → position in FILTERED view (which is
        # what iloc[] expects).
        positions: list = []
        if self._index_kind == "range" and not self._row_filters:
            # Unfiltered RangeIndex: labels are positional ints.
            start = self._index_start
            stop = self._index_stop
            for lab in labels:
                try:
                    lab_int = int(lab)
                except (TypeError, ValueError):
                    positions.append(None)
                    continue
                if start <= lab_int < stop:
                    positions.append(lab_int - start)
                else:
                    positions.append(None)
        else:
            # Filtered (any kind) OR materialized:
            # self.index → MaterializedIndex with correct labels;
            # get_positions_all returns filtered-view positions.
            # Build label→pos map once (O(N + K) not O(K*N)).
            idx = self.index
            label_to_pos: dict = {}
            for i, x in enumerate(idx):
                try:
                    if x not in label_to_pos:
                        label_to_pos[x] = i
                except TypeError:
                    continue
            for lab in labels:
                try:
                    positions.append(label_to_pos.get(lab, None))
                except TypeError:
                    positions.append(None)

        valid_positions = [p for p in positions if p is not None]
        k_valid = len(valid_positions)

        # All-missing shortcut: null-filled table (avoid take-on-empty)
        if k_valid == 0:
            n = len(labels)
            arrays = [
                pa.nulls(n, type=self._dataset.schema.field(c).type)
                for c in self._columns
            ]
            result_table = pa.Table.from_arrays(
                arrays, names=self._columns,
            )
            return self._wrap_result_table(result_table)

        # Memory check
        first_batch = None
        for b in self._iter_filtered_batches():
            first_batch = b
            break
        if first_batch is None:
            row_bytes = max(
                1, self._memory_guard.estimate_row_size_from_schema(
                    pa.schema([self._dataset.schema.field(c)
                               for c in self._columns]),
                )
            )
        else:
            row_bytes = self._memory_guard.reestimate_from_batch(
                first_batch,
            )
        est = k_valid * row_bytes
        state_limit = self._memory_guard.state_limit_bytes
        if est > state_limit:
            raise MemoryLimitExceeded(
                f"reindex: K={k_valid} × {row_bytes} B/row = "
                f"{est} B > state_limit {state_limit} B. "
                f"Reduce labels count or increase budget."
            )
        self._memory_guard.check_state(
            est, name=f"reindex(K={k_valid})",
        )

        # Gather valid rows via iloc[list]
        valid_df = self.iloc[valid_positions]
        valid_table = valid_df.head(k_valid)

        # Map: output[i] = valid_table[valid_idx_counter] if
        # positions[i] is not None, else null.
        valid_idx_arr: list = []
        valid_idx_counter = 0
        for pos in positions:
            if pos is None:
                valid_idx_arr.append(None)
            else:
                valid_idx_arr.append(valid_idx_counter)
                valid_idx_counter += 1

        try:
            result_table = valid_table.take(
                pa.array(valid_idx_arr, type=pa.int64())
            )
        except (pa.ArrowInvalid, pa.ArrowIndexError) as e:
            raise RuntimeError(
                f"reindex: Table.take with null indices failed: {e}. "
                f"This PyArrow build may not support null-safe take."
            ) from e

        return self._wrap_result_table(result_table)

    # ---------------- __getitem__ ----------------

    def __getitem__(self, key):'''


def add_ffill_bfill_reindex():
    print("\n=== add ffill + bfill + reindex methods ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, FFILL_ANCHOR_OLD, FFILL_ANCHOR_NEW,
        label="ffill_bfill_reindex",
    )
    if applied:
        assert_compiles(DATAFRAME_PATH)
        check_no_invalid_escape(DATAFRAME_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. tests ============

NEW_TESTS = '''"""v0.9.25i: Phase C batch C-3a — ffill, bfill, reindex."""
import math

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


# ============ ffill ============


def test_ffill_basic():
    df = _mk({"x": pa.array([1, None, None, 2, None],
                             type=pa.int64())})
    result = df.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1, 1, 1, 2, 2]


def test_ffill_leading_null_stays():
    df = _mk({"x": pa.array([None, 1, None, 2], type=pa.int64())})
    result = df.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [None, 1, 1, 2]


def test_ffill_all_null():
    df = _mk({"x": pa.array([None, None, None], type=pa.int64())})
    result = df.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [None, None, None]


def test_ffill_no_nulls():
    df = _mk({"x": [1, 2, 3]})
    result = df.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1, 2, 3]


def test_ffill_float():
    df = _mk({"x": pa.array([1.5, None, 2.5], type=pa.float64())})
    result = df.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1.5, 1.5, 2.5]


def test_ffill_multiple_columns():
    df = _mk({
        "a": pa.array([1, None, 3], type=pa.int64()),
        "b": pa.array([10.0, 20.0, None], type=pa.float64()),
    })
    result = df.ffill()
    out = result.head(10)
    assert out.column("a").to_pylist() == [1, 1, 3]
    assert out.column("b").to_pylist() == [10.0, 20.0, 20.0]


def test_ffill_columns_subset():
    df = _mk({
        "a": pa.array([1, None, 3], type=pa.int64()),
        "b": pa.array([10, None, 30], type=pa.int64()),
    })
    result = df.ffill(columns=["a"])
    out = result.head(10)
    assert out.column("a").to_pylist() == [1, 1, 3]
    assert out.column("b").to_pylist() == [10, None, 30]


def test_ffill_cross_batch(tmp_path):
    df = _mk_multi(
        tmp_path,
        {"x": pa.array([1, None, None, 2, None], type=pa.int64())},
        chunk_size=1,
    )
    result = df.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1, 1, 1, 2, 2]


def test_ffill_filtered():
    df = _mk({
        "x": pa.array([1, None, 3, None, 5], type=pa.int64()),
        "v": [10, 20, 30, 40, 50],
    })
    filtered = df[df["v"] > 15]
    result = filtered.ffill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [None, 3, 3, 5]


def test_ffill_empty():
    df = _mk({"x": [1, 2], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result = empty.ffill()
    out = result.head(10)
    assert out.num_rows == 0


def test_ffill_string():
    df = _mk({
        "s": pa.array(["a", None, "b"], type=pa.string()),
    })
    result = df.ffill()
    vals = result.head(10).column("s").to_pylist()
    assert vals == ["a", "a", "b"]


# ============ bfill ============


def test_bfill_basic():
    df = _mk({"x": pa.array([None, None, 3, None],
                             type=pa.int64())})
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [3, 3, 3, None]


def test_bfill_trailing_null_stays():
    df = _mk({"x": pa.array([1, None, None], type=pa.int64())})
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1, None, None]


def test_bfill_all_null():
    df = _mk({"x": pa.array([None, None], type=pa.int64())})
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [None, None]


def test_bfill_no_nulls():
    df = _mk({"x": [1, 2, 3]})
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1, 2, 3]


def test_bfill_float():
    df = _mk({"x": pa.array([None, 1.5, None], type=pa.float64())})
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1.5, 1.5, None]


def test_bfill_multiple_columns():
    df = _mk({
        "a": pa.array([None, 2, None], type=pa.int64()),
        "b": pa.array([10.0, None, 30.0], type=pa.float64()),
    })
    result = df.bfill()
    out = result.head(10)
    assert out.column("a").to_pylist() == [2, 2, None]
    assert out.column("b").to_pylist() == [10.0, 30.0, 30.0]


def test_bfill_columns_subset():
    df = _mk({
        "a": pa.array([None, 2, None], type=pa.int64()),
        "b": pa.array([None, 20, None], type=pa.int64()),
    })
    result = df.bfill(columns=["a"])
    out = result.head(10)
    assert out.column("a").to_pylist() == [2, 2, None]
    assert out.column("b").to_pylist() == [None, 20, None]


def test_bfill_cross_batch(tmp_path):
    df = _mk_multi(
        tmp_path,
        {"x": pa.array([None, None, 3, None], type=pa.int64())},
        chunk_size=1,
    )
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [3, 3, 3, None]


def test_bfill_filtered():
    df = _mk({
        "x": pa.array([None, 2, None, None, 5], type=pa.int64()),
        "v": [10, 20, 30, 40, 50],
    })
    filtered = df[df["v"] > 15]
    result = filtered.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [2, None, 5, 5]


def test_bfill_empty():
    df = _mk({"x": [1, 2], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result = empty.bfill()
    out = result.head(10)
    assert out.num_rows == 0


def test_bfill_preserves_original_order():
    df = _mk({"x": pa.array([1, None, 3, None, 5],
                             type=pa.int64())})
    result = df.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [1, 3, 3, 5, 5]


# ============ reindex ============


def test_reindex_all_present(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "b", "c"], "v": [10, 20, 30],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["a", "b", "c"])
    out = result.head(10)
    assert out.column("v").to_pylist() == [10, 20, 30]


def test_reindex_missing_label_nulls(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "b", "c"], "v": [10, 20, 30],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["a", "z", "c"])
    out = result.head(10)
    assert out.column("v").to_pylist() == [10, None, 30]


def test_reindex_order_matters(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "b", "c"], "v": [10, 20, 30],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["c", "a", "b"])
    out = result.head(10)
    assert out.column("v").to_pylist() == [30, 10, 20]


def test_reindex_duplicate_target_first_wins(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "b", "c"], "v": [10, 20, 30],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["a", "a"])
    out = result.head(10)
    assert out.column("v").to_pylist() == [10, 10]


def test_reindex_duplicate_source_first_wins(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "a", "b"], "v": [10, 20, 30],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["a"])
    out = result.head(10)
    assert out.column("v").to_pylist() == [10]


def test_reindex_range_index():
    df = _mk({"v": [10, 20, 30, 40, 50]})
    result = df.reindex([0, 2, 4])
    out = result.head(10)
    assert out.column("v").to_pylist() == [10, 30, 50]


def test_reindex_range_index_oob():
    df = _mk({"v": [10, 20, 30]})
    result = df.reindex([0, 100])
    out = result.head(10)
    assert out.column("v").to_pylist() == [10, None]


def test_reindex_filtered_range_index():
    """v0.9.25i-corrected (§13.4/266): filtered RangeIndex — labels
    are ORIGINAL positional ints, not filtered-view enumeration.

    df = [10, 20, 30, 40, 50]; filtered = df[df > 15] = rows at raw
    positions 1, 2, 3, 4 → filtered index labels [1, 2, 3, 4].
    reindex([2, 4]) → rows at labels 2 and 4 → v=[20, 40].
    """
    df = _mk({"v": [10, 20, 30, 40, 50]})
    filtered = df[df["v"] > 15]  # rows at raw positions 1-4
    result = filtered.reindex([2, 4])
    out = result.head(10)
    # Label 2 → raw pos 2 → v=30; label 4 → raw pos 4 → v=50
    # Wait — filtered rows are at raw [1, 2, 3, 4] with labels [1,2,3,4].
    # Label 2 = raw pos 2 = v=30. Label 4 = raw pos 4 = v=50.
    assert out.column("v").to_pylist() == [30, 50]


def test_reindex_empty_labels_raises():
    df = _mk({"v": [10, 20]})
    with pytest.raises(ValueError, match="non-empty"):
        df.reindex([])


def test_reindex_all_missing(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "b"], "v": [10, 20],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["x", "y", "z"])
    out = result.head(10)
    assert out.column("v").to_pylist() == [None, None, None]


def test_reindex_preserves_index(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "k": ["a", "b"], "v": [10, 20],
        }),
        src,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    ).set_index("k", drop=False)
    result = df.reindex(["a", "b"])
    assert result.count() == 2
'''

def add_tests():
    print("\n=== add tests/test_v0925i.py ===")
    p = Path(TESTS_PATH)
    if p.exists() and "test_ffill_basic" in p.read_text():
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

    print("\n=== pytest: v0925i + filters + sort + dataframe ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925i.py",
         "tests/test_sort.py",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
         "tests/test_loc.py",
         "tests/test_multiindex.py",
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
    print("=== v0.9.25i-corrected: C-3a ffill + bfill + reindex ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    add_ffill_bfill_reindex()
    add_tests()
    run_pytest()

    commit_with_guards(
        "v0.9.25i: Phase C batch C-3a — ffill + bfill + reindex. "
        "3 of 20 Pandas P0 features (after C-1 v0.9.25g, C-2 v0.9.25h). "
        "All pure compositions of existing primitives + Arrow kernels. "
        "No symbol moves, no whitelist changes, no new modules, no "
        "new memory models. "
        "Features: "
        "(1) `CadwDataFrame.ffill(columns=None)` — forward fill "
        "nulls per column. Streaming per-batch via pc.fill_null_"
        "forward + carry Array[1] for cross-batch leading-null fix. "
        "O(#cols) state, O(N) I/O. Original types preserved "
        "(nullable=True). Int stays int; string stays string. "
        "(2) `CadwDataFrame.bfill(columns=None)` — backward fill via "
        "reverse-sort (desc __orig_rid) + ffill-internal + re-sort "
        "(asc __orig_rid). 2× external_sort: O(N log N) I/O, O(batch) "
        "RAM. Heavier than ffill (documented). Trailing nulls stay "
        "null. "
        "(3) `CadwDataFrame.reindex(labels)` — row alignment to "
        "target labels. Unfiltered RangeIndex: labels are positional "
        "ints. Filtered (any kind) OR materialized: label→pos map "
        "built once from self.index (MaterializedIndex with correct "
        "labels); O(N + K). Missing → null row via Table.take with "
        "null-safe indices. All-missing shortcut: null-filled table "
        "(avoids take-on-empty). Bounded by K × row_size ≤ state_"
        "limit. MultiIndex → NotImplementedError. "
        "CORRECTIONS vs aborted v0.9.25i (0 mutation): "
        "(a) ANCHOR FIX: prior anchor assumed `def corr` follows "
        "pct_change; actual next is `# ---------------- __getitem__ "
        "----------------`. apply_or_fail fail-fast (§13.4/160): 0 "
        "mutations, HEAD stayed v0.9.25h. Correct anchor now verifies "
        "exact text before apply. "
        "(b) REINDEX FILTERED-RANGEINDEX BUG: prior code mapped "
        "labels [start, n_view) against FILTERED-VIEW positions. "
        "Filtered RangeIndex's labels are ORIGINAL positionals "
        "(positions + start). Fix: `_row_filters` check — unfiltered "
        "RangeIndex positional; else use self.index (Materialized"
        "Index, correct labels). Also O(K×N) per-label scan replaced "
        "with O(N+K) label→pos dict. Test test_reindex_filtered_"
        "range_index locks this in. "
        "(c) ALL-MISSING case: take on empty table + null indices "
        "would raise ArrowIndexError; added null-filled table "
        "shortcut. "
        "§13.4/266 (NEW): apply_or_fail anchor miss = 0-mutation "
        "safety, but manual reasoning during fix must check ALL "
        "affected call paths — anchor fix surfaced the reindex-"
        "filtered bug that would ship silently. Re-verify each "
        "feature's assumptions after anchor correction. "
        "§13.4/264 (REPL-first): verify_state does 9 checks BEFORE "
        "any edit — fill_null_forward exists + int/leading/string "
        "behavior; take with null indices; get_positions_all "
        "missing/dup; pa.array None; external_sort signature; "
        "RangeIndex sanity; anchor text presence. All PASSED. "
        "§13.4/260 preserved: no subprocess dogfood. §13.4/261 "
        "preserved: no symbol moves → no whitelist path updates. "
        "§13.4/262 preserved: pa.scalar typed scalar. §13.4/265: "
        "avoid copy-pasted asserts (checked sibling tests — bfill/"
        "ffill share helpers, different data, no copy-paste). "
        "DEFERRED: ffill/bfill limit= param → v0.9.25j; reindex "
        "MultiIndex → NotImplementedError; bfill pending-buffer op "
        "(O(N log N)-free alternative) → backlog. "
        "Tests (tests/test_v0925i.py): 33 total — ffill (11), "
        "bfill (11), reindex (11, incl. NEW test_reindex_filtered_"
        "range_index for correction (b)). "
        "Fast suite: ~2371 → ~2404 (+33). "
        "Next: Phase C C-3b (cut/qcut/duplicated).",
        self_path="scripts/patch_v0925i_ffill_bfill_reindex_corrected.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25i COMPLETE (3 features). Phase C: 9/20. "
          "Next: C-3b (cut/qcut/duplicated).")


if __name__ == "__main__":
    main()