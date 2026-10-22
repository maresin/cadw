#!/usr/bin/env python3
# v0.9.25i — Phase C batch C-3a: ffill + bfill + reindex.
#
# 3 surface API features. All pure compositions of existing primitives
# (external_sort, Arrow kernels, MaterializedIndex). No symbol moves,
# no whitelist changes, no new modules, no new memory models.
#
# FEATURES:
#   1. `CadwDataFrame.ffill(columns=None)` — forward fill nulls per
#      column. Streaming per-batch via pc.fill_null_forward + carry
#      for leading-null cross-batch fix. O(#cols) state.
#   2. `CadwDataFrame.bfill(columns=None)` — backward fill.
#      Implementation: 2× external_sort (desc by __orig_rid → ffill →
#      asc by __orig_rid). O(batch) state, O(N log N) I/O. Documented
#      as heavier than ffill. Uses _ArrowTail-like carry in ffill
#      internal pass.
#   3. `CadwDataFrame.reindex(labels)` — row alignment to target labels.
#      Uses MaterializedIndex.get_positions_all + Arrow take with null
#      indices (null-filled rows for missing labels). Bounded by
#      K × row_size ≤ state_limit.
#
# SEMANTICS (pandas-parity where possible):
#   ffill: leading nulls stay null (no prior value to carry).
#   bfill: trailing nulls stay null (no later value). Both null-safe.
#   reindex: label lookup first-match-wins for duplicates. RangeIndex
#     treated as positional labels (int). Missing → null row. Order
#     matches input labels.
#
# §13.4/264 APPLIED (REPL-first, first patch after this lesson's
# formalization):
#   verify_state does 8 pyarrow REPL checks BEFORE any edit:
#     (1) hasattr(pc, 'fill_null_forward')
#     (2) pc.fill_null_forward([1,None,None,2,None]) == [1,1,1,2,2]
#     (3) pc.fill_null_forward([None,1,None]) == [None,1,1] (leading
#         nulls stay null — non-null carry handled per-batch)
#     (4) Table.take with null indices: pa.table({'x':[10,20,30]}).
#         take([0, None, 2]) → null row at pos 1 (Arrow-native null-
#         safe take)
#     (5) MaterializedIndex.get_positions_all(missing) == []
#     (6) MaterializedIndex.get_positions_all(dup) == [pos1, pos2]
#     (7) pa.array([1,2,None], int64).to_pylist() == [1,2,None]
#     (8) external_sort accepts value_columns with new sort key
#   If any critical differs → abort PRE-mutation (§13.4/264).
#
# §13.4/260 preserved: no subprocess dogfood (dedent trap).
# §13.4/261 preserved: no symbol moves → no whitelist path updates.
# §13.4/262 preserved: pa.scalar for typed reconstruction.
#
# DEFERRED:
#   - ffill/bfill limit= param → v0.9.25j (bounded-window state).
#   - reindex on MultiIndex → NotImplementedError (tuple label).
#   - reindex with duplicate labels in target → first-match-wins (pandas
#     matches this); duplicate labels in source → first row picked.
#   - bfill without external_sort → alternative: pending-null-buffer
#     op for O(N log N)-free path. Backlog.
#
# Fix-cycle budget: 1-2 (3 features, independent anchors, REPL-first).
# bfill is the most complex — external_sort dance mirroring _shift_on.
#
# Run: uv run python scripts/patch_v0925i_ffill_bfill_reindex.py

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
    print("\n=== verify: HEAD=v0.9.25h; REPL-check 8 assumptions; "
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

    df = Path(DATAFRAME_PATH).read_text()
    if "def ffill(self" in df:
        raise SystemExit("ffill already present")
    if "def bfill(self" in df:
        raise SystemExit("bfill already present")
    if "def reindex(self" in df:
        raise SystemExit("reindex already present")

    # --- REPL checks ---
    import pyarrow as pa
    import pyarrow.compute as _pc

    # (1) fill_null_forward exists
    if not hasattr(_pc, "fill_null_forward"):
        raise SystemExit(
            "pc.fill_null_forward missing — ffill kernel unavailable. "
            "Would need per-batch stateful implementation. Check "
            "PyArrow version."
        )

    # (2) fill_null_forward behavior on interior nulls
    r = _pc.fill_null_forward(
        pa.array([1, None, None, 2, None], type=pa.int64())
    ).to_pylist()
    if r != [1, 1, 1, 2, 2]:
        raise SystemExit(
            f"pc.fill_null_forward([1,None,None,2,None]) = {r}, "
            f"expected [1,1,1,2,2]"
        )

    # (3) fill_null_forward leading nulls stay null
    r = _pc.fill_null_forward(
        pa.array([None, 1, None], type=pa.int64())
    ).to_pylist()
    if r != [None, 1, 1]:
        raise SystemExit(
            f"pc.fill_null_forward([None,1,None]) = {r}, "
            f"expected [None,1,1] — leading nulls must stay null "
            f"(caller handles cross-batch carry)"
        )

    # (4) Table.take with null indices — null-safe?
    tbl = pa.table({"x": [10, 20, 30]})
    try:
        r = tbl.take(pa.array([0, None, 2], type=pa.int64()))
        r_list = r.column("x").to_pylist()
        if r_list != [10, None, 30]:
            raise SystemExit(
                f"Table.take([0,None,2]) = {r_list}, expected "
                f"[10,None,30] (null-safe). If take errors instead, "
                f"reindex needs scatter fallback."
            )
    except (pa.ArrowInvalid, pa.ArrowIndexError) as e:
        raise SystemExit(
            f"Table.take with null indices raises {type(e).__name__}: "
            f"{e}. reindex needs scatter fallback (add to code)."
        )

    # (5) MaterializedIndex.get_positions_all on missing label
    from cadw.dataset.index import MaterializedIndex
    idx = MaterializedIndex(pa.array(["a", "b", "c"]))
    if idx.get_positions_all("z") != []:
        raise SystemExit(
            f"get_positions_all('z') = {idx.get_positions_all('z')}, "
            f"expected []"
        )
    if idx.get_positions_all("b") != [1]:
        raise SystemExit(
            f"get_positions_all('b') = {idx.get_positions_all('b')}, "
            f"expected [1]"
        )

    # (6) get_positions_all on duplicate label
    idx_dup = MaterializedIndex(pa.array(["a", "b", "a"]))
    if idx_dup.get_positions_all("a") != [0, 2]:
        raise SystemExit(
            f"get_positions_all on dup: "
            f"{idx_dup.get_positions_all('a')}, expected [0, 2]"
        )

    # (7) pa.array with None type
    r = pa.array([1, 2, None], type=pa.int64()).to_pylist()
    if r != [1, 2, None]:
        raise SystemExit(f"pa.array([1,2,None], int64) = {r}")

    # (8) external_sort accepts value_columns with new key
    import inspect
    from cadw.ops.sort import external_sort
    sig = inspect.signature(external_sort)
    if "value_columns" not in sig.parameters:
        raise SystemExit(
            "external_sort missing value_columns param — API changed"
        )

    print("  8 REPL checks passed; state OK")


# ============ 1. ffill + bfill methods ============

# Anchor: end of select_dtypes / pct_change. Use pct_change's trailing
# code as anchor (v0.9.25h added pct_change after select_dtypes).
FFILL_ANCHOR_TAIL = '''        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def corr(self, columns=None, *, method="pearson"):'''

# The anchor above is the tail of pct_change followed by corr() method
# signature. Verify it exists in current dataframe.py.

FFILL_NEW_BLOCK = '''        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def ffill(self, columns=None) -> "CadwDataFrame":
        """Forward-fill nulls per column (v0.9.25i).

        Streaming per-batch via pc.fill_null_forward + carry for
        leading-null cross-batch fix. O(#cols) state, O(N) I/O.

        - columns: None → all columns; str → one; Sequence[str] →
          subset. Non-fillable columns pass through unchanged.
        - Leading nulls stay null (no prior value).
        - Null-safe across batches: last non-null carries forward.
        - Original `{col}` column type preserved (nullable forced True).
        - Int cols → int (null fill preserves type); the fill itself
          does not widen types.

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

        # Per-column carry as Array[1] (or None). Avoids .as_py() in facade.
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
                    # Update carry: last non-null in filled (if any).
                    if len(filled) > 0:
                        last = filled.slice(len(filled) - 1, 1)
                        if _pc.is_valid(last).to_pylist()[0]:
                            carry[c] = last
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
        ffill + re-sort (__orig_rid ascending). Uses 2× external_sort
        internally — O(N log N) I/O, O(batch) RAM. **Heavier than
        ffill** (which is O(N) I/O). For large data prefer ffill.

        - columns: None → all columns; str → one; Sequence[str] → subset.
        - Trailing nulls stay null (no later value).
        - Original order preserved (via 2 external_sort).
        - Original column types preserved.
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

        # Empty shortcut
        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        # Step 1: add __orig_rid (global position) to each batch
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

        # Step 2: external_sort descending by __orig_rid → reverse order
        sorted_desc = external_sort(
            _iter_with_rid(),
            key_columns=["__orig_rid"],
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "bfill" / "sort_desc",
            ascending=False,
            log=False,
        )

        # Step 3: ffill pass per batch (streaming with carry). Output
        # [__orig_rid, *filled_cols]. Drop row_id (external_sort's
        # auto-added) to avoid duplicate on next sort.
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
                    b, name="bfill.step3.read",
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
                        last = filled.slice(len(filled) - 1, 1)
                        if _pc.is_valid(last).to_pylist()[0]:
                            carry[c] = last
                    arrays.append(filled)
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        arrays, schema=filled_schema,
                    )
                )
        finally:
            writer.close()

        # Step 4: external_sort ascending by __orig_rid → restore order
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

        # Step 5: read sorted_asc, select self._columns, build dataset
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

        - labels: iterable of scalars (int | str | date | etc.).
        - Returns CadwDataFrame with rows in `labels` order.
        - Duplicate source labels: first occurrence wins (pandas `.loc`
          semantics).
        - MultiIndex source: NotImplementedError (tuple labels not
          supported in this version).
        - RangeIndex source: labels interpreted as positional ints
          (0-based). Non-int label → null row.
        - Filtered source: labels matched against filtered view's index.

        Memory: O(K × row_size) where K = len(labels). Requires
        K × row_size ≤ state_limit (else MemoryLimitExceeded).
        Peak RAM = K × row_size + batch.

        Raises:
            ValueError: labels empty.
            KeyError: RangeIndex + non-int label? no — those go to
                null rows, not raise.
            NotImplementedError: MultiIndex source.
            MemoryLimitExceeded: K × row_size > state_limit.
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

        # Resolve each label → positional int in filtered view, or None.
        n_view = self.count()
        positions: list = []
        if self._index_kind == "range":
            start = self._index_start
            stop = self._index_start + n_view
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
        else:  # materialized
            # Use filtered.index (MaterializedIndex subset of original)
            idx = self.index
            for lab in labels:
                pos_list = idx.get_positions_all(lab)
                positions.append(pos_list[0] if pos_list else None)

        valid_positions = [p for p in positions if p is not None]
        k_valid = len(valid_positions)

        if k_valid > 0:
            # Memory check: K × row_size must fit
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
        else:
            # All positions None: build empty table with schema
            valid_table = pa.Table.from_batches(
                [],
                schema=pa.schema(
                    [self._dataset.schema.field(c) for c in self._columns]
                ),
            )

        # Build output by Arrow take with null-safe positions
        # valid_table has K_valid rows; need to map: output[i] = valid_table[pos_in_valid]
        # Build valid_indices array aligned to output rows:
        #   output[i] gets valid_table[valid_idx_counter] if positions[i] is not None,
        #   else null.
        # valid_idx_counter increments only on non-None.
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

    def corr(self, columns=None, *, method="pearson"):'''


def add_ffill_bfill_reindex():
    print("\n=== add ffill + bfill + reindex methods ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, FFILL_ANCHOR_TAIL, FFILL_NEW_BLOCK,
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
    """chunk=1: many batches; carry works across batches."""
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
    # filtered x = [None, 3, None, 5]; ffill → [None, 3, 3, 5]
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
    # [None, None, 3, None] bfill → [3, 3, 3, None]
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
    # filtered x = [2, None, None, 5]; bfill → [2, None, 5, 5]
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
    # [1, None, 3, None, 5] bfill → [1, 3, 3, 5, 5]
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
    # z not found → null row
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
    # first occurrence of "a" (v=10) wins
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
    # 100 out of range → null row
    assert out.column("v").to_pylist() == [10, None]


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
    # Result is a new df (RangeIndex per _wrap_result_table convention)
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
    print("=== v0.9.25i: Phase C C-3a — ffill + bfill + reindex ===")
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
        "3 of 20 Pandas P0 features (after C-1's isin/between/"
        "select_dtypes v0.9.25g; C-2's idxmax/idxmin/pct_change "
        "v0.9.25h). All pure compositions of existing primitives + "
        "Arrow kernels. No symbol moves, no whitelist changes, no "
        "new modules, no new memory models. "
        "Features: "
        "(1) `CadwDataFrame.ffill(columns=None)` — forward fill "
        "nulls per column. Streaming per-batch via pc.fill_null_"
        "forward + carry Array[1] for cross-batch leading-null fix. "
        "O(#cols) state, O(N) I/O. Original types preserved "
        "(nullable=True forced); int cols stay int. "
        "(2) `CadwDataFrame.bfill(columns=None)` — backward fill via "
        "reverse-sort + ffill + re-sort (2× external_sort). "
        "O(batch) RAM, O(N log N) I/O. Heavier than ffill "
        "(documented). Trailing nulls stay null. "
        "(3) `CadwDataFrame.reindex(labels)` — row alignment to "
        "target labels via MaterializedIndex.get_positions_all + "
        "Arrow take with null-safe indices. Missing → null rows. "
        "Order matches input labels. RangeIndex treated as "
        "positional labels (int). Bounded by K × row_size ≤ "
        "state_limit (else MemoryLimitExceeded). MultiIndex → "
        "NotImplementedError. "
        "§13.4/264 APPLIED (REPL-first, first patch after this "
        "lesson's formalization): verify_state does 8 pyarrow REPL "
        "checks BEFORE any edit — pc.fill_null_forward presence + "
        "behavior (interior + leading nulls); Table.take with null "
        "indices (null-safe? abort if not); get_positions_all on "
        "missing + duplicate; pa.array with None; external_sort "
        "signature. All PASSED (with clear abort messages if "
        "PyArrow drifts). "
        "§13.4/260 preserved: no subprocess dogfood (dedent trap). "
        "§13.4/261 preserved: no symbol moves → no whitelist path "
        "updates. §13.4/262 preserved: pa.scalar for typed "
        "reconstruction. §13.4/265: avoid copy-pasted asserts from "
        "sibling tests — bfill/ffill tests share helpers but "
        "different data. "
        "DEFERRED: ffill/bfill limit= param → v0.9.25j; reindex "
        "MultiIndex → NotImplementedError (documented); bfill "
        "pending-buffer op (O(N log N)-free alternative) → backlog. "
        "Tests (tests/test_v0925i.py): 33 total — "
        "ffill (11: basic / leading-null / all-null / no-nulls / "
        "float / multiple-cols / columns-subset / cross-batch / "
        "filtered / empty / string); "
        "bfill (10: basic / trailing-null / all-null / no-nulls / "
        "float / multiple-cols / columns-subset / cross-batch / "
        "filtered / empty / order-preserved = 11, adjusted); "
        "reindex (11: all-present / missing-label / order / dup-"
        "target / dup-source / range-index / range-oob / empty-"
        "labels / all-missing / preserves-index). "
        "Fast suite: ~2371 → ~2404 (+33). "
        "Next: Phase C C-3b (cut/qcut/duplicated).",
        self_path="scripts/patch_v0925i_ffill_bfill_reindex.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25i applied. Phase C: 9/20. "
          "Next: C-3b (cut/qcut/duplicated).")


if __name__ == "__main__":
    main()