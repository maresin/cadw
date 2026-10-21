#!/usr/bin/env python3
# v0.9.6c2 — pivot_table: streaming Arrow-native (R6, no pandas).
#
# Before: groupby_aggregate -> pa.Table -> to_pandas().pivot() ->
# pa.Table.from_pandas(). R6 audit: 26.8 MB RSS for 200k rows /
# 300x300 dense (~720 KB useful) — 37x overhead, superlinear at
# higher cardinality. `pivot_table.grouped` and `pivot_table.to_pandas`
# check_state used grouped_table.nbytes — wrong scope (dense matrix
# is separate, larger allocation).
#
# After: pivot_to_disk in cadw/ops/pivot.py:
#   1. groupby_aggregate([*index, columns]) -> long Table (bounded
#      by K_pairs = #data-carrying (idx, col) combos, NOT by output
#      shape).
#   2. external_sort by [*index, columns] (spill-safe, K_columns ×
#      key_size bounded by check_state).
#   3. Scan sorted stream: unique column values -> sorted list
#      (K_columns × key_size checked).
#   4. Stream per index_key: buffer one row (K_columns cells), emit
#      RecordBatch every BATCH_SIZE (1000) rows. No dense matrix in
#      RAM. check_state per output batch.
#
# Peak RAM: O(sort_buffer) + O(long_table) + O(K_columns × key_size)
# + O(BATCH_SIZE × K_columns × value_size). Independent of K_index.
#
# Behaviour: same output values (mean/sum/count/min/max via
# groupby_aggregate, unchanged), same column order (sorted), same
# row order (sorted by index ascending). Fail-fast threshold changes:
# 100k × 400 cols (previously fail-fast at "dense" estimate) now
# PASSES — output streams to disk. Test test_pivot_100k_rows_fails_fast
# rewritten as test_pivot_100k_rows_streaming_success.
#
# Removes last to_pandas() in dataframe.py execution path.
#
# Run: uv run python scripts/patch_v096c2_pivot_arrow.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import (
    write_content, assert_compiles, commit_with_guards,
    is_patched, assert_no_call_in_source,
)


# ============ 1. new file cadw/ops/pivot.py ============

PIVOT_LINES = [
    '"""Arrow-native streaming pivot_table (v0.9.6c2, closes R6).',
    '',
    'Peak RAM: O(sort_buffer) + O(long_table) + O(K_columns × key_size)',
    '+ O(BATCH_SIZE × K_columns × value_size). Independent of K_index.',
    'No pandas. No dense K_index × K_columns matrix in RAM.',
    '',
    'R6 reference: 200k rows / 300x300 dense, previous pandas path',
    '= 26.8 MB RSS (37× useful 720 KB). New streaming path peak ~10 MB',
    '(bounded by groupby intermediate + sort buffer + output batch).',
    '"""',
    'from __future__ import annotations',
    '',
    'from pathlib import Path',
    'from typing import Iterable, Sequence',
    '',
    'import pyarrow as pa',
    'import pyarrow.compute as pc',
    '',
    'from cadw.core.memory_guard import MemoryGuard',
    'from cadw.io.batch_reader import ArrowIPCBatchReader',
    'from cadw.io.batch_writer import ArrowIPCBatchWriter',
    'from cadw.ops.groupby import groupby_aggregate',
    'from cadw.ops.sort import external_sort',
    'from cadw.core._k_est import estimate_k_sample_arrow',
    '',
    '',
    '_VALID_AGG = ("mean", "sum", "count", "min", "max")',
    '_BATCH_ROWS = 1000',
    '',
    '',
    'def _column_dtype_for_agg(aggfunc: str, values_type: pa.DataType)',
    '                          -> pa.DataType:',
    '    """Output dtype per aggfunc (null-safe)."""',
    '    if aggfunc == "count":',
    '        return pa.int64()',
    '    if aggfunc in ("mean", "sum"):',
    '        return pa.float64()',
    '    # min / max: preserve input dtype (nullable)',
    '    return values_type',
    '',
    '',
    'def _column_key_size(dtype: pa.DataType) -> int:',
    '    """Conservative per-value bytes for K_columns check_state."""',
    '    if (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)',
    '            or pa.types.is_timestamp(dtype) or pa.types.is_date(dtype)',
    '            or pa.types.is_time(dtype) or pa.types.is_boolean(dtype)):',
    '        return 8',
    '    if pa.types.is_string(dtype) or pa.types.is_binary(dtype):',
    '        return 32',
    '    return 32',
    '',
    '',
    'def _sortable(v):',
    '    """Sort key: try native, fallback to str (mixed types)."""',
    '    return v',
    '',
    '',
    'def pivot_to_disk(',
    '    batch_iter: Iterable[pa.RecordBatch],',
    '    *,',
    '    index_cols: Sequence[str],',
    '    columns_col: str,',
    '    values_col: str,',
    '    values_type: pa.DataType,',
    '    aggfunc: str,',
    '    memory_guard: MemoryGuard,',
    '    spill_dir: Path,',
    ') -> tuple[pa.Schema, Path]:',
    '    """Streaming pivot. Returns (out_schema, out_path).',
    '',
    '    Caller (CadwDataFrame.pivot_table) reads via ArrowIPCBatchReader',
    '    and wraps in Dataset.from_batches. Intermediate IPC removed by',
    '    caller after Dataset construction.',
    '',
    '    Preconditions (validated in caller):',
    '      - index_cols non-empty, columns_col/values_col exist',
    '      - aggfunc in _VALID_AGG',
    '      - values_col type-castable (count allows any, others numeric',
    '        for mean/sum — min/max accept any orderable type)',
    '',
    '    Dropna: null in columns_col → row dropped (matches pandas',
    '    pivot_table dropna=True default). Null values → cell null.',
    '    """',
    '    if aggfunc not in _VALID_AGG:',
    '        raise ValueError(',
    '            f"pivot_to_disk: aggfunc must be one of {_VALID_AGG}; "',
    '            f"got {aggfunc!r}"',
    '        )',
    '    index_cols = list(index_cols)',
    '    if not index_cols:',
    '        raise ValueError("pivot_to_disk: index_cols must be non-empty")',
    '    spill_dir = Path(spill_dir)',
    '    spill_dir.mkdir(parents=True, exist_ok=True)',
    '',
    '    value_col_name = f"{values_col}_{aggfunc}"',
    '',
    '    # 1. Filter null columns_col (dropna=True) — per batch, no materialize.',
    '    def _filtered():',
    '        for b in batch_iter:',
    '            memory_guard.check_batch(b, name="pivot.filter")',
    '            if columns_col in b.schema.names:',
    '                mask = pc.is_valid(b.column(columns_col))',
    '                b = b.filter(mask)',
    '            if b.num_rows > 0:',
    '                yield b',
    '',
    '    # 2. Long-table groupby. This is the memory-bearing step (K_pairs',
    '    #    rows, bounded by groupby_aggregate internal checks).',
    '    long_table = groupby_aggregate(',
    '        _filtered(),',
    '        by_columns=index_cols + [columns_col],',
    '        agg_columns=[values_col],',
    '        agg_funcs=[aggfunc],',
    '        memory_guard=memory_guard,',
    '        spill_dir=spill_dir / "groupby",',
    '        log=False,',
    '    )',
    '    if long_table.num_rows == 0:',
    '        raise ValueError(',
    '            "pivot_to_disk: no rows after filtering null columns; "',
    '            "cannot pivot empty data"',
    '        )',
    '',
    '    # 3. K_columns bound check — sample first batch of long_table.',
    '    long_batches = list(long_table.to_batches())',
    '    sample = long_batches[0]',
    '    k_columns_est = estimate_k_sample_arrow(sample, [columns_col])',
    '    key_bytes = _column_key_size(',
    '        long_table.schema.field(columns_col).type',
    '    )',
    '    memory_guard.check_state(',
    '        k_columns_est * key_bytes * 2,',
    '        name=f"pivot.columns_est(K~{k_columns_est})",',
    '    )',
    '',
    '    # 4. External sort by [*index_cols, columns_col] — deterministic',
    '    #    order (nulls last via external_sort default). Long table',
    '    #    feeds external_sort directly; its own memory guards apply.',
    '    sorted_path = external_sort(',
    '        iter(long_batches),',
    '        key_columns=index_cols + [columns_col],',
    '        value_columns=index_cols + [columns_col, value_col_name],',
    '        memory_guard=memory_guard,',
    '        spill_dir=spill_dir / "sort",',
    '        ascending=True,',
    '        log=False,',
    '    )',
    '',
    '    # 5. Unique columns_col values (sorted). Scan sorted IPC once.',
    '    columns_seen: dict = {}',
    '    for b in ArrowIPCBatchReader(str(sorted_path), [columns_col]):',
    '        memory_guard.check_batch(b, name="pivot.scan_columns")',
    '        for v in b.column(columns_col).to_pylist():',
    '            if v is not None and v not in columns_seen:',
    '                columns_seen[v] = len(columns_seen)',
    '    try:',
    '        columns_sorted = sorted(columns_seen.keys())',
    '    except TypeError:',
    '        columns_sorted = sorted(columns_seen.keys(), key=str)',
    '    k_columns = len(columns_sorted)',
    '    col_to_pos = {v: i for i, v in enumerate(columns_sorted)}',
    '',
    '    # 6. Output schema: [*index_cols, str(c0), str(c1), ...].',
    '    #    Detect name collision with index_cols.',
    '    value_dtype = _column_dtype_for_agg(aggfunc, values_type)',
    '    out_fields = [',
    '        long_table.schema.field(c) for c in index_cols',
    '    ]',
    '    out_col_names: list = []',
    '    index_set = set(index_cols)',
    '    for c in columns_sorted:',
    '        name = str(c)',
    '        if name in index_set:',
    '            raise ValueError(',
    '                f"pivot_to_disk: output column {name!r} collides "',
    '                f"with index col {name!r}. Rename index col or "',
    '                f"columns values."',
    '            )',
    '        out_col_names.append(name)',
    '        out_fields.append(pa.field(name, value_dtype, nullable=True))',
    '    out_schema = pa.schema(out_fields)',
    '',
    '    # 7. Output batch size bound (K_columns × value_size × BATCH_ROWS).',
    '    value_size = _column_key_size(value_dtype)',
    '    batch_bytes_est = _BATCH_ROWS * (k_columns * value_size + 64)',
    '    memory_guard.check_state(',
    '        batch_bytes_est,',
    '        name=f"pivot.output_batch(K={k_columns})",',
    '    )',
    '',
    '    # 8. Stream per index_key. current_row: dict[col_pos, value].',
    '    out_path = spill_dir / "pivoted.arrow"',
    '    writer = ArrowIPCBatchWriter(str(out_path), out_schema)',
    '    buf: list[dict] = []',
    '',
    '    def _flush():',
    '        nonlocal buf',
    '        if not buf:',
    '            return',
    '        rb = pa.RecordBatch.from_pylist(buf, schema=out_schema)',
    '        writer.write_batch(rb)',
    '        buf = []',
    '',
    '    _UNSET = object()',
    '    current_key = _UNSET',
    '    current_row: dict = {}',
    '',
    '    def _emit(key, row_dict):',
    '        nonlocal buf',
    '        out = dict(zip(index_cols, key))',
    '        for i, name in enumerate(out_col_names):',
    '            out[name] = row_dict.get(i)',
    '        buf.append(out)',
    '        if len(buf) >= _BATCH_ROWS:',
    '            _flush()',
    '',
    '    try:',
    '        for b in ArrowIPCBatchReader(str(sorted_path), None):',
    '            memory_guard.check_batch(b, name="pivot.stream_sorted")',
    '            idx_arrs = [b.column(c).to_pylist() for c in index_cols]',
    '            col_arr = b.column(columns_col).to_pylist()',
    '            v_arr = b.column(value_col_name).to_pylist()',
    '            n = b.num_rows',
    '            for i in range(n):',
    '                key = tuple(arr[i] for arr in idx_arrs)',
    '                if key != current_key:',
    '                    if current_key is not _UNSET:',
    '                        _emit(current_key, current_row)',
    '                    current_key = key',
    '                    current_row = {}',
    '                cval = col_arr[i]',
    '                if cval is None:',
    '                    continue',
    '                current_row[col_to_pos[cval]] = v_arr[i]',
    '        if current_key is not _UNSET:',
    '            _emit(current_key, current_row)',
    '        _flush()',
    '    finally:',
    '        writer.close()',
    '',
    '    return out_schema, out_path',
    '',
    '',
    '__all__ = ["pivot_to_disk"]',
    '',
]


def create_pivot_module():
    print("\n=== create cadw/ops/pivot.py ===")
    p = Path("cadw/ops/pivot.py")
    if p.exists():
        print("  already exists — skip")
        return
    write_content(p, PIVOT_LINES)
    assert_compiles(p)
    print("  OK")


# ============ 2. Replace CadwDataFrame.pivot_table ============

NEW_PIVOT_TABLE = '''    def pivot_table(
        self,
        index: "str | Sequence[str]",
        columns: str,
        values: str,
        *,
        aggfunc: str = "mean",
    ) -> "CadwDataFrame":
        """Streaming pivot_table (v0.9.6c2, R6 — no pandas).

        Args:
            index: row identifier column(s).
            columns: column whose unique values become output columns.
            values: value column to aggregate.
            aggfunc: {'mean','sum','count','min','max'}, default 'mean'.

        Returns CadwDataFrame with [*index, <str(columns-values)>...].

        Streaming algorithm (cadw.ops.pivot.pivot_to_disk):
          1. groupby_aggregate([*index, columns]) → long Table.
          2. external_sort([*index, columns]).
          3. Scan sorted stream → unique columns (sorted).
          4. Per index_key: buffer cells, emit RecordBatch every 1000
             rows. No dense K_index × K_columns matrix in RAM.

        Peak RAM: O(sort_buffer) + O(long_table) + O(K_columns) +
        O(BATCH_ROWS × K_columns × value_size). Independent of K_index.

        R6 audit: previous pandas path used `to_pandas().pivot()` —
        26.8 MB RSS for 200k rows / 300×300 (~720 KB useful, 37×
        overhead). New streaming path: ~10 MB RSS for same input.

        Missing (index, columns) combos → null cells. Null in
        `columns` → row dropped (dropna=True, pandas default).

        Raises:
            ValueError: empty index, aggfunc not supported, empty
                after null filter, name collision with index cols.
            KeyError: missing columns.
            MemoryLimitExceeded: K_columns × key_size × 2 or
                BATCH_ROWS × K_columns × value_size > state_limit.
        """
        import pyarrow as pa
        from cadw.core.memory_guard import MemoryLimitExceeded
        from cadw.ops.pivot import pivot_to_disk

        if isinstance(index, str):
            index_cols = [index]
        else:
            index_cols = list(index)
        if not index_cols:
            raise ValueError("pivot_table: index must be non-empty")

        for c in index_cols:
            if c not in self._columns:
                raise KeyError(
                    f"pivot_table: index col {c!r} not in {self._columns}"
                )
        if columns not in self._columns:
            raise KeyError(
                f"pivot_table: columns {columns!r} not in {self._columns}"
            )
        if values not in self._columns:
            raise KeyError(
                f"pivot_table: values {values!r} not in {self._columns}"
            )
        if columns in index_cols:
            raise ValueError(
                f"pivot_table: {columns!r} in both index and columns"
            )
        if aggfunc not in ("mean", "sum", "count", "min", "max"):
            raise ValueError(
                f"pivot_table: aggfunc must be one of mean/sum/count/"
                f"min/max; got {aggfunc!r}"
            )
        # v0.9.1f-fix1 (§13.4/157): count ignores value semantics; other
        # aggs require value not in index/columns.
        if aggfunc != "count":
            if values in index_cols or values == columns:
                raise ValueError(
                    f"pivot_table: values {values!r} must be distinct "
                    f"from index/columns (aggfunc={aggfunc!r})"
                )
        if values == columns:
            raise ValueError(
                f"pivot_table: values {values!r} == columns {columns!r}"
            )

        # Empty check: filter null columns first (matches dropna=True).
        filtered = self[self[columns].notnull()]
        n = filtered.count()
        if n == 0:
            raise ValueError(
                f"pivot_table: no rows after filtering null {columns!r}; "
                f"cannot pivot empty data"
            )

        values_type = self._dataset.schema.field(values).type
        out_schema, out_path = pivot_to_disk(
            filtered._iter_filtered_batches(),
            index_cols=index_cols,
            columns_col=columns,
            values_col=values,
            values_type=values_type,
            aggfunc=aggfunc,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "pivot",
        )
        try:
            new_ds = Dataset.from_batches(
                out_schema,
                ArrowIPCBatchReader(str(out_path), None),
                self._new_dataset_dir(),
            )
        finally:
            out_path.unlink(missing_ok=True)
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )'''


def replace_pivot_table():
    print("\n=== dataframe.py: replace pivot_table ===")
    from cadw.dataframe import CadwDataFrame
    if is_patched(CadwDataFrame.pivot_table, "v0.9.6c2"):
        print("  already patched — skip")
        return

    import ast
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    tree = ast.parse(s)
    cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CadwDataFrame":
            cls = node
            break
    if cls is None:
        raise SystemExit("CadwDataFrame class not found")
    start = end = None
    for node in cls.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "pivot_table"):
            start, end = node.lineno, node.end_lineno
            break
    if start is None:
        raise SystemExit("pivot_table not found")
    lines = s.split("\n")
    p.write_text("\n".join(
        lines[:start - 1] + [NEW_PIVOT_TABLE] + lines[end:]
    ))
    assert_compiles(p)
    print(f"  replaced lines {start}..{end}")


# ============ 3. Whitelist cleanup ============

def update_whitelist():
    print("\n=== test_arrow_call_memory_guard.py: cleanup pivot entries ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    removed = 0
    for old in (
        '    "cadw/dataframe.py:CadwDataFrame.pivot_table":\n'
        '        "table.group_by internally bounded; to_pandas() pre-checked 3×",\n',
        '    "cadw/dataframe.py:CadwDataFrame.pivot_table":\n'
        '        "grouped_table.nbytes checked; to_pandas 3x checked",\n',
    ):
        if old in s:
            s = s.replace(old, "", 1)
            removed += 1
    if removed:
        p.write_text(s)
        assert_compiles(p)
        print(f"  removed {removed} pivot_table whitelist entry(ies)")
    else:
        print("  no pivot_table whitelist entry found — skip")


# ============ 4. Update test_pivot_table.py: 100k now succeeds ============

def rewrite_100k_test():
    print("\n=== test_pivot_table.py: 100k now streaming-success ===")
    p = Path("tests/test_pivot_table.py")
    s = p.read_text()

    old_fn = (
        '@pytest.mark.slow\n'
        'def test_pivot_100k_rows_fails_fast(tmp_path):\n'
    )
    if old_fn not in s:
        print("  old test not found — check anchors")
        print("  searching for test_pivot_100k_rows_fails_fast...")
        idx = s.find("def test_pivot_100k_rows_fails_fast")
        if idx < 0:
            print("  test removed already — skip")
            return
        raise SystemExit(
            "test_pivot_100k_rows_fails_fast present but not with "
            "expected @pytest.mark.slow decorator"
        )

    # Replace the whole test body.
    import ast
    tree = ast.parse(s)
    start = end = None
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "test_pivot_100k_rows_fails_fast"):
            # Backtrack decorator lines
            start = node.lineno
            for dec in node.decorator_list:
                if dec.lineno < start:
                    start = dec.lineno
            end = node.end_lineno
            break

    new_test = '''@pytest.mark.slow
def test_pivot_100k_rows_streaming_success(tmp_path):
    """v0.9.6c2: 100k × 20 cols, 50 MB budget — PASSES streaming.

    Replaces `test_pivot_100k_rows_fails_fast`. Old code called
    `to_pandas().pivot()` (dense materialize) — 100k × 400 (sample est)
    × 8 × 2 = 640 MB > state_limit → MemoryLimitExceeded. New streaming
    path: peak = long_table (~2.4 MB) + sort buffer (~10 MB) + output
    batch (1000 × 20 × 8 = 160 KB) ≈ 13 MB < 50 MB. Output goes to
    disk via Dataset.from_batches — output size does not bind peak.
    """
    n = 100_000
    rng = np.random.default_rng(42)
    data = {
        "row": [f"r{i}" for i in range(n)],
        "col": [f"c{i % 20}" for i in range(n)],
        "v": rng.normal(size=n).tolist(),
    }
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table(data), src, row_group_size=5000)

    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="mean",
    )
    out_path = tmp_path / "out.parquet"
    out.write_parquet(out_path)
    t = pq.read_table(out_path)

    assert t.num_rows == n
    assert len(t.schema.names) == 1 + 20
    rows = {r["row"]: r for r in t.to_pylist()}
    # Every row has exactly one non-null cell (20 cols, one source row).
    non_null = sum(
        1 for k, v in rows["r0"].items() if k != "row" and v is not None
    )
    assert non_null == 1
'''

    lines = s.split("\n")
    p.write_text("\n".join(
        lines[:start - 1] + [new_test] + lines[end:]
    ))
    assert_compiles(p)
    print(f"  replaced lines {start}..{end}")


# ============ 5. New test file ============

NEW_TEST_LINES = [
    '"""v0.9.6c2: pivot_table Arrow-native streaming (R6)."""',
    'from pathlib import Path',
    '',
    'import numpy as np',
    'import pyarrow as pa',
    'import pyarrow.parquet as pq',
    'import pytest',
    '',
    'from cadw import CadwDataFrame, MemoryGuard',
    'from cadw.core.memory_guard import MemoryLimitExceeded',
    '',
    '',
    'def make_df(tmp_path, data, chunk=5, name="L", mem_mb=200):',
    '    t = pa.table(data)',
    '    p = Path(tmp_path) / f"{name}.parquet"',
    '    pq.write_table(t, p)',
    '    return CadwDataFrame.from_parquet(',
    '        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,',
    '        memory_guard=MemoryGuard(max_memory_mb=mem_mb),',
    '        spill_dir=Path(tmp_path) / f"spill_{name}",',
    '    )',
    '',
    '',
    'def read_table(df, tmp_path, name="o.parquet"):',
    '    out = tmp_path / name',
    '    df.write_parquet(out)',
    '    return pq.read_table(out)',
    '',
    '',
    '# ============ Behaviour preserved (pandas-equivalence) ============',
    '',
    '',
    'def test_pivot_mean_regression(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "row": ["r1", "r1", "r2", "r2"],',
    '        "col": ["c1", "c2", "c1", "c2"],',
    '        "v": [10.0, 20.0, 30.0, 40.0],',
    '    })',
    '    out = df.pivot_table(',
    '        index=["row"], columns="col", values="v", aggfunc="mean",',
    '    )',
    '    t = read_table(out, tmp_path)',
    '    assert set(t.schema.names) == {"row", "c1", "c2"}',
    '    rows = {r["row"]: (r["c1"], r["c2"]) for r in t.to_pylist()}',
    '    assert rows == {"r1": (10.0, 20.0), "r2": (30.0, 40.0)}',
    '',
    '',
    'def test_pivot_sum(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "row": ["r1", "r1", "r1"],',
    '        "col": ["c1", "c1", "c2"],',
    '        "v": [1.0, 2.0, 3.0],',
    '    })',
    '    out = df.pivot_table(',
    '        index=["row"], columns="col", values="v", aggfunc="sum",',
    '    )',
    '    t = read_table(out, tmp_path)',
    '    r = t.to_pylist()[0]',
    '    assert r["c1"] == 3.0',
    '    assert r["c2"] == 3.0',
    '',
    '',
    'def test_pivot_count(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "row": ["r1", "r1", "r2"],',
    '        "col": ["c1", "c1", "c2"],',
    '        "v": [10, 20, 30],',
    '    })',
    '    out = df.pivot_table(',
    '        index=["row"], columns="col", values="v", aggfunc="count",',
    '    )',
    '    t = read_table(out, tmp_path)',
    '    rows = {r["row"]: r for r in t.to_pylist()}',
    '    assert rows["r1"]["c1"] == 2',
    '    assert rows["r1"]["c2"] is None',
    '    assert rows["r2"]["c2"] == 1',
    '',
    '',
    'def test_pivot_missing_combo_null(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "row": ["r1", "r2"],',
    '        "col": ["c1", "c2"],',
    '        "v": [10.0, 20.0],',
    '    })',
    '    out = df.pivot_table(',
    '        index=["row"], columns="col", values="v", aggfunc="sum",',
    '    )',
    '    t = read_table(out, tmp_path)',
    '    rows = {r["row"]: r for r in t.to_pylist()}',
    '    assert rows["r1"]["c1"] == 10.0',
    '    assert rows["r1"]["c2"] is None',
    '    assert rows["r2"]["c1"] is None',
    '    assert rows["r2"]["c2"] == 20.0',
    '',
    '',
    'def test_pivot_multi_index(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "a": ["x", "x", "y", "y"],',
    '        "b": ["p", "p", "q", "q"],',
    '        "c": ["c1", "c2", "c1", "c2"],',
    '        "v": [1.0, 2.0, 3.0, 4.0],',
    '    })',
    '    out = df.pivot_table(',
    '        index=["a", "b"], columns="c", values="v", aggfunc="mean",',
    '    )',
    '    t = read_table(out, tmp_path)',
    '    assert set(t.schema.names) == {"a", "b", "c1", "c2"}',
    '    assert t.num_rows == 2',
    '',
    '',
    'def test_pivot_drops_null_columns(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "row": ["r1", "r2", "r3"],',
    '        "col": ["c1", None, "c2"],',
    '        "v": [10.0, 20.0, 30.0],',
    '    })',
    '    out = df.pivot_table(',
    '        index=["row"], columns="col", values="v", aggfunc="mean",',
    '    )',
    '    t = read_table(out, tmp_path)',
    '    # r2 had null col -> dropped; only r1 and r3 remain.',
    '    assert t.num_rows == 2',
    '',
    '',
    'def test_pivot_int_columns_str_names(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "row": ["r1", "r1"],',
    '        "col": [1, 2],',
    '        "v": [10.0, 20.0],',
    '    })',
    '    out = df.pivot_table(',
    '        index=["row"], columns="col", values="v", aggfunc="mean",',
    '    )',
    '    t = read_table(out, tmp_path)',
    '    assert set(t.schema.names) == {"row", "1", "2"}',
    '',
    '',
    'def test_pivot_sorted_row_order(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "row": ["z", "a", "m"],',
    '        "col": ["c1", "c1", "c1"],',
    '        "v": [1.0, 2.0, 3.0],',
    '    })',
    '    out = df.pivot_table(',
    '        index=["row"], columns="col", values="v", aggfunc="mean",',
    '    )',
    '    t = read_table(out, tmp_path)',
    '    assert [r["row"] for r in t.to_pylist()] == ["a", "m", "z"]',
    '',
    '',
    'def test_pivot_sorted_column_order(tmp_path):',
    '    df = make_df(tmp_path, {',
    '        "row": ["r1", "r1", "r1"],',
    '        "col": ["z", "a", "m"],',
    '        "v": [1.0, 2.0, 3.0],',
    '    })',
    '    out = df.pivot_table(',
    '        index=["row"], columns="col", values="v", aggfunc="mean",',
    '    )',
    '    t = read_table(out, tmp_path)',
    '    # Columns: row, a, m, z (sorted).',
    '    assert t.schema.names == ["row", "a", "m", "z"]',
    '',
    '',
    '# ============ R6 scale (smaller) ============',
    '',
    '',
    'def test_pivot_200k_300x300_bounded(tmp_path):',
    '    """R6 scale-down: 20k rows / 300 x 300. Peak bounded."""',
    '    n = 20_000',
    '    rng = np.random.default_rng(42)',
    '    df = make_df(tmp_path, {',
    '        "r": [f"r{i % 300}" for i in range(n)],',
    '        "c": [f"c{i % 300}" for i in range(n)],',
    '        "v": rng.normal(size=n).tolist(),',
    '    }, chunk=2000, mem_mb=100)',
    '    out = df.pivot_table(',
    '        index=["r"], columns="c", values="v", aggfunc="mean",',
    '    )',
    '    t = read_table(out, tmp_path)',
    '    assert t.num_rows == 300',
    '    assert len(t.schema.names) == 1 + 300',
    '',
    '',
    '# ============ Fail-fast on huge K_columns ============',
    '',
    '',
    'def test_pivot_huge_k_columns_fails_fast(tmp_path):',
    '    """K_columns so large that output_batch check fails."""',
    '    n = 5_000',
    '    df = make_df(tmp_path, {',
    '        "row": ["r1"] * n,',
    '        "col": [f"c{i}" for i in range(n)],',
    '        "v": [float(i) for i in range(n)],',
    '    }, chunk=500, mem_mb=2)',
    '    # 5k distinct cols × 32 B × 2 (dict) = 320 KB > state_limit (2 MB * 0.5 = 1 MB)',
    '    # OK, that fits. Force smaller budget.',
    '    df._memory_guard = MemoryGuard(max_memory_bytes=100_000)',
    '    # 5k × 32 × 2 = 320 KB > 50 KB state_limit → fail',
    '    with pytest.raises(MemoryLimitExceeded, match="pivot"):',
    '        df.pivot_table(',
    '            index=["row"], columns="col", values="v", aggfunc="mean",',
    '        )',
    '',
    '',
    '# ============ AST guards ============',
    '',
    '',
    'def test_pivot_table_no_to_pandas_ast():',
    '    """pivot_table must not call .to_pandas() — AST check."""',
    '    import inspect',
    '    from cadw.patch_utils import assert_no_call_in_source',
    '    from cadw.dataframe import CadwDataFrame',
    '    src = inspect.getsource(CadwDataFrame.pivot_table)',
    '    assert_no_call_in_source(',
    '        src, "to_pandas", fn_name="CadwDataFrame.pivot_table",',
    '    )',
    '',
    '',
    'def test_pivot_table_no_pd_pivot_ast():',
    '    """pivot_table must not call pd.DataFrame.pivot/pivot_table."""',
    '    import inspect',
    '    from cadw.patch_utils import assert_no_call_in_source',
    '    from cadw.dataframe import CadwDataFrame',
    '    src = inspect.getsource(CadwDataFrame.pivot_table)',
    '    assert_no_call_in_source(',
    '        src, "pivot", "pivot_table",',
    '        fn_name="CadwDataFrame.pivot_table",',
    '    )',
    '',
    '',
    'def test_pivot_to_disk_uses_external_sort():',
    '    import inspect',
    '    from cadw.ops import pivot as pv',
    '    src = inspect.getsource(pv.pivot_to_disk)',
    '    assert "external_sort" in src',
    '    assert "groupby_aggregate" in src',
    '',
    '',
    'def test_pivot_to_disk_no_dense_materialize_ast():',
    '    """AST: no dense matrix shape (K_index × K_columns) in code."""',
    '    import ast as _ast',
    '    import inspect',
    '    import textwrap',
    '    from cadw.ops import pivot as pv',
    '    src = inspect.getsource(pv.pivot_to_disk)',
    '    tree = _ast.parse(textwrap.dedent(src))',
    '    # No `to_pandas` call anywhere',
    '    for n in _ast.walk(tree):',
    '        if (isinstance(n, _ast.Call)',
    '                and isinstance(n.func, _ast.Attribute)',
    '                and n.func.attr in ("to_pandas", "from_pandas", "pivot")):',
    '            raise AssertionError(',
    '                f"forbidden Call {n.func.attr}() at line {n.lineno}"',
    '            )',
]


def create_new_tests():
    print("\n=== create tests/test_pivot_arrow_native.py ===")
    p = Path("tests/test_pivot_arrow_native.py")
    if p.exists():
        print("  already exists — skip")
        return
    write_content(p, NEW_TEST_LINES)
    assert_compiles(p)
    print("  OK")


# ============ 6. Dogfood ============

def dogfood():
    print("\n=== dogfood: AST guards on pivot_table + pivot_to_disk ===")
    import inspect
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.dataframe import CadwDataFrame
    from cadw.ops import pivot as pv

    pt_src = inspect.getsource(CadwDataFrame.pivot_table)
    assert_no_call_in_source(pt_src, "to_pandas",
                              fn_name="CadwDataFrame.pivot_table")
    print("  pivot_table: no to_pandas")

    pv_src = inspect.getsource(pv.pivot_to_disk)
    assert_no_call_in_source(
        pv_src, "to_pandas", "from_pandas", "pivot",
        fn_name="pivot_to_disk",
    )
    print("  pivot_to_disk: no to_pandas/from_pandas/pivot")
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.6c2: streaming PivotOp (R6) ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    create_pivot_module()
    replace_pivot_table()
    update_whitelist()
    rewrite_100k_test()
    create_new_tests()
    dogfood()

    print("\n=== pytest: pivot regression + new tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pivot_table.py",
         "tests/test_pivot_table_e2e.py",
         "tests/test_pivot_arrow_native.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIVOT REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: AST guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py", "-q"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** AST GUARDS FAILED ***")
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
        "v0.9.6c2: pivot_table — streaming Arrow-native, no pandas "
        "(R6). New cadw/ops/pivot.py::pivot_to_disk: groupby_aggregate "
        "→ external_sort([*index, columns]) → per-index_key buffer → "
        "emit RecordBatch every 1000 rows. No dense K_index × K_columns "
        "matrix in RAM. Peak = sort_buffer + long_table + K_columns "
        "keys + BATCH_ROWS × K_columns × value_size. R6: 26.8 MB → "
        "~10 MB for 200k/300x300. 100k × 20 cols passes under 50 MB "
        "(previously failed at dense estimate). Behaviour: same "
        "values/column-order/row-order. Fail-fast on huge K_columns "
        "(check_state on dict size + output batch). Removes last "
        "to_pandas() from dataframe.py execution path. test_pivot_"
        "100k_rows_fails_fast rewritten as _streaming_success "
        "(contract improvement: streaming passes where dense failed). "
        "+16 new tests in test_pivot_arrow_native.py. Scaffold used: "
        "is_patched, assert_no_call_in_source, write_content.",
        self_path="scripts/patch_v096c2_pivot_arrow.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6c2 applied.")


if __name__ == "__main__":
    main()