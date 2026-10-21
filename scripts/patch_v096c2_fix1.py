#!/usr/bin/env python3
# v0.9.6c2-fix1 — three v0.9.6c2 issues.
#
# (1) cadw/ops/pivot.py pre-existed with SYNTAX ERROR:
#       File ".../cadw/ops/pivot.py", line 31
#         def _column_dtype_for_agg(aggfunc: str, values_type: pa.DataType)
#                                                                        ^
#       SyntaxError: expected ':'
#     Missing continuation line `-> pa.DataType:`. Source unknown —
#     not in my session log (possibly prior partial v0.9.6c2 attempt
#     or leftover from external source). `create_pivot_module()`
#     checked .exists() and skipped correct write. Fix: force-overwrite
#     from canonical PIVOT_LINES regardless of .exists(). Verify
#     import smoke in subprocess (§13.4/184, use -c).
#
# (2) `rewrite_100k_test` searched tests/test_pivot_table.py;
#     actual test lives in tests/test_pivot_table_e2e.py.
#     Fix: AST-based search in both files.
#
# (3) After replacing 100k test, `MemoryLimitExceeded` import in
#     test_pivot_table_e2e.py becomes unused. Remove.
#
# Dataframe.py: already patched by v0.9.6c2 (pivot_table replaced,
# imports pivot_to_disk lazily). Keep as-is.
#
# test_pivot_arrow_native.py: already created by v0.9.6c2. Keep as-is.
#
# Run: uv run python scripts/patch_v096c2_fix1.py

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
    write_content, assert_compiles, commit_with_guards,
    assert_no_call_in_source,
)


# ============ 1. canonical PIVOT_LINES (identical to v0.9.6c2) ============

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
    'def _column_dtype_for_agg(',
    '    aggfunc: str, values_type: pa.DataType,',
    ') -> pa.DataType:',
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
    '    def _filtered():',
    '        for b in batch_iter:',
    '            memory_guard.check_batch(b, name="pivot.filter")',
    '            if columns_col in b.schema.names:',
    '                mask = pc.is_valid(b.column(columns_col))',
    '                b = b.filter(mask)',
    '            if b.num_rows > 0:',
    '                yield b',
    '',
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
    '    value_size = _column_key_size(value_dtype)',
    '    batch_bytes_est = _BATCH_ROWS * (k_columns * value_size + 64)',
    '    memory_guard.check_state(',
    '        batch_bytes_est,',
    '        name=f"pivot.output_batch(K={k_columns})",',
    '    )',
    '',
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


def force_rewrite_pivot_py():
    print("\n=== force-rewrite cadw/ops/pivot.py ===")
    p = Path("cadw/ops/pivot.py")
    print(f"  pre-existing: {p.exists()}")
    if p.exists():
        s = p.read_text()
        print(f"  current size: {len(s)} chars, "
              f"{s.count(chr(10)) + 1} lines")
        # Diagnostic: show first 35 lines
        for i, line in enumerate(s.split(chr(10))[:35], 1):
            marker = " <<<" if i in (31, 32) else ""
            print(f"    {i:3d}: {line}{marker}")
    write_content(p, PIVOT_LINES)
    assert_compiles(p)
    print("  force-written OK")

    # Verify expected function names present.
    s = p.read_text()
    for name in ("pivot_to_disk", "_column_dtype_for_agg",
                 "_column_key_size", "_VALID_AGG"):
        if name not in s:
            raise SystemExit(f"post-write: {name} missing")
    print("  sanity: expected symbols present")

    # Import smoke (subprocess -c, §13.4/184).
    print("  import smoke...")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.ops.pivot import pivot_to_disk; "
         "assert callable(pivot_to_disk); "
         "print('import OK')"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        print("  STDERR:", r.stderr.strip()[-800:])
        raise SystemExit("import smoke failed")
    print("  " + r.stdout.strip())


# ============ 2. find+replace 100k test (AST-based) ============

NEW_100K_TEST = '''@pytest.mark.slow
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
    non_null = sum(
        1 for k, v in rows["r0"].items() if k != "row" and v is not None
    )
    assert non_null == 1
'''


def fix_100k_test():
    print("\n=== find+replace test_pivot_100k_rows_fails_fast ===")
    for path_str in (
        "tests/test_pivot_table_e2e.py",
        "tests/test_pivot_table.py",
    ):
        p = Path(path_str)
        if not p.exists():
            continue
        s = p.read_text()
        if "def test_pivot_100k_rows_fails_fast" not in s:
            continue
        print(f"  found in {path_str}")

        tree = ast.parse(s)
        start = end = None
        for node in tree.body:
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "test_pivot_100k_rows_fails_fast"):
                start = node.lineno
                for dec in node.decorator_list:
                    if dec.lineno < start:
                        start = dec.lineno
                end = node.end_lineno
                break

        if start is None:
            raise SystemExit(f"{path_str}: AST span not found")

        lines = s.split("\n")
        p.write_text("\n".join(
            lines[:start - 1] + [NEW_100K_TEST] + lines[end:]
        ))
        assert_compiles(p)
        print(f"  {path_str}: replaced lines {start}..{end}")
        return

    print("  not found in either file — may already be renamed")


# ============ 3. cleanup unused MemoryLimitExceeded ============

def cleanup_unused_imports():
    print("\n=== cleanup unused MemoryLimitExceeded ===")
    for path_str in (
        "tests/test_pivot_table_e2e.py",
        "tests/test_pivot_table.py",
    ):
        p = Path(path_str)
        if not p.exists():
            continue
        s = p.read_text()
        # Count occurrences of MemoryLimitExceeded as identifier (not
        # in a comment - simple count is good enough here).
        import re
        occurrences = len(re.findall(r"\bMemoryLimitExceeded\b", s))
        if occurrences == 0:
            continue
        # If the only occurrence is in the import line.
        import_line = "from cadw.core.memory_guard import MemoryLimitExceeded\n"
        if import_line in s and occurrences == 1:
            s = s.replace(import_line, "", 1)
            p.write_text(s)
            assert_compiles(p)
            print(f"  {path_str}: removed unused import")
        else:
            print(f"  {path_str}: {occurrences} occurrence(s) — keep")


# ============ 4. dogfood ============

def dogfood():
    print("\n=== dogfood: pivot_table + pivot_to_disk AST guards ===")
    import inspect
    from cadw.dataframe import CadwDataFrame
    from cadw.ops import pivot as pv

    pt_src = inspect.getsource(CadwDataFrame.pivot_table)
    assert_no_call_in_source(pt_src, "to_pandas",
                              fn_name="CadwDataFrame.pivot_table")
    print("  pivot_table: no to_pandas")

    pv_src = inspect.getsource(pv.pivot_to_disk)
    assert_no_call_in_source(
        pv_src, "to_pandas", "from_pandas",
        fn_name="pivot_to_disk",
    )
    print("  pivot_to_disk: no to_pandas/from_pandas")
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.6c2-fix1 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    force_rewrite_pivot_py()
    fix_100k_test()
    cleanup_unused_imports()
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
        "v0.9.6c2-fix1: force-overwrite cadw/ops/pivot.py "
        "(pre-existing broken file — SyntaxError at line 31: "
        "def _column_dtype_for_agg(...) missing '-> pa.DataType:' "
        "continuation; create_pivot_module() checked .exists() and "
        "skipped correct write; source unknown, not in my session "
        "log). Force-write canonical PIVOT_LINES + symbol sanity + "
        "subprocess import smoke (§13.4/184). Find+replace "
        "test_pivot_100k_rows_fails_fast in test_pivot_table_e2e.py "
        "(was searched in test_pivot_table.py). Cleanup unused "
        "MemoryLimitExceeded import in e2e. v0.9.6c2 complete.",
        self_path="scripts/patch_v096c2_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6c2 complete (fix1).")


if __name__ == "__main__":
    main()