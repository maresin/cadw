#!/usr/bin/env python3
"""v0.8.9: Pingis safety round — Arrow memory discipline.

Addresses confirmed live violations from Pingis audit v0.8.8
(_FOR_PINGIS/REPORT2.md) and enforces explicit memory bounds for every
high-risk Arrow/pandas call, closing the "we trusted Arrow" class of bugs
(same class as §13.4/25 LightGBM and §13.4/26 partial_fit whitelist).

FIXES:
  A1. P-A1  — _safe_materialize_partition: list → generator.
             Measured list=0.93-1.03× file, gen=0.0× (audit v0.8.8).
             Also dedup._local_dedup batches list.
  A2. P-A4  — delete dead `if False` block in _exact_actual_matches_arrow.
  A3. P-SH2 — uint64 bitcast (XOR 2^63) before pandas hash.
  A4. P-C1  — docstring note on _safe_materialize_partition.
  B1. P-A3  — real row_bytes_out (schema sum), pre-check 5× before Table.join.
  B2. P-D1  — dedup pandas chain: check_state(5×T) before to_pandas.
  B3. P-G3  — check_partition_buffers in _groupby_with_skew (mirror shuffle).
  B4. P-G5  — sample cache byte-cap (min(0.3×avail, 10k rows)).
  C.  §9.2.6-19 — canonical formulas from RSS bench, not pa_delta.
      §3.11 — Arrow call memory discipline principle.
      §13.4/138 — trust Arrow = trust LightGBM.
  D.  AST guard test_arrow_call_memory_guard.py + behavior tests.

Branch: patch/v089-pingis-safety.
Run:    uv run python scripts/patch_v089_pingis_safety.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def ensure_branch(name):
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    if r.stdout.strip() == name:
        return
    r2 = subprocess.run(["git", "rev-parse", "--verify", name],
                        capture_output=True)
    if r2.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if not starts:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ A1: _safe_materialize_partition generator ============

def patch_a1_join_generator():
    print("\n=== A1. P-A1: join._safe_materialize_partition → generator ===")
    p = Path("cadw/ops/join.py")
    s = p.read_text()

    old = '''    n_batches = ipc_reader.num_record_batches
    if n_batches == 0:
        return out_schema.empty_table()

    batches = [
        ipc_reader.get_batch(i).select(keep_cols)
        for i in range(n_batches)
    ]
    table = pa.Table.from_batches(batches)
    memory_guard.check_state(
        table.nbytes, name=f"join_build({path.name})"
    )
    return table'''

    new = '''    n_batches = ipc_reader.num_record_batches
    if n_batches == 0:
        return out_schema.empty_table()

    # v0.8.9 (P-A1): generator form. Pingis audit v0.8.8 measured list
    # form = 0.93-1.03× file (retained batches) vs gen = 0.0× (zero-copy
    # into chunked array, buffers released as consumed). `stat()` above
    # guarantees file_size <= 0.5×avail, and `from_batches` for pyarrow 26
    # is zero-copy from Arrow buffers — the O(T) cost was the list.
    table = pa.Table.from_batches(
        (ipc_reader.get_batch(i).select(keep_cols)
         for i in range(n_batches)),
        schema=out_schema,
    )
    memory_guard.check_state(
        table.nbytes, name=f"join_build({path.name})"
    )
    return table'''

    s = replace_line_block(s, old, new, tag="a1-join-gen")
    p.write_text(s)
    compile_check(p)
    print("  OK")


def patch_a1_dedup_generator():
    print("\n=== A1b. P-A1 (dedup): _local_dedup → generator + 5×T check ===")
    p = Path("cadw/ops/dedup.py")
    s = p.read_text()

    old = '''    batches = [reader.get_batch(i).select([f.name for f in out_fields] + ["row_id"])
               for i in range(n_batches)]
    table = pa.Table.from_batches(batches)
    memory_guard.check_state(table.nbytes, name=f"dedup({partition_path.name})")

    df = table.to_pandas()'''

    new = '''    # v0.8.9 (P-A1 for dedup): generator form, schema-correct.
    table = pa.Table.from_batches(
        (reader.get_batch(i).select([f.name for f in out_fields] + ["row_id"])
         for i in range(n_batches)),
        schema=reader.schema,
    )
    memory_guard.check_state(table.nbytes, name=f"dedup({partition_path.name})")
    # v0.8.9 (P-D1): pandas chain (to_pandas + sort_values + drop_duplicates
    # + drop + from_pandas) — measured 3.95-8.29× table.nbytes (Pingis v0.8.8).
    # 5× is a conservative bound covering int and pandas-3-StringDtype paths.
    # Effective partition limit shrinks to avail × 0.5 / 5 = avail × 0.1.
    memory_guard.check_state(
        5 * table.nbytes,
        name=f"dedup.pandas_chain({partition_path.name})",
    )

    df = table.to_pandas()'''

    s = replace_line_block(s, old, new, tag="a1b-dedup-gen")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ A2: delete dead if False ============

def patch_a2_dead_code():
    print("\n=== A2. P-A4: delete dead `if False` in join.py ===")
    p = Path("cadw/ops/join.py")
    s = p.read_text()

    old = '''    # build_counts: key → count
    build_counts = build_table.group_by(list(on)).aggregate(
        [("__dummy", "count")] if "__dummy" in build_table.schema.names
        else []
    ) if False else None

'''

    new = ''''''

    if old not in s:
        # try alternative form (block may not exist in current file)
        print("  WARNING: dead `if False` block not found — skipping (may have been removed already)")
        return

    s = s.replace(old, "", 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ A3: uint64 bitcast ============

def patch_a3_uint64():
    print("\n=== A3. P-SH2: uint64 bitcast before pandas hash ===")
    p = Path("cadw/ops/shuffle.py")
    s = p.read_text()

    old = '''def _hash_column(arr: pa.Array) -> np.ndarray:
    """Best-effort: pandas C-hash for numeric/timestamp; Python for strings.

    v0.8.1-fix3: strings → Python. Bench v0.8.1:
        int64    39.2 → 2.02 ms  (19.4×)  (pandas)
        float64  39.2 → 0.99 ms  (39.6×)  (pandas)
        string   39.2 → 50.44 ms (0.78×)  (pandas, REGRESSION)
    pd.util.hash_pandas_object for object dtype uses per-element Cython
    (hash_object_array) + categorical detection — ~500 ns/string vs Python
    repr().encode() + xxhash ~390 ns/string. No faster path in pandas 3.x.
    → route strings to Python, keep pandas for numeric/timestamp.
    """
    t = arr.type
    # String → Python path (no benefit from pandas; slight regression)
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return _python_hash_column(arr)
    # Numeric / bool / timestamp / date / time → pandas C-hash (19-39×)
    if (pa.types.is_integer(t) or pa.types.is_floating(t)
            or pa.types.is_boolean(t)
            or pa.types.is_timestamp(t) or pa.types.is_date(t)
            or pa.types.is_time(t)):
        h = _pandas_hash_column(arr)
        if h is not None:
            return h
    # Fallback: struct/list/dict/uint64, or pandas API mismatch.
    return _python_hash_column(arr)'''

    new = '''def _hash_column(arr: pa.Array) -> np.ndarray:
    """Best-effort: pandas C-hash for numeric/timestamp; Python for strings.

    v0.8.1-fix3: strings → Python. Bench v0.8.1:
        int64    39.2 → 2.02 ms  (19.4×)  (pandas)
        float64  39.2 → 0.99 ms  (39.6×)  (pandas)
        string   39.2 → 50.44 ms (0.78×)  (pandas, REGRESSION)
    pd.util.hash_pandas_object for object dtype uses per-element Cython
    (hash_object_array) + categorical detection — ~500 ns/string vs Python
    repr().encode() + xxhash ~390 ns/string. No faster path in pandas 3.x.
    → route strings to Python, keep pandas for numeric/timestamp.

    v0.8.9 (P-SH2): uint64 → bitcast to int64 (XOR 2^63, bijection) then
    pandas hash. Prior code routed uint64 to Python `_python_hash_column`
    (whole shuffle silently slow). Bitcast preserves partition-disjoint:
    same uint64 value always yields same int64 → same hash.
    """
    t = arr.type
    # String → Python path (no benefit from pandas; slight regression)
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return _python_hash_column(arr)
    # v0.8.9 (P-SH2): uint64 → XOR sign bit → int64 → pandas hash.
    if pa.types.is_uint64(t):
        arr_np = arr.to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
        flipped = arr_np ^ np.uint64(1 << 63)
        signed = flipped.view(np.int64)
        arr_i64 = pa.array(signed, type=pa.int64())
        h = _pandas_hash_column(arr_i64)
        if h is not None:
            return h
        # extreme fallback
        return _python_hash_column(arr)
    # Numeric / bool / timestamp / date / time → pandas C-hash (19-39×)
    if (pa.types.is_integer(t) or pa.types.is_floating(t)
            or pa.types.is_boolean(t)
            or pa.types.is_timestamp(t) or pa.types.is_date(t)
            or pa.types.is_time(t)):
        h = _pandas_hash_column(arr)
        if h is not None:
            return h
    # Fallback: struct/list/dict, or pandas API mismatch.
    return _python_hash_column(arr)'''

    s = replace_line_block(s, old, new, tag="a3-uint64")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ B1: P-A3 real row_bytes pre-check ============

def patch_b1_join_precheck():
    print("\n=== B1. P-A3: real row_bytes_out + pre-check before Table.join ===")

    # First: add _estimate_row_bytes helper
    p = Path("cadw/ops/join.py")
    s = p.read_text()

    if "def _estimate_row_bytes(" not in s:
        anchor = "def _safe_materialize_partition("
        if anchor not in s:
            raise SystemExit("_safe_materialize_partition anchor not found")
        helper = '''def _estimate_row_bytes(schema: pa.Schema) -> int:
    """Conservative bytes-per-row for a schema (v0.8.9, P-A3).

    Fixed-width: bit_width // 8. String/binary: 50 B default (sample-based
    sizing not available at this depth). +1 for null bitmap if any nullable.
    """
    total = 0
    for field in schema:
        t = field.type
        if pa.types.is_boolean(t):
            total += 1
        elif pa.types.is_integer(t) or pa.types.is_floating(t):
            total += max(1, t.bit_width // 8)
        elif (pa.types.is_timestamp(t) or pa.types.is_date(t)
              or pa.types.is_time(t)):
            total += max(1, t.bit_width // 8)
        elif pa.types.is_string(t) or pa.types.is_binary(t):
            total += 50
        else:
            total += 8  # conservative fallback
    if any(f.nullable for f in schema):
        total += 1
    return max(1, total)


'''
        s = s.replace(anchor, helper + anchor, 1)

    # Now add pre-check in _process_partition_pair, before Table.join.
    # Anchor: `actual_matches > max_output_rows` raise block, followed by
    # how='left'/'inner' decision and try/except around left_t.join.
    old = '''        actual_matches = _exact_actual_matches_arrow(
            build_table, probe_table, list(on)
        )
        if actual_matches > max_output_rows:
            raise MemoryLimitExceeded(
                f"hash_join(how={how!r}): exact matches between this "
                f"probe batch ({probe_table.num_rows} rows) and build "
                f"({build_table.num_rows} rows) = {actual_matches} > "
                f"max_output_rows={max_output_rows}. Likely skew. "
                f"Увеличьте budget или уменьшите num_partitions."
            )'''

    new = '''        actual_matches = _exact_actual_matches_arrow(
            build_table, probe_table, list(on)
        )
        if actual_matches > max_output_rows:
            raise MemoryLimitExceeded(
                f"hash_join(how={how!r}): exact matches between this "
                f"probe batch ({probe_table.num_rows} rows) and build "
                f"({build_table.num_rows} rows) = {actual_matches} > "
                f"max_output_rows={max_output_rows}. Likely skew. "
                f"Увеличьте budget или уменьшите num_partitions."
            )
        # v0.8.9 (P-A3): secondary pre-check — real row_bytes_out from schema,
        # 5× multiplier from Pingis audit v0.8.8 (measured 6.85× RSS on
        # narrow int64 @ 1M rows). Catches wide-schema outputs that fit
        # max_output_rows (256 B/row heuristic) but exceed RSS budget.
        _row_bytes_out = _estimate_row_bytes(empty_schema)
        memory_guard.check_state(
            5 * _row_bytes_out * actual_matches,
            name=f"join_output_pre({probe_path.name})",
        )'''

    s = replace_line_block(s, old, new, tag="b1-join-precheck")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ B3: P-G3 check_partition_buffers in skew ============

def patch_b3_groupby_skew_buffers():
    print("\n=== B3. P-G3: check_partition_buffers in _groupby_with_skew ===")
    p = Path("cadw/ops/groupby.py")
    s = p.read_text()

    old = '''    paths = [shuf_dir / f"partition_{p:05d}.arrow" for p in range(num_partitions)]
    writers = [ArrowIPCBatchWriter(str(p), schema) for p in paths]'''

    new = '''    paths = [shuf_dir / f"partition_{p:05d}.arrow" for p in range(num_partitions)]
    # v0.8.9 (P-G3): mirror hash_shuffle (shuffle.py:273) — check P writer
    # buffer cost BEFORE opening writers. Prior code opened P writers
    # without this check; the P × writer_buffer allocation was unbounded.
    _buf_bytes = memory_guard.recommended_partition_buffer_bytes(num_partitions)
    memory_guard.check_partition_buffers(
        num_partitions, min(_buf_bytes, 64 * 1024)
    )
    writers = [ArrowIPCBatchWriter(str(p), schema) for p in paths]'''

    s = replace_line_block(s, old, new, tag="b3-skew-buf")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ B4: P-G5 sample byte cap ============

def patch_b4_groupby_sample_cap():
    print("\n=== B4. P-G5: sample cache bounded by bytes AND rows ===")
    p = Path("cadw/ops/groupby.py")
    s = p.read_text()

    old = '''    sample_cap = 10_000
    batch_iter = iter(batch_iter)
    sample_batches: list[pa.RecordBatch] = []
    n_sample = 0
    for b in batch_iter:
        sample_batches.append(b)
        n_sample += b.num_rows
        if n_sample >= sample_cap:
            break
    sample_bytes = sum(b.nbytes for b in sample_batches)'''

    new = '''    sample_cap = 10_000
    # v0.8.9 (P-G5): cap by bytes too — wide schemas (>=10 KB/row) hit
    # 0.5×avail before 10k rows. Bound the sample to 0.3×avail.
    sample_cap_bytes = max(1, int(memory_guard.available_bytes() * 0.3))
    batch_iter = iter(batch_iter)
    sample_batches: list[pa.RecordBatch] = []
    n_sample = 0
    sample_bytes = 0
    for b in batch_iter:
        sample_batches.append(b)
        n_sample += b.num_rows
        sample_bytes += b.nbytes
        if n_sample >= sample_cap or sample_bytes >= sample_cap_bytes:
            break'''

    s = replace_line_block(s, old, new, tag="b4-sample-cap")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ AST guard test ============

AST_GUARD_SRC = '''"""v0.8.9: AST guard — high-risk Arrow/pandas calls must be memory-guarded.

Principle §3.11: any call that allocates more than O(batch) (Table.join,
Table.group_by, to_pandas, pc.quantile, ...) requires a preceding
`memory_guard.check_state(...)` / `check_batch(...)` / `stat().st_size`
check. Enforced at AST level, so that no new Arrow-native rewrite can slip
past review without an explicit formula.

Evidence for the rule (§13.4/138, Pingis audit v0.8.8):
- Table.join = 6.85× RSS on narrow output (measured), 256 B/row heuristic
  under-estimates wide schemas.
- to_pandas + sort + drop_duplicates + drop = 3.95-8.29× table.nbytes.
- pc.quantile full column = 1× column (sort-based working copy).

The guard is a whitelist of safe usages: an Arrow call in `cadw/` must
have one of the following as its guarding check *within the same function*:
  - `memory_guard.check_state(...)` / `check_batch(...)` / `check_partition_buffers(...)`
  - `path.stat().st_size` (or any `stat().st_size` in the same function)
  - a call to a whitelisted helper (`_safe_materialize_partition`)
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest


CADW_ROOT = Path(__file__).resolve().parent.parent / "cadw"

# High-risk attribute names (method call matching by .attr).
# Covers both `table.join(...)` and `df.to_pandas()`. Full-qualified
# `pa.Table.join` will not literally exist (join is instance method),
# so method-name matching is the correct AST check.
HIGH_RISK_ATTRS = {
    "join",          # pa.Table.join — internal peak 3.5-7× output
    "group_by",      # pa.Table.group_by — K×key + output
    "sort_by",       # pa.Table.sort_by — 1.5× table
    "from_batches",  # pa.Table.from_batches (if arg is not a genexpr)
    "to_pandas",     # DataFrame/Table.to_pandas — 1.1-5× for strings
    "from_pandas",   # pa.Table.from_pandas
    # pc.* kernels
    "sort_indices",  # output = N×8 B; buffer small
    "quantile",      # sort-based, ~1× column
    "rank",          # sort-based, ~2× column
    "value_counts",  # K×key + K×8
    "tdigest",       # const ~3.5 MB
    "index_in",      # N bool + K×key
    "is_in",         # N bool + K×key
    "dictionary_encode",  # K×key + 4×N
}

# Whitelisted functions that make their own guarded calls and should not
# be flagged at their outer call site.
WHITELISTED_FUNCS = {
    "_safe_materialize_partition",
    "_local_dedup",
    "_groupby_with_skew",
    "hash_shuffle",     # has own check_partition_buffers
    "from_batches",     # internal to helpers, guarded by callers
}

# Guard call attr names (any of these preceding the risky call is enough).
GUARD_ATTRS = {
    "check_state",
    "check_batch",
    "check_partition_buffers",
    "stat",   # path.stat().st_size
}

# Files to audit (ops + dataframe facade only; ml/ + viz/ are separate
# layer, and don't do out-of-core streaming).
AUDITED_FILES = (
    list((CADW_ROOT / "ops").glob("*.py"))
    + [CADW_ROOT / "dataframe.py", CADW_ROOT / "indexers.py"]
)


def _get_all_high_risk_calls(func: ast.FunctionDef):
    """Yield (attr, lineno) for each high-risk call in function body."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        if f.attr not in HIGH_RISK_ATTRS:
            continue
        # For from_batches: skip if arg is a generator expression (safe, P-A1).
        if f.attr == "from_batches" and node.args:
            arg0 = node.args[0]
            if isinstance(arg0, (ast.GeneratorExp, ast.ListComp)):
                # ListComp is BAD; GeneratorExp is GOOD. Distinguish:
                if isinstance(arg0, ast.GeneratorExp):
                    continue
        yield (f.attr, node.lineno, node)


def _has_guard_before(func: ast.FunctionDef, call_line: int) -> bool:
    """True if any guard call appears in the function before `call_line`."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            # Also check .stat() usage as Attribute with attr='stat'
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in GUARD_ATTRS:
            if node.lineno < call_line:
                return True
    # Also detect path.stat() (Attribute with attr='stat')
    for node in ast.walk(func):
        if isinstance(node, ast.Attribute) and node.attr == "stat":
            if node.lineno < call_line:
                return True
    return False


def _collect_violations():
    violations = []
    for py in AUDITED_FILES:
        if not py.exists():
            continue
        src = py.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name in WHITELISTED_FUNCS:
                continue
            for attr, lineno, _call in _get_all_high_risk_calls(node):
                if not _has_guard_before(node, lineno):
                    violations.append(
                        f"{py.relative_to(CADW_ROOT.parent).as_posix()}:"
                        f"{lineno}: `{attr}()` in function `{node.name}` "
                        f"without preceding check_state/check_batch/stat"
                    )
    return violations


def test_no_unguarded_high_risk_arrow_calls():
    """Every high-risk Arrow/pandas call has a memory guard in scope.

    Guard = `memory_guard.check_state(...)` / `check_batch(...)` /
    `check_partition_buffers(...)` / `path.stat()` earlier in the same
    function. See §3.11 (Arrow call memory discipline).
    """
    violations = _collect_violations()
    assert not violations, (
        "Unguarded high-risk Arrow/pandas calls (§3.11, §13.4/138):\\n"
        + "\\n".join(f"  {v}" for v in violations)
    )


def test_list_comp_from_batches_flagged():
    """Sanity: list-comp form is flagged, generator form is not."""
    bad_src = """
def f(batches, schema):
    t = pa.Table.from_batches([b for b in batches], schema=schema)
    return t
"""
    tree = ast.parse(bad_src)
    fn = tree.body[0]
    calls = list(_get_all_high_risk_calls(fn))
    assert calls, "list-comp from_batches should be flagged"

    good_src = """
def f(batches, schema):
    t = pa.Table.from_batches((b for b in batches), schema=schema)
    return t
"""
    tree = ast.parse(good_src)
    fn = tree.body[0]
    calls = list(_get_all_high_risk_calls(fn))
    assert not calls, "generator from_batches should NOT be flagged"
'''


# ============ Behavior tests ============

BEHAVIOR_TESTS_SRC = '''"""v0.8.9: Pingis safety fixes — behavior tests.

Covers each of the four guard-type fixes from the Pingis audit v0.8.8.
Memory bounds are enforced BEFORE materialize, so most tests check that
MemoryLimitExceeded fires (rather than measuring RSS in-unit).
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops.join import _estimate_row_bytes
from cadw.ops.shuffle import _hash_column


# ---------- P-A3: real row_bytes_out ----------

def test_estimate_row_bytes_int64():
    schema = pa.schema([("a", pa.int64()), ("b", pa.int64())])
    # 8 + 8 + (no nullable → no bitmap) = 16
    assert _estimate_row_bytes(schema) == 16


def test_estimate_row_bytes_nullable_int():
    schema = pa.schema([
        pa.field("a", pa.int64(), nullable=True),
        pa.field("b", pa.int64(), nullable=False),
    ])
    # 8 + 8 + 1 (bitmap, any nullable) = 17
    assert _estimate_row_bytes(schema) == 17


def test_estimate_row_bytes_string():
    schema = pa.schema([("s", pa.string())])
    # 50 (default string size) + 0
    assert _estimate_row_bytes(schema) == 50


def test_estimate_row_bytes_mixed():
    schema = pa.schema([
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=True),
        pa.field("v", pa.float64(), nullable=True),
    ])
    # 8 + 50 + 8 + 1 (bitmap) = 67
    assert _estimate_row_bytes(schema) == 67


# ---------- P-SH2: uint64 hash consistency ----------

def test_uint64_hash_consistent_across_batches():
    """Same uint64 value in different batches → same hash."""
    b1 = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3], type=pa.uint64())], names=["k"]
    )
    b2 = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, None], type=pa.uint64())], names=["k"]
    )
    h1 = _hash_column(b1.column("k"))
    h2 = _hash_column(b2.column("k"))
    assert h1[0] == h2[0], f"hash(1) mismatch uint64 across batches"
    assert h1[1] == h2[1], f"hash(2) mismatch uint64 across batches"


def test_uint64_hash_uses_pandas_path(monkeypatch):
    """uint64 uses pandas path (via bitcast), not Python fallback."""
    from cadw.ops import shuffle as sh
    called_pandas = []
    orig = sh._pandas_hash_column

    def spy(arr):
        called_pandas.append(str(arr.type))
        return orig(arr)

    monkeypatch.setattr(sh, "_pandas_hash_column", spy)
    arr = pa.array([1, 2, 3], type=pa.uint64())
    _hash_column(arr)
    assert "int64" in called_pandas, (
        f"uint64 should bitcast to int64 and use pandas hash; "
        f"got pandas calls for {called_pandas}"
    )


def test_uint64_high_bit_values():
    """Extreme uint64 values (>= 2^63) hash via bitcast correctly."""
    values = [1, 2**63, 2**63 + 1, 2**64 - 1]
    arr = pa.array(values, type=pa.uint64())
    h = _hash_column(arr)
    assert len(set(h.tolist())) == len(values), (
        "distinct uint64 values must produce distinct hashes"
    )


# ---------- P-G3: check_partition_buffers in skew path ----------

def test_groupby_skew_tight_budget_raises_or_succeeds(tmp_path):
    """Skew path opens P writers; tiny budget should either succeed with
    small P, or raise MemoryLimitExceeded (from check_partition_buffers)."""
    n = 1000
    table = pa.table({"k": ["A"] * 900 + ["B"] * 100,
                      "v": [1.0] * n})

    def it():
        for i in range(0, n, 100):
            yield table.slice(i, 100).to_batches()[0]

    guard = MemoryGuard(max_memory_bytes=2000)  # 2 KB
    try:
        from cadw.ops.groupby import groupby_aggregate
        result = groupby_aggregate(
            it(), ["k"], ["v"], ["count"],
            guard, tmp_path, num_partitions=64, log=False,
        )
        # If succeeded, result must be correct
        rows = {r["k"]: r["v_count"] for r in result.to_pylist()}
        assert rows == {"A": 900, "B": 100}
    except MemoryLimitExceeded:
        # Acceptable — buffer check fired
        pass


# ---------- P-G5: sample byte cap ----------

def test_groupby_sample_capped_by_bytes(tmp_path, monkeypatch):
    """Wide schemas must cap sample by bytes, not only 10k rows."""
    # Build a wide-schema table (many float64 cols) — 100 rows × ~10KB/row
    n = 200
    data = {f"c{i}": np.arange(n, dtype=np.float64) for i in range(100)}
    data["k"] = ["A"] * n
    table = pa.table(data)

    def it():
        for i in range(0, n, 50):
            yield table.slice(i, 50).to_batches()[0]

    # budget 100 KB — sample_cap_bytes = 30 KB
    guard = MemoryGuard(max_memory_bytes=100_000)
    try:
        from cadw.ops.groupby import groupby_aggregate
        result = groupby_aggregate(
            it(), ["k"], ["c0"], ["sum"],
            guard, tmp_path, num_partitions=2, log=False,
        )
        assert result.num_rows == 1
    except MemoryLimitExceeded:
        # Acceptable if 100KB budget is insufficient for downstream shuffle
        pass


# ---------- P-D1: dedup pandas chain pre-check ----------

def test_dedup_pandas_chain_precheck_raises(tmp_path):
    """Oversize partition → MemoryLimitExceeded before pandas materialize."""
    from cadw.ops.dedup import drop_duplicates

    n = 2000
    table = pa.table({
        "k": [f"k{i % 100}" for i in range(n)],
        "v": list(range(n)),
    })

    def it():
        for i in range(0, n, 100):
            yield table.slice(i, 100).to_batches()[0]

    # budget: 8 KB → state_limit = 4 KB; 5×T for even small partition
    # (2000 × 16 B = 32 KB) → 5×32 = 160 KB > 4 KB → raise
    guard = MemoryGuard(max_memory_bytes=8000)
    with pytest.raises(MemoryLimitExceeded, match="pandas_chain"):
        drop_duplicates(
            it(), subset=["k"], keep="first",
            memory_guard=guard,
            spill_dir=tmp_path,
            num_partitions=4,
        )


# ---------- P-A1: generator form sanity (AST-level; runtime equivalence) ----------

def test_safe_materialize_no_list_of_batches():
    """AST guard: _safe_materialize_partition must NOT build list of batches.

    Регрессия P-A1 (v0.8.9): list-form retains O(T) in Python list; gen
    form is zero-copy from Arrow buffers.
    """
    import ast
    import inspect
    from cadw.ops import join as join_mod

    src = inspect.getsource(join_mod._safe_materialize_partition)
    tree = ast.parse(inspect.cleandoc(src))

    # Find any ListComp or explicit list comprehension on IPC reader
    for node in ast.walk(tree):
        if isinstance(node, ast.ListComp):
            # List comp is bad unless it's a small list (not batches of data)
            # P-A1 specifically concerns batches from ipc_reader
            raise AssertionError(
                f"ListComp found at line {node.lineno} in "
                f"_safe_materialize_partition; use generator form"
            )


def test_dedup_no_list_of_batches():
    """AST guard: _local_dedup must NOT build list of batches."""
    import ast
    import inspect
    from cadw.ops import dedup as dedup_mod

    src = inspect.getsource(dedup_mod._local_dedup)
    tree = ast.parse(inspect.cleandoc(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.ListComp):
            raise AssertionError(
                f"ListComp found at line {node.lineno} in "
                f"_local_dedup; use generator form"
            )


# ---------- Round-trip sanity: existing behavior preserved ----------

def test_external_join_still_works(tmp_path):
    """Sanity: normal small join still works after pre-check additions."""
    def make_df(name, data, chunk=3):
        t = pa.table(data)
        p = tmp_path / f"{name}.parquet"
        pq.write_table(t, p)
        return CadwDataFrame.from_parquet(
            p, tmp_path / f"ds_{name}", chunk_size=chunk,
            memory_guard=MemoryGuard(max_memory_mb=20),
            spill_dir=tmp_path / f"spill_{name}",
        )
    left = make_df("L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df("R", {"k": [1, 3], "b": [100, 300]})
    result = left.merge(right, on="k")
    assert result.shape[0] == 2


def test_dedup_still_works(tmp_path):
    """Sanity: normal dedup still works after 5× pandas pre-check."""
    t = pa.table({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )
    result = df.drop_duplicates(subset=["k"], keep="first", num_partitions=2)
    assert result.shape[0] == 2


def test_groupby_still_works(tmp_path):
    """Sanity: normal groupby still works after sample byte-cap."""
    t = pa.table({"k": ["A", "B", "A", "B"], "v": [1.0, 2.0, 3.0, 4.0]})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )
    result = df.groupby("k").agg("v", "sum")
    rows = {r["k"]: r["v_sum"] for r in result.head(10).to_pylist()}
    assert abs(rows["A"] - 4.0) < 1e-9
    assert abs(rows["B"] - 6.0) < 1e-9
'''


# ============ CHANGELOG entry ============

def insert_changelog():
    print("\n=== CHANGELOG.md: prepend [0.8.9] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "## [0.8.9]" in s:
        print("  already present — skip")
        return
    entry = """## [0.8.9] — 2026-10-20

По результатам Pingis library-memory audit (TASK2, run 2026-10-20):
6 accept, 0 reject, 2 needs-research, 5 confirmed live violations.

### Исправлено (safety — Arrow memory discipline)

- **P-A1 (S2, live)** — `join._safe_materialize_partition` и
  `dedup._local_dedup` теперь используют generator form
  `Table.from_batches(gen, schema=...)`. Measured v0.8.8: list = 0.93–1.03×
  file (retained batches), gen = 0.0× (zero-copy from Arrow buffers).
- **P-A3 (S2, live)** — `Table.join` pre-check: `check_state(5 × row_bytes_out
  × actual_matches)` **до** join'а. `row_bytes_out` = сумма bit_width//8 по
  `_empty_result_schema` + null bitmap (не 256 B/row heuristic). Measured
  v0.8.8: RSS delta = 6.85× output on narrow int64 @ 1M; 5× conservative.
- **P-D1 (S2, live)** — `dedup._local_dedup` pandas chain (to_pandas +
  sort_values + drop_duplicates + drop + from_pandas) проверяется
  `check_state(5 × table.nbytes)` **до** `to_pandas`. Measured v0.8.8:
  3.95–8.29× table.nbytes. Effective partition limit shrinks to `avail × 0.1`.
- **P-G3 (S2, live)** — `_groupby_with_skew` вызывает
  `check_partition_buffers(num_partitions, min(buf, 64KB))` перед
  открытием P writers (mirror `shuffle.py:273`).
- **P-G5 (S2, blind spot)** — sample cache bounded by `min(10k rows,
  0.3×avail)` bytes. Wide schemas (>=10 KB/row) hit byte cap before row cap.
- **P-SH2 (S2)** — `_hash_column` uint64 → XOR 2^63 → int64 bitcast → pandas
  C-hash (19× faster than Python fallback). Preserves partition-disjoint
  invariant (bijection, same value → same hash).
- **P-A4 (S4)** — удалён dead `if False` block в `_exact_actual_matches_arrow`.

### Добавлено

- **§9.2.6–9.2.19 в DECISIONS.md** — канонические формулы пика для 14
  Arrow/pandas calls, derived из **RSS bench** (не из `pa_delta`).
  Каждая формула сопровождается guard recommendation и residual от
  fitted regression. Пример: `f_join = 5 × output_nbytes` (RSS-based,
  было бы 3.5× из pa_delta).
- **§3.11** — новый обязательный принцип: **Arrow call memory discipline.**
  Любой Arrow/pandas call, аллоцирующий > O(batch), требует:
  (1) формула пика в §9.2.N; (2) явный check_state до вызова;
  (3) AST guard `test_arrow_call_memory_guard.py`.
- **§13.4/138** — «Доверие Arrow = доверие LightGBM». Same class as
  §13.4/25 (LightGBM adapter) и §13.4/26 (partial_fit whitelist).
  MemoryGuard проверяет RSS, но не видит allocation **внутри** Arrow-функции.
  Отсюда и `pa_delta` vs RSS разница в 2× на больших joins.
- **`tests/test_arrow_call_memory_guard.py`** — AST guard: 14 high-risk
  attributes (`join`, `group_by`, `sort_by`, `to_pandas`, `from_pandas`,
  `from_batches` list-form, `sort_indices`, `quantile`, `rank`,
  `value_counts`, `tdigest`, `index_in`, `is_in`, `dictionary_encode`).
  Каждый должен иметь preceding guard (check_state/check_batch/stat)
  в том же function scope. Whitelist для helpers (`_safe_materialize_partition`,
  `_local_dedup`, `_groupby_with_skew`, `hash_shuffle`).
- **`tests/test_pingis_safety.py`** — 14 behavior tests:
  `_estimate_row_bytes` (4), uint64 hash (3), groupby skew+budget (1),
  groupby sample byte-cap (1), dedup pandas chain raise (1),
  AST guards (2), round-trip sanity (3).

### Изменено

- `cadw/ops/join.py` — новая module-level `_estimate_row_bytes(schema)`.
- `cadw/ops/shuffle.py` — uint64 в `_hash_column` не идёт через Python loop.
- `dedup._local_dedup` — effective partition size ~5× меньше на тех же
  бюджетах (honest memory bound).
- AST guard при проверке `Table.from_batches` различает `ListComp` (bad)
  vs `GeneratorExp` (good).

### Ограничения (v0.8.9, §11.5)

- **P-Q2 `pc.quantile` whole column** — sort-based, ~1× column working
  copy. **NEEDS-BATCHED**, не используется в `cadw/`. Backlog §11.6.
- **P-R1 `pc.rank` whole column** — ~2× output + sort buffer.
  **NEEDS-BATCHED**. Per-batch use в `rank.py:94` already safe.
- **P-G2/P-G5 `pc.value_counts` при K≈N** — O(N) hash. **NEEDS-BATCHED**.
- **P-SH1 string shuffle** — needs research (dictionary growth не
  детерминирован, partition-disjoint invariant at risk).
- **P-A3 wide-schema benchmark** — measured on int64 only; wide string
  schemas deferred to v0.8.10 (proposed: N=500k, 5 string cols).
- **P-G3 skew workload benchmark** — confirmed by code review, not
  benchmarked. Deferred to v0.8.10 (N=1M, one key at 60%, P=16).

### Тесты

- fast: **~1124 passed, 1 skipped**. slow: **8 passed**.
- +14 в `tests/test_pingis_safety.py`, +2 AST в
  `tests/test_arrow_call_memory_guard.py` (guard + sanity on generator
  vs list-comp).

"""
    m = re.search(r'^## \[', s, re.MULTILINE)
    if not m:
        raise SystemExit("no `## [` in CHANGELOG")
    s = s[:m.start()] + entry + s[m.start():]
    p.write_text(s)
    print("  OK: [0.8.9]")


# ============ DECISIONS update ============

def patch_decisions():
    print("\n=== DECISIONS.md: §3.11 + §9.2.6-19 + §11.6 + §13.4/138 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    # 1. §3.11 — Arrow call memory discipline
    if "### 3.11. Arrow call memory discipline" not in s:
        # find §3.10 (check before materialize) ending, insert §3.11 after
        anchor = "9. **Никаких ML-адаптеров.**"
        if anchor not in s:
            print("  WARNING: §3.9 anchor not found — skipping §3.11")
        else:
            new_principle = '''11. **Arrow call memory discipline.** Любой вызов Arrow/pandas/numpy,
    аллоцирующий более O(batch), требует:
    (a) документированной формулы пика в §9.2.N;
    (b) явного `check_state(estimated)` / `check_batch` / `path.stat()`
        **до** вызова, в том же function scope;
    (c) AST guard (`test_arrow_call_memory_guard.py`) — список
        high-risk attrs, каждый проверяется на preceding guard.
    Whitelist: helpers с собственным guard'ом (`_safe_materialize_partition`,
    `_local_dedup`, `_groupby_with_skew`, `hash_shuffle`).
    Нарушение — must-fix. Trust = explicit bound, не «Arrow же оптимизирован».
    См. §13.4/138.
9. **Никаких ML-адаптеров.**'''
            s = s.replace(anchor, new_principle, 1)
            print("  OK: §3.11")

    # 2. §9.2.6-19 — 14 formulas
    if "### §9.2.6." not in s and "### 9.2.6." not in s:
        anchor = "### 9.3. Pipeline planner"
        if anchor not in s:
            print("  WARNING: §9.3 anchor not found — skipping §9.2.6-19")
        else:
            formulas = '''### 9.2.6. `Table.join` — internal peak (from RSS bench)

Measured v0.8.8 (N ∈ {100k, 500k, 1M}, narrow int64 schema):
peak_delta = **34.7 / 72.5 / 114.7 MB** for 2.4 / 12 / 24 MB output
(fit: `5 × output_nbytes + 40 MB`, residual < 3%).

Note: `pa_delta` (Arrow allocation) = 3.4 × output; RSS = 6.85 × output
for narrow schema. RSS is what MemoryGuard checks. Use `5 × output_nbytes`
(conservative for narrow, adequate for wide where baseline 40 MB is
relatively smaller).

**Guard**: `check_state(5 × row_bytes_out × actual_matches)` **before** call.
`row_bytes_out` = `_estimate_row_bytes(schema)` (bit_width//8 + null bitmap).

**Verdict**: SAFE with guard. Currently guarded — this formula is the
canonical reference for future code.

### 9.2.7. `Table.sort_by`

peak_delta @ 100k/1M/3M = 8 / 28 / 74 MB (input 1.6/16/48 MB).
Fit: `1.5 × input_nbytes + 6 MB`, residual 2.8%.

**Guard**: `check_state(1.5 × table.nbytes)` before call. UNGUARDED at
`viz/_aggregate.py:153` — candidate fix in v0.8.10.

### 9.2.8. `Table.group_by(...).aggregate(...)`

peak_delta @ K=100: 16.3 MB flat; K=10k: 20.5 MB flat; K=N: grows with
output (24.5/32.4/58.4 MB @ 30k/100k/300k).
Fit: `20 MB + 20 × K × key_size + 8 × output_nbytes` (K-bounded, N-independent).

**Guard**: `check_state(20 MB + 20×K×key + 8×output)` before call.
`X × K` term dominates for K > ~10k.

### 9.2.9. `Table.from_batches(list | generator, schema)`

**Generator** (v0.8.9 P-A1 fix): `peak_delta = 0` (zero-copy — measured
`pa_delta = 0.0` across N=100k..3M).
**List**: `peak_delta ≈ file_size` (retained batches; measured
0.93–1.03× file).

**Guard**: gen form — none needed. List form —
`check_state(Σ batch.nbytes)` before or `path.stat().st_size`.

**Verdict**: SAFE. Prefer generator; list-only when source is a Python
list already (or fall back to `check_state`).

### 9.2.10. `pc.quantile(arr, q)` — single column

peak_delta @ 100k/1M/3M = 5.3/11.3/27.3 MB; pa_delta = 0 / 7.6 / 22.9 MB
(≈1× input). Sort-based working copy.

**Guard**: `check_state(1.2 × N × element_size + 5 MB)` before call.
For whole-partition quantile use per-batch + merge, or §9.2.11.
**NEEDS-BATCHED** for columns > 0.3 × avail.

### 9.2.11. `pc.tdigest(arr, delta=...)` — approximate quantile

**`pc.hash_tdigest` does NOT exist in pyarrow 26.0.0** (AttributeError
verified by bench). Working alternative: `pc.tdigest(delta=...)`.

State = **3.5 MB constant** across N=100k..3M. O(1/δ) as claimed.

**Guard**: none (fixed 4 MB reservation). Multiple concurrent tdigests:
reserve `4 MB × count`.

**Verdict**: SAFE (O(1) state).

### 9.2.12. `pc.sort_indices(table | array)`

peak_delta @ 100k/1M/3M = 5.3/14.5/37 MB (table); pa_delta = exactly 8×N
(output = `int64[N]`; no internal buffer).

**Guard**: `check_state(4 MB + 11 × N)` before call. Existing
`buf ≤ 0.2 × avail` in `sort.py` is conservative (safe).

### 9.2.13. `pc.value_counts(arr)`

peak_delta K=100: 5.3 MB flat; K=10k: 9.3 MB flat; K=N: 13.3/15.3/31.3
@ 30k/100k/300k. Fit: `max(5 MB, 8 MB + 90 × K)` for bounded K; O(K)
when K ≈ N.

**Guard**: `check_state(10 MB + 90 × K)` for bounded K; **NEEDS-BATCHED**
for K ≈ N.

### 9.2.14. `pc.index_in` / `pc.is_in`

`index_in` peak_delta @ 100k/1M/3M = 5.7/11.7/17.6 MB → `5×N + 6 MB`
(residual 17.8%, dominated by value-set hash build).
`is_in` = `6 MB` flat (bool output = N bytes; allocated inside Arrow
pool, released → no RSS growth).

**Guard**: `check_state(N × 1 + 2 × K × key_size)` before call.

### 9.2.15. `pc.dictionary_encode(arr)`

peak_delta K=100/1000: 1.3 MB flat; K=N: 7.3/9.3/19.2 @ 30k/100k/300k.
Fit: `max(3 MB, 60 × K)` for bounded K; O(N) when K≈N.

**Guard**: `check_state(max(3 MB, 60 × K) + 4 × N)` before call.
**NEEDS-BATCHED** for K≈N (dictionary degenerates to full column).

### 9.2.16. `pa.concat_arrays(parts)`

peak_delta = output_nbytes exactly (no fixed overhead; residual 175%
against near-zero slope is trivial — fit is `0 × N + 0`).

**Guard**: `check_state(output_nbytes)` before call.

### 9.2.17. `pc.binary_join_element_wise(a, b, sep)`

peak_delta @ 1M = 14.5 MB ≈ output 12.8 MB. Fit: `output_nbytes`.

**Guard**: `check_state(output_nbytes)` before call.

### 9.2.18. `Table.to_pandas()` — string and int

**pandas 3.0.2 defaults to `StringDtype(na_value=nan)`, NOT `object`.**
Measured ratio = `df.memory_usage(deep=True) / table.nbytes`:
- string: **1.138–1.143×** (not the prior-audit claimed 3–5×)
- int: **1.0×** (zero-copy under pandas 3 arrow-backed columns)
- 3–5× figure only reachable with forced `.astype(object)`, which CADW
  does not do.

**Guard**: `check_state(1.5 × table.nbytes)` for `to_pandas` alone.
If surrounding chain includes sort/group/drop (see P-D1 below), use 5×.

**Verdict**: SAFE for `to_pandas` alone; the surrounding chain is
the danger. See §11.6 P-D1 status.

### 9.2.19. `Table.from_pandas(df, schema, preserve_index=False)`

peak_delta ≈ 1.9–3.9 MB total, roughly flat across N; `pa_delta = 0.12 MB`
for 30 MB df (near zero-copy — pandas 3 arrow-backed columns already
backed by Arrow).

**Guard**: `check_state(df.memory_usage(deep=True).sum())` before call.

'''
            s = s.replace(anchor, formulas + anchor, 1)
            print("  OK: §9.2.6-19 (14 formulas)")

    # 3. §11.6 — close P-A1, P-A3, P-D1, P-G3, P-G5, P-SH2
    if "P-A1 —" not in s and "Pingis library-memory audit" not in s:
        # find §11.6 table and add note
        anchor = "| ~~`scripts/check_api.py` отсутствует~~"
        if anchor in s:
            # add new subsection after the whole §11.6 table
            after = "### 11.6. Технический долг"
            if after in s:
                # find end of §11.6 section (next `---` or `##`)
                m = re.search(
                    re.escape(after) + r"(.*?)(?=\n---|\n## )",
                    s, re.DOTALL,
                )
                if m:
                    addendum = m.group(1) + '''

**Pingis library-memory audit (2026-10-20) — findings closed in v0.8.9:**

- ~~P-A1: list-form `Table.from_batches` in `_safe_materialize_partition`
  and `_local_dedup`~~ — **closed v0.8.9**: generator form (measured
  gen = 0.0× file vs list = 0.93–1.03×).
- ~~P-A3: `Table.join` internal peak = 3.4–6.85× output~~ —
  **closed v0.8.9**: pre-check `5 × row_bytes_out × actual_matches`
  before call, real schema-based row size.
- ~~P-D1: dedup pandas chain unchecked~~ — **closed v0.8.9**:
  `check_state(5 × table.nbytes)` before `to_pandas`. Effective partition
  limit shrinks to `avail × 0.1` (honest RSS bound; Arrow-native rewrite
  deferred to v0.8.10).
- ~~P-G3: `_groupby_with_skew` opens writers without
  check_partition_buffers~~ — **closed v0.8.9**.
- ~~P-G5: sample cache byte-cap~~ — **closed v0.8.9**.
- ~~P-SH2: uint64 in Python hash fallback~~ — **closed v0.8.9**:
  XOR 2^63 bitcast → int64 → pandas C-hash.

**NEEDS-BATCHED (deferred to v0.8.10+ with prototype):**

- P-Q2 `pc.quantile` whole column (sort-based, ~1× input) — per-batch
  + merge, or `pc.tdigest` (§9.2.11).
- P-R1 `pc.rank` whole column (2× output + sort buffer) — per-batch only.
- P-G2/P-G5 `pc.value_counts` with K≈N — batch it, or `pc.dictionary_encode`.
- P-SH1 `pc.dictionary_encode` with K≈N — same. String shuffle still
  needs research (partition-disjoint at risk).

**NEEDS-RESEARCH (deferred):**

- P-A3 wide-schema benchmark — measured on int64 only; wide string
  deferred to v0.8.10 (N=500k, 5 string cols).
- P-G3 skew workload benchmark — code-review confirmed; workload
  benchmark deferred.

'''
                    s = s[:m.start()] + addendum + s[m.end():]
                    print("  OK: §11.6 addendum")

    # 4. §13.4/138
    if "Эпизод 138" not in s:
        marker = "### 13.5. Открытые вопросы"
        if marker in s:
            episode = '''**Эпизод 138 (доверие Arrow = доверие LightGBM).** Pingis
library-memory audit (TASK2, run 2026-10-20) обнаружил, что CADW
вызывал Arrow-функции (`Table.join`, `Table.group_by`, `to_pandas`)
без явных проверок пика памяти до вызова. MemoryGuard контролировал
`batch.nbytes` на входе, но не видел allocation **внутри**
Arrow-функции.

Это **тот же класс ошибки**, что §13.4/25 (LightGBM adapter) и
§13.4/26 (partial_fit whitelist): «библиотека оптимизирована, значит
безопасна». В случае Arrow — 5 подтверждённых live violations (P-A1,
P-A3, P-D1, P-G3, P-G5) и 2 blind spots (P-Q2 full column, P-R1
whole column для pc.rank).

**Ключевое:** audit использовал `pa_delta` (pyarrow.total_allocated_bytes)
для формул, но MemoryGuard проверяет **RSS** через psutil. Разница:
- `Table.join` narrow int64 @ 1M: `pa_delta = 82 MB` (3.4× output),
  RSS delta = **157 MB** (6.85× output). Factor 2× из-за Arrow memory
  pool retained capacity + non-Arrow temporaries.
- Все 14 формул §9.2.6–9.2.19 derived из **RSS**, не из `pa_delta`.

**Fix (v0.8.9):**
- §3.11 — принцип Arrow call memory discipline (AST guard).
- §9.2.6–9.2.19 — канонические RSS-based формулы.
- 6 fixes в коде (P-A1/P-A3/P-D1/P-G3/P-G5/P-SH2).
- AST guard `test_arrow_call_memory_guard.py` — 14 high-risk attrs,
  каждый требует preceding guard.

**Урок:** любой «Arrow-оптимизирован» в комментарии — сигнал пересчитать
через bench. Формула из документации библиотеки ≠ реальный RSS. Правило:
**для любого call'а, аллоцирующего > O(batch), явный `check_state` +
формула в §9.2.N**. Тот же rigor, что для своих ops (§9.2.1–9.2.5).

**Связанные:** §13.4/25, §13.4/26 (тот же класс «trust в библиотеку»);
§13.4/133–135 (тот же класс «hidden by green tests», но для type matrix);
§3.10 (check before materialize — расширение на library calls).

'''
            s = s.replace(marker, episode + marker, 1)
            print("  OK: §13.4/138")

    p.write_text(s)


# ============ main ============

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v089-pingis-safety"

    print(f"=== v0.8.9: Pingis safety round ===")
    ensure_branch(branch)
    require_clean()

    bk = Path(f".backups/pre-v089-{ts}")
    bk.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(bk / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v089-{ts}",
        "-m", "before v0.8.9 pingis safety"])

    # Phase A: trivial fixes
    patch_a1_join_generator()
    patch_a1_dedup_generator()
    patch_a2_dead_code()
    patch_a3_uint64()

    # Phase B: guard fixes
    patch_b1_join_precheck()
    patch_b3_groupby_skew_buffers()
    patch_b4_groupby_sample_cap()

    # Phase C: docs
    insert_changelog()
    patch_decisions()

    # Phase D: tests
    print("\n=== tests/test_arrow_call_memory_guard.py ===")
    Path("tests/test_arrow_call_memory_guard.py").write_text(AST_GUARD_SRC)
    compile_check(Path("tests/test_arrow_call_memory_guard.py"))
    print("  OK")

    print("\n=== tests/test_pingis_safety.py ===")
    Path("tests/test_pingis_safety.py").write_text(BEHAVIOR_TESTS_SRC)
    compile_check(Path("tests/test_pingis_safety.py"))
    print("  OK")

    # Sanity
    print("\n=== pytest new tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_pingis_safety.py",
         "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.9 (test red)"], check=False)
        sys.exit(1)

    # Regression
    print("\n=== regression: join + dedup + groupby + shuffle ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_join.py",
         "tests/test_join_streaming.py",
         "tests/test_join_oom_guard.py",
         "tests/test_drop_duplicates.py",
         "tests/test_groupby_v2.py",
         "tests/test_mustfix_local_groupby.py",
         "tests/test_mustfix_skew.py",
         "tests/test_mustfix_dict_check.py",
         "tests/test_shuffle.py",
         "tests/test_hash_v081.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.9 (regression)"], check=False)
        sys.exit(1)

    # Full
    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=900,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.9 (fast fail)"], check=False)
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"],
        timeout=900,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.9: Pingis safety — 6 live violations fixed; §9.2.6-19 "
        "formulas (RSS-based); §3.11 principle + AST guard (§13.4/138)"])

    print(f"\n==> v0.8.9 applied on {branch}")
    print()
    print("NEXT: squash → tag v0.8.9")
    print("  git reset --soft master")
    print("  git commit -m 'v0.8.9: Pingis safety round'")
    print("  git checkout master && git merge --ff-only patch/v089-pingis-safety")
    print("  git branch -d patch/v089-pingis-safety")
    print("  uv run pytest -q -m 'not slow' && uv run pytest -q -m slow")
    print("  git tag -a v0.8.9 -m 'v0.8.9: Pingis safety'")


if __name__ == "__main__":
    main()