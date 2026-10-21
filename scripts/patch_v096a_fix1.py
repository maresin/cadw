#!/usr/bin/env python3
# v0.9.6a-fix1 — fix header docstring anchor + import smoke test.
#
# Root cause: `old_header` string (module docstring) не совпал с
# файлом побайтово (emoji / em-dash / whitespace). Cosmetic anyway.
#
# Changes:
#   1. Skip block docstring replacement; single-line `.replace()` for
#      the contract bullet only.
#   2. AST span (`_stable_bytes` .. `compute_hashes`) — function names
#      are unambiguous (100% present in file).
#   3. Diagnostic dump if span not found (list of module-level defs).
#   4. Import smoke test after replacement (py_compile only checks
#      syntax; runtime NameError like missing `_dedup_columns` would
#      pass py_compile but fail on import).
#
# Run: uv run python scripts/patch_v096a_fix1.py

from __future__ import annotations
import ast
import os, subprocess, sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    write_content, assert_compiles, commit_with_guards,
)


# ============ 1. Replace hash functions in shuffle.py (robust) ============

NEW_HASH_BLOCK = [
    '# v0.9.6a: fixed hash for nulls (all nulls → one group, matching',
    '# pandas dropna=False groupby semantics). Magic value chosen after',
    '# splitmix64 avalanche — collision with hashed real value ~2^-64.',
    '_NULL_SENTINEL = np.uint64(0x1BADB002CAFEF00D)',
    '',
    '# XXH64 primes (from xxHash reference implementation, XXH64).',
    '_XXH64_P1 = np.uint64(11400714785074694791)',
    '_XXH64_P2 = np.uint64(14029467366897019727)',
    '_XXH64_P3 = np.uint64(1609587929392839161)',
    '_XXH64_P4 = np.uint64(9650029242287828579)',
    '_XXH64_P5 = np.uint64(2870177450012600261)',
    '',
    '',
    'def _rotl64(x: np.ndarray, r: int) -> np.ndarray:',
    '    """Rotate uint64 array left by r bits (r in [0, 63])."""',
    '    r_u = np.uint64(r)',
    '    return (x << r_u) | (x >> (np.uint64(64) - r_u))',
    '',
    '',
    'def _vectorized_hash_numeric(arr: pa.Array) -> np.ndarray:',
    '    """XXH64-round over 8-byte uint64 view. Vectorized, no pandas.',
    '',
    '    - int/uint/bool/timestamp/date/time: cast to int64 (or uint64)',
    '      → bitcast uint64 → XXH64 round.',
    '    - float: bitcast float64 → int64 → uint64 (preserves NaN bit',
    '      pattern; nulls become NaN via to_numpy → hash(NaN) — matches',
    '      pandas groupby(dropna=False) NaN/null coalescing for floats).',
    '    - nulls in int/temporal: overwritten with _NULL_SENTINEL (own',
    '      group, distinct from any integer value).',
    '',
    '    v0.9.6a: own XXH64-based hash. No pandas. Vectorized over the',
    '    batch (O(batch) numpy ops on uint64 arrays).',
    '    """',
    '    n = len(arr)',
    '    t = arr.type',
    '',
    '    if pa.types.is_floating(t):',
    '        np_arr = arr.to_numpy(zero_copy_only=False)  # float64, NaN for nulls',
    '        int_view = np_arr.view(np.uint64)',
    '        null_mask = None  # NaN bit pattern hashed; NaN/null coalesce',
    '    else:',
    '        if pa.types.is_boolean(t):',
    '            int_arr = pc.cast(arr, pa.int64())',
    '        elif (pa.types.is_timestamp(t) or pa.types.is_date(t)',
    '              or pa.types.is_time(t)):',
    '            int_arr = pc.cast(arr, pa.int64())',
    '        elif pa.types.is_integer(t):',
    '            # For uint types, keep unsigned representation to avoid',
    '            # int64 overflow (safe=True cast would raise for >= 2^63).',
    '            int_arr = arr  # already an Arrow array; will view below',
    '        else:',
    '            raise NotImplementedError(',
    '                f"_vectorized_hash_numeric: unsupported dtype {t}"',
    '            )',
    '        if int_arr.null_count > 0:',
    '            null_mask = pc.is_null(int_arr).to_numpy(zero_copy_only=False)',
    '            int_arr = pc.fill_null(int_arr, 0)',
    '        else:',
    '            null_mask = None',
    '        np_arr = int_arr.to_numpy(zero_copy_only=False)',
    '        if pa.types.is_unsigned_integer(t) or pa.types.is_boolean(t):',
    '            int_view = np_arr.astype(np.uint64, copy=False)',
    '        else:',
    '            int_view = np_arr.astype(np.int64, copy=False).view(np.uint64)',
    '',
    '    # XXH64 round (one 8-byte block, seed=0)',
    '    acc = np.full(n, _XXH64_P5, dtype=np.uint64)',
    '    acc = acc + np.uint64(8)  # += len(block_bytes)',
    '    k1 = int_view * _XXH64_P2',
    '    k1 = _rotl64(k1, 31)',
    '    k1 = k1 * _XXH64_P1',
    '    acc = acc ^ k1',
    '    acc = _rotl64(acc, 27) * _XXH64_P1 + _XXH64_P4',
    '',
    '    # XXH64 avalanche (final)',
    '    acc = acc ^ (acc >> np.uint64(33))',
    '    acc = acc * _XXH64_P2',
    '    acc = acc ^ (acc >> np.uint64(29))',
    '    acc = acc * _XXH64_P3',
    '    acc = acc ^ (acc >> np.uint64(32))',
    '',
    '    if null_mask is not None:',
    '        acc[null_mask] = _NULL_SENTINEL',
    '',
    '    return acc',
    '',
    '',
    'def _string_hash_column(arr: pa.Array) -> np.ndarray:',
    '    """Per-element xxhash.xxh64 over utf8/binary bytes. No pandas.',
    '',
    '    v0.9.6a: no repr()/encode() wrapper — column is homogeneous type',
    '    (all str or all bytes), so no cross-type hash-collision concern.',
    '    Cross-batch consistency: same string → same xxh64 output.',
    '    Nulls → _NULL_SENTINEL (own group).',
    '    """',
    '    n = len(arr)',
    '    out = np.empty(n, dtype=np.uint64)',
    '    for i, v in enumerate(arr.to_pylist()):',
    '        if v is None:',
    '            out[i] = _NULL_SENTINEL',
    '        else:',
    '            data = v.encode("utf-8") if isinstance(v, str) else v',
    '            out[i] = xxhash.xxh64(data, seed=0).intdigest()',
    '    return out',
    '',
    '',
    'def _stable_bytes(v) -> bytes:',
    '    """Fallback serialization for exotic types (dictionary/struct/list)."""',
    '    return repr(v).encode("utf-8")',
    '',
    '',
    'def _hash_column(arr: pa.Array) -> np.ndarray:',
    '    """Dispatch: vectorized XXH64 for fixed-width; per-element for strings.',
    '',
    '    v0.9.6a: no pandas. Covers all first-class Arrow dtypes',
    '    (int/float/bool/timestamp/date/time/string/binary). Exotic',
    '    (dictionary/struct/list) → Python loop, rare (not a partition key',
    '    column type in practice).',
    '    """',
    '    t = arr.type',
    '    if (pa.types.is_string(t) or pa.types.is_large_string(t)',
    '            or pa.types.is_binary(t) or pa.types.is_large_binary(t)):',
    '        return _string_hash_column(arr)',
    '    if (pa.types.is_integer(t) or pa.types.is_floating(t)',
    '            or pa.types.is_boolean(t)',
    '            or pa.types.is_timestamp(t) or pa.types.is_date(t)',
    '            or pa.types.is_time(t)):',
    '        return _vectorized_hash_numeric(arr)',
    '    # Exotic fallback (dictionary/struct/list)',
    '    values = arr.to_pylist()',
    '    hashes = [xxhash.xxh64(_stable_bytes(v), seed=0).intdigest()',
    '              for v in values]',
    '    return np.asarray(hashes, dtype=np.uint64)',
    '',
    '',
    'def compute_hashes(',
    '    batch: pa.RecordBatch,',
    '    key_columns: Sequence[str],',
    '    seed: int = 0,',
    ') -> pa.Array:',
    '    """Deterministic uint64 hash of key (boost-combine + splitmix64).',
    '',
    '    v0.9.6a: own XXH64-based hash for numeric/string columns; no pandas.',
    '    Seed applied via XOR before final splitmix64 avalanche (uniform',
    '    across dtypes; deterministic; XOR is bijection — no collision added).',
    '',
    '    Partition-disjoint invariant: same value → same hash within a call;',
    '    nulls → same fixed sentinel. Splitmix64 avalanche ensures uniform',
    '    low bits for partition assignment (v0.8.1-fix2: pandas string hash',
    '    had even low bits, wasting half the partitions).',
    '    """',
    '    if not key_columns:',
    '        raise ValueError("key_columns must be non-empty")',
    '',
    '    h_np: np.ndarray | None = None',
    '    for col in key_columns:',
    '        arr = batch.column(col)',
    '        ch = _hash_column(arr)',
    '        if h_np is None:',
    '            h_np = ch',
    '        else:',
    '            # boost::hash_combine (64-bit) with wrap-around',
    '            mixed = (',
    '                ch',
    '                + np.uint64(0x9E3779B97F4A7C15)',
    '                + (h_np << np.uint64(6))',
    '                + (h_np >> np.uint64(2))',
    '            )',
    '            h_np = h_np ^ mixed',
    '',
    '    assert h_np is not None',
    '    # Splitmix64 avalanche — uniform low bits for partition assignment.',
    '    if seed:',
    '        h_np = h_np ^ np.uint64(seed)',
    '    h_np = (h_np ^ (h_np >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)',
    '    h_np = (h_np ^ (h_np >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)',
    '    h_np = h_np ^ (h_np >> np.uint64(31))',
    '    return pa.array(h_np, type=pa.uint64())',
]


def replace_hash_module():
    print("\n=== shuffle.py: replace pandas hash with own XXH64 (AST-based) ===")
    p = Path("cadw/ops/shuffle.py")
    src = p.read_text()

    if "_NULL_SENTINEL" in src:
        print("  shuffle.py already v0.9.6a — skip")
        return

    # Cosmetic: single-line replace for the contract bullet (safe — exact
    # substring match, not multi-line block).
    old_line = (
        "  - детерминированный хэш (xxhash64, seed) — стабилен между прогонами;"
    )
    new_line = (
        "  - детерминированный XXH64-based хэш (v0.9.6a: own impl, no pandas);"
    )
    if old_line in src:
        src = src.replace(old_line, new_line, 1)
        print("  header bullet updated")
    else:
        print("  header bullet anchor missing (cosmetic, skip)")

    # AST span: _stable_bytes .. compute_hashes inclusive.
    tree = ast.parse(src)
    start = None
    end = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "_stable_bytes":
                start = node.lineno
            if node.name == "compute_hashes" and start is not None:
                end = node.end_lineno
                break

    if start is None or end is None:
        funcs = [
            n.name for n in tree.body if isinstance(n, ast.FunctionDef)
        ]
        print(f"  DIAG: module-level function names: {funcs}")
        raise SystemExit(
            f"AST span not found: start={start}, end={end}. "
            f"Expected _stable_bytes followed by compute_hashes."
        )

    print(f"  replacing lines {start}..{end} ({end - start + 1} lines)")

    lines = src.split("\n")
    new_lines = lines[:start - 1] + NEW_HASH_BLOCK + lines[end:]
    p.write_text("\n".join(new_lines))
    assert_compiles(p)
    print("  py_compile OK")

    # Import smoke test: py_compile only checks syntax; runtime NameError
    # (e.g. leftover reference to _python_hash_column) fails on import.
    print("  running import smoke test...")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import cadw.ops.shuffle as sh; "
         "assert hasattr(sh, '_vectorized_hash_numeric'); "
         "assert hasattr(sh, '_string_hash_column'); "
         "assert hasattr(sh, '_hash_column'); "
         "assert hasattr(sh, 'compute_hashes'); "
         "assert not hasattr(sh, '_pandas_hash_column'); "
         "assert not hasattr(sh, '_python_hash_column')"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        print("  STDERR: " + r.stderr.strip()[-800:])
        raise SystemExit("import smoke test failed — check stderr above")
    print("  import smoke OK")


# ============ 2. Whitelist cleanup (test_arrow_call_memory_guard.py) ============

def update_whitelist():
    print("\n=== test_arrow_call_memory_guard.py: remove pandas hash whitelist ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()

    old_entries = [
        '    "cadw/ops/shuffle.py:_pandas_hash_column":\n'
        '        "per-batch, caller has check_batch(extra_factor=3.0)",\n',
        '    "cadw/ops/shuffle.py:_python_hash_column":\n'
        "        \"per-batch Python loop, caller check_batch'd\",\n",
        '    "cadw/ops/shuffle.py:_hash_column":\n'
        '        "per-batch dispatch to one of the above",\n',
    ]
    removed = 0
    for entry in old_entries:
        if entry in s:
            s = s.replace(entry, "", 1)
            removed += 1
    if removed == 0:
        print("  no entries found — already clean?")
        return
    print(f"  removed {removed} entries")
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. Drop shuffle.py from pandas allow-list ============

def update_no_pandas_allowlist():
    print("\n=== test_no_pandas_in_ops.py: empty ALLOWED_PANDAS_MODULES ===")
    p = Path("tests/test_no_pandas_in_ops.py")
    s = p.read_text()

    old = (
        "ALLOWED_PANDAS_MODULES = {\n"
        "    # §9.2.20: pd.util.hash_pandas_object is the fastest hash for\n"
        "    # int/float/timestamp (19-40× Python fallback). Bounded per-batch\n"
        "    # by caller's check_batch(extra_factor=3.0).\n"
        '    "shuffle.py": "pd.util.hash_pandas_object — §9.2.20 perf exception",\n'
        "}\n"
    )
    new = (
        "ALLOWED_PANDAS_MODULES: dict[str, str] = {\n"
        "    # v0.9.6a: shuffle.py no longer imports pandas (own XXH64 hash).\n"
        "    # Every other module in cadw/ops/ was already pandas-free\n"
        "    # (v0.8.11a P-D1 Arrow-native dedup). Empty by design —\n"
        "    # any new entry requires explicit rationale + review.\n"
        "}\n"
    )
    if old not in s:
        print("  anchor missing — dump current ALLOWED_PANDAS_MODULES context:")
        idx = s.find("ALLOWED_PANDAS_MODULES")
        if idx >= 0:
            print("  " + s[idx:idx+400].replace("\n", "\n  "))
        print("  trying single-entry removal...")
        # fallback: remove only the shuffle.py entry, keep dict
        entry = '    "shuffle.py": "pd.util.hash_pandas_object — §9.2.20 perf exception",\n'
        if entry in s:
            s = s.replace(entry, "", 1)
            p.write_text(s)
            assert_compiles(p)
            print("  removed entry only (comment above remains)")
        else:
            print("  entry also missing — no-op")
        return
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 4. Delete test_shuffle_fallback_warning.py ============

def remove_fallback_test_file():
    print("\n=== delete tests/test_shuffle_fallback_warning.py ===")
    p = Path("tests/test_shuffle_fallback_warning.py")
    if not p.exists():
        print("  already absent — skip")
        return
    p.unlink()
    print("  removed")


# ============ 5. Update test_hash_v081.py ============

def update_test_hash_v081():
    print("\n=== test_hash_v081.py: remove pandas spies, add own-hash tests ===")
    p = Path("tests/test_hash_v081.py")
    src = p.read_text()

    pandas_test_names = {
        "test_pandas_path_used_for_int64",
        "test_string_uses_python_path",
        "test_numeric_uses_pandas_path",
        "test_uint64_uses_pandas_via_bitcast",
        "test_uint64_python_fallback_warns",
    }

    tree = ast.parse(src)
    to_remove = []
    lines = src.split("\n")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in pandas_test_names:
            start = node.lineno
            while start > 1 and lines[start - 2].strip().startswith("#"):
                start -= 1
            while start > 1 and lines[start - 2].strip() == "":
                start -= 1
            to_remove.append((start, node.end_lineno))

    if to_remove:
        to_remove.sort(reverse=True)
        for start, end in to_remove:
            del lines[start - 1:end]
        src = "\n".join(lines)
        print(f"  removed {len(to_remove)} pandas-specific tests")
    else:
        print("  no pandas tests found — skip removal")

    # Add own-hash tests
    if "def test_hash_no_pandas_import_ast" in src:
        print("  own-hash tests already present — skip")
    else:
        new_tests = [
            "",
            "",
            "# ============ v0.9.6a: own XXH64 hash, no pandas ============",
            "",
            "",
            "def test_hash_no_pandas_import_ast():",
            "    # AST: cadw/ops/shuffle.py has no `import pandas`.",
            "    import ast as _ast",
            "    import inspect",
            "    from cadw.ops import shuffle as sh",
            "    src = inspect.getsource(sh)",
            "    tree = _ast.parse(src)",
            "    for node in _ast.walk(tree):",
            "        if isinstance(node, _ast.Import):",
            "            for alias in node.names:",
            '                assert not alias.name.startswith("pandas"), (',
            '                    "shuffle.py must not import pandas (v0.9.6a)"',
            "                )",
            "        elif isinstance(node, _ast.ImportFrom):",
            "            if node.module:",
            '                assert not node.module.startswith("pandas"), (',
            '                    "shuffle.py must not import pandas (v0.9.6a)"',
            "                )",
            "",
            "",
            "def test_own_hash_removed_pandas_functions():",
            "    # Belt-and-suspenders: old function names gone from source.",
            "    import inspect",
            "    from cadw.ops import shuffle as sh",
            "    src = inspect.getsource(sh)",
            '    assert "_pandas_hash_column" not in src',
            '    assert "_python_hash_column" not in src',
            '    assert "_vectorized_hash_numeric" in src',
            '    assert "_string_hash_column" in src',
            "",
            "",
            "def test_own_hash_cross_batch_int64_with_and_without_nulls():",
            '    b1 = batch([1, 2, 3], names=["k"])',
            '    b2 = batch([1, 2, None], names=["k"])',
            '    h1 = compute_hashes(b1, ["k"]).to_pylist()',
            '    h2 = compute_hashes(b2, ["k"]).to_pylist()',
            "    assert h1[0] == h2[0]",
            "    assert h1[1] == h2[1]",
            "",
            "",
            "def test_own_hash_cross_batch_float64_with_and_without_nulls():",
            '    b1 = batch([1.5, 2.5, 3.5], names=["k"])',
            '    b2 = batch([1.5, 2.5, None], names=["k"])',
            '    h1 = compute_hashes(b1, ["k"]).to_pylist()',
            '    h2 = compute_hashes(b2, ["k"]).to_pylist()',
            "    assert h1[0] == h2[0]",
            "    assert h1[1] == h2[1]",
            "",
            "",
            "def test_own_hash_cross_batch_string_with_and_without_nulls():",
            '    b1 = batch(["a", "b"], names=["k"])',
            '    b2 = batch(["a", None], names=["k"])',
            '    h1 = compute_hashes(b1, ["k"]).to_pylist()',
            '    h2 = compute_hashes(b2, ["k"]).to_pylist()',
            "    assert h1[0] == h2[0]",
            "",
            "",
            "def test_own_hash_null_consistency():",
            '    b = batch([None, None, None], names=["k"])',
            '    h = compute_hashes(b, ["k"]).to_pylist()',
            "    assert h[0] == h[1] == h[2]",
            "",
            "",
            "def test_own_hash_uint64_extreme_values():",
            "    # uint64 values >= 2^63 must hash distinctively (safe=False bitcast).",
            "    values = [1, 2**63, 2**63 + 1, 2**64 - 1]",
            "    b = pa.RecordBatch.from_arrays(",
            '        [pa.array(values, type=pa.uint64())], names=["k"],',
            "    )",
            '    h = compute_hashes(b, ["k"]).to_pylist()',
            "    assert len(set(h)) == len(values), (",
            '        f"distinct uint64 values must hash distinctively; got {h}"',
            "    )",
            "",
            "",
            "def test_own_hash_timestamp_cross_batch():",
            "    import datetime as dt",
            "    ts = [dt.datetime(2020, 1, 1), dt.datetime(2020, 6, 1)]",
            "    b1 = pa.RecordBatch.from_arrays(",
            '        [pa.array(ts, type=pa.timestamp("s"))], names=["k"],',
            "    )",
            "    b2 = pa.RecordBatch.from_arrays(",
            '        [pa.array([ts[0], None], type=pa.timestamp("s"))], names=["k"],',
            "    )",
            '    h1 = compute_hashes(b1, ["k"]).to_pylist()',
            '    h2 = compute_hashes(b2, ["k"]).to_pylist()',
            "    assert h1[0] == h2[0]",
            "",
            "",
            "def test_own_hash_avalanche_uniform_low_bits():",
            "    # After splitmix64, low bits uniform for P in {2,4,8,16}.",
            "    import numpy as np_",
            "    n = 5000",
            '    values = [f"g{i:05d}" for i in range(n)]',
            '    b = batch(values, names=["k"])',
            '    h = np_.array(compute_hashes(b, ["k"]).to_pylist(),',
            "                  dtype=np_.uint64)",
            "    for P in (2, 4, 8, 16):",
            "        counts = np_.bincount(h & np_.uint64(P - 1), minlength=P)",
            "        expected = n / P",
            "        assert counts.min() >= expected * 0.7, (",
            "            f'P={P}: min bucket {counts.min()} < {expected*0.7}'",
            "        )",
            "        assert counts.max() <= expected * 1.3",
            "",
        ]
        src = src + "\n".join(new_tests)
    p.write_text(src)
    assert_compiles(p)
    print("  OK")


# ============ 6. Remove pandas test from test_pingis_safety.py ============

def remove_pingis_pandas_test():
    print("\n=== test_pingis_safety.py: remove test_uint64_hash_uses_pandas_path ===")
    p = Path("tests/test_pingis_safety.py")
    src = p.read_text()
    if "_pandas_hash_column" not in src:
        print("  no reference — skip")
        return

    tree = ast.parse(src)
    lines = src.split("\n")
    to_remove = []
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "test_uint64_hash_uses_pandas_path"):
            start = node.lineno
            while start > 1 and lines[start - 2].strip() == "":
                start -= 1
            to_remove.append((start, node.end_lineno))

    if not to_remove:
        print("  test not found — skip")
        return

    for start, end in sorted(to_remove, reverse=True):
        del lines[start - 1:end]
    p.write_text("\n".join(lines))
    assert_compiles(p)
    print("  OK")


# ============ 7. Quick timing benchmark ============

def bench_new_hash():
    print("\n=== bench: new hash timing ===")
    import numpy as np
    import pyarrow as pa
    import xxhash

    from cadw.ops.shuffle import _vectorized_hash_numeric

    n = 1_000_000
    arr = pa.array(np.arange(n, dtype=np.int64))

    # Vectorized (new)
    t0 = time.time()
    for _ in range(3):
        _vectorized_hash_numeric(arr)
    t_vec = (time.time() - t0) / 3
    print(f"  vectorized XXH64:   {t_vec*1000:6.1f} ms  ({n/t_vec/1e6:.1f} M rows/s)")

    # Python loop (old baseline)
    values = arr.to_pylist()
    t0 = time.time()
    for _ in range(3):
        [xxhash.xxh64(repr(v).encode()).intdigest() for v in values]
    t_py = (time.time() - t0) / 3
    print(f"  python loop:        {t_py*1000:6.1f} ms")
    print(f"  speedup vs py-loop: {t_py/t_vec:.1f}×")

    # Pandas (if installed)
    try:
        import pandas as pd
        s = arr.to_pandas()
        t0 = time.time()
        for _ in range(3):
            pd.util.hash_pandas_object(s, index=False).to_numpy(dtype=np.uint64)
        t_pd = (time.time() - t0) / 3
        print(f"  pandas C-hash:      {t_pd*1000:6.1f} ms")
        print(f"  vs pandas:          {t_pd/t_vec:.2f}× slower")
    except ImportError:
        print("  pandas:             not installed (good — not required)")


# ============ main ============

def main():
    print("=== v0.9.6a-fix1: own XXH64 hash, no pandas ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    replace_hash_module()
    update_whitelist()
    update_no_pandas_allowlist()
    remove_fallback_test_file()
    update_test_hash_v081()
    remove_pingis_pandas_test()

    print("\n=== pytest: hash tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_hash_v081.py", "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** HASH TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: shuffle tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_shuffle.py", "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** SHUFFLE TESTS FAILED ***")
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

    print("\n=== pytest: groupby + dedup (regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_groupby_v2.py",
         "tests/test_groupby_per_group.py",
         "tests/test_dedup_arrow_native.py",
         "tests/test_pingis_safety.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** GROUPBY/DEDUP/PINGIS REGRESSION FAILED ***")
        sys.exit(1)

    bench_new_hash()

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
        "v0.9.6a: own XXH64-based hash — no pandas dependency. "
        "_vectorized_hash_numeric (numpy element-wise over uint64 "
        "bitcast for int/uint/bool/float/timestamp/date/time) + "
        "_string_hash_column (per-element xxhash.xxh64, no repr/encode "
        "wrapper). Cross-batch consistency + null sentinel + "
        "splitmix64 avalanche preserved. Removed _pandas_hash_column, "
        "_python_hash_column. test_shuffle_fallback_warning.py deleted. "
        "Whitelist entries removed (no longer needed); "
        "ALLOWED_PANDAS_MODULES empty. Import smoke test added.",
        self_path="scripts/patch_v096a_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6a applied (fix1).")


if __name__ == "__main__":
    main()
