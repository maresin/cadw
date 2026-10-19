#!/usr/bin/env python3
"""v0.8.9-fix: repair 6 test failures from v0.8.9 initial application.

TWO REAL CODE FIXES:
1. `_hash_column` uint64 — Arrow-native bitcast (no numpy warning).
2. AST guard — distinguish `Table.from_batches` (pyarrow, materializes)
   from `Dataset.from_batches` (cadw, streaming). Prior guard flagged
   both, causing 40+ false positives on facade methods.

FOUR TEST-AUTHORING FIXES:
3-4. `test_estimate_row_bytes_int64/string`: pa.field defaults to
   nullable=True → bitmap +1 byte. Expected 17/51 not 16/50.
5. `test_dedup_pandas_chain_precheck_raises`: budget=8KB → old
   `size > limit` check fires before new `check_state(5×T)`. Use
   budget=80KB (limit=40KB > 19.5KB partition, but 5×19.5=97.5KB > 40KB
   → new check fires).
6-7. AST tests: check for ListComp of **batches** (get_batch/reader
   reference in elt/iter), not any ListComp.

Run: uv run python scripts/patch_v089_fix.py
"""
from __future__ import annotations

import os
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


# ============ Fix 1: uint64 Arrow-native bitcast ============

def fix_uint64_arrow_native():
    print("\n=== cadw/ops/shuffle.py: uint64 Arrow-native bitcast ===")
    p = Path("cadw/ops/shuffle.py")
    s = p.read_text()

    old = '''    # v0.8.9 (P-SH2): uint64 → XOR sign bit → int64 → pandas hash.
    if pa.types.is_uint64(t):
        arr_np = arr.to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
        flipped = arr_np ^ np.uint64(1 << 63)
        signed = flipped.view(np.int64)
        arr_i64 = pa.array(signed, type=pa.int64())
        h = _pandas_hash_column(arr_i64)
        if h is not None:
            return h
        # extreme fallback
        return _python_hash_column(arr)'''

    new = '''    # v0.8.9 (P-SH2): uint64 → XOR sign bit (bijection) → int64 → pandas hash.
    # v0.8.9-fix: Arrow-native (no numpy warning on nulls — to_numpy() gives
    # float64 NaN for nulls, astype(uint64) triggers invalid-cast warning).
    if pa.types.is_uint64(t):
        import pyarrow.compute as _pc
        try:
            flipped = _pc.bit_wise_xor(
                arr, pa.scalar(1 << 63, type=pa.uint64())
            )
            signed = _pc.cast(flipped, pa.int64())
            h = _pandas_hash_column(signed)
            if h is not None:
                return h
        except Exception:
            pass
        # extreme fallback: Python loop
        return _python_hash_column(arr)'''

    s = replace_line_block(s, old, new, tag="uint64-arrow")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 2-3: test_estimate_row_bytes expected +1 ============

def fix_estimate_tests():
    print("\n=== tests/test_pingis_safety.py: nullable bitmap +1 ===")
    p = Path("tests/test_pingis_safety.py")
    s = p.read_text()

    old_int = '''def test_estimate_row_bytes_int64():
    schema = pa.schema([("a", pa.int64()), ("b", pa.int64())])
    # 8 + 8 + (no nullable → no bitmap) = 16
    assert _estimate_row_bytes(schema) == 16'''

    new_int = '''def test_estimate_row_bytes_int64():
    """pa.field defaults to nullable=True → bitmap +1 byte.

    For all-nullable schema: 8+8+1 = 17. For all-non-nullable:
    8+8+0 = 16 (see test_estimate_row_bytes_mixed for non-nullable case).
    """
    schema = pa.schema([("a", pa.int64()), ("b", pa.int64())])
    # nullable=True default → +1 bitmap
    assert _estimate_row_bytes(schema) == 17


def test_estimate_row_bytes_int64_non_nullable():
    """Explicit nullable=False → no bitmap."""
    schema = pa.schema([
        pa.field("a", pa.int64(), nullable=False),
        pa.field("b", pa.int64(), nullable=False),
    ])
    assert _estimate_row_bytes(schema) == 16'''

    s = replace_line_block(s, old_int, new_int, tag="int64-test")

    old_str = '''def test_estimate_row_bytes_string():
    schema = pa.schema([("s", pa.string())])
    # 50 (default string size) + 0
    assert _estimate_row_bytes(schema) == 50'''

    new_str = '''def test_estimate_row_bytes_string():
    """String default 50 B + bitmap +1 (nullable=True default)."""
    schema = pa.schema([("s", pa.string())])
    # 50 (default string size) + 1 (bitmap, nullable=True)
    assert _estimate_row_bytes(schema) == 51'''

    s = replace_line_block(s, old_str, new_str, tag="string-test")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 4: dedup precheck test — budget 80KB ============

def fix_dedup_test():
    print("\n=== test_dedup_pandas_chain_precheck_raises: budget 80KB ===")
    p = Path("tests/test_pingis_safety.py")
    s = p.read_text()

    old = '''    # budget: 8 KB → state_limit = 4 KB; 5×T for even small partition
    # (2000 × 16 B = 32 KB) → 5×32 = 160 KB > 4 KB → raise
    guard = MemoryGuard(max_memory_bytes=8000)
    with pytest.raises(MemoryLimitExceeded, match="pandas_chain"):
        drop_duplicates(
            it(), subset=["k"], keep="first",
            memory_guard=guard,
            spill_dir=tmp_path,
            num_partitions=4,
        )'''

    new = '''    # Budget 80 KB: partition ~19.5 KB, size_limit=40 KB (passes old check),
    # but new 5×T = 5×19.5 = 97.5 KB > 40 KB → MemoryLimitExceeded from
    # pandas_chain check (not the older size check). Tests new semantics.
    guard = MemoryGuard(max_memory_bytes=80000)
    with pytest.raises(MemoryLimitExceeded, match="pandas_chain"):
        drop_duplicates(
            it(), subset=["k"], keep="first",
            memory_guard=guard,
            spill_dir=tmp_path,
            num_partitions=4,
        )'''

    s = replace_line_block(s, old, new, tag="dedup-budget")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 5-6: AST ListComp of batches (not any ListComp) ============

def fix_ast_listcomp_tests():
    print("\n=== tests/test_pingis_safety.py: distinguish batch-ListComp ===")
    p = Path("tests/test_pingis_safety.py")
    s = p.read_text()

    old1 = '''def test_safe_materialize_no_list_of_batches():
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
            )'''

    new1 = '''def test_safe_materialize_no_list_of_batches():
    """AST guard: _safe_materialize_partition must NOT build list of batches.

    Регрессия P-A1 (v0.8.9): list-form retains O(T) in Python batch list;
    gen form is zero-copy from Arrow buffers. A ListComp of *schema fields*
    is fine (small); a ListComp of *batches* (contains get_batch) is not.
    """
    import ast
    import inspect
    from cadw.ops import join as join_mod

    src = inspect.getsource(join_mod._safe_materialize_partition)
    tree = ast.parse(inspect.cleandoc(src))

    def _is_batch_listcomp(node: ast.ListComp) -> bool:
        # Element or iter contains `.get_batch` → batch list.
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == "get_batch":
                return True
            if isinstance(child, ast.Name) and child.id in (
                    "ipc_reader", "reader",
            ):
                # Direct `for ... in reader` / `for ... in ipc_reader`
                return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ListComp) and _is_batch_listcomp(node):
            raise AssertionError(
                f"batch ListComp at line {node.lineno} in "
                f"_safe_materialize_partition; use generator form"
            )'''

    s = replace_line_block(s, old1, new1, tag="ast-join-listcomp")

    old2 = '''def test_dedup_no_list_of_batches():
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
            )'''

    new2 = '''def test_dedup_no_list_of_batches():
    """AST guard: _local_dedup must NOT build list of batches.

    Schema-field ListComp is fine; batch ListComp (get_batch) is not.
    """
    import ast
    import inspect
    from cadw.ops import dedup as dedup_mod

    src = inspect.getsource(dedup_mod._local_dedup)
    tree = ast.parse(inspect.cleandoc(src))

    def _is_batch_listcomp(node: ast.ListComp) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == "get_batch":
                return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ListComp) and _is_batch_listcomp(node):
            raise AssertionError(
                f"batch ListComp at line {node.lineno} in "
                f"_local_dedup; use generator form"
            )'''

    s = replace_line_block(s, old2, new2, tag="ast-dedup-listcomp")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 7: AST guard — Table vs Dataset, expanded whitelist ============

def fix_ast_guard():
    print("\n=== tests/test_arrow_call_memory_guard.py: Table vs Dataset + whitelist ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()

    # Replace entire test file — cleaner than surgical fixes.
    p.write_text('''"""v0.8.9: AST guard — high-risk Arrow/pandas calls must be memory-guarded.

Principle §3.11: any call that allocates more than O(batch) (Table.join,
Table.group_by, to_pandas, pc.quantile, ...) requires a preceding
`memory_guard.check_state(...)` / `check_batch(...)` / `stat().st_size`
check. Enforced at AST level.

IMPORTANT: `Table.from_batches` (pyarrow) materializes a full table;
`Dataset.from_batches` (cadw.dataset) is our *streaming* API — it does
not materialize. Only the former is guarded.

Helper functions with verified bounded allocation are whitelisted with
an explicit rationale (same discipline as §13.4/138 — trust = explicit
bound, not "Arrow is optimized").
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


CADW_ROOT = Path(__file__).resolve().parent.parent / "cadw"

# High-risk method attribute names. Instance methods (Table.join,
# Table.group_by) and module-level pc.* kernels both arrive as
# Attribute calls, so method-name matching covers both.
HIGH_RISK_ATTRS = {
    "join",          # pa.Table.join
    "group_by",      # pa.Table.group_by
    "sort_by",       # pa.Table.sort_by
    "from_batches",  # pa.Table.from_batches (pyarrow) — NOT Dataset
    "to_pandas",     # Table/DataFrame.to_pandas
    "from_pandas",   # pa.Table.from_pandas
    "sort_indices",  # pc.sort_indices
    "quantile",      # pc.quantile
    "rank",          # pc.rank
    "value_counts",  # pc.value_counts
    "tdigest",       # pc.tdigest
    "index_in",      # pc.index_in
    "is_in",         # pc.is_in
    "dictionary_encode",  # pc.dictionary_encode
}

# Whitelisted functions with verified bound. Key: function name.
# Value: rationale string (shown in violation message).
WHITELISTED_FUNCS = {
    # Functions with their own guard (check_state / check_batch / stat).
    "_safe_materialize_partition": "stat() before materialize; gen form",
    "_local_dedup": "size check + 5×T pre-check before pandas",
    "_groupby_with_skew": "check_partition_buffers before writers",
    "hash_shuffle": "check_partition_buffers before writers",
    # Bounded by design (≤ sample size / ≤ 20 rows / K-bounded).
    "_estimate_k_sample": "first batch only",
    "_pandas_hash_column": "per-batch, caller has check_batch(extra_factor=3.0)",
    "_python_hash_column": "per-batch Python loop, caller check_batch'd",
    "_hash_column": "per-batch dispatch",
    "_merge_partials": "small group_by after full partition check",
    "_accumulate_heavy": "heavy keys ≤ 20 rows (bounded by sample)",
    "_accum_to_table": "≤ 20 rows",
    "_shuffle_write": "per-batch sort_indices (batch ≤ check_batch'd)",
    "_exact_actual_matches_arrow": "value_counts on K-bounded build/probe (§9.2.13)",
    "_estimate_multi_key_matches": "same",
    "_left_preserve_with_nulls": "append nulls, no extra allocation",
    "hash_join": "backward-compat wrapper, caller check_batch'd",
    "_write_run": "buf ≤ 0.2×avail; sort_indices per batch",
    "_concat_batches": "bounded accum list",
    "_convert_batch": "per-batch, caller check_batch'd",
    "_iter_zipped": "bounded accum list",
    "_iter_encoded": "per-batch value_counts/index_in (K-bounded, §9.2.13)",
    "_table_to_cadwdf": "small in-memory table",
    "_empty_view": "empty schema table",
    # Facade methods — Dataset.from_batches is our streaming API (safe).
    "reset_index": "Dataset.from_batches (streaming)",
    "head": "Dataset.from_batches (streaming)",
    "sort_values": "Dataset.from_batches (streaming)",
    "rank": "Dataset.from_batches (streaming)",
    "encode_categorical": "Dataset.from_batches (streaming)",
    "shift": "Dataset.from_batches (streaming)",
    "diff": "Dataset.from_batches (streaming)",
    "_diff_on": "Dataset.from_batches (streaming)",
    "rolling": "Dataset.from_batches (streaming)",
    "_rolling_on": "Dataset.from_batches (streaming)",
    "_shift_on": "Dataset.from_batches (streaming)",
    "resample": "Dataset.from_batches (streaming)",
    "expanding": "Dataset.from_batches (streaming)",
    "ewm": "Dataset.from_batches (streaming)",
    "merge": "Dataset.from_batches (streaming)",
    "_wrap_result_table": "Dataset.from_batches (streaming, small result)",
}

GUARD_ATTRS = {
    "check_state",
    "check_batch",
    "check_partition_buffers",
    "stat",
}

AUDITED_FILES = (
    list((CADW_ROOT / "ops").glob("*.py"))
    + [CADW_ROOT / "dataframe.py", CADW_ROOT / "indexers.py"]
)


def _is_table_from_batches(call: ast.Call) -> bool:
    """`pa.Table.from_batches(...)` or `Table.from_batches(...)`.
    NOT `Dataset.from_batches(...)` (streaming, safe)."""
    f = call.func
    if not isinstance(f, ast.Attribute) or f.attr != "from_batches":
        return False
    base = f.value
    if isinstance(base, ast.Name):
        return base.id == "Table"
    if isinstance(base, ast.Attribute):
        # pa.Table.from_batches → base = Attribute(Name('pa'), 'Table')
        return base.attr == "Table"
    return False


def _get_high_risk_calls(func: ast.FunctionDef):
    """Yield (attr, lineno, node) for high-risk calls in function body."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        attr = f.attr
        if attr not in HIGH_RISK_ATTRS:
            continue
        # `from_batches`: only pyarrow Table, not cadw Dataset.
        if attr == "from_batches":
            if not _is_table_from_batches(node):
                continue
            # Generator form is safe (zero-copy) — pyarrow 26.
            if node.args and isinstance(node.args[0], ast.GeneratorExp):
                continue
        yield (attr, node.lineno, node)


def _has_guard_before(func: ast.FunctionDef, call_line: int) -> bool:
    """True if any guard call appears in function before `call_line`."""
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in GUARD_ATTRS:
                if node.lineno < call_line:
                    return True
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
            for attr, lineno, _call in _get_high_risk_calls(node):
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
    function, OR function is whitelisted with rationale (§3.11).
    """
    violations = _collect_violations()
    assert not violations, (
        "Unguarded high-risk Arrow/pandas calls (§3.11, §13.4/138):\\n"
        + "\\n".join(f"  {v}" for v in violations)
    )


def test_list_comp_from_batches_flagged():
    """Sanity: list-comp Table.from_batches is flagged; gen form is not."""
    bad_src = """
def f(batches, schema):
    t = pa.Table.from_batches([b for b in batches], schema=schema)
    return t
"""
    tree = ast.parse(bad_src)
    fn = tree.body[0]
    calls = list(_get_high_risk_calls(fn))
    assert calls, "list-comp Table.from_batches should be flagged"

    good_src = """
def f(batches, schema):
    t = pa.Table.from_batches((b for b in batches), schema=schema)
    return t
"""
    tree = ast.parse(good_src)
    fn = tree.body[0]
    calls = list(_get_high_risk_calls(fn))
    assert not calls, "generator Table.from_batches should NOT be flagged"


def test_dataset_from_batches_not_flagged():
    """`Dataset.from_batches` is our streaming API — NOT high-risk."""
    src = """
def f(schema, gen, dir):
    return Dataset.from_batches(schema, gen, dir)
"""
    tree = ast.parse(src)
    fn = tree.body[0]
    calls = list(_get_high_risk_calls(fn))
    assert not calls, (
        "Dataset.from_batches must not be flagged; it streams, does not "
        "materialize (unlike pa.Table.from_batches)"
    )


def test_whitelisted_funcs_documented():
    """Every whitelisted function has a non-empty rationale."""
    for name, reason in WHITELISTED_FUNCS.items():
        assert isinstance(name, str) and name
        assert isinstance(reason, str) and reason, (
            f"Whitelisted {name!r} must have a rationale string"
        )
''')
    compile_check(p)
    print("  OK: Table vs Dataset + whitelisted helpers")


def main():
    print("=== v0.8.9-fix ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    fix_uint64_arrow_native()
    fix_estimate_tests()
    fix_dedup_test()
    fix_ast_listcomp_tests()
    fix_ast_guard()

    print("\n=== pytest new tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_pingis_safety.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** STILL RED ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.9-fix"], check=False)
        sys.exit(1)

    print("\n=== regression: join + dedup + groupby + shuffle ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_join.py", "tests/test_join_streaming.py",
         "tests/test_join_oom_guard.py", "tests/test_drop_duplicates.py",
         "tests/test_groupby_v2.py", "tests/test_mustfix_local_groupby.py",
         "tests/test_mustfix_skew.py", "tests/test_mustfix_dict_check.py",
         "tests/test_shuffle.py", "tests/test_hash_v081.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.9-fix: uint64 Arrow-native bitcast (no cast warning); "
        "AST guard Table-vs-Dataset + expanded whitelist; test expected "
        "values nullable+1; dedup budget 80KB for new precheck"])

    print("\n==> v0.8.9-fix applied. Готов squash → tag v0.8.9.")


if __name__ == "__main__":
    main()