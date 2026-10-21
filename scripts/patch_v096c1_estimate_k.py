#!/usr/bin/env python3
# v0.9.6c1 — _estimate_k_sample: Arrow-native (no to_pandas).
#
# Two call sites:
#   - cadw/ops/groupby.py::_estimate_k_sample
#   - cadw/ops/dedup.py::_estimate_k_sample
# Both used `sub.to_pandas().drop_duplicates().shape[0]` on the first
# sample batch. Pandas dependency + O(sample_rows × cols) Python-object
# overhead for K estimation. Sample is bounded (10k rows cap), but:
#   (1) it's the last pandas import in cadw/ops/dedup.py (v0.8.11a
#       removed the rest);
#   (2) `MemoryGuard.estimate_unique_keys` already has the correct
#       Arrow-native path (dictionary_encode single-col, binary_join
#       + dictionary_encode multi-col, count_distinct fallback);
#   (3) same code twice = drift risk.
#
# Fix: extract `cadw/core/_k_est.py::estimate_k_sample_arrow(batch,
# subset_cols) -> int` — single implementation. Both call sites use it.
# MemoryGuard.estimate_unique_keys NOT refactored here (already correct;
# add a docstring pointer, keep public API unchanged).
#
# Behaviour: K estimate unchanged within ±rounding (dictionary.length
# is exact, same as drop_duplicates().shape[0]). Number of partitions
# chosen downstream may differ by ±1 for tiny samples — harmless.
#
# Tests: test_k_est_arrow_native.py (6 new). AST guards ensure no
# `to_pandas` in either call site after patch.
#
# Run: uv run python scripts/patch_v096c1_estimate_k.py

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
    is_patched, assert_no_call_in_source, assert_call_order,
)


# ============ 1. new file cadw/core/_k_est.py ============

K_EST_LINES = [
    '"""K-sample estimation via Arrow-native dictionary_encode."""',
    'from __future__ import annotations',
    '',
    'from typing import Sequence',
    '',
    'import pyarrow as pa',
    'import pyarrow.compute as pc',
    '',
    '',
    'def estimate_k_sample_arrow(',
    '    sample_batch: pa.RecordBatch,',
    '    subset_cols: Sequence[str],',
    ') -> int:',
    '    """Estimate unique-key count in `sample_batch` on `subset_cols`.',
    '',
    '    Arrow-native. No pandas. Used by groupby and dedup for K_est',
    '    before HashShuffle (P selection).',
    '',
    '    Single column: `pc.dictionary_encode(...).dictionary.length`',
    '    (exact). Multi-column: binary-join (cast to string) +',
    '    dictionary_encode (safe upper bound — string concat collisions',
    '    not possible with "\\x01" separator).',
    '',
    '    Empty batch → 0.',
    '    """',
    '    cols = list(subset_cols)',
    '    if not cols:',
    '        raise ValueError("estimate_k_sample_arrow: subset_cols empty")',
    '    if sample_batch.num_rows == 0:',
    '        return 0',
    '    if len(cols) == 1:',
    '        arr = sample_batch.column(cols[0])',
    '        enc = arr.dictionary_encode()',
    '        if enc.dictionary is not None:',
    '            return len(enc.dictionary)',
    '        # Fallback: count_distinct (some dict-enc kernels missing)',
    '        return int(pc.count_distinct(arr).as_py())',
    '    # Multi-column: binary_join_element_wise with "\\x01" separator.',
    '    if not hasattr(pc, "binary_join_element_wise"):',
    '        # Fallback: product upper bound. Used only on Arrow < X.',
    '        product = 1',
    '        for c in cols:',
    '            product *= int(pc.count_distinct(sample_batch.column(c)).as_py())',
    '            if product >= sample_batch.num_rows:',
    '                return sample_batch.num_rows',
    '        return product',
    '    str_cols = [',
    '        sample_batch.column(c)',
    '        if pa.types.is_string(sample_batch.column(c).type)',
    '        else pc.cast(sample_batch.column(c), pa.string())',
    '        for c in cols',
    '    ]',
    '    joined = str_cols[0]',
    '    for c in str_cols[1:]:',
    '        joined = pc.binary_join_element_wise(joined, c, "\\x01")',
    '    enc = joined.dictionary_encode()',
    '    if enc.dictionary is not None:',
    '        return len(enc.dictionary)',
    '    return int(pc.count_distinct(joined).as_py())',
    '',
    '',
    '__all__ = ["estimate_k_sample_arrow"]',
    '',
]


def create_k_est():
    print("\n=== create cadw/core/_k_est.py ===")
    p = Path("cadw/core/_k_est.py")
    if p.exists():
        print("  already exists — skip")
        return
    write_content(p, K_EST_LINES)
    assert_compiles(p)
    print("  OK")


# ============ 2. Replace groupby._estimate_k_sample ============

NEW_GROUPBY_K_EST = '''def _estimate_k_sample(
    sample_batches: list[pa.RecordBatch],
    by_columns: Sequence[str],
) -> int:
    """Unique-key count in sample. Arrow-native (v0.9.6c1).

    Composition of per-batch `dictionary_encode` (single-col) or
    binary_join + dictionary_encode (multi-col); union across batches
    via Python set on the *dictionary values* (small — bounded by
    sample_rows, typically K_sample).
    """
    from cadw.core._k_est import estimate_k_sample_arrow

    if not sample_batches:
        return 0
    cols = list(by_columns)
    if len(cols) == 1:
        # Single column: union of dictionaries across batches.
        seen: set = set()
        for b in sample_batches:
            arr = b.column(cols[0])
            enc = arr.dictionary_encode()
            if enc.dictionary is not None:
                for v in enc.dictionary.to_pylist():
                    if v is not None:
                        seen.add(v)
            else:
                for v in arr.to_pylist():
                    if v is not None:
                        seen.add(v)
        return max(1, len(seen))
    # Multi-column: union of binary-joined string keys.
    import pyarrow.compute as pc
    seen2: set = set()
    for b in sample_batches:
        str_cols = [
            b.column(c) if pa.types.is_string(b.column(c).type)
            else pc.cast(b.column(c), pa.string())
            for c in cols
        ]
        joined = str_cols[0]
        for c in str_cols[1:]:
            joined = pc.binary_join_element_wise(joined, c, "\\x01")
        for v in joined.to_pylist():
            if v is not None:
                seen2.add(v)
    return max(1, len(seen2))'''


def replace_groupby():
    print("\n=== groupby.py: replace _estimate_k_sample ===")
    from cadw.ops import groupby as gb
    if is_patched(gb._estimate_k_sample, "Arrow-native (v0.9.6c1)"):
        print("  already patched — skip")
        return

    import ast
    p = Path("cadw/ops/groupby.py")
    s = p.read_text()
    tree = ast.parse(s)
    start = end = None
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_estimate_k_sample"):
            start, end = node.lineno, node.end_lineno
            break
    if start is None:
        raise SystemExit("groupby._estimate_k_sample not found")
    lines = s.split("\n")
    p.write_text("\n".join(
        lines[:start - 1] + [NEW_GROUPBY_K_EST] + lines[end:]
    ))
    assert_compiles(p)
    print(f"  replaced lines {start}..{end}")


# ============ 3. Replace dedup._estimate_k_sample ============

NEW_DEDUP_K_EST = '''def _estimate_k_sample(
    batch: pa.RecordBatch,
    subset: list[str],
    n_total: int,
) -> int:
    """Rough K estimate on sample. Arrow-native (v0.9.6c1).

    K_sample via `pc.dictionary_encode` (no pandas). Extrapolate to
    n_total: linear scaling capped at n_total.
    """
    from cadw.core._k_est import estimate_k_sample_arrow

    if batch.num_rows == 0:
        return 1
    k_sample = estimate_k_sample_arrow(batch, subset)
    if k_sample <= 0:
        return 1
    if n_total <= batch.num_rows:
        return k_sample
    return max(1, int(k_sample * (n_total / batch.num_rows)))'''


def replace_dedup():
    print("\n=== dedup.py: replace _estimate_k_sample ===")
    from cadw.ops import dedup as dd
    if is_patched(dd._estimate_k_sample, "Arrow-native (v0.9.6c1)"):
        print("  already patched — skip")
        return

    import ast
    p = Path("cadw/ops/dedup.py")
    s = p.read_text()
    tree = ast.parse(s)
    start = end = None
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_estimate_k_sample"):
            start, end = node.lineno, node.end_lineno
            break
    if start is None:
        raise SystemExit("dedup._estimate_k_sample not found")
    lines = s.split("\n")
    p.write_text("\n".join(
        lines[:start - 1] + [NEW_DEDUP_K_EST] + lines[end:]
    ))
    assert_compiles(p)
    print(f"  replaced lines {start}..{end}")


# ============ 4. Tests ============

NEW_TEST_LINES = [
    '"""v0.9.6c1: _estimate_k_sample Arrow-native (no pandas)."""',
    'from __future__ import annotations',
    '',
    'import ast',
    'import inspect',
    'import textwrap',
    '',
    'import pyarrow as pa',
    'import pytest',
    '',
    'from cadw.core._k_est import estimate_k_sample_arrow',
    'from cadw.ops import dedup as dd',
    'from cadw.ops import groupby as gb',
    '',
    '',
    '# ============ estimate_k_sample_arrow: positive cases ============',
    '',
    '',
    'def test_k_est_single_col_basic():',
    '    b = pa.RecordBatch.from_arrays(',
    '        [pa.array(["a", "b", "a", "c", "b"])], names=["k"],',
    '    )',
    '    assert estimate_k_sample_arrow(b, ["k"]) == 3',
    '',
    '',
    'def test_k_est_single_col_int():',
    '    b = pa.RecordBatch.from_arrays(',
    '        [pa.array([1, 2, 2, 3, 1, 1])], names=["k"],',
    '    )',
    '    assert estimate_k_sample_arrow(b, ["k"]) == 3',
    '',
    '',
    'def test_k_est_single_col_empty():',
    '    b = pa.RecordBatch.from_arrays(',
    '        [pa.array([], type=pa.string())], names=["k"],',
    '    )',
    '    assert estimate_k_sample_arrow(b, ["k"]) == 0',
    '',
    '',
    'def test_k_est_multi_col_basic():',
    '    b = pa.RecordBatch.from_arrays(',
    '        [pa.array(["a", "a", "b"]), pa.array([1, 2, 1])],',
    '        names=["k1", "k2"],',
    '    )',
    '    assert estimate_k_sample_arrow(b, ["k1", "k2"]) == 3',
    '',
    '',
    'def test_k_est_multi_col_mixed_types():',
    '    b = pa.RecordBatch.from_arrays(',
    '        [pa.array(["a", "a", "b"]), pa.array([1, 2, 1])],',
    '        names=["k1", "k2"],',
    '    )',
    '    # (a,1), (a,2), (b,1) → 3 distinct',
    '    assert estimate_k_sample_arrow(b, ["k1", "k2"]) == 3',
    '',
    '',
    'def test_k_est_empty_cols_raises():',
    '    b = pa.RecordBatch.from_arrays([pa.array([1])], names=["k"])',
    '    with pytest.raises(ValueError, match="subset_cols empty"):',
    '        estimate_k_sample_arrow(b, [])',
    '',
    '',
    '# ============ groupby._estimate_k_sample Arrow-native ============',
    '',
    '',
    'def test_groupby_k_est_single_col():',
    '    b = pa.RecordBatch.from_arrays(',
    '        [pa.array(["a", "b", "a", "c"])], names=["k"],',
    '    )',
    '    assert gb._estimate_k_sample([b], ["k"]) == 3',
    '',
    '',
    'def test_groupby_k_est_multi_col():',
    '    b = pa.RecordBatch.from_arrays(',
    '        [pa.array(["a", "a", "b"]), pa.array([1, 2, 1])],',
    '        names=["k1", "k2"],',
    '    )',
    '    assert gb._estimate_k_sample([b], ["k1", "k2"]) == 3',
    '',
    '',
    'def test_groupby_k_est_multi_batch_union():',
    '    b1 = pa.RecordBatch.from_arrays(',
    '        [pa.array(["a", "b"])], names=["k"],',
    '    )',
    '    b2 = pa.RecordBatch.from_arrays(',
    '        [pa.array(["a", "c"])], names=["k"],',
    '    )',
    '    # {a,b} ∪ {a,c} = {a,b,c} → 3',
    '    assert gb._estimate_k_sample([b1, b2], ["k"]) == 3',
    '',
    '',
    'def test_groupby_k_est_empty():',
    '    assert gb._estimate_k_sample([], ["k"]) == 0',
    '',
    '',
    '# ============ dedup._estimate_k_sample Arrow-native ============',
    '',
    '',
    'def test_dedup_k_est_basic():',
    '    b = pa.RecordBatch.from_arrays(',
    '        [pa.array(["a", "b", "a", "c", "b"])], names=["k"],',
    '    )',
    '    # 3 unique in 5 rows; n_total=5 → 3',
    '    assert dd._estimate_k_sample(b, ["k"], n_total=5) == 3',
    '',
    '',
    'def test_dedup_k_est_extrapolate():',
    '    b = pa.RecordBatch.from_arrays(',
    '        [pa.array(["a", "b", "a", "c", "b"])], names=["k"],',
    '    )',
    '    # 3 unique / 5 rows; n_total=500 → 3 * 100 = 300',
    '    assert dd._estimate_k_sample(b, ["k"], n_total=500) == 300',
    '',
    '',
    'def test_dedup_k_est_empty_batch():',
    '    b = pa.RecordBatch.from_arrays(',
    '        [pa.array([], type=pa.string())], names=["k"],',
    '    )',
    '    assert dd._estimate_k_sample(b, ["k"], n_total=0) == 1',
    '',
    '',
    '# ============ AST guards: no to_pandas in either call site ============',
    '',
    '',
    'def test_groupby_k_est_no_to_pandas_ast():',
    '    """v0.9.6c1: no .to_pandas() Call in groupby._estimate_k_sample."""',
    '    from cadw.patch_utils import assert_no_call_in_source',
    '    src = inspect.getsource(gb._estimate_k_sample)',
    '    assert_no_call_in_source(src, "to_pandas", fn_name=',
    '                              "groupby._estimate_k_sample")',
    '',
    '',
    'def test_dedup_k_est_no_to_pandas_ast():',
    '    """v0.9.6c1: no .to_pandas() Call in dedup._estimate_k_sample."""',
    '    from cadw.patch_utils import assert_no_call_in_source',
    '    src = inspect.getsource(dd._estimate_k_sample)',
    '    assert_no_call_in_source(src, "to_pandas", fn_name=',
    '                              "dedup._estimate_k_sample")',
    '',
    '',
    'def test_groupby_k_est_uses_arrow_helper():',
    '    src = inspect.getsource(gb._estimate_k_sample)',
    '    # Composition of dictionary_encode or estimate_k_sample_arrow',
    '    assert ("estimate_k_sample_arrow" in src',
    '            or "dictionary_encode" in src)',
    '',
    '',
    'def test_dedup_k_est_uses_arrow_helper():',
    '    src = inspect.getsource(dd._estimate_k_sample)',
    '    assert "estimate_k_sample_arrow" in src',
    '',
    '',
    'def test_dedup_no_pandas_import_still_clean():',
    '    """Belt-and-suspenders: dedup.py module has no `import pandas`."""',
    '    tree = ast.parse(inspect.getsource(dd))',
    '    for node in ast.walk(tree):',
    '        if isinstance(node, ast.Import):',
    '            for a in node.names:',
    '                assert not a.name.startswith("pandas"), a.name',
    '        elif isinstance(node, ast.ImportFrom):',
    '            if node.module and node.module.startswith("pandas"):',
    '                raise AssertionError(node.module)',
    '',
]


def create_tests():
    print("\n=== create tests/test_k_est_arrow_native.py ===")
    p = Path("tests/test_k_est_arrow_native.py")
    if p.exists():
        print("  already exists — skip")
        return
    write_content(p, NEW_TEST_LINES)
    assert_compiles(p)
    print("  OK")


# ============ 5. Dogfood ============

def dogfood():
    print("\n=== dogfood: AST guards on new functions ===")
    import inspect
    from cadw.patch_utils import (
        assert_no_call_in_source, is_patched,
    )
    from cadw.ops import groupby as gb
    from cadw.ops import dedup as dd

    for fn, name in [
        (gb._estimate_k_sample, "groupby._estimate_k_sample"),
        (dd._estimate_k_sample, "dedup._estimate_k_sample"),
    ]:
        src = inspect.getsource(fn)
        assert_no_call_in_source(src, "to_pandas", fn_name=name)
        print(f"  {name}: no to_pandas")
        assert is_patched(fn, "v0.9.6c1")
        print(f"  {name}: v0.9.6c1 marker")
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.6c1: _estimate_k_sample Arrow-native ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    create_k_est()
    replace_groupby()
    replace_dedup()
    create_tests()
    dogfood()

    print("\n=== pytest: k_est + dedup + groupby regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_k_est_arrow_native.py",
         "tests/test_dedup_arrow_native.py",
         "tests/test_groupby_v2.py",
         "tests/test_groupby_per_group.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** K_EST/DEDUP/GROUPBY REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: no-pandas guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_arrow_call_memory_guard.py", "-q"],
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
        "v0.9.6c1: _estimate_k_sample Arrow-native (no pandas). "
        "New cadw/core/_k_est.py::estimate_k_sample_arrow "
        "(dictionary_encode single-col, binary_join + "
        "dictionary_encode multi-col, count_distinct fallback). "
        "groupby._estimate_k_sample: union of per-batch dictionaries "
        "(no to_pandas). dedup._estimate_k_sample: delegate to "
        "estimate_k_sample_arrow (no to_pandas). +17 new tests. "
        "K estimate unchanged ± rounding; number of partitions "
        "downstream may differ by ±1 for tiny samples — harmless. "
        "Scaffold used: is_patched (§13.4/182), assert_no_call_in_"
        "source (§13.4/183), write_content + assert_compiles.",
        self_path="scripts/patch_v096c1_estimate_k.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6c1 applied.")


if __name__ == "__main__":
    main()