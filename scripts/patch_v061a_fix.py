#!/usr/bin/env python3
"""v0.6.1a-fix: доработать cadw/indexers.py (partial state after v0.6.1a failure).

v0.6.1a применил 4 файла (dataset/index.py, dataframe.py,
dataset/__init__.py, cadw/__init__.py), но упал на indexers.py —
p.write_text() не выполнился, все 3 подпатча потеряны.

Fix:
1. cadw/indexers.py: _rebuild_index (multi), _table_to_cadwdf (multi),
   _LocIndexer._materialized (multi-check) — через AST, не exact text.
2. tests/test_multiindex.py — если отсутствует.
3. Run pytest.

Run: uv run python scripts/patch_v061a_fix.py
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str),
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1000:]}")
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
        print(r.stderr[-1000:])
        sys.exit(1)
    print(f"  compile OK: {path}")


# ============ AST-based insert for multi check ============

def insert_multi_check_ast(s: str) -> tuple[str, bool]:
    """Insert multi check into _LocIndexer._materialized after range check.

    Uses AST — robust to comments/whitespace variations.
    Returns (new_source, changed).
    """
    tree = ast.parse(s)
    target = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_LocIndexer":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "_materialized":
                    target = method
                    break
            break
    if target is None:
        raise SystemExit("_LocIndexer._materialized not found via AST")

    # Check if multi check already present
    src_segment = ast.get_source_segment(s, target)
    if 'p._index_kind == "multi"' in (src_segment or ""):
        print("  _materialized: multi check already present — skip")
        return s, False

    # Find range check If
    for stmt in target.body:
        if not isinstance(stmt, ast.If):
            continue
        test = stmt.test
        if (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Attribute)
                and test.left.attr == "_index_kind"
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "range"):
            # Range check found. Insert multi check right after its body.
            last_stmt = stmt.body[-1]
            end_line = last_stmt.end_lineno  # 1-based
            lines = s.split("\n")
            if_lines = lines[stmt.lineno - 1]
            indent = len(if_lines) - len(if_lines.lstrip())
            indent_str = " " * indent
            new_lines = [
                "",
                f'{indent_str}if p._index_kind == "multi":',
                f'{indent_str}    raise NotImplementedError(',
                f'{indent_str}        "loc: MultiIndex not supported in v0.6.1a. "',
                f'{indent_str}        "Use reset_index + loc, or iloc. v0.6.1b adds loc[tuple]."',
                f'{indent_str}    )',
            ]
            out = "\n".join(lines[:end_line] + new_lines + lines[end_line:])
            return out, True

    raise SystemExit("range check If not found in _materialized")


# ============ Patch cadw/indexers.py ============

def patch_indexers():
    print("\n=== cadw/indexers.py ===")
    p = Path("cadw/indexers.py")
    s = p.read_text()

    # --- 1. _rebuild_index: multi-aware ---
    if "MaterializedMultiIndex" in s:
        print("  _rebuild_index: already multi-aware (skip)")
    else:
        old = '''def _rebuild_index(parent: "CadwDataFrame", raw_positions: list[int]):
    """Rebuild index for iloc[list] result (v0.4.0d — filter-compat).

    `raw_positions` — позиции в **unfiltered** dataset (из
    _iter_filtered_batches_with_positions).

    - parent RangeIndex → MaterializedIndex([start + p, ...]).
    - parent MaterializedIndex → subset `_index_values[p]`.
    """
    from cadw.dataset.index import MaterializedIndex

    if parent._index_kind == "range":
        vals = [parent._index_start + p for p in raw_positions]
        return MaterializedIndex(pa.array(vals, type=pa.int64()))

    src = parent._index_values
    vals = [src[p] for p in raw_positions]
    return MaterializedIndex(pa.array(vals, type=src.type))'''
        new = '''def _rebuild_index(parent: "CadwDataFrame", raw_positions: list[int]):
    """Rebuild index for iloc[list] result (v0.6.1a: multi-aware).

    - RangeIndex → MaterializedIndex([start+p, ...]).
    - MaterializedIndex → subset.
    - MaterializedMultiIndex → per-level subset.
    """
    from cadw.dataset.index import MaterializedIndex, MaterializedMultiIndex

    if parent._index_kind == "range":
        vals = [parent._index_start + p for p in raw_positions]
        return MaterializedIndex(pa.array(vals, type=pa.int64()))

    if parent._index_kind == "multi":
        assert parent._index_levels is not None
        idx = pa.array(raw_positions, type=pa.int64())
        new_levels = [lv.take(idx) for lv in parent._index_levels]
        return MaterializedMultiIndex(new_levels)

    src = parent._index_values
    vals = [src[p] for p in raw_positions]
    return MaterializedIndex(pa.array(vals, type=src.type))'''
        if old in s:
            s = s.replace(old, new, 1)
            print("  _rebuild_index: multi-aware")
        else:
            print("  WARN: _rebuild_index pattern not found (may need manual check)")

    # --- 2. _table_to_cadwdf: multi-aware ---
    old2 = '''    if new_index is not None:
        new._index_kind = "materialized"
        new._index_values = new_index.to_arrow()
        new._index_start = 0
        new._index_stop = len(new_index)
    return new'''
    new2 = '''    if new_index is not None:
        from cadw.dataset.index import MaterializedMultiIndex
        if isinstance(new_index, MaterializedMultiIndex):
            new._index_kind = "multi"
            new._index_values = None
            new._index_levels = list(new_index._levels)
        else:
            new._index_kind = "materialized"
            new._index_values = new_index.to_arrow()
            new._index_levels = None
        new._index_start = 0
        new._index_stop = len(new_index)
    return new'''
    if 'new._index_levels = list(new_index._levels)' in s:
        print("  _table_to_cadwdf: already multi-aware (skip)")
    elif old2 in s:
        s = s.replace(old2, new2, 1)
        print("  _table_to_cadwdf: multi-aware")
    else:
        print("  WARN: _table_to_cadwdf pattern not found")

    # --- 3. _LocIndexer._materialized: multi check via AST ---
    try:
        s, changed = insert_multi_check_ast(s)
        if changed:
            print("  _materialized: multi check inserted (AST)")
    except SystemExit as e:
        print(f"  ERROR in AST insert: {e}")
        raise

    p.write_text(s)
    compile_check(p)


# ============ Write tests ============

def write_tests():
    print("\n=== tests/test_multiindex.py ===")
    p = Path("tests/test_multiindex.py")
    if p.exists() and p.stat().st_size > 500:
        print("  already present — skip")
        return

    TESTS = '''"""v0.6.1a: MaterializedMultiIndex + set_index(list) + reset_index explode."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.dataset.index import (
    RangeIndex,
    MaterializedIndex,
    MaterializedMultiIndex,
)


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {
            "a": ["x", "x", "y", "y", "z"],
            "b": [1, 2, 1, 2, 1],
            "v": [10, 20, 30, 40, 50],
        }
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- MaterializedMultiIndex ----------

def test_multi_index_nlevels():
    idx = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    assert idx.nlevels == 2 and len(idx) == 2


def test_multi_index_getitem_int():
    idx = MaterializedMultiIndex([
        pa.array(["a", "b", "c"]), pa.array([1, 2, 3]),
    ])
    assert idx[0] == ("a", 1) and idx[-1] == ("c", 3)


def test_multi_index_getitem_slice():
    idx = MaterializedMultiIndex([
        pa.array(["a", "b", "c"]), pa.array([1, 2, 3]),
    ])
    sub = idx[1:3]
    assert isinstance(sub, MaterializedMultiIndex)
    assert list(sub) == [("b", 2), ("c", 3)]


def test_multi_index_iter():
    idx = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    assert list(idx) == [("a", 1), ("b", 2)]


def test_multi_index_to_arrow():
    idx = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    arr = idx.to_arrow()
    assert pa.types.is_struct(arr.type)
    assert arr.type.num_fields == 2
    assert arr[0].as_py() == {"level_0": "a", "level_1": 1}


def test_multi_index_equals():
    a = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    b = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    c = MaterializedMultiIndex([pa.array(["a", "c"]), pa.array([1, 2])])
    assert a.equals(b) and not a.equals(c)


def test_multi_index_has_duplicates():
    a = MaterializedMultiIndex([pa.array(["a", "b", "a"]), pa.array([1, 2, 1])])
    assert a.has_duplicates()
    b = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    assert not b.has_duplicates()


def test_multi_index_get_indexer():
    idx = MaterializedMultiIndex([
        pa.array(["a", "b", "c"]), pa.array([1, 2, 3]),
    ])
    assert idx.get_indexer(("b", 2)) == [1]
    assert idx.get_indexer(("z", 99)) == [-1]
    assert idx.get_indexer([("a", 1), ("c", 3)]) == [0, 2]


def test_multi_index_monotonic():
    a = MaterializedMultiIndex([
        pa.array(["a", "a", "b"]), pa.array([1, 2, 1]),
    ])
    assert a.is_monotonic_increasing()
    assert not a.is_monotonic_decreasing()


def test_multi_index_requires_two_levels():
    with pytest.raises(ValueError, match="at least 2"):
        MaterializedMultiIndex([pa.array(["a"])])


def test_multi_index_length_mismatch():
    with pytest.raises(ValueError, match="level 1"):
        MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1])])


# ---------- set_index multi ----------

def test_set_index_multi_drop_true(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index(["a", "b"], drop=True)
    assert isinstance(new.index, MaterializedMultiIndex)
    assert new.index.nlevels == 2
    assert list(new.index) == [("x", 1), ("x", 2), ("y", 1), ("y", 2), ("z", 1)]
    assert "a" not in new.columns and "b" not in new.columns
    assert "v" in new.columns
    assert "a" in df.columns


def test_set_index_multi_drop_false(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index(["a", "b"], drop=False)
    assert isinstance(new.index, MaterializedMultiIndex)
    assert "a" in new.columns and "b" in new.columns


def test_set_index_multi_inplace(tmp_path):
    df = make_df(tmp_path)
    result = df.set_index(["a", "b"], drop=True, inplace=True)
    assert result is df
    assert isinstance(df.index, MaterializedMultiIndex)


def test_set_index_single_still_works(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index("a", drop=False)
    assert isinstance(new.index, MaterializedIndex)
    assert list(new.index) == ["x", "x", "y", "y", "z"]


# ---------- reset_index multi ----------

def test_reset_index_multi_drop_true(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=True)
    restored = with_idx.reset_index(drop=True)
    assert isinstance(restored.index, RangeIndex)
    assert restored.index.stop == 5


def test_reset_index_multi_drop_false(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=True)
    restored = with_idx.reset_index(drop=False)
    assert isinstance(restored.index, RangeIndex)
    assert "level_0" in restored.columns and "level_1" in restored.columns
    out = tmp_path / "o.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["level_0"] for r in rows] == ["x", "x", "y", "y", "z"]
    assert [r["level_1"] for r in rows] == [1, 2, 1, 2, 1]


def test_reset_index_multi_collision(tmp_path):
    df = make_df(tmp_path, data={
        "a": ["x", "y"], "b": [1, 2],
        "level_0": [100, 200], "v": [10, 20],
    })
    with_idx = df.set_index(["a", "b"], drop=True)
    restored = with_idx.reset_index(drop=False)
    assert "level_0" in restored.columns
    assert "level_0_0" in restored.columns


# ---------- filter + iloc + parquet ----------

def test_index_multi_with_filter(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=False)
    filtered = with_idx[with_idx["v"] > 15]
    idx = filtered.index
    assert isinstance(idx, MaterializedMultiIndex)
    assert list(idx) == [("x", 2), ("y", 1), ("y", 2), ("z", 1)]


def test_iloc_list_preserves_multi_index(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=False)
    sub = with_idx.iloc[[0, 2, 4]]
    assert isinstance(sub.index, MaterializedMultiIndex)
    assert list(sub.index) == [("x", 1), ("y", 1), ("z", 1)]


def test_multi_index_via_parquet(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=True)
    restored = with_idx.reset_index(drop=False)
    out = tmp_path / "o.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 5
    assert "level_0" in rows[0] and "level_1" in rows[0]
'''
    p.write_text(TESTS)
    compile_check(p)


# ============ Run ============

def main():
    print("=== v0.6.1a-fix ===")

    # Status check
    print("\n=== git status ===")
    sh(["git", "status", "--short"])

    # Apply fixes
    patch_indexers()
    write_tests()

    # Run tests
    print("\n=== pytest tests/test_multiindex.py ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_multiindex.py", "-v"],
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP commit for debugging ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.6.1a-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.6.1a: MaterializedMultiIndex + set_index(list) + reset_index explode (fix)"])

    print("\n==> v0.6.1a complete.")


if __name__ == "__main__":
    main()