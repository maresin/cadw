#!/usr/bin/env python3
"""v0.6.1b: loc[tuple] — MultiIndex label access.

Что сделано:
- _LocIndexer._materialized: убрать NotImplementedError для multi;
  вернуть MaterializedMultiIndex (no-filter — из _index_levels;
  filtered — из p.index).
- Остальные методы (_slice, _many, __getitem__) работают через duck-typing
  (get_indexer / has_duplicates / is_monotonic / get_positions_all) —
  все есть у MaterializedMultiIndex.

Тесты (12): scalar, slice, open endpoints, list, missing, reverse, filter,
duplicates.

Run: uv run python scripts/patch_v061b_loc_multi.py
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
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-800:]}")
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
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


# ============ Patch _LocIndexer._materialized (AST-based) ============

def patch_materialized():
    print("\n=== cadw/indexers.py: _LocIndexer._materialized ===")
    p = Path("cadw/indexers.py")
    s = p.read_text()

    if "MaterializedMultiIndex(p._index_levels)" in s:
        print("  already multi-aware — skip")
        return

    tree = ast.parse(s)
    target = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_LocIndexer":
            for method in node.body:
                if (isinstance(method, ast.FunctionDef)
                        and method.name == "_materialized"):
                    target = method
                    break
            break
    if target is None:
        raise SystemExit("_LocIndexer._materialized not found via AST")

    lines = s.split("\n")
    start = target.lineno - 1
    end = target.end_lineno
    old_first = lines[start]
    indent_str = " " * (len(old_first) - len(old_first.lstrip()))

    I = indent_str
    new_body = "\n".join([
        f"{I}def _materialized(self):",
        f'{I}    """v0.6.1b: returns MaterializedIndex or MaterializedMultiIndex."""',
        f"{I}    from cadw.dataset.index import (",
        f"{I}        MaterializedIndex,",
        f"{I}        MaterializedMultiIndex,",
        f"{I}    )",
        f"{I}    p = self._parent",
        f'{I}    if p._index_kind == "range":',
        f"{I}        raise TypeError(",
        f'{I}            "loc: requires MaterializedIndex (after set_index). "',
        f'{I}            "RangeIndex — use iloc."',
        f"{I}        )",
        f"{I}    if p._row_filters:",
        f"{I}        idx = p.index",
        f"{I}        if not isinstance(idx, (MaterializedIndex, MaterializedMultiIndex)):",
        f"{I}            raise TypeError(",
        f'{I}                f"loc: filtered index is {{type(idx).__name__}}, "',
        f'{I}                f"expected MaterializedIndex or MaterializedMultiIndex"',
        f"{I}            )",
        f"{I}        return idx",
        f'{I}    if p._index_kind == "multi":',
        f"{I}        assert p._index_levels is not None",
        f"{I}        return MaterializedMultiIndex(p._index_levels)",
        f"{I}    assert p._index_values is not None",
        f"{I}    return MaterializedIndex(p._index_values)",
    ])

    new_lines = lines[:start] + new_body.split("\n") + lines[end:]
    s = "\n".join(new_lines)
    p.write_text(s)
    compile_check(p)
    print("  OK: _materialized — multi-aware (AST line-replace)")


# ============ tests ============

TESTS_SRC = '''"""v0.6.1b: loc[tuple] — MultiIndex label access."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.dataset.index import (
    MaterializedIndex,
    MaterializedMultiIndex,
    RangeIndex,
)


def make_multi_df(tmp_path, data=None, chunk=2):
    if data is None:
        data = {
            "a": ["x", "x", "y", "y", "z"],
            "b": [1, 2, 1, 2, 1],
            "v": [10, 20, 30, 40, 50],
        }
    t = pa.table(data)
    p = Path(tmp_path) / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, Path(tmp_path) / "ds", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / "spill",
    )
    return df.set_index(["a", "b"], drop=False)


def write_read(df, tmp_path, cols=None):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out).to_pylist()


# ---------- loc[tuple] scalar ----------

def test_loc_tuple_scalar(tmp_path):
    df = make_multi_df(tmp_path)
    row = df.loc[("x", 1)]
    assert isinstance(row, dict)
    assert row == {"a": "x", "b": 1, "v": 10}


def test_loc_tuple_scalar_middle(tmp_path):
    df = make_multi_df(tmp_path)
    row = df.loc[("y", 2)]
    assert row == {"a": "y", "b": 2, "v": 40}


def test_loc_tuple_missing_raises(tmp_path):
    df = make_multi_df(tmp_path)
    with pytest.raises(KeyError, match="not in index"):
        df.loc[("x", 99)]


# ---------- loc[tuple_lo:tuple_hi] slice ----------

def test_loc_tuple_slice_inclusive(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[("x", 1):("y", 2)]
    rows = write_read(sub, tmp_path)
    assert [(r["a"], r["b"], r["v"]) for r in rows] == [
        ("x", 1, 10), ("x", 2, 20), ("y", 1, 30), ("y", 2, 40),
    ]


def test_loc_tuple_slice_single(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[("x", 1):("x", 1)]
    rows = write_read(sub, tmp_path)
    assert [(r["a"], r["b"]) for r in rows] == [("x", 1)]


def test_loc_tuple_slice_open_start(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[:("y", 1)]
    rows = write_read(sub, tmp_path)
    assert [(r["a"], r["b"]) for r in rows] == [
        ("x", 1), ("x", 2), ("y", 1),
    ]


def test_loc_tuple_slice_open_stop(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[("y", 2):]
    rows = write_read(sub, tmp_path)
    assert [(r["a"], r["b"]) for r in rows] == [("y", 2), ("z", 1)]


def test_loc_tuple_slice_open_both(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[:]
    rows = write_read(sub, tmp_path)
    assert len(rows) == 5


def test_loc_tuple_slice_reverse_empty(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[("y", 2):("x", 1)]
    rows = write_read(sub, tmp_path)
    assert rows == []


def test_loc_tuple_slice_missing_endpoint_raises(tmp_path):
    df = make_multi_df(tmp_path)
    with pytest.raises(KeyError, match="not in index"):
        df.loc[("x", 99):("y", 1)]
    with pytest.raises(KeyError, match="not in index"):
        df.loc[("x", 1):("y", 99)]


# ---------- loc[list of tuples] ----------

def test_loc_tuple_list(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[[("x", 1), ("y", 2), ("z", 1)]]
    rows = write_read(sub, tmp_path)
    assert [(r["a"], r["b"]) for r in rows] == [
        ("x", 1), ("y", 2), ("z", 1),
    ]


def test_loc_tuple_list_missing_raises(tmp_path):
    df = make_multi_df(tmp_path)
    with pytest.raises(KeyError, match="not in index"):
        df.loc[[("x", 1), ("x", 99)]]


def test_loc_tuple_list_empty(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[[]]
    rows = write_read(sub, tmp_path)
    assert rows == []


# ---------- filtered view ----------

def test_loc_tuple_filtered(tmp_path):
    df = make_multi_df(tmp_path)
    filtered = df[df["v"] > 15]
    row = filtered.loc[("y", 1)]
    assert row == {"a": "y", "b": 1, "v": 30}
    with pytest.raises(KeyError):
        filtered.loc[("x", 1)]  # not in filtered


# ---------- duplicates → CadwDataFrame ----------

def test_loc_tuple_duplicates_returns_cadwdf(tmp_path):
    df = make_multi_df(tmp_path, data={
        "a": ["x", "x", "y"],
        "b": [1, 1, 2],
        "v": [10, 20, 30],
    })
    sub = df.loc[("x", 1)]
    assert isinstance(sub, CadwDataFrame)
    rows = write_read(sub, tmp_path)
    assert [r["v"] for r in rows] == [10, 20]


def test_loc_tuple_slice_duplicates_raises(tmp_path):
    df = make_multi_df(tmp_path, data={
        "a": ["x", "x", "y"],
        "b": [1, 1, 2],
        "v": [10, 20, 30],
    })
    with pytest.raises(NotImplementedError, match="unique index"):
        df.loc[("x", 1):("y", 2)]


# ---------- single-index regression ----------

def test_loc_single_index_still_works(tmp_path):
    """Single-index loc not broken by multi change."""
    t = pa.table({"k": ["a", "b", "c"], "v": [1, 2, 3]})
    p = Path(tmp_path) / "s.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, Path(tmp_path) / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / "spill",
    ).set_index("k", drop=False)

    assert isinstance(df.index, MaterializedIndex)
    assert df.loc["b"] == {"k": "b", "v": 2}


def test_loc_range_index_still_raises(tmp_path):
    """RangeIndex loc — TypeError (unchanged)."""
    t = pa.table({"v": [1, 2]})
    p = Path(tmp_path) / "r.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, Path(tmp_path) / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / "spill",
    )
    with pytest.raises(TypeError, match="requires MaterializedIndex"):
        df.loc[0]


def test_loc_tuple_on_single_index_raises_keyerror(tmp_path):
    """Tuple label on single-index — KeyError (not tuple index)."""
    t = pa.table({"k": ["a", "b"], "v": [1, 2]})
    p = Path(tmp_path) / "s2.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, Path(tmp_path) / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / "spill",
    ).set_index("k", drop=False)
    with pytest.raises(KeyError):
        df.loc[("a", 1)]


# ---------- multi via iloc list preserves ----------

def test_iloc_list_multi_loc_consistency(tmp_path):
    """iloc[list] and loc[list] agree on same rows."""
    df = make_multi_df(tmp_path)
    by_iloc = write_read(df.iloc[[0, 2, 4]], tmp_path)
    by_loc = write_read(
        df.loc[[("x", 1), ("y", 1), ("z", 1)]], tmp_path,
    )
    assert by_iloc == by_loc
'''


def write_tests():
    print("\n=== tests/test_loc_multi.py ===")
    p = Path("tests/test_loc_multi.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: 22 tests")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v061b-loc-multi"

    print("=== v0.6.1b: loc[tuple] ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v061b-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v061b-{ts_str}",
        "-m", "before patch v0.6.1b loc[tuple]"])

    patch_materialized()
    write_tests()

    print("\n=== pytest tests/test_loc_multi.py tests/test_loc.py "
          "tests/test_multiindex.py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_loc_multi.py",
         "tests/test_loc.py",
         "tests/test_multiindex.py",
         "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.6.1b (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.6.1b: loc[tuple] (MultiIndex label access; slice/list/scalar)"])

    print("\n==> v0.6.1b applied on", branch)


if __name__ == "__main__":
    main()