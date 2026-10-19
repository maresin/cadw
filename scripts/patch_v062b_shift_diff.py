#!/usr/bin/env python3
"""v0.6.2b: shift(k) + diff(k) — streamable lag/diff primitives.

Scope:
    df.shift(k=1, fill_value=None, *, columns=None)  — lag, k >= 0
    df.diff(k=1, *, columns=None)                    — first-difference, k >= 1

Streaming: O(k) memory. Batches iterated in position order
(_iter_filtered_batches). Output = same df with shifted/diffed columns
(shifted columns become nullable to allow fill_value=None).

k < 0 (lead) → NotImplementedError (v0.6.3).

Run: uv run python scripts/patch_v062b_shift_diff.py
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


# ============ Patch dataframe.py ============

def patch_dataframe():
    print("\n=== cadw/dataframe.py: shift + diff ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    if "def shift(" in s and "def diff(" in s:
        print("  shift+diff already present — skip")
        return
    if "def shift(" in s:
        raise SystemExit("shift already present but not diff — manual check")
    if "def diff(" in s:
        raise SystemExit("diff already present but not shift — manual check")

    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")

    methods = '''    # ---------------- shift / diff ----------------

    def shift(
        self,
        k: int = 1,
        fill_value=None,
        *,
        columns: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Shift values by `k` positions (v0.6.2b, lag).

        - k >= 0: `out[i] = x[i-k]` for i>=k, else `fill_value`.
        - k == 0: returns self (no-op).
        - k < 0 (lead): NotImplementedError (v0.6.3).
        - columns=None: all columns; else subset (others pass-through).
        - fill_value=None: null for first k rows. Non-None: that value.

        Streamable: O(k) memory (deque per column). Position-based,
        iterates filtered batches in row order.

        Shifted columns become **nullable** (fill = null).
        """
        from collections import deque

        if not isinstance(k, int):
            raise TypeError(f"shift: k must be int; got {type(k).__name__}")
        if k < 0:
            raise NotImplementedError(
                "shift: negative k (lead) not supported in v0.6.2b; "
                "use reverse + shift + reverse, or v0.6.3"
            )
        if k == 0:
            return self

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"shift: column {c!r} not in {self._columns}"
                )
        shift_cols = set(cols)

        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in shift_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        def _fill(size: int, type_) -> pa.Array:
            if fill_value is None:
                return pa.nulls(size, type=type_)
            return pa.array([fill_value] * size, type=type_)

        def _iter_shifted():
            bufs: dict[str, deque] = {
                c: deque(maxlen=k) for c in shift_cols
            }
            for b in self._iter_filtered_batches():
                n = b.num_rows
                arrays = []
                for c in self._columns:
                    col_vals = b.column(c)
                    if c not in shift_cols:
                        arrays.append(col_vals)
                        continue
                    prev = list(bufs[c])
                    fill_n = max(0, k - len(prev))
                    parts = []
                    if fill_n > 0:
                        parts.append(_fill(fill_n, col_vals.type))
                    if prev:
                        parts.append(pa.array(prev, type=col_vals.type))
                    if n > k:
                        parts.append(col_vals.slice(0, n - k))
                    out = (
                        pa.concat_arrays(parts) if parts
                        else pa.array([], type=col_vals.type)
                    )
                    if len(out) > n:
                        out = out.slice(0, n)
                    elif len(out) < n:
                        out = pa.concat_arrays([
                            out, pa.nulls(n - len(out), type=col_vals.type),
                        ])
                    arrays.append(out)
                    for v in col_vals.to_pylist():
                        bufs[c].append(v)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_shifted(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def diff(
        self,
        k: int = 1,
        *,
        columns: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """First difference: `out[i] = x[i] - x[i-k]` (v0.6.2b).

        - k >= 1. First k rows -> null.
        - columns=None: all; else subset (others pass-through).
        - Output columns nullable (first k = null).

        Streamable: O(k) memory.
        """
        from collections import deque

        if not isinstance(k, int):
            raise TypeError(f"diff: k must be int; got {type(k).__name__}")
        if k < 1:
            raise ValueError(f"diff: k must be >= 1; got {k}")

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"diff: column {c!r} not in {self._columns}"
                )
        diff_cols = set(cols)

        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in diff_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        import pyarrow.compute as _pc

        def _iter_diffed():
            bufs: dict[str, deque] = {
                c: deque(maxlen=k) for c in diff_cols
            }
            for b in self._iter_filtered_batches():
                n = b.num_rows
                arrays = []
                for c in self._columns:
                    col_vals = b.column(c)
                    if c not in diff_cols:
                        arrays.append(col_vals)
                        continue
                    prev = list(bufs[c])
                    fill_n = max(0, k - len(prev))
                    parts = []
                    if fill_n > 0:
                        parts.append(pa.nulls(fill_n, type=col_vals.type))
                    if prev:
                        parts.append(pa.array(prev, type=col_vals.type))
                    if n > k:
                        parts.append(col_vals.slice(0, n - k))
                    shifted = (
                        pa.concat_arrays(parts) if parts
                        else pa.array([], type=col_vals.type)
                    )
                    if len(shifted) > n:
                        shifted = shifted.slice(0, n)
                    elif len(shifted) < n:
                        shifted = pa.concat_arrays([
                            shifted,
                            pa.nulls(n - len(shifted), type=col_vals.type),
                        ])
                    diff_arr = _pc.subtract(col_vals, shifted)
                    arrays.append(diff_arr)
                    for v in col_vals.to_pylist():
                        bufs[c].append(v)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_diffed(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

'''
    s = s.replace(anchor, methods + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: shift + diff added")


# ============ tests ============

TESTS_SRC = '''"""v0.6.2b: shift(k) + diff(k)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {"v": [1, 2, 3, 4, 5], "cat": ["a", "b", "c", "d", "e"]}
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_col(df, tmp_path, col):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


# ---------- shift ----------

def test_shift_k1_basic(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1, columns=["v"])
    assert read_col(res, tmp_path, "v") == [None, 1, 2, 3, 4]


def test_shift_k2(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(2, columns=["v"])
    assert read_col(res, tmp_path, "v") == [None, None, 1, 2, 3]


def test_shift_fill_value(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(2, fill_value=0, columns=["v"])
    assert read_col(res, tmp_path, "v") == [0, 0, 1, 2, 3]


def test_shift_zero_is_noop(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(0, columns=["v"])
    # Same view
    assert res is df or read_col(res, tmp_path, "v") == [1, 2, 3, 4, 5]


def test_shift_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.shift(1, columns=["v"])
    assert read_col(df, tmp_path, "v") == [1, 2, 3, 4, 5]


def test_shift_string_column(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1, columns=["cat"])
    assert read_col(res, tmp_path, "cat") == [None, "a", "b", "c", "d"]


def test_shift_string_fill(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1, fill_value="MISSING", columns=["cat"])
    assert read_col(res, tmp_path, "cat") == [
        "MISSING", "a", "b", "c", "d",
    ]


def test_shift_all_columns_default(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1)
    assert read_col(res, tmp_path, "v") == [None, 1, 2, 3, 4]
    assert read_col(res, tmp_path, "cat") == [None, "a", "b", "c", "d"]


def test_shift_with_nulls_in_input(tmp_path):
    df = make_df(tmp_path, data={
        "v": [1, None, 3, None, 5],
        "cat": ["a", "b", "c", "d", "e"],
    })
    res = df.shift(1, columns=["v"])
    assert read_col(res, tmp_path, "v") == [None, 1, None, 3, None]


def test_shift_k_ge_n(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(10, fill_value=-1, columns=["v"])
    assert read_col(res, tmp_path, "v") == [-1, -1, -1, -1, -1]


def test_shift_negative_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(NotImplementedError, match="negative k"):
        df.shift(-1, columns=["v"])


def test_shift_missing_column_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.shift(1, columns=["nope"])


def test_shift_non_int_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="k must be int"):
        df.shift(1.5, columns=["v"])


def test_shift_filtered(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 2]  # 3, 4, 5
    res = filtered.shift(1, columns=["v"])
    assert read_col(res, tmp_path, "v") == [None, 3, 4]


# ---------- diff ----------

def test_diff_k1_basic(tmp_path):
    df = make_df(tmp_path)
    res = df.diff(1, columns=["v"])
    # 1, 2-1, 3-2, 4-3, 5-4 = 1,1,1,1,1; first null
    assert read_col(res, tmp_path, "v") == [None, 1, 1, 1, 1]


def test_diff_k2(tmp_path):
    df = make_df(tmp_path, data={"v": [1, 2, 4, 8, 16]})
    res = df.diff(2, columns=["v"])
    # None, None, 4-1, 8-2, 16-4 = None,None,3,6,12
    assert read_col(res, tmp_path, "v") == [None, None, 3, 6, 12]


def test_diff_with_nulls(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, None, 3.0, 5.0]})
    res = df.diff(1, columns=["v"])
    # 1, None, 3-1, 5-3 -> None, None, 2.0, 2.0 (pc.subtract null-propagates)
    vals = read_col(res, tmp_path, "v")
    assert vals[0] is None
    assert vals[1] is None
    assert abs(vals[2] - 2.0) < 1e-9
    assert abs(vals[3] - 2.0) < 1e-9


def test_diff_all_columns_default(tmp_path):
    df = make_df(tmp_path, data={"a": [1, 3, 6], "b": [10, 20, 40]})
    res = df.diff(1)
    assert read_col(res, tmp_path, "a") == [None, 2, 3]
    assert read_col(res, tmp_path, "b") == [None, 10, 20]


def test_diff_k0_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match=">= 1"):
        df.diff(0, columns=["v"])


def test_diff_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.diff(1, columns=["v"])
    assert read_col(df, tmp_path, "v") == [1, 2, 3, 4, 5]


def test_diff_subset_only(tmp_path):
    df = make_df(tmp_path, data={
        "a": [1, 2, 3], "b": ["x", "y", "z"],
    })
    res = df.diff(1, columns=["a"])
    assert read_col(res, tmp_path, "a") == [None, 1, 1]
    assert read_col(res, tmp_path, "b") == ["x", "y", "z"]


# ---------- combined ----------

def test_shift_then_diff(tmp_path):
    df = make_df(tmp_path, data={"v": [10, 20, 40, 80]})
    # shift(1) = [None,10,20,40]; diff(1) of shifted = [None,None,10,20]
    res = df.shift(1, columns=["v"]).diff(1, columns=["v"])
    assert read_col(res, tmp_path, "v") == [None, None, 10, 20]


def test_shift_dtypes_preserved(tmp_path):
    df = make_df(tmp_path, data={
        "v_int": [1, 2, 3],
        "v_float": [1.0, 2.0, 3.0],
        "v_str": ["a", "b", "c"],
    })
    res = df.shift(1)
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    schema = pq.read_schema(out)
    assert schema.field("v_int").type == pa.int64()
    assert schema.field("v_float").type == pa.float64()
    assert schema.field("v_str").type == pa.string()
'''


def write_tests():
    print("\n=== tests/test_shift_diff.py ===")
    p = Path("tests/test_shift_diff.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print(f"  OK: tests written")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v062b-shift-diff"

    print("=== v0.6.2b: shift + diff ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v062b-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v062b-{ts_str}",
        "-m", "before patch v0.6.2b shift+diff"])

    patch_dataframe()
    write_tests()

    print("\n=== pytest tests/test_shift_diff.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_shift_diff.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.6.2b (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.6.2b: shift(k) + diff(k) (streamable lag/diff; O(k) memory)"])

    print("\n==> v0.6.2b applied on", branch)


if __name__ == "__main__":
    main()