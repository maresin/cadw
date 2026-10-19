#!/usr/bin/env python3
"""v0.6.2a: encode_categorical — Arrow-native dictionary encoding.

Strategy:
    Pass 1: for each column, collect unique values (via value_counts).
            Sorted → deterministic ids.
    Pass 2: pc.index_in(col, value_set=...) per batch → int32.
    Returns (new_df, {col: sorted_values}). new_df has int32 columns
    (same name); original df unchanged.

Memory: O(K) per column (K = unique values), not O(N).
Idempotent: already-encoded (int/float) columns raise TypeError.

Run: uv run python scripts/patch_v062a_encode_categorical.py
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


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


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
        print(f"  branch: checkout {name}")
        sh(["git", "checkout", name])
    else:
        print(f"  branch: create {name}")
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
    print("\n=== cadw/dataframe.py: encode_categorical ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    # Insert before "# ---------------- merge ----------------"
    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")
    if "def encode_categorical" in s:
        print("  already present — skip")
        return

    method = '''    # ---------------- encode_categorical ----------------

    def encode_categorical(
        self,
        columns: "str | Sequence[str]",
    ) -> "tuple[CadwDataFrame, dict[str, list]]":
        """Encode categorical columns as int32 (v0.6.2a).

        - columns: str or Sequence[str]. Must be string/binary columns.
        - Pass 1: collect unique values (pc.value_counts per batch),
          sorted for determinism. Nulls excluded (become null ids).
        - Pass 2: pc.index_in(col, value_set=pa.array(sorted_values))
          per batch → int32 (null stays null).

        Returns:
            (encoded_df, reverse_maps): new CadwDataFrame with int32
            columns (same names); original df unchanged.
            reverse_maps[col] = sorted_values; id k → reverse_maps[col][k].

        Memory: O(K) per column, independent of N.
        Raises TypeError for non-string columns.
        """
        import pyarrow.compute as pc

        if isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("encode_categorical: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"encode_categorical: column {c!r} not in {self._columns}"
                )

        # Validate types — only string/binary encodable.
        for c in cols:
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_string(t) or pa.types.is_large_string(t)
                    or pa.types.is_binary(t) or pa.types.is_large_binary(t)):
                raise TypeError(
                    f"encode_categorical: column {c!r} has type {t}; "
                    f"only string/binary supported"
                )

        # Pass 1: collect unique values per column.
        reverse_maps: dict[str, list] = {}
        for col in cols:
            uniques: set = set()
            for b in self._iter_filtered_batches():
                arr = b.column(col)
                vc = pc.value_counts(arr)  # struct<values, counts>
                if isinstance(vc, pa.ChunkedArray):
                    vc = vc.combine_chunks()
                values = vc.field("values")
                for v in values.to_pylist():  # K per batch, not N
                    if v is not None:
                        uniques.add(v)
            try:
                sorted_vals = sorted(uniques)
            except TypeError:
                sorted_vals = sorted(uniques, key=lambda x: str(x))
            reverse_maps[col] = sorted_vals

        # Pass 2: encode.
        schema = pa.schema(
            [
                (
                    pa.field(c, pa.int32())
                    if c in cols
                    else self._dataset.schema.field(c)
                )
                for c in self._columns
            ]
        )

        def _iter_encoded():
            for b in self._iter_filtered_batches():
                arrays = []
                for c in self._columns:
                    if c in cols:
                        arr = b.column(c)
                        value_set = pa.array(reverse_maps[c], type=arr.type)
                        ids = pc.index_in(arr, value_set=value_set)
                        arrays.append(ids.cast(pa.int32()))
                    else:
                        arrays.append(b.column(c))
                yield pa.RecordBatch.from_arrays(arrays, schema=schema)

        new_ds = Dataset.from_batches(
            schema, _iter_encoded(), self._new_dataset_dir()
        )
        new_df = CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )
        return new_df, reverse_maps

'''
    s = s.replace(anchor, method + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: encode_categorical added")


# ============ tests ============

def write_tests():
    print("\n=== tests/test_encode_categorical.py ===")
    p = Path("tests/test_encode_categorical.py")
    p.write_text('''"""v0.6.2a: encode_categorical (Arrow-native, O(K) memory)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {
            "cat": ["a", "b", "c", "a", "b"],
            "v": [1, 2, 3, 4, 5],
        }
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_cols(df, tmp_path, cols):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out).select(cols).to_pandas()


def test_encode_basic(tmp_path):
    df = make_df(tmp_path)
    encoded, maps = df.encode_categorical("cat")
    assert maps["cat"] == ["a", "b", "c"]
    got = read_cols(encoded, tmp_path, ["cat", "v"])
    assert list(got["cat"]) == [0, 1, 2, 0, 1]
    assert list(got["v"]) == [1, 2, 3, 4, 5]


def test_encode_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.encode_categorical("cat")
    # Original unchanged
    got = read_cols(df, tmp_path, ["cat"])
    assert list(got["cat"]) == ["a", "b", "c", "a", "b"]


def test_encode_dtype_int32(tmp_path):
    df = make_df(tmp_path)
    encoded, _ = df.encode_categorical("cat")
    out = tmp_path / "o.parquet"
    encoded.write_parquet(out)
    t = pq.read_table(out)
    assert t.schema.field("cat").type == pa.int32()


def test_encode_deterministic_order(tmp_path):
    """Sorted values → deterministic ids across runs."""
    df = make_df(tmp_path)
    _, maps1 = df.encode_categorical("cat")
    _, maps2 = df.encode_categorical("cat")
    assert maps1["cat"] == maps2["cat"]


def test_encode_multi_columns(tmp_path):
    df = make_df(tmp_path, data={
        "a": ["x", "y", "x"],
        "b": ["p", "q", "p"],
        "v": [1, 2, 3],
    })
    encoded, maps = df.encode_categorical(["a", "b"])
    assert maps["a"] == ["x", "y"] and maps["b"] == ["p", "q"]
    got = read_cols(encoded, tmp_path, ["a", "b", "v"])
    assert list(got["a"]) == [0, 1, 0]
    assert list(got["b"]) == [0, 1, 0]


def test_encode_missing_column(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.encode_categorical("nope")


def test_encode_non_string_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="only string"):
        df.encode_categorical("v")


def test_encode_empty_columns_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        df.encode_categorical([])


def test_encode_nulls_skip(tmp_path):
    """Nulls: null in source → null id (not 0)."""
    df = make_df(tmp_path, data={
        "cat": ["a", None, "b", None, "a"],
        "v": [1, 2, 3, 4, 5],
    })
    encoded, maps = df.encode_categorical("cat")
    assert maps["cat"] == ["a", "b"]  # no None in map
    got = read_cols(encoded, tmp_path, ["cat"])
    # ids: 0, null, 1, null, 0
    import numpy as np
    vals = got["cat"].tolist()
    assert vals[0] == 0 and vals[2] == 1 and vals[4] == 0
    assert vals[1] is None or (isinstance(vals[1], float) and np.isnan(vals[1]))


def test_encode_multiple_batches(tmp_path):
    """Multi-batch: same value → same id across batches."""
    df = make_df(tmp_path, data={
        "cat": ["a", "b"] * 10,  # 20 rows
        "v": list(range(20)),
    }, chunk=3)
    encoded, maps = df.encode_categorical("cat")
    got = read_cols(encoded, tmp_path, ["cat"])
    assert list(got["cat"]) == [0, 1] * 10


def test_encode_then_roundtrip(tmp_path):
    """encode → save → load → reverse-map."""
    df = make_df(tmp_path)
    encoded, maps = df.encode_categorical("cat")
    out = tmp_path / "o.parquet"
    encoded.write_parquet(out)
    reloaded = CadwDataFrame.from_parquet(
        out, tmp_path / "ds2", chunk_size=3,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill2",
    )
    got = read_cols(reloaded, tmp_path, ["cat"])
    reverse = [maps["cat"][i] for i in got["cat"]]
    assert reverse == ["a", "b", "c", "a", "b"]
''')
    compile_check(p)
    print("  OK: tests written")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v062a-encode-categorical"

    print("=== v0.6.2a: encode_categorical ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v062a-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v062a-{ts_str}",
        "-m", "before patch v0.6.2a encode_categorical"])

    patch_dataframe()
    write_tests()

    print("\n=== pytest tests/test_encode_categorical.py ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_encode_categorical.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.6.2a (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.6.2a: encode_categorical (Arrow-native, O(K) memory; reverse map)"])

    print("\n==> v0.6.2a applied on", branch)


if __name__ == "__main__":
    main()