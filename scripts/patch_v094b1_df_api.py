#!/usr/bin/env python3
# v0.9.4b-1: DataFrame API — concat(0|1), nlargest/nsmallest,
# dtypes, info, memory_usage. Uses write_content for new files.
#
# Run: uv run python scripts/patch_v094b1_df_api.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    write_content, assert_compiles, commit_with_guards,
)


def sh(cmd, check=True):
    print("  $ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print("    " + line)
    if check and r.returncode != 0:
        print("    STDERR: " + r.stderr.strip()[-1500:])
        sys.exit(r.returncode)
    return r.returncode


# ============ DataFrame methods ============

DF_METHODS = [
    "    # ---------------- concat / nlargest / info ----------------",
    "",
    "    def nlargest(self, n, column):",
    "        # Sort desc by `column`, return first n rows.",
    "        # O(N log N) sort + head(n). For n << N, an optimized",
    "        # top-k (bounded heap) can be added later (11.6 backlog).",
    "        if not isinstance(n, int) or n < 1:",
    "            raise ValueError(",
    "                f\"nlargest: n must be int >= 1; got {n!r}\"",
    "            )",
    "        if column not in self._columns:",
    "            raise KeyError(",
    "                f\"nlargest: column {column!r} not in {self._columns}\"",
    "            )",
    "        sorted_df = self.sort_values(column, ascending=False)",
    "        return self._wrap_result_table(sorted_df.head(n))",
    "",
    "    def nsmallest(self, n, column):",
    "        # Sort asc by `column`, return first n rows.",
    "        if not isinstance(n, int) or n < 1:",
    "            raise ValueError(",
    "                f\"nsmallest: n must be int >= 1; got {n!r}\"",
    "            )",
    "        if column not in self._columns:",
    "            raise KeyError(",
    "                f\"nsmallest: column {column!r} not in {self._columns}\"",
    "            )",
    "        sorted_df = self.sort_values(column, ascending=True)",
    "        return self._wrap_result_table(sorted_df.head(n))",
    "",
    "    def dtypes(self):",
    "        # Return {col: dtype_str}. pandas-compatible shape (dict).",
    "        return {",
    "            c: str(self._dataset.schema.field(c).type)",
    "            for c in self._columns",
    "        }",
    "",
    "    def info(self, *, compute_non_null=False):",
    "        # Summary: total_rows, filtered_rows, per-column dtype.",
    "        # compute_non_null=True adds per-column non-null count",
    "        # (O(d) Reduce passes: one pass per column).",
    "        n_total = self._dataset.total_rows",
    "        n_filtered = self.count() if self._row_filters else n_total",
    "        cols = {}",
    "        for c in self._columns:",
    "            entry = {\"dtype\": str(self._dataset.schema.field(c).type)}",
    "            if compute_non_null:",
    "                from cadw.ops.reduce import reduce, CountOp",
    "                entry[\"non_null\"] = reduce(",
    "                    self._iter_filtered_batches(),",
    "                    CountOp(c),",
    "                    self._memory_guard,",
    "                    log=False,",
    "                )",
    "            cols[c] = entry",
    "        return {",
    "            \"total_rows\": n_total,",
    "            \"filtered_rows\": n_filtered,",
    "            \"n_columns\": len(self._columns),",
    "            \"columns\": cols,",
    "        }",
    "",
    "    def memory_usage(self):",
    "        # Upper-bound estimate: filtered_rows * per-column bit_width",
    "        # (+ 1 byte bitmap if nullable). Uses join._estimate_row_bytes",
    "        # (bit_width//8 + null bitmap for single-field schema).",
    "        from cadw.ops.join import _estimate_row_bytes",
    "        n = self.count()",
    "        per_col = {}",
    "        total = 0",
    "        for c in self._columns:",
    "            f = self._dataset.schema.field(c)",
    "            sz = _estimate_row_bytes(pa.schema([f]))",
    "            per_col[c] = n * sz",
    "            total += per_col[c]",
    "        return {\"total\": total, \"per_column\": per_col}",
    "",
    "    # ---------------- write ----------------",
]


def insert_df_methods():
    print("\n=== dataframe.py: insert concat/nlargest/info ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def nlargest(self, n, column):" in s:
        print("  already present - skip")
        return
    old = "    # ---------------- write ----------------\n"
    if old not in s:
        raise SystemExit("write anchor not found in dataframe.py")
    new = "\n".join(DF_METHODS) + "\n"
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ Module-level concat ============

CONCAT_BLOCK = [
    "def concat(",
    "    dfs,",
    "    axis: int = 0,",
    ") -> \"CadwDataFrame\":",
    "    # Concatenate CadwDataFrames along rows (axis=0) or columns (axis=1).",
    "    #",
    "    # axis=0 (rows):",
    "    #   Schemas must match exactly (same column names/types, same order).",
    "    #   Streams via chain(iter_batches); total rows = sum(inputs).",
    "    #",
    "    # axis=1 (columns):",
    "    #   Row counts must match. Column names must be unique across inputs.",
    "    #   Per-batch zip: all inputs iterate at fixed batch_size=10000 to",
    "    #   align chunk boundaries (independent of stored chunk layout).",
    "    #",
    "    # Preconditions:",
    "    #   - Row-level filters are applied BEFORE concat (iter_filtered).",
    "    #   - axis=1 positional alignment is valid only when filters preserve",
    "    #     original row order (always true in CADW: filters don't reorder).",
    "    #",
    "    # Raises:",
    "    #   ValueError: empty dfs, invalid axis, schema/row-count mismatch,",
    "    #     duplicate column names (axis=1).",
    "    #   TypeError: non-CadwDataFrame input.",
    "    from cadw.dataframe import CadwDataFrame as _CDF",
    "    if not dfs:",
    "        raise ValueError(\"concat: at least one df required\")",
    "    if axis not in (0, 1):",
    "        raise ValueError(",
    "            f\"concat: axis must be 0 or 1; got {axis!r}\"",
    "        )",
    "    dfs = list(dfs)",
    "    for d in dfs:",
    "        if not isinstance(d, _CDF):",
    "            raise TypeError(",
    "                f\"concat: all dfs must be CadwDataFrame; \"",
    "                f\"got {type(d).__name__}\"",
    "            )",
    "",
    "    first = dfs[0]",
    "",
    "    if axis == 0:",
    "        out_schema = pa.schema(",
    "            [first._dataset.schema.field(c) for c in first._columns]",
    "        )",
    "        for i, d in enumerate(dfs[1:], 1):",
    "            d_schema = pa.schema(",
    "                [d._dataset.schema.field(c) for c in d._columns]",
    "            )",
    "            if d_schema != out_schema:",
    "                raise ValueError(",
    "                    f\"concat(axis=0): schema mismatch between df[0] \"",
    "                    f\"and df[{i}]; expected {out_schema.names}/\"",
    "                    f\"{out_schema.types}, got {d_schema.names}/\"",
    "                    f\"{d_schema.types}\"",
    "                )",
    "",
    "        def _iter_rows():",
    "            for d in dfs:",
    "                yield from d._iter_arrow_batches()",
    "",
    "        new_ds = Dataset.from_batches(",
    "            out_schema, _iter_rows(), first._new_dataset_dir(),",
    "        )",
    "        return _CDF(",
    "            new_ds,",
    "            memory_guard=first._memory_guard,",
    "            spill_dir=first._spill_dir,",
    "        )",
    "",
    "    # axis == 1",
    "    n_first = first.count()",
    "    for i, d in enumerate(dfs[1:], 1):",
    "        n_i = d.count()",
    "        if n_i != n_first:",
    "            raise ValueError(",
    "                f\"concat(axis=1): row count mismatch: df[0]=\"",
    "                f\"{n_first}, df[{i}]={n_i}\"",
    "            )",
    "",
    "    seen = set()",
    "    out_fields = []",
    "    for d in dfs:",
    "        for c in d._columns:",
    "            if c in seen:",
    "                raise ValueError(",
    "                    f\"concat(axis=1): duplicate column {c!r} \"",
    "                    f\"across inputs\"",
    "                )",
    "            seen.add(c)",
    "            out_fields.append(d._dataset.schema.field(c))",
    "    out_schema = pa.schema(out_fields)",
    "    all_names = list(out_schema.names)",
    "",
    "    def _iter_cols():",
    "        iters = [d._iter_arrow_batches(batch_size=10000) for d in dfs]",
    "        for batches in zip(*iters):",
    "            n = batches[0].num_rows",
    "            for b in batches[1:]:",
    "                if b.num_rows != n:",
    "                    raise RuntimeError(",
    "                        f\"concat(axis=1): batch size mismatch at \"",
    "                        f\"streaming: {n} vs {b.num_rows}\"",
    "                    )",
    "            arrays = []",
    "            for b in batches:",
    "                for j in range(b.num_columns):",
    "                    arrays.append(b.column(j))",
    "            yield pa.RecordBatch.from_arrays(arrays, names=all_names)",
    "",
    "    new_ds = Dataset.from_batches(",
    "        out_schema, _iter_cols(), first._new_dataset_dir(),",
    "    )",
    "    return _CDF(",
    "        new_ds,",
    "        memory_guard=first._memory_guard,",
    "        spill_dir=first._spill_dir,",
    "    )",
    "",
]


def insert_concat():
    print("\n=== dataframe.py: insert module-level concat ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "\ndef concat(" in s:
        print("  already present - skip")
        return
    anchor = "def _concat_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:\n"
    if anchor not in s:
        raise SystemExit("_concat_batches anchor not found")
    block = "\n".join(CONCAT_BLOCK) + "\n"
    s = s.replace(anchor, block + anchor, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ Export from cadw __init__ ============

def update_init():
    print("\n=== cadw/__init__.py: export concat ===")
    p = Path("cadw/__init__.py")
    s = p.read_text()
    if "\"concat\"" in s:
        print("  already present - skip")
        return
    old_imp = "from .dataframe import CadwDataFrame\n"
    new_imp = "from .dataframe import CadwDataFrame, concat\n"
    if old_imp not in s:
        raise SystemExit("CadwDataFrame import anchor not found")
    s = s.replace(old_imp, new_imp, 1)
    # __all__
    old_all = '    "CadwDataFrame",\n'
    new_all = '    "CadwDataFrame", "concat",\n'
    if old_all not in s:
        raise SystemExit("__all__ anchor not found")
    s = s.replace(old_all, new_all, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ Tests ============

TESTS = [
    "# v0.9.4b-1: concat / nlargest / nsmallest / dtypes / info /",
    "# memory_usage. New DataFrame-level API surface.",
    "",
    "from pathlib import Path",
    "",
    "import pyarrow as pa",
    "import pyarrow.parquet as pq",
    "import pytest",
    "",
    "from cadw import CadwDataFrame, MemoryGuard, concat",
    "",
    "",
    "def make_df(tmp_path, data, chunk=10, name=\"L\", mem_mb=50):",
    "    t = pa.table(data)",
    "    p = Path(tmp_path) / f\"{name}.parquet\"",
    "    pq.write_table(t, p)",
    "    return CadwDataFrame.from_parquet(",
    "        p, Path(tmp_path) / f\"ds_{name}\", chunk_size=chunk,",
    "        memory_guard=MemoryGuard(max_memory_mb=mem_mb),",
    "        spill_dir=Path(tmp_path) / f\"spill_{name}\",",
    "    )",
    "",
    "",
    "def to_rows(df, tmp_path, name=\"out.parquet\"):",
    "    out = tmp_path / name",
    "    df.write_parquet(out)",
    "    return pq.read_table(out).to_pylist()",
    "",
    "",
    "# ============ concat axis=0 ============",
    "",
    "def test_concat_axis0_basic(tmp_path):",
    "    a = make_df(tmp_path, {\"k\": [1, 2, 3], \"v\": [10, 20, 30]}, name=\"a\")",
    "    b = make_df(tmp_path, {\"k\": [4, 5], \"v\": [40, 50]}, name=\"b\")",
    "    out = concat([a, b], axis=0)",
    "    rows = to_rows(out, tmp_path)",
    "    assert len(rows) == 5",
    "    assert [r[\"k\"] for r in rows] == [1, 2, 3, 4, 5]",
    "",
    "",
    "def test_concat_axis0_single(tmp_path):",
    "    a = make_df(tmp_path, {\"k\": [1, 2], \"v\": [10, 20]}, name=\"a\")",
    "    out = concat([a], axis=0)",
    "    rows = to_rows(out, tmp_path)",
    "    assert len(rows) == 2",
    "",
    "",
    "def test_concat_axis0_schema_mismatch(tmp_path):",
    "    a = make_df(tmp_path, {\"k\": [1], \"v\": [10]}, name=\"a\")",
    "    b = make_df(tmp_path, {\"k\": [2], \"w\": [20]}, name=\"b\")",
    "    with pytest.raises(ValueError, match=\"schema mismatch\"):",
    "        concat([a, b], axis=0)",
    "",
    "",
    "def test_concat_axis0_with_filter(tmp_path):",
    "    a = make_df(tmp_path, {\"k\": [1, 2, 3], \"v\": [10, 20, 30]}, name=\"a\")",
    "    b = make_df(tmp_path, {\"k\": [4, 5], \"v\": [40, 50]}, name=\"b\")",
    "    a_f = a[a[\"k\"] > 1]",
    "    out = concat([a_f, b], axis=0)",
    "    rows = to_rows(out, tmp_path)",
    "    assert [r[\"k\"] for r in rows] == [2, 3, 4, 5]",
    "",
    "",
    "# ============ concat axis=1 ============",
    "",
    "def test_concat_axis1_basic(tmp_path):",
    "    a = make_df(tmp_path, {\"x\": [1, 2, 3]}, name=\"a\")",
    "    b = make_df(tmp_path, {\"y\": [10, 20, 30]}, name=\"b\")",
    "    out = concat([a, b], axis=1)",
    "    rows = to_rows(out, tmp_path)",
    "    assert [r[\"x\"] for r in rows] == [1, 2, 3]",
    "    assert [r[\"y\"] for r in rows] == [10, 20, 30]",
    "",
    "",
    "def test_concat_axis1_single(tmp_path):",
    "    a = make_df(tmp_path, {\"x\": [1, 2]}, name=\"a\")",
    "    out = concat([a], axis=1)",
    "    rows = to_rows(out, tmp_path)",
    "    assert len(rows) == 2",
    "    assert rows[0][\"x\"] == 1",
    "",
    "",
    "def test_concat_axis1_row_mismatch(tmp_path):",
    "    a = make_df(tmp_path, {\"x\": [1, 2, 3]}, name=\"a\")",
    "    b = make_df(tmp_path, {\"y\": [10, 20]}, name=\"b\")",
    "    with pytest.raises(ValueError, match=\"row count mismatch\"):",
    "        concat([a, b], axis=1)",
    "",
    "",
    "def test_concat_axis1_column_collision(tmp_path):",
    "    a = make_df(tmp_path, {\"x\": [1, 2]}, name=\"a\")",
    "    b = make_df(tmp_path, {\"x\": [3, 4]}, name=\"b\")",
    "    with pytest.raises(ValueError, match=\"duplicate column\"):",
    "        concat([a, b], axis=1)",
    "",
    "",
    "def test_concat_empty_list_raises():",
    "    with pytest.raises(ValueError, match=\"at least one df\"):",
    "        concat([], axis=0)",
    "",
    "",
    "def test_concat_invalid_axis(tmp_path):",
    "    a = make_df(tmp_path, {\"x\": [1]}, name=\"a\")",
    "    with pytest.raises(ValueError, match=\"axis must be 0 or 1\"):",
    "        concat([a], axis=2)",
    "",
    "",
    "# ============ nlargest / nsmallest ============",
    "",
    "def test_nlargest_basic(tmp_path):",
    "    a = make_df(tmp_path, {\"x\": [3, 1, 4, 1, 5, 9, 2]}, name=\"a\")",
    "    top = a.nlargest(3, \"x\")",
    "    rows = to_rows(top, tmp_path)",
    "    assert [r[\"x\"] for r in rows] == [9, 5, 4]",
    "",
    "",
    "def test_nlargest_ngtN(tmp_path):",
    "    a = make_df(tmp_path, {\"x\": [1, 2, 3]}, name=\"a\")",
    "    top = a.nlargest(10, \"x\")",
    "    rows = to_rows(top, tmp_path)",
    "    assert [r[\"x\"] for r in rows] == [3, 2, 1]",
    "",
    "",
    "def test_nlargest_invalid_n(tmp_path):",
    "    a = make_df(tmp_path, {\"x\": [1]}, name=\"a\")",
    "    with pytest.raises(ValueError, match=\"n must be int >= 1\"):",
    "        a.nlargest(0, \"x\")",
    "",
    "",
    "def test_nlargest_missing_column(tmp_path):",
    "    a = make_df(tmp_path, {\"x\": [1]}, name=\"a\")",
    "    with pytest.raises(KeyError, match=\"nope\"):",
    "        a.nlargest(1, \"nope\")",
    "",
    "",
    "def test_nsmallest_basic(tmp_path):",
    "    a = make_df(tmp_path, {\"x\": [3, 1, 4, 1, 5, 9, 2]}, name=\"a\")",
    "    bot = a.nsmallest(3, \"x\")",
    "    rows = to_rows(bot, tmp_path)",
    "    assert [r[\"x\"] for r in rows] == [1, 1, 2]",
    "",
    "",
    "def test_nsmallest_ngtN(tmp_path):",
    "    a = make_df(tmp_path, {\"x\": [5, 3, 1]}, name=\"a\")",
    "    bot = a.nsmallest(10, \"x\")",
    "    rows = to_rows(bot, tmp_path)",
    "    assert [r[\"x\"] for r in rows] == [1, 3, 5]",
    "",
    "",
    "# ============ dtypes ============",
    "",
    "def test_dtypes_basic(tmp_path):",
    "    a = make_df(tmp_path, {\"i\": [1, 2], \"s\": [\"a\", \"b\"],",
    "                          \"f\": [1.0, 2.0]}, name=\"a\")",
    "    d = a.dtypes()",
    "    assert d[\"i\"] == \"int64\"",
    "    assert d[\"s\"] == \"string\"",
    "    assert d[\"f\"] == \"double\"",
    "",
    "",
    "# ============ info ============",
    "",
    "def test_info_basic(tmp_path):",
    "    a = make_df(tmp_path, {\"i\": [1, 2, 3], \"s\": [\"a\", \"b\", \"c\"]},",
    "                  name=\"a\")",
    "    inf = a.info()",
    "    assert inf[\"total_rows\"] == 3",
    "    assert inf[\"filtered_rows\"] == 3",
    "    assert inf[\"n_columns\"] == 2",
    "    assert inf[\"columns\"][\"i\"][\"dtype\"] == \"int64\"",
    "",
    "",
    "def test_info_compute_non_null(tmp_path):",
    "    a = make_df(tmp_path, {\"i\": [1, None, 3]}, name=\"a\")",
    "    inf = a.info(compute_non_null=True)",
    "    assert inf[\"columns\"][\"i\"][\"non_null\"] == 2",
    "",
    "",
    "def test_info_with_filter(tmp_path):",
    "    a = make_df(tmp_path, {\"i\": [1, 2, 3, 4]}, name=\"a\")",
    "    a_f = a[a[\"i\"] > 2]",
    "    inf = a_f.info()",
    "    assert inf[\"total_rows\"] == 4",
    "    assert inf[\"filtered_rows\"] == 2",
    "",
    "",
    "# ============ memory_usage ============",
    "",
    "def test_memory_usage_basic(tmp_path):",
    "    a = make_df(tmp_path, {\"i\": [1, 2, 3, 4]}, name=\"a\")",
    "    mu = a.memory_usage()",
    "    assert mu[\"total\"] > 0",
    "    assert mu[\"per_column\"][\"i\"] > 0",
    "",
    "",
    "def test_memory_usage_empty(tmp_path):",
    "    a = make_df(tmp_path, {\"i\": pa.array([], type=pa.int64())},",
    "                 name=\"a\")",
    "    mu = a.memory_usage()",
    "    assert mu[\"total\"] == 0",
    "",
]


def write_tests():
    print("\n=== tests/test_dataframe_v094b.py (new) ===")
    p = Path("tests/test_dataframe_v094b.py")
    write_content(p, TESTS)
    assert_compiles(p)
    print("  OK")


def main():
    print("=== v0.9.4b-1: DataFrame API ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    insert_df_methods()
    insert_concat()
    update_init()
    write_tests()

    print("\n=== pytest: new DataFrame tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_dataframe_v094b.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: dataframe regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
         "tests/test_sort.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
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
        "v0.9.4b-1: DataFrame API — concat(axis=0|1), nlargest/"
        "nsmallest, dtypes, info (compute_non_null opt), memory_usage "
        "(schema-based estimate); write_content dogfood for new tests",
    )
    print("\n==> v0.9.4b-1 applied.")


if __name__ == "__main__":
    main()
