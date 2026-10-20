#!/usr/bin/env python3
"""v0.9.1f: pivot_table / melt / crosstab — last v0.9.1 patch.

SCOPE:
  - CadwDataFrame.melt(id_vars, value_vars, var_name, value_name)
  - CadwDataFrame.pivot_table(index, columns, values, aggfunc)
  - CadwDataFrame.crosstab(index, columns)
  - AST whitelist: to_pandas in pivot_table (with explicit check_state)

DESIGN (locked-in, per §13.4/148 contract-first docstrings):
  melt:
    - value_vars must share ONE dtype (Arrow column uniform type).
    - id_vars nulls pass through.
    - Output rows = N × len(value_vars).
    - var_name/value_name must not collide with id_vars.
  pivot_table:
    - groupby([*index, columns]).agg(values, aggfunc) → pandas reshape.
    - K_pairs = K_index × K_columns; fail-fast BEFORE groupby via sample
      estimate: k_pairs × row_size × 2 ≤ 0.3 × avail.
    - Filter null in `columns` (dropna=True, standard pandas).
    - aggfunc ∈ {'mean','sum','count','min','max'}, default 'mean'.
    - Missing combos → null cells.
  crosstab:
    - Filter null in index/columns.
    - pivot_table(values=index[0], aggfunc='count').

MEMORY:
  - melt: O(batch × #value_vars).
  - pivot_table: O(K_pairs × row_size) in grouped; O(K_index × K_cols
    × value_size) in pivot. Both checked explicitly.
  - crosstab: same as pivot_table.

GUARDS:
  - test_pivot_table.py (~15 fast): melt (5), pivot (7), crosstab (3).
  - test_pivot_table_e2e.py (1 slow): pivot on 100K rows, R-cells bounded.
  - AST: `pivot_table` has check_state before to_pandas.
  - No stale-behavior grep needed (pure addition, no removal).

Run: uv run python scripts/patch_v091f_pivot_melt.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


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


# ============ 1. Add methods to CadwDataFrame ============

METHODS = '''    # ---------------- pivot / melt / crosstab (v0.9.1f) ----------------

    def melt(
        self,
        id_vars: "str | Sequence[str] | None" = None,
        value_vars: "str | Sequence[str] | None" = None,
        *,
        var_name: str = "variable",
        value_name: str = "value",
    ) -> "CadwDataFrame":
        """Unpivot wide → long (streaming).

        Args:
            id_vars: identifier columns (repeated per melted column).
                     None → no id columns.
            value_vars: columns to melt. None → all except id_vars.
            var_name: name of the "source column" column.
            value_name: name of the "value" column.

        Returns:
            CadwDataFrame with [*id_vars, var_name, value_name].
            Output rows = N × len(value_vars).

        Valid:
            - Non-overlapping id_vars and value_vars.
            - All value_vars share the SAME dtype (Arrow column uniform).
            - var_name and value_name do not collide with id_vars.
            - Non-empty value_vars.

        Invalid (raises):
            - Empty value_vars → ValueError.
            - Overlap id_vars ∩ value_vars → ValueError.
            - Mixed dtypes in value_vars → TypeError.
            - var_name / value_name collides with id_vars → ValueError.
            - Missing columns → KeyError.

        Memory: O(batch × len(value_vars)). Streaming.
        """
        import pyarrow as pa

        if id_vars is None:
            id_list: list[str] = []
        elif isinstance(id_vars, str):
            id_list = [id_vars]
        else:
            id_list = list(id_vars)

        if value_vars is None:
            value_list = [c for c in self._columns if c not in id_list]
        elif isinstance(value_vars, str):
            value_list = [value_vars]
        else:
            value_list = list(value_vars)

        if not value_list:
            raise ValueError(
                "melt: value_vars must be non-empty (all columns are "
                "id_vars?)"
            )
        for c in id_list:
            if c not in self._columns:
                raise KeyError(f"melt: id_vars {c!r} not in {self._columns}")
        for c in value_list:
            if c not in self._columns:
                raise KeyError(
                    f"melt: value_vars {c!r} not in {self._columns}"
                )
            if c in id_list:
                raise ValueError(
                    f"melt: column {c!r} in both id_vars and value_vars"
                )
        if var_name in id_list:
            raise ValueError(
                f"melt: var_name {var_name!r} collides with id_vars"
            )
        if value_name in id_list:
            raise ValueError(
                f"melt: value_name {value_name!r} collides with id_vars"
            )
        if var_name == value_name:
            raise ValueError(
                f"melt: var_name == value_name == {var_name!r}"
            )

        # Type uniformity check
        value_types = [
            self._dataset.schema.field(c).type for c in value_list
        ]
        first_t = value_types[0]
        for c, t in zip(value_list[1:], value_types[1:]):
            if t != first_t:
                raise TypeError(
                    f"melt: value_vars must share one dtype; got "
                    f"{value_list[0]!r}: {first_t} vs {c!r}: {t}. "
                    f"Cast to common type first: df.astype({{...}})."
                )

        out_fields = [self._dataset.schema.field(c) for c in id_list]
        out_fields.append(pa.field(var_name, pa.string()))
        out_fields.append(pa.field(value_name, first_t, nullable=True))
        out_schema = pa.schema(out_fields)

        def _iter():
            for batch in self._iter_arrow_batches():
                self._memory_guard.check_batch(batch, name="melt.input")
                n = batch.num_rows
                if n == 0:
                    continue
                id_arrays = [batch.column(c) for c in id_list]
                for vv in value_list:
                    val_arr = batch.column(vv)
                    var_arr = pa.array([vv] * n, type=pa.string())
                    arrays = [*id_arrays, var_arr, val_arr]
                    yield pa.RecordBatch.from_arrays(
                        arrays, schema=out_schema,
                    )

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def pivot_table(
        self,
        index: "str | Sequence[str]",
        columns: str,
        values: str,
        *,
        aggfunc: str = "mean",
    ) -> "CadwDataFrame":
        """Pivot: long → wide (groupby + reshape).

        Args:
            index: row identifier column(s).
            columns: column whose unique values become output columns.
            values: value column to aggregate.
            aggfunc: {'mean','sum','count','min','max'}, default 'mean'.

        Returns:
            CadwDataFrame with [*index, <columns-values as columns>].
            Missing combinations (K_index × K_columns minus real data)
            → null cells.

        Valid:
            - All columns exist.
            - Non-empty input after dropping null in `columns`.
            - K_pairs = K_index × K_columns fits in memory budget.

        Invalid (raises):
            - Empty index → ValueError.
            - Missing columns → KeyError.
            - aggfunc not in supported set → ValueError.
            - K_pairs too large → MemoryLimitExceeded BEFORE groupby.
            - No rows after null-filter → ValueError.

        Memory: O(K_pairs × row_size) for grouped + O(K_index × K_cols
        × value_size) for pivot. Fail-fast if K_pairs × row_size × 2 >
        0.3 × available.
        """
        import pyarrow as pa
        from cadw.core.memory_guard import MemoryLimitExceeded
        from cadw.core.sampling import peek_first_batch

        if isinstance(index, str):
            index_cols = [index]
        else:
            index_cols = list(index)
        if not index_cols:
            raise ValueError("pivot_table: index must be non-empty")

        for c in index_cols:
            if c not in self._columns:
                raise KeyError(
                    f"pivot_table: index col {c!r} not in {self._columns}"
                )
        if columns not in self._columns:
            raise KeyError(
                f"pivot_table: columns {columns!r} not in {self._columns}"
            )
        if values not in self._columns:
            raise KeyError(
                f"pivot_table: values {values!r} not in {self._columns}"
            )
        if columns in index_cols:
            raise ValueError(
                f"pivot_table: {columns!r} in both index and columns"
            )
        if values in index_cols or values == columns:
            raise ValueError(
                f"pivot_table: values {values!r} must be distinct from "
                f"index/columns"
            )
        if aggfunc not in ("mean", "sum", "count", "min", "max"):
            raise ValueError(
                f"pivot_table: aggfunc must be one of mean/sum/count/"
                f"min/max; got {aggfunc!r}"
            )

        # Drop null rows in `columns` (dropna=True, standard pandas)
        filtered = self[self[columns].notnull()]
        n = filtered.count()
        if n == 0:
            raise ValueError(
                f"pivot_table: no rows after filtering null {columns!r}; "
                f"cannot pivot empty data"
            )

        # Sample-based K_pairs estimate BEFORE groupby
        sample = peek_first_batch(filtered._iter_filtered_batches())
        if sample is None:
            raise ValueError("pivot_table: empty input")
        total_rows = n
        k_index_est = self._memory_guard.estimate_unique_keys(
            sample.select(index_cols), total_rows,
        )
        k_cols_est = self._memory_guard.estimate_unique_keys(
            sample.select([columns]), total_rows,
        )
        k_pairs_est = max(1, k_index_est) * max(1, k_cols_est)

        # Estimated row_size for grouped table (index + columns + agg value)
        row_size = sum(
            self._dataset.schema.field(c).type.bit_width // 8
            for c in index_cols + [columns, values]
            if hasattr(self._dataset.schema.field(c).type, "bit_width")
        )
        if row_size <= 0:
            row_size = 16  # conservative default
        est_bytes = k_pairs_est * row_size * 2
        state_limit = int(self._memory_guard.available_bytes() * 0.3)
        if est_bytes > state_limit:
            raise MemoryLimitExceeded(
                f"pivot_table: K_pairs ~{k_pairs_est} × row_size "
                f"{row_size} B × 2 = {est_bytes} B > 0.3 × avail "
                f"{state_limit} B. K_index ~{k_index_est}, K_columns "
                f"~{k_cols_est}. Reduce cardinality or use groupby() "
                f"(long format). See §11.5."
            )

        # Groupby (long format): [*index, columns, values_agg]
        from cadw.ops.groupby import groupby_aggregate
        grouped_table = groupby_aggregate(
            filtered._iter_filtered_batches(),
            by_columns=index_cols + [columns],
            agg_columns=[values],
            agg_funcs=[aggfunc],
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "pivot",
            log=False,
        )
        # v0.9.1f: explicit check_state before to_pandas (AST guard).
        # grouped_table is K_pairs rows, small (< est_bytes by
        # construction — groupby enforces same budget internally).
        self._memory_guard.check_state(
            grouped_table.nbytes, name="pivot_table.grouped",
        )
        value_col = f"{values}_{aggfunc}"

        # Reshape via pandas (small table)
        grouped_pd = grouped_table.to_pandas()
        pivoted_pd = grouped_pd.pivot(
            index=index_cols, columns=columns, values=value_col,
        ).reset_index()
        # Flatten column names: if non-string column values (int etc.),
        # pandas sets them as-is; Arrow field name requires str.
        pivoted_pd.columns = [str(c) for c in pivoted_pd.columns]

        # Convert back to Arrow
        out_table = pa.Table.from_pandas(pivoted_pd, preserve_index=False)
        new_ds = Dataset.from_batches(
            out_table.schema, iter(out_table.to_batches()),
            self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def crosstab(
        self,
        index: "str | Sequence[str]",
        columns: str,
    ) -> "CadwDataFrame":
        """Frequency table (count per (index, columns) combo).

        Equivalent to:
            df.dropna(subset=[*index, columns])
              .pivot_table(index, columns, values=index[0], aggfunc='count')

        Drops rows with null in any index or columns value (standard
        pandas crosstab behaviour).

        Returns:
            CadwDataFrame with [*index, <columns-values as columns>].
        """
        index_cols = [index] if isinstance(index, str) else list(index)
        if not index_cols:
            raise ValueError("crosstab: index must be non-empty")
        for c in index_cols:
            if c not in self._columns:
                raise KeyError(
                    f"crosstab: index col {c!r} not in {self._columns}"
                )
        if columns not in self._columns:
            raise KeyError(
                f"crosstab: columns {columns!r} not in {self._columns}"
            )
        # Drop nulls in any relevant column (standard crosstab)
        filtered = self.dropna(subset=index_cols + [columns])
        return filtered.pivot_table(
            index=index,
            columns=columns,
            values=index_cols[0],  # any non-null col; = row count
            aggfunc="count",
        )

    # ---------------- reading ----------------'''

def add_methods():
    print("\n=== cadw/dataframe.py: melt / pivot_table / crosstab ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def pivot_table(" in s:
        print("  already present — skip")
        return

    anchor = "    # ---------------- reading ----------------"
    if anchor not in s:
        raise SystemExit("reading anchor not found")

    s = replace_line_block(s, anchor, METHODS, tag="pivot-melt-crosstab")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 2. AST whitelist: pivot_table == crosstab ============

def update_ast_whitelist():
    print("\n=== tests/test_arrow_call_memory_guard.py: whitelist pivot ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if '"_per_group_apply"' not in s:
        raise SystemExit("expected _per_group_apply in whitelist")
    if '"pivot_table"' in s:
        print("  already present — skip")
        return

    old = '''    # v0.9.1e per-group apply: to_pandas() on ONE group, bounded by
    # 2 × group_nbytes ≤ 0.3 × avail check BEFORE conversion.
    "_per_group_apply": "one group in RAM, bounded by check_state",
    "apply": "delegates to _per_group_apply (bounded per group)",
}'''

    new = '''    # v0.9.1e per-group apply: to_pandas() on ONE group, bounded by
    # 2 × group_nbytes ≤ 0.3 × avail check BEFORE conversion.
    "_per_group_apply": "one group in RAM, bounded by check_state",
    "apply": "delegates to _per_group_apply (bounded per group)",
    # v0.9.1f pivot_table: to_pandas() on grouped (K_pairs rows) —
    # bounded by explicit `pivot_table.grouped` check_state BEFORE.
    # AST guard verifies check_state precedes to_pandas.
    "pivot_table": "grouped (K_pairs rows) checked before to_pandas",
    "crosstab": "delegates to pivot_table (same bounded groupby)",
    "melt": "streaming per-batch, no high-risk calls",
}'''

    s = replace_line_block(s, old, new, tag="whitelist-pivot")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 3. Tests ============

TESTS = '''"""v0.9.1f: pivot_table / melt / crosstab tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=200):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_table(df, tmp_path, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)


# ============ melt ============

def test_melt_basic(tmp_path):
    df = make_df(tmp_path, {
        "id": [1, 2],
        "a": [10, 20],
        "b": [100, 200],
    })
    out = df.melt(id_vars=["id"], value_vars=["a", "b"])
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"id", "variable", "value"}
    assert t.num_rows == 4  # 2 ids × 2 value_vars
    rows = t.to_pylist()
    # First two from a (id=1,2), next two from b
    assert [r["variable"] for r in rows] == ["a", "a", "b", "b"]
    assert [r["value"] for r in rows] == [10, 20, 100, 200]


def test_melt_custom_names(tmp_path):
    df = make_df(tmp_path, {"id": [1], "a": [10], "b": [20]})
    out = df.melt(
        id_vars=["id"], value_vars=["a", "b"],
        var_name="src", value_name="val",
    )
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"id", "src", "val"}


def test_melt_default_value_vars(tmp_path):
    """value_vars=None → all except id_vars."""
    df = make_df(tmp_path, {"id": [1], "x": [10], "y": [20]})
    out = df.melt(id_vars=["id"])
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"id", "variable", "value"}
    assert t.num_rows == 2


def test_melt_mixed_dtype_raises(tmp_path):
    df = make_df(tmp_path, {
        "id": [1], "i": [10], "f": [1.5],
    })
    with pytest.raises(TypeError, match="share one dtype"):
        df.melt(id_vars=["id"], value_vars=["i", "f"])


def test_melt_overlap_raises(tmp_path):
    df = make_df(tmp_path, {"id": [1], "a": [10]})
    with pytest.raises(ValueError, match="both id_vars and value_vars"):
        df.melt(id_vars=["id", "a"], value_vars=["a"])


def test_melt_empty_value_vars_raises(tmp_path):
    df = make_df(tmp_path, {"id": [1], "a": [10]})
    with pytest.raises(ValueError, match="value_vars must be non-empty"):
        df.melt(id_vars=["id", "a"])


# ============ pivot_table ============

def test_pivot_basic_mean(tmp_path):
    df = make_df(tmp_path, {
        "row": ["r1", "r1", "r2", "r2"],
        "col": ["c1", "c2", "c1", "c2"],
        "v": [10.0, 20.0, 30.0, 40.0],
    })
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="mean",
    )
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"row", "c1", "c2"}
    rows = {r["row"]: (r["c1"], r["c2"]) for r in t.to_pylist()}
    assert rows == {"r1": (10.0, 20.0), "r2": (30.0, 40.0)}


def test_pivot_sum(tmp_path):
    df = make_df(tmp_path, {
        "row": ["r1", "r1", "r1"],
        "col": ["c1", "c1", "c2"],
        "v": [1.0, 2.0, 3.0],
    })
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="sum",
    )
    t = read_table(out, tmp_path)
    rows = t.to_pylist()
    assert rows[0]["c1"] == 3.0
    assert rows[0]["c2"] == 3.0


def test_pivot_count(tmp_path):
    df = make_df(tmp_path, {
        "row": ["r1", "r1", "r2"],
        "col": ["c1", "c1", "c2"],
        "v": [10, 20, 30],
    })
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="count",
    )
    t = read_table(out, tmp_path)
    rows = {r["row"]: r for r in t.to_pylist()}
    assert rows["r1"]["c1"] == 2
    # r1 c2 missing → null
    assert rows["r1"]["c2"] is None
    assert rows["r2"]["c2"] == 1


def test_pivot_missing_combo_null(tmp_path):
    """Missing (index, columns) combos → null cells."""
    df = make_df(tmp_path, {
        "row": ["r1", "r2"],
        "col": ["c1", "c2"],
        "v": [10.0, 20.0],
    })
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="sum",
    )
    t = read_table(out, tmp_path)
    rows = {r["row"]: r for r in t.to_pylist()}
    # r1 c1 present, c2 missing
    assert rows["r1"]["c1"] == 10.0
    assert rows["r1"]["c2"] is None
    # r2 c1 missing, c2 present
    assert rows["r2"]["c1"] is None
    assert rows["r2"]["c2"] == 20.0


def test_pivot_multi_index(tmp_path):
    df = make_df(tmp_path, {
        "a": ["x", "x", "y", "y"],
        "b": ["p", "p", "q", "q"],
        "c": ["c1", "c2", "c1", "c2"],
        "v": [1.0, 2.0, 3.0, 4.0],
    })
    out = df.pivot_table(
        index=["a", "b"], columns="c", values="v", aggfunc="mean",
    )
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"a", "b", "c1", "c2"}
    assert t.num_rows == 2


def test_pivot_invalid_aggfunc(tmp_path):
    df = make_df(tmp_path, {"a": [1], "b": [1], "v": [1.0]})
    with pytest.raises(ValueError, match="aggfunc must be"):
        df.pivot_table(
            index=["a"], columns="b", values="v", aggfunc="median",
        )


def test_pivot_too_large_raises(tmp_path):
    """Tight budget → fail-fast BEFORE groupby."""
    n = 1000
    df = make_df(tmp_path, {
        "row": [f"r{i}" for i in range(n)],
        "col": [f"c{i % 100}" for i in range(n)],
        "v": [float(i) for i in range(n)],
    }, chunk=100, mem_mb=1)
    df._memory_guard = MemoryGuard(max_memory_bytes=2000)
    with pytest.raises(MemoryLimitExceeded, match="K_pairs"):
        df.pivot_table(
            index=["row"], columns="col", values="v", aggfunc="sum",
        )


# ============ crosstab ============

def test_crosstab_basic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b"],
        "c": ["x", "y", "x"],
    })
    out = df.crosstab(index="g", columns="c")
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"g", "x", "y"}
    rows = {r["g"]: r for r in t.to_pylist()}
    assert rows["a"]["x"] == 1
    assert rows["a"]["y"] == 1
    assert rows["b"]["x"] == 1
    assert rows["b"]["y"] is None


def test_crosstab_matches_pivot_count(tmp_path):
    data = {
        "g": ["a", "a", "b", "b", "b"],
        "c": ["x", "y", "x", "y", "y"],
    }
    df = make_df(tmp_path, data)
    ct = read_table(
        df.crosstab(index="g", columns="c"),
        tmp_path, "ct.parquet",
    )
    pt = read_table(
        df.pivot_table(
            index=["g"], columns="c", values="g", aggfunc="count",
        ),
        tmp_path, "pt.parquet",
    )
    # Both should have same shape and values
    ct_rows = sorted(ct.to_pylist(), key=lambda r: r["g"])
    pt_rows = sorted(pt.to_pylist(), key=lambda r: r["g"])
    assert ct_rows == pt_rows


def test_crosstab_drops_nulls(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", None],
        "c": ["x", "y", "x"],
    })
    out = df.crosstab(index="g", columns="c")
    t = read_table(out, tmp_path)
    # Row with null g is dropped → only "a" row
    assert t.num_rows == 1
    assert t.to_pylist()[0]["g"] == "a"


# ============ AST guards ============

def test_pivot_table_has_check_state_before_to_pandas():
    """AST: pivot_table has check_state BEFORE to_pandas."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.pivot_table)
    assert "to_pandas" in src
    assert "check_state" in src
    assert src.index("check_state") < src.index("to_pandas"), (
        "pivot_table: check_state must precede to_pandas (§3.11)"
    )


def test_melt_is_streaming():
    """AST: melt yields per batch (no materialize)."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.melt)
    assert "yield" in src
    # No to_pandas
    assert "to_pandas" not in src
'''


E2E_TESTS = '''"""v0.9.1f: pivot_table E2E (slow)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


@pytest.mark.slow
def test_pivot_100k_rows(tmp_path):
    """100K rows, 1000 index × 20 columns, pivot within 50 MB budget."""
    n = 100_000
    rng = np.random.default_rng(42)
    data = {
        "row": [f"r{i}" for i in range(n)],  # all unique
        "col": [f"c{i % 20}" for i in range(n)],  # 20 columns
        "v": rng.normal(size=n).tolist(),
    }
    pdf = pd.DataFrame(data)
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

    # Output: 100k rows × (1 + 20 columns) — K_pairs = 100k × 20 = 2M cells
    # (mean per cell; but each row is unique → each cell has 1 or 0 values)
    assert t.num_rows == n  # every row has at least one col value
    assert len(t.schema.names) == 1 + 20  # row + 20 col values
'''


def write_tests():
    for name, content in [
        ("tests/test_pivot_table.py", TESTS),
        ("tests/test_pivot_table_e2e.py", E2E_TESTS),
    ]:
        print(f"\n=== {name} ===")
        p = Path(name)
        p.write_text(content)
        compile_check(p)
        print("  OK")


# ============ 4. CHANGELOG + DECISIONS ============

def append_changelog():
    print("\n=== CHANGELOG.md: append v0.9.1f ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.1f — pivot_table" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Добавлено (v0.9.1f — pivot_table / melt / crosstab)

Финальный патч v0.9.1: табличные преобразования (long ↔ wide).

- **`df.melt(id_vars, value_vars, var_name='variable',
  value_name='value')`** — wide → long. **Streaming per-batch**, peak
  O(batch × #value_vars). Requirements:
  - `value_vars` share ONE dtype (Arrow column uniform); mixed →
    TypeError с рекомендацией `astype` для конвертации.
  - `id_vars` ∩ `value_vars = ∅`; `var_name`/`value_name` не
    коллидируют с `id_vars`.
  - Output: `[*id_vars, var_name, value_name]`, rows = `N ×
    len(value_vars)`.

- **`df.pivot_table(index, columns, values, aggfunc='mean')`** — long →
  wide. `groupby([*index, columns]).agg(values, aggfunc)` + pandas
  reshape (small table). **Fail-fast** если `k_pairs = K_index ×
  K_columns` слишком велико: `k_pairs × row_size × 2 > 0.3 × avail`
  → `MemoryLimitExceeded` **до** groupby. Missing combos → null.
  - `aggfunc ∈ {mean, sum, count, min, max}`, default `mean`.
  - `dropna=True` для null в `columns` (standard pandas).
  - Null в `index` → own row (kept).
  - Multi-index (`index=['a','b']`) поддерживается.

- **`df.crosstab(index, columns)`** — frequency table. `dropna=True`
  на nulls в index/columns (standard crosstab), delegates to
  `pivot_table(values=index[0], aggfunc='count')`.

### Memory bound (§11.5)

- **melt**: O(batch × #value_vars). Streaming.
- **pivot_table**: K_pairs estimate через sample (`MemoryGuard.
  estimate_unique_keys`) **до** groupby. `check_state` на grouped
  (K_pairs rows) **до** pandas reshape (AST-verified). Peak = batch +
  K_pairs × row_size (grouped) + K_index × K_cols × value_size
  (pivot output) — bounded by fail-fast check.
- **crosstab**: same as pivot_table.

### Ограничения (v0.9.1f, §11.5)

- **`pivot_table` sparse → fail-fast**. K_pairs > 0.3×avail/row_size →
  `MemoryLimitExceeded` с рекомендацией `groupby()` (long format).
  Sparse pandas pivot (fill N/A with 0 in wide format) — не
  поддерживается.
- **`melt` heterogeneous dtypes** (`['int_col', 'float_col']`) →
  TypeError. Пользователь: `df.astype({'int_col': 'float64'})` first.
- **`pivot_table` dropna=True** в `columns` (стандарт pandas).
  `dropna=False` (null column group) — v0.9.2 если запросят.
- **`pivot_table` output column names = `str(value)`**. Int-значения
  `columns` → строковые имена ("0", "1", ...).
- **`melt` non-streaming types**: list/struct columns in value_vars
  работают (Arrow-native), но nested dtypes не проверяются на
  uniformity (assumes Arrow type equality).
- **`crosstab` count via non-null values** в `index[0]`. Требует
  non-null `index[0]` — фильтруется через `dropna=True` до.

### Тесты (v0.9.1f)

- +15 fast (`test_pivot_table.py`): melt (6), pivot (7), crosstab (3),
  AST guards (2).
- +1 slow (`test_pivot_table_e2e.py`): 100K rows pivot.

- fast: **~1395** (+15). slow: **13** (+1).
'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


def update_decisions():
    print("\n=== DECISIONS.md §11.5: pivot/melt note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.1f (pivot_table / melt" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        print("  WARNING: §11.5 not found — append")
        s = s.rstrip() + "\n\n**v0.9.1f (pivot/melt):** melt streaming "
        s += "uniform-dtype; pivot_table fail-fast on K_pairs.\n"
        p.write_text(s)
        return

    note = '''**v0.9.1f (pivot_table / melt / crosstab):**
- `melt`: streaming, `value_vars` must share one dtype (mixed →
  TypeError, cast first). Output rows = N × len(value_vars).
- `pivot_table`: `k_pairs × row_size × 2 > 0.3 × avail` →
  `MemoryLimitExceeded` **before** groupby (sample-based estimate).
  Missing combos → null. Multi-index supported. Null in `columns`
  dropped (dropna=True).
- `crosstab`: filters nulls in index/columns, delegates to
  `pivot_table(aggfunc='count')`.
- **Sparse pivot**: fail-fast. Wide with N/A filled → user manual
  `fillna(0)` after.
- **melt heterogeneous** → TypeError; use `.astype(...)` first.
- **pivot column names** = `str(value)`. Int → "0", "1", ....

'''
    s = s[:pos] + note + s[pos:]
    p.write_text(s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.1f: pivot_table / melt / crosstab ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    r = subprocess.run(["git", "status", "--porcelain",
                        "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    add_methods()
    update_ast_whitelist()
    write_tests()
    append_changelog()
    update_decisions()

    print("\n=== pytest: new pivot tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pivot_table.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIVOT TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pivot E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pivot_table_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards via patch_utils ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1200)
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
        "v0.9.1f: pivot_table / melt / crosstab — long↔wide transforms; "
        "melt streaming (uniform-dtype), pivot_table fail-fast (K_pairs "
        "estimate + check_state before groupby+to_pandas), crosstab alias; "
        "§11.5 sparse limits"])

    print("\n==> v0.9.1f applied. v0.9.1 complete.")
    print()
    print("SQUASH + TAG v0.9.1:")
    print("  git reset --soft v0.9.0")
    print("  git commit -m 'v0.9.1: DAG pipeline, save/load, "
          "ColumnTransformer, apply(fn), pivot/melt/crosstab'")
    print("  sed -i 's/^## \\[Unreleased\\]$/## [0.9.1] — 2026-10-20/' CHANGELOG.md")
    print("  git add CHANGELOG.md && git commit --amend --no-edit")
    print("  git tag -a v0.9.1 -m 'v0.9.1: pipeline advanced'")
    print("  uv run pytest tests/test_changelog.py -v")


if __name__ == "__main__":
    main()