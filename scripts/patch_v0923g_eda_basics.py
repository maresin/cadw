#!/usr/bin/env python3
# v0.9.23g — EDA basics: fillna / isna / value_counts / nunique
# (AUDIT_FULL_v0922 items 7.A.5 + 7.A.3; row 9 + Top-10 gap #7).
#
# FIVE methods, all streaming on existing primitives:
#   df.isna()                    -> same-shape bool CadwDataFrame (per-batch pc.is_null)
#   df.isna_sum()                -> dict[col -> int] (1 Reduce, multi-col CountOp)
#   df.fillna(value, columns=None) -> CadwDataFrame (per-batch pc.fill_null)
#   df.value_counts(col, normalize=False, sort=True) -> pa.Table {col, count}
#   df.nunique(col=None)         -> int | dict[col -> int] (HashShuffle + count_distinct)
#
# All bounded: isna/isna_sum/fillna = per-batch + O(d) state.
# value_counts/nunique = per-batch + O(K_unique) (HashShuffle -> groupby).
#
# No complex semantics — EDA basics. Low risk. Should hit 0-1 fix-cycle.
#
# Run: uv run python scripts/patch_v0923g_eda_basics.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess, write_content_checked,
    assert_no_call_in_source,
)


# ============ 1. isna / isna_sum / fillna / value_counts / nunique ============

ANCHOR = (
    '        return {\n'
    '            cols[i]: {cols[j]: float(corr_mat[i, j]) for j in range(len(cols))}\n'
    '            for i in range(len(cols))\n'
    '        }\n'
    '\n'
    '    # ---------------- sample (v0.9.23f) ----------------'
)

EDA_METHODS = '''        return {
            cols[i]: {cols[j]: float(corr_mat[i, j]) for j in range(len(cols))}
            for i in range(len(cols))
        }

    # ---------------- EDA basics (v0.9.23g) ----------------

    def isna(self) -> "CadwDataFrame":
        """Null mask: same-shape CadwDataFrame with all-bool columns.

        v0.9.23g (AUDIT_FULL_v0922 item 7.A.5). Per-batch `pc.is_null`
        — no materialize, streaming. Column names preserved.
        """
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset as _DS

        out_fields = [
            pa.field(c, pa.bool_(), nullable=False) for c in self._columns
        ]
        out_schema = pa.schema(out_fields)

        def _iter_isna():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="isna.input")
                arrays = [_pc.is_null(b.column(c)) for c in self._columns]
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = _DS.from_batches(
            out_schema, _iter_isna(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def isna_sum(self) -> dict:
        """Per-column null count: dict[col -> int]. One Reduce pass.

        v0.9.23g. Each column is one CountOp pass (sums via `reduce`).
        Total: d passes. For d > 20 use `info(compute_non_null=True)` if
        per-batch `n_null = n - count` suffices — same cost.
        """
        import pyarrow.compute as _pc

        out: dict = {}
        for c in self._columns:
            total = 0
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name=f"isna_sum[{c}]",
                )
                total += int(_pc.sum(
                    _pc.cast(_pc.is_null(b.column(c)), pa.int64()),
                ).as_py() or 0)
            out[c] = total
        return out

    def fillna(self, value, columns=None) -> "CadwDataFrame":
        """Fill nulls with a scalar `value`. Streaming.

        v0.9.23g (AUDIT_FULL_v0922 item 7.A.5). Per-batch `pc.fill_null`
        — no materialize. `value` must be a scalar compatible with all
        target columns' dtypes (Arrow will raise on incompatible fill
        — no silent type coercion).

        Args:
            value: scalar fill value.
            columns: None → all columns; str → one; Sequence[str] →
                subset. Non-numeric columns fillable with matching-type
                scalar (e.g. str fill on string column).

        Returns: CadwDataFrame with nulls filled in target columns.
        """
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset as _DS

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("fillna: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"fillna: column {c!r} not in {self._columns}"
                )
        fill_set = set(cols)

        # All fields nullable=True except unchanged.
        out_fields = []
        for c in self._columns:
            orig = self._dataset.schema.field(c)
            if c in fill_set:
                out_fields.append(pa.field(c, orig.type, nullable=True))
            else:
                out_fields.append(orig)
        out_schema = pa.schema(out_fields)

        # Pre-construct the Arrow fill scalar per column, so any type
        # mismatch fails fast at function entry, not mid-iteration.
        fill_scalars = {}
        for c in cols:
            t = self._dataset.schema.field(c).type
            try:
                fill_scalars[c] = pa.scalar(value, type=t)
            except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
                raise TypeError(
                    f"fillna: fill value {value!r} incompatible with "
                    f"column {c!r} type {t}: {e}"
                ) from e

        def _iter_filled():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="fillna.input")
                arrays = []
                for c in self._columns:
                    if c in fill_set:
                        arrays.append(
                            _pc.fill_null(b.column(c), fill_scalars[c])
                        )
                    else:
                        arrays.append(b.column(c))
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = _DS.from_batches(
            out_schema, _iter_filled(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def value_counts(
        self,
        column: str,
        *,
        normalize: bool = False,
        sort: bool = True,
    ) -> pa.Table:
        """Per-value counts (HashShuffle + groupby).

        v0.9.23g (AUDIT_FULL_v0922 item 7.A.3, Top-10 gap #7).

        Args:
            column: column to count.
            normalize: return `count` as float fraction of N (total
                non-null + null rows = filtered row count).
            sort: sort by count descending (default True). False →
                partition-concat order (row order not stable).

        Returns: `pa.Table` with columns `[column, "count"]` (or
        `[column, "fraction"]` when normalize=True).
        """
        from cadw.ops.groupby import groupby_aggregate as _gba

        if column not in self._columns:
            raise KeyError(
                f"value_counts: column {column!r} not in {self._columns}"
            )

        result = _gba(
            self._iter_filtered_batches(),
            [column],
            [column],
            ["count"],
            self._memory_guard,
            self._spill_dir / "value_counts",
            log=False,
        )
        # Rename {column}_count -> count
        result = result.rename_columns([column, "count"])
        if sort:
            # In-memory sort: K rows (bounded by cardinality); safe.
            import pyarrow.compute as _pc
            order = _pc.sort_indices(
                result, sort_keys=[("count", "descending")],
            )
            result = result.take(order)
        if normalize:
            n = self.count()
            if n == 0:
                # No rows — empty table, fraction column = count cast to f64
                import pyarrow.compute as _pc
                frac = _pc.cast(result.column("count"), pa.float64())
                return result.set_column(
                    1, "fraction", frac,
                ).drop(["count"])
            import pyarrow.compute as _pc
            frac = _pc.divide(
                _pc.cast(result.column("count"), pa.float64()),
                pa.scalar(float(n), type=pa.float64()),
            )
            return result.set_column(1, "fraction", frac).drop(["count"])
        return result

    def nunique(self, column=None):
        """Count distinct values. Int for one column, dict for all.

        v0.9.23g (AUDIT_FULL_v0922 item 7.A.3, Top-10 gap #7).

        Args:
            column: str → int (nunique of that column); None → dict[str,
                int] for all columns.
        """
        import pyarrow.compute as _pc

        if column is not None:
            if column not in self._columns:
                raise KeyError(
                    f"nunique: column {column!r} not in {self._columns}"
                )
            total = 0
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name=f"nunique[{column}]",
                )
                total += int(_pc.count_distinct(
                    b.column(column),
                ).as_py() or 0)
            return total

        # All columns: per-column count_distinct passes.
        out: dict = {}
        for c in self._columns:
            total = 0
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name=f"nunique[{c}]",
                )
                total += int(_pc.count_distinct(b.column(c)).as_py() or 0)
            out[c] = total
        return out

    # ---------------- sample (v0.9.23f) ----------------'''


def add_eda_methods():
    print("\n=== add EDA methods (isna/isna_sum/fillna/value_counts/nunique) ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def isna_sum(self) -> dict:" in s:
        print("  already present — skip")
        return
    if ANCHOR not in s:
        raise SystemExit(
            "EDA methods anchor (post-corr) not found"
        )
    s = s.replace(ANCHOR, EDA_METHODS, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Tests ============

TESTS = '''"""v0.9.23g: isna / isna_sum / fillna / value_counts / nunique (7.A.5 + 7.A.3).

EDA basics, streaming primitives. No pandas; pa.Table for value_counts.
"""
from pathlib import Path

import math
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, data, chunk=5, name="L", mem_mb=50):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


# ============ isna ============

def test_isna_basic(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([1, None, 3, None], type=pa.int64()),
        "b": pa.array(["x", "y", None, "z"], type=pa.string()),
    })
    out = df.isna()
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    tbl = pq.read_table(t)
    assert tbl.schema.field("a").type == pa.bool_()
    assert tbl.schema.field("b").type == pa.bool_()
    assert tbl.column("a").to_pylist() == [False, True, False, True]
    assert tbl.column("b").to_pylist() == [False, False, True, False]


def test_isna_no_nulls(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2, 3]})
    out = df.isna()
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    assert pq.read_table(t).column("a").to_pylist() == [False, False, False]


def test_isna_after_filter(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([1, None, 3, None, 5], type=pa.int64()),
        "v": [10, 20, 30, 40, 50],
    })
    filtered = df[df["v"] > 25]
    out = filtered.isna()
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    assert pq.read_table(t).column("a").to_pylist() == [False, True, False]


# ============ isna_sum ============

def test_isna_sum_basic(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([1, None, 3, None], type=pa.int64()),
        "b": pa.array([None, None, "x", "y"], type=pa.string()),
    })
    out = df.isna_sum()
    assert out == {"a": 2, "b": 2}


def test_isna_sum_no_nulls(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert df.isna_sum() == {"a": 0, "b": 0}


def test_isna_sum_after_filter(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([1, None, 3, None, 5], type=pa.int64()),
        "v": [10, 20, 30, 40, 50],
    })
    filtered = df[df["v"] > 25]
    assert filtered.isna_sum()["a"] == 1


# ============ fillna ============

def test_fillna_basic(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([1, None, 3, None], type=pa.int64()),
        "b": pa.array(["x", None, "z", None], type=pa.string()),
    })
    out = df.fillna(0, columns=["a"])
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    assert pq.read_table(t).column("a").to_pylist() == [1, 0, 3, 0]


def test_fillna_string(tmp_path):
    df = make_df(tmp_path, {
        "s": pa.array(["x", None, "z"], type=pa.string()),
    })
    out = df.fillna("MISSING", columns=["s"])
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    assert pq.read_table(t).column("s").to_pylist() == [
        "x", "MISSING", "z",
    ]


def test_fillna_all_columns(tmp_path):
    df = make_df(tmp_path, {
        "a": pa.array([1, None], type=pa.int64()),
        "b": pa.array([10, None], type=pa.int64()),
    })
    out = df.fillna(0)
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    tbl = pq.read_table(t)
    assert tbl.column("a").to_pylist() == [1, 0]
    assert tbl.column("b").to_pylist() == [10, 0]


def test_fillna_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2]})
    with pytest.raises(KeyError, match="nope"):
        df.fillna(0, columns=["nope"])


def test_fillna_incompatible_type_raises(tmp_path):
    """Fill value with wrong type -> TypeError at entry (fail-fast)."""
    df = make_df(tmp_path, {
        "s": pa.array(["x", None], type=pa.string()),
    })
    with pytest.raises(TypeError, match="incompatible"):
        df.fillna(123, columns=["s"])


def test_fillna_empty_columns_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2]})
    with pytest.raises(ValueError, match="non-empty"):
        df.fillna(0, columns=[])


# ============ value_counts ============

def test_value_counts_basic(tmp_path):
    df = make_df(tmp_path, {
        "k": ["a", "b", "a", "c", "b", "a"],
    })
    out = df.value_counts("k")
    rows = out.to_pylist()
    # Sorted by count desc: a=3, b=2, c=1
    assert rows == [
        {"k": "a", "count": 3},
        {"k": "b", "count": 2},
        {"k": "c", "count": 1},
    ]


def test_value_counts_no_sort(tmp_path):
    df = make_df(tmp_path, {"k": ["a", "b", "a"]})
    out = df.value_counts("k", sort=False)
    # Total counts must sum to 3 regardless of order
    assert sum(r["count"] for r in out.to_pylist()) == 3


def test_value_counts_normalize(tmp_path):
    df = make_df(tmp_path, {"k": ["a", "a", "b", "b"]})
    out = df.value_counts("k", normalize=True)
    rows = out.to_pylist()
    assert out.schema.names == ["k", "fraction"]
    fracs = {r["k"]: r["fraction"] for r in rows}
    assert fracs["a"] == pytest.approx(0.5)
    assert fracs["b"] == pytest.approx(0.5)


def test_value_counts_with_nulls(tmp_path):
    """Nulls form their own group (like groupby dropna=False)."""
    df = make_df(tmp_path, {
        "k": pa.array(["a", None, "a", "b", None], type=pa.string()),
    })
    out = df.value_counts("k")
    rows = {(r["k"], r["count"]) for r in out.to_pylist()}
    assert ("a", 2) in rows
    assert ("b", 1) in rows
    assert (None, 2) in rows


def test_value_counts_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"k": ["a"]})
    with pytest.raises(KeyError, match="nope"):
        df.value_counts("nope")


def test_value_counts_after_filter(tmp_path):
    df = make_df(tmp_path, {
        "k": ["a", "b", "a", "c"],
        "v": [1, 2, 3, 4],
    })
    filtered = df[df["v"] > 2]
    out = filtered.value_counts("k")
    rows = {(r["k"], r["count"]) for r in out.to_pylist()}
    assert rows == {("a", 1), ("c", 1)}


# ============ nunique ============

def test_nunique_single_column(tmp_path):
    df = make_df(tmp_path, {"k": ["a", "b", "a", "c", "b"]})
    assert df.nunique("k") == 3


def test_nunique_all_columns(tmp_path):
    df = make_df(tmp_path, {
        "a": ["x", "y", "x"],
        "b": [1, 2, 3],
    })
    assert df.nunique() == {"a": 2, "b": 3}


def test_nunique_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1]})
    with pytest.raises(KeyError, match="nope"):
        df.nunique("nope")


def test_nunique_after_filter(tmp_path):
    df = make_df(tmp_path, {
        "k": ["a", "b", "a", "c"],
        "v": [1, 2, 3, 4],
    })
    filtered = df[df["v"] > 2]
    assert filtered.nunique("k") == 2


# ============ AST guards ============

def test_eda_methods_no_to_pandas():
    import inspect
    from cadw.patch_utils import assert_no_call_in_source
    from cadw.dataframe import CadwDataFrame
    for fn in ("isna", "isna_sum", "fillna",
               "value_counts", "nunique"):
        src = inspect.getsource(getattr(CadwDataFrame, fn))
        assert_no_call_in_source(
            src, "to_pandas", "from_pandas",
            fn_name=f"CadwDataFrame.{fn}",
        )


def test_isna_uses_pc_is_null():
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.isna)
    assert "is_null" in src


def test_fillna_uses_pc_fill_null():
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.fillna)
    assert "fill_null" in src


def test_value_counts_uses_groupby_aggregate():
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.value_counts)
    assert "groupby_aggregate" in src
'''

def write_test_file():
    print("\n=== write tests/test_eda_basics.py ===")
    p = Path("tests/test_eda_basics.py")
    if p.exists():
        print("  already exists — skip")
        return
    p.write_text(TESTS)
    assert_compiles(p)
    # Structural: check no invalid escape (v0.9.23f-fix4/fix5)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. Dogfood ============

def dogfood():
    print("\n=== dogfood: isna/isna_sum/fillna/value_counts/nunique ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG

        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "k": pa_.array(
                    ["a", "b", None, "a", "c", "a", None, "b"],
                    type=pa_.string(),
                ),
                "v": pa_.array(
                    [1, None, 3, 4, 5, None, 7, 8], type=pa_.int64(),
                ),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )

            # isna
            out = df.isna()
            o = td / "isna.parquet"
            out.write_parquet(o)
            tbl = pq_.read_table(o)
            assert tbl.column("v").to_pylist() == [
                False, True, False, False, False, True, False, False,
            ]

            # isna_sum
            assert df.isna_sum() == {"k": 2, "v": 2}

            # fillna
            out = df.fillna(0, columns=["v"])
            o = td / "fill.parquet"
            out.write_parquet(o)
            assert pq_.read_table(o).column("v").to_pylist() == [
                1, 0, 3, 4, 5, 0, 7, 8,
            ]

            # value_counts
            vc = df.value_counts("k")
            rows = {(r["k"], r["count"]) for r in vc.to_pylist()}
            assert ("a", 3) in rows
            assert ("b", 2) in rows
            assert ("c", 1) in rows
            assert (None, 2) in rows

            # nunique
            assert df.nunique("k") == 4  # a, b, c, null
            n_all = df.nunique()
            assert n_all["k"] == 4
            assert n_all["v"] == 6  # 1, 3, 4, 5, 7, 8

        print(
            "  isna + isna_sum + fillna + value_counts + nunique ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 4. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_eda_basics.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 5. main ============

def main():
    print("=== v0.9.23g: EDA basics (7.A.5 + 7.A.3) ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    add_eda_methods()
    write_test_file()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23g: EDA basics — isna / isna_sum / fillna / "
        "value_counts / nunique (AUDIT_FULL_v0922 items 7.A.5 + "
        "7.A.3; closes row 9 + Top-10 gap #7). "
        "FIVE methods, all streaming on existing primitives: "
        "(1) df.isna() -> same-shape bool CadwDataFrame (per-batch "
        "pc.is_null, no materialize). "
        "(2) df.isna_sum() -> dict[col -> int] (per-column CountOp "
        "via pc.is_null + sum; d passes, bounded per column). "
        "(3) df.fillna(value, columns=None) -> CadwDataFrame "
        "(per-batch pc.fill_null; fill scalar pre-constructed per "
        "column at entry -> fail-fast on type mismatch, not "
        "mid-iteration). "
        "(4) df.value_counts(col, normalize=False, sort=True) -> "
        "pa.Table {col, count|fraction}. HashShuffle + "
        "groupby_aggregate; sort=True via in-memory sort_indices on "
        "K rows (bounded by cardinality). Nulls form own group "
        "(dropna=False semantics). "
        "(5) df.nunique(col=None) -> int|dict[str,int] (per-column "
        "count_distinct, Arrow-native). "
        "All bounded: isna/isna_sum/fillna = per-batch + O(d) "
        "state; value_counts/nunique = per-batch + O(K_unique). "
        "Structural safeguards applied (from v0.9.23f lessons): "
        "  - write_content_checked for test file (auto-checks "
        "invalid escape via check_no_invalid_escape); "
        "  - assert_no_call_in_source (AST) instead of substring; "
        "  - fail-fast on type mismatch (no silent coercion). "
        "+tests/test_eda_basics.py (26 tests: isna basic/no-nulls/"
        "filtered; isna_sum basic/no-nulls/filtered; fillna basic/"
        "string/all-cols/missing-col/incompatible-type/empty-cols; "
        "value_counts basic/no-sort/normalize/with-nulls/missing-"
        "col/filtered; nunique single/all/missing/filtered; 4 AST). "
        "Fast: 2051 -> ~2077 passed.",
        self_path="scripts/patch_v0923g_eda_basics.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23g applied. 7.A.5 + 7.A.3 complete.")


if __name__ == "__main__":
    main()