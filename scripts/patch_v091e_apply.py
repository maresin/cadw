#!/usr/bin/env python3
"""v0.9.1e: per-group apply(fn) — prototype-validated (11 cases).

Prototype: scripts/proto_apply.py PASS 11/11. Composite-sort groups ≡
pandas groupby(g, dropna=False).

SCOPE:
  - CadwDataFrame._per_group_apply(group_cols, on_cols, fn, result_col,
    memory_guard) — composite-sort, one group in RAM at a time.
  - _GroupBy.apply(fn, result_col='apply_result', on=None) — wrapper.
  - Three uniform modes (scalar / Series / DataFrame).
  - Per-group memory bound: 2× group_nbytes ≤ 0.3 × avail BEFORE fn.

NOT included: transform(fn) — different semantics, v0.9.2.

GUARDS:
  - test_apply.py (~17 fast): 3 modes, nulls, multi-group, single-row,
    filter, determinism, on-col sort, errors (mixed modes, oversized,
    fn exception), AST guard (check_state before to_pandas).
  - test_apply_e2e.py (1 slow): 10K rows / 20 groups, pandas equiv.

Run: uv run python scripts/patch_v091e_apply.py
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


# ============ 1. _per_group_apply ============

def add_per_group_apply():
    print("\n=== cadw/dataframe.py: _per_group_apply ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def _per_group_apply(" in s:
        print("  already present — skip")
        return

    anchor = "    # ---------------- per-group shift/diff/rolling (v0.8.10) ----------------"

    new_method = '''    # ---------------- per-group apply (v0.9.1e) ----------------

    def _per_group_apply(
        self,
        group_cols: list[str],
        on_cols: list[str],
        fn,
        result_col: str,
        memory_guard: "MemoryGuard",
    ) -> "CadwDataFrame":
        """Composite-sort apply(fn) on each group (prototype-validated).

        Args:
            group_cols: group keys. Non-empty.
            on_cols: within-group sort keys (empty → row_id order).
            fn: callable(group_df: pd.DataFrame) → scalar | pd.Series |
                pd.DataFrame. Uniform mode across groups.
            result_col: column name for scalar mode.
            memory_guard: per-group memory check before fn().

        Peak RAM = batch + 2 × max(group_size × row_size) + fn's output.
        Bounded per group, independent of N.

        fn's own alloc is user responsibility (§3.9 extended).
        """
        import pyarrow as pa
        from cadw.io.batch_reader import ArrowIPCBatchReader
        from cadw.io.batch_writer import ArrowIPCBatchWriter
        from cadw.core.memory_guard import MemoryLimitExceeded

        if not group_cols:
            raise ValueError(
                "_per_group_apply: group_cols must be non-empty"
            )
        if not callable(fn):
            raise TypeError(
                f"_per_group_apply: fn must be callable; "
                f"got {type(fn).__name__}"
            )
        for gc in group_cols:
            if gc not in self._columns:
                raise KeyError(
                    f"_per_group_apply: group column {gc!r} not in "
                    f"{self._columns}"
                )
        for oc in on_cols:
            if oc not in self._columns:
                raise KeyError(
                    f"_per_group_apply: on column {oc!r} not in "
                    f"{self._columns}"
                )
        if not isinstance(result_col, str) or not result_col:
            raise ValueError(
                f"_per_group_apply: result_col must be non-empty str; "
                f"got {result_col!r}"
            )

        if self.count() == 0:
            out_fields = [
                self._dataset.schema.field(c) for c in group_cols
            ]
            out_fields.append(pa.field(result_col, pa.float64(), nullable=True))
            out_schema = pa.schema(out_fields)
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        sort_keys = list(group_cols) + list(on_cols)
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=sort_keys,
            value_columns=self._columns,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "per_group_apply" / "sort_by_keys",
            ascending=True,
            log=False,
        )

        out_schema_box: list = []
        out_mode_box: list = []
        buf_rows: list = []
        BATCH_ROWS = 1000
        applied_path = (
            self._spill_dir / "per_group_apply" / "applied.arrow"
        )
        applied_path.parent.mkdir(parents=True, exist_ok=True)
        writer_box: list = []

        def _flush_buf():
            nonlocal buf_rows
            if not buf_rows:
                return
            if not writer_box:
                writer_box.append(
                    ArrowIPCBatchWriter(str(applied_path),
                                        out_schema_box[0])
                )
            table = pa.Table.from_pylist(buf_rows,
                                         schema=out_schema_box[0])
            for b in table.to_batches():
                writer_box[0].write_batch(b)
            buf_rows = []

        def _emit_group(key: tuple, result):
            import pandas as pd
            nonlocal buf_rows

            mode = _detect_mode(result)
            if not out_mode_box:
                out_mode_box.append(mode)
                out_schema_box.append(
                    _build_out_schema(group_cols, self, mode,
                                      result, result_col)
                )
            elif mode != out_mode_box[0]:
                raise RuntimeError(
                    f"_per_group_apply: mixed return modes across "
                    f"groups: first group returned {out_mode_box[0]!r}, "
                    f"group {key!r} returned {mode!r}. fn() must "
                    f"return same type for all groups."
                )

            if mode == "scalar":
                row = {gc: v for gc, v in zip(group_cols, key)}
                row[result_col] = _normalize_scalar(result)
                buf_rows.append(row)
            elif mode == "series":
                row = {gc: v for gc, v in zip(group_cols, key)}
                for idx, val in result.items():
                    row[str(idx)] = (
                        None if pd.isna(val) else val
                    )
                buf_rows.append(row)
            else:  # dataframe
                for rec in result.to_dict(orient="records"):
                    row = {gc: v for gc, v in zip(group_cols, key)}
                    for c, v in rec.items():
                        row[str(c)] = (
                            None if pd.isna(v) else v
                        )
                    buf_rows.append(row)
            if len(buf_rows) >= BATCH_ROWS:
                _flush_buf()

        _UNSET = object()
        prev_key = _UNSET
        group_buf: list = []
        max_group_limit = int(
            memory_guard.available_bytes() * 0.3
        )

        def _flush_group():
            nonlocal group_buf
            if not group_buf:
                return
            gt = pa.Table.from_batches(group_buf)
            est = gt.nbytes * 2
            if est > max_group_limit:
                raise MemoryLimitExceeded(
                    f"_per_group_apply: group size {gt.nbytes} B "
                    f"× 2 (pandas) = {est} B > 0.3 × avail = "
                    f"{max_group_limit} B. Split by more group cols "
                    f"or use groupby().agg() if fn reduces. See "
                    f"§11.5."
                )
            memory_guard.check_state(
                est, name="per_group_apply.group",
            )
            gdf = gt.to_pandas()
            result = fn(gdf)
            _emit_group(tuple(prev_key), result)
            group_buf.clear()

        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                memory_guard.check_batch(
                    b, name="per_group_apply.sorted"
                )
                n = b.num_rows
                if n == 0:
                    continue
                group_vals = {
                    gc: b.column(gc).to_pylist() for gc in group_cols
                }
                start = 0
                for i in range(1, n):
                    gk = tuple(group_vals[gc][i] for gc in group_cols)
                    prev_gk = tuple(
                        group_vals[gc][i - 1] for gc in group_cols
                    )
                    if gk != prev_gk:
                        gk_start = tuple(
                            group_vals[gc][start] for gc in group_cols
                        )
                        piece = b.slice(start, i - start)
                        if gk_start != prev_key:
                            if prev_key is not _UNSET:
                                _flush_group()
                            prev_key = gk_start
                        group_buf.append(piece)
                        start = i
                gk_start = tuple(
                    group_vals[gc][start] for gc in group_cols
                )
                piece = b.slice(start, n - start)
                if gk_start != prev_key:
                    if prev_key is not _UNSET:
                        _flush_group()
                    prev_key = gk_start
                group_buf.append(piece)

            _flush_group()
            _flush_buf()
        finally:
            if writer_box:
                writer_box[0].close()

        if not out_schema_box:
            raise RuntimeError(
                "_per_group_apply: no groups produced output — "
                "unreachable after empty check"
            )

        def _iter_applied():
            for b in ArrowIPCBatchReader(str(applied_path), None):
                yield b

        new_ds = Dataset.from_batches(
            out_schema_box[0], _iter_applied(),
            self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    # ---------------- per-group shift/diff/rolling (v0.8.10) ----------------'''

    s = replace_line_block(s, anchor, new_method, tag="per-group-apply")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 2. Helpers at end of dataframe.py ============

HELPERS = '''

# ============ per-group apply helpers (v0.9.1e) ============

def _detect_mode(result) -> str:
    """Classify fn() output: 'scalar' / 'series' / 'dataframe'."""
    import pandas as pd
    import numpy as np

    if isinstance(result, pd.DataFrame):
        return "dataframe"
    if isinstance(result, pd.Series):
        return "series"
    if result is None:
        return "scalar"
    if isinstance(result, (bool, int, float, str)):
        return "scalar"
    if isinstance(result, (np.integer, np.floating, np.bool_)):
        return "scalar"
    raise TypeError(
        f"_per_group_apply: fn returned {type(result).__name__}; "
        f"expected scalar, pd.Series, or pd.DataFrame"
    )


def _normalize_scalar(v):
    """Convert numpy scalar / pandas NA to Python / None."""
    import numpy as np
    import pandas as pd

    if v is None:
        return None
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        v = v.item()
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def _infer_arrow_type_from_value(v):
    import pyarrow as pa
    import numpy as np

    if v is None:
        return pa.float64()
    if isinstance(v, (np.integer, int)):
        return pa.int64()
    if isinstance(v, (np.floating, float)):
        return pa.float64()
    if isinstance(v, (np.bool_, bool)):
        return pa.bool_()
    if isinstance(v, str):
        return pa.string()
    return pa.float64()


def _build_out_schema(group_cols, df, mode, first_result, result_col):
    """Output pa.Schema from group_cols + first group's result shape."""
    import pyarrow as pa

    fields = [df._dataset.schema.field(c) for c in group_cols]
    if mode == "scalar":
        v = _normalize_scalar(first_result)
        fields.append(pa.field(
            result_col, _infer_arrow_type_from_value(v), nullable=True,
        ))
    elif mode == "series":
        for idx in first_result.index:
            v = first_result.loc[idx]
            fields.append(pa.field(
                str(idx), _infer_arrow_type_from_value(v), nullable=True,
            ))
    else:  # dataframe
        for c in first_result.columns:
            v = first_result[c].iloc[0] if len(first_result) > 0 else None
            fields.append(pa.field(
                str(c), _infer_arrow_type_from_value(v), nullable=True,
            ))
    return pa.schema(fields)
'''


def add_helpers():
    print("\n=== cadw/dataframe.py: apply helpers ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "_detect_mode" in s:
        print("  already present — skip")
        return
    if not s.endswith("\n"):
        s += "\n"
    s += HELPERS
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 3. _GroupBy.apply wrapper ============

def add_groupby_apply():
    print("\n=== cadw/dataframe.py: _GroupBy.apply wrapper ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def apply(self, fn" in s:
        print("  already present — skip")
        return

    anchor = '''    def _normalize_on(self, on) -> list:
        if on is None:
            return []
        if isinstance(on, str):
            return [on]
        return list(on)'''

    new = '''    def apply(
        self,
        fn,
        *,
        result_col: str = "apply_result",
        on: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Per-group apply (v0.9.1e). Prototype-validated (11 cases).

        Args:
            fn: callable(group_df: pd.DataFrame) → scalar | pd.Series |
                pd.DataFrame. Uniform mode across groups.
            result_col: column name for scalar mode.
            on: within-group sort keys (None → row_id order).

        Returns: CadwDataFrame
            scalar → K rows, [*group_cols, result_col]
            Series → K rows, [*group_cols, *series.index]
            DataFrame → Σ|df_i| rows, [*group_cols, *df.columns]

        Order: sorted by group key.

        Memory: per-group bounded (2 × group_nbytes ≤ 0.3 × avail BEFORE
        fn). Oversized → MemoryLimitExceeded. fn's alloc — user resp.
        """
        on_cols = self._normalize_on(on)
        return self._parent._per_group_apply(
            self._by, on_cols, fn, result_col,
            memory_guard=self._parent._memory_guard,
        )

    def _normalize_on(self, on) -> list:
        if on is None:
            return []
        if isinstance(on, str):
            return [on]
        return list(on)'''

    s = replace_line_block(s, anchor, new, tag="groupby-apply")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 4. AST whitelist ============

def update_ast_whitelist():
    print("\n=== tests/test_arrow_call_memory_guard.py: whitelist apply ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if '"_per_group_apply"' in s:
        print("  already present — skip")
        return

    old = '''    "to_json": "params + state serialization, no df read",
    "from_json": "params + state deserialization, no df read",
}'''

    new = '''    "to_json": "params + state serialization, no df read",
    "from_json": "params + state deserialization, no df read",
    # v0.9.1e per-group apply: to_pandas() on ONE group, bounded by
    # 2 × group_nbytes ≤ 0.3 × avail check BEFORE conversion.
    "_per_group_apply": "one group in RAM, bounded by check_state",
    "apply": "delegates to _per_group_apply (bounded per group)",
}'''

    s = replace_line_block(s, old, new, tag="whitelist-apply")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 5. Tests ============

APPLY_TESTS = '''"""v0.9.1e: per-group apply(fn) tests."""
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


def test_apply_scalar_mean(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b", "b"], "v": [10, 20, 100, 200],
    })
    out = df.groupby("g").apply(lambda g: g["v"].mean())
    t = read_table(out, tmp_path)
    rows = {r["g"]: r["apply_result"] for r in t.to_pylist()}
    assert rows == {"a": 15.0, "b": 150.0}
    assert t.schema.names == ["g", "apply_result"]


def test_apply_scalar_custom_result_col(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "a", "b"], "v": [10, 20, 30]})
    out = df.groupby("g").apply(
        lambda g: g["v"].max(), result_col="max_v",
    )
    t = read_table(out, tmp_path)
    assert t.schema.names == ["g", "max_v"]


def test_apply_scalar_int_return(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b", "b", "b"], "v": [1, 2, 3, 4, 5],
    })
    out = df.groupby("g").apply(lambda g: len(g))
    t = read_table(out, tmp_path)
    rows = {r["g"]: r["apply_result"] for r in t.to_pylist()}
    assert rows == {"a": 2, "b": 3}
    assert t.schema.field("apply_result").type == pa.int64()


def test_apply_series_multi_output(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b", "b"], "v": [10, 20, 100, 200],
    })
    out = df.groupby("g").apply(
        lambda g: pd.Series({"mean": g["v"].mean(),
                             "max": g["v"].max()}),
    )
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"g", "mean", "max"}
    rows = {r["g"]: (r["mean"], r["max"]) for r in t.to_pylist()}
    assert rows == {"a": (15.0, 20), "b": (150.0, 200)}


def test_apply_dataframe_expand(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"], "v": [1, 2, 3, 10, 20],
    })
    def fn(g):
        return pd.DataFrame({
            "v_rank": np.arange(1, len(g) + 1),
            "v_shift": g["v"].tolist(),
        })
    out = df.groupby("g").apply(fn)
    t = read_table(out, tmp_path)
    assert set(t.schema.names) == {"g", "v_rank", "v_shift"}
    assert t.num_rows == 5


def test_apply_null_group(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", None, None], "v": [10, 20, 30, 40],
    })
    out = df.groupby("g").apply(lambda g: g["v"].sum())
    t = read_table(out, tmp_path)
    assert t.num_rows == 2


def test_apply_single_row_groups(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "b", "c"], "v": [10, 20, 30]})
    out = df.groupby("g").apply(lambda g: g["v"].iloc[0])
    t = read_table(out, tmp_path)
    rows = {r["g"]: r["apply_result"] for r in t.to_pylist()}
    assert rows == {"a": 10, "b": 20, "c": 30}


def test_apply_multi_group(tmp_path):
    df = make_df(tmp_path, {
        "s1": ["a", "a", "b", "b"],
        "s2": ["x", "y", "x", "y"],
        "v": [1, 2, 3, 4],
    })
    out = df.groupby(["s1", "s2"]).apply(lambda g: g["v"].sum())
    t = read_table(out, tmp_path)
    assert t.num_rows == 4
    assert set(t.schema.names) == {"s1", "s2", "apply_result"}


def test_apply_after_filter(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a", "b", "b"], "v": [1, 2, 3, 4, 5],
    })
    filtered = df[df["v"] > 1]
    out = filtered.groupby("g").apply(lambda g: g["v"].sum())
    t = read_table(out, tmp_path)
    rows = {r["g"]: r["apply_result"] for r in t.to_pylist()}
    assert rows == {"a": 5, "b": 9}


def test_apply_empty_input(tmp_path):
    df = make_df(tmp_path, {"g": ["a"], "v": [1]})
    empty = df[df["v"] > 100]
    out = empty.groupby("g").apply(lambda g: g["v"].sum())
    t = read_table(out, tmp_path)
    assert t.num_rows == 0


def test_apply_on_col_within_group(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "a"], "ts": [3, 1, 2], "v": [30, 10, 20],
    })
    out = df.groupby("g").apply(
        lambda g: g["v"].iloc[0], on="ts",
    )
    t = read_table(out, tmp_path)
    assert t.to_pylist()[0]["apply_result"] == 10


def test_apply_deterministic(tmp_path):
    df = make_df(tmp_path, {
        "g": ["b", "a", "b", "a"], "v": [1, 2, 3, 4],
    })
    t1 = read_table(
        df.groupby("g").apply(lambda g: g["v"].sum()),
        tmp_path, "o1.parquet",
    )
    t2 = read_table(
        df.groupby("g").apply(lambda g: g["v"].sum()),
        tmp_path, "o2.parquet",
    )
    assert t1.to_pylist() == t2.to_pylist()


def test_apply_fn_not_callable_raises(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "a"], "v": [1, 2]})
    with pytest.raises(TypeError, match="must be callable"):
        df.groupby("g").apply("not_a_function")


def test_apply_mixed_modes_raises(tmp_path):
    df = make_df(tmp_path, {
        "g": ["a", "a", "b", "b"], "v": [1, 2, 3, 4],
    })
    call_count = [0]
    def fn(g):
        call_count[0] += 1
        if call_count[0] == 1:
            return g["v"].sum()  # scalar
        return pd.Series({"x": 1})  # series
    with pytest.raises(RuntimeError, match="mixed return modes"):
        df.groupby("g").apply(fn)


def test_apply_unsupported_return_type_raises(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "a"], "v": [1, 2]})
    with pytest.raises(TypeError, match="expected scalar"):
        df.groupby("g").apply(lambda g: [1, 2, 3])


def test_apply_group_too_large_raises(tmp_path):
    df = make_df(
        tmp_path, {"g": ["a"] * 500, "v": list(range(500))},
        chunk=100, mem_mb=1,
    )
    df._memory_guard = MemoryGuard(max_memory_bytes=2000)
    with pytest.raises(MemoryLimitExceeded):
        df.groupby("g").apply(lambda g: g["v"].sum())


def test_apply_fn_exception_propagates(tmp_path):
    df = make_df(tmp_path, {"g": ["a", "a"], "v": [1, 2]})
    def bad_fn(g):
        raise ValueError("user error")
    with pytest.raises(ValueError, match="user error"):
        df.groupby("g").apply(bad_fn)


def test_apply_scalar_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(42)
    data = {
        "g": [f"k{i % 5}" for i in range(n)],
        "v": rng.integers(0, 100, size=n).tolist(),
    }
    df = make_df(tmp_path, data, chunk=20)
    out = df.groupby("g").apply(lambda g: g["v"].mean())
    t = read_table(out, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf.groupby("g")["v"].mean().reset_index()
    expected.columns = ["g", "apply_result"]
    got_df = t.to_pandas().sort_values("g").reset_index(drop=True)
    expected = expected.sort_values("g").reset_index(drop=True)
    np.testing.assert_allclose(
        got_df["apply_result"], expected["apply_result"], rtol=1e-9,
    )


def test_apply_ast_uses_composite_sort():
    """AST: _per_group_apply external_sort by group+on; check before to_pandas."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_apply)
    assert "external_sort" in src
    assert "to_pandas" in src
    assert "check_state" in src
    assert src.index("check_state") < src.index("to_pandas")
'''


APPLY_E2E_TESTS = '''"""v0.9.1e: per-group apply E2E (slow)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


@pytest.mark.slow
def test_apply_e2e_10k_rows(tmp_path):
    n = 10_000
    rng = np.random.default_rng(42)
    data = {
        "g": [f"k{i % 20}" for i in range(n)],
        "v": rng.normal(size=n).tolist(),
    }
    pdf = pd.DataFrame(data)
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table(data), src, row_group_size=1000)

    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=1000,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )
    out = df.groupby("g").apply(lambda g: g["v"].mean())
    out_path = tmp_path / "out.parquet"
    out.write_parquet(out_path)
    got = (pq.read_table(out_path).to_pandas()
           .sort_values("g").reset_index(drop=True))

    expected = pdf.groupby("g")["v"].mean().reset_index()
    expected.columns = ["g", "apply_result"]
    expected = expected.sort_values("g").reset_index(drop=True)

    np.testing.assert_allclose(
        got["apply_result"].to_numpy(),
        expected["apply_result"].to_numpy(),
        rtol=1e-9,
    )
    assert got.shape[0] == 20
'''


def write_tests():
    for name, content in [
        ("tests/test_apply.py", APPLY_TESTS),
        ("tests/test_apply_e2e.py", APPLY_E2E_TESTS),
    ]:
        print(f"\n=== {name} ===")
        p = Path(name)
        p.write_text(content)
        compile_check(p)
        print("  OK")


# ============ 6. CHANGELOG + DECISIONS ============

def append_changelog():
    print("\n=== CHANGELOG.md: append v0.9.1e ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.1e — apply(fn)" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Добавлено (v0.9.1e — per-group apply(fn))

- **`df.groupby('g').apply(fn, result_col='apply_result', on=None)`** —
  per-group user callable. Prototype-validated (11 cases A/B pandas
  vs composite-sort, §13.4/123).
  - **fn**: `callable(group_df: pd.DataFrame)` → scalar | `pd.Series` |
    `pd.DataFrame`. Uniform mode required across groups.
  - **Output**:
    - scalar → K rows, `[*group_cols, result_col]`
    - Series → K rows, `[*group_cols, *series.index]`
    - DataFrame → `Σ|df_i|` rows, `[*group_cols, *df.columns]`
  - **on**: within-group sort keys. `None` → row_id order.
  - **Order**: output sorted by group key.
- **`CadwDataFrame._per_group_apply(...)`** — internal: composite-sort
  by `[*group_cols, *on_cols]`, one group in RAM at a time, streaming
  output to IPC.

### Memory bound (§11.5)

- **Per-group bound**: `group_nbytes × 2 (pandas) ≤ 0.3 × available`.
  Oversized → `MemoryLimitExceeded` **before** `fn()`.
- **Peak RAM** = batch (input stream) + `2 × max(group_size × row_size)`
  + fn's output + small output-buffer batch. **Independent of N**.
- **fn's own alloc — user responsibility** (§3.9 extended): CADW bounds
  input to `fn`, not `fn`'s internals. `fn` that's O(N) per group → OOM
  is user's fault.

### Ограничения (v0.9.1e, §11.5)

- **`transform(fn)` NOT included** — semantically different (Series →
  same-length array for broadcast). v0.9.2.
- **Mixed return modes across groups** → RuntimeError.
- **Null in group key** → own group (dropna=False, matches pandas).
- **`fn` exceptions propagate** — no wrapping; user errors surface
  unchanged.
- **Output sorted by group key**, not original order.

### Тесты (v0.9.1e)

- +17 fast (`test_apply.py`): 3 modes, nulls, multi-group, single-row,
  filter, on-col sort, determinism, errors (fn not callable, mixed
  modes, unsupported type, oversized group, fn exception), pandas
  equivalence, AST guard.
- +1 slow (`test_apply_e2e.py`): 10K / 20 groups / mean, pandas equiv.

- fast: **~1378** (+17). slow: **12** (+1).
'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


def update_decisions():
    print("\n=== DECISIONS.md §11.5: apply note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.1e (apply(fn))" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        print("  WARNING: §11.5 not found — append")
        s = s.rstrip() + "\n\n**v0.9.1e (apply):** per-group fn, bounded "
        s += "2×group ≤ 0.3×avail. fn OOM = user's fault.\n"
        p.write_text(s)
        return

    note = '''**v0.9.1e (apply(fn)):** per-group user callable.
- Memory: per-group bounded (`2 × group_nbytes ≤ 0.3 × available`
  BEFORE `to_pandas`). Oversized → `MemoryLimitExceeded` before `fn()`.
- **fn's own alloc — user responsibility** (§3.9 extended).
- Three modes: scalar, pd.Series, pd.DataFrame. Uniform across groups
  (mixed → RuntimeError).
- Null in group key → own group (dropna=False).
- Output sorted by group key.
- **`transform(fn)` NOT included** — v0.9.2.
- **`fn` returning unsupported type** (list, dict) → TypeError.
- **fn exceptions propagate** — no wrapping.

'''
    s = s[:pos] + note + s[pos:]
    p.write_text(s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.1e: apply(fn) ===")

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

    add_per_group_apply()
    add_helpers()
    add_groupby_apply()
    update_ast_whitelist()
    write_tests()
    append_changelog()
    update_decisions()

    print("\n=== pytest: new apply tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_apply.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** APPLY TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: apply E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_apply_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline + CT regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_column_transformer.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
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
        "v0.9.1e: per-group apply(fn) — composite-sort, one group in "
        "RAM, prototype-validated (11 cases); three modes (scalar / "
        "Series / DataFrame); per-group memory bound 2× ≤ 0.3×avail; "
        "§11.5 constraints"])

    print("\n==> v0.9.1e applied.")
    print("NEXT: v0.9.1f (pivot_table / melt / crosstab). Then tag v0.9.1.")


if __name__ == "__main__":
    main()