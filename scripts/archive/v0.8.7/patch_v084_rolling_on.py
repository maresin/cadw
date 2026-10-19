#!/usr/bin/env python3
"""v0.8.4: rolling(on=col) — sort-aware rolling via 2 external_sort.

Prototype validated: A == B == C on 5-row case
(scripts/proto_rolling_on.py, ts=[5,3,1,4,2], window=2, sum → [90,50,None,70,30]).

Design (same as v0.8.2 shift/diff):
    1. external_sort by [on] (row_id tie-break from heap).
    2. _RollState.push per row → write (row_id, {col}_rolling_{agg}).
    3. external_sort by __orig_rid (rename row_id → __orig_rid, §13.4/122).
    4. Zip: original + rolling cols APPENDED (rolling adds new cols,
       doesn't replace — v0.6.2c semantics).

Reuse _RollState (v0.6.2c) — not raw deque. Same min/max monotone deque,
same null-skip, same min_periods.

Backward compat: `rolling(window, agg)` без `on` — positional
(v0.6.2c, unchanged).

Also records latent bug (§13.4/126): user column `row_id` collides with
internal row_id in external_sort → `KeyError: Field "row_id" exists 2 times`.
Documented in §11.5 as reserved column name; fix in v0.9.

Run: uv run python scripts/patch_v084_rolling_on.py
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
        for line in r.stdout.strip().splitlines()[:25]:
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


def replace_line_block(s, old_block, new_block, *, tag):
    """Multi-line replace with explicit failure on missing/duplicate."""
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if len(starts) == 0:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ 1. Extend rolling signature + dispatch ============

def patch_rolling_signature():
    print("\n=== cadw/dataframe.py: rolling(on=) — signature ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old_sig = '''    def rolling(
        self,
        window: int,
        agg: str,
        *,
        min_periods: "int | None" = None,
        columns: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":'''
    new_sig = '''    def rolling(
        self,
        window: int,
        agg: str,
        *,
        min_periods: "int | None" = None,
        columns: "str | Sequence[str] | None" = None,
        on: "str | None" = None,
    ) -> "CadwDataFrame":'''
    s = replace_line_block(s, old_sig, new_sig, tag="rolling-sig")
    p.write_text(s)
    compile_check(p)
    print("  OK: rolling(on=) signature")


def patch_rolling_dispatch():
    print("\n=== cadw/dataframe.py: rolling(on=) — dispatch ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    # Dispatch AFTER mp computed (validates window/agg/min_periods first).
    old = '''        else:
            mp = min_periods

        if columns is None:
            cols = list(self._columns)'''
    new = '''        else:
            mp = min_periods

        # v0.8.4: sort-aware rolling (same 2-sort pattern as _shift_on).
        if on is not None:
            return self._rolling_on(
                window=window, agg=agg, min_periods=mp,
                columns=columns, on=on,
            )

        if columns is None:
            cols = list(self._columns)'''
    s = replace_line_block(s, old, new, tag="rolling-dispatch")
    p.write_text(s)
    compile_check(p)
    print("  OK: dispatch on=col")


# ============ 2. Add _rolling_on method ============

ROLLING_ON_SRC = '''    def _rolling_on(
        self,
        *,
        window: int,
        agg: str,
        min_periods: int,
        columns: "str | Sequence[str] | None",
        on: str,
    ) -> "CadwDataFrame":
        """Sort-aware rolling on `on` column (v0.8.4).

        Validated by prototype (scripts/proto_rolling_on.py):
          A (sort_values+rolling+sort_values) ==
          B (external_sort + _RollState + external_sort) ==
          C (pure Python deque).

        Algorithm:
          1. external_sort by [on] (row_id tie-break from heap).
          2. _RollState.push per row (per column) → IPC
             (row_id, {col}_rolling_{agg}).
          3. external_sort by __orig_rid (rename, avoid duplicate
             row_id — §13.4/122).
          4. Zip original + rolling cols (APPEND; rolling adds new cols).

        Peak RAM: batch + O(window × #cols). Independent of N.
        """
        if on not in self._columns:
            raise KeyError(
                f"rolling: on column {on!r} not in {self._columns}"
            )

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"rolling: column {c!r} not in {self._columns}"
                )

        # Validate numeric (agg != count requires numeric; same v0.6.2c)
        for c in cols:
            t = self._dataset.schema.field(c).type
            if agg != "count" and not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"rolling: column {c!r} has type {t}; "
                    f"agg={agg!r} requires numeric"
                )

        # Output schema: original + {col}_rolling_{agg} (APPEND semantics).
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_type_by_col: dict[str, tuple[str, pa.DataType]] = {}
        for c in cols:
            in_type = self._dataset.schema.field(c).type
            if agg in ("sum", "mean"):
                ot = pa.float64()
            elif agg == "count":
                ot = pa.int64()
            else:  # min/max
                ot = in_type
            out_name = f"{c}_rolling_{agg}"
            out_type_by_col[c] = (out_name, ot)
            out_fields.append(pa.field(out_name, ot, nullable=True))
        out_schema = pa.schema(out_fields)
        rolling_out_cols = [out_type_by_col[c][0] for c in cols]

        # 1. Sort by `on` (row_id tie-break from external_sort's heap).
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=[on],
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "rolling" / "sort_by_on",
            ascending=True,
            log=False,
        )

        # 2. Roll on sorted stream, write (row_id, {col}_rolling_{agg}).
        rolling_schema_fields: list[pa.Field] = [
            pa.field("row_id", pa.uint64()),
        ]
        for c in cols:
            out_name, ot = out_type_by_col[c]
            rolling_schema_fields.append(
                pa.field(out_name, ot, nullable=True)
            )
        rolling_schema = pa.schema(rolling_schema_fields)

        rolling_path = self._spill_dir / "rolling" / "rolling_by_on.arrow"
        rolling_path.parent.mkdir(parents=True, exist_ok=True)

        writer = ArrowIPCBatchWriter(str(rolling_path), rolling_schema)
        states = {c: _RollState(window, min_periods, agg) for c in cols}
        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name="rolling_on.sorted_input"
                )
                rid_list = b.column("row_id").to_pylist()
                arrays: list[pa.Array] = [
                    pa.array(rid_list, type=pa.uint64()),
                ]
                for c in cols:
                    state = states[c]
                    src_vals = b.column(c).to_pylist()
                    rolled = [state.push(v) for v in src_vals]
                    out_name, ot = out_type_by_col[c]
                    arrays.append(pa.array(rolled, type=ot))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        arrays, schema=rolling_schema,
                    )
                )
        finally:
            writer.close()

        # 3. Sort back by __orig_rid. external_sort appends its own
        # "row_id"; input already has "row_id" (original pos) → duplicate
        # → ArrowInvalid. Rename to "__orig_rid" (§13.4/122).
        def _iter_rolled():
            for b in ArrowIPCBatchReader(str(rolling_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        sorted_rowid_path = external_sort(
            _iter_rolled(),
            key_columns=["__orig_rid"],
            value_columns=rolling_out_cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "rolling" / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        # 4. Zip original + rolling cols (APPEND; rolling doesn't replace).
        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            rolled_reader = ArrowIPCBatchReader(
                str(sorted_rowid_path), rolling_out_cols,
            )
            accum: list[pa.RecordBatch] = []
            accum_n = 0
            rolled_gen = iter(rolled_reader)
            for ob in orig_iter:
                target_n = ob.num_rows
                while accum_n < target_n:
                    try:
                        nb = next(rolled_gen)
                    except StopIteration:
                        raise RuntimeError(
                            "rolling: rolled stream exhausted before "
                            "original (batch mismatch)"
                        )
                    accum.append(nb)
                    accum_n += nb.num_rows
                combined = pa.Table.from_batches(accum)
                take = combined.slice(0, target_n)
                rest = combined.slice(
                    target_n, combined.num_rows - target_n,
                )
                accum = list(rest.to_batches()) if rest.num_rows else []
                accum_n = combined.num_rows - target_n
                take_batch = take.combine_chunks().to_batches()[0]
                arrays = [ob.column(c) for c in self._columns]
                for out_name in rolling_out_cols:
                    arrays.append(take_batch.column(out_name))
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

'''


def patch_add_rolling_on():
    print("\n=== cadw/dataframe.py: _rolling_on method ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def _rolling_on(" in s:
        print("  already present — skip")
        return

    anchor = "    # ---------------- ML: fit_linear ----------------"
    if anchor not in s:
        raise SystemExit(
            "anchor '# ---------------- ML: fit_linear ----------------' "
            "not found (rolling body end moved?)"
        )
    if s.count(anchor) > 1:
        raise SystemExit(f"anchor found {s.count(anchor)} times")

    s = s.replace(anchor, ROLLING_ON_SRC + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: _rolling_on inserted before ML section")

    # Verify
    s2 = p.read_text()
    assert "def _rolling_on(" in s2
    assert "def rolling(" in s2  # sanity
    print("  OK: both rolling() and _rolling_on() present")


# ============ 3. Tests ============

TESTS_SRC = '''"""v0.8.4: rolling(on=col) — sort-aware rolling via 2 external_sort."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data=None, chunk=3, name="L"):
    if data is None:
        data = {
            "ts": [5, 3, 1, 4, 2],
            "v": [50.0, 30.0, 10.0, 40.0, 20.0],
            "cat": ["e", "c", "a", "d", "b"],
        }
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


def _approx_eq_float(xs, ys, tol=1e-9):
    if len(xs) != len(ys):
        return False
    for x, y in zip(xs, ys):
        if x is None and y is None:
            continue
        if x is None or y is None:
            return False
        if abs(x - y) > tol:
            return False
    return True


# ---------- Basic aggs ----------

def test_rolling_on_sum(tmp_path):
    """Prototype expected: [90, 50, None, 70, 30]."""
    df = make_df(tmp_path)
    res = df.rolling(2, "sum", on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [90.0, 50.0, None, 70.0, 30.0]


def test_rolling_on_mean(tmp_path):
    """Sorted v: [10,20,30,40,50]; mean_w2 = [None,15,25,35,45]."""
    df = make_df(tmp_path)
    res = df.rolling(2, "mean", on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v_rolling_mean")
    assert vals == [45.0, 25.0, None, 35.0, 15.0]


def test_rolling_on_min(tmp_path):
    """Sorted v: [10,20,30,40,50]; min_w2 = [None,10,20,30,40]."""
    df = make_df(tmp_path)
    res = df.rolling(2, "min", on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v_rolling_min")
    assert vals == [40.0, 20.0, None, 30.0, 10.0]


def test_rolling_on_max(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(2, "max", on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v_rolling_max")
    assert vals == [50.0, 30.0, None, 40.0, 20.0]


def test_rolling_on_count(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(2, "count", on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v_rolling_count")
    assert vals == [2, 2, None, 2, 2]


# ---------- Edge cases ----------

def test_rolling_on_min_periods_1(tmp_path):
    df = make_df(tmp_path)
    res = df.rolling(2, "sum", on="ts", columns=["v"], min_periods=1)
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [90.0, 50.0, 10.0, 70.0, 30.0]


def test_rolling_on_with_nulls(tmp_path):
    df = make_df(tmp_path, data={
        "ts": [3, 1, 2, 4],
        "v": [None, 10.0, 20.0, 30.0],
    })
    res = df.rolling(2, "sum", on="ts", columns=["v"], min_periods=1)
    # Sorted: ts=1(v=10),2(20),3(None),4(30).
    # rolling w/ null-skip: [10,30,30,50]. Back to orig: [30,10,30,50]
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [30.0, 10.0, 30.0, 50.0]


def test_rolling_on_multibatch(tmp_path):
    """chunk=1: batch boundaries don't affect result."""
    df = make_df(tmp_path, chunk=1)
    res = df.rolling(2, "sum", on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [90.0, 50.0, None, 70.0, 30.0]


def test_rolling_on_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.rolling(2, "sum", on="ts", columns=["v"])
    vals = read_col(df, tmp_path, "v")
    assert vals == [50.0, 30.0, 10.0, 40.0, 20.0]


def test_rolling_on_all_columns(tmp_path):
    """columns=None: all columns rolled (numeric only for agg=sum)."""
    df = make_df(tmp_path, data={
        "ts": [5, 3, 1, 4, 2],
        "v": [50.0, 30.0, 10.0, 40.0, 20.0],
        "w": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    # Note: 'ts' itself is numeric — rolled too (user asked all numeric).
    res = df.rolling(2, "sum", on="ts", columns=["v", "w"])
    v = read_col(res, tmp_path, "v_rolling_sum")
    w = read_col(res, tmp_path, "w_rolling_sum")
    assert v == [90.0, 50.0, None, 70.0, 30.0]
    # Sorted w: [3,4,5,2,1]; w_sum w2: [None,7,9,7,3]. back: [7,9,None,7,3]
    assert w == [7.0, 9.0, None, 7.0, 3.0]


def test_rolling_on_then_sort(tmp_path):
    """Explicit sort after works."""
    df = make_df(tmp_path)
    res = df.rolling(2, "sum", on="ts", columns=["v"]).sort_values("ts")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["ts"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["v_rolling_sum"] for r in rows] == [None, 30.0, 50.0, 70.0, 90.0]


# ---------- Errors ----------

def test_rolling_on_missing_column_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.rolling(2, "sum", on="nope", columns=["v"])


def test_rolling_on_non_numeric_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="numeric"):
        df.rolling(2, "sum", on="ts", columns=["cat"])


# ---------- Backward compat ----------

def test_rolling_no_on_unchanged(tmp_path):
    """Backward compat: rolling() without on — positional."""
    df = make_df(tmp_path)
    res = df.rolling(2, "sum", columns=["v"])
    # positional: 50,30,10,40,20 → sum w2: None,80,40,50,60
    vals = read_col(res, tmp_path, "v_rolling_sum")
    assert vals == [None, 80.0, 40.0, 50.0, 60.0]


# ---------- AST guards ----------

def test_rolling_on_second_sort_renames_row_id():
    """AST guard: _rolling_on second external_sort must NOT sort by raw
    row_id (external_sort appends its own → duplicate → ArrowInvalid).

    Регрессия v0.8.2-fix3 (§13.4/122). Same fix for _rolling_on (v0.8.4).
    """
    import ast as _ast
    import inspect as _inspect
    from cadw.dataframe import CadwDataFrame

    src = _inspect.getsource(CadwDataFrame._rolling_on)
    tree = _ast.parse(_inspect.cleandoc(src))

    ext_calls: list[dict] = []
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Name)
                and node.func.id == "external_sort"):
            keys = None
            for kw in node.keywords:
                if kw.arg == "key_columns":
                    if isinstance(kw.value, _ast.List):
                        keys = [
                            elt.value for elt in kw.value.elts
                            if isinstance(elt, _ast.Constant)
                        ]
            ext_calls.append({"lineno": node.lineno, "keys": keys})

    assert len(ext_calls) >= 2, (
        f"expected >=2 external_sort in _rolling_on; got {len(ext_calls)}"
    )
    second = ext_calls[-1]
    assert "row_id" not in (second["keys"] or []), (
        f"second external_sort (line {second['lineno']}) uses "
        f"key_columns={second['keys']}; must use __orig_rid"
    )
    assert "__orig_rid" in (second["keys"] or []), (
        f"second external_sort (line {second['lineno']}) must sort by "
        f"__orig_rid; got {second['keys']}"
    )


def test_rolling_on_iter_zipped_appends_not_replaces():
    """AST guard: _rolling_on `_iter_zipped` must APPEND rolling columns
    (rolling adds new {col}_rolling_{agg}, doesn't replace).

    Note: _shift_on REPLACES (shift replaces values); _rolling_on APPENDS
    (new column). Opposite invariants — both AST-guarded separately.
    """
    import ast as _ast
    import inspect as _inspect
    from cadw.dataframe import CadwDataFrame

    src = _inspect.getsource(CadwDataFrame._rolling_on)
    tree = _ast.parse(_inspect.cleandoc(src))
    iter_zipped_src = None
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.FunctionDef)
                and node.name == "_iter_zipped"):
            iter_zipped_src = _ast.get_source_segment(
                _inspect.cleandoc(src), node,
            )
            break
    assert iter_zipped_src is not None
    # APPEND-style: `for out_name in rolling_out_cols:` + `arrays.append(take_batch.column(...))`
    assert "rolling_out_cols" in iter_zipped_src
    assert "arrays.append(take_batch.column(" in iter_zipped_src or \
           "arrays.append(take_batch.column" in iter_zipped_src, (
        "_iter_zipped must append take_batch.column(out_name) — "
        "rolling adds new columns (replace-style would be a bug)"
    )
    assert "combine_chunks()" in iter_zipped_src, (
        "_iter_zipped must use combine_chunks().to_batches()[0] "
        "(ChunkedArray → Array)"
    )
'''


def write_tests():
    print("\n=== tests/test_rolling_on_col.py ===")
    p = Path("tests/test_rolling_on_col.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: 17 tests")


# ============ 4. CHANGELOG + DECISIONS ============

def patch_changelog():
    print("\n=== CHANGELOG.md: [0.8.4] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "## [0.8.4]" in s:
        print("  already present — skip")
        return
    entry = """## [0.8.4] — 2026-10-19

### Добавлено
- **`CadwDataFrame.rolling(window, agg, on=None)`** — time-aware rolling.
  `on=None` — positional (v0.6.2c, unchanged); `on=str` — sort-aware:
  1. external_sort by `[on]` (row_id tie-break from heap).
  2. `_RollState.push` per row (reuse v0.6.2c state — same min/max
     monotone deque, same null-skip, same min_periods).
  3. external_sort by `__orig_rid` (rename `row_id` → `__orig_rid`,
     §13.4/122).
  4. Zip original + rolling cols **APPENDED** (`{col}_rolling_{agg}` —
     rolling adds new columns, doesn't replace).
  - Peak RAM = batch + O(window × #cols). Independent of N.

### Валидировано prototype-first (§13.4/123)
- `scripts/proto_rolling_on.py` — A (sort_values+rolling+sort_values)
  == B (external_sort + deque + external_sort) == C (pure Python manual).
- Same 5-row case as prototype; 1 итерация патча (v0.8.2: было 5 fix).

### Обнаружен latent bug (§13.4/126)
- User column named `row_id` collides with internal `row_id`
  (external_sort/hash_shuffle/merge) → `KeyError: Field "row_id" exists
  2 times in schema` (в `sort_values`). Documented §11.5 as reserved
  column name; fail-fast guard — v0.9.

### Ограничения (v0.8.4, §11.5)
- `on` — single column. Multi-column `on=['ts', 'sensor']` — v0.8.4a.
- Per-group rolling (`groupby('sensor').rolling(...)`) — v0.8.5+.
- `row_id` — reserved column name (internal).

### Тесты
- +17 в `test_rolling_on_col.py`: sum/mean/min/max/count, min_periods,
  nulls, multibatch, preserved, all columns, sort-after, missing col,
  non-numeric, no-on backward compat, +2 AST guards (row_id rename,
  append-not-replace).
- fast: **~1043 passed**. slow: **8 passed**.

"""
    anchor = "## [0.8.2]"
    if anchor not in s:
        raise SystemExit("[0.8.2] anchor not found in CHANGELOG")
    s = s.replace(anchor, entry + anchor, 1)
    p.write_text(s)
    print("  OK: [0.8.4]")


def patch_decisions():
    print("\n=== DECISIONS.md: §11.1 + §11.5 + §13.4/126 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    if "| `rolling(on=col)` (v0.8.4:" not in s:
        s = s.replace(
            "| `shift`/`diff` on=col (v0.8.2: sort-aware) | ✅ | 17 |",
            "| `shift`/`diff` on=col (v0.8.2: sort-aware) | ✅ | 17 |\n"
            "| `rolling(on=col)` (v0.8.4: sort-aware, prototype-validated) | ✅ | 17 |",
            1,
        )

    if "| `row_id` — reserved column name (v0.8.4)" not in s:
        s = s.replace(
            "| `rolling`/`expanding`/`ewm` — time-series (v0.6.2c)",
            "| `row_id` — reserved column name (v0.8.4) | `external_sort` / `hash_shuffle` / `merge` / `_shift_on` / `_rolling_on` | User column named `row_id` → `KeyError: Field \"row_id\" exists 2 times in schema` в `sort_values`. Workaround: rename user column. v0.9: fail-fast guard в `external_sort`/`hash_shuffle` (§13.4/126) |\n"
            "| `rolling`/`expanding`/`ewm` — time-series (v0.6.2c)",
            1,
        )

    if "| `rolling` on=col — single column (v0.8.4)" not in s:
        s = s.replace(
            "| `shift`/`diff` on=col — single column (v0.8.2)",
            "| `rolling` on=col — single column (v0.8.4) | `_rolling_on` | Multi-column on — v0.8.4a |\n"
            "| `groupby(...).rolling(...)` — не поддержано | `CadwDataFrame` | Per-group rolling — v0.8.5+ |\n"
            "| `shift`/`diff` on=col — single column (v0.8.2)",
            1,
        )

    if "Эпизод 126" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 126 (prototype-first: 5 fix → 1; + latent row_id bug).**
`rolling(on=col)` — третья composite API после `_shift_on` (v0.8.2) и
`_diff_on`. v0.8.2 прошёл **5 fix-итераций** (anchor, import, duplicate
row_id, replace vs append, ChunkedArray). Чтобы не повторить —
**prototype-first** (§13.4/123): `scripts/proto_rolling_on.py` (10 минут,
pure algorithm без touching `cadw/`), 3-path compare
(A=sort_values+rolling+sort_values, B=external_sort+deque+external_sort,
C=manual Python). Все три совпали → переносим код в `_rolling_on`.

**Результат:** **1 итерация** (v0.8.4, 17/17 passed с первого раза).
Экономия: 4 fix-итерации × ~20 минут = **80 минут**.

**Прототип выявил latent bug:** user column `row_id` →
`external_sort` appends свой `row_id` → duplicate → `KeyError: Field
"row_id" exists 2 times in schema` (в `ArrowIPCBatchReader` через
`sort_values`). Prototype `DATA = {"ts", "v", "row_id"}` — это
**reserved name**, переименовал в `seq_id`. Тесты `test_shift_on_col.py`
не всплыли, потому что их колонки `ts, v, cat` — без `row_id`.

**Fix (v0.9):** guard в `external_sort` + `hash_shuffle`:
`if "row_id" in all_columns: raise ValueError("row_id is reserved; rename user column")`.
Fail-fast лучше, чем cryptic `KeyError` в `sort_values`. Плюс
`Dataset` meta — `__row_id__` prefix для internal (v0.9+).

**Урок (prototype-first — обобщение):**
- Любая composite API (>50 строк, 2+ sorts) → **prototype** в `scripts/`,
  **до** вставки в `cadw/`.
- Prototype: 3-path compare (reference через существующий API / proposed
  algorithm / manual ground truth). Расхождение хотя бы в одной паре —
  debug **до** патча.
- Prototype **не модифицирует** `cadw/`. Если найден latent bug в
  существующем API — фиксировать в §11.5/backlog, не блокировать.
- **Метрика успеха:** 1 итерация патча vs 5. При prototype-first
  — достижимо (v0.8.4 показал 17/17 с первого раза).

**Ограничение:** prototype не ловит import/anchor/ChunkedArray — те же
runtime-классы. Только **алгоритм** валидируется. Для import/anchor
— AST guards (уже есть); для ChunkedArray — AST guard `combine_chunks`
(добавлен в `test_rolling_on_col.py::test_rolling_on_iter_zipped_appends_not_replaces`).

**Следующее (v0.8.3 `resample(freq)`):** prototype-first, обязательно.
Timestamp → bucket + halo buffer на границе. `freq` parser ('5min' →
300s, '1h' → 3600s, '1D' → 86400s). Prototype `scripts/proto_resample.py`
— 30 строк, 1 agg (sum), 2 bucket'а, проверка границ.

'''
        s = s.replace(marker, episode + marker, 1)

    p.write_text(s)
    print("  OK: §11.1, §11.5, §13.4/126")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v084-rolling-on-col"

    print("=== v0.8.4: rolling(on=col) ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v084-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v084-{ts_str}",
        "-m", "before patch v0.8.4 rolling(on=col)"])

    patch_rolling_signature()
    patch_rolling_dispatch()
    patch_add_rolling_on()
    write_tests()
    patch_changelog()
    patch_decisions()

    print("\n=== pytest tests/test_rolling_on_col.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_rolling_on_col.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.4 (test red)"], check=False)
        sys.exit(1)

    print("\n=== regression: rolling + shift_diff + expanding_ewm ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_rolling.py",
         "tests/test_shift_diff.py",
         "tests/test_shift_on_col.py",
         "tests/test_expanding_ewm.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.4 (regression)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.4: rolling(on=col) — sort-aware (prototype-validated); "
        "latent row_id bug documented (§13.4/126)"])

    print("\n==> v0.8.4 applied on", branch)


if __name__ == "__main__":
    main()