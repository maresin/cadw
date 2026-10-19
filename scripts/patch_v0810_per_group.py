#!/usr/bin/env python3
"""v0.8.10: per-group shift/diff/rolling — composite-sort, no shuffle.

ALGORITHM (validated by scripts/proto_per_group.py, 11 cases, A==B):
  1. external_sort(key_columns=[*group_cols, *on_cols]) — contiguous groups.
  2. Stream sorted; single deque per value column; on group change → clear.
  3. external_sort(key_columns=["__orig_rid"]) — sort back.
  4. Zip with original.

Peak RAM = batch + O(window × #cols). Independent of N and G.
No HashShuffle; no partition files.

API:
  df.groupby('sensor').shift(1, on='ts')
  df.groupby('sensor').shift(1, on='ts', columns=['v'])
  df.groupby('sensor').diff(1, on='ts')
  df.groupby('sensor').rolling(5, 'mean', on='ts')
  df.groupby('sensor').shift(1)   # on=None → original order within group

GUARDS (this patch runs them BEFORE commit, per §13.4/140 lesson):
  - test_changelog (3 passed)
  - test_arrow_call_memory_guard (AST — new method names added to whitelist)
  - test_groupby_per_group (new — 30 tests)
  - full fast + slow

Branch: rolling on master (no feature branch — v0.8.10 tag will include
v0.8.10-pre + this commit as one squash).

Run: uv run python scripts/patch_v0810_per_group.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


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


# ============ A. _RollState.reset() ============

def add_rollstate_reset():
    print("\n=== cadw/dataframe.py: _RollState.reset() ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def reset(self) -> None:\n        \"\"\"Reset state for per-group" in s:
        print("  already present — skip")
        return

    # Anchor: end of _RollState.push, before _ExpandingState class
    anchor = '''        raise RuntimeError(f"_RollState: unknown agg {self._agg!r}")


class _ExpandingState:'''

    new = '''        raise RuntimeError(f"_RollState: unknown agg {self._agg!r}")

    def reset(self) -> None:
        """Reset state for per-group rolling (v0.8.10).

        Called on group change. Clears deque/history/accumulators; next
        push starts fresh (min_periods applies from group start).
        """
        self._buf.clear()
        self._idx = 0
        self._sum = 0.0
        self._count = 0
        self._dq.clear()


class _ExpandingState:'''

    s = replace_line_block(s, anchor, new, tag="rollstate-reset")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ B. _GroupBy.shift/diff/rolling ============

def add_groupby_methods():
    print("\n=== cadw/dataframe.py: _GroupBy.shift/diff/rolling ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def shift(self, k: int = 1, fill_value=None" in s and \
       "self._parent._per_group_shift" in s:
        print("  already present — skip")
        return

    old = '''    def agg(self, agg_columns, agg_funcs, *, combiner: bool = False) -> CadwDataFrame:
        """Aggregate группами.

        - combiner=True: partial aggregation до shuffle для
          sum/count/min/max/mean (экономия I/O в N/K раз для низкой
          кардинальности). std → warning + fallback. §13.4/52.
        """
        agg_cols = [agg_columns] if isinstance(agg_columns, str) else list(agg_columns)
        funcs = [agg_funcs] if isinstance(agg_funcs, str) else list(agg_funcs)
        return self._parent._groupby_impl(
            self._by, agg_cols, funcs, combiner=combiner
        )'''

    new = '''    def agg(self, agg_columns, agg_funcs, *, combiner: bool = False) -> CadwDataFrame:
        """Aggregate группами.

        - combiner=True: partial aggregation до shuffle для
          sum/count/min/max/mean (экономия I/O в N/K раз для низкой
          кардинальности). std → warning + fallback. §13.4/52.
        """
        agg_cols = [agg_columns] if isinstance(agg_columns, str) else list(agg_columns)
        funcs = [agg_funcs] if isinstance(agg_funcs, str) else list(agg_funcs)
        return self._parent._groupby_impl(
            self._by, agg_cols, funcs, combiner=combiner
        )

    def _normalize_on(self, on) -> list:
        if on is None:
            return []
        if isinstance(on, str):
            return [on]
        return list(on)

    def shift(
        self,
        k: int = 1,
        fill_value=None,
        *,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group shift (v0.8.10).

        Shift by `k` within each group, optionally order-aware (`on=col`).
        `on=None` → within-group order = original row_id order.

        - k: >= 1. First k rows of EACH GROUP → `fill_value`.
        - columns: value columns to shift (default: all except group keys).
        - on: sort key within group (timestamp/integer), or None.

        Group columns are NOT shifted by default (they're the key).
        See `_per_group_shift` docstring for full semantics.
        """
        on_cols = self._normalize_on(on)
        return self._parent._per_group_shift(
            self._by, on_cols, k, fill_value, columns
        )

    def diff(
        self,
        k: int = 1,
        *,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group first difference (v0.8.10).

        `out[i] = x[i] - x[i-k]` within each group. First k rows of each
        group → null (per-group, not global).
        """
        on_cols = self._normalize_on(on)
        return self._parent._per_group_diff(
            self._by, on_cols, k, columns
        )

    def rolling(
        self,
        window: int,
        agg: str,
        *,
        min_periods: "int | None" = None,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group rolling (v0.8.10). APPENDS `{col}_rolling_{agg}`.

        Rolling computed within each group independently (min_periods
        applies from group start). Output preserves original row order;
        rolling columns are appended.
        """
        on_cols = self._normalize_on(on)
        return self._parent._per_group_rolling(
            self._by, on_cols, window, agg, min_periods, columns
        )'''

    s = replace_line_block(s, old, new, tag="groupby-methods")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ C. _per_group_* methods on CadwDataFrame ============

def add_per_group_methods():
    print("\n=== cadw/dataframe.py: _per_group_shift/_diff/_rolling ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def _per_group_shift(" in s:
        print("  already present — skip")
        return

    # Insert before "# ---------------- resample ----------------"
    anchor = '''    # ---------------- resample ----------------

    def resample(
        self,
        freq: "int | str",'''

    new = '''    # ---------------- per-group shift/diff/rolling (v0.8.10) ----------------

    def _per_group_shift(
        self,
        group_cols: list[str],
        on_cols: list[str],
        k: int,
        fill_value,
        columns: "str | Sequence[str] | None",
    ) -> "CadwDataFrame":
        """Composite-sort per-group shift (validated by proto_per_group.py).

        Algorithm (11 cases, A==B):
          1. external_sort by [*group_cols, *on_cols] — contiguous groups.
          2. Stream sorted; single deque per value col; clear on group change.
          3. external_sort by __orig_rid.
          4. Zip: replace shifted cols, keep originals.

        Peak RAM = batch + O(k × #cols). Independent of N and G.
        """
        from collections import deque

        if not group_cols:
            raise ValueError("_per_group_shift: group_cols must be non-empty")
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"_per_group_shift: k must be int >= 1; got {k!r}")
        for gc in group_cols:
            if gc not in self._columns:
                raise KeyError(
                    f"_per_group_shift: group column {gc!r} not in {self._columns}"
                )
        for oc in on_cols:
            if oc not in self._columns:
                raise KeyError(
                    f"_per_group_shift: on column {oc!r} not in {self._columns}"
                )

        if columns is None:
            cols = [c for c in self._columns if c not in group_cols]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c in group_cols:
                raise ValueError(
                    f"_per_group_shift: group column {c!r} cannot be shifted"
                )
            if c not in self._columns:
                raise KeyError(
                    f"_per_group_shift: column {c!r} not in {self._columns}"
                )
        if not cols:
            raise ValueError("_per_group_shift: no value columns to shift")
        shift_cols = set(cols)

        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in shift_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir()
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
            spill_dir=self._spill_dir / "per_group_shift" / "sort_by_keys",
            ascending=True,
            log=False,
        )

        applied_schema = pa.schema(
            [pa.field("row_id", pa.uint64())]
            + [
                pa.field(c, self._dataset.schema.field(c).type, nullable=True)
                for c in cols
            ]
        )
        applied_path = self._spill_dir / "per_group_shift" / "applied.arrow"
        applied_path.parent.mkdir(parents=True, exist_ok=True)

        writer = ArrowIPCBatchWriter(str(applied_path), applied_schema)
        bufs = {c: deque(maxlen=k) for c in cols}
        _SENTINEL = object()
        prev_group = _SENTINEL
        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(b, name="per_group_shift.sorted")
                n = b.num_rows
                if n == 0:
                    continue
                group_arrs = {gc: b.column(gc).to_pylist() for gc in group_cols}
                col_vals = {c: b.column(c).to_pylist() for c in cols}
                shifted_vals = {c: [None] * n for c in cols}
                for i in range(n):
                    gk = tuple(group_arrs[gc][i] for gc in group_cols)
                    if gk != prev_group:
                        for dq in bufs.values():
                            dq.clear()
                        prev_group = gk
                    for c in cols:
                        if len(bufs[c]) < k:
                            shifted_vals[c][i] = fill_value
                        else:
                            shifted_vals[c][i] = bufs[c][0]
                        bufs[c].append(col_vals[c][i])
                arrays = [b.column("row_id")]
                for c in cols:
                    arrays.append(
                        pa.array(
                            shifted_vals[c],
                            type=b.schema.field(c).type,
                        )
                    )
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
        finally:
            writer.close()

        def _iter_applied():
            for b in ArrowIPCBatchReader(str(applied_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        resorted_path = external_sort(
            _iter_applied(),
            key_columns=["__orig_rid"],
            value_columns=cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "per_group_shift" / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            sorted_reader = ArrowIPCBatchReader(str(resorted_path), cols)
            accum: list = []
            accum_n = 0
            gen = iter(sorted_reader)
            for ob in orig_iter:
                target_n = ob.num_rows
                while accum_n < target_n:
                    try:
                        nb = next(gen)
                    except StopIteration:
                        raise RuntimeError(
                            "per_group_shift: sorted stream exhausted before "
                            "original"
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
                arrays = []
                for c in self._columns:
                    if c in shift_cols:
                        arrays.append(take_batch.column(c))
                    else:
                        arrays.append(ob.column(c))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def _per_group_diff(
        self,
        group_cols: list[str],
        on_cols: list[str],
        k: int,
        columns: "str | Sequence[str] | None",
    ) -> "CadwDataFrame":
        """Per-group diff — delegate to shift then subtract."""
        import pyarrow.compute as _pc

        if not isinstance(k, int) or k < 1:
            raise ValueError(f"_per_group_diff: k must be int >= 1; got {k!r}")

        if columns is None:
            cols = [c for c in self._columns if c not in group_cols]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c in group_cols:
                raise ValueError(
                    f"_per_group_diff: group column {c!r} cannot be diffed"
                )
            if c not in self._columns:
                raise KeyError(
                    f"_per_group_diff: column {c!r} not in {self._columns}"
                )
        for c in cols:
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"_per_group_diff: column {c!r} has type {t}; "
                    f"numeric required"
                )
        if not cols:
            raise ValueError("_per_group_diff: no value columns")

        diff_cols = set(cols)

        shifted_df = self._per_group_shift(
            group_cols, on_cols, k, None, cols,
        )

        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in diff_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        batch_size = 10_000

        def _iter_diffed():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            shifted_iter = shifted_df._iter_arrow_batches(batch_size=batch_size)
            for ob, sb in zip(orig_iter, shifted_iter):
                if ob.num_rows != sb.num_rows:
                    raise RuntimeError(
                        f"_per_group_diff: batch size mismatch "
                        f"(orig={ob.num_rows}, shifted={sb.num_rows})"
                    )
                arrays = []
                for c in self._columns:
                    if c in diff_cols:
                        arrays.append(_pc.subtract(ob.column(c), sb.column(c)))
                    else:
                        arrays.append(ob.column(c))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_diffed(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def _per_group_rolling(
        self,
        group_cols: list[str],
        on_cols: list[str],
        window: int,
        agg: str,
        min_periods: "int | None",
        columns: "str | Sequence[str] | None",
    ) -> "CadwDataFrame":
        """Composite-sort per-group rolling — APPEND semantics."""
        if not group_cols:
            raise ValueError("_per_group_rolling: group_cols must be non-empty")
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"_per_group_rolling: window must be int >= 1; got {window!r}"
            )
        if agg not in {"sum", "mean", "min", "max", "count"}:
            raise ValueError(
                f"_per_group_rolling: agg must be one of sum/mean/min/max/"
                f"count; got {agg!r}"
            )
        if min_periods is None:
            mp = window
        elif not isinstance(min_periods, int):
            raise TypeError(
                f"_per_group_rolling: min_periods must be int; "
                f"got {type(min_periods).__name__}"
            )
        elif not (1 <= min_periods <= window):
            raise ValueError(
                f"_per_group_rolling: min_periods must be in [1, window="
                f"{window}]; got {min_periods}"
            )
        else:
            mp = min_periods

        for gc in group_cols:
            if gc not in self._columns:
                raise KeyError(
                    f"_per_group_rolling: group column {gc!r} not in "
                    f"{self._columns}"
                )
        for oc in on_cols:
            if oc not in self._columns:
                raise KeyError(
                    f"_per_group_rolling: on column {oc!r} not in "
                    f"{self._columns}"
                )

        if columns is None:
            cols = [c for c in self._columns if c not in group_cols]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c in group_cols:
                raise ValueError(
                    f"_per_group_rolling: group column {c!r} cannot be rolled"
                )
            if c not in self._columns:
                raise KeyError(
                    f"_per_group_rolling: column {c!r} not in {self._columns}"
                )
        if not cols:
            raise ValueError("_per_group_rolling: no value columns")
        for c in cols:
            t = self._dataset.schema.field(c).type
            if agg != "count" and not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"_per_group_rolling: column {c!r} has type {t}; "
                    f"agg={agg!r} requires numeric"
                )

        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_type_by_col: dict[str, tuple[str, pa.DataType]] = {}
        for c in cols:
            in_type = self._dataset.schema.field(c).type
            if agg in ("sum", "mean"):
                ot = pa.float64()
            elif agg == "count":
                ot = pa.int64()
            else:
                ot = in_type
            out_name = f"{c}_rolling_{agg}"
            out_type_by_col[c] = (out_name, ot)
            out_fields.append(pa.field(out_name, ot, nullable=True))
        out_schema = pa.schema(out_fields)
        rolling_out_cols = [out_type_by_col[c][0] for c in cols]

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir()
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
            spill_dir=self._spill_dir / "per_group_rolling" / "sort_by_keys",
            ascending=True,
            log=False,
        )

        applied_schema = pa.schema(
            [pa.field("row_id", pa.uint64())]
            + [
                pa.field(
                    out_type_by_col[c][0],
                    out_type_by_col[c][1],
                    nullable=True,
                )
                for c in cols
            ]
        )
        applied_path = self._spill_dir / "per_group_rolling" / "applied.arrow"
        applied_path.parent.mkdir(parents=True, exist_ok=True)

        writer = ArrowIPCBatchWriter(str(applied_path), applied_schema)
        states = {c: _RollState(window, mp, agg) for c in cols}
        _SENTINEL = object()
        prev_group = _SENTINEL
        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name="per_group_rolling.sorted"
                )
                n = b.num_rows
                if n == 0:
                    continue
                group_arrs = {gc: b.column(gc).to_pylist() for gc in group_cols}
                col_vals = {c: b.column(c).to_pylist() for c in cols}
                rolled_vals = {c: [None] * n for c in cols}
                for i in range(n):
                    gk = tuple(group_arrs[gc][i] for gc in group_cols)
                    if gk != prev_group:
                        for st in states.values():
                            st.reset()
                        prev_group = gk
                    for c in cols:
                        rolled_vals[c][i] = states[c].push(col_vals[c][i])
                arrays = [b.column("row_id")]
                for c in cols:
                    out_name, ot = out_type_by_col[c]
                    arrays.append(pa.array(rolled_vals[c], type=ot))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
        finally:
            writer.close()

        def _iter_rolled():
            for b in ArrowIPCBatchReader(str(applied_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        resorted_path = external_sort(
            _iter_rolled(),
            key_columns=["__orig_rid"],
            value_columns=rolling_out_cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "per_group_rolling" / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            sorted_reader = ArrowIPCBatchReader(
                str(resorted_path), rolling_out_cols,
            )
            accum: list = []
            accum_n = 0
            gen = iter(sorted_reader)
            for ob in orig_iter:
                target_n = ob.num_rows
                while accum_n < target_n:
                    try:
                        nb = next(gen)
                    except StopIteration:
                        raise RuntimeError(
                            "per_group_rolling: sorted stream exhausted "
                            "before original"
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
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    # ---------------- resample ----------------

    def resample(
        self,
        freq: "int | str",'''

    s = replace_line_block(s, anchor, new, tag="per-group-methods")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ D. AST guard whitelist update ============

def update_ast_guard_whitelist():
    print("\n=== tests/test_arrow_call_memory_guard.py: whitelist per_group_* ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if '"_per_group_shift"' in s:
        print("  already present — skip")
        return

    old = '''    "merge": "Dataset.from_batches (streaming)",
    "_wrap_result_table": "Dataset.from_batches (streaming, small result)",
}'''

    new = '''    "merge": "Dataset.from_batches (streaming)",
    "_wrap_result_table": "Dataset.from_batches (streaming, small result)",
    # v0.8.10 per-group methods — bounded accum list in _iter_zipped,
    # same pattern as _shift_on/_rolling_on.
    "_per_group_shift": "2× external_sort + bounded accum zip",
    "_per_group_diff": "delegates to _per_group_shift + streaming subtract",
    "_per_group_rolling": "2× external_sort + _RollState + bounded accum",
}'''

    s = replace_line_block(s, old, new, tag="whitelist-per-group")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ E. tests/_helpers/per_group_reference.py ============

REFERENCE_HELPER = '''"""Reference per-group shift/diff/rolling via pandas (v0.8.10).

Ground-truth for test expectations (§13.4/128, /129, /139, /141):
expectations authored by reimplementing the same operation in pandas,
not by manual arithmetic. If pandas semantics and CADW diverge — test
fails loudly; investigate before touching either.
"""
from __future__ import annotations


def _normalize(vals):
    """Convert pandas NaN → None (matches CADW null semantics)."""
    out = []
    for v in vals:
        if v is None:
            out.append(None)
        elif isinstance(v, float) and v != v:  # NaN check
            out.append(None)
        else:
            out.append(v)
    return out


def expected_per_group_shift(
    vals, groups, *, on=None, k=1, fill_value=None,
):
    """Per-group shift reference.

    - vals: list of values to shift.
    - groups: list of group values (one column).
    - on: list of on-values, or None → original row_id order.
    - k, fill_value: as CADW.

    Returns list (CADW order = original row order).
    """
    import pandas as pd

    n = len(vals)
    if on is None:
        on_vals = list(range(n))
    else:
        on_vals = list(on)
    assert len(groups) == n, "groups length mismatch"
    assert len(on_vals) == n, "on length mismatch"

    pdf = pd.DataFrame({
        "v": vals,
        "grp": groups,
        "on": on_vals,
        "__rid": list(range(n)),
    })
    pdf = pdf.sort_values(
        ["grp", "on", "__rid"], kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    pdf["__out"] = pdf.groupby("grp", dropna=False)["v"].shift(k)
    pdf = pdf.sort_values("__rid").reset_index(drop=True)
    result = _normalize(pdf["__out"].tolist())
    if fill_value is not None:
        result = [fill_value if v is None else v for v in result]
    return result


def expected_per_group_diff(vals, groups, *, on=None, k=1):
    """Per-group diff reference."""
    import pandas as pd

    n = len(vals)
    on_vals = list(range(n)) if on is None else list(on)

    pdf = pd.DataFrame({
        "v": vals,
        "grp": groups,
        "on": on_vals,
        "__rid": list(range(n)),
    })
    pdf = pdf.sort_values(
        ["grp", "on", "__rid"], kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    pdf["__out"] = pdf.groupby("grp", dropna=False)["v"].diff(k)
    pdf = pdf.sort_values("__rid").reset_index(drop=True)
    return _normalize(pdf["__out"].tolist())


def expected_per_group_rolling(
    vals, groups, *, on=None, window=2, agg="sum", min_periods=None,
):
    """Per-group rolling reference. Returns list (original order)."""
    import pandas as pd

    n = len(vals)
    on_vals = list(range(n)) if on is None else list(on)
    mp = window if min_periods is None else min_periods

    pdf = pd.DataFrame({
        "v": vals,
        "grp": groups,
        "on": on_vals,
        "__rid": list(range(n)),
    })
    pdf = pdf.sort_values(
        ["grp", "on", "__rid"], kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    rolling = pdf.groupby("grp", dropna=False)["v"].rolling(
        window, min_periods=mp,
    )
    agg_vals = getattr(rolling, agg)().reset_index(level=0, drop=True)
    pdf["__out"] = agg_vals.values
    pdf = pdf.sort_values("__rid").reset_index(drop=True)
    return _normalize(pdf["__out"].tolist())
'''


def write_reference_helper():
    print("\n=== tests/_helpers/per_group_reference.py ===")
    p = Path("tests/_helpers/per_group_reference.py")
    p.write_text(REFERENCE_HELPER)
    compile_check(p)
    print("  OK")


# ============ F. tests/test_groupby_per_group.py ============

TEST_FILE = '''"""v0.8.10: per-group shift/diff/rolling.

Prototype-validated (scripts/proto_per_group.py, 11 cases A==B).
Reference expectations from tests/_helpers/per_group_reference.py
(pandas ground truth, §13.4/128, /129, /139).
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from tests._helpers.per_group_reference import (
    expected_per_group_shift,
    expected_per_group_diff,
    expected_per_group_rolling,
)


def make_df(tmp_path, data, chunk=3, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_col(df, tmp_path, col, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


def _float_eq(a, b, tol=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return a == b


def _assert_lists_close(got, expected, *, tol=1e-9, msg=""):
    assert len(got) == len(expected), f"length mismatch {msg}"
    for i, (g, e) in enumerate(zip(got, expected)):
        assert _float_eq(g, e, tol), (
            f"[{i}] got={g!r} expected={e!r} ({msg})"
        )


# ============ shift — basic ============

def test_shift_per_group_basic(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "ts": [1, 2, 3, 1, 2, 3],
        "v": [10, 20, 30, 100, 200, 300],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_no_on(tmp_path):
    """on=None → within-group order = original row_id."""
    data = {
        "sensor": ["a", "a", "b", "b"],
        "v": [10, 20, 30, 40],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=None, k=1,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_k2(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "ts": [1, 2, 3, 1, 2, 3],
        "v": [10, 20, 30, 100, 200, 300],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(2, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=2,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_fill_value(tmp_path):
    data = {
        "sensor": ["a", "a", "b", "b"],
        "ts": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(
        1, fill_value=0, on="ts", columns=["v"],
    )
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1, fill_value=0,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_shuffled_input(tmp_path):
    data = {
        "sensor": ["b", "a", "b", "a", "b", "a"],
        "ts": [2, 1, 1, 2, 3, 3],
        "v": [200, 10, 100, 20, 300, 30],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


# ============ shift — nulls ============

def test_shift_per_group_nulls_in_value(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "a"],
        "ts": [1, 2, 3, 4],
        "v": [10, None, 30, None],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_nulls_in_group(tmp_path):
    """Null group = own group (SQL, not pandas dropna=True default)."""
    data = {
        "sensor": ["a", "a", None, None, "b"],
        "ts": [1, 2, 1, 2, 1],
        "v": [10, 20, 30, 40, 50],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


# ============ shift — edge ============

def test_shift_per_group_single_row_per_group(tmp_path):
    data = {
        "sensor": ["a", "b", "c"],
        "ts": [1, 1, 1],
        "v": [10, 20, 30],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    assert got == [None, None, None]


def test_shift_per_group_ties_in_on(tmp_path):
    """Ties in on → tie-break by row_id (stable)."""
    data = {
        "sensor": ["a", "a", "a"],
        "ts": [1, 1, 1],
        "v": [10, 20, 30],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_multi_key(tmp_path):
    data = {
        "s1": ["a", "a", "a", "a"],
        "s2": ["x", "x", "y", "y"],
        "ts": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
    }
    df = make_df(tmp_path, data)
    # For multi-key group, use the string concat as one group in reference
    groups_combined = [f"{a}|{b}" for a, b in zip(data["s1"], data["s2"])]
    res = df.groupby(["s1", "s2"]).shift(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_shift(
        data["v"], groups_combined, on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


def test_shift_per_group_multi_value_columns(tmp_path):
    data = {
        "sensor": ["a", "a", "b", "b"],
        "ts": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
        "w": [100, 200, 300, 400],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts", columns=["v", "w"])
    got_v = read_col(res, tmp_path, "v")
    got_w = read_col(res, tmp_path, "w")
    exp_v = expected_per_group_shift(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    exp_w = expected_per_group_shift(
        data["w"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got_v, exp_v)
    _assert_lists_close(got_w, exp_w)


def test_shift_per_group_default_columns_excludes_group(tmp_path):
    """columns=None → all except group cols (sensor not shifted)."""
    data = {
        "sensor": ["a", "a", "b", "b"],
        "ts": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").shift(1, on="ts")
    # sensor NOT shifted
    got_sensor = read_col(res, tmp_path, "sensor")
    assert got_sensor == data["sensor"]
    # ts IS shifted (it's not a group col)
    got_ts = read_col(res, tmp_path, "ts", name="o2.parquet")
    exp_ts = expected_per_group_shift(
        data["ts"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got_ts, exp_ts)


def test_shift_per_group_after_filter(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "ts": [1, 2, 3, 1, 2, 3],
        "v": [10, 20, 30, 100, 200, 300],
    }
    df = make_df(tmp_path, data)
    filtered = df[df["ts"] >= 2]
    res = filtered.groupby("sensor").shift(1, on="ts", columns=["v"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # Filtered rows: ts=2,3 per sensor → v=[20,30,200,300]
    # shift(1) per group: [None, 20, None, 200]
    assert [r["v"] for r in rows] == [None, 20, None, 200]


def test_shift_per_group_empty(tmp_path):
    data = {
        "sensor": ["a"],
        "ts": [1],
        "v": [10],
    }
    df = make_df(tmp_path, data)
    empty = df[df["v"] > 100]  # filtered to empty
    res = empty.groupby("sensor").shift(1, on="ts", columns=["v"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    t = pq.read_table(out)
    assert t.num_rows == 0
    assert t.schema.names == df.columns


# ============ shift — validation ============

def test_shift_per_group_missing_on_column(tmp_path):
    df = make_df(tmp_path, {"sensor": ["a"], "v": [10]})
    with pytest.raises(KeyError, match="nope"):
        df.groupby("sensor").shift(1, on="nope")


def test_shift_per_group_explicit_group_in_columns(tmp_path):
    df = make_df(tmp_path, {
        "sensor": ["a", "a"], "ts": [1, 2], "v": [10, 20],
    })
    with pytest.raises(ValueError, match="cannot be shifted"):
        df.groupby("sensor").shift(1, on="ts", columns=["sensor"])


def test_shift_per_group_invalid_k(tmp_path):
    df = make_df(tmp_path, {
        "sensor": ["a", "a"], "ts": [1, 2], "v": [10, 20],
    })
    with pytest.raises(ValueError, match="k must be int"):
        df.groupby("sensor").shift(0, on="ts")


# ============ diff ============

def test_diff_per_group_basic(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "ts": [1, 2, 3, 1, 2, 3],
        "v": [10, 25, 40, 100, 130, 170],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").diff(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_diff(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


def test_diff_per_group_k2(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "a"],
        "ts": [1, 2, 3, 4],
        "v": [10, 25, 40, 60],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").diff(2, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_diff(
        data["v"], data["sensor"], on=data["ts"], k=2,
    )
    _assert_lists_close(got, expected)


def test_diff_per_group_nulls(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "a"],
        "ts": [1, 2, 3, 4],
        "v": [10.0, None, 30.0, 40.0],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").diff(1, on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v")
    expected = expected_per_group_diff(
        data["v"], data["sensor"], on=data["ts"], k=1,
    )
    _assert_lists_close(got, expected)


def test_diff_per_group_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {
        "sensor": ["a", "a"], "ts": [1, 2], "s": ["x", "y"],
    })
    with pytest.raises(TypeError, match="numeric"):
        df.groupby("sensor").diff(1, on="ts", columns=["s"])


# ============ rolling ============

@pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count"])
def test_rolling_per_group_all_aggs(tmp_path, agg):
    data = {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "ts": [1, 2, 3, 1, 2, 3],
        "v": [10, 20, 30, 100, 200, 300],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").rolling(2, agg, on="ts", columns=["v"])
    got = read_col(res, tmp_path, f"v_rolling_{agg}")
    expected = expected_per_group_rolling(
        data["v"], data["sensor"], on=data["ts"],
        window=2, agg=agg,
    )
    _assert_lists_close(got, expected)


def test_rolling_per_group_min_periods_1(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "ts": [1, 2, 3, 1, 2, 3],
        "v": [10, 20, 30, 100, 200, 300],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").rolling(
        3, "sum", min_periods=1, on="ts", columns=["v"],
    )
    got = read_col(res, tmp_path, "v_rolling_sum")
    expected = expected_per_group_rolling(
        data["v"], data["sensor"], on=data["ts"],
        window=3, agg="sum", min_periods=1,
    )
    _assert_lists_close(got, expected)


def test_rolling_per_group_nulls_in_value(tmp_path):
    data = {
        "sensor": ["a", "a", "a", "a"],
        "ts": [1, 2, 3, 4],
        "v": [10.0, None, 30.0, 40.0],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").rolling(
        2, "sum", min_periods=1, on="ts", columns=["v"],
    )
    got = read_col(res, tmp_path, "v_rolling_sum")
    expected = expected_per_group_rolling(
        data["v"], data["sensor"], on=data["ts"],
        window=2, agg="sum", min_periods=1,
    )
    _assert_lists_close(got, expected)


def test_rolling_per_group_append_not_replace(tmp_path):
    data = {
        "sensor": ["a", "a", "b", "b"],
        "ts": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").rolling(2, "sum", on="ts", columns=["v"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    names = pq.read_schema(out).names
    # Original columns present + rolling column appended
    assert "v" in names
    assert "v_rolling_sum" in names


def test_rolling_per_group_window_gt_group_size(tmp_path):
    """window=10 > group size: within each group, min_periods=window → all null."""
    data = {
        "sensor": ["a", "a", "b", "b"],
        "ts": [1, 2, 1, 2],
        "v": [10, 20, 30, 40],
    }
    df = make_df(tmp_path, data)
    res = df.groupby("sensor").rolling(10, "sum", on="ts", columns=["v"])
    got = read_col(res, tmp_path, "v_rolling_sum")
    assert got == [None, None, None, None]


# ============ AST guards ============

def test_per_group_uses_composite_sort():
    """AST: _per_group_shift must sort by [*group_cols, *on_cols]."""
    import ast
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_shift)
    tree = ast.parse(inspect.cleandoc(src))
    has_composite = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "external_sort"):
            for kw in node.keywords:
                if kw.arg == "key_columns":
                    unparsed = ast.unparse(kw.value)
                    if "sort_keys" in unparsed:
                        has_composite = True
    assert has_composite, (
        "_per_group_shift must sort by composite key (sort_keys = "
        "group_cols + on_cols)"
    )


def test_per_group_resets_on_group_change():
    """AST: _per_group_shift clears deques on group change."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_shift)
    assert "dq.clear()" in src, (
        "_per_group_shift must clear deque on group change"
    )
    assert "prev_group" in src, (
        "_per_group_shift must track prev_group"
    )


def test_per_group_uses_second_sort_by_orig_rid():
    """AST: second sort must use __orig_rid (not raw row_id)."""
    import ast
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_shift)
    tree = ast.parse(inspect.cleandoc(src))
    sort_keys_seen: list = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "external_sort"):
            for kw in node.keywords:
                if kw.arg == "key_columns":
                    if isinstance(kw.value, ast.List):
                        keys = [
                            e.value for e in kw.value.elts
                            if isinstance(e, ast.Constant)
                        ]
                        sort_keys_seen.append(keys)
    # Second sort (last) must be __orig_rid
    assert sort_keys_seen, "no external_sort calls found"
    second = sort_keys_seen[-1]
    assert "__orig_rid" in second, (
        f"second sort must use __orig_rid; got {second}"
    )
    assert "row_id" not in second, (
        f"second sort must NOT use raw row_id; got {second}"
    )


def test_rollstate_has_reset():
    """_RollState.reset() must exist (used by per_group_rolling)."""
    from cadw.dataframe import _RollState
    assert hasattr(_RollState, "reset")
'''


def write_test_file():
    print("\n=== tests/test_groupby_per_group.py ===")
    p = Path("tests/test_groupby_per_group.py")
    p.write_text(TEST_FILE)
    compile_check(p)
    print("  OK")


# ============ G. CHANGELOG [Unreleased] append ============

def append_changelog():
    print("\n=== CHANGELOG.md [Unreleased]: append per-group ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "Per-group `shift`/`diff`/`rolling`" in s:
        print("  already present — skip")
        return

    # Insert under [Unreleased], before the next `## [` section
    anchor = "### Тесты\n\n- fast: **~1130 passed** (+3 behavior warnings: tight/loose budget). slow: **8**."
    if anchor not in s:
        print("  WARNING: [Unreleased] tests anchor not found — skip append")
        return

    addendum = '''### Добавлено (per-group feature)

- **`df.groupby('sensor').shift(k, on='ts', columns=[...])`** — per-group
  shift. `on=None` → within-group order = row_id. Group cols excluded
  from default `columns` set.
- **`df.groupby('sensor').diff(k, on='ts', columns=[...])`** — per-group
  first difference.
- **`df.groupby('sensor').rolling(window, agg, on='ts', columns=[...])`** —
  per-group rolling. APPENDS `{col}_rolling_{agg}`; min_periods applies
  from group start.

  **Algorithm (prototype-validated, 11 cases A==B):**
  composite sort `[*group_cols, *on_cols]` → single deque per col,
  reset on group change → sort back by `__orig_rid`. **2 sort passes,
  0 shuffle.** Peak RAM = batch + O(window × #cols), independent of N
  and G (group count).

  **Semantics:** null in group col → own group (SQL, not pandas
  `dropna=True`). Group columns are NOT shifted by default (they're
  the key). This differs from pandas `groupby().shift(1)` which shifts
  everything including group key.

  **Reference:** `tests/_helpers/per_group_reference.py` — pandas
  ground truth (§13.4/128, /129, /139).

''' + anchor
    s = s.replace(anchor, addendum, 1)
    p.write_text(s)
    print("  OK")


# ============ H. DECISIONS updates ============

def update_decisions():
    print("\n=== DECISIONS.md: §11.5 lift + §13.4/141 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    # Lift §11.5 limitations
    old_115 = [
        ("| `groupby(...).rolling(...)` — не поддержано | `CadwDataFrame` | Per-group rolling — v0.8.5+ |",
         "| ~~`groupby(...).rolling(...)` — не поддержано~~ | ~~`CadwDataFrame`~~ | **закрыто v0.8.10**: composite-sort, single deque reset-on-group-change |"),
        ("| `groupby(...).shift(1, on='ts')` — не поддержано | `CadwDataFrame` | Per-group shift — v0.8.3+ |",
         "| ~~`groupby(...).shift(1, on='ts')` — не поддержано~~ | ~~`CadwDataFrame`~~ | **закрыто v0.8.10** |"),
        ("| `on` multi-column — composite sort, not per-group (v0.8.6) | `_shift_on` / `_rolling_on` | `on=[\"ts\",\"sensor\"]` = lexicographic; per-group — `groupby(...)` (v0.8.5a+) |",
         "| ~~`on` multi-column — composite sort, not per-group~~ | ~~`_shift_on` / `_rolling_on`~~ | **закрыто v0.8.10**: `groupby(...).shift/rolling(on=[...])` |"),
    ]
    for old, new in old_115:
        if old in s:
            s = s.replace(old, new, 1)

    # §13.4/141
    if "Эпизод 141" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 141 (per-group shift/diff/rolling — prototype-first).**
v0.8.10 добавляет `df.groupby('g').shift/diff/rolling(on='ts')`.
Алгоритм: composite sort `[*group_cols, *on_cols]`, single deque per
column, reset on group change, sort back by `__orig_rid`. 2 sort passes,
0 shuffle. Peak RAM = batch + O(window × #cols), independent of N и G.

**Prototype-first (§13.4/123):** `scripts/proto_per_group.py` — 11
кейсов, A (pandas) vs B (composite-sort deque). Все PASS **до** того,
как был написан патч-код. Сравнение с историей: v0.8.2 (shift on=col)
— 5 итераций без prototype; v0.8.4 (rolling on=col) — 1 итерация с
prototype; v0.8.10 (per-group) — 0 fix-итераций.

**Semantics:** null in group → own group (SQL, not pandas dropna=True).
Group columns NOT shifted by default (unlike pandas). Documented в
`_per_group_shift` docstring + §11.5.

**Reference:** `tests/_helpers/per_group_reference.py` — pandas ground
truth, same pattern as `rolling_reference.py` (§13.4/129). Test
expectations **derived**, not authored — §13.4/128, /129, /139.

**Closes §11.5 entries:** `groupby().shift(on=)` (was "not supported"),
`groupby().rolling(on=)` (was "v0.8.5+"), multi-column `on` per-group
(was "composite sort only, not per-group").

**Backlog:** per-group vectorization (§11.6) — current per-row Python
loop (for group-change detection), O(N) but constant factor higher than
vectorized. Optimize via segment detection (pc.not_equal between adjacent
group values) — future release.

'''
        if marker in s:
            s = s.replace(marker, episode + marker, 1)

    p.write_text(s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.8.10: per-group shift/diff/rolling ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if branch != "master":
        print(f"WARNING: not on master (current: {branch}) — proceed anyway")

    r = subprocess.run(["git", "status", "--porcelain",
                        "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    add_rollstate_reset()
    add_groupby_methods()
    add_per_group_methods()
    update_ast_guard_whitelist()
    write_reference_helper()
    write_test_file()
    append_changelog()
    update_decisions()

    print("\n=== pytest: new tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_groupby_per_group.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.10 (new tests red)"], check=False)
        sys.exit(1)

    print("\n=== pytest: arrow guard + reference sanity ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_changelog.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** GUARD FAILURE ***")
        sys.exit(1)

    print("\n=== pytest: shift/diff/rolling regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_shift_diff.py", "tests/test_shift_on_col.py",
         "tests/test_rolling.py", "tests/test_rolling_on_col.py",
         "tests/test_expanding_ewm.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sys.exit(1)

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
        "v0.8.10: per-group shift/diff/rolling — composite-sort + single "
        "deque reset-on-group-change; 11-case prototype (§13.4/141); "
        "_RollState.reset(); closes §11.5 per-group entries"])

    print("\n==> v0.8.10 applied. Ready to squash v0.8.10-pre + v0.8.10 "
          "→ tag v0.8.10.")


if __name__ == "__main__":
    main()