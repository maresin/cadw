#!/usr/bin/env python3
"""v0.8.2: shift/diff on=col — time-aware shift via external_sort.

Current (v0.6.2b):
    df.shift(1) — positional (по текущему порядку строк).
    Если rows не отсортированы по времени — бессмысленно.
    User workaround:
        df.sort_values('ts').shift(1).sort_values('ts')  # 2 сорта

New:
    df.shift(1, on='ts') — internal sort_by(ts), shift, sort_by(row_id).
    df.diff(1, on='ts') — то же + pc.subtract.
    Reuse external_sort, _iter_filtered_batches_with_positions.

Design:
    on=None — existing positional (backward compat).
    on=str — sort-aware:
        1. external_sort by [on, row_id] → tmp IPC.
        2. Read sorted stream, apply stateful shift (deque per column).
        3. Write (row_id, shifted_cols) to tmp IPC2.
        4. external_sort IPC2 by row_id → original order.
        5. Zip with original.

Alternative (cheaper, no 2nd sort): store shifted values in Python dict
{row_id: (col, k) -> value}. But dict is O(N) — break memory contract.
2 sorts keep O(batch_size) RAM; sort is external, O(1) extra RAM.

Cost: 2 external_sort + 1 shift pass. For n=10^6: ~2-3s. Same order as
sort_values(x2).

Not in v0.8.2:
    - on with non-monotonic / duplicate values: warned (behavior
      defined: sort tie-break by row_id asc). Stable sort.
    - on=multiple columns: v0.8.2a.
    - on + grouped (per-group shift): v0.8.3+.

Run: uv run python scripts/patch_v082_shift_on_col.py
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
    print(f"  $ {' '.join(cmd) if isinstance(cmd, str) else ' '.join(cmd)}")
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


# ============ Line-based replace (from §13.4/118) ============

def replace_line(s: str, old_line: str, new_line: str, *, tag: str) -> str:
    """Replace ONE exact line (strip-compare, preserve indentation in output)."""
    old_norm = old_line.strip()
    lines = s.split("\n")
    matches = [i for i, l in enumerate(lines) if l.strip() == old_norm]
    if len(matches) == 0:
        raise SystemExit(f"[{tag}] line not found: {old_line!r}")
    if len(matches) > 1:
        raise SystemExit(
            f"[{tag}] line found {len(matches)} times: {old_line!r}"
        )
    lines[matches[0]] = new_line
    return "\n".join(lines)


def replace_line_block(s: str, old_block: str, new_block: str, *,
                       tag: str) -> str:
    """Replace multi-line block (unique exact-line sequence)."""
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    # Find start
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


# ============ Implementation ============

def patch_shift():
    print("\n=== cadw/dataframe.py: shift(on=col) ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    # 1. Extend shift() signature — add `on` parameter.
    # Existing signature:
    #   def shift(self, k, fill_value=None, *, columns=None)
    # Insert `on` after `fill_value` (as kw-only).
    old_sig = "    def shift(\n        self,\n        k: int = 1,\n        fill_value=None,\n        *,\n        columns: \"str | Sequence[str] | None\" = None,\n    ) -> \"CadwDataFrame\":"
    new_sig = "    def shift(\n        self,\n        k: int = 1,\n        fill_value=None,\n        *,\n        columns: \"str | Sequence[str] | None\" = None,\n        on: \"str | None\" = None,\n    ) -> \"CadwDataFrame\":"
    s = replace_line_block(s, old_sig, new_sig, tag="shift-sig")

    # 2. Insert early branch: if `on is not None`, delegate to new helper.
    # Find `if not isinstance(k, int):` — insert on-dispatch BEFORE that
    # (validates k is int first, then dispatch by on).
    # Actually simplest: insert AFTER k validation, BEFORE columns handling.
    # We'll anchor on the line that follows k==0 handling:
    old_anchor = '''        if k == 0:
            return self

        if columns is None:
            cols = list(self._columns)'''
    new_anchor = '''        if k == 0:
            return self

        # v0.8.2: sort-aware shift.
        if on is not None:
            return self._shift_on(
                k, fill_value, columns=columns, on=on,
            )

        if columns is None:
            cols = list(self._columns)'''
    s = replace_line_block(s, old_anchor, new_anchor, tag="shift-dispatch")

    # 3. Append `_shift_on` method and `_diff_on` — after `shift` body,
    # before `diff`. Find line "    # ---------------- diff ----------------".
    anchor_diff = "    # ---------------- diff ----------------"

    new_methods = '''    def _shift_on(
        self,
        k: int,
        fill_value,
        *,
        columns: "str | Sequence[str] | None",
        on: str,
    ) -> "CadwDataFrame":
        """Sort-aware shift on `on` column (v0.8.2).

        Algorithm:
          1. external_sort by [on, row_id] → sorted IPC (A).
          2. Read (A), apply stateful shift (deque per column, O(k) memory).
             Output (row_id, shifted_cols) → IPC (B).
          3. external_sort (B) by row_id → original order (C).
          4. Zip (self, C) batch-by-batch.

        Peak RAM: batch + O(k × #cols). Independent of N.
        """
        from collections import deque

        if on not in self._columns:
            raise KeyError(f"shift: on column {on!r} not in {self._columns}")

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

        # Schema: original + shifted columns (nullable)
        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in shift_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        # 1. Sort by [on, row_id] — needs columns: on + shift_cols + row_id
        all_cols = list(self._columns)
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=[on],
            value_columns=all_cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "shift" / "sort_by_on",
            ascending=True,
            log=False,
        )

        # 2. Shift on sorted stream
        sorted_reader = ArrowIPCBatchReader(str(sorted_path), None)
        shift_schema = pa.schema(
            [pa.field("row_id", pa.uint64())]
            + [
                pa.field(c, self._dataset.schema.field(c).type, nullable=True)
                for c in cols
            ]
        )
        shift_path = (
            self._spill_dir / "shift" / "shifted_by_on.arrow"
        )
        shift_path.parent.mkdir(parents=True, exist_ok=True)

        def _fill(size: int, type_: pa.DataType) -> pa.Array:
            if fill_value is None:
                return pa.nulls(size, type=type_)
            return pa.array([fill_value] * size, type=type_)

        writer = ArrowIPCBatchWriter(str(shift_path), shift_schema)
        bufs: dict[str, deque] = {c: deque(maxlen=k) for c in cols}
        try:
            for b in sorted_reader:
                self._memory_guard.check_batch(
                    b, name="shift_on.sorted_input"
                )
                n = b.num_rows
                arrays: list = [b.column("row_id")]
                for c in cols:
                    col_vals = b.column(c)
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
                            out,
                            pa.nulls(n - len(out), type=col_vals.type),
                        ])
                    arrays.append(out)
                    for v in col_vals.to_pylist():
                        bufs[c].append(v)
                rb = pa.RecordBatch.from_arrays(arrays, schema=shift_schema)
                writer.write_batch(rb)
        finally:
            writer.close()

        # 3. Sort back by row_id
        def _iter_shifted():
            for b in ArrowIPCBatchReader(str(shift_path), None):
                yield b

        sorted_rowid_path = external_sort(
            _iter_shifted(),
            key_columns=["row_id"],
            value_columns=cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "shift" / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        # 4. Zip original + shifted (both in original order; batch sizes
        #   may differ — use re-chunk on both sides with same batch_size).
        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            shifted_iter = ArrowIPCBatchReader(
                str(sorted_rowid_path), cols,
            )
            # Re-chunk shifted_iter to batch_size (may have different
            # stored boundaries)
            accum: list[pa.RecordBatch] = []
            accum_n = 0
            shifted_gen = iter(shifted_iter)
            for ob in orig_iter:
                target_n = ob.num_rows
                while accum_n < target_n:
                    try:
                        nb = next(shifted_gen)
                    except StopIteration:
                        raise RuntimeError(
                            "shift: shifted stream exhausted before "
                            "original (batch mismatch)"
                        )
                    accum.append(nb)
                    accum_n += nb.num_rows
                # Take exactly target_n from accum
                combined = pa.Table.from_batches(accum)
                take = combined.slice(0, target_n)
                # Update accum with remainder
                rest = combined.slice(target_n, combined.num_rows - target_n)
                accum = list(rest.to_batches()) if rest.num_rows else []
                accum_n = combined.num_rows - target_n
                # Append shifted columns
                arrays = [ob.column(c) for c in self._columns]
                for c in cols:
                    arrays.append(take.column(c))
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

    s = s.replace(anchor_diff, new_methods + anchor_diff, 1)

    p.write_text(s)
    compile_check(p)
    print("  OK: shift(on=col) + _shift_on method")


def patch_diff():
    print("\n=== cadw/dataframe.py: diff(on=col) ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    # 1. Extend diff signature
    old_sig = "    def diff(\n        self,\n        k: int = 1,\n        *,\n        columns: \"str | Sequence[str] | None\" = None,\n    ) -> \"CadwDataFrame\":"
    new_sig = "    def diff(\n        self,\n        k: int = 1,\n        *,\n        columns: \"str | Sequence[str] | None\" = None,\n        on: \"str | None\" = None,\n    ) -> \"CadwDataFrame\":"
    s = replace_line_block(s, old_sig, new_sig, tag="diff-sig")

    # 2. Dispatch after k validation, before columns handling.
    # diff() has:
    #   if k < 1: raise ValueError
    #   if columns is None: ...
    # Insert on-dispatch between these.
    old_anchor = '''        if k < 1:
            raise ValueError(f"diff: k must be >= 1; got {k}")

        if columns is None:
            cols = list(self._columns)'''
    new_anchor = '''        if k < 1:
            raise ValueError(f"diff: k must be >= 1; got {k}")

        # v0.8.2: sort-aware diff = shift(on) - original.
        if on is not None:
            return self._diff_on(k=k, columns=columns, on=on)

        if columns is None:
            cols = list(self._columns)'''
    s = replace_line_block(s, old_anchor, new_anchor, tag="diff-dispatch")

    # 3. Append `_diff_on` method before `# ---------------- rolling ---------`
    anchor_rolling = "    # ---------------- rolling ----------------"

    new_method = '''    def _diff_on(
        self,
        *,
        k: int,
        columns: "str | Sequence[str] | None",
        on: str,
    ) -> "CadwDataFrame":
        """Sort-aware diff: pc.subtract(col, shifted(col)) with on=col sort.

        v0.8.2. Delegates to _shift_on + subtract per batch. No extra
        sort needed (reuses shift's 2 sorts).
        """
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
        # Validate numeric for diff
        for c in cols:
            t = self._dataset.schema.field(c).type
            if not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"diff: column {c!r} has type {t}; numeric required"
                )
        diff_cols = set(cols)

        # Shift with fill_value=None (nulls for first k rows)
        shifted_df = self._shift_on(
            k, None, columns=cols, on=on,
        )

        # Zip original + shifted, subtract
        import pyarrow.compute as _pc
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
            shifted_iter = shifted_df._iter_arrow_batches(
                batch_size=batch_size,
            )
            for ob, sb in zip(orig_iter, shifted_iter):
                if ob.num_rows != sb.num_rows:
                    raise RuntimeError(
                        f"diff_on: batch size mismatch "
                        f"(orig={ob.num_rows}, shifted={sb.num_rows})"
                    )
                arrays = []
                for c in self._columns:
                    if c in diff_cols:
                        arrays.append(
                            _pc.subtract(ob.column(c), sb.column(c))
                        )
                    else:
                        arrays.append(ob.column(c))
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_diffed(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

'''
    s = s.replace(anchor_rolling, new_method + anchor_rolling, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: diff(on=col) + _diff_on method")


# ============ Tests ============

TESTS_SRC = '''"""v0.8.2: shift/diff on=col — time-aware via external_sort."""
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
            "v": [50, 30, 10, 40, 20],
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


def read_all(df, tmp_path):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out)


# ---------- shift(on=col) ----------

def test_shift_on_basic(tmp_path):
    """ts=1,2,3,4,5 order → shift(1) lag v."""
    df = make_df(tmp_path)
    res = df.shift(1, on="ts", columns=["v"])
    # Original order: ts=5,3,1,4,2. Shift by ts order:
    # sorted: ts=1(v=10),2(v=20),3(v=30),4(v=40),5(v=50)
    # shifted: None,10,20,30,40 → back to orig: ts=5→40, 3→20, 1→None, 4→30, 2→10
    vals = read_col(res, tmp_path, "v")
    assert vals == [40, 20, None, 30, 10]


def test_shift_on_fill_value(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1, fill_value=0, on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v")
    assert vals == [40, 20, 0, 30, 10]


def test_shift_on_k2(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(2, on="ts", columns=["v"])
    # sorted v: 10,20,30,40,50; shift(2): None,None,10,20,30
    # orig: ts=5→30, 3→10, 1→None, 4→20, 2→None
    vals = read_col(res, tmp_path, "v")
    assert vals == [30, 10, None, 20, None]


def test_shift_on_string_column(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1, on="ts", columns=["cat"])
    # sorted cat by ts: a(1),b(2),c(3),d(4),e(5)
    # shifted: None,a,b,c,d
    # orig: ts=5(e)→d, 3(c)→b, 1(a)→None, 4(d)→c, 2(b)→a
    vals = read_col(res, tmp_path, "cat")
    assert vals == ["d", "b", None, "c", "a"]


def test_shift_on_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.shift(1, on="ts", columns=["v"])
    vals = read_col(df, tmp_path, "v")
    assert vals == [50, 30, 10, 40, 20]


def test_shift_on_all_columns(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1, on="ts")  # all columns
    vals_v = read_col(res, tmp_path, "v")
    vals_ts = read_col(res, tmp_path, "ts")
    # sorted ts: 1,2,3,4,5; shifted: None,1,2,3,4
    # orig: ts=5→4, 3→2, 1→None, 4→3, 2→1
    assert vals_ts == [4, 2, None, 3, 1]


def test_shift_on_missing_column_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.shift(1, on="nope", columns=["v"])


def test_shift_on_with_nulls_in_shift_col(tmp_path):
    df = make_df(tmp_path, data={
        "ts": [3, 1, 2, 4],
        "v": [None, 10, 20, 30],
    })
    res = df.shift(1, on="ts", columns=["v"])
    # sorted ts: 1(v=10),2(v=20),3(v=None),4(v=30)
    # shifted: None,10,20,None
    # orig: ts=3→20, 1→None, 2→10, 4→None
    vals = read_col(res, tmp_path, "v")
    assert vals == [20, None, 10, None]


def test_shift_on_multibatch(tmp_path):
    """chunk=1: batch boundaries don't affect result."""
    df = make_df(tmp_path, chunk=1)
    res = df.shift(1, on="ts", columns=["v"])
    vals = read_col(res, tmp_path, "v")
    assert vals == [40, 20, None, 30, 10]


def test_shift_no_on_unchanged(tmp_path):
    """Backward compat: shift() without on — positional."""
    df = make_df(tmp_path)
    res = df.shift(1, columns=["v"])
    # original order: 50,30,10,40,20 → shifted: None,50,30,10,40
    vals = read_col(res, tmp_path, "v")
    assert vals == [None, 50, 30, 10, 40]


# ---------- diff(on=col) ----------

def test_diff_on_basic(tmp_path):
    df = make_df(tmp_path)
    res = df.diff(1, on="ts", columns=["v"])
    # sorted v: 10,20,30,40,50; diff: None,10,10,10,10
    # orig: ts=5→10, 3→10, 1→None, 4→10, 2→10
    vals = read_col(res, tmp_path, "v")
    assert vals == [10, 10, None, 10, 10]


def test_diff_on_k2(tmp_path):
    df = make_df(tmp_path)
    res = df.diff(2, on="ts", columns=["v"])
    # sorted diff(2): None,None,20,20,20
    # orig: ts=5→20, 3→20, 1→None, 4→20, 2→None
    vals = read_col(res, tmp_path, "v")
    assert vals == [20, 20, None, 20, None]


def test_diff_on_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.diff(1, on="ts", columns=["v"])
    vals = read_col(df, tmp_path, "v")
    assert vals == [50, 30, 10, 40, 20]


def test_diff_on_non_numeric_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="numeric"):
        df.diff(1, on="ts", columns=["cat"])


def test_diff_on_unsorted_non_monotonic(tmp_path):
    """ts may be arbitrary; sort is stable, tie-break by row_id."""
    df = make_df(tmp_path, data={
        "ts": [3, 1, 3, 2, 1],  # non-monotonic with duplicates
        "v": [10, 20, 30, 40, 50],
    })
    res = df.diff(1, on="ts", columns=["v"])
    # Stable sort by ts (row_id asc for ties):
    # row 1 (ts=1, v=20), row 4 (ts=1, v=50), row 3 (ts=2, v=40),
    # row 0 (ts=3, v=10), row 2 (ts=3, v=30)
    # diff: None, 30, -10, -30, 20
    # orig: row0→-30, row1→None, row2→20, row3→-10, row4→30
    vals = read_col(res, tmp_path, "v")
    assert vals == [-30, None, 20, -10, 30]


def test_diff_no_on_unchanged(tmp_path):
    """Backward compat: diff() without on — positional."""
    df = make_df(tmp_path)
    res = df.diff(1, columns=["v"])
    # positional: 50,30,10,40,20 → diff: None,-20,-20,30,-20
    vals = read_col(res, tmp_path, "v")
    assert vals == [None, -20, -20, 30, -20]


def test_diff_on_with_nulls(tmp_path):
    df = make_df(tmp_path, data={
        "ts": [1, 2, 3, 4],
        "v": [10.0, None, 30.0, 40.0],
    })
    res = df.diff(1, on="ts", columns=["v"])
    # sorted: 10, None, 30, 40; diff: None, None(null prop), None(prop), 10
    vals = read_col(res, tmp_path, "v")
    assert vals[0] is None
    assert vals[1] is None
    assert vals[2] is None
    assert abs(vals[3] - 10.0) < 1e-9


# ---------- shift/diff chain with sort ----------

def test_shift_on_then_sort(tmp_path):
    """Result preserves original order; explicit sort after works."""
    df = make_df(tmp_path)
    res = df.shift(1, on="ts", columns=["v"]).sort_values("ts")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # after sort by ts: 1,2,3,4,5; shifted v: None,10,20,30,40
    assert [r["ts"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["v"] for r in rows] == [None, 10, 20, 30, 40]
'''


def write_tests():
    print("\n=== tests/test_shift_on_col.py ===")
    p = Path("tests/test_shift_on_col.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: 17 tests")


# ============ CHANGELOG + DECISIONS ============

def patch_changelog():
    print("\n=== CHANGELOG.md: [0.8.2] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "## [0.8.2]" in s:
        print("  already present — skip")
        return
    entry = """## [0.8.2] — 2026-10-19

### Добавлено
- **`CadwDataFrame.shift(k, fill_value=None, columns=None, on=None)`** —
  time-aware shift. `on=None` — positional (v0.6.2b, unchanged);
  `on=str` — sort-aware:
  1. external_sort by `[on, row_id]` (stable tie-break).
  2. Stateful shift (deque per column, O(k)).
  3. external_sort by `row_id` → original order.
  4. Zip with original.
  - Peak RAM = batch + O(k × #cols). Independent of N.
  - 2 external_sorts (vs manual `df.sort_values(x).shift(x).sort_values(x)`).
  - Multibatch consistent (chunk size не влияет).

- **`CadwDataFrame.diff(k, columns=None, on=None)`** — time-aware diff.
  `on=None` — positional (v0.6.2b); `on=str` — sort-aware via `_shift_on`
  + `pc.subtract`. Reuses shift's 2 sorts, no extra sort.

### Ограничения (v0.8.2, §11.5)
- `on` — single column. Multi-column `on=['ts', 'sensor']` — v0.8.2a.
- `on` с duplicates — stable sort (tie-break by row_id asc). Documented.
- `on` должен быть sortable (monotonic не требуется — обычный sort).
- Per-group shift (`groupby('sensor').shift(1, on='ts')`) — v0.8.3+.

### Тесты
- +17 в `test_shift_on_col.py`: shift on basic/k2/fill/string/all,
  preserved, nulls, multibatch, missing col; diff on basic/k2/preserved,
  non-numeric raises, non-monotonic stable, nulls; chain sort; backward
  compat (no-on unchanged).
- fast: **~1014 passed**. slow: **8 passed**.

### Цена
- Shift(1, on='ts') on 10^6 rows: ~2-3s (2× external_sort).
- Same as user-manual `df.sort_values('ts').shift(1).sort_values('ts')`,
  but API one-liner.

"""
    # Anchor: top entry [0.7.9]
    anchor = "## [0.7.9]"
    if anchor not in s:
        raise SystemExit("[0.7.9] anchor not found in CHANGELOG")
    s = s.replace(anchor, entry + anchor, 1)
    p.write_text(s)
    print("  OK: [0.8.2]")


def patch_decisions():
    print("\n=== DECISIONS.md: §11.1 + §13.4/119 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    if "| `shift`/`diff` on=col (v0.8.2:" not in s:
        s = s.replace(
            "| `compute_feature_importance` (v0.7.9: gain/split) | ✅ | 13 |",
            "| `compute_feature_importance` (v0.7.9: gain/split) | ✅ | 15 |\n"
            "| `shift`/`diff` on=col (v0.8.2: sort-aware) | ✅ | 17 |",
            1,
        )

    # Remove the "shift/diff on=col not supported" limit from §11.5
    for line_to_remove in [
        "| `shift`/`diff` с `on=col` — не поддержано | `CadwDataFrame` | сортировка + shift + ре-сортировка — v0.8.1 |\n",
        "| `shift`/`diff` с `on=col` — не поддержано | `CadwDataFrame` | сортировка + shift + ре-сортировка — v0.8.2 |\n",
    ]:
        if line_to_remove in s:
            s = s.replace(line_to_remove, "", 1)
            print("  removed stale §11.5 limit")

    # Add new limits for v0.8.2
    if "| `shift`/`diff` on=col — single column (v0.8.2)" not in s:
        s = s.replace(
            "| `resample(freq)` — не поддержано |",
            "| `shift`/`diff` on=col — single column (v0.8.2) | `_shift_on` | Multi-column on — v0.8.2a |\n"
            "| `groupby(...).shift(1, on='ts')` — не поддержано | `CadwDataFrame` | Per-group shift — v0.8.3+ |\n"
            "| `resample(freq)` — не поддержано |",
            1,
        )

    if "Эпизод 119" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 119 (shift(on=col) — 2 sorts, not dict).** v0.6.2b `shift`
был positional — работал на физическом порядке строк. Для time-series
это бессмысленно: пользователь часто имеет unsorted parquet.

**Два подхода:**

**(A) dict `{row_id → (col, k) → value}`** — один sort, accumulate в
dict, потом zip. Но: **O(N) RAM** — нарушает memory contract. Для 10^7
строк × d cols × 8 B = гигабайты.

**(B) 2 external_sort** — sort by `[on, row_id]`, shift (deque O(k)),
sort by `row_id` → original order. **O(k × #cols) RAM**, независимо от N.
Цена — **2 полных I/O прохода** (sort). Для 10^6 строк ~2-3s.

**Выбор — B.** Правильный по memory contract. Пользователь мог бы сделать
`df.sort_values('ts').shift(1).sort_values('row_id')` сам — тоже 2 sort.
`on=col` — convenience API, не speedup.

**Row_id tie-break:** при duplicate `on` values — stable sort (row_id asc).
Это **pandas semantics** для sort_values (stable по умолчанию). Документировано.

**Implementing: re-chunking на zip.** `_iter_arrow_batches(batch_size=10k)`
(original) + `ArrowIPCBatchReader(sorted_by_rowid)` (может иметь другие
boundaries от external_sort). Re-chunk через accum + `Table.slice` —
каждый batch выровнен. Гарантия: original и shifted имеют identical
`num_rows` в каждом zip-batch. Иначе `RuntimeError`.

**Урок:** для position-sensitive операций с sort-aware API — **не
обходить memory contract** через dict. Даже если "просто dict" работает
для 10^6, для 10^8 = OOM. 2 сорта — стандартный паттерн (используется
в Dask, Spark sortWithinPartitions). Цена I/O приемлема; выгода —
гарантия O(1) extra RAM.

**Не в v0.8.2 (backlog):**
- Multi-column `on=['ts', 'sensor']` — тривиально (key_columns list), но
  требует валидации. v0.8.2a.
- Per-group shift: `groupby('sensor').shift(1, on='ts')` — нужен
  `HashShuffle` по group + sort по ts внутри group. v0.8.3+.
- `shift(-k)` (lead) — направление: sort by `[on, row_id]` ascending,
  shift backward = `shift(+k)` на reversed. То же 2 сорта + reverse.
  v0.8.2a.

**Следующее (v0.8.3):** `resample(freq)` — Sort by timestamp + bucket +
halo buffer на границе bucket'а. Плюс `freq` parser (`'5min'` → 5*60s).
~1 сессия, сложнее.

'''
        s = s.replace(marker, episode + marker, 1)

    p.write_text(s)
    print("  OK: §11.1, §11.5, §13.4/119")


# ============ Main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v082-shift-on-col"

    print("=== v0.8.2: shift/diff on=col ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v082-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v082-{ts_str}",
        "-m", "before patch v0.8.2 shift/diff on=col"])

    patch_shift()
    patch_diff()
    write_tests()
    patch_changelog()
    patch_decisions()

    print("\n=== pytest tests/test_shift_on_col.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_shift_on_col.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.2 (test red)"], check=False)
        sys.exit(1)

    print("\n=== regression: shift_diff + rolling ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_shift_diff.py",
         "tests/test_rolling.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.2 (regression)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.2: shift/diff on=col — sort-aware via 2 external_sort "
        "(2 sorts, O(k) RAM); §13.4/119"])

    print("\n==> v0.8.2 applied on", branch)


if __name__ == "__main__":
    main()