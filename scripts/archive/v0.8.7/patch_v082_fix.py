#!/usr/bin/env python3
"""v0.8.2-fix: вставить `_shift_on` method (missing anchor).

Причина: patch_v082 использовал `    # ---------------- diff ----------------`
как anchor — но в dataframe.py `shift` и `diff` под одним комментарием
`# ---------------- shift / diff ----------------`. str.replace — silent no-op.
`_diff_on` вставился (anchor `# ---------------- rolling ----------------`
существует), `_shift_on` — нет.

Fix: вставить `_shift_on` перед `    def _diff_on(` (он уже есть).
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


SHIFT_ON_SRC = '''    def _shift_on(
        self,
        k: int,
        fill_value,
        *,
        columns: "str | Sequence[str] | None",
        on: str,
    ) -> "CadwDataFrame":
        """Sort-aware shift on `on` column (v0.8.2).

        Algorithm:
          1. external_sort by [on, row_id] → sorted IPC.
          2. Read sorted stream, apply stateful shift (deque per column,
             O(k) memory). Output (row_id, shifted_cols) → IPC.
          3. external_sort by row_id → original order.
          4. Zip with original.

        Peak RAM: batch + O(k × #cols). Independent of N.
        """
        from collections import deque

        if on not in self._columns:
            raise KeyError(
                f"shift: on column {on!r} not in {self._columns}"
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

        # 1. Sort by [on, row_id]
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

        # 4. Zip (re-chunk to same batch_size)
        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            shifted_reader = ArrowIPCBatchReader(
                str(sorted_rowid_path), cols,
            )
            accum: list[pa.RecordBatch] = []
            accum_n = 0
            shifted_gen = iter(shifted_reader)
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
                combined = pa.Table.from_batches(accum)
                take = combined.slice(0, target_n)
                rest = combined.slice(
                    target_n, combined.num_rows - target_n,
                )
                accum = list(rest.to_batches()) if rest.num_rows else []
                accum_n = combined.num_rows - target_n
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


def insert_shift_on():
    print("\n=== cadw/dataframe.py: insert _shift_on before _diff_on ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    # Idempotent guard
    if "def _shift_on(" in s:
        print("  _shift_on already present — skip")
        return

    anchor = "    def _diff_on(\n"
    if anchor not in s:
        raise SystemExit(
            "anchor '    def _diff_on(' not found — unexpected state. "
            "Проверь вручную."
        )
    if s.count(anchor) > 1:
        raise SystemExit(f"anchor found {s.count(anchor)} times")

    s = s.replace(anchor, SHIFT_ON_SRC + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: _shift_on inserted before _diff_on")

    # Verify both methods now exist
    s2 = p.read_text()
    assert "def _shift_on(" in s2, "_shift_on missing after insert"
    assert "def _diff_on(" in s2, "_diff_on missing (unexpected)"
    print("  OK: both _shift_on and _diff_on present")


def main():
    print("=== v0.8.2-fix ===")
    insert_shift_on()

    print("\n=== pytest tests/test_shift_on_col.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_shift_on_col.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.2-fix (test red)"], check=False)
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
        sh(["git", "commit", "-m", "WIP: v0.8.2-fix (regression)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.2-fix: вставить _shift_on method (missing anchor '# diff' — "
        "в файле '# shift / diff')"])

    print("\n==> v0.8.2-fix complete. Готов squash → tag v0.8.2.")


if __name__ == "__main__":
    main()