#!/usr/bin/env python3
"""v0.8.2-fix4: _shift_on step 4 — replace (не append) + Array из Table.

Диагностика v0.8.2-fix3:
    ValueError: Schema and number of arrays unequal
    (в pyarrow/table.pyx: RecordBatch.from_arrays)

Причины (две):
    1. `out_schema` в _shift_on — `self._columns` (3 поля для default),
       shift **заменяет** колонки (как v0.6.2b shift), не аппендит.
       Но `_iter_zipped` делает `arrays = [ob.column(c) for c in self._columns]
       + [take.column(c) for c in cols]` = 3 + 1 = 4 → mismatch.
    2. `take.column(c)` на pa.Table возвращает ChunkedArray, а
       `pa.RecordBatch.from_arrays` требует Array.

Fix:
    - `_iter_zipped` — replace: для shifted columns берём `take_batch.column(c)`,
      для остальных — `ob.column(c)`.
    - `take.combine_chunks().to_batches()[0]` — ChunkedArray → Array.

Also:
    - AST guard `test_shift_on_iter_zipped_replace_not_append` — проверяет,
      что нет `arrays = [ob.column(c) for c in self._columns]` + append
      (pattern для правильного replace).

Run: uv run python scripts/patch_v082_fix4.py
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


def replace_line_block(s, old_block, new_block, *, tag):
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


def fix_shift_on_zip():
    print("\n=== cadw/dataframe.py: _shift_on _iter_zipped ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = '''                combined = pa.Table.from_batches(accum)
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
                )'''

    new = '''                combined = pa.Table.from_batches(accum)
                take = combined.slice(0, target_n)
                rest = combined.slice(
                    target_n, combined.num_rows - target_n,
                )
                accum = list(rest.to_batches()) if rest.num_rows else []
                accum_n = combined.num_rows - target_n
                # v0.8.2-fix4: (a) REPLACE shifted columns, not append —
                # out_schema has self._columns (shift replaces column value,
                # doesn't add new column; matches v0.6.2b semantics);
                # (b) Table.column() returns ChunkedArray; combine_chunks()
                # + to_batches()[0] → RecordBatch with Array columns.
                take_batch = take.combine_chunks().to_batches()[0]
                arrays = []
                for c in self._columns:
                    if c in shift_cols:
                        arrays.append(take_batch.column(c))
                    else:
                        arrays.append(ob.column(c))
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )'''

    s = replace_line_block(s, old, new, tag="shift-on-zip")
    p.write_text(s)
    compile_check(p)
    print("  OK: replace + combine_chunks().to_batches()[0]")


def add_ast_guard():
    print("\n=== tests/test_shift_on_col.py: +AST guard ===")
    p = Path("tests/test_shift_on_col.py")
    s = p.read_text()
    if "test_shift_on_iter_zipped_replace_not_append" in s:
        print("  already present — skip")
        return

    extra = '''

def test_shift_on_iter_zipped_replace_not_append():
    """AST guard: _shift_on `_iter_zipped` must REPLACE shifted columns,
    not append (shift replaces column values, doesn't add new column).

    Регрессия v0.8.2-fix4:
        Schema and number of arrays unequal — out_schema=3 (self._columns),
        arrays=4 (3 original + 1 shifted appended).

    Проверяем:
      - НЕТ `arrays = [ob.column(c) for c in self._columns]` (append-style).
      - ЕСТЬ `if c in shift_cols:` / `if c in cols:` (replace-style).
    """
    import ast as _ast
    import inspect as _inspect
    from cadw.dataframe import CadwDataFrame

    src = _inspect.getsource(CadwDataFrame._shift_on)
    # Find `_iter_zipped` inner function
    tree = _ast.parse(_inspect.cleandoc(src))
    iter_zipped_src = None
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.FunctionDef)
                and node.name == "_iter_zipped"):
            iter_zipped_src = _ast.get_source_segment(
                _inspect.cleandoc(src), node,
            )
            break
    assert iter_zipped_src is not None, "_iter_zipped not found in _shift_on"

    # Check for append-style pattern (bad)
    assert "arrays = [ob.column(c) for c in self._columns]" not in iter_zipped_src, (
        "_iter_zipped uses append-style (arrays = [ob.column...]) — must "
        "REPLACE shifted columns via `if c in shift_cols:`"
    )
    # Check for replace-style pattern (good)
    assert "if c in shift_cols:" in iter_zipped_src or "if c in cols:" in iter_zipped_src, (
        "_iter_zipped must use `if c in shift_cols:` (replace semantics)"
    )
    # Check ChunkedArray handling
    assert "combine_chunks()" in iter_zipped_src, (
        "_iter_zipped must use .combine_chunks().to_batches()[0] for "
        "Table→Array conversion (Table.column returns ChunkedArray)"
    )
'''

    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: AST guard (replace, not append)")


def main():
    print("=== v0.8.2-fix4 ===")
    fix_shift_on_zip()
    add_ast_guard()

    print("\n=== pytest tests/test_shift_on_col.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_shift_on_col.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.2-fix4 (test red)"], check=False)
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
        sh(["git", "commit", "-m", "WIP: v0.8.2-fix4 (regression)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.2-fix4: _shift_on _iter_zipped — replace shifted (not append); "
        "combine_chunks().to_batches()[0]; +AST guard"])

    print("\n==> v0.8.2-fix4 complete. Готов squash → tag v0.8.2.")


if __name__ == "__main__":
    main()