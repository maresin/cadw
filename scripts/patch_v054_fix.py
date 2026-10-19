#!/usr/bin/env python3
"""v0.5.4-fix: _read_partition_safe падал на empty partition;
AST guard ловил list( в docstring.

Fix:
1. _read_partition_safe: ipc.open_file + num_record_batches + generator,
   пустой — Table.from_batches([], schema=...).
2. AST guard: ищем ast.Call(list) в теле функции (не substring).

Run: uv run python scripts/patch_v054_fix.py
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
        print(f"    STDERR: {r.stderr.strip()[-1000:]}")
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
        print(r.stderr[-1000:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


# ============ Fix _read_partition_safe ============

def patch_read_partition_safe():
    print("\n=== cadw/ops/groupby.py: _read_partition_safe (fix empty) ===")
    p = Path("cadw/ops/groupby.py")
    s = p.read_text()

    # Current (broken from v0.5.4)
    old = '''    # v0.5.4 (P3/P10): single-pass Table.from_batches — Arrow освобождает
    # batches по мере сборки. Раньше `list(reader)` держал все batches
    # в Python-list до конца materialize (peak ≈ list + Table = 2×).
    # Теперь peak = Table only.
    reader = ArrowIPCBatchReader(str(partition_path), None)
    table = pa.Table.from_batches(reader)
    if table.num_rows == 0:
        return pa.Table.from_batches([], schema=table.schema) if table.schema else pa.Table.from_batches([], schema=pa.schema([]))
    if memory_guard is not None:
        memory_guard.check_state(
            table.nbytes,
            name=f"read_partition_safe({partition_path.name})",
        )
    return table'''

    new = '''    # v0.5.4-fix (P3/P10): single-pass materialize.
    # ArrowIPC file: schema доступна из ipc_reader без чтения batches.
    # Если партиция пустая (0 batches) — Table.from_batches требует schema.
    # Регулярный случай — generator (peak = Table only, не list + Table).
    import pyarrow.ipc as _ipc
    _reader = _ipc.open_file(str(partition_path))
    _schema = _reader.schema
    _n = _reader.num_record_batches
    if _n == 0:
        return pa.Table.from_batches([], schema=_schema)

    def _iter_batches():
        for i in range(_n):
            yield _reader.get_batch(i)

    table = pa.Table.from_batches(_iter_batches(), schema=_schema)
    if memory_guard is not None:
        memory_guard.check_state(
            table.nbytes,
            name=f"read_partition_safe({partition_path.name})",
        )
    return table'''

    s = replace_exact(s, old, new, tag="read-partition-safe-fix")
    p.write_text(s)
    compile_check(p)
    print("  OK: _read_partition_safe — empty-safe single-pass")


# ============ Fix AST guard ============

def fix_guard_test():
    print("\n=== tests/test_materialize_guard.py: AST guard ===")
    p = Path("tests/test_materialize_guard.py")
    s = p.read_text()

    old = '''def test_read_partition_safe_no_list_reader():
    """v0.5.4 (P3/P10): `_read_partition_safe` не делает `list(reader)`.

    Регрессия Pingis P3: `list(ArrowIPCBatchReader(...))` до materialize
    даёт peak ≈ list + Table = 2×. Fix: single-pass `Table.from_batches(
    reader)`.
    """
    import inspect
    from cadw.ops import groupby as gb_mod

    src = inspect.getsource(gb_mod._read_partition_safe)
    assert "list(" not in src, (
        "_read_partition_safe must not call list(reader) (P3 regression). "
        "Use pa.Table.from_batches(reader) for single-pass streaming."
    )
    assert "Table.from_batches" in src, (
        "_read_partition_safe must use Table.from_batches for materialize"
    )'''

    new = '''def test_read_partition_safe_no_list_reader():
    """v0.5.4 (P3/P10): `_read_partition_safe` не делает `list(reader)`.

    AST guard (не substring): ищем `ast.Call` с `func.id == "list"` в теле
    функции. Docstring содержит `list(ArrowIPCBatchReader(...))` как текст —
    не должен ловиться. §13.4/76.
    """
    import ast
    import inspect
    from cadw.ops import groupby as gb_mod

    src = inspect.getsource(gb_mod._read_partition_safe)
    tree = ast.parse(inspect.cleandoc(src))

    violations: list[int] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "list"):
            violations.append(node.lineno)

    assert not violations, (
        f"_read_partition_safe calls list() at lines {violations} "
        f"(P3 regression). Use Table.from_batches(reader) single-pass."
    )
    assert "Table.from_batches" in src, (
        "_read_partition_safe must use Table.from_batches for materialize"
    )'''

    s = replace_exact(s, old, new, tag="guard-ast")
    p.write_text(s)
    compile_check(p)
    print("  OK: AST guard (no substring)")


# ============ Run ============

def main():
    print("=== v0.5.4-fix ===")

    patch_read_partition_safe()
    fix_guard_test()

    print("\n=== pytest tests/test_materialize_guard.py "
          "tests/test_groupby_v2.py tests/test_mustfix_local_groupby.py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_materialize_guard.py",
         "tests/test_groupby_v2.py",
         "tests/test_mustfix_local_groupby.py",
         "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.5.4-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.5.4-fix: _read_partition_safe — empty-safe single-pass "
        "(ipc.open_file + num_record_batches); AST guard (no substring)"])

    print("\n==> v0.5.4 complete.")


if __name__ == "__main__":
    main()