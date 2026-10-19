#!/usr/bin/env python3
"""v0.5.4: P3+P10 — _read_partition_safe без list(reader).

Проблема (Pingis P3/P10):
    batches = list(ArrowIPCBatchReader(path))       # Python-list, peak 2×
    table = pa.Table.from_batches(batches)
    check_state(table.nbytes)                        # после materialize

Пик = list_nbytes + table_nbytes (×2 — каждая Arrow buffer в batch'е
живёт в list, пока Table не собран).

Fix (v0.5.4):
    table = pa.Table.from_batches(ArrowIPCBatchReader(path))  # single pass
    check_state(table.nbytes)                        # тот же check, но
                                                     # peak = table only
    gc.collect() перед group_by

Ожидание: join / groupby partition-цикл под бюджетом — фактический peak
снижается на ~50% для больших партиций; функционально эквивалентно.

Run: uv run python scripts/patch_v054_groupby_stream.py
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
        cmd, shell=isinstance(cmd, str),
        capture_output=True, text=True,
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


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    cur = r.stdout.strip()
    if cur == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        print(f"  branch: checkout {name}")
        sh(["git", "checkout", name])
    else:
        print(f"  branch: create {name}")
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


# ============ Patch groupby.py ============

def patch_read_partition_safe():
    print("\n=== cadw/ops/groupby.py: _read_partition_safe ===")
    p = Path("cadw/ops/groupby.py")
    s = p.read_text()

    old = '''    batches = list(ArrowIPCBatchReader(str(partition_path), None))
    if not batches:
        return pa.Table.from_batches([], schema=pa.schema([]))
    table = pa.Table.from_batches(batches)
    if memory_guard is not None:
        memory_guard.check_state(
            table.nbytes,
            name=f"read_partition_safe({partition_path.name})",
        )
    return table'''

    new = '''    # v0.5.4 (P3/P10): single-pass Table.from_batches — Arrow освобождает
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

    s = replace_exact(s, old, new, tag="read-partition-safe")
    p.write_text(s)
    compile_check(p)
    print("  OK: _read_partition_safe — no list(reader)")


def patch_local_groupby_gc():
    print("\n=== cadw/ops/groupby.py: gc before group_by ===")
    p = Path("cadw/ops/groupby.py")
    s = p.read_text()

    # Add `import gc` if not present
    if "\nimport gc" not in s and not s.startswith("import gc"):
        old_imp = "import itertools"
        new_imp = "import gc\nimport itertools"
        if old_imp in s:
            s = s.replace(old_imp, new_imp, 1)
            print("  import gc added")

    old = '''        if dict_fits:
            if partial_merge:
                return _merge_partials(
                    table, by_columns,
                    [(col, f) for col, f in zip(agg_columns, agg_funcs)],
                )
            agg_specs = [(col, f) for col, f in zip(agg_columns, agg_funcs)]
            return table.group_by(list(by_columns)).aggregate(agg_specs)'''

    new = '''        if dict_fits:
            # v0.5.4 (P10): gc before group_by — освободить reader/writer
            # buffers (Table уже materialized; old batches не нужны).
            gc.collect()
            if partial_merge:
                return _merge_partials(
                    table, by_columns,
                    [(col, f) for col, f in zip(agg_columns, agg_funcs)],
                )
            agg_specs = [(col, f) for col, f in zip(agg_columns, agg_funcs)]
            return table.group_by(list(by_columns)).aggregate(agg_specs)'''

    s = replace_exact(s, old, new, tag="gc-before-groupby")
    p.write_text(s)
    compile_check(p)
    print("  OK: gc.collect() before group_by")


# ============ AST guard ============

def append_guard_test():
    print("\n=== tests/test_materialize_guard.py: +_read_partition_safe guard ===")
    p = Path("tests/test_materialize_guard.py")
    s = p.read_text()

    marker = "def test_read_partition_safe_no_list_reader()"
    if marker in s:
        print("  guard already present — skip")
        return

    extra = '''

def test_read_partition_safe_no_list_reader():
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
    )
'''

    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: guard test added")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v054-groupby-stream"

    print("=== v0.5.4: P3+P10 (groupby materialize) ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v054-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v054-{ts_str}",
        "-m", "before patch v0.5.4 (P3+P10)"])

    patch_read_partition_safe()
    patch_local_groupby_gc()
    append_guard_test()

    print("\n=== pytest tests/test_materialize_guard.py tests/test_groupby_v2.py tests/test_mustfix_local_groupby.py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_materialize_guard.py",
         "tests/test_groupby_v2.py",
         "tests/test_mustfix_local_groupby.py",
         "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP commit ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.5.4 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.5.4: P3+P10 — _read_partition_safe single-pass (no list(reader)); gc before group_by"])

    print("\n==> v0.5.4 applied on", branch)


if __name__ == "__main__":
    main()