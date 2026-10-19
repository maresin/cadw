#!/usr/bin/env python3
"""v0.8.2-fix3: _shift_on второй sort — переименовать row_id → __orig_rid.

Диагностика v0.8.2-fix2:
    ArrowInvalid: Multiple matches for FieldRef.Name(row_id)
    in v: int64 / row_id: uint64 / row_id: uint64

Причина: external_sort ВСЕГДА делает append_column("row_id", ...).
Поток после step 2 (`shifted_by_on.arrow`) уже содержит row_id (сохранённый
original pos). Второй sort (by row_id) делает select(["v", "row_id"]) и
append ещё одного row_id → два row_id в одной таблице → pc.sort_indices
падает на FieldRef.Name(row_id).

Fix: перед вторым sort переименовать row_id → __orig_rid, сортировать по
__orig_rid. external_sort append'ит свой row_id (порядковая позиция,
игнорируем). Порядок __orig_rid ascending = original order.

Also:
    - AST guard `test_shift_on_no_duplicate_row_id` — проверяет, что во
      втором external_sort `_shift_on` key_columns НЕ содержит "row_id"
      (только `__orig_rid`).

Run: uv run python scripts/patch_v082_fix3.py
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


def fix_shift_on():
    print("\n=== cadw/dataframe.py: _shift_on second sort ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    # Before
    old_block = '''        # 3. Sort back by row_id
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
        )'''

    new_block = '''        # 3. Sort back by __orig_rid.
        # external_sort ALWAYS appends its own "row_id" column. Our stream
        # already has "row_id" (= original position) → two row_ids would
        # trigger ArrowInvalid: Multiple matches for FieldRef.Name(row_id).
        # Rename "row_id" → "__orig_rid" so external_sort's appended row_id
        # is a different column. Sort by __orig_rid ascending = original order.
        def _iter_shifted():
            for b in ArrowIPCBatchReader(str(shift_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        sorted_rowid_path = external_sort(
            _iter_shifted(),
            key_columns=["__orig_rid"],
            value_columns=cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "shift" / "sort_by_rowid",
            ascending=True,
            log=False,
        )'''

    s = replace_line_block(s, old_block, new_block, tag="shift-on-step3")
    p.write_text(s)
    compile_check(p)
    print("  OK: row_id → __orig_rid перед вторым sort")


def add_ast_guard():
    print("\n=== tests/test_shift_on_col.py: +AST guard ===")
    p = Path("tests/test_shift_on_col.py")
    s = p.read_text()
    if "test_shift_on_second_sort_renames_row_id" in s:
        print("  already present — skip")
        return

    extra = '''

def test_shift_on_second_sort_renames_row_id():
    """AST guard: _shift_on второй external_sort не должен использовать
    key_columns=["row_id"] — это вызовет duplicate row_id (ArrowInvalid).

    Регрессия v0.8.2-fix3: external_sort ВСЕГДА append_column("row_id", ...).
    Второй sort (by row_id) на потоке с уже-существующим row_id → два
    row_id → pc.sort_indices падает. Fix: переименовать row_id → __orig_rid.
    """
    import ast as _ast
    import inspect as _inspect
    from cadw.dataframe import CadwDataFrame

    src = _inspect.getsource(CadwDataFrame._shift_on)
    tree = _ast.parse(_inspect.cleandoc(src))

    # Find external_sort calls; extract key_columns arg
    external_sort_calls: list[dict] = []
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Name)
                and node.func.id == "external_sort"):
            keys_arg = None
            for kw in node.keywords:
                if kw.arg == "key_columns":
                    # Expect List(Constant("__orig_rid"))
                    if isinstance(kw.value, _ast.List):
                        keys_arg = [
                            elt.value for elt in kw.value.elts
                            if isinstance(elt, _ast.Constant)
                        ]
            external_sort_calls.append({"lineno": node.lineno, "keys": keys_arg})

    assert len(external_sort_calls) >= 2, (
        f"expected >=2 external_sort in _shift_on; got {len(external_sort_calls)}"
    )
    # Second sort (last in list, since first-by-on is earlier)
    second = external_sort_calls[-1]
    assert "row_id" not in (second["keys"] or []), (
        f"second external_sort (line {second['lineno']}) uses key_columns="
        f"{second['keys']}; must use __orig_rid (not raw row_id — "
        f"external_sort appends its own row_id → duplicate column)"
    )
    assert "__orig_rid" in (second["keys"] or []), (
        f"second external_sort (line {second['lineno']}) must sort by "
        f"__orig_rid; got {second['keys']}"
    )
'''

    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: AST guard (no duplicate row_id)")


def main():
    print("=== v0.8.2-fix3 ===")
    fix_shift_on()
    add_ast_guard()

    print("\n=== pytest tests/test_shift_on_col.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_shift_on_col.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.2-fix3 (test red)"], check=False)
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
        sh(["git", "commit", "-m", "WIP: v0.8.2-fix3 (regression)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.2-fix3: _shift_on step 3 — rename row_id → __orig_rid "
        "(external_sort appends its own row_id; duplicate → ArrowInvalid); "
        "+AST guard"])

    print("\n==> v0.8.2-fix3 complete. Готов squash → tag v0.8.2.")


if __name__ == "__main__":
    main()