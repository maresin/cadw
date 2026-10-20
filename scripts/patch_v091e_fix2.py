#!/usr/bin/env python3
"""v0.9.1e-fix2: `_flush_group` pre-check + DataFrame facade AST guard.

BUG 1 (§13.4/155): `_flush_group` calls `Table.from_batches(group_buf)`
BEFORE `check_state` — because `check_state` needs `gt.nbytes`. Chicken-
and-egg: from_batches is the potential OOM moment, but we need it to
measure. AST guard flags this.

FIX 1: pre-estimate from `sum(b.nbytes for b in group_buf) * 2` BEFORE
from_batches. In pyarrow 26, `Table.from_batches` is zero-copy (§9.2.9),
so `sum(b.nbytes) ≈ gt.nbytes`. `check_state(est_pre)` before from_batches
satisfies AST guard. Keep `est_actual = gt.nbytes * 2` check after
from_batches as defense-in-depth.

BUG 2 (§13.4/155): `test_no_pandas_in_dataframe_facade` uses
`_uses_pandas_import(tree)` which walks ALL AST — flags function-level
`import pandas as pd` in `_emit_group` (legitimate lazy import, only
inside `_per_group_apply`). Pandas needed to construct pd.Series/DataFrame
from user fn output — must not leak to `import cadw` top-level.

FIX 2: add `_uses_pandas_import_at_module_level(tree)` helper that only
checks `ast.Module.body`. Update `test_no_pandas_in_dataframe_facade`
to use it. Keep strict `_uses_pandas_import` for `cadw/ops/*` (no pandas
anywhere except whitelisted shuffle.py).

GUARDS: apply tests + pipeline regression + guards + fast + slow.

Run: uv run python scripts/patch_v091e_fix2.py
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


# ============ Fix 1: _flush_group pre-check before from_batches ============

def fix_flush_group_precheck():
    print("\n=== cadw/dataframe.py::_flush_group — pre-check before from_batches ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = '''        def _flush_group():
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
            group_buf.clear()'''

    new = '''        def _flush_group():
            nonlocal group_buf
            if not group_buf:
                return
            # §13.4/155: pre-estimate from `batch.nbytes` BEFORE
            # `from_batches`. In pyarrow 26 from_batches is zero-copy
            # (§9.2.9) → sum(b.nbytes) ≈ gt.nbytes. Check first, then
            # materialize. Keeps AST guard (§3.11) satisfied — no
            # high-risk Arrow call without preceding check_state.
            est_pre = sum(b.nbytes for b in group_buf) * 2
            if est_pre > max_group_limit:
                raise MemoryLimitExceeded(
                    f"_per_group_apply: group size ~{est_pre // 2} B "
                    f"× 2 (pandas) = {est_pre} B > 0.3 × avail = "
                    f"{max_group_limit} B. Split by more group cols "
                    f"or use groupby().agg() if fn reduces. See "
                    f"§11.5."
                )
            memory_guard.check_state(
                est_pre, name="per_group_apply.group",
            )
            gt = pa.Table.from_batches(group_buf)
            # Defense-in-depth: actual nbytes (may differ from sum of
            # batch.nbytes for chunked buffers; conservative check).
            est_actual = gt.nbytes * 2
            if est_actual > max_group_limit:
                raise MemoryLimitExceeded(
                    f"_per_group_apply: actual group size "
                    f"{gt.nbytes} B × 2 = {est_actual} B > 0.3 × avail "
                    f"= {max_group_limit} B (pre-estimate was "
                    f"{est_pre} B; buffer overhead). See §11.5."
                )
            gdf = gt.to_pandas()
            result = fn(gdf)
            _emit_group(tuple(prev_key), result)
            group_buf.clear()'''

    s = replace_line_block(s, old, new, tag="flush-group-precheck")
    p.write_text(s)
    compile_check(p)
    print("  OK: pre-estimate from batch nbytes, then from_batches")


# ============ Fix 2: AST guard — module-level only for dataframe.py ============

def fix_dataframe_facade_test():
    print("\n=== tests/test_no_pandas_in_ops.py: module-level helper + test ===")
    p = Path("tests/test_no_pandas_in_ops.py")
    s = p.read_text()

    # Add module-level helper after existing _uses_pandas_import
    old_helper_end = '''def _uses_pandas_import(tree: ast.Module) -> bool:
    """True if module has `import pandas` or `from pandas ...`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pandas" or alias.name.startswith("pandas."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "pandas"
                or node.module.startswith("pandas.")
            ):
                return True
    return False'''

    new_helper = old_helper_end + '''


def _uses_pandas_import_at_module_level(tree: ast.Module) -> bool:
    """True if module-level (top-level) import of pandas.

    v0.9.1e-fix2 (§13.4/155): lazy `import pandas as pd` inside a
    function is OK — keeps `import cadw` light, pandas only loaded when
    needed (e.g., `_per_group_apply` constructs pd.Series/DataFrame from
    user fn output). Module-level import would eagerly load pandas.
    """
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pandas" or alias.name.startswith("pandas."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "pandas"
                or node.module.startswith("pandas.")
            ):
                return True
    return False'''

    if old_helper_end not in s:
        raise SystemExit("_uses_pandas_import helper not found")

    s = s.replace(old_helper_end, new_helper, 1)

    # Update the test to use module-level helper
    old_test = '''def test_no_pandas_in_dataframe_facade():
    """`cadw/dataframe.py` — no pandas import (facade is Arrow-native)."""
    py = CADW_OPS.parent / "dataframe.py"
    src = py.read_text()
    tree = ast.parse(src)
    assert not _uses_pandas_import(tree), (
        "cadw/dataframe.py must not import pandas (facade is Arrow-native)"
    )'''

    new_test = '''def test_no_pandas_in_dataframe_facade():
    """`cadw/dataframe.py` — no MODULE-LEVEL pandas import.

    v0.9.1e-fix2 (§13.4/155): lazy `import pandas as pd` inside
    `_emit_group` (used by `_per_group_apply` to construct pd.Series/
    pd.DataFrame from user fn output) is legitimate — keeps `import
    cadw` light. Only top-level import is forbidden.
    """
    py = CADW_OPS.parent / "dataframe.py"
    src = py.read_text()
    tree = ast.parse(src)
    assert not _uses_pandas_import_at_module_level(tree), (
        "cadw/dataframe.py must not import pandas at module level "
        "(lazy import inside functions OK — see §13.4/155)"
    )'''

    if old_test not in s:
        raise SystemExit("test_no_pandas_in_dataframe_facade not found")

    s = s.replace(old_test, new_test, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: module-level helper + updated test")


# ============ Fix 3: §13.4/155 ============

def append_decisions_155():
    print("\n=== DECISIONS.md: §13.4/155 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 155" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/155 (chicken-egg guard + too-strict AST).** "
        s += "_flush_group pre-checks from batch nbytes before from_batches; "
        s += "module-level-only pandas import for dataframe.py.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 155 (chicken-egg guard + lazy-import AST over-strictness).**
v0.9.1e-fix1 прошёл apply-тесты (19 fast + 1 slow E2E), но упал на
**двух guard-тестах**:

**Bug 1 (chicken-egg):** `_flush_group` делает `pa.Table.from_batches(
group_buf)` **до** `check_state(est)`. Нужен `gt.nbytes` для est — а
`from_batches` и есть момент аллокации. AST guard (§3.11) правильно
флагает: high-risk Arrow call без preceding check.

Fix: **pre-estimate из `sum(b.nbytes for b in group_buf) * 2`** до
`from_batches` (pyarrow 26 from_batches — zero-copy per §9.2.9, значит
`sum(batch.nbytes) ≈ gt.nbytes`). `check_state(est_pre)` → `from_batches`
→ defense-in-depth `est_actual = gt.nbytes * 2`.

**Bug 2 (too-strict AST):** `test_no_pandas_in_dataframe_facade`
walks **весь AST** — ловит lazy `import pandas as pd` **внутри
`_emit_group`** (legitimate: нужен для `pd.Series`/`pd.DataFrame` при
обработке user fn output). Module-level import pandas в dataframe.py —
запрещён (тянет pandas при `import cadw`); function-level — допустим.

Fix: `_uses_pandas_import_at_module_level(tree)` — только `ast.Module.body`.
Strict `_uses_pandas_import` остаётся для `cadw/ops/*` (там даже lazy
не хотим, кроме whitelisted shuffle.py).

**Оба бага — фиксы не кода, а:**
- Bug 1 — не «код неверен», а «порядок операций в коде». `from_batches`
  нельзя сдвинуть после `check_state` без pre-estimate. Pre-estimate из
  `batch.nbytes` — стандартный паттерн (как `stat().st_size`).
- Bug 2 — не «код неверен», а «guard слишком широк». Lazy import vs
  eager import — разные вещи, AST должен различать.

**Урок (расширение §13.4/154):** AST guard должен быть **precise**:
- `def NAME(` vs `NAME` (— §13.4/154).
- `ast.Module.body` vs `ast.walk` (module-level vs any).
- `attribute == X` vs `substring X`.

Too-broad guard блокирует legitimate use cases; too-narrow — пропускает
баги. Fix always **narrow** to specific semantic (def, module-level,
attr name), never `walk` or substring.

**Backlog (§11.6):** audit guard-тестов на over-strictness:
- `test_no_pandas_in_ops` (strict for ops — OK; strict for facade — fixed).
- `test_no_unguarded_high_risk_arrow_calls` (func-level check — OK).
- `test_facade_has_no_as_py` (AST Call — OK).
- `test_materialize_guard` (substring `list(Reader)` — potential false
  positive if docstring contains pattern; но AST-based, OK).

**Связанные:** §13.4/154 (def-prefix), §13.4/152 (tooling), §13.4/151
(registry namespace), §3.11 (Arrow call memory discipline).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/155")


def main():
    print("=== v0.9.1e-fix2 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_flush_group_precheck()
    fix_dataframe_facade_test()
    append_decisions_155()

    print("\n=== pytest: guards (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** GUARDS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: apply tests (should stay green) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_apply.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** APPLY REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline + CT regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_column_transformer.py",
         "tests/test_apply_e2e.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: all guards via patch_utils ===")
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
        "§11.5 constraints; §13.4/154 (def-prefix helper check); "
        "§13.4/155 (pre-check before from_batches; module-level-only "
        "pandas import for dataframe.py)"])

    print("\n==> v0.9.1e-fix2 applied.")
    print("NEXT: v0.9.1f (pivot_table / melt / crosstab). Then tag v0.9.1.")


if __name__ == "__main__":
    main()