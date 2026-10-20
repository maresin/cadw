#!/usr/bin/env python3
"""v0.9.1e-fix1: missing helpers (naive `in s` check matched call site).

BUG (§13.4/154, same class as §13.4/53):
  `add_helpers()` in v0.9.1e used `if "_detect_mode" in s: skip`.
  `_detect_mode` IS in file — as a CALL in `_per_group_apply`
  (`_detect_mode(result)`), not as a DEFINITION. Check didn't
  distinguish. Helpers never appended → 14 tests fail with NameError.

FIX:
  - Append helpers block (checked by `def _detect_mode(` prefix,
    not raw string).
  - Add §13.4/154 docstring note.

GUARDS: apply tests + pipeline regression + guards + fast + slow.

Run: uv run python scripts/patch_v091e_fix1.py
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


def append_helpers():
    print("\n=== cadw/dataframe.py: append per-group apply helpers ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    # §13.4/154: check `def NAME(` prefix — call sites also contain NAME.
    required = [
        "def _detect_mode(",
        "def _normalize_scalar(",
        "def _infer_arrow_type_from_value(",
        "def _build_out_schema(",
    ]
    missing = [r for r in required if r not in s]
    if not missing:
        print("  all helpers present — skip")
        return
    print(f"  missing: {missing}")

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

    if not s.endswith("\n"):
        s += "\n"
    s += HELPERS
    p.write_text(s)
    compile_check(p)
    print("  OK: helpers appended")


def append_decisions_154():
    print("\n=== DECISIONS.md: §13.4/154 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 154" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/154 (call-site vs definition substring).** "
        s += "`if \"NAME\" in src` matches call sites; use `if \"def NAME(\"`.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 154 (call-site vs definition substring — §13.4/53 repeat).**
v0.9.1e: `add_helpers()` проверял `if "_detect_mode" in s: skip`.
`_detect_mode` **есть** в файле — как **вызов** в `_per_group_apply`
(`_detect_mode(result)`), не как **def**. Проверка не отличила →
helpers никогда не добавлены → 14 tests fail с `NameError`.

**Тот же класс, что §13.4/53** (из docs — `if "name" not in s` видел
call site и не добавлял def). Забыто через 140 эпизодов.

**Fix (v0.9.1e-fix1):** проверять `if "def NAME(" in s`. `def ` prefix
отличает определение от вызова.

**Правило (§13.4/154):** при проверке наличия **функции/класса** в
исходнике — проверять `"def NAME("` или `"class NAME("`, никогда
голое `"NAME"`. Аналогично для полей класса — `"self.NAME ="`, не
`".NAME"`.

**Backlog (§11.6):** audit всех patch scripts на `if "X" in s` where
X — function/class/attribute name. Regex check: `if f"def {name}(" in s`.
Сейчас ~30+ патчей с этим паттерном; все исторические, не трогаем.
Новые — по правилу.

**Связанные:** §13.4/53 (original discovery), §13.4/146 (first-run
bugs — that round also included a "call site matched" issue),
§13.4/153 (tooling caught its own bug — but this one not caught).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/154")


def main():
    print("=== v0.9.1e-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    append_helpers()
    append_decisions_154()

    print("\n=== pytest: apply tests (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_apply.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** APPLY TESTS STILL RED ***")
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
        "§11.5 constraints; §13.4/154 (def-prefix check for helpers)"])

    print("\n==> v0.9.1e-fix1 applied.")
    print("NEXT: v0.9.1f (pivot_table / melt / crosstab). Then tag v0.9.1.")


if __name__ == "__main__":
    main()