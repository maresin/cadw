#!/usr/bin/env python3
"""v0.9.2b-fix2: gate update() branches by _needs_* (not column presence).

BUG (§13.4/165): MetricsOp.update used column presence as gate:
    if self.y_pred_col is not None:    # runs regression block
For metrics=['accuracy'], y_pred='yp', y_true='cat' (string labels):
- regression block runs (WRONG — accuracy is classification).
- _to_f64('cat') → ValueError.

Correct gate: `_needs_regression` / `_needs_proba` / `_needs_classification`
(set in __init__ from requested metrics). Classification block already
correct (`if self._needs_classification:`); regression + log_loss blocks
left with old column-presence gate.

FIX: replace `if self.y_pred_col is not None:` → `if self._needs_regression:`
     replace `if self.y_proba_col is not None:` → `if self._needs_proba:`

NOTE: log_loss with string y still cryptic (ValueError: could not convert
'cat'). Honest — log_loss requires numeric binary y; user should encode.
Improvement backlog §13.4/165 note.

GUARDS: metrics fast + full suite + guards.

Run: uv run python scripts/patch_v092b_fix2.py
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


def fix_update_gates():
    print("\n=== cadw/ops/metrics.py::update — gate by _needs_* ===")
    p = Path("cadw/ops/metrics.py")
    s = p.read_text()

    # Regression block gate
    old_reg = '''        # Regression
        if self.y_pred_col is not None:
            y = _to_f64(batch.column(self.y_true_col))
            yp = _to_f64(batch.column(self.y_pred_col))'''

    new_reg = '''        # Regression (v0.9.2b-fix2, §13.4/165): gate by _needs_regression,
        # not by column presence. metrics=['accuracy'], y_pred='yp' with
        # string y_true would otherwise hit _to_f64('cat') → ValueError.
        if self._needs_regression:
            y = _to_f64(batch.column(self.y_true_col))
            yp = _to_f64(batch.column(self.y_pred_col))'''

    s = s.replace(old_reg, new_reg, 1)

    # Log loss block gate
    old_ll = '''        # Log loss (binary; y must be {0,1}, p in [0,1])
        if self.y_proba_col is not None:
            y = _to_f64(batch.column(self.y_true_col))'''

    new_ll = '''        # Log loss (binary; y must be {0,1}, p in [0,1])
        if self._needs_proba:
            y = _to_f64(batch.column(self.y_true_col))'''

    s = s.replace(old_ll, new_ll, 1)

    p.write_text(s)
    compile_check(p)
    print("  OK")


def append_decisions_165():
    print("\n=== DECISIONS.md: §13.4/165 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 165" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/165 (gate by metric needs, not column presence).** "
        s += "regression block ran for classification metrics with "
        s += "y_pred set → _to_f64('cat').\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 165 (gate by metric needs, not column presence).**
v0.9.2b: `MetricsOp.update` gated regression block by column presence
`if self.y_pred_col is not None:`. Для `metrics=['accuracy'],
y_pred='yp', y_true='cat'` (string labels): regression block runs →
`_to_f64('cat')` → `ValueError: could not convert string to float`.
1 test fails.

**Правильный gate:** `_needs_regression` / `_needs_proba` /
`_needs_classification` — установлены в `__init__` из requested
metrics. v092b добавил `_needs_classification` gate корректно, но
регрессию/логи оставил на старом `y_pred_col is not None`.

**Правило (§13.4/165, расширение 148, 162):** gate в `update` /
`finalize` — **by metric needs** (`_needs_*`), не by input presence.
Presence говорит «пользователь передал col», needs говорит «этот
блок действительно нужен». Первое — input, второе — semantic.

**Backlog (§11.6):** log_loss с string y по-прежнему cryptic error
(`could not convert 'cat'`). Улучшение: `MetricsOp._to_f64_numeric(
arr, "log_loss requires numeric binary y")` — context-specific
message. v0.9.2c.

**Связанные:** §13.4/148 (design vs validation), /162 (redundant check
masks specific), /163 (domain validation), /164 (pos_label ambiguity).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/165")


def main():
    print("=== v0.9.2b-fix2 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_update_gates()
    append_decisions_165()

    print("\n=== pytest: metrics fast (was 1 failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_metrics.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** METRICS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: metrics E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_metrics_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: reduce regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_reduce.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REDUCE REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
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
        "v0.9.2b: classification metrics — accuracy/precision/recall/f1 "
        "(binary + multiclass, average modes); MetricsOp tp/fp/fn state; "
        "ReduceOp.batch_memory_factor; §13.4/164 (pos_label=None→"
        "classes[-1]; np.divide where=); §13.4/165 (gate by _needs_* "
        "not column presence)"])

    print("\n==> v0.9.2b-fix2 applied. v0.9.2b complete.")
    print("NEXT: v0.9.2c (roc_auc + multiclass log_loss).")


if __name__ == "__main__":
    main()