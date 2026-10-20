#!/usr/bin/env python3
"""v0.9.2a-fix1: redundant pre-check masks per-metric validation.

BUG (§13.4/162, test vs code design):
  `metrics()` had:
      if y_pred is None and y_proba is None:
          raise ValueError("at least one of y_pred / y_proba required")
  Then `MetricsOp.__init__` ALSO validates per-metric:
      if needs_pred and y_pred_col is None:
          raise ValueError("y_pred_col required for ['r2']")
      if needs_proba and y_proba_col is None:
          raise ValueError("y_proba_col required for ['log_loss']")
  Generic check fires FIRST → specific message never shown.
  2 test failures (test expects "y_pred_col required" / "y_proba_col required").

  Plus: `test_metrics_no_pred_or_proba` duplicates
  `test_metrics_missing_y_pred_for_r2` (same scenario, different expected
  message). Bad test design — one canonical test per behavior.

FIX:
  1. Remove generic pre-check in `metrics()` — let `MetricsOp` raise
     specific per-metric error. Better UX ("you asked for r2 but passed
     no y_pred" > "at least one required").
  2. Delete `test_metrics_no_pred_or_proba` (redundant with
     `test_metrics_missing_y_pred_for_r2`).
  3. §13.4/162 — layered validation: don't check at outer layer what
     inner layer checks more precisely.
  4. Dogfood: `assert_no_stale_behavior(["at least one of y_pred"], tests/)`
     — after removing the message, no test should reference it.

GUARDS: metrics tests + full suite + guards.

Run: uv run python scripts/patch_v092a_fix1.py
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


# ============ Fix 1: remove generic pre-check ============

def remove_generic_check():
    print("\n=== cadw/dataframe.py::metrics — remove generic pre-check ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = '''        if not metrics:
            raise ValueError("metrics: metrics list is empty")

        if y_pred is None and y_proba is None:
            raise ValueError(
                "metrics: at least one of y_pred / y_proba required"
            )

        for c, label in ('''

    new = '''        if not metrics:
            raise ValueError("metrics: metrics list is empty")

        # §13.4/162: no generic "at least one required" check —
        # `MetricsOp.__init__` raises per-metric specific error
        # ("y_pred_col required for ['r2']" / "y_proba_col required for
        # ['log_loss']"). Better UX: specific > generic.

        for c, label in ('''

    s = replace_line_block(s, old, new, tag="remove-generic-check")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 2: delete redundant test ============

def remove_redundant_test():
    print("\n=== tests/test_metrics.py: delete test_metrics_no_pred_or_proba ===")
    p = Path("tests/test_metrics.py")
    s = p.read_text()

    old = '''def test_metrics_no_pred_or_proba(tmp_path):
    df = make_df(tmp_path, {"y": [1.0]})
    with pytest.raises(ValueError, match="at least one"):
        df.metrics(y_true="y", metrics=["r2"])


'''

    if old not in s:
        print("  already absent — skip")
        return
    s = s.replace(old, "", 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: deleted (redundant with test_metrics_missing_y_pred_for_r2)")


# ============ Fix 3: §13.4/162 ============

def append_decisions_162():
    print("\n=== DECISIONS.md: §13.4/162 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 162" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/162 (redundant check masks specific).** "
        s += "metrics() generic 'at least one' fires before MetricsOp "
        s += "specific 'y_pred_col required'.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 162 (redundant pre-check masks per-metric validation).**
v0.9.2a: `metrics()` raises generic `"at least one of y_pred / y_proba
required"` **before** `MetricsOp.__init__` успевает дать specific
`"y_pred_col required for ['r2']"` / `"y_proba_col required for
['log_loss']"`. 2 test failures (test expects specific, code gives
generic).

**Диагноз:** layered validation — outer layer defends against case,
inner layer validates more precisely. Generic check **fires first**,
specific message never shown.

**Fix:** убрать generic check. `MetricsOp` — единственный источник
правды для per-metric validation. UX: «you asked for r2 but passed no
y_pred» лучше чем «at least one required».

**Плюс:** `test_metrics_no_pred_or_proba` дублирует
`test_metrics_missing_y_pred_for_r2` (тот же сценарий `metrics(y_true=
"y", metrics=["r2"])` без y_pred). Один test ожидал generic, другой
specific — **несогласованные ожидания** на одно поведение. Delete
redundant.

**Правило (§13.4/162, расширение 148):** validation должна быть на
**одном уровне**, не distributed между caller и callee. Outer defends
только против того, что inner не может (например, `metrics` validates
column existence, потому что `MetricsOp` не знает про Dataset);
inner validates то, что знает лучше (specific metric requirements).
Не дублировать.

**Backlog (§11.6):** audit layered validations:
- `metrics()` — fixed.
- `Pipeline.fit` → `plan` → `step.estimate_state_bytes` — outer validates
  target presence; inner validates state. OK.
- `pivot_table` → `groupby_aggregate` — outer validates K_pairs; inner
  validates partition budget. OK (different levels).
- `apply(fn)` → `_per_group_apply` — outer validates fn callable; inner
  validates group size. OK.

**Связанные:** §13.4/148 (design intent vs validation), /156 (blanket
guard), /157 (hasattr pyarrow), /161 (AST substring).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/162")


# ============ Fix 4: dogfood — stale "at least one" grep ============

def dogfood_grep():
    """Dogfood: assert_no_stale_behavior for removed generic message."""
    print("\n=== dogfood (fatal): no 'at least one' in test_metrics ===")
    from cadw.patch_utils import assert_no_stale_behavior
    assert_no_stale_behavior(
        patterns=["at least one"],
        dirs=["tests/test_metrics.py", "tests/test_metrics_e2e.py"],
        label="at-least-one-stale",
    )
    print("  OK: no 'at least one' references in metrics tests")


def main():
    print("=== v0.9.2a-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    remove_generic_check()
    remove_redundant_test()
    append_decisions_162()
    dogfood_grep()

    print("\n=== pytest: metrics tests (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_metrics.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** METRICS TESTS STILL RED ***")
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
        "v0.9.2a: metrics — regression (r2/mse/rmse/mae) + binary "
        "log_loss via ReduceOp; single-pass O(1) state; NaN for empty; "
        "sklearn-compatible R² edge cases; §13.4/162 (redundant generic "
        "pre-check removed — per-metric specific error from MetricsOp)"])

    print("\n==> v0.9.2a-fix1 applied.")
    print("NEXT: v0.9.2b (classification metrics — confusion matrix).")


if __name__ == "__main__":
    main()