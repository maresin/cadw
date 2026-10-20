#!/usr/bin/env python3
"""v0.9.2c-fix1: 4 root causes behind 14 test failures.

BUG 1 (partial migration, §13.4/167): MetricsOp.__init__ checked
legacy `y_proba_col`, but callers passing only `y_proba_cols=[...]`
get y_proba_col=None → false "y_proba_col required". Fix: check
`self.y_proba_cols is None`.

BUG 2 (same class): CadwDataFrame.metrics facade iterated
`(y_proba, "y_proba")` in generic column-existence loop. y_proba
is now `str | Sequence[str]` — when list, `list not in columns` is
always True → KeyError. Fix: remove y_proba from generic loop;
specific per-column check already exists after normalization.

BUG 3 (test-authoring, §13.4/166 — 5th time in session):
`test_metrics_roc_auc_with_accuracy` expected 0.875 manually. Actual
AUC = 1.0 (all 4 pos/neg pairs correctly ordered). Fix: use
`expected_roc_auc` reference helper (no manual arithmetic).

BUG 4 (test regex vs message): message said "is computed by
CadwDataFrame.metrics" but test regex looks for "roc_auc.*separately".
Fix: add "separately" to message — clearer intent.

Also: silence SyntaxWarning in test regex — `"len\(y_proba\)"` →
`r"len\(y_proba\)"` (raw string).

GUARDS: metrics fast + E2E + reduce + full suite + guards.

Run: uv run python scripts/patch_v092c_fix1.py
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


# ============ Fix 1: MetricsOp.__init__ — unified y_proba_cols ============

def fix_metricsop_yproba_check():
    print("\n=== cadw/ops/metrics.py: check unified y_proba_cols ===")
    p = Path("cadw/ops/metrics.py")
    s = p.read_text()

    old = '''        if self._needs_proba and y_proba_col is None:
            raise ValueError(
                f"MetricsOp: y_proba_col required for "
                f"{sorted(ms & PROBA_METRICS)}"
            )'''

    new = '''        # v0.9.2c-fix1 (§13.4/167): check unified y_proba_cols, NOT
        # legacy y_proba_col. When only y_proba_cols=[...] is passed,
        # y_proba_col remains None even though proba IS provided.
        # Message kept as "y_proba_col" for v0.9.2a test regex compat.
        if self._needs_proba and self.y_proba_cols is None:
            raise ValueError(
                f"MetricsOp: y_proba_col required for "
                f"{sorted(ms & PROBA_METRICS)}"
            )'''

    s = replace_line_block(s, old, new, tag="metricsop-yproba-check")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 2: facade — remove y_proba from generic column loop ============

def fix_facade_column_loop():
    print("\n=== cadw/dataframe.py::metrics — y_proba loop bug ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old = '''        for c, label in (
            (y_true, "y_true"), (y_pred, "y_pred"), (y_proba, "y_proba")
        ):
            if c is not None and c not in self._columns:
                raise KeyError(
                    f"metrics: {label} column {c!r} not in {self._columns}"
                )'''

    new = '''        # v0.9.2c-fix1 (§13.4/167): y_proba is str | Sequence[str];
        # generic loop can't handle list (would check `list not in
        # columns` — always True). Specific per-column check below
        # (after y_proba_cols normalization).
        for c, label in ((y_true, "y_true"), (y_pred, "y_pred")):
            if c is not None and c not in self._columns:
                raise KeyError(
                    f"metrics: {label} column {c!r} not in {self._columns}"
                )'''

    s = replace_line_block(s, old, new, tag="facade-column-loop")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 3: test 0.875 → reference helper ============

def fix_test_roc_auc_with_accuracy():
    print("\n=== tests/test_metrics.py: roc_auc_with_accuracy use helper ===")
    p = Path("tests/test_metrics.py")
    s = p.read_text()

    old = '''def test_metrics_roc_auc_with_accuracy(tmp_path):
    """roc_auc + accuracy, both binary — different columns."""
    df = make_df(tmp_path, {
        "y": [0, 0, 1, 1], "yp": [0, 0, 1, 0], "p": [0.1, 0.2, 0.8, 0.4],
    })
    res = df.metrics(
        y_true="y", y_pred="yp", y_proba="p",
        metrics=["roc_auc", "accuracy"], classes=[0, 1],
    )
    assert res["accuracy"] == pytest.approx(0.75)
    assert res["roc_auc"] == pytest.approx(0.875)'''

    new = '''def test_metrics_roc_auc_with_accuracy(tmp_path):
    """roc_auc + accuracy, both binary — different columns.

    §13.4/166: expected AUC from `expected_roc_auc` reference, not
    manual arithmetic. y=[0,0,1,1], p=[0.1,0.2,0.8,0.4] → all 4
    pos/neg pairs correctly ordered → AUC = 1.0. Prior manual 0.875
    was wrong (5th test-authoring mistake in session — see §13.4/167).
    """
    from tests._helpers.metrics_reference import expected_roc_auc
    df = make_df(tmp_path, {
        "y": [0, 0, 1, 1], "yp": [0, 0, 1, 0], "p": [0.1, 0.2, 0.8, 0.4],
    })
    res = df.metrics(
        y_true="y", y_pred="yp", y_proba="p",
        metrics=["roc_auc", "accuracy"], classes=[0, 1],
    )
    assert res["accuracy"] == pytest.approx(0.75)
    assert res["roc_auc"] == pytest.approx(
        expected_roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.4])
    )'''

    s = replace_line_block(s, old, new, tag="test-roc-auc-accuracy")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 4: message "separately" + raw-string regex ============

def fix_metricsop_auc_message():
    print("\n=== cadw/ops/metrics.py: roc_auc message includes 'separately' ===")
    p = Path("cadw/ops/metrics.py")
    s = p.read_text()

    old = '''        if any(m in AUC_METRICS for m in self.metrics):
            raise ValueError(
                "MetricsOp: 'roc_auc' is computed by CadwDataFrame.metrics "
                "via external_sort (multipass). Do not pass roc_auc "
                "directly to MetricsOp."
            )'''

    new = '''        if any(m in AUC_METRICS for m in self.metrics):
            raise ValueError(
                "MetricsOp: 'roc_auc' is computed separately by "
                "CadwDataFrame.metrics() via external_sort (multipass). "
                "Do not pass roc_auc directly to MetricsOp. "
                "See §13.4/166."
            )'''

    s = replace_line_block(s, old, new, tag="metricsop-auc-message")
    p.write_text(s)
    compile_check(p)
    print("  OK")


def fix_test_regex_syntaxwarning():
    print("\n=== tests/test_metrics.py: raw-string regex (silence warning) ===")
    p = Path("tests/test_metrics.py")
    s = p.read_text()

    old = '        with pytest.raises(ValueError, match="len\\(y_proba\\)"):'
    new = '        with pytest.raises(ValueError, match=r"len\\(y_proba\\)"):'

    if old not in s:
        print("  NOT FOUND (already raw?) — skip")
        return
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 5. §13.4/167 + CHANGELOG addendum ============

def append_decisions_167():
    print("\n=== DECISIONS.md: §13.4/167 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 167" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/167 (partial-migration debt: dual field left "
        s += "active).** v092c introduced y_proba_cols; validation "
        s += "checked legacy y_proba_col.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 167 (partial migration: dual field left active; 5th
test-authoring).** v0.9.2c: 14 test failures, 4 root causes — 2 code
bugs (same class), 1 test-authoring (§13.4/166 protocol not applied),
1 message-regex mismatch.

**Bug 1 — partial migration of `y_proba_col` → `y_proba_cols`:**
v0.9.2c added `y_proba_cols: list[str]` (unified binary + multiclass).
Kept `y_proba_col: str | None` (legacy, backward compat). In
`__init__`, validation checked **legacy** `y_proba_col`:
```python
if self._needs_proba and y_proba_col is None:  # WRONG
```
Callers passing `y_proba_cols=[...]` but not `y_proba_col` get
`y_proba_col=None` → false "y_proba_col required". 6 tests fail.
Fix: `self.y_proba_cols is None`.

**Bug 2 — facade generic column loop not updated after type change:**
v0.9.2a `y_proba: str | None` (scalar). v0.9.2c `y_proba:
str | Sequence[str] | None` (can be list). Facade column-existence
loop still iterates `(y_proba, "y_proba")` and checks
`if y_proba not in self._columns`. For `y_proba=['p0','p1','p2']` →
`['p0','p1','p2'] not in ['y','p0','p1','p2']` = True → KeyError.
Specific per-column check already exists below — remove y_proba
from generic loop.

**Bug 3 — 5th test-authoring mistake in session (§13.4/166):**
`test_metrics_roc_auc_with_accuracy` expected 0.875 manually.
Correct AUC = 1.0 (all 4 pos/neg pairs correctly ordered).
The test-authoring chain: 128 (rolling) → 163 (log_loss semantic)
→ 164 (precision FP) → 166 (roc_auc prototype) → **167 (roc_auc mixed
test)**. Same anti-pattern. Fix: import `expected_roc_auc` helper.
Test was written BEFORE helper was imported — should've
derived expected from helper at authoring time.

**Bug 4 — message-regex mismatch:** message "computed by
CadwDataFrame.metrics via external_sort" vs test regex
`"roc_auc.*separately"`. Fix message: add "separately" — clearer
intent, matches test expectation.

**Правило (§13.4/167, расширение 149, 152, 160, 161):**
Когда поле X **deprecated** в пользу X_new:
- (a) **Все** validation checks — на X_new. Deprecated X — **только**
  для input compat (set from X_new[0] если X_new not None). Никогда
  не validation target.
- (b) **Все** generic loops / column-existence checks — обновить
  после type changes (str → str | list[str]).
- (c) **Test** на mixed-types (scalar + list) — обязательны при
  расширении типа. Facade y_proba (scalar) tests v0.9.2a **не**
  ловили list case.

**Связанные:** §13.4/149 (validation add/rename), /152 (stale
behavior), /160 (stale regex round 5), /161 (AST substring),
/166 (test-authoring).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/167")


def append_changelog_fix1():
    print("\n=== CHANGELOG.md: append v0.9.2c-fix1 note ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.2c-fix1" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Исправлено (v0.9.2c-fix1 — partial migration + test-authoring)

- **`MetricsOp.__init__`** — проверка `y_proba_cols` (unified) вместо
  legacy `y_proba_col`. v0.9.2c добавил `y_proba_cols` (list для
  multiclass/binary), но `if self._needs_proba and y_proba_col is None:`
  остался на legacy поле — все вызовы с `y_proba_cols=[...]` (без
  `y_proba_col`) получали false "y_proba_col required". 6 failures.
- **`CadwDataFrame.metrics`** — generic column-existence loop больше
  не включает `y_proba` (может быть list после v0.9.2c). Specific
  per-column check (after `y_proba_cols` normalization) уже был
  корректным. 6 failures.
- **`test_metrics_roc_auc_with_accuracy`** — expected AUC из
  `expected_roc_auc` helper, не manual 0.875 (правильно 1.0).
  **5-я** test-authoring ошибка в сессии (§13.4/128 → 163 → 164 →
  166 → 167).
- **`MetricsOp`** — message: "'roc_auc' is computed **separately**
  by CadwDataFrame.metrics()" (test regex `"roc_auc.*separately"`
  matches; clearer intent).
- **`tests/test_metrics.py`** — `match=r"len\\(y_proba\\)"` (raw
  string; silence SyntaxWarning).

### §13.4/167 — partial-migration pattern

X → X_new (deprecated):
- validation checks — **all** on X_new;
- generic loops / type-sensitive checks — **updated** after type
  widening (`str` → `str | list[str]`);
- **mixed-type tests** (scalar + list) — обязательны при расширении
  типа; v0.9.2a scalar tests не ловили list case.

'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.2c-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    # NOTE: NOT checking git status — v0.9.2c edits are present
    # (uncommitted); fix1 edits + v0.9.2c commit together.

    fix_metricsop_yproba_check()
    fix_facade_column_loop()
    fix_test_roc_auc_with_accuracy()
    fix_metricsop_auc_message()
    fix_test_regex_syntaxwarning()
    append_decisions_167()
    append_changelog_fix1()

    print("\n=== pytest: metrics fast (was 14 failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_metrics.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** METRICS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: metrics_reference ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_metrics_reference.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REFERENCE FAILED ***")
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

    print("\n=== pytest: reduce + sort regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_reduce.py", "tests/test_sort.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REDUCE/SORT REGRESSION FAILED ***")
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
        "v0.9.2c: roc_auc (binary + macro, sort-based per-threshold) + "
        "multiclass log_loss; MetricsOp rejects roc_auc (dispatched "
        "separately by facade); tests/_helpers/metrics_reference.py; "
        "§13.4/166 (np.trapezoid; ties; prototype reference); "
        "§13.4/167 (partial migration: y_proba_cols unified check; "
        "facade y_proba list; test-authoring 5th case)"])

    print("\n==> v0.9.2c applied (v092c + fix1 combined).")


if __name__ == "__main__":
    main()