#!/usr/bin/env python3
"""v0.9.2c-fix2: classification setup for multiclass log_loss (§13.4/168).

BUG (§13.4/168, same partial-migration class as /167):
Multiclass log_loss uses self._cls_to_idx for y_true → proba column
index. But classification setup block (that populates _classes and
_cls_to_idx) was gated on `if self._needs_classification:`. Pure
`metrics=["log_loss"]` with y_proba=[p0,p1,p2] sets
`_needs_classification=False` → `_cls_to_idx={}` → every y_true label
collected as "unknown" → ValueError.

FIX 1: gate also includes `_needs_multiclass_proba`. Multiclass
log_loss needs _classes/_cls_to_idx even without accuracy/etc.

FIX 2 (§3.11 honest peak): batch_memory_factor → 8.0 also for
multiclass proba. .to_pylist() on K+1 columns × batch rows — Python
object overhead. Prior 1.0 was dishonest for pure multiclass log_loss.

FIX 3 (defensive): _needs_regression + _needs_multiclass_proba →
ValueError. Domain conflict: regression y_true continuous; multiclass
log_loss y_true = class index (0..K-1). Mutually exclusive.

FIX 4 (test regex, fix1 missed): correct 4-space indent (was 8);
use raw string to silence SyntaxWarning.

GUARDS: metrics fast + E2E + reduce + full suite + guards.

Run: uv run python scripts/patch_v092c_fix2.py
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


# ============ Fix 1+3: classification setup gate + regression conflict ============

def fix_classification_gate():
    print("\n=== cadw/ops/metrics.py: gate includes multiclass proba ===")
    p = Path("cadw/ops/metrics.py")
    s = p.read_text()

    # Add regression + multiclass-proba conflict check (FIX 3),
    # and gate the classification setup block (FIX 1).

    old_conflict_anchor = '''        # §13.4/164: regression (continuous y_pred) and classification
        # (labels y_pred) can't share the same column.
        if self._needs_regression and self._needs_classification:
            raise ValueError(
                f"MetricsOp: regression ({sorted(ms & REGRESSION_METRICS)}) "
                f"and classification ({sorted(ms & CLASSIFICATION_METRICS)}) "
                f"cannot share y_pred column — one expects continuous "
                f"values, the other labels. Pass distinct calls."
            )'''

    new_conflict_anchor = '''        # §13.4/164: regression (continuous y_pred) and classification
        # (labels y_pred) can't share the same column.
        if self._needs_regression and self._needs_classification:
            raise ValueError(
                f"MetricsOp: regression ({sorted(ms & REGRESSION_METRICS)}) "
                f"and classification ({sorted(ms & CLASSIFICATION_METRICS)}) "
                f"cannot share y_pred column — one expects continuous "
                f"values, the other labels. Pass distinct calls."
            )
        # §13.4/168 (defensive): regression (continuous y_true) +
        # multiclass proba (y_true = class index 0..K-1) — contradictory
        # y_true semantics. Mutually exclusive.
        if self._needs_regression and self._needs_multiclass_proba:
            raise ValueError(
                f"MetricsOp: regression ({sorted(ms & REGRESSION_METRICS)}) "
                f"uses y_true as continuous target; multiclass y_proba "
                f"(len={len(self.y_proba_cols)}) uses y_true as class "
                f"index (0..K-1). Contradictory domain contracts — "
                f"cannot mix. Use separate calls."
            )'''

    s = replace_line_block(s, old_conflict_anchor, new_conflict_anchor,
                            tag="metricsop-regression-multiclass-conflict")

    # FIX 1: classification setup gate
    old_gate = '''        if self._needs_classification:
            if classes is None:
                # K=2 default
                self._classes = [0, 1]
            else:
                self._classes = list(classes)'''

    new_gate = '''        # §13.4/168: multiclass log_loss also needs _classes + _cls_to_idx
        # (y_true label → proba column index). Prior gate `if
        # self._needs_classification:` left them empty for pure
        # `metrics=["log_loss"]` → all labels treated as "unknown".
        _needs_class_setup = (
            self._needs_classification
            or (self._needs_proba and self._needs_multiclass_proba)
        )
        if _needs_class_setup:
            if classes is None:
                # K=2 default
                self._classes = [0, 1]
            else:
                self._classes = list(classes)'''

    s = replace_line_block(s, old_gate, new_gate, tag="classification-setup-gate")

    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 2: batch_memory_factor for multiclass proba ============

def fix_batch_memory_factor():
    print("\n=== cadw/ops/metrics.py: batch_memory_factor multiclass ===")
    p = Path("cadw/ops/metrics.py")
    s = p.read_text()

    old = '''    def batch_memory_factor(self) -> float:
        """§13.4/164: classification uses .to_pylist() per batch
        (~7× Python-object overhead). reduce() checks batch with this
        factor — honest peak RAM, not hidden."""
        return 8.0 if self._needs_classification else 1.0'''

    new = '''    def batch_memory_factor(self) -> float:
        """§13.4/164: classification uses .to_pylist() per batch
        (~7× Python-object overhead). reduce() checks batch with this
        factor — honest peak RAM, not hidden.

        §13.4/168: multiclass log_loss also does .to_pylist() on K+1
        columns (y_true + K proba cols). Same overhead class.
        """
        if self._needs_classification:
            return 8.0
        if self._needs_proba and self._needs_multiclass_proba:
            return 8.0
        return 1.0'''

    s = replace_line_block(s, old, new, tag="batch-memory-factor")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 4: test regex raw-string (correct 4-space indent) ============

def fix_test_regex():
    print("\n=== tests/test_metrics.py: raw-string regex (4-space indent) ===")
    p = Path("tests/test_metrics.py")
    s = p.read_text()

    old = '    with pytest.raises(ValueError, match="len\\(y_proba\\)"):'
    new = '    with pytest.raises(ValueError, match=r"len\\(y_proba\\)"):'

    if new in s and old not in s:
        print("  already raw — skip")
        return
    if old not in s:
        print(f"  WARNING: old pattern not found (4-space indent). "
              f"Searching 8-space fallback.")
        old = '        with pytest.raises(ValueError, match="len\\(y_proba\\)"):'
        new = '        with pytest.raises(ValueError, match=r"len\\(y_proba\\)"):'
        if old not in s:
            print("  NOT FOUND — skip (regex may be elsewhere)")
            return

    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 5: contract test for pure multiclass log_loss ============

def add_contract_tests():
    print("\n=== tests/test_metrics.py: contract tests for fix2 ===")
    p = Path("tests/test_metrics.py")
    s = p.read_text()
    if "test_metrics_multiclass_log_loss_no_classification_metrics" in s:
        print("  already present — skip")
        return

    new_tests = '''

# ============ v0.9.2c-fix2: contract tests (§13.4/168) ============

def test_metrics_multiclass_log_loss_no_classification_metrics(tmp_path):
    """Pure multiclass log_loss (no accuracy/etc.) populates _classes.

    §13.4/168: prior gate `if self._needs_classification:` left
    _cls_to_idx empty → all y_true labels treated as "unknown".
    """
    df = make_df(tmp_path, {
        "y": [0, 1, 2],
        "p0": [0.9, 0.1, 0.2],
        "p1": [0.05, 0.7, 0.2],
        "p2": [0.05, 0.2, 0.6],
    })
    res = df.metrics(
        y_true="y", y_proba=["p0", "p1", "p2"],
        metrics=["log_loss"], classes=[0, 1, 2],
    )
    assert res["log_loss"] > 0
    assert not math.isnan(res["log_loss"])


def test_metrics_multiclass_log_loss_with_regression_conflict(tmp_path):
    """§13.4/168 defensive: regression + multiclass proba → ValueError.

    r2 (continuous y_true) vs multiclass log_loss (y_true = class
    index 0..K-1) — contradictory domain contracts.
    """
    df = make_df(tmp_path, {
        "y": [0, 1, 2], "yp": [0.1, 0.5, 0.9],
        "p0": [0.9, 0.1, 0.2],
        "p1": [0.05, 0.7, 0.2],
        "p2": [0.05, 0.2, 0.6],
    })
    with pytest.raises(ValueError, match="Contradictory domain contracts"):
        df.metrics(
            y_true="y", y_pred="yp",
            y_proba=["p0", "p1", "p2"],
            metrics=["r2", "log_loss"], classes=[0, 1, 2],
        )


def test_metrics_multiclass_log_loss_k2_classes(tmp_path):
    """K=2 with explicit classes + multiclass proba list works.

    Binary in multiclass representation (y_proba=[p0, p1]).
    """
    df = make_df(tmp_path, {
        "y": [0, 1, 0, 1],
        "p0": [0.9, 0.1, 0.8, 0.2],
        "p1": [0.1, 0.9, 0.2, 0.8],
    })
    res = df.metrics(
        y_true="y", y_proba=["p0", "p1"],
        metrics=["log_loss"], classes=[0, 1],
    )
    assert res["log_loss"] > 0
    assert not math.isnan(res["log_loss"])


def test_metrics_batch_memory_factor_multiclass():
    """§13.4/168: batch_memory_factor = 8.0 for multiclass proba.

    .to_pylist() on K+1 columns × batch rows → Python-object overhead.
    Prior 1.0 was dishonest for pure multiclass log_loss.
    """
    from cadw.ops.metrics import MetricsOp
    op = MetricsOp(
        "y", y_proba_cols=["p0", "p1", "p2"],
        metrics=["log_loss"], classes=[0, 1, 2],
    )
    assert op.batch_memory_factor() == 8.0
'''
    if not s.endswith("\n"):
        s += "\n"
    s += new_tests
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 6: DECISIONS §13.4/168 ============

def append_decisions_168():
    print("\n=== DECISIONS.md: §13.4/168 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 168" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/168 (classification setup gate omitted multiclass "
        s += "log_loss).** _cls_to_idx empty for pure log_loss metrics.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 168 (classification setup gate omitted multiclass
log_loss; same partial-migration class as /167).** v0.9.2c-fix1: 2
tests fail — `multiclass_log_loss` and `multiclass_log_loss_known`.
Both: `ValueError: log_loss (multiclass) y_true has labels not in
classes []: [0, 1, 2]`.

**Bug:** multiclass log_loss в `update()` мапит y_true → proba column
index через `self._cls_to_idx`. Этот dict заполняется в
classification setup block, gated на `if self._needs_classification:`.
Pure `metrics=["log_loss"]` → `_needs_classification = False` →
`_cls_to_idx = {}` → все y_true labels уходят в `missing_yt`.

**Тот же partial-migration класс, что §13.4/167:** новое feature
(multiclass log_loss) добавлено, но gate классического блока не
расширен.

**Fix:**
- Gate: `_needs_classification OR (_needs_proba AND
  _needs_multiclass_proba)`. Multiclass log_loss использует
  `_cls_to_idx` — значит, нужна та же инициализация.
- `batch_memory_factor` → 8.0 для multiclass proba (`.to_pylist()`
  на K+1 columns; §3.11 honest peak).
- Defense: `_needs_regression AND _needs_multiclass_proba` → ValueError.
  Domain conflict: regression y_true continuous; multiclass log_loss
  y_true = class index. Mutually exclusive.
- Tests: (a) pure multiclass log_loss без classification metrics;
  (b) regression + multiclass conflict; (c) K=2 multiclass form;
  (d) batch_memory_factor contract.

**Правило (§13.4/168, расширение 167):** если feature F использует
internal state S, а S инициализируется под gate G — расширить G при
добавлении F, **даже если семантически F не входит в группу G.**
«Multiclass log_loss — не classification metric» истина по имени, но
**ложна по shared state**. Naming ≠ state-sharing.

**Backlog (§11.6):** audit всех internal-state gates после
v0.9.2 series:
- `_cls_to_idx` — classification + multiclass log_loss (fixed).
- `_classes` — same (fixed).
- `_k` — same (fixed; K-only state; also used by classification
  fail-fast `K×24+64 > state_limit`).
- `tp/fp/fn` — classification only (correct; log_loss O(1) state).
- `sum_log_loss/n_proba` — proba only (correct; binary + multiclass
  share).

**Связанные:** §13.4/167 (partial migration: y_proba_cols unified
check), /148 (design vs validation), /162 (redundant check masks
specific), /165 (gate by needs not presence).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/168")


# ============ Fix 7: CHANGELOG addendum ============

def append_changelog_fix2():
    print("\n=== CHANGELOG.md: append v0.9.2c-fix2 note ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.2c-fix2" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Исправлено (v0.9.2c-fix2 — multiclass log_loss gate)

- **`MetricsOp.__init__`** — classification setup block
  (`_classes`/`_cls_to_idx`/`_k`) gated на
  `_needs_classification OR (_needs_proba AND _needs_multiclass_proba)`.
  Pure `metrics=["log_loss"]` + multiclass `y_proba=[p0,p1,p2]` раньше
  оставлял `_cls_to_idx={}` → все y_true метки считались
  "неизвестными". 2 failures.
- **`MetricsOp.batch_memory_factor`** — 8.0 для multiclass proba
  (`.to_pylist()` на K+1 columns — §3.11 honest peak).
- **`MetricsOp.__init__`** — defensive: `_needs_regression +
  _needs_multiclass_proba` → ValueError (domain conflict: regression
  y_true continuous vs multiclass log_loss y_true class index).
- **`test_metrics.py`** — raw-string regex (4-space indent; v0.9.2c
  fix1 had 8-space, didn't match); 4 new contract tests
  (§13.4/168).

### §13.4/168 — partial migration: state-sharing ≠ naming

Feature F использует internal state S; S инициализируется под gate G.
Добавление F требует расширения G, даже если F не входит в
семантическую группу G. Пример: multiclass log_loss — не
classification metric (имя), но использует `_cls_to_idx` (state).

'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.2c-fix2 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_classification_gate()
    fix_batch_memory_factor()
    fix_test_regex()
    add_contract_tests()
    append_decisions_168()
    append_changelog_fix2()

    print("\n=== pytest: metrics fast (was 2 failing) ===")
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
        "facade y_proba list; test-authoring 5th case); "
        "§13.4/168 (multiclass log_loss _cls_to_idx; batch_memory_"
        "factor; regression+multiclass conflict)"])

    print("\n==> v0.9.2c applied (v092c + fix1 + fix2 combined).")


if __name__ == "__main__":
    main()