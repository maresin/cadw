#!/usr/bin/env python3
"""v0.9.2b-fix1: complete v0.9.2b (3 fails + 4 warnings, all in v092b).

STATE: v0.9.2b ran, edits applied (8 dirty files), 3 tests failed.
Commit not executed. This fix1 handles all 3 + RuntimeWarnings + commits
everything as one v0.9.2b.

BUGS (from v0.9.2b test run):
  1. test_metrics_precision_binary_pos_label_0 (test-authoring, §13.4/164):
     test comment claimed FP=0 for class 0, actual FP=1 (row 1 y=1,yp=0).
     Precision=1/2=0.5, not 1.0. Code correct.
  2. test_metrics_classification_k_too_large_raises (missing import):
     MemoryLimitExceeded not imported in tests/test_metrics.py.
  3. test_metrics_classification_string_labels (design, §13.4/164):
     pos_label=1 default; classes=['cat','dog'] → ValueError "not in
     classes". Fix: pos_label=None default → resolve classes[-1] (matches
     sklearn classes[1] for [0,1], resolves string classes friendly).
  4. RuntimeWarnings (division): `np.where(cond, a/b, 0)` evaluates a/b
     even when cond is False. Fix: np.divide(..., out=..., where=cond).

FIXES:
  1. test_precision_binary_pos_label_0: expected 0.5, correct comment.
  2. Add `from cadw.core.memory_guard import MemoryLimitExceeded`.
  3. MetricsOp + metrics(): pos_label: Optional = None; resolve to
     classes[-1] when None; validate explicit value.
  4. _finalize_classification: np.divide(..., out=..., where=...) for
     pp/pr/pf1.
  5. §13.4/164 — pos_label default ambiguity + test-authoring FP miss.

GUARDS: metrics fast + E2E + full suite + guards. Then commit
v0.9.2b + fix1 as one commit.

Run: uv run python scripts/patch_v092b_fix1.py
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


# ============ Fix 1: test expectation + comment ============

def fix_test_pos_label_0():
    print("\n=== tests/test_metrics.py: pos_label_0 expectation (0.5) ===")
    p = Path("tests/test_metrics.py")
    s = p.read_text()

    old = '''def test_metrics_precision_binary_pos_label_0(tmp_path):
    """pos_label=0: precision for class 0."""
    df = make_df(tmp_path, {
        "y": [1, 1, 1, 0, 0],
        "yp": [1, 0, 1, 1, 0],
    })
    # class 0: TP=1 (y=0,yp=0), FP=0, FN=1
    # precision(class0) = 1/(1+0) = 1.0
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["precision"], classes=[0, 1],
        average="binary", pos_label=0,
    )
    assert res["precision"] == pytest.approx(1.0)'''

    new = '''def test_metrics_precision_binary_pos_label_0(tmp_path):
    """pos_label=0: precision for class 0.

    y=[1,1,1,0,0], yp=[1,0,1,1,0]. For class 0:
      TP: (0,0) → row 4
      FP: (≠0, 0) → row 1 (y=1, yp=0)
      FN: (0, ≠0) → row 3 (y=0, yp=1)
    precision(0) = TP/(TP+FP) = 1/2 = 0.5 (§13.4/164: prior expected 1.0
    missed row 1 FP — test-authoring error, code correct).
    """
    df = make_df(tmp_path, {
        "y": [1, 1, 1, 0, 0],
        "yp": [1, 0, 1, 1, 0],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["precision"], classes=[0, 1],
        average="binary", pos_label=0,
    )
    assert res["precision"] == pytest.approx(0.5)'''

    s = replace_line_block(s, old, new, tag="fix-pos-label-0")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 2: missing import ============

def fix_missing_import():
    print("\n=== tests/test_metrics.py: import MemoryLimitExceeded ===")
    p = Path("tests/test_metrics.py")
    s = p.read_text()

    old = "from cadw import CadwDataFrame, MemoryGuard\n"
    new = ("from cadw import CadwDataFrame, MemoryGuard\n"
           "from cadw.core.memory_guard import MemoryLimitExceeded\n")

    if "from cadw.core.memory_guard import MemoryLimitExceeded" in s:
        print("  already present — skip")
        return

    if old not in s:
        raise SystemExit("import anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 3: pos_label Optional + resolve ============

def fix_pos_label_optional():
    print("\n=== cadw/ops/metrics.py: pos_label Optional + resolve ===")
    p = Path("cadw/ops/metrics.py")
    s = p.read_text()

    # Signature
    old_sig = "        pos_label: int = 1,\n    ) -> None:"
    new_sig = "        pos_label=None,\n    ) -> None:"
    if old_sig not in s:
        raise SystemExit("pos_label signature not found")
    s = s.replace(old_sig, new_sig, 1)

    # Resolve in classification setup
    old_resolve = '''            if self._average_method == "binary":
                if pos_label not in self._cls_to_idx:
                    raise ValueError(
                        f"MetricsOp: pos_label {pos_label!r} not in "
                        f"classes {self._classes}"
                    )'''

    new_resolve = '''            if self._average_method == "binary":
                # §13.4/164: pos_label=None default → resolve to
                # classes[-1] (matches sklearn's classes[1] for K=2
                # [0,1]; friendly for string classes).
                if pos_label is None:
                    self._pos_label = self._classes[-1]
                else:
                    if pos_label not in self._cls_to_idx:
                        raise ValueError(
                            f"MetricsOp: pos_label {pos_label!r} not "
                            f"in classes {self._classes}"
                        )
                    self._pos_label = pos_label
            else:
                self._pos_label = pos_label'''

    if old_resolve not in s:
        raise SystemExit("pos_label resolve block not found")
    s = s.replace(old_resolve, new_resolve, 1)

    # Remove the now-redundant self._pos_label = pos_label in __init__ preamble
    old_pre = '''        self._average_method: str = ""
        self._pos_label = pos_label'''
    new_pre = '''        self._average_method: str = ""
        self._pos_label = None'''
    if old_pre in s:
        s = s.replace(old_pre, new_pre, 1)

    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 4: divide warnings ============

def fix_divide_warnings():
    print("\n=== cadw/ops/metrics.py: np.divide(where=) no warnings ===")
    p = Path("cadw/ops/metrics.py")
    s = p.read_text()

    old = '''        # Per-class precision / recall / f1 (zero_division=0, sklearn)
        pp = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        pr = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        denom = pp + pr
        pf1 = np.where(denom > 0, 2.0 * pp * pr / denom, 0.0)'''

    new = '''        # Per-class precision / recall / f1 (zero_division=0, sklearn).
        # §13.4/164: np.divide(..., where=...) — avoids RuntimeWarning
        # from evaluating a/b on zero denominators inside np.where.
        pp = np.zeros_like(tp, dtype=np.float64)
        np.divide(tp, tp + fp, out=pp, where=(tp + fp) > 0)
        pr = np.zeros_like(tp, dtype=np.float64)
        np.divide(tp, tp + fn, out=pr, where=(tp + fn) > 0)
        denom = pp + pr
        pf1 = np.zeros_like(denom)
        np.divide(2.0 * pp * pr, denom, out=pf1, where=denom > 0)'''

    s = replace_line_block(s, old, new, tag="divide-where")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 5: dataframe.metrics pos_label Optional ============

def fix_facade_pos_label():
    print("\n=== cadw/dataframe.py::metrics — pos_label Optional ============")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    old_sig = '        pos_label: int = 1,\n    ) -> dict:'
    if old_sig not in s:
        raise SystemExit("facade pos_label signature not found")

    new_sig = '        pos_label=None,\n    ) -> dict:'
    s = s.replace(old_sig, new_sig, 1)

    # Update docstring note
    old_doc = '''            pos_label: positive class for average='binary'.'''
    new_doc = '''            pos_label: positive class for average='binary'. None →
                classes[-1] (K=2 default: [0,1]→1; string classes →
                last class). Explicit value must be in `classes`.'''

    if old_doc in s:
        s = s.replace(old_doc, new_doc, 1)

    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 6: §13.4/164 ============

def append_decisions_164():
    print("\n=== DECISIONS.md: §13.4/164 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 164" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/164 (pos_label default ambiguity; test-authoring FP "
        s += "miss).** pos_label=None → classes[-1]; test expectation 0.5 "
        s += "(not 1.0).\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 164 (pos_label default ambiguity + test-authoring FP
miss + np.where division warnings).** v0.9.2b: 3 test fails, 4
RuntimeWarnings.

**Bug 1 (test-authoring, §13.4/128-класс):**
`test_metrics_precision_binary_pos_label_0` — ручной расчёт:
y=[1,1,1,0,0], yp=[1,0,1,1,0]. Comment said «class 0: FP=0» — **wrong**.
Row 1 (y=1, yp=0) — это **FP для class 0** (модель предсказала 0, а
истина 1). Реально TP=1, FP=1, FN=1 → precision=0.5 (не 1.0).
**Code правильный**, test ожидал неверное. Класс — §13.4/128
(expectations authored, not derived).

**Bug 2 (missing import):** `MemoryLimitExceeded` не импортирован в
test_metrics.py.

**Bug 3 (design, §13.4/164):** `average='binary'` (auto для K=2)
требует `pos_label ∈ classes`. Default `pos_label=1` (int) не работает
для `classes=['cat','dog']`. Sklearn also fails (default int on string
y → error). Но API может быть **дружелюбнее**: `pos_label=None` →
`classes[-1]` — для `[0,1]` даёт `1` (= sklearn classes[1]), для
strings — last class. Explicit `pos_label` — validate.

**Bug 4 (warnings, latent):** `np.where((tp+fp)>0, tp/(tp+fp), 0)` —
`tp/(tp+fp)` evaluates for all elements, включая zeros → RuntimeWarning.
Fix: `np.divide(tp, tp+fp, out=pp, where=(tp+fp)>0)` — vectorized,
no warning. Same для pr, pf1.

**Урок (§13.4/164, усиление 128, 163):**
- **Manual arithmetic в test expectations** — 3-й случай в сессии
  (128 rolling, 163 log_loss semantic, 164 precision FP).
  Правило (уже было): reference helper или trace. Здесь применимо:
  `expected_precision(y_true, y_pred, classes, pos_label)` — helper.
- **pos_label default как int** — design assumption, ломается на
  strings. Optional=None + resolve pattern — consistent с sklearn
  (classes[1]), но friendlier.
- **np.where для zero-division** — anti-pattern. Всегда
  `np.divide(..., where=...)`.

**Связанные:** §13.4/128, /129, /163.

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/164")


# ============ main ============

def main():
    print("=== v0.9.2b-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    # NOTE: intentionally NOT checking git status — v0.9.2b edits are
    # present (dirty) from interrupted run. All v0.9.2b + fix1 edits
    # commit together.

    fix_test_pos_label_0()
    fix_missing_import()
    fix_pos_label_optional()
    fix_divide_warnings()
    fix_facade_pos_label()
    append_decisions_164()

    print("\n=== pytest: metrics fast (was 3 failing) ===")
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

    print("\n=== pytest: reduce regression (protocol) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_reduce.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REDUCE REGRESSION FAILED ***")
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
        "v0.9.2b: classification metrics — accuracy/precision/recall/f1 "
        "(binary + multiclass, average modes); MetricsOp tp/fp/fn state "
        "(24×K vs K² confusion); ReduceOp.batch_memory_factor protocol "
        "extension; §11.5 classification constraints; §13.4/164 (pos_label "
        "None→classes[-1]; test FP miss; np.divide where=)"])

    print("\n==> v0.9.2b applied (v092b + fix1 combined).")
    print("NEXT: v0.9.2c (roc_auc + multiclass log_loss).")


if __name__ == "__main__":
    main()