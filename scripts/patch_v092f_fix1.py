#!/usr/bin/env python3
"""v0.9.2f-fix1: standard_scaler_unseen_data — deterministic + reference
(§13.4/171, 7th test-authoring in session).

BUG A (manual arithmetic): test comment said "10.0/2.0 = 5.0", but
standardization gives (mean - mean_train)/std_train = 0, not mean/std.
Assertion `abs(mean - 5.0) < 0.5` was arithmetically wrong; actual
output mean ≈ -0.036 (correct).

BUG B (test doesn't distinguish frozen-vs-refit): both x_train and
x_test used rng.normal(loc=10.0, scale=2.0) — SAME distribution.
Frozen stats and refit-on-test give same output mean ≈ 0. Test could
not prove what its name claims ("unseen_data → frozen stats").

FIX:
  - Add tests/_helpers/scaler_reference.py — reference implementations
    for all 4 scalers (StandardScaler.transform(x_train, x_test),
    etc.). Reference-based pattern (§13.4/164-167 discipline: manual
    arithmetic in test bodies prohibited).
  - Replace test with deterministic data:
      x_train = [8.0, 12.0]  → mean=10, std(ddof=0)=2 (exact)
      x_test  = [12.0]       → frozen: (12-10)/2 = 1.0
                              refit:  (12-12)/1 = 0.0
    Frozen gives 1.0 (proves), refit gives 0.0 (would fail).
  - Assert exact 1.0 via reference helper.
  - §13.4/171 lesson: deterministic data + reference helper for
    exact-property tests; RNG only for statistical assertions
    ("does it converge?"), not "which stats were used?".

GUARDS: scalers fast + E2E + pipeline + full suite + guards.

Run: uv run python scripts/patch_v092f_fix1.py
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


# ============ 1. Reference helper ============

SCALER_REFERENCE = '''"""Reference scalers for tests (§13.4/164, /171).

Derived from sklearn definitions. Test bodies must use these — not
manual arithmetic.

Pattern (§13.4/171): for EXACT-property tests (e.g., "does transform
use FROZEN train stats?"), use deterministic data + reference helper.
RNG + statistical assertion only for CONVERGENCE tests ("does the
model learn?"), not "which stats were used?" — different distributions
are needed to distinguish frozen-vs-refit.
"""
from __future__ import annotations

import numpy as np


def _to_f64(xs) -> np.ndarray:
    return np.asarray(xs, dtype=np.float64)


def standard_scaler_transform(
    x_train, x_test, *, with_mean: bool = True, with_std: bool = True,
) -> np.ndarray:
    """(x - mean_train) / std_train (ddof=0). Degenerate std=0 → scale=1."""
    xtr = _to_f64(x_train)
    xte = _to_f64(x_test)
    mean = xtr.mean()
    var = xtr.var(ddof=0)
    std = float(np.sqrt(var)) if var > 0 else 1.0
    out = xte.copy()
    if with_mean:
        out = out - mean
    if with_std:
        out = out / std
    return out


def minmax_scaler_transform(
    x_train, x_test, *, feature_range=(0.0, 1.0),
) -> np.ndarray:
    """(x - min_train) * (high-low)/(max-min) + low. Constant → low."""
    xtr = _to_f64(x_train)
    xte = _to_f64(x_test)
    mn = xtr.min()
    mx = xtr.max()
    low, high = feature_range
    span = high - low
    data_range = mx - mn
    if data_range > 0:
        scale = span / data_range
        out = (xte - mn) * scale + low
    else:
        out = np.full_like(xte, low)
    return out


def maxabs_scaler_transform(x_train, x_test) -> np.ndarray:
    """x / max(|min_train|, |max_train|). All-zero → scale=1."""
    xtr = _to_f64(x_train)
    xte = _to_f64(x_test)
    max_abs = float(max(abs(xtr.min()), abs(xtr.max())))
    if max_abs == 0.0:
        max_abs = 1.0
    return xte / max_abs


def robust_scaler_transform(x_train, x_test) -> np.ndarray:
    """(x - median_train) / IQR_train. IQR=0 → scale=1."""
    xtr = _to_f64(x_train)
    xte = _to_f64(x_test)
    q25, q50, q75 = np.percentile(xtr, [25, 50, 75])
    iqr = float(q75 - q25)
    if iqr == 0.0:
        iqr = 1.0
    return (xte - q50) / iqr
'''


def write_scaler_reference():
    print("\n=== tests/_helpers/scaler_reference.py ===")
    p = Path("tests/_helpers/scaler_reference.py")
    p.write_text(SCALER_REFERENCE)
    compile_check(p)
    print("  OK")


# ============ 2. Replace broken test ============

def fix_unseen_test():
    print("\n=== tests/test_scalers.py: deterministic unseen test ===")
    p = Path("tests/test_scalers.py")
    s = p.read_text()

    old = '''def test_standard_scaler_unseen_data(tmp_path):
    rng = np.random.default_rng(42)
    x_train = rng.normal(loc=10.0, scale=2.0, size=200)
    x_test = rng.normal(loc=10.0, scale=2.0, size=100)
    df_train = make_df(tmp_path, {"x": x_train.tolist()}, name="train")
    df_test = make_df(tmp_path, {"x": x_test.tolist()}, name="test")
    s = StandardScaler()
    s.fit(df_train)
    out = s.transform(df_test)
    t = read_table(out, tmp_path)
    xs = np.array(_col(t, "x"), dtype=np.float64)
    # Test mean close to train mean (10.0/2.0 = 5.0) but not exactly
    assert abs(float(xs.mean()) - 5.0) < 0.5'''

    new = '''def test_standard_scaler_unseen_data(tmp_path):
    """Transform uses FROZEN train stats, not refit on test (§13.4/171).

    Deterministic data + reference helper. Prior test had 2 bugs:
    (a) same-distribution train/test can't distinguish frozen vs refit
        (both give output mean ≈ 0);
    (b) manual arithmetic `abs(mean - 5.0) < 0.5` was wrong —
        (mean - mean_train)/std = 0, not mean/std = 5.

    Setup: x_train = [8, 12] → mean=10, std(ddof=0)=2 (exact).
           x_test  = [12] → frozen: (12-10)/2 = 1.0
                            refit:  (12-12)/1 = 0.0 (would fail)
    """
    from tests._helpers.scaler_reference import standard_scaler_transform

    x_train = [8.0, 12.0]
    x_test = [12.0]
    df_train = make_df(tmp_path, {"x": x_train}, name="train")
    df_test = make_df(tmp_path, {"x": x_test}, name="test")
    s = StandardScaler()
    s.fit(df_train)
    out = s.transform(df_test)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    expected = standard_scaler_transform(x_train, x_test)
    assert len(xs) == 1
    assert abs(xs[0] - expected[0]) < 1e-9, (
        f"frozen-stats output: got {xs[0]:.6f}, expected "
        f"{expected[0]:.6f}. If ~0.0, transform refit on test (BUG)."
    )
    # Specific: (12 - 10) / 2 = 1.0
    assert abs(xs[0] - 1.0) < 1e-9'''

    s = replace_line_block(s, old, new, tag="unseen-test-deterministic")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 3. §13.4/171 lesson ============

def append_decisions_171():
    print("\n=== DECISIONS.md: §13.4/171 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 171" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/171 (test-authoring #7).** StandardScaler unseen "
        s += "test used RNG + wrong manual arithmetic.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 171 (test-authoring #7; deterministic-vs-statistical
test pattern).** v0.9.2f-fix1: 1 failure
`test_standard_scaler_unseen_data`. Two bugs in same test:

**Bug A (manual arithmetic, §13.4/128-class):** comment said
«Test mean close to train mean (10.0/2.0 = 5.0)», assertion
`abs(mean - 5.0) < 0.5`. But standardization is
`(x - mean_train) / std_train` → output mean = **0**, not `mean/std = 5`.
Actual result `mean ≈ -0.036` (correct). Manual arithmetic wrong.

**Bug B (test doesn't test what name claims):** `x_train` and
`x_test` both `rng.normal(loc=10.0, scale=2.0)`. Frozen-stats transform
and refit-on-test both give output mean ≈ 0 (trivially, since
distribution matches). Test called `unseen_data` but couldn't
distinguish frozen-vs-refit. Property under test ("which stats were
used?") **requires different distributions** to distinguish.

**Fix:**
- `tests/_helpers/scaler_reference.py` — reference transforms for
  all 4 scalers. Test bodies use reference (never manual arithmetic).
- Deterministic data: `x_train = [8, 12]` → mean=10, std=2 exactly.
  `x_test = [12]` → frozen: `(12-10)/2 = 1.0`, refit: `(12-12)/1 = 0.0`.
  Frozen gives 1.0, refit gives 0.0. Distinguishable. Exact assertion.

**Правило (§13.4/171, расширение 128, 163, 164, 166, 167):**

1. **For EXACT-property tests, use deterministic data.** The property
   «does transform use frozen stats?» is exact (which stats were used).
   RNG washes out the signal. Deterministic data + reference helper
   gives exact assertion. RNG + statistical tolerance only for
   «does the model converge/learn?» — not «which stats were used?».

2. **Test invariants, not derivable values.** «Output mean ≈ 0» is
   derivable from StandardScaler invariant (`(x-mean)/std` of the
   training set has mean 0). «Output mean ≈ 5.0 because mean/std = 5»
   is not derivable — it's manual arithmetic that happened to be wrong.

3. **Reference helper for every scaler transform.** Once one test
   derives value manually, every test needs same reference. Codify
   in `tests/_helpers/scaler_reference.py` (tested separately via
   scalers self-tests).

4. **7th test-authoring error in session** (128 → 163 → 164 → 166
   → 167 → 169 → 171). Pattern: manual arithmetic + RNG where
   exact-property test needed. The `tests/_helpers/*_reference.py`
   discipline (started §13.4/164 with `metrics_reference.py`,
   extended §13.4/169 with `rolling_reference.py`, now §13.4/171
   with `scaler_reference.py`) is the systematic fix: every test
   that can use a reference helper — must.

**Backlog (§11.6):**
- All scaler tests (once they exist) → import from `scaler_reference`.
- When GBDT ranking/quantile-loss tests are written → reference helper
  per loss.
- When TimeSeriesFeatures (composite) tests are written →
  `ts_features_reference.py`.

**Связанные:** §13.4/128 (rolling nulls), /163 (log_loss semantic),
/164 (precision FP; metrics_reference.py created), /166 (roc_auc
prototype), /167 (dual proba field), /169 (TSS+GBDT test data).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/171")


def append_changelog_fix1():
    print("\n=== CHANGELOG.md: v0.9.2f-fix1 note ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.2f-fix1" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Исправлено (v0.9.2f-fix1 — test-authoring #7)

- **`test_standard_scaler_unseen_data`** — deterministic data +
  reference helper. Prior test had 2 bugs:
  (a) same-distribution train/test — frozen-vs-refit indistinguishable
      (both output mean ≈ 0);
  (b) manual arithmetic `abs(mean - 5.0) < 0.5` — wrong: standardized
      mean = 0, not mean/std = 5.
  New: `x_train = [8, 12]`, `x_test = [12]` — frozen gives exactly
  1.0, refit gives 0.0. Exact assertion via reference.
- **`tests/_helpers/scaler_reference.py`** (new) — reference transforms
  for Standard/MinMax/MaxAbs/Robust. Codifies invariant
  `(x - mean_train) / std_train` per scaler; test bodies import
  reference, not manual arithmetic (§13.4/164 discipline extended
  to scalers).

### §13.4/171 — deterministic-vs-statistical test pattern

**For EXACT-property tests** (e.g., «does transform use frozen stats?»):
deterministic data + reference helper + exact assertion. RNG washes
out the property.

**For CONVERGENCE tests** («does the model learn?»): RNG + statistical
tolerance is fine.

7th test-authoring error in session (128 → 163 → 164 → 166 → 167 →
169 → 171). Systematic fix: reference helper per domain
(`metrics_reference.py`, `rolling_reference.py`, `scaler_reference.py`,
future `ts_features_reference.py`).

'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.2f-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    # NOTE: working tree dirty from v0.9.2f (patch aborted before commit
    # after test failure). Fix1 + v0.9.2f edits commit together.

    write_scaler_reference()
    fix_unseen_test()
    append_decisions_171()
    append_changelog_fix1()

    print("\n=== pytest: scalers fast (was 1 failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_scalers.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** SCALERS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: scalers E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_scalers_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline + cv regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_cv.py",
         "tests/test_register_step_type.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
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
                       timeout=1800)
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
        "v0.9.2f: scalers — StandardScaler/MinMaxScaler/MaxAbsScaler/"
        "RobustScaler; composed from Reduce+quantile; auto-registered "
        "with Pipeline; tests/_helpers/scaler_reference.py; §13.4/170 "
        "(composition, no prototype; AST whitelist over-broad); "
        "§13.4/171 (test-authoring #7: deterministic-vs-statistical; "
        "reference helper for exact-property tests)"])

    print("\n==> v0.9.2f applied (v092f + fix1 combined).")
    print("==> v0.9.2 COMPLETE (metrics + CV + scalers).")
    print("NEXT: git tag v0.9.2")


if __name__ == "__main__":
    main()