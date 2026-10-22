#!/usr/bin/env python3
# v0.9.25a-fix1 — _fit_one must accept sigma2 == 0.
#
# Bug: perfectly linear training data (e.g. y = 1..20 with d=1, D=1, s=2)
# produces twice-differenced series = all zeros → sigma2 = 0. Mathematically
# valid perfect fit, but _fit_one rejected (`if sigma2 <= 0`), so SARIMA
# could not fit on deterministic series.
#
# Surfaced by v0.9.25a dogfood (test_sarima_integrate_back_seasonal).
# v0.9.24 was lucky: every SARIMA test had D=0, so twice-differencing never
# happened and sigma2 was always > 0.
#
# Fix:
#   1. _fit_one: reject only sigma2 < 0 (or non-finite). Accept ==0.
#   2. AIC: log(max(sigma2, 1e-300)) → finite for sigma2=0 (log(0)=-inf
#      produced non-comparable -inf across candidates; still a valid "best"
#      signal but harder to reason about).
#
# v0.9.25a production changes were applied but NOT committed (dogfood failed
# before commit). This patch commits v0.9.25a + fix1 together.
#
# Run: uv run python scripts/patch_v0925a_fix1_sigma2_zero.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess,
)


# ============ 0. Verify v0.9.25a changes on disk ============

def verify_state():
    print("\n=== verify v0.9.25a changes on disk ===")
    src = Path("cadw/ts/models.py").read_text()
    checks = [
        "Undo seasonal differencing FIRST",        # v0.9.25a Bug 1
        "self._y_centered_ = y - self._y_mean",    # v0.9.25a Bug 2
        "if d or D:\n            return np.full(n, np.nan)",  # Bug 3
    ]
    for c in checks:
        if c not in src:
            raise SystemExit(
                f"cadw/ts/models.py missing v0.9.25a anchor: {c!r}. "
                f"Re-run v0.9.25a patch (its apply_or_fail may have "
                f"partially failed)."
            )

    if "if sigma2 <= 0 or not np.isfinite(sigma2):" in src:
        print("  v0.9.25a applied, sigma2<=0 check present (needs fix)")
    elif "if sigma2 < 0 or not np.isfinite(sigma2):" in src:
        print("  v0.9.25a + fix1 already applied")
    else:
        raise SystemExit(
            "sigma2 check anchor not found — unexpected state. "
            "Inspect cadw/ts/models.py _fit_one."
        )
    print("  OK")


# ============ 1. Fix _fit_one: accept sigma2 == 0 ============

OLD_FIT_ONE = '''        if sigma2 <= 0 or not np.isfinite(sigma2):
            return None, None, None, f"sigma2={sigma2!r}"
        n_valid = int(np.sum(~np.isnan(yd)))
        n_params = p + q + P + Q + 1
        aic = n_valid * np.log(sigma2) + 2 * n_params
        return (ar, ma, sigma2, yd), aic, sigma2, None
'''

NEW_FIT_ONE = '''        # v0.9.25a-fix1: allow sigma2 == 0 (perfect fit is valid —
        # e.g. exactly linear series with d=1,D=1 gives all-zero
        # twice-differenced and sigma2=0). Reject only negative or
        # non-finite. AIC uses log(max(sigma2, 1e-300)) so it stays
        # finite for sigma2=0 (log(0)=-inf produced non-comparable
        # -inf across candidates, even though -inf is a valid "best"
        # signal).
        if sigma2 < 0 or not np.isfinite(sigma2):
            return None, None, None, f"sigma2={sigma2!r}"
        n_valid = int(np.sum(~np.isnan(yd)))
        n_params = p + q + P + Q + 1
        sigma2_safe = max(sigma2, 1e-300)
        aic = n_valid * np.log(sigma2_safe) + 2 * n_params
        return (ar, ma, sigma2, yd), aic, sigma2, None
'''

def fix_fit_one():
    print("\n=== fix _fit_one: accept sigma2==0 ===")
    applied = apply_or_fail(
        "cadw/ts/models.py", OLD_FIT_ONE, NEW_FIT_ONE,
        label="fit_one_sigma2_zero",
    )
    print("  applied" if applied else "  already applied — skip")


# ============ 2. Add test pinning sigma2==0 case ============

NEW_TEST = '''

def test_sarima_perfect_linear_does_not_raise():
    """v0.9.25a-fix1: sigma2=0 (perfect fit) is valid, not an error.

    y = exactly linear, d=1, D=1 → twice-differenced = all zeros →
    sigma2 = 0. This surfaced in v0.9.25a dogfood: _fit_one rejected
    `sigma2 <= 0`, so SARIMA could not fit at all. Now accepted:
    perfect linear forecast = y[-1] + h (see integrate_back test).
    """
    y = list(np.arange(1.0, 21.0))
    df = _mk_df(y)
    m = SARIMAForecaster(
        order=(0, 1, 0), seasonal_order=(0, 1, 0), s=2, auto=False,
    ).fit(df, "y")
    fc = m.predict(3)
    assert np.all(np.isfinite(fc))
    assert m._sigma2 == 0.0
'''

def add_test():
    print("\n=== add test_sarima_perfect_linear_does_not_raise ===")
    p = Path("tests/test_ts_models.py")
    s = p.read_text()
    if "test_sarima_perfect_linear_does_not_raise" in s:
        print("  already present — skip")
        return
    p.write_text(s + NEW_TEST)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. Dogfood (clean, no dead code) ============

def dogfood():
    print("\n=== dogfood: 5 v0.9.25a invariants (clean) ===")

    def _verify():
        import numpy as _np
        import pyarrow as _pa
        from cadw import CadwDataFrame as _CDF
        from cadw.ts import SARIMAForecaster as _S

        def _mk(y):
            return _CDF.from_table(_pa.table({
                "y": _pa.array(list(y), type=_pa.float64()),
            }))

        # (1) D=1, s=2, d=1 on perfect linear [1..20] → [21, 22, 23]
        # (this is exactly the case that crashed sigma2=0)
        m = _S(
            order=(0, 1, 0), seasonal_order=(0, 1, 0), s=2, auto=False,
        ).fit(_mk(_np.arange(1.0, 21.0)), "y")
        fc = m.predict(3)
        assert _np.allclose(fc, [21.0, 22.0, 23.0], atol=1e-6), (
            f"Bug 1 (D→d order) + fix1 (sigma2=0) not closed: got {fc}"
        )
        assert m._sigma2 == 0.0, (
            f"expected sigma2=0 on perfect linear, got {m._sigma2}"
        )

        # (2) Same + mean=100 offset
        m = _S(
            order=(0, 1, 0), seasonal_order=(0, 1, 0), s=2, auto=False,
        ).fit(_mk(_np.arange(100.0, 120.0)), "y")
        fc = m.predict(3)
        assert _np.allclose(fc, [120.0, 121.0, 122.0], atol=1e-6), (
            f"Bug 1b (mean+D>0) not closed: got {fc}"
        )

        # (3) ARIMA(0,0,0) on mean=100 → ~100
        rng = _np.random.default_rng(42)
        m = _S(order=(0, 0, 0), auto=False).fit(
            _mk(100.0 + 0.5 * rng.normal(size=200)), "y",
        )
        fc = m.predict(5)
        assert all(80 < f < 120 for f in fc), (
            f"Bug 2 (no intercept) not closed: got {fc}"
        )

        # (4) predict_in_sample d=0: len=n, y-scale
        m = _S(order=(1, 0, 0), auto=False).fit(
            _mk(100.0 + rng.normal(size=50)), "y",
        )
        preds = m.predict_in_sample()
        assert len(preds) == 50, (
            f"Bug 3 (len) not closed: len={len(preds)}"
        )
        finite = preds[~_np.isnan(preds)]
        assert abs(finite.mean() - 100.0) < 5.0, (
            f"Bug 3 (scale) not closed: mean={finite.mean():.2f}"
        )

        # (5) predict_in_sample d=1 → all NaN
        m = _S(order=(0, 1, 0), auto=False).fit(
            _mk(_np.cumsum(rng.normal(size=50)) + 100.0), "y",
        )
        preds = m.predict_in_sample()
        assert len(preds) == 50 and _np.all(_np.isnan(preds)), (
            "d>0 predict_in_sample must be all-NaN"
        )

        print(
            "  5 invariants OK: D=1 order, D=1+mean, ARIMA(0,0,0) ~mean, "
            "in-sample len+scale, d>0 all-NaN; sigma2=0 accepted"
        )

    run_verify_in_subprocess(_verify)


# ============ 4. Pytest ============

def run_pytest():
    print("\n=== pre-flight: test_facade_has_no_as_py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py", "-q"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** FACADE AS_PY VIOLATION — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: targeted ts ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_ts_models.py", "tests/test_ts_thf.py",
         "tests/test_ts_backtest.py", "tests/test_ts_api.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 5. main ============

def main():
    print("=== v0.9.25a-fix1: _fit_one sigma2==0 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    fix_fit_one()
    assert_compiles(Path("cadw/ts/models.py"))
    add_test()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25a: SARIMA stabilization — 3 critical bugs + _fit_one "
        "sigma2==0 fix (Phase A of consolidation roadmap). "
        "Bug 1 (_integrate_back): reverse order D→d not d→D "
        "(_prep does d then D; integration must invert order). "
        "Bug 1b: seasonal_period_last from _difference(y_centered, d), "
        "not y_raw (wrong for d>0). "
        "Bug 2 (Hannan-Rissanen no intercept): center y before fit, "
        "+y_mean after integrate. ARIMA(0,0,0) on mean=100 no longer "
        "predicts all zeros. "
        "Bug 3 (_predict_in_sample): ForecasterBase contract — "
        "len=n_train and y-scale for d=0; documented all-NaN for d>0 "
        "(THF falls back to OLS, already handled). "
        "fix1 (surfaced by v0.9.25a dogfood): _fit_one rejected "
        "sigma2==0; perfectly linear training (d=1, D=1 → twice-"
        "differenced all zeros) is a valid perfect fit. Now accepts "
        "sigma2==0; AIC uses log(max(sigma2, 1e-300)) to stay finite. "
        "Prior tests all had D=0, so Bug 1 slipped through v0.9.24. "
        "+8 tests: D>0 integration, D>0+mean, ARIMA(0,0,0) nonzero "
        "mean, AR(1) mean-reversion on mean=100, predict_in_sample "
        "len+scale, predict_in_sample d>0 NaN, THF-WLS+SARIMA "
        "fallback, perfect-linear sigma2=0. "
        "Dogfood (fresh subprocess): 5 invariants verified. "
        "v0.9.24 TS toolkit now valid for D>0 and mean≠0 series. "
        "Fast suite: ~2261 → ~2269.",
        self_path="scripts/patch_v0925a_fix1_sigma2_zero.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25a+fix1 applied and committed. "
          "Next: v0.9.25b (B1 linear refill + B2 encode_cat K_est + "
          "external_sort fan-in floor).")


if __name__ == "__main__":
    main()