#!/usr/bin/env python3
# v0.9.25a — SARIMA stabilization (3 critical bugs from TS review).
#
# Bug 1 (§_integrate_back): порядок инверсии D→d, не d→D.
#   _prep делает d (regular) ЗАТЕМ D (seasonal). Обратная операция —
#   сначала D, потом d. Old code делал d→D → для D>0 прогноз
#   структурно неверный (не пойман: все SARIMA-тесты имели D=0).
#   Also: seasonal_period_last должен быть из _difference(y_raw, d),
#   не из y_raw (для d=0 совпадает, для d>0 — нет).
#
# Bug 2 (Hannan-Rissanen без intercept): ARMA CSS без intercept
#   предсказывает → 0 на stationary с mean≠0. Fix: центрирование
#   y перед fit, + y_mean в predict/predict_in_sample.
#
# Bug 3 (_predict_in_sample): возвращал len=n-d и в diff-scale.
#   Контракт ForecasterBase обещает len=n и y-scale. Fix: d>0 → all-NaN
#   (документированное ограничение; THF фолбэкается на OLS);
#   d=0 → n значений в y-scale.
#
# Test additions:
#   - test_sarima_integrate_back_seasonal (D=1, s=2)
#   - test_sarima_stationary_nonzero_mean
#   - test_sarima_predict_in_sample_length_and_scale (d=0)
#   - test_sarima_predict_in_sample_d1_returns_nan (d>0)
#
# Run: uv run python scripts/patch_v0925a_sarima_stabilize.py

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


# ============ 0. Verify v0.9.24 state ============

def verify_state():
    print("\n=== verify v0.9.24 SARIMA present ===")
    src = Path("cadw/ts/models.py").read_text()
    checks = [
        "class SARIMAForecaster",
        "def _integrate_back",
        "def _hannan_rissanen",
        "def _predict_in_sample",
        "return _arma_forecast(",  # current _predict_in_sample body
    ]
    for c in checks:
        if c not in src:
            raise SystemExit(
                f"cadw/ts/models.py missing anchor: {c!r}. "
                f"Re-run v0.9.24 base install."
            )
    print(f"  {len(checks)} anchors present — OK")


# ============ 1. Fix _integrate_back (order D→d + seas_last source) ======

OLD_INTEGRATE = '''def _integrate_back(y_forecast, y_levels, d, D, s, seasonal_period_last):
    """Undo differencing on forecasts.

    y_levels: tuple (L1, L2, ..., Ld) of last observed values of the
      d-differenced series at each integration level, from LEAST
      differenced (L1 = last y of original) down. For d=1, y_levels
      = (y[-1],). After cumsum of the d-differenced forecast, we add
      the (d-1)-differenced last value to get the (d-1)-differenced
      forecast, and so on. Specifically, for d=1: f = cumsum(f) +
      y[-1].
    seasonal_period_last: last s values of the series *before*
      seasonal differencing (i.e. after regular differencing). Used
      to undo seasonal differencing.
    """
    f = np.asarray(y_forecast, dtype=np.float64)
    # Undo regular differencing: for each of d diffs, add the level.
    for lvl in range(d):
        base = y_levels[lvl] if lvl < len(y_levels) else 0.0
        if not np.isfinite(base):
            base = 0.0
        f = np.cumsum(f) + base
    # Undo seasonal differencing: add y[T+h-s] recursively.
    for _ in range(D):
        out = np.empty_like(f)
        for h in range(len(f)):
            if h < s:
                out[h] = f[h] + seasonal_period_last[h]
            else:
                out[h] = f[h] + out[h - s]
        f = out
    return f
'''

NEW_INTEGRATE = '''def _integrate_back(y_forecast, y_levels, d, D, s, seasonal_period_last):
    """Undo differencing on forecasts.

    ORDER: reverse of _prep (which does d regular THEN D seasonal).
    So integration order is D seasonal FIRST, then d regular.
    v0.9.25a fix: prior code did d→D — structurally wrong for D>0
    (produced forecasts that were monotone-inflated by the seasonal
    level carried through regular cumsum). Not caught by tests because
    every SARIMA test in v0.9.24 had D=0.

    y_levels[lvl] = last observed value at lvl-th regular-diff level:
      lvl=0 -> y_raw[-1]            (original scale, centered)
      lvl=1 -> diff(y_raw)[-1]      (1st regular diff)
      lvl=2 -> diff(diff(y_raw))[-1] etc.
    seasonal_period_last = last s values of _difference(y_raw, d) —
      i.e., the series BEFORE seasonal diff, AFTER regular diff.
    v0.9.25a fix: prior caller passed y_raw[-s:] — wrong for d>0
    (y_raw has not been regular-differenced yet; _seasonal_difference
    is applied to diff(y_raw), so the head s values must come from
    diff(y_raw), not y_raw).
    """
    f = np.asarray(y_forecast, dtype=np.float64)
    # Undo seasonal differencing FIRST (reverse of _prep order).
    for _ in range(D):
        out = np.empty_like(f)
        for h in range(len(f)):
            if h < s:
                out[h] = f[h] + seasonal_period_last[h]
            else:
                out[h] = f[h] + out[h - s]
        f = out
    # Then undo regular differencing (d levels, outer→inner).
    for lvl in range(d):
        base = y_levels[lvl] if lvl < len(y_levels) else 0.0
        if not np.isfinite(base):
            base = 0.0
        f = np.cumsum(f) + base
    return f
'''

def fix_integrate_back():
    print("\n=== fix _integrate_back: order D→d, seas_last from diff(y) ===")
    applied = apply_or_fail(
        "cadw/ts/models.py", OLD_INTEGRATE, NEW_INTEGRATE,
        label="integrate_back_order",
    )
    print("  applied" if applied else "  already applied — skip")


# ============ 2. Fix SARIMAForecaster.fit — центрирование ============

OLD_FIT = '''    def fit(self, df, target_col, *, time_col=None, group_col=None):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        if len(y) < 4:
            raise ValueError("SARIMA: need at least 4 observations")
        self.y_raw_ = y.copy()
        self._fit(y)
        self._is_fitted = True
        return self
'''

NEW_FIT = '''    def fit(self, df, target_col, *, time_col=None, group_col=None):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        if len(y) < 4:
            raise ValueError("SARIMA: need at least 4 observations")
        self.y_raw_ = y.copy()
        # v0.9.25a: center to remove the mean. Hannan-Rissanen OLS has
        # no intercept; fitting on raw y with mean≠0 makes ARMA predict
        # toward 0 (e.g. ARIMA(0,0,0) on series with mean=100 predicted
        # all zeros). Centering is exact: add _y_mean back after
        # integration. For d>0 the integration level already captures
        # the mean; the +_y_mean at the end is still correct (same
        # offset in centered space).
        y_clean = y[~np.isnan(y)]
        self._y_mean = float(y_clean.mean()) if len(y_clean) else 0.0
        self._y_centered_ = y - self._y_mean
        self._fit(self._y_centered_)
        self._is_fitted = True
        return self
'''

def fix_fit_centering():
    print("\n=== fix SARIMAForecaster.fit: center y before _fit ===")
    applied = apply_or_fail(
        "cadw/ts/models.py", OLD_FIT, NEW_FIT,
        label="fit_centering",
    )
    print("  applied" if applied else "  already applied — skip")


# ============ 3. Fix _predict — y_levels from y_centered, +y_mean ======

OLD_PREDICT = '''    def _predict(self, steps):
        yd = self._yd
        # Remove leading NaNs for the ARMA recursion
        valid = ~np.isnan(yd)
        if not valid.any():
            raise RuntimeError("SARIMA: no valid observations after differencing")
        # Preserve tail only (head NaN is only from differencing warmup)
        y_clean = yd[valid]
        fc_diff = _arma_forecast(y_clean, self._ar, self._ma, self._sigma2, steps)
        # Undo differencing
        p, d, q = self._order_
        P, D, Q = self._seasonal_order_
        s = self.s
        if d or D:
            # Level values for integration: L1 = last value of the
            # original series (needed for the 1st cumsum level); for
            # d >= 2 we approximate deeper levels with 0 (CSS
            # limitation, documented).
            levels = []
            for lvl in range(d):
                if lvl == 0:
                    levels.append(float(self.y_raw_[-1]))
                else:
                    levels.append(0.0)
            seas_last = (
                self.y_raw_[-s:]
                if (D and s) else np.empty(0, dtype=np.float64)
            )
            fc = _integrate_back(
                fc_diff, tuple(levels), d, D, s,
                seas_last if len(seas_last) else np.zeros(0),
            )
        else:
            fc = fc_diff
        return fc.astype(np.float64)
'''

NEW_PREDICT = '''    def _predict(self, steps):
        yd = self._yd
        valid = ~np.isnan(yd)
        if not valid.any():
            raise RuntimeError("SARIMA: no valid observations after differencing")
        y_clean = yd[valid]
        fc_diff = _arma_forecast(y_clean, self._ar, self._ma, self._sigma2, steps)
        p, d, q = self._order_
        P, D, Q = self._seasonal_order_
        s = self.s
        if d or D:
            # v0.9.25a: build y_levels from centered series at each
            # regular-diff level. For d=2+, prior code used 0.0 for
            # levels >0 — structurally wrong (should be last diff value).
            levels = []
            yd_for_levels = self._y_centered_
            for lvl in range(d):
                v = float(yd_for_levels[-1]) if len(yd_for_levels) else 0.0
                levels.append(v)
                yd_for_levels = _difference(yd_for_levels, 1)
            # v0.9.25a: seasonal_period_last from _difference(y_centered, d)
            # — the series BEFORE seasonal diff, AFTER regular diff.
            # Prior code took self.y_raw_[-s:] — wrong for d>0.
            if D and s:
                yd_for_seas = (
                    _difference(self._y_centered_, d) if d
                    else self._y_centered_.copy()
                )
                seas_last = yd_for_seas[-s:] if len(yd_for_seas) >= s \
                    else np.zeros(s, dtype=np.float64)
            else:
                seas_last = np.empty(0, dtype=np.float64)
            fc = _integrate_back(
                fc_diff, tuple(levels), d, D, s, seas_last,
            )
        else:
            fc = fc_diff
        # v0.9.25a: restore the mean (centered in fit).
        return (fc + self._y_mean).astype(np.float64)
'''

def fix_predict():
    print("\n=== fix _predict: y_levels from centered, seas_last from diff ===")
    applied = apply_or_fail(
        "cadw/ts/models.py", OLD_PREDICT, NEW_PREDICT,
        label="predict_integration",
    )
    print("  applied" if applied else "  already applied — skip")


# ============ 4. Fix _predict_in_sample — contract-compliant ==========

OLD_IN_SAMPLE = '''    def _predict_in_sample(self):
        yd = self._yd
        valid = ~np.isnan(yd)
        if not valid.any():
            return np.full(len(self.y_raw_), np.nan)
        y_clean = yd[valid]
        return _arma_forecast(
            y_clean, self._ar, self._ma, self._sigma2, 0, in_sample=True,
        )
'''

NEW_IN_SAMPLE = '''    def _predict_in_sample(self):
        """In-sample prediction (v0.9.25a contract fix).

        ForecasterBase.predict_in_sample contract: len=n_train, values
        in original y scale.

        - d=0, D=0: returns one-step-ahead predictions in y scale.
        - d>0 or D>0: returns all-NaN. In-sample prediction at original
          y scale for a differenced model is not well-defined (each
          point would require its own integration base; the diff-head
          is not invertible in-sample without leakage). Documented
          limitation; TemporalHierarchy._fit_one already falls back to
          OLS when predict_in_sample returns all-NaN.
        """
        p, d, q = self._order_
        P, D, Q = self._seasonal_order_
        n = len(self.y_raw_)
        if d or D:
            return np.full(n, np.nan)
        yd = self._yd
        valid = ~np.isnan(yd)
        if not valid.any():
            return np.full(n, np.nan)
        y_clean = yd[valid]
        fc_diff = _arma_forecast(
            y_clean, self._ar, self._ma, self._sigma2, 0, in_sample=True,
        )
        if len(fc_diff) != n:
            return np.full(n, np.nan)
        return (fc_diff + self._y_mean).astype(np.float64)
'''

def fix_predict_in_sample():
    print("\n=== fix _predict_in_sample: len=n, y-scale ===")
    applied = apply_or_fail(
        "cadw/ts/models.py", OLD_IN_SAMPLE, NEW_IN_SAMPLE,
        label="predict_in_sample",
    )
    print("  applied" if applied else "  already applied — skip")


# ============ 5. Add tests ============

NEW_TESTS = '''

# ============ v0.9.25a: SARIMA integration + intercept + in-sample ======


def test_sarima_integrate_back_seasonal():
    """D>0: integration order must be D then d (reverse of _prep)."""
    # y = [1..20], d=1, D=1, s=2. Correct integrated forecast on a
    # perfectly linear series is [21, 22, 23].
    # Old (broken) order d-then-D gives [39, 40, 59] — structurally off.
    y = np.arange(1.0, 21.0)
    df = _mk_df(y)
    m = SARIMAForecaster(
        order=(0, 1, 0), seasonal_order=(0, 1, 0), s=2, auto=False,
    ).fit(df, "y")
    fc = m.predict(3)
    np.testing.assert_allclose(
        fc, [21.0, 22.0, 23.0], atol=1e-6,
        err_msg=f"seasonal integration order wrong: got {fc}",
    )


def test_sarima_integrate_back_seasonal_with_mean():
    """D>0 with non-zero mean: centering + correct order compose."""
    y = np.arange(100.0, 120.0)  # [100..119], mean≈109.5
    df = _mk_df(y)
    m = SARIMAForecaster(
        order=(0, 1, 0), seasonal_order=(0, 1, 0), s=2, auto=False,
    ).fit(df, "y")
    fc = m.predict(3)
    np.testing.assert_allclose(fc, [120.0, 121.0, 122.0], atol=1e-6)


def test_sarima_stationary_nonzero_mean():
    """d=0 on series with mean≈100 must predict ~100, not 0."""
    rng = np.random.default_rng(42)
    n = 200
    y = 100.0 + 0.5 * rng.normal(size=n)
    df = _mk_df(y)
    m = SARIMAForecaster(order=(0, 0, 0), auto=False).fit(df, "y")
    fc = m.predict(5)
    assert all(80 < f < 120 for f in fc), (
        f"ARIMA(0,0,0) on mean=100 series returned {fc} (should be ~100, "
        f"not 0; bug #2)"
    )


def test_sarima_ar1_nonzero_mean():
    """AR(1) with non-zero mean — long-horizon forecast ~ mean."""
    rng = np.random.default_rng(0)
    n = 500
    y = np.zeros(n)
    y[0] = 100.0
    for t in range(1, n):
        y[t] = 100.0 + 0.5 * (y[t - 1] - 100.0) + rng.normal(0, 0.5)
    df = _mk_df(y)
    m = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(df, "y")
    fc = m.predict(20)
    assert abs(fc[-1] - 100.0) < 5.0, (
        f"long-horizon AR(1) forecast {fc[-1]:.2f} far from mean 100"
    )


def test_sarima_predict_in_sample_length_and_scale():
    """d=0: len(predict_in_sample) == n_train, values in y scale."""
    rng = np.random.default_rng(0)
    n = 50
    y = 100.0 + rng.normal(size=n)
    df = _mk_df(y)
    m = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(df, "y")
    preds = m.predict_in_sample()
    assert len(preds) == len(y), (
        f"len(predict_in_sample)={len(preds)} != n_train={len(y)} "
        f"(contract violation, bug #3)"
    )
    finite = preds[~np.isnan(preds)]
    assert len(finite) > 0
    assert abs(finite.mean() - 100.0) < 5.0, (
        f"predict_in_sample mean={finite.mean():.2f} — should be ~100 "
        f"(y scale, not diff scale)"
    )


def test_sarima_predict_in_sample_d1_returns_nan():
    """d>0: predict_in_sample returns all-NaN (documented limitation)."""
    rng = np.random.default_rng(0)
    n = 50
    y = np.cumsum(rng.normal(size=n)) + 100.0
    df = _mk_df(y)
    m = SARIMAForecaster(order=(0, 1, 0), auto=False).fit(df, "y")
    preds = m.predict_in_sample()
    assert len(preds) == len(y)
    assert np.all(np.isnan(preds)), (
        "d>0 predict_in_sample must be all-NaN (documented); "
        "TemporalHierarchy falls back to OLS"
    )


def test_sarima_thf_wls_still_works_with_sarima():
    """TemporalHierarchy with SARIMA + WLS: fallback to OLS for d>0."""
    import warnings
    from cadw.ts import TemporalHierarchy
    rng = np.random.default_rng(0)
    n = 200
    y = np.cumsum(rng.normal(size=n)) + 100.0
    df = _mk_df(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fallback warning expected
        th = TemporalHierarchy(
            [1, 4, 16],
            lambda: SARIMAForecaster(order=(0, 1, 0), auto=False),
            reconciliation="optimal_wls",
        )
        th.fit(df, "y")
    fc = th.predict(8)
    assert len(fc) == 8
    assert np.all(np.isfinite(fc))
'''

def add_tests():
    print("\n=== add 7 v0.9.25a tests to test_ts_models.py ===")
    p = Path("tests/test_ts_models.py")
    s = p.read_text()
    if "test_sarima_integrate_back_seasonal" in s:
        print("  already present — skip")
        return
    p.write_text(s + NEW_TESTS)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 6. Dogfood ============

def dogfood():
    print("\n=== dogfood: SARIMA bugs closed (fresh subprocess) ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import numpy as _np
        import pyarrow as _pa
        from cadw import CadwDataFrame as _CDF
        from cadw.ts import SARIMAForecaster as _S

        def _fit_predict(y, **kw):
            t = _pa.table({"y": _pa.array(y, type=_pa.float64())})
            df = _CDF.from_table(t)
            m = _S(**kw).fit(df, "y")
            return m

        # (1) D=1, s=2, d=1 on linear -> [21, 22, 23]
        y = list(_np.arange(1.0, 21.0))
        m = _fit_predict(
            y, order=(0, 1, 0), seasonal_order=(0, 1, 0), s=2, auto=False,
        )
        fc = m.predict(3)
        assert _np.allclose(fc, [21.0, 22.0, 23.0], atol=1e-6), (
            f"BUG 1 not closed: D=1 integration gives {fc}, expected "
            f"[21, 22, 23]"
        )

        # (2) d=0, mean=100 -> predict ~ 100, not 0
        rng = _np.random.default_rng(42)
        y2 = (100.0 + 0.5 * rng.normal(size=200)).tolist()
        m2 = _fit_predict(y2, order=(0, 0, 0), auto=False)
        fc2 = m2.predict(5)
        assert all(80 < f < 120 for f in fc2), (
            f"BUG 2 not closed: ARIMA(0,0,0) on mean=100 returns {fc2}"
        )

        # (3) predict_in_sample for d=0 -> len=n, y scale
        m3 = _fit_predict(
            (100.0 + rng.normal(size=50)).tolist(),
            order=(1, 0, 0), auto=False,
        )
        preds = m3.predict_in_sample()
        assert len(preds) == 50, (
            f"BUG 3 not closed: len(predict_in_sample)={len(preds)}, "
            f"expected 50"
        )
        finite = preds[~_np.isnan(preds)]
        assert abs(finite.mean() - 100.0) < 5.0, (
            f"BUG 3 (scale): predict_in_sample mean={finite.mean()}, "
            f"expected ~100"
        )

        # (4) predict_in_sample for d=1 -> all NaN (documented)
        y4 = _np.cumsum(rng.normal(size=50)).tolist() + 100.0
        y4 = (y4 if isinstance(y4, list) else y4.tolist()) if hasattr(y4, "tolist") else y4
        y4_arr = _np.cumsum(rng.normal(size=50)) + 100.0
        m4 = _fit_predict(
            y4_arr.tolist(), order=(0, 1, 0), auto=False,
        )
        preds4 = m4.predict_in_sample()
        assert _np.all(_np.isnan(preds4)), (
            "d>0 predict_in_sample must be all-NaN"
        )

        print("  4 bugs closed: D=1 order, non-zero-mean, in-sample len+scale, "
              "d>0 NaN fallback")

    run_verify_in_subprocess(_verify)


# ============ 7. Pytest ============

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


# ============ 8. main ============

def main():
    print("=== v0.9.25a: SARIMA stabilization (3 critical bugs) ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    fix_integrate_back()
    fix_fit_centering()
    fix_predict()
    fix_predict_in_sample()
    assert_compiles(Path("cadw/ts/models.py"))
    add_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25a: SARIMA stabilization — 3 critical bugs "
        "(Phase A of consolidation roadmap). "
        "Bug 1 (_integrate_back): reverse order D→d not d→D "
        "(_prep does d then D; integration must invert order). "
        "Bug 1b: seasonal_period_last from _difference(y_centered, d), "
        "not y_raw (wrong for d>0). "
        "Bug 2 (Hannan-Rissanen no intercept): center y before fit, "
        "+y_mean after integrate. ARIMA(0,0,0) on mean=100 no longer "
        "predicts all zeros. "
        "Bug 3 (_predict_in_sample): ForecasterBase contract fixed — "
        "len=n_train and y-scale for d=0; documented all-NaN for d>0 "
        "(THF falls back to OLS, already handled). "
        "Prior tests all had D=0, so Bug 1 slipped through. "
        "+7 tests: D>0 integration, D>0+mean, ARIMA(0,0,0) nonzero mean, "
        "AR(1) mean-reversion on mean=100, predict_in_sample len+scale, "
        "predict_in_sample d>0 NaN, THF-WLS+SARIMA fallback. "
        "Dogfood (fresh subprocess): all 4 invariants verified. "
        "v0.9.24 TS toolkit now valid for D>0 and mean≠0 series. "
        "Fast suite: ~2261 → ~2268.",
        self_path="scripts/patch_v0925a_sarima_stabilize.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25a applied. Next: v0.9.25b (B1 refill + B2 encode_cat "
          "+ external_sort fan-in floor).")


if __name__ == "__main__":
    main()