"""Stage 7 tests: public API, TS accessor, registry, docs (v0.9.24)."""
import numpy as np
import pyarrow as pa
import pytest

import cadw
import cadw.ts as ts
from cadw import CadwDataFrame


def _df(n=20):
    ts_vals = [1700000000000000000 + i * 3600 * 10**9 for i in range(n)]
    return CadwDataFrame.from_table(pa.table({
        "y": pa.array([float(i) for i in range(1, n + 1)],
                      type=pa.float64()),
        "ts": pa.array(ts_vals, type=pa.timestamp("ns")),
        "g": pa.array(["a"] * n, type=pa.string()),
    }))




def _read_all_decisions():
    """v0.9.28-fix4 (§13.4/351): DECISIONS.md split → glob.
    Concatenate all topic files for tests asserting specific sections."""
    from pathlib import Path as _P
    _docs = _P(__file__).resolve().parent.parent / "docs" / "decisions"
    return "\n".join(f.read_text() for f in sorted(_docs.glob("*.md")))

def test_ts_all_present():
    for name in ts.__all__:
        assert hasattr(ts, name), f"cadw.ts missing {name}"
    # __all__ sorted, no duplicates
    assert ts.__all__ == sorted(ts.__all__)
    assert len(ts.__all__) == len(set(ts.__all__))


def test_accessor_registered():
    df = _df()
    acc = df.ts
    assert acc is not None
    for m in (
        "lag", "lead", "diff", "calendar", "cyclical", "fourier",
        "holiday", "make_features", "rolling_features", "crosscorr",
        "fit_forecaster", "backtest", "fit_temporal_hierarchy",
    ):
        assert callable(getattr(acc, m)), f"accessor missing {m}"


def test_accessor_returns_cadw_df():
    df = _df()
    out = df.ts.lag("y", lags=(1,))
    assert isinstance(out, CadwDataFrame)
    assert "y_lag1" in out.columns
    out2 = df.ts.calendar("ts", features=("month",))
    assert isinstance(out2, CadwDataFrame)
    assert "ts_month" in out2.columns
    out3 = df.ts.rolling_features("y", windows=(3,), stats=("mean",))
    assert isinstance(out3, CadwDataFrame)
    assert "y_roll3_mean" in out3.columns


def test_registry_names():
    from cadw.ts import get_forecaster
    for name in ("naive", "sarima", "gbdt"):
        f = get_forecaster(name)
        assert hasattr(f, "fit") and hasattr(f, "predict")
    with pytest.raises(ValueError):
        get_forecaster("nope")


def test_accessor_fit_forecaster_by_name():
    df = _df()
    f = df.ts.fit_forecaster("naive", "y")
    fc = f.predict(3)
    assert len(fc) == 3
    assert np.all(np.isfinite(fc))


def test_accessor_backtest():
    df = _df(40)
    summ, folds = df.ts.backtest(
        ts.NaiveForecaster(strategy="last"),
        target_col="y", time_col="ts",
        horizon=1, n_folds=2, strategy="expanding",
        train_size=10, step=1, metrics=("mae",),
    )
    assert len(summ) == 1 and summ[0]["metric"] == "mae"


def test_accessor_thf():
    df = _df(20)
    th = df.ts.fit_temporal_hierarchy(
        [1, 2], lambda: ts.NaiveForecaster(strategy="last"),
        target_col="y", time_col="ts", reconciliation="bottom_up",
    )
    fc = th.predict(2)
    assert len(fc) == 2


def test_readme_and_changelog_contain_ts_terms():
    import os
    readme = open("README.md", encoding="utf-8").read().lower()
    assert "time series" in readme or "time-series" in readme
    assert "forecaster" in readme
    changelog = open("CHANGELOG.md", encoding="utf-8").read().lower()
    assert "time series" in changelog or "time-series" in changelog
    decisions = _read_all_decisions().lower()
    assert "mint" in decisions
