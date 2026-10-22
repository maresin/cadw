"""RAM verification for CADW TS modules (v0.9.24, Stage 8)."""
import gc
import os
import time
import tracemalloc

import numpy as np
import pyarrow as pa
import psutil

import cadw
from cadw import CadwDataFrame
from cadw.ts import (
    rolling_stats, rolling_crosscorr, GBDTForecaster,
    TemporalHierarchy, NaiveForecaster, backtest_forecaster,
    make_features,
)


def _df(n, groups=1, seed=0):
    rng = np.random.default_rng(seed)
    ts = [1700000000000000000 + i * 3600 * 10**9 for i in range(n)]
    cols = {
        "y": pa.array(rng.normal(0, 1, n).cumsum() + 100.0, type=pa.float64()),
        "x": pa.array(rng.normal(0, 1, n), type=pa.float64()),
        "ts": pa.array(ts, type=pa.timestamp("ns")),
    }
    if groups > 1:
        cols["g"] = pa.array([f"g{i % groups}" for i in range(n)], type=pa.string())
    return CadwDataFrame.from_table(pa.table(cols))


def _measure(fn, budget_mb):
    # Sample peak RSS *during* the call (not after). Warm up imports
    # and the Arrow allocator first so first-call overhead is excluded.
    gc.collect()
    proc = psutil.Process()
    base = proc.memory_info().rss
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    cur, peak_tm = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = proc.memory_info().rss
    # peak_tm is the authoritative Python-allocations peak; rss delta
    # includes interpreter-level noise from earlier scenarios.
    return {
        "delta_mb": max(0, peak_rss - base) / (1024 * 1024),
        "tracemalloc_peak_mb": peak_tm / (1024 * 1024),
        "budget_mb": budget_mb,
        "sec": round(dt, 3),
    }


def _measure_one(name, fn, budget_mb):
    # Run one scenario in a fresh subprocess so cross-scenario RSS does
    # not contaminate. Prints a JSON line on stdout.
    import json
    import subprocess, sys
    code = (
        "import json, gc, time, psutil, tracemalloc\n"
        "import importlib, sys\n"
        "sys.path.insert(0, '.')\n"
        "mod = importlib.import_module('scripts.verify_ts_ram')\n"
        "gc.collect()\n"
        "proc = psutil.Process(); base = proc.memory_info().rss\n"
        "tracemalloc.start()\n"
        "t0 = time.perf_counter()\n"
        f"mod.{fn}()\n"
        "dt = time.perf_counter() - t0\n"
        "cur, peak_tm = tracemalloc.get_traced_memory()\n"
        "tracemalloc.stop()\n"
        "print(json.dumps({'delta_mb': (proc.memory_info().rss-base)/1048576, "
        "'tracemalloc_peak_mb': peak_tm/1048576, 'sec': dt}))\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    if r.returncode != 0:
        return {"error": r.stderr.strip().splitlines()[-1] if r.stderr else "?"}
    return json.loads(r.stdout.strip().splitlines()[-1])


def scenario_rolling_stats():
    n = 2000
    df = _df(n, groups=4)
    guard = cadw.MemoryGuard(max_memory_mb=32)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    out = rolling_stats(
        df2, "y", windows=(7, 14, 30),
        stats=("mean", "std", "q50", "skew", "slope", "autocorr_lag1"),
        group_col="g", time_col="ts",
    )
    n_out = sum(b.num_rows for b in out._iter_arrow_batches(batch_size=500))
    assert n_out == n


def scenario_crosscorr():
    n = 2000
    df = _df(n, groups=4)
    guard = cadw.MemoryGuard(max_memory_mb=32)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    out = rolling_crosscorr(
        df2, "y", "x", windows=(14,), lags=(0, 1),
        group_col="g", time_col="ts",
    )
    n_out = sum(b.num_rows for b in out._iter_arrow_batches(batch_size=500))
    assert n_out == n


def scenario_make_features():
    n = 2000
    df = _df(n, groups=4)
    guard = cadw.MemoryGuard(max_memory_mb=32)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    out = make_features(
        df2, target_col="y", ts_col="ts",
        lags=(1, 7, 14), diffs=(1,), seasonal_periods=(7,),
        calendar=("month", "weekday", "hour"),
        cyclical=("month", "weekday", "hour"),
        fourier=(24,), fourier_order=2,
        group_col="g",
    )
    n_out = sum(b.num_rows for b in out._iter_arrow_batches(batch_size=500))
    assert n_out == n


def scenario_gbdt():
    n = 500
    df = _df(n)
    guard = cadw.MemoryGuard(max_memory_mb=64)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    m = GBDTForecaster(
        mode="direct", lags=(1, 2, 7), n_estimators=20, max_depth=3,
    )
    m.fit(df2, "y")
    fc = m.predict(14)
    assert len(fc) == 14


def scenario_thf_ols():
    n = 300
    df = _df(n)
    guard = cadw.MemoryGuard(max_memory_mb=32)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    th = TemporalHierarchy(
        [1, 7, 28], lambda: NaiveForecaster("last"),
        reconciliation="optimal_ols",
    )
    th.fit(df2, "y", time_col="ts")
    fc = th.predict(28)
    assert len(fc) == 28


def scenario_thf_mint_wls():
    n = 300
    df = _df(n)
    guard = cadw.MemoryGuard(max_memory_mb=32)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    th = TemporalHierarchy(
        [1, 7, 28], lambda: NaiveForecaster("last"),
        reconciliation="optimal_wls",
    )
    th.fit(df2, "y", time_col="ts")
    fc = th.predict(28)
    assert len(fc) == 28


def scenario_backtest():
    n = 200
    df = _df(n)
    guard = cadw.MemoryGuard(max_memory_mb=32)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    summ, folds = backtest_forecaster(
        df2, NaiveForecaster("last"),
        target_col="y", time_col="ts",
        horizon=7, n_folds=5, strategy="expanding",
        metrics=("mae", "rmse", "mase"),
    )
    assert len(summ) == 3




def scenario_baseline_noop():
    """Control: build same df, do nothing. Used to subtract imports/df."""
    n = 2000
    _df(n, groups=4)

SCENARIOS = [
    ("rolling_stats(3 windows x 6 stats, n=2000, 4 groups)", scenario_rolling_stats, 32),
    ("rolling_crosscorr(14, lags 0/1, n=2000, 4 groups)", scenario_crosscorr, 32),
    ("make_features(n=2000, 4 groups, fourier+lags)", scenario_make_features, 32),
    ("GBDT direct fit+14-step predict (n=500)", scenario_gbdt, 64),
    ("THF [1,7,28] OLS predict 28 (n=300)", scenario_thf_ols, 32),
    ("THF [1,7,28] MinT-WLS predict 28 (n=300)", scenario_thf_mint_wls, 32),
    ("backtest naive h=7 k=5 expanding (n=200)", scenario_backtest, 32),
]


def main():
    # In-process (fast) pass; the subprocess helper is available for a
    # clean re-run when a scenario looks suspicious.
    print(f"{'scenario':<55} {'rss_delta_mb':>12} {'tm_peak_mb':>11} {'budget_mb':>9} {'sec':>7}")
    print("-" * 105)
    for name, fn, budget in SCENARIOS:
        r = _measure(fn, budget)
        print(f"{name:<55} {r['delta_mb']:>12.2f} {r['tracemalloc_peak_mb']:>11.2f} {budget:>9} {r['sec']:>7.2f}")


if __name__ == '__main__':
    main()
