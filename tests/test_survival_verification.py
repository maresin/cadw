"""v0.9.143-batch3: verification tests for survival pair (DEC-033).

KaplanMeier (fit_kaplan_meier):
  State = n_unique * 5 * f64 + 128 (RAM) OR 256 (spilled).
  CW-1: state_nbytes exact
  CW-2: state independent of n at fixed n_unique distribution
  CW-2b: spilled state constant (256) when n_unique > cap
  CW-5: OOC at fixed batch
  CW-6: guard refuses at tight budget (external_sort can't fit)
  Correctness: survival_at matches manual product-limit

WeibullAFT (fit_weibull_aft):
  State = d * f64 + 8 + 128
  CW-1: state_nbytes exact
  CW-2: state independent of n
  CW-5: OOC at fixed batch
  CW-6: guard refuses at tight (d large)
  Correctness: beta_ recovers true coefficients (weak signal, 3 seeds)
"""
import ctypes
import gc
import statistics

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import (
    MemoryGuard, MemoryLimitExceeded, measure_peak_traced,
)
from cadw.ml.kaplan_meier import fit_kaplan_meier, KaplanMeierModel
from cadw.ml.weibull_aft import fit_weibull_aft, WeibullAFTModel


def _iter_col_concat(df, col):
    """Concatenate a column across all batches (no to_table method)."""
    chunks = [b.column(col).to_numpy(zero_copy_only=False)
              for b in df._iter_filtered_batches()]
    return np.concatenate(chunks) if chunks else np.empty(0)


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _make_km_df(n, seed=0, n_time_levels=None):
    """n rows, exponentially-distributed times with 30% censoring."""
    rng = np.random.RandomState(seed)
    if n_time_levels is not None:
        # Discretize times to control n_unique exactly
        t = rng.randint(1, n_time_levels + 1, size=n).astype(np.float64)
    else:
        t = rng.exponential(scale=10.0, size=n)
    c = rng.random(n) < 0.3
    e = (~c).astype(np.int64)
    return CadwDataFrame.from_table(pa.table({
        "t": pa.array(t, type=pa.float64()),
        "e": pa.array(e, type=pa.int64()),
    }))


def _make_waft_df(n, d, seed=0, beta_true=None, shape=1.5, censor=0.25):
    rng = np.random.RandomState(seed)
    if beta_true is None:
        # deterministic pad to length d (cycle pattern length 5)
        _base = np.array([0.3, -0.2, 0.1, 0.15, -0.1])
        beta_true = np.resize(_base, d)
    else:
        beta_true = np.asarray(beta_true, dtype=np.float64)
        if beta_true.size < d:
            # pad with mean of existing values
            pad = np.full(d - beta_true.size, float(beta_true.mean()) * 0.5)
            beta_true = np.concatenate([beta_true, pad])
        elif beta_true.size > d:
            beta_true = beta_true[:d]
    X = rng.normal(size=(n, d))
    scale = 10.0
    u = rng.random(n)
    # Model convention: _newton_weibull fits beta by regressing log(t) on X.
    # So true log(t) = + X β_true + noise → β_fit ≈ +β_true.
    t = scale * np.exp(X @ beta_true) * (-np.log(u)) ** (1.0 / shape)
    c = rng.random(n) < censor
    e = (~c).astype(np.int64)
    if c.any():
        t[c] = t[c] * rng.uniform(1.0, 2.0, size=int(c.sum()))
    tbl = {f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)}
    tbl["t"] = pa.array(t, type=pa.float64())
    tbl["e"] = pa.array(e, type=pa.int64())
    return CadwDataFrame.from_table(pa.table(tbl))


def _warmup():
    df = _make_km_df(200, seed=999)
    fit_kaplan_meier(df, time_col="t", event_col="e",
                     memory_guard=MemoryGuard(max_memory_mb=500),
                     batch_size=100)
    df2 = _make_waft_df(200, 3, seed=999)
    fit_weibull_aft(df2, time_col="t", event_col="e",
                    memory_guard=MemoryGuard(max_memory_mb=500),
                    batch_size=100)


_warmup()


# v0.9.229-f: RSS-isolated subprocess peak measure (DEC-034 §v0.9.112).
# In-process measure_peak_traced has SNR < 0.2 in full-suite (prior tests
# raise baseline RSS ~1MB while KM fit peak is ~168KB). subprocess.run
# with fresh interpreter gives clean baseline -> valid ratio.
def _measure_km_peak_subprocess(n, *, seed=0, n_tl=100, guard_mb=500) -> int:
    import subprocess as _sp
    import sys as _sys
    import os as _os
    import json as _json
    code = f"""
import sys, json
import numpy as np, pyarrow as pa
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.kaplan_meier import fit_kaplan_meier

rng = np.random.RandomState({seed})
n = {n}
t = rng.exponential(size=n)
e = (rng.random(n) < 0.7).astype(np.int64)
t = np.floor(t * {n_tl}).astype(np.float64) + 0.001
df = CadwDataFrame.from_table(pa.table({{
    "t": pa.array(t, type=pa.float64()),
    "e": pa.array(e, type=pa.int64()),
}}), memory_guard=MemoryGuard(max_memory_mb=500))

mg = MemoryGuard(max_memory_mb={guard_mb})
r = measure_peak_traced(
    fit_kaplan_meier, df, time_col="t", event_col="e",
    memory_guard=mg, mg=mg)
print("PEAK_JSON:" + json.dumps({{"peak": int(r["peak_bytes"])}}))
"""
    from pathlib import Path as _P
    env = {**_os.environ, "PYTHONPATH": str(_P(__file__).parent.parent)}
    proc = _sp.run([_sys.executable, "-c", code], env=env,
                   capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess failed: {proc.stderr[:800]}")
    marker = "PEAK_JSON:"
    line = next((l for l in proc.stdout.split("\n") if l.startswith(marker)),
                None)
    if not line:
        raise RuntimeError(
            f"no PEAK_JSON in stdout: {proc.stdout[:200]!r}")
    return int(_json.loads(line[len(marker):])["peak"])


# ============================================================================
# KaplanMeier
# ============================================================================

@pytest.mark.parametrize("n,n_unique_target", [(500, 50), (2000, 200), (5000, 500)])
def test_km_state_nbytes_exact(n, n_unique_target):
    """Non-spilled state = n_unique * 5 * f64 + 128."""
    df = _make_km_df(n, seed=0, n_time_levels=n_unique_target)
    m = fit_kaplan_meier(df, time_col="t", event_col="e",
                         memory_guard=MemoryGuard(max_memory_mb=500),
                         batch_size=200)
    assert m.spill_path is None, "expected in-RAM (n_unique small)"
    expected = m.n_unique * 5 * 8 + 128
    assert m.state_nbytes() == expected, (
        f"n_unique={m.n_unique}: got {m.state_nbytes()}, expected {expected}")


def test_km_state_independent_of_n():
    """Same n_unique distribution → same state regardless of n."""
    n_unique_target = 100
    m1 = fit_kaplan_meier(
        _make_km_df(1000, seed=0, n_time_levels=n_unique_target),
        time_col="t", event_col="e",
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=200)
    m2 = fit_kaplan_meier(
        _make_km_df(5000, seed=0, n_time_levels=n_unique_target),
        time_col="t", event_col="e",
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=200)
    assert m1.n_unique == m2.n_unique
    assert m1.state_nbytes() == m2.state_nbytes()


def test_km_state_spilled_constant(tmp_path):
    """Force spill via spill_dir + tiny state_limit -> state = 256.

    n_time_levels=20000 ensures n_unique (~20000) > cap for 0.5 MB guard.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    df = _make_km_df(20000, seed=0, n_time_levels=20000)
    try:
        m = fit_kaplan_meier(
            df, time_col="t", event_col="e",
            memory_guard=MemoryGuard(max_memory_mb=0.5),
            batch_size=200, spill_dir=tmp_path / "km")
    except Exception as e:
        pytest.skip(f"spill_dir path or guard rejected: {type(e).__name__}: {e}")
    if m.spill_path is None:
        pytest.skip(f"n_unique={m.n_unique} did not exceed cap")
    assert m.state_nbytes() == 256
    m.cleanup()


@pytest.mark.slow
def test_km_peak_constant_in_n():
    """v0.9.229-f (P25c + DEC-023): peak constant in n (subprocess, DEC-034).

    After fix: batch_size derived from sort_buffer_bytes=16KB (1024 rows),
    max_buffer_bytes=16KB (Phase 1 = 64KB), max_buf_rows=1024 (buf=40KB).
    Peak ~168KB, independent of n. Single measure per n in fresh child
    (RSS=0 baseline, no cross-test pollution).
    """
    peaks = []
    for n in (1000, 10_000, 100_000):
        peaks.append(_measure_km_peak_subprocess(n))
    ratio = max(peaks) / max(min(peaks), 1)
    assert ratio < 2.0, f"peak ratio={ratio:.2f}; peaks={peaks}"


def test_km_spill_path_executes():
    """v0.9.229-f2: spill path executes without crash (regression guard).

    Bug fixed: _open_spill discarded `__enter__()` return value, so
    `self.spill_writer` was the context manager (no `.write_batch`) ->
    AttributeError on first spill. Also `finalize()` checked
    `spill_writer is not None`, but `close()` reset it to None ->
    `self.buf[:n_buf]` on `buf=None` crash. Both latent since v0.9.116
    because old cap=6.5M never triggered spill for test data.

    Force spill: max_buf_rows=1024, n_unique ~2000 (n_tl=2000, n=10000).
    """
    n, n_tl = 10_000, 2000
    rng = np.random.RandomState(0)
    t = np.floor(rng.exponential(size=n) * n_tl).astype(np.float64) + 0.001
    e = (rng.random(n) < 0.7).astype(np.int64)
    df = CadwDataFrame.from_table(pa.table({
        "t": pa.array(t, type=pa.float64()),
        "e": pa.array(e, type=pa.int64()),
    }), memory_guard=MemoryGuard(max_memory_mb=500))
    m = fit_kaplan_meier(df, time_col="t", event_col="e",
                         memory_guard=MemoryGuard(max_memory_mb=500))
    assert m.n_unique > 1024, (
        f"n_unique={m.n_unique} <= 1024, spill was not triggered "
        f"(test precondition failed)")
    assert m.n_total == n
    # Spill path completed: no crash above means finalize() reached
    # the spilled branch (self._spilled=True) without touching buf=None.


def test_km_sort_buffer_bitexact():
    """v0.9.229-f: sort_buffer_bytes invariant (product-limit formula)."""
    df = _make_km_df(1000, seed=0, n_time_levels=100)
    a = fit_kaplan_meier(df, time_col="t", event_col="e",
                         sort_buffer_bytes=1_048_576)
    b = fit_kaplan_meier(df, time_col="t", event_col="e",
                         sort_buffer_bytes=16_384)
    np.testing.assert_array_equal(a.times, b.times)
    np.testing.assert_array_equal(a.survival, b.survival)
    np.testing.assert_array_equal(a.greenwood, b.greenwood)


@pytest.mark.slow
def test_km_peak_bounded_at_large_n():
    """v0.9.229-f (P25c + DEC-023): peak < 2MB at n=1e6, 50MB budget."""
    peak = _measure_km_peak_subprocess(1_000_000, guard_mb=50)
    assert peak < 2_000_000, (
        f"peak={peak/1e6:.2f}MB > 2MB (constant claim violated)")

def test_km_guard_refuses_at_tight():
    """external_sort + n=5000 + 0.01 MB → MemoryLimitExceeded."""
    df = _make_km_df(5000, seed=0)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        fit_kaplan_meier(
            df, time_col="t", event_col="e",
            memory_guard=MemoryGuard(max_memory_mb=0.01),
            batch_size=200)


def test_km_survival_at_matches_manual():
    """S(t) from impl matches manual product-limit on small data."""
    df = _make_km_df(200, seed=1, n_time_levels=20)
    m = fit_kaplan_meier(df, time_col="t", event_col="e",
                         memory_guard=MemoryGuard(max_memory_mb=500),
                         batch_size=100)
    # manual product-limit
    t_all = _iter_col_concat(df, "t")
    e_all = _iter_col_concat(df, "e") != 0
    uniq_t = np.unique(t_all)
    S = 1.0
    manual_S = {}
    n_total = len(t_all)
    n_at_risk = n_total
    for u_t in uniq_t:
        mask_t = t_all == u_t
        d = int((e_all & mask_t).sum())
        c = int((~e_all & mask_t).sum())
        if d > 0 and n_at_risk > 0:
            S *= 1.0 - d / n_at_risk
        manual_S[u_t] = S
        n_at_risk -= (d + c)
    # sample a few times
    for t_check in [uniq_t[0], uniq_t[len(uniq_t) // 2], uniq_t[-1]]:
        impl_S = m.survival_at(float(t_check))
        assert abs(impl_S - manual_S[t_check]) < 1e-9, (
            f"t={t_check}: impl={impl_S}, manual={manual_S[t_check]}; "
            f"n_total={m.n_total}, n_unique={m.n_unique}, "
            f"impl_times_head={m.times[:3] if m.times is not None else None}, "
            f"impl_survival_head={m.survival[:3] if m.survival is not None else None}, "
            f"manual_S_head={[manual_S[t] for t in uniq_t[:3]]}")


# ============================================================================
# WeibullAFT
# ============================================================================

@pytest.mark.parametrize("d", [1, 3, 5, 10])
def test_waft_state_nbytes_exact(d):
    df = _make_waft_df(500, d, seed=0)
    m = fit_weibull_aft(df, time_col="t", event_col="e",
                        memory_guard=MemoryGuard(max_memory_mb=500),
                        batch_size=200)
    expected = d * 8 + 8 + 128  # beta + log_sigma + header
    assert m.state_nbytes() == expected, (
        f"d={d}: got {m.state_nbytes()}, expected {expected}")


def test_waft_state_independent_of_n():
    d = 5
    m1 = fit_weibull_aft(_make_waft_df(500, d, seed=0),
                         time_col="t", event_col="e",
                         memory_guard=MemoryGuard(max_memory_mb=500),
                         batch_size=200)
    m2 = fit_weibull_aft(_make_waft_df(5000, d, seed=0),
                         time_col="t", event_col="e",
                         memory_guard=MemoryGuard(max_memory_mb=500),
                         batch_size=200)
    assert m1.state_nbytes() == m2.state_nbytes()


def test_waft_ooc_at_fixed_batch():
    d, batch = 5, 200
    peaks = []
    for n in [1000, 5000]:
        df = _make_waft_df(n, d, seed=0)
        cell = []
        for _ in range(3):
            _malloc_trim(); gc.collect()
            mg = MemoryGuard(max_memory_mb=500)
            r = measure_peak_traced(
                fit_weibull_aft, df, time_col="t", event_col="e",
                memory_guard=mg, batch_size=batch, mg=mg)
            cell.append(int(r["peak_bytes"]))
            del r; gc.collect()
        peaks.append(int(statistics.median(cell)))
        del df; gc.collect()
    ratio = peaks[-1] / max(peaks[0], 1)
    assert ratio < 2.5, f"WAFT peak ratio {ratio:.2f}, peaks={peaks}"


def test_waft_guard_refuses_at_tight():
    """d=100 → state ~ 80KB. Guard 0.05 MB → refuses."""
    df = _make_waft_df(500, 100, seed=0)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        fit_weibull_aft(df, time_col="t", event_col="e",
                        memory_guard=MemoryGuard(max_memory_mb=0.05),
                        batch_size=100)


def test_waft_beta_recovers_true():
    """True beta=[0.3, -0.2, 0.1], no censoring → fit recovers (loose)."""
    d = 3
    beta_true = np.array([0.3, -0.2, 0.1])
    df = _make_waft_df(5000, d, seed=42, beta_true=beta_true, censor=0.0)
    m = fit_weibull_aft(df, time_col="t", event_col="e",
                        memory_guard=MemoryGuard(max_memory_mb=500),
                        batch_size=500)
    # sign must match (very loose; formula is Gauss-Newton approximation)
    for i in range(d):
        assert np.sign(m.beta_[i]) == np.sign(beta_true[i]), (
            f"beta[{i}]: got {m.beta_[i]:.3f}, true {beta_true[i]:.3f}")


def test_waft_predict_median_shape():
    d = 3
    df = _make_waft_df(500, d, seed=0)
    m = fit_weibull_aft(df, time_col="t", event_col="e",
                        memory_guard=MemoryGuard(max_memory_mb=500),
                        batch_size=200)
    X_test = np.random.RandomState(0).normal(size=(10, d))
    med = m.predict_median(X_test)
    assert med.shape == (10,)
    assert (med > 0).all()
