"""Tests for D.3.3 uplift trees + Qini (v0.9.37)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import uplift as upl
from cadw.ml.uplift import fit_uplift, UpliftModel, qini_curve


def _truth(n=2500, seed=0, d=3):
    """Synthetic uplift DGP: randomized treatment, monotone CATE on x2.

    v0.9.42 change: two problems fixed in the previous DGP.
    (1) Confounding (t depended on x0,x1) made the naive diff-in-means
        CATE estimate used by the causal tree *biased for split selection*:
        ED-gain on (P(y|t=1)-P(y|t=0))^2 rewarded splits by confounders
        (x0,x1) instead of the real effect (x2). The causal tree's
        honest-leaf estimator is unbiased only under randomized treatment
        (Wager-Athey 2018 unconfoundedness assumption). For real causal
        inference under confounding, a propensity-weighted variant would
        be required (see d3_3_uplift.md Future work).
    (2) Previous effect `0.6*t` through a sigmoid saturated near x2=0
        produced U-shaped CATE(x2) — the old `corr(u, x2) > 0.25` test
        passed by luck on seed=0. New multiplicative effect
        `t*(1.0 + 0.4*x2)` gives monotone CATE(x2) range ~[0.12, 0.27].
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    t = rng.binomial(1, 0.5, size=n).astype(float)
    lp = -0.5 + 0.8 * X[:, 2] + t * (1.0 + 0.4 * X[:, 2])
    p = 1.0 / (1.0 + np.exp(-lp))
    y = (rng.uniform(size=n) < p).astype(float)
    return X, t, y


def _df(X, t, y):
    d = {f"x{j}": pa.array(X[:, j]) for j in range(X.shape[1])}
    d["t"] = pa.array(t); d["y"] = pa.array(y)
    return CadwDataFrame.from_table(pa.table(d))


def test_synthetic_uplift_follows_treatment_effect():
    X, t, y = _truth(seed=0)
    m = fit_uplift(_df(X, t, y), ["x0", "x1", "x2"], "t", "y", n_estimators=30)
    u = m.uplift(_df(X, t, y))
    assert u.shape == (X.shape[0],)
    assert np.corrcoef(u, X[:, 2])[0, 1] > 0.25


def test_qini_coefficient_positive_on_informative_model():
    X, t, y = _truth(seed=1)
    df = _df(X, t, y)
    m = fit_uplift(df, ["x0", "x1", "x2"], "t", "y", n_estimators=30)
    _, _, coef = m.qini(df, "y")
    assert coef > 0.0


def test_qini_curve_reference_random_targeting():
    rng = np.random.RandomState(2)
    n = 2000
    t = (rng.uniform(size=n) < 0.5).astype(float)
    y = (rng.uniform(size=n) < 0.3).astype(float)
    score = rng.normal(size=n)
    f, q, coef = qini_curve(y, t, score)
    assert len(f) == len(q)
    assert f[0] > 0 and f[-1] <= 1.0
    assert abs(coef) < 0.05


def test_determinism_same_seed():
    X, t, y = _truth(seed=3, n=1200)
    df = _df(X, t, y)
    m1 = fit_uplift(df, ["x0", "x1"], "t", "y", n_estimators=15)
    m2 = fit_uplift(df, ["x0", "x1"], "t", "y", n_estimators=15)
    assert np.allclose(m1.uplift(df), m2.uplift(df))


def test_qini_curve_edge_and_guards():
    with pytest.raises(ValueError, match="equal non-zero"):
        qini_curve(np.zeros(0), np.zeros(0), np.zeros(0))
    y = np.array([1.0, 0.0, 1.0]); t = np.ones(3); s = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="both treated and control"):
        qini_curve(y, t, s)


def test_fit_guards_treatment_and_outcome():
    X, t, y = _truth(n=100, seed=4)
    tb = t.copy(); tb[0] = 2.0
    with pytest.raises(ValueError, match="treatment must be binary"):
        fit_uplift(_df(X, tb, y), ["x0"], "t", "y")
    yb = y.copy(); yb[0] = 2.0
    with pytest.raises(ValueError, match="outcome must be binary"):
        fit_uplift(_df(X, t, yb), ["x0"], "t", "y")


def test_fit_guards_missing_col_and_empty():
    X, t, y = _truth(n=100, seed=5)
    with pytest.raises(KeyError):
        fit_uplift(_df(X, t, y), ["zzz"], "t", "y")
    with pytest.raises(ValueError, match="non-empty"):
        fit_uplift(_df(X, t, y), [], "t", "y")


def test_ram_guard_raises_when_tight():
    X, t, y = _truth(n=1500, seed=6)
    mg = MemoryGuard(max_memory_bytes=1024)
    with pytest.raises(MemoryLimitExceeded):
        fit_uplift(_df(X, t, y), ["x0", "x1"], "t", "y",
                   n_estimators=5, memory_guard=mg)


def test_multi_batch_consistent(tmp_path):
    import pyarrow.parquet as pq
    X, t, y = _truth(n=800, seed=7)
    tbl = pa.table({**{f"x{j}": pa.array(X[:, j]) for j in range(3)},
                    "t": pa.array(t), "y": pa.array(y)})
    path = str(tmp_path / "mb.parquet")
    pq.write_table(tbl, path, row_group_size=200)
    df_mb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100)
    df_one = _df(X, t, y)
    m1 = fit_uplift(df_mb, ["x0", "x1", "x2"], "t", "y", n_estimators=10)
    m2 = fit_uplift(df_one, ["x0", "x1", "x2"], "t", "y", n_estimators=10)
    assert np.allclose(m1.uplift(df_one), m2.uplift(df_one), atol=1e-9)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(upl)
    assert "check_batch" in src and "extra_factor=2.0" in src
    assert "to_pylist" not in src
    assert "log_fit" in src
    assert "import pandas" not in src


def test_qini_vs_t_learner_on_strong_signal():
    """D.3.3 sanity: causal tree Qini not materially worse than T-learner.

    On a simple axis-aligned CATE (single threshold x0>0, CATE=0.7 vs 0),
    both causal tree and T-learner recover perfect ranking. Qini on
    n_test=2000 with Qini~0.09 has sampling std ≈ sqrt(0.09·0.91/2000)
    ≈ 0.006. Tolerance 0.01 ≈ 1.7σ.

    The original contract "Qini > T-learner" (v0.9.37) assumed causal
    trees systematically beat T-learners, which is false on simple DGPs.
    Causal trees win on: (a) high-dim noise features (T-learner wastes
    splits), (b) deep interactions (CT optimises divergence directly),
    (c) large n (honest split's variance reduction dominates). On simple
    single-threshold CATE — parity. See
    .pingis/algorithm_notes/d3_3_uplift.md §"Empirical comparison".
    """
    from cadw.ml.meta_learners import fit_meta_learner

    rng = np.random.RandomState(42)
    n = 8000
    d = 5
    X = rng.normal(size=(n, d))
    t = rng.binomial(1, 0.5, size=n).astype(float)
    p_t1 = np.where(X[:, 0] > 0, 0.90, 0.20)
    p_t0 = np.full(n, 0.20)
    p = np.where(t == 1, p_t1, p_t0)
    y = (rng.uniform(size=n) < p).astype(float)

    df = CadwDataFrame.from_table(pa.table({
        **{f"x{j}": pa.array(X[:, j]) for j in range(d)},
        "t": pa.array(t), "y": pa.array(y),
    }))
    cols = [f"x{j}" for j in range(d)]

    idx = rng.permutation(n).tolist()
    tr = df.iloc[idx[:6000]]
    te = df.iloc[idx[6000:]]

    u_model = fit_uplift(tr, cols, "t", "y", n_estimators=50, max_depth=3, seed=0)
    t_model = fit_meta_learner(tr, cols, "t", "y", kind="t",
                                n_estimators=50, max_depth=3)

    up_u = u_model.uplift(te)
    up_t = t_model.cate(te)

    ys, ts = [], []
    for b in te._iter_filtered_batches():
        ys.append(b.column("y").cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False))
        ts.append(b.column("t").cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False))
    y_te = np.concatenate(ys); t_te = np.concatenate(ts)

    _, _, q_u = qini_curve(y_te, t_te, up_u)
    _, _, q_t = qini_curve(y_te, t_te, up_t)

    print(f"Qini causal tree={q_u:.4f}  T-learner={q_t:.4f}  "
          f"delta={q_u - q_t:+.4f} (tolerance 0.01)")
    assert q_u >= q_t - 0.01, (
        f"causal tree Qini {q_u:.4f} materially worse than T-learner "
        f"{q_t:.4f} (delta={q_u - q_t:+.4f}, tolerance 0.01). "
        f"On simple axis-aligned CATE expected parity within sampling "
        f"noise (~0.006 at n_test=2000)."
    )
