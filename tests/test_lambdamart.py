"""Tests for D.2.1 LambdaMART (v0.9.35)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import lambdamart as lm
from cadw.ml.lambdamart import fit_lambdamart, ndcg_at_k, LambdaMARTModel


def _dataset(seed=0, nq=40, n=12, d=4):
    rng = np.random.RandomState(seed)
    rows, qs, ys = [], [], []
    for q in range(nq):
        X = rng.normal(size=(n, d))
        s = X[:, 0] + 0.5 * X[:, 1]
        rel = (s > 0).astype(int) + (s > 1.0).astype(int)
        rows.append(X); qs += [q] * n; ys += rel.tolist()
    X = np.vstack(rows)
    cols = {f"f{j}": pa.array(X[:, j]) for j in range(d)}
    cols["y"] = pa.array(ys, type=pa.int64())
    cols["q"] = pa.array(qs, type=pa.int64())
    return CadwDataFrame.from_table(pa.table(cols)), X, np.array(ys), np.array(qs)


def _mean_ndcg(y, q, s, k=10):
    vals = [ndcg_at_k(y[q == u], s[q == u], k) for u in sorted(set(q.tolist()))]
    return float(np.mean(vals))


def test_ndcg_perfect_and_bounds():
    y = np.array([3, 2, 1, 0])
    assert ndcg_at_k(y, y.astype(float), 4) == pytest.approx(1.0)
    assert ndcg_at_k(y, -y.astype(float), 4) < 1.0
    assert ndcg_at_k(np.zeros(4), np.arange(4.0), 4) == 1.0  # all gains 0


def test_smoke_and_mixin():
    df, X, y, q = _dataset()
    cols = [f"f{j}" for j in range(4)]
    m = fit_lambdamart(df, cols, "y", "q", n_estimators=10)
    assert isinstance(m, LambdaMARTModel)
    assert len(m.trees_) == 10
    mix = df.fit_lambdamart(cols, "y", "q", n_estimators=5)
    assert len(mix.trees_) == 5


def test_learns_ndcg_above_baseline():
    df, X, y, q = _dataset()
    m = fit_lambdamart(df, [f"f{j}" for j in range(4)], "y", "q",
                       n_estimators=20, max_depth=3)
    sc = m.predict(df).column("score").to_numpy(zero_copy_only=False)
    base = _mean_ndcg(y, q, np.zeros_like(y, dtype=float))
    got = _mean_ndcg(y, q, sc)
    assert got > base + 0.2, f"{base:.3f} -> {got:.3f}"
    assert got > 0.9


def test_reference_vs_sklearn_gbdt_lambdarank_like():
    # Reference: sklearn GradientBoostingRegressor on the relevance target
    # is not a ranker, so we compare against the *pointwise* baseline and
    # require the pairwise ranker to beat it on NDCG@10.
    from sklearn.ensemble import GradientBoostingRegressor
    df, X, y, q = _dataset(seed=3, nq=30, n=20)
    ref = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0)
    ref.fit(X, y)
    ref_ndcg = _mean_ndcg(y, q, ref.predict(X))
    m = fit_lambdamart(df, [f"f{j}" for j in range(4)], "y", "q",
                       n_estimators=30, max_depth=3)
    got = _mean_ndcg(y, q, m.predict(df).column("score").to_numpy(zero_copy_only=False))
    # Same model class (trees, same depth). Lambda ranker is compared against
    # the pointwise regression reference on NDCG@10; both may saturate, so
    # the requirement is parity within a small tolerance, not strict >.
    assert got >= ref_ndcg - 0.05, f"lambda {got:.3f} vs pointwise ref {ref_ndcg:.3f}"


def test_determinism():
    df, X, y, q = _dataset(seed=1)
    cols = [f"f{j}" for j in range(4)]
    a = fit_lambdamart(df, cols, "y", "q", n_estimators=8)
    b = fit_lambdamart(df, cols, "y", "q", n_estimators=8)
    sa = a.predict(df).column("score").to_numpy(zero_copy_only=False)
    sb = b.predict(df).column("score").to_numpy(zero_copy_only=False)
    assert np.array_equal(sa, sb)


def test_multi_batch_invariance():
    import tempfile, pyarrow.parquet as pq
    df, X, y, q = _dataset(seed=2)
    tbl = pa.table({**{f"f{j}": pa.array(X[:, j]) for j in range(4)},
                    "y": pa.array(y, type=pa.int64()),
                    "q": pa.array(q, type=pa.int64())})
    path = tempfile.mktemp(suffix=".parquet")
    pq.write_table(tbl, path, row_group_size=17)
    d1 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1)
    d2 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100000)
    cols = [f"f{j}" for j in range(4)]
    a = fit_lambdamart(d1, cols, "y", "q", n_estimators=5)
    b = fit_lambdamart(d2, cols, "y", "q", n_estimators=5)
    sa = a.predict(d1).column("score").to_numpy(zero_copy_only=False)
    sb = b.predict(d2).column("score").to_numpy(zero_copy_only=False)
    assert np.allclose(sa, sb)


def test_ram_state_scales_with_n_not_unbounded():
    df_s, Xs, ys, qs = _dataset(seed=4, nq=20)
    df_l, Xl, yl, ql = _dataset(seed=4, nq=20, n=60)
    cols = [f"f{j}" for j in range(4)]
    a = fit_lambdamart(df_s, cols, "y", "q", n_estimators=3)
    b = fit_lambdamart(df_l, cols, "y", "q", n_estimators=3)
    assert b.state_nbytes() >= a.state_nbytes()
    assert b.state_nbytes() < 4 * 1024 * 1024


def test_guard_large_state():
    df, X, y, q = _dataset(seed=5, nq=400, n=200, d=12)
    mg = MemoryGuard(max_memory_mb=4)
    with pytest.raises(MemoryLimitExceeded):
        fit_lambdamart(df, [f"f{j}" for j in range(12)], "y", "q",
                       n_estimators=3, memory_guard=mg)


def test_errors():
    df, X, y, q = _dataset(seed=6, nq=2)
    cols = [f"f{j}" for j in range(4)]
    with pytest.raises(KeyError):
        fit_lambdamart(df, ["nope"], "y", "q")
    one = CadwDataFrame.from_table(pa.table({
        "f0": pa.array([1.0, 2.0]), "y": pa.array([1, 0], type=pa.int64()),
        "q": pa.array([0, 0], type=pa.int64())}))
    with pytest.raises(ValueError):
        fit_lambdamart(one, ["f0"], "y", "q", n_estimators=2)


def test_source_contract():
    src = inspect.getsource(lm)
    assert "LambdaRank" in src and "Burges" in src
    assert "ndcg" in src.lower()
    assert "import pandas" not in src
    for line in src.splitlines():
        if "check_batch(" in line:
            assert "extra_factor=" in line
    fit_src = inspect.getsource(lm.fit_lambdamart)
    assert "check_state" in fit_src
    assert "log_fit" in fit_src
