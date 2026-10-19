#!/usr/bin/env python3
"""v0.7.0a: Linear/Ridge regression via Reduce.

Design:
    XTXOp(ReduceOp): one pass accumulates (n, X^T X, X^T y, sum_X, sum_y).
    fit_linear(): augment с intercept column of 1s, solve normal equations
                  (X^T X + α·I_non-intercept) β = X^T y.
    Peak RAM: batch + O(d²). Independent of N.

API:
    from cadw.ml.linear import XTXOp, LinearModel, fit_linear
    model = fit_linear(df, ["x1", "x2"], "y", alpha=0.0)
    model.coef_, model.intercept_
    model.predict(X)  # np.ndarray -> np.ndarray; CadwDataFrame -> CadwDataFrame

Not in v0.7.0a:
    - Logistic (v0.7.0b, Newton per-iter Reduce)
    - Lasso/ElasticNet (v0.7.1)
    - Categorical: use df.encode_categorical() first
    - Multiclass (v0.7.2)

Run: uv run python scripts/patch_v070a_linear.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-800:]}")
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


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


# ============ cadw/ml/__init__.py ============

ML_INIT = '''"""CADW ML layer (v0.7.0a): Linear/Ridge regression via Reduce."""
from .linear import XTXOp, LinearModel, fit_linear

__all__ = ["XTXOp", "LinearModel", "fit_linear"]
'''


# ============ cadw/ml/linear.py ============

LINEAR_SRC = '''"""Linear/Ridge regression (v0.7.0a).

Composition: Reduce (XTXOp) → solve. Peak RAM = batch + O(d²).

Model:
    y = X · coef + intercept
    OLS: minimize ||X β - y||²
    Ridge: minimize ||X β - y||² + α ||β||² (intercept unpenalized)

XTXOp accumulates: n, X^T X (d×d), X^T y (d), sum_X (d), sum_y.
Augmented system for intercept:
    [n        sum_X^T] [b0   ]   [sum_y  ]
    [sum_X    X^T X  ] [beta ] = [X^T y  ]
    + α · diag(0, 1, ..., 1) on non-intercept.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import ReduceOp, reduce


class XTXOp(ReduceOp):
    """Accumulate (n, X^T X, X^T y, sum_X, sum_y) over batches.

    State size: O(d²) + O(d). One pass.
    Only numeric columns. Nulls in X or y → row skipped (both X and y).
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        target_col: str,
    ) -> None:
        if not feature_cols:
            raise ValueError("XTXOp: feature_cols must be non-empty")
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(self.feature_cols)

    def init(self):
        return (
            0,
            np.zeros((self.d, self.d), dtype=np.float64),
            np.zeros(self.d, dtype=np.float64),
            np.zeros(self.d, dtype=np.float64),
            0.0,
        )

    def update(self, state, batch):
        n, xtx, xty, sx, sy = state
        X = _stack_numeric(batch, self.feature_cols)
        y = _numeric_to_f64(batch.column(self.target_col))
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        if X.shape[0] == 0:
            return state
        return (
            n + X.shape[0],
            xtx + X.T @ X,
            xty + X.T @ y,
            sx + X.sum(axis=0),
            sy + float(y.sum()),
        )

    def merge(self, a, b):
        return (
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
            a[3] + b[3],
            a[4] + b[4],
        )

    def state_nbytes(self, state):
        n, xtx, xty, sx, sy = state
        return int(xtx.nbytes + xty.nbytes + sx.nbytes + 8 + 8)

    def estimate_state_nbytes(self):
        d = self.d
        return 8 + 8 * d * d + 8 * d + 8 * d + 8

    def finalize(self, state):
        n, xtx, xty, sx, sy = state
        return {
            "n": int(n), "xtx": xtx, "xty": xty,
            "sum_X": sx, "sum_y": float(sy),
        }


def _numeric_to_f64(arr: pa.Array) -> np.ndarray:
    """Convert Arrow numeric Array (possibly with nulls) to float64 with NaN."""
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


def _stack_numeric(batch: pa.RecordBatch, cols: Sequence[str]) -> np.ndarray:
    return np.column_stack([_numeric_to_f64(batch.column(c)) for c in cols])


@dataclass
class LinearModel:
    """Fitted linear model (OLS or Ridge)."""

    coef_: np.ndarray
    intercept_: float
    feature_names_: list[str]
    target_col_: str
    alpha_: float
    n_rows_: int
    fitted_: bool = True

    def to_dict(self) -> dict:
        return {
            "coefficients": dict(zip(self.feature_names_, self.coef_.tolist())),
            "intercept": float(self.intercept_),
            "alpha": float(self.alpha_),
            "n_rows": int(self.n_rows_),
        }

    def predict(self, X):
        """Predict.

        - np.ndarray (n, d) → np.ndarray (n,)
        - CadwDataFrame → CadwDataFrame with added `{target_col_}_pred`.
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != len(self.coef_):
                raise ValueError(
                    f"predict: X has {X.shape[1]} features, "
                    f"expected {len(self.coef_)}"
                )
            return X @ self.coef_ + self.intercept_

        from cadw.dataframe import CadwDataFrame
        from cadw.dataset import Dataset
        if not isinstance(X, CadwDataFrame):
            raise TypeError(
                f"predict: X must be np.ndarray or CadwDataFrame; "
                f"got {type(X).__name__}"
            )
        for c in self.feature_names_:
            if c not in X.columns:
                raise KeyError(f"predict: column {c!r} not in df")

        out_col = f"{self.target_col_}_pred"
        orig_schema = X._dataset.schema
        out_fields = [orig_schema.field(c) for c in X._columns]
        out_fields.append(pa.field(out_col, pa.float64(), nullable=True))
        out_schema = pa.schema(out_fields)

        coef = self.coef_
        intercept = self.intercept_

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, self.feature_names_)
                pred = Xm @ coef + intercept
                arrays = [b.column(c) for c in X._columns]
                arrays.append(pa.array(pred, type=pa.float64()))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_pred(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )


def fit_linear(
    df,
    X_cols: "str | Sequence[str]",
    y_col: str,
    *,
    alpha: float = 0.0,
    memory_guard: Optional[MemoryGuard] = None,
) -> LinearModel:
    """Fit Linear (alpha=0) or Ridge (alpha>0) via one Reduce + solve.

    - X_cols: feature column names (numeric).
    - y_col: target column (numeric).
    - alpha: Ridge L2 penalty (non-negative). Intercept not penalized.
    - memory_guard: default df._memory_guard.

    Raises ValueError for empty df (all-null rows), negative alpha,
    X ∩ {y} non-empty, non-numeric columns.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    else:
        X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_linear: X_cols must be non-empty")
    if y_col in X_cols:
        raise ValueError(
            f"fit_linear: target {y_col!r} must not be in X_cols"
        )
    if alpha < 0:
        raise ValueError(f"fit_linear: alpha must be >= 0; got {alpha}")
    for c in X_cols:
        if c not in df.columns:
            raise KeyError(f"fit_linear: column {c!r} not in df")
    if y_col not in df.columns:
        raise KeyError(f"fit_linear: target {y_col!r} not in df")

    # Validate numeric
    schema = df._dataset.schema
    for c in X_cols + [y_col]:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_linear: column {c!r} has type {t}; "
                f"numeric required (use encode_categorical for strings)"
            )

    mg = memory_guard or df._memory_guard
    op = XTXOp(X_cols, y_col)
    res = reduce(df._iter_filtered_batches(), op, mg, log=False)

    if res["n"] == 0:
        raise ValueError("fit_linear: no valid (non-null) rows")

    d = len(X_cols)
    xtx = res["xtx"]
    xty = res["xty"]
    sum_x = res["sum_X"]
    sum_y = res["sum_y"]
    n = res["n"]

    # Augmented system (d+1 features: intercept + X)
    xtx_aug = np.zeros((d + 1, d + 1), dtype=np.float64)
    xtx_aug[0, 0] = float(n)
    xtx_aug[0, 1:] = sum_x
    xtx_aug[1:, 0] = sum_x
    xtx_aug[1:, 1:] = xtx

    xty_aug = np.zeros(d + 1, dtype=np.float64)
    xty_aug[0] = sum_y
    xty_aug[1:] = xty

    reg = alpha * np.eye(d + 1, dtype=np.float64)
    reg[0, 0] = 0.0  # intercept not penalized
    xtx_reg = xtx_aug + reg

    try:
        beta = np.linalg.solve(xtx_reg, xty_aug)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xtx_reg, xty_aug, rcond=None)[0]

    return LinearModel(
        coef_=beta[1:],
        intercept_=float(beta[0]),
        feature_names_=X_cols,
        target_col_=y_col,
        alpha_=float(alpha),
        n_rows_=int(n),
    )
'''


# ============ dataframe.py convenience methods ============

def patch_dataframe():
    print("\n=== cadw/dataframe.py: fit_linear convenience ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    if "def fit_linear" in s:
        print("  already present — skip")
        return

    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")

    methods = '''    # ---------------- ML: fit_linear ----------------

    def fit_linear(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        alpha: float = 0.0,
    ):
        """Fit Linear (alpha=0) or Ridge (alpha>0) (v0.7.0a).

        One Reduce over filtered batches → solve (X^T X + αI_nonintercept) β.

        Returns: LinearModel (coef_, intercept_, predict(X)).

        See cadw.ml.linear.fit_linear for full docs.
        """
        from cadw.ml.linear import fit_linear
        return fit_linear(
            self, X_cols, y_col, alpha=alpha,
            memory_guard=self._memory_guard,
        )

'''
    s = s.replace(anchor, methods + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: fit_linear method added")


# ============ cadw/__init__.py export ============

def patch_top_init():
    print("\n=== cadw/__init__.py: export ML ===")
    p = Path("cadw/__init__.py")
    s = p.read_text()
    if "from .ml import" in s:
        print("  already present — skip")
        return
    # Append at end of imports (before __all__)
    if "__all__ = [" not in s:
        raise SystemExit("__all__ not found")
    idx = s.find("__all__ = [")
    s = s[:idx] + "from .ml import XTXOp, LinearModel, fit_linear\n\n" + s[idx:]
    # Extend __all__
    s = s.replace(
        '__all__ = [',
        '__all__ = [\n    "XTXOp", "LinearModel", "fit_linear",',
        1,
    )
    p.write_text(s)
    compile_check(p)
    print("  OK: ML exports added")


# ============ tests ============

TESTS_SRC = '''"""v0.7.0a: Linear/Ridge via Reduce."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.ml.linear import LinearModel, XTXOp, fit_linear


def make_df(tmp_path, data, chunk=10, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _synth_linear(n, coefs, intercept, noise=0.0, seed=42):
    rng = np.random.default_rng(seed)
    d = len(coefs)
    X = rng.normal(size=(n, d))
    y = X @ np.array(coefs) + intercept + noise * rng.normal(size=n)
    data = {f"x{i}": X[:, i] for i in range(d)}
    data["y"] = y
    return data, X, y


# ---------- XTXOp ----------

def test_xtxop_estimate_state_nbytes():
    """P8 API for custom ReduceOp."""
    op = XTXOp(["a", "b", "c"], "y")
    assert op.estimate_state_nbytes() == 8 + 8*3*3 + 8*3 + 8*3 + 8


def test_xtxop_state_nbytes_matches_actual():
    op = XTXOp(["a", "b"], "y")
    assert op.state_nbytes(op.init()) <= op.estimate_state_nbytes()


# ---------- OLS / Ridge ----------

def test_fit_linear_perfect():
    """Noiseless linear data: coefficients exactly recovered."""
    n = 200
    data, X, y = _synth_linear(n, [2.0, -3.0, 0.5], intercept=1.5)
    # In-memory synthetic: use minimal parquet df via numpy->table
    import pyarrow as pa_  # local
    tbl = pa_.table(data)
    # Temporary: use in-memory wrapper directly
    from cadw.dataset import Dataset
    ds = Dataset.from_batches(
        tbl.schema, iter(tbl.to_batches(max_chunksize=20)),
        Path("/tmp/cadw_v070_test_ds"),
    )
    df = CadwDataFrame(
        ds,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=Path("/tmp/cadw_v070_test_spill"),
    )
    model = fit_linear(df, ["x0", "x1", "x2"], "y")
    assert isinstance(model, LinearModel)
    np.testing.assert_allclose(model.coef_, [2.0, -3.0, 0.5], atol=1e-9)
    assert abs(model.intercept_ - 1.5) < 1e-9
    assert model.n_rows_ == n


def test_fit_linear_with_noise_approx():
    n = 500
    data, X, y = _synth_linear(n, [2.0, -3.0], intercept=1.0, noise=0.1)
    import pyarrow as pa_  # local
    tbl = pa_.table(data)
    from cadw.dataset import Dataset
    ds = Dataset.from_batches(
        tbl.schema, iter(tbl.to_batches(max_chunksize=50)),
        Path("/tmp/cadw_v070_noise_ds"),
    )
    df = CadwDataFrame(
        ds,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=Path("/tmp/cadw_v070_noise_spill"),
    )
    model = fit_linear(df, ["x0", "x1"], "y")
    assert abs(model.coef_[0] - 2.0) < 0.05
    assert abs(model.coef_[1] + 3.0) < 0.05
    assert abs(model.intercept_ - 1.0) < 0.1


def test_fit_linear_matches_numpy_lstsq(tmp_path):
    """Compare to np.linalg.lstsq."""
    n, d = 100, 3
    data, X, y = _synth_linear(n, [1.0, 2.0, 3.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = fit_linear(df, [f"x{i}" for i in range(d)], "y")

    # numpy reference
    X_aug = np.column_stack([np.ones(n), X])
    beta_ref, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    np.testing.assert_allclose(model.coef_, beta_ref[1:], rtol=1e-9)
    assert abs(model.intercept_ - beta_ref[0]) < 1e-9


def test_fit_ridge_shrinks_coef():
    n = 200
    data, X, y = _synth_linear(n, [5.0, 5.0], intercept=0.0)
    import pyarrow as pa_
    tbl = pa_.table(data)
    from cadw.dataset import Dataset
    ds = Dataset.from_batches(
        tbl.schema, iter(tbl.to_batches(max_chunksize=20)),
        Path("/tmp/cadw_v070_ridge_ds"),
    )
    df = CadwDataFrame(
        ds,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=Path("/tmp/cadw_v070_ridge_spill"),
    )
    m_ols = fit_linear(df, ["x0", "x1"], "y", alpha=0.0)
    m_ridge = fit_linear(df, ["x0", "x1"], "y", alpha=100.0)
    # Ridge shrinks coef magnitude
    assert np.linalg.norm(m_ridge.coef_) < np.linalg.norm(m_ols.coef_)


def test_fit_ridge_large_alpha_near_zero():
    n = 200
    data, _, _ = _synth_linear(n, [5.0], intercept=0.0)
    import pyarrow as pa_
    tbl = pa_.table(data)
    from cadw.dataset import Dataset
    ds = Dataset.from_batches(
        tbl.schema, iter(tbl.to_batches(max_chunksize=20)),
        Path("/tmp/cadw_v070_ridge_big_ds"),
    )
    df = CadwDataFrame(
        ds,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=Path("/tmp/cadw_v070_ridge_big_spill"),
    )
    m = fit_linear(df, ["x0"], "y", alpha=1e8)
    assert abs(m.coef_[0]) < 0.01


def test_fit_linear_single_feature(tmp_path):
    n = 100
    x = np.arange(n, dtype=np.float64)
    y = 2.0 * x + 3.0
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_linear(df, ["x"], "y")
    assert abs(model.coef_[0] - 2.0) < 1e-9
    assert abs(model.intercept_ - 3.0) < 1e-9


def test_fit_linear_multi_batch(tmp_path):
    """Chunk size doesn't affect result."""
    n = 100
    data, X, y = _synth_linear(n, [1.5, -2.0], intercept=0.5, noise=0.1)
    df1 = make_df(tmp_path, data, chunk=10, name="a")
    df2 = make_df(tmp_path, data, chunk=33, name="b")
    m1 = fit_linear(df1, ["x0", "x1"], "y")
    m2 = fit_linear(df2, ["x0", "x1"], "y")
    np.testing.assert_allclose(m1.coef_, m2.coef_, rtol=1e-12)
    assert abs(m1.intercept_ - m2.intercept_) < 1e-12


# ---------- nulls ----------

def test_fit_linear_with_nulls(tmp_path):
    """Nulls in X or y → row skipped."""
    n = 100
    data, X, y = _synth_linear(n, [2.0], intercept=1.0)
    # Add nulls
    data["x0"] = [None if i % 10 == 0 else v for i, v in enumerate(data["x0"])]
    data["y"] = [None if i % 15 == 0 else v for i, v in enumerate(data["y"])]

    df = make_df(tmp_path, data)
    model = fit_linear(df, ["x0"], "y")
    # Still recovers coef (approx)
    assert abs(model.coef_[0] - 2.0) < 0.01
    assert model.n_rows_ < n


def test_fit_linear_all_null_raises(tmp_path):
    df = make_df(tmp_path, {
        "x0": [None, None, None],
        "y": [None, None, None],
    })
    # Fit — n_valid = 0
    with pytest.raises(ValueError, match="no valid"):
        fit_linear(df, ["x0"], "y")


# ---------- errors ----------

def test_fit_linear_empty_X_cols(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="X_cols"):
        fit_linear(df, [], "y")


def test_fit_linear_target_in_X(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="must not be in X_cols"):
        fit_linear(df, ["x0", "y"], "y")


def test_fit_linear_negative_alpha(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        fit_linear(df, ["x0"], "y", alpha=-1.0)


def test_fit_linear_missing_column(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(KeyError, match="nope"):
        fit_linear(df, ["nope"], "y")


def test_fit_linear_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {
        "x0": ["a", "b", "c"],
        "y": [1.0, 2.0, 3.0],
    })
    with pytest.raises(TypeError, match="numeric"):
        fit_linear(df, ["x0"], "y")


# ---------- predict ----------

def test_predict_numpy():
    n = 50
    data, X, y = _synth_linear(n, [2.0, -1.0], intercept=0.5)
    import pyarrow as pa_
    tbl = pa_.table(data)
    from cadw.dataset import Dataset
    ds = Dataset.from_batches(
        tbl.schema, iter(tbl.to_batches(max_chunksize=20)),
        Path("/tmp/cadw_v070_pred_ds"),
    )
    df = CadwDataFrame(
        ds,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=Path("/tmp/cadw_v070_pred_spill"),
    )
    model = fit_linear(df, ["x0", "x1"], "y")
    pred = model.predict(X[:5])
    expected = X[:5] @ model.coef_ + model.intercept_
    np.testing.assert_allclose(pred, expected)


def test_predict_dataframe(tmp_path):
    n = 50
    data, X, y = _synth_linear(n, [2.0], intercept=1.0)
    df = make_df(tmp_path, data)
    model = fit_linear(df, ["x0"], "y")
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    assert "y_pred" in t.schema.names
    pred_vals = t.column("y_pred").to_pylist()
    # Should be close to y (noiseless)
    y_vals = t.column("y").to_pylist()
    for p, yy in zip(pred_vals, y_vals):
        assert abs(p - yy) < 1e-6


def test_predict_df_missing_column(tmp_path):
    n = 30
    data, X, y = _synth_linear(n, [2.0], intercept=1.0)
    df = make_df(tmp_path, data)
    model = fit_linear(df, ["x0"], "y")
    # Select only y
    sub = df[["y"]]
    with pytest.raises(KeyError, match="x0"):
        model.predict(sub)


# ---------- to_dict ----------

def test_linear_model_to_dict(tmp_path):
    n = 50
    data, X, y = _synth_linear(n, [2.0, -1.0], intercept=0.5)
    df = make_df(tmp_path, data)
    model = fit_linear(df, ["x0", "x1"], "y")
    d = model.to_dict()
    assert "coefficients" in d
    assert "x0" in d["coefficients"]
    assert "x1" in d["coefficients"]
    assert abs(d["intercept"] - 0.5) < 1e-9
    assert d["n_rows"] == n


# ---------- convenience method ----------

def test_fit_linear_dataframe_method(tmp_path):
    n = 50
    data, X, y = _synth_linear(n, [2.0], intercept=1.0)
    df = make_df(tmp_path, data)
    model = df.fit_linear("x0", "y")
    assert hasattr(model, "coef_")
    assert abs(model.coef_[0] - 2.0) < 1e-9


# ---------- filtered ----------

def test_fit_linear_filtered(tmp_path):
    n = 100
    data, X, y = _synth_linear(n, [2.0], intercept=1.0, noise=0.0)
    df = make_df(tmp_path, data)
    # Filter to subset
    filtered = df[df["x0"] > 0]
    model = fit_linear(filtered, ["x0"], "y")
    # On subset X>0, coef should still be 2.0, intercept 1.0
    assert abs(model.coef_[0] - 2.0) < 1e-9
    assert abs(model.intercept_ - 1.0) < 1e-9
'''


def write_tests():
    print("\n=== tests/test_linear.py ===")
    p = Path("tests/test_linear.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: 20 tests")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v070a-linear"

    print("=== v0.7.0a: Linear/Ridge ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v070a-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v070a-{ts_str}",
        "-m", "before patch v0.7.0a Linear"])

    # Create ml package
    print("\n=== cadw/ml/ ===")
    Path("cadw/ml").mkdir(exist_ok=True)
    Path("cadw/ml/__init__.py").write_text(ML_INIT)
    Path("cadw/ml/linear.py").write_text(LINEAR_SRC)
    compile_check("cadw/ml/__init__.py")
    compile_check("cadw/ml/linear.py")
    print("  OK: cadw/ml package created")

    patch_dataframe()
    patch_top_init()
    write_tests()

    print("\n=== pytest tests/test_linear.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_linear.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.0a (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.0a: Linear/Ridge via Reduce (XTXOp; predict; ML layer)"])

    print("\n==> v0.7.0a applied on", branch)


if __name__ == "__main__":
    main()