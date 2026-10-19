#!/usr/bin/env python3
"""v0.7.0b: Logistic regression via iter-Reduce Newton IRLS.

Design:
    Each iteration: Reduce accumulates
        grad = X^T (p - y)
        hess = X^T W X  (W = diag(p*(1-p)))
        log-likelihood
    Then solve (hess + αI) step = grad; β ← β - step.
    Stop when max|step| < tol.

    Peak RAM: batch + O(d²). Independent of N.
    Iterations: typically 5-15 for well-conditioned data.

API:
    from cadw.ml.logistic import LogisticModel, fit_logistic, LogisticGradOp
    model = fit_logistic(df, ["x1", "x2"], "y", alpha=1.0, max_iter=100, tol=1e-6)
    model.coef_, model.intercept_, model.n_iter_, model.converged_
    model.predict_proba(X)  # np.ndarray -> [0,1]
    model.predict(X, threshold=0.5)  # np.ndarray -> {0,1}

    df.fit_logistic(X_cols, y_col, ...)  # convenience

Constraints:
    - y must be binary {0, 1}. Both classes required.
    - Nulls in X or y → row skipped.
    - alpha >= 0 (L2 penalty on non-intercept). Default 1.0 (stability).
    - Categorical → df.encode_categorical() first.

Not in v0.7.0b:
    - Multiclass softmax (v0.7.2)
    - L1 / ElasticNet (v0.7.1)

Run: uv run python scripts/patch_v070b_logistic.py
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


# ============ cadw/ml/logistic.py ============

LOGISTIC_SRC = '''"""Logistic regression (v0.7.0b).

Composition: iter-Reduce Newton IRLS + solve. Peak RAM = batch + O(d²).

Model:
    P(y=1 | x) = sigmoid(x · coef + intercept)

Optimization (Newton-Raphson on log-likelihood):
    grad = X^T (p - y)                    # d
    hess = X^T diag(p*(1-p)) X            # d×d
    step = solve(hess + α·I, grad)        # α on non-intercept
    β ← β - step

    Each iteration: one Reduce pass over df.

Stopping: ||step||_inf < tol, or max_iter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import ReduceOp, reduce


class LogisticGradOp(ReduceOp):
    """One Newton iteration: accumulate (grad, hess, log-likelihood).

    State: O(d²). Independent of N.

    Beta is passed from previous iteration; y must be binary {0,1}.
    Rows with null in X or y are skipped.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        target_col: str,
        beta: np.ndarray,
    ) -> None:
        if not feature_cols:
            raise ValueError("LogisticGradOp: feature_cols must be non-empty")
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.d = len(self.feature_cols) + 1  # + intercept
        self.beta = np.asarray(beta, dtype=np.float64)
        if self.beta.shape != (self.d,):
            raise ValueError(
                f"LogisticGradOp: beta shape {self.beta.shape}, "
                f"expected ({self.d},)"
            )

    def init(self):
        return (
            0,                                # n_valid
            np.zeros(self.d, dtype=np.float64),
            np.zeros((self.d, self.d), dtype=np.float64),
            0.0,                              # log-likelihood
        )

    def update(self, state, batch):
        n, grad, hess, ll = state
        X, y = _stack_and_validate(batch, self.feature_cols, self.target_col)
        if X.shape[0] == 0:
            return state
        X_aug = np.column_stack([np.ones(X.shape[0]), X])  # (B, d)
        z = X_aug @ self.beta                              # (B,)
        # Numerically stable sigmoid
        p = 0.5 * (1.0 + np.tanh(0.5 * z))                 # (B,)
        w = p * (1.0 - p)                                  # (B,)
        # Grad / hess
        grad_new = X_aug.T @ (p - y)
        hess_new = (X_aug * w[:, None]).T @ X_aug
        # Log-likelihood (clip for numerical stability)
        eps = 1e-15
        p_clip = np.clip(p, eps, 1.0 - eps)
        ll_new = float(
            (y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)).sum()
        )
        return (
            n + X.shape[0],
            grad + grad_new,
            hess + hess_new,
            ll + ll_new,
        )

    def merge(self, a, b):
        return (
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
            a[3] + b[3],
        )

    def state_nbytes(self, state):
        n, grad, hess, ll = state
        return int(grad.nbytes + hess.nbytes + 8 + 8)

    def estimate_state_nbytes(self):
        d = self.d
        return 8 + 8 * d + 8 * d * d + 8

    def finalize(self, state):
        n, grad, hess, ll = state
        return {
            "n": int(n), "grad": grad, "hess": hess, "ll": float(ll),
        }


def _stack_and_validate(
    batch: pa.RecordBatch,
    feature_cols: Sequence[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract numeric X, y; mask rows with null or non-binary y."""
    X_parts = [_column_to_f64(batch.column(c)) for c in feature_cols]
    X = np.column_stack(X_parts)
    y = _column_to_f64(batch.column(target_col))
    # Mask: no NaN in X or y; y in {0, 1}
    mask = (
        ~np.isnan(X).any(axis=1)
        & ~np.isnan(y)
        & ((y == 0.0) | (y == 1.0))
    )
    return X[mask], y[mask]


def _column_to_f64(arr: pa.Array) -> np.ndarray:
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


def _validate_binary_y(df, y_col: str) -> set:
    """Ensure y has exactly values in {0, 1}, both present. O(distinct)."""
    seen: set = set()
    for b in df._iter_filtered_batches():
        col = b.column(y_col)
        # ChunkedArray.unique returns Array
        try:
            uniq = col.unique()
        except Exception:
            uniq = None
        if uniq is not None:
            for v in uniq.to_pylist():
                if v is None:
                    continue
                fv = float(v)
                if fv != 0.0 and fv != 1.0:
                    raise ValueError(
                        f"fit_logistic: y must be binary (0/1); "
                        f"got {v!r}"
                    )
                seen.add(fv)
                if len(seen) == 2:
                    # Both classes present, but still scan rest for invalid
                    pass
        else:
            for v in col.to_pylist():
                if v is None:
                    continue
                fv = float(v)
                if fv != 0.0 and fv != 1.0:
                    raise ValueError(
                        f"fit_logistic: y must be binary (0/1); "
                        f"got {v!r}"
                    )
                seen.add(fv)
    if len(seen) < 2:
        raise ValueError(
            f"fit_logistic: y must have both classes 0 and 1; got {seen}"
        )
    return seen


@dataclass
class LogisticModel:
    """Fitted logistic regression model."""

    coef_: np.ndarray
    intercept_: float
    feature_names_: list[str]
    target_col_: str
    alpha_: float
    n_rows_: int
    n_iter_: int
    converged_: bool
    log_likelihood_: float
    fitted_: bool = True

    def to_dict(self) -> dict:
        return {
            "coefficients": dict(zip(self.feature_names_, self.coef_.tolist())),
            "intercept": float(self.intercept_),
            "alpha": float(self.alpha_),
            "n_rows": int(self.n_rows_),
            "n_iter": int(self.n_iter_),
            "converged": bool(self.converged_),
            "log_likelihood": float(self.log_likelihood_),
        }

    def predict_proba(self, X):
        """P(y=1|x). Returns np.ndarray for np.ndarray input; for
        CadwDataFrame adds `{target}_prob` column and returns CadwDataFrame.
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != len(self.coef_):
                raise ValueError(
                    f"predict_proba: X has {X.shape[1]} features, "
                    f"expected {len(self.coef_)}"
                )
            z = X @ self.coef_ + self.intercept_
            return 0.5 * (1.0 + np.tanh(0.5 * z))

        return self._predict_df(X, proba_only=True)

    def predict(self, X, threshold: float = 0.5):
        """Binary prediction (>= threshold → 1).

        For CadwDataFrame: adds BOTH `{target}_prob` and `{target}_pred`.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"predict: threshold must be in [0, 1]; got {threshold}"
            )
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            proba = self.predict_proba(X)
            return (proba >= threshold).astype(np.int64)
        return self._predict_df(X, proba_only=False, threshold=threshold)

    def _predict_df(self, X, *, proba_only: bool, threshold: float = 0.5):
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

        prob_col = f"{self.target_col_}_prob"
        pred_col = f"{self.target_col_}_pred"

        orig_schema = X._dataset.schema
        out_fields = [orig_schema.field(c) for c in X._columns]
        out_fields.append(pa.field(prob_col, pa.float64(), nullable=True))
        if not proba_only:
            out_fields.append(pa.field(pred_col, pa.int64(), nullable=True))
        out_schema = pa.schema(out_fields)

        coef = self.coef_
        intercept = self.intercept_

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = np.column_stack(
                    [_column_to_f64(b.column(c)) for c in self.feature_names_]
                )
                z = Xm @ coef + intercept
                prob = 0.5 * (1.0 + np.tanh(0.5 * z))
                arrays = [b.column(c) for c in X._columns]
                arrays.append(pa.array(prob, type=pa.float64()))
                if not proba_only:
                    pred = (prob >= threshold).astype(np.int64)
                    arrays.append(pa.array(pred, type=pa.int64()))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_pred(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )


def fit_logistic(
    df,
    X_cols,
    y_col: str,
    *,
    alpha: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    memory_guard: Optional[MemoryGuard] = None,
) -> LogisticModel:
    """Fit binary logistic regression via Newton-Raphson IRLS.

    - X_cols: feature columns (numeric). Use encode_categorical() for strings.
    - y_col: target column, must be binary {0,1}, both classes required.
    - alpha: L2 penalty (non-intercept). Default 1.0 (stability).
    - max_iter, tol: stopping criteria.
    - memory_guard: default df._memory_guard.

    Iterations: typically 5-15; each is one full Reduce pass.
    Peak RAM = batch + O(d²). Independent of N.

    Raises:
        ValueError: empty X, target in X, negative alpha, all-null rows,
                    non-binary y, only one class in y.
        KeyError: missing column.
        TypeError: non-numeric column.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    else:
        X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_logistic: X_cols must be non-empty")
    if y_col in X_cols:
        raise ValueError(
            f"fit_logistic: target {y_col!r} must not be in X_cols"
        )
    if alpha < 0:
        raise ValueError(f"fit_logistic: alpha must be >= 0; got {alpha}")
    if max_iter < 1:
        raise ValueError(f"fit_logistic: max_iter must be >= 1; got {max_iter}")
    if tol <= 0:
        raise ValueError(f"fit_logistic: tol must be > 0; got {tol}")
    for c in X_cols:
        if c not in df.columns:
            raise KeyError(f"fit_logistic: column {c!r} not in df")
    if y_col not in df.columns:
        raise KeyError(f"fit_logistic: target {y_col!r} not in df")

    schema = df._dataset.schema
    for c in X_cols + [y_col]:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_logistic: column {c!r} has type {t}; "
                f"numeric required (use encode_categorical for strings)"
            )

    # Validate binary y + both classes
    _validate_binary_y(df, y_col)

    mg = memory_guard or df._memory_guard
    d = len(X_cols) + 1  # + intercept
    beta = np.zeros(d, dtype=np.float64)
    ll_prev = -np.inf
    converged = False
    n_iter = 0
    n_valid = 0
    ll = 0.0

    for it in range(max_iter):
        n_iter = it + 1
        op = LogisticGradOp(X_cols, y_col, beta)
        res = reduce(df._iter_filtered_batches(), op, mg, log=False)
        n_valid = res["n"]
        if n_valid == 0:
            raise ValueError("fit_logistic: no valid (non-null, binary) rows")
        grad = res["grad"]
        hess = res["hess"]
        ll = res["ll"]

        reg = alpha * np.eye(d, dtype=np.float64)
        reg[0, 0] = 0.0  # intercept not penalized
        try:
            step = np.linalg.solve(hess + reg, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess + reg, grad, rcond=None)[0]

        step_inf = float(np.max(np.abs(step)))
        beta = beta - step

        if step_inf < tol:
            converged = True
            break

        # Δll check — safety net if Newton oscillates
        if abs(ll - ll_prev) < tol * 0.1 and it >= 1:
            converged = True
            break
        ll_prev = ll

    return LogisticModel(
        coef_=beta[1:],
        intercept_=float(beta[0]),
        feature_names_=list(X_cols),
        target_col_=y_col,
        alpha_=float(alpha),
        n_rows_=int(n_valid),
        n_iter_=int(n_iter),
        converged_=bool(converged),
        log_likelihood_=float(ll),
    )
'''


# ============ patch dataframe.py ============

def patch_dataframe():
    print("\n=== cadw/dataframe.py: fit_logistic method ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    if "def fit_logistic" in s:
        print("  already present — skip")
        return

    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")

    methods = '''    # ---------------- ML: fit_logistic ----------------

    def fit_logistic(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        alpha: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        """Fit binary logistic via Newton IRLS (v0.7.0b).

        Each Newton iteration = one Reduce pass. Typically 5-15 iterations.

        Returns LogisticModel (coef_, intercept_, predict, predict_proba).

        See cadw.ml.logistic.fit_logistic for full docs.
        """
        from cadw.ml.logistic import fit_logistic
        return fit_logistic(
            self, X_cols, y_col,
            alpha=alpha, max_iter=max_iter, tol=tol,
            memory_guard=self._memory_guard,
        )

'''
    s = s.replace(anchor, methods + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: fit_logistic method added")


# ============ cadw/ml/__init__.py: export ============

def patch_ml_init():
    print("\n=== cadw/ml/__init__.py: export logistic ===")
    p = Path("cadw/ml/__init__.py")
    p.write_text('''"""CADW ML layer (v0.7.0b)."""
from .linear import XTXOp, LinearModel, fit_linear
from .logistic import LogisticGradOp, LogisticModel, fit_logistic

__all__ = [
    "XTXOp", "LinearModel", "fit_linear",
    "LogisticGradOp", "LogisticModel", "fit_logistic",
]
''')
    compile_check(p)
    print("  OK: ml/__init__.py updated")


# ============ cadw/__init__.py: export ============

def patch_top_init():
    print("\n=== cadw/__init__.py: export logistic ===")
    p = Path("cadw/__init__.py")
    s = p.read_text()

    if "LogisticModel" in s:
        print("  already present — skip")
        return

    s = s.replace(
        "from .ml import XTXOp, LinearModel, fit_linear",
        "from .ml import (\n"
        "    XTXOp, LinearModel, fit_linear,\n"
        "    LogisticGradOp, LogisticModel, fit_logistic,\n"
        ")",
        1,
    )
    s = s.replace(
        '__all__ = [\n    "XTXOp", "LinearModel", "fit_linear",',
        '__all__ = [\n'
        '    "XTXOp", "LinearModel", "fit_linear",\n'
        '    "LogisticGradOp", "LogisticModel", "fit_logistic",',
        1,
    )
    p.write_text(s)
    compile_check(p)
    print("  OK: top-level exports added")


# ============ tests ============

TESTS_SRC = '''"""v0.7.0b: Logistic regression via iter-Reduce Newton IRLS."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.ml.logistic import LogisticGradOp, LogisticModel, fit_logistic


def make_df(tmp_path, data, chunk=20, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _synth_logistic(n, coefs, intercept, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    d = len(coefs)
    X = rng.normal(size=(n, d))
    z = X @ np.array(coefs) + intercept + noise * rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p).astype(np.int64)
    data = {f"x{i}": X[:, i] for i in range(d)}
    data["y"] = y
    return data, X, y


# ---------- op ----------

def test_logistic_grad_op_estimate_state_nbytes():
    op = LogisticGradOp(["a", "b", "c"], "y", np.zeros(4))
    assert op.estimate_state_nbytes() == 8 + 8*4 + 8*4*4 + 8


def test_logistic_grad_op_state_nbytes_matches():
    op = LogisticGradOp(["a", "b"], "y", np.zeros(3))
    assert op.state_nbytes(op.init()) <= op.estimate_state_nbytes()


# ---------- basic fit ----------

def test_fit_logistic_noisy(tmp_path):
    n = 1000
    data, X, y = _synth_logistic(n, [2.0, -1.5], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0", "x1"], "y", alpha=1.0)
    assert isinstance(model, LogisticModel)
    # Right sign
    assert model.coef_[0] > 0.5   # true +2
    assert model.coef_[1] < -0.3  # true -1.5
    # Reasonable accuracy
    proba = model.predict_proba(X)
    pred = (proba >= 0.5).astype(int)
    acc = (pred == y).mean()
    assert acc > 0.85


def test_fit_logistic_converges(tmp_path):
    n = 500
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0, noise=0.1)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0, max_iter=100, tol=1e-6)
    assert model.converged_
    assert model.n_iter_ < 100
    assert model.n_iter_ >= 1


def test_fit_logistic_multibatch_consistent(tmp_path):
    n = 300
    data, X, y = _synth_logistic(n, [1.0, -0.5], intercept=0.5)
    df1 = make_df(tmp_path, data, chunk=10, name="a")
    df2 = make_df(tmp_path, data, chunk=100, name="b")
    m1 = fit_logistic(df1, ["x0", "x1"], "y", alpha=1.0)
    m2 = fit_logistic(df2, ["x0", "x1"], "y", alpha=1.0)
    np.testing.assert_allclose(m1.coef_, m2.coef_, rtol=1e-6, atol=1e-6)


def test_fit_logistic_alpha_shrinks(tmp_path):
    n = 500
    data, X, y = _synth_logistic(n, [3.0, -3.0], intercept=0.0)
    df = make_df(tmp_path, data)
    m_small = fit_logistic(df, ["x0", "x1"], "y", alpha=0.01)
    m_large = fit_logistic(df, ["x0", "x1"], "y", alpha=100.0)
    assert np.linalg.norm(m_large.coef_) < np.linalg.norm(m_small.coef_)


# ---------- predict ----------

def test_predict_proba_numpy(tmp_path):
    n = 200
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_numpy(tmp_path):
    n = 200
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    pred = model.predict(X[:5])
    assert pred.shape == (5,)
    assert set(pred.tolist()) <= {0, 1}


def test_predict_df_adds_columns(tmp_path):
    n = 100
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob" in names
    assert "y_pred" in names


def test_predict_df_proba_only(tmp_path):
    n = 100
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    result = model.predict_proba(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "y_prob" in names
    assert "y_pred" not in names


def test_predict_threshold(tmp_path):
    n = 100
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    pred_default = model.predict(X, threshold=0.5)
    pred_low = model.predict(X, threshold=0.1)
    # Lower threshold → more positives (or equal)
    assert pred_low.sum() >= pred_default.sum()


def test_predict_invalid_threshold(tmp_path):
    n = 50
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    with pytest.raises(ValueError, match="threshold"):
        model.predict(X, threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        model.predict(X, threshold=-0.5)


# ---------- validation ----------

def test_fit_logistic_target_in_X(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 0.0], "y": [1, 0]})
    with pytest.raises(ValueError, match="must not be in X_cols"):
        fit_logistic(df, ["x0", "y"], "y")


def test_fit_logistic_empty_X(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0], "y": [1]})
    with pytest.raises(ValueError, match="X_cols"):
        fit_logistic(df, [], "y")


def test_fit_logistic_negative_alpha(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 0.0], "y": [1, 0]})
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        fit_logistic(df, ["x0"], "y", alpha=-1.0)


def test_fit_logistic_missing_column(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 0.0], "y": [1, 0]})
    with pytest.raises(KeyError, match="nope"):
        fit_logistic(df, ["nope"], "y")


def test_fit_logistic_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {"x0": ["a", "b"], "y": [1, 0]})
    with pytest.raises(TypeError, match="numeric"):
        fit_logistic(df, ["x0"], "y")


def test_fit_logistic_non_binary_y_raises(tmp_path):
    df = make_df(tmp_path, {
        "x0": [1.0, 2.0, 3.0, 4.0],
        "y": [2, 3, 2, 3],   # not 0/1
    })
    with pytest.raises(ValueError, match="binary"):
        fit_logistic(df, ["x0"], "y")


def test_fit_logistic_single_class_raises(tmp_path):
    df = make_df(tmp_path, {
        "x0": [1.0, 2.0, 3.0],
        "y": [1, 1, 1],
    })
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic(df, ["x0"], "y")


def test_fit_logistic_all_null_rows_raises(tmp_path):
    df = make_df(tmp_path, {
        "x0": pa.array([None, None], type=pa.float64()),
        "y": pa.array([None, None], type=pa.float64()),
    })
    # Will fail _validate_binary_y first (no values)
    with pytest.raises(ValueError):
        fit_logistic(df, ["x0"], "y")


# ---------- multi / convenience ----------

def test_fit_logistic_dataframe_method(tmp_path):
    n = 200
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    df = make_df(tmp_path, data)
    model = df.fit_logistic("x0", "y")
    assert hasattr(model, "coef_")
    assert hasattr(model, "predict_proba")


def test_logistic_model_to_dict(tmp_path):
    n = 200
    data, X, y = _synth_logistic(n, [1.0, -0.5], intercept=0.5)
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0", "x1"], "y")
    d = model.to_dict()
    assert "coefficients" in d
    assert "x0" in d["coefficients"]
    assert "n_iter" in d
    assert "converged" in d
    assert "log_likelihood" in d


def test_fit_logistic_filtered(tmp_path):
    n = 500
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0, noise=0.3)
    df = make_df(tmp_path, data)
    filtered = df[df["x0"] > -2.0]
    model = fit_logistic(filtered, ["x0"], "y", alpha=1.0)
    assert model.n_rows_ < n


def test_fit_logistic_with_nulls(tmp_path):
    n = 300
    data, X, y = _synth_logistic(n, [1.0], intercept=0.0)
    data["x0"] = [
        None if i % 10 == 0 else v for i, v in enumerate(data["x0"])
    ]
    df = make_df(tmp_path, data)
    model = fit_logistic(df, ["x0"], "y", alpha=1.0)
    # n_rows_ reflects valid count
    assert model.n_rows_ < n
    assert model.n_rows_ > 0
'''


def write_tests():
    print("\n=== tests/test_logistic.py ===")
    p = Path("tests/test_logistic.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: tests written")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v070b-logistic"

    print("=== v0.7.0b: Logistic ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v070b-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v070b-{ts_str}",
        "-m", "before patch v0.7.0b Logistic"])

    print("\n=== cadw/ml/logistic.py ===")
    Path("cadw/ml/logistic.py").write_text(LOGISTIC_SRC)
    compile_check("cadw/ml/logistic.py")
    print("  OK: logistic.py created")

    patch_dataframe()
    patch_ml_init()
    patch_top_init()
    write_tests()

    print("\n=== pytest tests/test_logistic.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_logistic.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.0b (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.0b: Logistic regression via iter-Reduce Newton IRLS"])

    print("\n==> v0.7.0b applied on", branch)


if __name__ == "__main__":
    main()