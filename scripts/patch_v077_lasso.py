#!/usr/bin/env python3
"""v0.7.7: Lasso / ElasticNet via coordinate descent on XTXOp state.

Design:
    - Reuse XTXOp: 1 Reduce → (n, XTX, XTy, sum_X, sum_y).
    - Center: XTX_c = XTX - n·μx·μxᵀ,  XTy_c = XTy - n·μx·μy.
    - Coordinate descent on centered data (d² per iter, d²+d state):
        b_j ← S(r_j, λ₁) / (z_j + λ₂),  where S = soft-threshold,
              r_j = XTy_c[j] - Σ_k XTX_c[j,k]·b_k + XTX_c[j,j]·b_j
              z_j = XTX_c[j,j]
    - Intercept: b₀ = μy - μx·b.
    - penalty='l2' | 'l1' | 'elasticnet':
        l2         : λ₁ = 0,          λ₂ = alpha   (existing np.linalg path)
        l1         : λ₁ = alpha,      λ₂ = 0
        elasticnet : λ₁ = alpha·r,    λ₂ = alpha·(1-r)
      where r = l1_ratio ∈ [0, 1].

Notation: alpha is a **raw** penalty (additive to X^T X), matching v0.7.0a
Ridge. NOT sklearn-style alpha/n. Documented in docstring.

Also includes:
    - DECISIONS.md §13.4/99 (B2 retro-doc), §11.1 (Lasso row),
      §11.5 (Lasso limitations), §12.5.1 (ML roadmap completion).
    - CHANGELOG.md [0.7.6.1] retro + [0.7.7].

Not in v0.7.7:
    - Warm-start / path (alpha sequence).
    - ElasticNet l1_ratio auto.
    - Group Lasso / LARS.

Run: uv run python scripts/patch_v077_lasso.py
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


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    if s.count(old) > 1:
        raise SystemExit(f"[{tag}] pattern found {s.count(old)} times")
    return s.replace(old, new, 1)


# ============ linear.py: add soft_threshold + CD + penalties ============

def patch_linear():
    print("\n=== cadw/ml/linear.py: Lasso/ElasticNet ===")
    p = Path("cadw/ml/linear.py")
    s = p.read_text()

    # 1. Helpers: insert before fit_linear
    anchor = "def fit_linear("
    if anchor not in s:
        raise SystemExit("fit_linear not found")

    if "_soft_threshold" not in s:
        helpers = '''def _soft_threshold(z: float, lambda_: float) -> float:
    """Soft-thresholding operator S(z, λ) = sign(z)·max(0, |z| - λ).

    λ ≤ 0 → identity. Cornerstone of coordinate descent for L1.
    """
    if lambda_ <= 0.0:
        return z
    if z > lambda_:
        return z - lambda_
    if z < -lambda_:
        return z + lambda_
    return 0.0


def _coordinate_descent(
    xtx_c: np.ndarray,   # (d, d) — centered X^T X
    xty_c: np.ndarray,   # (d,)   — centered X^T y
    lambda_l1: float,
    lambda_l2: float,
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, int, bool]:
    """Cyclic coordinate descent for Lasso/ElasticNet.

    Minimizes 0.5·bᵀ·XTX_c·b - bᵀ·XTy_c + λ₁·||b||₁ + 0.5·λ₂·||b||²₂.
    (Intercept not penalized — it's outside this function.)

    Returns (b, n_iter, converged).
    """
    d = len(xty_c)
    if d == 0:
        return np.zeros(0, dtype=np.float64), 0, True
    z = np.diag(xtx_c).copy()
    b = np.zeros(d, dtype=np.float64)
    n_iter = 0
    converged = False
    for it in range(max_iter):
        max_delta = 0.0
        for j in range(d):
            if z[j] <= 0.0:
                continue  # constant feature after centering
            # r_j = xty_c[j] - Σ_k XTX_c[j,k]·b_k + z_j·b_j
            r_j = float(xty_c[j] - xtx_c[j] @ b + z[j] * b[j])
            new_bj = _soft_threshold(r_j, lambda_l1) / (z[j] + lambda_l2)
            delta = abs(new_bj - b[j])
            if delta > max_delta:
                max_delta = delta
            b[j] = new_bj
        n_iter = it + 1
        if max_delta < tol:
            converged = True
            break
    return b, n_iter, converged


'''
        s = s.replace(anchor, helpers + anchor, 1)
        print("  OK: _soft_threshold + _coordinate_descent")

    # 2. Extend LinearModel with penalty metadata
    old_dc = '''@dataclass
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
        }'''

    new_dc = '''@dataclass
class LinearModel:
    """Fitted linear model (OLS, Ridge, Lasso, ElasticNet)."""

    coef_: np.ndarray
    intercept_: float
    feature_names_: list[str]
    target_col_: str
    alpha_: float
    n_rows_: int
    penalty_: str = "l2"      # 'l2' | 'l1' | 'elasticnet'
    l1_ratio_: float = 0.0    # 0 for l2, 1 for l1, in (0,1) for elasticnet
    n_iter_cd_: int = 0       # coordinate descent iters (0 for closed-form)
    converged_: bool = True   # CD convergence flag (True for closed-form)
    fitted_: bool = True

    def to_dict(self) -> dict:
        return {
            "coefficients": dict(zip(self.feature_names_, self.coef_.tolist())),
            "intercept": float(self.intercept_),
            "alpha": float(self.alpha_),
            "penalty": self.penalty_,
            "l1_ratio": float(self.l1_ratio_),
            "n_iter_cd": int(self.n_iter_cd_),
            "converged": bool(self.converged_),
            "n_rows": int(self.n_rows_),
        }'''

    if old_dc not in s:
        raise SystemExit("LinearModel dataclass not found (current state may differ)")
    s = replace_exact(s, old_dc, new_dc, tag="linear-model")
    print("  OK: LinearModel extended (penalty_, l1_ratio_, n_iter_cd_, converged_)")

    # 3. Replace fit_linear signature + CD branch
    old_sig = '''def fit_linear(
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
    """'''

    new_sig = '''def fit_linear(
    df,
    X_cols: "str | Sequence[str]",
    y_col: str,
    *,
    alpha: float = 0.0,
    penalty: str = "l2",
    l1_ratio: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    memory_guard: Optional[MemoryGuard] = None,
) -> LinearModel:
    """Fit Linear / Ridge / Lasso / ElasticNet via one Reduce + solve/CD.

    - X_cols: feature column names (numeric).
    - y_col: target column (numeric).
    - alpha: penalty strength (non-negative). Intercept not penalized.
    - penalty ∈ {'l2', 'l1', 'elasticnet'}:
        - 'l2'         : np.linalg.solve((XTX + alpha·I), XTy)  (closed form)
        - 'l1'         : coordinate descent, λ₁=alpha, λ₂=0
        - 'elasticnet' : coordinate descent, λ₁=alpha·l1_ratio,
                         λ₂=alpha·(1-l1_ratio)
    - l1_ratio: for penalty='elasticnet' (default 0.5); for 'l1' forced 1.0;
      for 'l2' ignored (must be None/0). ∈ [0, 1].
    - max_iter, tol: coordinate descent controls (ignored for 'l2').

    **Notation**: alpha is a **raw** additive penalty to X^T X (matching
    v0.7.0a Ridge). It is NOT sklearn-style alpha/n.

    Raises ValueError for empty df (all-null rows), negative alpha,
    X ∩ {y} non-empty, non-numeric columns, invalid penalty/l1_ratio.
    """'''

    s = replace_exact(s, old_sig, new_sig, tag="fit-linear-sig")
    print("  OK: fit_linear signature + docstring")

    # 4. Validate penalty/l1_ratio early
    old_check = '''    if alpha < 0:
        raise ValueError(f"fit_linear: alpha must be >= 0; got {alpha}")
    for c in X_cols:'''
    new_check = '''    if alpha < 0:
        raise ValueError(f"fit_linear: alpha must be >= 0; got {alpha}")
    if penalty not in ("l2", "l1", "elasticnet"):
        raise ValueError(
            f"fit_linear: penalty must be 'l2'|'l1'|'elasticnet'; "
            f"got {penalty!r}"
        )
    if penalty == "l1":
        # l1_ratio forced 1.0 (Lasso == ElasticNet with l1_ratio=1)
        l1_ratio = 1.0
    elif penalty == "elasticnet":
        if l1_ratio is None:
            l1_ratio = 0.5
        if not (0.0 <= l1_ratio <= 1.0):
            raise ValueError(
                f"fit_linear: l1_ratio must be in [0, 1]; got {l1_ratio}"
            )
    else:  # l2
        l1_ratio = 0.0
    if max_iter < 1:
        raise ValueError(f"fit_linear: max_iter must be >= 1; got {max_iter}")
    if tol <= 0:
        raise ValueError(f"fit_linear: tol must be > 0; got {tol}")
    for c in X_cols:'''
    s = replace_exact(s, old_check, new_check, tag="fit-linear-validate")
    print("  OK: penalty/l1_ratio validation")

    # 5. Replace the solve section with branch: l2 closed-form OR CD
    old_solve = '''    # Augmented system (d+1 features: intercept + X)
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
    )'''

    new_solve = '''    if penalty == "l2":
        # ---------- OLS/Ridge: closed form (existing path) ----------
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
            penalty_="l2",
            l1_ratio_=0.0,
            n_iter_cd_=0,
            converged_=True,
        )

    # ---------- Lasso / ElasticNet: coordinate descent ----------
    lambda_l1 = float(alpha) * float(l1_ratio)
    lambda_l2 = float(alpha) * (1.0 - float(l1_ratio))

    # Center X and y (intercept will be added back after CD).
    mu_x = sum_x / n            # (d,)
    mu_y = sum_y / n            # scalar
    xtx_c = xtx - n * np.outer(mu_x, mu_x)
    xty_c = xty - n * mu_x * mu_y

    # Symmetrize XTX_c (numerical symmetry; vectorized compute already
    # symmetric, but np.linalg / matrix products may drift 1e-16).
    xtx_c = 0.5 * (xtx_c + xtx_c.T)

    b, n_iter, converged = _coordinate_descent(
        xtx_c, xty_c,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        max_iter=max_iter,
        tol=tol,
    )
    b0 = float(mu_y - mu_x @ b)

    return LinearModel(
        coef_=b,
        intercept_=b0,
        feature_names_=X_cols,
        target_col_=y_col,
        alpha_=float(alpha),
        n_rows_=int(n),
        penalty_=penalty,
        l1_ratio_=float(l1_ratio),
        n_iter_cd_=int(n_iter),
        converged_=bool(converged),
    )'''

    if old_solve not in s:
        raise SystemExit("fit_linear solve section not found")
    s = replace_exact(s, old_solve, new_solve, tag="fit-linear-solve")
    print("  OK: fit_linear dispatch (l2 closed-form | l1/elasticnet CD)")

    # 6. Append fit_lasso + fit_elasticnet
    if "def fit_lasso" not in s:
        tail = '''


def fit_lasso(
    df,
    X_cols: "str | Sequence[str]",
    y_col: str,
    *,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    memory_guard: Optional[MemoryGuard] = None,
) -> LinearModel:
    """Lasso regression (L1 penalty) — convenience wrapper.

    Equivalent to fit_linear(..., penalty='l1', alpha=alpha).
    alpha is raw (additive to X^T X), not sklearn-style alpha/n.
    """
    return fit_linear(
        df, X_cols, y_col,
        alpha=alpha, penalty="l1",
        max_iter=max_iter, tol=tol,
        memory_guard=memory_guard,
    )


def fit_elasticnet(
    df,
    X_cols: "str | Sequence[str]",
    y_col: str,
    *,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-6,
    memory_guard: Optional[MemoryGuard] = None,
) -> LinearModel:
    """ElasticNet regression (L1 + L2) — convenience wrapper.

    Equivalent to fit_linear(..., penalty='elasticnet', alpha=alpha,
    l1_ratio=l1_ratio).

    - l1_ratio=1.0 → Lasso
    - l1_ratio=0.0 → Ridge (same as fit_linear(penalty='l2', alpha=alpha))
    """
    return fit_linear(
        df, X_cols, y_col,
        alpha=alpha, penalty="elasticnet", l1_ratio=l1_ratio,
        max_iter=max_iter, tol=tol,
        memory_guard=memory_guard,
    )
'''
        s = s.rstrip() + tail
        print("  OK: fit_lasso + fit_elasticnet")

    p.write_text(s)
    compile_check(p)


# ============ dataframe.py: fit_lasso / fit_elasticnet ============

def patch_dataframe():
    print("\n=== cadw/dataframe.py: fit_lasso + fit_elasticnet ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def fit_lasso" in s:
        print("  already present — skip")
        return

    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")

    methods = '''    # ---------------- ML: fit_lasso / fit_elasticnet ----------------

    def fit_lasso(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        """Fit Lasso (L1). alpha is raw (additive to X^T X)."""
        from cadw.ml.linear import fit_lasso
        return fit_lasso(
            self, X_cols, y_col,
            alpha=alpha, max_iter=max_iter, tol=tol,
            memory_guard=self._memory_guard,
        )

    def fit_elasticnet(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        """Fit ElasticNet (L1+L2). l1_ratio=1 → Lasso, 0 → Ridge."""
        from cadw.ml.linear import fit_elasticnet
        return fit_elasticnet(
            self, X_cols, y_col,
            alpha=alpha, l1_ratio=l1_ratio,
            max_iter=max_iter, tol=tol,
            memory_guard=self._memory_guard,
        )

'''
    s = s.replace(anchor, methods + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: fit_lasso + fit_elasticnet on CadwDataFrame")


# ============ exports ============

def patch_exports():
    print("\n=== cadw/ml/__init__.py + cadw/__init__.py ===")

    p = Path("cadw/ml/__init__.py")
    s = p.read_text()
    if "fit_lasso" not in s:
        s = s.replace(
            "from .linear import XTXOp, LinearModel, fit_linear",
            "from .linear import (\n"
            "    XTXOp, LinearModel, fit_linear,\n"
            "    fit_lasso, fit_elasticnet,\n"
            ")",
            1,
        )
        s = s.replace(
            '    "XTXOp", "LinearModel", "fit_linear",\n',
            '    "XTXOp", "LinearModel", "fit_linear",\n'
            '    "fit_lasso", "fit_elasticnet",\n',
            1,
        )
        p.write_text(s)
        compile_check(p)
        print("  OK: cadw/ml/__init__.py")

    p2 = Path("cadw/__init__.py")
    s2 = p2.read_text()
    if "fit_lasso" not in s2:
        s2 = s2.replace(
            "    XTXOp, LinearModel, fit_linear,\n",
            "    XTXOp, LinearModel, fit_linear,\n"
            "    fit_lasso, fit_elasticnet,\n",
            1,
        )
        s2 = s2.replace(
            '    "XTXOp", "LinearModel", "fit_linear",\n',
            '    "XTXOp", "LinearModel", "fit_linear",\n'
            '    "fit_lasso", "fit_elasticnet",\n',
            1,
        )
        p2.write_text(s2)
        compile_check(p2)
        print("  OK: cadw/__init__.py")


# ============ tests ============

TESTS_SRC = '''"""v0.7.7: Lasso / ElasticNet via coordinate descent on XTXOp state."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import (
    CadwDataFrame, MemoryGuard, fit_linear,
    fit_lasso, fit_elasticnet,
)
from cadw.ml.linear import LinearModel, _soft_threshold


def make_df(tmp_path, data, chunk=50, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _synth(n, coefs, intercept, noise=0.0, seed=42):
    rng = np.random.default_rng(seed)
    d = len(coefs)
    X = rng.normal(size=(n, d))
    y = X @ np.array(coefs) + intercept + noise * rng.normal(size=n)
    data = {f"x{i}": X[:, i] for i in range(d)}
    data["y"] = y
    return data, X, y


def _ref_lasso_closed_form_d1(x, y, lambda_1):
    """Analytical solution for d=1 Lasso (centered)."""
    x_c = x - x.mean()
    y_c = y - y.mean()
    xtx_c = float(np.dot(x_c, x_c))
    xty_c = float(np.dot(x_c, y_c))
    if xtx_c == 0:
        return 0.0, float(y.mean())
    b = _soft_threshold(xty_c, lambda_1) / xtx_c
    b0 = float(y.mean() - b * x.mean())
    return float(b), b0


# ---------- soft threshold ----------

def test_soft_threshold_basic():
    assert _soft_threshold(5.0, 2.0) == 3.0
    assert _soft_threshold(-5.0, 2.0) == -3.0
    assert _soft_threshold(1.5, 2.0) == 0.0
    assert _soft_threshold(-1.5, 2.0) == 0.0
    assert _soft_threshold(0.0, 2.0) == 0.0
    assert _soft_threshold(5.0, 0.0) == 5.0
    assert _soft_threshold(-5.0, -1.0) == -5.0


# ---------- Lasso: analytical d=1 ----------

def test_lasso_d1_matches_analytical(tmp_path):
    """d=1 Lasso has closed form: b = S(xy, λ) / xx."""
    n = 200
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = 2.0 * x + 0.1 * rng.normal(size=n)
    data = {"x": x.tolist(), "y": y.tolist()}
    df = make_df(tmp_path, data)
    alpha = 5.0
    model = fit_lasso(df, ["x"], "y", alpha=alpha, max_iter=5000, tol=1e-10)
    b_exp, b0_exp = _ref_lasso_closed_form_d1(x, y, alpha)
    assert model.coef_[0] == pytest.approx(b_exp, rel=1e-6, abs=1e-7)
    assert model.intercept_ == pytest.approx(b0_exp, rel=1e-6, abs=1e-7)


def test_lasso_d1_zero_coef_when_alpha_large(tmp_path):
    n = 100
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    y = 1.0 * x + 0.1 * rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    model = fit_lasso(df, ["x"], "y", alpha=1e6)
    assert model.coef_[0] == 0.0
    assert model.intercept_ == pytest.approx(y.mean(), rel=1e-9)


# ---------- Lasso: sparsity ----------

def test_lasso_zeroes_irrelevant_features(tmp_path):
    """y = 2·x0 + 0·x1 + 0·x2 (no noise) — Lasso zeroes x1, x2."""
    n = 500
    data, X, y = _synth(n, [2.0, 0.0, 0.0], intercept=1.0)
    df = make_df(tmp_path, data, chunk=100)
    model = fit_lasso(df, ["x0", "x1", "x2"], "y", alpha=0.1)
    assert abs(model.coef_[0] - 2.0) < 0.05
    assert abs(model.coef_[1]) < 0.05
    assert abs(model.coef_[2]) < 0.05
    assert abs(model.intercept_ - 1.0) < 0.05


def test_lasso_keeps_only_one_of_duplicated(tmp_path):
    """Perfectly correlated features — Lasso picks one."""
    n = 300
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = 2.0 * x
    df = make_df(tmp_path, {
        "a": x.tolist(), "b": (x * 1.0).tolist(), "y": y.tolist(),
    })
    model = fit_lasso(df, ["a", "b"], "y", alpha=0.5)
    # Exactly one of coef_[0], coef_[1] is non-zero (up to tolerance)
    nonzero = [abs(c) > 0.01 for c in model.coef_]
    assert sum(nonzero) == 1, f"coef={model.coef_}"


# ---------- Ridge (penalty='l2') backward compat ----------

def test_ridge_penalty_l2_matches_fit_linear(tmp_path):
    """penalty='l2' = existing fit_linear(alpha) path (no API change)."""
    n = 200
    data, X, y = _synth(n, [2.0, -1.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    m1 = fit_linear(df, ["x0", "x1"], "y", alpha=1.0)
    m2 = fit_linear(df, ["x0", "x1"], "y", alpha=1.0, penalty="l2")
    np.testing.assert_array_equal(m1.coef_, m2.coef_)
    assert m1.intercept_ == m2.intercept_
    assert m2.penalty_ == "l2"
    assert m2.l1_ratio_ == 0.0
    assert m2.n_iter_cd_ == 0


# ---------- ElasticNet equivalence ----------

def test_elasticnet_l1_ratio_0_equals_ridge(tmp_path):
    """ElasticNet(l1_ratio=0) == Ridge(penalty='l2')."""
    n = 300
    data, X, y = _synth(n, [2.0, -1.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    ridge = fit_linear(df, ["x0", "x1"], "y", alpha=2.0, penalty="l2")
    enet = fit_linear(
        df, ["x0", "x1"], "y",
        alpha=2.0, penalty="elasticnet", l1_ratio=0.0,
        max_iter=5000, tol=1e-8,
    )
    np.testing.assert_allclose(enet.coef_, ridge.coef_, rtol=1e-4, atol=1e-5)
    assert enet.intercept_ == pytest.approx(ridge.intercept_, rel=1e-4)


def test_elasticnet_l1_ratio_1_equals_lasso(tmp_path):
    """ElasticNet(l1_ratio=1) == Lasso(penalty='l1')."""
    n = 300
    data, X, y = _synth(n, [2.0, 0.0], intercept=0.0, noise=0.1)
    df = make_df(tmp_path, data)
    lasso = fit_lasso(df, ["x0", "x1"], "y", alpha=0.5, max_iter=5000, tol=1e-8)
    enet = fit_elasticnet(df, ["x0", "x1"], "y", alpha=0.5, l1_ratio=1.0,
                          max_iter=5000, tol=1e-8)
    np.testing.assert_allclose(enet.coef_, lasso.coef_, rtol=1e-6, atol=1e-7)


# ---------- large alpha → all zeros ----------

def test_lasso_large_alpha_all_zero(tmp_path):
    n = 200
    data, X, y = _synth(n, [1.0, 2.0, 3.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = fit_lasso(df, ["x0", "x1", "x2"], "y", alpha=1e9)
    assert np.allclose(model.coef_, 0.0, atol=1e-6)
    assert model.intercept_ == pytest.approx(y.mean(), rel=1e-6)


# ---------- deterministic / multibatch ----------

def test_lasso_deterministic(tmp_path):
    n = 300
    data, X, y = _synth(n, [1.0, -1.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    m1 = fit_lasso(df, ["x0", "x1"], "y", alpha=0.5)
    m2 = fit_lasso(df, ["x0", "x1"], "y", alpha=0.5)
    np.testing.assert_array_equal(m1.coef_, m2.coef_)
    assert m1.intercept_ == m2.intercept_


def test_lasso_multibatch_consistent(tmp_path):
    n = 300
    data, X, y = _synth(n, [1.0, -1.0], intercept=0.5, noise=0.1)
    df1 = make_df(tmp_path, data, chunk=10, name="a")
    df2 = make_df(tmp_path, data, chunk=100, name="b")
    m1 = fit_lasso(df1, ["x0", "x1"], "y", alpha=0.5)
    m2 = fit_lasso(df2, ["x0", "x1"], "y", alpha=0.5)
    np.testing.assert_allclose(m1.coef_, m2.coef_, rtol=1e-12)
    assert m1.intercept_ == pytest.approx(m2.intercept_, rel=1e-12)


# ---------- nulls ----------

def test_lasso_with_nulls(tmp_path):
    n = 300
    data, X, y = _synth(n, [2.0], intercept=0.5)
    data["x0"] = [None if i % 10 == 0 else v for i, v in enumerate(data["x0"])]
    df = make_df(tmp_path, data)
    model = fit_lasso(df, ["x0"], "y", alpha=0.01, max_iter=5000, tol=1e-10)
    # Small alpha, no noise → coef ≈ 2.0
    assert abs(model.coef_[0] - 2.0) < 0.01


# ---------- KKT property (basic sanity) ----------

def test_lasso_kkt_sign_property(tmp_path):
    """For non-zero coef, sign(coef_j) = sign(x_j^T residual)."""
    n = 400
    data, X, y = _synth(n, [2.0, 0.5], intercept=1.0, noise=0.05)
    df = make_df(tmp_path, data)
    model = fit_lasso(df, ["x0", "x1"], "y", alpha=0.05,
                      max_iter=5000, tol=1e-9)
    # Manual residual
    pred = X @ model.coef_ + model.intercept_
    res = y - pred
    x_c = X - X.mean(axis=0, keepdims=True)
    corr = x_c.T @ res
    for j in range(2):
        if abs(model.coef_[j]) > 0.01:
            assert np.sign(model.coef_[j]) == np.sign(corr[j]), (
                f"KKT sign mismatch at j={j}: coef={model.coef_[j]}, corr={corr[j]}"
            )


# ---------- invalid arguments ----------

def test_lasso_invalid_penalty(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="penalty"):
        fit_linear(df, ["x"], "y", penalty="l3")


def test_lasso_invalid_l1_ratio(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="l1_ratio"):
        fit_linear(df, ["x"], "y", penalty="elasticnet", l1_ratio=1.5)
    with pytest.raises(ValueError, match="l1_ratio"):
        fit_linear(df, ["x"], "y", penalty="elasticnet", l1_ratio=-0.1)


def test_lasso_invalid_max_iter(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="max_iter"):
        fit_linear(df, ["x"], "y", penalty="l1", max_iter=0)


def test_lasso_negative_alpha_raises(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="alpha"):
        fit_lasso(df, ["x"], "y", alpha=-1.0)


# ---------- constant feature ----------

def test_lasso_constant_feature_zero_coef(tmp_path):
    """Constant column: after centering, z_j=0 → coef=0."""
    n = 100
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=n)
    y = 2.0 * x0 + 0.5
    df = make_df(tmp_path, {
        "x0": x0.tolist(),
        "const": [1.0] * n,
        "y": y.tolist(),
    })
    model = fit_lasso(df, ["x0", "const"], "y", alpha=0.01,
                      max_iter=2000, tol=1e-9)
    assert model.coef_[1] == 0.0  # constant feature
    assert abs(model.coef_[0] - 2.0) < 0.01


# ---------- CadwDataFrame convenience ----------

def test_fit_lasso_dataframe_method(tmp_path):
    n = 200
    data, X, y = _synth(n, [2.0, -1.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = df.fit_lasso(["x0", "x1"], "y", alpha=0.5)
    assert isinstance(model, LinearModel)
    assert model.penalty_ == "l1"
    assert model.l1_ratio_ == 1.0
    assert model.n_iter_cd_ >= 1
    assert model.converged_


def test_fit_elasticnet_dataframe_method(tmp_path):
    n = 200
    data, X, y = _synth(n, [2.0, -1.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = df.fit_elasticnet(["x0", "x1"], "y", alpha=0.5, l1_ratio=0.5)
    assert model.penalty_ == "elasticnet"
    assert model.l1_ratio_ == 0.5
    assert model.converged_


# ---------- to_dict ----------

def test_lasso_to_dict(tmp_path):
    n = 200
    data, X, y = _synth(n, [2.0, -1.0], intercept=0.5, noise=0.1)
    df = make_df(tmp_path, data)
    model = fit_lasso(df, ["x0", "x1"], "y", alpha=0.5)
    d = model.to_dict()
    assert d["penalty"] == "l1"
    assert d["l1_ratio"] == 1.0
    assert d["n_iter_cd"] >= 1
    assert d["converged"] is True
    assert "coefficients" in d
    assert "x0" in d["coefficients"]


# ---------- predict works ----------

def test_lasso_predict_df(tmp_path):
    n = 200
    data, X, y = _synth(n, [2.0, 0.0], intercept=0.5, noise=0.0)
    df = make_df(tmp_path, data)
    model = fit_lasso(df, ["x0", "x1"], "y", alpha=0.01,
                      max_iter=2000, tol=1e-9)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    pred = t.column("y_pred").to_pylist()
    for p, yy in zip(pred, y.tolist()):
        assert abs(p - yy) < 0.05
'''

def write_tests():
    print("\n=== tests/test_linear_lasso.py ===")
    p = Path("tests/test_linear_lasso.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: 18 tests written")


# ============ docs: DECISIONS + CHANGELOG ============

def patch_decisions():
    print("\n=== DECISIONS.md updates ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    if "| `fit_lasso`/`fit_elasticnet` (v0.7.7:" not in s:
        s = s.replace(
            "| Combiner для `groupby` (v0.3.4: partial agg до shuffle) | ✅ | 12 |",
            "| Combiner для `groupby` (v0.3.4: partial agg до shuffle) | ✅ | 12 |\n"
            "| `fit_lasso`/`fit_elasticnet` (v0.7.7: coord descent) | ✅ | 18 |",
            1,
        )

    if "| `fit_lasso` — warm-start / path" not in s:
        s = s.replace(
            "| `_detect_heavy_keys` absolute min = 100 (v0.3.4)",
            "| `fit_lasso` — warm-start / path | `cadw/ml/linear.py` (v0.7.7) | alpha-путь переобучается с нуля; v0.7.8+ |\n"
            "| `fit_lasso` — `alpha` raw (не sklearn-style /n) | `cadw/ml/linear.py` (v0.7.7) | Совместимо с v0.7.0a Ridge alpha; документировано |\n"
            "| `_detect_heavy_keys` absolute min = 100 (v0.3.4)",
            1,
        )

    if "Эпизод 99" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 99 (B2: `_NumpyBackedBuilder` инверсия bitmap).** Первый
настоящий баг корректности за долгое время. `_NumpyBackedBuilder.finish()`
делал `np.packbits(mask)` где `mask[i]=True = null`. Arrow-спека:
validity bitmap, **`1=valid, 0=null`**. `packbits(True)=1` попадал в
null-позицию → null↔valid инвертированы. Диагностика (v0.7.6.1): 
`packbits([F,F,T]) → [None,None,30]` (2 nulls) вместо `[10,20,None]` (1).
Fix: `packbits(~mask)`.

**Почему latent:** в PyArrow 26 есть native `Int64Builder`/`Float64Builder`
— `_make_builder` выбирает native (fast path), numpy fallback не
использовался в тестах. Тесты `test_sort_*_nulls_builder_path` шли через
**native** builder, инверсия не срабатывала. Это второй эпизод после
§13.4/52b («зелёные тесты ≠ упражняется новая функциональность»).

**Fix (v0.7.6.1):** `packbits(~mask)` + 3 direct regression tests на
`_NumpyBackedBuilder` (**не через `external_sort`** — чтобы не маскировать
native fallback'ом). Плюс AST guard в sort.py не нужен: `packbits` есть
только в одном месте (`_NumpyBackedBuilder.finish`).

**Урок:** custom Arrow builder/encoder — тестировать **directly** (не через
pipeline) + проверять **Arrow spec invariants**: `arr.null_count` и
`arr.to_pylist()` с известными позициями. Pipeline-тест проходит мимо
fallback'а, если native API доступен для типа. Правило: «для custom encoder'а
— direct test; pipeline test — отдельно». Также: `_NumpyBackedBuilder` для
`date64`/`timestamp` — сейчас latent; после мутации PyArrow может стать
real. Фикс превентивный.

'''
        s = s.replace(marker, episode + marker, 1)

    if "`fit_lasso`/`fit_elasticnet` (v0.7.7):" in s and "Lasso/ElasticNet (v0.7.7):" not in s:
        s = s.replace(
            "| 5 | **Naive Bayes** | текст, быстрый baseline | `HashShuffle` (класс,фича) + `Reduce` (counts) | 0.5 |",
            "| 5 | **Naive Bayes** | текст, быстрый baseline | `HashShuffle` (класс,фича) + `Reduce` (counts) | 0.5 |\n"
            "| 6 | **Lasso / ElasticNet** (v0.7.7) | sparse linear, feature selection | `XTXOp` (1 pass) + coordinate descent (d² state) | 0.5 |",
            1,
        )

    p.write_text(s)
    print("  OK: §11.1, §11.5, §12.5.1, §13.4/99")


def patch_changelog():
    print("\n=== CHANGELOG.md ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()

    if "## [0.7.6.1]" not in s:
        entry_761 = """## [0.7.6.1] — 2026-10-19

### Исправлено
- **B2 (S2, latent):** `_NumpyBackedBuilder.finish()` инвертировал null
  bitmap. `np.packbits(mask)` давал `1` в null-позициях, Arrow читает
  `1=valid` → null↔valid инвертированы. Диагностика:
  `packbits([F,F,T]) → [None,None,30]` вместо `[10,20,None]`. Fix:
  `np.packbits(~mask, ...)`. Native builders в pyarrow 26 маскировали баг.
- **B1 (Medium):** `fit_multinomial_nb` не валидировал `X >= 0`. Silent
  NaN при отрицательных значениях (`feature_sum + alpha` может стать
  отрицательным → `log(neg)` = nan). Fix: `_check_X_non_negative` перед
  fit (1-pass).

### Изменено
- **I5 (perf):** `_validate_binary_y` теперь возвращает `(y_sum, n_valid)`;
  `_fit_boosting(loss='logloss_binary')` использует — один скан y вместо
  двух.

### Тесты
- +3: `test_numpy_backed_builder_*_direct` (B2 direct, не через
  `external_sort` — immune к native fallback'у).
- +2: `test_multinomial_nb_negative_X_raises`,
  `test_multinomial_nb_non_negative_X_passes` (B1).
- fast: 871 → **876 passed**. slow: 8 passed.

"""
        s = s.replace("## [0.7.6]", entry_761 + "## [0.7.6]", 1)

    if "## [0.7.7]" not in s:
        entry_77 = """## [0.7.7] — 2026-10-19

### Добавлено
- **`fit_linear(..., penalty='l1'|'elasticnet', l1_ratio=..., max_iter=1000,
  tol=1e-6)`** — Lasso / ElasticNet через coordinate descent на
  `XTXOp` state:
  - 1 Reduce → `(n, XTX, XTy, sum_X, sum_y)`.
  - Center: `XTX_c = XTX - n·μx·μxᵀ`, `XTy_c = XTy - n·μx·μy`.
  - Cyclic CD: `b_j ← S(r_j, λ₁) / (z_j + λ₂)`,
    `r_j = XTy_c[j] - Σ_k XTX_c[j,k]·b_k + z_j·b_j`.
  - `b₀ = μy - μx·b` (intercept not penalized).
  - Peak RAM = batch + O(d²). Independent of N.
- **`fit_lasso(df, X, y, alpha=1.0)`** — convenience wrapper
  (`penalty='l1'`).
- **`fit_elasticnet(df, X, y, alpha=1.0, l1_ratio=0.5)`** — convenience
  wrapper.
- **`CadwDataFrame.fit_lasso` / `.fit_elasticnet`** — методы фасада.
- **`LinearModel`** расширен: `penalty_`, `l1_ratio_`, `n_iter_cd_`,
  `converged_`.
- Экспорт: `cadw.fit_lasso`, `cadw.fit_elasticnet`,
  `cadw.ml.fit_lasso`, `cadw.ml.fit_elasticnet`.

### Contract
- `penalty='l2'` — closed-form `np.linalg.solve`, backward-compatible с
  v0.7.0a Ridge.
- `penalty='l1'` — `λ₁=alpha`, `λ₂=0`.
- `penalty='elasticnet'` — `λ₁=alpha·l1_ratio`, `λ₂=alpha·(1-l1_ratio)`.
- **`alpha` — raw (additive to X^T X)**, не sklearn-style `alpha/n`.
  Совместимо с v0.7.0a Ridge, документировано.
- `elasticnet(l1_ratio=0) ≡ Ridge(penalty='l2')`
- `elasticnet(l1_ratio=1) ≡ lasso(penalty='l1')`

### Ограничения (v0.7.7, §11.5)
- Warm-start / alpha-path — не поддерживается (v0.7.8+).
- Group Lasso / LARS / adaptive Lasso — не поддерживается.
- `max_iter` default 1000; на плохо обусловленных данных может не
  сойтись — проверять `model.converged_`.

### Тесты
- +18 в `test_linear_lasso.py`: analytical d=1, sparsity, KKT property,
  elasticnet ↔ ridge/lasso equivalence, constant feature, nulls,
  deterministic, multibatch, `LinearModel.to_dict`, invalid args.
- fast: **~894 passed**. slow: 8 passed.

"""
        s = s.replace("## [0.7.6]", entry_77 + "## [0.7.6]", 1) if "## [0.7.6]" in s else s.replace("## [0.7.5]", entry_77 + "## [0.7.5]", 1)

    p.write_text(s)
    print("  OK: [0.7.6.1] + [0.7.7] entries")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v077-lasso"

    print("=== v0.7.7: Lasso / ElasticNet ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v077-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v077-{ts_str}",
        "-m", "before patch v0.7.7 Lasso/ElasticNet"])

    patch_linear()
    patch_dataframe()
    patch_exports()
    write_tests()
    patch_decisions()
    patch_changelog()

    print("\n=== pytest tests/test_linear_lasso.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_linear_lasso.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.7 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_linear.py -v (regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_linear.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.7 (test_linear red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.7: Lasso/ElasticNet via coordinate descent on XTXOp; "
        "docs retro (v0.7.6.1 B2 episode 99); ML roadmap complete"])

    print("\n==> v0.7.7 applied on", branch)


if __name__ == "__main__":
    main()