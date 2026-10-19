#!/usr/bin/env python3
"""v0.7.1: k-means via iter-Reduce.

Design:
    StepOp: one full Reduce pass = assign + accumulate (k×d sums, k counts,
    inertia).
    Init: k-means++ (weighted by D², seed fixed).
    Stop: ||Δcentroid||_inf < tol or max_iter.

    Peak RAM: batch + O(k×d). Independent of N.

API:
    from cadw.ml.kmeans import KMeansStepOp, KMeansModel, fit_kmeans
    model = fit_kmeans(df, ["x1", "x2"], k=3)
    model.cluster_centers_, model.inertia_, model.n_iter_, model.converged_
    model.predict(df)  # adds column "cluster"

Not in v0.7.1:
    - predict_proba (soft assignment) — v0.7.2
    - categorical features — user encodes first
    - mini-batch k-means — full-batch only

Run: uv run python scripts/patch_v071a_kmeans.py
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


# ============ cadw/ml/kmeans.py ============

KMEANS_SRC = '''"""k-means clustering (v0.7.1).

Composition: iter-Reduce (assign + accumulate) + update.
Peak RAM = batch + O(k×d). Independent of N.

Algorithm:
    1. k-means++ init from first sample batch.
    2. Repeat until shift < tol or max_iter:
        a. Reduce: assign each row to nearest centroid; accumulate
           sums (k×d), counts (k), inertia = Σ ||x - c||².
        b. Update: centroids = sums / counts (empty → keep old).

    shift = max ||centroid_new - centroid_old||_inf.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import ReduceOp, reduce


class KMeansStepOp(ReduceOp):
    """One Reduce pass: assign rows to nearest centroid + accumulate.

    State: sums (k, d), counts (k), inertia (float). O(k×d).
    Centroid-array passed from previous iteration.
    Nulls in any feature → row skipped.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        centroids: np.ndarray,
    ) -> None:
        if not feature_cols:
            raise ValueError("KMeansStepOp: feature_cols must be non-empty")
        self.feature_cols = list(feature_cols)
        self.centroids = np.asarray(centroids, dtype=np.float64)
        if self.centroids.ndim != 2:
            raise ValueError(
                f"KMeansStepOp: centroids must be 2D; "
                f"got shape {self.centroids.shape}"
            )
        self.k, self.d = self.centroids.shape

    def init(self):
        return (
            np.zeros((self.k, self.d), dtype=np.float64),
            np.zeros(self.k, dtype=np.int64),
            0.0,
        )

    def update(self, state, batch):
        sums, counts, inertia = state
        X = _stack_numeric(batch, self.feature_cols)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        if X.shape[0] == 0:
            return state
        # Squared distances (n, k)
        # (n, k) = sum over d of (x - c)²
        diff = X[:, None, :] - self.centroids[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        labels = d2.argmin(axis=1)
        inertia += float(d2[np.arange(X.shape[0]), labels].sum())
        # Accumulate per-cluster sums / counts
        for j in range(self.k):
            m = labels == j
            if m.any():
                sums[j] += X[m].sum(axis=0)
                counts[j] += int(m.sum())
        return (sums, counts, inertia)

    def merge(self, a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def state_nbytes(self, state):
        sums, counts, _ = state
        return int(sums.nbytes + counts.nbytes + 8)

    def estimate_state_nbytes(self):
        k, d = self.k, self.d
        return 8 * k * d + 8 * k + 8

    def finalize(self, state):
        sums, counts, inertia = state
        return {
            "sums": sums, "counts": counts, "inertia": float(inertia),
        }


def _stack_numeric(
    batch: pa.RecordBatch, cols: Sequence[str],
) -> np.ndarray:
    parts = [_column_to_f64(batch.column(c)) for c in cols]
    return np.column_stack(parts)


def _column_to_f64(arr: pa.Array) -> np.ndarray:
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


def _kmeans_plus_plus_init(
    df, X_cols: Sequence[str], k: int, seed: int,
) -> np.ndarray:
    """k-means++ init from first sample batch. Deterministic with seed."""
    rng = np.random.default_rng(seed)
    # Grab first batch (small enough; if df huge — still 1 batch = bounded)
    sample: np.ndarray | None = None
    for b in df._iter_filtered_batches():
        X = _stack_numeric(b, X_cols)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        if X.shape[0] > 0:
            sample = X
            break
    if sample is None or sample.shape[0] == 0:
        raise ValueError("fit_kmeans: no valid (non-null) rows in df")

    n, d = sample.shape
    if n < k:
        raise ValueError(
            f"fit_kmeans: k={k} > available rows={n}"
        )

    # First centroid: random
    centers = np.empty((k, d), dtype=np.float64)
    first = rng.integers(0, n)
    centers[0] = sample[first]

    # Subsequent: weighted by D²
    closest_sq = ((sample - centers[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        probs = closest_sq / closest_sq.sum() if closest_sq.sum() > 0 else None
        idx = rng.choice(n, p=probs) if probs is not None else rng.integers(0, n)
        centers[j] = sample[idx]
        new_sq = ((sample - centers[j]) ** 2).sum(axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centers


@dataclass
class KMeansModel:
    """Fitted k-means model."""

    cluster_centers_: np.ndarray  # (k, d)
    feature_names_: list[str]
    inertia_: float
    n_iter_: int
    converged_: bool
    n_rows_: int
    fitted_: bool = True

    @property
    def k(self) -> int:
        return self.cluster_centers_.shape[0]

    def to_dict(self) -> dict:
        return {
            "cluster_centers": self.cluster_centers_.tolist(),
            "feature_names": self.feature_names_,
            "inertia": float(self.inertia_),
            "n_iter": int(self.n_iter_),
            "converged": bool(self.converged_),
            "n_rows": int(self.n_rows_),
        }

    def _labels_for_array(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.cluster_centers_[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        return d2.argmin(axis=1)

    def predict(self, X):
        """Predict cluster labels.

        - np.ndarray (n, d) → np.ndarray (n,) int64
        - CadwDataFrame → CadwDataFrame with `cluster` column (int64)
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != len(self.feature_names_):
                raise ValueError(
                    f"predict: X has {X.shape[1]} features, "
                    f"expected {len(self.feature_names_)}"
                )
            return self._labels_for_array(X).astype(np.int64)

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

        out_col = "cluster"
        orig_schema = X._dataset.schema
        out_fields = [orig_schema.field(c) for c in X._columns]
        out_fields.append(pa.field(out_col, pa.int64(), nullable=True))
        out_schema = pa.schema(out_fields)

        centers = self.cluster_centers_

        def _iter_pred():
            for b in X._iter_filtered_batches():
                Xm = _stack_numeric(b, self.feature_names_)
                n = Xm.shape[0]
                labels = np.full(n, -1, dtype=np.int64)  # -1 = null
                mask = ~np.isnan(Xm).any(axis=1)
                if mask.any():
                    Xv = Xm[mask]
                    diff = Xv[:, None, :] - centers[None, :, :]
                    d2 = (diff * diff).sum(axis=2)
                    labels[mask] = d2.argmin(axis=1)
                arrays = [b.column(c) for c in X._columns]
                # -1 → null
                arr = pa.array(
                    [None if v < 0 else int(v) for v in labels],
                    type=pa.int64(),
                )
                arrays.append(arr)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_pred(), X._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=X._memory_guard,
            spill_dir=X._spill_dir,
        )


def fit_kmeans(
    df,
    X_cols,
    k: int,
    *,
    n_init: int = 3,
    max_iter: int = 300,
    tol: float = 1e-4,
    seed: int = 42,
    memory_guard: Optional[MemoryGuard] = None,
) -> KMeansModel:
    """Fit k-means clustering via iter-Reduce.

    - X_cols: numeric feature columns.
    - k: number of clusters.
    - n_init: number of random inits (k-means++ seeded). Best inertia kept.
    - max_iter, tol: stopping criteria.
    - seed: for reproducibility.
    - memory_guard: default df._memory_guard.

    Peak RAM = batch + O(k×d). Independent of N.

    Raises:
        ValueError: empty X, k < 1, k > valid rows, all-null data.
        KeyError: missing column.
        TypeError: non-numeric column.
    """
    if isinstance(X_cols, str):
        X_cols = [X_cols]
    else:
        X_cols = list(X_cols)
    if not X_cols:
        raise ValueError("fit_kmeans: X_cols must be non-empty")
    if not isinstance(k, int):
        raise TypeError(f"fit_kmeans: k must be int; got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"fit_kmeans: k must be >= 1; got {k}")
    if n_init < 1:
        raise ValueError(f"fit_kmeans: n_init must be >= 1; got {n_init}")
    if max_iter < 1:
        raise ValueError(f"fit_kmeans: max_iter must be >= 1; got {max_iter}")
    if tol <= 0:
        raise ValueError(f"fit_kmeans: tol must be > 0; got {tol}")
    for c in X_cols:
        if c not in df.columns:
            raise KeyError(f"fit_kmeans: column {c!r} not in df")

    schema = df._dataset.schema
    for c in X_cols:
        t = schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"fit_kmeans: column {c!r} has type {t}; "
                f"numeric required (use encode_categorical for strings)"
            )

    mg = memory_guard or df._memory_guard
    d = len(X_cols)

    best: Optional[dict] = None
    for init_idx in range(n_init):
        init_seed = seed + init_idx
        centers = _kmeans_plus_plus_init(df, X_cols, k, init_seed)
        n_iter = 0
        converged = False
        inertia = float("inf")
        n_valid = 0

        for it in range(max_iter):
            n_iter = it + 1
            op = KMeansStepOp(X_cols, centers)
            res = reduce(df._iter_filtered_batches(), op, mg, log=False)
            n_valid = res["counts"].sum()
            if n_valid == 0:
                raise ValueError("fit_kmeans: no valid (non-null) rows")
            sums = res["sums"]
            counts = res["counts"]
            inertia = res["inertia"]

            # Update: mean per cluster; empty cluster keeps old
            new_centers = centers.copy()
            for j in range(k):
                if counts[j] > 0:
                    new_centers[j] = sums[j] / counts[j]

            shift = float(np.max(np.abs(new_centers - centers)))
            centers = new_centers

            if shift < tol:
                converged = True
                break

        if best is None or inertia < best["inertia"]:
            best = {
                "centers": centers.copy(),
                "inertia": float(inertia),
                "n_iter": int(n_iter),
                "converged": bool(converged),
                "n_rows": int(n_valid),
            }

    assert best is not None
    return KMeansModel(
        cluster_centers_=best["centers"],
        feature_names_=list(X_cols),
        inertia_=best["inertia"],
        n_iter_=best["n_iter"],
        converged_=best["converged"],
        n_rows_=best["n_rows"],
    )
'''


# ============ dataframe.py convenience ============

def patch_dataframe():
    print("\n=== cadw/dataframe.py: fit_kmeans method ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    if "def fit_kmeans" in s:
        print("  already present — skip")
        return

    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")

    methods = '''    # ---------------- ML: fit_kmeans ----------------

    def fit_kmeans(
        self,
        X_cols: "str | Sequence[str]",
        k: int,
        *,
        n_init: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        seed: int = 42,
    ):
        """Fit k-means clustering via iter-Reduce (v0.7.1).

        Each iteration = one Reduce pass (assign + accumulate). Peak
        RAM = batch + O(k×d).

        Returns KMeansModel (cluster_centers_, predict).
        """
        from cadw.ml.kmeans import fit_kmeans
        return fit_kmeans(
            self, X_cols, k,
            n_init=n_init, max_iter=max_iter, tol=tol, seed=seed,
            memory_guard=self._memory_guard,
        )

'''
    s = s.replace(anchor, methods + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: fit_kmeans method added")


# ============ exports ============

def patch_exports():
    print("\n=== cadw/ml/__init__.py + cadw/__init__.py ===")
    # ML init
    p = Path("cadw/ml/__init__.py")
    p.write_text('''"""CADW ML layer (v0.7.1)."""
from .linear import XTXOp, LinearModel, fit_linear
from .logistic import LogisticGradOp, LogisticModel, fit_logistic
from .kmeans import KMeansStepOp, KMeansModel, fit_kmeans

__all__ = [
    "XTXOp", "LinearModel", "fit_linear",
    "LogisticGradOp", "LogisticModel", "fit_logistic",
    "KMeansStepOp", "KMeansModel", "fit_kmeans",
]
''')
    compile_check(p)

    # Top-level
    p2 = Path("cadw/__init__.py")
    s = p2.read_text()
    if "KMeansModel" in s:
        print("  top-level already updated")
        return
    s = s.replace(
        "    LogisticGradOp, LogisticModel, fit_logistic,\n)",
        "    LogisticGradOp, LogisticModel, fit_logistic,\n"
        "    KMeansStepOp, KMeansModel, fit_kmeans,\n)",
        1,
    )
    s = s.replace(
        '    "LogisticGradOp", "LogisticModel", "fit_logistic",\n',
        '    "LogisticGradOp", "LogisticModel", "fit_logistic",\n'
        '    "KMeansStepOp", "KMeansModel", "fit_kmeans",\n',
        1,
    )
    p2.write_text(s)
    compile_check(p2)
    print("  OK: exports updated")


# ============ tests ============

TESTS_SRC = '''"""v0.7.1: k-means via iter-Reduce."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.ml.kmeans import KMeansModel, KMeansStepOp, fit_kmeans


def make_df(tmp_path, data, chunk=20, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _synth_kmeans(n, k_true, d=2, spread=0.5, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k_true, d)) * 5
    labels = rng.integers(0, k_true, size=n)
    X = centers[labels] + spread * rng.normal(size=(n, d))
    data = {f"x{i}": X[:, i] for i in range(d)}
    return data, X, centers


# ---------- op ----------

def test_kmeans_step_op_estimate_state():
    op = KMeansStepOp(["a", "b", "c"], np.zeros((3, 3)))
    assert op.estimate_state_nbytes() == 8*3*3 + 8*3 + 8


def test_kmeans_step_op_state_nbytes_matches():
    op = KMeansStepOp(["a", "b"], np.zeros((3, 2)))
    assert op.state_nbytes(op.init()) <= op.estimate_state_nbytes()


# ---------- basic fit ----------

def test_fit_kmeans_two_clusters(tmp_path):
    n = 500
    data, X, centers = _synth_kmeans(n, k_true=2, d=2, seed=1)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=2, n_init=1, seed=42)
    assert isinstance(model, KMeansModel)
    assert model.cluster_centers_.shape == (2, 2)
    assert model.inertia_ > 0
    assert model.converged_


def test_fit_kmeans_three_clusters(tmp_path):
    n = 500
    data, X, centers = _synth_kmeans(n, k_true=3, d=2, seed=2)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=3, n_init=3, seed=42)
    # Recovered centers should be close to true (up to permutation)
    from itertools import permutations
    best = min(
        float(np.linalg.norm(model.cluster_centers_ - np.array(p)))
        for p in permutations(centers)
    )
    assert best < 2.0, f"permutation error={best:.3f}"


def test_fit_kmeans_high_dimensional(tmp_path):
    n = 300
    data, X, centers = _synth_kmeans(n, k_true=3, d=5, seed=3)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, [f"x{i}" for i in range(5)], k=3, n_init=2)
    assert model.cluster_centers_.shape == (3, 5)
    assert model.converged_


def test_fit_kmeans_k1(tmp_path):
    n = 100
    data, X, _ = _synth_kmeans(n, k_true=1, d=2, seed=4)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=1)
    assert model.cluster_centers_.shape == (1, 2)
    # Cluster center = mean of all points
    np.testing.assert_allclose(
        model.cluster_centers_[0], X.mean(axis=0), atol=1e-6,
    )


def test_fit_kmeans_n_init_picks_best(tmp_path):
    n = 300
    data, X, _ = _synth_kmeans(n, k_true=3, d=2, seed=5)
    df = make_df(tmp_path, data)
    m1 = fit_kmeans(df, ["x0", "x1"], k=3, n_init=1, seed=42)
    m5 = fit_kmeans(df, ["x0", "x1"], k=3, n_init=5, seed=42)
    # More inits: inertia <= single init (never worse)
    assert m5.inertia_ <= m1.inertia_ + 1e-6


def test_fit_kmeans_deterministic_with_seed(tmp_path):
    n = 300
    data, X, _ = _synth_kmeans(n, k_true=3, d=2, seed=6)
    df = make_df(tmp_path, data)
    m1 = fit_kmeans(df, ["x0", "x1"], k=3, n_init=2, seed=42)
    m2 = fit_kmeans(df, ["x0", "x1"], k=3, n_init=2, seed=42)
    np.testing.assert_allclose(
        m1.cluster_centers_, m2.cluster_centers_, rtol=1e-9,
    )


# ---------- predict ----------

def test_predict_numpy(tmp_path):
    n = 200
    data, X, _ = _synth_kmeans(n, k_true=2, d=2, seed=7)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=2, seed=42)
    labels = model.predict(X[:10])
    assert labels.shape == (10,)
    assert set(labels.tolist()) <= {0, 1}


def test_predict_df_adds_cluster_column(tmp_path):
    n = 200
    data, X, _ = _synth_kmeans(n, k_true=3, d=2, seed=8)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=3, seed=42)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    names = pq.read_schema(out).names
    assert "cluster" in names


def test_predict_null_rows_marked(tmp_path):
    n = 100
    data, X, _ = _synth_kmeans(n, k_true=2, d=2, seed=9)
    data["x0"] = [None if i == 5 else v for i, v in enumerate(data["x0"])]
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=2)
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows[5]["cluster"] is None


def test_predict_consistent_with_fit(tmp_path):
    n = 300
    data, X, _ = _synth_kmeans(n, k_true=3, d=2, seed=10)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=3, seed=42)
    # Predict on same data — all labels should be non-null
    result = model.predict(df)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    assert t.column("cluster").null_count == 0


# ---------- with nulls in fit ----------

def test_fit_with_nulls_skips_rows(tmp_path):
    n = 200
    data, X, _ = _synth_kmeans(n, k_true=2, d=2, seed=11)
    data["x0"] = [None if i % 10 == 0 else v for i, v in enumerate(data["x0"])]
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=2)
    assert model.n_rows_ < n
    assert model.n_rows_ > 0


# ---------- validation ----------

def test_fit_kmeans_invalid_k(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "x1": [3.0, 4.0]})
    with pytest.raises(ValueError, match="k must be >= 1"):
        fit_kmeans(df, ["x0", "x1"], k=0)
    with pytest.raises(TypeError, match="k must be int"):
        fit_kmeans(df, ["x0", "x1"], k=1.5)


def test_fit_kmeans_missing_column(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0], "x1": [2.0]})
    with pytest.raises(KeyError, match="nope"):
        fit_kmeans(df, ["nope"], k=1)


def test_fit_kmeans_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, {"x0": ["a", "b"], "x1": [1.0, 2.0]})
    with pytest.raises(TypeError, match="numeric"):
        fit_kmeans(df, ["x0", "x1"], k=1)


def test_fit_kmeans_k_gt_n(tmp_path):
    df = make_df(tmp_path, {"x0": [1.0, 2.0], "x1": [3.0, 4.0]})
    with pytest.raises(ValueError, match="k=.*>.*available"):
        fit_kmeans(df, ["x0", "x1"], k=10)


# ---------- convenience ----------

def test_fit_kmeans_dataframe_method(tmp_path):
    n = 200
    data, X, _ = _synth_kmeans(n, k_true=2, d=2, seed=12)
    df = make_df(tmp_path, data)
    model = df.fit_kmeans(["x0", "x1"], k=2)
    assert hasattr(model, "cluster_centers_")


def test_kmeans_model_to_dict(tmp_path):
    n = 100
    data, X, _ = _synth_kmeans(n, k_true=2, d=2, seed=13)
    df = make_df(tmp_path, data)
    model = fit_kmeans(df, ["x0", "x1"], k=2)
    d = model.to_dict()
    assert "cluster_centers" in d
    assert "inertia" in d
    assert "n_iter" in d
    assert "converged" in d
'''


def write_tests():
    print("\n=== tests/test_kmeans.py ===")
    p = Path("tests/test_kmeans.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: tests written")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v071-kmeans"

    print("=== v0.7.1: k-means ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v071-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v071-{ts_str}",
        "-m", "before patch v0.7.1 k-means"])

    print("\n=== cadw/ml/kmeans.py ===")
    Path("cadw/ml/kmeans.py").write_text(KMEANS_SRC)
    compile_check("cadw/ml/kmeans.py")
    print("  OK: kmeans.py created")

    patch_dataframe()
    patch_exports()
    write_tests()

    print("\n=== pytest tests/test_kmeans.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_kmeans.py", "-v"]
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.1 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.1: k-means via iter-Reduce (k-means++ init; predict)"])

    print("\n==> v0.7.1 applied on", branch)


if __name__ == "__main__":
    main()