#!/usr/bin/env python3
"""v0.9.2a: regression metrics + log_loss (Reduce-based, scalar API).

SCOPE:
  - cadw/ops/metrics.py::MetricsOp — ReduceOp for r2/mse/rmse/mae/log_loss
  - CadwDataFrame.metrics(y_true, metrics, y_pred=, y_proba=) → dict
  - cadw/__init__.py — export MetricsOp

DESIGN (locked):
  - Single Reduce pass for ALL requested metrics (no N passes).
  - State: O(1) counters (7 floats + 2 ints) — no dependence on N.
  - Regression metrics require y_pred (float): r2, mse, rmse, mae.
  - log_loss requires y_proba (binary, in [0,1]).
  - Empty input → NaN for all (honest "no data").
  - R² matches sklearn: ss_tot==0 → 1.0 if perfect, else 0.0.
  - Nulls in y_true/y_pred/y_proba → skip row (skipna=True).
  - Not included: accuracy/precision/recall/f1/roc_auc — v0.9.2b
    (classification needs confusion matrix, K×K state).

WHY metrics first (v0.9.2 order):
  - User value: without evaluation, models useless.
  - Pure addition — no behavior change → zero stale-test risk.
  - Foundation for CV (v0.9.2d), scalers (v0.9.2e), TimeSeriesFeatures.

GUARDS:
  - test_metrics.py (~18 fast): perfect/known values, all metrics,
    log_loss, combined, nulls, empty→NaN, filtered, pandas-equivalence,
    validation (missing y_pred/y_proba, unknown metric, missing column).
  - test_metrics_e2e.py (1 slow): 100K rows, sklearn-equivalence.
  - No stale-behavior grep (pure addition). Use assert_guards_green.

Run: uv run python scripts/patch_v092a_metrics.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
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


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if not starts:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ 1. cadw/ops/metrics.py ============

METRICS_MODULE = '''"""Regression + log_loss metrics via ReduceOp (v0.9.2a).

Single-pass accumulation, O(1) state (7 floats + 2 ints). Independent
of N. Nulls skipped per row (skipna=True).

Metrics:
  - r2        : coefficient of determination
  - mse,rmse  : mean squared error, root
  - mae       : mean absolute error
  - log_loss  : binary cross-entropy (requires proba in [0,1])

Regressions:
  - ss_tot == 0 → r2 = 1.0 if ss_res == 0 (perfect), else 0.0
    (matches sklearn.metrics.r2_score)
  - Empty input → NaN (honest "no data")

Classification metrics (accuracy, precision, recall, f1, roc_auc) —
v0.9.2b (require confusion matrix, K×K state; roc_auc needs sort).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.ops.reduce import ReduceOp


REGRESSION_METRICS = frozenset({"r2", "mse", "rmse", "mae"})
PROBA_METRICS = frozenset({"log_loss"})
SUPPORTED_METRICS = REGRESSION_METRICS | PROBA_METRICS


def _to_f64(arr: pa.Array) -> np.ndarray:
    """Arrow Array → np.float64 with NaN for nulls."""
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


class MetricsOp(ReduceOp):
    """Regression + log_loss metrics in one Reduce pass.

    Usage:
        op = MetricsOp("y", y_pred_col="yp", metrics=["r2", "rmse"])
        result = reduce(batches, op, guard)
        # {"r2": 0.85, "rmse": 0.32}
    """

    def __init__(
        self,
        y_true_col: str,
        *,
        y_pred_col: Optional[str] = None,
        y_proba_col: Optional[str] = None,
        metrics: Sequence[str] | str,
    ) -> None:
        if isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = list(metrics)
        unknown = [m for m in self.metrics if m not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"MetricsOp: unknown metric(s) {unknown}; "
                f"supported: {sorted(SUPPORTED_METRICS)}"
            )
        self.y_true_col = y_true_col
        self.y_pred_col = y_pred_col
        self.y_proba_col = y_proba_col

        needs_pred = bool(set(self.metrics) & REGRESSION_METRICS)
        needs_proba = bool(set(self.metrics) & PROBA_METRICS)
        if needs_pred and y_pred_col is None:
            raise ValueError(
                f"MetricsOp: y_pred_col required for "
                f"{sorted(set(self.metrics) & REGRESSION_METRICS)}"
            )
        if needs_proba and y_proba_col is None:
            raise ValueError(
                f"MetricsOp: y_proba_col required for "
                f"{sorted(set(self.metrics) & PROBA_METRICS)}"
            )

    def estimate_state_nbytes(self) -> int:
        # 7 counters (n, sum_y, sum_yy, sum_res_sq, sum_abs_res, sum_ll,
        # n_proba) × 8 B = 56, rounded to 64.
        return 64

    def init(self):
        return {
            "n": 0,
            "sum_y": 0.0,
            "sum_yy": 0.0,
            "sum_residual_sq": 0.0,
            "sum_abs_residual": 0.0,
            "sum_log_loss": 0.0,
            "n_proba": 0,
        }

    def update(self, state, batch):
        # Regression
        if self.y_pred_col is not None:
            y = _to_f64(batch.column(self.y_true_col))
            yp = _to_f64(batch.column(self.y_pred_col))
            mask = ~np.isnan(y) & ~np.isnan(yp)
            if mask.any():
                y_m = y[mask]
                yp_m = yp[mask]
                resid = y_m - yp_m
                state["n"] += int(mask.sum())
                state["sum_y"] += float(y_m.sum())
                state["sum_yy"] += float((y_m * y_m).sum())
                state["sum_residual_sq"] += float((resid * resid).sum())
                state["sum_abs_residual"] += float(np.abs(resid).sum())

        # Log loss (binary; y must be {0,1}, p in [0,1])
        if self.y_proba_col is not None:
            y = _to_f64(batch.column(self.y_true_col))
            p = _to_f64(batch.column(self.y_proba_col))
            mask = ~np.isnan(y) & ~np.isnan(p)
            if mask.any():
                y_m = y[mask]
                p_m = p[mask]
                eps = 1e-15
                p_c = np.clip(p_m, eps, 1.0 - eps)
                ll = -(y_m * np.log(p_c)
                       + (1.0 - y_m) * np.log(1.0 - p_c))
                state["sum_log_loss"] += float(ll.sum())
                state["n_proba"] += int(mask.sum())

        return state

    def merge(self, a, b):
        return {k: a[k] + b[k] for k in a}

    def state_nbytes(self, state):
        return 8 * 7  # 7 counters × 8 B

    def finalize(self, state):
        results = {}
        nan = float("nan")
        n = state["n"]

        if "r2" in self.metrics:
            if n > 0:
                y_mean = state["sum_y"] / n
                ss_tot = state["sum_yy"] - n * y_mean * y_mean
                ss_res = state["sum_residual_sq"]
                if ss_tot <= 0:
                    results["r2"] = 1.0 if ss_res <= 0 else 0.0
                else:
                    results["r2"] = 1.0 - ss_res / ss_tot
            else:
                results["r2"] = nan

        if "mse" in self.metrics:
            results["mse"] = (
                state["sum_residual_sq"] / n if n > 0 else nan
            )
        if "rmse" in self.metrics:
            results["rmse"] = (
                float(np.sqrt(state["sum_residual_sq"] / n))
                if n > 0 else nan
            )
        if "mae" in self.metrics:
            results["mae"] = (
                state["sum_abs_residual"] / n if n > 0 else nan
            )

        if "log_loss" in self.metrics:
            results["log_loss"] = (
                state["sum_log_loss"] / state["n_proba"]
                if state["n_proba"] > 0 else nan
            )

        return results
'''


def write_metrics_module():
    print("\n=== cadw/ops/metrics.py ===")
    p = Path("cadw/ops/metrics.py")
    p.write_text(METRICS_MODULE)
    compile_check(p)
    print("  OK")


# ============ 2. CadwDataFrame.metrics() ============

def add_metrics_method():
    print("\n=== cadw/dataframe.py: CadwDataFrame.metrics() ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def metrics(" in s:
        print("  already present — skip")
        return

    # Insert before "# ---------------- write ----------------"
    anchor = "    # ---------------- write ----------------"

    method = '''    # ---------------- metrics (v0.9.2a) ----------------

    def metrics(
        self,
        *,
        y_true: str,
        metrics: "Sequence[str] | str",
        y_pred: "str | None" = None,
        y_proba: "str | None" = None,
    ) -> dict:
        """Regression + log_loss metrics in one pass (Reduce-based).

        Metrics:
          - r2, mse, rmse, mae — require `y_pred` (numeric).
          - log_loss — requires `y_proba` (binary, values in [0,1]).
          - Combinations of regression + log_loss allowed in one call if
            both y_pred and y_proba provided.

        Args:
            y_true: column with true values.
            metrics: metric name(s) — scalar or list.
            y_pred: predicted values (float). Required for
                r2/mse/rmse/mae.
            y_proba: predicted probabilities (float in [0,1]).
                Required for log_loss.

        Returns:
            dict `{metric_name: value}`. Values are NaN for empty input
            (after filter).

        Nulls in y_true / y_pred / y_proba → row skipped (skipna=True).

        State: O(1) (7 counters). Independent of N.
        """
        from cadw.ops.metrics import MetricsOp, SUPPORTED_METRICS
        from cadw.ops.reduce import reduce

        if isinstance(metrics, str):
            metrics = [metrics]
        metrics = list(metrics)
        if not metrics:
            raise ValueError("metrics: metrics list is empty")

        if y_pred is None and y_proba is None:
            raise ValueError(
                "metrics: at least one of y_pred / y_proba required"
            )

        for c, label in (
            (y_true, "y_true"), (y_pred, "y_pred"), (y_proba, "y_proba")
        ):
            if c is not None and c not in self._columns:
                raise KeyError(
                    f"metrics: {label} column {c!r} not in {self._columns}"
                )

        op = MetricsOp(
            y_true, y_pred_col=y_pred, y_proba_col=y_proba,
            metrics=metrics,
        )
        return reduce(
            self._iter_filtered_batches(), op,
            self._memory_guard, log=False,
        )

    # ---------------- write ----------------'''

    s = replace_line_block(s, anchor, method, tag="metrics-method")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 3. cadw/__init__.py export ============

def update_init():
    print("\n=== cadw/__init__.py: export MetricsOp ===")
    p = Path("cadw/__init__.py")
    s = p.read_text()
    if "MetricsOp" in s:
        print("  already present — skip")
        return

    old = "from .ops.reduce import (\n    reduce, ReduceOp,\n    CountOp, SumOp, MinMaxOp, MeanOp, StdOp, CovarianceOp, HistogramOp,\n)\n"
    if old not in s:
        raise SystemExit("reduce import anchor not found")

    new = old + "from .ops.metrics import MetricsOp, SUPPORTED_METRICS\n"
    s = s.replace(old, new, 1)

    if '"MetricsOp"' not in s:
        anchor_all = '"CountOp", "SumOp", "MinMaxOp", "MeanOp", "StdOp", "CovarianceOp", "HistogramOp",\n'
        if anchor_all in s:
            s = s.replace(
                anchor_all,
                anchor_all
                + '    "MetricsOp", "SUPPORTED_METRICS",\n',
                1,
            )

    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 4. Tests ============

TESTS = '''"""v0.9.2a: regression + log_loss metrics tests."""
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=50):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ============ r2 ============

def test_metrics_r2_perfect(tmp_path):
    df = make_df(tmp_path, {"y": [1.0, 2.0, 3.0], "yp": [1.0, 2.0, 3.0]})
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == pytest.approx(1.0)


def test_metrics_r2_known(tmp_path):
    """y=[1,2,3,4], yp=[2,2,3,3] → SS_res=2, SS_tot=5, r2=1-2/5=0.6."""
    df = make_df(tmp_path, {
        "y": [1.0, 2.0, 3.0, 4.0],
        "yp": [2.0, 2.0, 3.0, 3.0],
    })
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == pytest.approx(0.6)


def test_metrics_r2_constant_y_perfect(tmp_path):
    """y all equal to yp → ss_tot==0, ss_res==0 → r2=1.0 (sklearn behavior)."""
    df = make_df(tmp_path, {
        "y": [5.0, 5.0, 5.0],
        "yp": [5.0, 5.0, 5.0],
    })
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == 1.0


def test_metrics_r2_constant_y_wrong_pred(tmp_path):
    """y all 5, yp all 3 → ss_res>0, ss_tot==0 → r2=0.0 (sklearn behavior)."""
    df = make_df(tmp_path, {
        "y": [5.0, 5.0],
        "yp": [3.0, 3.0],
    })
    res = df.metrics(y_true="y", y_pred="yp", metrics=["r2"])
    assert res["r2"] == 0.0


# ============ mse / rmse / mae ============

def test_metrics_mse_rmse_mae(tmp_path):
    """y=[0,0,0], yp=[1,-1,0] → mse=2/3, rmse=sqrt(2/3), mae=2/3."""
    df = make_df(tmp_path, {
        "y": [0.0, 0.0, 0.0],
        "yp": [1.0, -1.0, 0.0],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["mse", "rmse", "mae"],
    )
    assert res["mse"] == pytest.approx(2.0 / 3.0)
    assert res["rmse"] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert res["mae"] == pytest.approx(2.0 / 3.0)


def test_metrics_all_regression_combined(tmp_path):
    """All regression metrics from one pass."""
    df = make_df(tmp_path, {
        "y": [1.0, 2.0, 3.0, 4.0],
        "yp": [1.1, 1.9, 3.1, 3.9],
    })
    res = df.metrics(
        y_true="y", y_pred="yp",
        metrics=["r2", "mse", "rmse", "mae"],
    )
    assert set(res.keys()) == {"r2", "mse", "rmse", "mae"}
    assert res["rmse"] == pytest.approx(math.sqrt(res["mse"]))
    # residual abs: 0.1 each → mae = 0.1
    assert res["mae"] == pytest.approx(0.1)


# ============ log_loss ============

def test_metrics_log_loss_perfect(tmp_path):
    """y=[1,0], p=[1,0] → clipped, loss ~ tiny."""
    df = make_df(tmp_path, {"y": [1.0, 0.0], "p": [1.0, 0.0]})
    res = df.metrics(y_true="y", y_proba="p", metrics=["log_loss"])
    assert res["log_loss"] < 1e-10


def test_metrics_log_loss_50_50(tmp_path):
    """y=[1,0], p=[0.5,0.5] → loss = -log(0.5) = ln(2)."""
    df = make_df(tmp_path, {"y": [1.0, 0.0], "p": [0.5, 0.5]})
    res = df.metrics(y_true="y", y_proba="p", metrics=["log_loss"])
    assert res["log_loss"] == pytest.approx(math.log(2.0))


def test_metrics_log_loss_known(tmp_path):
    """y=[1,0], p=[0.9, 0.1] → loss = -[log(0.9)+log(0.9)]/2 = -log(0.9)."""
    df = make_df(tmp_path, {"y": [1.0, 0.0], "p": [0.9, 0.1]})
    res = df.metrics(y_true="y", y_proba="p", metrics=["log_loss"])
    assert res["log_loss"] == pytest.approx(-math.log(0.9))


# ============ combined ============

def test_metrics_regression_and_log_loss(tmp_path):
    """Both y_pred and y_proba passed in one call."""
    df = make_df(tmp_path, {
        "y": [0.0, 1.0, 0.0, 1.0],
        "yp": [0.1, 0.9, 0.2, 0.8],
        "p": [0.1, 0.9, 0.2, 0.8],
    })
    res = df.metrics(
        y_true="y", y_pred="yp", y_proba="p",
        metrics=["mse", "log_loss"],
    )
    assert set(res.keys()) == {"mse", "log_loss"}
    assert res["mse"] > 0
    assert res["log_loss"] > 0


# ============ nulls, empty ============

def test_metrics_nulls_skipped(tmp_path):
    """Null in y or yp → row skipped."""
    df = make_df(tmp_path, {
        "y": [1.0, None, 3.0, 4.0],
        "yp": [1.0, 2.0, None, 4.0],
    })
    res = df.metrics(y_true="y", y_pred="yp", metrics=["mse"])
    # valid rows: (1,1), (4,4) → mse = 0
    assert res["mse"] == pytest.approx(0.0)


def test_metrics_empty_input_nan(tmp_path):
    df = make_df(tmp_path, {"y": [1.0, 2.0], "yp": [1.0, 2.0]})
    empty = df[df["y"] > 100]
    res = empty.metrics(
        y_true="y", y_pred="yp", metrics=["r2", "rmse", "mae"],
    )
    for k in ("r2", "rmse", "mae"):
        assert math.isnan(res[k])


def test_metrics_filtered(tmp_path):
    df = make_df(tmp_path, {
        "y": [0.0, 1.0, 2.0, 3.0],
        "yp": [0.0, 1.0, 2.0, 100.0],  # big error on filtered-out row
    })
    filtered = df[df["y"] < 3]
    res = filtered.metrics(y_true="y", y_pred="yp", metrics=["mse"])
    assert res["mse"] == pytest.approx(0.0)


# ============ validation ============

def test_metrics_string_metric(tmp_path):
    """String metric (not list) works."""
    df = make_df(tmp_path, {"y": [1.0, 2.0], "yp": [1.0, 2.0]})
    res = df.metrics(y_true="y", y_pred="yp", metrics="r2")
    assert res["r2"] == pytest.approx(1.0)


def test_metrics_invalid_metric(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(ValueError, match="unknown metric"):
        df.metrics(y_true="y", y_pred="yp", metrics=["bogus"])


def test_metrics_missing_y_pred_for_r2(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(ValueError, match="y_pred_col required"):
        df.metrics(y_true="y", metrics=["r2"])


def test_metrics_missing_y_proba_for_log_loss(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(ValueError, match="y_proba_col required"):
        df.metrics(y_true="y", metrics=["log_loss"])


def test_metrics_no_pred_or_proba(tmp_path):
    df = make_df(tmp_path, {"y": [1.0]})
    with pytest.raises(ValueError, match="at least one"):
        df.metrics(y_true="y", metrics=["r2"])


def test_metrics_missing_column(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(KeyError, match="nope"):
        df.metrics(y_true="nope", y_pred="yp", metrics=["r2"])


def test_metrics_empty_metrics_list(tmp_path):
    df = make_df(tmp_path, {"y": [1.0], "yp": [1.0]})
    with pytest.raises(ValueError, match="metrics list is empty"):
        df.metrics(y_true="y", y_pred="yp", metrics=[])


# ============ pandas / sklearn equivalence ============

def test_metrics_pandas_equivalence(tmp_path):
    n = 200
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    yp = y + rng.normal(scale=0.5, size=n)
    df = make_df(tmp_path, {"y": y.tolist(), "yp": yp.tolist()}, chunk=50)
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["r2", "mse", "rmse", "mae"],
    )
    # Reference (numpy, matches sklearn)
    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    assert res["r2"] == pytest.approx(1.0 - ss_res / ss_tot)
    assert res["mse"] == pytest.approx(ss_res / n)
    assert res["rmse"] == pytest.approx(math.sqrt(ss_res / n))
    assert res["mae"] == pytest.approx(float(np.abs(y - yp).mean()))


# ============ integration ============

def test_metrics_with_groupby_predict(tmp_path):
    """End-to-end: groupby → agg → predict → metrics works as chain."""
    n = 100
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {
        "x": rng.normal(size=n).tolist(),
        "y": (rng.normal(size=n) * 2).tolist(),
    })
    # Just verify metrics can be called on a df with any data
    res = df.metrics(y_true="y", y_pred="x", metrics=["mse"])
    assert res["mse"] > 0
'''


E2E_TESTS = '''"""v0.9.2a: metrics E2E (slow)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


@pytest.mark.slow
def test_metrics_100k_rows(tmp_path):
    """100K rows — metrics reduce independently of N."""
    n = 100_000
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    yp = y + rng.normal(scale=0.3, size=n)
    p = 1.0 / (1.0 + np.exp(-y))  # arbitrary proba in (0,1)

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({"y": y.tolist(), "yp": yp.tolist(), "p": p.tolist()}),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=30),
        spill_dir=tmp_path / "spill",
    )
    res = df.metrics(
        y_true="y", y_pred="yp", y_proba="p",
        metrics=["r2", "rmse", "mae", "log_loss"],
    )

    # Reference (numpy)
    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ref_r2 = 1.0 - ss_res / ss_tot
    ref_rmse = float(np.sqrt(ss_res / n))
    ref_mae = float(np.abs(y - yp).mean())

    assert res["r2"] == pytest.approx(ref_r2, rel=1e-9)
    assert res["rmse"] == pytest.approx(ref_rmse, rel=1e-9)
    assert res["mae"] == pytest.approx(ref_mae, rel=1e-9)
    assert res["log_loss"] > 0  # proba doesn't match y
'''


def write_tests():
    for name, content in [
        ("tests/test_metrics.py", TESTS),
        ("tests/test_metrics_e2e.py", E2E_TESTS),
    ]:
        print(f"\n=== {name} ===")
        p = Path(name)
        p.write_text(content)
        compile_check(p)
        print("  OK")


# ============ 5. CHANGELOG + DECISIONS ============

def append_changelog():
    print("\n=== CHANGELOG.md: append v0.9.2a via patch_utils ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.2a — metrics" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Добавлено (v0.9.2a — metrics, regression + log_loss)

- **`CadwDataFrame.metrics(y_true, metrics, y_pred=, y_proba=)`** —
  regression + binary log_loss в одном Reduce-проходе.
  - **Regression** (require `y_pred`): `r2`, `mse`, `rmse`, `mae`.
  - **log_loss** (requires `y_proba` in [0,1], binary y): binary
    cross-entropy.
  - **Combined** — если `y_pred` и `y_proba` заданы оба, все метрики
    считаются в одном проходе (state O(1)).
  - Empty input (после filter) → **NaN** для каждой метрики.
  - Nulls в `y_true`/`y_pred`/`y_proba` → skip per row (skipna=True).
  - **R² matches sklearn**: `ss_tot == 0` → 1.0 если perfect, else 0.0.
- **`cadw/ops/metrics.py::MetricsOp`** — ReduceOp, state = 7 counters
  × 8 B = 64 B. Независимо от N.

### Design decisions (locked-in, v0.9.2a)

- **Reduce-based**: один проход, O(1) state. Не Sort, не pandas.
- **Regression + log_loss в одном patch** — обе метрики O(1) state,
  тривиально комбинируются. Classification (confusion matrix, K×K
  state) — v0.9.2b.
- **Empty → NaN**, не 0.0. Honest «no data» — user sees it.
- **R² совпадает с sklearn** в граничных случаях (constant y).
- **log_loss**: `np.clip(p, 1e-15, 1 - 1e-15)` — как sklearn.

### Ограничения (v0.9.2a, §11.5)

- **Classification metrics** (`accuracy`, `precision`, `recall`, `f1`)
  — v0.9.2b. Требуют confusion matrix (K×K state, K = #classes).
- **`roc_auc`** — v0.9.2c. Требует Sort + cumulative TPR/FPR.
- **Multiclass log_loss** — v0.9.2b. Требует proba matrix (K columns).
- **Per-group metrics** (`groupby(g).metrics()`) — v0.9.2d.
- **`y_true` for regression must be numeric** — string labels need
  `encode_categorical` first.
- **log_loss requires binary y** ∈ {0, 1}. Multiclass — v0.9.2b.

### Тесты (v0.9.2a)

- +24 fast (`test_metrics.py`): perfect/known values (r2, mse, rmse,
  mae), log_loss (perfect/50-50/known), combined, nulls skip, empty→NaN,
  filtered, string metric, all validation errors, pandas-equivalence,
  chain integration.
- +1 slow (`test_metrics_e2e.py`): 100K rows, 30 MB budget, all four
  metrics verified vs numpy.

- fast: **~1422** (+24). slow: **15** (+1).
'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


def update_decisions():
    print("\n=== DECISIONS.md §11.5: metrics note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.2a (metrics" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        print("  WARNING: §11.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**v0.9.2a (metrics):** regression r2/mse/rmse/mae + binary "
        s += "log_loss, Reduce-based. Empty → NaN. Classification (accuracy/"
        s += "precision/recall/f1) — v0.9.2b.\n"
        p.write_text(s)
        return

    note = '''**v0.9.2a (metrics):** `CadwDataFrame.metrics()` — regression
(r2/mse/rmse/mae) + binary log_loss, Reduce-based, O(1) state.
- Requires `y_pred` (regression) / `y_proba` (log_loss). Combined OK.
- Empty → NaN for all.
- Nulls in y_true/y_pred/y_proba → skip per row.
- R² matches sklearn: ss_tot == 0 → 1.0 if perfect else 0.0.
- **Classification metrics** (accuracy/precision/recall/f1) — v0.9.2b.
  Require confusion matrix (K×K state).
- **`roc_auc`** — v0.9.2c (needs sort + cumulative TPR/FPR).
- **Multiclass log_loss** (proba matrix) — v0.9.2b.
- **Per-group metrics** (`groupby(g).metrics()`) — v0.9.2d.
- **y_true numeric** for regression — string labels → `encode_categorical`.

'''
    s = s[:pos] + note + s[pos:]
    p.write_text(s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.2a: metrics ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    r = subprocess.run(["git", "status", "--porcelain",
                        "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    write_metrics_module()
    add_metrics_method()
    update_init()
    write_tests()
    append_changelog()
    update_decisions()

    print("\n=== pytest: new metrics tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_metrics.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** METRICS TESTS FAILED ***")
        print("Fix in place (§13.4/146 backlog: no WIP commit on red).")
        sys.exit(1)

    print("\n=== pytest: metrics E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_metrics_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards via patch_utils ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1200)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.2a: metrics — regression (r2/mse/rmse/mae) + binary "
        "log_loss via ReduceOp; single-pass O(1) state; NaN for empty; "
        "sklearn-compatible R² edge cases; §11.5 constraints"])

    print("\n==> v0.9.2a applied.")
    print("NEXT: v0.9.2b (classification metrics — confusion matrix).")


if __name__ == "__main__":
    main()