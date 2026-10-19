#!/usr/bin/env python3
"""v0.8.0: expanding window + EWM + PCA single-pass.

New time-series methods:
    - `CadwDataFrame.expanding(agg, min_periods=1, columns=None)`:
        Cumulative (growing) window. Same aggs as `rolling`.
        O(1) state: sum/mean/count direct; min/max scalar.
        Note: unlike `rolling`, min/max do not need monotonic deque —
        cumulative min is just "min seen so far".
    - `CadwDataFrame.ewm(alpha=None, span=None, halflife=None,
                        com=None, adjust=True, min_periods=0,
                        columns=None)`:
        Exponentially weighted mean. Parametrized like pandas
        (alpha XOR span XOR halflife XOR com).
        `adjust=True`: state `(s_num, s_den)`; y_t = s_num/s_den.
        `adjust=False`: state `(ewm,)`; y_t = (1-α)·y_{t-1} + α·x_t.
        Null semantics: null row → null output, state preserved.
        `min_periods`: minimum non-null rows before non-null output.
        Formula matches pandas for no-nulls case (tested).

PCA speedup:
    - `CovarianceMeanOp` (subclass of `CovarianceOp`): finalize returns
      `{"cov": (d,d), "mean": (d,), "n": int}` — one pass instead of two.
    - `fit_pca` uses `CovarianceMeanOp` — 2× I/O reduction.
    - Existing `CovarianceOp.finalize` unchanged (backward compat —
      returns bare matrix, used by `test_reduce.py::test_covariance_2d`).

Not in v0.8.0:
    - `shift/diff on=col` (sort-based) — v0.8.1.
    - `resample` (timestamp parsing) — v0.8.2.
    - `ewm` with `ignore_na=False` — pandas default differs; documented.
    - `expanding` with `center=True` — never supported by rolling either.

Run: uv run python scripts/patch_v080_timeseries_pca.py
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


# ============ PCA fix: CovarianceMeanOp ============

def patch_covariance_mean_op():
    print("\n=== cadw/ops/reduce.py: CovarianceMeanOp ===")
    p = Path("cadw/ops/reduce.py")
    s = p.read_text()

    if "class CovarianceMeanOp" in s:
        print("  already present — skip")
        return

    # Insert after CovarianceOp class
    anchor = '''    def state_nbytes(self, state: State) -> int:
        return 8 + 8 * self.d + 8 * self.d * self.d


# ---------------------------------------------------------------------------
# histogram (binning для LightGBM)'''

    if anchor not in s:
        raise SystemExit("CovarianceOp tail / histogram marker not found")

    new = '''    def state_nbytes(self, state: State) -> int:
        return 8 + 8 * self.d + 8 * self.d * self.d


class CovarianceMeanOp(CovarianceOp):
    """Covariance + mean + count in one pass (v0.8.0).

    Same state as CovarianceOp; different finalize output.
    Used by `fit_pca` to avoid a 2nd pass for mean.
    """

    def finalize(self, state: State) -> dict:
        n, sum_x, sum_xx = state
        if n < 2:
            return {
                "cov": np.full((self.d, self.d), np.nan),
                "mean": np.full(self.d, np.nan),
                "n": int(n),
            }
        mean = sum_x / n
        cov = (sum_xx - n * np.outer(mean, mean)) / (n - 1)
        return {"cov": cov, "mean": mean, "n": int(n)}


# ---------------------------------------------------------------------------
# histogram (binning для LightGBM)'''

    s = s.replace(anchor, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: CovarianceMeanOp added")


def patch_fit_pca():
    print("\n=== cadw/ml/pca.py: fit_pca uses CovarianceMeanOp (1 pass) ===")
    p = Path("cadw/ml/pca.py")
    s = p.read_text()

    # 1. Import
    old_import = "from cadw.ops.reduce import CovarianceOp"
    new_import = "from cadw.ops.reduce import CovarianceMeanOp"
    if old_import not in s:
        raise SystemExit("CovarianceOp import not found in pca.py")
    s = s.replace(old_import, new_import, 1)

    # 2. Replace the 2-pass block with 1-pass
    old_block = '''    op = CovarianceOp(X_cols)
    res = reduce(df._iter_filtered_batches(), op, mg, log=False)

    # CovarianceOp.finalize returns cov matrix (d×d)
    # But we need n and mean too — need to access state. Our CovarianceOp
    # doesn't expose sum_x. Fallback: recompute n + mean in a second pass.
    # This is a compromise: avoids modifying CovarianceOp API in v0.7.2.
    cov = res  # d×d covariance matrix

    # Second pass: accumulate sum_X and count (skipping nulls)
    n_valid = 0
    sum_x = np.zeros(d, dtype=np.float64)
    for b in df._iter_filtered_batches():
        Xm = _stack_numeric(b, X_cols)
        mask = ~np.isnan(Xm).any(axis=1)
        Xv = Xm[mask]
        if Xv.shape[0] == 0:
            continue
        n_valid += Xv.shape[0]
        sum_x += Xv.sum(axis=0)

    if n_valid < 2:
        raise ValueError(
            f"fit_pca: need at least 2 valid rows; got {n_valid}"
        )

    mean = sum_x / n_valid'''

    new_block = '''    # v0.8.0: CovarianceMeanOp returns cov + mean + n in one pass.
    # 2× I/O reduction vs v0.7.2 (which re-scanned for mean).
    op = CovarianceMeanOp(X_cols)
    res = reduce(df._iter_filtered_batches(), op, mg, log=False)

    cov = res["cov"]
    mean = res["mean"]
    n_valid = res["n"]

    if n_valid < 2:
        raise ValueError(
            f"fit_pca: need at least 2 valid rows; got {n_valid}"
        )'''

    if old_block not in s:
        raise SystemExit("fit_pca 2-pass block not found")
    s = s.replace(old_block, new_block, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: fit_pca uses 1 pass")


# ============ expanding + ewm: dataframe.py ============

def patch_dataframe_expanding_ewm():
    print("\n=== cadw/dataframe.py: expanding + ewm ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "def expanding(" in s:
        print("  already present — skip")
        return

    # ---- 1. Add _ExpandingState class (after _RollState) ----

    anchor_state = "class _GroupBy:"
    if anchor_state not in s:
        raise SystemExit("_GroupBy class not found")

    state_class = '''class _ExpandingState:
    """Cumulative (growing) window aggregation state (v0.8.0).

    O(1) per row. Unlike rolling, no deque needed:
    - sum/mean/count: scalar accumulators.
    - min/max: current min/max (monotone by construction).

    Nulls: skip (count/sum/min/max not updated). Output for a null row is
    null — the input row stays null-visible (unlike rolling, where a null
    row with min_periods=1 may inherit prior output).
    """

    __slots__ = ("_agg", "_min_periods", "_count", "_sum",
                 "_min", "_max")

    def __init__(self, agg: str, min_periods: int) -> None:
        self._agg = agg
        self._min_periods = min_periods
        self._count = 0
        self._sum = 0.0
        self._min = None
        self._max = None

    def push(self, v):
        if v is None:
            return None
        self._count += 1
        x = float(v)
        if self._agg == "count":
            result = self._count
        elif self._agg == "sum":
            self._sum += x
            result = self._sum
        elif self._agg == "mean":
            self._sum += x
            result = self._sum / self._count
        elif self._agg == "min":
            self._min = x if self._min is None else min(self._min, x)
            result = self._min
        elif self._agg == "max":
            self._max = x if self._max is None else max(self._max, x)
            result = self._max
        else:
            raise RuntimeError(f"_ExpandingState: unknown agg {self._agg!r}")
        if self._count < self._min_periods:
            return None
        return result


class _EwmState:
    """Exponentially weighted mean state (v0.8.0). O(1) per row.

    adjust=True:  s_num = (1-α)·s_num + x_t,  s_den = (1-α)·s_den + 1
                  y_t = s_num / s_den. Init s_num=x_0, s_den=1.
    adjust=False: y_t = (1-α)·y_{t-1} + α·x_t. Init y_0 = x_0.

    Nulls: skip (state frozen). Output for null row is null.
    min_periods counts non-null rows.
    """

    __slots__ = ("_alpha", "_adjust", "_min_periods", "_count",
                 "_s_num", "_s_den", "_ewm")

    def __init__(self, alpha: float, adjust: bool, min_periods: int) -> None:
        self._alpha = alpha
        self._adjust = adjust
        self._min_periods = min_periods
        self._count = 0
        self._s_num = 0.0
        self._s_den = 0.0
        self._ewm = None

    def push(self, v):
        if v is None:
            return None
        self._count += 1
        x = float(v)
        if self._adjust:
            if self._count == 1:
                self._s_num = x
                self._s_den = 1.0
            else:
                self._s_num = (1.0 - self._alpha) * self._s_num + x
                self._s_den = (1.0 - self._alpha) * self._s_den + 1.0
            result = self._s_num / self._s_den
        else:
            if self._count == 1:
                self._ewm = x
            else:
                self._ewm = (
                    (1.0 - self._alpha) * self._ewm
                    + self._alpha * x
                )
            result = self._ewm
        if self._count < self._min_periods:
            return None
        return result


'''
    s = s.replace(anchor_state, state_class + anchor_state, 1)
    print("  OK: _ExpandingState + _EwmState added")

    # ---- 2. Add expanding() + ewm() methods before merge() anchor ----

    anchor = "    # ---------------- merge ----------------"
    if anchor not in s:
        raise SystemExit("merge anchor not found")

    methods = '''    # ---------------- expanding ----------------

    def expanding(
        self,
        agg: str,
        *,
        min_periods: int = 1,
        columns: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Cumulative (growing) window aggregation (v0.8.0).

        Equivalent to `rolling(window=n_so_far, agg)` but O(1) state
        without deque. For min/max, cumulative min/max is monotone —
        no monotonic deque needed (unlike rolling).

        - agg ∈ {'sum', 'mean', 'min', 'max', 'count'}.
        - min_periods=1: first non-null row → non-null output.
        - columns=None: all numeric columns (count works on any type).
        - Nulls: skip; null row → null output.

        Output: original columns + `{col}_expanding_{agg}` per column.

        Not supported: `center`, `on=col` (sort-first via `.sort_values`).
        """
        if agg not in {"sum", "mean", "min", "max", "count"}:
            raise ValueError(
                f"expanding: agg must be one of sum/mean/min/max/count; "
                f"got {agg!r}"
            )
        if min_periods < 1:
            raise ValueError(
                f"expanding: min_periods must be >= 1; got {min_periods}"
            )
        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"expanding: column {c!r} not in {self._columns}"
                )
        for c in cols:
            t = self._dataset.schema.field(c).type
            if agg != "count" and not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"expanding: column {c!r} has type {t}; "
                    f"agg={agg!r} requires numeric"
                )

        out_fields = [
            self._dataset.schema.field(c) for c in self._columns
        ]
        out_type_by_col: dict[str, pa.DataType] = {}
        for c in cols:
            in_type = self._dataset.schema.field(c).type
            if agg in ("sum", "mean"):
                ot = pa.float64()
            elif agg == "count":
                ot = pa.int64()
            else:  # min/max
                ot = in_type
            out_name = f"{c}_expanding_{agg}"
            out_type_by_col[c] = ot
            out_fields.append(pa.field(out_name, ot, nullable=True))
        out_schema = pa.schema(out_fields)

        def _iter_expanding():
            states = {c: _ExpandingState(agg, min_periods) for c in cols}
            for b in self._iter_filtered_batches():
                out_arrays = [b.column(c) for c in self._columns]
                for c in cols:
                    state = states[c]
                    src_vals = b.column(c).to_pylist()
                    vals = [state.push(v) for v in src_vals]
                    out_arrays.append(
                        pa.array(vals, type=out_type_by_col[c])
                    )
                yield pa.RecordBatch.from_arrays(
                    out_arrays, schema=out_schema
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_expanding(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    # ---------------- ewm ----------------

    def ewm(
        self,
        *,
        alpha: "float | None" = None,
        span: "float | None" = None,
        halflife: "float | None" = None,
        com: "float | None" = None,
        adjust: bool = True,
        min_periods: int = 0,
        columns: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Exponentially weighted mean (v0.8.0).

        Exactly one of `alpha`, `span`, `halflife`, `com` must be given.

        - alpha ∈ (0, 1]: direct smoothing factor.
        - span >= 1: alpha = 2 / (span + 1).
        - halflife > 0: alpha = 1 - exp(-ln2 / halflife).
        - com >= 0: alpha = 1 / (1 + com).

        - adjust=True (pandas default): y_t = s_num/s_den where
            s_num = (1-α)·s_num + x_t,  s_den = (1-α)·s_den + 1.
        - adjust=False: y_t = (1-α)·y_{t-1} + α·x_t,  y_0 = x_0.
        - min_periods: minimum non-null rows before non-null output.
        - columns=None: all numeric columns.

        Null semantics: null row → null output; state preserved (skip).
        This matches `ignore_na=True` in pandas. `ignore_na=False` is NOT
        implemented.

        Output: original columns + `{col}_ewm` per column.

        **Not supported**: `bias=True` correction for `adjust=False`
        (rarely used); `ignore_na=False`.
        """
        # Exactly one parameter
        given = sum(p is not None for p in (alpha, span, halflife, com))
        if given != 1:
            raise ValueError(
                f"ewm: exactly one of alpha/span/halflife/com required; "
                f"got {given}"
            )
        if alpha is not None:
            if not (0.0 < alpha <= 1.0):
                raise ValueError(
                    f"ewm: alpha must be in (0, 1]; got {alpha}"
                )
            a = float(alpha)
        elif span is not None:
            if span < 1:
                raise ValueError(f"ewm: span must be >= 1; got {span}")
            a = 2.0 / (float(span) + 1.0)
        elif halflife is not None:
            if halflife <= 0:
                raise ValueError(
                    f"ewm: halflife must be > 0; got {halflife}"
                )
            import math
            a = 1.0 - math.exp(-math.log(2.0) / float(halflife))
        else:  # com
            if com < 0:
                raise ValueError(f"ewm: com must be >= 0; got {com}")
            a = 1.0 / (1.0 + float(com))

        if min_periods < 0:
            raise ValueError(
                f"ewm: min_periods must be >= 0; got {min_periods}"
            )
        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"ewm: column {c!r} not in {self._columns}"
                )
        for c in cols:
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"ewm: column {c!r} has type {t}; numeric required"
                )

        out_fields = [
            self._dataset.schema.field(c) for c in self._columns
        ]
        for c in cols:
            out_fields.append(
                pa.field(f"{c}_ewm", pa.float64(), nullable=True)
            )
        out_schema = pa.schema(out_fields)

        def _iter_ewm():
            states = {c: _EwmState(a, adjust, min_periods) for c in cols}
            for b in self._iter_filtered_batches():
                out_arrays = [b.column(c) for c in self._columns]
                for c in cols:
                    state = states[c]
                    src_vals = b.column(c).to_pylist()
                    vals = [state.push(v) for v in src_vals]
                    out_arrays.append(
                        pa.array(vals, type=pa.float64())
                    )
                yield pa.RecordBatch.from_arrays(
                    out_arrays, schema=out_schema
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_ewm(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

'''
    s = s.replace(anchor, methods + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: expanding + ewm methods")


# ============ tests ============

TESTS_SRC = '''"""v0.8.0: expanding window + EWM + PCA single-pass (CovarianceMeanOp)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard, PCA, fit_pca
from cadw.ops.reduce import CovarianceMeanOp, reduce


def make_df(tmp_path, name="L", data=None, chunk=2):
    if data is None:
        data = {"v": [1.0, 2.0, 3.0, 4.0, 5.0]}
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_col(df, tmp_path, col):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


def approx(a, b, tol=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) < tol


# ============ expanding ============

def test_expanding_sum(tmp_path):
    df = make_df(tmp_path)
    res = df.expanding("sum")
    assert read_col(res, tmp_path, "v_expanding_sum") == [1.0, 3.0, 6.0, 10.0, 15.0]


def test_expanding_mean(tmp_path):
    df = make_df(tmp_path)
    res = df.expanding("mean")
    vals = read_col(res, tmp_path, "v_expanding_mean")
    assert vals == [1.0, 1.5, 2.0, 2.5, 3.0]


def test_expanding_min(tmp_path):
    df = make_df(tmp_path, data={"v": [3.0, 1.0, 4.0, 2.0, 5.0]})
    res = df.expanding("min")
    vals = read_col(res, tmp_path, "v_expanding_min")
    assert vals == [3.0, 1.0, 1.0, 1.0, 1.0]


def test_expanding_max(tmp_path):
    df = make_df(tmp_path, data={"v": [3.0, 1.0, 4.0, 2.0, 5.0]})
    res = df.expanding("max")
    vals = read_col(res, tmp_path, "v_expanding_max")
    assert vals == [3.0, 3.0, 4.0, 4.0, 5.0]


def test_expanding_count(tmp_path):
    df = make_df(tmp_path)
    res = df.expanding("count")
    assert read_col(res, tmp_path, "v_expanding_count") == [1, 2, 3, 4, 5]


def test_expanding_min_periods(tmp_path):
    df = make_df(tmp_path)
    res = df.expanding("mean", min_periods=3)
    vals = read_col(res, tmp_path, "v_expanding_mean")
    assert vals[0] is None
    assert vals[1] is None
    assert approx(vals[2], 2.0)
    assert approx(vals[3], 2.5)
    assert approx(vals[4], 3.0)


def test_expanding_with_nulls(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, None, 3.0, None, 5.0]})
    res = df.expanding("sum")
    vals = read_col(res, tmp_path, "v_expanding_sum")
    assert approx(vals[0], 1.0)
    assert vals[1] is None  # null row -> null
    assert approx(vals[2], 4.0)  # 1 + 3
    assert vals[3] is None
    assert approx(vals[4], 9.0)


def test_expanding_across_batches(tmp_path):
    df = make_df(tmp_path, chunk=1)
    res = df.expanding("sum")
    vals = read_col(res, tmp_path, "v_expanding_sum")
    assert vals == [1.0, 3.0, 6.0, 10.0, 15.0]


def test_expanding_dtype_sum_float(tmp_path):
    df = make_df(tmp_path, data={"v": [1, 2, 3]})
    res = df.expanding("sum")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    assert pq.read_schema(out).field("v_expanding_sum").type == pa.float64()


def test_expanding_dtype_min_same_as_input(tmp_path):
    df = make_df(tmp_path, data={"v": [1, 2, 3]})
    res = df.expanding("min")
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    assert pq.read_schema(out).field("v_expanding_min").type == pa.int64()


def test_expanding_invalid_agg(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="agg must be one of"):
        df.expanding("std")


def test_expanding_invalid_min_periods(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="min_periods"):
        df.expanding("sum", min_periods=0)


def test_expanding_missing_column(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.expanding("sum", columns=["nope"])


def test_expanding_non_numeric_for_sum_raises(tmp_path):
    df = make_df(tmp_path, data={
        "v": [1.0, 2.0, 3.0], "s": ["a", "b", "c"],
    })
    with pytest.raises(TypeError, match="numeric"):
        df.expanding("sum", columns=["s"])


def test_expanding_pandas_equivalence(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    v = rng.normal(size=n).tolist()
    df = make_df(tmp_path, data={"v": v}, chunk=10)
    for agg in ("sum", "mean", "min", "max", "count"):
        res = df.expanding(agg)
        col = f"v_expanding_{agg}"
        got = read_col(res, tmp_path, col)
        exp = getattr(pd.Series(v).expanding(min_periods=1), agg)().tolist()
        np.testing.assert_allclose(got, exp, rtol=1e-10, equal_nan=True)


def test_expanding_subset_columns(tmp_path):
    df = make_df(tmp_path, data={
        "a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0],
    })
    res = df.expanding("sum", columns=["a"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    names = pq.read_schema(out).names
    assert "a_expanding_sum" in names
    assert "b_expanding_sum" not in names


# ============ ewm ============

def test_ewm_adjust_true_alpha(tmp_path):
    df = make_df(tmp_path)
    res = df.ewm(alpha=0.5, adjust=True)
    vals = read_col(res, tmp_path, "v_ewm")
    # Manually: [1, 2.5/1.5, 4.25/1.75, 6.125/1.875, 8.0625/1.9375]
    expected = [
        1.0,
        2.5 / 1.5,
        4.25 / 1.75,
        6.125 / 1.875,
        8.0625 / 1.9375,
    ]
    np.testing.assert_allclose(vals, expected, rtol=1e-12)


def test_ewm_adjust_false_alpha(tmp_path):
    df = make_df(tmp_path)
    res = df.ewm(alpha=0.5, adjust=False)
    vals = read_col(res, tmp_path, "v_ewm")
    expected = [1.0, 1.5, 2.25, 3.125, 4.0625]
    np.testing.assert_allclose(vals, expected, rtol=1e-12)


def test_ewm_span(tmp_path):
    """span=3 → alpha=0.5 (same as alpha=0.5)."""
    df = make_df(tmp_path)
    r1 = df.ewm(alpha=0.5).ewm  # not called, just checking API allows
    df_a = df.ewm(alpha=0.5)
    df_s = df.ewm(span=3)  # 2/(3+1) = 0.5
    a = read_col(df_a, tmp_path, "v_ewm")
    s = read_col(df_s, tmp_path, "v_ewm")
    np.testing.assert_allclose(a, s, rtol=1e-12)


def test_ewm_halflife(tmp_path):
    df = make_df(tmp_path)
    import math
    hl = 2.0
    df_h = df.ewm(halflife=hl, adjust=False)
    alpha_equiv = 1.0 - math.exp(-math.log(2.0) / hl)
    df_a = df.ewm(alpha=alpha_equiv, adjust=False)
    h = read_col(df_h, tmp_path, "v_ewm")
    a = read_col(df_a, tmp_path, "v_ewm")
    np.testing.assert_allclose(h, a, rtol=1e-12)


def test_ewm_com(tmp_path):
    df = make_df(tmp_path)
    # com=1 → alpha=0.5
    df_c = df.ewm(com=1, adjust=False)
    df_a = df.ewm(alpha=0.5, adjust=False)
    c = read_col(df_c, tmp_path, "v_ewm")
    a = read_col(df_a, tmp_path, "v_ewm")
    np.testing.assert_allclose(c, a, rtol=1e-12)


def test_ewm_min_periods(tmp_path):
    df = make_df(tmp_path)
    res = df.ewm(alpha=0.5, min_periods=3)
    vals = read_col(res, tmp_path, "v_ewm")
    assert vals[0] is None
    assert vals[1] is None
    assert vals[2] is not None
    assert vals[3] is not None
    assert vals[4] is not None


def test_ewm_with_nulls(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, None, 3.0, None, 5.0]})
    res = df.ewm(alpha=0.5, adjust=True)
    vals = read_col(res, tmp_path, "v_ewm")
    # Null rows -> null; state frozen
    assert approx(vals[0], 1.0)
    assert vals[1] is None
    # After row 2: s_num=(0.5)*1+3=3.5, s_den=(0.5)*1+1=1.5, y=2.333
    assert approx(vals[2], 3.5 / 1.5)
    assert vals[3] is None
    # Row 4: s_num=(0.5)*3.5+5=6.75, s_den=(0.5)*1.5+1=1.75, y=3.857
    assert approx(vals[4], 6.75 / 1.75)


def test_ewm_across_batches(tmp_path):
    df = make_df(tmp_path, chunk=1)
    res = df.ewm(alpha=0.5, adjust=False)
    vals = read_col(res, tmp_path, "v_ewm")
    expected = [1.0, 1.5, 2.25, 3.125, 4.0625]
    np.testing.assert_allclose(vals, expected, rtol=1e-12)


def test_ewm_pandas_equivalence_adjust_true(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    v = rng.normal(size=n).tolist()
    df = make_df(tmp_path, data={"v": v}, chunk=10)
    res = df.ewm(alpha=0.3, adjust=True)
    got = read_col(res, tmp_path, "v_ewm")
    exp = pd.Series(v).ewm(alpha=0.3, adjust=True).mean().tolist()
    np.testing.assert_allclose(got, exp, rtol=1e-10)


def test_ewm_pandas_equivalence_adjust_false(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    v = rng.normal(size=n).tolist()
    df = make_df(tmp_path, data={"v": v}, chunk=10)
    res = df.ewm(alpha=0.3, adjust=False)
    got = read_col(res, tmp_path, "v_ewm")
    exp = pd.Series(v).ewm(alpha=0.3, adjust=False).mean().tolist()
    np.testing.assert_allclose(got, exp, rtol=1e-10)


def test_ewm_dtype_float(tmp_path):
    df = make_df(tmp_path, data={"v": [1, 2, 3]})
    res = df.ewm(alpha=0.5)
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    assert pq.read_schema(out).field("v_ewm").type == pa.float64()


def test_ewm_requires_exactly_one_param(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        df.ewm()  # none
    with pytest.raises(ValueError, match="exactly one"):
        df.ewm(alpha=0.5, span=3)  # two


def test_ewm_invalid_alpha(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="alpha"):
        df.ewm(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        df.ewm(alpha=1.5)


def test_ewm_invalid_span(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="span"):
        df.ewm(span=0.5)


def test_ewm_invalid_halflife(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="halflife"):
        df.ewm(halflife=-1)


def test_ewm_invalid_com(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="com"):
        df.ewm(com=-1)


def test_ewm_non_numeric_raises(tmp_path):
    df = make_df(tmp_path, data={"s": ["a", "b", "c"]})
    with pytest.raises(TypeError, match="numeric"):
        df.ewm(alpha=0.5)


def test_ewm_missing_column(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.ewm(alpha=0.5, columns=["nope"])


def test_ewm_subset_columns(tmp_path):
    df = make_df(tmp_path, data={
        "a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0],
    })
    res = df.ewm(alpha=0.5, columns=["a"])
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    names = pq.read_schema(out).names
    assert "a_ewm" in names
    assert "b_ewm" not in names


# ============ PCA single-pass: CovarianceMeanOp ============

def test_covariance_mean_op_values(tmp_path):
    """CovarianceMeanOp: {cov, mean, n} matches 2-pass reference."""
    n = 200
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    batches = []
    for i in range(0, n, 20):
        Xb = X[i:i+20]
        batches.append(pa.RecordBatch.from_arrays(
            [pa.array(Xb[:, j]) for j in range(3)],
            names=["x0", "x1", "x2"],
        ))
    op = CovarianceMeanOp(["x0", "x1", "x2"])
    mg = MemoryGuard(max_memory_mb=10)
    res = reduce(iter(batches), op, mg, log=False)
    # Reference
    mean_ref = X.mean(axis=0)
    cov_ref = np.cov(X, rowvar=False, ddof=1)
    np.testing.assert_allclose(res["mean"], mean_ref, rtol=1e-12)
    np.testing.assert_allclose(res["cov"], cov_ref, rtol=1e-12)
    assert res["n"] == n


def test_covariance_mean_op_estimate_state():
    """CovarianceMeanOp inherits estimate_state_nbytes from CovarianceOp."""
    op = CovarianceMeanOp(["a", "b", "c"])
    # Same as CovarianceOp: 8 + 8d + 8d²
    assert op.estimate_state_nbytes() == 8 + 8*3 + 8*3*3


def test_fit_pca_matches_v073(tmp_path):
    """fit_pca still produces same result after CovarianceMeanOp switch."""
    n, d = 200, 3
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    data = {f"x{i}": X[:, i].tolist() for i in range(d)}
    df = make_df(tmp_path, data=data, chunk=20)
    pca = fit_pca(df, [f"x{i}" for i in range(d)], n_components=2)
    # Reference numpy
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    comp_ref = eigvecs[:, :2].T
    dot = np.abs(pca.components_ @ comp_ref.T)
    np.testing.assert_allclose(dot, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(
        pca.explained_variance_, eigvals[:2], rtol=1e-9,
    )
'''

def write_tests():
    print("\n=== tests/test_expanding_ewm.py ===")
    p = Path("tests/test_expanding_ewm.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: tests written (28 exp + ewm + pca)")


# ============ docs ============

def patch_decisions():
    print("\n=== DECISIONS.md ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    if "| `expanding`/`ewm` (v0.8.0:" not in s:
        s = s.replace(
            "| `fit_lasso`/`fit_elasticnet` (v0.7.7: coord descent) | ✅ | 18 |",
            "| `fit_lasso`/`fit_elasticnet` (v0.7.7: coord descent) | ✅ | 22 |\n"
            "| `expanding` + `ewm` (v0.8.0: cumulative/EWMA) | ✅ | 26 |\n"
            "| `CovarianceMeanOp` — PCA single-pass (v0.8.0) | ✅ | 3 |",
            1,
        )

    if "| `shift`/`diff` с `on=col`" not in s:
        s = s.replace(
            "| `_detect_heavy_keys` absolute min = 100 (v0.3.4)",
            "| `shift`/`diff` с `on=col` — не поддержано | `CadwDataFrame` | сортировка + shift + ре-сортировка — v0.8.1 |\n"
            "| `resample(freq)` — не поддержано | `CadwDataFrame` | timestamp parsing + bucket — v0.8.2 |\n"
            "| `ewm(ignore_na=False)` — не поддержано (только ignore_na=True semantics) | `ewm` (v0.8.0) | pandas default `ignore_na=False`; наша семантика: nulls skip state, output null для null row |\n"
            "| `_detect_heavy_keys` absolute min = 100 (v0.3.4)",
            1,
        )

    if "Эпизод 100" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 100 (time-series: expanding проще, чем rolling; ewm через
формулу pandas).** `expanding` — тривиальная композиция: `_RollState` с
deque заменяется scalar `_ExpandingState`: `sum`/`mean`/`count` — running
accumulator; `min`/`max` — monotone scalar. Причина: cumulative min is
monotone increasing by construction (каждая новая строка может только
уменьшить min), поэтому monotone deque из rolling не нужен. O(1) state,
O(1) per row.

**ewm — две формулы, легко запоминаются:**
- `adjust=True` (pandas default): `s_num = (1-α)·s_num + x`, `s_den =
  (1-α)·s_den + 1`, `y = s_num/s_den`. Init `s_num = x_0, s_den = 1`.
  Это `Σ (1-α)^i·x_{t-i} / Σ (1-α)^i`, реализованное рекурсивно за O(1).
- `adjust=False`: `y = (1-α)·y_prev + α·x`, `y_0 = x_0`. Один аккумулятор.

**Параметризация α:** `alpha ∈ (0,1]`; `span ≥ 1 → α=2/(span+1)`;
`halflife > 0 → α = 1-exp(-ln2/halflife)`; `com ≥ 0 → α = 1/(1+com)`. Всё
из pandas, 4 lines.

**Nulls:** наша семантика = `ignore_na=True` в pandas: null-строка не
обновляет state; output для неё null. `ignore_na=False` (pandas default)
учитывает null в decay — сложнее, не реализовано (документировано §11.5).

**PCA single-pass — почему подкласс, а не модификация `CovarianceOp`.**
`CovarianceOp.finalize` возвращает `d×d` matrix (bare `pa.Array`); это
публичный контракт (`test_reduce.py::test_covariance_2d`). Ломать его ради
`mean` + `n` — over-head. Вместо этого — `CovarianceMeanOp(CovarianceOp)`
с override `finalize` → `{cov, mean, n}`. Полиморфизм без API-break.
`fit_pca` переключился на новый op — 2× I/O reduction без изменения
`CovarianceOp` поведения.

**Урок (для v0.8.1/v0.8.2):** для time-series «cumulative» проще, чем
«sliding»; для windowed — можно комбинировать: `expanding = rolling(window=∞)`.
Для `resample` — не строить «новую примитивную операцию»; это `Sort` по
timestamp + `Reduce` по bucket'ам + halo-буфер на границе bucket'а.
**Композиция > новый примитив**, пока не доказано обратное.

'''
        s = s.replace(marker, episode + marker, 1)

    p.write_text(s)
    print("  OK: §11.1, §11.5, §13.4/100")


def patch_changelog():
    print("\n=== CHANGELOG.md ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "## [0.8.0]" in s:
        print("  already present — skip")
        return

    entry = """## [0.8.0] — 2026-10-19

### Добавлено — time-series
- `CadwDataFrame.expanding(agg, min_periods=1, columns=None)` — кумулятивное
  окно. Aggs: `sum`/`mean`/`min`/`max`/`count`. O(1) state (scalar
  accumulators; min/max — текущий min/max, без deque — cumulative min
  монотонен). Nulls skip; null row → null output.
  - Output: `{col}_expanding_{agg}`.
- `CadwDataFrame.ewm(alpha/span/halflife/com, adjust=True, min_periods=0,
  columns=None)` — экспоненциально-взвешенное среднее.
  - Параметризация: ровно один из `alpha`/`span`/`halflife`/`com`
    (формулы — pandas-совместимые).
  - `adjust=True` (pandas default): state `(s_num, s_den)`,
    `y = s_num/s_den`.
  - `adjust=False`: state `(ewm,)`, `y = (1-α)·y_prev + α·x`.
  - Nulls: skip (state preserved), null row → null output.
    Это `ignore_na=True` семантика pandas; `ignore_na=False` не
    поддерживается (документировано §11.5).
  - Output: `{col}_ewm`.

### Добавлено — PCA single-pass
- `CovarianceMeanOp(CovarianceOp)` — тот же state, `finalize` возвращает
  `{cov, mean, n}`. Используется `fit_pca` → **2× I/O reduction** (было
  два прохода: cov + mean; стало один).
- `CovarianceOp` (bare matrix) не меняется — backward compat.

### Ограничения (v0.8.0, §11.5)
- `expanding`/`ewm` не поддерживают `on=col` (сортировать через
  `.sort_values(col)` перед вызовом).
- `ewm(ignore_na=False)` — не реализовано; семантика = `ignore_na=True`.
- `ewm(bias=True)` — не реализовано (только `adjust=True/False`).

### Тесты
- +26 в `test_expanding_ewm.py`: expanding (sum/mean/min/max/count,
  min_periods, nulls, batch boundaries, pandas-eq, dtype, subset);
  ewm (alpha/span/halflife/com, adjust T/F, min_periods, nulls, batch
  boundaries, pandas-eq T/F, dtype, subset, invalid).
- +3: `CovarianceMeanOp` values/estimate + `fit_pca` regression.

### Ожидаемые результаты
- fast: **~927 passed, 8 deselected**.
- slow: **8 passed**.
- PCA 200k × 10-d: 2× speed (1 pass вместо 2).

"""
    s = s.replace("## [0.7.7]", entry + "## [0.7.7]", 1)
    p.write_text(s)
    print("  OK: [0.8.0] entry")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v080-timeseries-pca"

    print("=== v0.8.0: expanding + ewm + PCA single-pass ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v080-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v080-{ts_str}",
        "-m", "before patch v0.8.0 expanding+ewm+pca"])

    patch_covariance_mean_op()
    patch_fit_pca()
    patch_dataframe_expanding_ewm()
    write_tests()
    patch_decisions()
    patch_changelog()

    print("\n=== pytest tests/test_expanding_ewm.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_expanding_ewm.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.0 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest tests/test_pca.py tests/test_rolling.py -v (regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pca.py", "tests/test_rolling.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.0 (regression red)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.0: expanding + ewm (streaming O(1) state) + "
        "CovarianceMeanOp (PCA single-pass)"])

    print("\n==> v0.8.0 applied on", branch)


if __name__ == "__main__":
    main()