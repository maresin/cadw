"""Distribution drift and A/B tests (D.6.2, v0.9.40).

Reference: Massey, *The Kolmogorov-Smirnov Test for Goodness of Fit*,
JASA 46(253), 1951 (two-sample KS); scipy.stats.ks_2samp / chi2_contingency
/ mannwhitneyu / ttest_ind as reference implementations; Population
Stability Index as defined in Yurdakul, *Statistical Properties of Population
Stability Index*, 2018; Deng, Xu, Kohavi, Walker, *Improving the Sensitivity
of Online Controlled Experiments by Utilizing Pre-Experiment Data*, WSDM
2013 (CUPED).
See DECISION_LOG for algorithm reference.

Drift: two-sample KS on numeric columns, chi-square on categorical
columns, and PSI with configurable bins.
A/B: Welch t-test, Mann-Whitney U, and CUPED-adjusted treatment effect.
All functions take pre-computed numpy arrays (the DataFrame-level bindings
stream the two columns into bounded arrays first).
"""
from __future__ import annotations

from typing import Any, Iterator, Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.core.spill import SpillDir
from cadw.core.warnings import (
    SlowOperationRequiresConfirm,
    SlowOperationWarning,
)
from cadw.ml._logging import log_fit
from cadw.ops.reduce import ReduceOp, reduce as _reduce
from cadw.ops.sort import external_sort, read_sorted
from cadw.ops.stats import (  # v0.9.153: moved canonical home
    ks_2samp, psi, chi2_categorical,
)


def welch_t(a, b):
    """Welch two-sample t-test: returns (statistic, p-value, diff)."""
    from scipy.stats import ttest_ind
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("welch_t: both samples need >= 2 observations")
    res = ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue), float(a.mean() - b.mean())


def mann_whitney(a, b):
    """Mann-Whitney U test: returns (statistic, p-value)."""
    from scipy.stats import mannwhitneyu
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("mann_whitney: both samples need >= 2 observations")
    res = mannwhitneyu(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def cuped(y, x, t):
    """CUPED-adjusted treatment effect (Deng et al. 2013).

    y: outcome, x: pre-experiment covariate, t: 0/1 treatment indicator.
    Returns dict with raw effect, adjusted effect and theta.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if y.size != x.size or y.size != t.size or y.size < 4:
        raise ValueError("cuped: y, x, t must be equal length >= 4")
    uniq = np.unique(t)
    if not np.array_equal(uniq, np.array([0.0, 1.0])):
        raise ValueError(f"cuped: t must be binary 0/1; got {uniq.tolist()}")
    theta = float(np.cov(x, y, bias=True)[0, 1] / np.var(x) if np.var(x) > 0 else 0.0)
    y_adj = y - theta * (x - x.mean())
    raw = float(y[t == 1].mean() - y[t == 0].mean())
    adj = float(y_adj[t == 1].mean() - y_adj[t == 0].mean())
    return {"raw_effect": raw, "adjusted_effect": adj, "theta": theta}


class DriftABModel:
    """Fitted A/B / drift model (v0.9.119).

    kind="drift": container only (state_nbytes=256), unchanged.
    kind="ab": OOC fit via Reduce (t/cuped) or external_sort (mw);
    result_ holds the test statistics; state_nbytes() is formula-based.
    """

    def __init__(self, kind, params, *, result=None, arms=2):
        self.kind_ = kind
        self.params_ = dict(params)
        self.result_ = dict(result) if result is not None else None
        self.arms_ = int(arms)
        self.fitted_ = True

    def state_nbytes(self):
        if self.kind_ != "ab" or self.result_ is None:
            return 256
        # per-arm: 6 f64 (n, sy, sy2, sx, sxy, sx2) = 48 bytes;
        # plus per-arm Python dict overhead (dict_entry_bytes per key x 6).
        from cadw.ml._memory_estimate import _load_constants
        c = _load_constants()
        de = int(c.get("dict_entry_bytes", 116))
        method = self.params_.get("method", "t")
        if method == "mw":
            # rank-scan state: r0,r1 (f64) + n0,n1 (i64) + rank_pos,cur_val (f64)
            return 5 * 8 + 256
        n_f64_per_arm = 3 if method == "t" else 6
        return int(self.arms_ * (n_f64_per_arm * 8 + 6 * de) + 256)

    def to_dict(self):
        out = {"kind": self.kind_, "params": self.params_}
        if self.result_ is not None:
            out["result"] = dict(self.result_)
        return out


def fit_drift(df, col, reference_col, *, method="ks", n_bins=10,
              memory_guard: Optional[MemoryGuard] = None):
    """Fit a drift estimator over two columns (current vs reference)."""
    if col not in df._columns:
        raise KeyError(f"fit_drift: column {col!r} not in {df._columns}")
    if reference_col not in df._columns:
        raise KeyError(f"fit_drift: column {reference_col!r} not in {df._columns}")
    if method not in ("ks", "psi"):
        raise ValueError(f"fit_drift: method must be 'ks' or 'psi'; got {method!r}")
    mg = memory_guard or df._memory_guard
    m = DriftABModel("drift", {"col": col, "reference_col": reference_col,
                                "method": method, "n_bins": n_bins})
    log_fit("drift", state_bytes=m.state_nbytes())
    return m


# ---------- v0.9.119: OOC ReduceOps for A/B ----------

class _ArmStatsOp(ReduceOp):
    """Per-arm streaming stats: (n, sy, sy2) for method='t'."""

    def __init__(self, outcome_col: str, treatment_col: str):
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col

    def estimate_state_nbytes(self) -> int:
        return 256

    def init(self):
        # dict[arm] -> np.array([n, sy, sy2], f64); arms encoded in batch.
        return {}

    def update(self, state, batch):
        y = batch.column(self.outcome_col).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        t = batch.column(self.treatment_col).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        mask = ~(np.isnan(y) | np.isnan(t))
        y = y[mask]; t = t[mask]
        for arm in (0.0, 1.0):
            sel = (t == arm)
            if not np.any(sel):
                continue
            ys = y[sel]
            acc = state.get(arm)
            if acc is None:
                acc = np.zeros(3, dtype=np.float64)
            acc[0] += ys.size
            acc[1] += float(ys.sum())
            acc[2] += float((ys * ys).sum())
            state[arm] = acc
        return state

    def merge(self, a, b):
        out = dict(a)
        for arm, acc_b in b.items():
            if arm in out:
                out[arm] = out[arm] + acc_b
            else:
                out[arm] = acc_b
        return out

    def finalize(self, state):
        return {int(k): v for k, v in state.items()}

    def state_nbytes(self, state):
        return int(len(state) * 3 * 8 + len(state) * 6 * 116 + 256)

    def batch_memory_factor(self) -> float:
        return 4.0   # y + t + mask + per-arm selection copy


class _CupedStatsOp(ReduceOp):
    """Per-arm streaming stats: (n, sy, sy2, sx, sxy, sx2) for cuped."""

    def __init__(self, outcome_col, treatment_col, covariate_col):
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.covariate_col = covariate_col

    def estimate_state_nbytes(self) -> int:
        return 512

    def init(self):
        return {}

    def update(self, state, batch):
        y = batch.column(self.outcome_col).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        x = batch.column(self.covariate_col).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        t = batch.column(self.treatment_col).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        mask = ~(np.isnan(y) | np.isnan(x) | np.isnan(t))
        y = y[mask]; x = x[mask]; t = t[mask]
        for arm in (0.0, 1.0):
            sel = (t == arm)
            if not np.any(sel):
                continue
            ys = y[sel]; xs = x[sel]
            acc = state.get(arm)
            if acc is None:
                acc = np.zeros(6, dtype=np.float64)
            acc[0] += ys.size
            acc[1] += float(ys.sum())
            acc[2] += float((ys * ys).sum())
            acc[3] += float(xs.sum())
            acc[4] += float((xs * ys).sum())
            acc[5] += float((xs * xs).sum())
            state[arm] = acc
        return state

    def merge(self, a, b):
        out = dict(a)
        for arm, acc_b in b.items():
            out[arm] = out.get(arm, np.zeros(6)) + acc_b
        return out

    def finalize(self, state):
        return {int(k): v for k, v in state.items()}

    def state_nbytes(self, state):
        return int(len(state) * 6 * 8 + len(state) * 6 * 116 + 256)

    def batch_memory_factor(self) -> float:
        return 5.0


def _welch_from_aggregates(agg: dict) -> dict:
    """Welch t from (n, sy, sy2) per arm 0/1."""
    a = agg.get(0); b = agg.get(1)
    if a is None or b is None or a[0] < 2 or b[0] < 2:
        return {"statistic": float("nan"), "pvalue": float("nan"),
                "diff": float("nan"), "n0": int(a[0] if a is not None else 0),
                "n1": int(b[0] if b is not None else 0)}
    n0, s0, s20 = a; n1, s1, s21 = b
    m0 = s0 / n0; m1 = s1 / n1
    v0 = (s20 - n0 * m0 * m0) / (n0 - 1)
    v1 = (s21 - n1 * m1 * m1) / (n1 - 1)
    se2 = v0 / n0 + v1 / n1
    tstat = (m1 - m0) / np.sqrt(se2) if se2 > 0 else float("nan")
    try:
        from scipy.stats import t as _t
        df = (se2 ** 2) / ((v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1)) if se2 > 0 else 1.0
        pval = 2.0 * _t.sf(abs(tstat), df) if np.isfinite(tstat) else float("nan")
    except Exception:
        pval = float("nan")
    return {"statistic": float(tstat), "pvalue": float(pval),
            "diff": float(m1 - m0), "n0": int(n0), "n1": int(n1)}


def _cuped_from_aggregates(agg: dict) -> dict:
    """CUPED theta + adjusted effect from per-arm 6-tuples."""
    a = agg.get(0); b = agg.get(1)
    if a is None or b is None or a[0] < 2 or b[0] < 2:
        return {"theta": float("nan"), "raw_effect": float("nan"),
                "adjusted_effect": float("nan"),
                "n0": int(a[0] if a is not None else 0),
                "n1": int(b[0] if b is not None else 0)}
    n0, sy0, sy20, sx0, sxy0, sx20 = a
    n1, sy1, sy21, sx1, sxy1, sx21 = b
    # Pooled mean(x), var(x), cov(x,y)
    n = n0 + n1
    mx = (sx0 + sx1) / n
    my = (sy0 + sy1) / n
    ex2 = (sx20 + sx21) / n
    exy = (sxy0 + sxy1) / n
    var_x = ex2 - mx * mx
    cov_xy = exy - mx * my
    theta = float(cov_xy / var_x) if var_x > 0 else 0.0
    # adjusted y = y - theta*(x - mx); per-arm mean(y_adj) = mean(y) - theta*(mean_x_arm - mx)
    raw = (sy1 / n1) - (sy0 / n0)
    adj = ((sy1 / n1) - theta * ((sx1 / n1) - mx)) -           ((sy0 / n0) - theta * ((sx0 / n0) - mx))
    return {"theta": theta, "raw_effect": float(raw),
            "adjusted_effect": float(adj), "n0": int(n0), "n1": int(n1)}


def _read_rank_stream(path, value_col: str):
    """Yield (y_chunk, ) from sorted IPC file; sorted by value_col."""
    for b in read_sorted(path, columns=[value_col]):
        yield b.column(value_col).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)


def _mann_whitney_ooc(df, outcome_col: str, treatment_col: str,
                     mg: MemoryGuard, batch_size: Optional[int]) -> dict:
    """Rank-sum U via external_sort (tie-aware), streaming.

    Sorts (value, arm) by value, then walks sorted stream:
    ties (equal value) processed as a block; sum of ranks 1..k via
    arithmetic mean — this is tie-corrected U statistic.
    """
    # Materialise (y, t) into (value, arm) sorted file.
    def _emit():
        it = df._iter_filtered_batches(batch_size=batch_size)
        for b in it:
            y = b.column(outcome_col).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            t = b.column(treatment_col).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            mask = ~(np.isnan(y) | np.isnan(t))
            if not np.any(mask):
                continue
            yield pa.RecordBatch.from_arrays(
                [pa.array(y[mask], type=pa.float64()),
                 pa.array(t[mask], type=pa.float64())],
                names=["y", "t"],
            )

    with SpillDir(guard=mg, name="ab_mw") as sdir:
        path = external_sort(
            _emit(), key_columns=["y"], value_columns=["t"], 
            memory_guard=mg, spill_dir=sdir.path("sort"),
        )
        # Streaming rank sum with tie blocking.
        r0 = 0.0; n0 = 0
        r1 = 0.0; n1 = 0
        rank_pos = 1.0
        cur_val = None
        block_arms: list[float] = []
        def _flush_block(val, arms, rank_start):
            k = len(arms)
            avg_rank = rank_start + (k - 1) / 2.0
            n0 = sum(1 for a in arms if a == 0.0)
            n1 = sum(1 for a in arms if a == 1.0)
            return (avg_rank * n0, n0, avg_rank * n1, n1, rank_start + k)
        for b in read_sorted(path, columns=["y", "t"]):
            ys = b.column("y").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            ts = b.column("t").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            for v, a in zip(ys, ts):
                if cur_val is None or v != cur_val:
                    if cur_val is not None:
                        d0, c0, d1, c1, next_pos = _flush_block(cur_val, block_arms, rank_pos)
                        r0 += d0; n0 += c0; r1 += d1; n1 += c1
                        rank_pos = next_pos
                    cur_val = v; block_arms = [a]
                else:
                    block_arms.append(a)
        if cur_val is not None:
            d0, c0, d1, c1, _ = _flush_block(cur_val, block_arms, rank_pos)
            r0 += d0; n0 += c0; r1 += d1; n1 += c1
    if n0 < 2 or n1 < 2:
        return {"statistic": float("nan"), "pvalue": float("nan"),
                "n0": int(n0), "n1": int(n1)}
    U1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n0 * n1 / 2.0
    sigma2 = n0 * n1 * (n0 + n1 + 1) / 12.0
    z = (U1 - mu) / np.sqrt(sigma2) if sigma2 > 0 else float("nan")
    try:
        from scipy.stats import norm as _norm
        p = 2.0 * _norm.sf(abs(z)) if np.isfinite(z) else float("nan")
    except Exception:
        p = float("nan")
    return {"statistic": float(min(U1, n0 * n1 - U1)), "pvalue": float(p),
            "n0": int(n0), "n1": int(n1)}


def estimate_ab_test_cost(
    df, outcome_col: str, treatment_col: str, *,
    covariate_col: Optional[str] = None, method: str = "t",
    memory_guard: Optional[MemoryGuard] = None,
) -> "CostEstimate":
    """RAM+time estimate for fit_ab_test (AP22, v0.9.119)."""
    from cadw.core.models import DataChars, CostEstimate
    mg = memory_guard or df._memory_guard
    n_rows = int(getattr(df._dataset, 'total_rows', 0)) if hasattr(df._dataset, "total_rows")         else int(df.shape[0]) if hasattr(df, "shape") else 0
    # Row size from first batch or schema.
    row_size = 16
    try:
        b = next(iter(df._iter_filtered_batches(batch_size=1)))
        row_size = mg.estimate_row_size_from_batch(b)
    except Exception:
        pass
    schema = df._dataset.schema if hasattr(df._dataset, "schema") else pa.schema([])
    dc = DataChars(schema=schema, n_rows=n_rows, row_size_bytes=row_size,
                   k_est=None, avg_key_size_bytes=None, size_of_key_entry_bytes=None)
    # state estimate
    de = 116
    try:
        from cadw.ml._memory_estimate import _load_constants
        de = int(_load_constants().get("dict_entry_bytes", 116))
    except Exception:
        pass
    if method == "mw":
        # SortMemoryModel peak is O(budget), not O(n): the sort buffer
        # is bounded by ~0.2*budget (external_sort phase-1 limit).
        # Rank-scan state adds 296 B.
        sort_dc = DataChars(
            schema=schema, n_rows=n_rows, row_size_bytes=row_size,
            k_est=2, avg_key_size_bytes=row_size,
            size_of_key_entry_bytes=row_size * 2,
        )
        try:
            sort_ce = mg.estimate_cost("sort", sort_dc)
            est_ram = int(sort_ce.est_ram_bytes) + 296
            est_time = float(sort_ce.est_time_seconds)
            if not sort_ce.fits_ram:
                return CostEstimate(
                    op_name="ab_test:mw",
                    est_ram_bytes=est_ram,
                    fits_ram=False,
                    est_ops=n_rows,
                    est_time_seconds=0.0,
                    time_class="unknown",
                    reason="sort buffer does not fit at this budget",
                )
        except Exception as exc:
            return CostEstimate(
                op_name="ab_test:mw",
                est_ram_bytes=0, fits_ram=False, est_ops=n_rows,
                est_time_seconds=0.0, time_class="unknown",
                reason=f"sort model unavailable: {exc}",
            )
        return CostEstimate(
            op_name="ab_test:mw",
            est_ram_bytes=int(est_ram),
            fits_ram=int(est_ram) <= int(mg.available_bytes() * 0.7),
            est_ops=n_rows,
            est_time_seconds=est_time,
            time_class=mg._classify_time(est_time) if est_time > 0 else "unknown",
        )

    n_f64 = 3 if method == "t" else 6
    state_bytes = 2 * (n_f64 * 8 + 6 * de) + 256
    op_name = "reduce"
    # Batch-memory factor: t=4x (y+t+mask+select), cuped=5x (y+t+x+mask+select).
    batch_factor = 4 if method == "t" else 5
    est_ram = state_bytes + batch_factor * dc.n_rows * 8
    rates = mg._load_throughput_rates()
    rate = (rates.get(op_name) or {}).get("rate_per_sec", 0)
    est_time = float(n_rows) / float(rate) if rate else 0.0
    return CostEstimate(
        op_name=f"ab_test:{method}",
        est_ram_bytes=int(est_ram),
        fits_ram=int(est_ram) <= int(mg.available_bytes() * 0.7),
        est_ops=n_rows,
        est_time_seconds=est_time,
        time_class=mg._classify_time(est_time) if est_time > 0 else "unknown",
    )


def fit_ab_test(df, outcome_col, treatment_col, *, covariate_col=None,
                method="t", memory_guard: Optional[MemoryGuard] = None,
                batch_size: Optional[int] = None,
                confirm_slow: bool = False, strict_slow: bool = False):
    """Fit an A/B comparison (v0.9.119, OOC).

    method='t':    streaming Welch t-test via Reduce.
    method='cuped':streaming CUPED via Reduce.
    method='mw':   rank-sum U via external_sort + streaming rank.

    Result available in model.result_. state_nbytes() is formula-based.
    """
    import warnings as _w
    if outcome_col not in df._columns:
        raise KeyError(f"fit_ab_test: column {outcome_col!r} not in {df._columns}")
    if treatment_col not in df._columns:
        raise KeyError(f"fit_ab_test: column {treatment_col!r} not in {df._columns}")
    if covariate_col is not None and covariate_col not in df._columns:
        raise KeyError(f"fit_ab_test: column {covariate_col!r} not in {df._columns}")
    if method not in ("t", "mw", "cuped"):
        raise ValueError(f"fit_ab_test: method must be 't', 'mw' or 'cuped'; got {method!r}")
    if method == "cuped" and covariate_col is None:
        raise ValueError("fit_ab_test: method='cuped' requires covariate_col")
    mg = memory_guard or df._memory_guard

    # AP22 pre-flight: cost estimate.
    cost = estimate_ab_test_cost(
        df, outcome_col, treatment_col,
        covariate_col=covariate_col, method=method, memory_guard=mg)
    if cost.time_class in ("slow", "very_slow"):
        if strict_slow and not confirm_slow:
            raise SlowOperationRequiresConfirm(
                f"fit_ab_test: est {cost.est_time_seconds:.1f}s "
                f"(time_class={cost.time_class}); pass confirm_slow=True")
        if not confirm_slow:
            _w.warn(
                f"fit_ab_test: est {cost.est_time_seconds:.1f}s "
                f"(time_class={cost.time_class})",
                SlowOperationWarning, stacklevel=2)
    if not cost.fits_ram:
        raise MemoryLimitExceeded(
            f"fit_ab_test: est_ram={cost.est_ram_bytes} B exceeds budget "
            f"{int(mg.available_bytes() * 0.7)} B")

    # v0.9.122: build op first, then batch_size from op.batch_memory_factor().
    # Prior (v0.9.119): expansion_factor=5.0 + state_bytes=1696 for ALL
    # methods. t's op factor is 4.0 (y+t+mask+select copies), cuped 5.0
    # (y+t+x+mask+select), mw uses sort path (not Reduce; 5.0 conservative).
    # state_bytes differs: t=1696, cuped=1744, mw=296 (rank-scan).
    _DE_PER_ARM = 116  # dict_entry_bytes (memory_constants.yaml)
    if method == "t":
        op = _ArmStatsOp(outcome_col, treatment_col)
        state_bytes = 256 + 2 * (3 * 8 + 6 * _DE_PER_ARM)   # 1696
        batch_factor = op.batch_memory_factor()              # 4.0
    elif method == "cuped":
        op = _CupedStatsOp(outcome_col, treatment_col, covariate_col)
        state_bytes = 256 + 2 * (6 * 8 + 6 * _DE_PER_ARM)   # 1744
        batch_factor = op.batch_memory_factor()              # 5.0
    else:  # mw
        op = None
        state_bytes = 5 * 8 + 256                            # 296 (rank-scan)
        batch_factor = 5.0  # sort path; SortMemoryModel v0.9.123

    # Derive batch_size from formula if not given.
    if batch_size is None:
        try:
            sample = next(iter(df._iter_filtered_batches(batch_size=8)))
            batch_size = mg.get_batch_rows(
                sample_batch=sample, state_bytes=state_bytes,
                expansion_factor=batch_factor)
        except StopIteration:
            batch_size = 1000

    it = df._iter_filtered_batches(batch_size=batch_size)
    if method == "t":
        agg = _reduce(it, op, mg, log=False)
        result = _welch_from_aggregates(agg)
    elif method == "cuped":
        agg = _reduce(it, op, mg, log=False)
        result = _cuped_from_aggregates(agg)
    else:  # mw
        result = _mann_whitney_ooc(df, outcome_col, treatment_col, mg, batch_size)

    m = DriftABModel("ab", {
        "outcome_col": outcome_col,
        "treatment_col": treatment_col,
        "covariate_col": covariate_col,
        "method": method,
        "batch_size": batch_size,
    }, result=result)
    log_fit("ab_test", state_bytes=m.state_nbytes())
    return m
