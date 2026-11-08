# CADW Temporal Hierarchical Forecasting (v0.9.24).
#
# THF (Athanasopoulos et al. 2017): fit forecasts at multiple temporal
# aggregation levels of the same series, then RECONCILE them so the
# aggregated forecasts sum consistently to the base forecasts.
#
# Temporal hierarchy:
#   levels = [m_0=1, m_1, m_2, ..., m_K]  (strictly increasing)
#   m_k = number of base periods aggregated into one observation at
#         level k. m_0=1 is the base level.
#   base series y[0..T) is aggregated to y_k at each level:
#     y_k[j] = sum(y[j*m_k : (j+1)*m_k])  (agg='sum' default)
#            = mean(y[j*m_k : (j+1)*m_k]) (agg='mean')
#
# Forecast horizon H is in BASE units. At level k we forecast
# n_k = ceil(H / m_k) observations (the last block at level k may
# cover fewer than m_k base steps).
#
# Summation matrix S maps reconciled BASE forecast (length H) to the
# stacked forecast across all levels (length sum_k n_k):
#   S is (sum_k n_k) x H.
#   Level-0 block (m_0=1, n_0=H): identity S[0:H, 0:H] = I.
#   Level-k block (k > 0): row j has 1s in columns
#     [j*m_k, min((j+1)*m_k, H)).
#
# Reconciliation methods:
#   'bottom_up':              use level-0 forecast directly (base).
#   'top_down_proportional':  use coarsest level (m_K) forecast, distribute
#                             by historical phase proportions
#                             probs[p] = mean(y[i*m_K + p] / y_K[i])
#                             over training i with y_K[i] != 0 (skip zeros,
#                             no divide-by-zero). Requires H % m_K == 0.
#   'optimal_ols':            W = I; y_bottom = pinv(S) @ y_all.
#   'optimal_wls':            Proper MinT-WLS with diagonal W where
#                             W[i] = residual variance of level k (i
#                             belongs to level k). Residual variance is
#                             estimated from each level's in-sample
#                             predictions (predict_in_sample) over rows
#                             where the prediction is finite. A small
#                             floor (1e-8) prevents zero weight on
#                             perfect in-sample fits. If a level has
#                             zero finite residuals (e.g. all-NaN
#                             in-sample predictions or single-row
#                             level), we fall back to OLS for the
#                             whole fit and warn once.
#
# pinv rationale:
#   S can have more rows than columns (sum_k n_k >= H), and the normal
#   equations matrix S^T S may be near-singular for highly redundant
#   aggregations. np.linalg.pinv uses SVD and is rank-safe; np.linalg.inv
#   can raise LinAlgError. Documented choice.
#
# RAM: O(sum_k n_k + (sum_k n_k)^2) for S and reconciliation. 
# Independent of N (training series length) at predict time. Forecaster
# fit goes through the existing MemoryGuard-aware machinery.
from __future__ import annotations

import numpy as np
import pyarrow as pa

from cadw import CadwDataFrame
from .models import _extract_series


_RECON_METHODS = (
    "bottom_up", "top_down_proportional", "optimal_ols", "optimal_wls",
)


def _aggregate(y, m, agg):
    n = len(y) // m
    if n == 0:
        return np.empty(0, dtype=np.float64)
    red = np.add.reduceat(y[:n * m], np.arange(0, n * m, m))
    if agg == "mean":
        return red / m
    return red


class TemporalHierarchy:
    """Temporal Hierarchical Forecasting (v0.9.24).

    Parameters:
      levels: strictly increasing list starting with 1; e.g. [1, 7, 30].
      forecaster_factory: zero-arg callable returning a fresh
        ForecasterBase instance. Called ONCE per level during fit.
      reconciliation: 'bottom_up' | 'top_down_proportional' |
        'optimal_ols' | 'optimal_wls'.
      agg: 'sum' (default) or 'mean'.

    API:
      fit(df, target_col, *, time_col=None, group_col=None)
      predict(steps)        -> reconciled base-level forecast (len=steps)
      predict_level(steps, level) -> reconciled forecast at level k
        (len=ceil(steps/levels[k]))
    """

    def __init__(
        self, levels, forecaster_factory,
        reconciliation="optimal_ols", agg="sum",
    ):
        if not levels or int(levels[0]) != 1:
            raise ValueError(
                f"TemporalHierarchy: levels must start with 1 "
                f"(base level); got {levels}"
            )
        lv = [int(m) for m in levels]
        for i in range(1, len(lv)):
            if lv[i] <= lv[i - 1]:
                raise ValueError(
                    f"TemporalHierarchy: levels must be strictly "
                    f"increasing; got {lv}"
                )
        if reconciliation not in _RECON_METHODS:
            raise ValueError(
                f"TemporalHierarchy: reconciliation must be one of "
                f"{_RECON_METHODS}; got {reconciliation!r}"
            )
        if agg not in ("sum", "mean"):
            raise ValueError(
                f"TemporalHierarchy: agg must be 'sum' or 'mean'; "
                f"got {agg!r}"
            )
        if not callable(forecaster_factory):
            raise TypeError(
                "TemporalHierarchy: forecaster_factory must be callable"
            )
        self.levels = lv
        self.forecaster_factory = forecaster_factory
        self.reconciliation = reconciliation
        self.agg = agg
        self._is_fitted = False

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        # v0.9.90-S5b: store guard for _fit_one (base path only).
        self._mg = getattr(df, "_memory_guard", None)
        # v0.9.129: reset cumulative estimate for multi-group check.
        self._cumulative_est = 0
        if group_col is not None:
            if group_col not in df._columns:
                raise KeyError(
                    f"TemporalHierarchy.fit: group_col {group_col!r} "
                    f"not in df"
                )
            seen = []
            for batch in df._iter_arrow_batches(batch_size=10_000):
                for v in batch.column(group_col).to_pylist():
                    if v not in seen:
                        seen.append(v)
            self.group_values_ = seen
            self._group_models = {}
            self._group_y_base = {}
            self._group_y_level = {}
            self._group_td_probs = {}
            self._group_col = group_col
            for gv in seen:
                y = _extract_series(
                    df, target_col, time_col=time_col,
                    group_col=group_col, group_value=gv,
                )
                y = y[~np.isnan(y)]
                self._fit_one(y, gv)
            self._is_multi = True
            self._is_fitted = True
            return self
        self._is_multi = False
        self._group_col = None
        y = _extract_series(df, target_col, time_col=time_col)
        y = y[~np.isnan(y)]
        self._fit_one(y, group=None)
        self._is_fitted = True
        return self

    def _fit_one(self, y, group):
        if len(y) < max(self.levels) * 2:
            raise ValueError(
                f"TemporalHierarchy: need at least "
                f"{max(self.levels) * 2} observations; got {len(y)}"
            )
        # v0.9.129: cumulative check across groups (single + multi).
        # Per-group peak = n*(1 + sum(1/m))*8. Cumulative bounded
        # by n_total <= 1e6.
        if getattr(self, "_mg", None) is not None:
            est = int(len(y) * (1.0 + sum(1.0 / m for m in self.levels))) * 8
            cumulative = getattr(self, "_cumulative_est", 0)
            self._mg.check_state(cumulative + est, name="thf.fit_one")
            self._cumulative_est = cumulative + est
        y_per_level = [_aggregate(y, m, self.agg) for m in self.levels]
        models = []
        resid_var = []
        for y_k in y_per_level:
            f = self.forecaster_factory()
            t = pa.table({"__y__": pa.array(y_k, type=pa.float64())})
            f.fit(CadwDataFrame.from_table(t), "__y__")
            models.append(f)
            # In-sample residual variance for MinT-WLS only.
            if self.reconciliation != "optimal_wls":
                continue
            try:
                preds = np.asarray(
                    f.predict_in_sample(), dtype=np.float64,
                )
            except Exception:
                preds = np.full(len(y_k), np.nan, dtype=np.float64)
            if len(preds) != len(y_k):
                preds = np.full(len(y_k), np.nan, dtype=np.float64)
            mask = np.isfinite(preds) & np.isfinite(y_k)
            if mask.sum() < 2:
                resid_var.append(None)
            else:
                r = y_k[mask] - preds[mask]
                v = float(np.var(r, ddof=1)) if len(r) > 1 else 0.0
                # Floor: 1e-8 to avoid zero-weight on perfect fits.
                v = max(v, 1e-8)
                resid_var.append(v)
        if any(v is None for v in resid_var):
            resid_var = None  # fallback OLS for the whole fit
            import warnings
            warnings.warn(
                "TemporalHierarchy: insufficient residuals for "
                "optimal_wls; falling back to optimal_ols",
                stacklevel=2,
            )
        td_probs = None
        if self.reconciliation == "top_down_proportional":
            m_top = self.levels[-1]
            y_top = y_per_level[-1]
            n = len(y_top)
            probs = np.zeros(m_top, dtype=np.float64)
            counts = np.zeros(m_top, dtype=np.float64)
            for i in range(n):
                if y_top[i] == 0:
                    continue
                block = y[i * m_top:(i + 1) * m_top]
                for p in range(m_top):
                    probs[p] += block[p] / y_top[i]
                    counts[p] += 1.0
            td_probs = np.where(
                counts > 0, probs / np.maximum(counts, 1.0),
                1.0 / m_top,
            )
        if group is not None:
            self._group_models[group] = models
            self._group_y_base[group] = y.copy()
            self._group_y_level[group] = y_per_level
            self._group_td_probs[group] = td_probs
            if not hasattr(self, "_group_resid_var"):
                self._group_resid_var = {}
            self._group_resid_var[group] = resid_var
        else:
            self.y_base_ = y.copy()
            self.y_per_level_ = y_per_level
            self.models_ = models
            self.top_down_probs_ = td_probs
            self._resid_var_ = resid_var

    def state_nbytes(self) -> int:
        """Persistent state bytes after fit.

        Single-group: y_base_(n*f64) + sum(y_per_level)(n*f64/m) +
        sum(forecaster.state_nbytes()) + td_probs + resid_var + 256.
        O(n_total) intrinsic — single-series architecture (like
        VAR/VECM). bounded_by_domain: n_total <= 1e6.
        """
        if not self._is_fitted:
            return 0
        total = 256

        def _model_state(m):
            return (m.state_nbytes() if hasattr(m, "state_nbytes")
                    else 4096)

        def _level_state(y_l, models, td, rv):
            s = sum(yk.nbytes for yk in y_l)
            s += sum(_model_state(m) for m in models)
            if td is not None:
                s += td.nbytes
            if rv is not None:
                s += len(rv) * 8
            return int(s)

        if self._is_multi:
            rv_map = getattr(self, "_group_resid_var", {})
            for g in self._group_y_base:
                total += self._group_y_base[g].nbytes
                total += _level_state(
                    self._group_y_level[g],
                    self._group_models[g],
                    self._group_td_probs.get(g),
                    rv_map.get(g),
                )
        else:
            total += self.y_base_.nbytes
            total += _level_state(
                self.y_per_level_, self.models_,
                self.top_down_probs_,
                getattr(self, "_resid_var_", None),
            )
        return int(total)

    def _build_S(self, steps):
        n_ks = [(steps + m - 1) // m for m in self.levels]
        total = sum(n_ks)
        S = np.zeros((total, steps), dtype=np.float64)
        offset = 0
        for m, n_k in zip(self.levels, n_ks):
            for j in range(n_k):
                c0 = j * m
                c1 = min(c0 + m, steps)
                if c0 < steps:
                    S[offset + j, c0:c1] = 1.0
            offset += n_k
        return S, n_ks

    def _reconcile(self, y_all, steps, group=None):
        """Reconcile a stacked forecast vector to the base level.

        Exposed for tests and for advanced users; predict() calls this.
        """
        S, n_ks = self._build_S(steps)
        if self.reconciliation == "bottom_up":
            return y_all[:steps].copy()
        if self.reconciliation == "top_down_proportional":
            m_top = self.levels[-1]
            if steps % m_top != 0:
                raise ValueError(
                    f"top_down_proportional requires steps % {m_top} "
                    f"== 0; got {steps}"
                )
            _, probs = self._models_for(group)
            y_top = y_all[-n_ks[-1]:]
            out = np.zeros(steps, dtype=np.float64)
            for j in range(len(y_top)):
                for p in range(m_top):
                    out[j * m_top + p] = y_top[j] * probs[p]
            return out
        if self.reconciliation == "optimal_ols":
            return np.linalg.pinv(S) @ y_all
        if self.reconciliation == "optimal_wls":
            # MinT-WLS: W[i] = residual variance of the level that
            # position i belongs to. From in-sample residuals; fall
            # back to OLS if the residual-variance weights are not
            # available (see _fit_one).
            rv = (
                self._group_resid_var[group]
                if group is not None else self._resid_var_
            )
            if rv is None:
                # Fallback: no residuals -> OLS projection.
                return np.linalg.pinv(S) @ y_all
            weights = []
            for k in range(len(self.levels)):
                weights.extend([rv[k]] * n_ks[k])
            w = np.asarray(weights, dtype=np.float64)
            # Guard: any non-finite or <=0 weight -> fallback OLS.
            if not np.all(np.isfinite(w)) or np.any(w <= 0):
                return np.linalg.pinv(S) @ y_all
            W_inv_S = S / w[:, None]  # W^-1 @ S
            W_inv_y = y_all / w
            A = S.T @ W_inv_S        # S^T W^-1 S
            b = S.T @ W_inv_y        # S^T W^-1 y_all
            return np.linalg.pinv(A) @ b
        raise RuntimeError(
            f"TemporalHierarchy: unknown reconciliation "
            f"{self.reconciliation!r}"
        )

    def _models_for(self, group):
        if self._is_multi:
            if group is None:
                raise ValueError(
                    "TemporalHierarchy.predict: fitted with "
                    "group_col, predict requires group="
                )
            if group not in self._group_models:
                raise KeyError(
                    f"TemporalHierarchy.predict: unknown group {group!r}"
                )
            return (
                self._group_models[group],
                self._group_td_probs[group],
            )
        return self.models_, self.top_down_probs_

    def predict(self, steps, *, group=None):
        if not self._is_fitted:
            raise RuntimeError(
                "TemporalHierarchy.predict called before fit"
            )
        if not isinstance(steps, int):
            raise TypeError(
                f"TemporalHierarchy.predict: steps must be int; "
                f"got {type(steps).__name__}"
            )
        if steps < 1:
            raise ValueError(
                f"TemporalHierarchy.predict: steps must be >= 1; "
                f"got {steps}"
            )
        models, _ = self._models_for(group)
        n_ks = [(steps + m - 1) // m for m in self.levels]
        y_hat = []
        for k, n_k in enumerate(n_ks):
            fc = np.asarray(models[k].predict(n_k), dtype=np.float64)
            if len(fc) != n_k:
                fc = fc[:n_k]
            y_hat.append(fc)
        y_all = np.concatenate(y_hat)
        return self._reconcile(y_all, steps, group=group)

    def predict_level(self, steps, level, *, group=None):
        if not (0 <= level < len(self.levels)):
            raise ValueError(
                f"TemporalHierarchy.predict_level: level must be in "
                f"[0, {len(self.levels) - 1}]; got {level}"
            )
        y_bottom = self.predict(steps, group=group)
        m = self.levels[level]
        n_k = (steps + m - 1) // m
        out = np.zeros(n_k, dtype=np.float64)
        for j in range(n_k):
            c0 = j * m
            c1 = min(c0 + m, steps)
            out[j] = y_bottom[c0:c1].sum()
        return out


__all__ = ["TemporalHierarchy"]
