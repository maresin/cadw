"""AnomalyPipeline: detrend / deseasonalize before anomaly detection.

Backlog trigger (v0.9.77b): 11 of 15 D3.x anomaly detectors assume
stationary baseline; on trended/seasonal series they produce false
positives (trend/seasonal points look like outliers). Solution:
preprocess y to remove trend/seasonality, run detector on residual,
map scores back to original timestamps.

Two modes (caller chooses explicitly; detectors are functions, not
classes, so capability flags are not part of the model API):

  mode='contextual' (default when STL/detrend available):
    - if seasonal_period >= 2: STL decompose (trend + seasonal +
      remainder), run detector on remainder (stationary), scores
      mapped back 1:1 (STL preserves length).
    - elif detrend=True: Differencer(lags=1), run detector on diffs,
      scores mapped back with first 1 timestamps excluded (unknown).
    Suited for CONTEXTUAL anomalies: values abnormal in context of
    normal trend+seasonal. Examples: spike above normal seasonal peak.

  mode='structural':
    - detect on raw series, no transform.
    Suited for STRUCTURAL anomalies: change in trend/level/seasonal
    itself. Native changepoint detectors (PELT, BOCPD, BinSeg, CUSUM,
    T-squared+Q) handle this correctly. Other detectors will see the
    change-point as outliers among normal trend points.

  mode='auto':
    - contextual if seasonal_period >= 2 or detrend=True,
      else structural.
    Safe default: on stationary data (no seasonal_period, no detrend)
    stays structural; on seasonal/trended data does the right thing.

Detector API:
  detector_fn must be callable with signature:
      detector_fn(series: np.ndarray, **detector_kwargs) -> result
  where result has attributes `decision_scores: np.ndarray` (length =
  len(series)) and `labels: np.ndarray` (0/1). All CADW tier1 anomaly
  detectors fit this contract (fit_dwt_mlead, fit_pelt, fit_bocpd,
  fit_hampel, fit_zms, fit_cusum, fit_stray, fit_s2g, fit_eif,
  fit_tspq, fit_sub_pca, fit_autoencoder, fit_kmeans_ad,
  fit_patch_memory_ad, fit_binseg, fit_kshape).

Result:
  AnomalyPipelineResult — same decision_scores / labels (length n)
  as original time series, plus metadata: mode, transforms_applied,
  base_result (for debugging).

References:
  - Backlog item 3 (v0.9.77): 'Anomaly stationarity assumption'.
  - Ren et al. (2019) "Time-Series Anomaly Detection Service at
    Microsoft" (SPECTRAL RESIDUAL, detrend inside).
  - Boniol & Palpanas (2020) S2G uses convolution (implicit
    normalization); equivalent to detrending for shape anomalies.

Pure numpy; no sklearn/statsmodels/scipy/torch (AST-guard). Lazy
import of transforms (circular-dep avoidance: transforms.py imports
ForecasterBase from models.py, not from ml/).

v0.9.87-ts-transforms-guard: fit_predict gains memory_guard kw.
Passes to stl_decompose (guarded ~8n alloc); differencing branch
guards np.diff(n-1). Structural mode has no additional n-sized
alloc. Known limitation: detector_fn may internally allocate O(n)
without guard — user should pass memory_guard to detector if it
accepts one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from cadw.core.memory_guard import MemoryLimitExceeded


@dataclass
class AnomalyPipelineResult:
    """Pipeline result.

    decision_scores : np.ndarray, shape (n,)
        Anomaly scores aligned to original series length (same as
        input y passed to fit_predict). If differencing was applied,
        the first `d` timestamps are 0 (unknown).
    labels : np.ndarray, shape (n,), int
        Binary labels (0/1), same alignment.
    mode : str
        'contextual' | 'structural' (resolved from user's spec).
    transforms_applied : list[str]
        Human-readable list of transformers applied, in order.
    base_result : object
        Original detector result (for debugging / score access).
    threshold : float | None
        Pipeline-level threshold (from base_result if present, else
        None; pipeline does not re-threshold).
    """
    decision_scores: np.ndarray
    labels: np.ndarray
    mode: str
    transforms_applied: list
    base_result: Any = None
    threshold: float | None = None


# ── Helpers ───────────────────────────────────────────────────────

def _extract_scores_and_labels(result):
    """Extract decision_scores and labels from detector result.

    Raises if result lacks the required attributes — the pipeline
    contract is strict (all tier1 detectors comply).
    """
    scores = getattr(result, "decision_scores", None)
    labels = getattr(result, "labels", None)
    if scores is None or labels is None:
        raise TypeError(
            f"AnomalyPipeline: detector result "
            f"{type(result).__name__} lacks decision_scores/labels "
            f"attributes (required by pipeline contract)."
        )
    return (np.asarray(scores, dtype=np.float64),
            np.asarray(labels, dtype=np.int64))


# ── AnomalyPipeline ──────────────────────────────────────────────

class AnomalyPipeline:
    """Contextual / structural anomaly detection pipeline.

    Parameters
    ----------
    detector_fn : callable
        Function with signature `detector_fn(series, **detector_kwargs)
        -> result`. Result must expose `.decision_scores` and `.labels`.
    mode : str, default='auto'
        'contextual' | 'structural' | 'auto'. See module docstring.
    seasonal_period : int | None, default=None
        If >= 2, STL decomposition applied (contextual mode).
    detrend : bool, default=False
        If True (and no seasonal_period), Differencer(lags=1) applied
        (contextual mode; length -1).
    **detector_kwargs
        Passed to detector_fn.
    """

    def __init__(self, detector_fn: Callable, *,
                 mode: str = "auto",
                 seasonal_period: int | None = None,
                 detrend: bool = False,
                 **detector_kwargs):
        if not callable(detector_fn):
            raise ValueError(
                "AnomalyPipeline: detector_fn must be callable")
        if mode not in ("auto", "contextual", "structural"):
            raise ValueError(
                f"AnomalyPipeline: mode={mode!r} not in "
                f"'auto' | 'contextual' | 'structural'")
        if seasonal_period is not None:
            if not isinstance(seasonal_period, int) or seasonal_period < 2:
                raise ValueError(
                    f"AnomalyPipeline: seasonal_period must be >= 2, "
                    f"got {seasonal_period!r}")
        self.detector_fn = detector_fn
        self.mode = mode
        self.seasonal_period = seasonal_period
        self.detrend = bool(detrend)
        self.detector_kwargs = dict(detector_kwargs)
        self._resolved_mode_ = None
        self._transforms_ = None

    def _resolve_mode(self) -> tuple[str, list]:
        """Determine (mode, transforms_list) from user's spec.

        - If mode='structural': no transforms.
        - If mode='contextual': use STL (if seasonal_period>=2) or
          Differencer (if detrend). If neither available, fall back
          to structural (with a silent no-op; user mis-specified).
        - If mode='auto': contextual if any transform available, else
          structural.
        """
        has_stl = (self.seasonal_period is not None
                   and self.seasonal_period >= 2)
        has_diff = self.detrend and not has_stl

        if self.mode == "structural":
            return "structural", []
        if self.mode == "contextual":
            if has_stl:
                return "contextual", ["stl"]
            if has_diff:
                return "contextual", ["diff_d1"]
            # Contextual requested but no transform available — fallback
            # to structural (honest: detector sees raw series).
            return "structural", []
        # mode='auto'
        if has_stl:
            return "contextual", ["stl"]
        if has_diff:
            return "contextual", ["diff_d1"]
        return "structural", []

    def fit_predict(self, series: np.ndarray, *,
                    memory_guard=None) -> AnomalyPipelineResult:
        y = np.asarray(series, dtype=np.float64).ravel()
        n = y.size
        if n < 4:
            raise ValueError(
                "AnomalyPipeline: need at least 4 observations")
        if np.any(~np.isfinite(y)):
            raise ValueError(
                "AnomalyPipeline: series contains non-finite values")

        mode, transforms = self._resolve_mode()
        self._resolved_mode_ = mode
        self._transforms_ = transforms

        # Structural: detect on raw.
        if mode == "structural":
            result = self.detector_fn(y, **self.detector_kwargs)
            scores, labels = _extract_scores_and_labels(result)
            if scores.size != n:
                raise ValueError(
                    f"AnomalyPipeline: detector returned "
                    f"decision_scores of size {scores.size} != n={n} "
                    f"in structural mode "
                    f"({type(result).__name__})")
            return AnomalyPipelineResult(
                decision_scores=scores,
                labels=labels,
                mode=mode,
                transforms_applied=transforms,
                base_result=result,
                threshold=getattr(result, "threshold", None),
            )

        # Contextual with STL: length preserved.
        if "stl" in transforms:
            from cadw.ts._decomposition import stl_decompose
            decomp = stl_decompose(
                y, period=self.seasonal_period,
                memory_guard=memory_guard)
            residual = decomp.remainder
            result = self.detector_fn(residual, **self.detector_kwargs)
            scores, labels = _extract_scores_and_labels(result)
            if scores.size != n:
                raise ValueError(
                    f"AnomalyPipeline: STL residual size n={n}, "
                    f"detector returned {scores.size}")
            return AnomalyPipelineResult(
                decision_scores=scores,
                labels=labels,
                mode=mode,
                transforms_applied=transforms,
                base_result=result,
                threshold=getattr(result, "threshold", None),
            )

        # Contextual with differencing: length loss d=1.
        if "diff_d1" in transforms:
            # v0.9.87-ts-transforms-guard: n-1 float64 diff
            if memory_guard is not None:
                est = (n - 1) * 8 + 1024
                try:
                    memory_guard.check_state(
                        est, name="anomaly_pipeline.diff_state")
                except MemoryLimitExceeded:
                    raise MemoryLimitExceeded(
                        f"AnomalyPipeline: differencing needs "
                        f"(n-1)*8 = {(n - 1) * 8} B; exceeds "
                        f"state_limit. n={n}. §13.4 v0.9.87")
            diffed = np.diff(y)
            result = self.detector_fn(diffed, **self.detector_kwargs)
            scores_d, labels_d = _extract_scores_and_labels(result)
            if scores_d.size != n - 1:
                raise ValueError(
                    f"AnomalyPipeline: differenced size n-1={n - 1}, "
                    f"detector returned {scores_d.size}")
            scores = np.zeros(n, dtype=np.float64)
            labels = np.zeros(n, dtype=np.int64)
            scores[1:] = scores_d
            labels[1:] = labels_d
            return AnomalyPipelineResult(
                decision_scores=scores,
                labels=labels,
                mode=mode,
                transforms_applied=transforms,
                base_result=result,
                threshold=getattr(result, "threshold", None),
            )

        # Should not reach here; _resolve_mode guarantees a branch.
        raise RuntimeError(
            f"AnomalyPipeline: unresolved mode={mode}, "
            f"transforms={transforms}")

    def get_params(self) -> dict:
        return {
            "detector_fn": getattr(self.detector_fn, "__name__",
                                   type(self.detector_fn).__name__),
            "mode_spec": self.mode,
            "resolved_mode": self._resolved_mode_,
            "seasonal_period": self.seasonal_period,
            "detrend": self.detrend,
            "transforms_applied": (list(self._transforms_)
                                   if self._transforms_ is not None
                                   else None),
            "detector_kwargs": dict(self.detector_kwargs),
        }

