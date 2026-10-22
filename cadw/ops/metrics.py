"""Regression + log_loss metrics via ReduceOp (v0.9.2a).

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

import math
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.ops.reduce import ReduceOp


REGRESSION_METRICS = frozenset({"r2", "mse", "rmse", "mae"})
PROBA_METRICS = frozenset({"log_loss"})
CLASSIFICATION_METRICS = frozenset({"accuracy", "precision", "recall", "f1"})
AUC_METRICS = frozenset({"roc_auc"})
SUPPORTED_METRICS = (
    REGRESSION_METRICS | PROBA_METRICS
    | CLASSIFICATION_METRICS | AUC_METRICS
)
VALID_AVERAGES = frozenset({"binary", "macro", "micro", "weighted"})

# module-level sentinel (§13.4/148 lesson: never `obj is object()`)
_ROC_AUC_UNSET = object()


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
        y_proba_col: Optional[str] = None,       # binary backward compat
        y_proba_cols: Optional[Sequence[str]] = None,  # multiclass/binary
        metrics: Sequence[str] | str,
        classes: Optional[Sequence] = None,
        average: Optional[str] = None,
        pos_label=None,
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
        # v0.9.2c: roc_auc is multipass (Sort-based); handled by
        # CadwDataFrame.metrics() facade, not MetricsOp.
        if any(m in AUC_METRICS for m in self.metrics):
            raise ValueError(
                "MetricsOp: 'roc_auc' is computed separately by "
                "CadwDataFrame.metrics() via external_sort (multipass). "
                "Do not pass roc_auc directly to MetricsOp. "
                "See §13.4/166."
            )
        self.y_true_col = y_true_col
        self.y_pred_col = y_pred_col
        self.y_proba_col = y_proba_col  # legacy single; also unified below

        # v0.9.2c: unified y_proba_cols list for binary + multiclass.
        if y_proba_cols is not None:
            self.y_proba_cols = list(y_proba_cols)
        elif y_proba_col is not None:
            self.y_proba_cols = [y_proba_col]
        else:
            self.y_proba_cols = None
        self._needs_multiclass_proba = (
            self.y_proba_cols is not None
            and len(self.y_proba_cols) > 1
        )

        ms = set(self.metrics)
        self._needs_regression = bool(ms & REGRESSION_METRICS)
        self._needs_proba = bool(ms & PROBA_METRICS)
        self._needs_classification = bool(ms & CLASSIFICATION_METRICS)

        # §13.4/164: regression (continuous y_pred) and classification
        # (labels y_pred) can't share the same column.
        if self._needs_regression and self._needs_classification:
            raise ValueError(
                f"MetricsOp: regression ({sorted(ms & REGRESSION_METRICS)}) "
                f"and classification ({sorted(ms & CLASSIFICATION_METRICS)}) "
                f"cannot share y_pred column — one expects continuous "
                f"values, the other labels. Pass distinct calls."
            )
        # §13.4/168 (defensive): regression (continuous y_true) +
        # multiclass proba (y_true = class index 0..K-1) — contradictory
        # y_true semantics. Mutually exclusive.
        if self._needs_regression and self._needs_multiclass_proba:
            raise ValueError(
                f"MetricsOp: regression ({sorted(ms & REGRESSION_METRICS)}) "
                f"uses y_true as continuous target; multiclass y_proba "
                f"(len={len(self.y_proba_cols)}) uses y_true as class "
                f"index (0..K-1). Contradictory domain contracts — "
                f"cannot mix. Use separate calls."
            )

        if self._needs_regression and y_pred_col is None:
            raise ValueError(
                f"MetricsOp: y_pred_col required for "
                f"{sorted(ms & REGRESSION_METRICS)}"
            )
        # v0.9.2c-fix1 (§13.4/167): check unified y_proba_cols, NOT
        # legacy y_proba_col. When only y_proba_cols=[...] is passed,
        # y_proba_col remains None even though proba IS provided.
        # Message kept as "y_proba_col" for v0.9.2a test regex compat.
        if self._needs_proba and self.y_proba_cols is None:
            raise ValueError(
                f"MetricsOp: y_proba_col required for "
                f"{sorted(ms & PROBA_METRICS)}"
            )
        if self._needs_classification and y_pred_col is None:
            raise ValueError(
                f"MetricsOp: y_pred_col required for classification "
                f"{sorted(ms & CLASSIFICATION_METRICS)}"
            )

        # Classification setup
        self._classes: list = []
        self._cls_to_idx: dict = {}
        self._k: int = 0
        self._average_method: str = ""
        self._pos_label = None

        # §13.4/168: multiclass log_loss also needs _classes + _cls_to_idx
        # (y_true label → proba column index). Prior gate `if
        # self._needs_classification:` left them empty for pure
        # `metrics=["log_loss"]` → all labels treated as "unknown".
        _needs_class_setup = (
            self._needs_classification
            or (self._needs_proba and self._needs_multiclass_proba)
        )
        if _needs_class_setup:
            if classes is None:
                # K=2 default
                self._classes = [0, 1]
            else:
                self._classes = list(classes)
            self._k = len(self._classes)
            if self._k < 2:
                raise ValueError(
                    f"MetricsOp: classification requires at least 2 "
                    f"classes; got {self._k}"
                )
            # Duplicate check
            if len(set(map(repr, self._classes))) != self._k:
                raise ValueError(
                    f"MetricsOp: classes must be unique; got "
                    f"{self._classes}"
                )
            self._cls_to_idx = {c: i for i, c in enumerate(self._classes)}

            # Average method: explicit or auto
            if average is None:
                self._average_method = (
                    "binary" if self._k == 2 else "macro"
                )
            elif average not in VALID_AVERAGES:
                raise ValueError(
                    f"MetricsOp: average must be one of "
                    f"{sorted(VALID_AVERAGES)}; got {average!r}"
                )
            else:
                self._average_method = average
            if self._average_method == "binary" and self._k != 2:
                raise ValueError(
                    f"MetricsOp: average='binary' requires K=2; "
                    f"got K={self._k}. Use 'macro'/'micro'/'weighted'."
                )
            if self._average_method == "binary":
                # §13.4/164: pos_label=None default → resolve to
                # classes[-1] (matches sklearn's classes[1] for K=2
                # [0,1]; friendly for string classes).
                if pos_label is None:
                    self._pos_label = self._classes[-1]
                else:
                    if pos_label not in self._cls_to_idx:
                        raise ValueError(
                            f"MetricsOp: pos_label {pos_label!r} not "
                            f"in classes {self._classes}"
                        )
                    self._pos_label = pos_label
            else:
                self._pos_label = pos_label

    def estimate_state_nbytes(self) -> int:
        # Base: 7 counters × 8 B = 56, rounded to 64.
        base = 64
        if self._needs_classification:
            # tp[K] + fp[K] + fn[K] int64 = 24 × K bytes
            return base + 24 * self._k
        return base

    def batch_memory_factor(self) -> float:
        """§13.4/164: classification uses .to_pylist() per batch
        (~7× Python-object overhead). reduce() checks batch with this
        factor — honest peak RAM, not hidden.

        §13.4/168: multiclass log_loss also does .to_pylist() on K+1
        columns (y_true + K proba cols). Same overhead class.
        """
        if self._needs_classification:
            return 8.0
        if self._needs_proba and self._needs_multiclass_proba:
            return 8.0
        return 1.0

    def init(self):
        state = {
            "n": 0,
            "sum_y": 0.0,
            "sum_yy": 0.0,
            "sum_residual_sq": 0.0,
            "sum_abs_residual": 0.0,
            "sum_log_loss": 0.0,
            "n_proba": 0,
        }
        if self._needs_classification:
            state["tp"] = np.zeros(self._k, dtype=np.int64)
            state["fp"] = np.zeros(self._k, dtype=np.int64)
            state["fn"] = np.zeros(self._k, dtype=np.int64)
        return state

    def update(self, state, batch):
        # Regression (v0.9.2b-fix2, §13.4/165): gate by _needs_regression,
        # not by column presence. metrics=['accuracy'], y_pred='yp' with
        # string y_true would otherwise hit _to_f64('cat') → ValueError.
        if self._needs_regression:
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

        # Log loss (multiclass v0.9.2c; else binary)
        if self._needs_proba and self._needs_multiclass_proba:
            # Multiclass: y_true encodes class index (via classes);
            # sum -log(p[y_idx]).
            yt_list = batch.column(self.y_true_col).to_pylist()
            proba_lists = [batch.column(c).to_pylist()
                           for c in self.y_proba_cols]
            missing_yt: list = []
            eps = 1e-15
            for i, v in enumerate(yt_list):
                if v is None:
                    continue
                idx = self._cls_to_idx.get(v, -1)
                if idx < 0:
                    if len(missing_yt) < 5:
                        missing_yt.append(v)
                    continue
                p = proba_lists[idx][i]
                if p is None:
                    continue
                p_c = min(max(float(p), eps), 1.0 - eps)
                state["sum_log_loss"] += -math.log(p_c)
                state["n_proba"] += 1
            if missing_yt:
                raise ValueError(
                    f"MetricsOp: log_loss (multiclass) y_true has "
                    f"labels not in classes {self._classes}: "
                    f"{missing_yt} (first 5)."
                )
        # Log loss (binary; y must be {0,1}, p in [0,1])
        elif self._needs_proba:
            y = _to_f64(batch.column(self.y_true_col))
            p = _to_f64(batch.column(self.y_proba_cols[0]))
            mask = ~np.isnan(y) & ~np.isnan(p)
            if mask.any():
                y_m = y[mask]
                p_m = p[mask]
                # §13.4/163: validate y ∈ {0, 1} — log_loss formula
                # assumes binary; continuous y gives mathematically
                # invalid (often negative) output silently.
                bad_mask = (y_m != 0.0) & (y_m != 1.0)
                if bad_mask.any():
                    bad_vals = y_m[bad_mask][:5].tolist()
                    raise ValueError(
                        f"MetricsOp: log_loss requires binary y_true "
                        f"∈ {{0, 1}}; got values {bad_vals} "
                        f"(first 5 of {int(bad_mask.sum())} invalid). "
                        f"Use `encode_categorical` for multiclass or "
                        f"threshold predictions. §13.4/163."
                    )
                eps = 1e-15
                p_c = np.clip(p_m, eps, 1.0 - eps)
                ll = -(y_m * np.log(p_c)
                       + (1.0 - y_m) * np.log(1.0 - p_c))
                state["sum_log_loss"] += float(ll.sum())
                state["n_proba"] += int(mask.sum())

        # Classification (v0.9.2b): TP/FP/FN per class
        if self._needs_classification:
            yt_list = batch.column(self.y_true_col).to_pylist()
            yp_list = batch.column(self.y_pred_col).to_pylist()
            missing_yt: list = []
            missing_yp: list = []
            yt_idx = np.full(len(yt_list), -1, dtype=np.int64)
            yp_idx = np.full(len(yp_list), -1, dtype=np.int64)
            for i, v in enumerate(yt_list):
                if v is None:
                    continue
                idx = self._cls_to_idx.get(v, -1)
                if idx < 0:
                    if len(missing_yt) < 5:
                        missing_yt.append(v)
                else:
                    yt_idx[i] = idx
            for i, v in enumerate(yp_list):
                if v is None:
                    continue
                idx = self._cls_to_idx.get(v, -1)
                if idx < 0:
                    if len(missing_yp) < 5:
                        missing_yp.append(v)
                else:
                    yp_idx[i] = idx
            if missing_yt:
                raise ValueError(
                    f"MetricsOp: y_true has labels not in classes "
                    f"{self._classes}: {missing_yt} (first 5)."
                )
            if missing_yp:
                raise ValueError(
                    f"MetricsOp: y_pred has labels not in classes "
                    f"{self._classes}: {missing_yp} (first 5)."
                )
            valid = (yt_idx >= 0) & (yp_idx >= 0)
            yt = yt_idx[valid]
            yp = yp_idx[valid]
            same = yt == yp
            diff = ~same
            np.add.at(state["tp"], yt[same], 1)
            np.add.at(state["fp"], yp[diff], 1)
            np.add.at(state["fn"], yt[diff], 1)

        return state

    def merge(self, a, b):
        return {k: a[k] + b[k] for k in a}

    def state_nbytes(self, state):
        base = 8 * 7  # 7 counters × 8 B
        if self._needs_classification:
            base += state["tp"].nbytes + state["fp"].nbytes + state["fn"].nbytes
        return base

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

        if self._needs_classification:
            results.update(self._finalize_classification(state))

        return results

    def _finalize_classification(self, state) -> dict:
        tp = state["tp"].astype(np.float64)
        fp = state["fp"].astype(np.float64)
        fn = state["fn"].astype(np.float64)

        # Per-class precision / recall / f1 (zero_division=0, sklearn).
        # §13.4/164: np.divide(..., where=...) — avoids RuntimeWarning
        # from evaluating a/b on zero denominators inside np.where.
        pp = np.zeros_like(tp, dtype=np.float64)
        np.divide(tp, tp + fp, out=pp, where=(tp + fp) > 0)
        pr = np.zeros_like(tp, dtype=np.float64)
        np.divide(tp, tp + fn, out=pr, where=(tp + fn) > 0)
        denom = pp + pr
        pf1 = np.zeros_like(denom)
        np.divide(2.0 * pp * pr, denom, out=pf1, where=denom > 0)

        total = float(tp.sum() + fp.sum())  # = n_total
        accuracy = float(tp.sum()) / total if total > 0 else float("nan")

        results: dict = {}
        for metric in self.metrics:
            if metric == "accuracy":
                results["accuracy"] = accuracy
            elif metric == "precision":
                results["precision"] = self._average_value(
                    pp, tp, fn, accuracy,
                )
            elif metric == "recall":
                results["recall"] = self._average_value(
                    pr, tp, fn, accuracy,
                )
            elif metric == "f1":
                results["f1"] = self._average_value(
                    pf1, tp, fn, accuracy,
                )
        return results

    def _average_value(self, per_class, tp, fn, accuracy):
        avg = self._average_method
        if avg == "micro":
            return accuracy  # single-label multiclass
        if avg == "macro":
            return float(per_class.mean())
        if avg == "weighted":
            support = tp + fn  # true count per class
            total_support = float(support.sum())
            if total_support <= 0:
                return float("nan")
            return float((per_class * support).sum() / total_support)
        if avg == "binary":
            pi = self._cls_to_idx[self._pos_label]
            return float(per_class[pi])
        raise RuntimeError(
            f"_average_value: unknown average {avg!r} (internal bug)"
        )


# ============ ROC-AUC (v0.9.2c) ============

def _pos_label_kind_error(y_type: "pa.DataType", pos_label) -> "str | None":
    """K1.4: return an error message if `pos_label` cannot be compared
    with a column of `y_type`, else None.

    Without this, `pa.scalar(pos_label, type=y.type)` raises a cryptic
    ArrowInvalid (e.g. "Could not convert yes with type str: tried to
    convert to int64") with no hint that the problem is a bad
    `pos_label`. We pre-check and raise a clear ValueError instead.
    """
    import pyarrow as pa

    t = y_type
    if pa.types.is_boolean(t):
        if not isinstance(pos_label, bool):
            return (
                f"roc_auc: pos_label {pos_label!r} incompatible with "
                f"y column type {t}; expected a bool scalar"
            )
        return None
    if pa.types.is_integer(t):
        if isinstance(pos_label, bool) or not isinstance(pos_label, int):
            return (
                f"roc_auc: pos_label {pos_label!r} incompatible with "
                f"y column type {t}; expected an int-compatible scalar"
            )
        return None
    if pa.types.is_floating(t):
        if isinstance(pos_label, bool) or not isinstance(
            pos_label, (int, float)
        ):
            return (
                f"roc_auc: pos_label {pos_label!r} incompatible with "
                f"y column type {t}; expected a numeric scalar"
            )
        return None
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        if not isinstance(pos_label, str):
            return (
                f"roc_auc: pos_label {pos_label!r} incompatible with "
                f"y column type {t}; expected a str scalar"
            )
        return None
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        if not isinstance(pos_label, (bytes, bytearray)):
            return (
                f"roc_auc: pos_label {pos_label!r} incompatible with "
                f"y column type {t}; expected a bytes scalar"
            )
        return None
    # Unknown/other type: let Arrow attempt and surface its own error.
    return None


def _compute_roc_auc_binary(
    stream_factory,
    *,
    y_true_col: str,
    y_proba_col: str,
    pos_label,
    spill_dir,
    memory_guard,
    log: bool = False,
) -> float:
    """ROC-AUC (binary ovr) via external_sort + streaming trapezoid.

    Validated by scripts/proto_roc_auc.py PASS 8/8. All tied probas →
    single threshold point (matches sklearn roc_curve).

    Peak RAM: batch + external_sort disk. Independent of N.

    Args:
        stream_factory: callable() → fresh iterable of RecordBatch
            (called twice: pass 1 counts, pass 2 trapezoid).
        y_true_col: binary label column (any two distinct values).
        y_proba_col: probability column.
        pos_label: value in y_true that denotes positive class.
        spill_dir: temp dir for external_sort.
        memory_guard: MemoryGuard for check_batch.
        log: verbose logging.

    Returns:
        float AUC in [0, 1], or NaN if n_pos==0 or n_neg==0.
    """
    import pyarrow as pa
    import pyarrow.compute as _pc
    from pathlib import Path
    from cadw.ops.sort import external_sort
    from cadw.io.batch_reader import ArrowIPCBatchReader

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: count n_pos / n_neg (both y and proba non-null)
    total_valid = 0
    n_pos = 0
    _validated = False
    for batch in stream_factory():
        memory_guard.check_batch(batch, name="roc_auc.count")
        y = batch.column(y_true_col)
        p = batch.column(y_proba_col)
        if not _validated:
            # K1.4: fail early and clearly on type mismatch, before
            # `pa.scalar(pos_label, type=y.type)` produces ArrowInvalid.
            err = _pos_label_kind_error(y.type, pos_label)
            if err is not None:
                raise ValueError(err)
            _validated = True
        both_valid = _pc.and_(_pc.is_valid(y), _pc.is_valid(p))
        y_f = y.filter(both_valid)
        total_valid += len(y_f)
        if len(y_f) > 0:
            eq = _pc.equal(y_f, pa.scalar(pos_label, type=y.type))
            n_pos += int(_pc.sum(_pc.cast(eq, pa.int64())).as_py() or 0)
    n_neg = total_valid - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Sort by proba desc (values: y_true)
    sorted_path = external_sort(
        stream_factory(),
        key_columns=[y_proba_col],
        value_columns=[y_true_col],
        memory_guard=memory_guard,
        spill_dir=spill_dir / "sort_by_proba",
        ascending=False,
        log=False,
    )

    # Pass 2: stream sorted, per-threshold grouping, trapezoid
    tp = 0
    fp = 0
    prev_fpr = 0.0
    prev_tpr = 0.0
    area = 0.0
    cur_score = _ROC_AUC_UNSET
    cur_tp = 0
    cur_fp = 0

    try:
        reader = ArrowIPCBatchReader(
            str(sorted_path), [y_proba_col, y_true_col]
        )
        for b in reader:
            memory_guard.check_batch(b, name="roc_auc.stream", extra_factor=7.0)
            probas = b.column(y_proba_col).to_pylist()
            ys = b.column(y_true_col).to_pylist()
            for s, y in zip(probas, ys):
                if s is None or y is None:
                    continue
                if cur_score is _ROC_AUC_UNSET or s != cur_score:
                    if cur_score is not _ROC_AUC_UNSET:
                        tp += cur_tp
                        fp += cur_fp
                        tpr = tp / n_pos
                        fpr = fp / n_neg
                        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
                        prev_fpr = fpr
                        prev_tpr = tpr
                    cur_score = s
                    cur_tp = 0
                    cur_fp = 0
                if y == pos_label:
                    cur_tp += 1
                else:
                    cur_fp += 1
        # Flush last group
        if cur_score is not _ROC_AUC_UNSET:
            tp += cur_tp
            fp += cur_fp
            tpr = tp / n_pos
            fpr = fp / n_neg
            area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
    finally:
        try:
            Path(sorted_path).unlink()
        except OSError:
            pass

    return float(area)


def _compute_roc_auc_macro(
    stream_factory,
    *,
    y_true_col: str,
    y_proba_cols: Sequence[str],
    classes: Sequence,
    spill_dir,
    memory_guard,
    log: bool = False,
) -> float:
    """Macro-average ROC-AUC (one-vs-rest, K binary passes).

    Peak RAM per pass: batch + Sort disk. Independent of N.
    NaN if no class produces valid AUC (all-pos or all-neg).
    """
    import math as _math
    from pathlib import Path

    spill_dir = Path(spill_dir)
    aucs: list = []
    for k, cls in enumerate(classes):
        auc = _compute_roc_auc_binary(
            stream_factory,
            y_true_col=y_true_col,
            y_proba_col=y_proba_cols[k],
            pos_label=cls,
            spill_dir=spill_dir / f"auc_k{k}",
            memory_guard=memory_guard,
            log=log,
        )
        aucs.append(auc)
    valid = [a for a in aucs if not _math.isnan(a)]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))
