"""Fairness metrics (D.1.4, Phase D, v0.9.35).

Group-wise confusion-matrix metrics from a single streaming pass:
demographic parity (DP), equal opportunity (EO, = TPR gap), FPR gap,
disparate impact.

Reference: Hardt, Price, Srebro, *Equality of Opportunity in
Supervised Learning*, NeurIPS 2016 (arXiv:1610.02413); fairlearn
`MetricFrame` formulas (BSD). See docs + .pingis/algorithm_notes/
d1_4_fairness.md.

Design: one Reduce-style pass. Per batch we build 0/1 masks and
`group_by(sensitive).aggregate(sum)` in Arrow — no row loops, no
.to_pylist() over rows. State is a per-group 4-counter dict, O(K)
with K = number of sensitive groups, independent of N.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.memory_guard import MemoryGuard


_GROUP_OVERHEAD = 160  # bytes per group entry (dict + list + str key)


class FairnessReport:
    """Per-group confusion counts + fairness gaps."""

    def __init__(self, groups: dict, metrics: dict, n: int):
        self.groups = groups
        self.metrics = metrics
        self.n_ = int(n)
        self.fitted_ = True

    def to_dict(self) -> dict:
        return {
            "n": self.n_,
            "groups": {str(k): dict(v) for k, v in self.groups.items()},
            "metrics": dict(self.metrics),
        }

    def state_nbytes(self) -> int:
        return len(self.groups) * _GROUP_OVERHEAD

    def __repr__(self) -> str:
        m = self.metrics
        return (
            f"FairnessReport(groups={len(self.groups)}, "
            f"dp={m.get('demographic_parity_difference'):.4f}, "
            f"eo={m.get('equal_opportunity_difference'):.4f})"
        )


def _binary_int(b, col: str, extra: float) -> pa.Array:
    """Validate a 0/1 column and return int64 mask-ready array."""
    arr = b.column(col)
    if not (pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type)
            or pa.types.is_boolean(arr.type)):
        raise TypeError(f"fairness: column {col!r} must be numeric/bool; got {arr.type}")
    if pa.types.is_boolean(arr.type):
        arr = arr.cast(pa.int64())
    else:
        arr = arr.cast(pa.int64(), safe=False)
    if arr.null_count:
        raise ValueError(f"fairness: column {col!r} has nulls; fill or filter first")
    mm = pc.min_max(arr)
    lo, hi = mm["min"].as_py(), mm["max"].as_py()
    if lo < 0 or hi > 1:
        raise ValueError(f"fairness: column {col!r} must be binary 0/1; got [{lo}, {hi}]")
    return arr


def _iter_group_counts(df, cols, mg):
    """Yield per-batch (group_value -> [tn, fp, fn, tp]) dicts."""
    sens, yt_col, yp_col = cols
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name="fairness.batch", extra_factor=3.0)
        yt = _binary_int(b, yt_col, 2.0)
        yp = _binary_int(b, yp_col, 2.0)
        s = b.column(sens)
        if s.null_count:
            raise ValueError(f"fairness: sensitive column {sens!r} has nulls")
        tp = pc.and_(pc.equal(yt, 1), pc.equal(yp, 1)).cast(pa.int64())
        fp = pc.and_(pc.equal(yt, 0), pc.equal(yp, 1)).cast(pa.int64())
        fn = pc.and_(pc.equal(yt, 1), pc.equal(yp, 0)).cast(pa.int64())
        tn = pc.and_(pc.equal(yt, 0), pc.equal(yp, 0)).cast(pa.int64())
        tbl = pa.table({"g": s, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
        agg = tbl.group_by("g").aggregate([
            ("tn", "sum"), ("fp", "sum"), ("fn", "sum"), ("tp", "sum"),
        ])
        gcol = agg.column("g")
        vals = {name: agg.column(name + "_sum").to_numpy(zero_copy_only=False)
                for name in ("tn", "fp", "fn", "tp")}
        # K rows (sensitive groups), not N — safe under the check_batch above.
        keys = gcol.to_pylist()  # K values per batch, not N
        out = {}
        for i, k in enumerate(keys):
            out[k] = [int(vals["tn"][i]), int(vals["fp"][i]),
                      int(vals["fn"][i]), int(vals["tp"][i])]
        yield out


def fairness_metrics(
    df,
    *,
    sensitive_col: str,
    y_true_col: str,
    y_pred_col: str,
    memory_guard: Optional[MemoryGuard] = None,
) -> FairnessReport:
    """Streaming group fairness metrics (DP / EO / TPR gap / FPR gap).

    Args:
        sensitive_col: group column (string or int, non-null).
        y_true_col: binary 0/1 true labels.
        y_pred_col: binary 0/1 predictions.

    Returns FairnessReport with per-group counts/rates and:
        demographic_parity_difference = max_g PPR_g - min_g PPR_g
        equal_opportunity_difference = max_g TPR_g - min_g TPR_g
        tpr_gap (alias of EO), fpr_gap
        disparate_impact = min_g PPR_g / max_g PPR_g

    State: O(K) counters, independent of N. Guarded per batch.
    """
    for c, lbl in ((sensitive_col, "sensitive_col"),
                   (y_true_col, "y_true_col"), (y_pred_col, "y_pred_col")):
        if c not in df._columns:
            raise KeyError(f"fairness: {lbl} {c!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    acc: dict = {}
    n_total = 0
    for part in _iter_group_counts(df, (sensitive_col, y_true_col, y_pred_col), mg):
        for k, v in part.items():
            cur = acc.get(k)
            if cur is None:
                acc[k] = v
            else:
                for j in range(4):
                    cur[j] += v[j]
        n_total += sum(sum(v) for v in part.values())
        mg.check_state(len(acc) * _GROUP_OVERHEAD, name="fairness.groups")
    if not acc:
        raise ValueError("fairness: no rows")

    groups: dict = {}
    for k in sorted(acc, key=lambda x: str(x)):
        tn, fp, fn, tp = acc[k]
        n = tn + fp + fn + tp
        pos = tp + fn
        neg = tn + fp
        groups[k] = {
            "n": n,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "ppr": (tp + fp) / n if n else float("nan"),
            "tpr": tp / pos if pos else float("nan"),
            "fpr": fp / neg if neg else float("nan"),
        }

    ppr = np.array([g["ppr"] for g in groups.values()], dtype=float)
    tpr = np.array([g["tpr"] for g in groups.values()], dtype=float)
    fpr = np.array([g["fpr"] for g in groups.values()], dtype=float)
    metrics = {
        "demographic_parity_difference": float(np.nanmax(ppr) - np.nanmin(ppr)),
        "equal_opportunity_difference": float(np.nanmax(tpr) - np.nanmin(tpr)),
        "tpr_gap": float(np.nanmax(tpr) - np.nanmin(tpr)),
        "fpr_gap": float(np.nanmax(fpr) - np.nanmin(fpr)),
    }
    hi = float(np.nanmax(ppr))
    metrics["disparate_impact"] = (
        float(np.nanmin(ppr) / hi) if hi > 0 else float("nan")
    )
    return FairnessReport(groups, metrics, n_total)
