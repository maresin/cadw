"""Reference rolling implementation for tests (§13.4/128).

WHY: two consecutive test-expectation errors in v0.8.4-fix (rolling+nulls)
were manual arithmetic mistakes. Reference function derives expectations
from a deque-based algorithm (ground truth) instead of hand-computation.

USAGE:
    from tests._helpers.rolling_reference import expected_rolling
    expected = expected_rolling(vals, window, agg, min_periods=...)
    assert got == expected

SEMANTICS (matches pandas Series.rolling):
- Positional window of size `window` over `vals` (list, 1D).
- Nulls consume window slot but contribute nothing to sum/min/max/mean.
- count = number of non-null in current window.
- min_periods applies to NON-NULL count, not window size:
    pd.Series([1, None, 3]).rolling(2, min_periods=2).sum()
    -> [NaN, NaN, NaN]  (each window has at most 1 non-null)
- Output: list[float | int | None], length == len(vals).
"""
from __future__ import annotations

from collections import deque


def expected_rolling(vals, window: int, agg: str, min_periods=None):
    if not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be int >= 1; got {window!r}")
    if agg not in ("sum", "mean", "min", "max", "count"):
        raise ValueError(
            f"agg must be one of sum/mean/min/max/count; got {agg!r}"
        )
    if min_periods is None:
        min_periods = window
    elif not (1 <= min_periods <= window):
        raise ValueError(
            f"min_periods must be in [1, window={window}]; got {min_periods}"
        )

    out = []
    buf = deque(maxlen=window)
    for v in vals:
        buf.append(v)
        non_null = [x for x in buf if x is not None]
        if len(non_null) < min_periods:
            out.append(None)
            continue
        if agg == "count":
            out.append(len(non_null))
        elif not non_null:
            out.append(None)
        elif agg == "sum":
            out.append(sum(non_null))
        elif agg == "mean":
            out.append(sum(non_null) / len(non_null))
        elif agg == "min":
            out.append(min(non_null))
        elif agg == "max":
            out.append(max(non_null))
    return out
