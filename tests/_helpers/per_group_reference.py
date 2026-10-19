"""Reference per-group shift/diff/rolling via pandas (v0.8.10).

Ground-truth for test expectations (§13.4/128, /129, /139, /141):
expectations authored by reimplementing the same operation in pandas,
not by manual arithmetic. If pandas semantics and CADW diverge — test
fails loudly; investigate before touching either.
"""
from __future__ import annotations


def _normalize(vals):
    """Convert pandas NaN → None (matches CADW null semantics)."""
    out = []
    for v in vals:
        if v is None:
            out.append(None)
        elif isinstance(v, float) and v != v:  # NaN check
            out.append(None)
        else:
            out.append(v)
    return out


def expected_per_group_shift(
    vals, groups, *, on=None, k=1, fill_value=None,
):
    """Per-group shift reference.

    - vals: list of values to shift.
    - groups: list of group values (one column).
    - on: list of on-values, or None → original row_id order.
    - k, fill_value: as CADW.

    Returns list (CADW order = original row order).
    """
    import pandas as pd

    n = len(vals)
    if on is None:
        on_vals = list(range(n))
    else:
        on_vals = list(on)
    assert len(groups) == n, "groups length mismatch"
    assert len(on_vals) == n, "on length mismatch"

    pdf = pd.DataFrame({
        "v": vals,
        "grp": groups,
        "on": on_vals,
        "__rid": list(range(n)),
    })
    pdf = pdf.sort_values(
        ["grp", "on", "__rid"], kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    pdf["__out"] = pdf.groupby("grp", dropna=False)["v"].shift(k)
    pdf = pdf.sort_values("__rid").reset_index(drop=True)
    result = _normalize(pdf["__out"].tolist())
    if fill_value is not None:
        result = [fill_value if v is None else v for v in result]
    return result


def expected_per_group_diff(vals, groups, *, on=None, k=1):
    """Per-group diff reference."""
    import pandas as pd

    n = len(vals)
    on_vals = list(range(n)) if on is None else list(on)

    pdf = pd.DataFrame({
        "v": vals,
        "grp": groups,
        "on": on_vals,
        "__rid": list(range(n)),
    })
    pdf = pdf.sort_values(
        ["grp", "on", "__rid"], kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    pdf["__out"] = pdf.groupby("grp", dropna=False)["v"].diff(k)
    pdf = pdf.sort_values("__rid").reset_index(drop=True)
    return _normalize(pdf["__out"].tolist())


def expected_per_group_rolling(
    vals, groups, *, on=None, window=2, agg="sum", min_periods=None,
):
    """Per-group rolling reference. Returns list (original order)."""
    import pandas as pd

    n = len(vals)
    on_vals = list(range(n)) if on is None else list(on)
    mp = window if min_periods is None else min_periods

    pdf = pd.DataFrame({
        "v": vals,
        "grp": groups,
        "on": on_vals,
        "__rid": list(range(n)),
    })
    pdf = pdf.sort_values(
        ["grp", "on", "__rid"], kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    rolling = pdf.groupby("grp", dropna=False)["v"].rolling(
        window, min_periods=mp,
    )
    agg_vals = getattr(rolling, agg)().reset_index(level=0, drop=True)
    pdf["__out"] = agg_vals.values
    pdf = pdf.sort_values("__rid").reset_index(drop=True)
    return _normalize(pdf["__out"].tolist())
