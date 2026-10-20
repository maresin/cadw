#!/usr/bin/env python3
"""v0.9.1e prototype: per-group apply(fn) — composite-sort validation.

WHY PROTOTYPE-FIRST (§13.4/123):
  v0.8.2 (shift on=col): 5 fix iterations, no prototype
  v0.8.4 (rolling on=col): 1 fix iteration, prototype
  v0.8.10 (per-group shift/diff/rolling): 0 fix iterations, prototype
  v0.9.1e (apply): prototype NOW, before implementation.

CORE ALGORITHM (both paths):
  Sort by [*group_cols, row_id] → iterate; on group change →
  flush group (fn(group_df) → emit). Same as v0.8.10 with on_cols=[].

THREE MODES (pandas df.groupby(g, dropna=False).apply(fn)):
  A. fn(group_df) → scalar        → K rows, [*group_cols, result_col]
  B. fn(group_df) → pd.Series     → K rows, [*group_cols, *series.index]
  C. fn(group_df) → pd.DataFrame  → Σ len(out_i) rows, [*group_cols, *df.columns]
  Mixed (scalar in one group, Series in another) → error

PROTOTYPE FOCUS:
  Since fn is the SAME in both paths, output identity follows from
  identical group subframes + order within group. So verify:
    1. Groups match (keys + subframes)
    2. Within-group order = row_id (deterministic)
    3. Null group handling (dropna=False semantics)
  Output formatting (scalar/Series/DataFrame → concat) is deterministic
  once (1)-(3) hold.

TEST CASES (10):
  1. Single group
  2. Two groups
  3. Null in group col
  4. Single-row groups
  5. Multi-col group
  6. Ties (same group + same row_id order)
  7. Null in value
  8. Empty input (after filter)
  9. Order preservation (row_id tie-break)
  10. Series return comparison

RUN: uv run python scripts/proto_apply.py
Expected: PASS on all 10, then implement patch_v091e_apply.py.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd


def path_a_pandas_groups(data: dict, group_cols: list):
    """Pandas reference: {group_key: subframe} (dropna=False)."""
    pdf = pd.DataFrame(data)
    pdf["__orig_rid"] = range(len(pdf))
    groups = {}
    for key, sub in pdf.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        # Sort within group by orig_rid (pandas preserves original order,
        # but be explicit for determinism in compare)
        sub = sub.sort_values("__orig_rid", kind="mergesort").reset_index(drop=True)
        groups[key] = sub.drop(columns=["__orig_rid"])
    return groups


def path_b_composite_groups(data: dict, group_cols: list):
    """Composite-sort: {group_key: subframe} in row_id order."""
    pdf = pd.DataFrame(data).copy()
    pdf["__orig_rid"] = range(len(pdf))
    sort_cols = list(group_cols) + ["__orig_rid"]
    pdf = pdf.sort_values(sort_cols, kind="mergesort",
                          na_position="last").reset_index(drop=True)

    groups = {}
    prev_key = object()
    group_rows: list = []
    for _, row in pdf.iterrows():
        key = tuple((None if pd.isna(row[c]) else row[c]) for c in group_cols)
        if key != prev_key:
            if group_rows:
                groups[prev_key] = pd.DataFrame(group_rows).drop(
                    columns=["__orig_rid"],
                ).reset_index(drop=True)
            group_rows = []
            prev_key = key
        group_rows.append(row.to_dict())
    if group_rows:
        groups[prev_key] = pd.DataFrame(group_rows).drop(
            columns=["__orig_rid"],
        ).reset_index(drop=True)
    return groups


def _norm_key(k):
    """Normalize NaN/None keys for comparison."""
    return tuple(
        (None if (isinstance(x, float) and np.isnan(x)) else x)
        for x in (k if isinstance(k, tuple) else (k,))
    )


def compare_groups(name, data, group_cols):
    a = path_a_pandas_groups(data, group_cols)
    b = path_b_composite_groups(data, group_cols)

    a_norm = {_norm_key(k): v for k, v in a.items()}
    b_norm = {_norm_key(k): v for k, v in b.items()}

    if set(a_norm.keys()) != set(b_norm.keys()):
        print(f"  FAIL {name}: key set mismatch")
        print(f"    A keys: {sorted(a_norm.keys(), key=repr)}")
        print(f"    B keys: {sorted(b_norm.keys(), key=repr)}")
        return False

    for k in a_norm:
        try:
            pd.testing.assert_frame_equal(
                a_norm[k][list(data.keys())],
                b_norm[k][list(data.keys())],
                check_dtype=False,
            )
        except AssertionError as e:
            print(f"  FAIL {name}: group {k!r} subframe mismatch")
            print(f"    A:\n{a_norm[k]}")
            print(f"    B:\n{b_norm[k]}")
            print(f"    {e}")
            return False

    print(f"  PASS {name} ({len(a_norm)} groups)")
    return True


def check_output_scalar(name, data, group_cols, fn):
    """Compare scalar apply output: [(key, result), ...]."""
    a_groups = path_a_pandas_groups(data, group_cols)
    b_groups = path_b_composite_groups(data, group_cols)

    a_out = {_norm_key(k): fn(v) for k, v in a_groups.items()}
    b_out = {_norm_key(k): fn(v) for k, v in b_groups.items()}

    a_sorted = sorted(a_out.items(), key=lambda kv: repr(kv[0]))
    b_sorted = sorted(b_out.items(), key=lambda kv: repr(kv[0]))

    for (ka, va), (kb, vb) in zip(a_sorted, b_sorted):
        try:
            np.testing.assert_allclose(float(va), float(vb), rtol=1e-9)
        except (AssertionError, TypeError):
            if va != vb:
                print(f"  FAIL {name}: group {ka!r}: A={va!r} B={vb!r}")
                return False
    print(f"  PASS {name} (scalar output, {len(a_out)} groups)")
    return True


def main():
    print("=== proto_apply: A (pandas groupby) vs B (composite-sort) ===\n")
    all_pass = True

    # 1. Single group
    all_pass &= compare_groups(
        "1. single group",
        {"g": ["a"] * 5, "v": [1, 2, 3, 4, 5]},
        ["g"],
    )

    # 2. Two groups
    all_pass &= compare_groups(
        "2. two groups",
        {"g": ["a", "a", "b", "b", "b"], "v": [1, 2, 3, 4, 5]},
        ["g"],
    )

    # 3. Null in group col (dropna=False → own group)
    all_pass &= compare_groups(
        "3. null in group",
        {"g": ["a", None, "b", None], "v": [1, 2, 3, 4]},
        ["g"],
    )

    # 4. Single-row groups
    all_pass &= compare_groups(
        "4. single-row groups",
        {"g": ["a", "b", "c"], "v": [10, 20, 30]},
        ["g"],
    )

    # 5. Multi-col group
    all_pass &= compare_groups(
        "5. multi-col group",
        {
            "s1": ["a", "a", "b", "b"],
            "s2": ["x", "y", "x", "y"],
            "v": [1, 2, 3, 4],
        },
        ["s1", "s2"],
    )

    # 6. Ties (same group, order by row_id)
    all_pass &= compare_groups(
        "6. ties in group",
        {"g": ["a"] * 4, "v": [10, 20, 30, 40]},
        ["g"],
    )

    # 7. Null in value (should pass through)
    all_pass &= compare_groups(
        "7. null in value",
        {"g": ["a", "a", "b"], "v": [1.0, None, 3.0]},
        ["g"],
    )

    # 8. Empty input — no groups from either path
    all_pass &= compare_groups(
        "8. empty input",
        {"g": pd.Series([], dtype="object"),
         "v": pd.Series([], dtype="int64")},
        ["g"],
    )

    # 9. Order preservation — verify first-row-in-group = first-rid in data
    data = {
        "g": ["b", "a", "b", "a", "b"],
        "v": [1, 2, 3, 4, 5],
    }
    groups = path_b_composite_groups(data, ["g"])
    a_vals = groups[("a",)]["v"].tolist()
    b_vals = groups[("b",)]["v"].tolist()
    if a_vals == [2, 4] and b_vals == [1, 3, 5]:
        print("  PASS 9. within-group order preserved")
    else:
        print(f"  FAIL 9. within-group order: a={a_vals}, b={b_vals}")
        all_pass = False

    # 10. Scalar output comparison (mean per group)
    all_pass &= check_output_scalar(
        "10. scalar mean per group",
        {"g": ["a", "a", "b", "b"], "v": [10, 20, 100, 200]},
        ["g"],
        lambda grp: grp["v"].mean(),
    )

    # 11. Series output comparison (describe-like)
    def series_fn(grp):
        return pd.Series({"mean": grp["v"].mean(), "max": grp["v"].max()})

    a_groups = path_a_pandas_groups(
        {"g": ["a", "a", "b", "b"], "v": [10, 20, 100, 200]}, ["g"],
    )
    b_groups = path_b_composite_groups(
        {"g": ["a", "a", "b", "b"], "v": [10, 20, 100, 200]}, ["g"],
    )
    a_series = {k: series_fn(v) for k, v in a_groups.items()}
    b_series = {k: series_fn(v) for k, v in b_groups.items()}
    if (a_series[("a",)].equals(b_series[("a",)])
            and a_series[("b",)].equals(b_series[("b",)])):
        print(f"  PASS 11. series output ({len(a_series)} groups)")
    else:
        print(f"  FAIL 11. series output")
        all_pass = False

    print()
    if all_pass:
        print("ALL PASS — composite-sort apply validated on 11 cases.")
        print()
        print("NEXT: uv run python scripts/patch_v091e_apply.py")
        return 0
    print("SOME FAILED — do NOT implement patch_v091e_apply.py yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())