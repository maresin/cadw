#!/usr/bin/env python3
"""v0.8.10 prototype: per-group shift/diff/rolling — 3-path validation.

WHY PROTOTYPE-FIRST (§13.4/123):
v0.8.2 (shift on=col) — 5 fix iterations, no prototype.
v0.8.4 (rolling on=col) — 1 fix iteration, prototype.
Prototype delta = algorithmic validation BEFORE code.

WHY COMPOSITE-SORT (Path B) AS CANDIDATE:
external_sort accepts arbitrary key_columns and stably sorts by
lexicographic composite. Sort by [group_cols..., on_col, row_id]:
contiguous groups mean ONE active deque, reset on group change.
Memory = O(window × #cols), NOT O(G × window × #cols). Cheaper than
shuffle-per-group (no partition files; 2 sort passes vs 3-4).

TWO PATHS (must match):
  A. pandas reference: sort_values([grp, on, orig_rid]), groupby(grp).shift(k),
     sort back by orig_rid. Well-tested pandas semantics.
  B. composite-sort algorithm: sort by [grp, on, orig_rid], single deque
     with reset-on-group-change, sort back by orig_rid. The path we'd
     implement via external_sort.

EXPECTED: A == B on all cases. If B != A on any case → investigate
WHY before patching (could be null ordering, tie-break, group-change
detection, or null propagation in diff).

TEST CASES (small, exhaustive):
  1. single group, shift(1)
  2. multi group, sorted input, shift(1)
  3. multi group, shuffled input, shift(1)
  4. ties in `on`, shift(1)
  5. nulls in value, shift(1)
  6. nulls in `on`, single group, shift(1)
  7. multi-key group, shift(1)
  8. diff(1)
  9. k=2
  10. single row per group, shift(1)
  11. multi group + null in `on`, shift(1)  ← na_position × group-boundary

Run: uv run python scripts/proto_per_group.py
"""
from __future__ import annotations

from collections import deque

import pandas as pd


def path_a_pandas(data, grp_cols, on_col, value_cols, k=1, op="shift"):
    """Reference: pandas native, sorted by (grp, on), apply, sort back."""
    pdf = pd.DataFrame(data).copy()
    pdf["__orig_rid"] = range(len(pdf))
    sort_cols = list(grp_cols) + [on_col, "__orig_rid"]
    pdf = pdf.sort_values(
        sort_cols, kind="mergesort", na_position="last",
    ).reset_index(drop=True)

    for vc in value_cols:
        if op == "shift":
            pdf[f"__out_{vc}"] = pdf.groupby(
                list(grp_cols), dropna=False,
            )[vc].shift(k)
        elif op == "diff":
            pdf[f"__out_{vc}"] = pdf.groupby(
                list(grp_cols), dropna=False,
            )[vc].diff(k)
        else:
            raise ValueError(f"unknown op {op!r}")

    pdf = pdf.sort_values("__orig_rid").reset_index(drop=True)
    return {vc: pdf[f"__out_{vc}"].tolist() for vc in value_cols}


def path_b_deque(data, grp_cols, on_col, value_cols, k=1, op="shift"):
    """Candidate: composite sort + single deque, reset on group change."""
    pdf = pd.DataFrame(data).copy()
    pdf["__orig_rid"] = range(len(pdf))
    sort_cols = list(grp_cols) + [on_col, "__orig_rid"]
    pdf = pdf.sort_values(
        sort_cols, kind="mergesort", na_position="last",
    ).reset_index(drop=True)

    # Single active state per value column, reset on group change
    out = {vc: [None] * len(pdf) for vc in value_cols}
    prev_group = object()  # sentinel (won't equal any real group)
    dqs = {vc: deque(maxlen=k) for vc in value_cols}

    for i, row in pdf.iterrows():
        group = tuple(
            (None if pd.isna(row[c]) else row[c]) for c in grp_cols
        )
        if group != prev_group:
            for vc in value_cols:
                dqs[vc].clear()
            prev_group = group

        for vc in value_cols:
            cur = None if pd.isna(row[vc]) else row[vc]
            if op == "shift":
                out[vc][i] = dqs[vc][0] if len(dqs[vc]) == k else None
                dqs[vc].append(cur)
            elif op == "diff":
                if len(dqs[vc]) < k:
                    out[vc][i] = None
                else:
                    prev = dqs[vc][0]
                    if prev is None or cur is None:
                        out[vc][i] = None
                    else:
                        out[vc][i] = cur - prev
                dqs[vc].append(cur)
            else:
                raise ValueError(f"unknown op {op!r}")

    # Reorder by original row_id
    reordered = {vc: [None] * len(pdf) for vc in value_cols}
    for i, row in pdf.iterrows():
        orig = int(row["__orig_rid"])
        for vc in value_cols:
            reordered[vc][orig] = out[vc][i]
    return reordered


def _norm(d):
    """Normalize values for comparison: NaN↔None, keep numeric as-is."""
    out = {}
    for vc, vals in d.items():
        normed = []
        for x in vals:
            if x is None:
                normed.append(None)
            elif isinstance(x, float) and pd.isna(x):
                normed.append(None)
            else:
                normed.append(x)
        out[vc] = normed
    return out


def _float_eq(a, b, tol=1e-9):
    """True if both None, or both numeric within tol."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return a == b


def check_case(name, data, grp_cols, on_col, value_cols, k=1, op="shift"):
    a = path_a_pandas(data, grp_cols, on_col, value_cols, k=k, op=op)
    b = path_b_deque(data, grp_cols, on_col, value_cols, k=k, op=op)
    a_n, b_n = _norm(a), _norm(b)

    ok = True
    diffs = []
    for vc in value_cols:
        if len(a_n[vc]) != len(b_n[vc]):
            ok = False
            diffs.append(f"  length mismatch {vc}: A={len(a_n[vc])} B={len(b_n[vc])}")
            continue
        for i, (av, bv) in enumerate(zip(a_n[vc], b_n[vc])):
            if not _float_eq(av, bv):
                ok = False
                diffs.append(f"  {vc}[{i}]: A={av!r} B={bv!r}")

    status = "PASS" if ok else "FAIL"
    print(f"  {status} {name}")
    for line in diffs[:5]:
        print(f"  {line}")
    return ok


def main():
    print("=== proto_per_group: A (pandas) vs B (composite-sort deque) ===\n")

    cases = [
        # 1. single group, shift(1)
        ("1. single group, shift(1)",
         {"sensor": ["a", "a", "a", "a"],
          "ts": [1, 2, 3, 4],
          "v": [10, 20, 30, 40]},
         ["sensor"], "ts", ["v"], 1, "shift"),

        # 2. multi group, sorted input
        ("2. multi group, sorted, shift(1)",
         {"sensor": ["a", "a", "b", "b"],
          "ts": [1, 2, 1, 2],
          "v": [10, 20, 30, 40]},
         ["sensor"], "ts", ["v"], 1, "shift"),

        # 3. multi group, shuffled input
        ("3. multi group, shuffled, shift(1)",
         {"sensor": ["b", "a", "b", "a"],
          "ts": [2, 1, 1, 2],
          "v": [40, 10, 30, 20]},
         ["sensor"], "ts", ["v"], 1, "shift"),

        # 4. ties in on (tie-break by orig_rid)
        ("4. ties in on, shift(1)",
         {"sensor": ["a", "a", "a"],
          "ts": [1, 1, 1],
          "v": [10, 20, 30]},
         ["sensor"], "ts", ["v"], 1, "shift"),

        # 5. nulls in value
        ("5. nulls in value, shift(1)",
         {"sensor": ["a", "a", "a", "a"],
          "ts": [1, 2, 3, 4],
          "v": [10, None, 30, None]},
         ["sensor"], "ts", ["v"], 1, "shift"),

        # 6. nulls in on, single group
        ("6. nulls in on, single group, shift(1)",
         {"sensor": ["a", "a", "a"],
          "ts": [1, None, 3],
          "v": [10, 20, 30]},
         ["sensor"], "ts", ["v"], 1, "shift"),

        # 7. multi-key group
        ("7. multi-key group, shift(1)",
         {"s1": ["a", "a", "a", "a"],
          "s2": ["x", "x", "y", "y"],
          "ts": [1, 2, 1, 2],
          "v": [10, 20, 30, 40]},
         ["s1", "s2"], "ts", ["v"], 1, "shift"),

        # 8. diff(1)
        ("8. multi group, diff(1)",
         {"sensor": ["a", "a", "b", "b"],
          "ts": [1, 2, 1, 2],
          "v": [10, 25, 100, 150]},
         ["sensor"], "ts", ["v"], 1, "diff"),

        # 9. shift(2)
        ("9. multi group, shift(2)",
         {"sensor": ["a", "a", "a", "b", "b", "b"],
          "ts": [1, 2, 3, 1, 2, 3],
          "v": [10, 20, 30, 100, 200, 300]},
         ["sensor"], "ts", ["v"], 2, "shift"),

        # 10. single row per group
        ("10. single row per group, shift(1)",
         {"sensor": ["a", "b", "c"],
          "ts": [1, 1, 1],
          "v": [10, 20, 30]},
         ["sensor"], "ts", ["v"], 1, "shift"),

        # 11. multi group + null in on (group-boundary × na_position)
        ("11. multi group + null in on, shift(1)",
         {"sensor": ["a", "a", "b", "b"],
          "ts": [1, None, 1, None],
          "v": [10, 20, 30, 40]},
         ["sensor"], "ts", ["v"], 1, "shift"),
    ]

    all_pass = True
    for name, data, grp, on, val, k, op in cases:
        ok = check_case(name, data, grp, on, val, k=k, op=op)
        all_pass = all_pass and ok

    print()
    if all_pass:
        print("ALL PASS — composite-sort approach validated on 11 cases.")
        print()
        print("NEXT: implement patch_v0810_per_group.py")
        print("      (single deque, reset-on-group-change, 2× external_sort).")
        print("      No shuffle. Memory O(window × #cols) — independent of G.")
        return 0

    print("SOME FAILED — do NOT implement v0.8.10 patch yet.")
    print("Investigate divergence above before proceeding.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())