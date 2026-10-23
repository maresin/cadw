# cadw/dataframe/_groupby.py
# v0.9.26k (B.1 step 5): _GroupBy class.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

from typing import Sequence

class _GroupBy:
    def __init__(self, parent: CadwDataFrame, by_cols: Sequence[str]) -> None:
        from ._core import CadwDataFrame
        self._parent = parent
        self._by = list(by_cols)

    def agg(self, agg_columns, agg_funcs, *, combiner: bool = False) -> CadwDataFrame:
        """Aggregate группами.

        - combiner=True: partial aggregation до shuffle для
          sum/count/min/max/mean (экономия I/O в N/K раз для низкой
          кардинальности). std → warning + fallback. §13.4/52.
        """
        from ._core import CadwDataFrame
        agg_cols = [agg_columns] if isinstance(agg_columns, str) else list(agg_columns)
        funcs = [agg_funcs] if isinstance(agg_funcs, str) else list(agg_funcs)
        return self._parent._groupby_impl(
            self._by, agg_cols, funcs, combiner=combiner
        )

    def apply(
        self,
        fn,
        *,
        result_col: str = "apply_result",
        on: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Per-group apply (v0.9.6b2: CADW-native, no pandas).

        Args:
            fn: callable(group_df: CadwDataFrame) → scalar |
                dict[str, scalar] | CadwDataFrame. Uniform
                mode across groups. For pd.Series-like output
                use a dict; for pd.DataFrame-like output use
                CadwDataFrame.from_table(pa_table).
            result_col: column name for scalar mode.
            on: within-group sort keys (None → row_id order).

        Returns: CadwDataFrame
            scalar → K rows, [*group_cols, result_col]
            Series → K rows, [*group_cols, *series.index]
            DataFrame → Σ|df_i| rows, [*group_cols, *df.columns]

        Order: sorted by group key.

        Memory: per-group bounded (2 × group_nbytes ≤ 0.3 × avail BEFORE
        fn). Oversized → MemoryLimitExceeded. fn's alloc — user resp.
        """
        on_cols = self._normalize_on(on)
        return self._parent._per_group_apply(
            self._by, on_cols, fn, result_col,
            memory_guard=self._parent._memory_guard,
        )

    def _normalize_on(self, on) -> list:
        if on is None:
            return []
        if isinstance(on, str):
            return [on]
        return list(on)

    def transform(self, agg_columns, agg_funcs):
        # Aggregate + broadcast back to original rows via
        # left-merge on group keys. Same signature as agg().
        #
        # Returns CadwDataFrame with all original columns + new
        # {col}_{func} columns (like pandas assign of transform).
        #
        # LIMITATION (§11.5): row order in the result is
        # partition-concatenation (df.merge semantics), not original.
        # For pandas-compatible order, add __orig_rid and sort_values
        # after (documented; v0.9.5 backlog for streaming order-preserving)
        agg_cols = (
            [agg_columns] if isinstance(agg_columns, str)
            else list(agg_columns)
        )
        funcs = (
            [agg_funcs] if isinstance(agg_funcs, str)
            else list(agg_funcs)
        )
        if len(agg_cols) != len(funcs):
            raise ValueError(
                "transform: agg_columns and agg_funcs must have "
                "same length; got " + str(len(agg_cols)) + " vs "
                + str(len(funcs))
            )
        agg_df = self._parent._groupby_impl(self._by, agg_cols, funcs)
        return self._parent.merge(agg_df, on=self._by, how="left")

    def cumsum(
        self,
        columns: "str | Sequence[str] | None" = None,
        *,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group cumulative sum. APPEND `{col}_cumsum`.

        v0.9.23e (AUDIT_FULL_v0922 item 7.A.6, Top-10 gap #5).
        Group order: sort by [*group_cols, *on_cols]; accumulator
        reset per group.
        """
        from ._core import CadwDataFrame
        on_cols = self._normalize_on(on)
        return self._parent._per_group_cumulative(
            self._by, on_cols, "sum", columns,
        )

    def cumprod(
        self,
        columns: "str | Sequence[str] | None" = None,
        *,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group cumulative product. APPEND `{col}_cumprod`."""
        from ._core import CadwDataFrame
        on_cols = self._normalize_on(on)
        return self._parent._per_group_cumulative(
            self._by, on_cols, "prod", columns,
        )

    def cummax(
        self,
        columns: "str | Sequence[str] | None" = None,
        *,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group cumulative max. APPEND `{col}_cummax`."""
        from ._core import CadwDataFrame
        on_cols = self._normalize_on(on)
        return self._parent._per_group_cumulative(
            self._by, on_cols, "max", columns,
        )

    def cummin(
        self,
        columns: "str | Sequence[str] | None" = None,
        *,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group cumulative min. APPEND `{col}_cummin`."""
        from ._core import CadwDataFrame
        on_cols = self._normalize_on(on)
        return self._parent._per_group_cumulative(
            self._by, on_cols, "min", columns,
        )

    def shift(
        self,
        k: int = 1,
        fill_value=None,
        *,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group shift (v0.8.10).

        Shift by `k` within each group, optionally order-aware (`on=col`).
        `on=None` → within-group order = original row_id order.

        - k: >= 1. First k rows of EACH GROUP → `fill_value`.
        - columns: value columns to shift (default: all except group keys).
        - on: sort key within group (timestamp/integer), or None.

        Group columns are NOT shifted by default (they're the key).
        See `_per_group_shift` docstring for full semantics.
        """
        from ._core import CadwDataFrame
        on_cols = self._normalize_on(on)
        return self._parent._per_group_shift(
            self._by, on_cols, k, fill_value, columns
        )

    def diff(
        self,
        k: int = 1,
        *,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group first difference (v0.8.10).

        `out[i] = x[i] - x[i-k]` within each group. First k rows of each
        group → null (per-group, not global).
        """
        from ._core import CadwDataFrame
        on_cols = self._normalize_on(on)
        return self._parent._per_group_diff(
            self._by, on_cols, k, columns
        )

    def rolling(
        self,
        window: int,
        agg: str,
        *,
        min_periods: "int | None" = None,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group rolling (v0.8.10). APPENDS `{col}_rolling_{agg}`.

        Rolling computed within each group independently (min_periods
        applies from group start). Output preserves original row order;
        rolling columns are appended.
        """
        from ._core import CadwDataFrame
        on_cols = self._normalize_on(on)
        return self._parent._per_group_rolling(
            self._by, on_cols, window, agg, min_periods, columns
        )
