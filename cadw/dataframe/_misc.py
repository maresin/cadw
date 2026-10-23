# cadw/dataframe/_misc.py
# v0.9.26l (B.1 step 6): _MiscMixin (leaf props/helpers).
# Symbols: unique, _normalize_on, _groupby_impl, ts, plot.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

from cadw.ops.groupby import groupby_aggregate


class _MiscMixin:
    """Small leaf properties/helpers (v0.9.26l)."""
    def unique(self, column: str) -> "CadwDataFrame":
        if column not in self._columns:
            raise KeyError(f"unique: column {column!r} not in selection")
        cols = list(self._columns)
        result = groupby_aggregate(
            self._iter_filtered_batches(),
            [column],
            [column],
            ["count"],
            self._memory_guard,
            self._spill_dir / "unique",
            log=False,
        )
        # результат: by=column, val_count; оставляем только сам столбец
        values_only = result.select([column])
        # восстанавливаем остальные колонки? нет — unique возвращает только колонку
        return self._wrap_result_table(values_only)


    def _normalize_on(
        self, on: "str | Sequence[str]", *, fn_name: str,
    ) -> list[str]:
        """Normalize `on` to list[str], validate each column exists.

        v0.8.6: multi-column `on` (`on=["ts", "sensor"]`) — lexicographic
        composite sort. Accepts str (backward compat) or Sequence[str].
        """
        if isinstance(on, str):
            on_cols = [on]
        else:
            on_cols = list(on)
        if not on_cols:
            raise ValueError(f"{fn_name}: on must be non-empty")
        for c in on_cols:
            if c not in self._columns:
                raise KeyError(
                    f"{fn_name}: on column {c!r} not in {self._columns}"
                )
        return on_cols


    def _groupby_impl(
        self, by_cols, agg_columns, agg_funcs, *, combiner: bool = False
    ) -> "CadwDataFrame":
        for c in list(by_cols) + list(agg_columns):
            if c not in self._columns:
                raise KeyError(
                    f"groupby: column {c!r} not in current selection {self._columns}"
                )
        result = groupby_aggregate(
            self._iter_filtered_batches(),
            by_cols,
            agg_columns,
            agg_funcs,
            self._memory_guard,
            self._spill_dir / "groupby",
            combiner=combiner,
            log=False,
        )
        return self._wrap_result_table(result)


    @property
    def ts(self):
        """Time-series accessor (v0.9.24): df.ts.lag / .rolling_features / .fit_forecaster / ...

        Thin wrappers over cadw.ts.*. No logic lives in the accessor.
        """
        from cadw.accessors._ts import _TsAccessor
        return _TsAccessor(self)


    @property
    def plot(self):
        """Plot accessor: df.plot.hist(col) / df.plot.plotly.hist(col)."""
        from cadw.plot_accessor import _PlotAccessor
        return _PlotAccessor(self)

