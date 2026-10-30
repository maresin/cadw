# cadw/dataframe/_aggregations.py
# v0.9.26n (B.1 step 8): _AggregationsMixin.
# Symbols: groupby, value_counts, corr, nunique, describe,
#   nlargest, nsmallest, isna, isna_sum, quantile, count.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import numpy as np
import pyarrow as pa
from ._groupby import _GroupBy
from cadw.dataframe_helpers import _empty_value_counts_table
from cadw.ops.quantile import quantile as _ops_quantile
from typing import Sequence


class _AggregationsMixin:
    """Aggregations / statistics mixin (v0.9.26n)."""
    def groupby(self, by: str | Sequence[str]) -> "_GroupBy":
        by_cols = [by] if isinstance(by, str) else list(by)
        return _GroupBy(self, by_cols)


    def value_counts(
        self,
        column: str,
        *,
        normalize: bool = False,
        sort: bool = True,
        dropna: bool = False,
    ) -> pa.Table:
        """Per-value counts (HashShuffle + groupby).

        v0.9.23g (AUDIT_FULL_v0922 item 7.A.3, Top-10 gap #7).

        Args:
            column: column to count.
            normalize: return `count` as float fraction of N (total
                non-null + null rows = filtered row count).
            sort: sort by count descending (default True). False →
                partition-concat order (row order not stable).

        Returns: `pa.Table` with columns `[column, "count"]` (or
        `[column, "fraction"]` when normalize=True).
        """
        import pyarrow.compute as _pc
        from cadw.ops.groupby import groupby_aggregate as _gba

        if column not in self._columns:
            raise KeyError(
                f"value_counts: column {column!r} not in {self._columns}"
            )

        col_type = self._dataset.schema.field(column).type
        # Empty-input fast path: groupby_aggregate raises ValueError
        # on empty input (v0.9.23g-fix2).
        if self.count() == 0:
            return _empty_value_counts_table(
                column, col_type, normalize=normalize,
            )

        # v0.9.23g-fix2: ones-sum, NOT count. groupby_aggregate(count)
        # gives non-null count per group (CountOp skips nulls) -> null
        # group gets 0. value_counts must count SIZE (every row,
        # including nulls). Ones-sum: each row contributes 1.
        def _with_one():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name="value_counts.input",
                )
                if b.num_rows == 0:
                    continue
                if dropna:
                    mask = _pc.is_valid(b.column(column))
                    b = b.filter(mask)
                    if b.num_rows == 0:
                        continue
                one = pa.array([1] * b.num_rows, type=pa.int64())
                yield b.append_column("__vco", one)

        try:
            result = _gba(
                _with_one(),
                [column],
                ["__vco"],
                ["sum"],
                self._memory_guard,
                self._spill_dir / "value_counts",
                log=False,
            )
        except ValueError as e:
            # groupby_aggregate raises ValueError on empty input. With
            # dropna=True and all-null column, filtered is empty ->
            # gba sees empty -> ValueError. Return empty table.
            # v0.9.23g-fix4: is_empty_input_error checks both
            # languages ("empty" / "пустой") — prior was English-only
            # and missed Russian message.
            from cadw.patch_utils import is_empty_input_error
            if is_empty_input_error(e):
                return _empty_value_counts_table(
                    column, col_type, normalize=normalize,
                )
            raise
        # v0.9.95: groupby_aggregate returns CadwDataFrame; convert
        # back to pa.Table for value_counts' pa.Table-based tail.
        _batches = list(result._iter_filtered_batches())
        if not _batches:
            return _empty_value_counts_table(
                column, col_type, normalize=normalize,
            )
        result = pa.Table.from_batches(_batches)
        result = result.rename_columns([column, "count"])
        if sort:
            order = _pc.sort_indices(
                result, sort_keys=[("count", "descending")],
            )
            result = result.take(order)
        if normalize:
            n = self.count()
            if n == 0:
                # Shouldn't reach (empty handled above); defensive.
                return _empty_value_counts_table(
                    column, col_type, normalize=True,
                )
            frac = _pc.divide(
                _pc.cast(result.column("count"), pa.float64()),
                pa.scalar(float(n), type=pa.float64()),
            )
            # v0.9.23g-fix2: set_column replaces index 1 -> no .drop.
            return result.set_column(1, "fraction", frac)
        return result


    def corr(self, columns=None, *, method="pearson"):
        """Correlation matrix (Pearson) between numeric columns.

        v0.9.23d (AUDIT_FULL_v0922 item 7.A.4). Composition of existing
        `CovarianceMeanOp` (one Reduce pass) + numpy normalization.
        O(d²) state, independent of N. Fail-fast if state > state_limit.

        Args:
            columns: numeric column names. None → auto-detect int/float.
                str → single column; Sequence[str] → those columns.
            method: 'pearson' (default). 'spearman' → NotImplementedError
                (needs per-column rank + Pearson on ranks; backlog P1).

        Returns:
            dict[str, dict[str, float]] — symmetric correlation matrix:
            `corr[a][b] = r`. Diagonal = 1.0 (or NaN for degenerate
            column). NaN for pairs involving a constant / all-null
            column.

        Raises:
            NotImplementedError: method != 'pearson'.
            ValueError: no numeric columns selected.
            KeyError: missing column.
            TypeError: non-numeric column requested explicitly.
            MemoryLimitExceeded: d too large for budget (CovarianceOp
                estimate 8+8d+8d² > state_limit).

        Example:
            >>> corr = df.corr()
            >>> corr['x1']['x2']         # Pearson r between x1, x2
            >>> corr['x1']['x1']         # == 1.0
        """
        from cadw.ops.reduce import CovarianceMeanOp, reduce

        if method != "pearson":
            raise NotImplementedError(
                f"corr: method={method!r} not supported. Only "
                f"'pearson' is implemented in v0.9.23d. 'spearman' "
                f"needs per-column rank (backlog P1)."
            )

        # Auto-detect numeric columns (None) or normalize (str / Sequence).
        if columns is None:
            cols = []
            for c in self._columns:
                t = self._dataset.schema.field(c).type
                if pa.types.is_integer(t) or pa.types.is_floating(t):
                    cols.append(c)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)

        if not cols:
            raise ValueError(
                "corr: no numeric columns selected (columns=None auto-"
                "detects int/float; found none)"
            )
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"corr: column {c!r} not in {self._columns}"
                )
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"corr: column {c!r} has type {t}; numeric required"
                )

        # One Reduce pass (O(d²) state, guarded in reduce()).
        op = CovarianceMeanOp(cols)
        res = reduce(
            self._iter_filtered_batches(), op, self._memory_guard, log=False,
        )
        cov = res["cov"]
        n = res["n"]

        nan = float("nan")
        if n < 2:
            # Not enough valid rows for any correlation.
            return {c: {c2: nan for c2 in cols} for c in cols}

        diag = np.diag(cov)
        std = np.sqrt(np.maximum(diag, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.outer(std, std)
            corr_mat = np.where(denom > 0, cov / denom, nan)
        # Symmetrize (numerical noise) and clip to [-1, 1].
        corr_mat = 0.5 * (corr_mat + corr_mat.T)
        corr_mat = np.clip(corr_mat, -1.0, 1.0)
        # Diagonal: 1.0 if var > 0, else NaN (constant / all-null).
        for i in range(len(cols)):
            corr_mat[i, i] = 1.0 if diag[i] > 0 else nan

        return {
            cols[i]: {cols[j]: float(corr_mat[i, j]) for j in range(len(cols))}
            for i in range(len(cols))
        }


    def nunique(self, column=None, *, dropna: bool = False):
        """Count distinct values. Int for one column, dict for all.

        v0.9.23g (AUDIT_FULL_v0922 item 7.A.3, Top-10 gap #7).

        Args:
            column: str → int (nunique of that column); None → dict[str,
                int] for all columns.
            dropna: if True, null is NOT counted as a distinct value
                (pandas-parity). Default False (CADW groupby default
                per §13.4/218 — null-key preservation).
        """
        from cadw.ops.groupby import groupby_aggregate as _gba

        # v0.9.23g-fix1: global nunique must use groupby_aggregate
        # (HashShuffle + count). Prior implementation summed per-batch
        # count_distinct -> WRONG on multi-batch (SILENT DATA
        # CORRUPTION: sum of per-batch uniques != global unique).
        # Example: [a, a] -> batch1=[a], batch2=[a] -> 1+1=2, actual=1.
        def _nunique_one(col_name: str, dropna_inner: bool) -> int:
            # groupby_aggregate over [col] → one row per unique value.
            # Number of output rows = global unique count.
            if dropna_inner:
                import pyarrow.compute as _pc
                def _filtered():
                    for b in self._iter_filtered_batches():
                        self._memory_guard.check_batch(
                            b, name=f"nunique[{col_name}]",
                        )
                        mask = _pc.is_valid(b.column(col_name))
                        f = b.filter(mask)
                        if f.num_rows > 0:
                            yield f
                it = _filtered()
            else:
                it = self._iter_filtered_batches()
            try:
                result = _gba(
                    it,
                    [col_name],
                    [col_name],
                    ["count"],
                    self._memory_guard,
                    self._spill_dir / f"nunique_{col_name}",
                    log=False,
                )
            except ValueError as e:
                # groupby_aggregate raises on empty input. dropna=True
                # + all-null column -> filtered is empty -> 0 uniques.
                # v0.9.23g-fix5: is_empty_input_error (both languages).
                from cadw.patch_utils import is_empty_input_error
                if is_empty_input_error(e):
                    return 0
                raise
            return result.count()  # v0.9.95: CadwDataFrame

        if column is not None:
            if column not in self._columns:
                raise KeyError(
                    f"nunique: column {column!r} not in {self._columns}"
                )
            return _nunique_one(column, dropna)

        # All columns: per-column groupby_aggregate.
        out: dict = {}
        for c in self._columns:
            out[c] = _nunique_one(c, dropna)
        return out


    def describe(self, columns=None, *, reservoir_size=10000):
        """Numeric summary — fused single-pass (v0.9.7b, R5: 109×).

        Single Reduce pass per column group: Welford (mean/M2/std)
        + min/max + reservoir (Algorithm R) for quantiles. Replaces
        5 Reduce passes + 1 external_sort per column.

        Peak RAM = batch + O(d × reservoir_size × 8B), independent
        of N. Default reservoir_size=10000 → quantiles exact for
        N ≤ reservoir_size, approximate above (§11.5).

        Args:
            columns: numeric columns (None → auto-detect int/float).
            reservoir_size: reservoir capacity per column
                (>= 1, default 10000). Larger → more exact quantiles,
                more memory (d × reservoir_size × 8B).

        Returns {col: {count, mean, std, min, 25%, 50%, 75%, max}}.
        Empty / all-null column → NaN for stat fields, count=0.

        For exact quantiles at any N, use `df.quantile()`
        (external_sort per column; slower but exact).
        """
        from cadw.ops.describe import describe_columns

        if not isinstance(reservoir_size, int) or reservoir_size < 1:
            raise ValueError(
                f"describe: reservoir_size must be int >= 1; "
                f"got {reservoir_size!r}"
            )
        if columns is None:
            cols = []
            for c in self._columns:
                t = self._dataset.schema.field(c).type
                if (pa.types.is_integer(t)
                        or pa.types.is_floating(t)):
                    cols.append(c)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    "describe: column " + repr(c) + " not in "
                    + str(self._columns)
                )
        return describe_columns(
            self._iter_filtered_batches(), cols, self._memory_guard,
            reservoir_size=reservoir_size,
        )


    def nlargest(self, n, column):
        """Streaming top-n largest (v0.9.7a, R4: 3946× vs external_sort).

        Replaces previous `sort_values(column, ascending=False).head(n)`
        (external_sort on all N, disk I/O) with per-batch top-k + merge
        (in-memory, P×k accumulator bounded by MemoryGuard).

        Peak RAM = batch + O(P × n × row_size), independent of N beyond
        P = num_batches. Fail-fast MemoryLimitExceeded when accumulator
        exceeds state_limit.

        Tie-break: stable by original row order (matches
        `sort_values(column, ascending=False)` semantics).

        Args:
            n: count, int >= 1.
            column: sort key column name.

        Returns CadwDataFrame with top-n rows, sorted descending.
        """
        from cadw.ops.topk import topk_to_disk

        if not isinstance(n, int) or n < 1:
            raise ValueError(
                f"nlargest: n must be int >= 1; got {n!r}"
            )
        if column not in self._columns:
            raise KeyError(
                f"nlargest: column {column!r} not in {self._columns}"
            )
        table = topk_to_disk(
            self._iter_filtered_batches(),
            column=column,
            n=n,
            kind="largest",
            memory_guard=self._memory_guard,
        )
        return self._wrap_result_table(table)


    def nsmallest(self, n, column):
        """Streaming top-n smallest (v0.9.7a, R4). See nlargest.

        Sorted ascending; stable tie-break by original row order.
        """
        from cadw.ops.topk import topk_to_disk

        if not isinstance(n, int) or n < 1:
            raise ValueError(
                f"nsmallest: n must be int >= 1; got {n!r}"
            )
        if column not in self._columns:
            raise KeyError(
                f"nsmallest: column {column!r} not in {self._columns}"
            )
        table = topk_to_disk(
            self._iter_filtered_batches(),
            column=column,
            n=n,
            kind="smallest",
            memory_guard=self._memory_guard,
        )
        return self._wrap_result_table(table)


    def isna(self) -> "CadwDataFrame":
        """Null mask: same-shape CadwDataFrame with all-bool columns.

        v0.9.23g (AUDIT_FULL_v0922 item 7.A.5). Per-batch `pc.is_null`
        — no materialize, streaming. Column names preserved.
        """
        from ._core import CadwDataFrame
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset as _DS

        # v0.9.23g-fix1 (§13.4/169): nullable=True. Dataset.
        # from_batches reconstructs per-column batches via
        # RecordBatch.from_arrays([col]) which always sets
        # nullable=True. nullable=False here would trigger
        # ArrowInvalid: Tried to write record batch with
        # different schema at write_batch.
        out_fields = [
            pa.field(c, pa.bool_(), nullable=True) for c in self._columns
        ]
        out_schema = pa.schema(out_fields)

        def _iter_isna():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="isna.input")
                arrays = [_pc.is_null(b.column(c)) for c in self._columns]
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = _DS.from_batches(
            out_schema, _iter_isna(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def isna_sum(self) -> dict:
        """Per-column null count: dict[col -> int]. One Reduce pass.

        v0.9.23g. Each column is one CountOp pass (sums via `reduce`).
        Total: d passes. For d > 20 use `info(compute_non_null=True)` if
        per-batch `n_null = n - count` suffices — same cost.
        """
        from cadw.ops.reduce import NullCountOp, reduce

        # v0.9.23g-fix6: NullCountOp (no .as_py in facade —
        # test_facade_has_no_as_py AST guard). batch.column(c).
        # null_count is O(1) metadata; d Reduce passes, same cost as
        # prior per-batch sum approach.
        out: dict = {}
        for c in self._columns:
            out[c] = reduce(
                self._iter_filtered_batches(),
                NullCountOp(c),
                self._memory_guard,
                log=False,
            )
        return out


    def quantile(
        self,
        column: str,
        q: float | Sequence[float],
        *,
        interpolation: str = "linear",
    ) -> float | list[float]:
        """Quantile(s) of a single column (v0.3.3).

        - q: scalar (returns float) или list (returns list[float]) ∈ [0, 1].
        - interpolation='linear' (pandas default; v0.3.3 only).
        - skipna=True (nulls не участвуют в n_valid и позициях).

        Sort: всегда ascending (квантиль от порядка не зависит).

        Для all-null или empty — NaN (или list of NaN).

        Другие интерполяции ('lower'/'higher'/'nearest'/'midpoint') —
        v0.3.4.
        """
        if column not in self._columns:
            raise KeyError(
                f"quantile: column {column!r} not in columns {self._columns}"
            )
        return _ops_quantile(
            self._iter_filtered_batches(),
            column=column,
            q=q,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "quantile",
            interpolation=interpolation,
            log=False,
        )


    def count(self) -> int:
        """Точное число строк с учётом фильтров (сканирует)."""
        if not self._row_filters:
            return self._dataset.total_rows
        n = 0
        for b in self._iter_filtered_batches():
            n += b.num_rows
        return n

    def to_pylist(self) -> list:
        """Materialize all rows as list of dicts (pa.Table-compatible).

        v0.9.95: matches pa.Table.to_pylist() API.

        Guarded per batch (check_batch extra_factor=3.0) — v0.9.25d
        scan requires every to_pylist site to be guarded.

        WARNING: O(n) RAM. For inspection only — use
        `_iter_filtered_batches()` for streaming consumers.
        """
        mg = self._memory_guard
        out = []
        for b in self._iter_filtered_batches():
            if mg is not None:
                mg.check_batch(b, name="to_pylist", extra_factor=3.0)
            out.extend(b.to_pylist())
        return out

