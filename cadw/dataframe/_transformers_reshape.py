# cadw/dataframe/_transformers_reshape.py
# v0.9.26p (B.1 step 10): _TransformersReshapeMixin.
# Symbols: astype, drop, rename, drop_duplicates,
#   duplicated, cut, qcut.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import pyarrow as pa
from cadw.dataset import Dataset
from typing import Sequence
from cadw.ops.dedup import drop_duplicates as _ops_drop_duplicates


class _TransformersReshapeMixin:
    """Reshape/dedup/bucketize mixin (v0.9.26p)."""
    def astype(
        self,
        dtype_map: dict,
        *,
        safe: bool = True,
    ) -> "CadwDataFrame":
        """Cast columns to new dtypes. Streaming, no materialization.

        - dtype_map: {col: pa.DataType | str-alias}
        - safe=True: raise on overflow (default). safe=False: wrap/truncate.
        """
        from ._core import CadwDataFrame
        import pyarrow.compute as _pc

        resolved: dict[str, pa.DataType] = {}
        for c, dt in dtype_map.items():
            if c not in self._columns:
                raise KeyError(
                    f"astype: column {c!r} not in {self._columns}"
                )
            if isinstance(dt, pa.DataType):
                resolved[c] = dt
            elif isinstance(dt, str):
                try:
                    resolved[c] = pa.type_for_alias(dt)
                except (ValueError, KeyError) as e:
                    raise ValueError(
                        f"astype: unknown dtype alias {dt!r}: {e}"
                    ) from e
            else:
                raise TypeError(
                    f"astype: dtype for {c!r} must be str or pa.DataType; "
                    f"got {type(dt).__name__}"
                )

        out_fields = []
        for c in self._columns:
            if c in resolved:
                out_fields.append(pa.field(c, resolved[c], nullable=True))
            else:
                out_fields.append(self._dataset.schema.field(c))
        out_schema = pa.schema(out_fields)

        def _iter():
            for b in self._iter_arrow_batches():
                self._memory_guard.check_batch(b, name="astype")
                arrays = []
                for c in self._columns:
                    if c in resolved:
                        arrays.append(_pc.cast(
                            b.column(c), resolved[c], safe=safe,
                        ))
                    else:
                        arrays.append(b.column(c))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def drop(self, cols) -> "CadwDataFrame":
        """Drop columns (selection minus). Lazy."""
        if isinstance(cols, str):
            cols = [cols]
        else:
            cols = list(cols)
        for c in cols:
            if c not in self._columns:
                raise KeyError(f"drop: column {c!r} not in {self._columns}")
        keep = [c for c in self._columns if c not in cols]
        return self[keep]


    def rename(self, mapping: dict[str, str]) -> "CadwDataFrame":
        """Rename columns. Materializes new Dataset (schema change).

        - mapping: {old: new}. All `old` must exist; `new` must not collide
          with an existing column (unless it's the column being renamed).
        """
        from ._core import CadwDataFrame
        if not isinstance(mapping, dict):
            raise TypeError(
                f"rename: mapping must be dict; got {type(mapping).__name__}"
            )
        for old, new in mapping.items():
            if old not in self._columns:
                raise KeyError(
                    f"rename: column {old!r} not in {self._columns}"
                )
            if new != old and new in self._columns:
                raise ValueError(
                    f"rename: target {new!r} collides with existing column"
                )
        new_names = [mapping.get(c, c) for c in self._columns]
        new_fields = []
        for c in self._columns:
            orig = self._dataset.schema.field(c)
            new_fields.append(
                pa.field(mapping.get(c, c), orig.type, nullable=orig.nullable)
            )
        new_schema = pa.schema(new_fields)

        def _iter():
            for b in self._iter_arrow_batches():
                yield b.rename_columns(new_names)

        new_ds = Dataset.from_batches(
            new_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def drop_duplicates(
        self,
        subset: str | Sequence[str] | None = None,
        keep: str | bool = "first",
        *,
        num_partitions: int | None = None,
    ) -> "CadwDataFrame":
        """Удалить дубликаты строк (v0.3.0).

        - subset=None: все колонки — как pandas.
        - subset=str | Sequence[str]: ключ(и) для определения дубликата.
        - keep: 'first' | 'last' | False (drop all duplicates).
        - num_partitions: None — auto из sample (по k_est на subset).

        **Порядок результата**: partition-concatenation, **не** pandas-order.
        Если нужен стабильный порядок — `.sort_values(...)` после. §11.5.

        **Skew**: при oversized partition — `MemoryLimitExceeded`;
        recursion не поддерживается в v0.3.0. Увеличьте `num_partitions`.
        """
        if keep not in ("first", "last", False):
            raise ValueError(
                f"drop_duplicates: keep must be 'first', 'last', or False; "
                f"got {keep!r}"
            )
        if subset is None:
            subset_cols = list(self._columns)
        elif isinstance(subset, str):
            subset_cols = [subset]
        else:
            subset_cols = list(subset)
        if not subset_cols:
            raise ValueError("drop_duplicates: subset must be non-empty")
        for c in subset_cols:
            if c not in self._columns:
                raise KeyError(
                    f"drop_duplicates: subset column {c!r} not in "
                    f"columns {self._columns}"
                )
        # subset=None означает dedup по всем; subset явный — dedup по списку,
        # но output сохраняет все колонки. Проверим совпадение с columns
        # для случая subset=None.
        if subset is None:
            subset_cols = list(self._columns)

        result = _ops_drop_duplicates(
            self._iter_filtered_batches(),
            subset=subset_cols,
            keep=keep,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "dedup",
            total_rows=self._dataset.total_rows,
            num_partitions=num_partitions,
            log=False,
        )
        return self._wrap_result_table(result)


    def duplicated(self, subset=None, keep="first") -> "CadwDataFrame":
        """Mark duplicate rows (v0.9.25j).

        Returns augmented CadwDataFrame: original columns + `duplicated`
        bool column (nullable=True). Row order = current filtered
        view order.

        - subset: None → all columns. str → one. Sequence[str] → subset.
        - keep: 'first' (default) → first occurrence per key is
          False (not duplicate); rest True.
          'last' → last occurrence False; earlier True.
          False → all rows in duplicate groups (count>1) True;
          unique-key rows False.

        NULL in subset: null == null for grouping (own group).

        Implementation: (1) HashShuffle + dedup on (subset, __rid)
        to find kept row_ids; (2) per-batch mask via
        pc.is_in(rid, kept_set); mask = NOT is_in.

        Peak RAM = partition_size (bounded by existing shuffle guard)
        + O(K_kept × 8 × 2) hash set for kept_rids. K_kept ≤ N_unique
        keys; guard fails fast if exceeds state_limit.

        Use to filter df: `df2 = df.duplicated(subset=["k"]);
        filtered = df2[df2["duplicated"] == True]`.
        Or extract mask: `df.duplicated()["duplicated"].to_list()`.
        """
        from ._core import CadwDataFrame
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset
        from cadw.ops.dedup import drop_duplicates as _ops_dedup

        if keep not in ("first", "last", False):
            raise ValueError(
                f"duplicated: keep must be 'first', 'last', or False; "
                f"got {keep!r}"
            )
        if subset is None:
            subset_cols = list(self._columns)
        elif isinstance(subset, str):
            subset_cols = [subset]
        else:
            subset_cols = list(subset)
        if not subset_cols:
            raise ValueError("duplicated: subset must be non-empty")
        for c in subset_cols:
            if c not in self._columns:
                raise KeyError(
                    f"duplicated: subset column {c!r} not in "
                    f"{self._columns}"
                )

        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field("duplicated", pa.bool_(), nullable=True))
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        # Pass 1: HashShuffle + dedup on (subset, __rid) → kept __rids
        def _subset_with_rid():
            for b, positions in self._iter_filtered_batches_with_positions():
                self._memory_guard.check_batch(
                    b, name="duplicated.shuffle", extra_factor=2.0,
                )
                sub_b = b.select(subset_cols)
                rid = pa.array(positions, type=pa.uint64())
                yield sub_b.append_column("__rid", rid)

        kept_table = _ops_dedup(
            _subset_with_rid(),
            subset=subset_cols,
            keep=keep,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "duplicated",
            total_rows=self._dataset.total_rows,
            log=False,
        )
        # v0.9.95: _ops_dedup returns CadwDataFrame; extract "__rid"
        # as a single pa.Array (concat across batches if multi).
        _rid_parts = [b.column("__rid") for b in
                      kept_table._iter_filtered_batches()]
        if _rid_parts:
            # pa.concat_arrays returns Array (not ChunkedArray);
            # .combine_chunks() is ChunkedArray method — not needed.
            kept_rids = pa.concat_arrays(_rid_parts)
        else:
            kept_rids = pa.array([], type=pa.uint64())
        _kept_count = kept_table.count()

        # Guard: kept_rids hash set must fit budget (~2x raw bytes)
        kept_est = max(1, kept_rids.nbytes) * 2
        self._memory_guard.check_state(
            kept_est,
            name=f"duplicated.kept_set(K_kept={_kept_count})",
        )

        # Pass 2: mask = NOT is_in(rid, kept_rids) per batch
        def _iter_augmented():
            for b, positions in self._iter_filtered_batches_with_positions():
                self._memory_guard.check_batch(
                    b, name="duplicated.mask", extra_factor=2.5,
                )
                rid = pa.array(positions, type=pa.uint64())
                is_kept = _pc.is_in(rid, value_set=kept_rids)
                dup = _pc.invert(is_kept)
                arrays = [b.column(c) for c in self._columns]
                arrays.append(dup)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_augmented(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def cut(
        self, column, bins, *, right: bool = True,
        include_lowest: bool = True,
    ):
        """Bin numeric column into discrete intervals (v0.9.25j).

        Args:
            column: numeric column to bin.
            bins: int → k equal-width bins spanning [min, max].
                list[float] → explicit edge values (ascending,
                len >= 2).
            right: True (default) → intervals are (a, b], matching
                pandas `cut(right=True)` default.
            include_lowest: True (default, CADW choice — DIFFERS from
                pandas) → lowest edge is included. Value == min goes
                to bin 0 instead of null. Implemented via
                np.nextafter(min, -inf) as first edge.

        Returns:
            (CadwDataFrame, edges: list[float]): augmented df with
            `{column}_bin` (int32 nullable) appended; edges list.

        Labels: 0..k-1. Out-of-range (below min_edge or above
        max_edge) → null. Null input → null label.

        Peak RAM = batch + O(k). Independent of N.
        """
        from ._core import CadwDataFrame
        import numpy as np
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset

        if not isinstance(right, bool):
            raise TypeError(
                f"cut: right must be bool; got {type(right).__name__}"
            )
        if not isinstance(include_lowest, bool):
            raise TypeError(
                f"cut: include_lowest must be bool; got "
                f"{type(include_lowest).__name__}"
            )
        if column not in self._columns:
            raise KeyError(
                f"cut: column {column!r} not in {self._columns}"
            )
        col_type = self._dataset.schema.field(column).type
        if not (
            pa.types.is_integer(col_type) or pa.types.is_floating(col_type)
        ):
            raise TypeError(
                f"cut: column {column!r} has type {col_type}; "
                f"numeric required"
            )

        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field(f"{column}_bin", pa.int32(), nullable=True))
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            ), []

        # Compute edges
        if isinstance(bins, int):
            if bins < 1:
                raise ValueError(
                    f"cut: bins (int) must be >= 1; got {bins}"
                )
            from cadw.ops.reduce import MinMaxOp, reduce
            min_val = reduce(
                self._iter_filtered_batches(),
                MinMaxOp(column, "min"),
                self._memory_guard, log=False,
            )
            max_val = reduce(
                self._iter_filtered_batches(),
                MinMaxOp(column, "max"),
                self._memory_guard, log=False,
            )
            if min_val is None or max_val is None:
                raise ValueError(
                    f"cut: column {column!r} has no non-null values"
                )
            min_f = float(min_val)
            max_f = float(max_val)
            if min_f == max_f:
                edges = [min_f, max_f]
            else:
                width = (max_f - min_f) / bins
                edges = [min_f + i * width for i in range(bins + 1)]
        else:
            edges = [float(b) for b in bins]
            if len(edges) < 2:
                raise ValueError(
                    f"cut: bins list must have >= 2 edges; got "
                    f"{len(edges)}"
                )
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    raise ValueError(
                        f"cut: bins must be strictly ascending; got "
                        f"{edges} at index {i}"
                    )

        edges_np = np.asarray(edges, dtype=np.float64)
        k_bins = len(edges) - 1

        # include_lowest: extend first edge to just below (via
        # np.nextafter) so that value == min goes to bucket 1 (bin 0)
        if include_lowest and len(edges_np) >= 2:
            edges_used = edges_np.copy()
            edges_used[0] = np.nextafter(edges_used[0], -np.inf)
        else:
            edges_used = edges_np

        def _iter():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name="cut.input", extra_factor=3.0,
                )
                arr = b.column(column)
                n = b.num_rows
                if n == 0:
                    yield b.select(self._columns)
                    continue
                null_np = (
                    arr.is_null().to_numpy(zero_copy_only=False)
                    if arr.null_count > 0 else None
                )
                arr_f64 = _pc.cast(arr, pa.float64())
                if arr.null_count > 0:
                    safe = _pc.fill_null(arr_f64, float("-inf"))
                else:
                    safe = arr_f64
                arr_np = safe.to_numpy(zero_copy_only=False)
                labels_np = (
                    np.digitize(arr_np, edges_used, right=right) - 1
                ).astype(np.int32)
                oob = (labels_np < 0) | (labels_np >= k_bins)
                if null_np is not None:
                    oob = oob | null_np
                if oob.any():
                    label_arr = pa.array(
                        labels_np, type=pa.int32(), mask=oob,
                    )
                else:
                    label_arr = pa.array(labels_np, type=pa.int32())
                arrays = [b.column(c) for c in self._columns]
                arrays.append(label_arr)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        ), edges


    def qcut(
        self, column, q, *, right: bool = True,
        include_lowest: bool = True,
    ):
        """Quantile-based binning of a numeric column (v0.9.25j).

        Args:
            column: numeric column to bin.
            q: int k → k equal-frequency bins (quantiles at
                [0, 1/k, ..., 1]). list[float] in (0, 1) → interior
                boundaries; edges = [min] + quantile(q) + [max].
            right: True (default) → intervals are (a, b].
            include_lowest: True (default) → lowest edge included.

        Returns: (CadwDataFrame, edges: list[float]).

        If quantile computation produces tied edges (insufficient
        distinct values in data), edges are deduped; fewer bins than
        requested. Raises ValueError if all values identical.

        Peak RAM = batch + O(N log N) I/O for external_sort
        (quantile). Independent of N for RAM.
        """
        from ._core import CadwDataFrame
        import numpy as np
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset

        if not isinstance(right, bool):
            raise TypeError(
                f"qcut: right must be bool; got {type(right).__name__}"
            )
        if not isinstance(include_lowest, bool):
            raise TypeError(
                f"qcut: include_lowest must be bool; got "
                f"{type(include_lowest).__name__}"
            )
        if column not in self._columns:
            raise KeyError(
                f"qcut: column {column!r} not in {self._columns}"
            )
        col_type = self._dataset.schema.field(column).type
        if not (
            pa.types.is_integer(col_type) or pa.types.is_floating(col_type)
        ):
            raise TypeError(
                f"qcut: column {column!r} has type {col_type}; "
                f"numeric required"
            )

        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field(f"{column}_bin", pa.int32(), nullable=True))
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            ), []

        # Build q positions (full range with 0.0 and 1.0)
        if isinstance(q, int):
            if q < 1:
                raise ValueError(
                    f"qcut: q (int) must be >= 1; got {q}"
                )
            q_positions = [i / q for i in range(q + 1)]
        elif isinstance(q, (list, tuple)):
            q_list = [float(v) for v in q]
            if not q_list:
                raise ValueError("qcut: q list must be non-empty")
            for qi in q_list:
                if not (0.0 <= qi <= 1.0):
                    raise ValueError(
                        f"qcut: q positions must be in [0, 1]; "
                        f"got {qi}"
                    )
            q_positions = [0.0] + q_list + [1.0]
        else:
            raise TypeError(
                f"qcut: q must be int or list[float]; got "
                f"{type(q).__name__}"
            )

        # Compute edges via quantile (uses external_sort for N>batch)
        edges = list(self.quantile(column, q_positions))
        # Handle NaN (all-null column)
        for e in edges:
            if e != e:  # NaN check
                raise ValueError(
                    f"qcut: column {column!r} has no non-null values"
                )

        # Dedupe tied edges (§13.4: quantile can tie on low-card data)
        unique_edges = sorted(set(edges))
        if len(unique_edges) < 2:
            _first_or_empty = (
                unique_edges[0] if unique_edges else "empty"
            )
            raise ValueError(
                f"qcut: all values identical ({_first_or_empty}); "
                f"cannot create bins"
            )
        if len(unique_edges) < len(edges):
            # Some quantile positions tied; use unique edges
            import logging
            logging.getLogger("cadw.dataframe").warning(
                "qcut: quantile edges contained ties (insufficient "
                "distinct values); %d unique edges instead of %d",
                len(unique_edges), len(edges),
            )
            edges = unique_edges

        edges_np = np.asarray(edges, dtype=np.float64)
        k_bins = len(edges) - 1

        if include_lowest and len(edges_np) >= 2:
            edges_used = edges_np.copy()
            edges_used[0] = np.nextafter(edges_used[0], -np.inf)
        else:
            edges_used = edges_np

        def _iter():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name="qcut.input", extra_factor=3.0,
                )
                arr = b.column(column)
                n = b.num_rows
                if n == 0:
                    yield b.select(self._columns)
                    continue
                null_np = (
                    arr.is_null().to_numpy(zero_copy_only=False)
                    if arr.null_count > 0 else None
                )
                arr_f64 = _pc.cast(arr, pa.float64())
                if arr.null_count > 0:
                    safe = _pc.fill_null(arr_f64, float("-inf"))
                else:
                    safe = arr_f64
                arr_np = safe.to_numpy(zero_copy_only=False)
                labels_np = (
                    np.digitize(arr_np, edges_used, right=right) - 1
                ).astype(np.int32)
                oob = (labels_np < 0) | (labels_np >= k_bins)
                if null_np is not None:
                    oob = oob | null_np
                if oob.any():
                    label_arr = pa.array(
                        labels_np, type=pa.int32(), mask=oob,
                    )
                else:
                    label_arr = pa.array(labels_np, type=pa.int32())
                arrays = [b.column(c) for c in self._columns]
                arrays.append(label_arr)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        ), edges

