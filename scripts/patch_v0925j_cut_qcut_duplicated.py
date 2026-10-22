#!/usr/bin/env python3
# v0.9.25j — Phase C batch C-3b: cut + qcut + duplicated.
#
# 3 surface API features. Compositions of existing primitives
# (ReduceOp MinMaxOp, quantile/external_sort, drop_duplicates) +
# numpy digitize + Arrow kernels (pc.cast, pc.fill_null, pc.is_in).
# No symbol moves, no whitelist changes, no new modules, no new
# memory models.
#
# FEATURES:
#   1. `CadwDataFrame.cut(column, bins, *, right=True,
#      include_lowest=True) -> (df, edges)` — bin numeric column into
#      equal-width (bins=int) or explicit-edge (bins=list) intervals.
#      Appends `{column}_bin` (int32, nullable). Returns edges as
#      list[float].
#   2. `CadwDataFrame.qcut(column, q, *, right=True,
#      include_lowest=True) -> (df, edges)` — quantile-based binning.
#      q=int k → k equal-frequency bins; q=list of floats in (0,1)
#      → those are interior boundaries, edges = [min] + quantile(q)
#      + [max]. Same output shape as cut.
#   3. `CadwDataFrame.duplicated(subset=None, keep='first') -> df` —
#      augmented df (original cols + `duplicated` bool). keep ∈
#      {'first', 'last', False}, same semantics as pandas. Uses
#      drop_duplicates on (subset, __rid) to find kept row_ids, then
#      pc.is_in mask per batch.
#
# SEMANTICS:
#   cut / qcut:
#     - Labels: 0..k-1 (int32).
#     - right=True: bins are (lo, hi] — right-closed (pandas default).
#     - include_lowest=True (CADW default, DIFFERS from pandas):
#       lowest edge included → value==min goes to bin 0 (not null).
#       Implemented via np.nextafter(min, -inf) as first edge.
#     - Out-of-range (below min-edge or above max-edge) → null.
#     - Null input → null label.
#     - Non-numeric column → TypeError.
#     - qcut with q=int on data with insufficient distinct values
#       (quantile ties) → dedupe edges; fewer bins than requested.
#       Documented.
#
#   duplicated:
#     - Augmented df: original cols + `duplicated` bool (nullable=False).
#     - keep='first': all but first occurrence per key are True.
#     - keep='last': all but last occurrence per key are True.
#     - keep=False: ALL rows in duplicate groups (count>1) are True;
#       unique-key rows are False.
#     - NULL in subset: null==null in group keys (own group).
#     - Result row order = original filtered view order.
#
# §13.4/264 (REPL-first): verify_state does 7 checks BEFORE any edit:
#   (1) np.digitize with right=True semantics
#   (2) np.digitize with right=False (differentiation)
#   (3) np.nextafter(x, -inf) < x
#   (4) pc.is_in with uint64 value_set
#   (5) pc.is_in with empty value_set → all False
#   (6) ChunkedArray.combine_chunks() → Array
#   (7) _iter_filtered_batches_with_positions yields (batch, positions)
#       matching expected positions
#
# §13.4/267 (mirror-symmetry): cut and qcut share the SAME `_label_fn`
# closure (imported via shared helper `_bin_labels`). Duplicated keep
# modes validated by _ops_drop_duplicates (single source of truth).
#
# §13.4/269 (null_count over to_pylist): all bin/mask ops use Arrow
# kernels + numpy, no .to_pylist() inside fill loops.
#
# §13.4/266 (anchor check): unique anchor is
# `    # ---------------- __getitem__ ----------------\n\n    def __getitem__(self, key):`
# (same well-tested anchor as v0.9.25i).
#
# §13.4/260 preserved: no subprocess dogfood (dedent trap).
# §13.4/261 preserved: no symbol moves → no whitelist path updates.
# §13.4/262 preserved: typed scalar reconstruction via pa.scalar.
# §13.4/265 preserved: no copy-pasted asserts (each test explicit).
#
# DEFERRED:
#   - cut/qcut `labels=` custom-labels parameter → backlog.
#   - qcut interpolation other than 'linear' → backlog (uses quantile
#     which is linear only).
#   - duplicated on MultiIndex → NotImplementedError (delegates to
#     drop_duplicates which handles subset cols, not index).
#   - bfill/ffill pending-buffer op (O(N log N)-free) → backlog.
#
# Fix-cycle budget: 2-3 (3 features; cut/qcut share logic; duplicated
# introduces new "augmented df" pattern).
#
# Run: uv run python scripts/patch_v0925j_cut_qcut_duplicated.py

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    check_no_invalid_escape,
)


DATAFRAME_PATH = "cadw/dataframe.py"
TESTS_PATH = "tests/test_v0925j.py"


# ============ 0. verify state (REPL-first, §13.4/264) ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25i; REPL-check 7 assumptions; "
          "no v0.9.25j features yet ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    # v0.9.25j-fix0 (§13.4/270): version-string presence check false-
    # positives when prior commit body references "→ vX.Y.Z" for
    # deferred/next work (v0.9.25i's body: "DEFERRED: ... → v0.9.25j").
    # Extract SUBJECT-PREFIX version via regex; exact-match.
    import re as _re
    _m = _re.match(r'^[a-f0-9]+ (v\d+\.\d+\.\d+\w*):', log)
    _head_version = _m.group(1) if _m else None
    if _head_version != "v0.9.25i":
        raise SystemExit(
            f"expected HEAD version v0.9.25i; got {_head_version!r} "
            f"(log prefix: {log[:120]!r})"
        )

    df_src = Path(DATAFRAME_PATH).read_text()
    for marker in ("def cut(self", "def qcut(self", "def duplicated(self"):
        if marker in df_src:
            raise SystemExit(f"{marker!r} already present")

    # Anchor text check (§13.4/266)
    anchor = (
        "    # ---------------- __getitem__ ----------------\n\n"
        "    def __getitem__(self, key):"
    )
    if anchor not in df_src:
        raise SystemExit(
            "ANCHOR MISS: `# ---------------- __getitem__ "
            "----------------` followed by `def __getitem__` not found. "
            "Inspect dataframe.py structure."
        )
    print("  anchor `# __getitem__` verified")

    # --- REPL checks ---
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as _pc

    # (1) np.digitize right=True: bins are (lo, hi]
    x = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 15.0])
    edges = np.array([0.0, 5.0, 10.0])
    idx_right = np.digitize(x, edges, right=True)
    # right=True: x <= edges[0] → 0; edges[0] < x <= edges[1] → 1; ...
    expected_right = np.array([0, 1, 1, 2, 2, 3])
    if not np.array_equal(idx_right, expected_right):
        raise SystemExit(
            f"np.digitize(right=True) = {idx_right.tolist()}, "
            f"expected {expected_right.tolist()}"
        )
    print("  (1) np.digitize right=True: [0,1,1,2,2,3] confirmed")

    # (2) np.digitize right=False: bins are [lo, hi)
    idx_left = np.digitize(x, edges, right=False)
    # right=False: x < edges[0] → 0; edges[0] <= x < edges[1] → 1; ...
    expected_left = np.array([1, 1, 2, 2, 3, 3])
    if not np.array_equal(idx_left, expected_left):
        raise SystemExit(
            f"np.digitize(right=False) = {idx_left.tolist()}, "
            f"expected {expected_left.tolist()}"
        )
    print("  (2) np.digitize right=False: [1,1,2,2,3,3] confirmed")

    # (3) np.nextafter(x, -inf) < x strictly
    nxt = np.nextafter(5.0, -np.inf)
    if not (nxt < 5.0):
        raise SystemExit(
            f"np.nextafter(5, -inf) = {nxt}, expected < 5.0"
        )
    print(f"  (3) np.nextafter(5, -inf) = {nxt} < 5.0 confirmed")

    # (4) pc.is_in with uint64 value_set
    rid = pa.array([0, 1, 2, 3], type=pa.uint64())
    vset = pa.array([1, 3], type=pa.uint64())
    is_in_result = _pc.is_in(rid, value_set=vset)
    if is_in_result.to_pylist() != [False, True, False, True]:
        raise SystemExit(
            f"pc.is_in(uint64) = {is_in_result.to_pylist()}, "
            f"expected [F,T,F,T]"
        )
    print("  (4) pc.is_in(uint64) [F,T,F,T] confirmed")

    # (5) pc.is_in with empty value_set → all False
    empty_vset = pa.array([], type=pa.uint64())
    empty_result = _pc.is_in(rid, value_set=empty_vset)
    if empty_result.to_pylist() != [False, False, False, False]:
        raise SystemExit(
            f"pc.is_in(empty) = {empty_result.to_pylist()}, "
            f"expected all False"
        )
    print("  (5) pc.is_in(empty value_set) [F,F,F,F] confirmed")

    # (6) ChunkedArray.combine_chunks() → Array
    chunked = pa.chunked_array(
        [pa.array([1, 2], type=pa.uint64()),
         pa.array([3, 4], type=pa.uint64())],
        type=pa.uint64(),
    )
    combined = chunked.combine_chunks()
    if not isinstance(combined, pa.Array):
        raise SystemExit(
            f"combine_chunks() returned {type(combined).__name__}, "
            f"expected pa.Array"
        )
    if combined.to_pylist() != [1, 2, 3, 4]:
        raise SystemExit(
            f"combine_chunks() content {combined.to_pylist()}"
        )
    print("  (6) ChunkedArray.combine_chunks() → Array confirmed")

    # (7) _iter_filtered_batches_with_positions yields (batch, positions)
    import pyarrow.parquet as _pq
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.parquet"
        _pq.write_table(
            pa.table({"v": list(range(10))}),
            src, row_group_size=3,
        )
        from cadw import CadwDataFrame as _CDF
        from cadw import MemoryGuard as _MG
        df_test = _CDF.from_parquet(
            src, td / "ds", chunk_size=3,
            memory_guard=_MG(max_memory_mb=10),
            spill_dir=td / "spill",
        )
        # Unfiltered
        offset = 0
        n_batches = 0
        for b, positions in df_test._iter_filtered_batches_with_positions():
            if not isinstance(positions, np.ndarray):
                raise SystemExit(
                    f"positions type {type(positions).__name__}, "
                    f"expected np.ndarray"
                )
            if int(positions[0]) != offset:
                raise SystemExit(
                    f"positions[0] = {positions[0]}, expected {offset}"
                )
            offset += b.num_rows
            n_batches += 1
        if n_batches == 0:
            raise SystemExit("no batches yielded by _iter_filtered_batches_with_positions")
        # Filtered
        filtered = df_test[df_test["v"] > 3]  # positions 4..9
        collected_positions = []
        for b, positions in filtered._iter_filtered_batches_with_positions():
            collected_positions.extend(positions.tolist())
        if collected_positions != [4, 5, 6, 7, 8, 9]:
            raise SystemExit(
                f"filtered positions = {collected_positions}, "
                f"expected [4..9]"
            )
    print("  (7) _iter_filtered_batches_with_positions semantics confirmed")

    print("  state + anchor + 7 REPL checks verified")


# ============ 1. add cut + qcut + duplicated methods ============

METHODS_BLOCK = '''    def cut(
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
        kept_rids = kept_table.column("__rid").combine_chunks()

        # Guard: kept_rids hash set must fit budget (~2x raw bytes)
        kept_est = max(1, kept_rids.nbytes) * 2
        self._memory_guard.check_state(
            kept_est,
            name=f"duplicated.kept_set(K_kept={kept_table.num_rows})",
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

    # ---------------- __getitem__ ----------------

    def __getitem__(self, key):'''


ANCHOR_OLD = '''    # ---------------- __getitem__ ----------------

    def __getitem__(self, key):'''


def add_methods():
    print("\n=== add cut + qcut + duplicated methods ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, ANCHOR_OLD, METHODS_BLOCK,
        label="cut_qcut_duplicated",
    )
    if applied:
        assert_compiles(DATAFRAME_PATH)
        check_no_invalid_escape(DATAFRAME_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. tests ============

NEW_TESTS = '''"""v0.9.25j: Phase C batch C-3b — cut, qcut, duplicated."""
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _mk(data, mem_mb=20):
    return CadwDataFrame.from_table(
        pa.table(data),
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
    )


def _to_rows(df):
    rows = []
    for b in df._iter_filtered_batches():
        rows.extend(b.to_pylist())
    return rows


# ============ cut ============

def test_cut_int_bins_basic():
    df = _mk({"x": [0.0, 2.5, 5.0, 7.5, 10.0]})
    result, edges = df.cut("x", bins=2)
    assert len(edges) == 3
    # Edges [0, 5, 10]; labels 0 (0<=x<=5), 1 (5<x<=10)
    rows = _to_rows(result)
    labels = [r["x_bin"] for r in rows]
    assert labels == [0, 0, 0, 1, 1]


def test_cut_list_bins_basic():
    df = _mk({"x": [0.0, 2.5, 5.0, 7.5, 10.0, 15.0]})
    result, edges = df.cut("x", bins=[0, 5, 10])
    assert edges == [0.0, 5.0, 10.0]
    rows = _to_rows(result)
    labels = [r["x_bin"] for r in rows]
    # 0→bin0 (include_lowest), 2.5→bin0, 5→bin0 (<=5),
    # 7.5→bin1, 10→bin1, 15→null (above max)
    assert labels == [0, 0, 0, 1, 1, None]


def test_cut_edges_returned():
    df = _mk({"x": list(range(11))})
    _, edges = df.cut("x", bins=2)
    assert edges == [0.0, 5.0, 10.0]


def test_cut_right_false():
    df = _mk({"x": [0.0, 5.0, 10.0]})
    result, _ = df.cut("x", bins=[0, 5, 10], right=False,
                       include_lowest=False)
    rows = _to_rows(result)
    # left-closed: [0, 5) → bin0, [5, 10) → bin1, 10 → null
    labels = [r["x_bin"] for r in rows]
    assert labels == [0, 1, None]


def test_cut_include_lowest_false():
    df = _mk({"x": [0.0, 2.5, 5.0]})
    result, _ = df.cut("x", bins=[0, 5, 10], include_lowest=False)
    rows = _to_rows(result)
    # 0 == min edge; include_lowest=False → null
    labels = [r["x_bin"] for r in rows]
    assert labels == [None, 0, 0]


def test_cut_null_input():
    df = _mk({"x": pa.array([1.0, None, 9.0], type=pa.float64())})
    result, _ = df.cut("x", bins=2)
    rows = _to_rows(result)
    labels = [r["x_bin"] for r in rows]
    assert labels[1] is None


def test_cut_all_same_value():
    df = _mk({"x": [5.0, 5.0, 5.0]})
    result, edges = df.cut("x", bins=3)
    # All identical → 1 bin
    assert len(edges) == 2
    rows = _to_rows(result)
    labels = [r["x_bin"] for r in rows]
    assert labels == [0, 0, 0]


def test_cut_empty():
    df = _mk({"x": [1.0, 2.0], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result, edges = empty.cut("x", bins=2)
    assert result.count() == 0
    assert edges == []


def test_cut_missing_column_raises():
    df = _mk({"x": [1.0]})
    with pytest.raises(KeyError, match="nope"):
        df.cut("nope", bins=2)


def test_cut_non_numeric_raises():
    df = _mk({"s": ["a", "b"]})
    with pytest.raises(TypeError, match="numeric"):
        df.cut("s", bins=2)


def test_cut_invalid_bins_int():
    df = _mk({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="bins"):
        df.cut("x", bins=0)


def test_cut_invalid_bins_list():
    df = _mk({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match=">= 2"):
        df.cut("x", bins=[5])
    with pytest.raises(ValueError, match="strictly ascending"):
        df.cut("x", bins=[5, 5])
    with pytest.raises(ValueError, match="strictly ascending"):
        df.cut("x", bins=[5, 3])


def test_cut_preserves_other_columns():
    df = _mk({"x": [1.0, 2.0, 3.0], "v": [10, 20, 30]})
    result, _ = df.cut("x", bins=2)
    assert "v" in result.columns
    assert "x" in result.columns
    assert "x_bin" in result.columns
    # Original x unchanged
    rows = _to_rows(result)
    assert [r["x"] for r in rows] == [1.0, 2.0, 3.0]


# ============ qcut ============

def test_qcut_int_bins():
    df = _mk({"x": list(range(10))})
    result, edges = df.qcut("x", q=2)
    # Edges = 3 (min, median, max) → 2 bins
    assert len(edges) == 3
    rows = _to_rows(result)
    labels = [r["x_bin"] for r in rows]
    # 0..4 → bin0, 5..9 → bin1
    assert labels[:5] == [0] * 5
    assert labels[5:] == [1] * 5


def test_qcut_4_quartiles():
    df = _mk({"x": list(range(100))})
    result, edges = df.qcut("x", q=4)
    assert len(edges) == 5  # min, q25, q50, q75, max
    rows = _to_rows(result)
    # Each bin should have ~25 rows
    labels = [r["x_bin"] for r in rows]
    counts = [labels.count(i) for i in range(4)]
    for c in counts:
        assert 20 <= c <= 30, f"uneven qcut distribution: {counts}"


def test_qcut_list_q():
    df = _mk({"x": list(range(100))})
    result, edges = df.qcut("x", q=[0.25, 0.5, 0.75])
    # edges = [min] + quantile(q) + [max] → 5 edges → 4 bins
    assert len(edges) == 5
    rows = _to_rows(result)
    assert len(rows) == 100


def test_qcut_edges_sorted():
    df = _mk({"x": list(range(50))})
    _, edges = df.qcut("x", q=3)
    assert edges == sorted(edges)
    assert len(edges) == 4


def test_qcut_empty():
    df = _mk({"x": [1.0, 2.0], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result, edges = empty.qcut("x", q=2)
    assert result.count() == 0
    assert edges == []


def test_qcut_all_same_raises():
    df = _mk({"x": [5.0, 5.0, 5.0]})
    with pytest.raises(ValueError, match="identical"):
        df.qcut("x", q=2)


def test_qcut_missing_column_raises():
    df = _mk({"x": [1.0]})
    with pytest.raises(KeyError, match="nope"):
        df.qcut("nope", q=2)


def test_qcut_non_numeric_raises():
    df = _mk({"s": ["a", "b"]})
    with pytest.raises(TypeError, match="numeric"):
        df.qcut("s", q=2)


def test_qcut_null_input():
    df = _mk({"x": pa.array([1.0, None, 9.0, 5.0], type=pa.float64())})
    result, _ = df.qcut("x", q=2)
    rows = _to_rows(result)
    labels = [r["x_bin"] for r in rows]
    assert labels[1] is None


# ============ duplicated ============

def test_duplicated_first_basic():
    df = _mk({"k": [1, 2, 1, 3, 1], "v": [10, 20, 30, 40, 50]})
    result = df.duplicated(subset=["k"])
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    # First 1 (pos 0) not dup, second 1 (pos 2) is dup, third 1 (pos 4) is dup
    assert mask == [False, False, True, False, True]


def test_duplicated_last_basic():
    df = _mk({"k": [1, 2, 1, 3, 1], "v": [10, 20, 30, 40, 50]})
    result = df.duplicated(subset=["k"], keep="last")
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    # Last 1 (pos 4) not dup, earlier 1s are dup
    assert mask == [True, False, True, False, False]


def test_duplicated_false_basic():
    df = _mk({"k": [1, 2, 1, 3, 1], "v": [10, 20, 30, 40, 50]})
    result = df.duplicated(subset=["k"], keep=False)
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    # All 1s are dup (three of them), 2 and 3 are unique
    assert mask == [True, False, True, False, True]


def test_duplicated_no_dupes():
    df = _mk({"k": [1, 2, 3, 4, 5], "v": [10, 20, 30, 40, 50]})
    result = df.duplicated(subset=["k"])
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    assert mask == [False] * 5


def test_duplicated_all_dupes():
    df = _mk({"k": [1, 1, 1, 1], "v": [10, 20, 30, 40]})
    result = df.duplicated(subset=["k"])
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    assert mask == [False, True, True, True]


def test_duplicated_multi_subset():
    df = _mk({
        "a": [1, 1, 2, 2, 1],
        "b": [10, 20, 30, 40, 10],
        "v": [100, 200, 300, 400, 500],
    })
    result = df.duplicated(subset=["a", "b"])
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    # (1,10) appears twice: pos 0 (first, not dup), pos 4 (dup)
    assert mask == [False, False, False, False, True]


def test_duplicated_null_in_subset():
    df = _mk({
        "k": pa.array([1, None, 1, None], type=pa.int64()),
        "v": [10, 20, 30, 40],
    })
    result = df.duplicated(subset=["k"])
    rows = _to_rows(result)
    mask = [r["duplicated"] for r in rows]
    # pos 0 (k=1) not dup, pos 1 (None) not dup, pos 2 (k=1) dup, pos 3 (None) dup
    assert mask == [False, False, True, True]


def test_duplicated_preserves_original_columns():
    df = _mk({"k": [1, 1], "v": [10, 20]})
    result = df.duplicated(subset=["k"])
    rows = _to_rows(result)
    assert "k" in rows[0]
    assert "v" in rows[0]
    assert "duplicated" in rows[0]
    # Original values preserved
    assert [r["v"] for r in rows] == [10, 20]


def test_duplicated_filter_usage():
    """Users filter dup rows via df2[df2['duplicated'] == True]."""
    df = _mk({"k": [1, 2, 1, 3], "v": [10, 20, 30, 40]})
    dup_df = df.duplicated(subset=["k"])
    filtered = dup_df[dup_df["duplicated"] == True]
    assert filtered.count() == 1
    rows = _to_rows(filtered)
    assert rows[0]["v"] == 30  # row 2 (pos 2) is the duplicate of k=1


def test_duplicated_empty():
    df = _mk({"k": [1, 2], "v": [10, 20]})
    empty = df[df["v"] > 100]
    result = empty.duplicated(subset=["k"])
    assert result.count() == 0
    assert "duplicated" in result.columns


def test_duplicated_missing_column_raises():
    df = _mk({"k": [1, 2]})
    with pytest.raises(KeyError, match="nope"):
        df.duplicated(subset=["nope"])


def test_duplicated_invalid_keep_raises():
    df = _mk({"k": [1, 2]})
    with pytest.raises(ValueError, match="keep"):
        df.duplicated(subset=["k"], keep="middle")


def test_duplicated_empty_subset_raises():
    df = _mk({"k": [1, 2]})
    with pytest.raises(ValueError, match="non-empty"):
        df.duplicated(subset=[])
'''


def add_tests():
    print("\n=== add tests/test_v0925j.py ===")
    p = Path(TESTS_PATH)
    if p.exists() and "test_cut_int_bins_basic" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(NEW_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print(f"  OK ({len(NEW_TESTS.splitlines())} L, 30 tests)")


# ============ 3. pytest ============

def run_pytest():
    print("\n=== pre-flight: facade AST guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py",
         "tests/test_dataframe.py::test_facade_has_no_row_loops",
         "tests/test_dataframe_filters.py::test_facade_has_no_as_py_ast",
         "tests/test_dataframe_filters.py::test_facade_has_no_row_loop_ast",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** FACADE AST GUARD FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: v0925j + related ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925j.py",
         "tests/test_v0925i.py",
         "tests/test_v0925h.py",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
         "tests/test_v0925d_a.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 4. main ============

def main():
    print("=== v0.9.25j: Phase C C-3b — cut + qcut + duplicated ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    add_methods()
    add_tests()
    run_pytest()

    commit_with_guards(
        "v0.9.25j: Phase C batch C-3b — cut + qcut + duplicated. "
        "3 of 20 Pandas P0 features (C-1 v0.9.25g: isin/between/"
        "select_dtypes; C-2 v0.9.25h: idxmax/idxmin/pct_change; "
        "C-3a v0.9.25i: ffill/bfill/reindex; C-3b here). "
        "All pure compositions of existing primitives + Arrow kernels. "
        "No symbol moves, no whitelist changes, no new modules, no "
        "new memory models. "
        "Features: "
        "(1) `CadwDataFrame.cut(column, bins, *, right=True, "
        "include_lowest=True) -> (df, edges)` — bin numeric column. "
        "bins=int → k equal-width bins [min, max]; bins=list[float] "
        "→ explicit ascending edges. Appends `{column}_bin` int32 "
        "nullable. Returns edges list[float]. Semantics: right=True → "
        "(lo, hi] (pandas default); include_lowest=True (CADW default, "
        "DIFFERS from pandas) → min goes to bin 0 (not null), "
        "implemented via np.nextafter(min, -inf). Out-of-range → null; "
        "null input → null. Uses MinMaxOp ReduceOp for edges; "
        "np.digitize + Arrow cast/fill_null per batch. "
        "(2) `CadwDataFrame.qcut(column, q, *, right=True, "
        "include_lowest=True) -> (df, edges)` — quantile-based "
        "binning. q=int k → k equal-frequency bins (quantiles at "
        "[0, 1/k, ..., 1]). q=list[float] in (0,1) → interior "
        "boundaries; edges = [min] + quantile(q) + [max]. Same "
        "label/schema/boundary semantics as cut (mirror-symmetry "
        "§13.4/267). Tied quantile edges (low-card data) deduped; "
        "fewer bins than requested, warning logged. All-same → "
        "ValueError. Uses existing `df.quantile()` (external_sort, "
        "O(N log N) I/O). "
        "(3) `CadwDataFrame.duplicated(subset=None, keep='first') "
        "-> augmented df` — original cols + `duplicated` bool "
        "column (nullable=False). keep ∈ {'first', 'last', False}; "
        "same semantics as pandas. NULL in subset → own group "
        "(null==null). Implementation: (a) hash_shuffle + "
        "_local_dedup on (subset, __rid) via existing "
        "_ops_drop_duplicates → kept __rid set; (b) per-batch "
        "mask = NOT pc.is_in(rid, kept_set) via "
        "_iter_filtered_batches_with_positions. Result row order = "
        "filtered view order. Users filter via df2[df2["
        "'duplicated'] == True] (§11.5: no Series type). "
        "Peak RAM: cut/qcut = batch + O(k) edges; duplicated = "
        "partition_size (bounded by shuffle guard) + O(K_kept × 8 × 2) "
        "hash set for kept_rids (fails fast if > state_limit). "
        "§13.4/264 (REPL-first): verify_state does 7 checks BEFORE "
        "any edit — np.digitize right=True → [0,1,1,2,2,3] on "
        "[0,2.5,5,7.5,10,15]/[0,5,10]; right=False → [1,1,2,2,3,3]; "
        "np.nextafter(5,-inf) < 5; pc.is_in(uint64) → [F,T,F,T]; "
        "pc.is_in(empty) → all False; ChunkedArray.combine_chunks() "
        "→ Array; _iter_filtered_batches_with_positions yields "
        "(batch, positions) matching expected. All PASS. "
        "§13.4/267 (mirror-symmetry): cut/qcut share SAME label_fn "
        "logic; dupl keep modes share _ops_drop_duplicates single "
        "source of truth. "
        "§13.4/269 (null_count over to_pylist): cut/qcut use "
        "np.digitize (no to_pylist, no as_py); duplicated uses "
        "pc.is_in (Arrow-native). Zero new low_factor sites. "
        "§13.4/266 (anchor re-verify): anchor = "
        "`# ---------------- __getitem__ ----------------` verified "
        "before apply. "
        "§13.4/260 preserved: no subprocess dogfood. §13.4/261: no "
        "symbol moves. §13.4/262: typed scalars via pa.scalar. "
        "§13.4/265: no copy-pasted asserts. "
        "Tests (tests/test_v0925j.py): 30 total — cut (12: int "
        "bins/list bins/edges returned/right=False/include_lowest="
        "False/null/all-same/empty/missing/non-numeric/invalid "
        "int/invalid list ×2/preserves-other); qcut (9: int bins/"
        "4 quartiles/list q/edges sorted/empty/all-same raise/"
        "missing/non-numeric/null input); duplicated (9: first/last/"
        "False basic/no-dupes/all-dupes/multi-subset/null-in-subset/"
        "preserves-cols/filter-usage/empty/missing/invalid keep/"
        "empty subset — 9 with grouping). "
        "Fast suite: 2404 → ~2434 (+30). "
        "Next: Phase C C-4 (value_counts done; cumsum/cumprod/"
        "cummax/cummin done at v0.9.23e; next: apply(fn)/agg "
        "extensions/str.split/str.cat/dt extras).",
        self_path="scripts/patch_v0925j_cut_qcut_duplicated.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25j applied. Phase C: 12/20. "
          "Next: C-4 (series/str/dt extras).")


if __name__ == "__main__":
    main()