# cadw/dataframe/__init__.py
# Package facade (v0.9.26y-fix1, post-B.1 split).
# Each re-export routed to actual source module (§13.4/331).

# Primary public surface
from ._core import CadwDataFrame
from ._exports import concat

# Class re-exports
from ._column_proxy import _ColumnProxy
from ._groupby import _GroupBy

# Helpers (v0.9.26w)
from ._helpers import (
    _apply_filter_with_positions,
    _build_out_schema,
    _detect_mode,
    _infer_arrow_type_from_value,
    _normalize_scalar,
)

# Backward-compat re-exports (pre-v0.9.26g surface)
from cadw.core.memory_guard import MemoryGuard
from cadw.core.sampling import data_chars_for_join, peek_first_batch
from cadw.dataframe_helpers import (
    _ArrowTail,
    _EwmState,
    _ExpandingState,
    _RollState,
    _concat_batches,
    _convert_batch,
    _empty_value_counts_table,
    _group_boundaries_arrow,
    _reservoir_positions,
    _weighted_reservoir_positions,
)
from cadw.dataset import (
    CadwIndex,
    Dataset,
    MaskFilter,
    MaterializedIndex,
    MaterializedMultiIndex,
    PredicateFilter,
    RangeIndex,
    RowFilter,
    SliceFilter,
)
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter, ParquetBatchWriter
from cadw.indexers import _AtIndexer, _IAtIndexer, _ILocIndexer, _LocIndexer
from cadw.ops.dedup import drop_duplicates as _ops_drop_duplicates
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.join import hash_join, hash_join_streaming
from cadw.ops.quantile import quantile as _ops_quantile
from cadw.ops.rank import VALID_METHODS, compute_ranks_to_disk
from cadw.ops.resample import parse_freq as _parse_freq
from cadw.ops.resample import resample_to_disk as _resample_to_disk
from cadw.ops.sort import external_sort

__all__ = ["CadwDataFrame", "concat"]
