"""CADW: out-of-core dataframe. См. DECISIONS.md."""
from .core.memory_guard import MemoryGuard, MemoryLimitExceeded
from .dataset import (
    Dataset,
    CadwIndex, RangeIndex, MaterializedIndex, MaterializedMultiIndex,
    RowFilter, PredicateFilter, AndFilter, OrFilter, NotFilter, SliceFilter,
)
from .dataframe import CadwDataFrame
from .io.batch_reader import ParquetBatchReader, ArrowIPCBatchReader
from .io.batch_writer import ParquetBatchWriter, ArrowIPCBatchWriter
from .ops.reduce import (
    reduce, ReduceOp,
    CountOp, SumOp, MinMaxOp, MeanOp, StdOp, CovarianceOp, HistogramOp,
)
from .ops.shuffle import (
    hash_shuffle, compute_hashes, estimate_num_partitions, read_partition_batches,
)
from .ops.sort import external_sort, read_sorted
from .ops.groupby import groupby_aggregate, write_groupby_result
from .ops.dedup import drop_duplicates
from .ops.rank import compute_ranks_to_disk, VALID_METHODS
from .ops.quantile import quantile

__all__ = [
    "CadwIndex",
    "RangeIndex",
    "MaterializedIndex",
    "MaterializedMultiIndex",
    "MemoryGuard", "MemoryLimitExceeded", "Dataset", "CadwDataFrame",
    "RowFilter", "PredicateFilter", "AndFilter", "OrFilter", "NotFilter", "SliceFilter",
    "ParquetBatchReader", "ArrowIPCBatchReader",
    "ParquetBatchWriter", "ArrowIPCBatchWriter",
    "reduce", "ReduceOp",
    "CountOp", "SumOp", "MinMaxOp", "MeanOp", "StdOp",
    "CovarianceOp", "HistogramOp",
    "hash_shuffle", "compute_hashes", "estimate_num_partitions",
    "read_partition_batches",
    "external_sort", "read_sorted",
    "groupby_aggregate", "write_groupby_result", "drop_duplicates", "compute_ranks_to_disk", "VALID_METHODS", "quantile",
]
