"""CADW: out-of-core dataframe. См. DECISIONS.md."""
from .core.memory_guard import MemoryGuard, MemoryLimitExceeded
from .dataset import Dataset
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

__all__ = [
    "MemoryGuard", "MemoryLimitExceeded", "Dataset",
    "ParquetBatchReader", "ArrowIPCBatchReader",
    "ParquetBatchWriter", "ArrowIPCBatchWriter",
    "reduce", "ReduceOp",
    "CountOp", "SumOp", "MinMaxOp", "MeanOp", "StdOp",
    "CovarianceOp", "HistogramOp",
    "hash_shuffle", "compute_hashes", "estimate_num_partitions",
    "read_partition_batches",
    "external_sort", "read_sorted",
    "groupby_aggregate", "write_groupby_result",
]
