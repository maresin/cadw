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

from .ml import (
    XTXOp, LinearModel, fit_linear,
    fit_lasso, fit_elasticnet,
    LogisticGradOp, LogisticModel, fit_logistic,
    KMeansStepOp, KMeansModel, fit_kmeans,
    PCA, fit_pca,
    GBDTRegressor, fit_gbdt_regressor,
    GBDTClassifier, fit_gbdt_classifier,
    GaussianNB, fit_gaussian_nb,
    MultinomialNB, fit_multinomial_nb,
    BernoulliNB, fit_bernoulli_nb,
)

from .viz import (
    hist as viz_hist,
    hist2d as viz_hist2d,
    box_stats as viz_box_stats,
    line_agg as viz_line_agg,
    sample as viz_sample,
)

__all__ = [
    "viz_hist", "viz_hist2d", "viz_box_stats",
    "viz_line_agg", "viz_sample",
    "XTXOp", "LinearModel", "fit_linear",
    "fit_lasso", "fit_elasticnet",
    "LogisticGradOp", "LogisticModel", "fit_logistic",
    "KMeansStepOp", "KMeansModel", "fit_kmeans",
    "PCA", "fit_pca",
    "GBDTRegressor", "fit_gbdt_regressor",
    "GBDTClassifier", "fit_gbdt_classifier",
    "GaussianNB", "fit_gaussian_nb",
    "MultinomialNB", "fit_multinomial_nb",
    "BernoulliNB", "fit_bernoulli_nb",
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
