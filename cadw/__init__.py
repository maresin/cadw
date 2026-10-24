"""CADW: out-of-core dataframe. См. DECISIONS.md."""
__version__ = "0.9.45"

from .core.memory_guard import MemoryGuard, MemoryLimitExceeded
from .dataset import (
    Dataset,
    CadwIndex, RangeIndex, MaterializedIndex, MaterializedMultiIndex,
    RowFilter, PredicateFilter, AndFilter, OrFilter, NotFilter, SliceFilter,
)
from .dataframe import CadwDataFrame, concat
from .io.batch_reader import ParquetBatchReader, ArrowIPCBatchReader
from .io.batch_writer import ParquetBatchWriter, ArrowIPCBatchWriter
from .ops.reduce import (
    reduce, ReduceOp,
    CountOp, SumOp, MinMaxOp, MeanOp, StdOp, CovarianceOp, HistogramOp,
    NullCountOp,
)
from .ops.metrics import (
    MetricsOp, SUPPORTED_METRICS,
    REGRESSION_METRICS, PROBA_METRICS, CLASSIFICATION_METRICS,
    AUC_METRICS,
)
from .ops.shuffle import (
    hash_shuffle, compute_hashes, estimate_num_partitions, read_partition_batches,
)
from .ops.sort import external_sort, read_sorted
from .ops.groupby import groupby_aggregate, write_groupby_result
from .ops.dedup import drop_duplicates
from .ops.rank import compute_ranks_to_disk, VALID_METHODS
from .ops.quantile import quantile
from .ops.resample import parse_freq, resample_to_disk
from .pipeline import (
    Pipeline,
    CastStep,
    DropnaStep,
    GBDTStep,
    ColumnTransformer,
    Transformer,
    Predictor,
    register_step_type,
    unregister_step_type,
    list_step_types,
)

from .cv import (
    KFold,
    GroupKFold,
    TimeSeriesSplit,
    cv_score,
)

from .scalers import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
)

from .ml import (
    XTXOp, LinearModel, fit_linear,
    fit_lasso, fit_elasticnet,
    LogisticGradOp, LogisticModel, fit_logistic,
    KMeansStepOp, KMeansModel, fit_kmeans,
    PCA, fit_pca,
    GBDTRegressor, fit_gbdt_regressor,
    GBDTClassifier, fit_gbdt_classifier,
    GBDTMulticlassClassifier, fit_gbdt_multiclass,
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
    "GBDTMulticlassClassifier", "fit_gbdt_multiclass",
    "GaussianNB", "fit_gaussian_nb",
    "MultinomialNB", "fit_multinomial_nb",
    "BernoulliNB", "fit_bernoulli_nb",
    "CadwIndex",
    "RangeIndex",
    "MaterializedIndex",
    "MaterializedMultiIndex",
    "MemoryGuard", "MemoryLimitExceeded", "Dataset", "CadwDataFrame", "concat",
    "RowFilter", "PredicateFilter", "AndFilter", "OrFilter", "NotFilter", "SliceFilter",
    "ParquetBatchReader", "ArrowIPCBatchReader",
    "ParquetBatchWriter", "ArrowIPCBatchWriter",
    "reduce", "ReduceOp",
    "CountOp", "SumOp", "MinMaxOp", "MeanOp", "StdOp",
    "CovarianceOp", "HistogramOp", "NullCountOp",
    "hash_shuffle", "compute_hashes", "estimate_num_partitions",
    "read_partition_batches",
    "external_sort", "read_sorted",
    "groupby_aggregate", "write_groupby_result", "drop_duplicates", "compute_ranks_to_disk", "VALID_METHODS", "quantile", "parse_freq", "resample_to_disk",
    # v0.9.1d pipeline exports
    "Pipeline", "CastStep", "DropnaStep", "GBDTStep",
    "ColumnTransformer", "Transformer", "Predictor",
    "register_step_type", "unregister_step_type", "list_step_types",
    # v0.9.2e cross-validation
    "KFold", "GroupKFold", "TimeSeriesSplit", "cv_score",
    # v0.9.2f scalers
    "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler",
]
