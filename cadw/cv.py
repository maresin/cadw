"""Cross-validation for out-of-core pipeline (v0.9.2e).

KFold: fold_id = row_id % k (shuffle=False, O(1) per row) or
       rng.permutation(n) (shuffle=True, materialized once).
TimeSeriesSplit: sort by time_col once, then positional fold slices
                 (no future leakage).

Both produce (train_df, test_df) pairs lazily via CadwDataFrame views.

Memory: O(N) int8 for fold_id array (KFold); O(1) state for TSS
        (sorted_df on disk, iloc slices are lazy).
Independent of N except fold_id array (N × 4 bytes for int32).
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa

from cadw.dataset import Dataset
from cadw.dataframe import CadwDataFrame


def _attach_column(
    df: CadwDataFrame, name: str, arrow_array: pa.Array,
) -> CadwDataFrame:
    """Materialize df's filtered view + attach Arrow array as column.

    Caller ensures len(arrow_array) == df.count() over filtered view.
    Errors if iterated rows != len(arrow_array).

    v0.9.2e-fix1 (§13.4/169): ALL output fields forced nullable=True.
    `Dataset.from_batches` reconstructs per-column singleton via
    `pa.RecordBatch.from_arrays([col], names=[field_name])` — always
    nullable=True. If out_schema has nullable=False for ANY field,
    writer target `pa.schema([field])` (nullable=False) mismatches
    singleton's (nullable=True) → ArrowInvalid. Convention: nullable
    =True everywhere, matching from_arrays default (all CADW ops do
    this: shift/diff/rank/expanding all use nullable=True).
    """
    base_cols = list(df._columns)
    out_fields = [
        pa.field(c, df._dataset.schema.field(c).type, nullable=True)
        for c in base_cols
    ]
    out_fields.append(pa.field(name, arrow_array.type, nullable=True))
    out_schema = pa.schema(out_fields)

    expected_n = len(arrow_array)

    def gen():
        off = 0
        for b in df._iter_filtered_batches():
            n_b = b.num_rows
            if off + n_b > expected_n:
                raise RuntimeError(
                    f"_attach_column: iterated rows exceed array length "
                    f"({off + n_b} > {expected_n})"
                )
            col = arrow_array.slice(off, n_b)
            off += n_b
            arrays = [b.column(c) for c in base_cols] + [col]
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)
        if off != expected_n:
            raise RuntimeError(
                f"_attach_column: iterated {off} rows != "
                f"array length {expected_n}"
            )

    new_ds = Dataset.from_batches(out_schema, gen(), df._new_dataset_dir())
    return CadwDataFrame(
        new_ds,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir,
    )


class KFold:
    """K-fold cross-validator for CadwDataFrame.

    shuffle=False (default): fold_id = arange(n) % k — deterministic,
    no RNG, O(1) per row.
    shuffle=True: rng.permutation(n) — materializes fold_id array
    (O(N) int32; ~4 MB for 1M rows). Different seed → different
    assignment.

    Both preserve row order within each fold (filters don't reorder).
    """

    __slots__ = ("n_splits", "shuffle", "random_state")

    def __init__(
        self,
        n_splits: int,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError(f"n_splits must be int >= 1; got {n_splits!r}")
        if not isinstance(shuffle, bool):
            raise TypeError(
                f"shuffle must be bool; got {type(shuffle).__name__}"
            )
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, df: CadwDataFrame,
    ) -> Iterator[tuple[int, CadwDataFrame, CadwDataFrame]]:
        """Yield (fold_i, train_df, test_df) for each fold.

        Yields lazy CadwDataFrame views (fold filter via PredicateFilter).
        Iteration over train/test materializes filtered batches.
        """
        n = df.count()
        if self.n_splits > n:
            raise ValueError(
                f"KFold: n_splits={self.n_splits} > n={n}"
            )

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            perm = rng.permutation(n)
            fold_id = np.empty(n, dtype=np.int32)
            fold_id[perm] = np.arange(n) % self.n_splits
        else:
            fold_id = (np.arange(n) % self.n_splits).astype(np.int32)

        fold_arr = pa.array(fold_id, type=pa.int32())
        augmented = _attach_column(df, "__fold", fold_arr)

        base_cols = list(df.columns)
        for i in range(self.n_splits):
            train = augmented[augmented["__fold"] != i][base_cols]
            test = augmented[augmented["__fold"] == i][base_cols]
            yield i, train, test


class TimeSeriesSplit:
    """Time-series cross-validator (no future leakage).

    Sort by `time_col` once (external_sort), then split sequentially:
      fold 0: train = [0, fs),     test = [fs, 2*fs)
      fold 1: train = [0, 2*fs),   test = [2*fs, 3*fs)
      ...
      fold k-1: train = [0, k*fs), test = [k*fs, n)  (test = remainder)

    fs = n // (n_splits + 1).

    Sort runs once per split() generator instantiation. If you call
    split() twice, you pay for sort twice — cache the sorted_df outside
    if reusing.
    """

    __slots__ = ("n_splits", "time_col")

    def __init__(self, n_splits: int, time_col: str) -> None:
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError(f"n_splits must be int >= 1; got {n_splits!r}")
        if not isinstance(time_col, str) or not time_col:
            raise ValueError(
                f"time_col must be non-empty str; got {time_col!r}"
            )
        self.n_splits = n_splits
        self.time_col = time_col

    def split(
        self, df: CadwDataFrame,
    ) -> Iterator[tuple[int, CadwDataFrame, CadwDataFrame]]:
        n = df.count()
        if n < self.n_splits + 1:
            raise ValueError(
                f"TimeSeriesSplit: n={n} < n_splits+1={self.n_splits + 1}"
            )
        if self.time_col not in df.columns:
            raise KeyError(
                f"TimeSeriesSplit: time_col {self.time_col!r} not in "
                f"{df.columns}"
            )

        sorted_df = df.sort_values(self.time_col, ascending=True)
        fs = n // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = fs * (i + 1)
            test_end = fs * (i + 2) if i < self.n_splits - 1 else n
            train = sorted_df.iloc[0:train_end]
            test = sorted_df.iloc[train_end:test_end]
            yield i, train, test


def cv_score(
    df: CadwDataFrame,
    cv,
    model_factory,
    *,
    X_cols: Sequence[str],
    y_col: str,
    metrics: Sequence[str] | str,
    pred_col: str | None = None,
    **metric_kwargs,
) -> dict:
    """Cross-validate a model over folds.

    Args:
        df: CadwDataFrame with features and target.
        cv: KFold | TimeSeriesSplit (or any object with .split(df)
            yielding (fold_i, train_df, test_df)).
        model_factory: callable() → fresh model. Model must implement
            ``fit(df, target=y_col)`` and ``predict(df) -> CadwDataFrame``
            (with prediction column named ``{target_col}_pred`` —
            GBDTStep/Logistic/Linear convention).
        X_cols: feature columns (used by model internally; passed to
            metrics? No — only y_col matters for metrics). Reserved.
        y_col: target column.
        metrics: list[str] | str — for CadwDataFrame.metrics().
        pred_col: prediction column name in predict output. None →
            f"{y_col}_pred".
        **metric_kwargs: passed to preds.metrics() (e.g., classes,
            average, pos_label, y_proba).

    Returns:
        dict:
            fold_scores: list[{metric: value}, ...]
            mean: {metric: nanmean}
            std:  {metric: nanstd}
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = list(metrics)
    if not metrics:
        raise ValueError("cv_score: metrics list is empty")
    if pred_col is None:
        pred_col = f"{y_col}_pred"

    fold_scores: list[dict] = []
    for fold_i, train, test in cv.split(df):
        model = model_factory()
        model.fit(train, target=y_col)
        preds = model.predict(test)
        # preds has all test columns + pred_col.
        score = preds.metrics(
            y_true=y_col, y_pred=pred_col, metrics=metrics,
            **metric_kwargs,
        )
        fold_scores.append(score)

    if not fold_scores:
        raise ValueError("cv_score: no folds produced scores")

    mean = {
        m: float(np.nanmean([fs[m] for fs in fold_scores]))
        for m in metrics
    }
    std = {
        m: float(np.nanstd([fs[m] for fs in fold_scores]))
        for m in metrics
    }
    return {"fold_scores": fold_scores, "mean": mean, "std": std}
