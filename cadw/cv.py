"""Cross-validation for out-of-core pipeline (v0.9.3a).

KFold fold assignment is **O(batch) state** in the default mode — no
O(N) arrays, no MemoryGuard violations:
  - shuffle=False: fold_id = row_id % k (per batch).
  - shuffle=True:  keyed PRNG splitmix64(row_id, seed) % k (per batch).
    Deterministic per row_id; split-quality (fold sizes ~binomial,
    mean n/k, per-fold variance O(sqrt(n/k))).
  - materialize=True (opt-in): old O(N) fold_id array with
    MemoryGuard.check_state(20·N). Exact-balanced shuffle for small N
    or when exact fold sizes matter.

TimeSeriesSplit: sort by time_col once, then positional fold slices
                 (no future leakage).

§13.4/173: pre-v0.9.3a docstring claimed "O(N) int8 fold_id" — actual
code was int32 fold_id + int64 arange/permutation temporaries
(12-20·N bytes) with NO MemoryGuard check at all. Guaranteed OOM
for N ≫ RAM — the exact scenario CADW exists for.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa

from cadw.dataset import Dataset
from cadw.dataframe import CadwDataFrame


# splitmix64 constants (deterministic avalanche, K1 pattern from
# shuffle.py; here for fold-id derivation).
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_DEFAULT_SEED = 42  # deterministic default for out-of-core reproducibility


def _fold_ids_from_row_ids(
    row_ids: np.ndarray, k: int, seed: int,
) -> np.ndarray:
    """Keyed splitmix64 hash of row_ids modulo k. Vectorized, uint64 math."""
    h = row_ids.astype(np.uint64) * _GOLDEN + np.uint64(seed)
    h = (h ^ (h >> np.uint64(30))) * _M1
    h = (h ^ (h >> np.uint64(27))) * _M2
    h = h ^ (h >> np.uint64(31))
    return (h % np.uint64(k)).astype(np.int32)


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
    =True everywhere.
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


def _attach_column_with_fn(
    df: CadwDataFrame, name: str, out_type: pa.DataType,
    fn,
) -> CadwDataFrame:
    """Streaming per-batch column attach (v0.9.3a, P-CV1).

    `fn(offset, n) -> pa.Array[n]` is called per batch with the running
    row offset. Memory = O(batch), independent of N. No O(N) array.

    All output fields nullable=True (§13.4/169).
    """
    base_cols = list(df._columns)
    out_fields = [
        pa.field(c, df._dataset.schema.field(c).type, nullable=True)
        for c in base_cols
    ]
    out_fields.append(pa.field(name, out_type, nullable=True))
    out_schema = pa.schema(out_fields)

    def gen():
        offset = 0
        for b in df._iter_filtered_batches():
            n_b = b.num_rows
            col = fn(offset, n_b)
            if len(col) != n_b:
                raise RuntimeError(
                    f"_attach_column_with_fn: fn returned {len(col)} "
                    f"rows, expected {n_b} (batch at offset {offset})"
                )
            offset += n_b
            arrays = [b.column(c) for c in base_cols] + [col]
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

    new_ds = Dataset.from_batches(out_schema, gen(), df._new_dataset_dir())
    return CadwDataFrame(
        new_ds,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir,
    )


class KFold:
    """K-fold cross-validator for CadwDataFrame (out-of-core).

    Default (shuffle=False): fold_id = row_id % k, computed per batch.
    O(batch) state, no O(N) arrays.

    shuffle=True (default mode, O(1) state): keyed PRNG
    splitmix64(row_id, seed) % k. Deterministic per row_id (same seed →
    same split). Fold sizes ~binomial, mean n/k, per-fold variance
    O(sqrt(n/k)) — split-quality, sufficient for CV.

    materialize=True (opt-in, O(N) RAM): old behavior — exact-balanced
    shuffle via rng.permutation(n). Requires MemoryGuard headroom
    (check_state(20·N) applied; raises MemoryLimitExceeded if budget
    insufficient). Use only when N is small or exact fold sizes matter.

    random_state=None (default) → seed=42 (deterministic reproducibility
    — same split across runs, machines, storage layouts).

    Row order within each fold is preserved (filters don't reorder).
    """

    __slots__ = ("n_splits", "shuffle", "random_state", "materialize")

    def __init__(
        self,
        n_splits: int,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
        materialize: bool = False,
    ) -> None:
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError(f"n_splits must be int >= 1; got {n_splits!r}")
        if not isinstance(shuffle, bool):
            raise TypeError(
                f"shuffle must be bool; got {type(shuffle).__name__}"
            )
        if not isinstance(materialize, bool):
            raise TypeError(
                f"materialize must be bool; got "
                f"{type(materialize).__name__}"
            )
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.materialize = materialize

    def split(
        self, df: CadwDataFrame,
    ) -> Iterator[tuple[int, CadwDataFrame, CadwDataFrame]]:
        """Yield (fold_i, train_df, test_df) for each fold.

        Yields lazy CadwDataFrame views (fold filter via PredicateFilter).
        """
        n = df.count()
        if self.n_splits > n:
            raise ValueError(
                f"KFold: n_splits={self.n_splits} > n={n}"
            )

        if self.materialize:
            # Opt-in exact-balanced shuffle. O(N) state, guarded.
            # Peak: int32 fold_id (4·N) + int64 arange (8·N) + for
            # shuffle=True rng.permutation (8·N) + np.arange temp (8·N).
            # Worst case (shuffle=True): ~20·N bytes transient. Use 20·N
            # as safe upper bound for check_state.
            df._memory_guard.check_state(
                20 * n, name="KFold.materialize",
            )
            if self.shuffle:
                seed = (_DEFAULT_SEED if self.random_state is None
                        else int(self.random_state))
                rng = np.random.default_rng(seed)
                perm = rng.permutation(n)
                fold_id = np.empty(n, dtype=np.int32)
                fold_id[perm] = np.arange(n) % self.n_splits
            else:
                fold_id = (np.arange(n) % self.n_splits).astype(np.int32)
            fold_arr = pa.array(fold_id, type=pa.int32())
            augmented = _attach_column(df, "__fold", fold_arr)
        else:
            # O(batch) state; per-batch fold_id.
            if self.shuffle:
                seed = (_DEFAULT_SEED if self.random_state is None
                        else int(self.random_state))
                k = self.n_splits
                def _fn(offset: int, n_b: int) -> pa.Array:
                    row_ids = np.arange(offset, offset + n_b,
                                        dtype=np.uint64)
                    ids = _fold_ids_from_row_ids(row_ids, k, seed)
                    return pa.array(ids, type=pa.int32())
            else:
                k = self.n_splits
                def _fn(offset: int, n_b: int) -> pa.Array:
                    ids = (np.arange(offset, offset + n_b) % k).astype(
                        np.int32)
                    return pa.array(ids, type=pa.int32())
            augmented = _attach_column_with_fn(
                df, "__fold", pa.int32(), _fn,
            )

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
