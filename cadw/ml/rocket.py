"""ROCKET: Random Convolutional Kernel Transform (D.4.x, v0.9.89).

Reference: Dempster, Petitjean, Webb, *ROCKET: Exceptionally fast and
accurate time series classification using random convolutional kernels*,
DMKD 2020.

CADW design (v0.9.89, OOC in n_kernels):

Kernel bank (state O(model), seeded): n_kernels random (weight ~ N(0,1),
bias ~ U(-1,1), dilation ∈ {1,2,4,...,max_dilation}).

Feature extraction (per series, streaming via cadw.ml._series.iter_series):
  For each dilation group, vectorized conv via strided windows:
      idx     = arange(n_out)[:, None] + arange(kernel_len)[None, :] * d
      windows = x[idx]                          (n_out, kernel_len)
      conv    = windows @ W_d.T + b_d[None, :]  (n_out, K_d)
      ppv     = (conv > 0).mean(axis=0)         (K_d,)
      max_    = conv.max(axis=0)                (K_d,)
  → 2 features per kernel → (2 * n_kernels,) feature vector per series.

Feature matrix (O(n_series * 2 * n_kernels * 8) on disk):
  SpillDir + ArrowIPCBatchWriter, each row = np.tobytes() blob.
  Reader reconstructs via np.frombuffer.

Classifier (state O(d * K), ANY d, ANY n_series):
  Streaming SGD ridge with RMSprop (cadw.ops.optim.sgd_streaming).
  State ~2 * d * K * 8 bytes (W + h).
  For d = 20000 (n_kernels=10000), binary: ~320 KB.

Memory at fit time:
  - Kernel bank: O(n_kernels * kernel_len) — hyperparameter-bounded.
  - Conv buffer: O(max_group_n_out * (kernel_len + K_d)) — per-series,
    per-dilation-group, guarded. Bounded by series length + K_d.
  - Feature matrix: on disk (SpillDir).
  - Classifier state: O(d * K) — independent of n and n_kernels^2.
  - Labels: O(n_series * K) — see "known limitation" below.

Known limitation (backlog): labels (classes and y_idx array) are
collected into RAM as O(n_series * 8). For n_series up to ~10^7 this is
< 80 MB and fine. Truly unbounded n_series would require a two-pass
labels protocol (HashShuffle by class + label spill in feature file);
deferred to audit backlog. Guards raise cleanly if label RAM exceeds
state_limit — no silent OOM.

Documented deviation from paper: SGD classifier instead of closed-form
Ridge. Accuracy within ~1-2% of closed-form on well-scaled ROCKET
features (PPV ∈ [0,1]); deterministic given batch order and epochs.
Trade-off: arbitrary n_kernels (state O(d) not O(d²)).
"""
from __future__ import annotations

from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import SpillDir
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ml._logging import log_fit
from cadw.ml._series import iter_series
from cadw.ops.optim import sgd_streaming


KERNEL_LEN = 9


def make_kernels(num_kernels: int = 1000,
                 kernel_len: int = KERNEL_LEN,
                 seed: int = 0,
                 max_dilation: int = 32,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROCKET kernel bank: (W (K, kernel_len), b (K,), dilations (K,)).

    Weights ~ N(0, 1); biases ~ U(-1, 1); dilations drawn uniformly
    from {1, 2, 4, ..., max_dilation}. Deterministic given `seed`.
    """
    if num_kernels < 1:
        raise ValueError("make_kernels: num_kernels must be >= 1")
    if kernel_len < 1:
        raise ValueError("make_kernels: kernel_len must be >= 1")
    if max_dilation < 1:
        raise ValueError("make_kernels: max_dilation must be >= 1")
    rng = np.random.RandomState(seed)
    W = rng.randn(num_kernels, kernel_len).astype(np.float64)
    b = rng.uniform(-1.0, 1.0, size=num_kernels).astype(np.float64)
    allowed, d = [], 1
    while d <= max_dilation:
        allowed.append(d)
        d *= 2
    allowed = np.array(allowed, dtype=np.int64)
    dilations = rng.choice(allowed, size=num_kernels).astype(np.int64)
    return W, b, dilations


def _rocket_features_series(x: np.ndarray,
                            W: np.ndarray,
                            b: np.ndarray,
                            dilations: np.ndarray,
                            kernel_len: int,
                            mg: MemoryGuard,
                            name: str,
                            ) -> np.ndarray:
    """(n_kernels, 2) PPV+MAX per series. Guarded per dilation group."""
    K, L = W.shape[0], x.size
    feats = np.zeros((K, 2), dtype=np.float64)
    if L <= 0:
        return feats
    for d in np.unique(dilations):
        mask = dilations == d
        K_d = int(mask.sum())
        W_d = W[mask]
        b_d = b[mask]
        n_out = L - (kernel_len - 1) * int(d)
        if n_out <= 0:
            continue
        # Peak inside group: idx (n_out, kernel_len) + windows (copy)
        # + conv (n_out, K_d) + bool mask (n_out, K_d).
        mg.check_state(2 * n_out * kernel_len * 8 + 2 * n_out * K_d * 8,
                       name=name + ".conv")
        idx = (np.arange(n_out, dtype=np.int64)[:, None]
               + np.arange(kernel_len, dtype=np.int64)[None, :] * int(d))
        windows = x[idx]                             # (n_out, kernel_len)
        conv = windows @ W_d.T + b_d[None, :]        # (n_out, K_d)
        feats[mask, 0] = (conv > 0.0).mean(axis=0)
        feats[mask, 1] = conv.max(axis=0)
    return feats


def _feature_schema() -> pa.Schema:
    return pa.schema([pa.field("features", pa.binary())])


def _write_rows(writer: ArrowIPCBatchWriter,
                rows: list,
                n_features: int,
                mg: MemoryGuard,
                name: str) -> None:
    """Write a list of feature rows to disk as one ArrowIPC batch."""
    n = len(rows)
    # Peak: blobs (n * n_features * 8) + ArrowIPC copy.
    mg.check_state(2 * n * n_features * 8, name=name + ".write")
    blobs = [r.tobytes() for r in rows]
    tbl = pa.table({"features": pa.array(blobs, type=pa.binary())})
    writer.write_batch(tbl.to_batches()[0])


def _read_batch(batch: pa.RecordBatch,
                n_features: int,
                mg: MemoryGuard,
                name: str) -> np.ndarray:
    """Reconstruct (b, n_features) float64 matrix from ArrowIPC batch.

    Guard choice (v0.9.90-S1-hotfix): check_batch, not check_state.
    Semantics: input is ONE pa.RecordBatch (already in RAM from
    ArrowIPCBatchReader); peak RAM is proportional to batch.nbytes,
    not to dataset N or K. check_batch(batch, extra_factor=3.0) checks
    that 3 * batch.nbytes fits in available_bytes, which matches the
    actual peak: batch (Arrow) + to_pylist (Python bytes objects,
    ~n * (33 + n_features*8) B) + out (np.empty, n * n_features * 8 B)
    ≈ 3 * n * n_features * 8. check_state was semantically wrong
    (state_limit = 0.5 * avail, intended for accumulators, not batches);
    classified as 'other-guard' by scripts/service/diagnose_unguarded_
    to_pylist.py.
    """
    n = batch.num_rows
    mg.check_batch(batch, name=name + ".read", extra_factor=3.0)
    rows = batch.column("features").to_pylist()
    out = np.empty((n, n_features), dtype=np.float64)
    for i, r in enumerate(rows):
        out[i] = np.frombuffer(r, dtype=np.float64)
    return out


def _accumulate_features_to_spill(df,
                                  value_col: str,
                                  id_col: Optional[str],
                                  W: np.ndarray,
                                  b: np.ndarray,
                                  dilations: np.ndarray,
                                  kernel_len: int,
                                  mg: MemoryGuard,
                                  name: str,
                                  sdir: SpillDir,
                                  ) -> Tuple[str, int]:
    """Stream series → compute features → write ArrowIPC to sdir."""
    path = str(sdir.path(name + "_features.arrow"))
    writer = ArrowIPCBatchWriter(path, _feature_schema())
    n_features = W.shape[0] * 2
    write_rows = max(1, mg.get_write_batch_size()
                     // max(1, n_features * 8))
    buf: list = []
    n_series = 0
    try:
        for _sid, x in iter_series(df, value_col, id_col, mg, name):
            f = _rocket_features_series(x, W, b, dilations, kernel_len,
                                        mg, name)
            buf.append(f.reshape(-1))
            n_series += 1
            if len(buf) >= write_rows:
                _write_rows(writer, buf, n_features, mg, name)
                buf = []
        if buf:
            _write_rows(writer, buf, n_features, mg, name)
    finally:
        writer.close()
    return path, n_series


def _feature_stream(path: str,
                    n_features: int,
                    mg: MemoryGuard,
                    name: str,
                    ) -> Callable[[], Iterator[Tuple[np.ndarray, int]]]:
    """Return callable producing fresh iterators over the feature file."""
    def _iter() -> Iterator[Tuple[np.ndarray, int]]:
        reader = ArrowIPCBatchReader(path, columns=["features"])
        try:
            idx = 0
            for batch in reader:
                X_b = _read_batch(batch, n_features, mg, name)
                yield X_b, idx
                idx += X_b.shape[0]
        finally:
            reader.close()
    return _iter


class RocketModel:
    """Fitted ROCKET feature transform + SGD ridge classifier."""

    def __init__(self, W, b, dilations, kernel_len,
                 weights, classes,
                 value_col, id_col, params):
        self.W_ = W
        self.b_ = b
        self.dilations_ = dilations
        self.kernel_len_ = kernel_len
        self.weights_ = weights          # (n_outputs, 2 * n_kernels)
        self.classes_ = classes
        self.value_col_ = value_col
        self.id_col_ = id_col
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self) -> int:
        kernel = (self.W_.size + self.b_.size + self.dilations_.size) * 8
        clf = self.weights_.size * 8
        return int(kernel + clf + 4096)

    def to_dict(self) -> dict:
        return {
            "params": self.params_,
            "value_col": self.value_col_,
            "id_col": self.id_col_,
            "num_kernels": int(self.W_.shape[0]),
            "kernel_len": int(self.kernel_len_),
            "classes": self.classes_,
        }

    def predict(self, df, memory_guard=None) -> np.ndarray:
        mg = memory_guard or df._memory_guard
        mg.check_state(self.state_nbytes(), name="rocket.predict.state")
        preds = []
        for _sid, x in iter_series(df, self.value_col_, self.id_col_,
                                   mg, "rocket.predict"):
            f = _rocket_features_series(x, self.W_, self.b_, self.dilations_,
                                        self.kernel_len_, mg, "rocket.predict")
            scores = self.weights_ @ f.reshape(-1)      # (K,) or (1,)
            if len(self.classes_) == 2 and scores.shape[0] == 1:
                preds.append(self.classes_[1 if scores[0] > 0.0 else 0])
            else:
                preds.append(self.classes_[int(np.argmax(scores))])
        return np.array(preds)


def fit_rocket(df,
               value_col: str,
               target_col: str,
               *,
               id_col: Optional[str] = None,
               num_kernels: int = 1000,
               kernel_len: int = KERNEL_LEN,
               seed: int = 0,
               max_dilation: int = 32,
               ridge_lambda: float = 1.0,
               sgd_lr: float = 0.01,
               sgd_epochs: int = 30,
               memory_guard: Optional[MemoryGuard] = None,
               ) -> RocketModel:
    """Fit ROCKET: stream features → SpillDir → SGD ridge (OOC in n_kernels)."""
    if value_col not in df._columns:
        raise KeyError(f"fit_rocket: column {value_col!r} not in {df._columns}")
    if target_col not in df._columns:
        raise KeyError(f"fit_rocket: column {target_col!r} not in {df._columns}")
    if id_col is not None and id_col not in df._columns:
        raise KeyError(f"fit_rocket: column {id_col!r} not in {df._columns}")

    mg = memory_guard or df._memory_guard

    # Kernel bank state guard BEFORE alloc.
    kernel_bytes = (num_kernels * kernel_len + 2 * num_kernels) * 8
    mg.check_state(kernel_bytes, name="rocket.kernels")
    W, b, dilations = make_kernels(num_kernels, kernel_len, seed,
                                   max_dilation)
    n_features = num_kernels * 2

    # Labels in RAM — O(n_series). Guarded; see module docstring.
    labels = []
    cur_id = None
    for b_ in df._iter_filtered_batches():
        mg.check_batch(b_, name="rocket.labels", extra_factor=3.0)
        if id_col is None:
            arr = b_.column(target_col).to_pylist()
            if arr and cur_id is None:
                labels.append(arr[0])
                cur_id = 0
            continue
        ids = b_.column(id_col).to_pylist()
        tgts = b_.column(target_col).to_pylist()
        i, n = 0, b_.num_rows
        while i < n:
            j = i
            while j < n and ids[j] == ids[i]:
                j += 1
            labels.append(tgts[i])
            i = j
    y_raw = np.asarray(labels)
    classes = np.unique(y_raw)
    K = int(classes.size)
    if K < 2:
        raise ValueError(f"fit_rocket: need >= 2 classes, got {K}")
    y_idx = np.searchsorted(classes, y_raw)

    # Encode target: binary → 1 output {-1,+1}; multiclass → K outputs one-hot.
    if K == 2:
        Y_target = (2.0 * y_idx - 1.0).reshape(-1, 1)
        n_outputs = 1
    else:
        Y_target = np.zeros((y_idx.size, K), dtype=np.float64)
        Y_target[np.arange(y_idx.size), y_idx] = 1.0
        n_outputs = K

    # Guard total fit-time RAM: classifier + labels.
    mg.check_state(2 * n_features * n_outputs * 8 + n_features * 8
                   + y_idx.size * n_outputs * 8,
                   name="rocket.fit.state")

    with SpillDir(guard=mg, name="rocket") as sdir:
        fpath, n_series = _accumulate_features_to_spill(
            df, value_col, id_col, W, b, dilations, kernel_len,
            mg, "rocket.fit", sdir)
        if n_series != y_idx.size:
            raise RuntimeError(
                f"fit_rocket: feature rows ({n_series}) != labels "
                f"({y_idx.size})")
        stream_iter = _feature_stream(fpath, n_features, mg, "rocket.fit")

        def _batches():
            for X_b, idx in stream_iter():
                yield X_b, Y_target[idx: idx + X_b.shape[0]]

        weights = sgd_streaming(
            _batches, n_features, n_outputs,
            lam=ridge_lambda, lr=sgd_lr, epochs=sgd_epochs,
            mg=mg, name="rocket.fit")

    model = RocketModel(W, b, dilations, kernel_len, weights, classes,
                        value_col, id_col,
                        {"num_kernels": num_kernels,
                         "kernel_len": kernel_len,
                         "seed": seed,
                         "max_dilation": max_dilation,
                         "ridge_lambda": ridge_lambda,
                         "sgd_lr": sgd_lr,
                         "sgd_epochs": sgd_epochs})
    log_fit("rocket", state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="rocket.model")
    return model
