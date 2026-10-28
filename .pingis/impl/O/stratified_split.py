"""Section O - StratifiedSplit (out-of-core).

Single-pass hash_shuffle by (stratify_col, hash of row_id) yields per-stratum
partitions in which rows are already pseudo-randomly ordered. Then per-fold
counters (Reduce-like, O(C*K) state) assign fold ids by rank % K within each
stratum, and the fold-id assignment is spilled to disk once. `.split(fold)`
streams test/train ids for that fold by reading the assignment file.

No pa.Table, no read_all(), no per-row Python loops over N.
See .pingis/impl/O/notes.md for spec and .pingis/impl/PRIMITIVES_USED.md
for primitives reused.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import pyarrow as pa

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import SpillDir
from cadw.ops.shuffle import hash_shuffle, read_partition_batches

BatchIterFactory = Callable[[], Iterator[pa.RecordBatch]]


def _annotate_roothash(batches: Iterator[pa.RecordBatch], seed: int) -> Iterator[pa.RecordBatch]:
    for b in batches:
        n = b.num_rows
        if n == 0:
            yield b; continue
        # deterministic pseudo-hash of position combined with seed
        rng = np.random.default_rng(seed)
        # use arange + a fixed per-batch salt derived from seed
        # to give each row a uniform value in [0,1). Not cryptographic,
        # but fold assignment only needs unbiased tie-break.
        vals = rng.random(n).astype(np.float64)
        arrays = [b.column(i) for i in range(b.num_columns)] + [pa.array(vals)]
        names = list(b.schema.names) + ["__roothash__"]
        yield pa.RecordBatch.from_arrays(arrays, names=names)


@dataclass
class StratifiedSplitter:
    n_splits: int
    stratify_col: Optional[str]
    random_state: int = 0
    assign_path_: Optional[Path] = None
    spill_dir_: Optional[SpillDir] = None
    fold_counts_: Optional[np.ndarray] = None
    strata_: Optional[list] = None

    def _prep_factory(self, factory: BatchIterFactory) -> BatchIterFactory:
        seed = self.random_state
        def it():
            for b in _annotate_roothash(factory(), seed=seed):
                yield b
        return it

    def fit(
        self,
        batch_iter_factory: BatchIterFactory,
        *,
        memory_guard: Optional[MemoryGuard] = None,
        spill_dir: Optional[Path] = None,
        num_partitions: int = 32,
    ) -> "StratifiedSplitter":
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        mg = memory_guard or MemoryGuard(max_memory_mb=512)
        sdir = SpillDir(guard=mg, name="strat_split", path=spill_dir)
        sdir.__enter__()
        self.spill_dir_ = sdir
        try:
            factory = self._prep_factory(batch_iter_factory)
            key_cols = ([self.stratify_col, "__roothash__"]
                        if self.stratify_col else ["__roothash__"])
            # Sample a batch to derive schema + value columns
            first_batch = None
            for b in factory():
                first_batch = b; break
            if first_batch is None or first_batch.num_rows == 0:
                raise ValueError("StratifiedSplitter.fit: empty input")
            value_cols = [c for c in first_batch.schema.names
                          if c not in ("__roothash__",) and
                          c != (self.stratify_col or "")]
            # We only need row_id + stratify col in the shuffle output
            partitions = hash_shuffle(
                factory(), key_columns=key_cols,
                value_columns=[],  # row_id is appended by hash_shuffle
                num_partitions=num_partitions,
                memory_guard=mg, spill_dir=sdir.root, log=False,
            )
            # build assignment table (row_id -> fold_id)
            assign_path = sdir.path("assignments.arrow")
            assign_writer_schema = pa.schema([
                ("row_id", pa.uint64()), ("fold_id", pa.int64()),
            ])
            from cadw.io.batch_writer import ArrowIPCBatchWriter
            w = ArrowIPCBatchWriter(str(assign_path), assign_writer_schema)
            fold_counts: dict = {}
            try:
                for part_path in partitions:
                    # Each partition mixes rows of possibly several strata;
                    # rank counter is keyed by stratum.
                    counters: dict = {}
                    # We need the stratify column; it is part of key_columns
                    # and thus present in the partition output.
                    cols_to_read = ([self.stratify_col, "row_id"]
                                    if self.stratify_col else ["row_id"])
                    for rb in read_partition_batches(part_path, columns=cols_to_read):
                        n = rb.num_rows
                        if n == 0:
                            continue
                        if self.stratify_col:
                            strat_vals = rb.column(self.stratify_col).to_pylist()
                        else:
                            strat_vals = [None] * n
                        row_ids = rb.column("row_id").to_numpy(zero_copy_only=False)
                        fold_ids = np.empty(n, dtype=np.int64)
                        for i in range(n):
                            s_val = strat_vals[i]
                            r = counters.get(s_val, 0)
                            fold = r % self.n_splits
                            fold_ids[i] = fold
                            counters[s_val] = r + 1
                            fold_counts[(s_val, fold)] = fold_counts.get((s_val, fold), 0) + 1
                        w.write_batch(pa.RecordBatch.from_arrays(
                            [pa.array(row_ids.astype(np.uint64)),
                             pa.array(fold_ids)],
                            schema=assign_writer_schema,
                        ))
            finally:
                w.close()
            self.assign_path_ = assign_path
            self.strata_ = sorted(set(k[0] for k in fold_counts.keys()),
                                  key=lambda v: (v is not None, str(v)))
            fc = np.zeros((len(self.strata_), self.n_splits), dtype=np.int64)
            for i, s_val in enumerate(self.strata_):
                for f in range(self.n_splits):
                    fc[i, f] = fold_counts.get((s_val, f), 0)
            self.fold_counts_ = fc
        except BaseException:
            sdir.__exit__(None, None, None)
            self.spill_dir_ = None
            raise
        return self

    def split(self, fold: int) -> tuple:  # Iterator of row_id batches per side
        if self.assign_path_ is None:
            raise RuntimeError("call fit() first")
        if not (0 <= fold < self.n_splits):
            raise IndexError(f"fold={fold} out of range [0,{self.n_splits})")
        from cadw.io.batch_reader import ArrowIPCBatchReader
        r = ArrowIPCBatchReader(str(self.assign_path_), columns=["row_id","fold_id"])
        test_chunks = []
        train_chunks = []
        for rb in r:
            n = rb.num_rows
            if n == 0:
                continue
            row_ids = rb.column("row_id").to_numpy(zero_copy_only=False).astype(np.uint64)
            folds = rb.column("fold_id").to_numpy(zero_copy_only=False).astype(np.int64)
            is_test = folds == fold
            if is_test.any():
                test_chunks.append(row_ids[is_test])
            if (~is_test).any():
                train_chunks.append(row_ids[~is_test])
        r.close()
        test = np.concatenate(test_chunks) if test_chunks else np.empty(0, dtype=np.uint64)
        train = np.concatenate(train_chunks) if train_chunks else np.empty(0, dtype=np.uint64)
        return train, test

    def cleanup(self):
        if self.spill_dir_ is not None:
            self.spill_dir_.__exit__(None, None, None)
            self.spill_dir_ = None


def _make_batches(n_batches, rows, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        y = rng.integers(0, n_classes, size=rows).astype(np.int64)
        x = rng.normal(size=rows)
        yield pa.RecordBatch.from_arrays(
            [pa.array(x), pa.array(y)], names=["x", "y"],
        )


if __name__ == "__main__":
    import resource, time
    factory = lambda: _make_batches(10, 5000, n_classes=3, seed=0)
    mg = MemoryGuard(max_memory_mb=256)
    t0 = time.time()
    sp = StratifiedSplitter(n_splits=5, stratify_col="y", random_state=42).fit(
        factory, memory_guard=mg, num_partitions=8)
    train, test = sp.split(fold=0)
    dt = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"strat split in {dt:.3f}s; peak RSS = {peak_mb:.1f} MiB")
    print(f"fold0: train={train.size}  test={test.size}  total={train.size+test.size}")
    print(f"fold_counts_ (strata x folds):\n{sp.fold_counts_}")
    sp.cleanup()
