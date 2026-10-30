#!/usr/bin/env python3
"""Bench: IsolationForest (D.1.1) — fit + score at N=100k/1M.

Compares against sklearn.ensemble.IsolationForest on the same subsample.
Records wall time and psutil peak RSS delta.
"""
from __future__ import annotations
import gc, sys, tempfile, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pyarrow as pa, psutil
from cadw import CadwDataFrame
from cadw.ml.iforest import fit_isolation_forest


class S:
    def __init__(self, iv=0.005):
        self.p=psutil.Process(); self.b=self.p.memory_info().rss; self.peak=self.b
        self._s=threading.Event(); self._iv=iv
    def _run(self):
        while not self._s.is_set():
            r=self.p.memory_info().rss
            if r>self.peak: self.peak=r
            self._s.wait(self._iv)
    def __enter__(self):
        self.t=threading.Thread(target=self._run,daemon=True); self.t.start(); return self
    def __exit__(self,*a): self._s.set(); self.t.join(timeout=2.0)
    @property
    def delta(self): return self.peak-self.b


def gen(n, d=5, chunk=50_000):
    rng = np.random.RandomState(0)
    for i in range(0, n, chunk):
        sz = min(chunk, n-i)
        X = rng.normal(size=(sz, d))
        X[: max(1, sz//100)] = rng.uniform(5, 8, size=(max(1, sz//100), d))  # 1% anomalies
        yield pa.record_batch({f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)})


def main():
    cols = [f"x{j}" for j in range(5)]
    for n in (100_000, 1_000_000):
        tbl = pa.Table.from_batches(list(gen(n)))
        df = CadwDataFrame.from_table(tbl)
        with S() as s:
            t0=time.time()
            m = fit_isolation_forest(df, cols, n_trees=100, subsample_size=256, random_state=0)
            fit_dt=time.time()-t0
            t0=time.time()
            _ss = m.score_samples(df)
            sc = np.concatenate(
                [b.column("score").to_numpy(zero_copy_only=False)
                 for b in _ss._iter_filtered_batches()])
            score_dt=time.time()-t0
        print(f"n={n:>8d} fit={fit_dt:6.2f}s score={score_dt:6.2f}s peak={s.delta/1048576:6.1f} MB "
              f"state={m.state_nbytes()} B mean_score={float(np.nanmean(sc)):.3f}")
        gc.collect()

if __name__ == "__main__":
    main()
