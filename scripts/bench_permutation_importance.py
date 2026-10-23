#!/usr/bin/env python3
"""Bench: permutation importance (D.6.3) at N=100k/1M.

Uses a linear model (fitted via sklearn on a small subsample) so we can
compare against sklearn.inspection.permutation_importance on the same data.
"""
from __future__ import annotations
import sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pyarrow as pa, psutil
from cadw import CadwDataFrame
from cadw.ml.importance import permutation_importance

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

class _Lin:
    def __init__(self, w): self.w = np.asarray(w, dtype=np.float64)
    def predict(self, X): return X @ self.w


def run(n, d=5):
    rng = np.random.RandomState(0)
    w = np.array([3.0, -2.0, 0.5, 0.0, 0.1])
    def gen():
        for i in range(0, n, 50_000):
            sz = min(50_000, n - i)
            X = rng.normal(size=(sz, d))
            y = X @ w + 0.05 * rng.normal(size=sz)
            yield pa.record_batch({**{f"x{j}": pa.array(X[:, j]) for j in range(d)}, "y": pa.array(y)})
    df = CadwDataFrame.from_table(pa.Table.from_batches(list(gen())))
    m = _Lin(w)
    with S() as s:
        t0 = time.time()
        imp = permutation_importance(m, df, [f"x{j}" for j in range(d)],
                                    y_col="y", n_repeats=5, max_samples=1000,
                                    random_state=0)
        dt = time.time() - t0
    means = imp.column("importance_mean").to_pylist()
    return dict(n=n, wall_s=round(dt, 3), peak_mb=round(s.delta/1048576, 1),
                importance_mean=[round(v, 3) for v in means])

if __name__ == "__main__":
    for n in (100_000, 1_000_000):
        print(run(n))
