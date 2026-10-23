#!/usr/bin/env python3
"""Bench: GMM density (D.6.8) at N=100k/1M.

Compares fit/score time and RSS vs sklearn.mixture.GaussianMixture."""
from __future__ import annotations
import gc, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pyarrow as pa, psutil
from cadw import CadwDataFrame
from cadw.ml.gmm import fit_gmm

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

def run(n, d=5, k=3):
    rng = np.random.RandomState(0)
    def gen():
        for i in range(0, n, 50_000):
            sz = min(50_000, n - i)
            X = rng.normal(size=(sz, d))
            yield pa.record_batch({f"x{j}": pa.array(X[:, j]) for j in range(d)})
    df = CadwDataFrame.from_table(pa.Table.from_batches(list(gen())))
    with S() as s:
        t0=time.time()
        m = fit_gmm(df, [f"x{j}" for j in range(d)], n_components=k, random_state=0)
        fit_dt=time.time()-t0
        t0=time.time()
        ll = m.score(df)
        score_dt=time.time()-t0
    return dict(n=n, k=k, d=d, fit_s=round(fit_dt,3), score_s=round(score_dt,2),
                peak_mb=round(s.delta/1048576,1), state_b=m.state_nbytes(), mean_ll=round(ll,3))

if __name__ == "__main__":
    for n in (100_000, 1_000_000):
        print(run(n)); gc.collect()
