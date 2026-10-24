#!/usr/bin/env python3
"""Bench: randsvd."""
from __future__ import annotations
import gc, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pyarrow as pa, psutil
from cadw import CadwDataFrame

class RSS:
    def __init__(self, iv=0.005):
        self.p = psutil.Process(); self.b = self.p.memory_info().rss; self.peak = self.b
        self._s = threading.Event(); self._iv = iv
    def _run(self):
        while not self._s.is_set():
            r = self.p.memory_info().rss
            if r > self.peak: self.peak = r
            self._s.wait(self._iv)
    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self._s.set(); self.t.join(timeout=2.0)
    @property
    def delta(self):
        return self.peak - self.b

from cadw.ml.randsvd import fit_randomized_svd

def make(n, d, k, seed=0):
    rng=np.random.RandomState(seed)
    Z=rng.normal(size=(n,k))
    X=Z@rng.normal(size=(k,d))+0.05*rng.normal(size=(n,d))
    cols=[f"x{j}" for j in range(d)]
    d_={c:pa.array(X[:,j]) for j,c in enumerate(cols)}
    return CadwDataFrame.from_table(pa.table(d_)), cols

def run(n, d, k):
    df, cols = make(n,d,k); gc.collect()
    with RSS() as s:
        t0=time.time(); m=fit_randomized_svd(df,cols,n_components=k); dt=time.time()-t0
        ratio=m.explained_variance_ratio(df)
    out=dict(n=n,d=d,k=k,fit_s=round(dt,3),sv0=round(float(m.singular_values_[0]),2),ratio0=round(float(ratio[0]),4),peak_mb=round(s.delta/1048576,1),state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__=="__main__":
    # sizes chosen to fit the 100 MB batch guard (extra_factor=3.0)
    res=[run(10_000,20,3),run(50_000,50,5),run(150_000,20,3)]
    p=Path(".pingis/benches/randsvd.json"); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"algo":"D.6.4","runs":res},indent=2)); print("wrote",p)
