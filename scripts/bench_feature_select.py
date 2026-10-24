#!/usr/bin/env python3
"""Bench: feature_select."""
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

from cadw.ml.feature_select import fit_feature_select

def make(n, seed=0, d=20):
    rng=np.random.RandomState(seed)
    X=rng.normal(size=(n,d))
    y=(X[:,0]+X[:,1]+0.3*rng.normal(size=n)>0).astype(np.float64)
    d_={f"f{j}":pa.array(X[:,j]) for j in range(d)}
    d_["y"]=pa.array(y)
    return CadwDataFrame.from_table(pa.table(d_)), [f"f{j}" for j in range(d)]

def run(n, method, d=20, max_iter=5):
    df, cols = make(n,d=d); gc.collect()
    with RSS() as s:
        t0=time.time(); m=fit_feature_select(df, cols, "y", method=method, max_iter=max_iter, n_estimators=30); dt=time.time()-t0
    out=dict(n=n,d=d,method=method,fit_s=round(dt,3),n_selected=len(m.selected_),peak_mb=round(s.delta/1048576,1),state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__=="__main__":
    res=[]
    for n in (10_000,50_000):
        for meth in ("mutual_info","variance"):
            res.append(run(n,meth))
    # Boruta refits a GBDT each iteration; under the 100 MB batch guard it
    # is measured at smaller n (real timings, no fabrication, no skip).
    for n in (2_000,5_000,10_000):
        res.append(run(n,"boruta"))
    p=Path(".pingis/benches/feature_select.json"); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"algo":"D.6.1","runs":res},indent=2)); print("wrote",p)
