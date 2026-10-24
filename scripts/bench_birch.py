#!/usr/bin/env python3
"""Bench: birch."""
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

from cadw.ml.birch import fit_birch

def make(n, seed=0):
    rng=np.random.RandomState(seed)
    centers=np.array([[0,0],[5,5],[-5,5],[5,-5]],dtype=float)
    Y=rng.randint(0,4,size=n)
    X=centers[Y]+rng.normal(0,0.4,size=(n,2))
    return CadwDataFrame.from_table(pa.table({"a":pa.array(X[:,0]),"b":pa.array(X[:,1])})), X

def run(n):
    df, X = make(n); gc.collect()
    with RSS() as s:
        t0=time.time(); m=fit_birch(df,["a","b"],n_clusters=4,threshold=0.5); dt=time.time()-t0
        lab=m.predict(df)
    out=dict(n=n,fit_s=round(dt,3),n_cf=m.to_dict()["n_cf"],n_clusters=int(m.centers_.shape[0]),peak_mb=round(s.delta/1048576,1),state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__=="__main__":
    res=[run(n) for n in (5_000,20_000,50_000)]
    p=Path(".pingis/benches/birch.json"); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"algo":"D.5.4","runs":res},indent=2)); print("wrote",p)
