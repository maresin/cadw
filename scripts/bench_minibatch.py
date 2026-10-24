#!/usr/bin/env python3
"""Bench: minibatch."""
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

from cadw.ml.minibatch import fit_minibatch_kmeans, fit_minibatch_gmm

def make(n, seed=0, d=2, k=3):
    rng=np.random.RandomState(seed)
    centers=rng.normal(scale=5.0,size=(k,d))
    Y=rng.randint(0,k,size=n)
    X=centers[Y]+rng.normal(0,0.5,size=(n,d))
    d_={f"x{j}":pa.array(X[:,j]) for j in range(d)}
    return CadwDataFrame.from_table(pa.table(d_)), X

def run_km(n, k=3):
    df,X=make(n,k=k); gc.collect()
    with RSS() as s:
        t0=time.time(); m=fit_minibatch_kmeans(df,["x0","x1"],n_clusters=k,batch_size=1024,max_iter=10); dt=time.time()-t0
        inertia=m.inertia(df)
    out=dict(n=n,model="kmeans",fit_s=round(dt,3),inertia=round(float(inertia),1),peak_mb=round(s.delta/1048576,1),state_b=m.state_nbytes())
    print(json.dumps(out)); return out

def run_gmm(n, k=3):
    df,X=make(n,k=k); gc.collect()
    with RSS() as s:
        t0=time.time(); m=fit_minibatch_gmm(df,["x0","x1"],n_components=k,batch_size=1024,max_iter=20); dt=time.time()-t0
        sc=m.score(df)
    out=dict(n=n,model="gmm",fit_s=round(dt,3),score=round(float(sc),4),peak_mb=round(s.delta/1048576,1),state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__=="__main__":
    res=[]
    for n in (20_000, 100_000, 500_000):
        res.append(run_km(n)); res.append(run_gmm(n))
    p=Path(".pingis/benches/minibatch.json"); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"algo":"D.5.3","runs":res},indent=2)); print("wrote",p)
