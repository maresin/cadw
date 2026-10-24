#!/usr/bin/env python3
"""Bench: chains."""
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

from cadw.ml.chains import fit_chains

def make(n, seed=0, d=3):
    rng=np.random.RandomState(seed)
    X=rng.normal(size=(n,d))
    a=(X[:,0]+0.3*rng.normal(size=n)>0).astype(np.float64)
    b=np.where(rng.uniform(size=n)<0.85,a,1.0-a).astype(np.float64)
    c=(-0.4*X[:,0]+0.5*X[:,2]+0.3*rng.normal(size=n)>0).astype(np.float64)
    d_={f"x{j}":pa.array(X[:,j]) for j in range(d)}
    d_["a"]=pa.array(a); d_["b"]=pa.array(b); d_["c"]=pa.array(c)
    return CadwDataFrame.from_table(pa.table(d_))

def run(n, chain, est=30):
    df=make(n); gc.collect()
    with RSS() as s:
        t0=time.time(); m=fit_chains(df,["x0","x1","x2"],["a","b","c"],chain=chain,n_estimators=est); dt=time.time()-t0
        sa=m.subset_accuracy(df)
    out=dict(n=n,chain=chain,fit_s=round(dt,2),subset_acc=round(float(sa),4),peak_mb=round(s.delta/1048576,1),state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__=="__main__":
    res=[]
    for n in (20_000, 100_000):
        for ch in (True, False):
            res.append(run(n, ch))
    p=Path(".pingis/benches/chains.json"); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"algo":"D.5.2","runs":res},indent=2)); print("wrote",p)
