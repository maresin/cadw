#!/usr/bin/env python3
"""Bench: ordinal_logit."""
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

from cadw.ml.ordinal import fit_ordinal_logit

def make(n, seed=0, d=2):
    rng=np.random.RandomState(seed)
    X=rng.normal(size=(n,d))
    lat=-0.5+1.2*X[:,0]-0.7*X[:,1]+rng.normal(0,0.5,n)
    y=np.digitize(lat,[-0.5,0.5,1.5]).astype(np.float64)
    d_={f"x{j}":pa.array(X[:,j]) for j in range(d)}
    d_["y"]=pa.array(y)
    return CadwDataFrame.from_table(pa.table(d_)), y

def run(n):
    df,y=make(n); gc.collect()
    with RSS() as s:
        t0=time.time(); m=fit_ordinal_logit(df,["x0","x1"],"y"); dt=time.time()-t0
        acc=float((m.predict(df)==y).mean())
    out=dict(n=n,fit_s=round(dt,3),n_iter=m.params_["n_iter"],success=m.params_["success"],acc=round(acc,4),peak_mb=round(s.delta/1048576,1),state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__=="__main__":
    res=[run(n) for n in (5_000, 50_000, 200_000)]
    p=Path(".pingis/benches/ordinal_logit.json"); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"algo":"D.5.1","runs":res},indent=2)); print("wrote",p)
