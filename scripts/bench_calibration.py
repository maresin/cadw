#!/usr/bin/env python3
"""Bench: calibration."""
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

from cadw.ml.calibration import fit_calibration

def make(n, seed=0, d=3):
    rng=np.random.RandomState(seed)
    X=rng.normal(size=(n,d))
    lp=-1+1.2*X[:,0]-0.5*X[:,1]
    y=(rng.uniform(size=n)<1/(1+np.exp(-lp))).astype(np.float64)
    d_={f"x{j}":pa.array(X[:,j]) for j in range(d)}
    d_["y"]=pa.array(y)
    return CadwDataFrame.from_table(pa.table(d_))

def run(n, method, est=50):
    df=make(n); gc.collect()
    with RSS() as s:
        t0=time.time(); m=fit_calibration(df,["x0","x1","x2"],"y",method=method,n_estimators=est); dt=time.time()-t0
        b=m.brier(df); e=m.ece(df)
    out=dict(n=n,method=method,fit_s=round(dt,2),brier=round(float(b),4),ece=round(float(e),4),peak_mb=round(s.delta/1048576,1),state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__=="__main__":
    res=[]
    for n in (20_000, 100_000):
        for meth in ("platt","isotonic"):
            res.append(run(n,meth))
    p=Path(".pingis/benches/calibration.json"); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"algo":"D.4.6","runs":res},indent=2)); print("wrote",p)
