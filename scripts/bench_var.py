#!/usr/bin/env python3
"""Bench: VAR."""
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

from cadw.ts.var import VARModel

def make(n, k=4, seed=0):
    rng=np.random.RandomState(seed)
    A=0.3*np.eye(k)+0.05*rng.normal(size=(k,k))
    Y=np.zeros((n,k))
    for t in range(1,n):
        Y[t]=A@Y[t-1]+rng.normal(0,1,k)
    d={f"y{j}":pa.array(Y[:,j]) for j in range(k)}
    return CadwDataFrame.from_table(pa.table(d)), [f"y{j}" for j in range(k)]

def run(n, k=4, p_order=2):
    df,cols=make(n,k); gc.collect()
    with RSS() as s_:
        t0=time.time(); m=VARModel(p=p_order).fit(df,cols); f=m.forecast_vector(8); dt=time.time()-t0
    out=dict(n=n, k=k, p=p_order, fit_forecast_s=round(dt,3), stable=m.stable_, peak_mb=round(s_.delta/1048576,1), state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__=="__main__":
    res=[run(1000,2,1),run(10000,4,2),run(50000,4,2)]
    p=Path(".pingis/benches/var.json"); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"algo":"D.4.5","runs":res},indent=2)); print("wrote",p)
