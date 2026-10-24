#!/usr/bin/env python3
"""Bench: THETA."""
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

from cadw.ts.theta import ThetaForecaster

def make(n, seed=0):
    t=np.arange(n)
    y=10+0.4*t+np.random.RandomState(seed).normal(0,0.5,n)
    return CadwDataFrame.from_table(pa.table({"y":pa.array(y)}))

def run(n):
    df=make(n); gc.collect()
    with RSS() as s_:
        t0=time.time(); m=ThetaForecaster().fit(df,"y"); f=m.predict(12); dt=time.time()-t0
    out=dict(n=n, fit_predict_s=round(dt,3), alpha=round(m.alpha_,3), sigma=round(float(np.sqrt(m.sigma2_)),4), peak_mb=round(s_.delta/1048576,1), state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__=="__main__":
    res=[run(n) for n in (1000,10000,100000)]
    p=Path(".pingis/benches/theta.json"); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"algo":"D.4.4","runs":res},indent=2)); print("wrote",p)
