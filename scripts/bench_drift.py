#!/usr/bin/env python3
"""Bench: drift."""
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

from cadw.ml.drift import ks_2samp, psi, chi2_categorical, welch_t, mann_whitney, cuped

def run(n):
    rng=np.random.RandomState(0)
    a=rng.normal(0,1,n); b=rng.normal(0.3,1,n); c1=rng.randint(0,4,n); c2=rng.randint(0,4,n)
    x=rng.normal(size=n); y=0.3+0.8*x+rng.normal(0,0.2,n); t=np.repeat([0,1],n//2)
    gc.collect()
    with RSS() as s:
        t0=time.time()
        r_ks=ks_2samp(a,b); r_psi=psi(a,b); r_chi=chi2_categorical(c1,c2); r_t=welch_t(a,b); r_mw=mann_whitney(a,b); r_cu=cuped(y,x,t)
        dt=time.time()-t0
    out=dict(n=n, total_s=round(dt,3), ks_p=float(r_ks[1]), psi=round(float(r_psi),4), chi2_p=float(r_chi[1]), t_p=float(r_t[1]), mw_p=float(r_mw[1]), cuped_theta=round(float(r_cu["theta"]),4), peak_mb=round(s.delta/1048576,1), state_b=256)
    print(json.dumps(out)); return out

if __name__=="__main__":
    res=[run(n) for n in (10_000,100_000,1_000_000)]
    p=Path(".pingis/benches/drift.json"); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"algo":"D.6.2","runs":res},indent=2)); print("wrote",p)
