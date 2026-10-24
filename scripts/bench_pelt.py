#!/usr/bin/env python3
"""Bench: PELT."""
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

from cadw.ml.pelt import fit_pelt
import ruptures as rpt

def make(n, seed=0):
    rng = np.random.RandomState(seed)
    y = np.concatenate([rng.normal(0,1,n//3), rng.normal(5,1,n//3), rng.normal(0,1,n - 2*(n//3))])
    return CadwDataFrame.from_table(pa.table({"v": pa.array(y)}))

def run(n):
    df = make(n); gc.collect()
    with RSS() as s:
        t0=time.time(); m=fit_pelt(df,"v",penalty=30.0); cps=m.detect(df)[0]; dt=time.time()-t0
        t1=time.time(); ref=rpt.Pelt(model="l2",min_size=2).fit(pa.Table.from_batches(list(df._iter_filtered_batches())).column("v").to_numpy(zero_copy_only=False)).predict(pen=30.0); dt_ref=time.time()-t1
    out=dict(n=n, fit_detect_s=round(dt,3), ruptures_s=round(dt_ref,3), cps=cps[:6], ruptures=ref[:-1], peak_mb=round(s.delta/1048576,1), state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__=="__main__":
    res=[run(n) for n in (1000,5000,20000)]
    p=Path(".pingis/benches/pelt.json"); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"algo":"D.4.2","runs":res},indent=2)); print("wrote",p)
