#!/usr/bin/env python3
"""Bench: geo."""
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

from cadw.ml.geo import fit_geo_agg, haversine

def make(n, seed=0):
    rng=np.random.RandomState(seed)
    lat=rng.uniform(55.7,55.8,n); lon=rng.uniform(37.5,37.7,n); v=rng.normal(size=n)
    return CadwDataFrame.from_table(pa.table({"lat":pa.array(lat),"lon":pa.array(lon),"v":pa.array(v)}))

def run(n, cell=500.0):
    df=make(n); gc.collect()
    with RSS() as s:
        t0=time.time(); m=fit_geo_agg(df,"lat","lon",cell_size_m=cell); k1,c1=m.hex_aggregate(df,agg="count"); k2,s2=m.hex_aggregate(df,value_col="v",agg="sum"); dt=time.time()-t0
    out=dict(n=n, cell_size_m=cell, cells=int(k1.shape[0]), total=float(c1.sum()), fit_agg_s=round(dt,3), peak_mb=round(s.delta/1048576,1), state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__=="__main__":
    res=[run(10_000),run(100_000),run(1_000_000)]
    p=Path(".pingis/benches/geo.json"); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"algo":"D.6.5","runs":res},indent=2)); print("wrote",p)
