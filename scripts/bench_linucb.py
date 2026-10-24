#!/usr/bin/env python3
"""Bench: tfidf."""
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
        return max(self.peak - self.b, 0)

from cadw.ml.linucb import fit_linucb

def make(n, d=8, K=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    arms = rng.randint(0, K, size=n)
    th = rng.normal(size=(K, d))
    r = np.array([X[i] @ th[arms[i]] for i in range(n)])
    dct = {f"x{j}": pa.array(X[:, j]) for j in range(d)}
    dct["arm"] = pa.array(arms.astype(np.int64)); dct["r"] = pa.array(r)
    return CadwDataFrame.from_table(pa.table(dct)), [f"x{j}" for j in range(d)]

def run(n):
    df, cols = make(n); gc.collect()
    with RSS() as s:
        t0 = time.time(); m = fit_linucb(df, cols, arm_col="arm", reward_col="r"); dt = time.time()-t0
        ch = m.choose(df)
    out = dict(n=n, fit_s=round(dt,3), arms=m.n_arms_, chosen=int(np.bincount(ch, minlength=m.n_arms_).tolist()[0]),
               peak_mb=round(s.delta/1048576,1), state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__ == "__main__":
    # sizes chosen to fit the 100 MB batch guard (extra_factor=3.0);
    # the guard is not relaxed
    res = [run(n) for n in (10_000, 50_000, 150_000)]
    p = Path(".pingis/benches/linucb.json"); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"algo": "D.6.7", "runs": res}, indent=2)); print("wrote", p)
