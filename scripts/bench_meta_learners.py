#!/usr/bin/env python3
"""Bench: D.3.2 T/S-learner + propensity (time, RSS, ATE accuracy)."""
from __future__ import annotations
import gc, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pyarrow as pa, psutil
from cadw import CadwDataFrame
from cadw.ml.meta_learners import fit_meta_learner


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


def make(n, seed=0, d=3, batch=50_000):
    rng = np.random.RandomState(seed)
    parts = []
    for i in range(0, n, batch):
        sz = min(batch, n - i)
        X = rng.normal(size=(sz, d))
        e = 1.0 / (1.0 + np.exp(-(0.5 * X[:, 0] - 0.3 * X[:, 1])))
        t = (rng.uniform(size=sz) < e).astype(np.float64)
        tau = 1.0 + 0.5 * X[:, 2]
        y = 2.0 * X[:, 0] + tau * t + rng.normal(scale=0.1, size=sz)
        cols = {f"x{j}": pa.array(X[:, j]) for j in range(d)}
        cols["t"] = pa.array(t); cols["y"] = pa.array(y)
        parts.append(pa.record_batch(cols))
    return CadwDataFrame.from_table(pa.Table.from_batches(parts))


def run(n, kind, d=3, est=50):
    df = make(n, d=d)
    cols = [f"x{j}" for j in range(d)]
    gc.collect()
    with RSS() as s:
        t0 = time.time()
        m = fit_meta_learner(df, cols, "t", "y", kind=kind, n_estimators=est)
        dt = time.time() - t0
        ate = m.ate(df)
    out = dict(n=n, d=d, kind=kind, n_estimators=est,
               fit_s=round(dt, 2), ate=round(float(ate), 4),
               peak_mb=round(s.delta / 1048576, 1),
               state_b=m.state_nbytes(),
               overlap=m.to_dict()["overlap"])
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    results = []
    for n in (20_000, 100_000):
        for kind in ("t", "s"):
            results.append(run(n, kind))
    outp = Path(".pingis/benches/meta_learners.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"algo": "D.3.2", "runs": results}, indent=2))
    print("wrote", outp)
