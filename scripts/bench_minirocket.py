#!/usr/bin/env python3
"""Bench: D.4.1 MiniRocket streaming features (time, RSS, feature count)."""
from __future__ import annotations
import gc, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pyarrow as pa, psutil
from cadw import CadwDataFrame
from cadw.ml.minirocket import fit_minirocket


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


def make(n_series, length, seed=0):
    rng = np.random.RandomState(seed)
    ids = np.repeat(np.arange(n_series, dtype=np.int64), length)
    vals = np.sin(np.arange(n_series * length) * 0.01) + rng.normal(scale=0.01, size=n_series * length)
    return CadwDataFrame.from_table(pa.table({
        "id": pa.array(ids, type=pa.int64()), "v": pa.array(vals)}))


def run(n_series, length, num_kernels=84):
    df = make(n_series, length)
    gc.collect()
    with RSS() as s:
        t0 = time.time()
        m = fit_minirocket(df, "v", id_col="id", num_kernels=num_kernels)
        F = m.transform(df)
        dt = time.time() - t0
    out = dict(n_series=n_series, length=length, num_kernels=num_kernels,
               feature_shape=list(F.shape), fit_transform_s=round(dt, 2),
               series_per_s=round(n_series / dt, 1),
               peak_mb=round(s.delta / 1048576, 1),
               state_b=m.state_nbytes())
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    results = [run(200, 500), run(1000, 500), run(500, 2000)]
    outp = Path(".pingis/benches/minirocket.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"algo": "D.4.1", "runs": results}, indent=2))
    print("wrote", outp)
