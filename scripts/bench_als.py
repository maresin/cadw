#!/usr/bin/env python3
"""Bench: D.2.2 implicit ALS at increasing interaction counts."""
from __future__ import annotations
import gc, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pyarrow as pa, psutil
from cadw import CadwDataFrame
from cadw.ml.als import fit_als

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


def run(n_users, n_items, per_user, k=16, iters=3, batch=100_000):
    rng = np.random.RandomState(0)
    us, its = [], []
    for u in range(n_users):
        items = rng.choice(n_items, size=per_user, replace=False)
        us.extend([u] * per_user); its.extend(items.tolist())
    batches = []
    for i in range(0, len(us), batch):
        batches.append(pa.record_batch({
            "u": pa.array(us[i:i + batch], type=pa.int64()),
            "i": pa.array(its[i:i + batch], type=pa.int64())}))
    df = CadwDataFrame.from_table(pa.Table.from_batches(batches))
    with RSS() as s:
        t0 = time.time()
        m = fit_als(df, "u", "i", n_factors=k, n_iter=iters, seed=0)
        dt = time.time() - t0
    return dict(n_users=n_users, n_items=n_items, interactions=len(us), k=k,
                iters=iters, fit_s=round(dt, 2),
                peak_mb=round(s.delta / 1048576, 1),
                state_b=m.state_nbytes())


if __name__ == "__main__":
    res = {}
    for nu, ni, pu in ((2000, 800, 20), (10000, 4000, 20)):
        key = f"{nu}x{ni}x{pu}"
        res[key] = run(nu, ni, pu); print(res[key]); gc.collect()
    Path(".pingis/benches").mkdir(parents=True, exist_ok=True)
    Path(".pingis/benches/als.json").write_text(json.dumps(res, indent=2))
    print("written")
