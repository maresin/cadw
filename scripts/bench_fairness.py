#!/usr/bin/env python3
"""Bench: D.1.4 fairness metrics — streaming vs numpy materialized."""
from __future__ import annotations
import gc, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pyarrow as pa, psutil
from cadw import CadwDataFrame
from cadw.ml.fairness import fairness_metrics

class S:
    def __init__(self, iv=0.005):
        self.p=psutil.Process(); self.b=self.p.memory_info().rss; self.peak=self.b
        self._s=threading.Event(); self._iv=iv
    def _run(self):
        while not self._s.is_set():
            r=self.p.memory_info().rss
            if r>self.peak: self.peak=r
            self._s.wait(self._iv)
    def __enter__(self):
        self.t=threading.Thread(target=self._run,daemon=True); self.t.start(); return self
    def __exit__(self,*a): self._s.set(); self.t.join(timeout=2.0)
    @property
    def delta(self): return self.peak-self.b

def run(n, ngroups=8, batch=50_000):
    rng = np.random.RandomState(0)
    batches = []
    for i in range(0, n, batch):
        sz = min(batch, n - i)
        batches.append(pa.record_batch({
            "g": pa.array([f"g{j}" for j in rng.randint(0, ngroups, size=sz)]),
            "yt": pa.array(rng.randint(0, 2, size=sz), type=pa.int64()),
            "yp": pa.array(rng.randint(0, 2, size=sz), type=pa.int64())}))
    df = CadwDataFrame.from_table(pa.Table.from_batches(batches))
    with S() as s:
        t0 = time.time()
        rep = fairness_metrics(df, sensitive_col="g", y_true_col="yt", y_pred_col="yp")
        dt = time.time() - t0
    return dict(n=n, groups=ngroups, time_s=round(dt, 3),
                peak_mb=round(s.delta/1048576, 1), state_b=rep.state_nbytes(),
                dp=round(rep.metrics["demographic_parity_difference"], 4),
                eo=round(rep.metrics["equal_opportunity_difference"], 4))

if __name__ == "__main__":
    out = {}
    for n in (100_000, 1_000_000):
        out[str(n)] = run(n); print(out[str(n)]); gc.collect()
    Path(".pingis/benches").mkdir(parents=True, exist_ok=True)
    Path(".pingis/benches/fairness.json").write_text(json.dumps(out, indent=2))
    print("written .pingis/benches/fairness.json")
