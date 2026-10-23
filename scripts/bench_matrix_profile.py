#!/usr/bin/env python3
"""Bench: D.1.2 Matrix Profile — bounded reservoir vs full brute force."""
from __future__ import annotations
import gc, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pyarrow as pa, psutil
from cadw import CadwDataFrame
from cadw.ml.matrix_profile import fit_matrix_profile

class RSS:
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

def run(n, w=32, S=4096, batch=100_000):
    rng = np.random.RandomState(0)
    batches = []
    for i in range(0, n, batch):
        sz = min(batch, n - i)
        x = np.sin(np.arange(sz) * 0.01) + rng.normal(scale=0.1, size=sz)
        batches.append(pa.record_batch({"v": pa.array(x)}))
    df = CadwDataFrame.from_table(pa.Table.from_batches(batches))
    with RSS() as s:
        t0 = time.time()
        m = fit_matrix_profile(df, "v", window=w, max_samples=S, random_state=0)
        dt = time.time() - t0
        t1 = time.time()
        sc = m.discord_scores()
        top = float(np.max(sc))
        tscore = time.time() - t1
    return dict(n=n, window=w, sample=S, fit_s=round(dt, 3),
                score_s=round(tscore, 4), peak_mb=round(s.delta / 1048576, 1),
                state_b=m.state_nbytes(), n_seen=m.n_seen_,
                sample_fraction=round(m.sample_fraction_, 5),
                max_discord=round(top, 4))

if __name__ == "__main__":
    out = {}
    for n in (100_000, 1_000_000):
        out[str(n)] = run(n); print(out[str(n)]); gc.collect()
    Path(".pingis/benches").mkdir(parents=True, exist_ok=True)
    Path(".pingis/benches/matrix_profile.json").write_text(json.dumps(out, indent=2))
    print("written")
