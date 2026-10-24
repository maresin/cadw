#!/usr/bin/env python3
"""Bench: D.2.3 hybrid MF + content vs plain ALS (quality + cost)."""
from __future__ import annotations
import gc, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pyarrow as pa, psutil
from cadw import CadwDataFrame
from cadw.ml.als import fit_als
from cadw.ml.hybrid_mf import fit_hybrid_mf

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


def run(U, I, c=6, k=8, iters=5, seed=0):
    rng = np.random.RandomState(seed)
    Xu = rng.normal(size=(U, k)); Ct = rng.normal(size=(I, c))
    Yi = Ct @ rng.normal(size=(c, k)) + 0.05 * rng.normal(size=(I, k))
    us, its = [], []
    for u in range(U):
        for i in range(I):
            if Xu[u] @ Yi[i] > 0.5:
                us.append(u); its.append(i)
    d = {"u": pa.array(us, type=pa.int64()), "i": pa.array(its, type=pa.int64())}
    for j in range(c):
        d[f"c{j}"] = pa.array([Ct[i, j] for i in its])
    df = CadwDataFrame.from_table(pa.table(d))
    cols = [f"c{j}" for j in range(c)]
    true = Xu @ Yi.T
    with RSS() as s:
        t0 = time.time()
        hyb = fit_hybrid_mf(df, "u", "i", cols, n_factors=k, n_iter=iters,
                            reg=0.05, alpha=5.0, ridge=0.5, seed=seed)
        hyb_s = time.time() - t0
    with RSS() as s2:
        t0 = time.time()
        base = fit_als(df, "u", "i", n_factors=k, n_iter=iters,
                       reg=0.05, alpha=5.0, seed=seed)
        als_s = time.time() - t0
    r_als = float(np.corrcoef(base.scores(0), true[0])[0, 1])
    r_hyb = float(np.corrcoef(hyb.scores(0), true[0])[0, 1])
    return dict(n_users=U, n_items=I, interactions=len(us), k=k, content=c,
                iters=iters, als_fit_s=round(als_s, 2),
                hybrid_fit_s=round(hyb_s, 2),
                hybrid_peak_mb=round(s.delta / 1048576, 1),
                als_peak_mb=round(s2.delta / 1048576, 1),
                state_b=hyb.state_nbytes(),
                corr_als_user0=round(r_als, 4),
                corr_hybrid_user0=round(r_hyb, 4),
                delta_corr=round(r_hyb - r_als, 4))


if __name__ == "__main__":
    res = {}
    for U, I in ((300, 200), (1200, 800)):
        key = f"{U}x{I}"
        res[key] = run(U, I); print(res[key]); gc.collect()
    Path(".pingis/benches").mkdir(parents=True, exist_ok=True)
    Path(".pingis/benches/hybrid_mf.json").write_text(json.dumps(res, indent=2))
    print("written")
