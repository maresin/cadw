#!/usr/bin/env python3
"""Bench: D.3.1 CoxPH vs lifelines (coef agreement, time, RSS)."""
from __future__ import annotations
import gc, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pyarrow as pa, psutil
from cadw import CadwDataFrame
from cadw.ml.coxph import fit_coxph

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


def make(n, d=5, seed=0, batch=50_000):
    rng = np.random.RandomState(seed)
    beta = np.linspace(0.6, -0.6, d)
    parts = []
    for i in range(0, n, batch):
        sz = min(batch, n - i)
        X = rng.normal(size=(sz, d))
        T = -np.log(rng.uniform(size=sz)) / np.exp(X @ beta) * 2.0
        C = rng.exponential(scale=3.0, size=sz)
        E = (T <= C).astype(np.int64)
        cols = {f"x{j}": pa.array(X[:, j]) for j in range(d)}
        cols["T"] = pa.array(np.minimum(T, C)); cols["E"] = pa.array(E, type=pa.int64())
        parts.append(pa.record_batch(cols))
    return CadwDataFrame.from_table(pa.Table.from_batches(parts)), X, beta


def run(n, d=5):
    df, X, beta = make(n, d)
    cols = [f"x{j}" for j in range(d)]
    with RSS() as s:
        t0 = time.time()
        m = fit_coxph(df, cols, "T", "E", max_iter=100)
        dt = time.time() - t0
    out = dict(n=n, d=d, fit_s=round(dt, 2), iters=m.n_iter_,
               converged=m.converged_, peak_mb=round(s.delta / 1048576, 1),
               state_b=m.state_nbytes(), c_index=round(m.concordance_index(df, "T", "E"), 4))
    try:
        import pandas as pd
        from lifelines import CoxPHFitter
        t = df._iter_filtered_batches()
        tbl = pa.Table.from_batches(list(t))
        pdf = pd.DataFrame({f"x{j}": tbl.column(f"x{j}").to_numpy() for j in range(d)})
        pdf["T"] = tbl.column("T").to_numpy(); pdf["E"] = tbl.column("E").to_numpy()
        t0 = time.time()
        ref = CoxPHFitter().fit(pdf, "T", "E")
        out["ref_fit_s"] = round(time.time() - t0, 2)
        out["ref_c_index"] = round(ref.concordance_index_, 4)
        out["coef_maxdiff"] = float(np.max(np.abs(m.coef_ - ref.params_.values)))
    except Exception as e:
        out["ref_error"] = type(e).__name__
    return out


if __name__ == "__main__":
    res = {}
    for n in (5_000, 50_000):
        res[str(n)] = run(n); print(res[str(n)]); gc.collect()
    Path(".pingis/benches").mkdir(parents=True, exist_ok=True)
    Path(".pingis/benches/coxph.json").write_text(json.dumps(res, indent=2))
    print("written")
