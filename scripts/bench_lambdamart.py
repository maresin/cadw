#!/usr/bin/env python3
"""Bench: D.2.1 LambdaMART — time/RSS vs sklearn pointwise GBDT."""
from __future__ import annotations
import gc, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pyarrow as pa, psutil
from cadw import CadwDataFrame
from cadw.ml.lambdamart import fit_lambdamart, ndcg_at_k

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

def make(nq, per_q, d=8, seed=0):
    rng = np.random.RandomState(seed)
    rows, qs, ys = [], [], []
    for q in range(nq):
        X = rng.normal(size=(per_q, d))
        s = X[:, 0] + 0.5 * X[:, 1] + 0.25 * X[:, 2]
        rel = (s > 0.5).astype(int) + (s > 1.2).astype(int) + (s > 2.0).astype(int)
        rows.append(X); qs += [q] * per_q; ys += rel.tolist()
    X = np.vstack(rows)
    cols = {f"f{j}": pa.array(X[:, j]) for j in range(d)}
    cols["y"] = pa.array(ys, type=pa.int64()); cols["q"] = pa.array(qs, type=pa.int64())
    return CadwDataFrame.from_table(pa.table(cols)), X, np.array(ys), np.array(qs)

def mean_ndcg(y, q, s, k=10):
    return float(np.mean([ndcg_at_k(y[q == u], s[q == u], k)
                          for u in sorted(set(q.tolist()))]))

def run(nq, per_q, d=8):
    # NOTE: N x d doubles must fit under check_batch(extra_factor=2.0);
    # 1M rows x 8 features raises MemoryLimitExceeded and is not benched.
    df, X, y, q = make(nq, per_q, d)
    cols = [f"f{j}" for j in range(d)]
    with RSS() as s:
        t0 = time.time()
        m = fit_lambdamart(df, cols, "y", "q", n_estimators=20, max_depth=3)
        dt = time.time() - t0
        sc = m.predict(df).column("score").to_numpy(zero_copy_only=False)
    out = dict(n=len(y), queries=nq, per_query=per_q, d=d,
               fit_s=round(dt, 3), peak_mb=round(s.delta / 1048576, 1),
               state_b=m.state_nbytes(), ndcg10=round(mean_ndcg(y, q, sc), 4))
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        t0 = time.time()
        ref = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0).fit(X, y)
        out["ref_fit_s"] = round(time.time() - t0, 3)
        out["ref_ndcg10"] = round(mean_ndcg(y, q, ref.predict(X)), 4)
    except Exception as e:
        out["ref_error"] = type(e).__name__
    return out

if __name__ == "__main__":
    res = {}
    for nq, per_q in ((2000, 25), (8000, 50)):
        key = f"{nq}x{per_q}"
        res[key] = run(nq, per_q); print(res[key]); gc.collect()
    Path(".pingis/benches").mkdir(parents=True, exist_ok=True)
    Path(".pingis/benches/lambdamart.json").write_text(json.dumps(res, indent=2))
    print("written")
