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

from cadw.ml.tfidf import fit_tfidf

WORDS = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel",
         "india","juliet","kilo","lima","mike","november","oscar","papa"]

def make(n, seed=0, L=8):
    rng = np.random.RandomState(seed)
    docs = []
    for _ in range(n):
        k = rng.randint(L//2, L+1)
        docs.append(" ".join(rng.choice(WORDS, size=k, replace=True)))
    return CadwDataFrame.from_table(pa.table({"text": pa.array(docs)}))

def run(n):
    df = make(n); gc.collect()
    with RSS() as s:
        t0 = time.time(); m = fit_tfidf(df, "text", vocab_size=1000); dt = time.time()-t0
        rows = m.transform(df, text_col="text")
    nnz = sum(len(i) for i, _ in rows)
    out = dict(n=n, fit_s=round(dt,3), vocab=len(m.vocab_), nnz=int(nnz),
               peak_mb=round(s.delta/1048576,1), state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__ == "__main__":
    res = [run(n) for n in (10_000, 100_000, 500_000)]
    p = Path(".pingis/benches/tfidf.json"); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"algo": "D.6.9", "runs": res}, indent=2)); print("wrote", p)
