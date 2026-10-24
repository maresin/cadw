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

from cadw.ml.lda import fit_lda

TOPICS = {0: ["apple","banana","cherry","grape"],
          1: ["engine","brake","wheel","clutch"],
          2: ["guitar","piano","violin","drum"]}

def make(n, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 3, size=n)
    docs = [" ".join(rng.choice(TOPICS[int(y[i])], size=6, replace=True)) for i in range(n)]
    return CadwDataFrame.from_table(pa.table({"text": pa.array(docs)}))

def run(n):
    df = make(n); gc.collect()
    with RSS() as s:
        t0 = time.time(); from cadw.core.memory_guard import MemoryGuard
        mg = MemoryGuard(max_memory_mb=200)
        m = fit_lda(df, "text", n_topics=3, vocab_size=1000, seed=0, memory_guard=mg); dt = time.time()-t0
        g = m.transform(df, text_col="text")
        ppl = m.perplexity(df, text_col="text")
    out = dict(n=n, fit_s=round(dt,3), topics=m.n_topics_, vocab=len(m.vocab_),
               perplexity=round(float(ppl),3), peak_mb=round(s.delta/1048576,1),
               state_b=m.state_nbytes())
    print(json.dumps(out)); return out

if __name__ == "__main__":
    res = [run(n) for n in (2_000, 10_000, 50_000)]
    p = Path(".pingis/benches/lda.json"); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"algo": "D.6.6", "runs": res}, indent=2)); print("wrote", p)
