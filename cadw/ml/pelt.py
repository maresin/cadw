"""PELT change-point detection (D.4.2, v0.9.38).

Reference: Killick, Fearnhead, Eckley, *Optimal detection of changepoints
with a linear computational cost*, JASA 107(500), 2012 (PELT);
Truong, Oudre, Vayatis, *Selective review of offline change point detection
methods*, Signal Processing 167, 2020 (cost functions / pruning).
See .pingis/algorithm_notes/d4_2_pelt.md.

Exact PELT with the L2 (mean-shift) cost and the standard pruning rule.
Memory is O(window): the recursion horizon can be bounded by `max_window`,
which is what the backlog item means by "online" -- a bounded window keeps
O(window) state instead of O(n). With `max_window=None` the recursion is
exact full-series PELT and matches ruptures.Pelt(model="l2").
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit


def pelt(y, penalty, min_size=2, max_window=None):
    """PELT on a 1-D array. Returns change-point indices (excluding 0 and n)."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n == 0:
        raise ValueError("pelt: empty series")
    if not np.all(np.isfinite(y)):
        raise ValueError("pelt: series contains non-finite values")
    if penalty <= 0:
        raise ValueError(f"pelt: penalty must be > 0, got {penalty}")
    if min_size < 1:
        raise ValueError(f"pelt: min_size must be >= 1, got {min_size}")
    if max_window is not None and max_window < 2 * min_size:
        raise ValueError("pelt: max_window must be >= 2 * min_size")

    cs = np.concatenate([[0.0], np.cumsum(y)])
    cs2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def cost(a, b):
        s = b - a
        if s <= 1:
            return 0.0
        sm = cs[b] - cs[a]
        sm2 = cs2[b] - cs2[a]
        return float(sm2 - sm * sm / s)

    F = np.full(n + 1, np.inf, dtype=np.float64)
    cp = np.full(n + 1, -1, dtype=np.int64)
    F[0] = -float(penalty)
    R = [0]
    for t in range(1, n + 1):
        lo = 0 if max_window is None else max(0, t - max_window)
        best = np.inf
        best_s = -1
        keep = []
        for s in R:
            if s < lo:
                continue
            if t - s < min_size:
                keep.append(s)
                continue
            fsc = F[s] + cost(s, t)
            c = fsc + penalty
            if c < best:
                best = c
                best_s = s
            # Killick et al. 2012 pruning: s can be discarded for all
            # future t if F(s) + C(s,t) + K > F(t); keep it otherwise.
            if fsc + penalty <= best:
                keep.append(s)
        F[t] = best
        cp[t] = best_s
        keep.append(t)
        R = keep

    out = []
    t = n
    while t > 0:
        s = int(cp[t])
        if s <= 0:
            break
        out.append(s)
        t = s
    out.reverse()
    return out


def _collect_series(df, value_col, id_col, mg, name):
    """Yield (sid, np.ndarray) one series at a time, O(longest series)."""
    if id_col is None:
        vals = []
        n = 0
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name=name, extra_factor=3.0)
            vals.append(b.column(value_col).cast(pa.float64(), safe=False)
                        .to_numpy(zero_copy_only=False))
            n += b.num_rows
            mg.check_state(n * 8, name=name + ".state")
        if n == 0:
            raise ValueError("pelt: empty input")
        yield 0, np.concatenate(vals)
        return
    cur = None
    buf = []
    seen = set()
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name=name, extra_factor=3.0)
        ids = b.column(id_col).to_pylist()
        arr = b.column(value_col).cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        i = 0
        n = b.num_rows
        while i < n:
            j = i
            while j < n and ids[j] == ids[i]:
                j += 1
            sid = ids[i]
            if cur is None:
                cur = sid
            if sid != cur:
                if sid in seen:
                    raise ValueError("pelt: input must be grouped/sorted by id_col")
                seen.add(cur)
                yield cur, np.concatenate(buf)
                cur = sid
                buf = []
            buf.append(arr[i:j])
            mg.check_state(sum(x.size for x in buf) * 8, name=name + ".state")
            i = j
    if cur is None:
        raise ValueError("pelt: empty input")
    if buf:
        yield cur, np.concatenate(buf)


class PELTModel:
    """Fitted PELT parameters (detection itself is deterministic)."""

    def __init__(self, value_col, id_col, params):
        self.value_col_ = value_col
        self.id_col_ = id_col
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return 256

    def to_dict(self):
        return {"params": self.params_, "value_col": self.value_col_,
                "id_col": self.id_col_}

    def detect(self, df, memory_guard=None):
        """Return {series_id: [change_point, ...]} for every series."""
        mg = memory_guard or df._memory_guard
        pen = self.params_["penalty"]
        ms = self.params_["min_size"]
        mw = self.params_["max_window"]
        out = {}
        for sid, y in _collect_series(df, self.value_col_, self.id_col_, mg, "pelt.detect"):
            out[sid] = pelt(y, pen, min_size=ms, max_window=mw)
        return out

    def change_points(self, df, memory_guard=None):
        """Flat list of (series_id, index) pairs."""
        d = self.detect(df, memory_guard)
        return [(sid, i) for sid, cps in d.items() for i in cps]


def fit_pelt(df, value_col, *, id_col=None, penalty=10.0, min_size=2,
             max_window=None, memory_guard: Optional[MemoryGuard] = None):
    """Fit (validate) a PELT detector and bind its parameters."""
    if value_col not in df._columns:
        raise KeyError(f"fit_pelt: column {value_col!r} not in {df._columns}")
    if id_col is not None and id_col not in df._columns:
        raise KeyError(f"fit_pelt: column {id_col!r} not in {df._columns}")
    if penalty <= 0:
        raise ValueError(f"fit_pelt: penalty must be > 0, got {penalty}")
    mg = memory_guard or df._memory_guard
    m = PELTModel(value_col, id_col,
                  {"penalty": float(penalty), "min_size": int(min_size),
                   "max_window": max_window})
    n = 0
    for _sid, y in _collect_series(df, value_col, id_col, mg, "pelt.fit"):
        if y.size < 2 * min_size:
            raise ValueError(f"fit_pelt: series shorter than 2*min_size={2*min_size}")
        n += y.size
    if n == 0:
        raise ValueError("fit_pelt: empty input")
    log_fit("pelt", state_bytes=m.state_nbytes())
    mg.check_state(m.state_nbytes(), name="pelt.model")
    return m
