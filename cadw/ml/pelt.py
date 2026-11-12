"""PELT change-point detection (D.4.2, v0.9.38).

Reference: Killick, Fearnhead, Eckley, *Optimal detection of changepoints
with a linear computational cost*, JASA 107(500), 2012 (PELT);
Truong, Oudre, Vayatis, *Selective review of offline change point detection
methods*, Signal Processing 167, 2020 (cost functions / pruning).
See DECISION_LOG for algorithm reference.

Exact PELT with the L2 (mean-shift) cost and the standard pruning rule.
Memory is O(window): the recursion horizon can be bounded by `max_window`,
which is what the backlog item means by "online" -- a bounded window keeps
O(window) state instead of O(n). With `max_window=None` the recursion is
exact full-series PELT and matches ruptures.Pelt(model="l2").
"""
from __future__ import annotations

from typing import Optional

from pathlib import Path
import numpy as np
import pyarrow as pa

from cadw.core.approximation import ApproximationInfo, ApproximationMixin
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


def _iter_series_batches(df, value_col, id_col, memory_guard, name,
                         batch_size=None):
    """Yield (series_id, batch) for streaming PELT.

    For id_col=None: yields (None, batch) with only value_col.
    For id_col!=None: fallback to _collect_series per series (bounded by
    guard; multi-series OOC is deferred to v0.9.181+).
    """
    if id_col is None:
        for b in df._iter_filtered_batches(batch_size=batch_size):
            memory_guard.check_batch(
                b, name=f"{name}.input", extra_factor=1.5)
            # Select only value_col (single-column batch for streaming PELT)
            arr = b.column(value_col)
            single = pa.RecordBatch.from_arrays([arr], names=["v"])
            yield None, single
    else:
        # Multi-series fallback (materializes per-series).
        for sid, y in _collect_series(df, value_col, id_col, memory_guard, name):
            arr = pa.array(y, type=pa.float64())
            single = pa.RecordBatch.from_arrays([arr], names=["v"])
            yield sid, single


def _resolve_w_eff(mw_raw, n_s, W_max, ms, strict):
    """Resolve effective window for one series. Returns (W_eff, list[ApproximationInfo]).

    See PELT docs for semantics.
    """
    infos = []
    if mw_raw == "auto" or mw_raw is None:
        W_eff = min(n_s, W_max)
        if strict and mw_raw is None and W_eff < n_s:
            from cadw.core.memory_guard import MemoryLimitExceeded
            raise MemoryLimitExceeded(
                f"pelt: max_window=None (unbounded) with n={n_s} > "
                f"W_max={W_max}; strict=True refuses.",
                model="fit_pelt",
                params={"max_window": None, "strict": True},
                min_required_bytes=n_s * 8 * 5,
                available_bytes=None,
                options=[
                    f"increase budget to fit n={n_s}",
                    "set strict=False to auto-adjust with warning",
                    f"set max_window=int (<= {W_max}) explicitly",
                ],
            )
        if W_eff < n_s:
            # W_eff < n_s => data loss (changepoints beyond W not detected).
            # Always "window_truncated" regardless of requested value type:
            # "auto" also truncates if picked W < n; "None" adjusts unbounded
            # request down to W_max < n.
            infos.append(ApproximationInfo(
                type="window_truncated",
                axis="n",
                requested=mw_raw,
                effective=W_eff,
                reason="budget_limit",
                lost_range=(0, n_s - W_eff),
                accuracy_impact="changepoints beyond W_eff not detected",
            ))
        return W_eff, infos
    if isinstance(mw_raw, int):
        if mw_raw < 2 * ms:
            raise ValueError(
                f"pelt: max_window={mw_raw} < 2*min_size={2 * ms}"
            )
        W_eff = min(mw_raw, n_s, W_max)
        if W_eff < mw_raw:
            if W_eff == n_s and W_eff < W_max:
                reason = "n_smaller_than_requested"
            else:
                reason = "budget_limit"
            if strict:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"pelt: max_window={mw_raw} adjusted to {W_eff} "
                    f"({reason}); strict=True refuses.",
                    model="fit_pelt",
                    params={"max_window": mw_raw, "strict": True},
                    min_required_bytes=mw_raw * 8 * 3,
                    available_bytes=None,
                    options=[
                        f"increase budget to fit W={mw_raw}",
                        "set strict=False to auto-adjust with warning",
                    ],
                )
            infos.append(ApproximationInfo(
                type="window_adjusted", axis="n",
                requested=mw_raw, effective=W_eff, reason=reason,
                accuracy_impact="DP horizon reduced",
            ))
        elif W_eff < n_s:
            infos.append(ApproximationInfo(
                type="window_truncated", axis="n",
                requested=mw_raw, effective=W_eff, reason="explicit",
                lost_range=(0, n_s - W_eff),
                accuracy_impact="changepoints beyond W not detected",
            ))
        return W_eff, infos
    raise ValueError(
        f"pelt: max_window must be 'auto'|int|None, got {type(mw_raw).__name__}"
    )


def _pelt_streaming_batches(batch_iter, penalty, min_size, W_eff,
                            memory_guard, spill_dir=None):
    """Streaming PELT: O(W_eff) RAM, O(n * W_eff) time.

    batch_iter: iterator of batches, each with single f64 column at index 0.
    Returns: list of change point indices (excluding 0, n).

    Algorithm: standard PELT DP with:
      - ring buffer of last W_eff (cs, cs2, F) values  (3 * W_eff * f64)
      - backtrack array (n * i64) written to a temp file
    No full-series arrays are materialized.
    """
    import tempfile
    import os as _os

    cap = int(W_eff) + 1
    memory_guard.check_state(
        3 * cap * 8 + 256, name="pelt.streaming.state")
    cs_buf = np.zeros(cap, dtype=np.float64)
    cs2_buf = np.zeros(cap, dtype=np.float64)
    F_buf = np.full(cap, np.inf, dtype=np.float64)

    if spill_dir is not None:
        try:
            Path(spill_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            spill_dir = None
    if spill_dir is None:
        bt_path = Path(tempfile.mktemp(suffix=".pelt_bt.bin"))
    else:
        bt_path = Path(spill_dir) / f"pelt_bt_{_os.getpid()}_{id(batch_iter)}.bin"

    cs_total = 0.0
    cs2_total = 0.0
    t = 0

    bt_handle = open(bt_path, "wb")
    try:
        for batch in batch_iter:
            memory_guard.check_batch(
                batch, name="pelt.streaming.batch", extra_factor=2.0)
            col = batch.column(0)
            y_arr = col.to_numpy(zero_copy_only=False).astype(
                np.float64, copy=False)
            for y_val in y_arr:
                t += 1
                cs_total += y_val
                cs2_total += y_val * y_val
                slot_t = t % cap
                cs_buf[slot_t] = cs_total
                cs2_buf[slot_t] = cs2_total

                lo = max(0, t - int(W_eff))
                hi = t - int(min_size)
                if hi < lo:
                    F_buf[slot_t] = np.inf
                    bt_handle.write(np.int64(-1).tobytes())
                    continue

                a_range = np.arange(lo, hi + 1, dtype=np.int64)
                slots_a = a_range % cap

                # a==0 special: cs[0]=0, cs2[0]=0, F[0]=-penalty
                cs_a = np.where(a_range == 0, 0.0, cs_buf[slots_a])
                cs2_a = np.where(a_range == 0, 0.0, cs2_buf[slots_a])
                F_a = np.where(a_range == 0, -float(penalty), F_buf[slots_a])

                s = (t - a_range).astype(np.float64)
                ds = cs_total - cs_a
                ds2 = cs2_total - cs2_a
                costs = F_a + ds2 - ds * ds / s + float(penalty)

                bi = int(np.argmin(costs))
                F_buf[slot_t] = costs[bi]
                bt_handle.write(np.int64(a_range[bi]).tobytes())
    finally:
        bt_handle.close()

    # Backtrack from n to 0 via file (random access, but monotone-decreasing)
    cps: list[int] = []
    cur = t
    with open(bt_path, "rb") as bt_f:
        while cur > 0:
            bt_f.seek((cur - 1) * 8)
            raw = bt_f.read(8)
            if len(raw) < 8:
                break
            a = int(np.frombuffer(raw, dtype=np.int64)[0])
            if a <= 0:
                break
            if a < cur - 1:
                cps.append(a)
            cur = a
    cps.reverse()
    try:
        bt_path.unlink()
    except Exception:
        pass
    return cps


class PELTModel(ApproximationMixin):
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
        """Return {series_id: [change_point, ...]} for every series.

        v0.9.180: single-series (id_col=None) — true OOC streaming;
        W_eff resolved from n_total. Multi-series (id_col!=None) — per-
        series W_eff + numpy fallback (bounded_by_domain); streaming
        multi-series deferred v0.9.181+.
        """
        mg = memory_guard or df._memory_guard
        self._clear_approximation()

        pen = self.params_["penalty"]
        ms = int(self.params_["min_size"])
        mw_raw = self.params_["max_window"]
        strict = bool(self.params_.get("strict", False))

        W_max = mg.get_window_size(unit_bytes=16, target_fraction=0.25)
        approx_infos: list = []
        out: dict = {}

        if self.id_col_ is None:
            # === Single-series streaming (true OOC) ===
            n_total = int(df.count())
            W_eff, infos = _resolve_w_eff(mw_raw, n_total, W_max, ms, strict)
            approx_infos.extend(infos)

            try:
                bs = mg.get_batch_rows(expansion_factor=1.5)
            except Exception:
                bs = None
            _spill = getattr(df, "_spill_dir", None)

            def _yield_batches():
                for _, b in _iter_series_batches(
                        df, self.value_col_, None, mg, "pelt.detect",
                        batch_size=bs):
                    yield b

            out[0] = _pelt_streaming_batches(
                _yield_batches(), pen, ms, W_eff, mg, spill_dir=_spill,
            )
        else:
            # === Multi-series per-series (bounded_by_domain fallback) ===
            for sid, y in _collect_series(
                    df, self.value_col_, self.id_col_, mg, "pelt.detect"):
                n_s = int(y.size)
                if n_s < 2 * ms:
                    continue
                w_eff_s, infos = _resolve_w_eff(
                    mw_raw, n_s, W_max, ms, strict)
                approx_infos.extend(infos)
                mg.check_state(
                    (3 * n_s + 2 * w_eff_s) * 8 + 256,
                    name=f"pelt.detect[{sid}]",
                )
                pelt_mw = w_eff_s if w_eff_s < n_s else None
                out[sid] = pelt(y, pen, min_size=ms, max_window=pelt_mw)

        if approx_infos:
            self._record_approximations_batch(approx_infos)
        return out

    def change_points(self, df, memory_guard=None):
        """Flat list of (series_id, index) pairs."""
        d = self.detect(df, memory_guard)
        return [(sid, i) for sid, cps in d.items() for i in cps]


def fit_pelt(df, value_col, *, id_col=None, penalty=10.0, min_size=2,
             max_window: "str | int | None" = "auto",
             strict: bool = False,
             memory_guard: Optional[MemoryGuard] = None):
    """Fit (validate) a PELT detector and bind its parameters.

    v0.9.179: max_window supports "auto"|int|None.
      - "auto" (default): W_eff = min(n_series, W_max) per-series.
      - int W: W_eff = min(W, n_series, W_max); strict=True raises if W > W_max.
      - None: unbounded request; W_eff = min(n_series, W_max); strict=True
        raises if W_max < n_series.

    W_max = memory_guard.get_window_size(unit_bytes=16, target_fraction=0.25).
    Raises MemoryLimitExceeded if W_max < 2*min_size (insufficient_no_alt).
    """
    if value_col not in df._columns:
        raise KeyError(f"fit_pelt: column {value_col!r} not in {df._columns}")
    if id_col is not None and id_col not in df._columns:
        raise KeyError(f"fit_pelt: column {id_col!r} not in {df._columns}")
    if penalty <= 0:
        raise ValueError(f"fit_pelt: penalty must be > 0, got {penalty}")
    if not (max_window == "auto" or max_window is None
            or isinstance(max_window, int)):
        raise ValueError(
            f"fit_pelt: max_window must be 'auto'|int|None, "
            f"got {type(max_window).__name__}"
        )
    if isinstance(max_window, int) and max_window < 2 * min_size:
        raise ValueError(
            f"fit_pelt: max_window={max_window} must be >= 2*min_size={2*min_size}"
        )

    mg = memory_guard or df._memory_guard

    # Insufficient check: can't even fit minimum window
    W_max = mg.get_window_size(unit_bytes=16, target_fraction=0.25)
    if W_max < 2 * min_size:
        from cadw.core.memory_guard import MemoryLimitExceeded
        raise MemoryLimitExceeded(
            f"fit_pelt: W_max={W_max} (from budget) < 2*min_size="
            f"{2*min_size}. Increase memory_guard budget or reduce min_size.",
            model="fit_pelt",
            params={"min_size": int(min_size)},
            min_required_bytes=(3 * 2 * min_size * 8 + 2 * 2 * min_size * 8),
            available_bytes=mg.available_bytes(),
            options=[
                f"increase budget (min_required ~= {2 * min_size * 40} B)",
                f"reduce min_size (currently {min_size})",
            ],
        )

    m = PELTModel(value_col, id_col,
                  {"penalty": float(penalty), "min_size": int(min_size),
                   "max_window": max_window, "strict": bool(strict)})

    if id_col is None:
        # v0.9.180: single-series OOC — no _collect_series.
        n = int(df.count())
        if n == 0:
            raise ValueError("fit_pelt: empty input")
        if n < 2 * min_size:
            raise ValueError(
                f"fit_pelt: series shorter than 2*min_size={2*min_size}"
            )
    else:
        # Multi-series fallback (bounded_by_domain n<=1e6).
        n = 0
        for _sid, y in _collect_series(df, value_col, id_col, mg, "pelt.fit"):
            if y.size < 2 * min_size:
                raise ValueError(
                    f"fit_pelt: series shorter than 2*min_size={2*min_size}"
                )
            n += y.size
        if n == 0:
            raise ValueError("fit_pelt: empty input")

    log_fit("pelt", state_bytes=m.state_nbytes())
    mg.check_state(m.state_nbytes(), name="pelt.model")
    return m
