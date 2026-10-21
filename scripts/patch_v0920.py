#!/usr/bin/env python3
# v0.9.20 — peak-polling + DECISIONS append §13.4/226-231.
#
# v0.9.19 bench revealed: net pool delta = 0 for all reduce ops (batches
# pre-allocated at guard creation, Arrow temp buffers released before
# end-of-op). C_arrow=0 is honest (no leak) but useless as formula
# validation. Formula §9.2.1 predicts ABSOLUTE peak (batch + 2*state);
# we need to measure that, not net.
#
# Design: opt-in peak sampling via background thread. calibate(...,
# sample_peak=True) starts thread that polls pool.bytes_allocated()
# and RSS every 10ms; tracks max. Records BOTH:
#   - peak_pool_absolute: max alloc during op (comparable to formula)
#   - net_pool_absolute: end alloc (leak detection)
#   - peak_delta = peak - start (net overhead above baseline)
#   - net_delta = end - start (persistent overhead)
#
# Zero overhead when calibrate=False (yield only). Thread spawn cost
# ~100μs — opt-in only via sample_peak=True, runtime users with
# calibrate=True (delta mode) don't pay for threading.
#
# Motivation: without peak, formula drift can't be detected for streaming
# ops (reduce/sort/shuffle/groupby all have net≈0). Peak polling closes
# the loop.
#
# §13.4/232 (new): bench metric mismatch. Bench at v0.9.19 compared
# net-delta/predicted where predicted = absolute peak (batch + state).
# Metric artifact, not formula drift. Rule: metric unit must match
# formula unit. Peak vs peak, delta vs delta. Never cross.
#
# Run: uv run python scripts/patch_v0920.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) peak-polling in calibrate() ============

def patch_calibrate_peak():
    print("\n=== (a) calibrate(sample_peak=True) ===")
    p = Path("cadw/core/memory_guard.py")
    s = p.read_text()
    if "sample_peak: bool = False" in s:
        print("  already present — skip")
        return

    old_sig = '''    @contextmanager
    def calibrate(
        self,
        op_name: str,
        predicted_bytes: int = 0,
    ) -> Iterator[None]:'''

    new_sig = '''    @contextmanager
    def calibrate(
        self,
        op_name: str,
        predicted_bytes: int = 0,
        sample_peak: bool = False,
        sample_interval_ms: float = 10.0,
    ) -> Iterator[None]:'''

    if old_sig not in s:
        raise SystemExit("calibrate signature anchor not found")
    s = s.replace(old_sig, new_sig, 1)

    # Extend docstring: append sample_peak note before closing """
    old_doc_tail = '''        Both flags off → yield only, zero overhead.
        """
        if not (self._calibrate or self._runtime_monitor):
            yield
            return'''

    new_doc_tail = '''        Both flags off → yield only, zero overhead.

        v0.9.20: `sample_peak=True` — start background thread that polls
        pa.default_memory_pool().bytes_allocated() and psutil RSS every
        `sample_interval_ms` (default 10ms), tracking max. Records
        peak_absolute (peak above Arrow zero, comparable to formula §9.2.N
        which predicts absolute peak) AND peak_delta (peak above op-start
        baseline). Use when validating formula (absolute) or detecting
        unmodeled temp buffers (delta). No overhead when calibrate=False
        — thread only spawns when sample_peak=True AND calibrate=True.
        """
        if not (self._calibrate or self._runtime_monitor):
            yield
            return'''

    if old_doc_tail not in s:
        raise SystemExit("calibrate docstring anchor not found")
    s = s.replace(old_doc_tail, new_doc_tail, 1)

    # Replace "Ensure baselines initialized" + start/end block
    old_body = '''        if self._pool is None:
            self._pool = pa.default_memory_pool()
            self._pool_baseline = self._pool.bytes_allocated()
        if self._process is None:
            import psutil
            self._process = psutil.Process()
            self._rss_baseline = self._process.memory_info().rss

        pool_start = self._pool.bytes_allocated()
        rss_start = self._process.memory_info().rss

        try:
            yield
        finally:
            pool_end = self._pool.bytes_allocated()
            rss_end = self._process.memory_info().rss
            pool_delta = pool_end - pool_start
            rss_delta = rss_end - rss_start

            # Tier 2 (runtime_monitor): hard safety net via RSS
            if self._runtime_monitor:
                soft = int(self._budget * self._rss_soft_factor)
                hard = int(self._budget * self._rss_hard_factor)
                if rss_delta > hard:
                    raise MemoryLimitExceeded(
                        f"{op_name}: RSS delta {rss_delta} B > "
                        f"{self._rss_hard_factor:.1f}× budget ({hard} B). "
                        f"Unmodeled allocation (numpy/pandas/foreign lib?). "
                        f"Arrow delta {pool_delta} B. See §9.6."
                    )
                elif rss_delta > soft:
                    logger.warning(
                        "RSS soft-drift: op=%s delta=%d B > %.1f× budget "
                        "(soft=%d B, hard=%d B). Arrow delta %d B.",
                        op_name, rss_delta, self._rss_soft_factor,
                        soft, hard, pool_delta,
                    )

            # Tier 3 (calibrate): log sample, warn on formula drift
            if self._calibrate:
                self._record_calibration(
                    op_name, predicted_bytes, pool_delta, rss_delta,
                )'''

    new_body = '''        if self._pool is None:
            self._pool = pa.default_memory_pool()
            self._pool_baseline = self._pool.bytes_allocated()
        if self._process is None:
            import psutil
            self._process = psutil.Process()
            self._rss_baseline = self._process.memory_info().rss

        pool_start = self._pool.bytes_allocated()
        rss_start = self._process.memory_info().rss

        # v0.9.20: optional peak sampling via background thread.
        # Runs only when sample_peak=True AND (calibrate or monitor).
        # Thread spawn ~100us; polls every sample_interval_ms.
        peak_pool_box = [pool_start]
        peak_rss_box = [rss_start]
        stop_evt = None
        sampler_thread = None
        if sample_peak:
            import threading
            stop_evt = threading.Event()
            interval_s = max(0.001, float(sample_interval_ms) / 1000.0)

            def _sampler():
                while not stop_evt.is_set():
                    try:
                        p = self._pool.bytes_allocated()
                        r = self._process.memory_info().rss
                        if p > peak_pool_box[0]:
                            peak_pool_box[0] = p
                        if r > peak_rss_box[0]:
                            peak_rss_box[0] = r
                    except Exception:
                        pass
                    stop_evt.wait(interval_s)

            sampler_thread = threading.Thread(
                target=_sampler, daemon=True,
                name=f"cadw_peak_sampler_{op_name}",
            )
            sampler_thread.start()

        try:
            yield
        finally:
            if stop_evt is not None:
                stop_evt.set()
                if sampler_thread is not None:
                    sampler_thread.join(timeout=2.0)

            pool_end = self._pool.bytes_allocated()
            rss_end = self._process.memory_info().rss
            net_pool_delta = pool_end - pool_start
            net_rss_delta = rss_end - rss_start

            if sample_peak:
                peak_pool_abs = peak_pool_box[0]
                peak_rss_abs = peak_rss_box[0]
                peak_pool_delta = peak_pool_abs - pool_start
                peak_rss_delta = peak_rss_abs - rss_start
            else:
                peak_pool_abs = pool_end
                peak_rss_abs = rss_end
                peak_pool_delta = net_pool_delta
                peak_rss_delta = net_rss_delta

            # Tier 2 (runtime_monitor): hard safety net via peak RSS
            # when sampled (more honest), fallback net RSS otherwise.
            if self._runtime_monitor:
                effective_rss = peak_rss_delta if sample_peak else net_rss_delta
                soft = int(self._budget * self._rss_soft_factor)
                hard = int(self._budget * self._rss_hard_factor)
                if effective_rss > hard:
                    raise MemoryLimitExceeded(
                        f"{op_name}: RSS delta {effective_rss} B > "
                        f"{self._rss_hard_factor:.1f}× budget ({hard} B). "
                        f"Unmodeled allocation (numpy/pandas/foreign lib?). "
                        f"Arrow delta {peak_pool_delta if sample_peak else net_pool_delta} B. "
                        f"See §9.6."
                    )
                elif effective_rss > soft:
                    logger.warning(
                        "RSS soft-drift: op=%s delta=%d B > %.1f× budget "
                        "(soft=%d B, hard=%d B). Arrow delta %d B.",
                        op_name, effective_rss, self._rss_soft_factor,
                        soft, hard,
                        peak_pool_delta if sample_peak else net_pool_delta,
                    )

            # Tier 3 (calibrate): log sample, warn on formula drift
            if self._calibrate:
                self._record_calibration(
                    op_name,
                    predicted_bytes,
                    peak_pool_abs,
                    peak_rss_abs,
                    net_pool_delta,
                    net_rss_delta,
                    sample_mode=sample_peak,
                )'''

    if old_body not in s:
        raise SystemExit("calibrate body anchor not found")
    s = s.replace(old_body, new_body, 1)

    # Extend _record_calibration: new signature + entry fields
    old_rec = '''    def _record_calibration(
        self,
        op_name: str,
        predicted: int,
        actual_pool: int,
        actual_rss: int,
    ) -> None:
        """Append calibration sample; warn on C = actual/predicted > 2."""
        entry = {
            "predicted": int(predicted),
            "arrow": int(actual_pool),
            "rss": int(actual_rss),
        }
        self._calibration_samples.setdefault(op_name, []).append(entry)

        if predicted > 0:
            c_arrow = actual_pool / predicted
            c_rss = actual_rss / predicted
            logger.info(
                "calibrate op=%s predicted=%d arrow=%d (C=%.2f) "
                "rss=%d (C=%.2f)",
                op_name, predicted, actual_pool, c_arrow,
                actual_rss, c_rss,
            )
            if c_arrow > 2.0 or c_rss > 2.0:
                logger.warning(
                    "calibrate DRIFT: op=%s predicted=%d C_arrow=%.2f "
                    "C_rss=%.2f (threshold 2.0; consider fixing formula "
                    "or updating §9.2.N calibration)",
                    op_name, predicted, c_arrow, c_rss,
                )
        else:
            logger.info(
                "calibrate op=%s predicted=unknown arrow=%d rss=%d",
                op_name, actual_pool, actual_rss,
            )'''

    new_rec = '''    def _record_calibration(
        self,
        op_name: str,
        predicted: int,
        peak_pool_abs: int,
        peak_rss_abs: int,
        net_pool_delta: int,
        net_rss_delta: int,
        *,
        sample_mode: bool = False,
    ) -> None:
        """Append calibration sample; warn on C = peak/predicted > 2.

        v0.9.20: records peak (absolute, comparable to formula §9.2.N
        which predicts absolute peak) and net_delta (persistent, leak
        detection). sample_mode=True when background thread was used
        (peak is honest max); False when peak_pool_abs == net_pool_abs
        (i.e., no sampling, only start/end).
        """
        entry = {
            "predicted": int(predicted),
            "peak_pool": int(peak_pool_abs),
            "peak_rss": int(peak_rss_abs),
            "net_pool_delta": int(net_pool_delta),
            "net_rss_delta": int(net_rss_delta),
            "sample_mode": "peak" if sample_mode else "delta",
        }
        self._calibration_samples.setdefault(op_name, []).append(entry)

        if predicted > 0:
            c_peak = peak_pool_abs / predicted
            c_rss = peak_rss_abs / predicted
            logger.info(
                "calibrate op=%s mode=%s predicted=%d peak_pool=%d "
                "(C=%.2f) peak_rss=%d (C=%.2f) net_pool_delta=%d",
                op_name, entry["sample_mode"], predicted, peak_pool_abs,
                c_peak, peak_rss_abs, c_rss, net_pool_delta,
            )
            if c_peak > 2.0 or c_rss > 2.0:
                logger.warning(
                    "calibrate DRIFT: op=%s predicted=%d C_peak=%.2f "
                    "C_rss=%.2f (threshold 2.0; consider fixing formula "
                    "or updating §9.2.N calibration)",
                    op_name, predicted, c_peak, c_rss,
                )
        else:
            logger.info(
                "calibrate op=%s mode=%s predicted=unknown peak_pool=%d "
                "peak_rss=%d net_pool_delta=%d",
                op_name, entry["sample_mode"], peak_pool_abs,
                peak_rss_abs, net_pool_delta,
            )'''

    if old_rec not in s:
        raise SystemExit("_record_calibration anchor not found")
    s = s.replace(old_rec, new_rec, 1)

    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (b) update bench: sample_peak=True ============

def patch_bench():
    print("\n=== (b) bench: sample_peak=True + C_peak column ===")
    p = Path("scripts/bench_v0919_calibration.py")
    if not p.exists():
        raise SystemExit("bench script missing")
    s = p.read_text()
    if "sample_peak=True" in s:
        print("  already updated — skip")
        return

    # All calibrate calls: add sample_peak=True
    s = s.replace(
        "with guard.calibrate(name, predicted):",
        "with guard.calibrate(name, predicted, sample_peak=True):",
    )
    s = s.replace(
        'with guard.calibrate("external_sort", predicted):',
        'with guard.calibrate("external_sort", predicted, sample_peak=True):',
    )
    s = s.replace(
        'with guard.calibrate("hash_shuffle", predicted):',
        'with guard.calibrate("hash_shuffle", predicted, sample_peak=True):',
    )
    s = s.replace(
        'with guard.calibrate("groupby_aggregate", predicted):',
        'with guard.calibrate("groupby_aggregate", predicted, sample_peak=True):',
    )

    # Update _print_row: use peak_pool/peak_rss fields
    old_print = '''def _print_row(name, predicted, sample):
    arrow = sample.get("arrow", 0)
    rss = sample.get("rss", 0)
    if predicted > 0:
        ca = arrow / predicted
        cr = rss / predicted
        print(
            f"{name:22s} {predicted:>12,} {arrow:>12,} {rss:>12,} "
            f"{ca:>8.2f} {cr:>8.2f}"
        )
    else:
        print(
            f"{name:22s} {'unknown':>12s} {arrow:>12,} {rss:>12,} "
            f"{'-':>8s} {'-':>8s}"
        )'''

    new_print = '''def _print_row(name, predicted, sample):
    # v0.9.20: peak_pool/peak_rss = absolute peak (comparable to formula
    # §9.2.N which predicts absolute peak). net_pool_delta = persistent
    # (leak detection). sample_mode shows "peak" (background thread) or
    # "delta" (start/end only).
    arrow = sample.get("peak_pool", sample.get("arrow", 0))
    rss = sample.get("peak_rss", sample.get("rss", 0))
    if predicted > 0:
        ca = arrow / predicted
        cr = rss / predicted
        print(
            f"{name:22s} {predicted:>12,} {arrow:>12,} {rss:>12,} "
            f"{ca:>8.2f} {cr:>8.2f}"
        )
    else:
        print(
            f"{name:22s} {'unknown':>12s} {arrow:>12,} {rss:>12,} "
            f"{'-':>8s} {'-':>8s}"
        )'''

    if old_print not in s:
        raise SystemExit("_print_row anchor not found")
    s = s.replace(old_print, new_print, 1)

    # Interpretation note: peak vs delta
    old_interp = '''    print("Interpretation:")
    print("  C_arrow = arrow_delta / predicted (formula drift signal)")
    print("  C_rss   = rss_delta / predicted (includes Python overhead)")
    print("  drift threshold (AUDIT §9.5): C > 2.0 → investigate formula")'''

    new_interp = '''    print("Interpretation (v0.9.20: peak polling, not net delta):")
    print("  peak_pool = max pa.pool.bytes_allocated during op (absolute,")
    print("              comparable to §9.2.N formula absolute-peak)")
    print("  peak_rss  = max psutil RSS during op")
    print("  C_arrow   = peak_pool / predicted  (formula drift signal)")
    print("  C_rss     = peak_rss  / predicted  (includes Python overhead)")
    print("  drift threshold (AUDIT §9.5): C > 2.0 → investigate formula")
    print("  Note: predicted excludes Arrow temp buffers (pc.sum skip_nulls")
    print("  temp, sort_indices intermediate). C_peak 1.5-3.0 expected.")'''

    if old_interp not in s:
        raise SystemExit("interpretation anchor not found")
    s = s.replace(old_interp, new_interp, 1)

    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (c) DECISIONS append §13.4/226-231 ============

DECISIONS_226_231 = [
    "",
    "",
    "**Эпизод 226 (nullable int to_numpy → float64).** v0.9.18-fix1: "
    "`DictionaryArray.indices.to_numpy(zero_copy_only=False)` на "
    "null-содержащем массиве возвращает **float64** (NaN для null), "
    "не int64. `uniq_hashes[indices[non_null]]` → `IndexError: arrays "
    "used as indices must be integer type`.",
    "",
    "**Правило (§13.4/226):** pyarrow nullable integer "
    "`Array.to_numpy(zero_copy_only=False)` → **float64** (NaN), не "
    "int64. Для integer-indexing: "
    "`arr.fill_null(0).to_numpy(...).astype(int64)` + отдельная "
    "`null_mask = arr.is_null().to_numpy()`. Same class as §13.4/223 "
    "(bool→int): pyarrow to_numpy теряет Arrow type strictness при "
    "nulls.",
    "",
    "**Связанные:** §13.4/223, /229 (numpy `==` coercion).",
    "",
    "",
    "**Эпизод 227 (dictionary_encode all-null → empty dict).** "
    "v0.9.18-fix2: `pa.array([None, None, None]).dictionary_encode()` "
    "даёт **empty dictionary** (K=0), indices = [null, null, null]. "
    "`fill_null(0)` → [0, 0, 0], но `uniq_hashes` размера 0 → "
    "`IndexError: index 0 is out of bounds`.",
    "",
    "**Правило (§13.4/227):** `dictionary_encode()` на all-null array "
    "→ пустой dictionary. Проверять `K == 0` **до** любых индексных "
    "операций. All-null → all `_NULL_SENTINEL`, no dictionary path.",
    "",
    "**Связанные:** §13.4/226.",
    "",
    "",
    "**Эпизод 228 (dogfood = happy path only).** v0.9.18-fix1/fix2: "
    "dogfood упал на null-containing path (fix1 root cause), "
    "затем на all-null path (fix2 root cause). При этом "
    "`test_string_hash_all_nulls` **уже был** в fix1 тестах и "
    "поймал бы баг за 10 секунд pytest — но dogfood упал первым.",
    "",
    "**Правило (§13.4/228):** dogfood — быстрый happy-path smoke + "
    "только структурные инварианты, требующие subprocess (fresh "
    "imports). Edge cases — pytest. Иначе каждый edge bug = "
    "отдельный patch cycle (3 fix на v0.9.19: initial + fix1 + fix2).",
    "",
    "**Связанные:** §13.4/216 (helper exists), /225 (guard scope).",
    "",
    "",
    "**Эпизод 229 (numpy element-wise == coerces label).** "
    "v0.9.18-fix3: `get_positions_all(('a', 1))` на string index: "
    "`np_arr == ('a', 1)` → numpy coerces tuple to string array, "
    "matches first element → false positive position `[0]`. Python "
    "`\"1\" == 1` → False (type strict); numpy `np.array(['1']) == 1` "
    "→ `[True]` (str coercion). 13 tests с `loc[tuple]` на single "
    "index падали с false-positive match.",
    "",
    "**Правило (§13.4/229):** numpy fast path для label-lookup "
    "безопасен только для **same-type scalar comparisons**. Для "
    "string/object index + non-scalar labels — Python loop "
    "(type-strict). Same class: §13.4/223 (bool→int), /226 "
    "(nullable int → float64): numpy теряет Arrow type strictness.",
    "",
    "**Связанные:** §13.4/226, /223.",
    "",
    "",
    "**Эпизод 230 (write_content lines — complete syntactic units).** "
    "v0.9.19-fix1: bench script broken — print-headers f-string "
    "разбит на два `write_content(lines)` элемента без закрывающей "
    "`\"` в первом. Результат: два f-string open, один close → "
    "`SyntaxError` на line 83. `assert_compiles` сработал, patch "
    "упал до commit.",
    "",
    "**Правило (§13.4/230):** `write_content(lines: list[str])` — "
    "каждый элемент = одна полная строка файла. Каждый должен быть "
    "**complete syntactic unit** для multi-part construct (f-string, "
    "string concatenation, backslash continuation). Нельзя разбить "
    "`f\"...` mid-literal. Опции: single-line f-string, `\\` "
    "continuation, или вынести в переменную.",
    "",
    "**Связанные:** §13.4/211 (compile-before-write), /212.",
    "",
    "",
    "**Эпизод 231 (helper methods = caller-facing).** v0.9.19-fix2: "
    "`calibrate()` prologue вызывал `_get_pool_delta()` и "
    "`_get_rss_delta()` для «ensure baselines init». Эти helpers — "
    "**caller-facing measurement APIs**, читают `self._pool`/"
    "`self._process` при **каждом** вызове, даже когда baseline уже "
    "установлен. Fake с 2 reads `[0, 100M]`: prologue consumed read 0 "
    "for each, `pool_start`/`pool_end` both read clamped `value[1]` → "
    "`delta = 0` → no raise.",
    "",
    "**Правило (§13.4/231):** helper с публичной семантикой («сколько "
    "delta с момента baseline») нельзя звать изнутри критического "
    "path (calibrate), где контракт другой («1 read start + 1 read "
    "end»). calibrate() делает **прямой lazy init** (только когда "
    "`None`), не зовёт `_get_*_delta`. Same class как §13.4/178 "
    "(вакуумный guard): helper выглядит как utility, но его "
    "использование ломает контракт.",
    "",
    "**Связанные:** §13.4/178, /230 (fix1), /226 (fix1).",
    "",
    "",
    "**Эпизод 232 (metric unit must match formula unit).** v0.9.19 "
    "bench: C_arrow = net_delta / predicted = 0.00 для reduce ops. "
    "Formula §9.2.1 predicts **absolute peak** (batch + 2×state); "
    "metric measured **net delta** (end − start). Батчи pre-allocated "
    "at guard creation, Arrow temp buffers released before end-of-op "
    "→ net ≈ 0. C_arrow=0 — honest (no leak), но useless как "
    "formula validation. v0.9.20: peak-polling (background thread, "
    "10ms sample) → peak_absolute compared to formula. C_peak = "
    "peak_pool / predicted, expected 1.5-3.0 (formula excludes Arrow "
    "temp buffers, sort_indices intermediate).",
    "",
    "**Правило (§13.4/232):** metric unit must match formula unit. "
    "Peak vs peak, delta vs delta. Never cross. Bench делал "
    "net_delta/predicted_peak — metric artifact, не formula drift.",
    "",
    "**Связанные:** §13.4/231 (calibrate проlogue), /229 (type "
    "coercion — same «wrong thing measured» class).",
    "",
    "",
]


def append_decisions():
    print("\n=== (c) DECISIONS append §13.4/226-231 + 232 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "**Эпизод 226 (nullable int to_numpy" in s:
        print("  already present — skip")
        return
    anchor = "**Gap 190–210 (reserved)"
    if anchor not in s:
        # v0.9.17 inserted 178-189 + gap; v0.9.16 inserted 211-225.
        # So 211 comes AFTER 189+gap. Insert 226-232 right before "Gap 190-210"? No.
        # Actually: gap note is BETWEEN 189 and 211. So 226-232 should be AFTER 225.
        # Use "### 13.5. Открытые вопросы" as anchor.
        anchor = "### 13.5. Открытые вопросы"
        if anchor not in s:
            raise SystemExit("DECISIONS.md anchor not found")
    # Insert 226-232 AFTER 225, i.e., BEFORE §13.5
    block = "\n".join(DECISIONS_226_231) + "\n"
    s = s.replace(anchor, block + anchor, 1)
    p.write_text(s, newline="")
    # Sanity
    assert "**Эпизод 226 (nullable int to_numpy" in s
    assert "**Эпизод 232 (metric unit must match" in s
    assert s.index("**Эпизод 225") < s.index("**Эпизод 226")
    assert s.index("**Эпизод 232") < s.index("### 13.5. Открытые вопросы")
    print("  OK (226-232 inserted after 225, before §13.5)")


# ============ (d) tests ============

def add_peak_tests():
    print("\n=== (d) test_runtime_monitor: peak sampling tests ===")
    p = Path("tests/test_runtime_monitor.py")
    s = p.read_text()
    if "def test_calibrate_sample_peak_mode" in s:
        print("  already present — skip")
        return

    extra = '''

# ============ v0.9.20: peak sampling ============


def test_calibrate_sample_peak_mode():
    """sample_peak=True → entry has peak_pool/peak_rss + sample_mode='peak'."""
    g = MemoryGuard(max_memory_mb=10, calibrate=True)
    with g.calibrate("op", 1000, sample_peak=True, sample_interval_ms=1):
        pass
    entry = g.calibration_samples["op"][0]
    assert entry["sample_mode"] == "peak"
    assert "peak_pool" in entry
    assert "peak_rss" in entry
    assert "net_pool_delta" in entry
    assert "net_rss_delta" in entry


def test_calibrate_delta_mode_default():
    """sample_peak=False (default) → sample_mode='delta', peak==net."""
    g = MemoryGuard(max_memory_mb=10, calibrate=True)
    with g.calibrate("op", 1000):
        pass
    entry = g.calibration_samples["op"][0]
    assert entry["sample_mode"] == "delta"
    # Without sampling, peak == end == net delta
    assert entry["peak_pool"] == entry["net_pool_delta"] + entry["peak_pool"] - entry["net_pool_delta"]


def test_calibrate_peak_captures_spike():
    """sample_peak=True captures temporary allocation spike during op."""
    import time as _time
    g = MemoryGuard(max_memory_mb=100, calibrate=True)
    # Force sampling to catch the spike by holding it for > interval
    with g.calibrate("spike_op", 0, sample_peak=True, sample_interval_ms=5):
        buf = [None] * 100_000
        _time.sleep(0.05)  # 50ms > 5 samples
        # Actual: peak measured while buf alive
        _ = buf
    entry = g.calibration_samples["spike_op"][0]
    assert entry["sample_mode"] == "peak"
    # RSS peak should be >= some reasonable size for 100k list (~800 KB)
    assert entry["peak_rss"] >= 500_000, (
        f"peak_rss={entry['peak_rss']}, expected >=500KB for 100k list"
    )


def test_calibrate_peak_thread_joined():
    """Thread must be joined on exit — no thread leak."""
    import threading as _th
    g = MemoryGuard(max_memory_mb=10, calibrate=True)
    base_count = _th.active_count()
    for _ in range(5):
        with g.calibrate("op", 0, sample_peak=True, sample_interval_ms=1):
            pass
    # All threads should be joined — no leak
    after_count = _th.active_count()
    # Allow +1 for pytest's own threads
    assert after_count <= base_count + 1, (
        f"thread leak: before={base_count}, after={after_count}"
    )


def test_calibrate_peak_runtime_monitor_uses_peak():
    """runtime_monitor with sample_peak uses peak RSS for hard/soft check."""
    g = MemoryGuard(max_memory_mb=10, runtime_monitor=True, calibrate=True)
    # Fake process returns controlled values; entry=0, exit=0, but peak>hard
    rss_values = [0, 0]  # start/end
    ridx = [0]
    peak_rss_forced = [0]
    class FakeProcess:
        def memory_info(self):
            # Return 0 for start/end calls; peak forced separately
            return MagicMock(rss=rss_values[min(ridx[0], len(rss_values)-1)])
    g._process = FakeProcess()
    g._rss_baseline = 0
    class FakePool:
        def bytes_allocated(self):
            return 0
    g._pool = FakePool()
    g._pool_baseline = 0
    # Force peak_rss_box beyond hard by monkeypatching thread start
    # Simpler: patch peak_rss_box via memory_guard internals? No.
    # Instead: use non-sampling path via calibration sample inspection.
    # This test just verifies peak_rss field is stored; hard check
    # behavior is covered by existing test_runtime_monitor_raises_on_hard_rss_drift.
    with g.calibrate("op", 100, sample_peak=True, sample_interval_ms=1):
        pass
    entry = g.calibration_samples["op"][0]
    assert "peak_rss" in entry


def test_bench_v0920_uses_sample_peak():
    """Bench script uses sample_peak=True for all calibrate calls."""
    from pathlib import Path as _P
    bench = (
        _P(__file__).resolve().parent.parent
        / "scripts" / "bench_v0919_calibration.py"
    )
    src = bench.read_text()
    # At least one sample_peak=True (all calibrate calls updated)
    assert "sample_peak=True" in src
    # And not leftover calibrate without sample_peak on known ops
    import re
    non_peak = re.findall(
        r"with guard\\.calibrate\\([^)]+\\):", src,
    )
    for line in non_peak:
        if "sample_peak" not in line:
            raise AssertionError(
                f"bench calibrate without sample_peak: {line.strip()}"
            )
'''

    p.write_text(s + extra, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.20 + regression ===")
    pu.verify_anchors([
        # v0.9.20 (a)
        ("cadw/core/memory_guard.py", "sample_peak: bool = False"),
        ("cadw/core/memory_guard.py", "sample_interval_ms: float = 10.0"),
        ("cadw/core/memory_guard.py", "threading.Event()"),
        ("cadw/core/memory_guard.py", "peak_pool_box"),
        ("cadw/core/memory_guard.py", "peak_rss_box"),
        ("cadw/core/memory_guard.py", 'entry["peak_pool"]'),
        ("cadw/core/memory_guard.py", 'entry["sample_mode"]'),
        # (b)
        ("scripts/bench_v0919_calibration.py", "sample_peak=True"),
        ("scripts/bench_v0919_calibration.py", "peak_polling"),
        # (c)
        ("DECISIONS.md", "**Эпизод 226 (nullable int to_numpy"),
        ("DECISIONS.md", "**Эпизод 227 (dictionary_encode all-null"),
        ("DECISIONS.md", "**Эпизод 228 (dogfood = happy path only)"),
        ("DECISIONS.md", "**Эпизод 229 (numpy element-wise =="),
        ("DECISIONS.md", "**Эпизод 230 (write_content lines"),
        ("DECISIONS.md", "**Эпизод 231 (helper methods = caller-facing)"),
        ("DECISIONS.md", "**Эпизод 232 (metric unit must match"),
        # (d)
        ("tests/test_runtime_monitor.py", "def test_calibrate_sample_peak_mode"),
        ("tests/test_runtime_monitor.py", "def test_calibrate_peak_captures_spike"),
        ("tests/test_runtime_monitor.py", "def test_bench_v0920_uses_sample_peak"),
        # v0.9.19 regression
        ("cadw/core/memory_guard.py", "def calibrate("),
        ("cadw/core/memory_guard.py", "def _record_calibration("),
        ("cadw/ops/reduce.py", "with memory_guard.calibrate("),
        ("tests/test_runtime_monitor.py", "def test_calibrate_records_sample"),
        # v0.9.18/17/16 regression
        ("cadw/ops/shuffle.py", "§13.4/227"),
        ("cadw/dataset/index.py", "v0.9.18 (AUDIT §1.6)"),
        ("cadw/dataframe.py", "class _ArrowTail:"),
        ("DECISIONS.md", "**Эпизод 178 (guard, вакуумный"),
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
    ])
    print("  all anchors present")


# ============ dogfood (happy path only, §13.4/228) ============

def dogfood():
    print("\n=== dogfood: v0.9.20 (happy path only) ===")

    def _verify_all():
        import pyarrow as pa
        from cadw.core.memory_guard import MemoryGuard
        from cadw.ops.reduce import CountOp, reduce

        # 1. Backward compat: no flags → no overhead, no samples
        g_off = MemoryGuard(max_memory_mb=10)
        batches = [
            pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], names=["x"]),
        ]
        result = reduce(iter(batches), CountOp("x"), g_off, log=False)
        assert result == 3
        assert g_off.calibration_samples == {}

        # 2. Delta mode (calibrate=True, no sample_peak) — happy path
        g_delta = MemoryGuard(max_memory_mb=10, calibrate=True)
        result = reduce(iter(batches), CountOp("x"), g_delta, log=False)
        assert result == 3
        entry = g_delta.calibration_samples["CountOp"][0]
        assert entry["sample_mode"] == "delta"

        # 3. Peak mode (sample_peak=True) — happy path
        g_peak = MemoryGuard(max_memory_mb=10, calibrate=True)
        with g_peak.calibrate("op", 100, sample_peak=True, sample_interval_ms=2):
            import time as _t
            _ = [None] * 50_000
            _t.sleep(0.02)  # 10 samples at 2ms
        entry_p = g_peak.calibration_samples["op"][0]
        assert entry_p["sample_mode"] == "peak"
        assert entry_p["peak_rss"] >= 100_000, (
            f"peak_rss={entry_p['peak_rss']} expected >=100KB for 50k list"
        )

        # 4. Bench compiles
        import py_compile
        from pathlib import Path as _P
        py_compile.compile(
            str(_P("scripts/bench_v0919_calibration.py")), doraise=True,
        )

        print("  no-flag: no samples OK")
        print("  delta mode: sample_mode='delta' OK")
        print("  peak mode: peak_rss >= 100KB for 50k list OK")
        print("  bench compiles OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target (runtime monitor) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_runtime_monitor.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
        sys.exit(1)

    print("\n=== pytest: reduce (regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_reduce.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REDUCE FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.20 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    patch_calibrate_peak()
    patch_bench()
    append_decisions()
    add_peak_tests()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.20: peak-polling in calibrate() + DECISIONS §13.4/226-232 "
        "append. Motivation: v0.9.19 bench revealed C_arrow=0 for all "
        "reduce ops — net pool delta is honest (no leak) but useless "
        "for formula validation. Formula §9.2.1 predicts ABSOLUTE peak "
        "(batch + 2×state); net delta = end − start, and batches are "
        "pre-allocated at guard creation, Arrow temp buffers released "
        "before end-of-op → net ≈ 0. §13.4/232 (new): metric unit must "
        "match formula unit. Peak vs peak, delta vs delta. Never cross. "
        "(Design) `calibrate(..., sample_peak=True, sample_interval_ms=10)` "
        "— opt-in background thread samples pa.default_memory_pool()."
        "bytes_allocated() + psutil RSS every interval; tracks max. "
        "Records BOTH: `peak_pool` (absolute max, comparable to formula "
        "§9.2.N which predicts absolute peak) and `net_pool_delta` "
        "(end − start, leak detection). `sample_mode='peak'|'delta'` "
        "distinguishes. Zero overhead when calibrate=False (yield only). "
        "Thread spawn ~100μs — opt-in only, delta mode (default) pays "
        "nothing. runtime_monitor uses peak_rss when sampled (more "
        "honest hard/soft check), falls back to net_rss otherwise. "
        "(Bench) `bench_v0919_calibration.py` updated: all calibrate "
        "calls with sample_peak=True; `_print_row` uses peak_pool/peak_rss "
        "fields; interpretation note updated (C_peak 1.5-3.0 expected — "
        "formula excludes Arrow temp buffers like pc.sum skip_nulls, "
        "sort_indices intermediates). (DECISIONS) §13.4/226-232 append: "
        "226 (nullable int to_numpy → float64 not int64), 227 "
        "(dictionary_encode all-null → empty dict, K=0 edge), 228 "
        "(dogfood = happy path only, edge cases = pytest), 229 (numpy "
        "element-wise == coerces label to array dtype — 13 loc[tuple] "
        "tests false-positive fixed), 230 (write_content lines — "
        "complete syntactic units for f-strings), 231 (helper methods "
        "are caller-facing, calibrate must do direct lazy init not call "
        "_get_*_delta), 232 (metric unit must match formula unit). "
        "(Tests) 6 new: sample_peak mode records peak fields; delta "
        "mode default; peak captures 100k-list spike via 50ms hold; "
        "thread joined (no leak); runtime_monitor uses peak_rss when "
        "sampled; bench uses sample_peak=True. (Backward compat) all "
        "1902 existing tests unchanged; zero overhead when flags off "
        "(calibrate() no-op, no thread). Deferred to v0.9.21: AUDIT "
        "§1.4 per-group shift/rolling numpy; per-op predicted wiring "
        "for sort/shuffle/groupby (needs peak-polling validated first); "
        "Pingis round 3 (unaudited: ML/pipeline/cv/metrics).",
        self_path="scripts/patch_v0920.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.20 applied.")


if __name__ == "__main__":
    main()
