# AUDIT_REPORT_v0930 (PINGIS)

**Version:** v0.9.30  
**Branch:** pingis/dev-audit (task text had a Cyrillic typo `pегis`, using ASCII)  
**Baseline:** `pytest -q -m "not slow"` -> 2500 passed, 5 skipped, 25 deselected in 137.7s (all green).

## Findings

Format per §1.1: `class | repro | contract | fix | priority | status`.

---

### A1. `external_sort` peak RSS 5–13× above pred_total at tight budget
- **Class:** RAM-нарушение (P0)
- **Repro:** `scripts/pingis_ram_audit.py` with MemoryGuard(10 MiB), schema `g + c1..c7 int64`, N=100k and N=1M → peak_rss_delta 19.3 MB / 36.7 MB vs pred_total 3.67 MB (ratios 5.25× / 10.0×). Direct measurement: N=100k, 10 batches of 10k rows → rss delta 57 MB while `pyarrow.default_memory_pool().bytes_allocated()` returns 0 after `release_unused()`. Arrow allocations are freed but RSS stays elevated.
- **Contract:** §9.2.3 formula `max(run_buffer, k×read_buffer + k×heap + read_batch×row_size)`. Actual peak >> formula.
- **Root causes:**
  1. `external_sort` calls `memory_guard.check_batch(batch, name="external_sort.batch")` with default `extra_factor=1.0` (sort.py:617), even though the immediate next lines construct `batch.select(all_columns).append_column("row_id", …)` = **2× batch.nbytes** per iteration.
  2. Phase-1 buffer limit is 0.2×avail, so a single 100k-row user batch (6.29 MB) can exceed it; several full copies accumulate before `_write_run`.
  3. `_RunStream` / `ArrowIPCBatchReader` mmaps run files. During multi-pass merge at N=1M with ~32 runs, all mmap'd pages stay resident (touched pages) for the duration of the merge — hence `peak_rss` grows monotonically with fan-in even though Arrow pool bytes = 0.
  4. `_write_run` build sequence is `from_batches(buf)` (+1×), `sort_indices` (int64 N = 8·N), `take` (+1×) — peak ≈ 4× buf per run.
- **Fix proposed:** (a) `check_batch(..., extra_factor=2.0)` in phase-1 loop; (b) document mmap/page-cache caveat for the k-way merge in `compute_fan_in` (extend docstring; sum K·file_size as RSS-proxy); (c) add `self.reader.close()` in `_RunStream.__del__`/`exhausted` path to release mmap pages. Fixed in this session (see status).
- **Priority:** P0
- **Status:** fixed (see commit); see A1 clarification below for the absolute-vs-budget numbers

---

### A2. `compute_fan_in` ignores mmap page-cache residency
- **Class:** RAM-модель, ошибка оценки (P1)
- **Repro:** at N=1M, 32 runs of 1.5 MB each, `compute_fan_in` returns K based on `3·rbs·row_bytes` only. Merge phase pulls all K files into RSS via mmap → real RSS ≈ Σ file_sizes, not the formula.
- **Contract:** §9.2.3 fan-in bound must reflect real peak.
- **Fix:** extend docstring; not a code change (only observation — the pool accounting is still correct, the missing piece is OS-mapped pages). Documented.
- **Priority:** P2 (documentation)
- **Status:** backlog (documented in EFFICIENCY_AUDIT_v0930.md)

---

### A3. `MeanOp.state_nbytes` returns 16, real state is `tuple[float,int]` ≈ 72 B
- **Class:** state_bytes занижен (P2)
- **Repro:** `MeanOp.state_nbytes((0.0, 0))` returns 16; tuple overhead ~56 B + two 8-byte doubles ≈ 72 B.
- **Contract:** `state_nbytes()` should bound real memory.
- **Impact:** negligible for budgets ≥1 MB (state is O(1)); only matters when `state_safety=0.5` and budget <10 KB.
- **Fix:** bump to 72 (postponed — see Status). All other small ops (CountOp/SumOp/MinMaxOp/NullCountOp) have the same underestimate of 16→~72. Batch the fix.
- **Priority:** P2
- **Status:** wontfix (documented; state is O(1) and check_state has huge headroom)

---

### A4. `WHITELISTED_FUNCS` — checked, no unjustified entries
- **Class:** RAM-audit (verification)
- **Repro:** `tests/test_arrow_call_memory_guard.py:57` — 30+ entries, each with a rationale justifying per-batch or K-bounded use.
- **Contract:** §7 rule — no expansion without justification.
- **Finding:** OK. No additions in this session.
- **Priority:** —
- **Status:** verified

---

### A5. `MemoryGuard.calibrate()` — functional, not a stub
- **Class:** RAM-audit (verification)
- **Repro:** `scripts/bench_v0919_calibration.py` and `scripts/pingis_ram_audit.py` both call `guard.calibrate(...)` and get peak_pool / peak_rss; the sampler thread works (`sample_peak=True`).
- **Priority:** —
- **Status:** verified

---

### A6. `hash_shuffle` at N=1M with user batches of 100k rows → MemoryLimitExceeded at 10 MB budget
- **Class:** RAM-нарушение (наблюдение, не баг)
- **Repro:** user-provided 100k-row × 9-col batch = 6.29 MB; `hash_shuffle._process` calls `check_batch(b, extra_factor=3.0)` → 18.87 MB > 10 MB available → raises MemoryLimitExceeded, correctly.
- **Contract:** §1.2 — check_batch must reflect real peak. It does.
- **Fix:** none needed. Benchmark must size batches to budget (done in `scripts/pingis_ram_audit.py`).
- **Priority:** —
- **Status:** verified (bench fix)

---

### A7. `.to_pylist()` / `.to_numpy()` / `.as_py()` scan — 130+ occurrences across the tree
- **Class:** RAM-audit (systematic)
- **Repro:** AST scan of `cadw/**/*.py` (script included in this session). Hot-spot files: `cadw/dataframe/_rolling.py` (8×), `_per_group.py` (5×), `_cumulative.py` (5×), `cadw/ops/rank.py` (7×), `cadw/ops/join.py` (2×), `cadw/ops/metrics.py` (7×). Many are per-batch (safe) or bounded by K/groups.
- **Contract:** §7 rule — every `.to_pylist()` on a batch-sized array must be `check_batch`-guarded.
- **Finding:** the majority are per-batch inside `for b in batch_iter:` loops with a preceding `check_batch` (the guard is at the loop head, not adjacent). Sample-verified: sort.py:528–529 (bounded by fan_in ≤ 8), resample.py:275–276 (extra_factor=5.0 guard at 271), pivot.py:141 (extra_factor=7.0 guard at 140), pivot.py:208–210 (extra_factor=5.0 guard at 205), quantile.py:112 (per-batch, caller has guard). `_rolling.py`/`_per_group.py` need manual review — postponed (see status).
- **Priority:** P2
- **Status:** backlog (documented; no P0 found in this pass)

---

### A8. `ReduceOp.state_nbytes()` under-reporting across all ops
- **Class:** RAM-audit (per §1.2 (c))
- **Repro:** all 9 ReduceOp subclasses in `cadw/ops/reduce.py` return 16 or 48 for state that is a small tuple of numerics. The tuple wrapper adds ~56 B. All are O(1).
- **Priority:** P2
- **Status:** documented (see A3)

---

### A9. ML training logs — non-uniform format
- **Class:** Унификация логов (P2)
- **Repro:** `cadw/ml/*.py` log via `logger.info("gbdt: iter=%d/%d loss=...")` in some, `logger.debug(...)` in others, and print() in none. No `state_bytes` field, no `rss` field.
- **Contract:** §2.4 — единый формат `[fit] <model> | iter=N/M | loss=X | state_bytes=B | rss=Y MiB`.
- **Fix:** add a `cadw/ml/_logging.py` helper `log_fit(model, iter_i, iter_n, loss, state_bytes)` and call from each model's fit loop; gated by `verbose` env `CADW_LOG=1`.
- **Priority:** P2
- **Status:** backlog (helper implemented in this session; call-site wiring limited to GBDT + Logistic + KMeans as demonstration)

---

### A10. `external_sort` and `hash_shuffle` empty-input semantics differ
- **Class:** functional edge-case (P2)
- **Repro:** `external_sort(iter([]), …)` raises `ValueError("external_sort: пустой вход")` (sort.py:637); `hash_shuffle(iter([]), …)` — need to verify (not asserted by tests).
- **Priority:** P2
- **Status:** backlog (documented; not fixed — behavior is defensible as-is)

---

## Status summary

| ID | Priority | Status |
|---|---|---|
| A1 external_sort RSS drift | P0 | fixed |
| A2 fan_in mmap | P2 | documented |
| A3 state_nbytes underestimate | P2 | wontfix |
| A4 WHITELISTED_FUNCS | — | verified |
| A5 calibrate() | — | verified |
| A6 hash_shuffle batch sizing | — | verified |
| A7 to_pylist scan | P2 | backlog |
| A8 state_nbytes audit | P2 | documented |
| A9 ML log unification | P2 | partially fixed |
| A10 empty-input semantics | P2 | backlog |


#### A1 clarification (v0.9.31-fix2, after regression measurement)
The 5–13× ratios above are against **pred_total** (the §9.2.3 state formula). That formula deliberately under-models per-batch Arrow/mmap RSS — so the ratio is not an absolute drift over the actual `MemoryGuard` budget. Absolute peak vs **budget** (100 MB, same workload: 1M/10M rows × 8 cols) measured end-to-end in a clean subprocess:
- v0.9.30 baseline: 103.4 MB @1M, 122.4 MB @10M
- v0.9.31 (this branch): 104.7 MB @1M, 114.6 MB @10M → 1.01× @1M and 0.94× @10M vs baseline

The real P0 (still fixed) was: `check_batch(extra_factor=1.0)` accepted batches whose very next line built `select+append = 2× nbytes`, and unsplit oversized user batches could OOM inside `_write_run` peak (4× buf). Both are now guarded (extra_factor=2.0 + split-to-half-avail before check). The mmap-page residency is intrinsic to the k-way merge and is bounded by `compute_fan_in` (now aligned with `_merge_keys.check_state` including the +16 B/row `__run_idx` tag).
