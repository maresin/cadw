# CADW — Bug backlog (v0.9.33)

**Sources:** `.pingis/audits/AUDIT_REPORT_v0930.md`,
`.pingis/audits/AUDIT_v0931.md`,
`docs/roadmap_research/part1_report.md` §1-§2.

**Patch:** v0.9.34 — one bugfix pass, no deferral. If any fix
requires split → D.x.y-1, D.x.y-2 sub-patches, still no deferral.

| ID | Priority | Bug | File | Patch |
|---|---|---|---|---|
| **B1** | **P0** | `_RunStream.refill` quadratic `pa.concat_tables([buffer, chunk])` — до 320 GB copies @ rbs=1M | `cadw/ops/sort.py` | v0.9.34 |
| **B2** | **P0** | `encode_categorical` accumulates Python-set of uniques (O(K), ~600 MB @ 10M uniq) | `cadw/dataframe/_ml_bindings.py` | v0.9.34 |
| **B3** | P1 | ROC-AUC pass2 row-wise `zip(probas, ys)` (CPU-linear) | `cadw/ops/metrics.py:639` | v0.9.34 |
| **B7** | P1 | `_string_hash_column` Python-loop over dict (O(K) per batch) | `cadw/ops/shuffle.py:177` | v0.9.34 |
| **A9** | P1 | ML log wiring — 3/6 models (linear / pca / naive_bayes not wired) | `cadw/ml/{linear,pca,naive_bayes}.py` | v0.9.34 |
| **B4** | P2 | `memory_guard.py:700` fallback `len(set(...))` without extra_factor | `cadw/core/memory_guard.py:700` | v0.9.34 |
| **A10** | P2 | `external_sort` vs `hash_shuffle` empty-input contract not documented | docstrings | v0.9.34 |
| **B6** | P2 | README version drift (v0.8.8 / 0.9.24 / `__version__`) | `README.md` | v0.9.34 |

---

## Per-bug detail

### B1 (P0) — Quadratic `_RunStream.refill`

- **Repro:** 2000 chunks of 1 row: repeated concat = 0.078 s;
  `concat_tables(parts_list)` = 0.0003 s (~250× + superlinear).
- **Contract:** reference `docs/roadmap_research/part1_report.md §2.1`.
- **Impact:** at rbs=200k: 12.9 GB copies; @1M: up to 320 GB (est).
  Peak RSS correct, but I/O pressure → page-cache swap.
- **Fix:** linear buffer via `parts_list` + single `concat_tables`.
- **Test:** `tests/test_refill_linear.py` — bench 2000 chunks; assert
  time ≤ 2× single concat; assert copies ≤ rbs.
- **Bench artifact:** `.pingis/benches/v0934_b1_refill.json`.

### B2 (P0) — `encode_categorical` Python-set

- **Repro:** 10M unique strings — Python-set ~600 MB, not seen by
  MemoryGuard (per-batch ~130 KB).
- **Contract:** `check_state()` before Pass1.
- **Fix:** `estimate_k_sample_arrow` → `check_state(K_est ×
  (value_bytes + 50))`; or Arrow dictionary IPC + join (Pass1 O(1)
  RAM).
- **Test:** `tests/test_encode_categorical_high_card.py` — 100k
  uniques in 10 MB budget → raises `MemoryLimitExceeded`.
- **Bench artifact:** `.pingis/benches/v0934_b2_encode.json`.

### B3 (P1) — ROC-AUC pass2 Python loop

- **Repro:** 500k rows: 0.073 s Python loop vs 0.168 s Arrow
  group_by (but +72 MB, K=N).
- **Fix:** vectorize via `pc` primitives; or keep if not bottleneck —
  **but** must document *why* kept (memory tradeoff).
- **Test:** `tests/test_roc_auc_streaming.py` — 1M rows bench;
  memory ≤ current; time ≤ current.
- **Bench artifact:** `.pingis/benches/v0934_b3_rocauc.json`.

### B7 (P1) — `_string_hash_column` Python loop

- **Repro:** O(K) Python objects per batch on dict keys.
- **Fix:** `dictionary_encode` + hash dict once (Arrow native).
- **Test:** `tests/test_string_hash_native.py` — output equal to
  current; faster on high K.
- **Bench artifact:** `.pingis/benches/v0934_b7_hash.json`.

### A9 (P1) — ML log wiring (3 remaining)

- **Contract:** uniform `[fit] <model> | iter=N/M | loss=X |
  state_bytes=B | rss=Y MiB` via `cadw.ml._logging.log_fit`.
- **Fix:** call `log_fit` in `fit_linear` (one-shot fit — no iter;
  log final only), `fit_pca`, `fit_gaussian_nb` / `fit_multinomial_nb`
  / `fit_bernoulli_nb`.
- **Test:** `tests/test_ml_logs_uniform.py` — grep `log_fit` in all 6
  `cadw/ml/*.py`; run each fit with `CADW_LOG=1`; parse stdout format.

### B4 (P2) — `memory_guard` fallback factor

- **Fix:** add `extra_factor=4` in `len(set(...))` fallback (part1
  §2.3 factor for utf8).
- **Test:** inline test verifies factor.

### A10 (P2) — empty-input contract

- **Fix:** docstrings for `external_sort`, `hash_shuffle`:
  both raise `ValueError` on empty input.
- **Test:** explicit `pytest.raises(ValueError)`.

### B6 (P2) — README version drift

- **Fix:** `README.md` — pull version from `cadw.__version__`
  (single source).
- **Test:** grep that README contains `__version__` string (no
  hardcoded number).

---

## Closure (v0.9.34)

| ID | Final status | Evidence |
|---|---|---|
| B1 | already-fixed-at-baseline | `6fa5e49` v0.9.25c; repro PASS + bench |
| B2 | already-fixed-at-baseline | `6fa5e49` v0.9.25c; repro PASS + bench |
| B3 | rejected-with-measurement | Python loop 0.026s@500k vs Arrow 0.104s; +K=N memory |
| B7 | already-fixed-at-baseline | v0.9.18; repro PASS + bench |
| A9 | fixed | `log_fit` wired into linear/pca/all 3 NB fitters |
| B4 | fixed | `_UNIQUE_FALLBACK_FACTOR=4` + WHITELIST justification |
| A10 | fixed | empty-input contract docstrings + tests |
| B6 | fixed | README version drift removed |

Artifacts: `.pingis/audits/v0934_bugfix_report.md`,
`.pingis/benches/v0934_*.json`, `.pingis/experiments/v0934_notes.md`.

