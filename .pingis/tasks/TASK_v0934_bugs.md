# TASK v0.9.34 — Bug fix pass (8 items, no deferral)

**Branch:** `pingis/dev-audit` (same as v0.9.32)
**Baseline:** HEAD = `38c8eb7 v0.9.32`, tests: 2557 passed, 2 skipped.
**Patch:** **one commit** `v0.9.34: bugfix pass (8 items)`.
**Rejection/deferral:** NOT ALLOWED. Each item is either fixed or
rejected with **explicit proof** (repro + measurement showing it's not
a bug). "wontfix" without measurement → rejected commit.

---

## Bug list (8 items)

Exact list from `docs/BUG_BACKLOG.md`. Full repro + contract per bug
is there.

| # | ID | Priority | Bug | File |
|---|---|---|---|---|
| 1 | B1 | P0 | Quadratic `_RunStream.refill` | `cadw/ops/sort.py` |
| 2 | B2 | P0 | `encode_categorical` Python-set | `cadw/dataframe/_ml_bindings.py` |
| 3 | B3 | P1 | ROC-AUC pass2 Python loop | `cadw/ops/metrics.py:639` |
| 4 | B7 | P1 | `_string_hash_column` Python loop | `cadw/ops/shuffle.py:177` |
| 5 | A9 | P1 | ML log wiring (3/6) | `cadw/ml/{linear,pca,naive_bayes}.py` |
| 6 | B4 | P2 | `memory_guard` fallback factor | `cadw/core/memory_guard.py:700` |
| 7 | A10 | P2 | empty-input contract docs | docstrings |
| 8 | B6 | P2 | README version drift | `README.md` |

---

## Requirements per bug

For each item:

1. **Repro first.** Write `tests/test_v0934_<id>_<name>.py` (or
   inline test) with the failing case from `docs/BUG_BACKLOG.md`.
   Run it. Confirm it fails **for the documented reason**.

2. **Fix.** Apply the documented fix (see `BUG_BACKLOG.md` per-item
   section).

3. **Test green.** After fix, run test — must pass.

4. **Fast suite.** `uv run pytest -q -m "not slow"` → **all green**
   (2557+ passed, no regressions).

5. **Bench artifact.** For B1/B2/B3/B7: `scripts/bench_v0934_<id>.py`
   → output `.pingis/benches/v0934_<id>.json` with before/after
   numbers (time, peak RSS, copies).

6. **Commit message.** One final commit includes per-bug one paragraph
   (what was bug, what fixed, numbers before/after).

---

## Reporting structure

All artifacts under `.pingis/`:

```
.pingis/
├── audits/
│   └── v0934_bugfix_report.md         ← summary: 8 items status
├── benches/
│   ├── v0934_b1_refill.json
│   ├── v0934_b2_encode.json
│   ├── v0934_b3_rocauc.json
│   └── v0934_b7_hash.json
└── experiments/
    └── v0934_notes.md                  ← edge cases / surprises
```

`v0934_bugfix_report.md` format:

```markdown
# v0.9.34 — Bug fix report

| ID | Status | Fix | Before | After | Test |
|---|---|---|---|---|---|
| B1 | fixed | linear parts_list | 12.9 GB copies | 0.13 GB copies | pass |
| B2 | fixed | K_est + check_state | OOM @ 10M uniq | MemoryLimitExceeded early | pass |
| B3 | fixed / rejected (measurement) | ... | ... | ... | pass |
...
```

---

## Anti-patterns (auto-reject)

- **Skipping test.** "Test would fail without fix, but I trust it" →
  reject.
- **Wontfix without measurement.** "This is fine" → reject unless
  measured (RSS/time/bug-free).
- **Partial fix with deferral.** "Fixed B1, B2 partial, rest next
  patch" → reject. Fix all 8, or provide explicit rejection proof
  for the deferred ones.
- **Changing test expectations to pass.** Test suite is contract.
- **Silent expansion of WHITELISTED_FUNCS.** No — each new entry must
  be justified inline.

---

## Acceptance

- All 8 bugs fixed (or rejected with measurement).
- Fast suite: 2557+ passed, 0 failed.
- Slow suite: 25+ passed, 0 failed.
- Bench artifacts present in `.pingis/benches/`.
- One commit `v0.9.34: bugfix pass (8 items)` on `pingis/dev-audit`.
- Report `v0934_bugfix_report.md` complete.

After acceptance → proceed to TASK_v0935 (algorithms).
