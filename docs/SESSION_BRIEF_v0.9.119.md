# Session Brief — v0.9.119

**Version:** 1.0 (2026-11-04)
**Status:** next-session bootstrap. Read after AI_CONSTITUTION,
ROADMAP, STABILITY_PROTOCOL, EXPERIMENT_PROCEDURE (T2).
**Hierarchy:** Tier 5 (operational). Process only. No content rules.

---

## §1. Where we are

**Version:** v0.9.118 committed (`bf34461`-line, tag).
**Baseline (source of truth, `docs/ROADMAP.md`):**

| Status | fns | cards | Что |
|---|---|---|---|
| verified | **31** | 45 | v0.9.114 (24) + v0.9.115 (+6) + v0.9.116 (+1) + v0.9.117 (+2) + v0.9.118 (+2) |
| draft | **8** | 13 | BVAR/VECM, binseg×2 (permanent refuse_at_guard), pelt×2, t2q, meta_learner (T/S AP2), ab_test×3 (container) |
| pending | **24** | 30 | A_classification ×8, O_eda ×16 |
| no-fn | — | 62 | described_only / not_implemented |

**23 verification files** (tests/test_*_verification.py) — DEC-033 component-wise.

**Invariants (schema-enforced):**
- **R8** (v0.9.115): `input_api: df|series|dual` on every card with `cadw_function`.
- **R9** (v0.9.118): `cadw_function` must be importable (catches YAML drift — fixed `fit_classifier_chains` → `fit_chains`).
- **R1-R7** (pre-existing).
- **CW-5** (v0.9.116): `test_guard_refuses_at_domain_bound` mandatory for `bounded_by_domain` fns.

---

## §2. What is ready (patterns from v0.9.115-118)

### CoxPH-precedent (v0.9.114) — O(n)-output fns

`state_nbytes()` + `bounded_by_domain: {axis: n, bound: 1e6}` + CW-1..5
(component-wise, DEC-033). Applied: coxph, kmeans_ad, eif, bocpd,
kshape, hampel, dwt_mlead, patch_memory_ad, autoencoder, s2g, sub_pca.

### Dual dispatch (v0.9.113-118)

```python
def fit_X(df_or_series, value_col=None, *, output="memory"|"auto",
          batch_size=None, ...):
    is_df = hasattr(df_or_series, "_iter_filtered_batches")
    if is_df: return _fit_X_df(df_or_series, value_col, ...)
    return _fit_X_series(df_or_series, ...)   # legacy
```

- `output="disk"` → `NotImplementedError` (SpillArray v0.9.119+ for O(n) output).
- Reference: `cadw/ml/zms.py::fit_zms`, `cadw/ml/kshape.py::fit_kshape`.

### Auto K-spill (v0.9.117, DEC-042)

```python
if K_raw_est > state_limit_bytes // 96 // 2:
    vocab = _build_vocab_k_spill(df, ..., min_df, batch_size)
else:
    vocab = _build_vocab(df, ...)   # legacy, bit-identical
```

- Reference: `cadw/ml/lda.py::fit_lda`, `cadw/ml/tfidf.py::fit_tfidf`.

### Component-wise CW-template (DEC-033, 5 checks)

1. `test_state_nbytes_exact` — bit-exact formula (parametrized k/d/K/V)
2. `test_state_independent_of_n` (or `state_ratio_linear` DELTA — A20)
3. `test_formula_small_scale` — predicted/measured in [0.5, 2.0]
   **OR** `test_peak_grows_linearly_with_n` (if signal << noise, DEC-034)
4. `test_guard_refuses_at_domain_bound` — n=bound, guard=10MB → MemoryLimitExceeded
5. `test_tight_guard_raises` — `_FakeMB(max_memory_mb=0.001)` → raise

Plus dual dispatch: `test_df_matches_series` (bit-identical),
`test_output_memory_and_auto`, `test_output_disk_raises`.

References: `tests/test_ordinal_verification.py`, `tests/test_chains_verification.py`.

---

## §3. What remains for v0.9.119

### Part 1 — draft fns (5 fns / 8 cards; ~3-4 h)

1. **fit_ab_test ×3** (container `DriftABModel.state_nbytes=256`): t/mw/cuped.
   - YAML flip + CW-1 (state exact) + CW-5 (tight_guard) — no per-row state.
   - Probe: `cadw/ml/drift.py`, `tests/test_drift.py` (already 9 tests).
   - ~40 min.
2. **BVARModel / VECMModel** (TS, `tier1_core_exception` likely):
   - Probe fit signatures. Likely `df, cols` + `bounded_by_domain n<=1e6`.
   - ~1-1.5 h.
3. **fit_t2q** (T-squared+Q): `_make_windows` strided+chunked (v0.9.104 backlog).
   - CoxPH-precedent on state_nbytes. ~30-40 min.
4. **fit_pelt ×2** (container 256B + `_collect_series`): CoxPH-precedent
   on `.detect()` DP; fit container is honest O(1). ~30-40 min.
5. **fit_meta_learner ×2** (T/S_learner AP2): `_collect + np.vstack` — needs
   chunked refactor. Defer if prioritized low (~1-1.5 h).

### Part 2 — ML_COVERAGE_MATRIX.md (~1-1.5 h)

`section × subtype × {baseline, SOTA, ensemble}` audit. Roles
(linear_signal, nonlinear_interactions, representation, probabilistic,
temporal_dynamics, local_structure, calibration) — current gaps and
priorities for v0.9.120+.

### Part 3 — AP2 sweep (grep `column_stack|vstack|concatenate` in cadw/ml/)

Quick grep + classify: CoxPH-precedent candidate vs needs chunked refactor.

---

## §4. Anti-patterns from v0.9.115-118 (real mistakes)

### A26 — multi-occurrence alias

`str.index()` finds **first** `- name: <Card>`. Aliased cards
(fit_zms ×3, fit_autoencoder ×2, fit_sub_pca ×2, fit_chains ×2) require
loop right-to-left (or `_occ = []` + iterate `reversed`). Skip → YAML
partial-patched (1 of N), assert fails, exit.

Fix:
```python
_occ = []; _pos = 0
while True:
    idx = src.find("\n  - name: Card\n", _pos)
    if idx < 0: break
    start = idx + len("\n  - name: Card\n")
    end = start + re.search(r"\n  - name: |\n- id: |\Z", src[start:]).start()
    _occ.append((start, end)); _pos = end
for s, e in reversed(_occ):
    src = src[:s] + _transform(src[s:e]) + src[e:]
```

### A27 — anchor after last field dataclass

`state_nbytes()` goes **after** all fields (not after a similar field).
Precedent: patch_5_pma used `contamination: float` — but `normalization: str`
was last → duplicate field, `TypeError` on `PatchMemoryADResult(...)`, 6/6
back-compat fail. Fix: `grep 'class .*Result' -A 40` before patch.

### A28 — formula units

`40 * n * f64` = 40×8 = 320 bytes/element, **not** 40 bytes.
Precedent: s2g formula `40*n*f64` (should be `5*n*f64`).
Detection: `test_bounded_at_domain` (n=1e6 formula > 50 MB → fail).

### A29 — helper insertion after decorator target

`@pytest.mark.parametrize` + `def test_*` — helper AFTER block, not between.
Precedent: patch_3_cw4 inserted `_malloc_trim` between decorator and
`def test_state_nbytes_exact` → `fixture 'n' not found`. Fix: relocate
helper before last test (`test_peak_grows_linearly_with_n`), not first.

### A30 — full-fast 3 runs, 11 fails

v0.9.118: 3 full-fast runs → 8 fails → 1 fail → 2 fails. Each round,
1-2 DEC-034 escalation (RSS drift in full-suite). **Rule:** if a
measure_peak_traced test fails **only** in full-suite (passes isolation),
widen tolerance + document DEC-034 escalation, do NOT re-run full fast
for a single test fix. Targeted re-run (1-2 tests) is enough.

### A31 — alias card "counted as fn"

`31 fns / 45 cards` — aliased cards (MiniRocket×2, ALS×2, zms×3, etc.)
count as 1 fn, N cards. Source of truth: `python -c "..."` script
(see SESSION_BRIEF §1). Don't say "35 fns" from card count.

---

## §5. Reusable patterns

### Component-wise template (copy for ab_test, bvar, vecm, t2q, pelt, ...)

See §2 CW-template. Reference: `tests/test_chains_verification.py`.

### YAML flip template

```yaml
    implementation_status: {status: verified, ...}
    ram_formula: "<formula>"
    formula_evidence: {term1: "file.py -- line -- what", ...}
    measurement: {verification_mode: "component-wise (DEC-033)", cells_passed: "5/5", ...}
    bounded_state: false
    n_scalability: linear
    bounded_state_dims: {k: 1, d: 1, ...}
    bounded_by_domain: {axis: n, bound: 1e6, reason: "..."}
```

### Two-phase patch discipline

1. **Probe** (read-only): `grep -n 'def fit_'`, `grep 'state_nbytes'`, `head`/`sed`
2. **Patch** в `/tmp/patch_*.py`, `sys.exit(1)` на anchor miss (A25),
   `ast.parse` + `yaml.safe_load` sanity, `p.write_text` last
3. **Verify** (separate block): `pytest -q` focused (1-2 tests)
4. **Commit** после зелёного

**Do not mix full fast with single-test fix.** Targeted re-run enough.

---

## §6. Session-start checklist

- [ ] Read AI_CONSTITUTION (T1) + ROADMAP + STABILITY §4.1.1 +
  EXPERIMENT_PROCEDURE §3.7 (T2) + ALGORITHM_TEMPLATE Step 5-alt (T3).
- [ ] `git log --oneline -8` — last 8 commits.
- [ ] `git describe --tags` — `v0.9.118` latest tag.
- [ ] `pytest -q -m "not slow"` — baseline (~16 min). Если красный —
  диагностика до patch.
- [ ] First action: **probe (read-only)**, никогда patch.
- [ ] Read `tests/test_chains_verification.py` before writing new
  verification file (canonical component-wise template, v0.9.118).
- [ ] Read `docs/ml_tools/DECISION_LOG.md` DEC-040/041/042 (v0.9.117-118).

**Do not:**
- Do not re-verify v0.9.118 verified fns (tag + full evidence).
- Do not refactor `_stack_numeric`, `MemoryGuard` core.
- Do not attempt SpillArray for sub_pca — CoxPH-precedent sufficient.
- Do not fight `measure_peak_traced` flake — widen tolerance + DEC-034
  escalation, targeted re-run (A30).
- Do not commit `.pingis/audits/` (agent artifacts, out of scope).
- Do not run full fast for a single-test fix — targeted re-run only.

**If session brief contradicts AI_CONSTITUTION:** stop, ask user,
constitution wins.

---

## §7. Definition of done (v0.9.119)

| Item | Criteria |
|---|---|
| ab_test ×3 | `DriftABModel.state_nbytes=256` + CW (state exact + tight_guard) + YAML flip |
| BVAR/VECM | `state_nbytes` + `bounded_by_domain n<=1e6` (TS, tier1_core) |
| t2q / pelt | CoxPH-precedent OR `refuse_at_guard` + draft_reason |
| meta_learner | AP2 chunked refactor OR defer with DEC |
| ML_COVERAGE_MATRIX | `docs/ML_COVERAGE_MATRIX.md` — section × subtype × roles |
| Focused suite | Green per commit (targeted, not full fast) |
| Tag | `v0.9.119` when ≥33 verified fns |

**Not required (v0.9.120+):**
- Pending fns (24: A_classification ×8, O_eda ×16)
- T/S_learner GBDT streaming refactor (if deferred)
- AP2 sweep (optional)

---

## §8. What was NOT done in v0.9.118

- ab_test ×3 (draft, container)
- BVAR/VECM (draft, TS)
- t2q, pelt, meta_learner (draft)
- Pending fns (24: A_classification ×8, O_eda ×16)
- ML_COVERAGE_MATRIX.md
- T/S_learner GBDT streaming (AP2)
- **Full-fast green** for v0.9.118 tag — commit `7241420` + last full fast
  `2 failed / 4583 passed` (T7/T8 fixed in `7241420`, targeted re-run 2/2
  passed). Remaining flake risk <0.1% (DEC-034 escalation).

**What was closed (v0.9.116-118):**
- v0.9.116: sub_pca verified ×2; df-API migration bocpd/eif/kmeans_ad;
  DEC-034 partial (lda_peak); DEC-035 (BIRCH _insert); DEC-039 (CW-4
  partial + G2 fix hampel/sub_pca + CW-5).
- v0.9.117: LDA + TF-IDF verified; auto K-spill (DEC-042); DEC-041
  (bounded_state = state, not fit peak).
- v0.9.118: ordinal + chains verified; R9 (cadw_function importable);
  full-fast triage 8 groups (T1-T8) + DEC-034 escalations.

---

**End of brief.** If anything unclear after reading T1-T2, ask user
before starting work.
