# Session Brief — v0.9.129

**Version:** 1.0 (2026-11-06)
**Status:** next-session bootstrap. Read after AI_CONSTITUTION,
ROADMAP, STABILITY_PROTOCOL, EXPERIMENT_PROCEDURE (T2).
**Hierarchy:** Tier 5 (operational). Process only. No content rules.

---

## §1. Where we are

**Version:** v0.9.128 committed (`077b06f`, tag).
**TAXONOMY FROZEN** (DEC-052, AI_CONSTITUTION §5.3): 58/58 subtypes
have ≥3 diverse candidates or explicit NOT-ENSEMBLE.

**Baseline (source of truth, `docs/ROADMAP.md`):**

| Status | fns | cards |
|---|---|---|
| verified | 32 | 48 |
| draft | 14 | 17 |
| described_only | — | 61 |
| inference-only | — | 6 |
| last-priority (S) | — | 5 |

**9 recent tags:** v0.9.119 … v0.9.128.

**Invariants:**
- R8 (input_api), R9 (cadw_function importable), R10 (public fit_* covered,
  ml + ts), CW-5 (guard-refuses-at-domain-bound), DEC-033 component-wise.

**Weak machine** (DEC-050): no GPU, 8-16 GB RAM. Not "2.8 GB budget".

---

## §2. What is FROZEN (do NOT touch)

**Card additions STOPPED at v0.9.128** (DEC-052). Post-freeze allowed only:

- `status: draft → verified` (formula from code AP23 + measurement DEC-033)
- `ram_formula` / `formula_evidence` completion for existing draft cards
- `cross_ref` field formalization (markdown → schema field)
- `ensemble_recommendation` refinements

**Do not** add new candidates to subtypes. **Do not** create new sections.
**Do not** create `tier: ensemble` cards without explicit DEC (would
duplicate `ensemble_recommendation` — DEC-050).

**Do not resurrect:** K_vision, L_graph, U_automl, TabPFN, BERT, BERTopic,
NormalizingFlow, Chronos-Bolt-Small/Base, Timer-S1, t0-beta, AutoGluon-TS.
Deleted v0.9.126 by weak-machine / non-tabular / non-OOC criteria.

---

## §3. Scope v0.9.129 (3 parts, ~3-4 hours total)

### Part A — Verify 4 ts drafts (1-1.5 h)

**Targets:** `EnsembleForecaster`, `OTMThetaForecaster`,
`TemporalHierarchy`, `TransformedTargetForecaster` (D_time_series.yaml
subtype D1, current `status: draft` with `draft_reason: "verified in
cadw.ts vX; YAML ram_formula pending v0.9.128"`).

**Probe first:**
```bash
grep -n "def fit\|def predict\|state_nbytes\|state_bytes\|check_state\|check_batch" \
  cadw/ts/{ensemble,otm,thf,transforms}.py
```

**For each card:**
1. Formula from code (AP23): sum of state + batch transient terms.
2. `formula_evidence` per term (`file.py:line`).
3. `measurement: {verification_mode: component-wise (DEC-033)}`.
4. Test file `tests/test_<card>_verification.py` (CW-1..5 from
   `tests/test_chains_verification.py` template).
5. Status `draft → verified`.

### Part B — `fit_stacking` pilot (1-1.5 h)

**Target:** `cadw/ml/stacking.py::fit_stacking`.

```python
def fit_stacking(df, base_models, meta_learner, *, cv=5, output="auto", ...):
    """OOC k-fold stacking. base_models: list of (name, fit_fn, predict_fn).
    meta_learner: same. Returns StackingModel with .predict_proba(df).
    OOC: O(batch) state per fold; OOF predictions stored on disk (SpillArray).
    """
```

**Test:** A1 (LogReg + GBDT + NB → stacked LogReg) on synthetic
binary classification. Compare stacking accuracy vs best base (should be
within 1-2% on well-tuned bases; improvement on noisy data).

**Not a card** — infrastructure function (like `fit_ab_test`). Not in
YAML. `primitive_reuse: [spill_array]`.

### Part C (bonus, if time) — `cross_ref` schema field (1 h)

Formalize DEC-049 markdown `# cross-ref: <home>.<card>` into
`ModelCard.cross_ref: Optional[str]`. Add `_check("cross_ref", ...)` in
schema. Add `test_R11_cross_ref_consistency` (canonical home must exist).
Convert 6 existing markdown comments.

---

## §4. Anti-patterns (from v0.9.119-128, real mistakes)

### A1. `git add -A` picks up agent artifacts
- **Happened:** v0.9.126 pulled 108 files `.pingis/audits/` (~9.5k lines).
- **Fix:** explicit `git add <list>`. Never `-A` unless `.gitignore` verified.
- **Detection:** `git status --short | grep -v "expected-paths"`.

### A2. Insertion before `from __future__`
- **Happened:** v0.9.127b broke `test_ml_tools_schema.py` (SyntaxError).
- **Fix:** PEP 236 — `from __future__` is first statement after docstring.
- Check: `assert not lines[i-1].strip().startswith("from __future__")`.

### A3. `sed 's/^def x($/def x /'` removed opening paren
- **Happened:** v0.9.128 — f-string became module-level → NameError.
- **Fix:** never sed-edit Python parens. Use `re.sub` with explicit match,
  AST-check after.

### A4. `^[ 	]*metrics:` matched `metrics:` inside `benchmarks:` list
- **Happened:** v0.9.128 — cards inserted into `benchmarks:` list,
  `BenchmarkRef got 'tier'`.
- **Fix:** anchor boundary to `models:` block end (indent-aware), not
  `metrics:`.

### A5. Duplicate `no_public_benchmark:` prefix in template
- **Happened:** v0.9.128 — `reason` already starts with prefix.
- **Fix:** template `note: '{reason}'` (reason includes prefix).

### A6. Role-shorthand vs full name
- **Happened:** v0.9.128 — `roles: [linear]` vs `linear_signal`.
- **Fix:** `ROLES` enum is canonical; `linear`, `nonlinear`, `local` fail.

### A7. Naive `importable()` in probe = false-positive
- **Happened:** v0.9.126 — 37 «dangling» reported, actual = 0 (R9 green).
- **Cause:** `_load_cards` indexes by 3 keys (function, module::function, name);
  probe `importable("OneVsRest_Logistic")` = False because card name not callable.
- **Fix:** use R9 mechanism or filter `cadw_function` field only.

### A8. Per-method expansion_factor hardcoded
- **Happened:** v0.9.119 — `fit_ab_test` used `expansion_factor=5.0` for
  all methods (t=4.0, cuped=5.0, mw=5.0). Batch inconsistent with planner.
- **Fix:** `op.batch_memory_factor()` per method. Build op before batch_size.

### A9. Missing anchor metric in new subtype
- **Happened:** v0.9.128 — 5 cards used `metric:` not in declared
  `subtype.metrics`. Schema requires anchor covered.
- **Fix:** probe `subtype.metrics` first, choose metric from that list.

### A10. ROADMAP order broken (prepend vs append)
- **Happened:** v0.9.128 prepended before v0.9.127 (summary insertion point).
- **Fix:** verify `grep -n "^\*\*v0.9" ROADMAP.md` — ascending v0.9.NNN.

---

## §5. Reusable patterns (from this session)

### Insertion helper (indent-aware)

```python
def _reindent(text, from_indent, to_indent):
    """Shift non-blank lines by delta."""
    delta = to_indent - from_indent
    if delta == 0: return text
    return "".join(
        (" " * max(0, (len(l) - len(l.lstrip(" "))) + delta) + l.lstrip(" ")
         if l.strip() else l)
        for l in text.splitlines(keepends=True)
    )

def insert_card(src, subtype_id, card_yaml):
    """Insert before `metrics:` at subtype level. Handles id=0/2."""
    pat = re.compile(rf'^([ \t]*)- id: {re.escape(subtype_id)}\s*$', re.MULTILINE)
    m = pat.search(src)
    id_indent = len(m.group(1))
    # next subtype boundary (same/less indent)
    # models: at id_indent+2, cards at various
    # boundary = first line after models: with indent <= models_indent
    ...
```

Reference: `tests/test_ml_tools_schema.py::test_R10_cadw_ts_covered`
(v0.9.127) — TS_UTILITY_WHITELIST for alias-cases.

### Component-wise CW-template

5 checks per model: `test_state_nbytes_exact`, `test_state_bounded_for_n_ge_s`
(or `test_state_ratio_linear` DELTA), `test_batch_coef_isolated`,
`test_formula_small_scale` (or `test_peak_grows_linearly_with_n` if
signal << noise), `test_ooc_batch_fixed` (or `test_tight_guard_raises`
if OOC unmeasurable).

Reference: `tests/test_chains_verification.py` (v0.9.118 canonical).

---

## §6. Session-start checklist

- [ ] Read AI_CONSTITUTION (§1-§5, especially §5.3 freeze).
- [ ] Read ROADMAP (§Current, §Open post-freeze, §Next 3 sessions).
- [ ] Read STABILITY_PROTOCOL §4.1.1 + EXPERIMENT_PROCEDURE §3.7.
- [ ] `git log --oneline -8` — last 8 commits.
- [ ] `git describe --tags` — `v0.9.128` expected.
- [ ] `git status --short` — clean (no `.pingis/audits/` untracked — should
  be ignored).
- [ ] `pytest -q tests/test_ml_tools_schema.py` — 108 passed expected.
- [ ] **First action: probe (read-only)**, никогда patch.

**Do not:**
- Add cards to subtypes (freeze).
- Resurrect deleted sections (K/L/U, TabPFN, BERT, etc.).
- Run full-fast for single-test fix (A30 from prior brief — targeted only).
- Use `git add -A` (A1 from this brief).
- Edit parens with `sed` (A3).

**If brief contradicts AI_CONSTITUTION:** stop, ask user, constitution wins.

---

## §7. Definition of done (v0.9.129)

| Item | Criteria |
|---|---|
| 4 ts drafts | `status: verified` + `ram_formula` + `formula_evidence` + CW-1..5 |
| `fit_stacking` | `cadw/ml/stacking.py` + 1 test (A1 LogReg+GBDT+NB) + `primitive_reuse: [spill_array]` |
| `cross_ref` (bonus) | `ModelCard.cross_ref` + `test_R11` + 6 markdown → field |
| Tests | Targeted green per commit (not full-fast per A30) |
| Tag | `v0.9.129` when 4 drafts → verified |

**Not required (v0.9.130+):**
- `expansion_factor` migration (18 call sites, high risk)
- `.pingis/impl/` promotion (11 models, 2-3 sessions)
- Pending fns (A_classification ×8 + O_eda ×16)

---

## §8. What was NOT done in v0.9.119-128

**Verified as draft-target (numpy implementable, later):**
- 42 described_only candidates (all v0.9.128)
- 4 ts drafts (v0.9.129 scope)
- 11 `.pingis/impl/` models (v0.9.131 promotion)
- 24 pending fns (A_classification ×8, O_eda ×16)

**Deferred with rationale:**
- 5 S_generative last-priority (v0.9.130+)
- 6 inference-only (import via to_batches, fit not ours)
- 8 NOT-ENSEMBLE (fusion impossible/meaningless)

**Infrastructure deferred:**
- `fit_stacking` (v0.9.129 pilot)
- `cross_ref` schema field (v0.9.129 bonus)
- `expansion_factor` migration (v0.9.130)
- ML_COVERAGE_MATRIX.md (backlog)

**Closed in this session:**
- Pipeline planner v2 (v0.9.119)
- `solve_batch` inverse (v0.9.119)
- R10 coverage gate (v0.9.121, extended ts v0.9.127)
- `cadw.ts` coverage gap (v0.9.127)
- Weak-machine policy + cleanup (v0.9.126)
- TAXONOMY FROZEN (v0.9.128)

---

**End of brief.** If anything unclear after reading T1-T2, ask user
before starting work.
