# TASK v0.9.35+ — Algorithm implementation (24 items, no deferral)

**Branch:** `pingis/dev-audit`
**Baseline:** HEAD = `v0.9.34` (after bug pass).
**Patches:** 24 individual patches (D.1.2, D.1.4, D.2.1, ..., D.6.9).
Version bumps: v0.9.35 → v0.9.36+ (each patch +0.0.1).
**Rejection/deferral:** Only for **explicitly out-of-scope** T5-T10
(torch-only, listed in `docs/ALGORITHM_BACKLOG.md`). Everything else
must be implemented.

---

## Algorithm list (24 items)

Exact list from `docs/ALGORITHM_BACKLOG.md`. Full requirements
(API, RAM, tests, bench, docs) — same file, "Universal requirements".

| # | ID | Priority | Complexity | Name | Patch |
|---|---|---|---|---|---|
| 1 | D.1.2 | P0 | M | Matrix Profile streaming | v0.9.35 |
| 2 | D.1.4 | P1 | S | fairness metrics DP/EO/TPR-gap | v0.9.35 |
| 3 | D.2.1 | P0 | M | LambdaMART lambdarank | v0.9.35 |
| 4 | D.2.2 | P0 | M | ALS (implicit) iterative Reduce | v0.9.36 |
| 5 | D.2.3 | P1 | M | hybrid MF + content | v0.9.36 |
| 6 | D.3.1 | P0 | M | CoxPH partial likelihood + Sort | v0.9.36 |
| 7 | D.3.2 | P0 | M | T/S-learner + propensity | v0.9.37 |
| 8 | D.3.3 | P1 | M | uplift trees (Qini) | v0.9.37 |
| 9 | D.4.1 | P0 | M | MiniRocket streaming | v0.9.37 |
| 10 | D.4.2 | P0 | M | PELT online change-point | v0.9.38 |
| 11 | D.4.3 | P1 | M | ETS (Holt-Winters) | v0.9.38 |
| 12 | D.4.4 | P1 | S | Theta-method | v0.9.38 |
| 13 | D.4.5 | P1 | M | VAR(p) | v0.9.38 |
| 14 | D.4.6 | P1 | M | calibration Platt/isotonic | v0.9.39 |
| 15 | D.5.1 | P1 | M | ordinal logistic | v0.9.39 |
| 16 | D.5.2 | P1 | S | classifier chains | v0.9.39 |
| 17 | D.5.3 | P1 | M | MiniBatchKMeans / GMM (EM) | v0.9.39 |
| 18 | D.5.4 | P1 | L | BIRCH CF-tree streaming | v0.9.40 |
| 19 | D.6.1 | P1 | M | mutual_info / variance / Boruta | v0.9.40 |
| 20 | D.6.2 | P1 | M | drift + A/B tests | v0.9.40 |
| 21 | D.6.4 | P1 | M | randomized SVD | v0.9.40 |
| 22 | D.6.5 | P1 | S | geo-aggregations h3/hexbin | v0.9.40 |
| 23 | D.6.6 | P1 | L | LDA | v0.9.41 |
| 24 | D.6.7 | P1 | S | LinUCB contextual bandit | v0.9.41 |
| 25 | D.6.9 | P0 | M | TF-IDF on disk | v0.9.41 |

**Batching:** S items (D.1.4, D.4.4, D.5.2, D.6.5, D.6.7) — 2-3 per
patch. M items — 1-2 per patch. L items (D.5.4, D.6.6) — 1 per patch,
may split D.x.y-1/D.x.y-2.

**Verify:** before starting each patch — `git log --oneline -1` =
previous version, run baseline fast suite (~170s), note counts.

---

## Requirements per algorithm

Exact same as `docs/ALGORITHM_BACKLOG.md` "Universal requirements":

1. Model + fit function in `cadw/ml/<name>.py` or
   `cadw/ops/<name>.py`.
2. Mixin method in `cadw/dataframe/_ml_bindings.py` (§13.4/329
   append pattern).
3. ≥8 tests (synthetic, reference, edge, determinism, multi-batch,
   RAM, guard, source contract).
4. Bench script `scripts/bench_<name>.py` — 1M rows, peak RSS vs
   reference; artifact `.pingis/benches/<name>.json`.
5. `.pingis/algorithm_notes/<name>.md` — design + sources +
   benchmarks.
6. Update `docs/ML_TASKS_TAXONOMY.md` — status ⏳ → ✓.
7. Update `docs/roadmap.json` — D.x.y → done.
8. Uniform ML log (`log_fit` helper) if iterative.
9. CHANGELOG entry (### v0.9.XX).

---

## Special requirements

### Research first (per algorithm)

Before writing code:

1. **Read the paper.** arXiv / journal. Cite key equations in
   `.pingis/algorithm_notes/<name>.md`.
2. **Read reference library source** (sklearn, statsmodels, lifelines,
   etc.). Note API contract and any surprising edge cases.
3. **Fetch benchmarks.** Kaggle, papers with code, UCR archive, etc.
   Record in algorithm_notes.
4. **Design RAM model.** State size, per-batch peak, `check_batch`
   factors. Document before coding.
5. **Pick primitive composition.** Reduce / HashShuffle / Sort —
   which fits, which doesn't.

### Sources to check (mandatory)

| Task class | Must read |
|---|---|
| ML models | sklearn docs + paper + Kaggle solution |
| TS forecast | statsmodels docs + M4/M5 comp writeups |
| Anomaly | sklearn docs + ODDS benchmark + paper |
| Survival | lifelines + scikit-survival + paper |
| Causal | EconML + CausalML docs + papers |
| Recsys | implicit lib + Kaggle RecSys comps |
| LTR | LightGBM lambdarank docs + LETOR paper |
| TS class | sktime + UCR archive + MiniRocket paper |
| Drift | evidently + alibi-detect + KS/PSI papers |
| Text | sklearn TfidfVectorizer + sklearn source |

If a public benchmark exists (KDD Cup, Kaggle, UCR, M4/M5, LETOR,
ODD, MovieLens) — cite the number in algorithm_notes. Never invent
benchmarks — if no verified source, say "no verified citation".

### Substitution / addition (allowed, justified)

- **Substitution:** if a better algorithm exists (paper + benchmark +
  RAM class equal-or-better), substitute. Must include:
  - Original plan vs substitute name
  - Paper source
  - Benchmark comparing
  - RAM model comparison
- **Addition:** may add algorithms not in list if they unlock a task
  type. Justify in `.pingis/experiments/<name>.md`.

### External dependencies

- If an algorithm requires a package > 10 MB (e.g. `h3`, `stumpy`):
  - Default: implement natively (preferred per project constraint).
  - If native is impractical: propose dependency in the patch message;
    user approves/rejects. Do not silently install.
- Small deps (< 10 MB, pure Python, MIT/BSD/Apache) may be added if
  the alternative would be > 300 LOC of custom code.

---

## Reporting structure

```
.pingis/
├── algorithm_notes/
│   ├── d1_2_matrix_profile.md
│   ├── d1_4_fairness.md
│   └── ... (24 files)
├── benches/
│   ├── d1_2_matrix_profile.json
│   └── ... (24 files)
├── experiments/
│   ├── v0935_summary.md           ← 24-item progress table
│   └── <name>.md                  ← per-algorithm edge cases
└── audits/
    └── v0935_algo_report.md       ← final summary (when all done)
```

`v0935_summary.md`:
```markdown
# v0.9.35+ — Algorithm progress

| ID | Version | Status | Tests | Bench (RSS/time) | Reference |
|---|---|---|---|---|---|
| D.1.2 | v0.9.35 | ✓ | 9 | 45 MB / 2.1 s | stumpy: 42 MB / 1.9 s |
| D.1.4 | v0.9.35 | ✓ | 8 | 2 MB / 0.05 s | fairlearn: 2 MB / 0.04 s |
| ... |
| D.6.9 | v0.9.41 | ✓ | 10 | 120 MB / 8 s | sklearn: 130 MB / 6 s |
```

---

## Anti-patterns (auto-reject)

- **Skip tests.** Every algorithm ≥8 tests, no exceptions.
- **`pytest.skip` / `pytest.mark.skip`** unless external tool absent
  AND noted in report.
- **Fake benchmark.** Numbers from empty data, single batch, or
  cherry-picked.
- **No reference comparison.** "I trust my implementation" → reject.
- **Silent algorithm substitution.** Replacing plan without
  justification + benchmark → revert.
- **`import pandas` in `cadw/`.** Only in tests.
- **`.to_pylist()` on batch without `check_batch(extra_factor≥2)`**
  → reject (AST guard).
- **Materializing full dataset.** Peak RSS > 2× `pred_total` → reject.
- **Partial deliverable.** "D.1.2 mostly works" → reject. Complete or
  split into sub-patches (`D.x.y-1`, `D.x.y-2`) with **both landed**.
- **Hardcoded test expectations matched to current code.** Not
  contract-driven. Use `sklearn`/`statsmodels` reference.

---

## Acceptance (per patch)

Each patch:

- All algorithm's tests pass.
- Fast suite green (2557+ passed).
- Slow suite green (25+ passed).
- Bench artifact under `.pingis/benches/`.
- Algorithm_notes under `.pingis/algorithm_notes/`.
- ML log uniform (if iterative).
- `docs/ML_TASKS_TAXONOMY.md` status update.
- `docs/roadmap.json` item → done.
- CHANGELOG entry.
- One commit.

## Acceptance (whole task)

When all 24 items done:

- Fast suite: 2557+ 24×8 = **~2750 passed**.
- Slow suite: 25+ passed.
- All `D.x.y` items done in `docs/roadmap.json`.
- `.pingis/audits/v0935_algo_report.md` complete.
- Phase D → `completed` in `docs/roadmap.json`.
- Tag `v0.9.41` (or actual final version).

After acceptance → Phase E (recipes) + Phase G (tech debt close) per
`docs/roadmap_research/consolidated_roadmap.md`.
