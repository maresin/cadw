# TASK v0.9.50 — M3: migrate legacy A–H (v2 → v3)

**Branch:** pingis/dev-audit
**Baseline:** HEAD = v0.9.49.1, tests 3077 passed, 2 skipped.
**Mode:** migration + enrichment (web fetch). Edit existing cards, no new cards.
**Resume:** allowed (§Resume).

## Goal

Migrate 6 legacy sections (A, D, E, F, G, H) from v2 schema
(`benchmark_result` singular, no attribution / algorithm_source_url /
task_type) to v3 (list of `BenchmarkResult` with two URLs + attribution).
Close all 53 `[LEGACY]` warnings. Do NOT create new cards; edit existing.

## Sections & card counts (from `.pingis/audits/v0.9.48_final_report.md`)

| Sec | File | Cards |
|---|---|---|
| A | A_classification.yaml | 18 |
| D | D_time_series.yaml | 19 |
| E | E_anomaly.yaml | 5 |
| F | F_recsys.yaml | 4 |
| G | G_learning_to_rank.yaml | 2 |
| H | H_causal.yaml | 5 |

Total: 53 cards. One checkpoint commit per section.

## Per-card recipe

For each `model` in the YAML:

1. **Convert** `benchmark_result: {source, url, metric, value, note}`
   → `benchmark_results: [ { … v3 … } ]`.
2. **Fill v3 fields** on the converted entry:
   - `benchmark: <old source>` (rename)
   - `benchmark_url: <old url>` (rename)
   - `algorithm: <model.name>` — e.g. "LogisticRegression", "GBDT",
     "IForest". NOT "cadw.*" for the literature entry.
   - `algorithm_source_url: <canonical paper | docs | code>` — **REQUIRED**
     for literature entries. Fetch and record in url_log.
   - `algorithm_source_kind: paper | docs | code | pseudocode | description`
     (prefer `paper` for classical methods, `docs` for library baselines)
   - `attribution: literature`
   - `task_type: <from subtype>` — e.g. "binary classification",
     "univariate forecasting", "unsupervised anomaly"
   - `n_datasets: null`, `std: null` (legacy had no such info)
   - `metric: <old metric>`, `value: <old value>`, `note: <old note>` +
     suffix `[legacy v0.9.44, migrated v0.9.50]`
3. **Fetch** `benchmark_url` and `algorithm_source_url`. Log both to
   `.pingis/audits/v0.9.50_url_log.jsonl`.
4. **Update value** if fetched number differs from legacy value:
   - value: <new fetched number>
   - note: `<old note> [updated v0.9.50: was <old> → <new> per fetched quote: '<snippet>']`
5. **If fetch failed** (404/429/timeout) AND no alternative URL works:
   - keep legacy `value` unchanged (do NOT null it — legacy data is precious)
   - `algorithm_source_url: ""` (empty)
   - `algorithm_source_kind: "description"`
   - note: `<old note> [legacy v0.9.44, source not re-verified: <reason>]`
   - audit_url_log will emit `[LEGACY_SOURCE]` WARN (not ERROR) for
     `algorithm_source_url=""` on `attribution=literature`
6. **For 5 M2-covered models**: add a second entry `attribution=cadw_reproduced`:
   - `benchmark: <same as literature entry>`
   - `benchmark_url: .pingis/benches/reproduction/<name>.json`
   - `algorithm: cadw.ml.<module>.fit_<fn>`
   - `algorithm_source_url: cadw/ml/<module>.py`, kind=code
   - `reproduces: <literature algorithm>`
   - `value: <from JSON>`, `n_datasets: 1`
   - `note: "M2 parity, verdict=consistent, tol=<atol|rtol>"`

   Models (only if `.pingis/benches/reproduction/<name>.json` exists):
   - E.E1.IForest → iforest.json (sklearn.IsolationForest)
   - A.A3.GBDT_L2 → gbdt_regressor.json (sklearn.GradientBoostingRegressor)
   - D.D2.VAR → var.json (statsmodels.VAR)
   - F.F1.ALS → als.json (implicit.als.AlternatingLeastSquares)
   - F.F2.ALS → als.json (same file, both F1 and F2 use ALS)

7. **For legacy CADW-measured cards** (D.3.3.CausalTree, D.6.6.LDA,
   D.6.8.GMM, D.1.2.MatrixProfile — verify which exist in CADW):
   add `cadw_measured` entry:
   - `benchmark: cadw_internal`
   - `benchmark_url: .pingis/benches/<name>.json` (if file exists)
   - `algorithm: cadw.ml.<module>`, kind=code
   - `value: <from JSON>` or null with note
   - Skip if the bench file does not exist; note in card "no local bench yet".

8. **Fix obvious bugs in legacy YAML**: e.g. D.D2.VAR has
   `metric: wrmssе` with **Cyrillic е** (U+0435). Fix to Latin `wrmse`.
   Check that the fixed metric is in `benchmarks.yaml[M5].metrics`.

## URL log (mandatory)

Same format as v0.9.48. Append to `.pingis/audits/v0.9.50_url_log.jsonl`:
```jsonl
{"ts": "…", "url": "…", "status": 200, "bytes": N,
 "for_card": "A.A1.GBDT", "algorithm": "GBDT",
 "metric": "roc_auc", "value": 0.842,
 "quote": "<literal snippet from page>"}
{"ts": "…", "url": "…", "status": 429, "for_card": "D.D1.SARIMA",
 "note": "rate limited; will try alternate source next pass"}
```

If both URLs fetch fail and you run out of alternates → record both
failures, set empty `algorithm_source_url`, audit produces WARN.

## Algorithm source URL candidates (guidance, not exhaustive)

For classical methods prefer canonical paper DOI/arXiv; for library
baselines prefer official docs; for CADW-internal, cadw/ml/*.py.

Examples:
- LogisticRegression → sklearn docs (linear_model.LogisticRegression)
- GBDT → Friedman 2001 Annals of Statistics (DOI 10.1214/aos/1013203451)
  OR LightGBM NeurIPS 2017 paper (for reference impl, use canonical)
- IForest → Liu/Ting/Zhou ICDM 2008 (DOI 10.1109/ICDM.2008.17)
- ETS → Hyndman et al. Springer book (or statsmodels docs)
- SARIMA → statsmodels docs or Box-Jenkins
- VAR → statsmodels docs
- NaiveForecaster → Hyndman "Forecasting: principles and practice"
- T_learner / S_learner → Künzel et al. PNAS 2019
- CoxPH → Cox 1972 JRSS-B
- LDA → Blei/Ng/Jordan JMLR 2003

If unsure — use `description` kind with a GitHub/docs URL.

## Anti-hallucination rules

Same as v0.9.48 — but **ONE EXCEPTION for legacy**:

**Legacy data preservation rule (v0.9.50):** if `fetch` fails and you
cannot find an alternate URL, DO NOT null out legacy `value`. Keep it,
set `algorithm_source_url=""`, note `[legacy v0.9.44, source not
re-verified: <reason>]`. Legacy value came from a PINGIS v0.9.44 source,
which is degraded but not fabricated. audit_url_log will treat empty
`algorithm_source_url` on `attribution=literature` as WARN
`[LEGACY_SOURCE]`, not ERROR.

For new fields added this task (algorithm_source_url, cadw_reproduced
entries, cadw_measured entries): strict v0.9.48 rule applies — no value
without url_log 200.

## Resume protocol

Same as v0.9.48:
1. `git status` — commit or stash
2. `cat .pingis/audits/v0.9.50_progress.json`
3. Audit existing YAML — cards with v2 schema still? (grep `benchmark_result:`)
4. Continue from first v2 card
5. Re-run gates, commit, update progress.json

## Hard gates per section

```bash
pytest -q tests/test_ml_tools_schema.py    # green
pytest -q tests/test_render_matrix.py      # green
uv run python scripts/render_matrix.py
uv run python scripts/audit_url_log.py \
  .pingis/audits/v0.9.50_url_log.jsonl docs/ml_tools/*.yaml  # 0 errors
# After migration: LEGACY warnings for migrated section must be GONE
```

## Acceptance (after all 6 sections + cleanup)

- `audit_url_log.py` → **0 errors, 0 LEGACY** (all 53 warnings closed;
  may have WARN LEGACY_SOURCE for unfetchable URLs, count them in report)
- `pytest tests/test_ml_tools_schema.py tests/test_render_matrix.py` green
- Schema v3 for all 13 sections (no `benchmark_result:` singular remains)
- **Remove backward-compat shim in schema.py**:
  - `ModelCard.benchmark_result` property (deprecated accessor)
  - `BenchmarkResult.source` / `BenchmarkResult.url` properties
  - shim branch in `_parse_benchmark_results` (only v3 path remains)
  - shim branch in `_parse_benchmark_result` (reject v2 raw)
  - Verify no test depends on the removed properties (grep).
- `.pingis/audits/v0.9.50_final_report.md`:
  - per section: cards migrated, fetched URLs, empty algorithm_source_url count
  - all cards with `[LEGACY_SOURCE]` (reason + attempted URL)
  - all cadw_reproduced entries added (model → bench json)
  - all cadw_measured entries added
  - all legacy bugs fixed (e.g. wrmssе → wrmse)

## Do NOT

- implement models in `cadw/ml/*.py`
- add deps to `pyproject.toml`
- delete legacy values if fetch fails (keep + flag, per exception rule)
- touch I/N/J/M/O/L/K YAML (v3 already, do not re-edit)
- downgrade attribution after setting
- add new cards (only migrate + enrich existing)

## Not in scope

- Phase F (Light NNs)
- Phase H (Long-tail)
- M2.2 (LOF/OCSVM)
