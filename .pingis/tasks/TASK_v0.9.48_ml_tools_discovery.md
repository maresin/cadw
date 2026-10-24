# TASK v0.9.48 — ML-tools matrix: sections I / N / J / M / O / L / K

**Branch:** pingis/dev-audit
**Baseline:** HEAD = v0.9.47b, tests 2978 passed.
**Mode:** discovery + research (web access, fetch). **Cards only, no code.**
**Resume:** allowed (§Resume).

## Goal

Fill 7 remaining matrix sections. For each model card record:
- **multi-benchmark** results (typical 2-3, minimum 1 for niche)
- **multi-metric** per benchmark (all published leaderboard metrics)
- **per-algorithm** rows (top-3 implementations, not just winner)
- **two URLs** per result (benchmark_url + algorithm_source_url)
- **attribution** per result (literature / cadw_reproduced / cadw_measured)

## Ordering (7 checkpoints, 1 per section)

| # | Section | Subtypes | Cards (est.) | Fetches (est.) |
|---|---|---|---|---|
| 1 | **I** survival | I1 time-to-event | 5-8 | ~15 |
| 2 | **N** density | N1 unsupervised density | 4-6 | ~10 |
| 3 | **J** NLP | J1 classification, J2 embeddings, J3 topic | 8-12 | ~25 |
| 4 | **M** bandits | M1 contextual bandit | 4-6 | ~10 |
| 5 | **O** EDA / cleaning | O1-O11 | 40-50 | ~100 |
| 6 | **L** graph | L1 node embeddings | 2-4 | ~6 |
| 7 | **K** vision | K1 summary | 2-4 | ~6 |

After each section:
1. `pytest -q tests/test_ml_tools_schema.py` — green
2. `uv run python scripts/render_matrix.py`
3. `uv run python scripts/audit_url_log.py .pingis/audits/v0.9.48_url_log.jsonl docs/ml_tools/*.yaml`
   — 0 errors (LEGACY warnings OK)
4. Update `.pingis/audits/v0.9.48_progress.json`
5. Commit `v0.9.48.<N>: <section> complete`

## Per-card recipe

1. **Subtype** — from `docs/roadmap_research/part2_taxonomy.md`.
2. **Models per subtype**:
   - **baseline** (>= 1) — canonical method (CoxPH, GMM, TF-IDF, LinUCB, describe, ...)
   - **sota** (>= 1, ideally 2-3) — top-3 from `paperswithcode.com/sota/<task>`
   - if a model already exists in `cadw/ml/` → `verified_in_cadw: true`,
     `cadw_module`, `cadw_function`
3. **Per benchmark_result** (list):
   - `benchmark` — registry name from `benchmarks.yaml`
   - `benchmark_url` — leaderboard URL (fetched + quoted) OR `.pingis/...` for cadw_*
   - `algorithm` — WHO: "LightGBM", "sklearn.IsolationForest", "cadw.ml.gbdt.fit_gbdt_regressor"
   - `algorithm_source_url` — paper / github / docs
   - `algorithm_source_kind` — paper | code | docs | pseudocode | description
   - `attribution` — literature | cadw_reproduced | cadw_measured
   - `reproduces` — iff cadw_reproduced: literature algorithm name
   - `task_type` — "binary classification", "TS anomaly", ...
   - `n_datasets` — canonical count for (benchmark, task_type)
   - `metric`, `value`, `std`, `note` (fetch date + quote)
4. **Roles** — from `docs/ml_tools/enums.py` (7 total).
5. **purpose** — 2-3 sentences, must state **inductive bias**.
6. **real_life_examples** — 3 concrete use cases.
7. **cadw_module / cadw_function** — exact (verify with `grep "def fit_X"`).

## URL log (mandatory)

Every fetch appends to `.pingis/audits/v0.9.48_url_log.jsonl`:

```jsonl
{"ts": "2026-10-24T10:15:33Z", "url": "https://paperswithcode.com/sota/...",
 "status": 200, "bytes": 48213, "for_card": "A1.GBDT",
 "algorithm": "LightGBM", "metric": "roc_auc", "value": 0.842,
 "quote": "LightGBM 0.842 ± 0.048 on 15 binary tasks"}
{"ts": "...", "url": "...", "status": 403, "for_card": "I1.DeepSurv",
 "note": "paywall"}
```

## Anti-hallucination rules

### Value discipline
- **fetched + quoted** → `value: X.XX`, `note: "fetched <date>; quoted: '...'"`
- **fetched, number not extractable** (JS render, paywall) →
  `value: null`, `note: "fetch_ok_no_number: <url>"`
- **fetch failed** → `value: null`, `note: "fetch_failed: <url> status=<code>"`
- **no public leaderboard** for this subtype → `benchmark: cadw_internal` +
  `attribution: cadw_measured` if CADW verifies; otherwise `value: null,
  note: "no_public_benchmark"`
- **NEVER** write `value: X.XX` without a corresponding `url_log.jsonl`
  entry with `status=200`, matching `for_card`, `metric`, `value`.
  Even if "you know it from training data" — fetch it. If not fetchable,
  `value: null`.

### Multi-benchmark / multi-metric / multi-algorithm

- **Mainstream subtype** → 2-3 benchmarks, >= 1 sota algorithm per benchmark.
- **Niche subtype** → 1 benchmark acceptable; note reason in section
  `benchmarks[].note` ("only accepted benchmark", "no public leaderboard").
- **Per-algorithm rows**: for the SOTA benchmark, record **top-3** implementations
  (e.g. LightGBM, XGBoost, CatBoost) as separate literature entries. This
  shows CADW's relative position; do not only record the winner.
- **Per-dataset**: `n_datasets` + `std` from leaderboard if published.
  Per-dataset stories (highest/lowest) go in `note` prose, not structured.

### SOTA age
- top-3 leaderboard entries on fetch date
- if all top-3 papers older than 2022 → `note: "leaderboard_oldest_<year>"`,
  human review

### K / L scope
- K (vision) and L (graph) — torch-only, **out of scope**
- 2-4 cards per section, `tier3_export`, `not_implemented_in_cadw: true`
- `note: "out of scope: torch required"`

### cadw_internal benchmark
For models whose only verified number is CADW's own measurement
(memory, runtime, synthetic DGP). `benchmark: cadw_internal`,
`benchmark_url: ".pingis/benches/<name>.json"` or `"tests/test_<name>.py"`,
`attribution: cadw_measured`. NOT comparable to external leaderboards.

## Resume protocol

If context exhausted mid-section:
1. `git status` — uncommitted changes? If yes, either commit or `git stash`.
2. Read `.pingis/audits/v0.9.48_progress.json`:
   ```json
   {"last_completed": "I", "in_progress": "N",
    "completed_cards": ["I1.CoxPH", "I1.KaplanMeier", ...]}
   ```
3. Audit existing YAML for `in_progress` section:
   - cards with `value: null` and reason?
   - cards with fetched but unreviewed quotes?
4. Continue from first incomplete card, do NOT redo completed.
5. Re-run gates (schema + render + audit_url_log).
6. Commit + update progress.json.

## Hard gates (per section commit)

```bash
pytest -q tests/test_ml_tools_schema.py    # green
pytest -q tests/test_render_matrix.py      # green
uv run python scripts/render_matrix.py
uv run python scripts/audit_url_log.py \
  .pingis/audits/v0.9.48_url_log.jsonl docs/ml_tools/*.yaml  # 0 errors
```

## Acceptance gates (after all 7 sections)

- `pytest -q tests/test_ml_tools_schema.py` → 20+ passed
- `pytest -q tests/test_render_matrix.py` → 20+ passed
- `render_matrix.py` → 13 sections, ~80+ tasks under `docs/tasks/`
- `audit_url_log.py` → 0 errors (LEGACY warnings OK)
- `.pingis/audits/v0.9.48_final_report.md`:
  - per section: subtypes, cards, benchmarks, % with fetched value
  - list of ALL `value: null` with reason
  - list of ALL SOTA cards with fetched quote + paper link
  - list of ALL `cadw_measured` entries (these need M2 reproduction later)

## Do NOT

- implement models in `cadw/ml/*.py`
- add deps to `pyproject.toml`
- touch A/D/E/F/G/H YAML (accepted; may add `cadw_reproduced` entries
  only if `verified_in_cadw: true` and M2 confirms)
- update `docs/ML_TASKS_TAXONOMY.md` (deprecated)
- write `value: X.XX` without url_log entry
- downgrade `attribution` after setting it

## Not in scope

- M2.2 (LOF/OCSVM) — Phase H native
- Phase F (Light NNs) — separate task
- Reproduction for new models — separate task (needs reference)
