# PINGIS TASK — CADW v0.9.2 post-release audit + roadmap

**Type:** RAM-correctness audit + perf/correctness review + gap analysis
**Branch:** `pingis/v0.9.2-audit` from `master` @ `v0.9.2` tag
**Working dir:** main working directory (can edit `cadw/`, `tests/`, `scripts/`)
**Output dir:** `_FOR_PINGIS/` (gitignored; do NOT commit output)
**Maintainer:** maresin

---

## Context

CADW — out-of-core tabular framework for data larger than RAM. Three
primitives: `Reduce`, `HashShuffle`, `Sort`. **Core guarantee:**
bounded per-batch memory; op either fits or raises `MemoryLimitExceeded`
**before** any O(N) materialization.

**Version:** v0.9.2 (tagged 2026-10-20). 1516 fast + 20 slow tests, all
green on master. 3 main layers + ML + CV + scalers + pipeline + viz.

**Read before working (in this order):**
1. `DECISIONS.md` §3 (principles, especially §3.10 "check before
   materialize" and §3.11 "Arrow call memory discipline").
2. `DECISIONS.md` §9 (MemoryGuard + memory models per op).
3. `DECISIONS.md` §11.4 (must-fix history — all closed), §11.5
   (documented limitations), §11.6 (tech debt).
4. `DECISIONS.md` §13.4 (lessons, episodes 1..172 — long; skim for
   pattern classes: partial-migration, stale-test, nullability,
   test-authoring).
5. `CHANGELOG.md` top ~300 lines (recent v0.9.x changes).
6. `_FOR_PINGIS/REPORT.md` + `analysis_ram.md` (prior audit v0.8.8).

**Known-good invariants (do NOT break):**
- Public API of existing ops (`groupby`, `merge`, `sort_values`, etc.)
  backward-compatible.
- Every `MemoryLimitExceeded` is raised BEFORE any allocation > budget.
- AST guards in `tests/test_arrow_call_memory_guard.py`,
  `tests/test_materialize_guard.py`, `tests/test_no_pandas_in_ops.py`
  stay green.
- All fast + slow tests green after each change.

**Do NOT touch:**
- `CHANGELOG.md`, `DECISIONS.md` (maintainer's domain).
- `pyproject.toml`, `uv.lock` (dependency pins).
- `master` branch (commit only on `pingis/v0.9.2-audit`).
- Existing test files' semantics (you may ADD tests, not delete/replace).

---

## Priorities (you may reorder — see "Proposed plan" at end)

### K1. Known backlog — v0.9.3 hygiene (~2-3 h total)

These 4 are already identified in `DECISIONS.md §11.6` and can be
fixed without new design decisions. Small, surgical, testable.

#### K1.1 — AST whitelist by qualname, not function name

`tests/test_arrow_call_memory_guard.py` `WHITELISTED_FUNCS` matches
`node.name in WHITELISTED_FUNCS` — i.e. ALL methods called `fit` /
`transform` / `predict` anywhere in `cadw/` are whitelisted globally.
Effect: `RobustScaler.fit` (which calls `df.quantile` — high-risk,
Sort-based) is not flagged. Same for any future `*_fit` method.

**Fix:** `_collect_violations` should walk `ClassDef` nodes and build
`{ClassName}.{method_name}` qualname; whitelist by qualname
(`"Pipeline.fit": "delegates to step.fit, per-step guard"`,
`"RobustScaler.fit": "calls df.quantile; quantile self-guards"`, etc.).
Remove generic `"fit"`, `"transform"`, `"predict"` whitelist entries.

Rationale: current whitelist gives **false confidence** — it hides real
gaps. §13.4/170.

#### K1.2 — `_pandas_hash_column` silent fallback

`cadw/ops/shuffle.py::_pandas_hash_column`:
```python
try:
    ...
    return h.to_numpy(dtype=np.uint64)
except Exception:
    return None
```
On ANY exception → `None` → `_hash_column` falls through to Python loop
(~30× slower, no warning). If pandas API changes (e.g. pandas 4.x
drops `types_mapper`), perf regression silent, only caught by benchmark.

**Fix:** `logging.warning` on fallback (with exception repr);
optional `strict=True` kwarg to raise. NOT a behavior change
(exception still swallowed by default).

#### K1.3 — `_exact_actual_matches_arrow` per-probe-batch recompute

`cadw/ops/join.py::_exact_actual_matches_arrow` recomputes
`build_table.value_counts` for every probe batch in
`_process_partition_pair`. For partition-pair with P batches on probe
side, `build_table.value_counts` is computed P times.

**Fix:** cache `value_counts(build)` per partition-pair. Either:
(a) pass pre-computed `build_counts` into `_process_partition_pair`
from `_join_setup`; or (b) module-level memo on `(build_table_arrow_
pointer_id, on_cols)`. (a) is cleaner — no global state.

Perf win: ~B× on `value_counts(build)` call. Real for P large
(~10-16 typical), moderate.

#### K1.4 — `pa.scalar(pos_label, type=y.type)` type mismatch

`cadw/ops/metrics.py::_compute_roc_auc_binary`:
```python
eq = _pc.equal(y_f, pa.scalar(pos_label, type=y.type))
```
If `y.type = int64` and `pos_label = "yes"` (user error) →
`ArrowInvalid: Could not convert yes with type str: tried to convert to
int64` — cryptic error, no context.

**Fix:** upfront validation `_validate_pos_label_type(y.type, pos_label)`
— clear message "roc_auc: pos_label 'yes' incompatible with y column
type int64; expected int-compatible scalar".

---

### K2. RAM correctness audit (CORE — highest priority)

**The guarantee:** for every op, `peak_RAM(op) ≤ budget × safety` OR
raise `MemoryLimitExceeded` BEFORE any O(N) allocation. Verify that
this holds for **every** code path in `cadw/`.

Read each file in:
- `cadw/ops/` — `reduce.py`, `shuffle.py`, `sort.py`, `groupby.py`,
  `join.py`, `dedup.py`, `rank.py`, `quantile.py`, `resample.py`,
  `metrics.py`, `histogram.py`
- `cadw/ml/` — `linear.py`, `logistic.py`, `kmeans.py`, `pca.py`,
  `gbdt.py`, `naive_bayes.py`
- `cadw/scalers.py`, `cadw/cv.py`, `cadw/pipeline.py`
- `cadw/dataframe.py` (facade — accessors, per-group apply,
  pivot_table, melt, crosstab, shift/diff/rolling/resample)

For each op, look for:
- `to_pylist()` / `to_pandas()` / `as_py()` in hot path (per-row)
- `list(reader)` / `Table.from_batches(...)` on unbounded input
- `out_batches.extend(...)` / Python list accumulator that grows to N
- Python dict accumulator (`accum[key] = ...` with N keys)
- `check_state` / `check_batch` called AFTER materialize
- `RowFilter`/`SliceFilter` composition that materializes filtered view
  when caller expected streaming
- **Materialize duplicate**: `_attach_column` in `cv.py` (v0.9.2e)
  duplicates df for KFold; check if it's bounded or if a streaming
  `Dataset.add_column` alternative is needed.

For each finding:
- **P-ID**, **severity (S1** guaranteed OOM realistic N; **S2** data-
  dependent OOM; **S3** inefficiency **S4** cosmetic)
- **file:line**
- **peak formula** (`f(N, d, K, row_size, ...)`)
- **reproducer sketch** (small benchmark snippet)

**Verify claim with benchmark** — don't trust comments. Example: P-D1
(dedup) was claimed 5×, actual 3-5× for strings, 1× for int — always
measure.

**Re-verify known v0.8.8 findings** (from `_FOR_PINGIS/REPORT.md`):
- P-A1 (`_safe_materialize_partition` generator vs list)
- P-A3 (`Table.join` internal peak)
- P-D1 (dedup — now Arrow-native v0.8.11a; verify it still holds)
- P-G3 (`_groupby_with_skew` writer buffers)
- P-G5 (sample cache byte cap)

If closed properly — mark as resolved. If new issue — new P-ID.

---

### K3. Performance audit

Not OOM-class. Perf-relevant only. Look for:
- **Redundant recompute** (K1.3 is one; scan for others —
  `_exact_actual_matches_arrow` was found by code review, likely more)
- **Python loop over Arrow data** where `pc.*` exists
- **`to_pylist()` in inner loop** where can be `to_numpy()` or
  Arrow-native
- **`_coordinate_descent` in `linear.py`** — Retracted claim in
  v0.9.2e: naive O(d²) and residual O(d²) are the same asymptote
  (BLAS dot vs axpy). BUT — warm-start over alpha path (Lasso path)
  gives 10-100× real. Feature, not optimization; ask maintainer
  before implementing.
- **Hash shuffle perf** — `_pandas_hash_column` fast path vs Python
  fallback — already covered by K1.2.
- **join partition loop** — `_process_partition_pair` opens/reads/merges
  per pair; could it batch partition pairs? Measure first.

Severity: S3 (inefficiency) or S4 (cosmetic). No S1/S2 here.

---

### K4. Correctness audit

Beyond K1.4. Look for:
- **`pa.scalar` type coercions** elsewhere (`metrics.py`, `join.py`,
  `dataframe.py`) — same class as K1.4.
- **Nullability conventions** (§13.4/169) — `_attach_column` in cv.py
  fixed; other places? Search for `nullable=False`.
- **State gating** (`_needs_*` pattern from §13.4/165) — any remaining
  "gate by presence, not by semantic"? Search `if self.*_col is not
  None:` in metrics.py, pipeline.py.
- **Partial migration** (§13.4/167) — dual fields (`y_proba_col` +
  `y_proba_cols`) — is validation always on the new field? Any other
  dual fields (schema vs `_column_paths`, `columns` vs `_dataset.
  columns`, `index_values` vs `_index_levels`)?
- **Silent fallbacks** — `except Exception: pass` in hot paths
  (K1.2 is one; grep for `except Exception` in cadw/).
- **AST guard coverage** — after K1.1 fix, are there high-risk calls
  still missed? Spot-check `RobustScaler.fit` (should flag after fix,
  then whitelisted by qualname), `Pipeline.transform` (delegates;
  should be whitelisted by qualname).
- **Documentation vs actual** — §11.5 claims like "resample date64
  through parquet → date32" — verify each. §9.2.N memory formulas —
  verify at least 2 with quick benchmark.

---

### K5. Missing pandas functionality — gap analysis

User-facing features still absent (from `## Полный каталог` in prior
session). **Classify each** as HIGH/MEDIUM/LOW/REJECTED with rationale.

**HIGH candidates** (block real DS workflows — verify):
- `describe / info / dtypes / memory_usage` — Jupyter inspection
- `concat(axis=0|1)` — multi-file ingest (critical for out-of-core)
- `groupby().transform(str_agg)` — `agg + merge` composition (ready,
  ~2h)
- `groupby().nlargest / nsmallest` — Sort + head per group
  (composition, ~1h)
- `from_csv / from_parquet_partitioned` — ingest beyond parquet
- `StratifiedKFold` — for classification with rare classes

**MEDIUM candidates:**
- `fillna` / `dropna` / `isin` / `between` — trivial compositions
- `tail` / `sort_index` — trivial
- `nunique / value_counts / mode / median / cov / corr` — Reduce-based
- `pivot_table` done; `get_dummies` missing
- `cut / qcut` — binning
- `cumsum / cumprod / cummax / cummin` — expanding already covers,
  direct API nicer
- `pct_change` — shift + divide
- `duplicated` — bool mask
- `astype / select_dtypes` — astype done, select_dtypes missing
- `dt.floor/ceil/round`, `to_datetime`, `strftime` — accessors
- `str.strip/lstrip/rstrip`, `starts_with/ends_with`, `replace`,
  `slice`, `cat` — accessors

**LOW / REJECTED:**
- `get_dummies` on high-cardinality (OOM risk)
- `str.split / extract / findall` (list-type not supported — §11.5)
- `groupby().apply` already exists (v0.9.1e) — verify it works for
  common cases, or is it broken?
- `TargetEncoder` — needs OOF prototype (§13.4/123)
- `TimeSeriesFeatures` composite — big scope; features composed from
  existing primitives

For each HIGH/MEDIUM candidate: **~effort estimate** + **1-sentence
design** + **prototype-needed? (yes/no)**.

---

## Constraints (hard)

- **Do NOT** change public API of existing ops.
- **Do NOT** modify `CHANGELOG.md` or `DECISIONS.md`.
- **Do NOT** modify `pyproject.toml` or `uv.lock`.
- **Do NOT** push to remote.
- **Do NOT** commit `_FOR_PINGIS/` files.
- **Do NOT** touch `master` branch — only `pingis/v0.9.2-audit`.
- **Every commit:** fast + slow tests green.
- **Any new feature:** ≥1 fast test + (if O(N)-ish) ≥1 slow E2E test.

## Deliverables (in `_FOR_PINGIS/`)

### `REPORT.md` (executive)
- Run date, CADW version, branch, commit hash
- **Findings summary:** K new, M resolved, N unchanged
- **Priority matrix:** what to do next (from your findings), with
  estimated effort
- **Status of prior v0.8.8 findings** (P-A1, P-A3, P-D1, P-G3, P-G5)
- **Your recommendation** for v0.9.3, v0.9.4 scope

### `analysis_ram.md` (core)
- Full P-catalog: one section per op
- For each: file:line, peak formula, benchmark, verdict SAFE / UNSAFE /
  NEEDS-BATCHED / NEEDS-RESEARCH
- Cross-reference §9.2.N if formula known — verify/correct
- Re-verification table for P-A1, P-A3, P-D1, P-G3, P-G5

### `analysis_perf.md`
- Findings P#-ID, severity S3/S4
- Perf formulas where measurable (e.g., ×N where N=...)

### `analysis_correctness.md`
- Findings C#-ID
- Every `pa.scalar`, `except Exception`, `if self.*_col` gated by
  presence (not semantic)
- AST guard coverage gaps after K1.1 fix

### `gap_pandas.md`
- Table: feature | status | priority | effort | prototype-needed | notes
- Include all HIGH/MEDIUM/LOW/REJECTED from prior catalog

### `bench/` (extend v0.8.8 benchmarks)
- `bench/bench_ram_v092.py` — new findings
- `bench/RESULTS.md` — table of measurements

### `proto/` (if needed)
- Only for algorithmic prototypes (e.g., TargetEncoder OOF)
- A (naive) vs B (candidate), PASS criterion

### `patches/` (optional)
- If K1 fixes implemented: patch scripts (for maintainer to review
  before applying to master)

## Process

1. **Setup**:
   ```bash
   cd /home/maresin/Dev/sandbox/temp_cadw/Cadw_framework
   git checkout -b pingis/v0.9.2-audit master
   uv run pytest -q -m "not slow"   # confirm baseline green
   uv run pytest -q -m slow          # confirm baseline green
   ```

2. **K1 fixes** first (surgical, low risk, tests green after each).
   Commit each fix separately with clear message.

3. **K2 RAM audit** — read each op file, benchmark suspicious findings,
   write `analysis_ram.md` incrementally. Prioritize ops most used
   (groupby, join, sort, dedup, cv, scalers).

4. **K3+K4** — spot-check, write findings.

5. **K5** — gap analysis (no implementation, only classification).

6. **REPORT.md** — executive + priority matrix.

7. **Commit** each fix to `pingis/v0.9.2-audit`. Push? No — maintainer
   merges.

## Proposed plan (you may reorder)

Suggested sequence:

1. **Day 1**: Setup + K1.1-K1.4 (4 small fixes, ~2h). Report back if
   any K1 surprise.
2. **Day 2-3**: K2 RAM audit on `ops/` (core 3 primitives + join +
   groupby + dedup). 60% of time here.
3. **Day 3-4**: K2 on `ml/`, `scalers.py`, `cv.py`, `dataframe.py`.
4. **Day 4**: K3 perf spot-check + K4 correctness.
5. **Day 5**: K5 gap analysis + REPORT.md + priority matrix.

If you find something that changes priority (e.g., S1 in an op I
marked SAFE) — **bump it to top**, note in REPORT.md.

## Acceptance criteria

- [ ] `pingis/v0.9.2-audit` branch exists, all commits on it.
- [ ] Fast + slow test suite green on branch tip.
- [ ] K1.1-K1.4 fixed (or documented why not, if design decision needed).
- [ ] `REPORT.md` with executive summary + priority matrix.
- [ ] `analysis_ram.md` — every op in `cadw/ops/` + `cadw/ml/` +
      `cadw/scalers.py` + `cadw/cv.py` audited, formula + verdict.
- [ ] `analysis_perf.md` — at least 5 findings OR "no findings after
      full review" (justify).
- [ ] `analysis_correctness.md` — at least 5 findings OR "no findings
      after full review" (justify).
- [ ] `gap_pandas.md` — ≥15 missing features classified.
- [ ] No changes to `CHANGELOG.md`, `DECISIONS.md`, `master` branch.
- [ ] `_FOR_PINGIS/` output gitignored (verify with `git status`).

## Re-run criteria

Trigger new audit when:
- v0.9.3 tag released (or any new feature tag).
- New op added to `cadw/ops/` or `cadw/ml/`.
- Arrow version bumps in `pyproject.toml` (`uv.lock` regen — `pc.*`
  availability may change; check `test_api_availability.py`).
- ~2-3 months since last audit (whichever first).

## Notes for this run

- **The `_coordinate_descent` retraction**: earlier maintainer claim
  "1000× speedup possible via residual maintenance" is **WRONG** —
  naive O(d²) and residual O(d²) same asymptote. Don't pursue.
  Real win: warm-start over alpha path (feature, not optimization).
  Ask maintainer before implementing.
- **The AST guard over-broad issue** (K1.1) — fix is safe; after fix
  may surface previously-hidden violations. If so, add to whitelist
  with qualname + rationale (do NOT delete high-risk calls).
- **`_FOR_PINGIS/` is gitignored** — verify with `git status --ignored
  | grep _FOR_PINGIS`.
- **`scripts/patch_v092_release.py`** — untracked from maintainer's
  last operation; can leave it (it's already been run) or move to
  `_FOR_PINGIS/old_scripts/`.
- **`bench/`** — prior v0.8.8 benchmarks in `_FOR_PINGIS/bench/`;
  extend, don't duplicate (`bench_join.py`, `bench_suite.py`).