# Glossary — canonical terms

**Purpose:** one canonical definition per term. Every doc and every
patch uses these terms with these meanings. Synonyms are forbidden.

**Hierarchy:** Tier 4 (reference). Complements
`AI_CONSTITUTION.md` (Tier 1), `STABILITY_PROTOCOL.md` (Tier 2),
`ALGORITHM_TEMPLATE.md` (Tier 3). Does NOT override any rule.
On conflict — higher tier wins.

---

## Status and lifecycle

**`status`** — YAML field under `implementation_status`. One of
`draft`, `verified`, `tier1_core_exception`. Orthogonal to
`verified_in_cadw` (algorithmic fidelity to source).

**`draft`** — function is in development or has known gap. Formula
may not exist, stability not confirmed, or backlog open. Re-opening
expected. Every `draft` card has `# draft_reason:` OR entry in
`ROADMAP` with target version.

**`verified`** — stability confirmed (§STABILITY §2), formula
derived from code (§3), matrix measurement passed 9/9 cells (§4).
Re-opening is a red flag (means earlier closure was dishonest).

**`tier1_core_exception`** — TEMPORARY status value (v0.9.102).
Function is `tier_level: tier1_core`, OOC on all axes except n, and n
is bounded by problem domain (single-series TS, n≤10⁶ typical). Tested
bound documented in card. Replaced in v0.9.103 by `bounded_by_domain`
field + `status: verified`. See AI_CONSTITUTION §5.2, DEC-011.

**`bounded_by_domain`** (planned v0.9.103) — YAML field, structured:
`{axis: n, bound: 10^6, reason: "..."}`. Orthogonal to `status`; a
`verified` card may also be `bounded_by_domain`. Replaces
`tier1_core_exception` status.

**`verified_in_cadw`** — boolean. Algorithm implementation matches
source/paper (fidelity). Independent of `status`. A `draft` card can
have `verified_in_cadw: true` (algorithm right, formula missing).

---

## Stability

**`stable`** — all three: (a) OOC confirmed via n-sweep
(`peak(10⁵)/peak(10³) < 2.0`), (b) tight-guard raises
`MemoryLimitExceeded` (spill check), (c) no `# BACKLOG:` marker
anywhere mentioning function.

**`OOC`** — out-of-core. RAM independent of at least one axis (n, d,
K, n_groups). RAM = `O(batch) + O(state)`, where state bounded by
user params (d, k, w, n_estimators). Not "all axes OOC" — always
say which axis.

**`spill`** — write state/buffer to disk (Arrow IPC or Parquet) when
budget exceeded. `MemoryGuard`-driven. Never silent OOM; always
`MemoryLimitExceeded` raise before.

**`refuse_at_guard`** — documented, tested behavior where algorithm
raises `MemoryLimitExceeded` at a specific n threshold (no spill).
Legacy; removed from status enum v0.9.102 (was `n_scalability: refuse_at_guard`).

---

## Formula and measurement

**`ram_formula`** — YAML field. String expression: sum of memory
terms. Grammar: `+ - * / **`, identifiers (params and typed
constants). No `=` sign. Example:
`"2 * d * d * f64 + 3 * batch * row_size + batch * d * f64"`.

**`formula_evidence`** — YAML field. Dict mapping term name to code
reference: `"<file.py>:<line> — <what alloc>"`. Enables verification:
`sed -n '<line>p' <file>` should show the alloc.

**`typed constants`** — named scalars from `memory_constants.yaml`.
Examples: `f64=8`, `f32=4`, `i32=4`, `i8=1`, `bool=1`,
`dict_entry_bytes=116`, `py_str_overhead_bytes=49`. Version-stamped
with Python/numpy/arrow versions.

**`state`** — memory that persists across batches during fit. Sum of
state terms in formula. Typically `O(d²)`, `O(k·d)`, `O(w)`.

**`transient`** — memory that exists only during batch processing.
Typical: input batch, working arrays. NOT a separate formula
component — it's just the `batch * row_size * N` terms from
`mg.check_batch` and `np.empty((batch, ...))`.

**`bottleneck`** — formula term with largest contribution at given
params. Reported in `estimate_memory` output. Users use this to
understand what to reduce (d? k? batch?).

**`measurement`** — YAML field. Dict: `{matrix, cells_passed,
max_ratio_deviation, date, env}`. Records what measurement proved
formula right.

**`matrix`** — 3×3 (or larger) grid of measurement cells: `n ∈ {10³,
10⁴, 10⁵}`, `d ∈ {10, 100, 1000}`. Each cell asserts
`0.7 ≤ predicted/measured ≤ 1.4`.

**`measure_peak(func, *args)`** — dev tool. Runs func, returns
`{py_bytes, arrow_bytes, rss_bytes, total_bytes}`. Not for runtime
(1-10s overhead). CI-only + calibration.

**`calibration`** — isolated activity: derive formula from code,
verify with measurement. Fixed batch, explicit n/d/k. No
MemoryGuard.

**`runtime`** — activity of `estimate_memory` / `solve_batch`: given
guard budget + params, return predicted peak bytes OR chosen batch
size. Uses formula but inverted (for solve).

**`expansion_factor`** — legacy heuristic in `MemoryGuard.get_batch_rows`.
Value ≈ 2.5. Deprecated v0.9.103+ (replaced by `solve_batch`).

**`solve_batch(op, budget, **params)`** — runtime function (v0.9.103+).
Inverse of `evaluate_formula`: given budget, return largest batch
such that formula(batch, ...) ≤ budget.

---

## Data and types

**`batch`** — input chunk of rows. Size chosen by `MemoryGuard` at
runtime; fixed `batch=1000` at calibration (experimental control).

**`row_size`** — bytes per input row, from `MemoryGuard.row_size_bytes`
(computed from schema + sample). Mixed-type schema = sum of per-column
type_size.

**`n`** — total dataset rows (or rows in a partition, per context).

**`d`** — number of features.

**`k`** — number of classes / clusters / groups (per context).

**`w`** — window size (for windowed ops: rolling, sliding PCA, matrix
profile).

**`n_groups`** — number of distinct group keys (for groupby, ts
per-entity).

**`n_estimators`** — number of base learners (for GBDT, random forest,
chains).

---

## Quality gates

**`G1-G11`** — `AI_CONSTITUTION` §4. AST-scan gates, run in fast suite:

- G1: no `np.zeros((n,))` unguarded
- G2: `check_state` BEFORE alloc (not after)
- G3: no `sliding_window_view(...).copy()`
- G4: YAML schema completeness (`bounded_state: false` + `tier1_core`
  not allowed except documented)
- G5: no `np.zeros((hyperparam,))` (bounded state for hyperparams)
- G6: no `list.append` + `np.vstack`/`concat` in batch loop
- G7-G10: dual-branch self-tests, spill_ratio, formula accuracy
- G11: transitive-migration (caller uses CadwDataFrame API, not
  legacy pa.Table)

**`AP<N>`** — anti-pattern, numbered. Current: AP1–AP23. See
`AI_CONSTITUTION` §5.

---

## Primitives and patterns

**`Reduce`** — primitive for aggregation. `cadw/ops/reduce.py`.
State `O(d)`. Examples: sum, mean, XTX, covariance.

**`HashShuffle`** — primitive for partitioning. `cadw/ops/shuffle.py`.
State `O(n / P)`. Examples: groupby, join build side, dedup.

**`Sort`** — primitive for global order. `cadw/ops/sort.py`. State
`O(k · rbs)`. Examples: global median, top-k, k-fold, rank.

**`SpillBuffer`** — primitive for `pa.Table` output. `cadw/core/spill.py`.
RAM `O(batch)`, disk `O(n)`. Returns `AutoResult` (memory or disk).

**`SpillArray`** — primitive for numpy output. `cadw/core/spill.py`.
RAM `O(batch × K)`, disk `O(n × K)`. Returns `AutoArrayResult`.

**`chunked strided`** — pattern for windowed ops. `sliding_window_view`
(0 bytes) + chunked iteration (`np.ascontiguousarray` on chunk, not
on full view). RAM `O(chunk × w)`. Reference: `test_hampel.py`.

**`guard call`** — a call to `mg.check_state(bytes, name=...)` or
`mg.check_batch(b, name=..., extra_factor=...)` in code. **Every
guard call is a formula term.**

---

## Process

**`probe`** — read-only phase of session. Only commands, no edits.
Output pasted back.

**`patch`** — write phase. Script to `/tmp/patch_NNN.py`, run, verify
in same block.

**`review`** — human reads `git diff --stat` + sample of changed file.
Approves or corrects.

**`commit`** — only after explicit "go". `git add`, `commit`,
`tag`.

**`session brief`** — markdown doc at session start (`docs/SESSION_BRIEF_vX.Y.Z.md`).
States goals, scope, anti-scope. Must not contradict
`AI_CONSTITUTION`. If conflict, constitution wins; ask user.
