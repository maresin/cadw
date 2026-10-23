# §15. Time series (v0.9.24)

> v0.9.28 B.2 split — original DECISIONS.md lines 5385-5645.


### 15.1 Forecaster API

Common `ForecasterBase`: fit(df, target_col, *, time_col=None,
 group_col=None) / predict(steps) / predict_in_sample(). Predict before
 fit raises RuntimeError; steps<1 raises ValueError. This is the
 contract every forecaster satisfies; test_ts_models asserts method
 presence and arity for all classes.

### 15.2 GBDT default = direct, deferred training

GBDTForecaster default mode='direct': h independent models, one per
 horizon. Horizons are determined by predict(steps): fit(df) with no
 horizons defers training; predict(10) trains exactly 10 horizon models
 and caches them. No silent extrapolation (was: reuse last horizon model
 — removed). 'recursive' mode is available but propagates errors
 (feed-back); 'multi_output' is h-independent (CADW GBDT has no native
 multi-target) — documented.

### 15.3 SARIMA uses CSS, not MLE

SARIMAForecaster fits via conditional sum of squares / Hannan-Rissanen
(two-step AR-approx + OLS on ARMA), not full MLE. Auto-ARIMA grid is
bounded: p,q in 0..2, d in 0..1; P,D,Q in 0..1 when seasonal. AIC can
UNDER-DIFFERENCE trending series (level-stationary ARMA often wins by
AIC on a random walk). Recommendation documented: set order=(p,1,q)
explicitly for trending/unit-root data. Integration back from d,D
uses L1=last observed y; L2+ levels default to 0.0 for d>=2 (CSS limit).

### 15.4 MinT-WLS uses residual-variance weights

THF reconciliation 'optimal_wls' is proper MinT with diagonal W where
W[i] = in-sample residual variance of the level that position i belongs
to. Residuals come from each level model's predict_in_sample over rows
where the prediction is finite. A 1e-8 floor prevents zero weight on
perfect in-sample fits. If ANY level has <2 finite residual pairs, the
whole fit falls back to OLS and emits a one-time warning. This replaced
an earlier heuristic (w[i]=1/m_k) that was NOT MinT. Reconciled y =
S (S^T W^-1 S)^-1 S^T W^-1 y_all with np.linalg.pinv (rank-safe).

### 15.5 sort_values ascending-list (pre-existing bug, fixed)

df.sort_values(by=list, ascending=list) previously treated a non-empty
list ascending as truthy -> all columns ascending. Silently broke mixed-
direction sorts. Fixed in v0.9.24: ascending now accepts a per-column
sequence, length-checked against by; both the in-memory fast path and
external_sort use the resolved sequence. Not a breaking change for the
scalar case.

#### §13.4/249 — hasattr reveal ≠ rebuild (Phase C close)

*Also referenced as §13.4/279 in v0.9.25m-diagnostic commit
message (Phase C lessons are recorded in commit messages
§13.4/262–279; canonical DECISIONS.md backfill deferred to
v0.9.26 docs-debt sweep).*

**Context (v0.9.25m-diagnostic):** the "final batch" C-5 patch
(`dropna` + `fillna`) had `verify_state` abort with
`CadwDataFrame.dropna already present`. Both features shipped in
v0.9.23g (EDA-basics batch, went through 6 fix-cycles — highest
in the project). Diagnostic revealed:

- `dropna(how="any", subset=None)` — lazy notnull() filter-compose.
  API matches pandas parameter order (`how` before `subset`).
- `fillna(value, columns=None)` — streaming `pc.fill_null`.
  Scalar value only; no dict `{col: val}` variant. Dict is P2.
- **12 tests** already cover both: `test_eda_basics.py`
  (5 dropna + 7 fillna), `test_facade_transformers.py` (6 dropna),
  `test_pipeline.py` (2 dropna).

**Response:** diagnostic-only patch, 0 mutations. Do not rewrite
working hardened code to match imagined spec. Do not add P2
`fillna(dict)` to a 6-fix-cycle-hardened module for marginal
benefit — `df.fillna(0).fillna({"s": "?"})` composes today.

**§13.4/249 rule:** when `hasattr` verify (§13.4/278) reveals
pre-existing implementation:

1. **Diagnostic-only** first — `inspect.getsource` + signature +
   tests + git log + CHANGELOG. No mutation.
2. **Accept existing API as-is** — actual signature may differ from
   speculative spec; matching actual is correct.
3. **Assess genuine gaps** — P2 dict-fillna is real gap; compose
   works today.
4. **Don't touch hardened code** — 6 fix-cycles vs 0-mutation abort
   is the cost signal.
5. **Test-only extension** is safe middle path if coverage thin
   (not needed here — 12 existing tests).

**Cost model:** C-5 (v0.9.23g) took 6 fix-cycles + ongoing coverage
vs v0.9.25m verify (0 attempts, 0 mutation). The 20/20 target was
already met — v0.9.25m is a verification pass, not a feature patch.

**Phase C close (v0.9.25n):** 20/20 Pandas P0 features. 12
fix-cycle attempts across 6 batches. 1 production bug total
(C-1 `type_alias`, §13.4/262). 92% churn in test-tooling.
Defense layers that emerged: §13.4/264 (REPL-first),
§13.4/266 (post-anchor re-verify), §13.4/278 (hasattr vs
source-text), §13.4/249 (hasattr reveal ≠ rebuild).

#### §13.4/280 — file-state assertions need REPL-first too

**Context (v0.9.25n-corrected):** my first v0.9.25n attempt
aborted in `verify_state` with "expected §13.4/270+, got 248."
My assumption: "Phase C commit messages reference §13.4/262–279,
therefore DECISIONS.md is at 270+." **Wrong.** Those numbers were
only in commit message text; DECISIONS.md canonical was at 248
(v0.9.22b era). The gap 249–279 = ~30 lessons of docs debt.

**Root:** I applied §13.4/264 (REPL-first) to **library behavior**
(`pc.divide` semantics) but not to **file state**. The same
speculative-assumption anti-pattern, different domain.

**§13.4/280 rule:** file-state assertions in `verify_state` use
**actual inspection** (regex/regex-max/getsource), not inference
from commit messages, git log, or sibling files. If actual differs
from expectation → diagnostic, not assertion. Where "expected ≥ N"
is speculative, downgrade to "abort only if < lower_bound or
> upper_bound" (loose sanity, not target). Commit messages and
git log are NOT authoritative for file contents.

**Symmetric to §13.4/279:** hasattr reveal ≠ rebuild (state
revealed is what actually runs). §13.4/280: file-state reveal ≠
commit-message state (state read from file is what's actually
there). Both require **accept actual, don't infer**.

#### §13.4/281 — Phase C lessons index (docs-debt backfill)

**Context:** Phase C (v0.9.25g–l) recorded meta-lessons §13.4/250–280
in **commit messages only**, not DECISIONS.md. Users reading
DECISIONS.md saw §13.4/249 (v0.9.22 era); users reading commit
messages saw up to §13.4/280. This block consolidates the lessons
with commit references so DECISIONS.md is self-contained.
Individual lesson sub-blocks not separately appended — this index
is the canonical record.

| # | Lesson (1-line) | Commit | Status |
|---|---|---|---|
| 262 | PyArrow dtype string repr ≠ user-facing name → `pa.type_for_alias` first, `str(t)` fallback | v0.9.25g-fix1 | applied |
| 263 | Foreign dtype vocabulary (pandas `string[pyarrow]`) → reject with clear ValueError | v0.9.25g-fix2 | applied |
| 264 | REPL-first: 5-second `python -c` behavior check BEFORE writing test assumptions | v0.9.25g-fix3 | applied |
| 265 | Inline `wait, X` comment = signal to STOP and delete nearby assert (stale copy-paste) | v0.9.25h-fix1 | applied |
| 266 | Anchor fix (apply_or_fail miss) → re-verify each feature's assumptions, not just anchor | v0.9.25i-corrected | applied |
| 277 | Accessor `__init__` gate fires before method body — match actual raising layer | v0.9.25l-fix1 | applied |
| 278 | `hasattr(imported_class, 'method')` beats source-text markers for presence checks | v0.9.25l-fix1-corrected | applied |
| 279 | `hasattr` reveal ≠ rebuild — diagnostic-only response when feature already shipped | v0.9.25m-diagnostic | applied |
| 280 | File-state assertions: actual inspection, NOT inference from commit messages / git log | v0.9.25n-corrected | applied |
| 281 | Commit-subject checks: exact version token, NOT substring of `--oneline` line | v0.9.25o-fix1 | applied |

**Meta-lesson numbers 250–261 and 267–276:** not referenced in
Phase C commit messages (may have been reserved for other lessons
or skipped). If later lessons surface referencing those numbers,
cross-reference this index.

**Unified family:** §13.4/278, 280, 281 are one pattern —
"string match as proxy for state → format-fragile". Prefer
structured extraction (AST, `hasattr`, exact-token equality,
regex-with-anchors) over substring `in`. Each was learned
iteratively on a distinct miss; consolidated here to close the
docs-debt.

**Debt closure:** CHANGELOG v0.9.25 "docs debt" note (deferred to
v0.9.26) is resolved by this block. Future Phase D+ lessons append
sequentially from §13.4/282 onward.

#### §13.4/290 — Phase A closeout meta-lessons index (docs-debt backfill)

**Context:** Phase A closeout (v0.9.26b-d) surfaced 8 new meta-lessons
§13.4/282-289 in commit messages (some also in CHANGELOG v0.9.26).
This block consolidates them into a single DECISIONS.md entry,
mirroring how §13.4/281 indexed Phase C lessons §13.4/262-280.
Same docs-debt pattern; closes for this batch.

| # | Lesson (1-line) | Commit | Status |
|---|---|---|---|
| 282 | Phase completion checks iterate ALL items, not `patches` list (many-to-one) | v0.9.26b | applied |
| 283 | Roadmap item with specific count + absent audit -> cannot close by inference; defer to later phase | v0.9.26d | applied |
| 284 | Roadmap item names may be mechanistically misleading (`to_pylist` vs `check_batch` kwarg) - verify mechanism, not name | v0.9.26d | applied |
| 285 | Post-mutation validators must match file type: `json.loads` for .json, `ast.parse` only for .py; `check_no_invalid_escape` on Markdown fails on em-dash | v0.9.26d | applied |
| 286 | New state class requires BOTH whitelist extension AND field-consistency assertion (`deferred` -> `deferred_to` present) | v0.9.26d-fix2 | applied |
| 287 | Test the script's own compilation (`ast.parse(open(f).read())`) BEFORE `uv run python` - catches Unicode/quote mangling pre-run | v0.9.26d-fix2-recovery | applied |
| 288 | Discovery before patch: `str.find()` invariant marker, extract actual indent/context, print, then patch. Never hard-code surrounding whitespace | v0.9.26d-fix2-recovery2 | applied |
| 289 | State mutation -> ALL downstream consumers stale, not just the obvious test (test_collect_project_modes vs test_roadmap); invariant lives in consumer | v0.9.26d-fix3 | applied |

**Unified family:** §13.4/278, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289 are one pattern - "assume a specific representation/summary matches semantics." New lessons continue from §13.4/291 onward.

**Docs debt status:** Phase C lessons 262-280 indexed in §13.4/281 (v0.9.26a). Phase A closeout lessons 282-289 indexed here (§13.4/290, v0.9.26e). Baseline is now current through v0.9.26e.

#### §13.4/337 — Phase B.1 lessons index (docs-debt backfill)

**Context:** B.1 `dataframe.py` split (20+ patches, v0.9.26g-y2)
recorded 40 meta-lessons §13.4/296-336 in **commit messages only**.
Same docs-debt pattern as Phase A closeout (§13.4/290) and Phase C
lessons (§13.4/281). This block consolidates them into one
DECISIONS entry. Individual sub-blocks not separately appended —
this index is the canonical record.

**Note on numbering:** §13.4/291-295 are unused (reserved/skipped).
Phase B lessons continue from §13.4/296 (as referenced in commit
messages). This index sits at §13.4/337 (past all Phase B
lesson refs), mirroring Phase A closeout pattern (§13.4/290 = past
§13.4/282-289 refs).

| # | Lesson (1-line) | Patch | Status |
|---|---|---|---|
| 296 | Mixin extraction scope; `inspect.getsource(pkg)` reads `__init__.py`, not `_core.py` | v0.9.26g | applied |
| 297 | Metadata composite fields (`"path1, path2"`) require substring, not value replacement | v0.9.26g | applied |
| 298 | Same as 297 (dup tracking) | v0.9.26g | applied |
| 299 | Lazy import for circular `_core<->_exports` (mixin needs helper from parent) | v0.9.26h | applied |
| 300 | Off-by-one indent check (`< len(indent)+1`); fixed to `< len(indent)` | v0.9.26h2 | applied |
| 301 | Guard (`assert X not in src`) vs assertion (`assert X in src`) — opposite semantics under glob | v0.9.26h2 | applied |
| 302 | Already-extracted helpers in `dataframe_helpers.py`; discovery must not fail | v0.9.26i-corrected | applied |
| 303 | `inspect.getsource` returns indented source; `ast.parse` needs `textwrap.dedent` | v0.9.26i-fix1 | applied |
| 304 | Silent no-op exception (`except SyntaxError: return src`) hides failures | v0.9.26i-fix1 | applied |
| 305 | Multi-line signatures: insert point = sig-end, not `def+1` | v0.9.26i-fix2 | applied |
| 306 | Bootstrap lockout: recovery script must not `import cadw` on module level | v0.9.26i-fix3 | applied |
| 307 | Extraction misses module-level helper deps (bare `_helper()` calls) | v0.9.26i-fix4 | applied |
| 308 | `method_locals` needs nested `FunctionDef/ClassDef` names + nested args | v0.9.26i-fix4b | applied |
| 309 | Discovery: no parent-scope subtraction (imports are scope-local) | v0.9.26j-corrected | applied |
| 310 | `ImportFrom.asname` preserved in import map (`import X as Y`) | v0.9.26j-corrected | applied |
| 311 | Extraction scope must include whitelist-grep (all candidates) | v0.9.26j-fix1 | applied |
| 312 | Function-name typo → NameError before mutation; pre-execute grep check | v0.9.26l-fix1 | applied |
| 313 | Sanity: `hasattr` only, no API guesses (`df['x'].unique` wrong) | v0.9.26l-fix2 | applied |
| 314 | `TARGET_SYMBOLS` from diagnostic blueprint, not from Pandas API | v0.9.26m-fix1 | applied |
| 315 | `method_locals` must catch all binding constructs (nested args, `ExceptHandler.name`) | v0.9.26m-fix1 | applied |
| 316 | `python -c` with `for`/`try` needs newlines, not `;` | v0.9.26m-fix1 | applied |
| 317 | Sanity runtime API probe = false abort; use tests as authoritative | v0.9.26m-fix2 | applied |
| 318 | Whitelist keys class-qualified (`{file}:{Class}.{method}`) | v0.9.26m-fix3 | applied |
| 319 | Relative imports: `ast.ImportFrom.level` = dots count; emit `from {'.'*level}{module}` | v0.9.26n-fix1 | applied |
| 320 | `ImportFrom.asname` in emission (`from X import Y as Z`) | v0.9.26n-fix1 | applied |
| 321 | Repeated bug: sanity `for` pattern from broken source (fixed source copied from failed) | v0.9.26n-fix1 | applied |
| 322 | Extended lazy injection: handle all lazy names per method, not just `CadwDataFrame` | v0.9.26r | applied |
| 323 | Monkeypatch target follows symbol after extraction (`_core` → `_join`) | v0.9.26u-fix1 | applied |
| 324 | Nested whitelist keys need regex lookahead, not plain replace | v0.9.26v | applied |
| 325 | Shared helper extraction needs `_core.py` re-export for lazy imports | v0.9.26w | applied |
| 326 | Helpers > 100 LOC → extract; < 100 LOC → leave with re-export | v0.9.26w | applied |
| 327 | Test-fix diagnostic prints actual matched pattern, not guessed | v0.9.26w-fix1 | applied |
| 328 | Import cleanup = last step after all extractions | v0.9.26y | applied |
| 329 | Append-to-existing-mixin: idempotency via class body names + import dedupe | v0.9.26x2 | applied |
| 330 | Re-export reorganization: `__init__` imports from actual source module | v0.9.26y | applied |
| 331 | Auto-generated `__init__.py` needs full rewrite post-split, not edit | v0.9.26y-fix1 | applied |
| 332 | Sanity distinguishes `CadwDataFrame` methods vs `_ColumnProxy` (proxy accessor) methods | v0.9.26y-fix2 | applied |
| 333 | Sanity method list from blueprint, not Pandas API (`tail` never existed) | v0.9.26y-fix3 | applied |
| 334 | Backlog features in dedicated feature patch, not refactor patch | v0.9.26y2 | applied |
| 335 | `sys.path.insert(0, str(ROOT))` after `os.chdir(ROOT)` — mandatory boilerplate | v0.9.26y2-fix1 | applied |
| 336 | New methods in extraction-target modules: guard-compliant by construction (mirror existing pattern) | v0.9.26y2-fix1 | applied |

**Unified pattern** (building on §13.4/278-290): *"assume a summary
representation matches semantics."* Every lesson is a specialization
of "the string/count/scope I assumed is not the actual runtime/scope
semantics." Phase B extraction at scale (20 patches) forced
incremental learning of: AST-vs-string, scope-vs-parent, whitelist
patterns, import forms, monkeypatch targets, `python -c` syntax
boundaries, guard invariants.

**Docs debt status:** §13.4/291-295 unused; §13.4/296-336 referenced
from Phase B commits; this index = §13.4/337. Future Phase D+
lessons continue from §13.4/338.
