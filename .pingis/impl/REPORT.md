# CADW — models for remaining ML sections (I/N/J/M/O)

## 1. Goal
For the CADW project (read-only; no code changes to `cadw/` or `docs/`), select ML models for the sections that were still `described_only` after the first reconstruction round — sections **I (Survival), J (NLP), M (Contextual bandits), N (Unsupervised density), O (EDA / cleaning / FE)** — and provide reference out-of-core algorithms reusing CADW primitives under a fixed 2.8 GB RAM budget. L (Time series) and K (Recommendation) are already covered by `cadw/ts/*` and `cadw/ml/*` (torch-based models are out of scope for a bounded-RAM Python reference; see §7).

All work lives under `.pingis/impl/`; application code was not modified.

## 2. Files

```
.pingis/impl/
├── INVENTORY.md              # stage 1: gap inventory for I/N/J/M/O
├── PRIMITIVES_USED.md        # stage 2.5: cadw primitive API used by impls
├── SMOKE_RESULTS.md          # stage 5: smoke-test table (9/9 passed, peak RSS)
├── REPORT.md                 # this file
├── I/  selection.md notes.md kaplan_meier.py weibull_aft.py
├── J/  selection.md notes.md complement_nb.py nmf.py
├── M/  selection.md notes.md bandits.py
├── N/  selection.md notes.md kernel_density.py
├── O/  selection.md notes.md reservoir.py stratified_split.py one_hot_encoder.py
└── smoke/run_smoke.py
```

## 3. Models selected per section (summary)

Full sources, benchmark URLs and `verified_in_cadw` / `described_only` classification per candidate are in the corresponding `.pingis/impl/<Sec>/selection.md`. Algorithm-level specs (state class / state type / bounded-state proof / n-scaling / primitive reuse) are in `.pingis/impl/<Sec>/notes.md`.

| Section | Implemented (working .py) | Selection rationale |
|---|---|---|
| I — Survival | KaplanMeier, WeibullAFT | Cover non-parametric baseline (KM) + parametric regression (Weibull AFT). Both have closed-form Newton steps on sufficient statistics → O(d^2) state, independent of N. RandomSurvivalForest/DeepSurv/DeepHit excluded (O(N) per tree or torch). |
| J — NLP | ComplementNB, NMF | Cover classification on sparse (nnz) input + unsupervised topic extraction. BERT/SentenceTransformer/BERTopic excluded (torch); Word2Vec deferred (out-of-core SGNS is nontrivial; not needed to prove the point). |
| M — Bandits | EpsilonGreedy, UCB1, ThompsonSampling | All three are O(A) sufficient-statistic algorithms; a shared `ArmStatsOp` reducer covers them. NeuralUCB excluded (torch). LinUCB already in `cadw.ml.linucb`. |
| N — Density | KernelDensity | Selected as the only out-of-core-plausible model in this section. Center set is reservoir-sampled (K << N), state = O(K·d), score = streaming log-sum-exp. HDBSCAN excluded (MST O(n log n) is incompatible with 2.8 GB); NormalizingFlow excluded (torch). GaussianMixture / BIRCH already in `cadw.ml`. |
| O — EDA / cleaning / FE | ReservoirSample, StratifiedSplit, OneHotEncoder | Cover (a) uniform sampling, (b) cross-row shuffled splitting, (c) external-sort–based category encoding. Deferred: KNNImputer, KBinsDiscretizer, GridSearch, RandomizedSearch — see §4. Pandera / GreatExpectations / Optuna / SHAPTree / Evidently excluded (wrappers or tier3_export). |

## 4. Deferred O/* — one-line rationales

- **O2.KNNImputer** — correctness under bounded RAM requires a partitioned top-k join over complete rows. Plumbing is proved by N/kernel_density.py (partition + streaming chunk log-sumexp) and O/stratified_split.py (`hash_shuffle` + `read_partition_batches`). Not duplicated. Algorithm spec remains in `O/notes.md`.
- **O3.KBinsDiscretizer** — per-column external sort for exact quantiles is structurally identical to O/one_hot_encoder.py's per-column vocab collection; the sort+read_sorted plumbing is already proved. Spec in `O/notes.md`.
- **O7.GridSearch / O7.RandomizedSearch** — orchestration wrappers around `estimator.fit` / `estimator.predict`. No new memory model; the constraint is inherited from the underlying estimator. Specs in `O/notes.md`.

No `NotImplemented` stubs were shipped; deferred entries exist only as specs, per the “quality over coverage” decision.

## 5. What “out-of-core” means here (invariants enforced by all 9 impls)

1. **No `pa.Table`, no `read_all()`** — checked across the whole `.pingis/impl/` tree.
2. **State is bounded** — every impl exposes `estimate_state_nbytes()` / an equivalent bound and calls `MemoryGuard.check_state(...)` **before** allocating dense state (e.g. `WeibullAFT` raises `MemoryLimitExceeded` if `8·d²` > `state_limit_bytes`; `ComplementNB` raises `ValueError` if `8·C·V` exceeds it).
3. **Streaming input** — every impl takes `batch_iter_factory: Callable[[], Iterator[pa.RecordBatch]]` so multi-pass algorithms (KM sort → scan; Weibull pass-per-iteration; NMF pass-per-epoch) re-stream the source instead of buffering.
4. **Spill, don't accumulate** — when a piece of state can exceed `state_limit_bytes`, it is written to a `SpillDir` via `spill_array_writer` / read back with `spill_array_reader` (I/KM step function; N/KD centers; O/stratified_split assignment file).
5. **External primitives** used as-is, not re-implemented: `external_sort` + `read_sorted` (I/KM, O/OHE), `hash_shuffle` + `read_partition_batches` (O/stratified_split), `Reduce` / `ReduceOp` (M/bandits, I/WeibullAFT, J/ComplementNB).

## 6. Smoke tests

See `.pingis/impl/SMOKE_RESULTS.md`. Runner: `.pingis/impl/smoke/run_smoke.py`. Each impl is invoked as a `python <impl>` subprocess and its own `ru_maxrss` is read from its stdout line “peak RSS = X MiB” (Linux KiB → MiB; this is `resource.getrusage(RUSAGE_SELF).ru_maxrss`). The runner fails if any impl returns non-zero.

**Result: 9/9 passed.** Peak RSS across all impls stays in **112–274 MiB**, far below the 2.8 GB budget.

### Note on WeibullAFT smoke output
The smoke test for `I/weibull_aft.py` asserts **plumbing only** (state bounded, run completes, parameters are finite and `sigma > 0`). It does **not** assert statistical convergence — the printed `iter=1` reflects one Newton step on synthetic censored data and is not a quality claim about the algorithm. Real usage would run `max_iter=50` with a proper convergence check on the log-likelihood; that path is implemented but not exercised by the smoke test.

## 7. L / K — out of scope (rationale)

- **L (Time series)** — the section is already covered by `cadw/ts/*` (ETS, AutoETS, Theta, AutoTheta, VAR, THF, OTM, decomposition, backtest). None of the missing candidates identified in `INVENTORY.md` add a new memory-model problem: they are either (a) already implemented in `cadw/ts`, or (b) torch-only (e.g. DeepAR-style neural forecasters). No new algorithm under 2.8 GB is needed.
- **K (Recommendation)** — already covered by `cadw/ml/als.py`, `hybrid_mf.py`, `s2g.py`, `tspq.py`. The missing candidates (LightFM, neural collaborative filtering, transformer recommenders) are either thin wrappers over the existing out-of-core ALS/HybridMF paths, or torch-based (tier3_export). No reference algorithm can be written that would reuse the CADW primitives in a genuinely new way.

Both sections are therefore documented here rather than given cards / new code.

## 8. Quality over coverage — trade-off

Target for this pass was **3–4 solid out-of-core algorithms per section**, not exhaustive coverage. Delivered: I=2, J=2, M=3, N=1, O=3 working impls (11 model classes total; M/bandits.py contains 3 algorithms). Every implemented algorithm:

- imports standalone (`uv run python .pingis/impl/<Sec>/<file>.py`),
- runs end-to-end on synthetic data in <10 s,
- reports its own peak RSS,
- cites a concrete original source in `<Sec>/selection.md`,
- has a one-page spec with state-class / bounded-state proof in `<Sec>/notes.md`,
- reuses at least one CADW primitive (`Reduce`, `external_sort`, `hash_shuffle`, `SpillDir`, or `MemoryGuard.check_state`).

The 4 deferred O/* are intentional — their memory-model problems are already solved by N/KD and O/stratified_split, so implementing them again would add code without adding algorithmic content.

## 9. Process notes

- **No code changes** to `cadw/` or `docs/`; `git status` on the repo root is unchanged for those paths.
- Primitive API surface was read from source (not from docs), and every documented entry in `PRIMITIVES_USED.md` names the actual module and function. Cross-checked by the critic against source at stage 2.5.
- Stage 4/5 loop: write impl → run `python <impl>` → confirm non-zero exit code is absent and peak RSS is in budget → mark stage. All 9 impls went through this loop before STAGE 4 DONE was declared.
