# CADW — Algorithm backlog (v0.9.33)

**Sources:** `docs/roadmap_research/part2_todo_roadmap.md` (NEEDS
N1-N40, TODO T1-T12),
`docs/roadmap_research/consolidated_roadmap.md` §6 (Phase D),
`.pingis/algorithm_notes/algorithm_notes.md`.

**Already done:** 4 (D.1.1 IForest v0.9.31, D.1.3 cost-sensitive
GBDT v0.9.32, D.6.3 perm importance v0.9.31, D.6.8 GMM density
v0.9.31).

**Pending:** 25 items. All must be implemented — see
`.pingis/tasks/TASK_v0935_algorithms.md` for exact assignment.

---

## Pending table

| ID | Priority | Complexity | Name | State | Test criterion |
|---|---|---|---|---|---|
| D.1.2 | P0 | M | Matrix Profile streaming | O(window) | finds 5 anomalies on synthetic |
| D.1.4 | P1 | S | fairness metrics DP/EO/TPR-gap | O(K) | = fairlearn on reference |
| D.2.1 | P0 | M | LambdaMART lambdarank (GBDT) | O(model) | NDCG@10 on LETOR-like |
| D.2.2 | P0 | M | ALS (implicit) iterative Reduce | O((U+V)·k) | RMSE/NDCG ~ implicit ALS |
| D.2.3 | P1 | M | hybrid MF + content | O(model) | NDCG > ALS-only |
| D.3.1 | P0 | M | CoxPH partial likelihood + Sort | O(d) | c-index ~ lifelines |
| D.3.2 | P0 | M | T/S-learner + propensity | O(model) | ATE within CI on synthetic |
| D.3.3 | P1 | M | uplift trees (Qini) | O(trees) | Qini-coef > T-learner |
| D.4.1 | P0 | M | MiniRocket streaming | O(#kernels) fixed | accuracy within 2% of reference |
| D.4.2 | P0 | M | PELT online change-point | O(window) | detects 3 points on synthetic |
| D.4.3 | P1 | M | ETS (Holt-Winters) | O(k) seasonal | MASE ~ statsmodels |
| D.4.4 | P1 | S | Theta-method | O(1) | MASE on M3-like |
| D.4.5 | P1 | M | VAR(p) | O(p·k²) | RMSE ~ statsmodels |
| D.4.6 | P1 | M | calibration Platt/isotonic | O(d) / O(N_bins) | Brier/ECE not worse |
| D.5.1 | P1 | M | ordinal logistic cumulative link | O(d·K) | MAE ~ mord OrdinalRidge |
| D.5.2 | P1 | S | classifier chains | O(K·model) | subset accuracy ≥ OvR |
| D.5.3 | P1 | M | MiniBatchKMeans / GMM (EM) | O(K·d) / O(K·d²) | inertia/likelihood ~ sklearn |
| D.5.4 | P1 | L | BIRCH CF-tree streaming | O(#CF nodes) | = sklearn Birch on sample |
| D.6.1 | P1 | M | mutual_info / variance / Boruta | O(d) | top-k = sklearn |
| D.6.2 | P1 | M | drift (KS/PSI/Chi2), A/B (t/MW/CUPED) | O(1) | p-value ~ scipy |
| D.6.4 | P1 | M | randomized SVD | O(N·d·k) | explained var ~ sklearn PCA |
| D.6.5 | P1 | S | geo-aggregations h3/hexbin | O(1) | map = h3-equivalent |
| D.6.6 | P1 | L | LDA (variational/EM) | O(K·V) | perplexity ~ gensim sample |
| D.6.7 | P1 | S | LinUCB contextual bandit | O(d²·arms) | regret ~ baseline |
| D.6.9 | P0 | M | TF-IDF on disk | O(V) | linear classifier = sklearn |

---

## Universal requirements per algorithm

**Location:**
- Model class + `fit_<name>` in `cadw/ml/<name>.py` (models) or
  `cadw/ops/<name>.py` (primitives).
- Mixin method `df.fit_<name>(...)` appended to
  `cadw/dataframe/_ml_bindings.py` (§13.4/329 append pattern).

**API contract:**
- `class <Name>Model` with `.predict(df)`, `.predict_proba(df)` if
  applicable; `.to_dict()` for serialization.
- `fit_<name>(df, X_cols, y_col, *, ..., memory_guard=None) -> <Name>Model`.
- All numeric inputs via `_stack_numeric` / `_column_to_f64`
  (batched).
- Every `check_batch` call must include `extra_factor ≥ 2.0`
  (measured for the specific op).

**RAM constraints:**
- State O(1) / O(d²) / O(model) — no materialization of full dataset.
- `batch_memory_factor()` override if per-batch peak > yielded
  `batch.nbytes` (F9 pattern, v0.9.32).
- `check_state(est_state_bytes)` preflight before any O(n) allocation.

**Tests (≥8 per algorithm):**
1. Synthetic known-output test.
2. Reference-comparison vs sklearn/statsmodels/lifelines (accuracy
   within documented tolerance).
3. Edge: empty input, 1 row, all null, all same.
4. Determinism: same seed → same result.
5. Multi-batch: chunk=1 vs chunk=1000 → same result.
6. RAM: peak RSS via `psutil` ≤ 2× `MemoryGuard.pred_total`.
7. Guard: high-cardinality / large-N → `MemoryLimitExceeded` if over
   budget (not crash).
8. Source contract: `inspect.getsource` matches design docstring
   (semantic).

**Bench (mandatory):**
- `scripts/bench_<name>.py` — N=1M synthetic; peak RSS, time vs
  reference.
- Output: `.pingis/benches/<name>.json` with raw numbers.

**Docs:**
- `.pingis/algorithm_notes/<name>.md` — design rationale, sources
  (paper + docs + benchmarks), edge cases, deferral reasons (if any
  — must be explicit time estimate).
- `docs/ML_TASKS_TAXONOMY.md` — update status from ⏳ to ✓.
- `docs/tasks/XX_<name>.md` (optional Phase E) — recipe.

**Logs (iterative models):**
- `log_fit` from `cadw.ml._logging` per iteration; gated by
  `CADW_LOG=1`.
- Format: `[fit] <name> | iter=N/M | loss=X.XXXXXX | state_bytes=B |
  rss=Y MiB`.

**Version:** each algorithm = 1 patch (or D.x.y-1 / D.x.y-2 if >500
LOC). Version bumps +0.0.1.

**No refusal:** PINGIS implements all 25 items. If a specific item
needs an external package > 10 MB (e.g. `h3`), PINGIS documents the
requirement in the patch message + proposes dependency; user
approves/rejects. But default: implement natively.

**Substitution allowed** with justification:
- If a better algorithm exists (paper + benchmark), PINGIS may
  substitute, but must:
  1. Cite source (paper + docs + benchmark).
  2. Show benchmark vs original plan.
  3. Prove RAM class equal or better.
  4. Same or better test coverage.

**Addition allowed** with justification:
- PINGIS may add algorithms not in the list if they unlock a task
  type or improve an existing one. Must be justified in patch
  message + `.pingis/experiments/<name>.md`.

---

## Out-of-scope (T5-T10)

Explicitly NOT implemented (torch-only or degenerate):

- **T5** NCF / SASRec — torch batching only (via
  `to_batches(format='torch')`).
- **T6** BERT / LLM fine-tune — torch.
- **T7** CV (vision models) — torch.
- **T8** GNN — torch (classic Reduce label-prop OK, but not GNN).
- **T9** RL (non-bandit) — torch.
- **T10** Generative (VAE/GAN/diffusion/LLM) — torch.

Recipes for out-of-scope tasks live in
`docs/roadmap_research/part2_todo_roadmap.md §3`.
