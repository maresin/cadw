# CADW Roadmap

**Document version:** 1.0
**Last updated:** 2026-10-22
**Current CADW version:** v0.9.25c
**Phase:** A (Stabilization) — in progress
**Source consolidation:** AUDIT_FULL_v0922 §7, part1_report.md,
part2_taxonomy.md, part2_todo_roadmap.md, TS review, prior sessions.

---

## §0 Conventions

**Phase** — a set of interdependent items, closed by a small number of
patches (vX.Y.Za, vX.Y.Zb, ...). Phases run A → B → C → ... in order.
Each phase has a status: `planned`, `in_progress`, `done`.

**Priority markers** (within a phase):
- `P0` — blocks whole task types or has critical bugs. Must ship.
- `P1` — strong perf / quality win, does not block.
- `P2` — extension, ship on demand.

**Complexity:**
- `S` — < 100 LOC, single file
- `M` — 100–300 LOC, may touch 2–3 files
- `L` — > 300 LOC or cross-module refactor

**Patch id** — `vX.Y.Z{a..f}` — sub-patch within a phase.

**Item id** — `{PHASE}.{N}` (e.g. `A.1`, `D.3.2`). Referenced from
`docs/roadmap.json`.

**Reference sources:**
- `AUDIT_FULL_v0922.md` — round-4 audit (34 items, 7.A/7.B/7.C classes)
- `part1_report.md` — RAM / efficiency audit (B1-B8, E1-E10, A1-A10)
- `part2_taxonomy.md` — 112-row DS task taxonomy × CADW coverage
- `part2_todo_roadmap.md` — N1-N40 (NEEDS) + T1-T12 (TODO plans)

---

## §1 Principles (hard filters for new items)

1. **Universality via recipes, not model zoo.** A task type is covered
   if there is ONE primitive + ONE recipe. Not "30 anomaly detectors"
   — one IForest + composition recipe.
2. **Every algorithm is either state O(1)/O(d²), or a sample+stream
   wrapper.** Full O(N²) / deep-learning implementations stay outside
   core, reachable via `to_batches(format='torch')`.
3. **New model only if it covers a task class unreachable by
   composition.** E.g. `CoxPH` = Reduce + Sort → ship it. `DeepSurv`
   — only via `to_batches(format='torch')` + recipe.
4. **Light NNs (≤5M params) — only for ensemble diversification** where
   GBDT systematically loses (TS classification, sequence recsys).
   Training on batches, outside core.
5. **Infrastructure refactor ≤ features.** `dataframe.py` split and
   `DECISIONS.md` split are blockers for context budget, not purity.

---

## §2 Phase map

| Phase | Version range | Items | Patches | Depends on |
|---|---|---|---|---|
| A Stabilization | v0.9.25 | 10 | 1+a | — |
| B Infrastructure | v0.9.26–v0.9.27 | 8 | 2–3 | A |
| C Pandas P0 + ML essentials | v0.9.28–v0.9.30 | 7 | 3 | B |
| D Algorithm coverage P1 | v0.9.31–v0.9.36 | 24 | 6–8 | C |
| E Recipes (docs) | parallel with D | 12 docs | 1–2 | C |
| F Light NNs | v0.9.37 | 3 | 1 | D |
| G Tech debt close | v0.9.38 | 8 | 2 | B, C |
| H P2 / long-tail | v0.9.39+ | 15+ | ongoing | G |

---

## §3 Phase A — Stabilization (v0.9.25)

**Goal:** fix critical bugs found by audits; add no new functionality.

### A.1 SARIMA integration order (D→d) — `v0.9.25a` — **done**
Reverse order: `_prep` does d then D; integration must invert to D then d.
`seasonal_period_last` from `_difference(y_centered, d)`, not `y_raw`.

### A.2 SARIMA intercept (Hannan-Rissanen) — `v0.9.25a` — **done**
Center y before fit, +y_mean after integrate. ARIMA(0,0,0) on mean=100
no longer predicts zeros.

### A.3 SARIMA `predict_in_sample` contract — `v0.9.25a` — **done**
len=n_train, y-scale for d=0; documented all-NaN for d>0 (THF falls
back to OLS, already handled).

### A.3b `_fit_one` sigma2==0 — `v0.9.25a-fix1` — **done**
Perfect linear training (d=1, D=1 → twice-differenced zeros) is valid
perfect fit. Reject only sigma2<0. AIC uses log(max(sigma2, 1e-300)).

### A.4 B1 refill linear — `v0.9.25c` — **done**
`ops/sort.py:_RunStream.refill` quadratic concat → linear list-append.
12.9 GB copied @ rbs=200k → 0.13 GB.

### A.5 B2 encode_categorical guard — `v0.9.25c` — **done**
K_est by sample + `check_state(K_est × (value_bytes + 50))` before Pass1.
Currently O(K) Python-set is invisible to MemoryGuard.

### A.6 18 to_pylist sites without extra_factor — `v0.9.25d` — planned
Table of file:line + factor (3.5–12). AST guard.

### A.7 external_sort fan-in rbs (K=8 @ 32MB/8cols/640B) — `v0.9.25c` — **done**
24.4s @ N=500k/8 cols / 32 MB budget → ~350 ms. See AUDIT §6.6.

### A.8 Missing TS tests — `v0.9.25a` — **done**
D>0, predict_in_sample, external_sort ascending-list, top-down
`agg='mean'` — all added by v0.9.25a.

### A.9 B5 dead code `_append_to_builders` — `v0.9.25e` — planned
ops/sort.py:314. Exported but never called in merge.

### A.10 B4 `memory_guard` fallback factor — `v0.9.25e` — planned
core/memory_guard.py:696. `len(set(col.to_pylist()))` without factor.

---

## §4 Phase B — Infrastructure (v0.9.26–v0.9.27)

**Goal:** remove context-transfer pain + test-latency pain. No
functional changes.

### B.1 `dataframe.py` split — `v0.9.26` — planned
7134 lines → 9 modules (`dataframe.py` facade + `_io.py`, `_index.py`,
`_reshape.py`, `_window.py`, `_apply.py`, `_ml.py`, `_stats.py`,
`_fe.py`). Re-export for backward compat. AST guard adapts to
facade-only.

### B.2 `DECISIONS.md` split — `v0.9.27` — planned
5380 lines → `docs/{architecture,reference,decisions,recipes}/` tree.
`DECISIONS.md` becomes index. See v0.9.25b docs/index.md scaffold.

### B.3 Tests grouping — `v0.9.27` — planned
Pytest markers: `slow`, `guards`, `core`, `ops`, `ml`, `ts`,
`pipeline`, `facade`, `viz`, `compat`. Auto-mark by path via
`pytest_collection_modifyitems`. Add `pytest-xdist`; `pytest -m "ts"`
≈ 3s, full fast ≈ 40s on 4 cores.

### B.4 `collect_project.py` phase-aware — `v0.9.25c` — planned
New modes: `--phase A`, `--module cadw/ts/models.py`,
`--doc docs/roadmap.md`. Reads `docs/roadmap.json`. Interfaces via
`ast.parse` (signatures + `__all__`).

### B.5 Folder structure — `v0.9.27` — planned
`scripts/{service,_measure,_prototypes,_tmp,archive/patches/}`.
`_FOR_PINGIS/_archive/`, `_FOR_PINGIS/_scratch/`.

### B.6 Cleanup — `v0.9.25e` — planned
`git rm --cached ts_review_snapshot.md`; `.backups/` removal; move
old `scripts/patch_v09{0,1,2}*.py` → `archive/patches/`; move
`_FOR_PINGIS/_out_*.json`, `_measure_*.py`, `_trace_*.py` → `_archive/`.

### B.7 Recipes scaffold — `v0.9.25b` — **done**
`docs/index.md` links to `docs/recipes/` (empty; Phase E fills).

### B.8 `docs/roadmap.json` manifest — `v0.9.25b` — **done**

---

## §5 Phase C — Pandas P0 + ML essentials (v0.9.28–v0.9.30)


**Status:** COMPLETED (v0.9.25, 2026-10-23)
**Goal:** close last 2 P0 pandas + 2 ML essentials from taxonomy (N1, N2).

| ID | Item | Patch | Complexity | Task types |
|---|---|---|---|---|
| C.1 | `df.assign(name, expr)` streaming | v0.9.28 | M | taxonomy row 16 |
| C.2 | `encode_frequency(col)` | v0.9.29 | S | row 24 |
| C.3 | `encode_onehot(cols, prefix)` | v0.9.29 | M | row 23 |
| C.4 | `encode_target(col, y, n_splits)` OOF | v0.9.30 | M | row 26 |
| C.5 | `sample_weight` in GBDT/Logistic/Linear | v0.9.30 | M | A1-A3, A6, G1, H1-H2, I1 |
| C.6 | `pinball loss` in GBDT | v0.9.30 | M | A7, D1-D2 |
| C.7 | `metrics(y_proba_cols=[...])` wrapper | v0.9.28 | S | metrics |

**Acceptance:** cat FE no leakage; sample_weight = sklearn-equivalent;
pinball coverage ≈ q.

---

## §6 Phase D — Algorithm coverage P1 (v0.9.31–v0.9.36)

**Goal:** cover task types currently NEEDS. All on primitives, state
O(1..model). Grouped by task type.

### D.1 Anomaly (v0.9.31)
- D.1.1 IForest on batches (N5) — P0, S, state t·O(ψ)
- D.1.2 Matrix Profile (STUMPY-like) streaming (N6) — P0, M, O(window)
- D.1.3 cost-sensitive GBDT + threshold tuning (N22) — P1, S, O(model)
- D.1.4 fairness metrics DP/EO/TPR-gap (N28) — P1, S, O(K)

### D.2 Ranking / Recsys (v0.9.32)
- D.2.1 LambdaMART lambdarank objective (N8) — P0, M, O(model)
- D.2.2 ALS (implicit) iterative Reduce (N7) — P0, M, O((U+V)·k)
- D.2.3 hybrid MF + content (N23) — P1, M, O(model)

### D.3 Survival / Causal (v0.9.33)
- D.3.1 CoxPH partial likelihood + Sort (N10) — P0, M, O(d)
- D.3.2 T/S-learner + propensity (N9) — P0, M, O(model)
- D.3.3 uplift trees (Qini) (N24) — P1, M, O(trees)

### D.4 TS extension (v0.9.34)
- D.4.1 MiniRocket streaming (N3) — P0, M, O(#kernels) fixed
- D.4.2 PELT online change-point (N4) — P0, M, O(window)
- D.4.3 ETS (Holt-Winters) (N18) — P1, M, O(k)
- D.4.4 Theta-method (N19) — P1, S, O(1)
- D.4.5 VAR(p) (N20) — P1, M, O(p·k²)
- D.4.6 calibration Platt/isotonic (N33) — P1, M, O(d) / O(N_bins)

### D.5 Classification extensions (v0.9.35)
- D.5.1 ordinal logistic (N13) — P1, M, O(d·K)
- D.5.2 classifier chains (N14) — P1, S, O(K·model)
- D.5.3 MiniBatchKMeans / GMM (N15) — P1, M, O(K·d) / O(K·d²)
- D.5.4 BIRCH CF-tree (N16) — P1, L, O(#CF-nodes)

### D.6 Features / selection / interpretability / stats (v0.9.36)
- D.6.1 mutual_info / var threshold / Boruta (N25) — P1, M, O(d)
- D.6.2 drift (KS/PSI/Chi2), A/B (t/MW/CUPED) (N26) — P1, M, O(1)
- D.6.3 permutation importance (N27) — P1, S, O(d·model)
- D.6.4 randomized SVD (N17) — P1, M, O(N·d·k)
- D.6.5 geo-aggregations h3/hexbin (N29) — P1, S, O(1)
- D.6.6 LDA (N30) — P1, L, O(K·V) for V ≤ 2e5
- D.6.7 LinUCB contextual bandit (N31) — P1, S, O(d²·arms)
- D.6.8 GMM density (N32) — P1, S, O(K·d²)
- D.6.9 TF-IDF / text hashing on disk (N11) — P0, M, O(V)

---

## §7 Phase E — Recipes (docs, parallel with D)

**Goal:** "разделение функционала по типам задач" as documentation.

`docs/recipes/` tree:
- `_template.md`
- `classification.md` (A1-A2, A4-A6)
- `regression.md` (A3, A7)
- `clustering.md` (B1-B3)
- `anomaly.md` (E1-E3)
- `timeseries.md` (D1-D5)
- `ranking.md` (G1)
- `survival.md` (I1)
- `causal.md` (H1-H2)
- `recsys.md` (F1-F2)
- `text.md` (J1-J5)
- `tabular-fe.md` (O1-O13)

Each recipe: "Available today / Roadmap / Recipe (numbered steps) /
pandas/sklearn equivalent table".

---

## §8 Phase F — Light NNs (v0.9.37, optional)

Only where GBDT loses systematically. Training on batches, outside core.

- F.1 MLP tabular (2-3 layers, ~0.05-2M)
- F.2 N-BEATS blocks for TS
- F.3 AE-reconstruction for anomaly

Not doing: NCF/SASRec, BERT fine-tune, CV — via
`to_batches(format='torch')` + recipe only.

---

## §9 Phase G — Tech debt close (v0.9.38)

- G.1 kmeans++ reservoir (+36.9% quality) — §11.6
- G.2 RobustScaler.fit single sort (10s → minutes @ d=100) — §11.6
- G.3 silent hash fallback warning — §11.6
- G.4 Arrow-native rank null path — 7.B.4
- G.5 API ergonomics: groupby.agg dict + rolling trailing — AUDIT §3.5
- G.6 `.cache('name')` + `_viz_cache` — 7.B.1, 7.B.2
- G.7 ML checkpoint — 7.B.3
- G.8 SumOp int64 overflow → decimal128 opt-in — §11.5

---

## §10 Phase H — P2 / long-tail (v0.9.39+)

- **Lazy expression** — 7.C.1, 7.C.2
- **Statistical tests** — N34 IterativeImputer, N40 t-digest
- **Long-tail algorithms** — 7.C.3 ACF/PACF, 7.C.4 STL, 7.C.5 GMM/EM,
  7.C.6 IForest, 7.C.7 GLM, N37 partial_fit, N38 Arrow-dict-IPC
- **Serialization / interop** — 7.C.9 to_csv/json, 7.C.10 pickle,
  7.C.7 polars / arrow-c-stream
- **TODO-class (sample+stream)** — T1 linkage, T2 t-SNE/UMAP,
  T3 HDBSCAN, T4 One-Class SVM, T11 causal forest, T12 RSF
- **TODO out-of-scope (recipes only)** — T5 NCF/SASRec, T6 BERT,
  T7 CV, T8 GNN, T9 RL, T10 generative

---

## §11 Cross-reference: task type → phase

| Taxonomy | Task type | Available now | Phase D | Phase H |
|---|---|---|---|---|
| A1 | binary class | GBDT, Logistic, NB | D.5.4, C.5 | — |
| A2 | multiclass | GBDT-multiclass | D.5.1 | — |
| A3 | regression | Linear/Ridge/Lasso/GBDT | C.6 | — |
| A4 | multilabel | OvR manual | D.5.4 | — |
| A5 | ordinal | — | D.5.1 | — |
| A6 | imbalanced | — | C.5, D.1.3 | — |
| A7 | quantile reg | — | C.6 | — |
| B1 | clustering | kmeans | D.5.3 | T1, T3 |
| B2 | hierarchical | — | — | T1 (recipe) |
| B3 | BIRCH | — | D.5.4 | — |
| C1 | PCA | PCA | D.6.4 | — |
| C2 | t-SNE/UMAP | — | — | T2 (recipe) |
| C3 | Autoencoder | via to_batches | — | F.3 |
| D1 | TS univariate | Naive, SARIMA, GBDT | D.4.3, D.4.4 | — |
| D2 | TS multivariable | — | D.4.5 | — |
| D3 | TS hierarchical | THF MinT-WLS | — | — |
| D4 | TS classification | — | D.4.1 | F.2 |
| D5 | change-point | — | D.4.2 | — |
| E1 | anomaly unsup | — | D.1.1 | T4 |
| E2 | anomaly sup | GBDT | D.1.3 | — |
| E3 | TS anomaly | — | D.1.2 | — |
| F1 | recsys MF | — | D.2.2 | T5 |
| F2 | hybrid recsys | — | D.2.3 | — |
| G1 | learning to rank | — | D.2.1 | — |
| H1 | causal | — | D.3.2 | T11 |
| H2 | uplift | — | D.3.3 | — |
| I1 | survival | — | D.3.1 | T12 |
| J1 | text class | — | D.6.9 | T6 |
| J4 | topic model | — | D.6.6 | — |
| K1-K4 | CV | — | — | T7 (recipe) |
| L1 | GNN | — | — | T8 (recipe) |
| M1 | RL | — | D.6.7 | T9 (recipe) |
| N1 | generative | — | D.6.8 | T10 (recipe) |
| O1-O13 | EDA/FE/validation | describe/hist/quantile/vc | D.6.1-3, C.1-4 | — |

---

## §12 Roadmap change log

- **2026-10-22 v1.0** — Initial consolidation from AUDIT_FULL_v0922 §7,
  part1/part2 audits, TS review, prior sessions. Phase A split into
  subpatches a..e. Phase map laid out.

---

*Maintained in `docs/roadmap.json` (machine-readable) and this file
(human-readable). Update both when adding/renaming items.*
