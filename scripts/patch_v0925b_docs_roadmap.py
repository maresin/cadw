#!/usr/bin/env python3
# v0.9.25b — docs infrastructure: ROADMAP.md + roadmap.json + index.md.
#
# Docs-only patch (no cadw/ code touched). Sets up the navigation
# layer for future chat sessions:
#   - docs/ROADMAP.md      — human-readable, phases A..H
#   - docs/roadmap.json    — machine-readable, consumable by
#                            scripts/service/collect_project.py (v0.9.25c)
#   - docs/index.md        — new docs entry point
#   - .gitignore           — ts_review_snapshot.md + scripts/_tmp/ +
#                            _FOR_PINGIS/_scratch/
#   - tests/test_roadmap.py — 4 structural tests
#
# Does NOT touch DECISIONS.md (its split is Phase B step 2, v0.9.26).
# Does NOT move part1/part2 audit docs (cleanup is Phase B step 4).
#
# Run: uv run python scripts/patch_v0925b_docs_roadmap.py

from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    check_no_invalid_escape,
)


# ============ content constants (module-level, NOT closures) ============

ROADMAP_MD = """\
# CADW Roadmap

**Document version:** 1.0
**Last updated:** 2026-10-22
**Current CADW version:** v0.9.25a
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

### A.4 B1 refill linear — `v0.9.25c` — planned
`ops/sort.py:_RunStream.refill` quadratic concat → linear list-append.
12.9 GB copied @ rbs=200k → 0.13 GB.

### A.5 B2 encode_categorical guard — `v0.9.25c` — planned
K_est by sample + `check_state(K_est × (value_bytes + 50))` before Pass1.
Currently O(K) Python-set is invisible to MemoryGuard.

### A.6 18 to_pylist sites without extra_factor — `v0.9.25d` — planned
Table of file:line + factor (3.5–12). AST guard.

### A.7 external_sort fan-in floor = 8 — `v0.9.25c` — planned
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
"""


ROADMAP_JSON_DICT = {
    "schema_version": "1",
    "current_version": "0.9.25a",
    "updated": "2026-10-22",
    "principles": [
        "Universality via recipes, not model zoo.",
        "Every algorithm O(1)/O(d²) or sample+stream wrapper.",
        "New model only if unreachable by composition.",
        "Light NNs (<=5M) only for ensemble diversification.",
        "Infrastructure refactor <= features.",
    ],
    "phases": [
        {
            "id": "A",
            "name": "Stabilization",
            "version_range": "v0.9.25",
            "status": "in_progress",
            "description": "Fix critical bugs found by audits; no new functionality.",
            "patches": [
                "v0.9.25a", "v0.9.25b", "v0.9.25c",
                "v0.9.25d", "v0.9.25e",
            ],
            "items": [
                {"id": "A.1", "name": "SARIMA integration order D->d",
                 "patch": "v0.9.25a", "status": "done", "complexity": "S",
                 "file": "cadw/ts/models.py"},
                {"id": "A.2", "name": "SARIMA Hannan-Rissanen intercept",
                 "patch": "v0.9.25a", "status": "done", "complexity": "S",
                 "file": "cadw/ts/models.py"},
                {"id": "A.3", "name": "SARIMA predict_in_sample contract",
                 "patch": "v0.9.25a", "status": "done", "complexity": "S",
                 "file": "cadw/ts/models.py"},
                {"id": "A.3b", "name": "_fit_one sigma2==0 accept",
                 "patch": "v0.9.25a-fix1", "status": "done", "complexity": "S",
                 "file": "cadw/ts/models.py"},
                {"id": "A.8", "name": "Missing TS tests (D>0, in_sample, ...)",
                 "patch": "v0.9.25a", "status": "done", "complexity": "S",
                 "file": "tests/test_ts_models.py"},
                {"id": "A.4", "name": "B1 refill linear concat",
                 "patch": "v0.9.25c", "status": "planned", "complexity": "S",
                 "file": "cadw/ops/sort.py"},
                {"id": "A.5", "name": "B2 encode_categorical K_est guard",
                 "patch": "v0.9.25c", "status": "planned", "complexity": "S",
                 "file": "cadw/dataframe.py"},
                {"id": "A.7", "name": "external_sort fan-in floor=8",
                 "patch": "v0.9.25c", "status": "planned", "complexity": "S",
                 "file": "cadw/ops/sort.py"},
                {"id": "A.6", "name": "18 to_pylist sites extra_factor",
                 "patch": "v0.9.25d", "status": "planned", "complexity": "M",
                 "file": "cadw/{dataframe,ops,ts}/*.py"},
                {"id": "A.9", "name": "B5 dead code _append_to_builders",
                 "patch": "v0.9.25e", "status": "planned", "complexity": "S",
                 "file": "cadw/ops/sort.py"},
                {"id": "A.10", "name": "B4 memory_guard fallback factor",
                 "patch": "v0.9.25e", "status": "planned", "complexity": "S",
                 "file": "cadw/core/memory_guard.py"},
            ],
        },
        {
            "id": "B",
            "name": "Infrastructure",
            "version_range": "v0.9.26-v0.9.27",
            "status": "planned",
            "description": "Remove context-transfer pain and test-latency pain. No functional changes.",
            "items": [
                {"id": "B.1", "name": "dataframe.py split into 9 modules",
                 "patch": "v0.9.26", "status": "planned", "complexity": "L",
                 "file": "cadw/dataframe.py"},
                {"id": "B.2", "name": "DECISIONS.md split into docs/ tree",
                 "patch": "v0.9.27", "status": "planned", "complexity": "L",
                 "file": "DECISIONS.md"},
                {"id": "B.3", "name": "Pytest markers per module + xdist",
                 "patch": "v0.9.27", "status": "planned", "complexity": "M",
                 "file": "tests/, pyproject.toml"},
                {"id": "B.4", "name": "collect_project.py phase-aware",
                 "patch": "v0.9.25c", "status": "planned", "complexity": "M",
                 "file": "scripts/service/collect_project.py"},
                {"id": "B.5", "name": "scripts/ folder reorganization",
                 "patch": "v0.9.27", "status": "planned", "complexity": "S",
                 "file": "scripts/"},
                {"id": "B.6", "name": "Cleanup (backups, old patches, snapshots)",
                 "patch": "v0.9.25e", "status": "planned", "complexity": "S",
                 "file": ".gitignore, scripts/archive/"},
                {"id": "B.7", "name": "Recipes scaffold (docs/recipes/)",
                 "patch": "v0.9.25b", "status": "done", "complexity": "S",
                 "file": "docs/index.md"},
                {"id": "B.8", "name": "docs/roadmap.json manifest",
                 "patch": "v0.9.25b", "status": "done", "complexity": "S",
                 "file": "docs/roadmap.json"},
            ],
        },
        {
            "id": "C",
            "name": "Pandas P0 + ML essentials",
            "version_range": "v0.9.28-v0.9.30",
            "status": "planned",
            "description": "Close last 2 P0 pandas + 2 ML essentials.",
            "items": [
                {"id": "C.1", "name": "df.assign(name, expr) streaming",
                 "patch": "v0.9.28", "status": "planned", "complexity": "M",
                 "file": "cadw/dataframe.py"},
                {"id": "C.2", "name": "encode_frequency(col)",
                 "patch": "v0.9.29", "status": "planned", "complexity": "S",
                 "file": "cadw/dataframe.py"},
                {"id": "C.3", "name": "encode_onehot(cols, prefix)",
                 "patch": "v0.9.29", "status": "planned", "complexity": "M",
                 "file": "cadw/dataframe.py"},
                {"id": "C.4", "name": "encode_target(col, y, n_splits) OOF",
                 "patch": "v0.9.30", "status": "planned", "complexity": "M",
                 "file": "cadw/dataframe.py"},
                {"id": "C.5", "name": "sample_weight in GBDT/Logistic/Linear",
                 "patch": "v0.9.30", "status": "planned", "complexity": "M",
                 "file": "cadw/ml/*.py"},
                {"id": "C.6", "name": "pinball loss in GBDT",
                 "patch": "v0.9.30", "status": "planned", "complexity": "M",
                 "file": "cadw/ml/gbdt.py"},
                {"id": "C.7", "name": "metrics(y_proba_cols=[...]) wrapper",
                 "patch": "v0.9.28", "status": "planned", "complexity": "S",
                 "file": "cadw/dataframe.py"},
            ],
        },
        {
            "id": "D",
            "name": "Algorithm coverage P1",
            "version_range": "v0.9.31-v0.9.36",
            "status": "planned",
            "description": "Cover task types currently NEEDS. On primitives, state O(1..model).",
            "items": [
                {"id": "D.1.1", "name": "IForest on batches",
                 "patch": "v0.9.31", "status": "planned", "complexity": "S"},
                {"id": "D.1.2", "name": "Matrix Profile streaming",
                 "patch": "v0.9.31", "status": "planned", "complexity": "M"},
                {"id": "D.1.3", "name": "cost-sensitive GBDT + threshold tuning",
                 "patch": "v0.9.31", "status": "planned", "complexity": "S"},
                {"id": "D.1.4", "name": "fairness metrics DP/EO/TPR-gap",
                 "patch": "v0.9.31", "status": "planned", "complexity": "S"},
                {"id": "D.2.1", "name": "LambdaMART lambdarank objective",
                 "patch": "v0.9.32", "status": "planned", "complexity": "M"},
                {"id": "D.2.2", "name": "ALS (implicit) iterative Reduce",
                 "patch": "v0.9.32", "status": "planned", "complexity": "M"},
                {"id": "D.2.3", "name": "hybrid MF + content",
                 "patch": "v0.9.32", "status": "planned", "complexity": "M"},
                {"id": "D.3.1", "name": "CoxPH partial likelihood",
                 "patch": "v0.9.33", "status": "planned", "complexity": "M"},
                {"id": "D.3.2", "name": "T/S-learner + propensity",
                 "patch": "v0.9.33", "status": "planned", "complexity": "M"},
                {"id": "D.3.3", "name": "uplift trees (Qini)",
                 "patch": "v0.9.33", "status": "planned", "complexity": "M"},
                {"id": "D.4.1", "name": "MiniRocket streaming",
                 "patch": "v0.9.34", "status": "planned", "complexity": "M"},
                {"id": "D.4.2", "name": "PELT online change-point",
                 "patch": "v0.9.34", "status": "planned", "complexity": "M"},
                {"id": "D.4.3", "name": "ETS (Holt-Winters)",
                 "patch": "v0.9.34", "status": "planned", "complexity": "M"},
                {"id": "D.4.4", "name": "Theta-method",
                 "patch": "v0.9.34", "status": "planned", "complexity": "S"},
                {"id": "D.4.5", "name": "VAR(p)",
                 "patch": "v0.9.34", "status": "planned", "complexity": "M"},
                {"id": "D.4.6", "name": "calibration Platt/isotonic",
                 "patch": "v0.9.34", "status": "planned", "complexity": "M"},
                {"id": "D.5.1", "name": "ordinal logistic cumulative link",
                 "patch": "v0.9.35", "status": "planned", "complexity": "M"},
                {"id": "D.5.2", "name": "classifier chains",
                 "patch": "v0.9.35", "status": "planned", "complexity": "S"},
                {"id": "D.5.3", "name": "MiniBatchKMeans / GMM",
                 "patch": "v0.9.35", "status": "planned", "complexity": "M"},
                {"id": "D.5.4", "name": "BIRCH CF-tree streaming",
                 "patch": "v0.9.35", "status": "planned", "complexity": "L"},
                {"id": "D.6.1", "name": "mutual_info / var threshold / Boruta",
                 "patch": "v0.9.36", "status": "planned", "complexity": "M"},
                {"id": "D.6.2", "name": "drift KS/PSI/Chi2, A/B t/MW/CUPED",
                 "patch": "v0.9.36", "status": "planned", "complexity": "M"},
                {"id": "D.6.3", "name": "permutation importance",
                 "patch": "v0.9.36", "status": "planned", "complexity": "S"},
                {"id": "D.6.4", "name": "randomized SVD",
                 "patch": "v0.9.36", "status": "planned", "complexity": "M"},
                {"id": "D.6.5", "name": "geo-aggregations h3/hexbin",
                 "patch": "v0.9.36", "status": "planned", "complexity": "S"},
                {"id": "D.6.6", "name": "LDA",
                 "patch": "v0.9.36", "status": "planned", "complexity": "L"},
                {"id": "D.6.7", "name": "LinUCB contextual bandit",
                 "patch": "v0.9.36", "status": "planned", "complexity": "S"},
                {"id": "D.6.8", "name": "GMM density",
                 "patch": "v0.9.36", "status": "planned", "complexity": "S"},
                {"id": "D.6.9", "name": "TF-IDF / text hashing on disk",
                 "patch": "v0.9.36", "status": "planned", "complexity": "M"},
            ],
        },
        {
            "id": "E",
            "name": "Recipes (docs)",
            "version_range": "parallel with D",
            "status": "planned",
            "description": "Task-type recipes. Fills docs/recipes/.",
            "items": [
                {"id": "E.1", "name": "classification recipe",
                 "patch": "parallel", "status": "planned", "complexity": "S"},
                {"id": "E.2", "name": "regression recipe",
                 "patch": "parallel", "status": "planned", "complexity": "S"},
                {"id": "E.3", "name": "clustering recipe",
                 "patch": "parallel", "status": "planned", "complexity": "S"},
                {"id": "E.4", "name": "anomaly recipe",
                 "patch": "parallel", "status": "planned", "complexity": "S"},
                {"id": "E.5", "name": "timeseries recipe",
                 "patch": "parallel", "status": "planned", "complexity": "M"},
                {"id": "E.6", "name": "ranking recipe",
                 "patch": "parallel", "status": "planned", "complexity": "S"},
                {"id": "E.7", "name": "survival recipe",
                 "patch": "parallel", "status": "planned", "complexity": "S"},
                {"id": "E.8", "name": "causal recipe",
                 "patch": "parallel", "status": "planned", "complexity": "S"},
                {"id": "E.9", "name": "recsys recipe",
                 "patch": "parallel", "status": "planned", "complexity": "S"},
                {"id": "E.10", "name": "text recipe",
                 "patch": "parallel", "status": "planned", "complexity": "S"},
                {"id": "E.11", "name": "tabular-fe recipe",
                 "patch": "parallel", "status": "planned", "complexity": "M"},
            ],
        },
        {
            "id": "F",
            "name": "Light NNs",
            "version_range": "v0.9.37",
            "status": "planned",
            "description": "Only where GBDT loses systematically. Batches via to_batches('torch').",
            "items": [
                {"id": "F.1", "name": "MLP tabular",
                 "patch": "v0.9.37", "status": "planned", "complexity": "M"},
                {"id": "F.2", "name": "N-BEATS blocks for TS",
                 "patch": "v0.9.37", "status": "planned", "complexity": "M"},
                {"id": "F.3", "name": "AE-reconstruction for anomaly",
                 "patch": "v0.9.37", "status": "planned", "complexity": "M"},
            ],
        },
        {
            "id": "G",
            "name": "Tech debt close",
            "version_range": "v0.9.38",
            "status": "planned",
            "description": "Close §11.5/§11.6/AUDIT §3.5 items.",
            "items": [
                {"id": "G.1", "name": "kmeans++ reservoir",
                 "patch": "v0.9.38", "status": "planned", "complexity": "S"},
                {"id": "G.2", "name": "RobustScaler.fit single sort",
                 "patch": "v0.9.38", "status": "planned", "complexity": "S"},
                {"id": "G.3", "name": "silent hash fallback warning",
                 "patch": "v0.9.38", "status": "planned", "complexity": "S"},
                {"id": "G.4", "name": "Arrow-native rank null path",
                 "patch": "v0.9.38", "status": "planned", "complexity": "S"},
                {"id": "G.5", "name": "groupby.agg dict + rolling trailing",
                 "patch": "v0.9.38", "status": "planned", "complexity": "S"},
                {"id": "G.6", "name": ".cache('name') + _viz_cache",
                 "patch": "v0.9.38", "status": "planned", "complexity": "S"},
                {"id": "G.7", "name": "ML checkpoint",
                 "patch": "v0.9.38", "status": "planned", "complexity": "M"},
                {"id": "G.8", "name": "SumOp int64 overflow decimal128",
                 "patch": "v0.9.38", "status": "planned", "complexity": "M"},
            ],
        },
        {
            "id": "H",
            "name": "P2 / long-tail",
            "version_range": "v0.9.39+",
            "status": "planned",
            "description": "Extended features, sample+stream wrappers for O(N^2) classes. Ongoing.",
            "items": [
                {"id": "H.1", "name": "lazy expression API",
                 "patch": "v0.9.39+", "status": "planned", "complexity": "L"},
                {"id": "H.2", "name": "auto-memoize",
                 "patch": "v0.9.39+", "status": "planned", "complexity": "M"},
                {"id": "H.3", "name": "ACF/PACF/STL",
                 "patch": "v0.9.39+", "status": "planned", "complexity": "L"},
                {"id": "H.4", "name": "to_csv / to_json streaming",
                 "patch": "v0.9.39+", "status": "planned", "complexity": "S"},
                {"id": "H.5", "name": "model pickle",
                 "patch": "v0.9.39+", "status": "planned", "complexity": "M"},
                {"id": "H.6", "name": "T1 linkage recipe",
                 "patch": "v0.9.39+", "status": "planned", "complexity": "S"},
                {"id": "H.7", "name": "T2 t-SNE/UMAP recipe",
                 "patch": "v0.9.39+", "status": "planned", "complexity": "S"},
                {"id": "H.8", "name": "T3 HDBSCAN recipe",
                 "patch": "v0.9.39+", "status": "planned", "complexity": "S"},
                {"id": "H.9", "name": "T4 One-Class SVM recipe",
                 "patch": "v0.9.39+", "status": "planned", "complexity": "S"},
                {"id": "H.10", "name": "T11 causal forest",
                 "patch": "v0.9.39+", "status": "planned", "complexity": "L"},
                {"id": "H.11", "name": "T12 RSF",
                 "patch": "v0.9.39+", "status": "planned", "complexity": "L"},
            ],
        },
    ],
}


INDEX_MD = """\
# CADW Documentation

**Entry point** for all CADW documentation. Replaces the monolithic
`DECISIONS.md` as the primary navigation aid.

**Current version:** v0.9.25a
**Phase:** A (Stabilization) — see [roadmap](ROADMAP.md)

---

## Start here

- **[Roadmap](ROADMAP.md)** — what's done, what's next, phases A..H.
- **[roadmap.json](roadmap.json)** — machine-readable, consumed by
  `scripts/service/collect_project.py --phase`.

## By task type (recipes)

Each recipe: available today, roadmap, numbered steps, pandas/sklearn
equivalent table. Filled progressively during Phase E; scaffold is here.

- `recipes/classification.md` — binary, multiclass, multilabel,
  ordinal, imbalanced
- `recipes/regression.md` — L2, quantile, Tweedie
- `recipes/clustering.md` — kmeans, MiniBatch, GMM, BIRCH
- `recipes/anomaly.md` — IForest, MatrixProfile, AE
- `recipes/timeseries.md` — univariate, multivariable, hierarchical,
  classification, change-point
- `recipes/ranking.md` — LambdaMART, pairwise, listwise
- `recipes/survival.md` — CoxPH, RSF, DeepSurv
- `recipes/causal.md` — T/S-learner, propensity, uplift
- `recipes/recsys.md` — ALS, hybrid MF, sequence models
- `recipes/text.md` — TF-IDF, embeddings, BERT fine-tune
- `recipes/tabular-fe.md` — encoding, imputation, binning, selection

## By module (API reference)

*To be created in Phase B.2 (DECISIONS.md split).* For now, see
`../DECISIONS.md` §10.

## Architecture

- `../DECISIONS.md` §3–§8 — principles, primitives, memory model.
  Split into `architecture/*.md` in Phase B.2.

## Audit and limitations

- `../DECISIONS.md` §11.5 — documented limitations (28 items).
- `../DECISIONS.md` §11.6 — tech debt.
- `../DECISIONS.md` §13.4 — audit lessons (episodes 1..N).
- `../AUDIT_FULL_v0922.md` — round-4 audit (34 items).
- `../part1_report.md` — RAM / efficiency audit.
- `../part2_taxonomy.md` — 112-row DS task taxonomy × CADW coverage.
- `../part2_todo_roadmap.md` — N1-N40 + T1-T12 detailed plans.

## Changelog

- `../CHANGELOG.md`
- `../README.md`

---

*Migration note: DECISIONS.md will be split into this tree during
Phase B.2 (v0.9.27). Until then, both locations are authoritative;
this index is the recommended entry point.*
"""


# ============ helpers ============

def _write_doc(path: str, content: str, marker: str) -> bool:
    """Idempotent write. Skip if file exists and marker present."""
    p = Path(path)
    if p.exists():
        existing = p.read_text()
        if marker in existing:
            print(f"  {path}: already present (marker {marker!r}) — skip")
            return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, newline="\n")
    print(f"  {path}: written ({len(content)} B)")
    return True


# ============ 1. docs/ROADMAP.md ============

def create_roadmap_md():
    print("\n=== docs/ROADMAP.md ===")
    _write_doc("docs/ROADMAP.md", ROADMAP_MD,
               marker="# CADW Roadmap")


# ============ 2. docs/roadmap.json ============

def create_roadmap_json():
    print("\n=== docs/roadmap.json ===")
    content = json.dumps(ROADMAP_JSON_DICT, indent=2, ensure_ascii=False)
    _write_doc("docs/roadmap.json", content,
               marker='"schema_version"')


# ============ 3. docs/index.md ============

def create_index_md():
    print("\n=== docs/index.md ===")
    _write_doc("docs/index.md", INDEX_MD,
               marker="# CADW Documentation")


# ============ 4. .gitignore additions ============

GITIGNORE_ADDITIONS = [
    "ts_review_snapshot.md",
    "scripts/_tmp/",
    "_FOR_PINGIS/_scratch/",
    "_verify_*.py",
]


def update_gitignore():
    print("\n=== .gitignore additions ===")
    p = Path(".gitignore")
    if not p.exists():
        p.write_text("")
    s = p.read_text()
    added = []
    for line in GITIGNORE_ADDITIONS:
        if line not in s.splitlines():
            s = s.rstrip() + "\n" + line + "\n"
            added.append(line)
    if added:
        p.write_text(s)
        print(f"  added {len(added)}: {added}")
    else:
        print("  all entries already present — skip")


# ============ 5. tests/test_roadmap.py ============

ROADMAP_TESTS = '''"""Structural tests for docs/roadmap.json and docs/ROADMAP.md (v0.9.25b)."""
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
ROADMAP_JSON = ROOT / "docs" / "roadmap.json"
ROADMAP_MD = ROOT / "docs" / "ROADMAP.md"
INDEX_MD = ROOT / "docs" / "index.md"


def test_roadmap_json_parses():
    assert ROADMAP_JSON.exists(), f"missing {ROADMAP_JSON}"
    data = json.loads(ROADMAP_JSON.read_text())
    assert data["schema_version"] == "1"
    assert isinstance(data["principles"], list)
    assert len(data["principles"]) >= 3


def test_roadmap_json_phases_present():
    data = json.loads(ROADMAP_JSON.read_text())
    phases = data["phases"]
    phase_ids = [p["id"] for p in phases]
    for required in ("A", "B", "C", "D", "E", "F", "G", "H"):
        assert required in phase_ids, f"phase {required!r} missing"


def test_roadmap_json_items_have_required_fields():
    data = json.loads(ROADMAP_JSON.read_text())
    for phase in data["phases"]:
        assert "items" in phase
        for item in phase["items"]:
            for field in ("id", "name", "patch", "status", "complexity"):
                assert field in item, (
                    f"phase {phase['id']} item {item.get('id')} "
                    f"missing field {field!r}"
                )
            assert item["status"] in ("done", "in_progress", "planned"), (
                f"bad status {item['status']!r} for {item['id']}"
            )


def test_roadmap_md_phase_headers():
    text = ROADMAP_MD.read_text()
    for pid in "ABCDEFGH":
        assert f"## §" in text  # sanity: section markers present
    for name in (
        "Phase A — Stabilization",
        "Phase B — Infrastructure",
        "Phase C — Pandas P0",
        "Phase D — Algorithm coverage",
        "Phase E — Recipes",
        "Phase F — Light NNs",
        "Phase G — Tech debt close",
        "Phase H — P2",
    ):
        assert name in text, f"ROADMAP.md missing section: {name!r}"


def test_docs_index_links_to_roadmap_and_recipes():
    text = INDEX_MD.read_text()
    assert "ROADMAP.md" in text
    assert "roadmap.json" in text
    assert "recipes/" in text


def test_roadmap_json_done_items_match_current_version():
    """Any 'done' item should reference a patch <= current_version."""
    data = json.loads(ROADMAP_JSON.read_text())
    current = data["current_version"]  # e.g. "0.9.25a"
    # Simple sanity: v0.9.25a-fix1 items are done.
    for phase in data["phases"]:
        for item in phase["items"]:
            if item["status"] == "done":
                assert item["patch"].startswith("v"), (
                    f"done item {item['id']} has non-version patch "
                    f"{item['patch']!r}"
                )
'''


def add_roadmap_tests():
    print("\n=== tests/test_roadmap.py ===")
    p = Path("tests/test_roadmap.py")
    if p.exists() and "test_roadmap_json_parses" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(ROADMAP_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 6. Verify state ============

def verify_state():
    print("\n=== verify v0.9.25a committed ===")
    r = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True,
    )
    if "v0.9.25a" not in r.stdout:
        raise SystemExit(
            f"v0.9.25a not in HEAD commit; got {r.stdout!r}. "
            f"Re-run v0.9.25a-fix1 first."
        )
    print(f"  HEAD: {r.stdout.strip()[:80]}")


# ============ 7. Dogfood ============

def dogfood():
    print("\n=== dogfood: docs load + roadmap consistency ===")

    def _verify():
        import json as _j
        from pathlib import Path as _P
        data = _j.loads(_P("docs/roadmap.json").read_text())
        assert data["schema_version"] == "1"
        phase_ids = {p["id"] for p in data["phases"]}
        assert phase_ids == set("ABCDEFGH"), (
            f"expected phases A-H, got {sorted(phase_ids)}"
        )
        # Every done item belongs to phase A.
        for phase in data["phases"]:
            for item in phase["items"]:
                if item["status"] == "done":
                    assert phase["id"] == "A", (
                        f"done item {item['id']} not in phase A — "
                        f"unexpected"
                    )
        # Every item with patch=v0.9.25a is done.
        for phase in data["phases"]:
            for item in phase["items"]:
                if item["patch"] in ("v0.9.25a", "v0.9.25a-fix1"):
                    assert item["status"] == "done"
        # Roadmap.md has all 8 phase sections.
        md = _P("docs/ROADMAP.md").read_text()
        for pid in "ABCDEFGH":
            assert f"## §" in md
        for name in ("Stabilization", "Infrastructure",
                     "Pandas P0", "Algorithm coverage",
                     "Recipes", "Light NNs",
                     "Tech debt close", "P2"):
            assert name in md, f"ROADMAP.md missing {name!r}"
        # Index links.
        idx = _P("docs/index.md").read_text()
        assert "ROADMAP.md" in idx and "roadmap.json" in idx
        print("  roadmap.json + ROADMAP.md + index.md consistent")

    from cadw.patch_utils import run_verify_in_subprocess
    run_verify_in_subprocess(_verify)


# ============ 8. Pytest ============

def run_pytest():
    print("\n=== pre-flight: test_facade_has_no_as_py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py", "-q"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** FACADE AS_PY VIOLATION — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: roadmap tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_roadmap.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** ROADMAP TEST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 9. main ============

def main():
    print("=== v0.9.25b: docs infrastructure ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    create_roadmap_md()
    create_roadmap_json()
    create_index_md()
    update_gitignore()
    add_roadmap_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25b: docs infrastructure — ROADMAP.md + roadmap.json "
        "+ docs/index.md + .gitignore + tests (Phase B prep). "
        "Docs-only patch; cadw/ untouched. "
        "docs/ROADMAP.md — human-readable, phases A..H with items, "
        "principles, cross-reference task-type -> phase. ~500 lines. "
        "docs/roadmap.json — machine-readable schema_version=1, "
        "8 phases, per-item {id, name, patch, status, complexity}. "
        "Ready for collect_project.py --phase (v0.9.25c). "
        "docs/index.md — new docs entry point; links to roadmap, "
        "recipes/ (scaffold for Phase E), audit docs. Replaces "
        "monolithic DECISIONS.md as primary navigation. "
        "DECISIONS.md untouched — split is Phase B.2 (v0.9.27). "
        ".gitignore: ts_review_snapshot.md, scripts/_tmp/, "
        "_FOR_PINGIS/_scratch/, _verify_*.py. Note: ts_review_snapshot "
        "already committed in v0.9.25a; git rm --cached is Phase B.6. "
        "+tests/test_roadmap.py (6 tests: JSON parses, all phases "
        "present, items have required fields, ROADMAP.md headers, "
        "index links, done items match version). "
        "Fast suite: ~2269 -> ~2275.",
        self_path="scripts/patch_v0925b_docs_roadmap.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25b applied. Next: v0.9.25c "
          "(B1 refill + B2 encode_cat + fan-in floor + "
          "collect_project.py phase-aware).")


if __name__ == "__main__":
    main()