# Консолидированный roadmap CADW

Свёл 4 источника + предыдущую сессию в один документ, сгруппированный по **фазам** (а не по модулям или приоритетам). Фаза = набор взаимозависимых работ, который закрывается набором патчей. Внутри фазы — приоритеты P0/P1/P2 как маркеры.

**Источники, вошедшие в консолидацию:**
- `AUDIT_FULL_v0922 §7` — 34 item (7.A/7.B/7.C)
- `part1_report.md` — B1-B8, §2.3 (18 call-sites), A1-A10, B1-B5, C1-C8 (частично пересекается)
- `part2_taxonomy.md` + `part2_todo_roadmap.md` — N1-N40 (NEEDS), T1-T12 (TODO)
- Предыдущая сессия — 3 critical SARIMA бага, missing tests, refactor asks

---

## §1 Принципы (жёсткие фильтры для новых items)

1. **Универсальность через рецепты, не через зоопарк.** Задача покрыта, если есть **один** примитив + рецепт. Не «30 anomaly detectors» — 1 IForest + рецепт композиции.
2. **Каждый алгоритм либо state O(1)/O(d²), либо обёртка «sample + stream».** Полные реализации O(N²)/глубоких — не в ядре.
3. **Новая модель — только если покрывает класс задач, недостижимый композицией существующих.** Например, `CoxPH` = Reduce + Sort — берём; `DeepSurv` — только через `to_batches(format='torch')`.
4. **Лёгкие NN (≤5M) — только для диверсификации ансамбля** там, где GBDT систематически уступает (TS-классификация, sequence recsys). Обучение — батчами, вне ядра.
5. **Infrastructure refactor ≤ features.** `dataframe.py` split и `DECISIONS.md` split — **блокеры** следующих фаз, а не «purity».

---

## §2 Карта фаз

| Phase | Версия | Объём | Патчей | Зависит от | Зачем |
|---|---|---|---|---|---|
| **A** Stabilization | v0.9.25 | ~10 items | 1 | — | Блокеры: TS-баги, B1/B2/B8 |
| **B** Infrastructure | v0.9.26–v0.9.27 | ~8 items | 2–3 | A | `dataframe.py` split, `DECISIONS.md` split, тесты, tooling, cleanup |
| **C** Pandas P0 + ML essentials | v0.9.28–v0.9.30 | ~6 items | 3 | B | `assign`, cat FE, `sample_weight`, `pinball`, `y_proba_cols` |
| **D** Algorithm coverage P1 | v0.9.31–v0.9.36 | ~20 items | 6–8 | C | Anomaly / recsys / ranking / survival / causal / TS-ext / text |
| **E** Recipes (docs) | параллельно D | ~12 docs | 1–2 | C | Recipes by task type + API reference + migration |
| **F** Light NNs | v0.9.37 | ~3 items | 1 | D | N-BEATS, 1D-CNN, AE — опционально |
| **G** Tech debt close | v0.9.38 | ~8 items | 2 | B, C | kmeans++, RobustScaler, silent-fallback, API-ergonomics, §11.5 |
| **H** P2 / long-tail | v0.9.39+ | ~15 items | ongoing | G | Lazy expr, memoize, STL, GMM, grid search, pickle, … |

---

## §3 Phase A — Stabilization (v0.9.25, 1 патч)

**Цель:** починить критические баги, ничего нового.

| ID | Item | Source | Severity | LOC | Test |
|---|---|---|---|---|---|
| A.1 | `SARIMA._integrate_back`: порядок `D` затем `d`, `seasonal_period_last` из `_difference(y_raw, d)` | TS-review | **critical** (D>0 broken) | ~30 | +тест с `D=1, d=1, s=2` |
| A.2 | SARIMA: центрирование / intercept (Hannan-Rissanen без intercept → прогноз → 0 на stationary с mean≠0) | TS-review | **critical** | ~10 | +тест с `mean=5` |
| A.3 | `SARIMA._predict_in_sample`: вернуть длину `n_train` в масштабе `y` (сейчас — diff-шкала, длина `n-1`); тихо ломает THF-WLS | TS-review | **critical** | ~20 | +тест длины и масштаба |
| A.4 | B1: `_RunStream.refill` — линейный `concat_tables(parts_list)` | part1 | high (12.9 GB копий @ rbs=200k) | ~15 | test_refill_linear_time |
| A.5 | B2: `encode_categorical` — `K_est` по sample + `check_state(K_est × (value_bytes + 50))` до Pass1 | part1 | high (O(K) Python-set) | ~25 | test_encode_categorical_high_card_guard |
| A.6 | B8: 18 call-sites `to_pylist` без `extra_factor` (таблица в part1 §2.3, factor 3.5–12) | part1 | medium RAM | ~40 | AST guard проверяет `extra_factor` ≥ 3.0 при `to_pylist` |
| A.7 | `external_sort` fan-in floor = 8 (проверить, применено ли; в `v0.9.23c_merge_arrow` не упоминается) | AUDIT §6.6 | high (24s @ 500k) | ~5 | test_sort_fan_in_floor + bench |
| A.8 | Missing TS tests: `D>0`, `predict_in_sample`, external_sort ascending-list, top-down `agg='mean'` | TS-review | high | ~50 | сами тесты |
| A.9 | B5: удалить dead code `_append_to_builders` (ops/sort.py:314) | part1 | low | ~15 | — |
| A.10 | B4: `memory_guard.py:696` fallback `len(set(...))` — `extra_factor=4` | part1 | low | ~3 | — |

**Acceptance:** `pytest -m "not slow"` зелёный; все 3 critical SARIMA бага закрыты тестами; RAM @ 1.9 GB подтверждён для refill и encode_categorical.

---

## §4 Phase B — Infrastructure (v0.9.26–v0.9.27)

**Цель:** снять боль «трудно передавать контекст в сессию» и «тесты 2-3 мин». Никаких функциональных изменений.

### B.1 `dataframe.py` split (v0.9.26) — A1 из старого roadmap

7134 строки → 9 модулей. Подробный дизайн в моём предыдущем ответе. Ключевое:
- `dataframe.py` facade ~800
- `_io.py` / `_index.py` / `_reshape.py` / `_window.py` / `_apply.py` / `_ml.py` / `_stats.py` / `_fe.py`
- Re-export в `__init__.py` для backward compat (`from cadw.dataframe import _ArrowTail` продолжает работать)
- AST-guard `test_facade_has_no_as_py` — только facade, не 7000 строк

**Acceptance:** 2261 тест зелёный без изменений; AST-guard пересобран; `from cadw.dataframe import X` работает для 100+ public/internal имён.

### B.2 `DECISIONS.md` split (v0.9.27)

5380+ строк → `docs/` дерево. Структура ниже (§12).

### B.3 Tests grouping (v0.9.27)

Pytest markers per модуль + xdist + per-test timing assertion.

### B.4 `collect_project.py` phase-aware (v0.9.27)

`--phase A`, `--module NAME`, `--doc NAME` читает manifest `docs/roadmap.json`.

### B.5 Folder structure (v0.9.27)

```
scripts/
├── service/          # stable entry points (collect_project, verify_*, bench_*)
├── _tmp/             # temp scripts (gitignored)
├── _measure/         # measurement harnesses
├── _prototypes/      # prototypes
├── archive/patches/  # old patches (v0.1 to current phase - 2)
├── patch_v0_9_2*.py  # current cycle
```

### B.6 Cleanup (v0.9.27)

См. §16.

### B.7 Recipes scaffold (v0.9.27)

Папка `docs/recipes/` с 11 файлами-заглушками + шаблон + 1 полный пример (classification).

### B.8 Roadmap manifest (v0.9.27)

`docs/roadmap.json` — машиночитаемая карта фаз.

**Acceptance:** 
- `uv run python scripts/service/collect_project.py --phase A` работает
- `pytest -m "not slow and ts"` — TS-only за ~3s
- `pytest -n auto -m "not slow"` — parallel, если тесты parallel-safe
- `ls docs/` — новая структура, `DECISIONS.md` в корне — index

---

## §5 Phase C — Pandas P0 + ML essentials (v0.9.28–v0.9.30, 3 патча)

**Цель:** закрыть последние 2 P0 pandas + 2 ML essentials из taxonomy N1/N2. Они нужны в **каждом** DS-проекте.

| ID | Item | Source | Task types | LOC | Patch |
|---|---|---|---|---|---|
| C.1 | `df.assign(name, expr)` — streaming Arrow-expression | 7.A.7 | все numeric FE (taxonomy row 16) | ~200 | v0.9.28 |
| C.2 | `encode_frequency(col)` | 7.A.8 / N12 (partially) | cat FE (row 24) | ~80 | v0.9.29 |
| C.3 | `encode_onehot(cols, prefix)` — sparse dict + streaming expansion, K_kept capped by guard | 7.A.8 | cat FE (row 23) | ~100 | v0.9.29 |
| C.4 | `encode_target(col, y, n_splits)` — OOF через KFold | 7.A.8 / N12 | cat FE (row 26) | ~130 | v0.9.30 |
| C.5 | N1: `sample_weight` в `fit_gbdt_*` / `fit_logistic` / `fit_linear` | part2 P0 | A1-A3, A6, G1, H1-H2, I1 | ~150 | v0.9.30 |
| C.6 | N2: `pinball loss` в `fit_gbdt_regressor(loss='quantile', alpha=q)` | part2 P0 | A7, D1-D2 | ~80 | v0.9.30 |
| C.7 | `metrics(y_proba_cols=[...])` wrapper (из AUDIT §3.5) | AUDIT §3.5 / N | metrics | ~30 | v0.9.28 |

**C.1 — самый большой, отдельный патч.** C.2-C.4 — три по 80-130 LOC, можно в один патч с проверкой. C.5 — Reduce+веса, аккуратно (переписать GBDT `_grad_hess` с весами).

**Acceptance:** 
- `df.assign("log_x", pc.log(df["x"]))` работает, streaming
- `sample_weight` в GBDT = sklearn-эквивалент на эталоне
- `pinball` покрытие ~q на синтетике
- cat FE — нет leakage, = category_encoders на эталоне

---

## §6 Phase D — Algorithm coverage P1 (v0.9.31–v0.9.36)

**Цель:** покрыть типы задач, которые сейчас NEEDS. Все — на примитивах, state O(1..O(model)). Организовано **по типам задач**.

### D.1 Anomaly (v0.9.31)

| ID | Item | Source | State | Зачем |
|---|---|---|---|---|
| D.1.1 | N5: Isolation Forest на батчах | part2 P0 | t·O(ψ) | самый масштабируемый unsupervised detector |
| D.1.2 | N6: Matrix Profile (STUMPY-подобный) streaming | part2 P0 | O(window) | TS anomaly, IoT |
| D.1.3 | N22: cost-sensitive GBDT + threshold tuning | part2 P1 | O(model) | supervised fraud — композиция на C.5 |
| D.1.4 | N28: fairness-метрики (DP/EO/TPR-gap) | part2 P1 | O(K) | production compliance |

### D.2 Ranking / Recsys (v0.9.32)

| ID | Item | Source | State |
|---|---|---|---|
| D.2.1 | N8: LambdaMART lambdarank objective в GBDT | part2 P0 | O(model) |
| D.2.2 | N7: ALS (implicit) — итеративный Reduce | part2 P0 | O((U+V)·k) |
| D.2.3 | N23: hybrid MF + content (ALS + GBDT-ранкер) | part2 P1 | O(model) |

### D.3 Survival / Causal (v0.9.33)

| ID | Item | Source | State |
|---|---|---|---|
| D.3.1 | N10: CoxPH — Reduce partial likelihood + Sort | part2 P0 | O(d) |
| D.3.2 | N9: T/S-learner + propensity | part2 P0 | O(model) |
| D.3.3 | N24: uplift trees (Qini) | part2 P1 | O(trees) |

### D.4 TS extension (v0.9.34)

| ID | Item | Source | State |
|---|---|---|---|
| D.4.1 | N3: MiniRocket как streaming-преобразование | part2 P0 | O(#kernels) fixed |
| D.4.2 | N4: PELT online change-point | part2 P0 | O(window) |
| D.4.3 | N18: ETS (Holt-Winters) | part2 P1 | O(k) сезонностей |
| D.4.4 | N19: Theta-метод | part2 P1 | O(1) |
| D.4.5 | N20: VAR(p) | part2 P1 | O(p·k²) |
| D.4.6 | N33: calibration (Platt / isotonic) | part2 P1 | O(d) / O(N_bins) |

### D.5 Classification extensions (v0.9.35)

| ID | Item | Source | State |
|---|---|---|---|
| D.5.1 | N13: ordinal logistic (cumulative link) | part2 P1 | O(d·K) |
| D.5.2 | N14: classifier chains | part2 P1 | O(K·model) |
| D.5.3 | N15: MiniBatchKMeans / GMM (EM) | part2 P1 | O(K·d) / O(K·d²) |
| D.5.4 | N16: BIRCH CF-tree streaming | part2 P1 | O(#CF-nodes) |

### D.6 Features / selection / interpretability / stats (v0.9.36)

| ID | Item | Source | State |
|---|---|---|---|
| D.6.1 | N25: mutual_info / variance threshold / Boruta | part2 P1 | O(d) |
| D.6.2 | N26: drift (KS/PSI/Chi2), A/B (t/MW/CUPED) | part2 P1 | O(1) |
| D.6.3 | N27: permutation importance | part2 P1 | O(d·model) |
| D.6.4 | N17: randomized SVD (PCA для больших d) | part2 P1 | O(N·d·k) |
| D.6.5 | N29: geo-агрегации (h3/hexbin) | part2 P1 | O(1) |
| D.6.6 | N30: LDA (variational / EM) | part2 P1 | O(K·V) — влезает для V≤2e5 |
| D.6.7 | N31: LinUCB contextual bandit | part2 P1 | O(d²·arms) |
| D.6.8 | N32: GMM density | part2 P1 | O(K·d²) |
| D.6.9 | N11: TF-IDF / хэширование текста на диске | part2 P0 | O(V) |

**Acceptance:** каждый item = отдельный патч или подпатч; тесты = sklearn/statsmodels-эквивалент на эталоне.

---

## §7 Phase E — Recipes (docs, параллельно D)

**Цель:** «разделение функционала по типам задач» на уровне документации.

```
docs/recipes/
├── _template.md
├── classification.md      # A1-A2, A4-A6 — полный
├── regression.md          # A3, A7
├── clustering.md          # B1-B3
├── anomaly.md             # E1-E3
├── timeseries.md          # D1-D5
├── ranking.md             # G1
├── survival.md            # I1
├── causal.md              # H1-H2
├── recsys.md              # F1-F2
├── text.md                # J1-J5
├── tabular-fe.md          # O1-O13
```

**Структура каждого рецепта:**
```markdown
# Classification

## Available today
- ✓ `fit_gbdt_classifier` — binary, multiclass
- ✓ `fit_logistic`
- ✓ NB (3 variants)
- ✓ `metrics` с `y_proba_cols=`

## Roadmap
- D.5.1 ordinal logistic (v0.9.35)
- C.5 sample_weight (v0.9.30)

## Recipe: imbalanced binary (fraud)
1. Load
2. dropna/fillna
3. GBDT с sample_weight=... (после C.5) / взвешенный threshold (сейчас)
4. metrics: precision/recall/f1 + roc_auc
5. Backtesting: KFold(stratify_by='y')

## pandas / sklearn equivalent
| pandas/sklearn | CADW | status |
|---|---|---|
| `sklearn.GradientBoostingClassifier` | `fit_gbdt_classifier` | ✓ |
| `class_weight='balanced'` | — (C.5 в плане) | roadmap |
```

**Acceptance:** 11 рецептов + индекс, каждый покрывает 1-3 taxonomy rows; в конце каждого — «что доступно сейчас, что в roadmap, что вне scope».

---

## §8 Phase F — Light NNs (v0.9.37, опционально)

Только там, где GBDT систематически уступает:

| ID | Item | Why | Как |
|---|---|---|---|
| F.1 | MLP tabular (2-3 слоя, ~0.05-2M) | диверсификация ансамбля | `to_batches(format='torch')` |
| F.2 | N-BEATS-блоки для TS | долгие зависимости | батчи |
| F.3 | AE-reconstruction для anomaly | нелинейная граница | батчи |

**Не делаем:** NCF/SASRec (recsys), BERT fine-tune (text), CV — только через `to_batches(format='torch')` + рецепт в документации.

**Acceptance:** обучается на батчах, RAM ≤ budget, рецепт в `docs/recipes/` с benchmark vs GBDT.

---

## §9 Phase G — Tech debt close (v0.9.38)

| ID | Item | Source | LOC |
|---|---|---|---|
| G.1 | kmeans++ reservoir (закрывает quality gap +36.9%) | §11.6 / AUDIT_ML | ~50 |
| G.2 | RobustScaler.fit одним sort с col_id (10s → минуты @ d=100) | §11.6 | ~30 |
| G.3 | silent hash fallback: `logging.warning` + optional `strict=True` | §11.6 | ~10 |
| G.4 | Arrow-native rank null path (7.B.4) | 7.B.4 | ~40 |
| G.5 | API-ergonomics: `df.groupby('g').agg({'x': 'sum'})` dict + `df.rolling(10).mean()` trailing | AUDIT §3.5 | ~80 |
| G.6 | `.cache('name')` + `_viz_cache` (7.B.1 + 7.B.2) | 7.B.1, 7.B.2 | ~80 |
| G.7 | ML checkpoint (7.B.3) | 7.B.3 | ~150 |
| G.8 | `SumOp` int64 overflow → `decimal128` опция | §11.5 | ~60 |

**Acceptance:** каждая closes конкретный §11.5/§11.6/AUDIT §3.5 item.

---

## §10 Phase H — P2 / long-tail (v0.9.39+)

Из `AUDIT §7.3` (7.C.1–7.C.11) + `part2 §1 P2` (N34–N40) + `part2 §2 TODO` (T1–T12):

| Класс | Items |
|---|---|
| **Lazy expression** | 7.C.1 Lazy API (300-500 LOC), 7.C.2 auto-memoize |
| **Statistical tests** | N26 drift/drift, N34 IterativeImputer |
| **Long-tail algorithms** | 7.C.3 ACF/PACF, 7.C.4 STL, 7.C.5 GMM/EM, 7.C.6 IForest, 7.C.7 GLM, N37 partial_fit, N38 Arrow-dict-IPC, N40 t-digest |
| **Serialization / interop** | 7.C.9 to_csv/json, 7.C.10 pickle, 7.C.7 polars/arrow-c-stream |
| **TODO-класс (sample+stream)** | T1 linkage, T2 t-SNE/UMAP, T3 HDBSCAN, T4 One-Class SVM, T11 causal forest, T12 RSF |
| **TODO (вне scope)** | T5 NCF/SASRec, T6 BERT, T7 CV, T8 GNN, T9 RL, T10 generative — только рецепты |

**Принцип:** это не «сделаем всё», а «есть список известных расширений; делаем по запросу/фидбеку». T5-T10 — только recipes, без реализации.

---

## §11 Cross-reference: task type → phases

| Taxonomy row | Тип | Доступно сейчас | Phase D item | Phase H item |
|---|---|---|---|---|
| A1 | binary class | GBDT, Logistic, NB | D.5.4 (chains), C.5 | — |
| A2 | multiclass | GBDT-multiclass | D.5.1 ordinal | — |
| A3 | regression | Linear/Ridge/Lasso/GBDT | C.6 (quantile) | — |
| A4 | multilabel | OvR вручную | D.5.4 chains | — |
| A5 | ordinal | — | D.5.1 | — |
| A6 | imbalanced | — | C.5 sample_weight, D.1.3 | — |
| A7 | quantile reg | — | C.6 pinball | — |
| B1 | clustering | kmeans | D.5.3 MiniBatch/GMM | T1/T3 recipes |
| B2 | hierarchical | — | — | T1 recipe |
| B3 | BIRCH | — | D.5.4 | — |
| C1 | PCA | PCA | D.6.4 randomized SVD | — |
| C2 | t-SNE/UMAP | — | — | T2 recipe |
| C3 | Autoencoder | через `to_batches('torch')` | — | F.3 optional |
| D1 | TS univariate | Naive, SARIMA, GBDT | D.4.3 ETS, D.4.4 Theta | — |
| D2 | TS multivariable | — | D.4.5 VAR | — |
| D3 | TS hierarchical | THF MinT-WLS | — | — |
| D4 | TS classification | — | D.4.1 MiniRocket | F.2 N-BEATS |
| D5 | Change-point | — | D.4.2 PELT | — |
| E1 | Anomaly unsupervised | — | D.1.1 IForest | T4 One-Class SVM |
| E2 | Anomaly supervised | GBDT | D.1.3 | — |
| E3 | TS anomaly | — | D.1.2 MatrixProfile | — |
| F1 | Recsys MF | — | D.2.2 ALS | T5 NCF/SASRec |
| F2 | Hybrid recsys | — | D.2.3 | — |
| G1 | Learning to rank | — | D.2.1 LambdaMART | — |
| H1 | Causal | — | D.3.2 T/S-learner | T11 causal forest |
| H2 | Uplift | — | D.3.3 | — |
| I1 | Survival | — | D.3.1 CoxPH | T12 RSF |
| J1 | Text classification | — | D.6.9 TF-IDF | T6 BERT |
| J4 | Topic modeling | — | D.6.6 LDA | — |
| K1-K4 | CV | — | — | T7 (recipes) |
| L1 | GNN | — | — | T8 (recipes) |
| M1 | RL | — | D.6.7 bandits | T9 (recipes) |
| N1 | Generative | — | D.6.8 GMM density | T10 (out) |
| O1-O13 | EDA/FE/validation | describe/hist/quantile/value_counts | D.6.1-D.6.3, D.6.5, C.1-C.4 | — |

---

## §12 Docs split proposal

```
docs/
├── index.md                    # was DECISIONS.md, links to all
├── architecture/
│   ├── principles.md           # §3 (principles, no ML adapters)
│   ├── primitives.md           # §5-§8 (Reduce, HashShuffle, Sort)
│   └── memory-model.md         # §9 (planner, op models)
├── reference/                  # API reference by module
│   ├── dataframe.md
│   ├── ops.md
│   ├── ml.md
│   ├── ts.md
│   ├── pipeline.md
│   ├── cv.md
│   └── viz.md
├── recipes/                    # task-type recipes (Phase E)
│   ├── _template.md
│   ├── classification.md
│   ├── regression.md
│   ├── clustering.md
│   ├── anomaly.md
│   ├── timeseries.md
│   ├── ranking.md
│   ├── survival.md
│   ├── causal.md
│   ├── recsys.md
│   ├── text.md
│   └── tabular-fe.md
├── decisions/                  # ADR-style (was §13 selective)
│   ├── 0001-no-ml-adapters.md
│   ├── 0002-explicit-disk-contract.md
│   ├── 0010-own-gbdt.md
│   ├── 0011-timeseries.md      # v0.9.24 decisions
│   ├── 0012-sort-values-ascending-list.md
│   └── ...
├── audit-lessons.md            # §13.4 (episodes 1..N)
├── limitations.md              # §11.5 (28 items)
├── roadmap.md                  # this
├── tests.md                    # test marker guide
├── changelog.md                # link to CHANGELOG.md
└── migration-from-pandas.md
```

**`DECISIONS.md` → `docs/index.md`** — 50-100 строк, links. Остальное — в дереве.

---

## §13 Tests grouping proposal

**Markers** (добавить в `pyproject.toml [tool.pytest.ini_options].markers`):

```toml
markers = [
    "slow: E2E, >1s (already exists)",
    "guards: AST/architecture guard tests",
    "core: MemoryGuard, Dataset, models_*, planner",
    "ops: reduce, shuffle, sort, groupby, join, dedup, rank, quantile, resample, pivot, topk, describe, metrics",
    "ml: linear, logistic, kmeans, pca, gbdt, naive_bayes",
    "ts: timeseries toolkit",
    "pipeline: pipeline, cv, scalers, accessors",
    "facade: dataframe, indexers, concat",
    "viz: viz adapters",
    "compat: pandas-equivalence",
]
```

**Разметка:** не 2261 файл — автоматически по пути (`tests/test_ts_*` → ts, `tests/test_gbdt_*` → ml). Через `pytest_collection_modifyitems` hook в `conftest.py`:

```python
def pytest_collection_modifyitems(config, items):
    markers = {
        "test_ts_": "ts",
        "test_gbdt_": "ml",
        "test_linear": "ml",
        ...
    }
    for item in items:
        for prefix, marker in markers.items():
            if item.nodeid.startswith(f"tests/{prefix}"):
                item.add_marker(marker)
                break
```

**Workflow:**

```bash
# 1. Итерация при патче в ts/models.py
uv run pytest -m "ts or guards" --timeout=60

# 2. Итерация при патче в ops/join.py  
uv run pytest -m "ops or guards"

# 3. Промежуточный gate
uv run pytest -m "not slow" -x -n auto

# 4. Финальный gate перед commit (как сейчас)
uv run pytest -m "not slow"

# 5. Только slow / e2e
uv run pytest -m slow
```

**`pytest-xdist`**: 131s → ~40-50s на 4 ядра. Требует всех тестов parallel-safe (tmp_path у всех уже есть). Проверить: `uv add --dev pytest-xdist` + `pytest -n auto`.

**Per-test timing assertion** (в `conftest.py`):

```python
def pytest_runtest_logreport(report):
    if report.when == "call" and report.duration > 1.0:
        if "slow" not in report.keywords:
            warnings.warn(f"{report.nodeid} took {report.duration:.2f}s — mark it slow")
```

**Acceptance:** 
- `pytest -m "ts"` — ~3s
- `pytest -m "ml"` — ~15s
- `pytest -m "not slow"` — ~130s
- `pytest -n auto -m "not slow"` — ~40s

---

## §14 Folder structure proposal

```
# Root (unchanged)
README.md, CHANGELOG.md, DECISIONS.md (index), pyproject.toml, uv.lock

# New: docs/ (см. §12)
docs/

# scripts/ — reorganized
scripts/
├── service/                        # stable entry points
│   ├── collect_project.py
│   ├── verify_ts_ram.py
│   ├── verify_whitelist.py
│   └── _lib.sh
├── _measure/                       # measurement harnesses
│   ├── measure_stage4.py
│   ├── measure_index.py
│   └── bench_*.py
├── _prototypes/                    # algorithmic prototypes
│   ├── proto_apply.py
│   ├── proto_roc_auc.py
│   └── ...
├── _tmp/                           # temp scripts (gitignored)
│   └── .gitkeep
├── archive/
│   ├── v0.1.0/                     # (existing)
│   ├── v0.8.7/                     # (existing)
│   └── patches/                    # ← NEW: old patch_v09*
│       ├── v0.9.0-0.9.20/
│       └── ...
└── patch_v0_9_25_*.py              # ← current phase only

# _FOR_PINGIS/ — keep, but clean old run artifacts
_FOR_PINGIS/
├── AUDIT_FULL_v0922.md             # latest audit
├── part1_report.md
├── part2_taxonomy.md
├── part2_todo_roadmap.md
├── ts_review_snapshot.md
├── _archive/                       # ← NEW: old audit runs
│   ├── v0.8.8_library_memory/
│   ├── v0.9.14_round1/
│   └── v0.9.22_round3/
└── _scratch/                       # current work (gitignored)
```

**`.gitignore` additions:**
```
scripts/_tmp/
_FOR_PINGIS/_scratch/
.backups/
```

---

## §15 `collect_project.py` proposal

**Новые режимы:**

```bash
# Phase-based (reads docs/roadmap.json)
uv run python scripts/service/collect_project.py --phase A
# → cadw/ts/models.py (full) + 4 tests (full) + 2 docs (full) + facade (interface) + core (interface)

# Module-based
uv run python scripts/service/collect_project.py --module cadw/ts/models.py
# → cadw/ts/models.py + test_ts_models.py + facade (interface) + __init__.py

# Module + tests only (fastest for review)
uv run python scripts/service/collect_project.py --module cadw/ts/models.py --no-core

# Doc-based
uv run python scripts/service/collect_project.py --doc docs/recipes/timeseries.md

# Full project (existing behavior)
uv run python scripts/service/collect_project.py --mode smart
```

**`docs/roadmap.json`** — machine-readable:

```json
{
  "phases": {
    "A": {
      "version": "v0.9.25",
      "description": "Stabilization",
      "modules": ["cadw/ts/models.py", "cadw/ops/sort.py", "cadw/dataframe.py", "cadw/core/memory_guard.py"],
      "tests": ["tests/test_ts_models.py", "tests/test_sort_merge_p*.py", "tests/test_ts_features.py"],
      "docs": ["docs/roadmap.md"],
      "interfaces": ["cadw/dataframe.py", "cadw/__init__.py"],
      "core": false
    },
    "B": {
      "version": "v0.9.26-v0.9.27",
      "description": "Infrastructure refactor",
      "modules": ["cadw/dataframe.py", "cadw/dataframe/_io.py", "..."],
      "core": true
    }
  }
}
```

`--module` — читает `cadw/<path>` + `tests/test_<basename>*.py` + interface-list + `__init__.py` + facade signatures. Без core по умолчанию.

`--phase` — читает манифест.

**Acceptance:** 
- `--phase A` = ~20 файлов, ~30 KB (vs ~2 MB full)
- `--module cadw/ts/models.py` = ~5 файлов, ~10 KB
- Interfaces извлекаются через `ast.parse` (только `def`/`class`/`import`/`__all__`)

---

## §16 Cleanup checklist (Phase B.6)

```bash
# 1. Backups — delete all, use git
rm -rf .backups/

# 2. Old patches — move to archive, keep current phase + previous 2
mkdir -p scripts/archive/patches/v0.9.0-0.9.20
git mv scripts/patch_v09{0,1,2}*.py scripts/archive/patches/v0.9.0-0.9.20/ 2>/dev/null || true
# keep: scripts/patch_v0923*.py (last 3)
# move: everything older

# 3. _FOR_PINGIS/archive old runs
mkdir -p _FOR_PINGIS/_archive
git mv _FOR_PINGIS/_out_*.json _FOR_PINGIS/_archive/ 2>/dev/null || true
git mv _FOR_PINGIS/_measure_*.py _FOR_PINGIS/_archive/ 2>/dev/null || true
git mv _FOR_PINGIS/_trace_*.py _FOR_PINGIS/_archive/ 2>/dev/null || true

# 4. scripts/ reorganization
mkdir -p scripts/{service,_measure,_prototypes,_tmp}
git mv scripts/collect_project.py scripts/service/
git mv scripts/verify_ts_ram.py scripts/service/
git mv scripts/run_ts_ram.py scripts/service/
git mv scripts/diag_rolling.py scripts/_measure/
git mv scripts/_measure_*.py scripts/_measure/
git mv scripts/proto_*.py scripts/_prototypes/
git mv scripts/_lib.sh scripts/service/

# 5. .gitignore
echo "scripts/_tmp/" >> .gitignore
echo "_FOR_PINGIS/_scratch/" >> .gitignore

# 6. Version sync — B6
# README.md: change v0.8.8 / 0.9.24 to __version__
# cadw/__init__.py: __version__ = "0.9.24" (after Phase A)
# All docs point to single source
```

**Acceptance:** `git status` clean; `scripts/` содержит только текущий цикл (≤5 patch_); `_FOR_PINGIS/` — только актуальные артефакты.

---

## §17 Immediate next action

**Начать с Phase A (v0.9.25), 1 патч, ~250 LOC + тесты.** Это блокирует всё остальное (3 critical SARIMA бага делают TS-тулкит невалидным для `D>0` и ломают THF-WLS).

**Порядок работ:**
1. **A.1–A.3** (SARIMA bugs) — 60 LOC + 3 теста. Критично.
2. **A.4–A.6** (B1, B2, B8) — 80 LOC + 3 теста. Высокие RAM-риски.
3. **A.7** (external_sort fan-in floor) — проверить, применить ли.
4. **A.8** (missing tests) — 50 LOC тестов.
5. **A.9–A.10** (B5 dead code, B4 fallback factor) — cleanup.
6. Commit `v0.9.25: stabilization — 3 critical SARIMA bugs + B1/B2/B8 audit`.

**Потом — Phase B (v0.9.26) = `dataframe.py` split.** Это блокер для всего, что идёт дальше (без split каждый следующий патч требует 50k+ токенов контекста).

---

## Что в этой consolidation **не** вошло (осознанно)

- **T5/T6/T7** (NCF/SASRec, BERT fine-tune, CV) — только recipes в Phase E, не реализация. Согласуется с принципом «не раздувать зоопарк».
- **T8/T9/T10** (GNN, RL, generative) — только recipes, out of scope.
- **7.C.1 lazy expression API** — отложен в Phase H P2; аргумент в §5.0 AUDIT §5 (fusion для wall time, не RAM) остаётся в силе.
- **External sort heap → Arrow batched merge** — уже сделано в v0.9.23c; §A.7 = только floor tuning, если надо.
- **`sklearn` список «заимствовать»** — не существует; в `§3.9` принцип «своё, не адаптер». Каждый D.* item — своё.

---

## Следующий шаг — что можно сделать сейчас

Если готов — могу сгенерировать сразу:

1. **`docs/ROADMAP.md`** — полный документ (этот текст развёрнутый, ~400-500 строк)
2. **`docs/roadmap.json`** — machine-readable манифест фаз
3. **`docs/index.md`** — новый DECISIONS index
4. **`scripts/service/collect_project.py`** — phase-aware version
5. **`scripts/patch_v0925_stabilize.py`** — patch-скрипт для Phase A

Или сначала — обсудить какие-то items (например, приоритеты внутри Phase D, или что именно считать «recipe» vs «implementation»). Какие-то фазы готов пересмотреть, если есть сомнения.