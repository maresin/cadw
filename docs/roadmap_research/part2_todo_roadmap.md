# CADW: TODO-roadmap по итогам таксономии (Часть 2)

Дата: 2026-10-22. Связан с docs/part1_report.md (A/B/C-список) и docs/part2_taxonomy.md (полная таксономия).

Правило: каждый тип задач, помеченный **TODO** в таксономии, получает здесь **конкретный план** (стриминговый вариант / reduce-алгоритм / дистилляция), а не «TODO».
Каждый пункт **NEEDS** — конкретная функция, которая ложится на примитивы Reduce/HashShuffle/Sort и влезает в 1.9 GB.

Столбцы: приоритет (P0 — блокирует типы задач на больших данных; P1 — сильный perf/качество; P2 — расширение), тип задач, что делаем, как (примитив/state/RAM), критерий готовности.

## 0. Сводка

- **Уже OK** (влезает, есть реализация): табличные классификация/регрессия на собственных GBDT/Linear/NB, k-means, PCA (через cov), time-series (naive/SARIMA/GBDT/THF/backtest), EDA/агрегации, CV-сплиты, imputation (mean/median), TS-фичи.
- **NEEDS** (реализуемо на примитивах, конкретная функция): ~40 пунктов, из них P0 — 12 (см. §1).
- **TODO** (в текущем виде не решается при 1.9 GB; план — стриминг/сэмпл/дистилляция): 12 пунктов (§2).

---

## 1. NEEDS -> конкретные функции (реализуемо на примитивах)

### P0 — блокируют целые типы задач на больших данных

| # | Тип задач | Что | Как (примитив, state, RAM) | Готово когда |
|---|---|---|---|---|
| N1 | A1,A2,A3,A6, G1, H1, H2, I1 | **sample_weight в GBDT/Logistic/Linear** | Reduce с весами: state O(1)/O(d), гистограммы градиентов уже считаются — веса в сумму/квантиль | weighted fit = sklearn-эквивалент на эталоне |
| N2 | A7, D1, D2 | **pinball/quantile loss в GBDT** | градиент/гессиан pinball на батч; state O(1) | quantile-прогноз с покрытием ~q на синтетике |
| N3 | D4 | **MiniRocket как streaming-преобразование** | случайные свёртки на батч, PPV, state O(#kernels)=фикс; затем GBDT/linear на признаках | UCR-подобный датасет: accuracy в пределах 2% от эталона |
| N4 | D5 | **PELT online change-point** | state O(window), Reduce по префиксным суммам; штраф BIC | точно детектит 3 точки на синтетике |
| N5 | E1 | **Isolation Forest на батчах** | t деревьев, subsample ψ=256; state = t·O(ψ), влезает | AUC на ODDS-подобном сэмпле >= sklearn IForest |
| N6 | E3 | **Matrix Profile (STUMPY-подобный) streaming** | state O(window), внешний sort для top-k соседей | находит 5 аномалий на синтетике |
| N7 | F1 | **ALS (implicit) как итеративный Reduce** | чередование: фикс V — решаем U (Reduce по user); state O((U+V)·k) | RMSE/NDCG на MovieLens-сэмпле ~ implicit ALS |
| N8 | G1 | **LambdaMART lambdarank objective** | пары внутри query-группы, HashShuffle по query_id, градиенты в GBDT | NDCG@10 на LETOR-подобном наборе |
| N9 | H1 | **T/S-learner + propensity** | две GBDT (по группам) или GBDT+propensity (logistic есть); state O(model) | ATE-оценка в пределах CI на синтетике с известным эффектом |
| N10 | I1 | **CoxPH как Reduce** | partial likelihood: сортировка по времени (Sort), risk-set суммы (Reduce); state O(d) | c-index ~ lifelines CoxPH на эталоне |
| N11 | J1 | **TF-IDF/хэширование текста на диске** | HashShuffle по токену, counts Reduce; разреженный словарь на диске | linear text classifier = sklearn-эквивалент по accuracy |
| N12 | O3 | **target/frequency encoding на диске** | HashShuffle по категории + Reduce (sum,count); out-of-fold через KFold | нет утечки, совпадает с category_encoders на эталоне |

### P1 — сильный perf/качество, но не блокирует тип

| # | Тип | Что | Как | Готово когда |
|---|---|---|---|---|
| N13 | A5 | ordinal logistic (cumulative link) | Reduce: K-1 порогов, state O(d·K) | MAE по рейтингам ~ mord OrdinalRidge |
| N14 | A4 | classifier chains | OvR + последовательное добавление предсказаний; Reduce по метке | subset accuracy >= OvR baseline |
| N15 | B1 | MiniBatchKMeans / GMM (EM) | центроиды Reduce; GMM — sum/sum-sq по кластеру | inertia/GMM-likelihood = sklearn MiniBatch |
| N16 | B3 | BIRCH CF-tree streaming | CF-nodes state O(#nodes), merge по порогу | совпадает с Birch на сэмпле |
| N17 | C1 | randomized SVD (PCA для больших d) | O(N·d·k): батчи A·Ω, затем малый QR/SVD | объяснённая дисперсия ~ sklearn PCA |
| N18 | D1 | ETS (Holt-Winters) | state O(k) сезонностей, Reduce по рядам | MASE ~ statsmodels ETS |
| N19 | D1 | Theta-метод | state O(1), Reduce | M3-подобный MASE |
| N20 | D2 | VAR(p) | state O(p·k²), Reduce по времени | RMSE ~ statsmodels VAR |
| N21 | D1/D2 | N-BEATS-блоки (лёгкая NN) | `to_batches(format='torch')`, обучение вне примитивов, инференс батчами | MAE ~ эталон на синтетике |
| N22 | E2 | cost-sensitive GBDT (веса — N1) + threshold tuning | Reduce по калибровке | PR-AUC >= baseline |
| N23 | F2 | hybrid MF+content | ALS + content features (N12) в GBDT-ранкер | NDCG лучше ALS-only |
| N24 | H2 | uplift trees (Qini) | критерий split по uplift; state O(trees) | Qini-коэф > T-learner |
| N25 | O4 | mutual_info / variance threshold / Boruta | биннинг+Reduce; shadow features+GBDT | top-k совпадает с sklearn |
| N26 | O5/O10/O11 | drift (KS/PSI/Chi2), A/B-тесты (t/Mann-Whitney/CUPED) | Reduce O(1) state | p-value ~ scipy на эталоне |
| N27 | O8 | permutation importance | батчи: перемешать колонку, predict, метрика — Reduce | ранжирование = sklearn permutation_importance |
| N28 | O9 | fairness-метрики (DP/EO/TPR-gap) | Reduce по (group, pred) | = fairlearn на эталоне |
| N29 | O13 | гео-агрегации (h3/hexbin) | Reduce по h3-ячейке (hexbin уже есть) | карта = h3-эквивалент |
| N30 | J4 | LDA (variational/EM) | итеративный Reduce, state O(K·V); для V<=2e5, K<=50 — влезает (float32 ~ 40 MB) | perplexity ~ gensim на сэмпле |
| N31 | M1 | LinUCB contextual bandit | state O(d²) на arm, Reduce по обновлениям | regret ~ baseline на синтетике |
| N32 | N1 | GMM density (EM) | Reduce sum/sum-sq, state O(K·d²) | log-likelihood ~ sklearn GMM |
| N33 | A1 | calibration (Platt/isotonic) | Platt — 2-параметрическая logistic (Reduce); isotonic — PAVA на батчах (Sort+Reduce) | Brier/ECE не хуже sklearn CalibratedClassifierCV |

### P2 — расширения/паритет

| # | Тип | Что | Как |
|---|---|---|---|
| N34 | O2 | IterativeImputer (EM на корреляциях) | state O(d²); Reduce по парам |
| N35 | O6 | стратифицированный reservoir sampling | расширение `core/sampling.py` (есть) |
| N36 | O7 | CV-обёртка для hyperparameter search | Pipeline+CV уже есть; добавить grid/random |
| N37 | O12 | partial_fit-интерфейс на GBDT/Logistic | онлайн-обновление state O(model) |
| N38 | A10 | Arrow dictionary IPC для category | словарь на диск, join |
| N39 | B2 (join) | кэш value_counts per partition-pair | кэш per pair |
| N40 | A1 | approx_quantile (t-digest/KLL) | merge-able state O(1/δ) |

---

## 2. TODO -> план для больших данных + 1.9 GB

| # | Тип | Почему в текущем виде не решается | План (стриминг/reduce/дистилляция) | Критерий |
|---|---|---|---|---|
| T1 | B2 иерархическая кластеризация | linkage O(N²) in-memory | **Сэмпл 5-10k → linkage → assign остальных по nearest centroid батчами** (mini-batch assign). Альтернатива: BIRCH (N16) как первый уровень, linkage на CF-центроидах. | дендрограмма на сэмпле + потоковое назначение, RAM O(sample²) |
| T2 | C2 t-SNE/UMAP | kNN-граф O(N·k) | **Сэмпл 5-10k → UMAP/t-SNE (fit) → transform батчами** (UMAP умеет `transform`); для очень больших — faiss-ANN для kNN на батчах, затем параметрический UMAP. | визуализация на сэмпле + стабильное вложение на потоке |
| T3 | B1 HDBSCAN | kNN-граф + MST O(N) памяти | **faiss-ANN-сэмпл** + HDBSCAN на ядре; assignment остальных по core-distance. Либо BIRCH→HDBSCAN на CF. | кластеры = HDBSCAN на сэмпле, RAM O(sample·k) |
| T4 | E1 One-Class SVM | kernel O(N²) | **Stochastic OC-SVM** (SGD на батчах) или сэмпл для fit + decision_function батчами. Оценка: OC-SVM fit только на сэмпле 10-20k. | AUC ~ sklearn OC-SVM на сэмпле |
| T5 | F1 NCF / SASRec | deep, state O(params) | Не портируем в примитивы: **torch-батчи** через `to_batches(format='torch')`, обучение вне CADW; если нужен out-of-core — distillation в ALS (N7). | NDCG >= ALS-baseline |
| T6 | J1/J5 fine-tune BERT/LLM | глубокое обучение | Вне scope. Только стриминг данных в torch (батчи). Для дешёвого варианта — **дистилляция в fastText/линейный на TF-IDF (N11)**. | linear/fastText в пределах X% от BERT (фиксируем порог по задаче) |
| T7 | K1-K3 CV (detection/segmentation) | глубокое обучение | Вне scope; torch-батчи. Для классификации на больших данных — **дистилляция в лёгкий backbone (MobileNet/Distil) или в эмбеддинги + GBDT**. | лёгкая модель >= X% от тяжёлой |
| T8 | L1 GNN (node/graph) | глубокое, разреженные графы | **Классические итерации на Reduce**: label propagation, PageRank, degree/community (через разреженные операции батчами). GNN — вне scope, только torch-батчи (sampling соседей на диске). | метрики = networkx на эталоне |
| T9 | M1 RL | среда+агент | Вне scope. **Bandits (LinUCB, N31) — Reduce**, это покрывает практическую часть (recsys/ads). | regret ~ baseline |
| T10 | N1-N2 генеративные (VAE/GAN/diffusion/LLM) | глубокая генерация | Вне scope. Для плотностной оценки — **GMM (N32)**; для табличных синтетических данных — CTGAN вне CADW через батчи. | — (явно вне scope) |
| T11 | H1 causal forest | O(trees) — реализуемо, но не приоритет | Расширение GBDT: honest splits + критерий по treatment-эффекту (как uplift N24, но с honest). | ATE ~ EconML CausalForest на сэмпле |
| T12 | I1 Random Survival Forest | O(trees) | Расширение GBDT: split по log-rank; state O(trees·leaves). | c-index ~ scikit-survival RSF на сэмпле |

---

## 3. Лёгкие нейросети: где включаем в арсенал (не переключаясь на чужой фреймворк)

Принцип: NN используем только там, где (a) на таблицах GBDT уступает незначительно, но NN даёт **диверсификацию ансамбля**, или (b) тип задач принципиально нейросетевой и есть компактный вариант (<5M параметров), обучаемый батчами из `to_batches(format='torch')` без материализации данных.

| Тип задач | Лёгкая NN | Параметры | Зачем в ансамбле |
|---|---|---|---|
| Табличная классификация/регрессия (A1-A3) | MLP 2-3 слоя / TabNet-lite | 0.05-2M | нелинейные взаимодействия, отличные от деревьев |
| Временные ряды (D1-D2) | N-BEATS-блоки, dilated-CNN | 0.1-1M | долгие зависимости, сезонность |
| Классификация TS (D4) | 1D-CNN / InceptionTime-lite | 0.1-1M | форма, а не интервалы |
| Anomaly (E1) | AE-reconstruction | 0.1-1M | нелинейная граница |
| Текст (J1) | fastText / MiniLM / DistilBERT | 0.1-66M | контекст vs лексика |
| Рекомендации (F1) | SASRec-lite | 0.1-2M | последовательности |
| Survival (I1) | DeepSurv MLP | ~0.1M | нелинейный hazard |
| Uplift (H2) | T-learner MLP | ~0.1M | диверсификация |
| Графы (L1) | (не берём) | — | только классические Reduce |
| CV (K1-K3) | MobileNet/Distil (inference), дообучение — вне | 1-5M | только если K-задача критична |

Правило: **любая лёгкая NN обучается на батчах, инференс — тоже батчами; в память данные целиком не поднимаются**. Это совместимо с текущей архитектурой `to_batches(format='torch')`.

---

## 4. Что в первую очередь (порядок работ)

1. **P0-фиксы из part1**: B1 (refill линейный), B2 (encode_categorical K_est+check_state), A5 (check_python_objects / extra_factor по типу) — дёшево, снимает главный RAM-риск.
2. **N1, N2, N12** (веса, pinball, target-encoding) — открывают A6/A7/D-квантили/O3 и нужны почти везде.
3. **N3, N5, N7, N8** — открывают целые типы (TS-классификация, anomaly, recsys, ranking).
4. **N9, N10** — causal/survival (важные, но узкие).
5. **T1-T4, T8** — обёртки «сэмпл+поток» для типов, которые принципиально O(N²)/глубокие; реализуются как отдельный слой поверх уже готовых примитивов, без переписывания ядра.
6. **Лёгкие NN (N21, N33)** — в последнюю очередь, как опция для ансамбля.

---

## 5. Трассировка «таксономия -> roadmap»

Каждый TODO/NEEDS из part2_taxonomy.md имеет здесь строку:
- A1-A7 -> N1,N2,N13,N14,N33
- B1-B3 -> N15,N16,T1,T3
- C1-C3 -> N17,T2,N21
- D1-D5 -> N2,N18,N19,N20,N21,N3,N4
- E1-E3 -> N5,N6,N22,T4
- F1-F2 -> N7,N23,T5
- G1 -> N8
- H1-H2 -> N9,N24,T11
- I1 -> N10,T12
- J1-J5 -> N11,T6,T8? (J — T6)
- K1-K4 -> T7
- L1-L2 -> T8
- M1 -> N31,T9
- N1-N2 -> N32,T10
- O1-O13 -> N12,N25,N26,N27,N28,N29,N30,N34-N37
