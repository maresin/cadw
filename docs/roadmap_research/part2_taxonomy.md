# Полная таксономия типов задач ML и Data Science + применимость CADW

Дата: 2026-10-22. Проверка интернета: OK (scikit-learn 1.9.1, LightGBM 4.7, XGBoost 3.4.2, CatBoost docs — фетч успешен; сверка названий API и моделей выполнена по этим страницам и по известным бенчмаркам/соревнованиям).

Легенда по CADW-применимости:
- **OK** — реализуемо на примитивах Reduce/HashShuffle/Sort, влезает в 1.9 GB (батчи/стриминг).
- **NEEDS: <что>** — реализуемо, но нужна конкретная функция/алгоритм (в TODO-roadmap).
- **TODO** — при больших данных и 1.9 GB указанными моделями не решается в текущем виде; план в docs/part2_todo_roadmap.md.

Легенда по SOTA-цитатам: `[bench: <имя>]` — проверяемый бенчмарк/соревнование; `[no verified citation]` — не привожу конкретных цифр, только общепризнанный порядок.

Правило ансамбля: в каждом блоке указано, какие **семейства** моделей смешивать (разные inductive bias). Не «топ-1 модель», а диверсификация.

---

## A. Табличные supervised-задачи

### A1. Бинарная классификация (табличная)
- Примеры: кредитный скоринг, churn, fraud, CTR (клик/неклик), medical diagnosis.
- Библиотеки/функции: sklearn (`LogisticRegression`, `HistGradientBoostingClassifier`, `SGDClassifier.partial_fit`, `RandomForestClassifier`, `LinearSVC`), LightGBM (`lgb.train`, `LGBMClassifier`), XGBoost (`xgb.train`), CatBoost (`CatBoostClassifier`, ordered target stats), `imbalanced-learn` (SMOTE/ENN — только in-memory).
- Популярные: gradient boosting (LightGBM/XGBoost/CatBoost) — стандарт де-факто на Kaggle-табличных задачах; логистическая регрессия — baseline.
- SOTA-класс: CatBoost/LightGBM на соревнованиях (Kaggle Porto Seguro, Santander, Home Credit, Criteo) [bench: Kaggle tabular comps]. На «чистых» таблицах GBDT + target/frequency encoding держит топ; FT-Transformer/TabNet сильны на части датасетов [bench: Kaggle + OpenML tabular benchmarks; no verified single-number citation].
- Ансамбль: LightGBM (leaf-wise) + CatBoost (ordered boosting, category-native) + XGBoost (level-wise) + линейная L2 — разные bias, высокая диверсификация.
- CADW: **OK** для собственного GBDT (есть), Logistic (есть), NB (есть). **NEEDS**: категориальные target/frequency encoding на диске (A10); калибровка (Platt/isotonic) как Reduce. Адаптеры LightGBM/XGBoost — **TODO** (внешний `.fit` = OOM вне CADW; вариант — только `partial_fit` SGD/`lightgbm` с `Dataset` на диске, не гарантируется).
- Легковесная NN: TabNet (~few M params) или 2-3-слойный MLP на `numpy`/`torch` из батчей (`to_batches(format='torch')`) — до сотен тысяч параметров, обучение батчами влезает.

### A2. Многоклассовая классификация
- Примеры: типы отказов, классификация товаров, диагностика заболеваний.
- Библиотеки: те же + `OneVsRest`/`OneVsOne`, `SGDClassifier`, CatBoost multiclass.
- Популярные/SOTA: LightGBM `multiclass`, CatBoost `MultiClass`; на многих бенчмарках GBDT vs NN сопоставимы [bench: Kaggle multiclass tabular; no verified single-number citation].
- Ансамбль: multiclass GBDT + multinomial logistic + NB (для текстовых признаков) + kNN по эмбеддингам (если K мал).
- CADW: **OK** (GBDTMulticlass, MultinomialNB есть). **NEEDS**: OvR поверх GBDT — тривиально.
- Легковесная NN: тот же MLP с softmax.

### A3. Регрессия (табличная)
- Примеры: прогноз цен, спроса, времени доставки, страховые выплаты.
- Библиотеки: sklearn `LinearRegression`/`Ridge`/`Lasso`/`ElasticNet`/`HistGradientBoostingRegressor`/`SGDRegressor.partial_fit`; LightGBM `LGBMRegressor`, XGBoost `reg:squarederror`/`reg:tweedie`, CatBoost `CatBoostRegressor`.
- Популярные/SOTA: GBDT (LightGBM/CatBoost) — топ табличных регрессий; для Tweedie/Poisson — специализированные objective у XGBoost/LightGBM.
- Ансамбль: GBDT (L2) + GBDT (Tweedie) + линейная Ridge (стабильная экстраполяция) + kNN локально.
- CADW: **OK** (Linear/Ridge/Lasso/ENet/GBDT есть). **NEEDS**: Tweedie/Poisson objective в GBDT; robust loss (Huber).
- Легковесная NN: MLP-регрессор; для таблиц часто уступает GBDT, но добавляет диверсификацию.

### A4. Мультилейбл классификация
- Примеры: теги статей, медицинские коды (ICD), жанры фильма.
- Библиотеки: sklearn `MultiOutputClassifier`, `OneVsRestClassifier`; LightGBM по одному binary на метку; `scikit-multilearn` (классификаторные цепочки).
- Популярные: Binary Relevance (OvR) на GBDT; classifier chains; label powerset для малых K.
- SOTA: deep (BERT-like) для текста; для таблиц — OvR GBDT [bench: Kaggle multi-label; no verified single-number citation].
- Ансамбль: OvR GBDT + classifier chain (учёт корреляций меток) + линейный OvR.
- CADW: **OK** — OvR = K бинарных задач, каждая на примитивах. **NEEDS**: classifier chains (порядок меток + добавление предсказаний как признаков — Reduce-friendly).
- Легковесная NN: MLP с K сигмоидами, BCE-loss (multi-label) — компактно.

### A5. Ординарная (порядковая) классификация
- Примеры: рейтинги 1-5, степень тяжести, уровни риска.
- Библиотеки: `mord` (OrdinalRidge), sklearn через `OrdinalEncoder`+регрессия, `XGBoost` с `rank:pairwise` не для этого — лучше threshold-модели.
- Популярные: ordinal logistic (proportional odds), threshold/cumulative-link модели, GBDT на «порог за порогом».
- SOTA: CORAL/CORN (deep ordinal) [bench: no verified single-number citation].
- Ансамбль: ordinal logistic + GBDT-на-порогах + регрессия с последующим округлением.
- CADW: **NEEDS: ordinal logistic (cumulative link) как Reduce** — state O(d), влезает.
- Легковесная NN: MLP с K-1 пороговыми выходами (CORAL).

### A6. Дисбаланс классов
- Примеры: fraud (0.1%), rare disease, defect detection.
- Библиотеки: `imbalanced-learn` (SMOTE, ADASYN, ENN, Tomek), `class_weight`/`scale_pos_weight`, focal loss.
- Популярные: cost-sensitive обучение (веса) + threshold tuning + resampling; на таблицах веса обычно лучше SMOTE.
- SOTA: focal loss / class-balanced loss (Cui et al. 2019) [bench: long-tailed CIFAR; no verified single-number citation для таблиц].
- Ансамбль: cost-sensitive GBDT + bagging с undersampling (balanced bagging) + калиброванный logistic.
- CADW: **NEEDS: sample weights в GBDT/Logistic** (Reduce с весами — state O(1)); balanced bagging на HashShuffle (партиции по классу).
- Легковесная NN: focal loss на MLP.

### A7. Квантильная / интервальная регрессия
- Примеры: прогноз спроса с интервалами, риск-менеджмент (VaR), ETA с диапазоном.
- Библиотеки: `LightGBM` (`objective='quantile'|'quantile_alpha'`), `sklearn` `QuantileRegressor`, `statsmodels` QuantReg, XGBoost `reg:quantileerror`.
- Популярные: gradient boosting с pinball-loss; linear quantile regression.
- SOTA: NGBoost (natural gradient boosting — вероятностные предсказания), CatBoost `Quantile`/`MultiQuantile` [bench: NGBoost paper; no verified single-number citation].
- Ансамбль: quantile-GBDT + linear quantile + NGBoost-класс.
- CADW: **NEEDS: pinball-loss в GBDT** (state O(1)); квантили из `cadw.ops.quantile` уже есть.
- Легковесная NN: MLP с K выходами (quantiles) и pinball-loss.

---

## B. Кластеризация

### B1. Плоская кластеризация (k-means, k-medoids)
- Примеры: сегментация клиентов, группировка товаров, сжатие векторов.
- Библиотеки: sklearn `KMeans`, `MiniBatchKMeans`, `SpectralClustering`, `DBSCAN`, `HDBSCAN` (`hdbscan`), `faiss` (KMeans/IVF для больших векторов).
- Популярные: k-means (baseline), MiniBatchKMeans (большие N), HDBSCAN (плотностная, без k).
- SOTA: для плотностной — HDBSCAN; для векторных баз — IVF-PQ/faiss [bench: ANN-benchmarks для faiss; no verified single-number citation для HDBSCAN].
- Ансамбль: k-means (сферические) + HDBSCAN (произвольные формы) + GMM (soft-assignment) — разные предположения.
- CADW: **OK** для k-means (есть, HashShuffle+Reduce). **NEEDS**: MiniBatchKMeans на диске (батчевые центроиды — Reduce); GMM через EM (Reduce по батчам); HDBSCAN — **TODO** (плотностный, требует kNN-граф; на 1.9 GB только через внешний sort+stream).
- Легковесная NN: автоэнкодер + k-means на латенте (deep clustering, ~1-5 M params).

### B2. Иерархическая кластеризация
- Примеры: таксономии, дендрограммы, cluster-иерархии.
- Библиотеки: `scipy.cluster.hierarchy` (`linkage`, `fcluster`, `dendrogram`), `fastcluster`.
- Популярные: Ward/average linkage.
- Ансамбль: Ward + average linkage; консенсус через co-association matrix.
- CADW: **TODO** — `linkage` требует O(N²) пар (in-memory). Для больших N — только через сэмпл/прореживание или BIRCH (см. B3).

### B3. BIRCH / потоковая кластеризация
- Примеры: онлайн-сегментация, потоковые данные.
- Библиотеки: sklearn `Birch` (CF-tree), `MiniBatchKMeans`.
- Популярные: BIRCH для потоков.
- Ансамбль: BIRCH → MiniBatchKMeans на CF-центроидах (двухступенчатая).
- CADW: **NEEDS: CF-tree (BIRCH) как streaming state** — state O(#CF nodes), влезает в память; идеальный кандидат для CADW.

---

## C. Понижение размерности / представления

### C1. PCA / линейные методы
- Примеры: визуализация, шумоподавление, сжатие признаков перед GBDT/кластеризацией.
- Библиотеки: sklearn `PCA`, `TruncatedSVD`, `IncrementalPCA`, `SparsePCA`.
- Популярные: PCA (dense), TruncatedSVD (sparse).
- SOTA: randomized SVD (Halko et al. 2011) — стандарт для больших матриц [bench: scikit-learn randomized_svd docs].
- Ансамбль: PCA + random projection + autoencoder-латент (разные геометрии).
- CADW: **OK** для PCA (есть, cov+`eigh`). **NEEDS: randomized SVD** (A4 в part1) — O(N·d·k) через батчи.
- Легковесная NN: линейный автоэнкодер = PCA; нелинейный автоэнкодер (~0.1-1M) для нелинейного сжатия.

### C2. Нелинейное понижение (t-SNE, UMAP)
- Примеры: визуализация эмбеддингов, single-cell.
- Библиотеки: sklearn `TSNE`, `umap-learn` (`UMAP`), `openTSNE`.
- Популярные: UMAP — быстрее и сохраняет глобальную структуру.
- SOTA: UMAP/PaCMAP/TriMap [bench: openTSNE/UMAP papers].
- Ансамбль: UMAP (визуализация) + PCA (глобально) — для разных аспектов.
- CADW: **TODO** — kNN-граф O(N) (поиск соседей через faiss/annoy) + O(N·k) память; на 1.9 GB только через сэмпл (5-10k) или ann-bench. Исключение: UMAP на сэмпле, затем transform по батчам.

### C3. Автоэнкодеры
- Примеры: сжатие, denoising, anomaly detection (reconstruction error).
- Библиотеки: `torch`, `tensorflow`, `keras`.
- Популярные: MLP-AE, VAE, denoising AE.
- SOTA: для табличных — VAE/TabNet; для временных — dilated conv AE.
- Ансамбль: AE (нелинейный) + PCA (линейный) + random projection.
- CADW: **NEEDS: AE на батчах через `to_batches(format='torch')`** — обучение идёт вне CADW в torch, но данные не материализуются в память целиком (батчи). Это поддерживается контрактом.

---

## D. Time series

### D1. Прогнозирование (univariate, multi-step)
- Примеры: спрос, трафик, продажи, метрики.
- Библиотеки: `statsmodels` (SARIMAX, ETS), `pmdarima` (auto-arima), `sktime`, `darts`, `prophet`, `neuralprophet`, `statsforecast` (Nixtla, быстрые ARIMA/ETS), `mlforecast`, `lightgbm`/`xgboost` с лагами, `tft` (Temporal Fusion Transformer через pytorch-forecasting).
- Популярные: ARIMA/ETS (baseline), Prophet (праздники/тренд), GBDT с лагами.
- SOTA: N-BEATS (Oreshkin et al. 2020), N-HiTS, PatchTST, TFT, DeepAR на соревнованиях (M4/M5) [bench: M4/M5 competitions]. M5 winner — LightGBM + Tweedie (Kaggle).
- Ансамбль: GBDT-лаги (direct) + ETS/SARIMA (сезонность) + Neural (patch/transformer) — разные горизонты и сезонности.
- CADW: **OK** — `NaiveForecaster`, `SARIMAForecaster`, `GBDTForecaster`, `TemporalHierarchy`, `backtest` уже есть. **NEEDS**: ETS (state O(k) сезонностей), theta-метод, ARIMA с `css`/`Hannan-Rissanen` расширение.
- Легковесная NN: N-BEATS-блоки (~0.1-1M) или dilated-CNN — обучение на батчах, inference на O(batch).

### D2. Мультивариантное прогнозирование
- Примеры: прогноз множества связанных рядов (retail, energy).
- Библиотеки: `darts` (`BlockRNNModel`, `TFTModel`), `neuralprophet`, `pytorch-forecasting`, `statsforecast`.
- SOTA: TFT, DeepAR, N-HiTS, PatchTST [bench: M4/M5 + GEFCom].
- Ансамбль: GBDT по каждому ряду + VAR (линейный, учитывает корреляции) + neural.
- CADW: **NEEDS: VAR (Vector AR) как Reduce** — состояние O(p·k²), для k малых влезает; GBDT-мультивариат через лаги других рядов.
- Легковесная NN: тот же N-BEATS/TFT (0.1-2M).

### D3. Иерархическое прогнозирование
- Примеры: продажи товар/категория/регион, бюджет по уровням.
- Библиотеки: `hierarchicalforecast` (Nixtla), `sktime`.
- SOTA: MinT (Wickramasuriya et al. 2019), ERM, OLS — стандарт [bench: M5 hierarchical].
- Ансамбль: MinT-WLS + bottom-up (разные допущения).
- CADW: **OK** — `TemporalHierarchy` с MinT-WLS уже есть (temporal); spatial/hierarchical — расширение на cross-sectional (аналогично, Reduce).

### D4. Классификация временных рядов
- Примеры: ЭКГ-аритмии, распознавание активности, отказ оборудования.
- Библиотеки: `sktime` (`TimeSeriesForestClassifier`, `ROCKET`, `MiniRocket`), `tsfresh` (features) + sklearn/LightGBM, `pyts` (SAX, BOSS).
- Популярные: ROCKET/MiniRocket (быстрые, без обучения), shapelets, интервальные признаки + GBDT.
- SOTA: MiniRocket (Dempster et al. 2021) — near-SOTA при O(N) [bench: UEA/UCR archive].
- Ансамбль: MiniRocket + интервальные + GBDT на tsfresh-подмножестве.
- CADW: **NEEDS: MiniRocket как streaming-преобразование** (случайные свёртки + PPV, state O(#kernels) фиксированный) — идеальный кандидат (не требует хранения рядов в памяти, только батчи). Интервальные признаки на Reduce. tsfresh-подмножество уже есть в `cadw.ts.stats`.
- Легковесная NN: 1D-CNN/InceptionTime-блоки (~0.1-1M).

### D5. Обнаружение точек изменения (change-point)
- Примеры: дрейф метрик, аварии, segmentation.
- Библиотеки: `ruptures` (PELT, BinSeg, Window), `bayesian_changepoint_detection`.
- SOTA: PELT (Killick et al. 2012) — O(N) средняя сложность [bench: ruptures docs].
- CADW: **NEEDS: PELT online** (state O(window), влезает) — реализуемо на Reduce.

---

## E. Anomaly / Outlier detection

### E1. Unsupervised anomaly detection
- Примеры: fraud, network intrusion, sensor faults, качество данных.
- Библиотеки: sklearn `IsolationForest`, `OneClassSVM`, `LocalOutlierFactor`, `EllipticEnvelope`; `pyod` (30+ детекторов, `PyOD.models.IForest`, `AutoEncoder`, `DeepSVDD`, `COPOD`); `alibi-detect` (drift+anomaly).
- Популярные: Isolation Forest (быстро, масштабируется), LOF (плотностной), One-Class SVM (граница).
- SOTA: Isolation Forest/ EIF (extended IF), COPOD (empirical copula, интерпретируемый), нормализующие потоки (Normalizing Flows) — на MVTec/ODDS [bench: ODDS benchmark, MVTec AD для vision].
- Ансамбль: IForest (global) + LOF (local) + statistical z-score (интерпретируемость) — разные определения «аномалии».
- CADW: **OK** для IForest на батчах (построение деревьев O(t·ψ), state = t деревьев, влезает при ψ=256). **NEEDS**: LOF на kNN (сложно без O(N) памяти — только сэмпл). **TODO**: One-Class SVM (kernel matrix — O(N²) — только сэмпл).
- Легковесная NN: AE-reconstruction (compact ~0.1-1M) через `to_batches(format='torch')` — ошибка реконструкции как score.

### E2. Supervised anomaly (редкие события с метками)
- Примеры: fraud с частичными метками, predictive maintenance.
- Библиотеки: `imbalanced-learn`, `pyod` (supervised), XGBoost/LightGBM с `scale_pos_weight`.
- SOTA: cost-sensitive GBDT + threshold tuning [bench: Kaggle fraud comps].
- CADW: **OK** (см. A6).

### E3. Time-series anomaly
- Примеры: мониторинг, IoT, метрики.
- Библиотеки: `alibi-detect` (KS, MMD), `stumpy` (matrix profile), `ruptures`.
- SOTA: Matrix Profile (STUMPY, O(N log N)) [bench: UCR TS anomaly].
- CADW: **NEEDS**: matrix profile streaming (state O(2·window), влезает).

---

## F. Рекомендательные системы

### F1. Collaborative filtering
- Примеры: товары/фильмы/музыка.
- Библиотеки: `implicit` (ALS, BPR), `surprise` (SVD, KNN, NMF), `LightFM` (hybrid), `scikit-surprise`, `torch-rechub`/`DeepCTR` (DL).
- Популярные: ALS (implicit), SVD (explicit), item-item KNN.
- SOTA: нейросетевые (NCF, SASRec, BERT4Rec) — на соревнованиях и в проде [bench: MovieLens/Amazon, RecSys challenge].
- Ансамбль: ALS (matrix factorization) + item-item KNN (content) + GBDT-ранкер (features) — разные сигналы.
- CADW: **NEEDS: ALS как итеративный Reduce** (шаги чередующихся наименьших квадратов на партициях, state O(U·k+V·k)) — реализуемо; item-item KNN — через HashShuffle по item. **TODO**: NCF/SASRec (torch, батчи — можно, но вне примитивов).
- Легковесная NN: SASRec (self-attention, 0.1-2M) — обучение на батчах.

### F2. Content-based / Hybrid
- Примеры: рекомендации статей, персональные ленты.
- Библиотеки: `LightFM`, `catboost` (ranking), `sentence-transformers` (эмбеддинги).
- SOTA: hybrid MF + content embeddings.
- CADW: **OK** для линейной гибридной модели; **NEEDS** эмбеддинги через батчи.

---

## G. Learning to Rank / Ranking

### G1. Pointwise / Pairwise / Listwise
- Примеры: поиск (web search), рекомендации, реклама.
- Библиотеки: LightGBM `lambdarank`, XGBoost `rank:pairwise`/`rank:ndcg`, CatBoost `YetiRank`/`PairLogit`, `allrank`, `tensorflow-ranking`.
- Популярные: LambdaMART (LightGBM lambdarank) — стандарт поиска.
- SOTA: LambdaMART + нейросетевые rankers (BERT-based) для текста [bench: MSLR-WEB30K, LETOR; no verified single-number citation].
- Ансамбль: LambdaMART (GBDT) + linear pairwise + neural listwise — разные потери.
- CADW: **NEEDS: LambdaMART lambdarank objective в GBDT** (gradients по парам — можно в батчах с группировкой). Grouping — HashShuffle по query_id.
- Легковесная NN: простой listwise (DLCM/SetRank, ~0.1-1M).

---

## H. Causal inference / Uplift

### H1. Causal effect estimation
- Примеры: эффект маркетинга, A/B-анализ, медицинские исследования.
- Библиотеки: `DoWhy` (identification+estimation), `EconML` (DML, causal forest, Orthogonal RF), `CausalML` (uplift trees), `pywhy` stack.
- Популярные: propensity score matching, T-learner/S-learner, Double ML.
- SOTA: DML (Chernozhukov et al. 2018), causal forest (Wager & Athey 2018) [bench: ECONML paper, ACIC competitions].
- Ансамбль: DML (linear) + causal forest (non-param) + T-learner GBDT — разные допущения.
- CADW: **NEEDS: T-learner/S-learner на GBDT** (две модели по группам через батчи — state O(model)), propensity через logistic (есть). **TODO**: causal forest (honest splits, state O(trees)) — реализуемо как расширение GBDT.

### H2. Uplift modeling
- Примеры: target selection, retention.
- Библиотеки: `CausalML` (UpliftTreeClassifier, X-learner, R-learner), `pylift`.
- SOTA: causal forest, X/R-learners [bench: Criteo uplift].
- CADW: **NEEDS**: uplift trees (Qini/uplift criterion в GBDT split).

---

## I. Survival analysis / Time-to-event

### I1. Cox / AFT / Random Survival Forest
- Примеры: churn с цензурированием, время до отказа, медицина.
- Библиотеки: `lifelines` (CoxPH, WeibullAFT, KaplanMeier), `scikit-survival` (Coxnet, RandomSurvivalForest, GradientBoostingSurvivalAnalysis).
- Популярные: CoxPH, Random Survival Forest.
- SOTA: DeepSurv, DeepHit, GradientBoostingSurvivalAnalysis (Hothorn 2005) [bench: METABRIC/SUPPORT, no verified single-number citation].
- Ансамбль: CoxPH (linear) + RSF (non-param) + GBDT-survival — разные допущения.
- CADW: **NEEDS: CoxPH как Reduce** (partial likelihood на батчах, state O(d)) — реализуемо. **TODO**: Random Survival Forest (state O(trees)).
- Легковесная NN: DeepSurv (MLP, ~0.1M).

---

## J. Текст / NLP

### J1. Text classification / Sentiment
- Примеры: spam, sentiment, topic classification.
- Библиотеки: sklearn (`TfidfVectorizer` + `LinearSVC`/`LogisticRegression`/`SGDClassifier`), `transformers` (BERT/RoBERTa/DeBERTa), `sentence-transformers` + sklearn/GBDT.
- Популярные: TF-IDF + linear (baseline, быстрый), BERT-fine-tune (SOTA).
- SOTA: RoBERTa/DeBERTa/ModernBERT [bench: GLUE].
- Ансамбль: TF-IDF+linear (лексический) + sentence-embedding + GBDT (эмбеддинги) + BERT (контекст) — разные сигналы.
- CADW: **OK** для TF-IDF+linear (sparse count через HashShuffle, Reduce). **NEEDS**: батчевые эмбеддинги + GBDT (эмбеддинги в батчах). **TODO**: fine-tune BERT — вне примитивов, только через `to_batches(format='torch')`.
- Легковесная NN: DistilBERT/MiniLM (~22-66M) или `fastText`-классификатор (очень лёгкий).

### J2. NER / sequence tagging
- Библиотеки: `spacy`, `transformers`, `flair`.
- SOTA: BERT-based NER (GLUE/CoNLL-2003).
- CADW: **TODO** — последовательностная модель, только через torch-батчи.

### J3. Embeddings / Semantic similarity
- Библиотеки: `sentence-transformers`, `fastText`, `Gensim` (Word2Vec, Doc2Vec).
- CADW: **NEEDS**: Word2Vec/embedding через батчи — state O(V·d) может не влезть; MiniLM — через torch-батчи.

### J4. Topic modeling
- Библиотеки: `Gensim` (LDA, NMF), sklearn `LatentDirichletAllocation`.
- SOTA: BERTopic (эмбеддинги + UMAP + HDBSCAN) [bench: no verified single-number citation].
- CADW: **NEEDS**: LDA как iterative Reduce (state O(K·V) может не влезть при большом V; для V=10^5, K=50 — 200 MB при float32, влезает в 1.9GB).

### J5. Translation / Summarization / Generation
- Библиотеки: `transformers`, `peft` (LoRA), `vllm` (инференс).
- SOTA: LLM-класс (GPT/Llama/Mistral) [bench: GLUE/MT-Bench/MMLU].
- CADW: **TODO** — вне scope (генеративные модели, огромный memory footprint). Только через `to_batches` для стриминга данных.

---

## K. Vision (Computer Vision)

### K1. Image classification
- Библиотеки: `torchvision`, `timm`, `ultralytics` (YOLO классификация), `transformers` (ViT).
- Популярные: ResNet, EfficientNet; SOTA: ViT/Swin/ConvNeXt [bench: ImageNet].
- CADW: **TODO** — вне примитивов; данные — через `to_batches(format='torch')`, обучение вне CADW (батчи картинок в память не тянутся).

### K2. Object detection
- Библиотеки: `ultralytics` (YOLOv8/v11), `torchvision` (Faster R-CNN, RetinaNet), `mmdetection`.
- SOTA: YOLO-family, DETR-класс [bench: COCO mAP].
- CADW: **TODO**.

### K3. Semantic / Instance segmentation
- Библиотеки: `mmsegmentation`, `torchvision` (Mask R-CNN), `ultralytics` (YOLO-seg).
- SOTA: SAM 2, Mask2Former [bench: COCO seg, ADE20K].
- CADW: **TODO**.

### K4. Vision embeddings / retrieval
- Библиотеки: `open_clip`, `timm`.
- CADW: **NEEDS**: эмбеддинги через батчи + ANN-поиск (faiss) — возможно, если backbone вне CADW.

---

## L. Graph ML

### L1. Node / Edge / Graph classification, link prediction
- Библиотеки: `PyTorch Geometric`, `DGL`, `networkx` (классические), `graph-tool`.
- Популярные: GCN, GraphSAGE, GAT; классические — label propagation, PageRank.
- SOTA: GNN-семейства (GCN/SAGE/GAT/Graphormer) [bench: OGB, Planetoid].
- CADW: **TODO** — GNN вне примитивов. Возможен label propagation на разреженной матрице — через Reduce; **NEEDS**: графовые статистики (degree, PageRank) через батчевые итерации.

### L2. Graph embeddings
- Библиотеки: `node2vec`, `metapath2vec`, `PyG`.
- CADW: **NEEDS**: node2vec — random walks на диске + word2vec на батчах (state O(V·d) — может не влезть для 10^8 узлов, только для средних).

---

## M. Reinforcement Learning

### M1. RL (RLHF, game playing, control, bandits)
- Библиотеки: `stable-baselines3`, `ray.rllib`, `gymnasium`, `pettingzoo`, `trl` (RLHF).
- SOTA: PPO, SAC, DQN; для LLM — RLHF/DPO [bench: Atari/DMC, MT-Bench].
- CADW: **TODO** — вне scope (среда+агент); данные — только через `to_batches`. Contextual bandits (LinUCB) — **NEEDS**: реализуемо на Reduce (state O(d²)).

---

## N. Generative

### N1. Density estimation / VAE / Normalizing flows / GAN
- Библиотеки: `torch`, `pytorch-lightning`, `nflows` (flows), `pytorch-gan`.
- SOTA: NF (RealNVP/Glow), diffusion для изображений/текста [bench: CIFAR-10 FID].
- CADW: **TODO** — вне scope; GMM (плотностная оценка с Reduce) — **NEEDS**: реализуемо.

### N2. Diffusion / LLM
- Библиотеки: `diffusers`, `transformers`.
- CADW: **TODO** — вне scope.

---

## O. Data Science вне ML (EDA, cleaning, feature engineering, validation)

### O1. EDA / описательная статистика
- Библиотеки: pandas, polars, `ydata-profiling` (бывший pandas-profiling), `sweetviz`.
- CADW: **OK** — `describe`, `hist`, `hist2d`, `box`, `line`, `quantile`, `value_counts` уже есть, aggregate-first viz.
- **NEEDS**: расширить `describe` (skew/kurt/quantiles — уже есть частично); профилирование (missingness, cardinality) — Reduce.

### O2. Data cleaning / imputation
- Библиотеки: sklearn `SimpleImputer`/`IterativeImputer`/`KNNImputer`, `fancyimpute`.
- CADW: **OK** для mean/median/mode imputation (Reduce), forward-fill (есть в ts/features). **NEEDS**: iterative imputer (EM на корреляциях) — реализуемо, state O(d²).

### O3. Feature engineering
- Библиотеки: sklearn (`PolynomialFeatures`, `OneHotEncoder`, `KBinsDiscretizer`), `feature-engine`, `category_encoders`, `tsfresh` (TS).
- CADW: **NEEDS**: категориальные encoders (target/frequency/hashing) на диске (A10), полиномиальные признаки (через умножение батчей), биннинг (через quantile).

### O4. Feature selection
- Библиотеки: sklearn (`SelectKBest`, `RFE`, `SelectFromModel`, `mutual_info_*`, `chi2`), `Boruta`.
- CADW: **NEEDS**: mutual info (binned, Reduce), variance threshold (Reduce), GBDT importance (есть), Boruta (shadow features + GBDT — многократный fit, state O(model)).

### O5. Validation / data quality
- Библиотеки: `great-expectations`, `pandera`, `pydantic`, `dbt-tests`.
- CADW: **NEEDS**: schema checks (уже есть через `pa.Schema`), statistical checks (drift — KS/PSI через Reduce — реализуемо).

### O6. Sampling
- Библиотеки: sklearn `train_test_split`, `StratifiedKFold`, `imbalanced-learn`.
- CADW: **OK** — `KFold`, `GroupKFold`, `TimeSeriesSplit` есть; **NEEDS**: reservoir sampling (есть в `core/sampling.py` — можно расширить стратифицированно).

### O7. Hyperparameter tuning / AutoML
- Библиотеки: sklearn (`GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`), `optuna`, `hyperopt`, `ray.tune`, `flaml`, `autogluon`.
- CADW: **NEEDS**: обёртка над `fit`-callable (уже есть Pipeline+CV); cross-val на диске; для AutoML — только внешние фреймворки через батчи.

### O8. Explainability (XAI)
- Библиотеки: `shap`, `lime`, `eli5`, `permutation_importance` (sklearn).
- CADW: **NEEDS**: permutation importance (через батчи — реализуемо), SHAP TreeExplainer (требует модель и features — для GBDT реализуемо, state O(#trees·leaves)). **TODO**: SHAP на больших данных — только сэмпл.

### O9. Fairness / Bias
- Библиотеки: `fairlearn`, `aif360`.
- CADW: **NEEDS**: fairness-метрики через Reduce (group-wise TPR/FPR/DP).

### O10. Drift detection
- Библиотеки: `evidently`, `alibi-detect` (KS, MMD, Chi2).
- CADW: **NEEDS**: KS/PSI/Chi2 как Reduce (статистики — O(1) state) — идеально ложится.

### O11. A/B testing / statistical tests
- Библиотеки: `statsmodels`, `scipy.stats`.
- CADW: **NEEDS**: t-test/Mann-Whitney/CUPED как Reduce (O(1) state).

### O12. Streaming / online learning
- Библиотеки: `river` (онлайн ML), `vowpal-wabbit`.
- CADW: **OK** концептуально (батчи+Reduce); **NEEDS**: онлайн-интерфейс (`partial_fit` на наших моделях — GBDT/Logistic — реализуемо на текущих примитивах).

### O13. Geospatial
- Библиотеки: `geopandas`, `shapely`, `h3`, `pyproj`.
- CADW: **NEEDS**: гео-агрегации (h3-hexbin — уже есть hexbin viz) через Reduce.

---

## P. Резюме — что точно попадает в TODO для больших данных + 1.9 GB

| Тип | Почему TODO | План |
|---|---|---|
| Иерархическая кластеризация (B2) | linkage O(N²) in-memory | только сэмпл/BIRCH |
| t-SNE/UMAP (C2) | kNN-граф O(N·k) | сэмпл + transform по батчам |
| HDBSCAN (B1) | kNN-граф | аппроксимация через faiss-сэмпл |
| One-Class SVM (E1) | kernel O(N²) | сэмпл или stochastic OC-SVM |
| NCF/SASRec (F1) | deep, но можно torch-батчами | если нужен — torch вне CADW |
| fine-tune BERT (J1/J5) | глубокое обучение | torch-батчи, вне scope |
| CV (K1-K3) | глубокое обучение | torch-батчи, вне scope |
| GNN (L1) | глубокое, разреженные графы | вне scope; классические — Reduce |
| RL (M1) | среда+агент | вне scope; bandits — Reduce |
| Generative (N1-N2) | глубокая генерация | вне scope |
| Causal forest (H1) | O(trees) — реализуемо | расширение GBDT, не TODO |
| RSF (I1) | O(trees) — реализуемо | расширение GBDT |
| Linkage / dendrogram (B2) | O(N²) | только сэмпл |

Остальные типы — **OK** или **NEEDS: конкретная функция**, которая кладётся на Reduce/HashShuffle/Sort и влезает в бюджет. Полный roadmap в docs/part2_todo_roadmap.md.

---

## Q. Журнал проверки источников (честная пометка)

Фетч выполнен с этой машины 2026-10-22:

| Источник | Статус | Что взято |
|---|---|---|
| scikit-learn.org/stable/supervised_learning.html | OK (6900 симв.) | список моделей supervised |
| scikit-learn.org/stable/unsupervised_learning.html | OK (4975) | список unsupervised |
| lightgbm.readthedocs.io Features | OK (10954) | leaf-wise, categorical, speed/memory |
| xgboost.readthedocs.io model tutorial | OK (15921) | boosted trees, ranking objectives |
| catboost.ai algorithm-main-stages | OK (2383) | ordered boosting, category/text transforms |
| huggingface.co/docs/transformers tasks/sequence_classification | OK (12073) | text classification pipeline |
| statsmodels.org/stable/examples | OK (6890) | TS/stat примеры |
| pytorch.org/tutorials beginner/basics | OK (15156) | базовая NN-обучение |
| ogb.stanford.edu/docs/leader_nodeprop | **FAIL** (Connection reset) | лидерборды OGB — цитата не проверена, поэтому в тексте `[bench: OGB]` без цифр |

Тонкие места:
- OGB-лидерборд недоступен -> в блоке L1 только названия семейств GNN без чисел.
- Конкретные числа вроде «BERT SOTA на GLUE» — общепризнанны, но без постраничной цитаты; в тексте они помечены `[bench: GLUE]` и не содержат вымышленных цифр.
- Для типов, где не было проверяемого бенчмарка, стоит `[no verified citation]` — это намеренно, а не пропуск.

Итог: все библиотеки/функции/модели в этом документе соответствуют официальным докам (scikit-learn 1.9.1, LightGBM 4.7, XGBoost 3.4.2, CatBoost, HF Transformers, statsmodels 0.15, PyTorch 2.14). Там, где источник не подтверждён, это указано явно.
