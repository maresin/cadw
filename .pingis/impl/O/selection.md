# Section O - EDA / cleaning / FE

Benchmarks: sklearn_reference https://scikit-learn.org/stable/modules/ ; OpenML_Benchmarks https://www.openml.org/search?type=benchmark

## IMPLEMENT

### O2.KNNImputer [baseline]
- algorithm_source_url: https://scikit-learn.org/stable/modules/impute.html
- ref: Troyanskaya et al. 2001, Missing value estimation methods for DNA microarrays
- metric: rmse (imputation error)
- state_class: O(K*d) k-NN index per NaN row batch; bounded_state: true
- state_type: partitioned (HashShuffle by row block) + reduce (per-row k-NN over surviving dims)
- primitive_reuse: HashShuffle (row blocks), Reduce (distance accumulation), guard.check_state
- algorithm: stream batches, impute each block independently using k nearest complete rows (k stored in state, complete rows sampled to K). If complete pool > state_limit, HashShuffle complete rows to partitions and accumulate top-k via Reduce.

### O3.OneHotEncoder [baseline]
- algorithm_source_url: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html (fetched 200)
- ref: sklearn docs
- metric: accuracy
- state_class: O(V) vocabulary; bounded_state: true (SpillDir for V categories if > state_limit)
- state_type: aggregation (unique values) + elementwise (encode)
- primitive_reuse: Reduce (unique values per column), SpillDir for V > state_limit
- algorithm: pass 1 collect unique values per column via sorted batches (Sort or set Reduce); pass 2 encode streaming batches to sparse indicator. Unknown categories mapped to extra column.

### O3.KBinsDiscretizer [baseline]
- algorithm_source_url: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html (fetched 200)
- ref: sklearn docs
- metric: accuracy
- state_class: O(d * n_bins) quantile edges; bounded_state: true
- state_type: cross_row (Sort) - per-column quantile edges
- primitive_reuse: Sort (external_sort per column for exact quantiles), Reduce (per-batch counts for uniform)
- algorithm: for each column, Sort by value, stream chunk medians to get n_bins-1 edges; encode via bisect in streaming pass.

### O6.StratifiedSplit [baseline]
- algorithm_source_url: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html (fetched 200)
- ref: sklearn docs
- metric: (auc stability)
- state_class: O(K_classes); bounded_state: true
- state_type: cross_row (HashShuffle by (class, hash of row id))
- primitive_reuse: HashShuffle (by class label), Sort (optional for hash), Reduce (per-fold counts)
- algorithm: HashShuffle rows by (class, rand_seed(row_id)); within each class partition assign stratified fractions; emit test/train batch assignments to SpillDir.

### O6.ReservoirSample [baseline]
- algorithm_source_url: https://en.wikipedia.org/wiki/Reservoir_sampling (fetched 200)
- ref: Vitter 1985, Random sampling with a reservoir
- metric: (auc stability)
- state_class: O(K) reservoir; bounded_state: true
- state_type: elementwise (online)
- primitive_reuse: none needed (Python loop or numpy vector per batch), guard.check_state(K)
- algorithm: Algorithm R - reservoir of size K, for i>K pick random j=rand(0..i); if j<K replace. Vectorized per batch with RNG rng.integers.

### O7.GridSearch [baseline]
- algorithm_source_url: https://scikit-learn.org/stable/modules/grid_search.html (fetched 200)
- ref: sklearn docs
- metric: accuracy
- state_class: O(|grid|) (params only); bounded_state: true
- state_type: elementwise (loop over candidate fits, each own out-of-core path)
- primitive_reuse: reuses cadw.cv folds + Reduce; depends on estimator
- algorithm: enumerate grid, for each call estimator fit/predict (its own out-of-core path) + score via Reduce; keep best.

### O7.RandomizedSearch [baseline]
- algorithm_source_url: https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization (fetched 200)
- ref: Bergstra & Bengio 2012, Random Search for Hyper-Parameter Optimization
- metric: accuracy
- state_class: O(n_iter)
- state_type: elementwise
- primitive_reuse: same as GridSearch
- algorithm: sample n_iter param draws from distributions, fit/score, keep best.

## SKIP (tier3 / heavy)
- O1.YDataProfiling - wrapper; described_only
- O2.IterativeImputer - MICE iterative; possible out-of-core but complex; described_only for now
- O3.PolynomialFeatures - combinatorial d^2 column explosion; state_class O(d^2) state fits but the actual expansion blows up; skip as described_only
- O5.PanderaSchema - schema check, described_only
- O5.GreatExpectationsSuite - heavy; tier3_export
- O6.SMOTE - interpolation over k-NN; described_only (possible but not priority)
- O7.OptunaTPE - external; described_only
- O8.SHAPTreeExplainer - exact tree shap needs O(leaves); described_only
- O9.FairlearnParity - group sums; trivial; described_only
- O10.EvidentlyDrift - heavy; tier3_export
- O1.Describe/Hist/ValueCounts + O2.SimpleImputer + O4.* + O8.PermutationImportance + O9.FairnessMetrics + O10.DriftStats + O11.* - verified_in_cadw; no new algorithm
