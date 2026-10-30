# cadw/dataframe/_ml_bindings.py
# v0.9.26v (B.1 step 16): _MLBindingsMixin.
# Symbols: encode_categorical, metrics, fit_* (10).
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import pyarrow as pa
from cadw.dataset import Dataset


class _MLBindingsMixin:
    """ML fit_* bindings + metrics + encode (v0.9.26v)."""
    def encode_categorical(
        self,
        columns: "str | Sequence[str]",
    ) -> "tuple[CadwDataFrame, dict[str, list]]":
        """Encode categorical columns as int32 (v0.6.2a).

        - columns: str or Sequence[str]. Must be string/binary columns.
        - Pass 1: collect unique values (pc.value_counts per batch),
          sorted for determinism. Nulls excluded (become null ids).
        - Pass 2: pc.index_in(col, value_set=pa.array(sorted_values))
          per batch → int32 (null stays null).

        Returns:
            (encoded_df, reverse_maps): new CadwDataFrame with int32
            columns (same names); original df unchanged.
            reverse_maps[col] = sorted_values; id k → reverse_maps[col][k].

        Memory: O(K) per column, independent of N.
        Raises TypeError for non-string columns.
        """
        from ._core import CadwDataFrame
        from cadw.core.memory_guard import MemoryLimitExceeded
        import pyarrow.compute as pc

        if isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("encode_categorical: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"encode_categorical: column {c!r} not in {self._columns}"
                )

        # Validate types — only string/binary encodable.
        for c in cols:
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_string(t) or pa.types.is_large_string(t)
                    or pa.types.is_binary(t) or pa.types.is_large_binary(t)):
                raise TypeError(
                    f"encode_categorical: column {c!r} has type {t}; "
                    f"only string/binary supported"
                )

        # Pass 1: collect unique values per column.
        reverse_maps: dict[str, list] = {}
        for col in cols:
            uniques: set = set()
            for b in self._iter_filtered_batches():
                arr = b.column(col)
                # v0.9.90-S3 (G2): pre-check BEFORE set grows. Worst
                # case this batch adds b.num_rows unique values.
                # 180 B/entry = 50 (str) + ~40 (set slot)
                # + ~90 (list copy margin for sorted()).
                est_bytes = (len(uniques) + b.num_rows) * 180
                try:
                    self._memory_guard.check_state(
                        est_bytes,
                        name=f"encode_categorical({col})",
                    )
                except MemoryLimitExceeded as e:
                    k_est = len(uniques) + b.num_rows
                    raise MemoryLimitExceeded(
                        f"encode_categorical({col!r}): reverse_maps "
                        f"is O(K) by API contract (list of unique "
                        f"values), independent of N. K~{k_est:,} "
                        f"values, ~{est_bytes / 1e6:.1f} MB. "
                        f"Options: (a) top-K via "
                        f"pc.value_counts(...).top_k with K << 10^6; "
                        f"(b) wait for v0.9.102 lazy reverse_maps "
                        f"(SpillMap on disk). "
                        f"Original: {e}"
                    ) from e
                vc = pc.value_counts(arr)  # struct<values, counts>
                if isinstance(vc, pa.ChunkedArray):
                    vc = vc.combine_chunks()
                values = vc.field("values")
                for v in values.to_pylist():  # K per batch, not N
                    if v is not None:
                        uniques.add(v)
            try:
                sorted_vals = sorted(uniques)
            except TypeError:
                sorted_vals = sorted(uniques, key=lambda x: str(x))
            reverse_maps[col] = sorted_vals

        # Pass 2: encode.
        schema = pa.schema(
            [
                (
                    pa.field(c, pa.int32())
                    if c in cols
                    else self._dataset.schema.field(c)
                )
                for c in self._columns
            ]
        )

        def _iter_encoded():
            for b in self._iter_filtered_batches():
                arrays = []
                for c in self._columns:
                    if c in cols:
                        arr = b.column(c)
                        value_set = pa.array(reverse_maps[c], type=arr.type)
                        ids = pc.index_in(arr, value_set=value_set)
                        arrays.append(ids.cast(pa.int32()))
                    else:
                        arrays.append(b.column(c))
                yield pa.RecordBatch.from_arrays(arrays, schema=schema)

        new_ds = Dataset.from_batches(
            schema, _iter_encoded(), self._new_dataset_dir()
        )
        new_df = CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )
        return new_df, reverse_maps


    def metrics(
        self,
        *,
        y_true: str,
        metrics: "Sequence[str] | str",
        y_pred: "str | None" = None,
        y_proba: "str | Sequence[str] | None" = None,
        y_proba_cols: "Sequence[str] | None" = None,
        classes: "Sequence | None" = None,
        average: "str | None" = None,
        pos_label=None,
    ) -> dict:
        """Regression + log_loss + classification metrics in one pass.

        Metrics:
          - r2, mse, rmse, mae — require `y_pred` (numeric, continuous).
          - log_loss — requires `y_proba` (binary, values in [0,1]).
          - accuracy, precision, recall, f1 — require `y_pred` (labels,
            numeric or string).

        Args:
            y_true: column with true values.
            metrics: metric name(s) — scalar or list.
            y_pred: predicted values (float or labels — depends on
                metric group).
            y_proba: predicted probabilities. str for binary
                (single column, float in [0,1]); list[str] for
                multiclass (K columns, K classes — requires
                `classes`).
            y_proba_cols: alias for `y_proba` when passing a
                list of columns. Both work; use whichever you
                prefer (matches MetricsOp parameter name).
                Passing both → ValueError.
            classes: explicit class list for classification. K=2 default
                when classes=None.
            average: 'binary' | 'macro' | 'micro' | 'weighted' for
                precision/recall/f1. Auto: 'binary' (K=2) / 'macro' (K>2).
            pos_label: positive class for average='binary'. None →
                classes[-1] (K=2 default: [0,1]→1; string classes →
                last class). Explicit value must be in `classes`.

        Returns:
            dict `{metric_name: value}`. NaN for empty input.

        Restrictions:
          - Regression (r2/mse/rmse/mae) and classification
            (accuracy/...) cannot share y_pred in one call.
          - Classification K×24+64 > state_limit → MemoryLimitExceeded.

        State: O(1) + O(K) (classification). Independent of N.
        """
        from cadw.ops.metrics import (
            AUC_METRICS, CLASSIFICATION_METRICS, REGRESSION_METRICS,
            SUPPORTED_METRICS, MetricsOp,
        )
        from cadw.ops.reduce import reduce

        if isinstance(metrics, str):
            metrics = [metrics]
        metrics = list(metrics)
        if not metrics:
            raise ValueError("metrics: metrics list is empty")
        unknown = [m for m in metrics if m not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"metrics: unknown metric(s) {unknown}; "
                f"supported: {sorted(SUPPORTED_METRICS)}"
            )

        # v0.9.2c-fix1 (§13.4/167): y_proba is str | Sequence[str];
        # generic loop can't handle list (would check `list not in
        # columns` — always True). Specific per-column check below
        # (after y_proba_cols normalization).
        for c, label in ((y_true, "y_true"), (y_pred, "y_pred")):
            if c is not None and c not in self._columns:
                raise KeyError(
                    f"metrics: {label} column {c!r} not in {self._columns}"
                )

        # §13.4/164: fail-fast for classification K too large — before
        # constructing MetricsOp (clear message, not "state limit").
        ms = set(metrics)
        if (ms & CLASSIFICATION_METRICS) or (ms & AUC_METRICS):
            K = 2 if classes is None else len(classes)
            est = 64 + 24 * K
            state_limit = self._memory_guard.state_limit_bytes
            if est > state_limit:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"metrics: K={K} → state {est} B > state_limit "
                    f"{state_limit} B. Reduce class count or increase "
                    f"budget. §13.4/164."
                )

        # v0.9.23b: y_proba_cols alias (matches MetricsOp param
        # name; AUDIT_FULL_v0922 item 7.A.11.3 / c11 Jupyter
        # gap). Existing `y_proba` (str|Sequence) and new
        # `y_proba_cols` (Sequence[str]) both work; passing
        # both → ValueError (ambiguous); str → TypeError (would
        # iterate char-by-char — footgun).
        if y_proba_cols is not None:
            if y_proba is not None:
                raise ValueError(
                    "metrics: pass either y_proba= or "
                    "y_proba_cols=, not both"
                )
            if isinstance(y_proba_cols, str):
                raise TypeError(
                    "metrics: y_proba_cols must be a "
                    "sequence of column names; got single "
                    f"string {y_proba_cols!r}. Use "
                    "y_proba='col' for single-column "
                    "proba, or y_proba_cols=['col'] for a "
                    "list."
                )
            y_proba = list(y_proba_cols)
            if not y_proba:
                raise ValueError(
                    "metrics: y_proba_cols must be non-empty"
                )

        # v0.9.2c: y_proba normalization (str → [str], list → list).
        if y_proba is None:
            y_proba_cols = None
        elif isinstance(y_proba, str):
            y_proba_cols = [y_proba]
        else:
            y_proba_cols = list(y_proba)
            if not y_proba_cols:
                raise ValueError(
                    "metrics: y_proba list is empty"
                )

        # Column existence for all y_proba cols
        if y_proba_cols is not None:
            for pc_ in y_proba_cols:
                if pc_ not in self._columns:
                    raise KeyError(
                        f"metrics: y_proba column {pc_!r} not in "
                        f"{self._columns}"
                    )

        is_multiclass_proba = (
            y_proba_cols is not None and len(y_proba_cols) > 1
        )
        if is_multiclass_proba:
            if classes is None:
                raise ValueError(
                    f"metrics: multiclass y_proba (len="
                    f"{len(y_proba_cols)}) requires `classes`"
                )
            if len(y_proba_cols) != len(classes):
                raise ValueError(
                    f"metrics: len(y_proba)={len(y_proba_cols)} != "
                    f"len(classes)={len(classes)}"
                )

        # roc_auc: separate handling (multipass Sort-based)
        has_roc_auc = "roc_auc" in metrics
        other_metrics = [m for m in metrics if m != "roc_auc"]

        # Conflict: roc_auc + regression on same y_true (binary labels vs
        # continuous target — mutually exclusive semantics).
        if has_roc_auc and (set(other_metrics) & REGRESSION_METRICS):
            raise ValueError(
                f"metrics: roc_auc requires binary labels y_true, "
                f"regression ({sorted(set(other_metrics) & REGRESSION_METRICS)}) "
                f"requires continuous — cannot mix. Use separate calls."
            )
        if has_roc_auc and y_proba_cols is None:
            raise ValueError("metrics: roc_auc requires y_proba")

        result: dict = {}

        if other_metrics:
            op = MetricsOp(
                y_true,
                y_pred_col=y_pred,
                y_proba_cols=y_proba_cols,
                metrics=other_metrics,
                classes=classes,
                average=average,
                pos_label=pos_label,
            )
            result.update(reduce(
                self._iter_filtered_batches(), op,
                self._memory_guard, log=False,
            ))

        if has_roc_auc:
            from cadw.ops.metrics import (
                _compute_roc_auc_binary, _compute_roc_auc_macro,
            )
            if is_multiclass_proba:
                result["roc_auc"] = _compute_roc_auc_macro(
                    stream_factory=lambda: self._iter_filtered_batches(),
                    y_true_col=y_true,
                    y_proba_cols=y_proba_cols,
                    classes=list(classes),
                    spill_dir=self._spill_dir / "metrics_roc_auc",
                    memory_guard=self._memory_guard,
                )
            else:
                pos_roc = pos_label if pos_label is not None else 1
                result["roc_auc"] = _compute_roc_auc_binary(
                    stream_factory=lambda: self._iter_filtered_batches(),
                    y_true_col=y_true,
                    y_proba_col=y_proba_cols[0],
                    pos_label=pos_roc,
                    spill_dir=self._spill_dir / "metrics_roc_auc",
                    memory_guard=self._memory_guard,
                )

        return result


    def fit_linear(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        alpha: float = 0.0,
    ):
        """Fit Linear (alpha=0) or Ridge (alpha>0) (v0.7.0a).

        One Reduce over filtered batches → solve (X^T X + αI_nonintercept) β.

        Returns: LinearModel (coef_, intercept_, predict(X)).

        See cadw.ml.linear.fit_linear for full docs.
        """
        from cadw.ml.linear import fit_linear
        return fit_linear(
            self, X_cols, y_col, alpha=alpha,
            memory_guard=self._memory_guard,
        )


    def fit_logistic(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        alpha: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        """Fit binary logistic via Newton IRLS (v0.7.0b).

        Each Newton iteration = one Reduce pass. Typically 5-15 iterations.

        Returns LogisticModel (coef_, intercept_, predict, predict_proba).

        See cadw.ml.logistic.fit_logistic for full docs.
        """
        from cadw.ml.logistic import fit_logistic
        return fit_logistic(
            self, X_cols, y_col,
            alpha=alpha, max_iter=max_iter, tol=tol,
            memory_guard=self._memory_guard,
        )


    def fit_kmeans(
        self,
        X_cols: "str | Sequence[str]",
        k: int,
        *,
        n_init: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        seed: int = 42,
    ):
        """Fit k-means clustering via iter-Reduce (v0.7.1).

        Each iteration = one Reduce pass (assign + accumulate). Peak
        RAM = batch + O(k×d).

        Returns KMeansModel (cluster_centers_, predict).
        """
        from cadw.ml.kmeans import fit_kmeans
        return fit_kmeans(
            self, X_cols, k,
            n_init=n_init, max_iter=max_iter, tol=tol, seed=seed,
            memory_guard=self._memory_guard,
        )


    def fit_pca(
        self,
        X_cols: "str | Sequence[str]",
        n_components: "int | None" = None,
    ):
        """Fit PCA via Reduce(CovarianceOp) + eigh (v0.7.2).

        Returns PCA with sklearn-compatible attributes:
            components_ (k, d), mean_, explained_variance_,
            explained_variance_ratio_, singular_values_.

        pca.transform(df) → df with pc0..pc{k-1} columns.
        pca.inverse_transform(df) → reconstructed feature columns.
        """
        from cadw.ml.pca import fit_pca
        return fit_pca(
            self, X_cols, n_components,
            memory_guard=self._memory_guard,
        )


    def fit_gaussian_nb(
        self, X_cols: "str | Sequence[str]", y_col: str,
        *, var_smoothing: float = 1e-9,
    ):
        from cadw.ml.naive_bayes import fit_gaussian_nb
        return fit_gaussian_nb(
            self, X_cols, y_col, var_smoothing=var_smoothing,
            memory_guard=self._memory_guard,
        )


    def fit_multinomial_nb(
        self, X_cols: "str | Sequence[str]", y_col: str,
        *, alpha: float = 1.0,
    ):
        from cadw.ml.naive_bayes import fit_multinomial_nb
        return fit_multinomial_nb(
            self, X_cols, y_col, alpha=alpha,
            memory_guard=self._memory_guard,
        )


    def fit_bernoulli_nb(
        self, X_cols: "str | Sequence[str]", y_col: str,
        *, alpha: float = 1.0,
    ):
        from cadw.ml.naive_bayes import fit_bernoulli_nb
        return fit_bernoulli_nb(
            self, X_cols, y_col, alpha=alpha,
            memory_guard=self._memory_guard,
        )


    def fit_gbdt_regressor(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        bins: int = 32,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        min_gain: float = 0.0,
    ):
        """Fit GBDT regression (L2 loss, level-wise) — v0.7.3a.

        Peak RAM = batch + O(n × 32B) internal state + O(nodes × feat × bins).
        O(n) state is inherent to Newton-based boosting (XGBoost/LightGBM
        same). Guarded via MemoryGuard.

        Returns GBDTRegressor (trees_, predict).
        """
        from cadw.ml.gbdt import fit_gbdt_regressor
        return fit_gbdt_regressor(
            self, X_cols, y_col,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, bins=bins,
            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
            min_gain=min_gain,
            memory_guard=self._memory_guard,
        )


    def fit_gbdt_classifier(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        bins: int = 32,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        min_gain: float = 0.0,
    ):
        """Fit GBDT binary classification (logloss) — v0.7.5.

        y must be {0, 1}, both classes present.
        Returns GBDTClassifier (predict_proba, predict).
        """
        from cadw.ml.gbdt import fit_gbdt_classifier
        return fit_gbdt_classifier(
            self, X_cols, y_col,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, bins=bins,
            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
            min_gain=min_gain,
            memory_guard=self._memory_guard,
        )


    def fit_gbdt_multiclass(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        bins: int = 32,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        min_gain: float = 0.0,
    ):
        """Fit GBDT multiclass classification (softmax) — v0.7.8.

        y must be {0, 1, ..., K-1}, K >= 3. For binary — fit_gbdt_classifier.
        """
        from cadw.ml.gbdt import fit_gbdt_multiclass
        return fit_gbdt_multiclass(
            self, X_cols, y_col,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, bins=bins,
            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
            min_gain=min_gain,
            memory_guard=self._memory_guard,
        )

    def fit_naive_bayes(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        variant: str = "gaussian",
        alpha: float = 1.0,
        var_smoothing: float = 1e-9,
    ):
        """Fit Naive Bayes (v0.7.6).

        - variant='gaussian'    : continuous X; params: var_smoothing.
        - variant='multinomial' : X >= 0 counts; params: alpha.
        - variant='bernoulli'   : X in {0,1}; params: alpha.

        y must be numeric (int/float). For categorical target — encode.
        Returns GaussianNB | MultinomialNB | BernoulliNB.
        """
        from cadw.ml.naive_bayes import (
            _VARIANTS,
            fit_gaussian_nb, fit_multinomial_nb, fit_bernoulli_nb,
        )
        if variant not in _VARIANTS:
            raise ValueError(
                f"fit_naive_bayes: variant must be one of {_VARIANTS}; "
                f"got {variant!r}"
            )
        if variant == "gaussian":
            return fit_gaussian_nb(
                self, X_cols, y_col, var_smoothing=var_smoothing,
                memory_guard=self._memory_guard,
            )
        if variant == "multinomial":
            return fit_multinomial_nb(
                self, X_cols, y_col, alpha=alpha,
                memory_guard=self._memory_guard,
            )
        return fit_bernoulli_nb(
            self, X_cols, y_col, alpha=alpha,
            memory_guard=self._memory_guard,
        )


    def fairness_metrics(
        self,
        *,
        sensitive_col: str,
        y_true_col: str,
        y_pred_col: str,
    ):
        """Streaming group fairness metrics (v0.9.35, D.1.4).

        DP / equal-opportunity (TPR gap) / FPR gap / disparate impact
        from a single pass; state O(K) groups.

        See cadw.ml.fairness.fairness_metrics.
        """
        from cadw.ml.fairness import fairness_metrics
        return fairness_metrics(
            self,
            sensitive_col=sensitive_col,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            memory_guard=self._memory_guard,
        )

    def fit_matrix_profile(
        self,
        value_col: str,
        *,
        window: int = 32,
        max_samples: int = 4096,
        random_state: "int | None" = 0,
    ):
        """Streaming Matrix Profile (v0.9.35, D.1.2).

        Bounded reservoir sample of z-normalized subsequences + exact block
        STOMP over the sample. State O(max_samples * window), independent
        of N. Discord scores: 1 - mP/sqrt(2w).
        """
        from cadw.ml.matrix_profile import fit_matrix_profile
        return fit_matrix_profile(
            self, value_col, window=window, max_samples=max_samples,
            random_state=random_state, memory_guard=self._memory_guard,
        )


    def fit_lambdamart(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        query_col: str,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        bins: int = 32,
        sigma: float = 1.0,
        ndcg_k: int = 10,
    ):
        """Fit a LambdaMART ranker (v0.9.35, D.2.1).

        y_col: graded relevance (int >= 0). query_col: query / group id.
        Returns LambdaMARTModel with .predict(df) -> score column.
        """
        from cadw.ml.lambdamart import fit_lambdamart
        return fit_lambdamart(
            self, X_cols, y_col, query_col,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, bins=bins, sigma=sigma,
            ndcg_k=ndcg_k, memory_guard=self._memory_guard,
        )


    def fit_als(
        self,
        user_col: str,
        item_col: str,
        *,
        weight_col: "str | None" = None,
        n_factors: int = 16,
        n_iter: int = 10,
        reg: float = 0.1,
        alpha: float = 40.0,
        seed: int = 0,
    ):
        """Implicit-feedback ALS (v0.9.36, D.2.2). Out-of-core via
        external_sort per sweep; state O((U+I)*k)."""
        from cadw.ml.als import fit_als
        return fit_als(
            self, user_col, item_col, weight_col=weight_col,
            n_factors=n_factors, n_iter=n_iter, reg=reg, alpha=alpha,
            seed=seed, memory_guard=self._memory_guard,
        )


    def fit_hybrid_mf(
        self,
        user_col: str,
        item_col: str,
        content_cols: "str | Sequence[str]",
        *,
        n_factors: int = 16,
        n_iter: int = 10,
        reg: float = 0.1,
        alpha: float = 40.0,
        ridge: float = 1.0,
        seed: int = 0,
    ):
        """Hybrid MF + content (v0.9.36, D.2.3). ALS factors plus a
        ridge-learned content residual; cold items score from content."""
        from cadw.ml.hybrid_mf import fit_hybrid_mf
        return fit_hybrid_mf(
            self, user_col, item_col, content_cols,
            n_factors=n_factors, n_iter=n_iter, reg=reg, alpha=alpha,
            ridge=ridge, seed=seed, memory_guard=self._memory_guard,
        )


    def fit_coxph(
        self,
        X_cols: "str | Sequence[str]",
        time_col: str,
        event_col: str,
        *,
        max_iter: int = 50,
        tol: float = 1e-7,
    ):
        """Cox proportional hazards (v0.9.36, D.3.1). Breslow partial
        likelihood, Newton-Raphson, guarded streaming pre-pass."""
        from cadw.ml.coxph import fit_coxph
        return fit_coxph(
            self, X_cols, time_col, event_col,
            max_iter=max_iter, tol=tol, memory_guard=self._memory_guard,
        )


    def fit_lasso(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        """Fit Lasso (L1). alpha is raw (additive to X^T X)."""
        from cadw.ml.linear import fit_lasso
        return fit_lasso(
            self, X_cols, y_col,
            alpha=alpha, max_iter=max_iter, tol=tol,
            memory_guard=self._memory_guard,
        )


    def fit_elasticnet(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        """Fit ElasticNet (L1+L2). l1_ratio=1 → Lasso, 0 → Ridge."""
        from cadw.ml.linear import fit_elasticnet
        return fit_elasticnet(
            self, X_cols, y_col,
            alpha=alpha, l1_ratio=l1_ratio,
            max_iter=max_iter, tol=tol,
            memory_guard=self._memory_guard,
        )


    def fit_meta_learner(
        self,
        X_cols,
        treatment_col: str,
        y_col: str,
        *,
        kind: str = "t",
        n_estimators: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        propensity_alpha: float = 1.0,
    ):
        """T/S-learner + propensity (v0.9.37, D.3.2)."""
        from cadw.ml.meta_learners import fit_meta_learner as _f
        return _f(
            self, X_cols, treatment_col, y_col, kind=kind,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, propensity_alpha=propensity_alpha,
        )

    def fit_uplift(
        self,
        X_cols,
        treatment_col: str,
        y_col: str,
        *,
        n_estimators: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.1,
    ):
        """Uplift trees + Qini (v0.9.37, D.3.3)."""
        from cadw.ml.uplift import fit_uplift as _f
        return _f(
            self, X_cols, treatment_col, y_col, n_estimators=n_estimators,
            max_depth=max_depth, learning_rate=learning_rate,
        )

    def fit_minirocket(
        self,
        value_col: str,
        *,
        id_col: "str | None" = None,
        num_kernels: int = 84,
        seed: int = 0,
        max_dilation: int = 32,
    ):
        """MiniRocket random-conv features (v0.9.37, D.4.1)."""
        from cadw.ml.minirocket import fit_minirocket as _f
        return _f(
            self, value_col, id_col=id_col, num_kernels=num_kernels,
            seed=seed, max_dilation=max_dilation,
        )

    def fit_pelt(
        self,
        value_col: str,
        *,
        id_col: "str | None" = None,
        penalty: float = 10.0,
        min_size: int = 2,
        max_window: "int | None" = None,
    ):
        """PELT change-point detection (v0.9.38, D.4.2)."""
        from cadw.ml.pelt import fit_pelt as _f
        return _f(self, value_col, id_col=id_col, penalty=penalty,
                  min_size=min_size, max_window=max_window)

    def fit_ets(
        self,
        target_col: str,
        *,
        time_col: "str | None" = None,
        group_col: "str | None" = None,
        seasonal_period: int = 1,
        seasonal: str = "additive",
        trend: bool = True,
        damped: bool = False,
    ):
        """ETS / Holt-Winters forecaster (v0.9.38, D.4.3)."""
        from cadw.ts.ets import ETSForecaster
        return ETSForecaster(seasonal_period=seasonal_period, seasonal=seasonal,
                             trend=trend, damped=damped).fit(
            self, target_col, time_col=time_col, group_col=group_col)

    def fit_theta(
        self,
        target_col: str,
        *,
        time_col: "str | None" = None,
        group_col: "str | None" = None,
        seasonal: bool = False,
        seasonal_period: int = 1,
    ):
        """Theta-method forecaster (v0.9.38, D.4.4)."""
        from cadw.ts.theta import ThetaForecaster
        return ThetaForecaster(seasonal=seasonal,
                               seasonal_period=seasonal_period).fit(
            self, target_col, time_col=time_col, group_col=group_col)

    def fit_var(
        self,
        target_col,
        *,
        time_col: "str | None" = None,
        group_col: "str | None" = None,
        p: int = 1,
    ):
        """VAR(p) vector autoregression (v0.9.38, D.4.5)."""
        from cadw.ts.var import VARModel
        return VARModel(p=p).fit(self, target_col,
                                 time_col=time_col, group_col=group_col)

    def fit_calibration(
        self,
        X_cols,
        y_col: str,
        *,
        method: str = "platt",
        n_estimators: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        n_bins: int = 256,
    ):
        """Platt / isotonic probability calibration (v0.9.39, D.4.6)."""
        from cadw.ml.calibration import fit_calibration as _f
        return _f(self, X_cols, y_col, method=method,
                  n_estimators=n_estimators, max_depth=max_depth,
                  learning_rate=learning_rate, n_bins=n_bins)

    def fit_ordinal_logit(
        self,
        X_cols,
        y_col: str,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
    ):
        """Ordinal logistic cumulative link (v0.9.39, D.5.1)."""
        from cadw.ml.ordinal import fit_ordinal_logit as _f
        return _f(self, X_cols, y_col, max_iter=max_iter, tol=tol)

    def fit_chains(
        self,
        X_cols,
        label_cols,
        *,
        chain: bool = True,
        n_estimators: int = 30,
        max_depth: int = 3,
        learning_rate: float = 0.1,
    ):
        """Classifier chains / one-vs-rest (v0.9.39, D.5.2)."""
        from cadw.ml.chains import fit_chains as _f
        return _f(self, X_cols, label_cols, chain=chain,
                  n_estimators=n_estimators, max_depth=max_depth,
                  learning_rate=learning_rate)

    def fit_minibatch_kmeans(
        self,
        X_cols,
        *,
        n_clusters: int = 8,
        batch_size: int = 1024,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 0,
    ):
        """MiniBatch KMeans (v0.9.39, D.5.3)."""
        from cadw.ml.minibatch import fit_minibatch_kmeans as _f
        return _f(self, X_cols, n_clusters=n_clusters, batch_size=batch_size,
                  max_iter=max_iter, tol=tol, seed=seed)

    def fit_minibatch_gmm(
        self,
        X_cols,
        *,
        n_components: int = 3,
        batch_size: int = 1024,
        max_iter: int = 50,
        tol: float = 1e-4,
        seed: int = 0,
        reg: float = 1e-6,
    ):
        """MiniBatch EM GMM, diagonal covariance (v0.9.39, D.5.3)."""
        from cadw.ml.minibatch import fit_minibatch_gmm as _f
        return _f(self, X_cols, n_components=n_components,
                  batch_size=batch_size, max_iter=max_iter, tol=tol,
                  seed=seed, reg=reg)

    def fit_birch(self, X_cols, *, n_clusters: int = 8, threshold: float = 0.5,
                  branching_factor: int = 50, max_iter: int = 20):
        """BIRCH CF-tree streaming clustering (v0.9.40, D.5.4)."""
        from cadw.ml.birch import fit_birch as _f
        return _f(self, X_cols, n_clusters=n_clusters, threshold=threshold,
                  branching_factor=branching_factor, max_iter=max_iter)

    def fit_feature_select(self, X_cols, y_col: str, *, method: str = "mutual_info",
                           task: str = "classification", variance_threshold: float = 0.0,
                           max_iter: int = 20, n_estimators: int = 50, seed: int = 0):
        """Feature selection: mutual_info / variance / Boruta (v0.9.40, D.6.1)."""
        from cadw.ml.feature_select import fit_feature_select as _f
        return _f(self, X_cols, y_col, method=method, task=task,
                  variance_threshold=variance_threshold, max_iter=max_iter,
                  n_estimators=n_estimators, seed=seed)

    def fit_drift(self, col: str, reference_col: str, *, method: str = "ks", n_bins: int = 10):
        """Drift estimator (v0.9.40, D.6.2)."""
        from cadw.ml.drift import fit_drift as _f
        return _f(self, col, reference_col, method=method, n_bins=n_bins)

    def fit_ab_test(self, outcome_col: str, treatment_col: str, *,
                    covariate_col: "str | None" = None, method: str = "t"):
        """A/B test container (v0.9.40, D.6.2)."""
        from cadw.ml.drift import fit_ab_test as _f
        return _f(self, outcome_col, treatment_col,
                  covariate_col=covariate_col, method=method)

    def fit_randomized_svd(self, X_cols, *, n_components: int = 8,
                           n_oversamples: int = 10, n_power_iter: int = 1, seed: int = 0):
        """Randomized SVD (v0.9.40, D.6.4)."""
        from cadw.ml.randsvd import fit_randomized_svd as _f
        return _f(self, X_cols, n_components=n_components,
                  n_oversamples=n_oversamples, n_power_iter=n_power_iter, seed=seed)

    def fit_geo_agg(self, lat_col: str, lon_col: str, *, cell_size_m: float = 1000.0,
                    origin_lat: "float | None" = None,
                    origin_lon: "float | None" = None):
        """Geo-aggregation config (v0.9.40, D.6.5)."""
        from cadw.ml.geo import fit_geo_agg as _f
        return _f(self, lat_col, lon_col, cell_size_m=cell_size_m,
                  origin_lat=origin_lat, origin_lon=origin_lon)

    def fit_lda(self, text_col: str, *, n_topics: int = 10, vocab_size: int = 1000,
                max_iter: int = 10, alpha: float = 0.1, beta: float = 0.01,
                tau0: float = 64.0, kappa: float = 0.7, seed: int = 0):
        """Online variational Bayes LDA (v0.9.41, D.6.6)."""
        from cadw.ml.lda import fit_lda as _f
        return _f(self, text_col, n_topics=n_topics, vocab_size=vocab_size,
                  max_iter=max_iter, alpha=alpha, beta=beta, tau0=tau0,
                  kappa=kappa, seed=seed)

    def fit_linucb(self, X_cols, *, arm_col: str, reward_col: str,
                   alpha: float = 1.0, l2: float = 1.0):
        """Disjoint LinUCB sufficient statistics (v0.9.41, D.6.7)."""
        from cadw.ml.linucb import fit_linucb as _f
        return _f(self, X_cols, arm_col=arm_col, reward_col=reward_col,
                  alpha=alpha, l2=l2)

    def fit_tfidf(self, text_col: str, *, vocab_size: int = 10000,
                  sublinear_tf: bool = True, min_df: int = 1,
                  norm: "str | None" = "l2"):
        """Streaming TF-IDF vectorizer (v0.9.41, D.6.9)."""
        from cadw.ml.tfidf import fit_tfidf as _f
        return _f(self, text_col, vocab_size=vocab_size,
                  sublinear_tf=sublinear_tf, min_df=min_df, norm=norm)
