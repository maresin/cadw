"""CADWBoost main class."""
from __future__ import annotations

import numpy as np

from .binning import apply_binning, compute_binning
from .boosting import boost_dart, boost_standard
from .losses import make_loss


class CADWBoost:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        tree_mode: str = "symmetric",
        max_depth: int = 6,
        split_mode: str = "histogram",
        binning: str = "quantile",
        max_bins: int = 255,
        leaf_predictor: str = "constant",
        loss: str = "mse",
        loss_kwargs: dict | None = None,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        sampling: str = "none",
        goss_top_rate: float = 0.2,
        goss_other_rate: float = 0.1,
        reg_lambda: float = 1.0,
        min_gain: float = 1e-6,
        min_child_weight: float = 1.0,
        path_smoothing: float = 0.0,
        max_delta_step: float = 0.0,
        dart_rate: float = 0.0,
        missing_bin: bool = False,
        efb: bool = False,
        efb_max_conflicts: int = 0,
        random_state: int = 0,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tree_mode = tree_mode
        self.max_depth = max_depth
        self.split_mode = split_mode
        self.binning = binning
        self.max_bins = max_bins
        self.leaf_predictor = leaf_predictor
        self.loss = loss
        self.loss_kwargs = loss_kwargs or {}
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.sampling = sampling
        self.goss_top_rate = goss_top_rate
        self.goss_other_rate = goss_other_rate
        self.reg_lambda = reg_lambda
        self.min_gain = min_gain
        self.min_child_weight = min_child_weight
        self.path_smoothing = path_smoothing
        self.max_delta_step = max_delta_step
        self.dart_rate = dart_rate
        self.missing_bin = missing_bin
        self.efb = efb
        self.efb_max_conflicts = efb_max_conflicts
        self.random_state = random_state

        # Fitted
        self.trees_ = []
        self.tree_weights_ = []
        self.thresholds_ = []
        self.n_bins_per_feature_ = None
        self.has_missing_ = None
        self.init_pred_ = 0.0
        self.n_features_in_ = 0
        self.feature_names_ = []
        self.loss_ = None
        self.fitted_ = False
        self.train_loss_ = []
        self.imp_split_ = None
        self.imp_gain_ = None
        self.bundles_ = None      # EFB bundles (if efb=True)

    def _loss_instance(self):
        lk = dict(self.loss_kwargs)
        if self.loss == "huber" and "delta" not in lk:
            lk["delta"] = 1.0
        if self.loss == "quantile" and "quantile" not in lk:
            lk["quantile"] = 0.5
        if self.loss == "focal":
            lk.setdefault("gamma", 2.0)
            lk.setdefault("alpha", 0.25)
        if self.loss == "tweedie" and "power" not in lk:
            lk["power"] = 1.5
        return make_loss(self.loss, **lk)

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.n_features_in_ = d
        self.feature_names_ = (
            list(feature_names) if feature_names
            else [f"x{j}" for j in range(d)])

        self.loss_ = self._loss_instance()
        self.init_pred_ = self.loss_.init_pred(y)

        b = compute_binning(X, self.max_bins, self.binning,
                            missing_bin=self.missing_bin)
        self.thresholds_ = b.thresholds
        self.has_missing_ = b.has_missing_per_feature
        codes = b.codes
        n_bins = b.n_bins_per_feature
        rng = np.random.RandomState(self.random_state)

        # EFB: bundle mutually exclusive features (LightGBM-style)
        if self.efb and d > 2:
            from .efb import apply_bundles, build_bundles
            bundles = build_bundles(
                codes, n_bins,
                max_conflicts=self.efb_max_conflicts, rng=rng)
            # Only apply if meaningful reduction
            if len(bundles) < d:
                codes, n_bins = apply_bundles(codes, bundles)
                self.bundles_ = bundles
        self.n_bins_per_feature_ = n_bins
        if self.dart_rate > 0:
            trees, weights, pred = boost_dart(
                self, codes, X, y, self.init_pred_,
                n_bins, rng, self.missing_bin)
        else:
            trees, weights, pred = boost_standard(
                self, codes, X, y, self.init_pred_,
                n_bins, rng, self.missing_bin)

        self.trees_ = trees
        self.tree_weights_ = np.array(weights)
        self._compute_importance()
        self.fitted_ = True
        return self

    def predict(self, X):
        if not self.fitted_:
            raise RuntimeError("CADWBoost not fitted")
        return self.loss_.predict_from_raw(self._raw_predict(X))

    def _raw_predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        codes = apply_binning(X, self.thresholds_, self.missing_bin)
        if self.bundles_ is not None:
            from .efb import apply_bundles
            codes, _ = apply_bundles(codes, self.bundles_)
        n = X.shape[0]
        raw = np.full(n, self.init_pred_)
        for tree, w in zip(self.trees_, self.tree_weights_):
            if w == 0:
                continue
            raw = raw + w * tree.predict(
                codes, X if tree.linear else None)
        return raw

    def predict_proba(self, X):
        if self.loss_.task != "binary":
            raise ValueError(
                f"predict_proba only for binary losses (logistic/focal); "
                f"got task={self.loss_.task}")
        raw = self._raw_predict(X)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))

    def _compute_importance(self):
        d = self.n_features_in_
        split = np.zeros(d)
        gain = np.zeros(d)
        for tree in self.trees_:
            if tree.mode == "symmetric":
                for lvl in range(tree.max_depth):
                    f = int(tree.splits[lvl, 0])
                    if f >= 0:
                        split[f] += 1
                        if tree.split_gains is not None:
                            gain[f] += float(tree.split_gains[lvl])
            else:
                for f in tree.feature:
                    if f >= 0:
                        split[f] += 1
                # leaf-wise split_gains not tracked yet
                if tree.split_gains is not None:
                    for i, f in enumerate(tree.feature):
                        if f >= 0:
                            gain[f] += float(tree.split_gains[i])
        self.imp_split_ = split
        self.imp_gain_ = gain

    def feature_importance(self, kind: str = "gain"):
        if not self.fitted_:
            raise RuntimeError("not fitted")
        if kind == "split":
            arr = self.imp_split_
        elif kind == "gain":
            arr = self.imp_gain_
        else:
            raise ValueError(f"unknown importance kind {kind!r}")
        return dict(zip(self.feature_names_, arr))

    def state_nbytes(self):
        tree_state = 0
        for t in self.trees_:
            if t.values is not None:
                tree_state += t.values.nbytes
            if t.splits is not None:
                tree_state += t.splits.nbytes
            if t.coefs is not None:
                tree_state += t.coefs.nbytes
        bin_state = sum(a.nbytes for a in self.thresholds_)
        return int(tree_state + bin_state + 256 + self.tree_weights_.nbytes)

    def save_json(self, path):
        """Save fitted model to JSON file."""
        from .serialize import save_json as _save
        _save(self, path)

    @classmethod
    def load_json(cls, path):
        """Load model from JSON file (returns fitted CADWBoost)."""
        from .serialize import load_json as _load
        return _load(path, cls)

    def fit_df(self, df, X_cols, y_col, *, mode="memory",
               memory_guard=None):
        """Fit from CadwDataFrame. mode="memory" (n≤10^6) or "ooc"."""
        from .dataframe import fit_df as _fit_df
        return _fit_df(self, df, X_cols, y_col, mode=mode,
                       memory_guard=memory_guard)

    def predict_df(self, df, X_cols, *, output="numpy", memory_guard=None):
        """Predict on CadwDataFrame, per-batch streaming."""
        from .dataframe import predict_df as _predict_df
        return _predict_df(self, df, X_cols, output=output,
                           memory_guard=memory_guard)

    def predict_proba_df(self, df, X_cols, *, memory_guard=None):
        """Streaming predict_proba (binary only)."""
        from .dataframe import predict_proba_df as _pp
        return _pp(self, df, X_cols, memory_guard=memory_guard)

    def contributions(self, X, feature_names=None):
        """Per-sample, per-feature contributions (simplified TreeSHAP).

        Sum of contributions + init_pred ≈ prediction (on raw/logit scale).
        See shap.py for details.
        """
        from .shap import feature_contributions
        return feature_contributions(self, X, feature_names)

    def contributions_summary(self, X):
        """Global feature importances from contributions (|mean|, mean, std)."""
        from .shap import summarize_contributions
        contrib = self.contributions(X)
        return summarize_contributions(contrib, self.feature_names_)
