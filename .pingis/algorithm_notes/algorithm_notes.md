# Algorithm notes - Phase D picks (v0.9.32)

## D.1.3 Cost-sensitive GBDT + threshold tuning

### Why
- Imbalanced classification (taxonomy A6) is a P0 task type not yet covered. Two cheap levers:
  1. **Class weight / scale_pos_weight** injected into per-sample gradient/hessian. Per Friedman 2001, weights multiply grad/hess, so scaling is exact. sklearn GBDT exposes both sample_weight and class_weight.
  2. **Post-hoc threshold tuning** on predict_proba — the cheapest test-time lever (O(1) additional state, caller picks threshold). Backed by sklearn roc_curve/precision_recall_curve and Fawcett 2006 ROC analysis.

### Memory
- Weight adds **no state**: only a scalar pos_weight. Gradients are already float64 per row.
- Threshold selection: `predict(X, threshold=...)` — scalar arg, no sweep state.
- Fit-time memory: identical to existing GBDT (F9 fix applies).

### Sources
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html  (sample_weight, class_weight)
- https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting (framework, loss form)
- Friedman 2001, Annals of Statistics (canonical grad/hess form)
- Fawcett 2006, Pattern Recognition Letters (ROC / threshold rationale)
- https://xgboost.readthedocs.io/en/stable/parameter.html (scale_pos_weight = neg/pos)

### Decision
Add `scale_pos_weight: float = 1.0` to `fit_gbdt_classifier`, threaded through `_fit_boosting`. For logloss_binary: per-row weight = pos_weight if y==1 else 1.0. Multiply g and h by weight in `_grad_hess`.

Add `predict(X, threshold=0.5)` to `GBDTClassifier` — threshold applied to sigmoid(pred). predict_proba unchanged.

### Alternatives considered
- **Focal loss** (Lin 2017): stronger down-weighting of easy examples; rejected — new loss variant, harder to justify without target metric.
- **SMOTE / resampling**: needs materialized synthetic rows; conflicts with <=1 batch constraint unless streaming. Deferred.
- **Cost-sensitive per-class loss**: equivalent to scale_pos_weight for binary.

## D.3.1 CoxPH - NOT IMPLEMENTED THIS SESSION

Reason: context/RAM budget. CoxPH needs ~400-500 LOC + Sort-based risk-set primitive + tests; deferred to a dedicated session; roadmap entry kept as planned.

Sources:
- https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html
- Cox 1972 JRSS B
- Efron 1977 JASA
