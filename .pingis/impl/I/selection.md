# Section I - Survival

Benchmark: CoxPH_Survival (sksurv datasets) https://github.com/sebp/scikit-survival/tree/master/sksurv/datasets

## IMPLEMENT

### I1.KaplanMeier [baseline]
- algorithm_source_url: https://en.wikipedia.org/wiki/Kaplan-Meier_estimator
- ref: https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html
- metric: c_index
- state_class: O(u) unique times (spill); bounded_state: true
- state_type: cross_row (Sort + Reduce)
- primitive_reuse: Sort, Reduce, SpillDir
- algorithm: sort (t,e) by t; stream at-risk/event counts; S(t)=prod(1-d_i/n_i); spill step function if u>state_limit

### I1.WeibullAFT [baseline]
- algorithm_source_url: https://en.wikipedia.org/wiki/Accelerated_failure_time_model
- ref: https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html
- metric: c_index
- state_class: O(d^2); bounded_state: true
- state_type: aggregation + iterative (Newton on sufficient stats)
- primitive_reuse: Reduce (XTWX, XTWy), MemoryGuard.check_state(8*d*d)
- algorithm: multi-pass Reduce + Newton step on (beta, log sigma); same pattern as cadw.ml.coxph

## SKIP (tier3_export, no algorithm)
- I1.RandomSurvivalForest - O(N) bootstrap per tree; card stays described_only
- I1.DeepSurv - torch; not implementable
- I1.DeepHit - torch; not implementable
