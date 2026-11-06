# Audit v0.9.112 - SUMMARY

## Counts

| Metric | Count |
|---|---|
| Total audited | 25 |
| `verified-ready` | 0 |
| `needs-patch` | 14 |
| `draft` | 6 |
| `tier1_core_exception` | 5 |
| Priority high | 6 |
| Priority medium | 15 |
| Priority low | 4 |
| Confidence high | 9 |
| Confidence medium | 16 |
| Confidence low | 0 |

## Slot resolutions (01, 02, 05)

- **slot_01:** `A_classification.yaml:GBDT:fit_gbdt_classifier` - bucket=`needs-patch`
- **slot_02:** `A_classification.yaml:GBDT_L2:fit_gbdt_regressor` - bucket=`needs-patch`
- **slot_05:** `A_classification.yaml:GBDT_multiclass:fit_gbdt_classifier` - bucket=`needs-patch`

> Note: the 3 free slots landed on the same `cadw/ml/gbdt.py` module, so cards 01/02/05 share one finding. These were the first 3 alphabetical non-explicit pending candidates.

## Group patterns

### Group A - ReduceOps / small state (GBDT, LDA, CoxPH, ALS, T_learner)
- Cards: 01, 02, 03, 04, 05, 24, 25
- Common finding: all have state guards and streaming loops; blockers are YAML/measurement, not algorithm.
- Recommendation: formula + component-wise test; AP2 in T_learner needs refactor.

### Group B - Anomaly / windowed (ZMS, Hampel, Sub-PCA, KShape, EIF, Autoencoder, KMeansAD, PatchMemoryAD, S2G, MiniRocket, DWT-MLEAD, BOCPD)
- Cards: 06-17
- Common finding: all use chunked windows; 8 of 12 return O(n) per-point scores/labels (DEC-013 class).
- Recommendation: SpillArray batch (06,07,08,14,16); others formula+measure.

### Group C - TS forecasters (ETS, AutoETS, Theta, AutoTheta, SARIMA, Naive)
- Cards: 18-23
- Common finding: single-series, O(n) series retained; `tier1_core_exception` wart must become `bounded_by_domain` (DEC-011).
- Recommendation: docs-only for 18-22; NaiveForecaster also needs an O(1)->O(n) YAML correction.

### Group D - Other
- Cards: 24 (ALS), 25 (T_learner)

## YAML lies found (count: 7)

| Card | Field | YAML says | Code shows | Evidence |
|---|---|---|---|---|
| `GBDT` | `state_class / bounded_state / n_scalability` | `O(batch) / true / constant` | O(n) internal state: est_n_state = n_total*32 | `cadw/ml/gbdt.py:997` |
| `GBDT_L2` | `state_class / bounded_state / n_scalability` | `O(batch) / true / constant` | O(n) internal state: est_n_state = n_total*32 | `cadw/ml/gbdt.py:997` |
| `GBDT_multiclass` | `state_class / bounded_state / n_scalability` | `O(batch) / true / constant` | np.full(n,-1) + np.full((n,K),nan); guard n_total*8*2 | `cadw/ml/gbdt.py:1512-1513, :1598` |
| `LDA` | `bounded_state_dims` | `{d: 1}` | {n_topics, vocab_size} - persistent K*V*f64 model | `cadw/ml/lda.py:368, :87-90` |
| `CoxPH` | `state_class / bounded_state / n_scalability` | `O(n) / false / linear` | O(d^2 + pending); n streamed via external_sort + _iter_filtered_batches | `cadw/ml/coxph.py:460, :474-500` |
| `NaiveForecaster` | `state_class / bounded_state / n_scalability / bounded_state_dims` | `O(1) / true / constant / {}` | O(n): self.y_ = y (full series retained) | `cadw/ts/models.py:220 (docstring :187)` |
| `T_learner` | `bounded_state / n_scalability` | `true / constant` | O(n*d): Xs.append + np.vstack(Xs) in _collect | `cadw/ml/meta_learners.py:137-144` |

## ROADMAP<->YAML drift (out of scope - human patches)

| ROADMAP says | YAML reality | Recommendation for human |
|---|---|---|
| `fit_pca` (CovarianceOp) | **no YAML card** | add card, OR remove from ROADMAP |
| `fit_kmeans` (k knob) | **no YAML card** | add card, OR remove from ROADMAP |
| `fit_tfidf` | status=`draft` (J_nlp) | audit draft separately |

## Recent OOC work without status flip

| Card | Function | Last commit | What it did |
|---|---|---|---|
| `ZMS` | `fit_zms` | `v0.9.91-S6a strided+chunked` | OOC fix merged, verification pending |
| `Hampel` | `fit_hampel` | `v0.9.91-S6a strided+chunked` | OOC fix merged, verification pending |
| `Sub-PCA` | `fit_sub_pca` | `v0.9.87-P4a streaming` | OOC fix merged, verification pending |
| `KShapeAD` | `fit_kshape` | `v0.9.87-P3 streaming` | OOC fix merged, verification pending |
| `EIF` | `fit_eif` | `v0.9.91-S6b-eif chunked fit` | OOC fix merged, verification pending |
| `Autoencoder` | `fit_autoencoder` | `v0.9.91-S6d chunked SGD` | OOC fix merged, verification pending |
| `KMeansAD` | `fit_kmeans_ad` | `v0.9.91-S6c chunked OOC` | OOC fix merged, verification pending |
| `PatchMemoryAD` | `fit_patch_memory_ad` | `v0.9.87-P4d SpillDir+memmap` | OOC fix merged, verification pending |
| `S2G` | `fit_s2g` | `v0.9.87-P4c streaming+dedup` | OOC fix merged, verification pending |
| `DWT-MLEAD` | `fit_dwt_mlead` | `v0.9.87-P4b streaming mahalanobis` | OOC fix merged, verification pending |
| `BOCPD` | `fit_bocpd` | `v0.9.87-fix1c rolling history` | OOC fix merged, verification pending |
| `ALS` | `fit_als` | `v0.9.87-P5a-fix-als streaming dict` | OOC fix merged, verification pending |
| `LDA` | `fit_lda` | `v0.9.87-P5a-fix-lda sparse E-step` | OOC fix merged, verification pending |
| `T_learner` | `fit_meta_learner` | `v0.9.90-S3 G2 fixes` | OOC fix merged, verification pending |

## Common blockers

| Blocker | Count | Cards |
|---|---|---|
| `implementation_status.status` missing | 25 | all |
| `ram_formula`/`formula_evidence`/`measurement` missing | 25 | all |
| No verification test `tests/test_<fn>_verification.py` | 25 | all |
| O(n) output (DEC-013 class) | 9 | ZMS, Hampel, Sub-PCA, S2G, DWT-MLEAD, BOCPD, EIF, Autoencoder, KMeansAD |
| AP2 (append+concat) | 4 | Sub-PCA, MiniRocket(transform), DWT-MLEAD, T_learner |
| AP1 (np.zeros/full(n)) | 3 | DWT-MLEAD, GBDT_multiclass, SARIMA |
| `tier1_core_exception` wart (DEC-011) | 5 | ETS, AutoETS, Theta, AutoTheta, SARIMA |
| No `batch_size` kwarg | 2 | GBDT (x3 cards), CoxPH |

## Do-not-re-solve (skip in v0.9.112)

- `fit_zms` x3 - DEC-013 (O(n) output; SpillArray needed)
- `fit_pelt` - DEC-014 (bounded_by_domain is modifier)
- `fit_cusum` - DEC-030
- verified cards (16) - already closed
- described_only (19) and not_implemented_in_cadw (tier3_export) - out of scope

## Suggested batches for v0.9.112

### Batch 1 - quick wins (<=30 min each)
- 15 MiniRocket - cheap flip (fixed kernel state)
- 24 ALS - cheap flip (formula already in guard)
- 09 KShapeAD - formula already written at kshape.py:443-449

### Batch 2 - medium (<=2h each)
- 03 LDA, 04 CoxPH, 10 EIF, 11 Autoencoder, 12 KMeansAD, 13 PatchMemoryAD, 17 BOCPD, 25 T_learner (refactor)
- 01/02/05 GBDT - needs batch_size + state-guard rework

### Batch 3 - needs-prereq (SpillArray, DEC-013)
- 06 ZMS, 07 Hampel, 08 Sub-PCA, 14 S2G, 16 DWT-MLEAD

### Batch 4 - tier1_core_exception (docs only, no code)
- 18 ETS, 19 AutoETS, 20 Theta, 21 AutoTheta, 22 SARIMA (bounded_by_domain)
- 23 NaiveForecaster - also fix O(1)->O(n) YAML claim

## Open questions for human

- Q1: 5 explicit `cadw_function` names in the task table are not module-level functions: `fit_auto_ets`, `fit_auto_theta`, `fit_sarima`, `fit_naive_forecaster`, `fit_meta_learners`. Actual YAML `cadw_function` values are class names (`AutoETSForecaster`, `AutoThetaForecaster`, `SARIMAForecaster`, `NaiveForecaster`) and `fit_meta_learner`. Should the task table be aligned to YAML?
- Q2: ETS/Theta are only reachable through DataFrame bindings `cadw/dataframe/_ml_bindings.py:850` and `:867`. Should these become module-level `fit_ets`/`fit_theta` wrappers for consistency with other tier1_core cards?
- Q3: 9 cards return O(n) per-point scores/labels (DEC-013 class). Is a shared SpillArray output type planned for v0.9.112, or should these stay `draft`?
- Q4: 5 cards still carry the `tier1_core_exception` wart (DEC-011). Approve replacing with `bounded_by_domain` in v0.9.112?
- Q5: All 25 cards have no `ram_formula`/`formula_evidence`/`measurement`. Should formula+measurement be done as one batch by module family?

