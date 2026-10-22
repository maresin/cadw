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
