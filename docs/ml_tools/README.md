# ML tools matrix (M1, v0.9.44)

Per-section YAML cards describing "what to use and why" for each ML
task type: baselines, SOTA, metric roles, ensemble rules.

**Start here:** `enums.py` for the fixed vocabulary; `schema.py` for
the data model; then any `<Section>_*.yaml`.

## Files

| File | Purpose |
|---|---|
| `enums.py` | Roles (7), tier levels (3), impl flags (4), combination (3) |
| `schema.py` | dataclasses + `validate_matrix()` |
| `benchmarks.yaml` | Global registry of benchmarks (name → url, modality, metrics, warning) |
| `A_classification.yaml` | A1-A7 |
| `D_time_series.yaml` | D1, D2, D3.1, D3.2, D3.3, D4, D5 |
| (E, F, H) | Filled in v0.9.44b |

## Roles (invariant set)

- `linear_signal` — additive-linear in features
- `nonlinear_interactions` — trees / GBDT / kernel
- `local_structure` — per-neighbourhood (kNN / LOF / Matrix Profile)
- `probabilistic` — NB / GMM
- `temporal_dynamics` — ARIMA / ETS / PELT
- `representation` — PCA / randSVD / MiniRocket
- `calibration` — Platt / isotonic

Ensembles should have ≤2 models per role (`max_per_role` in each rule).

## Tier levels

- `tier1_core` — implemented natively on CADW primitives
- `tier2_wrapper` — thin wrapper over a light pure-Python library
- `tier3_export` — SOTA reached only via `to_batches` (not in CADW)

**Rule**: `tier3_export` models MUST NOT appear in `ensemble_recommendation`.

## Metric combination

- `anchor` — the primary ranking metric; exactly ONE per subtype
- `diagnostic` — printed alongside; not combined into a score
- `pareto_axis` — one axis of a Pareto front (discrimination × calibration, etc.)

**Never sum two diagnostics into one scalar**. If a single number is
required, use the `anchor`. Diagnostics are for diagnosing, not scoring.

## Benchmark comparability

`benchmarks.yaml` gives a `warning` per benchmark. Rule:

**Numbers from different benchmarks are NEVER comparable.** Different
datasets, different protocols, different metric definitions. Read
`warning` before copying any number into a report.

## Adding a new model card

1. Pick `subtype` in the section YAML.
2. Add a model under `models:` with `name`, `tier`, `tier_level`,
   `implementation_status`, `roles`, `purpose`, `real_life_examples`,
   `config`, `benchmark_result`, `cadw_module`, `cadw_function`.
3. `purpose` must be 2-3 sentences explaining the model's inductive bias.
4. `real_life_examples` — at least one concrete use case.
5. Run `pytest -q tests/test_ml_tools_schema.py` — validator catches
   missing fields, unknown roles, non-verified models in ensembles.

## Status legend

- `verified_in_cadw: true` — model implemented + tested + bench
- `verified_by_agent: true` — agent has read source / paper and confirmed
- `described_only: true` — described here but not yet implemented
- `not_implemented_in_cadw: true` — reached only via `to_batches`

If both `verified_*` are false → **not eligible for ensembles**.
