"""Fixed enumerations for the ML-tools matrix (M1, v0.9.44).

Keep these small and stable: adding a role/tier/combo rewrites every
YAML card that touches it. See docs/ml_tools/README.md for the rationale.
"""
from __future__ import annotations

# === Model roles (7) ===
# Orthogonal inductive biases a model brings to an ensemble.
ROLES = (
    "linear_signal",            # additive-linear in features (LogReg, Ridge)
    "nonlinear_interactions",   # trees/GBDT/kernel; interaction-heavy
    "local_structure",          # kNN/LOF/MatrixProfile; per-neighbourhood
    "probabilistic",            # NB/GMM; explicit likelihood / priors
    "temporal_dynamics",        # ARIMA/ETS/PELT; explicit time structure
    "representation",           # PCA/randSVD/MiniRocket; learned projection
    "calibration",              # Platt/isotonic; post-hoc score mapping
)

# === Tier levels (3) ===
TIER_LEVELS = (
    "tier1_core",     # implementable on CADW primitives (Reduce/Shuffle/Sort)
    "tier2_wrapper",  # thin wrapper over a light pure-Python library
    "tier3_export",   # SOTA, reached only via to_batches (not implemented here)
)

# === Implementation status flags (4) ===
# `verified_in_cadw` and `verified_by_agent` may both be False; in that case
# the card is reference-only and MUST NOT appear in ensemble_recommendation.
IMPL_FLAGS = (
    "verified_in_cadw",
    "verified_by_agent",
    "described_only",
    "not_implemented_in_cadw",
)

# === Metric combination (3) ===
COMBINATION = (
    "anchor",       # primary ranking metric for this subtype; ONE per subtype
    "diagnostic",   # printed alongside, NOT folded into a score
    "pareto_axis",  # one axis of a Pareto front (e.g. discrimination × calib)
)

TIERS = ("baseline", "sota", "ensemble")
DIRECTIONS = ("higher", "lower", "target")

# === Attribution (3) — WHO produced the number ===
# literature:     number comes from external paper / leaderboard (fetched).
# cadw_reproduced: CADW reproduced it, parity with literature proven via M2.
# cadw_measured:  CADW measured it, no external reference (memory/runtime/synthetic).
ATTRIBUTIONS = (
    "literature",
    "cadw_reproduced",
    "cadw_measured",
)

# === Algorithm source kind (5) — WHERE the algorithm is documented ===
# paper:       peer-reviewed paper (arxiv, JMLR, NeurIPS, ...).
# code:        source code (github, cadw/ml/*.py).
# docs:        official documentation (sklearn, PyOD, lifelines, ...).
# pseudocode:  algorithm box in a paper, no full code.
# description: prose only (blog, review, manual).
SOURCE_KINDS = (
    "paper",
    "code",
    "docs",
    "pseudocode",
    "description",
)


# === v0.9.51.1 (v0.3): model classes (TAB taxonomy, 5) ===
MODEL_CLASSES = (
    "non_learning",
    "ml",
    "dl_from_scratch",
    "llm_based",
    "pretrained_ts",
)

# === Learning modes (6) ===
LEARNING_MODES = (
    "unsupervised",
    "semi_supervised",
    "self_supervised",
    "supervised",
    "zero_shot",
    "few_shot",
)

# === Anomaly types (6) ===
ANOMALY_TYPES = (
    "point",
    "collective",
    "contextual",
    "changepoint",
    "shape",
    "none",
)

# === Data modality (4) ===
DATA_MODALITIES = (
    "uni",
    "multi",
    "both",
    "n/a",
)

# === Leakage risks (5) ===
LEAKAGE_RISKS = (
    "none",
    "low",
    "medium",
    "high",
    "unknown",
)

# === Evaluation scope (3) ===
EVALUATION_SCOPES = (
    "real_benchmark",
    "synthetic",
    "both",
)

# === Source libraries (12) ===
SOURCE_LIBRARIES = (
    "pyod", "sktime", "merlion", "orion", "kats", "alibi_detect",
    "stumpy", "ruptures", "telemanom", "banpei", "paddlex", "native",
)

# === Coverage (2) ===
COVERAGE = ("broad", "narrow")

# === Task family (5) ===
TASK_FAMILIES = (
    "supervised_tabular", "time_series", "unsupervised",
    "specialized", "applicability",
)

# === Task type (14) ===
TASK_TYPES = (
    "classification", "regression", "forecast", "anomaly", "density",
    "recsys", "ltr", "causal", "survival", "bandit",
    "nlp", "vision", "graph", "eda",
)
