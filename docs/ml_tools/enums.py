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
