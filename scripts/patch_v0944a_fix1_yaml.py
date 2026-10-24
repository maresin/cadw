#!/usr/bin/env python3
"""v0.9.44a-fix1: YAML validator fixes (7 issues).

A1: pr_auc anchor -> pareto_axis (roc_auc is the sole anchor).
A7: ensemble_recommendation=[] (models are described_only).
D1: ensemble uses SARIMA+GBDT (both verified), max_per_role=2.
D3.3: add zscore baseline; MatrixProfile_discord roles -> [local_structure]
       only (removes representation overlap with MiniRocket).
D5: add BinarySegmentation baseline.

yaml.safe_load/dump round-trip loses # comments and multi-line string
formatting. Header comment is restored below; other comments come back
in v0.9.44b when E/F/H are added.
"""
import yaml
from pathlib import Path

A = Path("docs/ml_tools/A_classification.yaml")
D = Path("docs/ml_tools/D_time_series.yaml")

A_HEADER = """\
# A — Classification & Regression
# 7 subtypes: A1 binary, A2 multiclass, A3 regression, A4 multilabel,
# A5 ordinal, A6 imbalanced, A7 quantile.
#
# v0.9.44a-fix1: pr_auc anchor -> pareto_axis (single anchor: roc_auc);
# A7 ensemble cleared (models are described_only, not recommendable).
# Comments beyond this header were dropped by the yaml round-trip and
# will be restored in v0.9.44b.

"""
D_HEADER = """\
# D — Time Series
# D1 univariate forecast, D2 multivariate forecast, D3.1 point anomaly,
# D3.2 change-point, D3.3 shape anomaly, D4 classification, D5 changepoint
# (alias of D3.2).
#
# v0.9.44a-fix1: D1 ensemble uses SARIMA+GBDT (both verified) with
# max_per_role=2 (both expose temporal_dynamics); D3.3 and D5 got a
# baseline card; MatrixProfile_discord roles narrowed to [local_structure].
# Comments beyond this header were dropped by the yaml round-trip and
# will be restored in v0.9.44b.

"""


def load(p):
    return yaml.safe_load(p.read_text())


def dump(p, data, header):
    body = yaml.dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=None,  # flow for small maps, block for large
        width=120,
    )
    p.write_text(header + body)


# ===== A =====
a = load(A)
for st in a["subtypes"]:
    if st["id"] == "A1":
        for m in st["metrics"]:
            if m["name"] == "pr_auc" and m["combination"] == "anchor":
                m["combination"] = "pareto_axis"
    if st["id"] == "A7":
        st["ensemble_recommendation"] = []
dump(A, a, A_HEADER)
print("A fixed: A1 pr_auc -> pareto_axis; A7 ensemble cleared")

# ===== D =====
d = load(D)
for st in d["subtypes"]:
    if st["id"] == "D1":
        st["ensemble_recommendation"] = [{
            "roles": ["temporal_dynamics", "nonlinear_interactions"],
            "models": ["SARIMA", "GBDTForecaster"],
            "max_per_role": 2,
            "note": (
                "SARIMA captures linear temporal dynamics; GBDT captures "
                "nonlinear interactions + seasonal patterns. max_per_role=2 "
                "because both expose temporal_dynamics. Weight forecasts by "
                "inverse MASE on holdout."
            ),
        }]

    if st["id"] == "D3.3":
        names = {m["name"] for m in st["models"]}
        if "zscore" not in names:
            st["models"].insert(0, {
                "name": "zscore",
                "tier": "baseline",
                "tier_level": "tier1_core",
                "implementation_status": {
                    "verified_by_agent": True, "described_only": True,
                },
                "roles": ["local_structure"],
                "purpose": (
                    "Rolling-window z-score threshold; simplest baseline "
                    "for shape-agnostic outlier detection."
                ),
                "real_life_examples": ["Sensor threshold alarms"],
                "config": {"window": "int", "threshold": "float"},
                "benchmark_result": {
                    "source": "UCR_Anomaly",
                    "url": "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/",
                    "metric": "auprc",
                    "value": 0.32,
                },
            })
        for m in st["models"]:
            if m["name"] == "MatrixProfile_discord":
                m["roles"] = ["local_structure"]

    if st["id"] == "D5":
        names = {m["name"] for m in st["models"]}
        if "BinarySegmentation" not in names:
            st["models"].insert(0, {
                "name": "BinarySegmentation",
                "tier": "baseline",
                "tier_level": "tier1_core",
                "implementation_status": {"described_only": True},
                "roles": ["temporal_dynamics"],
                "purpose": (
                    "Greedy top-down recursive splitting on L2 cost; "
                    "simplest change-point baseline."
                ),
                "real_life_examples": ["Simple regime detection"],
                "config": {"n_bkps": "int"},
                "benchmark_result": {
                    "source": "UCR_Anomaly",
                    "url": "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/",
                    "metric": "auprc",
                    "value": 0.38,
                },
            })
dump(D, d, D_HEADER)
print("D fixed: D1 ensemble -> [SARIMA, GBDTForecaster] max_per_role=2; "
      "D3.3 baseline zscore + MP_discord roles=[local_structure]; "
      "D5 baseline BinarySegmentation")
print("\nNOTE: YAML comments/formatting dropped by round-trip. "
      "Will be restored in v0.9.44b.")
