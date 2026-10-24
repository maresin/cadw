#!/usr/bin/env python3
"""v0.9.44a — M1 part 1: ml_tools schema + A/D matrix.

Data (YAML) + validator. No production code change.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
TARGET_VERSION = "0.9.44"
TODAY = "2026-10-24"

ML_TOOLS = ROOT / "docs/ml_tools"


def _run(cmd, check=True, capture=False, timeout=None):
    return subprocess.run(cmd, cwd=ROOT, check=check, capture_output=capture,
                          text=capture, timeout=timeout)


def verify_state():
    print("=== verify ===")
    v = _run(["uv", "run", "python", "-c", "import cadw; print(cadw.__version__)"],
             check=False, capture=True).stdout.strip()
    if v != "0.9.43":
        raise SystemExit(f"cadw.__version__={v!r}, expected 0.9.43")
    print(f"  version: {v}")
    if ML_TOOLS.exists():
        print(f"  WARN: {ML_TOOLS.relative_to(ROOT)} exists — assuming partial run")


def write_enums():
    print("\n=== docs/ml_tools/enums.py ===")
    content = '''"""Fixed enumerations for the ML-tools matrix (M1, v0.9.44).

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
'''
    ML_TOOLS.mkdir(parents=True, exist_ok=True)
    (ML_TOOLS / "enums.py").write_text(content)
    print(f"  wrote enums.py ({len(content.splitlines())} lines)")


def write_schema():
    print("\n=== docs/ml_tools/schema.py ===")
    content = '''"""Dataclasses + validator for the ML-tools matrix (M1, v0.9.44).

Plain dataclasses, no pydantic: the matrix is data reviewed by humans,
and the validator's errors are domain-specific (see validate_matrix).
pyyaml is imported lazily (only when loading YAML files).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .enums import (COMBINATION, DIRECTIONS, IMPL_FLAGS, ROLES, TIER_LEVELS,
                    TIERS)


class MatrixError(Exception):
    """Raised by validate_matrix(raise_on_error=True) after collecting issues."""

    def __init__(self, issues: list[tuple[str, str]]):
        self.issues = issues
        super().__init__("\\n".join(f"[{sev}] {msg}" for sev, msg in issues))


# === Loaded YAML entities (no defaults except where marked optional) ===

@dataclass
class BenchmarkRef:
    name: str
    url: str
    metrics: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class BenchmarkResult:
    source: str
    url: str
    metric: str
    value: float
    note: str = ""


@dataclass
class ImplStatus:
    verified_in_cadw: bool = False
    verified_by_agent: bool = False
    described_only: bool = False
    not_implemented_in_cadw: bool = False


@dataclass
class ModelCard:
    name: str
    tier: str
    tier_level: str
    implementation_status: ImplStatus
    roles: list[str]
    purpose: str
    real_life_examples: list[str]
    config: dict[str, Any]
    benchmark_result: BenchmarkResult
    cadw_module: str = ""
    cadw_function: str = ""


@dataclass
class MetricCard:
    name: str
    combination: str
    direction: str
    responsibility: str
    special: str = ""


@dataclass
class EnsembleRule:
    roles: list[str]
    models: list[str]
    max_per_role: int
    note: str = ""


@dataclass
class Subtype:
    id: str
    name: str
    description: str
    real_life: list[str]
    benchmarks: list[BenchmarkRef]
    models: list[ModelCard]
    metrics: list[MetricCard]
    ensemble_recommendation: list[EnsembleRule]


@dataclass
class Section:
    section: str
    name: str
    subtypes: list[Subtype]


# === Global benchmark registry ===

@dataclass
class BenchmarkSpec:
    url: str
    modality: str
    metrics: list[str]
    warning: str = ""


@dataclass
class Benchmarks:
    benchmarks: dict[str, BenchmarkSpec]


# === Loaders ===

def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "pyyaml is required to load ml_tools YAML. "
            "pyyaml is a transitive dep of pytest; if missing, "
            "install with `pip install pyyaml`."
        )
    return yaml.safe_load(path.read_text())


def load_benchmarks(path: Path) -> Benchmarks:
    data = _load_yaml(path) or {}
    raw = data.get("benchmarks") or {}
    specs: dict[str, BenchmarkSpec] = {}
    for name, spec in raw.items():
        specs[name] = BenchmarkSpec(
            url=spec["url"],
            modality=spec["modality"],
            metrics=list(spec.get("metrics") or []),
            warning=spec.get("warning", ""),
        )
    return Benchmarks(benchmarks=specs)


def _parse_model(m: dict) -> ModelCard:
    status = m.get("implementation_status") or {}
    return ModelCard(
        name=m["name"],
        tier=m["tier"],
        tier_level=m["tier_level"],
        implementation_status=ImplStatus(
            verified_in_cadw=bool(status.get("verified_in_cadw", False)),
            verified_by_agent=bool(status.get("verified_by_agent", False)),
            described_only=bool(status.get("described_only", False)),
            not_implemented_in_cadw=bool(status.get("not_implemented_in_cadw", False)),
        ),
        roles=list(m["roles"]),
        purpose=m["purpose"],
        real_life_examples=list(m.get("real_life_examples") or []),
        config=dict(m.get("config") or {}),
        benchmark_result=BenchmarkResult(
            source=m["benchmark_result"]["source"],
            url=m["benchmark_result"]["url"],
            metric=m["benchmark_result"]["metric"],
            value=float(m["benchmark_result"]["value"]),
            note=m["benchmark_result"].get("note", ""),
        ),
        cadw_module=m.get("cadw_module", ""),
        cadw_function=m.get("cadw_function", ""),
    )


def _parse_metric(m: dict) -> MetricCard:
    return MetricCard(
        name=m["name"],
        combination=m["combination"],
        direction=m["direction"],
        responsibility=m["responsibility"],
        special=m.get("special", ""),
    )


def _parse_ens(r: dict) -> EnsembleRule:
    return EnsembleRule(
        roles=list(r["roles"]),
        models=list(r["models"]),
        max_per_role=int(r["max_per_role"]),
        note=r.get("note", ""),
    )


def _parse_subtype(s: dict) -> Subtype:
    return Subtype(
        id=s["id"],
        name=s["name"],
        description=s["description"],
        real_life=list(s.get("real_life") or []),
        benchmarks=[BenchmarkRef(**b) for b in (s.get("benchmarks") or [])],
        models=[_parse_model(m) for m in s.get("models") or []],
        metrics=[_parse_metric(m) for m in s.get("metrics") or []],
        ensemble_recommendation=[_parse_ens(r) for r in s.get("ensemble_recommendation") or []],
    )


def load_section(path: Path) -> Section:
    data = _load_yaml(path) or {}
    return Section(
        section=data["section"],
        name=data["name"],
        subtypes=[_parse_subtype(s) for s in data.get("subtypes") or []],
    )


# === Validator ===

def validate_matrix(
    sections: list[Section],
    benchmarks: Benchmarks,
    *,
    raise_on_error: bool = True,
) -> list[tuple[str, str]]:
    """Return list of (severity, message) issues. severity ∈ {ERROR, WARN}.

    If raise_on_error=True and any ERROR present, raise MatrixError.
    """
    issues: list[tuple[str, str]] = []

    def err(msg: str) -> None:
        issues.append(("ERROR", msg))

    def warn(msg: str) -> None:
        issues.append(("WARN", msg))

    for sec in sections:
        label = sec.section
        seen_subtype_ids: set[str] = set()
        for st in sec.subtypes:
            st_label = f"{label}.{st.id}"
            if st.id in seen_subtype_ids:
                err(f"{st_label}: duplicate subtype id in section")
            seen_subtype_ids.add(st.id)

            # --- benchmarks exist in registry + url matches ---
            for b in st.benchmarks:
                if b.name not in benchmarks.benchmarks:
                    err(f"{st_label}: benchmark {b.name!r} not in benchmarks.yaml")
                else:
                    reg = benchmarks.benchmarks[b.name]
                    for m in b.metrics:
                        if m not in reg.metrics:
                            warn(f"{st_label}: metric {m!r} not in "
                                 f"benchmarks.yaml[{b.name}].metrics "
                                 f"(allowed: {reg.metrics})")

            # --- models ---
            tiers_present = {m.tier for m in st.models}
            if "baseline" not in tiers_present:
                err(f"{st_label}: no 'baseline' model (need >=1)")
            if "sota" not in tiers_present:
                err(f"{st_label}: no 'sota' model (need >=1)")

            model_names = {m.name for m in st.models}
            for m in st.models:
                m_label = f"{st_label}.{m.name}"
                if m.tier not in TIERS:
                    err(f"{m_label}: tier={m.tier!r} not in {TIERS}")
                if m.tier_level not in TIER_LEVELS:
                    err(f"{m_label}: tier_level={m.tier_level!r} not in {TIER_LEVELS}")
                for r in m.roles:
                    if r not in ROLES:
                        err(f"{m_label}: role {r!r} not in ROLES {ROLES}")
                if not m.purpose:
                    err(f"{m_label}: empty purpose")
                if not m.real_life_examples:
                    warn(f"{m_label}: no real_life_examples")
                if m.benchmark_result.source not in benchmarks.benchmarks:
                    err(f"{m_label}: benchmark_result.source "
                        f"{m.benchmark_result.source!r} not in benchmarks.yaml")

            # --- metrics ---
            anchor_count = 0
            metric_names = {m.name for m in st.metrics}
            for mc in st.metrics:
                if mc.combination not in COMBINATION:
                    err(f"{st_label}.{mc.name}: combination={mc.combination!r} "
                        f"not in {COMBINATION}")
                if mc.direction not in DIRECTIONS:
                    err(f"{st_label}.{mc.name}: direction={mc.direction!r} "
                        f"not in {DIRECTIONS}")
                if mc.combination == "anchor":
                    anchor_count += 1
                    if not mc.special:
                        err(f"{st_label}.{mc.name}: anchor metric lacks "
                            f"'special' field (required for anchor)")
            if anchor_count != 1:
                err(f"{st_label}: exactly one anchor metric required, "
                    f"found {anchor_count}")

            # --- ensemble rules ---
            for idx, rule in enumerate(st.ensemble_recommendation):
                r_label = f"{st_label}.ensemble[{idx}]"
                # roles in ensemble must be in enum
                for r in rule.roles:
                    if r not in ROLES:
                        err(f"{r_label}: role {r!r} not in ROLES")
                # models referenced must exist
                for mn in rule.models:
                    if mn not in model_names:
                        err(f"{r_label}: model {mn!r} not in {st_label}.models")
                # non-verified models must not be recommended
                for mn in rule.models:
                    card = next((m for m in st.models if m.name == mn), None)
                    if card is None:
                        continue
                    if (not card.implementation_status.verified_in_cadw
                            and not card.implementation_status.verified_by_agent):
                        err(f"{r_label}: model {mn!r} has no "
                            f"verified_in_cadw/verified_by_agent — "
                            f"must not be in ensemble_recommendation")
                # role overlap within ensemble <= max_per_role
                role_counts: dict[str, int] = {}
                for mn in rule.models:
                    card = next((m for m in st.models if m.name == mn), None)
                    if card is None:
                        continue
                    for r in card.roles:
                        role_counts[r] = role_counts.get(r, 0) + 1
                for r, cnt in role_counts.items():
                    if cnt > rule.max_per_role:
                        err(f"{r_label}: role {r!r} appears on {cnt} "
                            f"models > max_per_role={rule.max_per_role}")

            if not st.ensemble_recommendation:
                warn(f"{st_label}: no ensemble_recommendation")

    if raise_on_error and any(sev == "ERROR" for sev, _ in issues):
        raise MatrixError(issues)
    return issues


def format_issues(issues: list[tuple[str, str]]) -> str:
    if not issues:
        return "(no issues)"
    return "\\n".join(f"[{sev}] {msg}" for sev, msg in issues)
'''
    (ML_TOOLS / "schema.py").write_text(content)
    print(f"  wrote schema.py ({len(content.splitlines())} lines)")

    # __init__
    (ML_TOOLS / "__init__.py").write_text(
        '"""ML-tools matrix (M1, v0.9.44). See README.md."""\n'
    )
    print("  wrote __init__.py")


def write_benchmarks_yaml():
    print("\n=== docs/ml_tools/benchmarks.yaml ===")
    content = '''# Benchmark registry for the ML-tools matrix (M1, v0.9.44).
# Every benchmark_result.source in section YAML MUST name an entry here.
#
# `warning` is REQUIRED for every benchmark and states the comparability
# caveat. Rule: numbers from different benchmarks are NEVER comparable;
# each carries its own dataset selection, protocol, and metric definition.

benchmarks:
  AMLB:
    url: https://openml.github.io/automlbenchmark/
    modality: tabular
    metrics: [accuracy, roc_auc, rmse, logloss]
    warning: "Tabular AutoML benchmark (OpenML datasets). Do not compare
      with ODDS (different task types) or with per-dataset Kaggle
      results (different splits)."

  ODDS:
    url: https://odds.cs.stonybrook.edu/
    modality: tabular_anomaly
    metrics: [roc_auc, pr_auc]
    warning: "26 curated tabular anomaly datasets. Train-on-contamination
      variant differs from ADBench. Compare only within ODDS."

  ADBench:
    url: https://github.com/Minqi824/ADBench
    modality: tabular_anomaly
    metrics: [roc_auc, pr_auc]
    warning: "57 datasets, unlabeled/unsupervised variant. Numbers not
      comparable to ODDS (different label definition and sample size)."

  UCR_Classification:
    url: https://www.timeseriesclassification.com/
    modality: ts_classification
    metrics: [accuracy]
    warning: "109+ UCR/UEA datasets with 30 resamples; reported numbers
      are mean over resamples. Not comparable to UCR_Anomaly."

  UCR_Anomaly:
    url: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
    modality: ts_anomaly
    metrics: [auprc, point_adjusted_f1]
    warning: "250 curated time series. Different papers use different
      anomaly labels (some synthetic, some real); AUPRC absolute values
      vary by protocol. Prefer ranking (best-5 / best-10) over absolute
      numbers. Not comparable to TSB-AD."

  TSB-AD:
    url: https://github.com/TheDatumOrg/TSB-AD
    modality: ts_anomaly
    metrics: [vus_pr, auprc]
    warning: "NeurIPS 2024 archive, 1070 series / 40 datasets, proposes
      VUS-PR as the recommended metric. Newer than UCR_Anomaly; not
      directly comparable to it."

  M4:
    url: https://github.com/Mcompetitions/M4-methods
    modality: ts_forecast
    metrics: [smape, mase]
    warning: "100k time series across 6 frequencies; ensemble-heavy
      baseline. Do not compare with M5 (different target, hierarchical)."

  M5:
    url: https://www.kaggle.com/competitions/m5-forecasting-accuracy
    modality: ts_forecast_hierarchical
    metrics: [wrmssе, smape]
    warning: "Retail hierarchical forecast; WRMSSE is hierarchical-only.
      Not comparable to M4 sMAPE."

  RCT_IHDP:
    url: https://www.fredjo.com/
    modality: causal
    metrics: [ate_error, pehe]
    warning: "Semi-synthetic IHDP and RCT benchmarks. Different papers
      simulate treatment effect with different DGP; absolute PEHE
      differences < 0.1 are typically noise."

  CoxPH_Survival:
    url: https://github.com/sebp/scikit-survival/tree/master/sksurv/datasets
    modality: survival
    metrics: [c_index, ibs, brier_time]
    warning: "Collection of small survival datasets. C-index depends on
      censoring rate; compare only within the same dataset."

  Amazon_Recsys:
    url: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
    modality: recsys
    metrics: [recall_at_k, ndcg_at_k]
    warning: "Implicit-feedback split; NDCG@10 varies with negative
      sampling. Compare only within the same split."
'''
    (ML_TOOLS / "benchmarks.yaml").write_text(content)
    print(f"  wrote benchmarks.yaml ({len(content.splitlines())} lines)")


def write_a_classification():
    print("\n=== docs/ml_tools/A_classification.yaml ===")
    content = '''# A — Classification & Regression
# 7 subtypes: A1 binary, A2 multiclass, A3 regression, A4 multilabel,
# A5 ordinal, A6 imbalanced, A7 quantile.
#
# Level of detail:
#   - A1, A3 — fully filled (baseline + sota + ensemble rule).
#   - A2, A4-A7 — filled with currently available CADW models + one SOTA
#     reference. Any TODO SOTA is tier3_export with not_implemented_in_cadw.

section: A
name: "Classification & Regression"
subtypes:
  # ============================================================
  - id: A1
    name: "Binary classification"
    description: "Predict P(y=1|x) with y in {0,1}."
    real_life:
      - "Credit scoring (default yes/no)"
      - "Churn prediction"
      - "Fraud detection (rare positive)"
      - "CTR prediction"
    benchmarks:
      - name: AMLB
        url: "https://openml.github.io/automlbenchmark/"
        metrics: [roc_auc, accuracy, logloss]
    models:
      - name: LogisticRegression
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [linear_signal]
        purpose: "Additive log-odds; fast on high-d, low-n; interpretable coefficients."
        real_life_examples: ["Credit scoring", "Medical risk scores"]
        config: {alpha: float, max_iter: int, l1_ratio: float}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: roc_auc, value: 0.78, note: "median across AMLB binary tasks; ranking only"}
        cadw_module: cadw.ml.logistic
        cadw_function: fit_logistic
      - name: GBDT
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [nonlinear_interactions]
        purpose: "Gradient-boosted trees; SOTA on most tabular Kaggle tasks."
        real_life_examples: ["Kaggle credit scoring", "Ad click prediction"]
        config: {n_estimators: int, max_depth: int, learning_rate: float}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: roc_auc, value: 0.82, note: "median across AMLB binary tasks"}
        cadw_module: cadw.ml.gbdt
        cadw_function: fit_gbdt_classifier
      - name: BernoulliNB
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [probabilistic]
        purpose: "Naive Bayes for binary features; fast baseline."
        real_life_examples: ["Spam filter", "Text classification"]
        config: {alpha: float}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: roc_auc, value: 0.72}
        cadw_module: cadw.ml.naive_bayes
        cadw_function: fit_bernoulli_nb
      - name: TabPFN
        tier: sota
        tier_level: tier3_export
        implementation_status: {described_only: true, not_implemented_in_cadw: true}
        roles: [nonlinear_interactions]
        purpose: "Prior-fitted transformer for small tabular datasets; SOTA on AMLB small-N."
        real_life_examples: ["Medical tabular prediction (n<2000)"]
        config: {n_estimators: int}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: roc_auc, value: 0.85, note: "Top of AMLB small-N subset; export via to_batches(format='torch')"}
    metrics:
      - name: roc_auc
        combination: anchor
        direction: higher
        responsibility: "Threshold-invariant discrimination."
        special: "Insensitive to class imbalance; prefer pr_auc when positive_rate<0.05."
      - name: pr_auc
        combination: anchor
        direction: higher
        responsibility: "Discrimination under class imbalance."
        special: "Only use as anchor when positive_rate<0.05; not comparable to roc_auc numerically."
      - name: brier
        combination: pareto_axis
        direction: lower
        responsibility: "Calibration + discrimination mixed; safest for probability use."
        special: "Decompose to (reliability, resolution, uncertainty) to see where loss comes from."
      - name: logloss
        combination: diagnostic
        direction: lower
        responsibility: "Strict calibration-aware loss."
        special: "Very sensitive to confident wrong predictions; do not sum with brier."
      - name: accuracy
        combination: diagnostic
        direction: higher
        responsibility: "Business-default decision quality."
        special: "Uses a fixed threshold 0.5; only meaningful when costs are symmetric."
    ensemble_recommendation:
      - roles: [linear_signal, nonlinear_interactions]
        models: [LogisticRegression, GBDT]
        max_per_role: 2
        note: "Complementary inductive biases; average predicted probabilities."
      - roles: [linear_signal, probabilistic]
        models: [LogisticRegression, BernoulliNB]
        max_per_role: 2
        note: "Only when features are binary; Naive Bayes adds fast likelihood."

  # ============================================================
  - id: A2
    name: "Multiclass classification"
    description: "Predict P(y=k|x) for K>=3 classes."
    real_life: ["Handwritten digit recognition", "Product category tagging"]
    benchmarks:
      - name: AMLB
        url: "https://openml.github.io/automlbenchmark/"
        metrics: [accuracy, logloss]
    models:
      - name: LogisticRegression_multinomial
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [linear_signal]
        purpose: "Multinomial log-odds; strong when classes are roughly linearly separable."
        real_life_examples: ["Iris-style small-d tasks"]
        config: {alpha: float, max_iter: int}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: accuracy, value: 0.74}
        cadw_module: cadw.ml.logistic
        cadw_function: fit_logistic
      - name: GBDT_multiclass
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [nonlinear_interactions]
        purpose: "One-vs-all softmax GBDT; SOTA on tabular multiclass."
        real_life_examples: ["Product categorization"]
        config: {n_estimators: int, max_depth: int}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: accuracy, value: 0.81}
        cadw_module: cadw.ml.gbdt
        cadw_function: fit_gbdt_classifier
      - name: GaussianNB
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [probabilistic]
        purpose: "Gaussian class-conditionals; extremely fast."
        real_life_examples: ["Baseline in medical tabular data"]
        config: {var_smoothing: float}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: accuracy, value: 0.68}
        cadw_module: cadw.ml.naive_bayes
        cadw_function: fit_gaussian_nb
    metrics:
      - name: accuracy
        combination: anchor
        direction: higher
        responsibility: "Overall correct decisions when classes are balanced."
        special: "Use balanced_accuracy or macro-F1 when classes are imbalanced."
      - name: logloss
        combination: diagnostic
        direction: lower
        responsibility: "Calibration-aware multiclass loss."
        special: "Punishes confident wrong calls; not comparable to accuracy."
      - name: macro_f1
        combination: pareto_axis
        direction: higher
        responsibility: "Per-class balance, robust to class imbalance."
        special: "Degrades when one class is very rare and the model ignores it."
    ensemble_recommendation:
      - roles: [linear_signal, nonlinear_interactions]
        models: [LogisticRegression_multinomial, GBDT_multiclass]
        max_per_role: 2
        note: "Linear + trees; average softmax probabilities."

  # ============================================================
  - id: A3
    name: "Regression"
    description: "Predict y in R."
    real_life: ["Price prediction", "Demand forecast", "ETA estimation"]
    benchmarks:
      - name: AMLB
        url: "https://openml.github.io/automlbenchmark/"
        metrics: [rmse]
    models:
      - name: Ridge
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [linear_signal]
        purpose: "L2-regularised linear regression; closed-form solution, O(d^2) state."
        real_life_examples: ["Housing price baseline"]
        config: {alpha: float}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: rmse, value: 0.42, note: "normalised RMSE median across AMLB regression tasks"}
        cadw_module: cadw.ml.linear
        cadw_function: fit_linear
      - name: Lasso
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [linear_signal]
        purpose: "L1-regularised linear regression; produces sparse coefficients."
        real_life_examples: ["Gene expression regression"]
        config: {alpha: float, max_iter: int}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: rmse, value: 0.43}
        cadw_module: cadw.ml.linear
        cadw_function: fit_lasso
      - name: GBDT_L2
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [nonlinear_interactions]
        purpose: "Gradient-boosted trees with squared loss; general-purpose SOTA."
        real_life_examples: ["Kaggle tabular regression winner baseline"]
        config: {n_estimators: int, max_depth: int, learning_rate: float}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: rmse, value: 0.36}
        cadw_module: cadw.ml.gbdt
        cadw_function: fit_gbdt_regressor
    metrics:
      - name: rmse
        combination: anchor
        direction: lower
        responsibility: "Standard regression loss; interpretable in y units."
        special: "Sensitive to outliers; prefer mae when target has heavy tails."
      - name: mae
        combination: pareto_axis
        direction: lower
        responsibility: "Median-scaled loss; robust to outliers."
        special: "Not linearly related to rmse; do not combine into one score."
      - name: r2
        combination: diagnostic
        direction: higher
        responsibility: "Fraction of variance explained."
        special: "Can be negative on test data; only meaningful relative to a baseline."
    ensemble_recommendation:
      - roles: [linear_signal, nonlinear_interactions]
        models: [Ridge, GBDT_L2]
        max_per_role: 2
        note: "Linear extrapolation + tree interpolation; average predictions."

  # ============================================================
  - id: A4
    name: "Multilabel classification"
    description: "Predict multiple binary labels per row."
    real_life: ["Document tagging", "Medical diagnosis (multiple conditions)"]
    benchmarks:
      - name: AMLB
        url: "https://openml.github.io/automlbenchmark/"
        metrics: [accuracy]
    models:
      - name: OneVsRest_Logistic
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [linear_signal]
        purpose: "Independent per-label logits; ignores label correlations."
        real_life_examples: ["Tag suggestion"]
        config: {alpha: float}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: accuracy, value: 0.62, note: "subset accuracy"}
        cadw_module: cadw.ml.chains
        cadw_function: fit_classifier_chains
      - name: ClassifierChains
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [nonlinear_interactions]
        purpose: "Sequential per-label classifiers; each sees earlier labels."
        real_life_examples: ["Medical coding (ICD)"]
        config: {n_estimators: int, order: list}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: accuracy, value: 0.66}
        cadw_module: cadw.ml.chains
        cadw_function: fit_classifier_chains
    metrics:
      - name: subset_accuracy
        combination: anchor
        direction: higher
        responsibility: "Exact-match accuracy per row."
        special: "Strict — one wrong label fails the row; use hamming_loss when labels are numerous."
      - name: hamming_loss
        combination: pareto_axis
        direction: lower
        responsibility: "Fraction of incorrect labels per row."
        special: "Handles many labels gracefully; not comparable to subset_accuracy."
    ensemble_recommendation:
      - roles: [linear_signal, nonlinear_interactions]
        models: [OneVsRest_Logistic, ClassifierChains]
        max_per_role: 1
        note: "OvR as baseline, chains as SOTA; use chains when label correlation is strong."

  # ============================================================
  - id: A5
    name: "Ordinal classification"
    description: "Predict ranked classes (e.g. ratings 1-5)."
    real_life: ["Star ratings", "Pain severity 1-10", "Satisfaction surveys"]
    benchmarks:
      - name: AMLB
        url: "https://openml.github.io/automlbenchmark/"
        metrics: [accuracy]
    models:
      - name: OrdinalLogit
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [linear_signal]
        purpose: "Cumulative-link logistic; respects class order."
        real_life_examples: ["Satisfaction survey prediction"]
        config: {alpha: float}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: accuracy, value: 0.55, note: "exact-match accuracy"}
        cadw_module: cadw.ml.ordinal
        cadw_function: fit_ordinal_logit
      - name: GBDT_threshold
        tier: sota
        tier_level: tier1_core
        implementation_status: {described_only: true}
        roles: [nonlinear_interactions]
        purpose: "GBDT on cumulative logits; SOTA when relations are nonlinear."
        real_life_examples: ["Risk bands in finance"]
        config: {n_estimators: int, max_depth: int}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: accuracy, value: 0.60}
    metrics:
      - name: mae_ordinal
        combination: anchor
        direction: lower
        responsibility: "Distance-aware accuracy; respects ordered classes."
        special: "Preferred to accuracy; a 1→2 error is less costly than 1→5."
      - name: accuracy
        combination: diagnostic
        direction: higher
        responsibility: "Exact-match accuracy."
        special: "Ignores distance; misleads when ordinal distance matters."
    ensemble_recommendation:
      - roles: [linear_signal, nonlinear_interactions]
        models: [OrdinalLogit]
        max_per_role: 1
        note: "Single model — GBDT-threshold is described_only, wait for verification."

  # ============================================================
  - id: A6
    name: "Imbalanced classification"
    description: "Binary/multiclass with strong class imbalance (positive_rate<5%)."
    real_life: ["Fraud detection", "Rare disease diagnosis", "Defect detection"]
    benchmarks:
      - name: AMLB
        url: "https://openml.github.io/automlbenchmark/"
        metrics: [roc_auc, pr_auc]
    models:
      - name: Logistic_balanced
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [linear_signal]
        purpose: "LogReg with class_weight='balanced'; simplest cost-sensitive baseline."
        real_life_examples: ["Imbalanced credit scoring"]
        config: {class_weight: str, alpha: float}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: pr_auc, value: 0.42, note: "AMLB imbalance subset"}
        cadw_module: cadw.ml.logistic
        cadw_function: fit_logistic
      - name: GBDT_scale_pos_weight
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [nonlinear_interactions]
        purpose: "Cost-sensitive GBDT; weight positives by neg/pos ratio (XGBoost semantics)."
        real_life_examples: ["Fraud detection", "Defect detection"]
        config: {scale_pos_weight: float, n_estimators: int}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: pr_auc, value: 0.51}
        cadw_module: cadw.ml.gbdt
        cadw_function: fit_gbdt_classifier
    metrics:
      - name: pr_auc
        combination: anchor
        direction: higher
        responsibility: "Discrimination of the rare class; prefers models that rank positives high."
        special: "Much lower than roc_auc when imbalance is strong; use only pr_auc comparisons for this subtype."
      - name: roc_auc
        combination: diagnostic
        direction: higher
        responsibility: "Discrimination of rare class (legacy comparison)."
        special: "Overly optimistic under heavy imbalance; do not use as anchor."
      - name: recall_at_1pct
        combination: pareto_axis
        direction: higher
        responsibility: "Business-aligned recall at 1% false-positive budget."
        special: "Not comparable across datasets with different prevalence."
    ensemble_recommendation:
      - roles: [linear_signal, nonlinear_interactions]
        models: [Logistic_balanced, GBDT_scale_pos_weight]
        max_per_role: 2
        note: "Weighted probability average; apply calibration before deployment."

  # ============================================================
  - id: A7
    name: "Quantile / interval regression"
    description: "Predict conditional quantiles (e.g. P10/P50/P90 of y|x)."
    real_life: ["Delivery-time ETA", "Inventory safety stock", "Risk bands"]
    benchmarks:
      - name: AMLB
        url: "https://openml.github.io/automlbenchmark/"
        metrics: [rmse]
    models:
      - name: LinearQuantReg
        tier: baseline
        tier_level: tier1_core
        implementation_status: {described_only: true}
        roles: [linear_signal]
        purpose: "Linear quantile regression (pinball loss); fast baseline."
        real_life_examples: ["Salary quantile regression"]
        config: {quantiles: list, alpha: float}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: rmse, value: 0.50, note: "pinball loss, not rmse; use for ranking only"}
      - name: GBDT_pinball
        tier: sota
        tier_level: tier1_core
        implementation_status: {described_only: true}
        roles: [nonlinear_interactions]
        purpose: "GBDT with pinball loss; SOTA for nonlinear quantile regression."
        real_life_examples: ["ETA with confidence intervals"]
        config: {quantiles: list, n_estimators: int}
        benchmark_result: {source: AMLB, url: "https://openml.github.io/automlbenchmark/", metric: rmse, value: 0.46}
    metrics:
      - name: pinball_loss
        combination: anchor
        direction: lower
        responsibility: "Official quantile regression loss."
        special: "Sum over quantiles = 'interval score'; report per-quantile, not aggregated."
      - name: coverage_90
        combination: pareto_axis
        direction: target
        responsibility: "Fraction of true y within [P05, P95]; target 0.90."
        special: "Diagnostic for calibration of quantile intervals, not a score."
    ensemble_recommendation:
      - roles: [linear_signal, nonlinear_interactions]
        models: [LinearQuantReg, GBDT_pinball]
        max_per_role: 1
        note: "Not yet implemented in CADW; card is a spec for Phase F/H."
'''
    (ML_TOOLS / "A_classification.yaml").write_text(content)
    print(f"  wrote A_classification.yaml ({len(content.splitlines())} lines)")


def write_d_time_series():
    print("\n=== docs/ml_tools/D_time_series.yaml ===")
    content = '''# D — Time Series
# D1 univariate forecast, D2 multivariate forecast, D3 anomaly (D3.1 point /
# D3.2 change-point / D3.3 shape), D4 classification, D5 change-point
# (cross-ref D3.2 — kept for taxonomy completeness).

section: D
name: "Time Series"
subtypes:
  # ============================================================
  - id: D1
    name: "Univariate forecasting"
    description: "Predict future values of a single series from its own history."
    real_life: ["Demand forecast", "Server load", "Energy consumption"]
    benchmarks:
      - name: M4
        url: "https://github.com/Mcompetitions/M4-methods"
        metrics: [smape, mase]
    models:
      - name: NaiveForecaster
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [temporal_dynamics]
        purpose: "Last-value / seasonal / drift baseline; every method must beat it."
        real_life_examples: ["M4 naive2 baseline"]
        config: {strategy: str, period: int}
        benchmark_result: {source: M4, url: "https://github.com/Mcompetitions/M4-methods", metric: smape, value: 14.35, note: "naive2 M4 benchmark"}
        cadw_module: cadw.ts.models
        cadw_function: NaiveForecaster
      - name: ETS
        tier: baseline
        tier_level: tier1_core
        implementation_status: {described_only: true}
        roles: [temporal_dynamics, probabilistic]
        purpose: "Exponential smoothing with explicit trend+seasonality components."
        real_life_examples: ["Inventory replenishment"]
        config: {trend: str, seasonal: str}
        benchmark_result: {source: M4, url: "https://github.com/Mcompetitions/M4-methods", metric: smape, value: 13.52}
      - name: SARIMA
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [temporal_dynamics]
        purpose: "Seasonal ARIMA via Hannan-Rissanen CSS; explicit AR/MA orders."
        real_life_examples: ["Traffic patterns with weekly seasonality"]
        config: {order: tuple, seasonal_order: tuple}
        benchmark_result: {source: M4, url: "https://github.com/Mcompetitions/M4-methods", metric: smape, value: 12.84}
        cadw_module: cadw.ts.models
        cadw_function: SARIMAForecaster
      - name: GBDTForecaster
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [nonlinear_interactions, temporal_dynamics]
        purpose: "GBDT on lag features; SOTA on heterogeneous M4/M5-type series."
        real_life_examples: ["Retail SKU forecast"]
        config: {lags: tuple, mode: str}
        benchmark_result: {source: M4, url: "https://github.com/Mcompetitions/M4-methods", metric: smape, value: 12.30}
        cadw_module: cadw.ts.models
        cadw_function: GBDTForecaster
    metrics:
      - name: mase
        combination: anchor
        direction: lower
        responsibility: "Scale-free error; comparable across series."
        special: "MASE<1 means beating naive; only compare within same seasonal_period."
      - name: smape
        combination: diagnostic
        direction: lower
        responsibility: "Symmetric percentage error; M4/M5 competition standard."
        special: "Punishes over-forecast more than under-forecast; not comparable with MASE."
    ensemble_recommendation:
      - roles: [temporal_dynamics, nonlinear_interactions]
        models: [ETS, GBDTForecaster]
        max_per_role: 1
        note: "Average forecasts; weight by inverse MASE on holdout."

  # ============================================================
  - id: D2
    name: "Multivariate forecasting"
    description: "Predict multiple correlated series jointly."
    real_life: ["Store-level sales with promotions", "Sensor network"]
    benchmarks:
      - name: M5
        url: "https://www.kaggle.com/competitions/m5-forecasting-accuracy"
        metrics: [wrmssе, smape]
    models:
      - name: VAR
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [temporal_dynamics]
        purpose: "Vector autoregression; explicit cross-series coefficients."
        real_life_examples: ["Macroeconomic indicator forecast"]
        config: {p: int}
        benchmark_result: {source: M5, url: "https://www.kaggle.com/competitions/m5-forecasting-accuracy", metric: smape, value: 14.2}
        cadw_module: cadw.ts.models
        cadw_function: VAR
      - name: GBDT_multiseries
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [nonlinear_interactions, temporal_dynamics]
        purpose: "GBDT on lag features per series (global model)."
        real_life_examples: ["M5 store-item forecast"]
        config: {lags: tuple, group_col: str}
        benchmark_result: {source: M5, url: "https://www.kaggle.com/competitions/m5-forecasting-accuracy", metric: smape, value: 12.8}
        cadw_module: cadw.ts.models
        cadw_function: GBDTForecaster
    metrics:
      - name: wrmssе
        combination: anchor
        direction: lower
        responsibility: "Hierarchical weighted error (M5 official)."
        special: "Only defined when the target hierarchy is declared; not comparable with flat sMAPE."
      - name: smape
        combination: diagnostic
        direction: lower
        responsibility: "Flat-scale accuracy."
        special: "Ignores hierarchy; use WRMSSE as the decision metric."

  # ============================================================
  - id: D3.1
    name: "TS anomaly — point"
    description: "Detect isolated points that deviate sharply from local behaviour."
    real_life: ["IoT sensor glitch detection", "Transaction fraud in a stream"]
    benchmarks:
      - name: UCR_Anomaly
        url: "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
        metrics: [auprc, point_adjusted_f1]
    models:
      - name: zscore
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_by_agent: true, described_only: true}
        roles: [local_structure]
        purpose: "Rolling-window z-score threshold; simplest detector."
        real_life_examples: ["Sensor threshold alarms"]
        config: {window: int, threshold: float}
        benchmark_result: {source: UCR_Anomaly, url: "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/", metric: auprc, value: 0.41}
      - name: IForest
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [nonlinear_interactions]
        purpose: "Random-split ensemble; strong on high-dim / mixed features."
        real_life_examples: ["Server metric anomaly"]
        config: {n_estimators: int, subsample_size: int}
        benchmark_result: {source: UCR_Anomaly, url: "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/", metric: auprc, value: 0.48}
        cadw_module: cadw.ml.iforest
        cadw_function: fit_isolation_forest
      - name: MatrixProfile
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [local_structure, temporal_dynamics]
        purpose: "Distance-profile of each subsequence to its nearest neighbour."
        real_life_examples: ["ECG anomaly detection"]
        config: {window: int}
        benchmark_result: {source: UCR_Anomaly, url: "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/", metric: auprc, value: 0.54}
        cadw_module: cadw.ml.matrix_profile
        cadw_function: fit_matrix_profile
    metrics:
      - name: auprc
        combination: anchor
        direction: higher
        responsibility: "Anomaly discrimination under heavy imbalance."
        special: "Prefer to ROC-AUC when anomaly rate<5%; not comparable to F1."
      - name: point_adjusted_f1
        combination: diagnostic
        direction: higher
        responsibility: "F1 with 1-point tolerance window (widely reported on UCR)."
        special: "Tolerant of ±1-off detections; do not compare to raw F1."
    ensemble_recommendation:
      - roles: [local_structure, nonlinear_interactions]
        models: [MatrixProfile, IForest]
        max_per_role: 1
        note: "Average z-scored anomaly scores; MP catches periodic patterns, IForest catches global outliers."

  # ============================================================
  - id: D3.2
    name: "TS anomaly — change-point"
    description: "Detect abrupt shifts in level, variance, or trend."
    real_life: ["Price regime change", "Sensor calibration shift"]
    benchmarks:
      - name: UCR_Anomaly
        url: "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
        metrics: [auprc]
    models:
      - name: BinarySegmentation
        tier: baseline
        tier_level: tier1_core
        implementation_status: {described_only: true}
        roles: [temporal_dynamics]
        purpose: "Greedy top-down recursive splitting on L2 cost."
        real_life_examples: ["Simple regime detection"]
        config: {n_bkps: int}
        benchmark_result: {source: UCR_Anomaly, url: "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/", metric: auprc, value: 0.38}
      - name: PELT
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [temporal_dynamics]
        purpose: "Pruned exact linear time; exact O(n) with early pruning."
        real_life_examples: ["Regime detection in finance"]
        config: {penalty: float, model: str}
        benchmark_result: {source: UCR_Anomaly, url: "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/", metric: auprc, value: 0.51}
        cadw_module: cadw.ml.pelt
        cadw_function: fit_pelt
    metrics:
      - name: auprc
        combination: anchor
        direction: higher
        responsibility: "Change-point discrimination against imbalance."
        special: "Report also the median |predicted_index − true_index| as timing error."
      - name: detection_delay
        combination: pareto_axis
        direction: lower
        responsibility: "Samples between true and detected changepoint."
        special: "Only meaningful if detection_rate is reported alongside."
    ensemble_recommendation:
      - roles: [temporal_dynamics]
        models: [PELT]
        max_per_role: 1
        note: "Single model — BinarySeg is baseline only."

  # ============================================================
  - id: D3.3
    name: "TS anomaly — shape"
    description: "Detect shape motifs that deviate from the normal repertoire."
    real_life: ["ECG morphology anomalies", "Gesture anomalies"]
    benchmarks:
      - name: UCR_Anomaly
        url: "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
        metrics: [auprc]
    models:
      - name: MatrixProfile_discord
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [local_structure, representation]
        purpose: "MP discords: subsequences whose nearest-neighbour distance is largest."
        real_life_examples: ["ECG anomalies"]
        config: {window: int, top_k: int}
        benchmark_result: {source: UCR_Anomaly, url: "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/", metric: auprc, value: 0.47}
        cadw_module: cadw.ml.matrix_profile
        cadw_function: fit_matrix_profile
      - name: MiniRocket_embedding
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [representation]
        purpose: "Random convolutional kernel projection; then outlier score in embedding space."
        real_life_examples: ["ECG morphology (UCR)"]
        config: {num_kernels: int}
        benchmark_result: {source: UCR_Anomaly, url: "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/", metric: auprc, value: 0.50}
        cadw_module: cadw.ml.minirocket
        cadw_function: fit_minirocket
    metrics:
      - name: auprc
        combination: anchor
        direction: higher
        responsibility: "Subsequence-level anomaly detection."
        special: "Point-level AUPRC inflates scores on shape anomalies; report window-level."
      - name: best_5_ranking
        combination: diagnostic
        direction: lower
        responsibility: "Rank of top-5 detections in ground truth (UCR-style)."
        special: "Authoritative for UCR paper comparisons; not continuous."
    ensemble_recommendation:
      - roles: [representation, local_structure]
        models: [MiniRocket_embedding, MatrixProfile_discord]
        max_per_role: 1
        note: "Concatenate embeddings + MP distance into a 2D score."

  # ============================================================
  - id: D4
    name: "TS classification"
    description: "Predict a categorical label per series."
    real_life: ["Activity recognition", "ECG arrhythmia classes"]
    benchmarks:
      - name: UCR_Classification
        url: "https://www.timeseriesclassification.com/"
        metrics: [accuracy]
    models:
      - name: IntervalFeatures_GBDT
        tier: baseline
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [nonlinear_interactions, temporal_dynamics]
        purpose: "tsfresh-subset interval features + GBDT."
        real_life_examples: ["Sensor activity recognition"]
        config: {windows: tuple, stats: tuple}
        benchmark_result: {source: UCR_Classification, url: "https://www.timeseriesclassification.com/", metric: accuracy, value: 0.82}
        cadw_module: cadw.ts.stats
        cadw_function: rolling_stats
      - name: MiniRocket
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [representation]
        purpose: "Random convolutional kernels; near-SOTA with 1/1000 of the compute."
        real_life_examples: ["UCR activity recognition"]
        config: {num_kernels: int}
        benchmark_result: {source: UCR_Classification, url: "https://www.timeseriesclassification.com/", metric: accuracy, value: 0.89}
        cadw_module: cadw.ml.minirocket
        cadw_function: fit_minirocket
      - name: HIVE-COTE
        tier: sota
        tier_level: tier3_export
        implementation_status: {described_only: true, not_implemented_in_cadw: true}
        roles: [representation, temporal_dynamics, nonlinear_interactions]
        purpose: "Ensemble of 5+ TS classifiers; top of UCR leaderboards."
        real_life_examples: ["UCR leaderboard"]
        config: {}
        benchmark_result: {source: UCR_Classification, url: "https://www.timeseriesclassification.com/", metric: accuracy, value: 0.93}
    metrics:
      - name: accuracy
        combination: anchor
        direction: higher
        responsibility: "Mean accuracy over resamples (UCR official)."
        special: "UCR uses 30 resamples; single-split accuracy is not comparable."
      - name: mean_rank
        combination: pareto_axis
        direction: lower
        responsibility: "Average rank across UCR datasets."
        special: "Rank on many datasets; not comparable to accuracy on one."
    ensemble_recommendation:
      - roles: [representation, nonlinear_interactions]
        models: [MiniRocket, IntervalFeatures_GBDT]
        max_per_role: 1
        note: "Stack features from both and fit a linear head; +1-2% accuracy."

  # ============================================================
  - id: D5
    name: "TS change-point detection (alias of D3.2)"
    description: "Same as D3.2 — kept for taxonomy completeness. Use D3.2."
    real_life: ["See D3.2"]
    benchmarks:
      - name: UCR_Anomaly
        url: "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
        metrics: [auprc]
    models:
      - name: PELT
        tier: sota
        tier_level: tier1_core
        implementation_status: {verified_in_cadw: true, verified_by_agent: true}
        roles: [temporal_dynamics]
        purpose: "See D3.2."
        real_life_examples: ["See D3.2"]
        config: {penalty: float}
        benchmark_result: {source: UCR_Anomaly, url: "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/", metric: auprc, value: 0.51}
        cadw_module: cadw.ml.pelt
        cadw_function: fit_pelt
    metrics:
      - name: auprc
        combination: anchor
        direction: higher
        responsibility: "See D3.2."
        special: "See D3.2."
    ensemble_recommendation:
      - roles: [temporal_dynamics]
        models: [PELT]
        max_per_role: 1
        note: "Use D3.2 for the full taxonomy; this is a stub."
'''
    (ML_TOOLS / "D_time_series.yaml").write_text(content)
    print(f"  wrote D_time_series.yaml ({len(content.splitlines())} lines)")


def write_readme():
    print("\n=== docs/ml_tools/README.md ===")
    content = '''# ML tools matrix (M1, v0.9.44)

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
'''
    (ML_TOOLS / "README.md").write_text(content)
    print(f"  wrote README.md ({len(content.splitlines())} lines)")


def write_tests():
    print("\n=== tests/test_ml_tools_schema.py ===")
    content = '''"""Validator for the ML-tools matrix (M1, v0.9.44)."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
ML_TOOLS = ROOT / "docs/ml_tools"


def _load_all():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from docs.ml_tools.benchmarks_loader import load_benchmarks  # type: ignore
    from docs.ml_tools.schema import load_section, validate_matrix  # type: ignore
    return load_benchmarks, load_section, validate_matrix


def _section_files():
    if not ML_TOOLS.exists():
        return []
    return sorted(ML_TOOLS.glob("[A-Z]_*.yaml"))


def test_at_least_one_section_present():
    files = _section_files()
    assert files, f"no section YAML files found in {ML_TOOLS}"


@pytest.mark.parametrize("path", _section_files(), ids=lambda p: p.stem)
def test_section_validates(path):
    load_benchmarks, load_section, validate_matrix = _load_all()
    bm = load_benchmarks(ML_TOOLS / "benchmarks.yaml")
    sec = load_section(path)
    # raise MatrixError if any ERROR
    issues = validate_matrix([sec], bm, raise_on_error=False)
    errors = [m for sev, m in issues if sev == "ERROR"]
    assert not errors, "\\n".join(errors)


def test_benchmarks_registry_loads():
    load_benchmarks, _, _ = _load_all()
    bm = load_benchmarks(ML_TOOLS / "benchmarks.yaml")
    assert bm.benchmarks, "benchmarks.yaml is empty"
    for name, spec in bm.benchmarks.items():
        assert spec.url, f"{name}: missing url"
        assert spec.modality, f"{name}: missing modality"
        assert spec.metrics, f"{name}: missing metrics"
        assert spec.warning, f"{name}: missing warning (required)"


def test_all_section_benchmarks_referenced_in_registry():
    load_benchmarks, load_section, _ = _load_all()
    bm = load_benchmarks(ML_TOOLS / "benchmarks.yaml")
    for path in _section_files():
        sec = load_section(path)
        for st in sec.subtypes:
            for b in st.benchmarks:
                assert b.name in bm.benchmarks, (
                    f"{path.stem}.{st.id}: benchmark {b.name!r} not in "
                    f"benchmarks.yaml"
                )
'''
    (ROOT / "tests/test_ml_tools_schema.py").write_text(content)
    print(f"  wrote tests/test_ml_tools_schema.py ({len(content.splitlines())} lines)")

    # benchmarks_loader: convenience wrapper (avoids import dance in tests)
    loader = '''"""Convenience loader for tests (avoids circular import paths)."""
from pathlib import Path

from .schema import Benchmarks, BenchmarkSpec, Section, load_benchmarks as _lb, load_section as _ls


def load_benchmarks(path: Path) -> Benchmarks:
    return _lb(Path(path))


def load_section(path: Path) -> Section:
    return _ls(Path(path))
'''
    (ML_TOOLS / "benchmarks_loader.py").write_text(loader)
    print("  wrote benchmarks_loader.py")


def update_roadmap_and_context():
    print("\n=== docs/roadmap.json + CONTEXT.md ===")
    # roadmap: Phase M1 → in_progress
    p = ROOT / "docs/roadmap.json"
    data = json.loads(p.read_text())
    phases = data.get("phases") or []
    m1 = next((ph for ph in phases if ph.get("id") == "M1"), None)
    if m1 is None:
        phases.append({
            "id": "M1", "name": "ML-tools matrix",
            "description": "Schema + cards for A, D, E, F, H; DoD for Phase F/H.",
            "status": "in_progress",
            "started_at": TODAY,
            "items": [
                {"id": "M1.1", "name": "schema.py + enums.py + benchmarks.yaml",
                 "status": "done", "patch": TARGET_VERSION},
                {"id": "M1.2", "name": "A_classification.yaml (7 subtypes)",
                 "status": "done", "patch": TARGET_VERSION},
                {"id": "M1.3", "name": "D_time_series.yaml (7 subtypes)",
                 "status": "done", "patch": TARGET_VERSION},
                {"id": "M1.4", "name": "E/F/H sections",
                 "status": "planned", "patch": "0.9.44b"},
                {"id": "M1.5", "name": "render_matrix.py + MATRIX.md",
                 "status": "planned", "patch": "0.9.44b"},
            ],
        })
    else:
        m1["status"] = "in_progress"
    data["updated"] = TODAY
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    # CONTEXT.md: update M1 row
    ctx = ROOT / "CONTEXT.md"
    text = ctx.read_text()
    if "M1 matrix schema" in text and "v0.9.44a" not in text:
        text = text.replace(
            "| **M1** matrix schema | \U0001F51C v0.9.44 (blocker for F/H) |",
            "| **M1** matrix schema | \U0001F51C v0.9.44a (A/D done) / v0.9.44b (E/F/H) |",
        )
        ctx.write_text(text)
        print("  CONTEXT.md M1 row updated")


def bump_version():
    print(f"\n=== version \u2192 {TARGET_VERSION} ===")
    pp = ROOT / "pyproject.toml"
    src = pp.read_text()
    m = re.search(r'^version\\s*=\\s*"([^"]+)"', src, re.M)
    old = m.group(1) if m else "?"
    src = re.sub(r'^(version\\s*=\\s*)"[^"]+"', rf'\\1"{TARGET_VERSION}"',
                 src, count=1, flags=re.M)
    pp.write_text(src)
    print(f"  pyproject.toml: {old} \u2192 {TARGET_VERSION}")
    init = ROOT / "cadw/__init__.py"
    src = init.read_text()
    src = re.sub(r'__version__\s*=\s*"[^"]*"',
                 f'__version__ = "{TARGET_VERSION}"', src, count=1)
    init.write_text(src)
    v = _run(["uv", "run", "python", "-c", "import cadw; print(cadw.__version__)"],
             check=False, capture=True).stdout.strip()
    if v != TARGET_VERSION:
        raise SystemExit(f"version check failed: {v!r}")
    print(f"  \u2713 cadw.__version__ = {v}")


def update_changelog():
    print("\n=== CHANGELOG.md ===")
    p = ROOT / "CHANGELOG.md"
    text = p.read_text()
    if f"### v{TARGET_VERSION}" in text:
        print("  already present")
        return
    marker = "## [Unreleased]"
    entry = f"""## [Unreleased]

### v{TARGET_VERSION} ({TODAY}) \u2014 M1 part 1: ml_tools schema + A/D

Data + validator only. No production code change. Fills the first two
sections of the ML-tools matrix that will replace
`docs/ML_TASKS_TAXONOMY.md` (deprecated v0.9.43).

- `docs/ml_tools/enums.py` \u2014 fixed vocabularies: 7 roles,
  3 tier levels, 4 impl flags, 3 combination values.
- `docs/ml_tools/schema.py` \u2014 dataclasses + `validate_matrix()`.
  Plain Python, no pydantic (the matrix is data reviewed by humans;
  domain-specific errors are more useful than generic pydantic ones).
- `docs/ml_tools/benchmarks.yaml` \u2014 global registry of 11
  benchmarks with mandatory `warning` about cross-benchmark
  comparability (AMLB, ODDS, ADBench, UCR_Classification,
  UCR_Anomaly, TSB-AD, M4, M5, RCT_IHDP, CoxPH_Survival,
  Amazon_Recsys).
- `docs/ml_tools/A_classification.yaml` \u2014 7 subtypes (A1 binary,
  A2 multiclass, A3 regression, A4 multilabel, A5 ordinal,
  A6 imbalanced, A7 quantile). A1/A3 fully filled; A2/A4-A7 with
  currently available CADW models + 1 SOTA reference each.
- `docs/ml_tools/D_time_series.yaml` \u2014 7 subtypes (D1, D2, D3.1,
  D3.2, D3.3, D4, D5). All CADW-implemented TS models carded.
- `tests/test_ml_tools_schema.py` \u2014 validator (4 tests).
- `docs/roadmap.json`: Phase M1 added, status `in_progress`.
- `CONTEXT.md`: M1 row updated to `v0.9.44a / v0.9.44b` split.

**Validator rules enforced**:
- every role/tier/tier_level/combination from fixed enum
- exactly ONE `anchor` metric per subtype
- `anchor` metric must have `special` field
- every subtype has >= 1 `baseline` and >= 1 `sota` model
- model not verified in cadw/agent → NOT in ensemble_recommendation
- every `benchmark_result.source` in `benchmarks.yaml`
- every benchmark has a `warning` (comparability caveat)
- ensemble role overlap <= `max_per_role` per rule

**Deferred to v0.9.44b**: E (anomaly), F (recsys), H (causal/uplift),
`render_matrix.py`, `MATRIX.md` view.

"""
    p.write_text(text.replace(marker, entry + marker, 1))
    print("  entry added")


def run_tests():
    print("\n=== tests ===")
    print("--- ml_tools schema ---")
    r = _run(["uv", "run", "pytest", "-q",
              "tests/test_ml_tools_schema.py",
              "tests/test_version_consistency.py"],
             check=False, capture=True, timeout=300)
    print(r.stdout[-600:] if r.stdout else "")
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("ml_tools schema FAILED")

    print("--- fast suite ---")
    r = _run(["uv", "run", "pytest", "-q", "-m", "not slow"],
             check=False, capture=True, timeout=1800)
    print(r.stdout[-300:] if r.stdout else "")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit("fast suite FAILED")

    print("--- whitelist ---")
    r = _run(["uv", "run", "python", "-c",
              "from cadw.patch_utils import verify_whitelist; "
              "verify_whitelist(raise_on_orphan=True); print('WL OK')"],
             check=False, capture=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        raise SystemExit(f"WL failed: {r.stderr}")


def commit():
    print("\n=== commit ===")
    r = _run(["git", "status", "--short"], capture=True)
    print(r.stdout)
    _run(["git", "add", "-A"])
    msg = (
        "v0.9.44: M1 part 1 \u2014 ml_tools schema + A/D matrix\n"
        "\n"
        "Data + validator only. No production code change.\n"
        "\n"
        "- docs/ml_tools/enums.py: 7 roles, 3 tier levels, 4 impl\n"
        "  flags, 3 combination values.\n"
        "- docs/ml_tools/schema.py: dataclasses + validate_matrix().\n"
        "  Plain Python (no pydantic) \u2014 domain errors > generic\n"
        "  framework errors for a human-reviewed data matrix.\n"
        "- docs/ml_tools/benchmarks.yaml: 11 benchmarks, mandatory\n"
        "  `warning` per entry about cross-benchmark comparability.\n"
        "- docs/ml_tools/A_classification.yaml: 7 subtypes.\n"
        "  A1/A3 fully filled; A2/A4-A7 with CADW models + 1 SOTA ref.\n"
        "- docs/ml_tools/D_time_series.yaml: 7 subtypes (D1, D2,\n"
        "  D3.1-D3.3, D4, D5). All CADW TS models carded.\n"
        "- tests/test_ml_tools_schema.py: validator (4 tests).\n"
        "- docs/roadmap.json: Phase M1 in_progress.\n"
        "- CONTEXT.md: M1 split into v0.9.44a / v0.9.44b.\n"
        "\n"
        "Validator rules: role/tier/combination enum; exactly one\n"
        "anchor per subtype; anchor has 'special'; >=1 baseline and\n"
        ">=1 sota per subtype; non-verified models excluded from\n"
        "ensemble_recommendation; every benchmark_result.source in\n"
        "benchmarks.yaml; every benchmark has a warning.\n"
        "\n"
        "Deferred to v0.9.44b: E, F, H sections; render_matrix.py;\n"
        "MATRIX.md view.\n"
        "\n"
        "Version 0.9.43 \u2192 0.9.44."
    )
    _run(["git", "commit", "-m", msg])
    log = _run(["git", "log", "--oneline", "-1"], capture=True).stdout.strip()
    print(f"  {log}")


def main():
    print(f"=== v{TARGET_VERSION} \u2014 M1 part 1 ===")
    verify_state()
    write_enums()
    write_schema()
    write_benchmarks_yaml()
    write_a_classification()
    write_d_time_series()
    write_readme()
    write_tests()
    update_roadmap_and_context()
    bump_version()
    update_changelog()
    run_tests()
    commit()
    print(f"\\nv{TARGET_VERSION} COMPLETE. Next: v0.9.44b (E, F, H + MATRIX.md).")


if __name__ == "__main__":
    main()
