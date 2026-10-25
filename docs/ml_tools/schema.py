"""Dataclasses + validator for the ML-tools matrix (M1, v0.9.44).

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
        super().__init__("\n".join(f"[{sev}] {msg}" for sev, msg in issues))


# === Loaded YAML entities (no defaults except where marked optional) ===

@dataclass
class BenchmarkRef:
    name: str
    url: str
    metrics: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class BenchmarkResult:
    """v3 schema (v0.9.47b): separates number source from algorithm source.

    - benchmark / benchmark_url: WHERE the NUMBER comes from
    - algorithm / algorithm_source_url / algorithm_source_kind: WHERE the
      algorithm is documented (paper / code / docs / pseudocode / description)
    - attribution: literature | cadw_reproduced | cadw_measured
    - reproduces: iff cadw_reproduced, names the literature algorithm
      (must match an `algorithm` in a sibling literature entry)
    """
    benchmark: str
    benchmark_url: str
    algorithm: str = ""
    algorithm_source_url: str = ""
    algorithm_source_kind: str = "description"
    attribution: str = "literature"
    reproduces: str | None = None
    task_type: str = ""
    n_datasets: int | None = None
    metric: str = ""
    value: float | None = None
    std: float | None = None
    note: str = ""


@dataclass
class ImplStatus:
    verified_in_cadw: bool = False
    verified_by_agent: bool = False
    described_only: bool = False
    not_implemented_in_cadw: bool = False


# v0.9.51.1: leakage info (optional, backward compatible).
@dataclass
class LeakageInfo:
    pretraining_corpus: str | None = None
    pretraining_overlap_risk: str = "unknown"
    temporal_split: bool | None = None
    target_leakage_checked: bool | None = None
    notes: str | None = None


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
    benchmark_results: list[BenchmarkResult]
    cadw_module: str = ""
    cadw_function: str = ""
    # v0.9.51.1: v0.3 optional fields (all None by default)
    task_family: str | None = None
    task_type: str | None = None
    model_class: str | None = None
    learning_mode: str | None = None
    anomaly_type: str | None = None
    domain: list[str] | None = None
    data_modality: str | None = None
    source_library: str | None = None
    library_benchmark: str | None = None
    reference_implementation: bool | None = None
    evaluation_scope: str | None = None
    dgp_source: str | None = None
    dgp_overlap_risk: str | None = None
    leakage: LeakageInfo | None = None
    coverage: str | None = None
    pareto_axes: list[str] | None = None


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

def _load_yaml(path: str | Path) -> Any:
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "pyyaml is required to load ml_tools YAML. "
            "pyyaml is a transitive dep of pytest; if missing, "
            "install with `pip install pyyaml`."
        )
    return yaml.safe_load(Path(path).read_text())


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


def _parse_benchmark_result(raw: dict, model_name: str) -> BenchmarkResult:
    """Parse one v3 benchmark_result entry."""
    new_val = raw.get("value")
    new_std = raw.get("std")
    return BenchmarkResult(
        benchmark=raw["benchmark"],
        benchmark_url=raw["benchmark_url"],
        algorithm=raw.get("algorithm", ""),
        algorithm_source_url=raw.get("algorithm_source_url", ""),
        algorithm_source_kind=raw.get("algorithm_source_kind", "description"),
        attribution=raw.get("attribution", "literature"),
        reproduces=raw.get("reproduces"),
        task_type=raw.get("task_type", ""),
        n_datasets=raw.get("n_datasets"),
        metric=raw.get("metric", ""),
        value=float(new_val) if new_val is not None else None,
        std=float(new_std) if new_std is not None else None,
        note=raw.get("note", ""),
    )


def _parse_benchmark_results(m: dict) -> list[BenchmarkResult]:
    """Parse v3 `benchmark_results` list. Empty/missing -> [] ."""
    raw = m.get("benchmark_results") or []
    return [_parse_benchmark_result(r, m["name"]) for r in raw]


def _parse_leakage(raw) -> LeakageInfo | None:
    """v0.9.51.1: parse optional leakage block (backward compatible)."""
    if not raw:
        return None
    return LeakageInfo(
        pretraining_corpus=raw.get("pretraining_corpus"),
        pretraining_overlap_risk=raw.get("pretraining_overlap_risk", "unknown"),
        temporal_split=raw.get("temporal_split"),
        target_leakage_checked=raw.get("target_leakage_checked"),
        notes=raw.get("notes"),
    )


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
        benchmark_results=_parse_benchmark_results(m),
        cadw_module=m.get("cadw_module", ""),
        cadw_function=m.get("cadw_function", ""),
        task_family=m.get("task_family"),
        task_type=m.get("task_type"),
        model_class=m.get("model_class"),
        learning_mode=m.get("learning_mode"),
        anomaly_type=m.get("anomaly_type"),
        domain=m.get("domain"),
        data_modality=m.get("data_modality"),
        source_library=m.get("source_library"),
        library_benchmark=m.get("library_benchmark"),
        reference_implementation=m.get("reference_implementation"),
        evaluation_scope=m.get("evaluation_scope"),
        dgp_source=m.get("dgp_source"),
        dgp_overlap_risk=m.get("dgp_overlap_risk"),
        leakage=_parse_leakage(m.get("leakage")),
        coverage=m.get("coverage"),
        pareto_axes=m.get("pareto_axes"),
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

def _validate_benchmark_result(br: BenchmarkResult, card: ModelCard,
                              m_label: str, err) -> None:
    """v0.9.47b: attribution-specific validation rules for one entry."""
    from .enums import ATTRIBUTIONS, SOURCE_KINDS

    lbl = f"{m_label}.{br.benchmark}.{br.metric}"

    if br.attribution not in ATTRIBUTIONS:
        err(f"{lbl}: attribution={br.attribution!r} not in {ATTRIBUTIONS}")
        return

    if br.algorithm_source_kind and br.algorithm_source_kind not in SOURCE_KINDS:
        err(f"{lbl}: algorithm_source_kind={br.algorithm_source_kind!r} "
            f"not in {SOURCE_KINDS}")

    # Note required (explains fetch quote, failure, or measurement evidence)
    if not (br.note or "").strip():
        if br.value is not None:
            err(f"{lbl}: value={br.value} but note empty")
        else:
            err(f"{lbl}: value=null but note empty (must explain)")

    if br.std is not None and (br.n_datasets is None or br.n_datasets < 2):
        err(f"{lbl}: std={br.std} requires n_datasets >= 2")

    if br.attribution == "literature":
        if br.algorithm.startswith("cadw."):
            err(f"{lbl}: literature attribution cannot use cadw.* algorithm")
        if br.reproduces is not None:
            err(f"{lbl}: literature attribution must not set 'reproduces'")

    elif br.attribution == "cadw_reproduced":
        if not br.algorithm.startswith("cadw."):
            err(f"{lbl}: cadw_reproduced algorithm must be cadw.*, "
                f"got {br.algorithm!r}")
        if br.reproduces is None:
            err(f"{lbl}: cadw_reproduced requires 'reproduces'")
        else:
            # v0.9.50.1: 'reproduces' must name an algorithm that appears
            # in a literature entry within the same card. Match by algorithm
            # only — benchmark/metric may differ (e.g. cadw_internal RMSE
            # on synthetic DGP vs M5 sMAPE on real data for VAR).
            lit_algs = {r.algorithm for r in card.benchmark_results
                        if r.attribution == "literature"}
            if br.reproduces not in lit_algs:
                err(f"{lbl}: reproduces={br.reproduces!r} not among "
                    f"literature algorithms in this card "
                    f"(available: {sorted(lit_algs)})")

    elif br.attribution == "cadw_measured":
        if not br.algorithm.startswith("cadw."):
            err(f"{lbl}: cadw_measured algorithm must be cadw.*")
        if br.benchmark != "cadw_internal":
            err(f"{lbl}: cadw_measured must use benchmark=cadw_internal")


def _validate_model_v03(m: ModelCard, m_label: str, err, warn) -> None:
    """v0.9.51.1: v0.3 optional-field validation.

    All rules fire only if the relevant field is set — existing cards
    with no v0.3 fields pass unchanged (backward compat).
    """
    from .enums import (ANOMALY_TYPES, COVERAGE, DATA_MODALITIES,
                        EVALUATION_SCOPES, LEAKAGE_RISKS, LEARNING_MODES,
                        MODEL_CLASSES, SOURCE_LIBRARIES, TASK_FAMILIES,
                        TASK_TYPES)

    def _check(name, value, allowed):
        if value is not None and value not in allowed:
            err(f"{m_label}: {name}={value!r} not in {allowed}")

    _check("task_family", m.task_family, TASK_FAMILIES)
    _check("task_type", m.task_type, TASK_TYPES)
    _check("model_class", m.model_class, MODEL_CLASSES)
    _check("learning_mode", m.learning_mode, LEARNING_MODES)
    _check("anomaly_type", m.anomaly_type, ANOMALY_TYPES)
    _check("data_modality", m.data_modality, DATA_MODALITIES)
    _check("source_library", m.source_library, SOURCE_LIBRARIES)
    _check("evaluation_scope", m.evaluation_scope, EVALUATION_SCOPES)
    _check("coverage", m.coverage, COVERAGE)
    _check("dgp_overlap_risk", m.dgp_overlap_risk, LEAKAGE_RISKS)

    if m.leakage is not None:
        if m.leakage.pretraining_overlap_risk not in LEAKAGE_RISKS:
            err(f"{m_label}: leakage.pretraining_overlap_risk="
                f"{m.leakage.pretraining_overlap_risk!r} not in "
                f"{LEAKAGE_RISKS}")

    # Rule A: pretrained/llm require leakage block
    if m.model_class in ("pretrained_ts", "llm_based"):
        if m.leakage is None:
            err(f"{m_label}: model_class={m.model_class!r} requires "
                f"'leakage' block (pretraining_corpus + overlap_risk)")
        elif m.leakage.pretraining_overlap_risk == "high":
            warn(f"{m_label}: leakage.pretraining_overlap_risk='high' "
                 f"— should be excluded or explicitly documented")

    # Rule B: cadw_reproduced requires evaluation_scope='synthetic'
    has_repro = any(br.attribution == "cadw_reproduced"
                    for br in m.benchmark_results)
    if has_repro and m.evaluation_scope is not None:
        if m.evaluation_scope != "synthetic":
            err(f"{m_label}: cadw_reproduced requires "
                f"evaluation_scope='synthetic', "
                f"got {m.evaluation_scope!r}")

    # Rule C: supervised TS requires temporal_split=True
    if (m.task_type in ("forecast", "anomaly")
            and m.learning_mode == "supervised"
            and m.leakage is not None
            and m.leakage.temporal_split is False):
        err(f"{m_label}: task_type={m.task_type!r} + "
            f"learning_mode='supervised' requires temporal_split=True")


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
                _validate_model_v03(m, m_label, err, warn)
                for br in m.benchmark_results:
                    if br.benchmark not in benchmarks.benchmarks:
                        err(f"{m_label}: benchmark {br.benchmark!r} "
                            f"not in benchmarks.yaml")
                    _validate_benchmark_result(br, m, m_label, err)

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
    return "\n".join(f"[{sev}] {msg}" for sev, msg in issues)
