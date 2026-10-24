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

    # --- Backward-compat shim (removed in v0.9.47c) ---
    @property
    def source(self) -> str:
        return self.benchmark

    @property
    def url(self) -> str:
        return self.benchmark_url


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
    benchmark_results: list[BenchmarkResult]
    cadw_module: str = ""
    cadw_function: str = ""

    # --- Backward-compat shim (removed in v0.9.47c) ---
    @property
    def benchmark_result(self) -> BenchmarkResult | None:
        """Deprecated: use .benchmark_results list."""
        return self.benchmark_results[0] if self.benchmark_results else None


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
    """Parse one benchmark_result entry (old v0.9.44 or new v0.9.47b schema).

    Old schema shim: source/url/metric/value/note -> v3 fields with
    algorithm=model_name, algorithm_source_kind=description,
    attribution=literature. Note gets a [migrated] suffix.
    """
    if "source" in raw and "benchmark" not in raw:
        old_note = raw.get("note", "")
        suffix = "[migrated v0.9.47b: algorithm/algorithm_source_url unverified]"
        migrated_note = (old_note + " " + suffix).strip() if old_note else suffix
        old_val = raw["value"]
        return BenchmarkResult(
            benchmark=raw["source"],
            benchmark_url=raw["url"],
            algorithm=model_name,
            algorithm_source_url="",
            algorithm_source_kind="description",
            attribution="literature",
            reproduces=None,
            task_type="",
            n_datasets=None,
            metric=raw["metric"],
            value=float(old_val) if old_val is not None else None,
            std=None,
            note=migrated_note,
        )
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
    """Accept either old singular `benchmark_result` or new list `benchmark_results`."""
    if "benchmark_results" in m and m["benchmark_results"]:
        return [_parse_benchmark_result(r, m["name"]) for r in m["benchmark_results"]]
    if "benchmark_result" in m:
        return [_parse_benchmark_result(m["benchmark_result"], m["name"])]
    return []


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

def _validate_benchmark_result(br: BenchmarkResult, m_label: str, err) -> None:
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

    elif br.attribution == "cadw_measured":
        if not br.algorithm.startswith("cadw."):
            err(f"{lbl}: cadw_measured algorithm must be cadw.*")
        if br.benchmark != "cadw_internal":
            err(f"{lbl}: cadw_measured must use benchmark=cadw_internal")


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
                for br in m.benchmark_results:
                    if br.benchmark not in benchmarks.benchmarks:
                        err(f"{m_label}: benchmark {br.benchmark!r} "
                            f"not in benchmarks.yaml")
                    _validate_benchmark_result(br, m_label, err)

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
