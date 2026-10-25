"""Validator for the ML-tools matrix (M1, v0.9.44; v0.9.48.8 coverage)."""
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
    issues = validate_matrix([sec], bm, raise_on_error=False)
    errors = [m for sev, m in issues if sev == "ERROR"]
    assert not errors, "\n".join(errors)


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


# ============================================================
# v0.9.48.8: per-subtype coverage tests.
# ============================================================


def _all_subtypes_full():
    _, load_section, _ = _load_all()
    out = []
    for path in _section_files():
        sec = load_section(path)
        for st in sec.subtypes:
            out.append((path, sec.section, st))
    return out


@pytest.mark.parametrize(
    "path,section_id,subtype",
    _all_subtypes_full(),
    ids=lambda x: (x.id if hasattr(x, "id") else (x.stem if hasattr(x, "stem") else str(x))),
)
def test_each_subtype_has_benchmarks_and_roles(path, section_id, subtype):
    load_benchmarks, _, _ = _load_all()
    bm = load_benchmarks(ML_TOOLS / "benchmarks.yaml")
    label = f"{section_id}.{subtype.id}"
    assert subtype.benchmarks, f"{label}: no benchmarks declared"
    for b in subtype.benchmarks:
        assert b.name in bm.benchmarks, (
            f"{label}: benchmark {b.name!r} not in benchmarks.yaml"
        )
    for m in subtype.models:
        assert m.roles, f"{label}.{m.name}: no roles declared"


def test_ensemble_recommendation_only_verified_in_cadw():
    # Scoped to sections filled in v0.9.48 (I,N,J,M,O,L,K). Legacy A-H are 
    # v2-schema and must not be modified per TASK_v0.9.48.
    NEW_SECTIONS = set('I N J M O L K'.split())
    errors = []
    for path, section_id, st in _all_subtypes_full():
        if section_id not in NEW_SECTIONS:
            continue
        by_name = {m.name: m for m in st.models}
        for i, rule in enumerate(st.ensemble_recommendation):
            for mn in rule.models:
                m = by_name.get(mn)
                if m is None:
                    errors.append(
                        f"{section_id}.{st.id}.ensemble[{i}]: model {mn!r} "
                        f"not among the subtype models"
                    )
                    continue
                if not m.implementation_status.verified_in_cadw:
                    errors.append(
                        f"{section_id}.{st.id}.ensemble[{i}]: model {mn!r} "
                        f"is not verified_in_cadw"
                    )
    assert not errors, "".join(errors)


def test_subtype_metric_names_cover_benchmark_result_metrics():
    NEW_SECTIONS = set('I N J M O L K'.split())
    errors = []
    for path, section_id, st in _all_subtypes_full():
        if section_id not in NEW_SECTIONS:
            continue
        declared = {mc.name for mc in st.metrics}
        used = set()
        for m in st.models:
            for br in m.benchmark_results:
                if br.metric:
                    used.add(br.metric)
        missing = used - declared
        if missing:
            errors.append(
                f"{section_id}.{st.id}: benchmark_results use undeclared "
                f"metric(s) {sorted(missing)}; declared={sorted(declared)}"
            )
    assert not errors, "".join(errors)
