"""Validator for the ML-tools matrix (M1, v0.9.44; v0.9.48.8 coverage)."""
from __future__ import annotations

import sys
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


# ============================================================
# v0.9.50.1: validator enforces 'reproduces' sibling match
# ============================================================

def test_cadw_reproduced_reproduces_must_match_literature_algorithm():
    """cadw_reproduced.reproduces must name an algorithm present in a
    literature entry of the same card. Guards against fabricated parity
    claims ('CADW reproduces Algorithm-X' when Algorithm-X has no
    literature row in this card)."""
    from docs.ml_tools.schema import (
        BenchmarkResult, ModelCard, ImplStatus, validate_matrix,
        Section, Subtype, MetricCard,
    )
    from docs.ml_tools.schema import Benchmarks, BenchmarkSpec

    lit = BenchmarkResult(
        benchmark="AMLB", benchmark_url="https://amlb.example/",
        algorithm="LightGBM", algorithm_source_url="https://papers.example/lg.pdf",
        algorithm_source_kind="paper", attribution="literature",
        task_type="binary classification", metric="roc_auc",
        value=0.842, note="fetched; quote",
    )
    bad_repro = BenchmarkResult(
        benchmark="AMLB", benchmark_url=".pingis/benches/reproduction/gbdt.json",
        algorithm="cadw.ml.gbdt.fit_gbdt_regressor",
        algorithm_source_url="cadw/ml/gbdt.py", algorithm_source_kind="code",
        attribution="cadw_reproduced",
        reproduces="XGBoost",  # NOT in literature siblings
        task_type="binary classification", metric="roc_auc",
        value=0.838, note="M2",
    )
    card = ModelCard(
        name="GBDT", tier="sota", tier_level="tier1_core",
        implementation_status=ImplStatus(verified_in_cadw=True),
        roles=["nonlinear_interactions"], purpose="x",
        real_life_examples=["y"], config={},
        benchmark_results=[lit, bad_repro],
    )
    st = Subtype(
        id="A1", name="binary", description="d", real_life=[],
        benchmarks=[], models=[card], metrics=[
            MetricCard(name="roc_auc", combination="anchor",
                       direction="higher", responsibility="r", special="s"),
        ], ensemble_recommendation=[],
    )
    sec = Section(section="A", name="A", subtypes=[st])
    bm = Benchmarks(benchmarks={
        "AMLB": BenchmarkSpec(url="https://amlb.example/", modality="tabular",
                              metrics=["roc_auc"], warning="w"),
    })
    issues = validate_matrix([sec], bm, raise_on_error=False)
    errors = [msg for sev, msg in issues if sev == "ERROR"]
    assert any("reproduces='XGBoost' not among literature algorithms" in e
               for e in errors), errors

# ============================================================
# v0.9.51.1: v0.3 schema extension — backward compatible
# ============================================================


def _make_card(**kwargs):
    """Minimal valid ModelCard for v0.3 tests (one literature entry)."""
    from docs.ml_tools.schema import ModelCard, ImplStatus, BenchmarkResult
    defaults = dict(
        name="TestModel", tier="sota", tier_level="tier1_core",
        implementation_status=ImplStatus(verified_in_cadw=True),
        roles=["nonlinear_interactions"], purpose="test",
        real_life_examples=["test"], config={},
        benchmark_results=[BenchmarkResult(
            benchmark="AMLB", benchmark_url="https://example.com/",
            algorithm="x", algorithm_source_url="https://example.com/x",
            algorithm_source_kind="paper", attribution="literature",
            task_type="test", metric="roc_auc",
            value=0.5, note="test",
        )],
    )
    defaults.update(kwargs)
    return ModelCard(**defaults)


def _make_section(card):
    """Wrap a test card in a Subtype with a dummy baseline + metric card."""
    from docs.ml_tools.schema import (Section, Subtype, MetricCard,
                                       ModelCard, ImplStatus)
    baseline = ModelCard(
        name="__baseline__", tier="baseline", tier_level="tier1_core",
        implementation_status=ImplStatus(verified_in_cadw=True),
        roles=["linear_signal"], purpose="b",
        real_life_examples=["b"], config={}, benchmark_results=[],
    )
    st = Subtype(
        id="X1", name="t", description="d", real_life=[],
        benchmarks=[], models=[baseline, card],
        metrics=[MetricCard(
            name="roc_auc", combination="anchor", direction="higher",
            responsibility="r", special="s",
        )],
        ensemble_recommendation=[],
    )
    return Section(section="X", name="X", subtypes=[st])


def _make_benchmarks():
    from docs.ml_tools.schema import Benchmarks, BenchmarkSpec
    return Benchmarks(benchmarks={
        "AMLB": BenchmarkSpec(
            url="https://amlb.example/", modality="tabular",
            metrics=["roc_auc", "accuracy", "pr_auc"],
            warning="w",
        ),
    })


def _validate_card(card):
    from docs.ml_tools.schema import validate_matrix
    sec = _make_section(card)
    return validate_matrix([sec], _make_benchmarks(), raise_on_error=False)


def _errors_of(issues):
    return [m for sev, m in issues if sev == "ERROR"]


def _warns_of(issues):
    return [m for sev, m in issues if sev == "WARN"]


def test_v03_all_sections_still_validate():
    """v0.3 fields are optional; existing 13 sections must still pass."""
    load_benchmarks, load_section, validate_matrix = _load_all()
    bm = load_benchmarks(ML_TOOLS / "benchmarks.yaml")
    for path in _section_files():
        sec = load_section(path)
        issues = validate_matrix([sec], bm, raise_on_error=False)
        errors = _errors_of(issues)
        assert not errors, f"{path.stem}:\n" + "\n".join(errors)


def test_v03_enum_fields_accept_known_values():
    card = _make_card(
        task_family="time_series", task_type="anomaly",
        model_class="ml", learning_mode="unsupervised",
        anomaly_type="point", data_modality="uni",
        source_library="pyod", evaluation_scope="real_benchmark",
        coverage="broad",
    )
    errors = _errors_of(_validate_card(card))
    assert not errors, errors


def test_v03_model_class_invalid_value_raises():
    card = _make_card(model_class="quantum_model")
    errors = _errors_of(_validate_card(card))
    assert any("model_class='quantum_model'" in e for e in errors), errors


def test_v03_learning_mode_invalid_value_raises():
    card = _make_card(learning_mode="telepathic")
    errors = _errors_of(_validate_card(card))
    assert any("learning_mode='telepathic'" in e for e in errors), errors


def test_v03_pretrained_requires_leakage_info():
    card = _make_card(model_class="pretrained_ts")
    errors = _errors_of(_validate_card(card))
    assert any("requires 'leakage' block" in e for e in errors), errors


def test_v03_pretrained_with_leakage_passes():
    from docs.ml_tools.schema import LeakageInfo
    card = _make_card(
        model_class="pretrained_ts",
        leakage=LeakageInfo(
            pretraining_corpus="unknown",
            pretraining_overlap_risk="unknown",
        ),
    )
    errors = _errors_of(_validate_card(card))
    assert not errors, errors


def test_v03_pretrained_high_overlap_warns():
    from docs.ml_tools.schema import LeakageInfo
    card = _make_card(
        model_class="pretrained_ts",
        leakage=LeakageInfo(
            pretraining_corpus="leaky",
            pretraining_overlap_risk="high",
        ),
    )
    warns = _warns_of(_validate_card(card))
    assert any("overlap_risk='high'" in w for w in warns), warns


def test_v03_cadw_reproduced_wrong_scope_raises():
    from docs.ml_tools.schema import BenchmarkResult
    card = _make_card(evaluation_scope="real_benchmark")
    card.benchmark_results.append(BenchmarkResult(
        benchmark="AMLB", benchmark_url=".pingis/benches/x.json",
        algorithm="cadw.ml.x", algorithm_source_url="cadw/ml/x.py",
        algorithm_source_kind="code", attribution="cadw_reproduced",
        reproduces="x", task_type="test", metric="roc_auc",
        value=0.5, note="test",
    ))
    errors = _errors_of(_validate_card(card))
    # v0.9.51.3: validator message updated to allow both scopes
    assert any("evaluation_scope in ('synthetic', 'both')" in e
               for e in errors), errors


def test_v03_cadw_reproduced_synthetic_scope_passes():
    from docs.ml_tools.schema import BenchmarkResult
    card = _make_card(evaluation_scope="synthetic")
    card.benchmark_results.append(BenchmarkResult(
        benchmark="AMLB", benchmark_url=".pingis/benches/x.json",
        algorithm="cadw.ml.x", algorithm_source_url="cadw/ml/x.py",
        algorithm_source_kind="code", attribution="cadw_reproduced",
        reproduces="x", task_type="test", metric="roc_auc",
        value=0.5, note="test",
    ))
    errors = _errors_of(_validate_card(card))
    assert not any("evaluation_scope" in e for e in errors), errors


def test_v03_supervised_ts_temporal_split_false_raises():
    from docs.ml_tools.schema import LeakageInfo
    card = _make_card(
        task_type="anomaly", learning_mode="supervised",
        leakage=LeakageInfo(temporal_split=False),
    )
    errors = _errors_of(_validate_card(card))
    assert any("temporal_split=True" in e for e in errors), errors


def test_v03_supervised_ts_temporal_split_unset_no_error():
    from docs.ml_tools.schema import LeakageInfo
    card = _make_card(
        task_type="anomaly", learning_mode="supervised",
        leakage=LeakageInfo(temporal_split=None),
    )
    errors = _errors_of(_validate_card(card))
    assert not any("temporal_split" in e for e in errors), errors


def test_v03_audit_leakage_script_present():
    """audit_leakage.py exists, runs clean on current sections."""
    import subprocess
    script = ROOT / "scripts" / "audit_leakage.py"
    assert script.exists(), f"missing {script}"
    yaml_files = [str(p) for p in _section_files()]
    r = subprocess.run(
        [sys.executable, str(script)] + yaml_files,
        capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        f"audit_leakage failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )


def test_v03_audit_leakage_catches_pretrained_without_leakage(tmp_path):
    """audit_leakage.py errors on pretrained_ts with no leakage block."""
    import subprocess
    bad_yaml = tmp_path / "X_bad.yaml"
    bad_yaml.write_text("""section: X
name: X
subtypes:
  - id: X1
    name: t
    description: d
    models:
      - name: P
        tier: sota
        tier_level: tier3_export
        model_class: pretrained_ts
""")
    script = ROOT / "scripts" / "audit_leakage.py"
    r = subprocess.run(
        [sys.executable, str(script), str(bad_yaml)],
        capture_output=True, text=True,
    )
    assert r.returncode == 1, (
        f"expected exit 1, got {r.returncode}\nSTDOUT:\n{r.stdout}"
    )
    assert "PRETRAINED_NO_LEAKAGE" in r.stdout, r.stdout

def test_v03_cadw_reproduced_both_scope_passes():
    """v0.9.51.2: 'both' is valid for cards mixing real + synthetic."""
    from docs.ml_tools.schema import BenchmarkResult
    card = _make_card(evaluation_scope="both")
    card.benchmark_results.append(BenchmarkResult(
        benchmark="cadw_internal", benchmark_url=".pingis/benches/x.json",
        algorithm="cadw.ml.x", algorithm_source_url="cadw/ml/x.py",
        algorithm_source_kind="code", attribution="cadw_reproduced",
        reproduces="x", task_type="test", metric="roc_auc",
        value=0.5, note="test",
    ))
    errors = _errors_of(_validate_card(card))
    assert not any("evaluation_scope" in e for e in errors), errors

# ─── R1-R4: status system enforcement (v0.9.103) ────────────────
# See AI_CONSTITUTION §5 (AP21), STABILITY_PROTOCOL §6.
# R1: status enum valid
# R2: verified requires ram_formula
# R3: verified + # BACKLOG: in module code → fail
# R4: draft requires draft_reason

_ALLOWED_STATUSES = {"draft", "verified", "tier1_core_exception"}


def _iter_verified_in_cadw_cards():
    """Yield (file_path, card, model_label) for each verified card."""
    load_benchmarks, load_section, validate_matrix = _load_all()
    for path in _section_files():
        sec = load_section(path)
        for subtype in sec.subtypes:
            for m in subtype.models:
                if not m.implementation_status.verified_in_cadw:
                    continue
                yield path, m, f"{subtype.id}.{m.name}"


def test_R1_status_enum():
    """R1: status, when present, in {draft, verified, tier1_core_exception}."""
    bad = []
    for path, m, label in _iter_verified_in_cadw_cards():
        s = m.implementation_status.status
        if s is None:
            continue  # pending, not yet assigned
        if s not in _ALLOWED_STATUSES:
            bad.append(f"{label}: status={s!r} (allowed: {sorted(_ALLOWED_STATUSES)})")
    assert not bad, "\n".join(bad)


def test_R2_verified_requires_ram_formula():
    """R2: status=verified implies ram_formula present, parseable,
    with all names resolvable from constants or bounded_state_dims."""
    import ast
    import yaml
    from cadw.ml._memory_estimate import _load_constants
    constants = set(_load_constants().keys())
    bad = []
    for path, m, label in _iter_verified_in_cadw_cards():
        if m.implementation_status.status != "verified":
            continue
        d = yaml.safe_load(path.read_text())
        found = None
        for st in d.get("subtypes", []):
            for c in st.get("models", []):
                if c.get("name") == m.name:
                    found = c
                    break
        formula = (found or {}).get("ram_formula")
        if not formula:
            bad.append(f"{label}: verified but no ram_formula")
            continue
        try:
            tree = ast.parse(formula, mode="eval")
        except SyntaxError as e:
            bad.append(f"{label}: ram_formula syntax error: {e}")
            continue
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        dims = (found or {}).get("bounded_state_dims") or {}
        # v0.9.107: `batch` and `row_size` are auto-injected by
        # `estimate_from_yaml` at runtime (guard.min_batch_rows()
        # and guard.row_size_bytes). Not constants (vary per call),
        # not axes in bounded_state_dims (describe peak, not state).
        # Formula system gained these 2 implicit names in v0.9.105.
        allowed = constants | set(dims.keys()) | {"batch", "row_size"}
        extra = names - allowed
        if extra:
            bad.append(f"{label}: ram_formula names {sorted(extra)} "
                       f"not in constants/dims")
    assert not bad, "\n".join(bad)



def test_R3_verified_no_backlog_in_code():
    """R3: status=verified ⟹ no # BACKLOG: in cadw module source."""
    import re
    bad = []
    root = Path(__file__).parent.parent
    for path, m, label in _iter_verified_in_cadw_cards():
        if m.implementation_status.status != "verified":
            continue
        mod = m.cadw_module or ""
        fn = m.cadw_function or ""
        if not mod.startswith("cadw."):
            continue
        parts = mod.split(".")
        fpath = root / "/".join(parts[:-1]) / f"{parts[-1]}.py"
        if not fpath.exists():
            continue
        src = fpath.read_text()
        # Find function body
        mm = re.search(rf"^def\s+{re.escape(fn)}\b", src, re.M)
        if not mm:
            continue
        tail = src[mm.start():]
        nxt = re.search(r"^(def |class )", tail[1:], re.M)
        body = tail[: nxt.start() + 1] if nxt else tail
        if "# BACKLOG:" in body:
            bad.append(f"{label}: verified but '# BACKLOG:' in {mod}::{fn}")
    assert not bad, "\n".join(bad)


def test_R4_draft_requires_draft_reason():
    """R4: status=draft ⟹ draft_reason present."""
    bad = []
    for path, m, label in _iter_verified_in_cadw_cards():
        if m.implementation_status.status != "draft":
            continue
        if not (m.implementation_status.draft_reason or "").strip():
            bad.append(f"{label}: status=draft but no draft_reason")
    assert not bad, "\n".join(bad)


def test_R5_aliases_consistent_status():
    """R5: aliases of the same cadw_function must agree on status.

    YAML can have multiple cards for one fn (different ML-taxonomy slots:
    classification, imbalanced, cost-sensitive...). If one alias is
    `verified` and another is `pending`, the aggregate claim is
    inconsistent — users reading the other alias may think the fn is
    not verified.

    Rule: for each fn, all its cards must share the same
    implementation_status.status. (`draft` vs `draft` OK, `draft` vs
    `verified` NOT OK — split verification is not allowed.)

    Exception: `described_only: true` cards are reference-only and
    excluded (no `cadw_function`).
    """
    import yaml
    from collections import defaultdict
    from pathlib import Path as _P

    YAML_DIR = _P(__file__).parent.parent / "docs" / "ml_tools"
    SKIP = {"benchmarks.yaml", "memory_constants.yaml"}

    by_fn = defaultdict(list)  # fn -> [(file, name, status)]
    for yp in sorted(YAML_DIR.glob("*.yaml")):
        if yp.name in SKIP:
            continue
        try:
            data = yaml.safe_load(yp.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        for st in data.get("subtypes", []):
            for card in st.get("models", []):
                fn = card.get("cadw_function")
                if not fn:
                    continue
                impl = card.get("implementation_status") or {}
                status = impl.get("status", "(pending)")
                by_fn[fn].append((yp.name, card.get("name") or "?", status))

    bad = []
    for fn, rows in by_fn.items():
        if len(rows) < 2:
            continue
        statuses = {r[2] for r in rows}
        if len(statuses) > 1:
            bad.append(f"{fn}: aliases have inconsistent status {statuses}")
            for r in rows:
                bad.append(f"  {r[0]:<28} {r[1]:<32} {r[2]}")

    assert not bad, "\n".join(bad)
