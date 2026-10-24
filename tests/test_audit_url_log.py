"""Tests for scripts/audit_url_log.py (v0.9.47b)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "audit_url_log", ROOT / "scripts/audit_url_log.py"
)
audit_mod = importlib.util.module_from_spec(_spec)
sys.modules["audit_url_log"] = audit_mod
_spec.loader.exec_module(audit_mod)  # type: ignore


def _write_section(tmp: Path, section: str, cards: list[dict]) -> Path:
    """cards: list of model dicts with 'name' and 'benchmark_results'."""
    data = {
        "section": section,
        "name": f"Section {section}",
        "subtypes": [
            {"id": "X1", "name": "Subtype", "description": "d",
             "real_life": [], "benchmarks": [],
             "models": cards,
             "metrics": [], "ensemble_recommendation": []},
        ],
    }
    p = tmp / f"{section}_test.yaml"
    p.write_text(json.dumps(data))  # JSON valid as YAML
    return p


def _write_log(tmp: Path, entries: list[dict]) -> Path:
    p = tmp / "url_log.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return p


# ============================================================
# 1. OK case: literature with proper url_log
# ============================================================

def test_ok_literature(tmp_path):
    card = {
        "name": "GBDT",
        "benchmark_results": [{
            "benchmark": "AMLB",
            "benchmark_url": "https://amlb.example/sota",
            "algorithm": "LightGBM",
            "algorithm_source_url": "https://papers.example/lightgbm.pdf",
            "algorithm_source_kind": "paper",
            "attribution": "literature",
            "reproduces": None,
            "task_type": "binary classification",
            "n_datasets": 15,
            "metric": "roc_auc",
            "value": 0.842,
            "std": 0.048,
            "note": "fetched 2026-10-24; quote 'LightGBM 0.842'",
        }],
    }
    ypath = _write_section(tmp_path, "A", [card])
    lpath = _write_log(tmp_path, [
        {"url": "https://amlb.example/sota", "status": 200,
         "for_card": "A.X1.GBDT", "metric": "roc_auc", "value": 0.842},
        {"url": "https://papers.example/lightgbm.pdf", "status": 200,
         "for_card": "A.X1.GBDT"},
    ])
    errors, warnings = audit_mod.audit([str(ypath)], str(lpath), root=tmp_path)
    assert errors == [], errors


# ============================================================
# 2. HALLUCINATION: value but no url_log entry
# ============================================================

def test_hallucination_missing_bench_url(tmp_path):
    card = {
        "name": "GBDT",
        "benchmark_results": [{
            "benchmark": "AMLB",
            "benchmark_url": "https://amlb.example/sota",
            "algorithm": "LightGBM",
            "algorithm_source_url": "https://papers.example/lightgbm.pdf",
            "algorithm_source_kind": "paper",
            "attribution": "literature",
            "metric": "roc_auc",
            "value": 0.842,
            "note": "claimed 0.842",
        }],
    }
    ypath = _write_section(tmp_path, "A", [card])
    lpath = _write_log(tmp_path, [])  # empty log
    errors, _ = audit_mod.audit([str(ypath)], str(lpath), root=tmp_path)
    assert len(errors) == 2  # both bench_url and algo_url missing
    assert any("HALLUCINATION" in e and "benchmark_url" in e for e in errors)
    assert any("HALLUCINATION" in e and "algorithm_source_url" in e for e in errors)


def test_hallucination_wrong_metric_in_log(tmp_path):
    card = {
        "name": "GBDT",
        "benchmark_results": [{
            "benchmark": "AMLB",
            "benchmark_url": "https://amlb.example/sota",
            "algorithm": "LightGBM",
            "algorithm_source_url": "https://papers.example/lg.pdf",
            "algorithm_source_kind": "paper",
            "attribution": "literature",
            "metric": "roc_auc",
            "value": 0.842,
            "note": "x",
        }],
    }
    ypath = _write_section(tmp_path, "A", [card])
    lpath = _write_log(tmp_path, [
        # same URL, same for_card, but logged for a DIFFERENT metric
        {"url": "https://amlb.example/sota", "status": 200,
         "for_card": "A.X1.GBDT", "metric": "logloss"},
        {"url": "https://papers.example/lg.pdf", "status": 200,
         "for_card": "A.X1.GBDT"},
    ])
    errors, _ = audit_mod.audit([str(ypath)], str(lpath), root=tmp_path)
    assert any("HALLUCINATION" in e and "benchmark_url" in e for e in errors)


# ============================================================
# 3. NULL_NO_NOTE
# ============================================================

def test_null_value_requires_note(tmp_path):
    card = {
        "name": "TabPFN",
        "benchmark_results": [{
            "benchmark": "AMLB",
            "benchmark_url": "https://amlb.example/sota",
            "algorithm": "TabPFN",
            "algorithm_source_url": "https://papers.example/tabpfn.pdf",
            "algorithm_source_kind": "paper",
            "attribution": "literature",
            "metric": "roc_auc",
            "value": None,
            "note": "",  # empty — should error
        }],
    }
    ypath = _write_section(tmp_path, "A", [card])
    lpath = _write_log(tmp_path, [])
    errors, _ = audit_mod.audit([str(ypath)], str(lpath), root=tmp_path)
    assert any("NULL_NO_NOTE" in e for e in errors)


# ============================================================
# 4. CADW_ARTIFACT_MISSING / CADW_VERDICT
# ============================================================

def test_cadw_artifact_missing(tmp_path):
    card = {
        "name": "GBDT",
        "benchmark_results": [{
            "benchmark": "AMLB",
            "benchmark_url": ".pingis/benches/reproduction/missing.json",
            "algorithm": "cadw.ml.gbdt.fit_gbdt_regressor",
            "algorithm_source_url": "cadw/ml/gbdt.py",
            "algorithm_source_kind": "code",
            "attribution": "cadw_reproduced",
            "reproduces": "LightGBM",
            "metric": "roc_auc",
            "value": 0.838,
            "note": "M2",
        }],
    }
    ypath = _write_section(tmp_path, "A", [card])
    lpath = _write_log(tmp_path, [])
    errors, _ = audit_mod.audit([str(ypath)], str(lpath), root=tmp_path)
    assert any("CADW_ARTIFACT_MISSING" in e and "missing.json" in e for e in errors)


def test_cadw_verdict_divergent(tmp_path):
    # create both artifacts
    bench = tmp_path / ".pingis/benches/reproduction/gbdt.json"
    bench.parent.mkdir(parents=True)
    bench.write_text(json.dumps({"verdict": "divergent", "model": "GBDT"}))
    algo = tmp_path / "cadw/ml/gbdt.py"
    algo.parent.mkdir(parents=True)
    algo.write_text("# stub\n")

    card = {
        "name": "GBDT",
        "benchmark_results": [{
            "benchmark": "AMLB",
            "benchmark_url": ".pingis/benches/reproduction/gbdt.json",
            "algorithm": "cadw.ml.gbdt.fit_gbdt_regressor",
            "algorithm_source_url": "cadw/ml/gbdt.py",
            "algorithm_source_kind": "code",
            "attribution": "cadw_reproduced",
            "reproduces": "LightGBM",
            "metric": "roc_auc",
            "value": 0.838,
            "note": "M2",
        }],
    }
    ypath = _write_section(tmp_path, "A", [card])
    lpath = _write_log(tmp_path, [])
    errors, _ = audit_mod.audit([str(ypath)], str(lpath), root=tmp_path)
    assert any("CADW_VERDICT" in e and "divergent" in e for e in errors)


def test_cadw_reproduced_ok(tmp_path):
    bench = tmp_path / ".pingis/benches/reproduction/gbdt.json"
    bench.parent.mkdir(parents=True)
    bench.write_text(json.dumps({"verdict": "consistent", "model": "GBDT"}))
    algo = tmp_path / "cadw/ml/gbdt.py"
    algo.parent.mkdir(parents=True)
    algo.write_text("# stub\n")

    card = {
        "name": "GBDT",
        "benchmark_results": [{
            "benchmark": "AMLB",
            "benchmark_url": ".pingis/benches/reproduction/gbdt.json",
            "algorithm": "cadw.ml.gbdt.fit_gbdt_regressor",
            "algorithm_source_url": "cadw/ml/gbdt.py",
            "algorithm_source_kind": "code",
            "attribution": "cadw_reproduced",
            "reproduces": "LightGBM",
            "metric": "roc_auc",
            "value": 0.838,
            "note": "M2",
        }],
    }
    ypath = _write_section(tmp_path, "A", [card])
    lpath = _write_log(tmp_path, [])
    errors, _ = audit_mod.audit([str(ypath)], str(lpath), root=tmp_path)
    assert errors == [], errors


# ============================================================
# 5. LEGACY v2 YAML: warning, not error
# ============================================================

def test_legacy_v2_yaml_warns(tmp_path):
    """v2 cards (singular benchmark_result) emit LEGACY warning, no error."""
    card = {
        "name": "OldModel",
        "benchmark_result": {
            "source": "AMLB",
            "url": "https://amlb.example/sota",
            "metric": "roc_auc",
            "value": 0.7,
            "note": "old v2 note",
        },
    }
    ypath = _write_section(tmp_path, "A", [card])
    lpath = _write_log(tmp_path, [])
    errors, warnings = audit_mod.audit([str(ypath)], str(lpath), root=tmp_path)
    assert errors == [], errors
    assert any("[LEGACY]" in w and "OldModel" in w for w in warnings)


# ============================================================
# 6. benchmarks.yaml (no 'section' key) — skipped, not crashed
# ============================================================

def test_registry_yaml_skipped(tmp_path):
    """Non-section YAML (benchmarks registry) must not crash audit."""
    import json as _json
    registry = tmp_path / "benchmarks.yaml"
    registry.write_text(_json.dumps({"benchmarks": {
        "AMLB": {"url": "https://amlb.example/", "modality": "tabular",
                 "metrics": ["roc_auc"]},
    }}))
    lpath = _write_log(tmp_path, [])
    errors, warnings = audit_mod.audit([str(registry)], str(lpath), root=tmp_path)
    assert errors == []
    assert warnings == []
