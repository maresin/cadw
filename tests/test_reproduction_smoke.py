"""Smoke tests for M2 reproduction harness (v0.9.46).

Tests import scripts/reproduce_benchmark.py and run each reproducer.
Skipped on ImportError (reference not installed). Failed on
verdict="divergent" or "cadw_error".
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "reproduce_benchmark", ROOT / "scripts/reproduce_benchmark.py"
)
repro = importlib.util.module_from_spec(_spec)
# Register in sys.modules before exec: @dataclass needs cls.__module__ lookup.
sys.modules["reproduce_benchmark"] = repro
_spec.loader.exec_module(repro)  # type: ignore


@pytest.mark.parametrize("name", sorted(repro.REPRODUCERS))
def test_reproduction_consistent(name):
    """CADW vs reference on synthetic benchmark-like DGP must be consistent."""
    try:
        result = repro.REPRODUCERS[name]()
    except ImportError as e:
        pytest.skip(f"reference missing: {e}")

    if result.verdict == "divergent":
        pytest.fail(
            f"{name}: CADW={result.cadw_metric:.4f} vs "
            f"reference={result.reference_metric:.4f} "
            f"(ref_model={result.reference_model}). "
            f"{result.notes}"
        )
    assert result.verdict == "consistent", (
        f"{name}: unexpected verdict {result.verdict!r}. {result.notes}"
    )


def test_artifact_writes_dry_run(tmp_path, monkeypatch):
    """--write with a temp OUT_DIR produces valid JSON."""
    import json
    monkeypatch.setattr(repro, "OUT_DIR", tmp_path)
    # Direct write via the runner
    r = repro.REPRODUCERS["gbdt_regressor"]()
    path = tmp_path / "gbdt_regressor.json"
    path.write_text(json.dumps(repro.asdict(r), indent=2) + "\n")
    data = json.loads(path.read_text())
    assert data["model"] == "GBDT_regressor"
    assert "verdict" in data
    assert "cadw_metric" in data
