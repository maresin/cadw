"""v0.9.120: calibrate_throughput.py — dry-run + write behaviour."""
from __future__ import annotations

import pytest
import yaml

from scripts.calibrate_throughput import _write_rates


SAMPLE_YAML = """\
# Header comment to preserve
# Second line

profiles:
  default:
    calibrated: "2026-01-01"
    env: {python: "3.12", numpy: "2.4", pyarrow: "26"}
    rates:
      reduce:        {ops_unit: elements, rate_per_sec: 1.0e7}
      sort:          {ops_unit: elements, rate_per_sec: 2.0e6}
      hash_shuffle:  {ops_unit: elements, rate_per_sec: 5.0e5}
      filter:        {ops_unit: elements, rate_per_sec: 5.0e7}
"""


def test_write_rates_updates_reduce(tmp_path):
    yp = tmp_path / "throughput.yaml"
    yp.write_text(SAMPLE_YAML, encoding="utf-8")
    _write_rates({"reduce": 9.99e7}, yp)
    data = yaml.safe_load(yp.read_text(encoding="utf-8"))
    assert data["profiles"]["default"]["rates"]["reduce"]["rate_per_sec"] == 9.99e7
    assert data["profiles"]["default"]["rates"]["sort"]["rate_per_sec"] == 2.0e6
    assert data["profiles"]["default"]["env"]["python"] == "3.12"


def test_write_rates_preserves_header_comment(tmp_path):
    yp = tmp_path / "throughput.yaml"
    yp.write_text(SAMPLE_YAML, encoding="utf-8")
    _write_rates({"reduce": 1.0}, yp)
    text = yp.read_text(encoding="utf-8")
    assert text.startswith("# Header comment to preserve")
    assert "# Second line" in text


def test_write_rates_updates_calibrated(tmp_path):
    yp = tmp_path / "throughput.yaml"
    yp.write_text(SAMPLE_YAML, encoding="utf-8")
    _write_rates({"reduce": 1.0}, yp, calibrated="2026-11-05")
    data = yaml.safe_load(yp.read_text(encoding="utf-8"))
    assert data["profiles"]["default"]["calibrated"] == "2026-11-05"


def test_write_rates_missing_file_raises(tmp_path):
    yp = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        _write_rates({"reduce": 1.0}, yp)


def test_write_rates_preserves_static_defaults(tmp_path):
    yp = tmp_path / "throughput.yaml"
    yp.write_text(SAMPLE_YAML, encoding="utf-8")
    _write_rates({"reduce": 1.0, "sort": 2.0, "hash_shuffle": 3.0}, yp)
    data = yaml.safe_load(yp.read_text(encoding="utf-8"))
    assert data["profiles"]["default"]["rates"]["filter"]["rate_per_sec"] == 5.0e7
