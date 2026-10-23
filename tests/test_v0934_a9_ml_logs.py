"""v0.9.34 (A9): uniform log_fit wiring for linear/pca/naive_bayes.

Contract (BUG_BACKLOG A9): every cadw.ml model uses `cadw.ml._logging.log_fit`.
Format: `[fit] <model> | iter=N/M | loss=X | state_bytes=B | rss=Y MiB`.
"""
import inspect
from pathlib import Path

import pytest


MODEL_FILES = [
    "cadw/ml/kmeans.py",
    "cadw/ml/logistic.py",
    "cadw/ml/gbdt.py",
    "cadw/ml/linear.py",
    "cadw/ml/pca.py",
    "cadw/ml/naive_bayes.py",
]


def test_all_ml_files_import_log_fit():
    for f in MODEL_FILES:
        src = Path(f).read_text(encoding="utf-8")
        assert "log_fit" in src, f"{f} does not reference log_fit"


def test_log_fit_format(capsys, monkeypatch):
    monkeypatch.setenv("CADW_LOG", "1")
    from cadw.ml._logging import log_fit
    log_fit("unit", iter_i=1, iter_n=3, loss=0.5, state_bytes=1024)
    out = capsys.readouterr().out
    assert out.startswith("[fit] unit |")
    assert "iter=1/3" in out
    assert "loss=0.500000" in out
    assert "state_bytes=1024" in out
    assert "rss=" in out and "MiB" in out
