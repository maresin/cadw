"""Convenience loader for tests (avoids circular import paths)."""
from pathlib import Path

from .schema import Benchmarks, BenchmarkSpec, Section, load_benchmarks as _lb, load_section as _ls


def load_benchmarks(path: Path) -> Benchmarks:
    return _lb(Path(path))


def load_section(path: Path) -> Section:
    return _ls(Path(path))
