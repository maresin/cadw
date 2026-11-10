"""v0.9.176: resolve_params skeleton tests (L1 + L5 wired)."""
import pytest

from cadw.core import ApproximationInfo, resolve_params
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded


def test_resolve_params_user_fits():
    mg = MemoryGuard(max_memory_mb=1000)
    params = {"solver": "closed", "d": 100}
    def min_ram(p):
        return 100 * 100 * 8 if p["solver"] == "closed" else 100 * 8
    effective, approx = resolve_params(
        "fit_linear", mg, params, auto_switch=[], min_ram_fn=min_ram
    )
    assert effective == params
    assert approx == []


def test_resolve_params_no_min_ram_fn():
    mg = MemoryGuard(max_memory_mb=1)
    params = {"solver": "closed"}
    effective, approx = resolve_params(
        "fit_linear", mg, params, min_ram_fn=None
    )
    assert effective == params
    assert approx == []


def test_resolve_params_no_alternatives_raises():
    mg = MemoryGuard(max_memory_mb=0.1)
    params = {"d": 10000}
    def min_ram(p):
        return 10000 * 10000 * 8
    with pytest.raises(MemoryLimitExceeded) as exc_info:
        resolve_params("fit_linear", mg, params, auto_switch=[], min_ram_fn=min_ram)
    e = exc_info.value
    assert e.model == "fit_linear"
    assert e.min_required_bytes == 800_000_000
    assert e.available_bytes == pytest.approx(100_000, rel=0.1)
    assert len(e.options) >= 2


def test_resolve_params_L1_solver_switch():
    mg = MemoryGuard(max_memory_mb=50)
    params = {"solver": "closed", "d": 1000}
    def min_ram(p):
        return 1000 * 1000 * 8 if p["solver"] == "closed" else 1000 * 8
    auto_switch = [{
        "from": {"solver": "closed"},
        "to": {"solver": "sgd"},
        "level": "L1",
        "approximation_type": "solver_switch",
        "axis": "d",
        "accuracy_impact": "convergence DEC-033",
    }]
    effective, approx = resolve_params(
        "fit_linear", mg, params, auto_switch=auto_switch, min_ram_fn=min_ram
    )
    # 8 MB fits in 50 MB budget -> no switch
    assert effective == params
    assert approx == []


def test_resolve_params_L1_forced_switch():
    mg = MemoryGuard(max_memory_mb=5)
    params = {"solver": "closed", "d": 2000}
    def min_ram(p):
        return 2000 * 2000 * 8 if p["solver"] == "closed" else 2000 * 8
    auto_switch = [{
        "from": {"solver": "closed"},
        "to": {"solver": "sgd"},
        "level": "L1",
        "approximation_type": "solver_switch",
        "axis": "d",
        "accuracy_impact": "convergence",
    }]
    effective, approx = resolve_params(
        "fit_linear", mg, params, auto_switch=auto_switch, min_ram_fn=min_ram
    )
    # 32 MB closed > 5 MB; sgd = 16 KB fits -> switch
    assert effective["solver"] == "sgd"
    assert len(approx) == 1
    assert approx[0].type == "solver_switch"


def test_resolve_params_L2_not_implemented():
    mg = MemoryGuard(max_memory_mb=1)
    params = {"n_estimators": 1000}
    def min_ram(p):
        return 100_000_000_000
    auto_switch = [{
        "from": {"n_estimators": 1000},
        "to": {"mode": "round_by_round"},
        "level": "L2",
        "approximation_type": "round_by_round",
        "axis": "n",
    }]
    with pytest.raises(NotImplementedError, match="L2"):
        resolve_params("fit_gbdt", mg, params,
                       auto_switch=auto_switch, min_ram_fn=min_ram)


def test_resolve_params_all_candidates_too_big():
    mg = MemoryGuard(max_memory_mb=0.001)
    params = {"solver": "closed", "d": 1000}
    def min_ram(p):
        return 1000 * 1000 * 8 if p["solver"] == "closed" else 1000 * 8
    auto_switch = [{
        "from": {"solver": "closed"},
        "to": {"solver": "sgd"},
        "level": "L1",
    }]
    with pytest.raises(MemoryLimitExceeded) as exc_info:
        resolve_params("fit_linear", mg, params,
                       auto_switch=auto_switch, min_ram_fn=min_ram)
    assert "all auto-switch candidates exceed budget" in str(exc_info.value)
