"""v0.9.156: tests for cadw.ops.optim.{grid_search, random_search} (L14)."""
import numpy as np
import pytest

from cadw.ops.optim import grid_search, random_search
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded


# ============================================================================
# grid_search
# ============================================================================

def test_grid_search_basic():
    """Best combo picked from 2x3 grid."""
    grid = {"a": [1, 2], "b": [10, 20, 30]}
    def scorer(p):
        return p["a"] * 100 + p["b"]
    result = grid_search(grid, scorer, maximize=True)
    assert result["best_params"] == {"a": 2, "b": 30}
    assert result["best_score"] == 230
    assert len(result["results"]) == 6


def test_grid_search_minimize():
    grid = {"x": [1, 5, 3]}
    def scorer(p):
        return (p["x"] - 3) ** 2
    result = grid_search(grid, scorer, maximize=False)
    assert result["best_params"] == {"x": 3}
    assert result["best_score"] == 0


def test_grid_search_single_param():
    result = grid_search({"lr": [0.1, 0.01, 0.001]},
                         lambda p: -abs(p["lr"] - 0.01),
                         maximize=True)
    assert result["best_params"] == {"lr": 0.01}


def test_grid_search_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        grid_search({}, lambda p: 0)
    with pytest.raises(ValueError, match="empty"):
        grid_search({"x": []}, lambda p: 0)


def test_grid_search_scorer_called_once_per_combo():
    calls = []
    grid = {"a": [1, 2], "b": [3, 4, 5]}
    def scorer(p):
        calls.append(dict(p))
        return 0.0
    grid_search(grid, scorer)
    assert len(calls) == 6
    # all combinations present
    seen = {(c["a"], c["b"]) for c in calls}
    assert seen == {(1, 3), (1, 4), (1, 5),
                    (2, 3), (2, 4), (2, 5)}


def test_grid_search_guard_refuses():
    grid = {"a": list(range(200)), "b": list(range(200))}
    with pytest.raises(MemoryLimitExceeded):
        grid_search(grid, lambda p: 0.0,
                    memory_guard=MemoryGuard(max_memory_mb=0.1))


# ============================================================================
# random_search
# ============================================================================

def test_random_search_uniform_int():
    """Best params near target for integer uniform."""
    dist = {"x": (0, 10)}
    def scorer(p):
        return -abs(p["x"] - 7)
    result = random_search(dist, scorer, n_iter=200, maximize=True, seed=0)
    assert 5 <= result["best_params"]["x"] <= 9
    assert result["best_score"] >= -2.0


def test_random_search_uniform_float():
    dist = {"lr": (0.001, 1.0)}
    def scorer(p):
        return -abs(p["lr"] - 0.1)
    result = random_search(dist, scorer, n_iter=300, maximize=True, seed=42)
    assert 0.05 <= result["best_params"]["lr"] <= 0.2


def test_random_search_categorical():
    dist = {"kernel": ["linear", "rbf", "poly"]}
    scores = {"linear": 1.0, "rbf": 5.0, "poly": 3.0}
    def scorer(p):
        return scores[p["kernel"]]
    result = random_search(dist, scorer, n_iter=50, maximize=True, seed=1)
    assert result["best_params"]["kernel"] == "rbf"


def test_random_search_reproducible():
    """Same seed -> same results."""
    dist = {"x": (0, 100), "y": ("a", "b", "c")} if False else {"x": (0, 100)}
    def scorer(p):
        return float(p["x"])
    r1 = random_search(dist, scorer, n_iter=20, seed=7)
    r2 = random_search(dist, scorer, n_iter=20, seed=7)
    assert r1["best_params"] == r2["best_params"]
    assert [r["score"] for r in r1["results"]] == [r["score"] for r in r2["results"]]


def test_random_search_different_seeds():
    """Different seeds -> likely different results."""
    dist = {"x": (0, 1000)}
    def scorer(p):
        return float(p["x"])
    r1 = random_search(dist, scorer, n_iter=20, seed=1)
    r2 = random_search(dist, scorer, n_iter=20, seed=2)
    # Very unlikely to pick exact same 20 draws
    assert r1["results"] != r2["results"]


def test_random_search_callable_spec():
    """Callable spec receives rng."""
    def custom_dist(rng):
        return int(rng.integers(0, 5))
    result = random_search({"x": custom_dist}, lambda p: p["x"],
                            n_iter=20, seed=3)
    assert 0 <= result["best_params"]["x"] <= 4


def test_random_search_minimize():
    dist = {"x": (0, 100)}
    result = random_search(dist, lambda p: (p["x"] - 50) ** 2,
                            n_iter=200, maximize=False, seed=4)
    assert abs(result["best_params"]["x"] - 50) < 20


def test_random_search_invalid_args():
    with pytest.raises(ValueError, match="non-empty"):
        random_search({}, lambda p: 0, n_iter=5)
    with pytest.raises(ValueError, match="n_iter"):
        random_search({"x": (0, 1)}, lambda p: 0, n_iter=0)
    with pytest.raises(ValueError, match="spec must be"):
        random_search({"x": "invalid"}, lambda p: 0, n_iter=5)


def test_random_search_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        random_search({"x": []}, lambda p: 0, n_iter=5)


def test_random_search_guard_refuses():
    dist = {"x": (0, 100)}
    with pytest.raises(MemoryLimitExceeded):
        random_search(dist, lambda p: 0.0, n_iter=10_000,
                      memory_guard=MemoryGuard(max_memory_mb=0.1))
