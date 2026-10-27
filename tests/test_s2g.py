"""Tests for S2G (Series2Graph) — paper-faithful rewrite."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadw.ml.s2g import (S2GResult, _aggregate_to_timestamps,
                        _angles_from_points, _build_graph,
                        _compute_degrees, _embed_shape,
                        _local_convolution, _path_normalcy,
                        _quantize_to_nodes, fit_s2g)

ROOT = Path(__file__).parent.parent


def _sine_segments(n=5000, period=100, n_segments=5, seg_len=30,
                   seed=0, amplitude=5.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.05 * rng.standard_normal(n)
    starts = rng.choice(n - seg_len, size=n_segments, replace=False)
    pos = []
    for s in starts:
        x[s:s + seg_len] = amplitude
        pos.extend(range(s, s + seg_len))
    return x, np.array(sorted(pos))


def _sine_motif_anomalies(n=5000, period=100, n_motifs=5,
                          motif_len=60, seed=0):
    """Sine + embedded motifs (double-freq sine segments).

    S2G's primary use case: shape/motif anomalies in periodic data.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.05 * rng.standard_normal(n)
    starts = rng.choice(n - motif_len - 10, size=n_motifs,
                        replace=False) + 5
    pos = []
    for s in starts:
        lt = np.arange(motif_len)
        x[s:s + motif_len] = np.sin(2 * np.pi * lt / (period // 2)) \
            + 0.05 * rng.standard_normal(motif_len)
        pos.extend(range(s, s + motif_len))
    return x, np.array(sorted(pos))


# ── Unit: local convolution ─────────────────────────────────────

def test_local_convolution_shape():
    rng = np.random.default_rng(0)
    windows = rng.standard_normal((100, 50))
    z = _local_convolution(windows, lam=16)
    assert z.shape == (100, 35)  # 50 - 16 + 1


def test_local_convolution_values():
    """Mean of consecutive lam points, correct boundaries."""
    x = np.arange(10.0)
    windows = x.reshape(1, 10)
    z = _local_convolution(windows, lam=3)
    # z[0] = mean([0,1,2]) = 1.0
    # z[1] = mean([1,2,3]) = 2.0
    # z[7] = mean([7,8,9]) = 8.0
    np.testing.assert_allclose(z[0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


def test_local_convolution_bad_lam():
    windows = np.random.default_rng(0).standard_normal((10, 50))
    with pytest.raises(ValueError, match="convolution_size"):
        _local_convolution(windows, lam=100)


# ── Unit: embedding & angles ────────────────────────────────────

def test_embed_shape_dimensions():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((200, 35))
    points, comps, mean = _embed_shape(Z)
    assert points.shape == (200, 2)
    assert comps.shape == (35, 3)
    assert mean.shape == (35,)


def test_embed_shape_deterministic():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((100, 20))
    p1, c1, m1 = _embed_shape(Z)
    p2, c2, m2 = _embed_shape(Z)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(c1, c2)


def test_angles_range():
    points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    angles = _angles_from_points(points)
    np.testing.assert_allclose(angles, [0.0, np.pi/2, np.pi, -np.pi/2],
                               atol=1e-9)


# ── Unit: nodes & graph ─────────────────────────────────────────

def test_quantize_uniform_bins():
    angles = np.array([-np.pi + 0.01, -np.pi/2, 0.0, np.pi/2, np.pi - 0.01])
    ids = _quantize_to_nodes(angles, n_nodes=4)
    # 4 bins over [−π, π]: edges at -π, -π/2, 0, π/2, π
    assert ids.min() >= 0
    assert ids.max() < 4


def test_build_graph_no_self_loops():
    """W only counts transitions with src ≠ dst."""
    ids = np.array([0, 1, 0, 2, 2, 0])
    W = _build_graph(ids, n_nodes=3)
    # transitions: 0->1, 1->0, 0->2, 2->2 (skip), 2->0
    assert W[0, 1] == 1.0
    assert W[1, 0] == 1.0
    assert W[0, 2] == 1.0
    assert W[2, 2] == 0.0  # self-loop excluded
    assert W[2, 0] == 1.0
    assert W.sum() == 4.0  # 5 transitions, 1 is self


def test_compute_degrees_distinct_edges():
    ids = np.array([0, 1, 0, 2, 2, 0, 1, 0])
    W = _build_graph(ids, n_nodes=3)
    deg = _compute_degrees(W)
    # node 0: out_edges {1, 2}, in_edges {1, 2} -> 4 distinct
    # node 1: out {0}, in {0} -> 2
    # node 2: out {0}, in {0} -> 2
    assert deg[0] == 4.0
    assert deg[1] == 2.0
    assert deg[2] == 2.0


# ── Unit: path normalcy ─────────────────────────────────────────

def test_path_normalcy_simple():
    W = np.zeros((3, 3))
    W[0, 1] = 5.0
    deg = np.array([1.0, 1.0, 0.0])
    # path 0->1: normalcy = 5 * (deg[0] - 1) = 5 * 0 = 0
    # deg=1 means only this edge, so (deg - 1) = 0 -> normalcy 0
    p = np.array([0, 1])
    assert _path_normalcy(p, W, deg) == 0.0


def test_path_normalcy_with_higher_degree():
    W = np.zeros((4, 4))
    W[0, 1] = 5.0
    W[0, 2] = 3.0
    W[2, 1] = 2.0
    W[1, 3] = 1.0
    deg = np.array([2.0, 2.0, 2.0, 1.0])  # each has 2 distinct incident
    # path 0->1->3: normalcy(0->1) = 5 * (2-1) = 5
    #                normalcy(1->3) = 1 * (2-1) = 1
    # mean = 3.0
    p = np.array([0, 1, 3])
    assert abs(_path_normalcy(p, W, deg) - 3.0) < 1e-9


def test_path_normalcy_self_loops_skipped():
    W = np.zeros((3, 3))
    W[0, 1] = 2.0
    deg = np.array([1.0, 1.0, 0.0])
    # path 0->0->1: only edge 0->1 counted (0->0 skipped)
    p = np.array([0, 0, 1])
    # normalcy = 2 * (1 - 1) = 0
    assert _path_normalcy(p, W, deg) == 0.0


# ── Aggregation ─────────────────────────────────────────────────

def test_aggregate_to_timestamps_max():
    scores = np.array([1.0, 2.0, 3.0])
    starts = np.array([0, 2, 4])
    # query window = 3, n = 8
    # q0 covers [0:3], q1 covers [2:5], q2 covers [4:7]
    # coverage: [0]=1, [1]=1, [2]=max(1,2)=2, [3]=2, [4]=max(2,3)=3, [5]=3, [6]=3, [7]=0
    out = _aggregate_to_timestamps(scores, starts, 3, 8)
    np.testing.assert_allclose(out, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 0.0])


# ── fit_s2g contract ────────────────────────────────────────────

def test_fit_s2g_shape_and_dtypes():
    x, _ = _sine_segments(n=2000, n_segments=3, seg_len=30, seed=0)
    r = fit_s2g(x, window_size=50, query_window_size=75)
    assert isinstance(r, S2GResult)
    assert r.decision_scores.shape == (2000,)
    assert r.labels.shape == (2000,)
    assert r.window_size == 50
    assert r.query_window_size == 75
    assert r.convolution_size == 16  # 50 // 3
    assert r.transition_matrix.shape == (r.n_nodes, r.n_nodes)
    assert r.degrees.shape == (r.n_nodes,)


def test_fit_s2g_deterministic():
    x, _ = _sine_segments(n=2000, n_segments=3, seg_len=30, seed=3)
    r1 = fit_s2g(x)
    r2 = fit_s2g(x)
    np.testing.assert_array_equal(r1.decision_scores, r2.decision_scores)
    np.testing.assert_array_equal(r1.node_ids, r2.node_ids)


def test_fit_s2g_short_series_raises():
    with pytest.raises(ValueError, match="< window_size"):
        fit_s2g(np.arange(30.0), window_size=50)


def test_fit_s2g_query_shorter_than_window_raises():
    x = _sine_segments(n=500, n_segments=2, seed=1)[0]
    with pytest.raises(ValueError, match="query_window_size"):
        fit_s2g(x, window_size=50, query_window_size=30)


def test_fit_s2g_bad_n_nodes_raises():
    x = _sine_segments(n=500, n_segments=2, seed=1)[0]
    with pytest.raises(ValueError, match="n_nodes"):
        fit_s2g(x, n_nodes=1)


# ── Quality: motif + segment detection ──────────────────────────

def test_fit_s2g_detects_motifs():
    """Motif anomalies — S2G primary use case. Query path through
    anomalous nodes gets low normalcy, high score.

    Requires sufficient n for graph to have enough transitions:
    at n=5000 motif recall ~0.15 (sparse graph); at n=50000 ~0.34
    (bench level). Test uses n=20000 for balance (recall ~0.25-0.30).
    Paper VUS-PR 0.39 is averaged over 250 UCR datasets — our 0.34
    at n=50000 is in same range.
    """
    x, pos = _sine_motif_anomalies(n=20000, n_motifs=10, motif_len=60,
                                   seed=0)
    r = fit_s2g(x, contamination=0.05)
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.25, f"motif recall={recall:.2f} on {len(pos)} pts"


def test_fit_s2g_detects_segments():
    x, pos = _sine_segments(n=5000, n_segments=5, seg_len=30, seed=0)
    r = fit_s2g(x, contamination=0.05)
    recall = float(np.mean(r.labels[pos] == 1))
    assert recall >= 0.3, f"segment recall={recall:.2f} on {len(pos)} pts"


def test_fit_s2g_constant_series():
    x = np.ones(500) * 3.0
    r = fit_s2g(x)
    # Constant → all windows identical → z identical → angles all same →
    # all node ids same → W has no non-self edges → all path normalcy = 0
    # → all scores = 0 → threshold 0, labels (0 > 0) = 0
    assert r.labels.sum() == 0


def test_fit_s2g_memory_guard_logic():
    from cadw.core.memory_guard import MemoryLimitExceeded

    class _FakeMB:
        max_memory_mb = 0.001

    x = np.random.default_rng(0).standard_normal(5000)
    with pytest.raises(MemoryLimitExceeded):
        fit_s2g(x, window_size=50, guard=_FakeMB())


def test_s2g_no_bench_deps_in_source():
    """AST-based check: no forbidden imports in actual statements."""
    import ast
    src = (ROOT / "cadw/ml/s2g.py").read_text()
    tree = ast.parse(src)
    forbidden = ("sktime", "sklearn", "scipy", "torch", "pyod", "aeon")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden, f"forbidden: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            assert top not in forbidden, f"forbidden: {node.module}"

def test_fit_s2g_guard_covers_score_queries(monkeypatch):
    """v0.9.87-fix1b: est_bytes covers both pattern_windows AND
    _score_queries windows.

    n=10k, w=64 (lam=21): old est ~7 MB (old formula), new est
    ~20 MB. Guard at 10 MB -> new raises BEFORE calling
    _score_queries; old would have allocated first then raised (or
    passed and OOM'd during _score_queries).
    """
    from cadw.ml import s2g as s2g_mod
    from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded

    called = {"n": 0}
    orig = s2g_mod._score_queries

    def spy(*a, **kw):
        called["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(s2g_mod, "_score_queries", spy)

    guard = MemoryGuard(max_memory_mb=10)
    x = np.random.default_rng(0).standard_normal(10_000)
    with pytest.raises(MemoryLimitExceeded):
        s2g_mod.fit_s2g(x, window_size=64, guard=guard)
    assert called["n"] == 0, (
        f"guard should fire BEFORE _score_queries; "
        f"_score_queries was called {called['n']} time(s)")

