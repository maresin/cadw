"""v0.9.143: verification tests for cadw.ml.bandits (DEC-033, DEC-058).

Component-wise verification of EpsilonGreedy / UCB1 / ThompsonSampling.
State = O(n_arms), bounded. Row width is 16 B (arm i64 + reward f64) —
batch signal is small vs noise; primary evidence = state_nbytes_exact
+ OOC (peak ratio at fixed batch). CW-3 coef extraction skipped,
documented in YAML measurement.note.
"""
import ctypes
import gc
import statistics

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import (
    MemoryGuard, MemoryLimitExceeded, measure_peak_traced,
)
from cadw.ml.bandits import (
    fit_epsilon_greedy, fit_ucb1, fit_thompson_sampling,
)


FITTERS = {
    "eg": fit_epsilon_greedy,
    "ucb": fit_ucb1,
    "ts": fit_thompson_sampling,
}

EXPECTED_STATE_PER_ARM = {"eg": 2, "ucb": 2, "ts": 4}  # f64 per arm


def _make_df(n, n_arms=5, seed=0):
    rng = np.random.RandomState(seed)
    true_p = np.linspace(0.2, 0.8, n_arms)
    arms = rng.randint(0, n_arms, size=n).astype(np.int64)
    rewards = (rng.random(n) < true_p[arms]).astype(np.float64)
    return CadwDataFrame.from_table(pa.table({
        "arm": pa.array(arms, type=pa.int64()),
        "reward": pa.array(rewards, type=pa.float64()),
    }))


def _iter_col_concat(df, col):
    chunks = [b.column(col).to_numpy(zero_copy_only=False)
              for b in df._iter_filtered_batches()]
    return np.concatenate(chunks) if chunks else np.empty(0)


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    df = _make_df(200, n_arms=3, seed=999)
    for fn in FITTERS.values():
        fn(df, arm_col="arm", reward_col="reward", n_arms=3,
           memory_guard=MemoryGuard(max_memory_mb=500), batch_size=50)


_warmup()


# ---- CW-1: state_nbytes exact ----

@pytest.mark.parametrize("policy", ["eg", "ucb", "ts"])
@pytest.mark.parametrize("n_arms", [1, 5, 50, 200])
def test_state_nbytes_exact(policy, n_arms):
    df = _make_df(1000, n_arms=n_arms)
    model = FITTERS[policy](
        df, arm_col="arm", reward_col="reward", n_arms=n_arms,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=200,
    )
    expected = EXPECTED_STATE_PER_ARM[policy] * n_arms * 8 + 128
    assert model.state_nbytes() == expected, (
        f"{policy} n_arms={n_arms}: got {model.state_nbytes()}, "
        f"expected {expected}")


# ---- CW-2: state independent of n ----

@pytest.mark.parametrize("policy", ["eg", "ucb", "ts"])
def test_state_independent_of_n(policy):
    n_arms = 20
    m1 = FITTERS[policy](
        _make_df(1000, n_arms=n_arms, seed=0),
        arm_col="arm", reward_col="reward", n_arms=n_arms,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=200)
    m2 = FITTERS[policy](
        _make_df(10_000, n_arms=n_arms, seed=0),
        arm_col="arm", reward_col="reward", n_arms=n_arms,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=200)
    assert m1.state_nbytes() == m2.state_nbytes()


# ---- CW-5: OOC at fixed batch ----

@pytest.mark.parametrize("policy", ["eg", "ucb", "ts"])
def test_ooc_batch_fixed(policy):
    n_arms, batch = 20, 200
    peaks = []
    for n in [1000, 5000]:
        df = _make_df(n, n_arms=n_arms, seed=0)
        cell = []
        for _ in range(3):
            _malloc_trim(); gc.collect()
            mg = MemoryGuard(max_memory_mb=500)
            r = measure_peak_traced(
                FITTERS[policy], df,
                arm_col="arm", reward_col="reward", n_arms=n_arms,
                memory_guard=mg, batch_size=batch, mg=mg)
            cell.append(int(r["peak_bytes"]))
            del r; gc.collect()
        peaks.append(int(statistics.median(cell)))
        del df; gc.collect()
    ratio = peaks[-1] / max(peaks[0], 1)
    assert ratio < 2.0, f"{policy}: ratio {ratio:.2f}, peaks={peaks}"


# ---- guard refuses at tight budget ----

@pytest.mark.parametrize("policy", ["eg", "ucb", "ts"])
def test_guard_refuses_at_tight(policy):
    df = _make_df(200, n_arms=10, seed=0)
    with pytest.raises(MemoryLimitExceeded):
        FITTERS[policy](
            df, arm_col="arm", reward_col="reward", n_arms=10_000,
            memory_guard=MemoryGuard(max_memory_mb=0.01), batch_size=100)


# ---- correctness: q_, n_ match manual ----

@pytest.mark.parametrize("policy", ["eg", "ucb", "ts"])
def test_q_matches_manual(policy):
    n_arms = 7
    df = _make_df(5000, n_arms=n_arms, seed=1)
    model = FITTERS[policy](
        df, arm_col="arm", reward_col="reward", n_arms=n_arms,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=500)
    arms = _iter_col_concat(df, "arm")
    rews = _iter_col_concat(df, "reward")
    sums = np.zeros(n_arms, dtype=np.float64)
    counts = np.zeros(n_arms, dtype=np.int64)
    np.add.at(sums, arms, rews)
    np.add.at(counts, arms, 1)
    q_expected = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    np.testing.assert_allclose(model.q_, q_expected, rtol=1e-10)
    np.testing.assert_array_equal(model.n_, counts)


def test_ts_alpha_beta_from_successes():
    n_arms = 5
    df = _make_df(2000, n_arms=n_arms, seed=2)
    model = fit_thompson_sampling(
        df, arm_col="arm", reward_col="reward", n_arms=n_arms,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=200)
    arms = _iter_col_concat(df, "arm")
    rews = _iter_col_concat(df, "reward")
    succ = np.zeros(n_arms, dtype=np.int64)
    counts = np.zeros(n_arms, dtype=np.int64)
    np.add.at(succ, arms, (rews > 0.5).astype(np.int64))
    np.add.at(counts, arms, 1)
    np.testing.assert_array_equal(
        model.alpha_, 1.0 + succ.astype(np.float64))
    np.testing.assert_array_equal(
        model.beta_, 1.0 + (counts - succ).astype(np.float64))


@pytest.mark.parametrize("policy", ["eg", "ucb", "ts"])
def test_online_select_update(policy):
    n_arms = 5
    df = _make_df(1000, n_arms=n_arms, seed=3)
    model = FITTERS[policy](
        df, arm_col="arm", reward_col="reward", n_arms=n_arms,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=200)
    a = model.select()
    assert 0 <= a < n_arms
    n_before = int(model.n_[a]); q_before = float(model.q_[a])
    model.update(a, 1.0)
    assert int(model.n_[a]) == n_before + 1
    expected = q_before + (1.0 - q_before) / int(model.n_[a])
    assert abs(float(model.q_[a]) - expected) < 1e-12


def test_n_arms_sample_infer():
    df = _make_df(2000, n_arms=8, seed=4)
    model = fit_ucb1(
        df, arm_col="arm", reward_col="reward", n_arms=None,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=1000)
    assert model.n_arms >= 1
    assert model.n_arms == len(model.q_)
