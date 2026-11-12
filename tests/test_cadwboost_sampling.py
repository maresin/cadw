"""v0.9.224-a: CadwBoost sampling (none/random/GOSS) tests."""
import numpy as np
from cadw.ml.cadwboost.sampling import (
    sample_none, sample_random, sample_goss, SampleMask)


def test_sample_none_full():
    sm = sample_none(1000)
    assert sm.mask.shape == (1000,)
    assert sm.mask.all(), "sample_none should keep all rows"
    assert sm.weights.shape == (1000,)
    np.testing.assert_allclose(sm.weights, 1.0)


def test_sample_random_rate():
    rng = np.random.RandomState(0)
    sm = sample_random(10000, 0.3, rng)
    assert sm.mask.shape == (10000,)
    kept = int(sm.mask.sum())
    # 0.3 * 10000 = 3000, allow 10% deviation
    assert 2700 <= kept <= 3300, f"kept={kept} (expected ~3000)"


def test_sample_goss_covers_top_gradient():
    """GOSS: top by |grad| always kept; weights applied."""
    n = 1000
    rng = np.random.RandomState(0)
    grad = rng.normal(size=n)
    sm = sample_goss(n, grad, top_rate=0.2, other_rate=0.1, rng=rng)
    # top 20% by |grad| = 200 rows, always in mask
    top_idx = np.argsort(-np.abs(grad))[:200]
    assert sm.mask[top_idx].all(), "top 200 by |grad| must be kept"
    # weights: top rows weight=1.0; others scaled by (1-top)/other
    assert sm.weights[top_idx].max() <= 1.0 + 1e-6
    # mask size roughly 200 + 0.1*800 = 280
    kept = int(sm.mask.sum())
    assert 200 <= kept <= 400, f"kept={kept} (expected ~280)"
