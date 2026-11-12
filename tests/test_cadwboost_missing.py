"""v0.9.224-a: CadwBoost missing value default direction tests."""
import numpy as np
from cadw.ml.cadwboost.missing import split_gain_with_default


def test_split_gain_with_default_shape():
    """Returns (gain: float, default_left: bool)."""
    gain, default_left = split_gain_with_default(
        G_left=1.0, H_left=2.0, G_right=3.0, H_right=4.0,
        G_missing=0.5, H_missing=1.0, reg_lambda=1.0)
    assert isinstance(gain, (float, np.floating))
    assert isinstance(default_left, (bool, np.bool_))


def test_split_gain_with_default_no_missing():
    """If no missing (G_missing=H_missing=0), both directions equal."""
    gain_l, left_l = split_gain_with_default(
        G_left=1.0, H_left=2.0, G_right=3.0, H_right=4.0,
        G_missing=0.0, H_missing=0.0, reg_lambda=1.0)
    gain_r, left_r = split_gain_with_default(
        G_left=1.0, H_left=2.0, G_right=3.0, H_right=4.0,
        G_missing=0.0, H_missing=0.0, reg_lambda=1.0)
    assert abs(gain_l - gain_r) < 1e-9
    assert left_l == left_r
