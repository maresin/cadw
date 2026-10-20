"""Root test configuration (v0.9.1a).

Autouse fixture: close matplotlib figures after each test.

Reason: v0.9.0c added 8 plot-accessor tests, tipping the ~20-figure
threshold. Pre-existing seaborn/viz tests did not close figures either.
Fix is global — every test gets figure cleanup, no more RuntimeWarning.
"""
import pytest


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each test.

    matplotlib default retains figures in `plt._pylab_helpers.Gcf` until
    explicit close. Across a 1300-test suite that accumulates memory and
    emits "More than 20 figures have been opened" RuntimeWarning.
    """
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass
