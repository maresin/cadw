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


# v0.9.17 (§13.4/225 + third-party tracking): filter specific
# MatplotlibDeprecationWarning from seaborn/categorical.py:700
# (seaborn < 0.14 passes vert=True to ax.bxp; matplotlib 3.11
# deprecates vert in favor of orientation). Third-party
# transition — CADW code correct, upstream fix pending.
# Tracking: §11.6 upgrade to seaborn>=0.14 backlog.
import warnings as _warnings
_warnings.filterwarnings(
    'ignore',
    message='vert: bool was deprecated in Matplotlib 3.11',
    category=DeprecationWarning,
)


# v0.9.29-fix1 B.3: auto-marking by nodeid (§13.4/354/356).
# item.nodeid = "tests/test_ts_features.py::test_lag_basic"
# — stable across pytest 7/8/9 (fspath removed in pytest 9,
#  path added in 7 but nodeid never changes).
# Additive: pytest dedupes identical markers.
def pytest_collection_modifyitems(config, items):
    import os as _os
    import pytest as _pt
    for item in items:
        nodeid = item.nodeid
        fname = nodeid.split("::")[0]
        base = _os.path.basename(fname)

        if base.startswith("test_ts_"):
            item.add_marker(_pt.mark.ts)
        if base.startswith("test_ml_") or any(
            base.startswith(p) for p in (
                "test_gbdt", "test_kmeans", "test_linear",
                "test_logistic", "test_naive_bayes", "test_pca",
                "test_scalers", "test_metrics", "test_cv",
                "test_facade_transformers",
                "test_column_transformer",
            )
        ):
            item.add_marker(_pt.mark.ml)
        if base.startswith("test_viz_") or base == "test_plot_accessor.py":
            item.add_marker(_pt.mark.viz)
        if (base.startswith("test_pipeline")
                or base.startswith("test_cv")):
            item.add_marker(_pt.mark.pipeline)
        if base.startswith("test_e2e") or base == "test_oom_resistance.py":
            item.add_marker(_pt.mark.e2e)
        if any(base.startswith(p) for p in (
            "test_groupby", "test_join", "test_sort",
            "test_dedup", "test_rank", "test_quantile",
            "test_resample", "test_pivot", "test_reduce",
            "test_shuffle", "test_topk", "test_histogram",
            "test_merge", "test_anti_join", "test_semi_join",
        )):
            item.add_marker(_pt.mark.ops)
