"""Test measurement reporting + pipeline trace (visible with log_cli=true).

Usage:
    with pipeline_trace() as t:
        pipe.fit_transform(df)
    report("test", writes=t.writes, scans=t.scans, **)
"""
import logging
from contextlib import contextmanager

_logger = logging.getLogger("cadw.test.measurement")


def report(test_name: str, **metrics) -> None:
    if not metrics:
        _logger.info("[%s]", test_name)
        return
    parts = []
    for k, v in metrics.items():
        parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
    _logger.info("[%s] %s", test_name, " ".join(parts))


class PipelineTrace:
    """Instrument Dataset.from_batches (writes) + _iter_filtered_batches (scans)."""

    def __init__(self):
        self.writes = 0
        self.scans = 0
        self.write_log: list[dict] = []

    def __repr__(self):
        return (f"PipelineTrace(writes={self.writes}, scans={self.scans})")


@contextmanager
def pipeline_trace(label: str = ""):
    """Monkey-patch Dataset.from_batches (count writes) and
    CadwDataFrame._iter_filtered_batches (count scans) for duration.
    Restores on exit. Raises if patches don't apply.
    """
    from cadw.dataset import Dataset
    from cadw.dataframe import CadwDataFrame

    trace = PipelineTrace()

    orig_from_batches = Dataset.from_batches.__func__
    orig_iter_batches = CadwDataFrame._iter_filtered_batches

    @classmethod
    def _patched_from_batches(cls, schema, batches_iter, base_dir):
        ds = orig_from_batches(cls, schema, batches_iter, base_dir)
        trace.writes += 1
        trace.write_log.append({
            "write_n": trace.writes,
            "rows": ds._total_rows,
            "cols": list(ds.schema.names),
        })
        _logger.info(
            "trace: write #%d rows=%d cols=%s",
            trace.writes, ds._total_rows, list(ds.schema.names))
        return ds

    def _patched_iter_batches(self, *a, **kw):
        trace.scans += 1
        return orig_iter_batches(self, *a, **kw)

    Dataset.from_batches = _patched_from_batches
    CadwDataFrame._iter_filtered_batches = _patched_iter_batches
    try:
        yield trace
    finally:
        Dataset.from_batches = classmethod(orig_from_batches)  # type: ignore
        CadwDataFrame._iter_filtered_batches = orig_iter_batches
        _logger.info(
            "trace[%s]: final writes=%d scans=%d",
            label or "?", trace.writes, trace.scans)
