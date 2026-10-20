"""K1.4: roc_auc pos_label type mismatch → clear ValueError, not
cryptic ArrowInvalid.

Before: `pa.scalar(pos_label, type=y.type)` inside the pass-1 loop
raised `ArrowInvalid: Could not convert yes with type str: tried to
convert to int64` with no hint that `pos_label` was the problem.
Now: upfront `_pos_label_kind_error` validation raises a clear
ValueError naming both the value and the column type.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from cadw.ops.metrics import _pos_label_kind_error


@pytest.mark.parametrize(
    "y_type, pos_label, should_error",
    [
        (pa.int64(), "yes", True),
        (pa.int64(), 1, False),
        (pa.int64(), 1.0, True),
        (pa.int64(), True, True),
        (pa.int32(), 0, False),
        (pa.float64(), 1, False),
        (pa.float64(), 0.5, False),
        (pa.float64(), "p", True),
        (pa.string(), "yes", False),
        (pa.string(), 1, True),
        (pa.large_string(), "p", False),
        (pa.bool_(), True, False),
        (pa.bool_(), 1, True),
        (pa.binary(), b"p", False),
        (pa.binary(), "p", True),
        (pa.null(), "anything", False),  # unknown → let Arrow decide
    ],
)
def test_pos_label_kind_error_matrix(y_type, pos_label, should_error):
    err = _pos_label_kind_error(y_type, pos_label)
    if should_error:
        assert err is not None, (y_type, pos_label)
        assert "pos_label" in err and "roc_auc" in err
    else:
        assert err is None, (y_type, pos_label, err)


def test_message_names_value_and_type():
    err = _pos_label_kind_error(pa.int64(), "yes")
    assert err is not None
    assert "'yes'" in err and "int64" in err


def _guard():
    from cadw.core.memory_guard import MemoryGuard

    return MemoryGuard(max_memory_bytes=256 * 1024 * 1024)


def _batches(tbl, bs=64):
    for i in range(0, tbl.num_rows, bs):
        yield tbl.slice(i, bs).to_batches()[0]


def test_roc_auc_rejects_incompatible_pos_label(tmp_path):
    from cadw.ops.metrics import _compute_roc_auc_binary

    tbl = pa.table({"y": [0, 1, 0, 1], "p": [0.1, 0.9, 0.2, 0.8]})
    with pytest.raises(ValueError, match="pos_label"):
        _compute_roc_auc_binary(
            lambda: _batches(tbl),
            y_true_col="y",
            y_proba_col="p",
            pos_label="yes",
            spill_dir=tmp_path,
            memory_guard=_guard(),
        )


def test_roc_auc_accepts_compatible_pos_label(tmp_path):
    from cadw.ops.metrics import _compute_roc_auc_binary

    tbl = pa.table({"y": [0, 1, 0, 1], "p": [0.1, 0.9, 0.2, 0.8]})
    auc = _compute_roc_auc_binary(
        lambda: _batches(tbl),
        y_true_col="y",
        y_proba_col="p",
        pos_label=1,
        spill_dir=tmp_path,
        memory_guard=_guard(),
    )
    assert auc == pytest.approx(1.0)
