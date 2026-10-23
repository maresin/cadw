"""Tests for D.1.4 fairness metrics (v0.9.35)."""
from __future__ import annotations
import ast, inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.fairness import FairnessReport, fairness_metrics

def _df(sens, yt, yp):
    return CadwDataFrame.from_table(pa.table({
        "g": pa.array(sens),
        "yt": pa.array(yt, type=pa.int64()),
        "yp": pa.array(yp, type=pa.int64())}))

def _manual(sens, yt, yp):
    sens = np.asarray(sens); yt = np.asarray(yt); yp = np.asarray(yp)
    out = {}
    for g in sorted(set(sens.tolist())):
        m = sens == g
        d = dict(n=int(m.sum()),
                 tp=int((m & (yt == 1) & (yp == 1)).sum()),
                 fp=int((m & (yt == 0) & (yp == 1)).sum()),
                 fn=int((m & (yt == 1) & (yp == 0)).sum()),
                 tn=int((m & (yt == 0) & (yp == 0)).sum()))
        d["ppr"] = (d["tp"] + d["fp"]) / d["n"]
        d["tpr"] = d["tp"] / (d["tp"] + d["fn"])
        d["fpr"] = d["fp"] / (d["fp"] + d["tn"])
        out[g] = d
    return out


def _fm(df, *, sensitive_col="g", y_true_col="yt", y_pred_col="yp",
        memory_guard=None):
    return fairness_metrics(df, sensitive_col=sensitive_col,
                            y_true_col=y_true_col, y_pred_col=y_pred_col,
                            memory_guard=memory_guard)

def test_synthetic_known_confusion():
    rep = _fm(_df(["A","A","A","A","B","B","B","B"],
                               [1,1,0,1, 1,1,0,0],
                               [1,1,1,0, 0,0,0,0]))
    a = rep.groups["A"]; b = rep.groups["B"]
    assert (a["tp"], a["fp"], a["fn"], a["tn"]) == (2, 1, 1, 0)
    assert (b["tp"], b["fp"], b["fn"], b["tn"]) == (0, 0, 2, 2)
    assert a["ppr"] == pytest.approx(3/4)
    assert a["tpr"] == pytest.approx(2/3)
    assert b["tpr"] == pytest.approx(0.0)
    assert rep.metrics["equal_opportunity_difference"] == pytest.approx(2/3)
    assert rep.metrics["demographic_parity_difference"] == pytest.approx(3/4)
    assert rep.metrics["tpr_gap"] == rep.metrics["equal_opportunity_difference"]
    assert rep.metrics["disparate_impact"] == pytest.approx(0.0)

def test_reference_vs_numpy_bruteforce():
    rng = np.random.RandomState(0)
    n = 2000
    sens = rng.choice(["g0", "g1", "g2"], size=n).tolist()
    yt = rng.randint(0, 2, size=n); yp = rng.randint(0, 2, size=n)
    rep = _fm(_df(sens, yt.tolist(), yp.tolist()))
    ref = _manual(sens, yt, yp)
    for g, d in ref.items():
        for k in ("n", "tp", "fp", "fn", "tn"):
            assert rep.groups[g][k] == d[k]
        assert rep.groups[g]["ppr"] == pytest.approx(d["ppr"])
        assert rep.groups[g]["tpr"] == pytest.approx(d["tpr"])
        assert rep.groups[g]["fpr"] == pytest.approx(d["fpr"])
    ppr = np.array([d["ppr"] for d in ref.values()])
    tpr = np.array([d["tpr"] for d in ref.values()])
    assert rep.metrics["demographic_parity_difference"] == pytest.approx(ppr.max() - ppr.min())
    assert rep.metrics["equal_opportunity_difference"] == pytest.approx(tpr.max() - tpr.min())

def test_perfectly_fair_predictor_has_zero_gaps():
    # Construct data where both groups have identical PPR/TPR/FPR:
    # each group gets the same (yt, yp) pattern.
    pat_yt = [1, 1, 0, 0, 1, 0, 0, 1]
    pat_yp = [1, 1, 0, 0, 0, 0, 0, 1]
    sens = ["a"] * 8 + ["b"] * 8
    yt = pat_yt * 2
    yp = pat_yp * 2
    rep = _fm(_df(sens, yt, yp))
    assert rep.metrics["demographic_parity_difference"] < 1e-12
    assert rep.metrics["equal_opportunity_difference"] < 1e-12
    assert rep.metrics["fpr_gap"] < 1e-12

def test_mixin_binding():
    df = _df(["A","A","B","B"], [1,0,1,0], [1,0,1,0])
    rep = df.fairness_metrics(sensitive_col="g", y_true_col="yt", y_pred_col="yp")
    assert isinstance(rep, FairnessReport)
    assert rep.n_ == 4

def test_determinism_and_dict_roundtrip():
    sens = ["x"]*10 + ["y"]*10
    yt = [1,0]*10; yp = [1,1,0,0,1,1,0,0,1,1]*2
    a = _fm(_df(sens, yt, yp))
    b = _fm(_df(sens, yt, yp))
    assert a.metrics == b.metrics
    d = a.to_dict()
    assert d["groups"]["x"]["n"] == 10
    assert d["n"] == 20

def test_multi_batch_chunk_invariance():
    import tempfile
    import pyarrow.parquet as pq
    rng = np.random.RandomState(2)
    n = 400
    tbl = pa.table({"g": pa.array(rng.choice(["p","q"], size=n).tolist()),
                    "yt": pa.array(rng.randint(0,2,size=n), type=pa.int64()),
                    "yp": pa.array(rng.randint(0,2,size=n), type=pa.int64())})
    path = tempfile.mktemp(suffix=".parquet")
    pq.write_table(tbl, path, row_group_size=7)
    r1 = _fm(CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1),
                          sensitive_col="g", y_true_col="yt", y_pred_col="yp")
    r2 = _fm(CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1000),
                          sensitive_col="g", y_true_col="yt", y_pred_col="yp")
    assert r1.metrics == r2.metrics
    assert r1.groups == r2.groups

def test_state_independent_of_n_and_ram():
    small = _fm(_df(["a","b"]*5, [1,0]*5, [1,0]*5))
    big = _fm(_df(["a","b"]*5000, [1,0]*5000, [1,0]*5000))
    assert small.state_nbytes() == big.state_nbytes()
    assert big.state_nbytes() < 4096

def test_guard_high_cardinality_groups():
    mg = MemoryGuard(max_memory_mb=4)
    n = 20000
    df = _df([f"g{i}" for i in range(n)], [1]*n, [1]*n)
    with pytest.raises(MemoryLimitExceeded):
        _fm(df, sensitive_col="g", y_true_col="yt",
                         y_pred_col="yp", memory_guard=mg)

def test_errors_non_binary_and_missing():
    df = _df(["a","a"], [2,0], [1,0])
    with pytest.raises(ValueError):
        _fm(df, sensitive_col="g", y_true_col="yt", y_pred_col="yp")
    with pytest.raises(KeyError):
        _fm(df, sensitive_col="zzz", y_true_col="yt", y_pred_col="yp")
    empty = CadwDataFrame.from_table(pa.table({
        "g": pa.array([], type=pa.string()),
        "yt": pa.array([], type=pa.int64()),
        "yp": pa.array([], type=pa.int64())}))
    with pytest.raises(ValueError):
        _fm(empty, sensitive_col="g", y_true_col="yt", y_pred_col="yp")

def test_ast_guard_no_pandas_no_unguarded_to_pylist():
    import cadw.ml.fairness as fm
    src = inspect.getsource(fm)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("pandas")
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("pandas")
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        fsrc = ast.get_source_segment(src, fn) or ""
        if ".to_pylist()" in fsrc:
            assert "check_batch(" in fsrc, f"{fn.name}: unguarded to_pylist"
            for line in fsrc.splitlines():
                if "check_batch(" in line:
                    assert "extra_factor=" in line
                    val = float(line.split("extra_factor=")[1].split(",")[0].rstrip(")"))
                    assert val >= 2.0

def test_source_contract():
    import cadw.ml.fairness as fm
    src = inspect.getsource(fm)
    assert "Hardt" in src
    assert "demographic_parity_difference" in src
    assert "equal_opportunity_difference" in src
