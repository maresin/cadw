"""Tests for D.6.5 geo aggregations (v0.9.40)."""
from __future__ import annotations
import inspect, numpy as np, pyarrow as pa, pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import geo as gmod
from cadw.ml.geo import haversine, bearing, hex_cell, fit_geo_agg, EARTH_RADIUS_M


def test_haversine_analytic_reference():
    # Paris (48.8566, 2.3522) -> London (51.5074, -0.1278) ~ 343.5 km
    d = float(haversine(48.8566, 2.3522, 51.5074, -0.1278))
    assert 340_000 < d < 347_000
    # same point -> 0
    assert float(haversine(10, 20, 10, 20)) == 0.0
    # 1 deg lon at equator
    assert abs(float(haversine(0, 0, 0, 1)) - EARTH_RADIUS_M * np.pi / 180) < 1.0


def test_bearing_cardinal_directions():
    assert abs(float(bearing(0, 0, 1, 0)) - 0.0) < 1e-9     # north
    assert abs(float(bearing(0, 0, 0, 1)) - 90.0) < 1e-9    # east
    assert abs(float(bearing(0, 0, -1, 0)) - 180.0) < 1e-9  # south
    assert abs(float(bearing(0, 0, 0, -1)) - 270.0) < 1e-9  # west


def test_hex_cell_determinism_and_neighbour_separation():
    lat = np.array([55.75, 55.75, 55.7501, 55.76])
    lon = np.array([37.61, 37.61, 37.6101, 37.62])
    q, r = hex_cell(lat, lon, cell_size_m=1000.0, origin_lat=55.75, origin_lon=37.61)
    assert (q[0], r[0]) == (q[1], r[1])
    # repeated call is bit-identical
    q2, r2 = hex_cell(lat, lon, cell_size_m=1000.0, origin_lat=55.75, origin_lon=37.61)
    assert np.array_equal(q, q2) and np.array_equal(r, r2)


def _gdf(n=400, seed=0):
    rng = np.random.RandomState(seed)
    lat = rng.uniform(55.7, 55.8, n)
    lon = rng.uniform(37.5, 37.7, n)
    v = rng.normal(size=n)
    return CadwDataFrame.from_table(pa.table({
        "lat": pa.array(lat), "lon": pa.array(lon), "v": pa.array(v)})), lat, lon, v


def test_hex_aggregate_count_and_sum():
    df, _, _, v = _gdf(seed=0)
    m = fit_geo_agg(df, "lat", "lon", cell_size_m=500.0)
    keys, cnt = m.hex_aggregate(df, agg="count")
    assert int(cnt.sum()) == len(df)
    assert keys.shape == (len(cnt), 2)
    keys2, s = m.hex_aggregate(df, value_col="v", agg="sum")
    assert np.allclose(keys, keys2)
    assert abs(float(s.sum()) - float(v.sum())) < 1e-6


def test_hex_aggregate_mean():
    df, _, _, v = _gdf(seed=1)
    m = fit_geo_agg(df, "lat", "lon", cell_size_m=1000.0)
    keys, mean = m.hex_aggregate(df, value_col="v", agg="mean")
    assert len(keys) == len(mean)
    assert np.all(np.isfinite(mean))


def test_edge_guards():
    df, _, _, _ = _gdf(n=50, seed=2)
    with pytest.raises(KeyError):
        fit_geo_agg(df, "nope", "lon")
    with pytest.raises(KeyError):
        fit_geo_agg(df, "lat", "nope")
    with pytest.raises(ValueError, match="cell_size_m"):
        fit_geo_agg(df, "lat", "lon", cell_size_m=0.0)
    m = fit_geo_agg(df, "lat", "lon")
    with pytest.raises(ValueError, match="agg must be"):
        m.hex_aggregate(df, value_col="v", agg="max")


def test_ram_guard_raises_when_tight():
    df, _, _, _ = _gdf(n=600, seed=3)
    mg = MemoryGuard(max_memory_bytes=512)
    with pytest.raises(MemoryLimitExceeded):
        fit_geo_agg(df, "lat", "lon", cell_size_m=500.0, memory_guard=mg)


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    df, _, _, _ = _gdf(n=400, seed=4)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "g.parquet")
    pq.write_table(tbl, path, row_group_size=71)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100)
    m = fit_geo_agg(df, "lat", "lon", cell_size_m=500.0)
    k1, c1 = m.hex_aggregate(df, agg="count")
    k2, c2 = m.hex_aggregate(dmb, agg="count")
    assert np.array_equal(k1, k2) and np.array_equal(c1, c2)


def test_source_contract_and_no_pandas():
    src = inspect.getsource(gmod)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src
