"""Geospatial aggregations and distance (D.6.5, v0.9.40).

Reference: Sinnott, *Virtues of the Haversine*, Sky and Telescope 68(2),
1984 (haversine distance); Uber Engineering, *H3: A Hierarchical Hexagonal
Geospatial Index*, 2018 (hexagonal binning as used by h3); scipy.spatial
cKDTree as reference for nearest-neighbour lookups.
See .pingis/algorithm_notes/d6_5_geo.md.

Deliberately no h3 dependency: the hex aggregation is the h3-equivalent
pentagon-free hexagonal binning on a local equirectangular projection
(valid within the documented cell radius). Earth radius is the mean radius
6 371 008.8 m, matching the haversine constant used in geopy.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ml._logging import log_fit

EARTH_RADIUS_M = 6371008.8


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (vectorised, haversine formula)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=np.float64))
                              for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing in degrees (0..360) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=np.float64))
                              for x in (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def hex_cell(lat, lon, cell_size_m=1000.0, origin_lat=0.0, origin_lon=0.0):
    """Hexagonal bin id on a local equirectangular projection (h3-equivalent).

    Two integers (q, r) in axial coordinates; the same (q, r) for a point
    stays stable across repeated calls and across batches.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    # metres east of origin (equirectangular, valid within cell radius)
    x = np.radians(lon - origin_lon) * EARTH_RADIUS_M * np.cos(np.radians(origin_lat))
    y = np.radians(lat - origin_lat) * EARTH_RADIUS_M
    # hex grid with flat-topped cells of width cell_size_m: sqrt(3) spacing
    size = float(cell_size_m)
    r = np.round(y / (1.5 * size)).astype(np.int64)
    q = np.round(x / (np.sqrt(3.0) * size) - 0.5 * r).astype(np.int64)
    return q, r


def _stack_numeric_safe(b, cols):
    from cadw.ml.gbdt import _stack_numeric
    return _stack_numeric(b, cols)


class GeoAggModel:
    """Fitted geo-aggregation container (cell size and origin)."""

    def __init__(self, lat_col, lon_col, params):
        self.lat_col_ = lat_col
        self.lon_col_ = lon_col
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return 256

    def to_dict(self):
        return {"lat_col": self.lat_col_, "lon_col": self.lon_col_,
                "params": self.params_}

    def _collect(self, df, mg):
        lats, lons = [], []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="geo.collect", extra_factor=3.0)
            lats.append(b.column(self.lat_col_).cast(pa.float64(), safe=False)
                        .to_numpy(zero_copy_only=False))
            lons.append(b.column(self.lon_col_).cast(pa.float64(), safe=False)
                        .to_numpy(zero_copy_only=False))
        return np.concatenate(lats), np.concatenate(lons)

    def hex_aggregate(self, df, *, value_col=None, agg="count", memory_guard=None):
        """Group points by hex cell and aggregate an optional value column."""
        mg = memory_guard or df._memory_guard
        lat, lon = self._collect(df, mg)
        if lat.size == 0:
            raise ValueError("geo.hex_aggregate: empty input")
        q, r = hex_cell(lat, lon, self.params_["cell_size_m"],
                        self.params_["origin_lat"], self.params_["origin_lon"])
        keys = np.column_stack([q, r])
        if value_col is None or agg == "count":
            uniq, counts = np.unique(keys, axis=0, return_counts=True)
            return uniq, counts.astype(np.float64)
        vals = []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="geo.values", extra_factor=3.0)
            vals.append(b.column(value_col).cast(pa.float64(), safe=False)
                        .to_numpy(zero_copy_only=False))
        v = np.concatenate(vals)
        if v.size != lat.size:
            raise ValueError("geo.hex_aggregate: value length mismatch")
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        out = np.zeros(uniq.shape[0], dtype=np.float64)
        if agg == "sum":
            np.add.at(out, inv, v)
        elif agg == "mean":
            cnt = np.zeros_like(out)
            np.add.at(out, inv, v); np.add.at(cnt, inv, 1.0)
            out = out / np.maximum(cnt, 1.0)
        else:
            raise ValueError(f"geo.hex_aggregate: agg must be 'count', 'sum' or 'mean'; got {agg!r}")
        return uniq, out


def fit_geo_agg(df, lat_col, lon_col, *, cell_size_m=1000.0,
                origin_lat=None, origin_lon=None,
                memory_guard: Optional[MemoryGuard] = None):
    """Fit a geo-aggregation config (cell size and origin)."""
    if lat_col not in df._columns:
        raise KeyError(f"fit_geo_agg: column {lat_col!r} not in {df._columns}")
    if lon_col not in df._columns:
        raise KeyError(f"fit_geo_agg: column {lon_col!r} not in {df._columns}")
    if cell_size_m <= 0:
        raise ValueError(f"fit_geo_agg: cell_size_m must be > 0; got {cell_size_m}")
    mg = memory_guard or df._memory_guard
    if origin_lat is None or origin_lon is None:
        lats, lons = [], []
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name="geo.fit", extra_factor=3.0)
            lats.append(b.column(lat_col).cast(pa.float64(), safe=False)
                        .to_numpy(zero_copy_only=False))
            lons.append(b.column(lon_col).cast(pa.float64(), safe=False)
                        .to_numpy(zero_copy_only=False))
        la = np.concatenate(lats) if lats else np.zeros(0)
        lo = np.concatenate(lons) if lons else np.zeros(0)
        if la.size == 0:
            raise ValueError("fit_geo_agg: empty input")
        if np.isnan(la).any() or np.isnan(lo).any():
            raise ValueError("fit_geo_agg: nulls in lat/lon")
        origin_lat = float(la.mean())
        origin_lon = float(lo.mean())
    model = GeoAggModel(lat_col, lon_col,
                        {"cell_size_m": float(cell_size_m),
                         "origin_lat": float(origin_lat),
                         "origin_lon": float(origin_lon)})
    log_fit("geo_agg", state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="geo.model")
    return model
