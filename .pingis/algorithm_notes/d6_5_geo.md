# D.6.5 Geospatial aggregation (haversine, bearing, hex binning)

## Paper
Sinnott, *Virtues of the Haversine*, Sky and Telescope 68(2), 1984.
Uber H3 / hexagonal binning (Brodsky, *H3: Uber Hexagonal Hierarchical
Spatial Index*, 2018). Reference in tests: closed-form haversine on the
unit sphere (analytic exact) and scipy spherical distance.

## Docs
- `cadw.ml.geo.haversine(lat1,lon1,lat2,lon2)`, `bearing(...)`,
  `hex_cell(lat,lon,cell_size_m,origin_lat,origin_lon)`.
- `fit_geo_agg(df, lat_col, lon_col, cell_size_m=1000.0, agg="count"|
  "sum"|"mean", value_col=None)` -> `GeoAggModel`.
- `CadwDataFrame.fit_geo_agg(...)`.

## Design
- Haversine in metres on the mean Earth radius; vectorised over batches.
- Hex binning on a flat local projection around the origin (cell_size_m).
- Streaming aggregation: per-cell count/sum maintained incrementally; state
  = #cells, independent of n.

## Benchmark (`.pingis/benches/geo.json`, cell_size_m=500)
| n | cells | total | fit_agg_s | peak_mb | state_b |
|---|---|---|---|---|---|
| 10 000 | 233 | 10 000 | 0.017 | 6.2 | 256 |
| 100 000 | 233 | 100 000 | 0.185 | 8.2 | 256 |
| 1 000 000 | 233 | 1 000 000 | 2.450 | 107.6 | 256 |
Reference: haversine matches the analytic great-circle distance exactly
(tolerance 1e-9 in tests); bearing cardinal directions verified. Peak RSS
at n=1M is 108 MB, under budget.

## RAM model
State = #cells * 8 bytes (256 B measured for 233 cells). Streaming per-cell
count/sum; every batch passes `check_batch(extra_factor>=2.0)`.
