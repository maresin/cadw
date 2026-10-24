# D.4.2 PELT change-point detection

## Paper
Killick, Fearnhead, Eckley, *Optimal detection of changepoints with a linear
computational cost*, JASA 107(500), 2012. Truong, Oudre, Vayatis, *Selective
review of offline change point detection methods*, Signal Processing 167, 2020.

## Docs
- `cadw.ml.pelt.fit_pelt(df, value_col, id_col=None, penalty=10.0, min_size=2, max_window=None)`
- `CadwDataFrame.fit_pelt(...)` binding.
- Model: `detect(df)` -> {series_id: [cp, ...]}, `change_points(df)` -> flat list.

## Design
- Exact PELT with the L2 (mean-shift) cost, cost evaluated in O(1) from
  cumulative sums of y and y^2.
- Killick pruning rule: discard s when F(s)+C(s,t)+K > F(t).
- `max_window` bounds the recursion horizon -> O(window) state (the "online"
  reading of the backlog line); with `max_window=None` the recursion is exact
  full-series PELT.
- Single series streamed; id_col input must be grouped/sorted (guarded).

## Benchmark (`.pingis/benches/pelt.json`, penalty=30)
| n | ours s | ruptures s | cps (ours) | cps (ruptures) | peak_mb | state_b |
|---|---|---|---|---|---|---|
| 1 000 | 0.005 | 0.144 | [334, 666] | [335, 665] | 5.0 | 256 |
| 5 000 | 0.021 | 4.134 | [1666, 3332] | [1665, 3335] | 0.4 | 256 |
| 20 000 | 0.101 | 73.572 | [6666, 13332] | [6665, 13330] | 1.9 | 256 |
Reference: ruptures.Pelt(model="l2", min_size=2). Same segmentations within
boundary ties (<=3 samples), 40-730x faster.

## RAM model
State = 256 B (parameters only). Detection holds the series (O(n)) plus the
F/cp arrays O(n); `max_window` bounds R and the cost horizon. check_batch on
every streamed batch.
