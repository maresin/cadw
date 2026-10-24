# D.2.3 — Hybrid matrix factorization + content

**Version:** v0.9.36 · **Module:** `cadw/ml/hybrid_mf.py` · **Mixin:** `df.fit_hybrid_mf(...)`
**Complexity:** M · **State:** O((U+I)·k + I·c)

## Sources

- Forbes, Zhu, *Content-Boosted Collaborative Filtering*, UAI 2001.
- Melville, Mooney, Nagarajan, *Content-Boosted CF*, AAAI 2002.
- Schein et al., *Methods and Metrics for Cold-Start Recommendations*, SIGIR 2002.

## Design

ALS (D.2.2) gives item factors `Y`. Content columns are z-scored into `C`
(rows in the model item order) and a ridge map is fitted from content to
the MF factor on items that have interactions:

```
B = (C_w^T C_w + ridge*I)^-1 C_w^T Y_w
item_repr = Y + C @ B        # both k-dim: a residual correction
score(u, i) = item_repr_i . x_u
```

**Why a residual, not a concatenation.** `[Y, C@B]` is (k+c)-dim while the
user factor stays k-dim, so scoring breaks (this was an actual bug caught by
the test suite). Adding the content prediction as a residual keeps the
representation in the same user space, still one dot product per item, and
shrinks the noise in the MF factor for low-interaction items.

Cold start: `content_only_scores(user, content_rows)` scores never-seen
items from content alone (`(C @ B) . x_u`).

## RAM model

| Component | Bytes |
|---|---|
| ALS base (D.2.2) | `(U+I)*k*8` |
| Content matrix `C` | `I*c*8` |
| Content map `B` | `c*k*8` |

`check_state(C.nbytes)` before materialising C. C is built from a streamed
pass keeping one row per item, so it is O(I*c) and independent of the
interaction count. `log_fit("hybrid_mf", loss=content-fit MSE, ...)` after the
ridge solve.

## Benchmark (measured)

`scripts/bench_hybrid_mf.py` -> `.pingis/benches/hybrid_mf.json`, k=8, c=6,
5 sweeps. Item factors are constructed as a function of content plus 5%
noise, so content is informative by construction. Metric: Pearson
correlation between model scores and true scores for user 0, over all items.

| users x items | interactions | ALS corr | hybrid corr | delta | ALS fit | hybrid fit | hybrid peak |
|---|---|---|---|---|---|---|---|
| 300 x 200 | 27 871 | 0.1607 | 0.1862 | +0.0255 | 6.06 s | 6.08 s | 7.8 MB |
| 1 200 x 800 | 446 391 | 0.2171 | 0.2971 | +0.0799 | 96.44 s | 96.93 s | 79.4 MB |

The gain grows with interaction count and costs ~0.5% extra fit time. On
data where content is not informative this reduces to plain ALS with a
near-zero B; the ridge penalty prevents harm.

## Tests (10)

smoke/shapes/state · hybrid beats ALS on content-correlated data ·
closed-form ridge reference for B · cold-start content-only scoring ·
determinism · multi-batch invariance · state scales with items/users not
interactions · guard on the content matrix · errors (missing column, empty
content list) · source contract.

## Edge cases

- Items with no interactions still get a content vector; a `seen` mask
  excludes them from the ridge fit.
- Large `ridge` -> B -> 0, degrades gracefully to plain ALS.
- Non-numeric content raises `TypeError` at the cast.
