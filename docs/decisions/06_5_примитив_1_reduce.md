# 5. Примитив 1: `Reduce`

> v0.9.28 B.2 split — original DECISIONS.md lines 184-247.


### 5.1. Назначение и граница

Свёртка потока батчей в один аккумулятор. Состояние **не зависит от N**.

| Подходит | Не подходит |
|---|---|
| `count/sum/min/max/mean` | `median`, точные `quantile` |
| `var/std` (Welford) | `nunique`, `mode` |
| `cov/corr` (d×d) | `distinct` |
| `histogram` (binning LightGBM) | |
| `partial_fit`-состояние | |

### 5.2. Контракт

```python
class ReduceOp:
    def init(self) -> State
    def update(self, state, batch) -> State
    def merge(self, a, b) -> State
    def finalize(self, state) -> Any
    def state_nbytes(self, state) -> int

def reduce(batch_iter, op, memory_guard, *, log=True) -> Any
```

**Правило:** `update`/`merge` — только векторизованные (`pa.compute`, `np`).
Построчные `as_py()` запрещены.

**Инвариант:** результат не зависит от `batch_size` (проверяется параметризованными тестами).

### 5.3. Числовой пример (mean, batch_size=2)

`x = [10,20,30,40,50]`. State `(sum, count)`:

| Шаг | В RAM | state |
|---|---|---|
| init | — | `(0, 0)` |
| batch #1 | 2 числа | `(30, 2)` |
| batch #2 | 2 числа | `(100, 4)` |
| batch #3 | 1 число | `(150, 5)` |
| finalize | — | `30.0` |

**Reduce — единственный примитив без записи на диск.** Если реализация
пишет — архитектурная ошибка (тест `test_reduce_does_not_spill`).

### 5.4. Реализации

| Операция | Состояние | Замечание |
|---|---|---|
| `CountOp` | `(n_valid, n_null)` | skipna |
| `SumOp` | `(sum_f64, n_valid)` | промоушен float64; int64 overflow не обрабатывается (§11.5) |
| `MinMaxOp` | `(scalar, valid)` | `pc.min_max` |
| `MeanOp` | `(sum, n)` | — |
| `StdOp` | `(n, mean, M2)` | Welford (Chan et al.) |
| `CovarianceOp` | `(n, sum_x[d], sum_xx[d,d])` | O(d²); при большом d — §11.5 |
| `HistogramOp` | `counts[B]` | требует min/max (двухпроходный) |

Welford: на `1e9 + [1,2,3]` даёт `std=1.0`; наивная формула — мусор.
Числовая устойчивость — часть контракта.

---
