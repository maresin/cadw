# 6. Примитив 2: `HashShuffle`

> v0.9.28 B.2 split — original DECISIONS.md lines 248-306.


### 6.1. Инвариант

Строки раскладываются по P файлам так, что все строки с одинаковым ключом —
в одном файле. Партиции **не пересекаются по множеству ключей**
(хэш-коллизии разных ключей — норма; downstream различает по значению).

Применяется для: `groupby`, `join`, `unique`, `drop_duplicates`,
`set operations`, `pivot`, `target encoding`, `stratified sampling`.

### 6.2. `row_id` — обязательное поле

Каждая строка несёт `row_id` (глобальная позиция в исходном датасете).
Критическая деталь: при `pc.sort_indices` `row_id` строки на позиции `i` —
это `order_np[i] + offset`, а не `i + offset`.

### 6.3. Детерминированный хэш

- `xxhash.xxh64(_stable_bytes(v), seed).intdigest()`.
- `_stable_bytes(v) = repr(v).encode('utf-8')` — различает типы.
- Multi-key: boost `hash_combine` — `h ^= ch + 0x9E3779B97F4A7C15 + (h<<6) + (h>>2)`.
- Арифметика — NumPy `uint64` (Arrow `pc.add`/`pc.shift_left` возбуждают
  `ArrowInvalid` на overflow).
- NULL хэшируется детерминированно, отдельная группа.

### 6.4. Выбор P

1. **Sample:** `K_est = K_sample * N / s`; `P = ceil(K_est * safety / gpp)`,
   `safety=1.5`, `gpp = available*0.6 / bytes_per_key`.
2. **Рекурсия** при переполнении партиции (02b1): `_local_groupby` при
   `sum(b.nbytes) > available*0.5` вызывает `hash_shuffle(seed=depth+1)`,
   `MAX_DEPTH=5`; при превышении — `MemoryLimitExceeded` с указанием на skew.
3. **Skew-изоляция** (02b2): `_detect_heavy_keys` по sample (>5%) →
   тяжёлые агрегируются потоково (`_accumulate_heavy`), лёгкие — обычный
   shuffle + `_local_groupby`; конкатенация.

### 6.5. Числовой пример (P=4)

При `h(A)%4=0, h(B)%4=1, h(C)%4=2, h(D)%4=3`:

| partition | строки |
|---|---|
| 0 | (A,10,0), (A,11,4) |
| 1 | (B,20,1), (B,21,5) |
| 2 | (C,30,2), (C,31,6) |
| 3 | (D,40,3), (D,41,7) |

`A` только в 0, `B` только в 1, … — агрегация независима, конкатенация.

### 6.6. Тесты (13/13)

Инвариант «for each key, all rows in exactly one partition»;
`row_id` восстанавливает исходные данные; P>N; P=1; пустой вход → `ValueError`;
стабильность между батчами; seed меняет хэш; `estimate_num_partitions`;
`check_partition_buffers` срабатывает.

---
