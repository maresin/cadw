#!/usr/bin/env bash
# patch_09_drop_partial_fit: удалить partial_fit и sklearn-зависимость.
# После 09 ML-слой CADW = to_batches(format=...) только (экспорт).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-09-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_09 (drop partial_fit)"

BR="patch/09-drop-partial-fit"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- 1. dataframe.py: удалить partial_fit, _SAFE_PARTIAL_FIT ----------
python3 - <<'PY'
import re
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# 1a. Удалить whitelist (_SAFE_PARTIAL_FIT)
m = re.search(
    r"_SAFE_PARTIAL_FIT = frozenset\(\{.*?\}\)\n\n\n",
    s,
    re.DOTALL,
)
if not m:
    raise SystemExit("_SAFE_PARTIAL_FIT not found")
s = s.replace(m.group(0), "", 1)

# 1b. Удалить блок partial_fit (от комментария до следующего метода)
m = re.search(
    r"    # -+ partial_fit -+\n.*?(?=    # -+ helpers -+)",
    s,
    re.DOTALL,
)
if not m:
    raise SystemExit("partial_fit block not found")
s = s.replace(m.group(0), "", 1)

p.write_text(s)
print("dataframe.py: partial_fit и _SAFE_PARTIAL_FIT удалены")
PY

# ---------- 2. Удалить ML-тесты ----------
rm -f tests/test_ml_integration.py
rm -f tests/test_sklearn_extras.py
echo "removed: tests/test_ml_integration.py, tests/test_sklearn_extras.py"

# ---------- 3. test_api_availability.py: убрать sklearn-секцию ----------
python3 - <<'PY'
import re
from pathlib import Path
p = Path("tests/test_api_availability.py")
s = p.read_text()

m = re.search(
    r"# -+ sklearn -+\n.*?(?=\Z)",
    s,
    re.DOTALL,
)
if not m:
    raise SystemExit("sklearn section not found")
s = s.replace(m.group(0), "").rstrip() + "\n"

# Обновить docstring
s = s.replace(
    "Pre-flight guard: API PyArrow / sklearn / xxhash, которые CADW",
    "Pre-flight guard: API PyArrow / xxhash, которые CADW",
)
s = s.replace(
    "24 эпизода отладки, ~40% из них —\nAPI-контракты. Классы: параметры",
    "24 эпизода отладки, ~40% из них —\nAPI-контракты. После патча 09 sklearn удалён из CADW; остаются только\nPyArrow и xxhash. Классы: параметры",
)
p.write_text(s)
print("test_api_availability.py: sklearn section removed")
PY

# ---------- 4. pyproject.toml: убрать scikit-learn ----------
python3 - <<'PY'
from pathlib import Path
p = Path("pyproject.toml")
s = p.read_text()
s = s.replace('    "scikit-learn>=1.4",\n', '')
p.write_text(s)
print("pyproject.toml: scikit-learn removed")
PY

# ---------- 5. README.md: ML-секция → экспорт ----------
python3 - <<'PY'
import re
from pathlib import Path
p = Path("README.md")
s = p.read_text()

# Удалить всю "## ML-интеграция" секцию
m = re.search(r"## ML-интеграция\n.*?(?=\n## |\Z)", s, re.DOTALL)
if m:
    s = s.replace(m.group(0), "", 1)

# Вставить короткую "## ML и внешние библиотеки" перед "## Документация"
new_ml = """## ML и внешние библиотеки

CADW **не является ML-фреймворком**. Он даёт `to_batches(format=...)` для
экспорта батчей в вашу модель. Если ваша модель OOM-ит — это её состояние,
а не CADW. Гарантия out-of-core действует только на код CADW, не на
сторонние библиотеки.

```python
for batch in df.to_batches(format="numpy", batch_size=10_000):
    # batch: dict[str, np.ndarray]
    process_batch(batch)
```

**Не поддерживается:**
- адаптеры к `lgb.train`, `xgb.fit`, `PCA.fit`, `HistGradientBoosting.fit`
  и любым `fit`-методам без `partial_fit` — они OOM-ят внутри библиотеки,
  `MemoryGuard` не спасает;
- `partial_fit`-адаптеры — whitelist невозможен без доверия к модели.

**Планируется** (v0.3+): собственные алгоритмы на примитивах:
GBDT (level-wise, аналог `HistGradientBoosting`), Linear, k-means, PCA,
Naive Bayes. См. `DECISIONS.md` §12.5.

"""
anchor = "## Документация"
if anchor in s:
    s = s.replace(anchor, new_ml + anchor, 1)
else:
    s = s.rstrip() + "\n\n" + new_ml

# Обновить статус
s = s.replace(
    "`unique`, `sort_values`, `groupby`, `partial_fit` (sklearn, потоково), "
    "E2E на 200k строк / 20 MB budget. См. `DECISIONS.md` §11.1.",
    "`unique`, `sort_values`, `groupby`, `to_batches(format=...)`, "
    "E2E на 200k строк / 20 MB budget. ML — вне scope CADW. "
    "См. `DECISIONS.md` §11.1.",
)

p.write_text(s)
print("README.md: ML-секция → 'экспорт батчей', sklearn удалён")
PY

# ---------- 6. DECISIONS.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

# 6.1 §3.9 — расширить (не только O(N)-adapters, но и partial_fit)
s = s.replace(
    "9. **Никаких адаптеров к O(N)-алгоритмам.** Любая внешняя функция,\n"
    "   которой нужен весь датасет в RAM (`lgb.train`, `PCA.fit`,\n"
    "   `XGBRegressor.fit`, любой `fit` без `partial_fit`), несовместима с\n"
    "   целью CADW. `MemoryGuard` не спасёт: OOM происходит **внутри**\n"
    "   сторонней библиотеки, до возврата управления. Такие функции не\n"
    "   подключаются как адаптеры; если нужны — пишутся на примитивах.\n"
    "   Исключение — `partial_fit` у моделей с **доказанным** O(1)-state\n"
    "   (см. §11.5, whitelist в `CadwDataFrame`).",
    "9. **Никаких ML-адаптеров.** CADW не подключает сторонние ML-модели\n"
    "   через адаптеры, ни через `fit`, ни через `partial_fit`.\n"
    "   - `fit`-модели (`lgb.train`, `PCA.fit`, `XGBRegressor.fit`,\n"
    "     `HistGradientBoosting.fit`) требуют весь датасет в RAM —\n"
    "     OOM происходит **внутри** библиотеки, `MemoryGuard` не спасает.\n"
    "   - `partial_fit`-модели (`SGDClassifier`, `StandardScaler`,\n"
    "     `MiniBatchKMeans`, `IncrementalPCA`) **могут** быть O(1)-state,\n"
    "     но whitelist — политика на доверии: любая модель вне whitelist\n"
    "     или модель, чей state вырос, OOM-ит без предупреждения от CADW.\n"
    "     Ответственность за state модели не должна перекладываться на\n"
    "     CADW.\n"
    "   - **Единственный ML-интерфейс CADW — `to_batches(format=...)`.**\n"
    "     Это экспорт: CADW выдаёт батчи, пользователь сам решает. Если\n"
    "     его модель OOM-ит — это его модель, CADW не создаёт иллюзию.\n"
    "   - ML-функции, которые действительно нужны в out-of-core, пишутся\n"
    "     на примитивах (v0.3: GBDT, Linear, k-means, PCA, NB).",
    1,
)

# 6.2 §4.1 уровень 6
s = s.replace(
    "│ 6. ML: to_batches(format), partial_fit (O(1)-state models)    │",
    "│ 6. Экспорт: to_batches(format='arrow'|'pandas'|'numpy'|'torch') │",
    1,
)

# 6.3 §11.1: убрать ML-строки, добавить чистую
s = s.replace(
    "| ML: `to_batches(format)` + `partial_fit` (06a) | ✅ | 12 |\n",
    "",
    1,
)
s = s.replace(
    "| sklearn extras (`MiniBatchKMeans`, `IncrementalPCA`) | ✅ | 4 + 1 (slow) |\n",
    "",
    1,
)
s = s.replace(
    "| ~~LightGBM `to_lightgbm`~~ | ❌ удалён (08: O(N), противоречит §3.9) | 0 |",
    "| Экспорт батчей: `to_batches(format=...)` | ✅ | (в CadwDataFrame) |\n"
    "| ~~`partial_fit`~~ | ❌ удалён (09: whitelist — политика на доверии, §3.9) | 0 |\n"
    "| ~~LightGBM `to_lightgbm`~~ | ❌ удалён (08: O(N), §3.9) | 0 |",
    1,
)

# 6.4 §11.5: удалить partial_fit-ограничения, обновить ML-policy
for line in [
    "| `PCA` (batch-only) не имеет `partial_fit`; `IncrementalPCA.partial_fit(X)` без `n_features` (выводится из `X.shape[1]`) | `CadwDataFrame.partial_fit` | для batch-only `PCA` — `to_batches(format='numpy')` → `np.vstack` → `PCA.fit` (не out-of-core). `n_features` — API `LatentDirichletAllocation`/`SGDOneClassSVM`, не PCA |\n",
    "| `SGDClassifier.partial_fit` требует `classes=[...]` на первом вызове | `CadwDataFrame.partial_fit` | `classes=[0, 1]` в kwargs первого вызова |\n",
    "| Категориальные признаки для LightGBM — int-encoded, не строки | `to_lightgbm` | пользователь кодирует сам; `encode_categorical` — кандидат v0.2 |\n",
]:
    s = s.replace(line, "", 1)

old_policy_start = "**Политика ML (после патча 08).**"
old_policy_end = "### 11.6. Технический долг"
if old_policy_start in s:
    # Заменить весь блок policy
    import re
    m = re.search(
        r"\*\*Политика ML \(после патча 08\)\.\*\*.*?(?=### 11\.6)",
        s, re.DOTALL,
    )
    if m:
        new_policy = """**Политика ML (после патча 09).** CADW **не является ML-фреймворком**.
Он предоставляет примитивы (Reduce, HashShuffle, Sort) и фасад
(CadwDataFrame). ML-функциональность делится на три класса:

1. **Экспорт батчей** (`to_batches(format='arrow'|'pandas'|'numpy'|'torch')`):
   поддерживается. Ответственность за OOM в сторонней модели — на
   пользователе. CADW не создаёт иллюзию, что «обучил на больших данных».

2. **Адаптеры к ML-моделям** (`fit`, `partial_fit`, `lgb.train`,
   `PCA.fit`, `SGDClassifier.partial_fit`): **не поддерживаются** (§3.9).
   Причины: `fit` — O(N)-state, OOM внутри библиотеки; `partial_fit` —
   whitelist на доверии, любая модель вне whitelist или модель с выросшим
   state OOM-ит без предупреждения. Разделение ответственности: state
   модели — её ответственность, не CADW.

3. **Собственные ML-реализации на примитивах** (v0.3+):
   - GBDT (level-wise, аналог `HistGradientBoosting`) —
     classification/regression/ranking/time series;
   - Linear (OLS, Ridge, Lasso, Logistic) — `CovarianceOp` + numpy.linalg;
   - k-means — `HashShuffle` (по кластеру) + `Reduce` (центроиды);
   - PCA — `CovarianceOp` (уже есть) + `eigh`;
   - Naive Bayes — `HashShuffle` (класс,фича) + `Reduce` (counts).

   Итого ~8–11 сессий на весь ML-слой. Каждая функция — композиция
   примитивов, не адаптер, гарантия out-of-core сохраняется.

"""
        s = s.replace(m.group(0), new_policy, 1)

# 6.5 §11.4: обновить формулировку регрессии
s = s.replace(
    "**Регрессия (обнаружено в патче 08, не закрыто):** в `_local_groupby`\n"
    "`list(ArrowIPCBatchReader(...))` материализует всю партицию **до**\n"
    "`check_state`. Для партиций ~500 MB — OOM до срабатывания проверки.\n"
    "Настоящий streaming: `partition_path.stat().st_size` (Arrow IPC без\n"
    "сжатия ≈ in-memory) → если > `available*0.5` — рекурсия без чтения.\n"
    "**Статус: must-fix, вынесен в отдельный патч.**",
    "**Регрессия (обнаружено в патче 08, must-fix, не закрыто):** в\n"
    "`_local_groupby` вызов `list(ArrowIPCBatchReader(...))` материализует\n"
    "всю партицию **до** `check_state`. Для партиции ~500 MB — OOM до\n"
    "срабатывания проверки. Это **не** закрывает пункт 1 из 02b1\n"
    "(«materialize до check»), а воспроизводит его на другом уровне.\n"
    "Настоящий streaming: `partition_path.stat().st_size` (Arrow IPC без\n"
    "сжатия ≈ in-memory) → если > `available*0.5` — рекурсия без чтения.\n"
    "**Статус: must-fix, патч 10. После 10 — аудит всех примитивов на\n"
    "паттерн «materialize до check_state».**",
    1,
)

# 6.6 §12.5 — переписать
old_125 = "### 12.5. Приоритет 4 — ML (06a ✅ `partial_fit`; 06b ❌ LightGBM удалён; v0.3 — свой GBDT)"
new_125 = "### 12.5. Приоритет 4 — ML (вне scope v0.1; v0.3 — собственные реализации)"
s = s.replace(old_125, new_125, 1)

# Найти и заменить подпункты 18-20
import re
m = re.search(
    r"18\. ~~`to_batches.*?(?=\n###|\Z)",
    s, re.DOTALL,
)
if m:
    new_items = """18. ✅ **Экспорт батчей:** `to_batches(format='arrow'|'pandas'|'numpy'|'torch')`
    (06a). Единственный ML-интерфейс CADW.
19. ❌ **`partial_fit`:** удалён (09). Whitelist — политика на доверии, §3.9.
20. ❌ **LightGBM `to_lightgbm`:** удалён (08). O(N), §3.9.

### 12.5.1. v0.3 — собственные ML-алгоритмы на примитивах

| # | Алгоритм | Покрывает | Примитивы | Сессий |
|---|---|---|---|---|
| 1 | **GBDT** (level-wise, аналог `HistGradientBoosting`) | classification (bin/multi), regression, ranking, time series (lag-features) | `HistogramOp` (binning) + `HashShuffle` (узел,фича,бин) + `Reduce` (sum_grad,sum_hess,count) + `Sort` (квантили) | 5–8 |
| 2 | **Linear** (OLS, Ridge, Lasso, Logistic) | baseline, интерпретируемость | `Reduce` (X^T X, X^T y = `CovarianceOp`-стиль) + `numpy.linalg.solve` | 1 |
| 3 | **k-means** | кластеризация, сегментация | `HashShuffle` (по кластеру) + `Reduce` (центроиды) + итерации | 1 |
| 4 | **PCA** | понижение размерности | `CovarianceOp` (есть) + `numpy.linalg.eigh` + transform | 0.5 |
| 5 | **Naive Bayes** | текст, быстрый baseline | `HashShuffle` (класс,фича) + `Reduce` (counts) | 0.5 |

**Итого ~8–11 сессий** — весь ML-слой. Специализированные (Isolation
Forest, ALS, ARIMA, LambdaMART) — v0.5+, тоже композиции.

"""
    s = s.replace(m.group(0), new_items, 1)

# 6.7 §13.3: удалить адаптер partial_fit
for line in [
    "| `IncrementalPCA` | `PCA.partial_fit` | у `PCA` нет `partial_fit` |\n",
]:
    s = s.replace(line, "", 1)
if "| Никаких ML-адаптеров | `partial_fit` whitelist |" not in s:
    s = s.replace(
        "| `_unique_columns(value, key)` | передавать как есть | `Multiple matches` |",
        "| `_unique_columns(value, key)` | передавать как есть | `Multiple matches` |\n"
        "| Никаких ML-адаптеров (§3.9) | `partial_fit` whitelist | whitelist — политика на доверии; любая модель вне whitelist → OOM без предупреждения |\n"
        "| Свой GBDT на примитивах (v0.3) | Адаптер к LightGBM / XGBoost / HGB | адаптер O(N) → OOM в сторонней библиотеке; `MemoryGuard` не спасает |",
        1,
    )

# 6.8 §13.4: добавить эпизод 26
episode_26 = """**Эпизод 26 (архитектурный, ключевой).** `partial_fit`-адаптер был
следующим шагом после `to_lightgbm`. Аргумент «whitelist безопасных
моделей» (SGD, StandardScaler, MiniBatchKMeans, IncrementalPCA) выглядел
разумным: эти модели имеют O(1)-state. Но whitelist — **политика на
доверии**, а не гарантия. Модель в whitelist может изменить state в новой
версии; новая модель может быть добавлена в whitelist по имени без
проверки. Более того, `partial_fit` создаёт **иллюзию**: пользователь
думает, что CADW «обучил на 500 ГБ», а фактически CADW выдал батчи в
sklearn, который мог OOM-нуть внутри. **Урок:** единственный
безопасный ML-интерфейс — **экспорт батчей** (`to_batches`).
Ответственность за OOM в сторонней модели — на пользователе, а не на
CADW. Всё, что требует out-of-core ML, пишется на примитивах (v0.3). Это
укрепляет §3.9: не только `fit`-адаптеры, но и `partial_fit`-адаптеры
запрещены. `partial_fit` может вернуться только как **собственная
реализация** на примитивах (например, `SGD` на `Reduce`), а не как
обёртка над sklearn.

"""
marker_135 = "### 13.5. Открытые вопросы"
s = s.replace(marker_135, episode_26 + marker_135, 1)

# 6.9 §14 резюме
s = s.replace(
    "- **Работает:** `MemoryGuard` (25), `Reduce` (33), `HashShuffle` (13),",
    "- **Работает:** `MemoryGuard` (25), `Reduce` (33), `HashShuffle` (13),",
)
s = s.replace(
    "**v0.1.2 (патч 08):** `to_lightgbm` удалён (§3.9). ML — только\n"
    "`partial_fit` для O(1)-state моделей. GBDT — на примитивах (v0.3).",
    "**v0.1.2 (патч 08):** `to_lightgbm` удалён (§3.9).\n\n"
    "**v0.1.3 (патч 09):** `partial_fit` удалён полностью (§3.9). ML-слой\n"
    "CADW = `to_batches(format=...)` только. Собственные ML-алгоритмы —\n"
    "v0.3 (GBDT, Linear, k-means, PCA, NB) на примитивах.",
    1,
)

p.write_text(s)
print("DECISIONS.md updated: §3.9 (ML), §4.1, §11.1/11.4/11.5, §12.5, §13.3/13.4/26, §14")
PY

# ---------- 7. CHANGELOG.md: v0.1.3 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()

new_entry = """## [0.1.3] — 2026-10-18

### Удалено
- `CadwDataFrame.partial_fit` — адаптер к `sklearn.partial_fit`. Причина:
  whitelist (`_SAFE_PARTIAL_FIT`) — политика на доверии; любая модель вне
  whitelist или модель с выросшим state OOM-ит без предупреждения от CADW.
  См. DECISIONS.md §3.9, §13.4 эпизод 26.
- `_SAFE_PARTIAL_FIT` whitelist.
- `tests/test_ml_integration.py` (11 + 1 slow).
- `tests/test_sklearn_extras.py` (4 + 1 slow).
- sklearn-секция из `tests/test_api_availability.py` (7 assertions).
- `scikit-learn` из `[dependency-groups]` (dev).
- ML-секции из README, заменены на «ML и внешние библиотеки» с policy.

### Добавлено в DECISIONS.md
- §3.9: «Никаких ML-адаптеров» (не только `fit`, но и `partial_fit`).
- §11.5: расширенная «Политика ML» (3 класса: экспорт / адаптеры запрещены
  / собственные реализации v0.3).
- §12.5.1: план v0.3 — GBDT, Linear, k-means, PCA, NB на примитивах (~8–11
  сессий на весь ML-слой).
- §13.4 эпизод 26: whitelist `partial_fit` — политика на доверии, не
  гарантия.

### Единственный ML-интерфейс CADW
- `to_batches(format='arrow'|'pandas'|'numpy'|'torch')` — экспорт батчей.
  Ответственность за OOM в сторонней модели — на пользователе.

"""

marker = "## [0.1.2]"
if marker in s:
    s = s.replace(marker, new_entry + marker, 1)
else:
    s = s.replace("## [0.1.0]", new_entry + "## [0.1.0]", 1)

p.write_text(s)
print("CHANGELOG.md: v0.1.3 entry added")
PY

# ---------- 8. uv lock && sync ----------
echo "=== uv lock && uv sync ==="
uv lock
uv sync

# ---------- 9. Проверка, что sklearn нигде не осталось ----------
echo "=== grep sklearn / partial_fit (должно быть 0) ==="
if grep -rn "sklearn\|partial_fit" cadw/ --include="*.py" 2>/dev/null; then
  echo "  WARNING: sklearn/partial_fit остатки в cadw/"
  exit 1
else
  echo "  OK: 0 совпадений в cadw/"
fi

# ---------- 10. Тесты ----------
echo "=== pytest (fast) ==="
set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
uv run pytest -q -m slow
RC_SLOW=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then echo "fast: rc=$RC_FAST"; exit 1; fi
if [[ $RC_SLOW -ne 0 ]]; then echo "slow: rc=$RC_SLOW"; exit 1; fi

# ---------- 11. Commit ----------
git add -A
git commit -m "09: drop partial_fit (whitelist — политика на доверии, §3.9); ML = to_batches only; ML-policy в DECISIONS"

echo "==> sha256"
sha256sum cadw/dataframe.py tests/test_api_availability.py DECISIONS.md
echo "==> 09 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"