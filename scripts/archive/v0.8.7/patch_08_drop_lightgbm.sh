#!/usr/bin/env bash
# patch_08_drop_lightgbm: удалить to_lightgbm (O(N) adapter), whitelist partial_fit,
# policy в DECISIONS, обновить CHANGELOG/README.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-08-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_08 (drop lightgbm)"

BR="patch/08-drop-lightgbm"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- 1. dataframe.py: удалить to_lightgbm, whitelist partial_fit ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# 1a. Удалить блок LightGBM целиком (от комментария до _wrap_result_table)
import re
pattern = re.compile(
    r"    # -+ LightGBM -+\n.*?(?=    # -+ helpers -+)",
    re.DOTALL,
)
if not pattern.search(s):
    raise SystemExit("to_lightgbm block not found")
s = pattern.sub("", s)

# 1b. Добавить whitelist safe partial_fit models (module-level, перед классом)
whitelist = '''_SAFE_PARTIAL_FIT = frozenset({
    # linear (SGD)
    "SGDClassifier", "SGDRegressor", "SGDOneClassSVM",
    "PassiveAggressiveClassifier", "PassiveAggressiveRegressor",
    "Perceptron",
    # scalers
    "StandardScaler", "MinMaxScaler", "RobustScaler",
    # clustering
    "MiniBatchKMeans",
    # NB
    "MultinomialNB", "BernoulliNB", "ComplementNB",
    # decomposition (safe for d < 1000; см. DECISIONS §11.5)
    "IncrementalPCA",
    # neural (sklearn, small)
    "MLPClassifier", "MLPRegressor",
})


'''
anchor = "class CadwDataFrame:"
if anchor not in s:
    raise SystemExit("class CadwDataFrame not found")
s = s.replace(anchor, whitelist + anchor, 1)

# 1c. Обновить docstring partial_fit: warning
old_doc = '''        """Потоковое обучение модели с методом .partial_fit.

        - X_cols: колонки признаков (str или list[str]);
        - y_col=None: модель без y (StandardScaler, PCA). partial_fit(X, **kw).
        - y_col=str: partial_fit(X, y, **kw).

        Внутри: per-batch конвертация X (np.column_stack) и y (.to_numpy);
        никаких построчных циклов. Модель обучается батч за батчем.

        Возвращает обученную модель (тот же объект, изменённый in-place).
        """'''

new_doc = '''        """Потоковое обучение модели с методом .partial_fit.

        - X_cols: колонки признаков (str или list[str]);
        - y_col=None: модель без y (StandardScaler, IncrementalPCA).
          partial_fit(X, **kw);
        - y_col=str: partial_fit(X, y, **kw).

        Внутри: per-batch конвертация X (np.column_stack) и y (.to_numpy);
        никаких построчных циклов. Модель обучается батч за батчем.

        **Контракт безопасности.** partial_fit должен давать O(1)-state
        относительно N (числа строк). Это не контроль со стороны CADW —
        это ответственность модели. Модели с O(N)-state (любая модель, где
        partial_fit накапливает все данные) приведут к OOM независимо от
        MemoryGuard. Для моделей, не входящих в _SAFE_PARTIAL_FIT,
        выдаётся RuntimeWarning. См. DECISIONS.md §11.5.

        Возвращает обученную модель (тот же объект, изменённый in-place).
        """'''

if old_doc not in s:
    raise SystemExit("partial_fit docstring not found")
s = s.replace(old_doc, new_doc, 1)

# 1d. Добавить warning в тело partial_fit (после проверки колонок)
old_body = '''        n_batches = 0
        for b in self._iter_arrow_batches(batch_size=batch_size):
            X = np.column_stack(
                [b.column(c).to_numpy(zero_copy_only=False) for c in x_cols]
            )
            if y_col is not None:'''
new_body = '''        _cls = type(model).__name__
        if _cls not in _SAFE_PARTIAL_FIT:
            import warnings
            warnings.warn(
                f"CadwDataFrame.partial_fit: model class {_cls!r} is not in "
                f"_SAFE_PARTIAL_FIT. Only safe if its partial_fit keeps O(1) "
                f"state w.r.t. N. Otherwise OOM possible despite MemoryGuard. "
                f"See DECISIONS.md §11.5.",
                RuntimeWarning,
                stacklevel=2,
            )

        n_batches = 0
        for b in self._iter_arrow_batches(batch_size=batch_size):
            X = np.column_stack(
                [b.column(c).to_numpy(zero_copy_only=False) for c in x_cols]
            )
            if y_col is not None:'''
if old_body not in s:
    raise SystemExit("partial_fit body not found")
s = s.replace(old_body, new_body, 1)

p.write_text(s)
print("dataframe.py: to_lightgbm removed, partial_fit whitelist + warning")
PY

# ---------- 2. Удалить test_lightgbm.py ----------
rm -f tests/test_lightgbm.py
echo "removed: tests/test_lightgbm.py"

# ---------- 3. test_api_availability.py: убрать LightGBM assertions ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_api_availability.py")
s = p.read_text()

# Удалить секцию LightGBM
import re
pattern = re.compile(
    r"# -+ LightGBM -+\n.*?\Z",
    re.DOTALL,
)
if not pattern.search(s):
    raise SystemExit("LightGBM section in test_api_availability not found")
s = pattern.sub("", s).rstrip() + "\n"

# Обновить docstring (убрать LightGBM)
s = s.replace(
    "Pre-flight guard: API PyArrow / sklearn / LightGBM / xxhash, которые CADW",
    "Pre-flight guard: API PyArrow / sklearn / xxhash, которые CADW",
)
s = s.replace(
    "использует. Если внешняя библиотека изменит сигнатуру — этот файл упадёт",
    "использует. Если внешняя библиотека изменит сигнатуру — этот файл упадёт",
)
p.write_text(s)
print("test_api_availability.py: LightGBM section removed")
PY

# ---------- 4. pyproject.toml: убрать lightgbm из dev ----------
python3 - <<'PY'
from pathlib import Path
p = Path("pyproject.toml")
s = p.read_text()
s = s.replace('    "lightgbm>=4.0",\n', '')
p.write_text(s)
print("pyproject.toml: lightgbm removed from [dependency-groups] dev")
PY

# ---------- 5. README.md: убрать LightGBM-секцию ----------
python3 - <<'PY'
from pathlib import Path
import re
p = Path("README.md")
s = p.read_text()

# Удалить "### LightGBM: честный downsampling" секцию
pattern = re.compile(
    r"### LightGBM: честный downsampling\n.*?(?=### |\Z)",
    re.DOTALL,
)
s = pattern.sub("", s)

# Удалить "### Категориальные признаки (LightGBM)" секцию
pattern2 = re.compile(
    r"### Категориальные признаки \(LightGBM\)\n.*?(?=### |\Z)",
    re.DOTALL,
)
s = pattern2.sub("", s)

# Убрать упоминания LightGBM из статуса
s = s.replace(
    "Реализовано: `CadwDataFrame`, `Dataset`, `RowFilter`, ленивые `__getitem__`, "
    "`unique`, `sort_values`, `groupby`, E2E на 200k строк / 20 MB budget. "
    "См. `DECISIONS.md` §11.1.",
    "Реализовано: `CadwDataFrame`, `Dataset`, `RowFilter`, ленивые `__getitem__`, "
    "`unique`, `sort_values`, `groupby`, `partial_fit` (sklearn, потоково), "
    "E2E на 200k строк / 20 MB budget. См. `DECISIONS.md` §11.1.",
)

# Добавить явную policy в ML-секцию
s = s.replace(
    "### sklearn: partial_fit (потоковое обучение)",
    "### sklearn: partial_fit (потоковое обучение)\n\n"
    "**Политика ML.** CADW поддерживает только модели с **O(1)-state** "
    "относительно числа строк (истинно инкрементальные). Модели, которым "
    "нужен весь датасет в RAM (`PCA.fit`, `lgb.train`, любой `fit` без "
    "`partial_fit`), **не поддерживаются** — они OOM-ят независимо от "
    "`MemoryGuard`. См. `DECISIONS.md` §3.9 и §11.5.",
)

p.write_text(s)
print("README.md: LightGBM removed, ML policy added")
PY

# ---------- 6. DECISIONS.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

# 6.1 §3: новый принцип 3.9
s = s.replace(
    "8. **Честный фасад.** `CadwDataFrame` не реализует алгоритмы; guard-тесты\n"
    "   (`test_facade_has_no_as_py`, `test_facade_has_no_row_loop_ast`) ловят\n"
    "   регресс на уровне AST.",
    "8. **Честный фасад.** `CadwDataFrame` не реализует алгоритмы; guard-тесты\n"
    "   (`test_facade_has_no_as_py`, `test_facade_has_no_row_loop_ast`) ловят\n"
    "   регресс на уровне AST.\n"
    "9. **Никаких адаптеров к O(N)-алгоритмам.** Любая внешняя функция,\n"
    "   которой нужен весь датасет в RAM (`lgb.train`, `PCA.fit`,\n"
    "   `XGBRegressor.fit`, любой `fit` без `partial_fit`), несовместима с\n"
    "   целью CADW. `MemoryGuard` не спасёт: OOM происходит **внутри**\n"
    "   сторонней библиотеки, до возврата управления. Такие функции не\n"
    "   подключаются как адаптеры; если нужны — пишутся на примитивах.\n"
    "   Исключение — `partial_fit` у моделей с **доказанным** O(1)-state\n"
    "   (см. §11.5, whitelist в `CadwDataFrame`).",
    1,
)

# 6.2 §4.1 уровень 6: переформулировать
s = s.replace(
    "│ 6. ML: to_batches(format), partial_fit, to_lightgbm          │",
    "│ 6. ML: to_batches(format), partial_fit (O(1)-state models)    │",
    1,
)

# 6.3 §11.1: удалить строку LightGBM, добавить "removed"
s = s.replace(
    "| LightGBM `to_lightgbm(max_rows)` (06b, честный downsampling) | ✅ | 11 |\n",
    "",
    1,
)
s = s.replace(
    "| sklearn extras (`MiniBatchKMeans`, `IncrementalPCA`) | ✅ | 4 + 1 (slow) |",
    "| sklearn extras (`MiniBatchKMeans`, `IncrementalPCA`) | ✅ | 4 + 1 (slow) |\n"
    "| ~~LightGBM `to_lightgbm`~~ | ❌ удалён (08: O(N), противоречит §3.9) | 0 |",
    1,
)

# 6.4 §11.5: удалить LightGBM-строку, добавить policy
s = s.replace(
    "| `to_lightgbm(max_rows=N)` материализует до N строк | `CadwDataFrame` | LightGBM требует весь датасет в RAM; N — под бюджет; для большего — distributed LightGBM |\n",
    "",
    1,
)
s = s.replace(
    "| Категориальные признаки для LightGBM — int-encoded, не строки | `to_lightgbm` | пользователь кодирует сам; `encode_categorical` — кандидат v0.2 |\n",
    "",
    1,
)
s = s.replace(
    "| `to_lightgbm(max_rows)` (downsampling) | `lgb.Sequence` | LightGBM всё равно O(N) на гистограммах; Sequence мало документирован |\n",
    "",
    1,
)

# Добавить блок ML-policy в §11.5 (после таблицы)
marker_115_end = "### 11.6. Технический долг"
ml_policy = """**Политика ML (после патча 08).** CADW не является ML-фреймворком.
Он предоставляет примитивы и фасад; ML-алгоритмы делятся на три класса:

1. **Истинно инкрементальные** (`partial_fit` с O(1)-state):
   `SGDClassifier`, `StandardScaler`, `MiniBatchKMeans`,
   `IncrementalPCA` (d<1000), `MultinomialNB`, …
   → поддерживаются через `CadwDataFrame.partial_fit`.
   Whitelist — `_SAFE_PARTIAL_FIT` в `dataframe.py`.
   Модели вне whitelist → `RuntimeWarning` (не блок).

2. **Batch-only, O(N)-state** (`fit` без `partial_fit`):
   `PCA`, `RandomForest`, `XGBoost`, `LightGBM` (`lgb.train`),
   `HistGradientBoosting`, … → **не поддерживаются** (§3.9).
   Причина: OOM внутри библиотеки, `MemoryGuard` не спасает.

3. **Собственные реализации на примитивах** (v0.3+):
   GBDT (level-wise, аналог `HistGradientBoosting`),
   `PCA` (через `CovarianceOp` — уже есть),
   k-means (итеративный `Reduce` центроидов).
   Каждая — композиция `Reduce`/`HashShuffle`/`Sort`, не адаптер.

"""
s = s.replace(marker_115_end, ml_policy + marker_115_end, 1)

# 6.5 §11.4: добавить новый must-fix (регрессия _local_groupby)
s = s.replace(
    "**Итоговый статус (после 02b2-fix3):**",
    "**Регрессия (обнаружено в патче 08, не закрыто):** в `_local_groupby`\n"
    "`list(ArrowIPCBatchReader(...))` материализует всю партицию **до**\n"
    "`check_state`. Для партиций ~500 MB — OOM до срабатывания проверки.\n"
    "Настоящий streaming: `partition_path.stat().st_size` (Arrow IPC без\n"
    "сжатия ≈ in-memory) → если > `available*0.5` — рекурсия без чтения.\n"
    "**Статус: must-fix, вынесен в отдельный патч.**\n\n"
    "**Итоговый статус (после 02b2-fix3):**",
    1,
)

# 6.6 §12.5: переписать ML-часть
s = s.replace(
    "### 12.5. Приоритет 4 — ML и визуализация (06a ✅, 06b ✅ LightGBM)",
    "### 12.5. Приоритет 4 — ML (06a ✅ `partial_fit`; 06b ❌ LightGBM удалён; v0.3 — свой GBDT)",
    1,
)
s = s.replace(
    "18. `to_batches(format='numpy'/'torch')`.\n"
    "19. `partial_fit`-интеграция (SGD, MiniBatchKMeans).\n"
    "20. LightGBM binning через `HistogramOp` + `Dataset`-адаптер.",
    "18. ~~`to_batches(format='numpy'/'torch')`~~ — ✅ (06a).\n"
    "19. ~~`partial_fit`-интеграция~~ — ✅ (06a, whitelist).\n"
    "20. ~~LightGBM binning через `HistogramOp` + `Dataset`-адаптер~~ — ❌\n"
    "    удалён (08, §3.9). Заменено на: собственный GBDT на примитивах\n"
    "    (v0.3): `HistogramOp` (binning) + `HashShuffle` по (узел,фича,бин)\n"
    "    + `Reduce` (sum_grad/sum_hess/count). Level-wise, аналог\n"
    "    `sklearn.HistGradientBoosting`.",
    1,
)

# 6.7 §13.3: удалить решение про to_lightgbm, добавить отказ от адаптеров
s = s.replace(
    "| `to_lightgbm(max_rows)` (downsampling) | `lgb.Sequence` | LightGBM всё равно O(N) на гистограммах; Sequence мало документирован |\n",
    "",
    1,
)
s = s.replace(
    "| `IncrementalPCA` | `PCA.partial_fit` | у `PCA` нет `partial_fit` |",
    "| `IncrementalPCA` | `PCA.partial_fit` | у `PCA` нет `partial_fit` |\n"
    "| Свой GBDT на примитивах (v0.3) | Адаптер к LightGBM / XGBoost / HGB | адаптер O(N) → OOM в сторонней библиотеке; `MemoryGuard` не спасает |",
    1,
)

# 6.8 §13.4: добавить эпизод 25
episode_25 = """**Эпизод 25 (архитектурный, важный).** `to_lightgbm(max_rows=N)` был
реализован как «честный downsampling» — с явным `max_rows` и документированным
ограничением. Но это **противоречит цели проекта**: пользователь думает, что
обучил модель на 500 ГБ, а фактически — на 1 млн строк. `lgb.train` — O(N)-state
(все градиенты в RAM), `lgb.Sequence` не помогает (LightGBM всё равно строит
гистограммы по всему датасету), форк LightGBM — вне нашей задачи.
**Урок:** адаптер к O(N)-алгоритму — иллюзия out-of-core. `MemoryGuard` не
может предотвратить OOM, который происходит **внутри** сторонней библиотеки,
после возврата управления в её код. Поэтому принцип §3.9: no adapters to
O(N)-algorithms. ML-функции, требующие весь датасет, должны быть
реализованы на примитивах (v0.3: свой GBDT) или не реализованы вовсе.
`partial_fit` — исключение, потому что это **истинно инкрементальный**
паттерн с O(1)-state (whitelist).

"""
marker_135 = "### 13.5. Открытые вопросы"
s = s.replace(marker_135, episode_25 + marker_135, 1)

# 6.9 §14 резюме
s = s.replace(
    "- **Работает:** `MemoryGuard` (25), `Reduce` (33), `HashShuffle` (13),",
    "- **Работает:** `MemoryGuard` (25), `Reduce` (33), `HashShuffle` (13),",
)
s = s.replace(
    "**v0.1.0 зафиксирован: каркас production-ready. Дальнейшие функции —\nкомпозиции готовых примитивов, не переделка ядра.**",
    "**v0.1.0 зафиксирован: каркас production-ready. Дальнейшие функции —\nкомпозиции готовых примитивов, не переделка ядра.**\n\n"
    "**v0.1.2 (патч 08):** `to_lightgbm` удалён (§3.9). ML — только\n"
    "`partial_fit` для O(1)-state моделей. GBDT — на примитивах (v0.3).",
    1,
)

p.write_text(s)
print("DECISIONS.md updated: §3.9, §11.1/11.4/11.5, §12.5, §13.3/13.4, §14")
PY

# ---------- 7. CHANGELOG.md: v0.1.2 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()

new_entry = """## [0.1.2] — 2026-10-18

### Удалено
- `CadwDataFrame.to_lightgbm` — адаптер к LightGBM. Причина: `lgb.train`
  требует O(N)-state (все градиенты в RAM), OOM происходит внутри
  библиотеки, `MemoryGuard` не спасает. Противоречит §3.9 «no adapters to
  O(N)-algorithms». См. DECISIONS.md §13.4 эпизод 25.
- `tests/test_lightgbm.py` (11 + 1 slow тестов).
- `lightgbm` из `[dependency-groups]` (dev).
- LightGBM-секции из README, DECISIONS §4.1/§11.5/§12.5/§13.3.

### Изменено
- `CadwDataFrame.partial_fit`: добавлен whitelist `_SAFE_PARTIAL_FIT`
  (SGD, StandardScaler, MiniBatchKMeans, IncrementalPCA, NB, MLP).
  Модели вне whitelist → `RuntimeWarning` с указанием на §11.5.
  Docstring обновлён: контракт безопасности (модель должна иметь
  O(1)-state w.r.t. N, это ответственность модели, не CADW).
- `tests/test_api_availability.py`: убрана LightGBM-секция.

### Добавлено в DECISIONS.md
- §3.9: «Никаких адаптеров к O(N)-алгоритмам».
- §11.5: блок «Политика ML» (3 класса: incremental / batch-only /
  own implementation).
- §11.4: обнаружена регрессия `_local_groupby` (materialize до
  `check_state`) — вынесена как must-fix.
- §13.4 эпизод 25: почему адаптер к LightGBM — ошибка.

### Планы
- v0.3: свой GBDT (level-wise, аналог `sklearn.HistGradientBoosting`) на
  примитивах: `HistogramOp` (binning) + `HashShuffle` (узел,фича,бин) +
  `Reduce` (sum_grad/sum_hess/count).

"""

marker = "## [0.1.1]"
if marker in s:
    s = s.replace(marker, new_entry + marker, 1)
else:
    s = s.replace("## [0.1.0]", new_entry + "## [0.1.0]", 1)

p.write_text(s)
print("CHANGELOG.md: v0.1.2 entry added")
PY

# ---------- 8. uv lock + sync (убрать lightgbm) ----------
echo "=== uv lock && uv sync ==="
uv lock
uv sync

# ---------- 9. Проверка, что LightGBM нигде не осталось ----------
echo "=== grep LightGBM (должно быть 0) ==="
if grep -rn "lightgbm\|lgb\." cadw/ tests/ 2>/dev/null | grep -v "test_api_availability"; then
  echo "  WARNING: LightGBM остатки (кроме test_api_availability)"
else
  echo "  OK: 0 совпадений в cadw/ и tests/ (кроме test_api_availability)"
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
git commit -m "08: drop to_lightgbm (O(N) adapter), whitelist partial_fit, ML-policy in DECISIONS (§3.9, §11.5, §13.4/25)"

echo "==> sha256"
sha256sum cadw/dataframe.py tests/test_api_availability.py DECISIONS.md
echo "==> 08 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"