# DECISIONS.md — Index

> **Full decisions split into `docs/decisions/*.md` for context-transfer.**
> Original monolithic 288 KB / 5645 lines → 24 per-topic files (target ≤ 800 lines each).

## Reading guide

| # | File | Topic | Lines |
|---|---|---|---|
| 00 | [00_decisionsmd_архитектурные_решения_cadw.md](docs/decisions/00_decisionsmd_архитектурные_решения_cadw.md) | DECISIONS.md — Архитектурные решения CADW | 11 |
| 01 | [01_0_метаданные_и_статус.md](docs/decisions/01_0_метаданные_и_статус.md) | 0. Метаданные и статус | 19 |
| 02 | [02_1_что_такое_cadw_и_для_кого.md](docs/decisions/02_1_что_такое_cadw_и_для_кого.md) | 1. Что такое CADW и для кого | 26 |
| 03 | [03_2_глоссарий.md](docs/decisions/03_2_глоссарий.md) | 2. Глоссарий | 20 |
| 04 | [04_3_ключевые_принципы_обязательные.md](docs/decisions/04_3_ключевые_принципы_обязательные.md) | 3. Ключевые принципы (обязательные) | 55 |
| 05 | [05_4_архитектура_уровни_и_модули.md](docs/decisions/05_4_архитектура_уровни_и_модули.md) | 4. Архитектура: уровни и модули | 64 |
| 06 | [06_5_примитив_1_reduce.md](docs/decisions/06_5_примитив_1_reduce.md) | 5. Примитив 1: `Reduce` | 66 |
| 07 | [07_6_примитив_2_hashshuffle.md](docs/decisions/07_6_примитив_2_hashshuffle.md) | 6. Примитив 2: `HashShuffle` | 61 |
| 08 | [08_7_примитив_3_sort.md](docs/decisions/08_7_примитив_3_sort.md) | 7. Примитив 3: `Sort` | 36 |
| 09 | [09_8_почему_три_примитива_а_не_n.md](docs/decisions/09_8_почему_три_примитива_а_не_n.md) | 8. Почему три примитива, а не N | 22 |
| 10 | [10_9_memoryguard_планировщик_памяти.md](docs/decisions/10_9_memoryguard_планировщик_памяти.md) | 9. `MemoryGuard`: планировщик памяти | 635 |
| 11 | [11_10_dataset_rowfilter_cadwdataframe_реали.md](docs/decisions/11_10_dataset_rowfilter_cadwdataframe_реали.md) | 10. Dataset, RowFilter, CadwDataFrame (реализованы) | 59 |
| 12 | [12_11_степень_реализации.md](docs/decisions/12_11_степень_реализации.md) | 11. Степень реализации | 443 |
| 13 | [13_12_дорожная_карта.md](docs/decisions/13_12_дорожная_карта.md) | 12. Дорожная карта | 33 |
| 14 | [14_13_история_обсуждения_и_уроки_131_почему.md](docs/decisions/14_13_история_обсуждения_и_уроки_131_почему.md) | 13. История обсуждения и уроки | 39 |
| 15a | [15a_13_4_уроки_part1.md](docs/decisions/15a_13_4_уроки_part1.md) | 13.4. Уроки — часть 1/4 | 525 |
| 15b | [15b_13_4_уроки_part2.md](docs/decisions/15b_13_4_уроки_part2.md) | 13.4. Уроки — часть 2/4 | 530 |
| 15c | [15c_13_4_уроки_part3.md](docs/decisions/15c_13_4_уроки_part3.md) | 13.4. Уроки — часть 3/4 | 530 |
| 15d | [15d_13_4_уроки_part4.md](docs/decisions/15d_13_4_уроки_part4.md) | 13.4. Уроки — часть 4/4 | 532 |
| 16 | [16_пример.md](docs/decisions/16_пример.md) | Пример: | 565 |
| 17 | [17_пример.md](docs/decisions/17_пример.md) | generic). | 565 |
| 18 | [18_пример.md](docs/decisions/18_пример.md) |  | 565 |
| 19 | [19_14_резюме_для_восстановления_контекста.md](docs/decisions/19_14_резюме_для_восстановления_контекста.md) | 14. Резюме для восстановления контекста | 36 |
| 20 | [20_15_time_series_v0924.md](docs/decisions/20_15_time_series_v0924.md) | §15. Time series (v0.9.24) | 262 |

## §13.4 Lessons — cross-reference

§13.4 lessons live in `14_*` (13.1), `15a-d_*` (§13.4 parts). Consolidated indices per phase within 15x files.

```bash
grep -rn "§13.4/274" docs/decisions/
```

## Common queries

| Need | Read |
|---|---|
| Intro / principles / architecture | `00-05_*.md` |
| Reduce / Shuffle / Sort primitives | `06-09_*.md` |
| MemoryGuard / OOM / formulas | `10_9_memoryguard*.md` |
| DataFrame status / roadmap | `11-13_*.md` |
| §13.4/NNN lesson | `14_*`, `15a-d_*.md` |
| TS features (§15) | `20_15_time_series*.md` |

## Why split

v0.9.28 B.2 — Phase B goal is *remove context-transfer pain*. DECISIONS.md at 288 KB alone was 1/3 of typical context window. Now each topic loads independently.

## Legacy

Full original preserved in git:
```bash
git show v0.9.27:DECISIONS.md | wc -l  # 5645 lines
```
