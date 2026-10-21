#!/usr/bin/env python3
# v0.9.16 — DECISIONS.md §13.4/211–225 append + release tag v0.9.14.
#
# Scope:
#   (1) DECISIONS.md: append episodes 211–225 before `### 13.5.
#       Открытые вопросы` — 15 meta-lessons from v0.9.7–v0.9.15.
#       Code-comments (§13.4/211, 217, 218, 220, 221, 223, 225)
#       already reference these — DECISIONS.md lags.
#       Numbers 178–210 — gap (planned v0.9.7–v0.9.13 lessons,
#       rolled into 211–225 consolidated set). Backlog: audit
#       v0.9.7–v0.9.13 patches for missing distinct episodes.
#   (2) Tag v0.9.14 — annotated release tag on v0.9.16 commit
#       (contains __version__="0.9.14" + CHANGELOG 0.9.14 section
#       + all v0.9.7–v0.9.16 changes).
#
# Deferred to v0.9.17:
#   - A1: dataframe.py split (5404 lines → 7 modules, ~4h, risky).
#   - Pingis formal round 3 (fresh bench + snapshot).
#   - §11.5 backlog: N6, R3 (design limitation, not bugs).
#
# Run: uv run python scripts/patch_v0916.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (1) DECISIONS.md §13.4/211–225 append ============

DECISIONS_APPEND_LINES = [
    "",
    "**Эпизод 211 (compile-before-write в apply_or_fail).** "
    "v0.9.10-fix3 (T2.1): `apply_or_fail` делал "
    "`p.write_text(s.replace(old, new))`, затем `assert_compiles(p)`. "
    "Если `new_s` синтаксически некорректен (например, bad anchor "
    "оставил невалидный indent) — файл уже на диске. Compile падает, "
    "файл corrupt, `SystemExit`.",
    "",
    "**Правило (§13.4/211):** любой write-helper с post-write "
    "валидацией должен компилировать in-memory **до** `write_text`. "
    "На `SyntaxError` — `SystemExit`, файл нетронут.",
    "",
    "**Связанные:** §13.4/221 (format-aware validation), /222 "
    "(multi-step abort).",
    "",
    "",
    "**Эпизод 212 (pre-import restore order).** v0.9.10-fix3: "
    "patch-script падал на `import cadw.patch_utils` — "
    "`cadw/__init__.py` тащит `dataframe.py`, который был corrupt "
    "(IndentationError). Восстановление (`git checkout`) было в "
    "`main()`, после import.",
    "",
    "**Правило (§13.4/212):** повреждённый module state "
    "восстанавливается **до** первого `import` этого модуля или "
    "его importing chain. Import errors — bytecode-level, не "
    "ловится try/except выше.",
    "",
    "**Связанные:** §13.4/211.",
    "",
    "",
    "**Эпизод 213 (negative substring → AST).** v0.9.10-fix4: "
    "dogfood `assert \"combine_chunks\" not in src_concat` падал на "
    "комментарии `# vs from_batches + combine_chunks + to_batches`. "
    "Класс: negative substring check не различает код и комментарий.",
    "",
    "**Правило (§13.4/213):** negative checks (`\"X\" not in src`) "
    "должны быть AST: `ast.walk(tree)` + `ast.Call` node filter. "
    "Комментарии и docstring invisible для `ast.Call`.",
    "",
    "**Связанные:** §13.4/161 (substring without context), /217 "
    "(precise AST), /206 (anchor exact fragment of new).",
    "",
    "",
    "**Эпизод 214 (idempotency check misfires).** v0.9.10-fix5: "
    "rollback-test в `apply_or_fail` использовал `old=\"    x = 1\\n\"`, "
    "`new=\"x = 1\\n\"`. `new in s` — True (substring), idempotency "
    "path возвращает False, не пишет. Тест ожидал SystemExit, "
    "получил False.",
    "",
    "**Правило (§13.4/214):** test data для `apply_or_fail` "
    "rollback **не должен** иметь `new` как substring `old` (или "
    "vice versa). Реальные патчи безопасны (`new` — long "
    "marker-tagged block, `old` — short source).",
    "",
    "**Связанные:** §13.4/211.",
    "",
    "",
    "**Эпизод 215 (dedent before ast.parse).** v0.9.11-fix1: "
    "dogfood `ast.parse(inspect.getsource(CadwDataFrame._per_group_apply))` "
    "→ `IndentationError: unexpected indent`. "
    "`inspect.getsource` на class method возвращает source с "
    "class-level indent; `ast.parse` требует zero-indent.",
    "",
    "**Правило (§13.4/215):** `textwrap.dedent(inspect.getsource(fn))` "
    "перед `ast.parse` / `compile` / top-level temp-file write. "
    "Three recidivisms session: v0.9.10-fix3 (temp-file), "
    "v0.9.11-fix1 (dogfood ast.parse).",
    "",
    "**Связанные:** §13.4/216, /211.",
    "",
    "",
    "**Эпизод 216 (existing helper exists — use it).** v0.9.11-fix1: "
    "`patch_utils.parse_method_source` (v0.9.6e2 §13.4/191) уже "
    "делал `_ast.parse(_tw.dedent(getsource(fn)))`. Dogfood "
    "изобрёл inline raw `_ast.parse(src)` с тем же intent — и "
    "получил баг. Second recurrence: v0.9.10-fix1 (apply_or_fail "
    "exists, но `c1_fix` использовал inline).",
    "",
    "**Правило (§13.4/216):** если helper для класса операций "
    "существует — **использовать**, не писать заново. Meta-guard: "
    "`test_no_inline_dedent_getsource_outside_helper` (v0.9.15 "
    "T2.7).",
    "",
    "**Связанные:** §13.4/210 (helper exists, use it), /225 "
    "(guard scope).",
    "",
    "",
    "**Эпизод 217 (precise AST anti-pattern).** v0.9.12-fix1: "
    "dogfood `assert not as_py_calls` упал на новом "
    "`_detect_heavy_keys` — в нём легитимно есть **один** "
    "`.as_py()` (aggregate `int(pc.sum(...).as_py())`). "
    "F12-4 bug был про **per-row** `.as_py()` в "
    "`for i in range(num_rows)`, не про aggregate.",
    "",
    "**Правило (§13.4/217):** negative AST guard должен "
    "специфицировать (a) что искать, (b) где именно (receiver "
    "type, nesting), а не только имя метода. Для F12-4: fail on "
    "`X[i].as_py()` (Subscript receiver), allow "
    "`fn(...).as_py()` (Call receiver).",
    "",
    "**Связанные:** §13.4/213 (negative → AST), /216 (helper).",
    "",
    "",
    "**Эпизод 218 (Kleene null в boolean mask).** v0.9.12-fix2: "
    "`_compute_heavy_mask` строил mask через "
    "`pc.equal(null, non_null)` = null (Kleene logic). "
    "`batch.filter(null)` treats null как false → null-key rows "
    "дропаются из **heavy_batch**. `invert(null)=null` → "
    "также дропаются из **light_batch**. Итого: **silent data "
    "loss** в skew path `groupby` — все null-key rows теряются. "
    "v0.3.4 → v0.9.12-fix2, ~3 месяца в проде, ни одного теста.",
    "",
    "**Правило (§13.4/218):** Kleene-logic (null-propagating) "
    "arrow predicates над nullable input требуют **явного** "
    "`fill_null(default)` перед `filter`. Null в boolean mask = "
    "«unknown» = filter drops = silent data loss. Каждый "
    "`pc.equal/and_/or_/not_equal` над nullable input → "
    "terminate with `fill_null(default)`.",
    "",
    "**Связанные:** §13.4/223 (order in isinstance), /217.",
    "",
    "",
    "**Эпизод 219 (caller-bounded rationale needs caller-count "
    "guard).** v0.9.12-fix3: `_detect_heavy_keys` whitelisted "
    "как «sample_batches bounded by groupby_aggregate». Если "
    "появится второй caller с unbounded input — whitelist "
    "становится ложью. Meta-test "
    "`test_detect_heavy_keys_single_caller` (AST: exactly 1 "
    "call site).",
    "",
    "**Правило (§13.4/219):** whitelist с caller-bounded "
    "rationale требует caller-count guard. Иначе через полгода "
    "кто-то добавит второй caller — whitelist останется, станет "
    "ложью.",
    "",
    "**Связанные:** §13.4/170 (whitelist by qualname).",
    "",
    "",
    "**Эпизод 220 (shadowed module import).** v0.9.13-fix1: "
    "`from cadw.ops import quantile as q_mod` даёт **функцию** "
    "`quantile` (ре-экспортирована в `cadw/ops/__init__.py`), не "
    "submodule. `q_mod._stream_values_at_positions` → "
    "AttributeError: 'function' object has no attribute. Python "
    "`from pkg import name` предпочитает attribute над "
    "submodule в `sys.modules[pkg.name]`.",
    "",
    "**Правило (§13.4/220):** если тесту/dogfood нужен **модуль** "
    "для introspection — direct submodule import: "
    "`from pkg.name import symbol`. Shadowed names в CADW: "
    "`reduce`, `quantile` (functions re-exported с same-name "
    "submodules). Meta-guard: "
    "`test_no_shadowed_module_import_in_tests`.",
    "",
    "**Связанные:** §13.4/216, /225.",
    "",
    "",
    "**Эпизод 221 (compile-before-write assumes Python target).** "
    "v0.9.14-fix1: `apply_or_fail` падал на "
    "`pyproject.toml version 0.1.0→0.9.14` — "
    "`compile(new_s, path, 'exec')` на TOML → "
    "`SyntaxError: cannot assign to expression`. "
    "§13.4/211 assumes Python target.",
    "",
    "**Правило (§13.4/221):** format-aware validation. "
    "`apply_or_fail(validate=None|'python'|'toml'|'json'|'none')`, "
    "default inferred from extension. `.py`/`.pyi` → python "
    "compile; `.toml` → `tomllib.loads`; `.json` → "
    "`json.loads`; остальное → none.",
    "",
    "**Связанные:** §13.4/211.",
    "",
    "",
    "**Эпизод 222 (multi-step patch abort leaves partial state).** "
    "v0.9.14-fix2: v0.9.14 (initial) died at U3 (compile step). "
    "Order: F12-6a (OK) → F12-6b (OK) → F12-7 (OK) → U3 (DIED) "
    "→ T2.2 (never) → tests (never) → commit (never). Partial "
    "state on disk. v0.9.14-fix1 completed U3 + validate= + T2.9 "
    "tests, но T2.2 + F12-6/T2.2 tests остались. fix2 доделал.",
    "",
    "**Правило (§13.4/222):** multi-step patch abort оставляет "
    "state в частично применённом виде. Verify_state должен "
    "проверять **все** anchors (idempotent-safe); fix-scripts "
    "видят, что осталось. Критические шаги — первыми (U3 до "
    "tests), декларативные — последними.",
    "",
    "**Связанные:** §13.4/211, /212.",
    "",
    "",
    "**Эпизод 223 (bool before int in isinstance chain).** "
    "v0.9.14-fix3: `_infer_arrow_type_from_value` проверял "
    "`isinstance(v, int)` **до** `bool`. Python `bool` — subclass "
    "of `int`: `isinstance(True, int)` = True. `True` inferred "
    "как int64 → `from_pylist([{'x': True}], schema=int64)` → "
    "`ArrowTypeError: Expected integer, got bool`. F12-6a "
    "bool-widen branch was **dead code** — schema никогда не "
    "был pa.bool_. Mixed bool/float «работал» случайно "
    "(True→int64, 2.5→int branch widen→float64); all-bool → "
    "ArrowTypeError.",
    "",
    "**Правило (§13.4/223):** в type-inference chains / "
    "dispatchers / validators — **bool BEFORE int**. Python "
    "`bool` = int subclass; `np.bool_` — не subclass of "
    "`np.integer` (modern numpy), но Python bool IS subclass of "
    "int. Порядок: None → bool/np.bool_ → int → float → str.",
    "",
    "**Связанные:** §13.4/218 (Kleene null), /175 (formula scope).",
    "",
    "",
    "**Эпизод 224 (version markers in new code blocks).** "
    "v0.9.15-fix1: `dedent_method_source` (v0.9.15 T2.6) в "
    "docstring ссылается на §13.4/215/216, но **без** "
    "`v0.9.15` marker. verify_state anchor `(\"cadw/patch_utils.py\", "
    "\"v0.9.15\")` фейлится. v0.9.7–v0.9.14 патчи следовали "
    "паттерну — `vX.Y.Z` в каждом новом block. v0.9.15 нарушил.",
    "",
    "**Правило (§13.4/224):** каждый новый/изменённый code "
    "block (docstring или comment) должен содержать version "
    "marker `vX.Y.Z (Tn)`. Даёт: grep-tracking, verify anchor, "
    "blame annotation. Verify требует — код должен содержать.",
    "",
    "**Связанные:** §13.4/206 (anchor exact fragment of new), "
    "/161.",
    "",
    "",
    "**Эпизод 225 (static guard scope = «code that will run "
    "again»).** v0.9.15-fix2: T2.7 meta-guard "
    "`test_no_inline_dedent_getsource_outside_helper` сканировал "
    "`[cadw/, scripts/]` → 13 false-positive violations в "
    "historical patch-scripts (v0.9.6–v0.9.14) + "
    "`SyntaxWarning: invalid escape sequence '\\(`'` из "
    "`_ast.parse` на их docstring'ах.",
    "",
    "**Правило (§13.4/225):** static guards для "
    "«reinvented-helper» паттернов scope'ятся к **production "
    "code** (`cadw/`), не к historical artifacts (`scripts/`). "
    "Ephemeral one-shot scripts, которые отработали — вне "
    "forward-looking conventions. Convention для будущих "
    "patch-scripts — helper docstring + code review. Плюс: "
    "`_ast.parse` на неотфильтрованных файлах требует "
    "`warnings.catch_warnings() + simplefilter('ignore', "
    "SyntaxWarning)` (Python 3.12+).",
    "",
    "**Связанные:** §13.4/216 (existing helper, use it), /213 "
    "(negative → AST).",
    "",
    "",
]


def decisions_append():
    print("\n=== (1) DECISIONS.md: §13.4/211–225 append ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "**Эпизод 211 (compile-before-write" in s:
        print("  already present — skip")
        return
    anchor = "### 13.5. Открытые вопросы"
    if anchor not in s:
        raise SystemExit("DECISIONS.md anchor '### 13.5. Открытые вопросы' not found")
    block = "\n".join(DECISIONS_APPEND_LINES) + "\n"
    # Insert BEFORE §13.5 (after §13.4 ends)
    new_block = block + anchor
    s = s.replace(anchor, new_block, 1)
    p.write_text(s, newline="")
    # Sanity: order 13.4 < 211-225 < 13.5
    assert "**Эпизод 211 (compile-before-write" in s
    assert "**Эпизод 225 (static guard scope" in s
    assert s.index("Эпизод 211") < s.index("Эпизод 225")
    assert s.index("Эпизод 225") < s.index("### 13.5. Открытые вопросы")
    print("  OK (episodes 211–225 inserted before §13.5)")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.16 + regression ===")
    pu.verify_anchors([
        # v0.9.16 (DECISIONS append)
        ("DECISIONS.md", "**Эпизод 211 (compile-before-write"),
        ("DECISIONS.md", "**Эпизод 217 (precise AST anti-pattern)"),
        ("DECISIONS.md", "**Эпизод 218 (Kleene null в boolean mask)"),
        ("DECISIONS.md", "**Эпизод 220 (shadowed module import)"),
        ("DECISIONS.md", "**Эпизод 221 (compile-before-write assumes"),
        ("DECISIONS.md", "**Эпизод 222 (multi-step patch abort"),
        ("DECISIONS.md", "**Эпизод 223 (bool before int"),
        ("DECISIONS.md", "**Эпизод 224 (version markers"),
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
        # v0.9.15 regression
        ("cadw/patch_utils.py", "v0.9.15 (T2.6)"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
        ("cadw/__init__.py", '__version__ = "0.9.14"'),
        ("tests/test_patch_utils.py", "roots = [_P('cadw')]"),
        # v0.9.14-fix3
        ("cadw/dataframe.py", "v0.9.14-fix3 (§13.4/223)"),
        ("tests/test_apply.py", "test_apply_all_bool_stays_bool"),
        # v0.9.14
        ("cadw/dataframe.py", "v0.9.14 (F12-6a)"),
        ("cadw/patch_utils.py", "v0.9.14 (T2.2, §13.4/217)"),
        ("cadw/patch_utils.py", "v0.9.14-fix1 (T2.9, §13.4/221)"),
        ("pyproject.toml", 'version = "0.9.14"'),
        # v0.9.13/12/11/10/9
        ("cadw/ops/quantile.py", "v0.9.13 (F12-5, S3 perf)"),
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/patch_utils.py", "def apply_or_fail("),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.16 via subprocess ===")

    def _verify_all():
        from pathlib import Path as _P

        # DECISIONS append: all 15 episodes present + §13.5 intact
        d = _P("DECISIONS.md").read_text()
        for n in range(211, 226):
            assert f"**Эпизод {n} " in d, (
                f"DECISIONS.md missing episode {n}"
            )
        # Order: 211 < 225 < §13.5
        assert d.index("**Эпизод 211 ") < d.index("**Эпизод 225 ")
        assert d.index("**Эпизод 225 ") < d.index("### 13.5. Открытые вопросы")
        # §13.4 was before §13.5 originally — 211 follows §13.4 end
        assert d.index("### 13.4") < d.index("**Эпизод 211 ")

        # v0.9.15 regression — CHANGELOG, version
        import cadw
        assert cadw.__version__ == "0.9.14"
        cl = _P("CHANGELOG.md").read_text()
        assert "## [0.9.14] — 2026-10-21" in cl

        # v0.9.15 T2.7 scope
        t27 = _P("tests/test_patch_utils.py").read_text()
        assert "roots = [_P('cadw')]" in t27
        assert "roots = [_P('cadw'), _P('scripts')]" not in t27

        print("  DECISIONS 211–225 all present, ordered before §13.5 OK")
        print("  CHANGELOG 0.9.14 section intact OK")
        print("  __version__ = 0.9.14 OK")
        print("  T2.7 scope cadw/ only intact OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_changelog.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=900,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ Tag v0.9.14 ============

def tag_release():
    print("\n=== (2) Tag v0.9.14 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--verify", "refs/tags/v0.9.14"],
        capture_output=True,
    )
    if r.returncode == 0:
        print("  tag v0.9.14 already exists — skip")
        return
    subprocess.run(
        [
            "git", "tag", "-a", "v0.9.14",
            "-m",
            "v0.9.14: audit round 2 close. 21 original findings "
            "(R1-R6, B1-B4, N1-N7) + 13 discovered, 3 silent "
            "data-loss bugs fixed (C1 int truncation, §13.4/218 "
            "null-key drop 3 months in prod, §13.4/223 bool→int "
            "misinference), 15 new §13.4 meta-lessons (211-225). "
            "16 patches v0.9.7-v0.9.16.",
        ],
        check=True,
    )
    print("  tag v0.9.14 created")
    # Verify
    r = subprocess.run(
        ["git", "tag", "-n", "1", "v0.9.14"],
        capture_output=True, text=True, check=True,
    )
    print(f"  {r.stdout.strip()}")


# ============ main ============

def main():
    print("=== v0.9.16 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    decisions_append()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.16: DECISIONS.md §13.4/211–225 append (15 new "
        "meta-lessons from v0.9.7-v0.9.15) + release tag v0.9.14. "
        "DECISIONS.md on disk was at v0.9.6 baseline (last episode "
        "#177); v0.9.7-v0.9.15 code-comments referenced §13.4/211, "
        "217, 218, 220, 221, 223, 225 but episodes were not "
        "documented. Append 15 consolidated meta-lessons before "
        "§13.5: 211 (compile-before-write), 212 (pre-import "
        "restore order), 213 (negative → AST), 214 (idempotency "
        "check misfires), 215 (dedent before ast.parse), 216 "
        "(existing helper exists — use it), 217 (precise AST "
        "anti-pattern), 218 (Kleene null → fill_null before "
        "filter — silent data-loss 3 months in prod), 219 "
        "(caller-bounded rationale needs caller-count guard), "
        "220 (shadowed module import), 221 (compile-before-write "
        "assumes Python target — format-aware validate=), 222 "
        "(multi-step abort leaves partial state), 223 (bool "
        "before int in isinstance chain), 224 (version markers "
        "in new code blocks), 225 (static guard scope = 'code "
        "that will run again'). Numbers 178–210 — gap (planned "
        "v0.9.7-v0.9.13 lessons, rolled into 211-225 "
        "consolidated set; backlog: audit patches for missing "
        "distinct episodes). Tag v0.9.14 annotated on v0.9.16 "
        "commit (contains __version__='0.9.14' + CHANGELOG "
        "0.9.14 section + all v0.9.7-v0.9.16 changes).",
        self_path="scripts/patch_v0916.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.16 applied.")

    # Tag AFTER commit — tag points to release commit
    tag_release()


if __name__ == "__main__":
    main()
