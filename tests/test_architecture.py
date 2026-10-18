"""Архитектурные guard-тесты (патч 05: legacy удалён)."""
import pathlib

import cadw


def test_no_legacy_package():
    """cadw._legacy удалён; не должно быть попыток его импортировать."""
    root = pathlib.Path(cadw.__file__).parent
    assert not (root / "_legacy").exists(), "cadw/_legacy/ must not exist"


def test_main_code_does_not_reference_legacy():
    """main-код (cadw/*.py) не содержит 'cadw._legacy' или '_legacy.'."""
    root = pathlib.Path(cadw.__file__).parent
    offenders: list[str] = []
    for p in root.rglob("*.py"):
        if "_legacy" in str(p.relative_to(root)):
            continue
        src = p.read_text(encoding="utf-8")
        if "cadw._legacy" in src or "from ._legacy" in src:
            offenders.append(str(p.relative_to(root)))
    assert not offenders, f"main code references _legacy: {offenders}"


def test_legacy_tests_removed():
    """tests/_legacy/ удалён вместе с legacy."""
    tests_root = pathlib.Path(__file__).parent
    assert not (tests_root / "_legacy").exists(), "tests/_legacy/ must not exist"
