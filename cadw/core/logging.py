# cadw/core/logging.py
"""Единая настройка логирования CADW."""
import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Возвращает логгер с префиксом cadw."""
    return logging.getLogger(f"cadw.{name}")


def configure(level: int = logging.INFO) -> None:
    """Настраивает корневой логгер cadw. Вызывать один раз при старте."""
    root = logging.getLogger("cadw")
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)