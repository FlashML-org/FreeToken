"""Lightweight logging setup that is safe to import without the CUDA stack."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

_LOG_LEVEL = None


def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool | None = None,
    use_tp_rank: bool | None = None,
):
    """Initialize the logger for the module with colors and pretty formatting."""
    import logging
    import os
    import sys

    global _LOG_LEVEL
    if _LOG_LEVEL is None:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = level or os.getenv("LOG_LEVEL", "").upper()
        _LOG_LEVEL = level_map.get(level, logging.INFO)

    if strip_file:
        suffix = os.path.basename(suffix)
    if suffix:
        suffix = f"|{suffix}"

    if use_pid is None:
        use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")
    if use_pid:
        suffix = f"|pid={os.getpid()}{suffix}"

    tp_info = None

    class ColorFormatter(logging.Formatter):
        """Formatter with colored levels and optional process/rank suffixes."""

        COLORS = {
            "DEBUG": "\033[36m",
            "INFO": "\033[32m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
            "CRITICAL": "\033[35m",
        }
        RESET = "\033[0m"
        BOLD = "\033[1m"

        def format(self, record):
            nonlocal tp_info
            try:
                from freetoken.distributed import try_get_tp_info

                tp_info = tp_info or try_get_tp_info()
            except Exception:  # noqa: BLE001 -- torch may be absent on Metal
                tp_info = None
            timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
            if tp_info is not None and use_tp_rank is not False:
                real_suffix = f"{suffix}|core|rank={tp_info.rank}"
            else:
                real_suffix = suffix
            timestamp = timestamp.format(suffix=real_suffix)
            level_color = self.COLORS.get(record.levelname, "")
            colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"
            return f"{self.BOLD}{timestamp}{self.RESET} {colored_level} {record.getMessage()}"

    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    def _call_rank0(msg, *args, _which, **kwargs):
        from freetoken.distributed import try_get_tp_info

        nonlocal tp_info
        tp_info = tp_info or try_get_tp_info()
        if tp_info is None or tp_info.is_primary():
            getattr(logger, _which)(msg, *args, **kwargs)

    if TYPE_CHECKING:

        class WrapperLogger(logging.Logger):
            def info_rank0(self, msg, *args, **kwargs): ...
            def warning_rank0(self, msg, *args, **kwargs): ...
            def debug_rank0(self, msg, *args, **kwargs): ...
            def critical_rank0(self, msg, *args, **kwargs): ...

        return WrapperLogger(name)

    logger.info_rank0 = partial(_call_rank0, _which="info")
    logger.debug_rank0 = partial(_call_rank0, _which="debug")
    logger.critical_rank0 = partial(_call_rank0, _which="critical")
    logger.warning_rank0 = partial(_call_rank0, _which="warning")
    return logger
