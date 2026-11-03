from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

_LOG_LEVEL = None
# module-level: init_logger runs again for the same name and rebuilds the logger's methods
_LOGGED_ONCE: set[tuple] = set()


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
        LEVEL_MAP = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = level or os.getenv("LOG_LEVEL", "").upper()
        _LOG_LEVEL = LEVEL_MAP.get(level, logging.INFO)

    if strip_file:
        suffix = os.path.basename(suffix)

    if suffix:
        suffix = f"|{suffix}"

    if use_pid is None:
        use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")

    if use_pid:
        pid = os.getpid()
        suffix = f"|pid={pid}{suffix}"

    tp_info = None

    # Color formatter class
    class ColorFormatter(logging.Formatter):
        """Formatter with colors and pretty output"""

        # ANSI color codes
        COLORS = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        RESET = "\033[0m"
        BOLD = "\033[1m"

        def format(self, record):
            from freetoken.distributed import try_get_tp_info

            # Format timestamp like SGLang: [YYYY-MM-DD|HH:MM:SS|pid=1234]
            timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
            nonlocal tp_info
            tp_info = tp_info or try_get_tp_info()
            if tp_info is not None and use_tp_rank is not False:
                real_suffix = f"{suffix}|core|rank={tp_info.rank}"
            else:
                real_suffix = suffix
            timestamp = timestamp.format(suffix=real_suffix)

            # Get color for log level
            level_color = self.COLORS.get(record.levelname, "")

            # Format the message
            colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"
            message = record.getMessage()

            # Pretty format: [timestamp] LEVEL message
            return f"{self.BOLD}{timestamp}{self.RESET} {colored_level} {message}"

    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    def _call_rank0(msg, *args, _which, **kwargs):
        from freetoken.distributed import try_get_tp_info

        nonlocal tp_info
        tp_info = tp_info or try_get_tp_info()
        # No TP set yet (e.g. a unit test or a tool that loads weights without distributed
        # init) -> treat as a single rank (primary) and log, rather than crashing.
        if tp_info is None or tp_info.is_primary():
            getattr(logger, _which)(msg, *args, **kwargs)

    def _call_once(msg, *args, _which, **kwargs):
        key = (name, _which, msg, args)
        if key in _LOGGED_ONCE:
            return
        _LOGGED_ONCE.add(key)
        _call_rank0(msg, *args, _which=_which, **kwargs)

    if TYPE_CHECKING:

        class WrapperLogger(logging.Logger):
            """Custom logger to handle the color formatter."""

            def info_rank0(self, msg, *args, **kwargs): ...
            def warning_rank0(self, msg, *args, **kwargs): ...
            def debug_rank0(self, msg, *args, **kwargs): ...
            def critical_rank0(self, msg, *args, **kwargs): ...
            # rank 0 only, once per distinct (msg, args) in this process; args must hash by value (not exceptions or tensors)
            def debug_once(self, msg, *args, **kwargs): ...
            def info_once(self, msg, *args, **kwargs): ...
            def warning_once(self, msg, *args, **kwargs): ...

        return WrapperLogger(name)
    else:
        logger.info_rank0 = partial(_call_rank0, _which="info")
        logger.debug_rank0 = partial(_call_rank0, _which="debug")
        logger.critical_rank0 = partial(_call_rank0, _which="critical")
        logger.warning_rank0 = partial(_call_rank0, _which="warning")
        logger.debug_once = partial(_call_once, _which="debug")
        logger.info_once = partial(_call_once, _which="info")
        logger.warning_once = partial(_call_once, _which="warning")
        return logger
