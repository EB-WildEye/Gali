"""
Gali - Professional Logging Implementation
============================================
Provides a pre-configured logger factory for consistent, readable
log output across all Gali modules.

Logging verbosity is driven by ``settings.ENV``:
  • ``"dev"``  → DEBUG level  (maximum detail)
  • ``"prod"`` → WARNING level (errors & warnings only)

Usage:
    from gali.src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

import logging
import sys

from gali.config.settings import settings

# ── Resolve default level from environment mode ───────────────────────
_ENV_LEVEL_MAP = {
    "dev": logging.DEBUG,
    "prod": logging.WARNING,
}
_DEFAULT_LEVEL = _ENV_LEVEL_MAP.get(settings.ENV, logging.INFO)


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """
    Create and return a named logger with a standardised formatter.

    Args:
        name:  Typically ``__name__`` of the calling module.
        level: Explicit logging level override.
               If *None*, the level is derived from ``settings.ENV``.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    resolved_level = level if level is not None else _DEFAULT_LEVEL
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger already exists
    if not logger.handlers:
        logger.setLevel(resolved_level)

        # ── Console Handler ────────────────────────────────────────────
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(resolved_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Prevent log propagation to the root logger (avoids duplicates)
        logger.propagate = False

    return logger
