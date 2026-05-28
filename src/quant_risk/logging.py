# src/quant_risk/logging.py
"""
Logging helpers for the quant_risk package.

Usage (library code)
--------------------
from quant_risk.logging import get_logger
logger = get_logger(__name__)
logger.info("fetched %d rows from %s", n, source)

Usage (application code -- scripts, entry points)
-------------------------------------------------
from quant_risk.logging import configure_file_logging
configure_file_logging()   # call once at startup

All quant_risk.* loggers write to logs/fund_liquidity.log with automatic
rotation at 5 MB, keeping 3 backup files (20 MB total cap).
"""

import logging
from logging.handlers import RotatingFileHandler
from quant_risk.config import LOGS_DIR

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger for library code.

    Adds a NullHandler so callers never see 'No handler found' warnings.
    The application is responsible for attaching real handlers via
    configure_file_logging().

    Parameters
    ----------
    name : str
        Typically __name__ of the calling module.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def configure_file_logging(
    filename: str = "fund_liquidity.log",
    level: int = logging.DEBUG,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """
    Attach a rotating file handler to the quant_risk package root logger.

    Safe to call multiple times -- skips setup if a FileHandler is already
    attached, so repeated calls at startup do not add duplicate handlers.

    Parameters
    ----------
    filename : str
        Log file name, written under LOGS_DIR (project_root/logs/).
    level : int
        Logging level. Default logging.DEBUG captures both INFO fetch
        events and DEBUG cache hit/miss records.
    max_bytes : int
        Rotate when the file reaches this size. Default 5 MB.
    backup_count : int
        Number of rotated files to keep before deleting oldest.
        Total disk cap = (backup_count + 1) * max_bytes. Default 3.
    """
    root = logging.getLogger("quant_risk")

    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        LOGS_DIR / filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
