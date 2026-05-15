"""
Shared logging setup for MEMIT experiments.
All experiment logs are written to logs/ under the project root.
"""

from datetime import datetime
from pathlib import Path
import logging
from typing import Optional

LOGS_DIR = Path("/data1/D-PIKE/memit-main/memit-main/logs")
_LOGGER_NAME = "memit"
_logger: Optional[logging.Logger] = None
_file_handler: Optional[logging.FileHandler] = None


def _ensure_logs_dir(log_dir: Optional[Path] = None) -> Path:
    """Create logs directory if it does not exist."""
    d = log_dir or LOGS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logging(
    log_dir: Optional[Path] = None,
    log_basename: Optional[str] = None,
) -> logging.Logger:
    """
    Configure the shared MEMIT logger to write to both a timestamped file
    under log_dir and to stdout. Idempotent: safe to call multiple times;
    reconfigures the logger each time (e.g. new run => new log file).

    :param log_dir: Directory for log files. Default: LOGS_DIR.
    :param log_basename: Base name for log file (without .log). 
        Default: evaluate_YYYYMMDD_HHMMSS.
    :return: The configured logger.
    """
    global _logger, _file_handler

    log_dir = log_dir or LOGS_DIR
    _ensure_logs_dir(log_dir)
    if log_basename is None:
        log_basename = f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = log_dir / f"{log_basename}.log"

    log = logging.getLogger(_LOGGER_NAME)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _file_handler = logging.FileHandler(log_path, encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(fmt)
    log.addHandler(_file_handler)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    _logger = log
    log.info("Logging initialized. Log file: %s", log_path)
    return log


def get_logger() -> logging.Logger:
    """Return the shared MEMIT logger. Use setup_logging() first when running evaluate."""
    log = logging.getLogger(_LOGGER_NAME)
    if not log.handlers:
        log.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        log.addHandler(ch)
    return log
