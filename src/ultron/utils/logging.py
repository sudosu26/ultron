"""Logging setup for ULTRON.

Logging levels: Minimal, Normal, Debug.
API keys, passwords, and tokens are NEVER logged.
"""

import logging
import sys
from pathlib import Path

from ultron.utils.paths import LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Words that must never appear in log output.
_SENSITIVE_PATTERNS = (
    "api_key", "apikey", "authorization", "bearer", "token",
    "password", "secret", "credential",
)


class SensitiveFilter(logging.Filter):
    """Redact log records that contain sensitive key names."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().lower()
        for pattern in _SENSITIVE_PATTERNS:
            if pattern in message:
                record.msg = "[REDACTED — sensitive content]"
                record.args = ()
                return True
        return True


def setup_logging(level: str = "Normal") -> logging.Logger:
    """Configure ULTRON logging.

    level: 'Minimal', 'Normal', or 'Debug'.
    """
    levels = {
        "Minimal": logging.WARNING,
        "Normal": logging.INFO,
        "Debug": logging.DEBUG,
    }
    numeric_level = levels.get(level, logging.INFO)

    logger = logging.getLogger("ultron")
    logger.setLevel(numeric_level)
    logger.handlers.clear()

    sensitive_filter = SensitiveFilter()

    # Console handler.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    console.addFilter(sensitive_filter)
    logger.addHandler(console)

    # File handler.
    log_file = LOGS_DIR / "ultron.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    file_handler.addFilter(sensitive_filter)
    logger.addHandler(file_handler)

    return logger