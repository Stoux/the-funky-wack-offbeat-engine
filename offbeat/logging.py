from loguru import logger
import os
import sys


def setup_logging():
    """Configure loguru with sane defaults.

    - Logs to stdout by default.
    - Log level controlled via OFFBEAT_LOG_LEVEL (default: INFO)
    - Formats in a readable, timestamped pattern.
    """
    level = os.getenv("OFFBEAT_LOG_LEVEL", "INFO")

    # Remove default handlers to avoid duplicate logs on re-init
    logger.remove()

    logger.add(
        sys.stdout,
        level=level,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {process} | {name}:{function}:{line} | {message}",
    )

    return logger
