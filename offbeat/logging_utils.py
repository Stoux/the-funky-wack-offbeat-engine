import logging
import os
import sys
import time
from typing import Optional, Dict

DEFAULT_LOG_LEVEL = os.getenv("OFFBEAT_LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_HZ = float(os.getenv("OFFBEAT_LOG_MAX_HZ", "1"))  # max lines per key per second
DEFAULT_EVERY_N = int(os.getenv("OFFBEAT_LOG_EVERY_N", "100"))


class RateLimiter:
    def __init__(self, max_hz: float):
        self.min_interval = 1.0 / max_hz if max_hz > 0 else 0.0
        self._last: Dict[str, float] = {}

    def allow(self, key: str) -> bool:
        if self.min_interval <= 0:
            return True
        now = time.time()
        last = self._last.get(key, 0.0)
        if now - last >= self.min_interval:
            self._last[key] = now
            return True
        return False


class ProgressLogger(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger, job_id: Optional[str] = None, max_hz: float = DEFAULT_LOG_HZ):
        super().__init__(logger, extra={"job_id": job_id})
        self.rate = RateLimiter(max_hz=max_hz)

    def info_throttled(self, key: str, msg: str, *args, **kwargs):
        if self.rate.allow(key):
            self.info(msg, *args, **kwargs)

    def every_n(self, n: int, i: int, msg: str, *args, **kwargs):
        if n > 0 and i % n == 0:
            self.info(msg, *args, **kwargs)

    def every_n_throttled(self, key: str, n: int, i: int, msg: str, *args, **kwargs):
        if n > 0 and i % n == 0 and self.rate.allow(key):
            self.info(msg, *args, **kwargs)


def _plain_formatter() -> logging.Formatter:
    class _Adapter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            job_id = getattr(record, "job_id", None)
            record.job = f"job_id={job_id} " if job_id else ""

            # Safe formatting supporting both `{}` and `%` styles.
            msg = record.msg
            args = record.args if isinstance(record.args, tuple) else ()

            formatted = None
            # 1) Try brace-style formatting if it looks like a `{}` template
            if isinstance(msg, str) and ("{" in msg and "}" in msg):
                try:
                    formatted = msg.format(*args)
                except Exception:
                    formatted = None
            # 2) Fallback to percent-style formatting
            if formatted is None:
                try:
                    if args:
                        formatted = str(msg % args)
                    else:
                        formatted = str(msg)
                except Exception:
                    # 3) Last resort: concatenate stringified pieces
                    try:
                        formatted = " ".join([str(msg)] + [str(a) for a in args])
                    except Exception:
                        formatted = str(msg)

            record.message = formatted
            # Important: avoid default logging applying `%` again anywhere downstream
            record.args = ()

            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)) + \
                   f" [{record.levelname}] {record.name} {record.job}{record.message}"
    return _Adapter()


def setup_logging(level: str = DEFAULT_LOG_LEVEL) -> logging.Logger:
    logger = logging.getLogger("offbeat")
    if logger.handlers:
        return logger
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_plain_formatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(module_name: str, job_id: Optional[str] = None, max_hz: Optional[float] = None) -> ProgressLogger:
    base = setup_logging()
    child = logging.getLogger(f"offbeat.{module_name}")
    child.setLevel(base.level)
    return ProgressLogger(child, job_id=job_id, max_hz=(max_hz if max_hz is not None else DEFAULT_LOG_HZ))
