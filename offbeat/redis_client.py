import json
from typing import Any, Tuple, Optional
import redis
from loguru import logger
from .config import settings


def get_client() -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def pop_job_blocking(timeout: int = 5) -> Optional[Tuple[str, Any]]:
    """Pop a job from the analysis queue, blocking with timeout.

    Returns (queue_name, parsed_json) or None on timeout.
    """
    r = get_client()
    result = r.brpop(settings.analysis_queue, timeout=timeout)
    if not result:
        return None
    q, data = result
    try:
        payload = json.loads(data)
    except Exception as e:
        logger.error(f"Malformed job JSON from queue {q}: {e}")
        return q, None
    return q, payload


def push_result(result: dict) -> None:
    r = get_client()
    r.lpush(settings.results_queue, json.dumps(result))
