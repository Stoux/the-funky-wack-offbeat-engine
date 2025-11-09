"""Quick Redis connectivity self-check.

Usage:
    python scripts/redis_check.py

This will connect to the Redis instance from OFFBEAT_REDIS_URL and perform a PING.
Exits with code 0 on success, non-zero on failure.
"""
from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from offbeat.redis_client import get_client
from offbeat.config import settings


def main() -> int:
    try:
        r = get_client()
        pong = r.ping()
        if pong:
            print(f"Redis OK at {settings.redis_url}")
            return 0
        print("Redis ping returned falsy value")
        return 2
    except Exception as e:
        print(f"Redis check failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
