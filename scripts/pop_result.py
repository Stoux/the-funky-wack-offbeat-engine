#!/usr/bin/env python3
"""
Pop a single analysis result from the Redis results queue and print it.

Usage:
    python scripts/pop_result.py [--timeout 10]

Environment:
    OFFBEAT_REDIS_URL, OFFBEAT_RESULTS_QUEUE (see .env.example)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from offbeat.config import settings
from offbeat.redis_client import get_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=10, help="BRPOP timeout seconds")
    args = parser.parse_args()

    r = get_client()
    result = r.brpop(settings.results_queue, timeout=args.timeout)
    if not result:
        print(f"No result within {args.timeout}s from '{settings.results_queue}'")
        return
    q, data = result
    try:
        payload = json.loads(data)
    except Exception:
        payload = data
    print(json.dumps(payload, indent=2) if isinstance(payload, dict) else str(payload))


if __name__ == "__main__":
    main()
