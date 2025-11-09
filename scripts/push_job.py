#!/usr/bin/env python3
"""
Push a single analysis job into the Redis analysis queue.

Usage:
    python scripts/push_job.py /absolute/path/to/audio.wav [--job-id 1]

Environment:
    OFFBEAT_REDIS_URL, OFFBEAT_ANALYSIS_QUEUE (see .env.example)
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
    parser.add_argument("file_path", help="Absolute path to a WAV file for the PoC")
    parser.add_argument("--job-id", type=int, default=1, help="Job ID to use")
    parser.add_argument(
        "--title", type=str, default=None, help="Optional first cue title to include"
    )
    args = parser.parse_args()

    fp = os.path.abspath(args.file_path)
    payload = {"job_id": args.job_id, "file_path": fp}
    if args.title:
        payload["cue_tracks"] = [{"title": args.title, "start_time_sec": 0.0}]

    r = get_client()
    r.lpush(settings.analysis_queue, json.dumps(payload))
    print(
        f"Pushed job to '{settings.analysis_queue}' at {settings.redis_url}:\n" + json.dumps(payload, indent=2)
    )


if __name__ == "__main__":
    main()
