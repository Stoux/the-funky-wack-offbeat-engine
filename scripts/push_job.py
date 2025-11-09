#!/usr/bin/env python3
"""
Push a single analysis job into the Redis analysis queue.

Usage:
    # Preferred (v2): pass a path relative to the shared mount
    python scripts/push_job.py user-uploads/liveset.wav [--job-id 1]

    # Legacy: absolute path (will be converted to relative if under shared mount)
    python scripts/push_job.py /absolute/path/to/audio.wav [--job-id 1]

Environment:
    OFFBEAT_REDIS_URL, OFFBEAT_ANALYSIS_QUEUE (see .env)
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
    parser.add_argument("path", help="Relative path under shared mount (v2) or absolute path (legacy)")
    parser.add_argument("--job-id", type=int, default=1, help="Job ID to use")
    parser.add_argument(
        "--title", type=str, default=None, help="Optional first cue title to include"
    )
    args = parser.parse_args()

    raw = args.path
    payload = {"job_id": args.job_id}

    # Decide whether to send relative_path (preferred) or fall back to file_path
    if os.path.isabs(raw):
        abs_path = os.path.abspath(raw)
        # If under the shared mount, convert to relative
        try:
            sm = os.path.abspath(settings.shared_mount_path)
        except Exception:
            sm = "/mnt/audio-storage"
        if abs_path.startswith(sm.rstrip("/") + os.sep):
            rel = os.path.relpath(abs_path, sm)
            payload["relative_path"] = rel
        else:
            # Outside shared mount: keep legacy absolute path for dev convenience
            payload["file_path"] = abs_path
    else:
        # Treat as relative path under the shared mount
        payload["relative_path"] = raw

    if args.title:
        payload["cue_tracks"] = [{"title": args.title, "start_time_sec": 0.0}]

    r = get_client()
    r.lpush(settings.analysis_queue, json.dumps(payload))
    print(
        f"Pushed job to '{settings.analysis_queue}' at {settings.redis_url}:\n" + json.dumps(payload, indent=2)
    )


if __name__ == "__main__":
    main()
