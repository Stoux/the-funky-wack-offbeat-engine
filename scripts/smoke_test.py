"""Smoke test for PoC queue + one processing task.

This script bypasses Redis and exercises the job validation and result assembly
using the current modules. It generates a tiny temporary WAV file so that the
Global Analysis PoC can compute a real duration value, then prints the final JSON.

Usage:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import wave
import struct
import math

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from offbeat.analysis.global_analysis import run_global_analysis
from offbeat.analysis.track_detection import detect_tracks
from offbeat.worker import assemble_result, chunks_to_track_results


def _write_tiny_wav(path: str, seconds: float = 1.0, sr: int = 16000) -> None:
    n_samples = int(seconds * sr)
    freq = 440.0
    amp = 0.3
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        for i in range(n_samples):
            sample = int(max(-1.0, min(1.0, amp * math.sin(2 * math.pi * freq * i / sr))) * 32767)
            wf.writeframes(struct.pack('<h', sample))


def main():
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "tiny.wav")
        _write_tiny_wav(wav_path, seconds=1.0)
        job = {
            "job_id": 1,
            "file_path": wav_path,
            # "cue_tracks": [{"title": "Intro", "start_time_sec": 0.0}],
        }

        global_ctx = run_global_analysis(job)
        chunks = detect_tracks(global_ctx, job)
        tracks = chunks_to_track_results(chunks, global_ctx, job)
        final = assemble_result(job, global_ctx, tracks)
        print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
