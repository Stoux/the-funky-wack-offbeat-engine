"""Per-track analysis stub.

Real Spleeter/librosa processing will be added later. The stubs below expose
signatures used by the future worker orchestration.
"""
from typing import Dict, Any


def init_spleeter():
    # No-op in scaffolding phase
    return None


def teardown_spleeter():
    # No-op in scaffolding phase
    return None


def analyze_track_chunk(ctx, chunk) -> Dict[str, Any]:
    # Minimal placeholder structure mirroring the target schema
    return {
        "track_id": getattr(chunk, "track_id", 0),
        "title": getattr(chunk, "title", "Unknown"),
        "cue_start_time_sec": float(getattr(chunk, "start_sec", 0.0)),
        "transition_period_sec": getattr(chunk, "transition", {"start": None, "end": None, "duration": None}) or {"start": None, "end": None, "duration": None},
        "analysis": {
            "duration_sec": float(max(0.0, getattr(chunk, "end_sec", 0.0) - getattr(chunk, "start_sec", 0.0))),
            "key": None,
            "loudness_lufs_total": None,
            "loudness_lufs_bass": None,
            "average_brightness": None,
            "has_vocals": None,
            "vocal_energy_rms": None,
        },
    }
