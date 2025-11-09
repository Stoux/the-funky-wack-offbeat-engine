"""Track detection PoC.

For the current PoC we create a single chunk spanning the detectable audio
range so downstream code has a non-zero duration to work with.

- In pure_audio_guess mode: start at trimmed_start_sec (default 0.0), end at duration_sec.
- In cue_correlated mode: start at the first cue start_time_sec if provided, else
  trimmed_start_sec; end at duration_sec.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Chunk:
    track_id: int
    title: str
    start_sec: float
    end_sec: float
    transition: Optional[Dict[str, float]]


def detect_tracks(ctx, job) -> List[Chunk]:
    mode = getattr(ctx, "analysis_mode", "pure_audio_guess")
    duration = float(getattr(ctx, "duration_sec", 0.0) or 0.0)
    trimmed_start = float(getattr(ctx, "trimmed_start_sec", 0.0) or 0.0)

    # Determine start time based on mode and cue data
    cue_tracks = job.get("cue_tracks") or []
    if mode == "cue_correlated" and cue_tracks:
        start_sec = float(cue_tracks[0].get("start_time_sec", trimmed_start) or trimmed_start)
        default_title = f"{cue_tracks[0].get('title') or 'Track 1 (from cue)'}"
    else:
        start_sec = trimmed_start
        default_title = "Guessed Track 1"

    end_sec = duration if duration and duration > start_sec else start_sec

    title = (cue_tracks[0].get("title") if cue_tracks else None) or (
        default_title if mode == "pure_audio_guess" else (cue_tracks[0].get("title") if cue_tracks else "Track 1 (from cue)")
    )

    return [Chunk(track_id=0, title=title, start_sec=start_sec, end_sec=end_sec, transition=None)]
