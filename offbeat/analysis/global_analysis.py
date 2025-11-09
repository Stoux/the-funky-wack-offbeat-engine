"""Global analysis PoC.

Implements a minimal processing task for the PoC: compute audio duration for
WAV files using the stdlib `wave` module. This keeps the repository light
(no heavy analysis deps) while providing a real, testable output value
(`duration_sec`).
"""
from dataclasses import dataclass
from typing import List, Dict
import contextlib
import wave
import os

from ..config import settings


@dataclass
class GlobalContext:
    duration_sec: float | None
    trimmed_start_sec: float | None
    beat_grid_times: List[float]
    bpm_curve: List[Dict[str, float]]
    analysis_mode: str
    threads: int


def _duration_from_wav(file_path: str) -> float:
    """Return duration in seconds for a WAV file using stdlib.

    Raises:
        FileNotFoundError: if file does not exist
        ValueError: if file is not a readable WAV or has invalid params
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with contextlib.closing(wave.open(file_path, "rb")) as wf:
            frames = wf.getnframes()
            framerate = wf.getframerate()
            if framerate <= 0:
                raise ValueError("Invalid WAV: zero framerate")
            return float(frames) / float(framerate)
    except wave.Error as e:
        raise ValueError(f"Unsupported or invalid WAV file: {e}")


def run_global_analysis(job: dict) -> GlobalContext:
    """Compute basic global context for PoC.

    - Computes duration for WAV files via stdlib.
    - Sets trimmed_start_sec to 0.0 (no trimming in PoC).
    - Keeps beat grid and bpm curve empty.
    - Chooses analysis mode based on presence of cue_tracks.
    """
    mode = "cue_correlated" if job.get("cue_tracks") else "pure_audio_guess"
    duration = _duration_from_wav(job["file_path"])  # may raise; worker will catch and report
    return GlobalContext(
        duration_sec=duration,
        trimmed_start_sec=0.0,
        beat_grid_times=[],
        bpm_curve=[],
        analysis_mode=mode,
        threads=int(getattr(settings, "threads", 3)),
    )
