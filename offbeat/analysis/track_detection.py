"""Track detection: pure-audio novelty-based segmentation (pivot).

Per product pivot, we always perform pure-audio guessing and ignore cue tracks.
This implementation computes a combined novelty curve and uses peak picking to
propose boundaries. Conservative defaults aim for DJ track-length segments.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import librosa

from ..config import settings


@dataclass
class Chunk:
    track_id: int
    title: str
    start_sec: float
    end_sec: float
    transition: Optional[Dict[str, float]]


def detect_tracks(ctx, job) -> List[Chunk]:
    # Pivot: ignore cues and always use pure audio detection
    return _detect_pure_audio(ctx, job)


def _minmax_scale(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax - xmin <= 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def _novelty_supercurve(y: np.ndarray, sr: int, hop_length: int) -> tuple[np.ndarray, np.ndarray]:
    # Rhythmic novelty: onset strength
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Timbral novelty: MFCC deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    dm = np.mean(np.abs(np.diff(mfcc, axis=1)), axis=0)
    # Harmonic novelty: chroma deltas
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    dc = np.mean(np.abs(np.diff(chroma, axis=1)), axis=0)

    # Align lengths
    L = min(len(oenv), len(dm), len(dc))
    if L <= 1:
        # Degenerate: fall back to onset envelope
        times = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=hop_length)
        return _minmax_scale(oenv), times
    oenv = oenv[:L]
    dm = dm[:L]
    dc = dc[:L]

    oenv_n = _minmax_scale(oenv)
    dm_n = _minmax_scale(dm)
    dc_n = _minmax_scale(dc)

    supercurve = (oenv_n + dm_n + dc_n) / 3.0
    times = librosa.frames_to_time(np.arange(L), sr=sr, hop_length=hop_length)
    return supercurve, times


def _detect_pure_audio(ctx, job) -> List[Chunk]:
    duration = float(getattr(ctx, "duration_sec", 0.0) or 0.0)
    trimmed_start = float(getattr(ctx, "trimmed_start_sec", 0.0) or 0.0)
    y = getattr(ctx, "y", None)
    sr = int(getattr(ctx, "sr", 22050) or 22050)
    hop = int(getattr(settings, "hop_length", 512))

    # Guards
    if y is None or len(y) == 0 or duration <= trimmed_start:
        start_sec = trimmed_start
        end_sec = duration if duration and duration > start_sec else start_sec
        return [Chunk(track_id=0, title="Guessed Track 1", start_sec=start_sec, end_sec=end_sec, transition=None)]

    curve, times_rel = _novelty_supercurve(y, sr, hop)

    # Peak picking parameters
    min_gap_sec = max(30.0, float(getattr(settings, "min_track_duration_sec", 120)))
    wait = int(max(1, min_gap_sec * sr / hop))
    pre_max = post_max = int(max(1, 5.0 * sr / hop))
    pre_avg = post_avg = int(max(1, 5.0 * sr / hop))
    delta = float(getattr(settings, "peak_delta", 0.3))

    try:
        peaks = librosa.util.peak_pick(curve, pre_max, post_max, pre_avg, post_avg, delta, wait)
    except Exception:
        peaks = np.array([], dtype=int)

    # Boundaries with guards
    boundaries: List[float] = [trimmed_start]
    for p in peaks:
        t_abs = float(times_rel[int(p)] + trimmed_start)
        if t_abs - boundaries[-1] >= min_gap_sec and t_abs < duration:
            boundaries.append(t_abs)
    if boundaries[-1] < duration:
        boundaries.append(duration)

    # Build chunks
    chunks: List[Chunk] = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
        if e <= s:
            continue
        if (e - s) < (0.5 * min_gap_sec):
            # Discard very short
            continue
        chunks.append(
            Chunk(
                track_id=len(chunks),
                title=f"Guessed Track {len(chunks)+1}",
                start_sec=float(s),
                end_sec=float(e),
                transition=None,
            )
        )

    if not chunks:
        chunks = [Chunk(track_id=0, title="Guessed Track 1", start_sec=trimmed_start, end_sec=duration, transition=None)]
    return chunks
