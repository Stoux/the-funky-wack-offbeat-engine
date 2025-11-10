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
from ..logging_utils import get_logger


@dataclass
class Chunk:
    track_id: int
    title: str
    start_sec: float
    end_sec: float
    transition: Optional[Dict[str, float]]


def detect_tracks(ctx, job) -> List[Chunk]:
    # Pivot: ignore cues and always use pure audio detection
    log = get_logger(
        "analysis.track_detection",
        job_id=str(job.get("job_id") or job.get("id") or job.get("relative_path") or job.get("file_path") or "?")
    )
    log.info("track_detection: start (pure-audio pivot)")
    try:
        chunks = _detect_pure_audio(ctx, job)
        log.info("track_detection: done; chunks=%d", len(chunks))
        return chunks
    except Exception:
        log.exception("track_detection: failed; falling back to single-span chunk")
        # Attempt to provide a single chunk fallback
        duration = float(getattr(ctx, "duration_sec", 0.0) or 0.0)
        trimmed_start = float(getattr(ctx, "trimmed_start_sec", 0.0) or 0.0)
        return [Chunk(track_id=0, title="Guessed Track 1", start_sec=trimmed_start, end_sec=duration, transition=None)]


def _minmax_scale(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax - xmin <= 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def _novelty_supercurve(y: np.ndarray, sr: int, hop_length: int, log=None) -> tuple[np.ndarray, np.ndarray]:
    """Revised fused novelty curve:
    - Base spectral novelty from STFT onset strength (larger FFT)
    - Beat-synchronous chroma key-change distance
    - Normalize, fuse by weighted addition (0.7/0.3)
    - Return curve (length T) and frame times for hop_length
    """
    if log is None:
        log = get_logger("analysis.track_detection")
    log.info("supercurve: computing STFT+onset (n_fft=4096 hop=%d)", hop_length)
    # 1) Base spectral novelty
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=hop_length))
    base_nov = librosa.onset.onset_strength(S=S, sr=sr, hop_length=hop_length)

    # Smooth ~2 seconds in frames
    sigma_frames = max(1, int(2 * sr / hop_length))
    try:
        from scipy.ndimage import gaussian_filter1d  # type: ignore
        base_nov_s = gaussian_filter1d(base_nov, sigma=sigma_frames)
        log.info("supercurve: gaussian smoothing applied (sigma_frames=%d)", sigma_frames)
    except Exception:
        # Fallback: moving average with window ~ 2 seconds
        w = max(1, sigma_frames)
        if w > 1:
            kernel = np.ones(w, dtype=float) / float(w)
            base_nov_s = np.convolve(base_nov, kernel, mode="same")
        else:
            base_nov_s = base_nov
        log.info("supercurve: fallback moving-average smoothing (win=%d)", w)

    # 2) Key-change novelty via beat-synchronous chroma
    try:
        log.info("supercurve: beat_track + chroma_cqt for key-jump")
        _tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        if beats is None or len(beats) < 2:
            key_jump = np.zeros(1, dtype=float)
            log.info("supercurve: insufficient beats for key-jump (count=%s)", 0 if beats is None else len(beats))
        else:
            Cb = librosa.util.sync(chroma, beats, aggregate=np.median)
            if Cb.shape[1] < 2:
                key_jump = np.zeros(1, dtype=float)
                log.info("supercurve: beat-synced chroma too short (frames=%d)", Cb.shape[1])
            else:
                key_jump = np.linalg.norm(np.diff(Cb, axis=1), axis=0)
                log.info("supercurve: computed key-jump len=%d", len(key_jump))
    except Exception:
        key_jump = np.zeros(1, dtype=float)
        log.info("supercurve: key-jump computation failed; using zeros")

    # 3) Align and fuse
    if len(key_jump) <= 1:
        key_up = np.zeros_like(base_nov_s)
        log.info("supercurve: key curve unavailable; zeros used")
    else:
        key_up = np.interp(
            np.linspace(0, len(key_jump) - 1, num=len(base_nov_s)),
            np.arange(len(key_jump)),
            key_jump,
        )
        log.info("supercurve: upsampled key curve to len=%d", len(base_nov_s))

    base_n = _minmax_scale(base_nov_s)
    key_n = _minmax_scale(key_up)
    supercurve = 0.7 * base_n + 0.3 * key_n
    log.info("supercurve: fused (weights 0.7/0.3); len=%d", len(supercurve))

    times = librosa.frames_to_time(np.arange(len(supercurve)), sr=sr, hop_length=hop_length)
    return supercurve, times


def _detect_pure_audio(ctx, job) -> List[Chunk]:
    log = get_logger("analysis.track_detection", job_id=str(job.get("job_id") or job.get("id") or job.get("relative_path") or job.get("file_path") or "?"))
    duration = float(getattr(ctx, "duration_sec", 0.0) or 0.0)
    trimmed_start = float(getattr(ctx, "trimmed_start_sec", 0.0) or 0.0)
    y = getattr(ctx, "y", None)
    sr = int(getattr(ctx, "sr", 22050) or 22050)
    hop = int(getattr(settings, "hop_length", 512))

    log.info("detect_pure_audio: params sr=%d hop=%d duration=%.2fs trimmed_start=%.2fs", sr, hop, duration, trimmed_start)

    # Guards
    if y is None or len(y) == 0 or duration <= trimmed_start:
        start_sec = trimmed_start
        end_sec = duration if duration and duration > start_sec else start_sec
        log.info("detect_pure_audio: guard fallback to single chunk [%.2f, %.2f]", start_sec, end_sec)
        return [Chunk(track_id=0, title="Guessed Track 1", start_sec=start_sec, end_sec=end_sec, transition=None)]

    curve, times_rel = _novelty_supercurve(y, sr, hop, log=log)

    # Peak picking parameters (dynamic thresholding via MAD)
    min_gap_sec = max(30.0, float(getattr(settings, "min_track_duration_sec", 120)))
    distance_frames = int(max(1, min_gap_sec * sr / hop))

    # Dynamic threshold
    median = float(np.median(curve)) if curve.size else 0.0
    mad = float(np.median(np.abs(curve - median))) if curve.size else 0.0
    k = float(getattr(settings, "peak_mad_k", 1.5))
    dyn_thr = median + k * mad
    log.info("detect_pure_audio: threshold median=%.4f mad=%.4f k=%.2f dyn_thr=%.4f distance_frames=%d", median, mad, k, dyn_thr, distance_frames)

    # Prefer scipy.signal.find_peaks; fallback to librosa peak_pick if needed
    try:
        import scipy.signal as sig  # type: ignore
        peaks, _ = sig.find_peaks(curve, height=dyn_thr, distance=distance_frames)
        log.info("detect_pure_audio: scipy.find_peaks found=%d", len(peaks))
    except Exception:
        pre_max = post_max = int(max(1, 5.0 * sr / hop))
        pre_avg = post_avg = int(max(1, 5.0 * sr / hop))
        delta = float(getattr(settings, "peak_delta", 0.3))
        try:
            peaks = librosa.util.peak_pick(curve, pre_max, post_max, pre_avg, post_avg, delta, distance_frames)
            log.info("detect_pure_audio: librosa.peak_pick found=%d", len(peaks))
        except Exception:
            peaks = np.array([], dtype=int)
            log.info("detect_pure_audio: peak picking failed; defaulting to zero peaks")

    # Boundaries with guards
    boundaries: List[float] = [trimmed_start]
    for p in peaks:
        t_abs = float(times_rel[int(p)] + trimmed_start)
        if t_abs - boundaries[-1] >= min_gap_sec and t_abs < duration:
            boundaries.append(t_abs)
    if boundaries[-1] < duration:
        boundaries.append(duration)
    log.info("detect_pure_audio: boundaries(%d)=%s", len(boundaries), [round(b, 2) for b in boundaries])

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
        log.info("detect_pure_audio: no valid segments; single chunk fallback [%.2f, %.2f]", trimmed_start, duration)
    else:
        log.info("detect_pure_audio: produced %d chunks", len(chunks))

    # Ensure chronological order and sequential IDs
    chunks.sort(key=lambda c: (float(getattr(c, "start_sec", 0.0) or 0.0), float(getattr(c, "end_sec", 0.0) or 0.0)))
    for idx, ch in enumerate(chunks):
        ch.track_id = idx
        ch.title = f"Guessed Track {idx+1}"
    log.info("detect_pure_audio: ordered chunks chronologically; ids reassigned 0..%d", len(chunks)-1)
    return chunks
