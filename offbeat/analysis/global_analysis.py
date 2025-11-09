"""Global analysis (beats + BPM curve).

Loads audio with librosa, trims leading silence, computes a global beat grid
and a downsampled BPM curve. Returns a context reused by later phases.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import os

import numpy as np
import librosa

from ..config import settings

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None  # type: ignore

# Lazy import within function for Spleeter to avoid hard dependency in simple tests


def _moving_average(arr: List[float], window: int) -> List[float]:
    if not arr or window <= 1:
        return list(arr)
    window = int(window)
    if window <= 1:
        return list(arr)
    pad = window // 2
    padded = np.pad(np.array(arr, dtype=float), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return [float(x) for x in smoothed]


def _rate_limit(arr: List[float], max_delta_per_sec: float) -> List[float]:
    if not arr or max_delta_per_sec is None or max_delta_per_sec <= 0:
        return list(arr)
    out: List[float] = [float(arr[0])]
    max_d = float(max_delta_per_sec)
    for i in range(1, len(arr)):
        prev = out[-1]
        target = float(arr[i])
        delta = target - prev
        if delta > max_d:
            target = prev + max_d
        elif delta < -max_d:
            target = prev - max_d
        out.append(float(target))
    return out


def _clip_range(arr: List[float], lo: float, hi: float) -> List[float]:
    if not arr:
        return []
    lo = float(lo)
    hi = float(hi)
    return [float(min(max(x, lo), hi)) for x in arr]


@dataclass
class GlobalContext:
    duration_sec: Optional[float]
    trimmed_start_sec: Optional[float]
    beat_grid_times: List[float]
    # Per-second BPM curve; index i corresponds to time i seconds.
    bpm_curve: List[float]
    analysis_mode: str
    threads: int
    # Cached audio for downstream use
    y: Optional[np.ndarray] = None
    sr: Optional[int] = None
    # v2 additions
    stems_arrays: Optional[Dict[str, np.ndarray]] = None  # mono arrays for slicing
    stems: Dict[str, str] = None  # relative paths to saved stems


def _per_second_resample(times: np.ndarray, values: np.ndarray, duration_sec: float) -> List[float]:
    """Return a per-second BPM list where index i corresponds to time i seconds.

    We pick the nearest-previous frame value for each integer second from 0..floor(duration).
    """
    if times.size == 0 or values.size == 0 or duration_sec <= 0:
        return []
    # Ensure increasing times
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    # Build integer-second grid from 0 to floor(duration)
    last_sec = int(max(0, np.floor(duration_sec)))
    grid = np.arange(0, last_sec + 1, dtype=float)
    # Map each second to the closest index at or before that time
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return [float(values[i]) for i in idx]


def run_global_analysis(job: dict) -> GlobalContext:
    """Compute global context with librosa.

    - Load mono audio at configured sample rate.
    - Trim leading silence; record time offset.
    - Beat track on trimmed audio; convert beat times to absolute.
    - Compute dynamic tempo over time and downsample to ~1 Hz.
    - Force analysis mode to 'pure_audio_guess' per product pivot.
    """
    # Resolve absolute input path based on v2 (prefer relative_path under shared mount)
    rel = str(job.get("relative_path", "") or "")
    file_path = None
    if rel:
        file_path = os.path.join(settings.shared_mount_path, rel)
    else:
        file_path = job.get("file_path")
    if not file_path or not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load mono at target SR
    target_sr = int(getattr(settings, "sample_rate", 22050))
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)

    # Trim leading/trailing silence per v2
    try:
        y_trim, (start_idx, end_idx) = librosa.effects.trim(y, top_db=int(getattr(settings, "silence_top_db", 40)))
        time_offset_sec = float(start_idx) / float(sr)
    except Exception:
        y_trim = y
        time_offset_sec = 0.0
    duration_sec = float(len(y_trim)) / float(sr)

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y_trim, sr=sr)
    beat_times_rel = librosa.frames_to_time(beat_frames, sr=sr)
    beat_times_abs = (beat_times_rel + time_offset_sec).astype(float).tolist()

    # v2: attempt full-file Spleeter separation and save stems as OGG
    stems_arrays: Optional[Dict[str, np.ndarray]] = None
    stems_paths: Dict[str, str] = {}
    try:
        from spleeter.separator import Separator  # type: ignore
        sep = Separator("spleeter:4stems")
        # Spleeter expects stereo; duplicate mono to two channels
        stereo = np.stack([y_trim, y_trim], axis=-1)
        pred = sep.separate(stereo, sample_rate=sr)  # type: ignore[arg-type]
        # Convert to mono arrays
        stems_arrays = {k: (v.mean(axis=1) if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in pred.items()}

        # Determine base relative path and abs output dir
        if rel:
            base_rel_dir, base_rel_name = os.path.split(rel)
        else:
            # derive rel from absolute by stripping shared mount prefix
            try:
                base_rel = os.path.relpath(file_path, settings.shared_mount_path)
            except Exception:
                base_rel = os.path.basename(file_path)
            base_rel_dir, base_rel_name = os.path.split(base_rel)
        base_name, _ = os.path.splitext(base_rel_name)

        def _save_stem(name: str, arr: np.ndarray) -> str:
            out_rel = os.path.join(base_rel_dir, f"{base_name}_{name}_global.ogg")
            out_abs = os.path.join(settings.shared_mount_path, out_rel)
            os.makedirs(os.path.dirname(out_abs), exist_ok=True)
            if sf is not None and isinstance(arr, np.ndarray) and arr.size:
                sf.write(out_abs, arr.astype(np.float32), sr, format='OGG', subtype='VORBIS')
            return out_rel

        # Save all four
        for k in ("vocals", "drums", "bass", "other"):
            arr = stems_arrays.get(k) if stems_arrays else None
            if isinstance(arr, np.ndarray) and arr.size:
                stems_paths[k] = _save_stem(k, arr)
    except Exception:
        # Spleeter or save failed; keep stems empty
        stems_arrays = None
        stems_paths = {}

    # Prefer beat-interval derived BPM (more stable than onset tempogram)
    hop_length = int(getattr(settings, "hop_length", 512))

    bpm_curve: List[float] = []
    if len(beat_times_rel) >= 4:
        # Instantaneous tempo from consecutive beat intervals
        intervals = np.diff(beat_times_rel)  # seconds per beat
        # Guard against zeros
        intervals = intervals[intervals > 1e-6]
        if intervals.size >= 1:
            inst_bpm = 60.0 / intervals
            # Reference BPM as median of instantaneous BPMs
            ref_bpm = float(np.median(inst_bpm)) if inst_bpm.size else None

            # Fold BPMs near plausible musical range and around reference
            def _fold_to_ref(bpm: float, ref: Optional[float]) -> float:
                if not np.isfinite(bpm) or bpm <= 0:
                    return 0.0
                # Bring into [60, 200]
                while bpm < 60.0:
                    bpm *= 2.0
                while bpm > 200.0:
                    bpm *= 0.5
                if ref and ref > 0:
                    # Try small-ratio candidates to reduce octave errors
                    candidates = [bpm * 0.5, bpm * (2.0 / 3.0), bpm, bpm * 1.5, bpm * 2.0]
                    candidates = [c for c in candidates if 60.0 <= c <= 200.0]
                    if candidates:
                        # Choose candidate closest to reference
                        bpm = min(candidates, key=lambda c: abs(c - ref))
                return bpm

            folded = np.array([_fold_to_ref(float(v), ref_bpm) for v in inst_bpm], dtype=float)

            # Smooth over a sliding window of ~8-12 beats (choose 10)
            win_beats = 10
            if folded.size >= win_beats:
                kernel = np.ones(win_beats, dtype=float) / float(win_beats)
                pad = win_beats // 2
                padded = np.pad(folded, (pad, pad), mode="edge")
                bpm_smooth_beats = np.convolve(padded, kernel, mode="valid")
            else:
                bpm_smooth_beats = folded

            # Map beat-centered times to absolute seconds
            # Place each instantaneous tempo at the midpoint of the interval it describes
            times_mid = beat_times_rel[:-1] + (np.diff(beat_times_rel) / 2.0) + time_offset_sec
            times_mid = times_mid[: len(bpm_smooth_beats)]

            # Resample to per-second
            bpm_curve = _per_second_resample(times_mid.astype(float), bpm_smooth_beats.astype(float), duration_sec)

    if not bpm_curve:
        # Fallback: onset-based per-frame tempo series if beats insufficient
        oenv = librosa.onset.onset_strength(y=y_trim, sr=sr, hop_length=hop_length)
        bpm_series = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length, aggregate=None)
        # Reference BPM from global median
        ref_bpm = float(np.median(bpm_series)) if bpm_series.size else None

        def _fold_bpm(bpm: float, ref: Optional[float]) -> float:
            if not np.isfinite(bpm) or bpm <= 0:
                return 0.0
            while bpm < 60.0:
                bpm *= 2.0
            while bpm > 200.0:
                bpm *= 0.5
            if ref and ref > 0:
                candidates = [bpm * 0.5, bpm * (2.0 / 3.0), bpm, bpm * 1.5, bpm * 2.0]
                candidates = [c for c in candidates if 60.0 <= c <= 200.0]
                if candidates:
                    bpm = min(candidates, key=lambda c: abs(c - ref))
            return bpm

        bpm_folded = np.array([_fold_bpm(float(x), ref_bpm) for x in bpm_series], dtype=float)
        # Smooth using a moving average window (~5 seconds)
        win_frames = int(max(1, round((5.0 * sr) / hop_length)))
        if win_frames > 1 and bpm_folded.size >= win_frames:
            kernel = np.ones(win_frames, dtype=float) / float(win_frames)
            pad = win_frames // 2
            padded = np.pad(bpm_folded, (pad, pad), mode="edge")
            bpm_smooth = np.convolve(padded, kernel, mode="valid")
        else:
            bpm_smooth = bpm_folded
        frame_times = librosa.frames_to_time(np.arange(len(bpm_smooth)), sr=sr, hop_length=hop_length) + time_offset_sec
        bpm_smooth = np.clip(bpm_smooth, 60.0, 200.0)
        bpm_curve = _per_second_resample(frame_times.astype(float), bpm_smooth.astype(float), duration_sec)

    # Post-process BPM curve to reduce unrealistic second-to-second jitter
    if bpm_curve:
        # Moving average over N seconds
        win_sec = int(getattr(settings, "bpm_smooth_seconds", 8) or 0)
        if win_sec > 1:
            bpm_curve = _moving_average(bpm_curve, win_sec)
        # Rate-of-change limiting (max delta per second)
        max_dps = float(getattr(settings, "bpm_max_dps", 1.5))
        bpm_curve = _rate_limit(bpm_curve, max_dps)
        # Clamp to configured range
        bpm_lo = float(getattr(settings, "bpm_min", 60))
        bpm_hi = float(getattr(settings, "bpm_max", 190))
        bpm_curve = _clip_range(bpm_curve, bpm_lo, bpm_hi)

    mode = "pure_audio_guess"  # pivot: ignore cues and force pure audio

    return GlobalContext(
        duration_sec=duration_sec,
        trimmed_start_sec=time_offset_sec,
        beat_grid_times=[float(t) for t in beat_times_abs],
        bpm_curve=bpm_curve,
        analysis_mode=mode,
        threads=int(getattr(settings, "threads", 3)),
        y=y_trim,
        sr=sr,
        stems_arrays=stems_arrays,
        stems=stems_paths,
    )
