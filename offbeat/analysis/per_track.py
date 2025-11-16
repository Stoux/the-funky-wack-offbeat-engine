"""Per-track analysis with Spleeter-backed features (mandatory).

Design notes:
- Spleeter is required. If it is not installed or cannot initialize, the
  system must fail fast: init_spleeter() raises RuntimeError.
- Loudness via pyloudnorm is required. If pyloudnorm is not installed, the
  system must fail fast during import.
- When both are available, we compute stems-based features.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
from offbeat.logging_utils import get_logger

try:
    from librosa.feature.rhythm import tempo as lr_tempo  # librosa >= 0.10
except Exception:  # pragma: no cover - older librosa
    import librosa  # type: ignore
    lr_tempo = librosa.beat.tempo

logger = get_logger("analysis.per_track")

try:
    import librosa  # lightweight and already a dependency
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

try:  # loudness is mandatory
    import pyloudnorm as pyln  # type: ignore
    logger.info("pyloudnorm available: enabling LUFS computation")
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pyloudnorm is required but not installed or failed to import. "
        "Please add 'pyloudnorm' to your environment. Original error: {}".format(e)
    )

# Optional Spleeter
_separator = None  # type: ignore
_have_spleeter = False

try:  # import gated, may fail on CI
    from spleeter.separator import Separator  # type: ignore
    _have_spleeter = True
except Exception:  # pragma: no cover
    Separator = None  # type: ignore
    _have_spleeter = False

# Key detection helpers (lightweight, no external deps beyond librosa/numpy)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
ALL_KEYS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
KEY_PROFILES = {**{f"{ALL_KEYS[i]} Major": np.roll(MAJOR_PROFILE, i) for i in range(12)},
                **{f"{ALL_KEYS[i]} Minor": np.roll(MINOR_PROFILE, i) for i in range(12)}}



def init_spleeter():
    """Initialize Spleeter Separator (mandatory).

    Behavior:
    - If Spleeter is not installed/importable, raise RuntimeError.
    - If Separator construction fails, raise RuntimeError.
    - Safe to call multiple times (no-op if already initialized).
    """
    global _separator
    if not _have_spleeter:
        raise RuntimeError("Spleeter is required but not installed. Please install 'spleeter' and its dependencies.")
    if _separator is not None:
        return None
    try:
        # 4 stems gives access to vocals and bass for simple features
        _separator = Separator("spleeter:4stems")
    except Exception as e:
        _separator = None
        raise RuntimeError(f"Failed to initialize Spleeter Separator: {e}")
    return None


def teardown_spleeter():
    # Nothing special to teardown in CPU mode; keep placeholder for symmetry
    return None


def _safe_duration(chunk) -> float:
    start = float(getattr(chunk, "start_sec", 0.0) or 0.0)
    end = float(getattr(chunk, "end_sec", start) or start)
    return float(max(0.0, end - start))


def _slice_chunk_audio(ctx, chunk) -> tuple[np.ndarray, int]:
    """Return y_chunk (mono) and sr with guards."""
    y = getattr(ctx, "y", None)
    sr = int(getattr(ctx, "sr", 22050) or 22050)
    if y is None:
        logger.warning("_slice_chunk_audio: ctx.y is None; returning empty slice at sr={}", sr)
        return np.zeros(0, dtype=float), sr
    if librosa is None:
        logger.warning("_slice_chunk_audio: librosa not available; returning empty slice at sr={}", sr)
        return np.zeros(0, dtype=float), sr
    if len(y) == 0:
        logger.info("_slice_chunk_audio: ctx.y is empty; returning empty slice at sr={}", sr)
        return np.zeros(0, dtype=float), sr
    t0 = float(getattr(ctx, "trimmed_start_sec", 0.0) or 0.0)
    start = max(float(getattr(chunk, "start_sec", 0.0) or 0.0) - t0, 0.0)
    end = max(float(getattr(chunk, "end_sec", 0.0) or 0.0) - t0, start)
    s = int(max(0, round(start * sr)))
    e = int(max(s, round(end * sr)))
    s = min(s, len(y))
    e = min(e, len(y))
    y_chunk = y[s:e]
    if y_chunk.ndim > 1:
        # ensure mono
        y_chunk = np.mean(y_chunk, axis=-1)
    logger.debug("_slice_chunk_audio: start_sec={:.3f} end_sec={:.3f} -> samples [{}:{}] size={} sr={}",
                 float(getattr(chunk, "start_sec", 0.0) or 0.0),
                 float(getattr(chunk, "end_sec", 0.0) or 0.0), s, e, y_chunk.size, sr)
    return y_chunk.astype(float, copy=False), sr


def _compute_brightness(y_chunk: np.ndarray, sr: int) -> Optional[float]:
    if librosa is None or y_chunk.size == 0:
        return None
    try:
        centroid = librosa.feature.spectral_centroid(y=y_chunk, sr=sr)
        return float(np.mean(centroid)) if centroid.size else None
    except Exception:
        return None


def _compute_loudness(y_chunk: np.ndarray, sr: int) -> Optional[float]:
    if pyln is None:
        logger.debug("_compute_loudness: pyloudnorm missing; cannot compute LUFS (size={} sr={})", y_chunk.size, sr)
        return None
    if y_chunk.size == 0:
        logger.debug("_compute_loudness: empty input; cannot compute LUFS (sr={})", sr)
        return None
    try:
        meter = pyln.Meter(sr)
        lufs = float(meter.integrated_loudness(y_chunk))
        logger.debug("_compute_loudness: computed LUFS={:.3f} (size={} sr={})", lufs, y_chunk.size, sr)
        return lufs
    except Exception as e:
        logger.warning("_compute_loudness: exception during LUFS computation: {}", e)
        return None


def _separate_stems(y_chunk: np.ndarray, sr: int) -> Optional[dict]:
    """Deprecated in v2: no per-chunk separation. Kept for backward compatibility.
    Return None to force usage of precomputed global stems.
    """
    return None


def _slice_stem(ctx, stem_name: str, start_sec: float, end_sec: float) -> tuple[np.ndarray, int]:
    stems = getattr(ctx, "stems_arrays", None) or {}
    sr = int(getattr(ctx, "sr", 22050) or 22050)
    t0 = float(getattr(ctx, "trimmed_start_sec", 0.0) or 0.0)
    start = max(float(start_sec) - t0, 0.0)
    end = max(float(end_sec) - t0, start)
    s = int(max(0, round(start * sr)))
    e = int(max(s, round(end * sr)))
    arr = stems.get(stem_name)
    if not isinstance(arr, np.ndarray) or arr.size == 0:
        logger.info("_slice_stem: missing/empty stem '{}' -> returning empty slice (sr={} window=[{:.3f},{:.3f}] samples=[{}:{}])",
                    stem_name, sr, start_sec, end_sec, s, e)
        return np.zeros(0, dtype=float), sr
    s = min(s, len(arr))
    e = min(e, len(arr))
    sl = arr[s:e].astype(float, copy=False)
    logger.debug("_slice_stem: '{}' samples [{}:{}] size={} sr={}", stem_name, s, e, sl.size, sr)
    return sl, sr


def _local_guide_tempo(ctx, start_sec: float, end_sec: float) -> Optional[float]:
    """Return a robust local guide tempo (median BPM) from ctx.bpm_curve over
    the absolute window [start_sec, end_sec].

    Note: bpm_curve is per-second sampled relative to trimmed_start_sec==0.
    We convert absolute times to trimmed-relative by subtracting ctx.trimmed_start_sec.
    Falls back to None if window is empty or highly unstable.
    """
    curve = list(getattr(ctx, "bpm_curve", []) or [])
    if not curve:
        return None
    t0 = float(getattr(ctx, "trimmed_start_sec", 0.0) or 0.0)
    # Convert absolute seconds to indices in per-second bpm_curve
    rel_start = max(0.0, float(start_sec) - t0)
    rel_end = max(rel_start, float(end_sec) - t0)
    i0 = int(max(0, np.floor(rel_start)))
    i1 = int(max(i0, np.ceil(rel_end)))
    i0 = min(i0, len(curve))
    i1 = min(i1, len(curve))
    if i1 <= i0:
        return None
    window = curve[i0:i1]
    if not window:
        return None
    med = float(np.median(window))
    if not np.isfinite(med) or med <= 0:
        return None
    # Instability check via IQR
    try:
        q75, q25 = np.percentile(window, 75), np.percentile(window, 25)
        iqr = float(q75 - q25)
        if iqr > 8.0:  # too wobbly; transition/breakdown
            return None
    except Exception:
        pass

    # Normalize the local median toward a musically plausible octave.
    try:
        ref = getattr(ctx, "t_global", None)
        adjusted = med
        if ref is not None and np.isfinite(ref) and ref > 0:
            cands = [med * 0.5, med, med * 1.5, med * 2.0]
            cands = [c for c in cands if 60.0 <= c <= 200.0]
            if cands:
                adjusted = float(min(cands, key=lambda c: abs(c - float(ref))))
        else:
            # If no reference, avoid half-time by preferring [90, 180] when obvious
            if 60.0 <= med < 85.0 and (med * 2.0) <= 200.0:
                adjusted = med * 2.0
        if abs(adjusted - med) >= 1e-6:
            logger.debug("_local_guide_tempo: adjusted median from %.2f -> %.2f (ref=%s)", med, adjusted, ref)
            med = adjusted
    except Exception:
        pass

    # Log debug info about the chosen window
    try:
        logger.debug(
            "_local_guide_tempo: abs=[%.2f,%.2f] rel=[%.2f,%.2f] idx=[%d:%d] win_len=%d med=%.2f",
            float(start_sec), float(end_sec), rel_start, rel_end, i0, i1, (i1 - i0), med,
        )
    except Exception:
        pass
    return med


def _stable_bpm(y_drum: np.ndarray, sr: int, t_global: Optional[float] = None) -> Optional[float]:
    if librosa is None or y_drum.size == 0:
        return None
    try:
        # Use finer hop for improved resolution
        hop_length = 256
        onset_env = librosa.onset.onset_strength(y=y_drum, sr=sr, hop_length=hop_length)

        # Normalize/adjust guide toward the musically plausible octave
        raw_guide = float(t_global) if (t_global is not None and np.isfinite(t_global) and t_global > 0) else None
        guide = raw_guide
        if guide is not None:
            # If guide is in a half/double octave, prefer the one in [90, 180] when possible
            if guide < 90.0 and (guide * 2.0) <= 200.0:
                guide = guide * 2.0
                logger.debug("_stable_bpm: elevated low guide %.2f -> %.2f to avoid half-time bias", raw_guide, guide)
            elif guide > 180.0 and (guide / 2.0) >= 60.0:
                guide = guide / 2.0
                logger.debug("_stable_bpm: reduced high guide %.2f -> %.2f to avoid double-time bias", raw_guide, guide)

        # 1) Multi-candidate selection near guide using tempogram peaks
        if guide is not None:
            cands = lr_tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=None)
            if cands is not None and len(cands) > 0:
                expanded: list[float] = []
                for t in cands:
                    tt = float(t)
                    expanded.extend([tt / 2.0, tt, tt * 2.0])
                # Gate to a band around guide to avoid octave slips (slightly wider than before)
                lo = max(60.0, 0.67 * guide)
                hi = min(200.0, 1.5 * guide)
                expanded = [float(t) for t in expanded if lo <= float(t) <= hi]
                if expanded:
                    choice = float(min(expanded, key=lambda t: abs(t - guide)))
                    logger.debug("_stable_bpm: using candidate selection near guide=%.2f -> %.3f (hop=%d)", guide, choice, hop_length)
                    return choice

        # 2) Fallback: bias beat_track toward guide with matching hop_length
        if guide is not None:
            tempo_bt, _ = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=hop_length,  # critical to match onset_env
                start_bpm=guide,
                tightness=100.0,
                trim=False,
            )
            cands = [tempo_bt / 2.0, float(tempo_bt), float(tempo_bt) * 2.0]
            lo = max(60.0, 0.67 * guide)
            hi = min(200.0, 1.5 * guide)
            cands = [c for c in cands if lo <= c <= hi]
            if cands:
                choice = float(min(cands, key=lambda t: abs(t - guide)))
                logger.debug("_stable_bpm: beat_track biased to guide=%.2f -> %.3f (hop=%d)", guide, choice, hop_length)
                return choice
            logger.debug("_stable_bpm: beat_track returned tempo=%.3f without valid candidates (hop=%d)", float(tempo_bt), hop_length)
            return float(tempo_bt)

        # 3) Last resort: unbiased beat_track with correct hop_length
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        tempo = float(tempo)
        # Reconcile last-resort tempo toward guide if available, else simple half-time fix
        if np.isfinite(tempo) and tempo > 0:
            if guide is not None:
                # Snap to nearest of {T/2, T, 2T} around the (adjusted) guide
                cands = [tempo / 2.0, tempo, tempo * 2.0]
                # Keep within plausible range
                cands = [c for c in cands if 60.0 <= c <= 200.0]
                if cands:
                    tempo = float(min(cands, key=lambda t: abs(t - guide)))
            else:
                if tempo < 90.0:
                    tempo *= 2.0
        logger.debug("_stable_bpm: last-resort beat_track tempo=%.3f (hop=%d)", tempo, hop_length)
        return tempo
    except Exception as e:
        logger.warning("_stable_bpm: exception during tempo estimation: {}", e)
        return None


def _find_musical_key(y_harm: np.ndarray, sr: int) -> str:
    if librosa is None or y_harm.size == 0:
        return "Unknown"
    try:
        chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
        prof = np.sum(chroma, axis=1)
        if np.sum(prof) == 0:
            return "Unknown"
        prof = prof / np.sum(prof)
        best = max(KEY_PROFILES.items(), key=lambda kv: np.corrcoef(prof, kv[1])[0, 1])[0]
        return best
    except Exception:
        return "Unknown"


def _camelot_from_musical(musical: Optional[str]) -> Optional[str]:
    if not musical or musical == "Unknown":
        return None
    try:
        parts = str(musical).split()
        tonic = parts[0]
        mode = parts[1] if len(parts) > 1 else None
        camelot_map_minor = {
            'A': '8A', 'E': '9A', 'B': '10A', 'F#': '11A', 'C#': '12A',
            'G#': '1A', 'D#': '2A', 'A#': '3A', 'F': '4A', 'C': '5A', 'G': '6A', 'D': '7A'
        }
        camelot_map_major = {
            'C': '8B', 'G': '9B', 'D': '10B', 'A': '11B', 'E': '12B',
            'B': '1B', 'F#': '2B', 'C#': '3B', 'G#': '4B', 'D#': '5B', 'A#': '6B', 'F': '7B'
        }
        if mode == 'Major':
            return camelot_map_major.get(tonic)
        if mode == 'Minor':
            return camelot_map_minor.get(tonic)
        return None
    except Exception:
        return None


def analyze_track_chunk(ctx, chunk) -> Dict[str, Any]:
    # Defaults
    duration = _safe_duration(chunk)
    y_chunk, sr = _slice_chunk_audio(ctx, chunk)
    start = float(getattr(chunk, "start_sec", 0.0) or 0.0)
    end = float(getattr(chunk, "end_sec", start) or start)

    # Slice stems from precomputed global stems
    drums, _ = _slice_stem(ctx, "drums", start, end)
    bass, _ = _slice_stem(ctx, "bass", start, end)
    other, _ = _slice_stem(ctx, "other", start, end)
    vocals, _ = _slice_stem(ctx, "vocals", start, end)

    logger.info(
        "analyze_track_chunk: track_id={} window=[{:.3f},{:.3f}] dur={:.3f}s sr={} sizes: y={} drums={} bass={} other={} vocals={}",
        int(getattr(chunk, "track_id", 0) or 0), start, end, duration, sr,
        y_chunk.size, drums.size, bass.size, other.size, vocals.size,
    )

    # Features
    lufs_total = _compute_loudness(y_chunk, sr)
    lufs_bass = _compute_loudness(bass, sr)
    vocal_rms = float(np.sqrt(np.mean(vocals.astype(float) ** 2))) if vocals.size else None
    has_vocals = (vocal_rms is not None and vocal_rms > 0.02)
    brightness = _compute_brightness(y_chunk, sr)
    # Derive local guide tempo from global bpm_curve over this chunk's window; fallback to global
    t_local = _local_guide_tempo(ctx, start, end)
    t_global = getattr(ctx, "t_global", None)
    guide = t_local if (t_local is not None) else t_global
    logger.info(
        "bpm guide window: start=%.2f end=%.2f t_local=%s t_global=%s", start, end, t_local, t_global
    )
    stable_bpm = _stable_bpm(drums, sr, guide)
    musical_key = _find_musical_key(bass + other if (bass.size and other.size) else y_chunk, sr)
    camelot_key = _camelot_from_musical(musical_key)

    logger.info(
        "analyze_track_chunk: results track_id={} LUFS_total={} LUFS_bass={} brightness={} has_vocals={} vocal_rms={} stable_bpm={} key_musical={} key_camelot={} guide_bpm={}",
        int(getattr(chunk, "track_id", 0) or 0), lufs_total, lufs_bass, brightness, has_vocals, vocal_rms, stable_bpm, musical_key, camelot_key, guide,
    )

    return {
        "track_id": int(getattr(chunk, "track_id", 0) or 0),
        "start_time_sec": float(getattr(chunk, "start_sec", 0.0) or 0.0),
        "analysis": {
            "duration_sec": duration,
            "stable_bpm": stable_bpm,
            "key": {
                "musical": musical_key,
                "camelot": camelot_key,
            },
            "loudness_lufs_total": lufs_total,
            "loudness_lufs_bass": lufs_bass,
            "average_brightness": brightness,
            "has_vocals": has_vocals,
            "vocal_energy_rms": vocal_rms,
        },
    }
