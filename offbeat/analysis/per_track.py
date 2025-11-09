"""Per-track analysis with Spleeter-backed features (mandatory).

Design notes:
- Spleeter is required. If it is not installed or cannot initialize, the
  system must fail fast: init_spleeter() raises RuntimeError.
- When Spleeter is available, we compute stems-based features. Some metrics
  (e.g., loudness via pyloudnorm) are optional and will be None if the optional
  dependency is missing, but Spleeter itself is mandatory.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np

try:
    import librosa  # lightweight and already a dependency
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

try:  # optional loudness
    import pyloudnorm as pyln  # type: ignore
except Exception:  # pragma: no cover
    pyln = None  # type: ignore

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
    if y is None or librosa is None or len(y) == 0:
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
    if pyln is None or y_chunk.size == 0:
        return None
    try:
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(y_chunk))
    except Exception:
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
        return np.zeros(0, dtype=float), sr
    s = min(s, len(arr))
    e = min(e, len(arr))
    return arr[s:e].astype(float, copy=False), sr


def _stable_bpm(y_drum: np.ndarray, sr: int) -> Optional[float]:
    if librosa is None or y_drum.size == 0:
        return None
    try:
        onset_env = librosa.onset.onset_strength(y=y_drum, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo = float(tempo)
        if tempo < 90:
            tempo *= 2.0
        return tempo
    except Exception:
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
        # Map to Camelot notation
        try:
            name_parts = best.split()
            tonic = name_parts[0]
            mode = name_parts[1]  # 'Major' or 'Minor'
            camelot_map_minor = {
                'A': '8A', 'E': '9A', 'B': '10A', 'F#': '11A', 'C#': '12A',
                'G#': '1A', 'D#': '2A', 'A#': '3A', 'F': '4A', 'C': '5A', 'G': '6A', 'D': '7A'
            }
            camelot_map_major = {
                'C': '8B', 'G': '9B', 'D': '10B', 'A': '11B', 'E': '12B',
                'B': '1B', 'F#': '2B', 'C#': '3B', 'G#': '4B', 'D#': '5B', 'A#': '6B', 'F': '7B'
            }
            code = (camelot_map_major if mode == 'Major' else camelot_map_minor).get(tonic)
            if code:
                pretty = f"{tonic} {mode}"
                return f"{code} / {pretty}"
        except Exception:
            pass
        return best
    except Exception:
        return "Unknown"


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

    # Features
    lufs_total = _compute_loudness(y_chunk, sr)
    lufs_bass = _compute_loudness(bass, sr)
    vocal_rms = float(np.sqrt(np.mean(vocals.astype(float) ** 2))) if vocals.size else None
    has_vocals = (vocal_rms is not None and vocal_rms > 0.02)
    brightness = _compute_brightness(y_chunk, sr)
    stable_bpm = _stable_bpm(drums, sr)
    key = _find_musical_key(bass + other if (bass.size and other.size) else y_chunk, sr)

    return {
        "track_id": int(getattr(chunk, "track_id", 0) or 0),
        "start_time_sec": float(getattr(chunk, "start_sec", 0.0) or 0.0),
        "analysis": {
            "duration_sec": duration,
            "stable_bpm": stable_bpm,
            "key": key,
            "loudness_lufs_total": lufs_total,
            "loudness_lufs_bass": lufs_bass,
            "average_brightness": brightness,
            "has_vocals": has_vocals,
            "vocal_energy_rms": vocal_rms,
        },
    }
