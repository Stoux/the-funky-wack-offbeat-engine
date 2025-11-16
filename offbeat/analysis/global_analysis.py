"""Global analysis (beats + BPM curve).

Loads audio with librosa, trims leading silence, computes a global beat grid
and a downsampled BPM curve. Returns a context reused by later phases.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import time

import numpy as np
import librosa
try:
    from librosa.feature.rhythm import tempo as lr_tempo  # librosa >= 0.10
except Exception:  # pragma: no cover - older librosa
    lr_tempo = librosa.beat.tempo

from ..config import settings
from ..logging_utils import get_logger



def _overlap_fade_windows(fade: int):
    if fade <= 0:
        return None, None
    fi = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade, dtype=np.float32))
    return fi, fi[::-1]


def _separate_chunked(sep, stereo: np.ndarray, sr: int, chunk_sec: int, overlap_sec: int, supports_sr_kw: bool, log) -> dict:
    """Chunked Spleeter separation with overlap-add (mono per stem).
    - Processes stereo (N,2) in windows of chunk_sec with overlap_sec.
    - Calls sep.separate() per chunk (with/without sample_rate kwarg).
    - Folds each returned stem to mono and overlap-adds into full-length arrays.
    Returns: dict[str, np.ndarray] mapping stem name -> mono waveform (float32, len N).
    """
    n = int(stereo.shape[0])
    if n == 0:
        return {}
    win = int(max(1, chunk_sec) * sr)
    fade = int(max(0, overlap_sec) * sr)
    hop = max(1, win - fade)
    fi, fo = _overlap_fade_windows(fade)

    # Prepare accumulators lazily when first chunk returns keys
    acc: dict[str, np.ndarray] = {}
    weight = np.zeros((n,), dtype=np.float32)

    start = 0
    chunk_idx = 0
    while start < n:
        end = min(n, start + win)
        chunk = stereo[start:end, :]
        # Call Spleeter
        pred = sep.separate(chunk, sample_rate=sr) if supports_sr_kw else sep.separate(chunk)
        if not isinstance(pred, dict):
            raise RuntimeError("Spleeter returned unexpected type for chunk")
        # Initialize accumulators for keys on first chunk
        if not acc:
            for k in pred.keys():
                acc[k] = np.zeros((n,), dtype=np.float32)
        # Fold to mono and place with fades
        for k, v in pred.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 2 and v.shape[1] == 2:
                    mono = v.mean(axis=1).astype(np.float32)
                elif v.ndim == 1:
                    mono = v.astype(np.float32)
                else:
                    # Try mean over smallest axis
                    mono = v.mean(axis=int(np.argmin(v.shape))).astype(np.float32)
            else:
                continue
            m = mono.shape[0]
            if fade > 0 and m > 0:
                if start > 0:
                    mono[:min(fade, m)] *= fi[:min(fade, m)]
                if end < n:
                    w = min(fade, m)
                    mono[m - w:m] *= fo[fade - w:fade]
            acc[k][start:start + m] += mono
        # Weighting for overlaps
        wv = np.ones((end - start,), dtype=np.float32)
        if fade > 0:
            if start > 0:
                w = min(fade, wv.size)
                wv[:w] *= fi[:w]
            if end < n:
                w = min(fade, wv.size)
                wv[-w:] *= fo[fade - w:fade]
        weight[start:end] += wv

        chunk_idx += 1
        log.info("spleeter: processed chunk %d start=%.2fs end=%.2fs", chunk_idx, start/float(sr), end/float(sr))
        start += hop

    # Normalize by weight
    weight[weight == 0] = 1.0
    for k in list(acc.keys()):
        acc[k] = (acc[k] / weight).astype(np.float32)
    log.info("spleeter: chunked separation complete; chunks=%d", chunk_idx)
    return acc


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
    stems_sr: Optional[Dict[str, int]] = None  # true sample rates per stem name
    # Guide tempo derived from full mix (BPM)
    t_global: Optional[float] = None
    # Drum-based beat grid in absolute seconds (from untrimmed stems)
    drum_beat_times: Optional[List[float]] = None


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
    log = get_logger("analysis.global_analysis", job_id=str(job.get("job_id") or job.get("id") or job.get("relative_path") or job.get("file_path") or "?"))

    # Resolve absolute input path based on v2 (prefer relative_path under shared mount)
    rel = str(job.get("relative_path", "") or "")
    file_path = None
    if rel:
        file_path = os.path.join(settings.shared_mount_path, rel)
    else:
        file_path = job.get("file_path")
    if not file_path or not os.path.isfile(file_path):
        log.error("Input file missing: rel=%r path=%r", rel, file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load mono at target SR
    target_sr = int(getattr(settings, "sample_rate", 22050))
    log.info("Loading audio mono sr=%d path=%s", target_sr, file_path)
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    log.info("Loaded audio: sr=%d samples=%d dur=%.2fs", sr, len(y), (len(y)/sr) if sr else -1.0)

    # Trim leading/trailing silence per v2
    try:
        y_trim, (start_idx, end_idx) = librosa.effects.trim(y, top_db=int(getattr(settings, "silence_top_db", 40)))
        time_offset_sec = float(start_idx) / float(sr)
        log.info("Trimmed silence: start_idx=%d end_idx=%d offset=%.3fs", start_idx, end_idx, time_offset_sec)
    except Exception:
        log.exception("Silence trim failed; using untrimmed audio")
        y_trim = y
        time_offset_sec = 0.0
    duration_sec = float(len(y_trim)) / float(sr)

    # Beat tracking
    log.info("Beat tracking with librosa.beat.beat_track")
    tempo, beat_frames = librosa.beat.beat_track(y=y_trim, sr=sr)
    beat_times_rel = librosa.frames_to_time(beat_frames, sr=sr)
    beat_times_abs = (beat_times_rel + time_offset_sec).astype(float).tolist()
    if len(beat_times_rel) < 4:
        log.warning("Insufficient beats detected: count=%d; will use onset-based fallback", len(beat_times_rel))

    # Global guide tempo (T_global) using finer hop for resolution
    try:
        hop_glob = 256
        onset_env_full = librosa.onset.onset_strength(y=y_trim, sr=sr, hop_length=hop_glob)
        tg_arr = lr_tempo(onset_envelope=onset_env_full, sr=sr, hop_length=hop_glob, aggregate=np.median)
        t_global = float(tg_arr[0]) if hasattr(tg_arr, "__len__") else float(tg_arr)
        log.info("Global guide tempo (T_global)=%.2f BPM", t_global)
    except Exception:
        log.exception("Failed to compute T_global; continuing without guide tempo")
        t_global = None

    # v2: attempt full-file Spleeter separation and save stems as WAV
    stems_arrays: Optional[Dict[str, np.ndarray]] = None
    stems_paths: Dict[str, str] = {}
    stems_sr_map: Optional[Dict[str, int]] = None

    # Pre-log helpful context for troubleshooting
    try:
        log.info(
            "Spleeter preflight: sr=%s y_trim.shape=%s dtype=%s duration=%.2fs rel=%r shared_mount=%r",
            sr,
            getattr(y_trim, "shape", None),
            getattr(y_trim, "dtype", None),
            duration_sec,
            rel or None,
            getattr(settings, "shared_mount_path", None),
        )
    except Exception:
        # Don't fail on logging errors
        pass

    try:
        try:
            from spleeter.separator import Separator  # type: ignore
        except Exception as e:
            log.exception("Spleeter import failed. Is 'spleeter' installed with TensorFlow? Error=%s", e)
            raise

        try:
            sep = Separator("spleeter:4stems")
            log.debug("Spleeter Separator created with 4stems model")
        except Exception as e:
            log.exception("Failed to create Spleeter Separator. Error=%s", e)
            raise

        # Spleeter: use full-file stereo at 44.1 kHz for better quality and model compatibility
        try:
            # Load original file as stereo at native SR (untrimmed)
            y_native, sr_native = librosa.load(file_path, sr=None, mono=False)
            if not isinstance(y_native, np.ndarray):
                raise TypeError("Loaded audio is not a numpy array")

            # librosa with mono=False returns (C, N)
            if y_native.ndim == 1:
                # mono (N,) → channels-first (2, N) by duplicating
                channels_first = np.stack([y_native, y_native], axis=0)
            elif y_native.ndim == 2:
                channels_first = y_native  # (C, N)
                C, N = channels_first.shape
                if C == 1:
                    channels_first = np.vstack([channels_first, channels_first])  # (2, N)
                elif C > 2:
                    channels_first = channels_first[:2, :]  # keep only first 2 channels
            else:
                raise ValueError(f"Unexpected audio array ndim={y_native.ndim}")

            # Resample per channel to Spleeter SR (44100)
            spleeter_sr = 44100
            if int(sr_native) != spleeter_sr:
                resampled_chans = [
                    librosa.resample(channels_first[i, :], orig_sr=int(sr_native), target_sr=spleeter_sr)
                    for i in range(2)
                ]
                # Pad/truncate to equal length
                max_len = max(len(c) for c in resampled_chans)
                resampled_chans = [
                    np.pad(c, (0, max_len - len(c)), mode="edge") if len(c) < max_len else c[:max_len]
                    for c in resampled_chans
                ]
                channels_first = np.stack(resampled_chans, axis=0)  # (2, N')

            # Convert to (N, 2) float32 for Spleeter
            stereo = channels_first.T.astype(np.float32)
            log.debug(
                "Prepared stereo for Spleeter: in_shape=%s native_sr=%s out_shape=%s out_sr=%s dur=%.2fs",
                getattr(y_native, "shape", None), sr_native, getattr(stereo, "shape", None), spleeter_sr,
                (stereo.shape[0] / float(spleeter_sr)) if spleeter_sr and isinstance(stereo, np.ndarray) else -1.0,
            )
        except Exception as e:
            log.exception("Failed to prepare stereo array for Spleeter. Error=%s", e)
            raise

        try:
            # Some Spleeter versions accept sample_rate kwarg; others don't. Prefer feature test then fallback.
            import inspect
            # Feature-detect whether this Spleeter version supports the 'sample_rate' kwarg
            try:
                sep_sig = inspect.signature(sep.separate)
            except Exception:
                sep_sig = None
            supports_sr_kw = bool(sep_sig and "sample_rate" in sep_sig.parameters)

            # Chunked separation to cap memory usage; keep 4 stems
            chunk_sec = int(getattr(settings, "spleeter_chunk_sec", 60))
            overlap_sec = int(getattr(settings, "spleeter_overlap_sec", 5))
            log.info(
                "Spleeter chunked mode: chunk_sec=%ds overlap_sec=%ds supports_sr_kw=%s", chunk_sec, overlap_sec, supports_sr_kw
            )
            pred = _separate_chunked(sep, stereo, spleeter_sr, chunk_sec=chunk_sec, overlap_sec=overlap_sec, supports_sr_kw=supports_sr_kw, log=log)

            if not isinstance(pred, dict) or not pred:
                raise RuntimeError("Spleeter returned no stems")
            # Convert stem arrays to mono robustly regardless of layout, with diagnostics
            stems_arrays = {}
            for k, v in pred.items():
                if not isinstance(v, np.ndarray):
                    log.warning("Stem '%s' value is not ndarray: %r", k, type(v))
                    continue
                log.debug("Raw stem '%s' shape=%s dtype=%s", k, getattr(v, "shape", None), getattr(v, "dtype", None))
                mono: Optional[np.ndarray]
                if v.ndim == 1:
                    mono = v
                elif v.ndim == 2:
                    # Prefer averaging along the axis that looks like channels (size==2 or the smaller axis)
                    if v.shape[1] == 2:
                        mono = v.mean(axis=1)
                    elif v.shape[0] == 2:
                        mono = v.mean(axis=0)
                    else:
                        chan_axis = int(np.argmin(v.shape))
                        mono = v.mean(axis=chan_axis)
                    # If something went wrong and we produced a tiny vector, try the other axis
                    if mono.size <= 2:
                        other_axis = 0 if (v.ndim == 2 and (v.shape[1] == 2 or int(np.argmin(v.shape)) == 1)) else 1
                        try:
                            alt = v.mean(axis=other_axis)
                            if alt.size > mono.size:
                                log.warning("Mono fold for stem '%s' looked wrong (size=%d). Using alternate axis result size=%d.", k, mono.size, alt.size)
                                mono = alt
                        except Exception:
                            pass
                else:
                    log.warning("Unexpected stem array shape for '%s': %s", k, getattr(v, "shape", None))
                    continue
                stems_arrays[k] = mono.astype(np.float32)
                log.debug("Folded stem '%s' to mono shape=%s seconds=%.2f", k, getattr(mono, "shape", None), (mono.size/float(44100)))
            stem_sr = spleeter_sr
            # Record per-stem sample rates (all equal to Spleeter's output rate)
            stems_sr_map = {k: int(stem_sr) for k in stems_arrays.keys()}
            log.info("Spleeter separation done. Stems=%s sr=%d", sorted(stems_arrays.keys()), stem_sr)
        except Exception as e:
            log.exception("Spleeter separation step failed. Error=%s", e)
            raise

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
            out_rel = os.path.join(base_rel_dir, f"{base_name}_{name}_global.wav")
            out_abs = os.path.join(settings.shared_mount_path, out_rel)
            os.makedirs(os.path.dirname(out_abs), exist_ok=True)

            # Sanity: compute duration
            n = int(arr.size) if isinstance(arr, np.ndarray) else 0
            dur = (n / float(stem_sr)) if stem_sr and n else 0.0
            log.debug(
                "Preparing to write stem '%s': shape=%s dtype=%s seconds=%.3f path=%s",
                name,
                getattr(arr, "shape", None),
                getattr(arr, "dtype", None),
                dur,
                out_abs,
            )

            if not (isinstance(arr, np.ndarray) and arr.size > int(stem_sr)):
                # Require at least 1 second of audio to avoid tiny corrupted files
                log.warning("Stem '%s' too short or invalid (shape=%s seconds=%.3f); skipping write", name, getattr(arr, "shape", None), dur)
            else:
                try:
                    import wave
                    # Convert float32 [-1,1] to int16 PCM
                    data = np.clip(arr.astype(np.float32), -1.0, 1.0)
                    pcm16 = (data * 32767.0).astype(np.int16)
                    log.info("About to write WAV stem '%s' to %s", name, out_abs)
                    with wave.open(out_abs, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(int(stem_sr))
                        wf.writeframes(pcm16.tobytes())
                    log.info("Wrote stem '%s' to %s (sr=%s shape=%s seconds=%.3f)", name, out_abs, stem_sr, getattr(arr, 'shape', None), dur)
                except Exception as e:
                    log.exception("Failed writing stem '%s' to %s. Error=%s", name, out_abs, e)
            return out_rel

        # Save all four
        for k in ("vocals", "drums", "bass", "other"):
            arr = stems_arrays.get(k) if stems_arrays else None
            if isinstance(arr, np.ndarray) and arr.size:
                stems_paths[k] = _save_stem(k, arr)
            else:
                log.warning("Stem '%s' missing or empty; not saving", k)
    except Exception:
        # Spleeter or save failed; keep stems empty
        log.exception("Spleeter separation/save failed; continuing without stems")
        stems_arrays = None
        stems_paths = {}

    log.info("Global analysis pre-BPM phase reached; stems_paths=%s", stems_paths)

    # Optional: compute drum-based beat grid directly from drums stem (absolute times)
    drum_beat_times_abs: List[float] = []
    try:
        y_drums = stems_arrays.get("drums") if isinstance(stems_arrays, dict) else None
        if isinstance(y_drums, np.ndarray) and y_drums.size > 0:
            try:
                sr_drums = int(stem_sr)  # set in Spleeter block
            except Exception:
                sr_drums = sr  # fallback to mix SR if unavailable
            # Ensure mono float array
            yd = y_drums.astype(float, copy=False)
            _tempo_d, drum_frames = librosa.beat.beat_track(y=yd, sr=sr_drums)
            drum_times = librosa.frames_to_time(drum_frames, sr=sr_drums)
            drum_beat_times_abs = drum_times.astype(float).tolist()
            log.info("Computed drum-based beat grid: count=%d", len(drum_beat_times_abs))
        else:
            log.info("No drums stem available for drum-based beat grid")
    except Exception:
        log.exception("Drum beat tracking failed; continuing without drum timestamps")

    # Prefer beat-interval derived BPM (more stable than onset tempogram)
    hop_length = int(getattr(settings, "hop_length", 512))

    bpm_curve: List[float] = []

    # Primary: PLP-based dynamic tempo curve (time-varying guide)
    try:
        # Prefer drums stem for PLP if available; fallback to full mix
        y_plp = y_trim
        sr_plp = sr
        try:
            if isinstance(stems_arrays, dict) and isinstance(stems_arrays.get("drums"), np.ndarray) and stems_arrays.get("drums").size:
                y_plp = stems_arrays.get("drums")  # mono already
                try:
                    sr_plp = int(stem_sr)  # set by spleeter block
                except Exception:
                    sr_plp = sr
                log.info("PLP source: using drums stem (sr=%s size=%s)", sr_plp, getattr(y_plp, "size", None))
            else:
                log.info("PLP source: using full mix (sr=%s size=%s)", sr_plp, getattr(y_plp, "size", None))
        except Exception:
            log.info("PLP source selection failed; using full mix")
            y_plp = y_trim
            sr_plp = sr

        # Adaptive hop_length to bound frame count for long sets
        def _choose_hop(len_samples: int, sr_local: int, target_max_frames: int = 100_000) -> int:
            # Try 256, 512, 1024, 2048 and pick the smallest hop with frames <= cap
            for h in (256, 512, 1024, 2048):
                frames = int(np.ceil(len_samples / float(h))) if h > 0 else 0
                if frames <= target_max_frames:
                    return h
            return 2048

        hop_plp = _choose_hop(len(y_plp), sr_plp)
        kwargs_plp = dict(tempo_min=60, tempo_max=200, win_length=256)

        # Preflight diagnostics
        try:
            import librosa as _lb
            lb_ver = getattr(_lb, "__version__", "?")
        except Exception:
            lb_ver = "?"
        est_frames = int(np.ceil(len(y_plp) / hop_plp)) if sr_plp else -1
        log.info(
            "PLP preflight: sr=%d dur=%.2fs hop=%d est_frames=%d librosa=%s source=%s",
            sr_plp, (len(y_plp)/float(sr_plp)) if sr_plp else -1.0, hop_plp, est_frames, lb_ver,
            "drums" if y_plp is not y_trim else "mix",
        )

        # Onset envelope and stats
        oenv_plp = librosa.onset.onset_strength(y=y_plp, sr=sr_plp, hop_length=hop_plp)
        try:
            o_min = float(np.nanmin(oenv_plp))
            o_max = float(np.nanmax(oenv_plp))
            o_mean = float(np.nanmean(oenv_plp))
            o_nans = int(np.isnan(oenv_plp).sum())
            log.info("PLP onset env stats: shape=%s min=%.3g max=%.3g mean=%.3g nan_ct=%d", getattr(oenv_plp, "shape", None), o_min, o_max, o_mean, o_nans)
        except Exception:
            pass
        # Sanitize envelope to avoid NaN/Inf issues in PLP
        if np.isnan(oenv_plp).any() or np.isinf(oenv_plp).any():
            nan_ct = int(np.isnan(oenv_plp).sum())
            log.warning("PLP onset envelope had invalid values; sanitizing (nan_ct=%d)", nan_ct)
            oenv_plp = np.nan_to_num(oenv_plp, nan=0.0, posinf=0.0, neginf=0.0)
            # Rebase to >=0
            try:
                oenv_plp = oenv_plp - float(np.min(oenv_plp))
            except Exception:
                pass

        # Optional: PLP self-test on synthetic click (diagnostics only)
        try:
            if bool(getattr(settings, "plp_selftest", False)) or bool(getattr(settings, "plp_diag", False)):
                import math
                dur_test = 5.0
                sr_test = int(sr_plp)
                t = np.arange(int(dur_test * sr_test)) / float(sr_test)
                # 2 Hz click => 120 BPM
                y_click = np.zeros_like(t, dtype=np.float32)
                y_click[(np.arange(int(dur_test*2)) * int(sr_test/2)).clip(max=y_click.size-1)] = 1.0
                oenv_test = librosa.onset.onset_strength(y=y_click, sr=sr_test, hop_length=hop_plp)
                log.info("PLP selftest: oenv_test shape=%s min=%.3g max=%.3g", getattr(oenv_test, 'shape', None), float(np.min(oenv_test)), float(np.max(oenv_test)))
                # Try a minimal PLP call to ensure runtime path works
                try:
                    _ = librosa.beat.plp(onset_envelope=oenv_test, sr=sr_test, hop_length=hop_plp)
                    log.info("PLP selftest: librosa.beat.plp(minimal) succeeded")
                except Exception as e_st:
                    log.warning("PLP selftest failed on librosa.beat.plp(minimal): %s: %s", type(e_st).__name__, str(e_st))
        except Exception:
            pass

        # Discover supported kwargs via signature to avoid TypeError
        import inspect
        def _filter_kwargs(func, kwargs: dict) -> dict:
            try:
                sig = inspect.signature(func)
                allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
                return allowed
            except Exception:
                return kwargs

        # Try legacy API first, with signature-aware kwargs; then feature.rhythm.plp
        plp = None
        used_method = None
        try:
            try:
                kw1 = _filter_kwargs(librosa.beat.plp, kwargs_plp)
                plp = librosa.beat.plp(onset_envelope=oenv_plp, sr=sr_plp, hop_length=hop_plp, **kw1)
                used_method = f"librosa.beat.plp(kwargs:{list(kw1.keys())})"
                log.debug("librosa.beat.plp with filtered kwargs succeeded (shape=%s)", getattr(plp, "shape", None))
            except TypeError as e_kw:
                log.warning("librosa.beat.plp rejected kwargs (TypeError: %s); retrying without kwargs", str(e_kw))
                plp = librosa.beat.plp(onset_envelope=oenv_plp, sr=sr_plp, hop_length=hop_plp)
                used_method = "librosa.beat.plp(minimal)"
                log.debug("librosa.beat.plp without kwargs succeeded (shape=%s)", getattr(plp, "shape", None))
        except Exception as e1:
            log.warning("librosa.beat.plp failed (%s: %s); trying librosa.feature.rhythm.plp fallback", type(e1).__name__, str(e1))
            try:
                from librosa.feature.rhythm import plp as plp_new
                try:
                    kw2 = _filter_kwargs(plp_new, kwargs_plp)
                    plp = plp_new(onset_envelope=oenv_plp, sr=sr_plp, hop_length=hop_plp, **kw2)
                    used_method = f"feature.rhythm.plp(kwargs:{list(kw2.keys())})"
                    log.info("Used librosa.feature.rhythm.plp with filtered kwargs")
                except TypeError as e_kw2:
                    log.warning("feature.rhythm.plp rejected kwargs (TypeError: %s); retrying without kwargs", str(e_kw2))
                    plp = plp_new(onset_envelope=oenv_plp, sr=sr_plp, hop_length=hop_plp)
                    used_method = "feature.rhythm.plp(minimal)"
                    log.info("Used librosa.feature.rhythm.plp without kwargs")
            except Exception:
                plp = None

        bpm_series = None
        frame_count = None
        if plp is not None:
            # Convert PLP to dominant BPM per frame with robust guards to avoid IndexError
            try:
                # Ensure 2D shape (bins x frames)
                # Normalize dimensionality; prefer shape=(bins, frames)
                if isinstance(plp, np.ndarray) and plp.ndim == 1:
                    # Interpret 1D as frames along time; set bins=1
                    plp = plp.reshape((1, -1))
                if not isinstance(plp, np.ndarray) or plp.ndim != 2 or plp.size == 0:
                    raise RuntimeError(f"Invalid PLP array shape={getattr(plp, 'shape', None)}")

                orig_shape = tuple(plp.shape)
                bins = int(plp.shape[0])
                frames = int(plp.shape[1])
                # If frames collapsed to 1 but we expected many, it's likely transposed; fix by transpose
                if frames <= 1 and bins > 4:
                    try:
                        # Use precomputed estimate to decide
                        if isinstance(est_frames, int) and est_frames > 4:
                            plp = plp.T
                            log.info("PLP array transposed from %s to %s to correct frames dimension (est_frames=%s)", orig_shape, tuple(plp.shape), est_frames)
                            bins = int(plp.shape[0])
                            frames = int(plp.shape[1])
                    except Exception:
                        pass

                bpm_grid = librosa.tempo_frequencies(bins, sr=sr_plp, hop_length=hop_plp)
                idx = np.argmax(plp, axis=0)
                # Diagnostics for indices
                try:
                    idx_min = int(np.min(idx)) if idx.size else -1
                    idx_max = int(np.max(idx)) if idx.size else -1
                    log.debug("PLP argmax indices: min=%s max=%s bins=%d frames=%d", idx_min, idx_max, bins, frames)
                except Exception:
                    pass
                # Safe take to avoid IndexError on rare boundary conditions
                bpm_series = np.take(bpm_grid, idx, mode='clip').astype(float)
                frame_count = frames
                log.info("PLP succeeded via %s; frames=%d bins=%d (orig_shape=%s)", used_method, frame_count, bins, orig_shape)
            except Exception as e_conv:
                # Re-raise as RuntimeError so outer handler logs and manages fallback/diagnostics
                raise RuntimeError(f"PLP-to-BPM conversion failed: {type(e_conv).__name__}: {str(e_conv)}")
        else:
            # As a robust fallback, compute a tempogram-based dynamic tempo curve
            try:
                tempogram = librosa.feature.tempogram(onset_envelope=oenv_plp, sr=sr_plp, hop_length=hop_plp)
                if isinstance(tempogram, np.ndarray) and tempogram.size:
                    bpm_grid = librosa.tempo_frequencies(tempogram.shape[0], sr=sr_plp, hop_length=hop_plp)
                    idx = np.argmax(tempogram, axis=0)
                    bpm_series = bpm_grid[idx].astype(float)
                    frame_count = tempogram.shape[1]
                    used_method = "tempogram"
                    log.info("Tempogram-based tempo curve computed as PLP fallback: frames=%d bins=%d", frame_count, tempogram.shape[0])
                else:
                    raise RuntimeError("Empty tempogram result")
            except Exception as e_tmp:
                # Let outer except handle ultimate fallback
                raise RuntimeError(f"Tempogram fallback failed: {type(e_tmp).__name__}: {str(e_tmp)}")

        # Fold BPMs into plausible musical range [60, 200]
        def _fold(b: float) -> float:
            if not np.isfinite(b) or b <= 0:
                return 0.0
            while b < 60.0:
                b *= 2.0
            while b > 200.0:
                b *= 0.5
            return b
        bpm_series = np.array([_fold(float(x)) for x in bpm_series], dtype=float)

        # Reconcile each frame w.r.t reference (prefer t_global) to reduce half/double slips
        ref = None
        try:
            ref = float(t_global) if (t_global is not None and np.isfinite(t_global) and t_global > 0) else None
        except Exception:
            ref = None
        if ref is not None:
            def _reconcile_to_ref(b: float, ref_val: float) -> float:
                cands = [b * 0.5, b * (2.0 / 3.0), b, b * 1.5, b * 2.0]
                cands = [c for c in cands if 60.0 <= c <= 200.0]
                return float(min(cands, key=lambda c: abs(c - ref_val))) if cands else b
            bpm_series = np.array([_reconcile_to_ref(float(x), ref) for x in bpm_series], dtype=float)
            log.info("Applied %s reconciliation toward ref=%.2f for %d frames", used_method or "PLP", ref, bpm_series.size)

        # Frame times to absolute seconds (align to global timeline)
        frame_times_abs = librosa.frames_to_time(np.arange(frame_count), sr=sr_plp, hop_length=hop_plp) + time_offset_sec
        # Resample to per-second grid over trimmed duration
        bpm_curve = _per_second_resample(frame_times_abs.astype(float), bpm_series.astype(float), duration_sec)
        if bpm_curve:
            log.info("%s-based tempo curve computed: points=%d (hop=%d source=%s)", (used_method or "PLP").upper(), len(bpm_curve), hop_plp, "drums" if y_plp is not y_trim else "mix")
    except Exception as e:
        log.exception("PLP tempo curve computation failed; will fall back to beat/onset methods (exc=%s)", type(e).__name__)

    if (not bpm_curve) and len(beat_times_rel) >= 4:
        # Instantaneous tempo from consecutive beat intervals
        intervals = np.diff(beat_times_rel)  # seconds per beat
        # Guard against zeros
        intervals = intervals[intervals > 1e-6]
        if intervals.size >= 1:
            inst_bpm = 60.0 / intervals
            # Reference BPM: prefer T_global if available, else median of instantaneous BPMs
            if t_global is not None and np.isfinite(t_global) and t_global > 0:
                ref_bpm = float(t_global)
                log.info("Beat-interval fallback: using T_global=%.2f as reference for folding", ref_bpm)
            else:
                ref_bpm = float(np.median(inst_bpm)) if inst_bpm.size else None
                if ref_bpm is not None:
                    log.info("Beat-interval fallback: using median(inst_bpm)=%.2f as reference for folding", ref_bpm)

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
        bpm_series = lr_tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length, aggregate=None)
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
        stems_sr=stems_sr_map,
        t_global=t_global,
        drum_beat_times=drum_beat_times_abs,
    )
