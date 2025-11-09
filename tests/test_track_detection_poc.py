import os
import tempfile
import wave
import struct
import math

from offbeat.analysis.global_analysis import run_global_analysis
from offbeat.analysis.track_detection import detect_tracks
from offbeat.worker import chunks_to_track_results


def _write_tiny_wav(path: str, seconds: float = 0.6, sr: int = 8000) -> None:
    n_samples = int(seconds * sr)
    freq = 330.0
    amp = 0.25
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        for i in range(n_samples):
            val = amp * math.sin(2 * math.pi * freq * i / sr)
            sample = int(max(-1.0, min(1.0, val)) * 32767)
            wf.writeframes(struct.pack('<h', sample))


def test_detect_tracks_returns_single_full_span_chunk_with_positive_duration():
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "tiny.wav")
        _write_tiny_wav(wav_path, seconds=0.6)
        job = {"job_id": 11, "file_path": wav_path}

        ctx = run_global_analysis(job)
        chunks = detect_tracks(ctx, job)
        assert len(chunks) == 1
        ch = chunks[0]
        # end should be >= start and > start for positive duration
        assert ch.end_sec >= ch.start_sec

        # Convert into TrackResult to check computed analysis duration
        tracks = chunks_to_track_results(chunks, ctx, job)
        assert len(tracks) == 1
        tr = tracks[0]
        assert tr.analysis.duration_sec is not None
        assert tr.analysis.duration_sec >= 0.0
        # Expect positive duration for a valid tiny wav
        assert tr.analysis.duration_sec > 0.0
