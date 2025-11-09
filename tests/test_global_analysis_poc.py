import os
import tempfile
import wave
import struct
import math
from offbeat.analysis.global_analysis import run_global_analysis


def _write_tiny_wav(path: str, seconds: float = 0.5, sr: int = 8000) -> None:
    n_samples = int(seconds * sr)
    freq = 220.0
    amp = 0.3
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        for i in range(n_samples):
            sample = int(max(-1.0, min(1.0, amp * math.sin(2 * math.pi * freq * i / sr))) * 32767)
            wf.writeframes(struct.pack('<h', sample))


def test_run_global_analysis_returns_duration():
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "tiny.wav")
        _write_tiny_wav(wav_path, seconds=0.5)
        job = {"job_id": 7, "file_path": wav_path}
        ctx = run_global_analysis(job)
        assert ctx.duration_sec is not None and ctx.duration_sec > 0
        assert ctx.trimmed_start_sec == 0.0
        assert ctx.analysis_mode in ("pure_audio_guess", "cue_correlated")
