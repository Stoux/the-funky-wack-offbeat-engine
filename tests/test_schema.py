from offbeat.schema import JobPayload, CueTrack
import pytest


def test_jobpayload_valid_with_cue_tracks():
    payload = {
        "job_id": 123,
        "file_path": "/tmp/audio.wav",
        "cue_tracks": [
            {"title": "Intro", "start_time_sec": 2.5},
            {"title": "Next", "start_time_sec": 600.0},
        ],
    }
    jp = JobPayload(**payload)
    assert jp.job_id == 123
    assert jp.file_path.endswith("audio.wav")
    assert isinstance(jp.cue_tracks, list)
    assert isinstance(jp.cue_tracks[0], CueTrack)


def test_jobpayload_invalid_file_path():
    with pytest.raises(Exception):
        JobPayload(job_id=1, file_path="")

    with pytest.raises(Exception):
        JobPayload(job_id=1, file_path=None)  # type: ignore[arg-type]
