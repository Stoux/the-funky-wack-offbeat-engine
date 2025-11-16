import json
from offbeat.analysis.global_analysis import GlobalContext
from offbeat.worker import assemble_result, chunks_to_track_results
from offbeat.analysis.track_detection import Chunk


def test_assemble_result_shape_pure_audio_mode():
    job = {"job_id": 42, "file_path": "/tmp/a.wav"}
    ctx = GlobalContext(
        duration_sec=None,
        trimmed_start_sec=None,
        beat_grid_times=[],
        bpm_curve=[],
        threads=3,
    )

    chunks = [Chunk(track_id=0, title="Guessed Track 1", start_sec=0.0, end_sec=0.0, transition=None)]
    tracks = chunks_to_track_results(chunks, ctx, job)
    final = assemble_result(job, ctx, tracks)

    assert final["job_id"] == 42
    assert final["status"] == "completed"
    assert "results" in final
    assert "global" in final["results"]
    assert "tracks" in final["results"]
    # validate some keys exist
    g = final["results"]["global"]
    assert set(["duration_sec", "trimmed_start_sec", "global_beat_grid_timestamps", "global_drum_beat_grid_timestamps", "stems"]).issubset(g.keys())
    assert isinstance(final["results"]["tracks"], list)
