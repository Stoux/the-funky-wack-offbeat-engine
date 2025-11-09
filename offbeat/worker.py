from __future__ import annotations

import signal
import time
from typing import Any, Dict, List

from loguru import logger

from .logging import setup_logging
from .config import settings
from .schema import (
    JobPayload,
    FinalResult,
    GlobalResult,
    TrackResult,
    TransitionPeriod,
    TrackAnalysis,
)
from .redis_client import pop_job_blocking, push_result
from .analysis.global_analysis import run_global_analysis
from .analysis.track_detection import detect_tracks


def _to_global_result(ctx: Any) -> GlobalResult:
    """Convert a GlobalContext-like object into a GlobalResult Pydantic model."""
    # Supports either a dataclass with attributes or already a GlobalResult
    if isinstance(ctx, GlobalResult):
        return ctx
    # Fallback: read attributes from stub dataclass
    return GlobalResult(
        duration_sec=getattr(ctx, "duration_sec", None),
        trimmed_start_sec=getattr(ctx, "trimmed_start_sec", None),
        global_beat_grid_timestamps=list(getattr(ctx, "beat_grid_times", []) or []),
        global_bpm_curve=list(getattr(ctx, "bpm_curve", []) or []),
        analysis_mode=getattr(ctx, "analysis_mode", "pure_audio_guess"),
    )


def chunks_to_track_results(chunks: List[Any], ctx: Any, job: Dict[str, Any]) -> List[TrackResult]:
    results: List[TrackResult] = []
    for ch in chunks:
        start = float(getattr(ch, "start_sec", 0.0) or 0.0)
        end = float(getattr(ch, "end_sec", start) or start)
        duration = max(0.0, end - start)
        trans = getattr(ch, "transition", None) or {}
        transition = TransitionPeriod(
            start=trans.get("start"),
            end=trans.get("end"),
            duration=trans.get("duration"),
        ) if isinstance(trans, dict) else TransitionPeriod(start=None, end=None, duration=None)
        analysis = TrackAnalysis(
            duration_sec=duration,
            key=None,
            loudness_lufs_total=None,
            loudness_lufs_bass=None,
            average_brightness=None,
            has_vocals=None,
            vocal_energy_rms=None,
        )
        title = getattr(ch, "title", None) or (
            (job.get("cue_tracks") or [{}])[0].get("title")
            or ("Guessed Track 1" if getattr(ctx, "analysis_mode", "pure_audio_guess") == "pure_audio_guess" else "Track 1 (from cue)")
        )
        results.append(
            TrackResult(
                track_id=int(getattr(ch, "track_id", len(results)) or len(results)),
                title=title,
                cue_start_time_sec=start,
                transition_period_sec=transition,
                analysis=analysis,
            )
        )
    return results


def assemble_result(job: Dict[str, Any], global_ctx: Any, tracks: List[TrackResult]) -> Dict[str, Any]:
    global_result = _to_global_result(global_ctx)
    return FinalResult(
        job_id=job["job_id"],
        status="completed",
        results={
            "global": global_result.dict(),
            "tracks": [t.dict() for t in tracks],
        },
    ).dict()


_running = True


def _handle_sigterm(signum, frame):
    global _running
    logger.info("Received shutdown signal, stopping worker loop...")
    _running = False


def main():
    setup_logging()
    logger.info("Starting The Offbeat Engine worker (scaffold/core infra)...")
    logger.info(f"Redis URL: {settings.redis_url}")
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    while _running:
        try:
            popped = pop_job_blocking(timeout=5)
            if not popped:
                continue  # timeout, loop again
            queue_name, job = popped
            if job is None:
                # Malformed JSON; push a failed result placeholder and continue
                logger.error("Received malformed job payload (JSON parse error)")
                continue

            # Validate job
            try:
                jp = JobPayload(**job)
            except Exception as e:
                logger.error(f"Job validation failed: {e}")
                push_result(
                    {
                        "job_id": job.get("job_id", -1),
                        "status": "failed",
                        "error": f"validation_error: {e}",
                    }
                )
                continue

            start_time = time.time()
            logger.info(f"Processing job_id={jp.job_id} from queue={queue_name}")

            # Phase 1: Global (module stub)
            global_ctx = run_global_analysis(job)
            # Phase 2: Track detection (module stub)
            chunks = detect_tracks(global_ctx, job)
            tracks = chunks_to_track_results(chunks, global_ctx, job)

            # Assemble and push
            final = assemble_result(job, global_ctx, tracks)
            push_result(final)
            elapsed = time.time() - start_time
            logger.info(f"Completed job_id={jp.job_id} in {elapsed:.2f}s")

        except Exception as e:
            logger.exception("Unexpected error in worker loop")
            # Attempt to push a failure result if we have a job context
            try:
                job_id = job.get("job_id", -1) if isinstance(job, dict) else -1
                push_result({"job_id": job_id, "status": "failed", "error": str(e)})
            except Exception:
                # Ignore if we can't even push result
                pass
            # Brief backoff
            time.sleep(1)

    logger.info("Worker stopped.")


if __name__ == "__main__":
    main()
