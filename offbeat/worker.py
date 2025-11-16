from __future__ import annotations

import signal
import time
from typing import Any, Dict, List, Iterable

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
        stems=dict(getattr(ctx, "stems", {}) or {}),
        global_bpm_curve=[],
        analysis_mode="pure_audio_guess",
    )


def chunks_to_track_results(chunks: List[Any], ctx: Any, job: Dict[str, Any]) -> List[TrackResult]:
    # Ensure chronological order by start time, break ties by end time
    ordered = sorted(
        list(chunks or []),
        key=lambda c: (
            float(getattr(c, "start_sec", 0.0) or 0.0),
            float(getattr(c, "end_sec", 0.0) or 0.0),
        ),
    )
    results: List[TrackResult] = []
    for idx, ch in enumerate(ordered):
        start = float(getattr(ch, "start_sec", 0.0) or 0.0)
        end = float(getattr(ch, "end_sec", start) or start)
        duration = max(0.0, end - start)
        analysis = TrackAnalysis(
            duration_sec=duration,
        )
        results.append(
            TrackResult(
                track_id=idx,
                start_time_sec=start,
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
    logger.info(f"Mode: {settings.mode}")
    logger.info(f"Redis URL: {settings.redis_url}")
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    # Import heavy per-track dependencies lazily to allow module import in light contexts/tests
    try:
        from .analysis.per_track import init_spleeter, teardown_spleeter, analyze_track_chunk
    except Exception as e:
        logger.error(f"Per-track analysis dependencies failed to import: {e}")
        # Fail fast: loudness/spleeter are mandatory for worker mode
        raise

    # Initialize Spleeter (ML is always required)
    logger.info("Initializing Spleeter (ML is mandatory)...")
    try:
        init_spleeter()
        logger.info("Spleeter initialized successfully. ML features are ENABLED.")
    except Exception as e:
        logger.error(f"Spleeter initialization failed (ML is mandatory): {e}")
        # In all modes we fail fast, as ML is required
        raise
    try:
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

                # Phase 3: Per-track analysis in parallel
                try:
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    results_dicts = []
                    with ThreadPoolExecutor(max_workers=getattr(global_ctx, "threads", settings.threads)) as pool:
                        futures = [pool.submit(analyze_track_chunk, global_ctx, ch) for ch in chunks]
                        for fut in as_completed(futures):
                            results_dicts.append(fut.result())
                    # Convert dicts to TrackResult models
                    tracks = [TrackResult(**d) for d in results_dicts]
                except Exception as e:
                    logger.warning(f"Per-track analysis failed, falling back to duration-only: {e}")
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
                # In DEV mode, re-raise to surface errors during development
                if settings.is_dev:
                    logger.error("DEV mode enabled: re-raising exception from worker loop")
                    raise
                # Brief backoff
                time.sleep(1)
    finally:
        try:
            # teardown_spleeter is available if import above succeeded
            teardown_spleeter()
        except Exception:
            pass

    logger.info("Worker stopped.")



if __name__ == "__main__":
    main()
